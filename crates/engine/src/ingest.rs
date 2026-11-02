//! JSONL ingestion: streaming reader, options, progress counters, and the
//! process-local job registry.
//!
//! # JSONL format
//!
//! One JSON object per line: `{"id": "<string>", "vector": [f32, ...]}`.
//! Extra fields are ignored. Blank lines are skipped. Lines longer than
//! [`crate::limits::MAX_JSONL_LINE_BYTES`] stop the job (the reader cannot
//! resynchronise safely).
//!
//! # Error policy ([`OnError`])
//!
//! * `fail_fast` (default): the first invalid line stops the job, which is
//!   reported as `failed` with its line number. Records **before** that line
//!   remain indexed and are persisted (there is no rollback).
//! * `skip`: invalid lines are counted as rejected (the first
//!   [`MAX_REJECTED_SAMPLES`] are reported with line numbers) and ingestion
//!   continues.
//!
//! Job status lives in memory only; it is lost when the server restarts.

use crate::collection::Collection;
use crate::error::EngineError;
use crate::limits::MAX_JSONL_LINE_BYTES;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Rejected lines reported per job.
pub const MAX_REJECTED_SAMPLES: usize = 20;
/// Jobs retained in the registry; older finished jobs are evicted.
pub const MAX_RETAINED_JOBS: usize = 1_000;
/// Default checkpoint interval for incremental ingestion.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnError {
    #[default]
    FailFast,
    Skip,
}

#[derive(Debug, Clone, Copy)]
pub struct IngestOptions {
    pub on_error: OnError,
    /// Incremental mode persists metadata every `checkpoint_every` records.
    pub checkpoint_every: usize,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            on_error: OnError::FailFast,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
        }
    }
}

/// Live counters, readable while a job runs.
#[derive(Debug, Default)]
pub struct IngestProgress {
    pub records_read: AtomicU64,
    pub inserted: AtomicU64,
    pub updated: AtomicU64,
    pub rejected: AtomicU64,
    pub current_line: AtomicU64,
    /// Nodes linked into the graph during an initial bulk build.
    pub nodes_linked: AtomicU64,
}

/// A parsed record.
#[derive(Debug, Clone, Deserialize)]
pub struct IngestRecord {
    pub id: String,
    pub vector: Vec<f32>,
}

/// One item produced by an ingestion source.
#[derive(Debug)]
pub enum SourceItem {
    Record {
        line: u64,
        record: IngestRecord,
    },
    /// A malformed line; handled according to [`OnError`].
    Invalid {
        line: u64,
        reason: String,
    },
    /// An unrecoverable read error; always stops the job.
    Fatal {
        line: u64,
        reason: String,
    },
}

/// Streams [`SourceItem`]s from a JSONL reader without loading the file.
pub struct JsonlRecords<R: BufRead> {
    reader: R,
    line: u64,
    buf: Vec<u8>,
    done: bool,
}

impl<R: BufRead> JsonlRecords<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line: 0,
            buf: Vec::new(),
            done: false,
        }
    }
}

impl<R: BufRead> Iterator for JsonlRecords<R> {
    type Item = SourceItem;

    fn next(&mut self) -> Option<SourceItem> {
        loop {
            if self.done {
                return None;
            }
            self.buf.clear();
            let limit = (MAX_JSONL_LINE_BYTES as u64) + 1;
            let read = (&mut self.reader)
                .take(limit)
                .read_until(b'\n', &mut self.buf);
            let n = match read {
                Ok(n) => n,
                Err(e) => {
                    self.done = true;
                    return Some(SourceItem::Fatal {
                        line: self.line + 1,
                        reason: format!("read error: {e}"),
                    });
                }
            };
            if n == 0 {
                self.done = true;
                return None;
            }
            self.line += 1;
            if self.buf.len() > MAX_JSONL_LINE_BYTES && self.buf.last() != Some(&b'\n') {
                self.done = true;
                return Some(SourceItem::Fatal {
                    line: self.line,
                    reason: format!(
                        "line exceeds the maximum length of {MAX_JSONL_LINE_BYTES} bytes"
                    ),
                });
            }
            while matches!(self.buf.last(), Some(b'\n') | Some(b'\r')) {
                self.buf.pop();
            }
            if self.buf.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            return Some(match serde_json::from_slice::<IngestRecord>(&self.buf) {
                Ok(record) => SourceItem::Record {
                    line: self.line,
                    record,
                },
                Err(e) => SourceItem::Invalid {
                    line: self.line,
                    reason: format!("invalid record: {e}"),
                },
            });
        }
    }
}

/// Converts in-memory records into a source (benchmarks and tests).
pub fn records_source(records: Vec<IngestRecord>) -> impl Iterator<Item = SourceItem> {
    records
        .into_iter()
        .enumerate()
        .map(|(i, record)| SourceItem::Record {
            line: i as u64 + 1,
            record,
        })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RejectedRecord {
    pub line: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum IngestMode {
    /// The collection was empty: vectors are loaded first, the medoid is
    /// chosen, then every node is linked; metadata is written once at the end.
    #[default]
    InitialBuild,
    /// The collection already had data: each record is upserted (fully
    /// linked) under the write lock; metadata is checkpointed periodically.
    Incremental,
}

/// Final outcome of an ingestion run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IngestSummary {
    pub mode: IngestMode,
    pub records_read: u64,
    pub inserted: u64,
    pub updated: u64,
    pub rejected: u64,
    pub rejected_samples: Vec<RejectedRecord>,
    /// Set when the run stopped early or failed. Client-safe text.
    pub error: Option<String>,
}

impl IngestSummary {
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }

    pub(crate) fn reject(&mut self, progress: &IngestProgress, line: u64, reason: String) {
        self.rejected += 1;
        progress.rejected.fetch_add(1, Ordering::Relaxed);
        if self.rejected_samples.len() < MAX_REJECTED_SAMPLES {
            self.rejected_samples.push(RejectedRecord { line, reason });
        }
    }

    pub(crate) fn fail(&mut self, message: String) {
        if self.error.is_none() {
            self.error = Some(message);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug)]
struct JobInner {
    state: JobState,
    created_at_ms: u64,
    started_at_ms: Option<u64>,
    finished_at_ms: Option<u64>,
    mode: Option<IngestMode>,
    error: Option<String>,
    rejected_samples: Vec<RejectedRecord>,
}

/// One ingestion job.
#[derive(Debug)]
pub struct IngestJob {
    pub id: String,
    pub collection: String,
    pub on_error: OnError,
    pub progress: IngestProgress,
    inner: Mutex<JobInner>,
}

/// Serializable job status returned by the API.
#[derive(Debug, Clone, Serialize)]
pub struct JobSnapshot {
    pub job_id: String,
    pub collection: String,
    pub state: JobState,
    pub on_error: OnError,
    pub mode: Option<IngestMode>,
    pub records_read: u64,
    pub records_inserted: u64,
    pub records_updated: u64,
    pub records_rejected: u64,
    pub current_line: u64,
    pub nodes_linked: u64,
    pub created_at_ms: u64,
    pub started_at_ms: Option<u64>,
    pub finished_at_ms: Option<u64>,
    pub error: Option<String>,
    pub rejected_samples: Vec<RejectedRecord>,
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl IngestJob {
    fn new(id: String, collection: String, on_error: OnError) -> Self {
        Self {
            id,
            collection,
            on_error,
            progress: IngestProgress::default(),
            inner: Mutex::new(JobInner {
                state: JobState::Queued,
                created_at_ms: now_ms(),
                started_at_ms: None,
                finished_at_ms: None,
                mode: None,
                error: None,
                rejected_samples: Vec::new(),
            }),
        }
    }

    fn with_inner<T>(&self, f: impl FnOnce(&mut JobInner) -> T) -> T {
        // A poisoned job mutex only means a panic happened while updating
        // plain status fields; the data is still usable for reporting.
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut guard)
    }

    pub fn state(&self) -> JobState {
        self.with_inner(|i| i.state)
    }

    pub fn mark_running(&self) {
        self.with_inner(|i| {
            i.state = JobState::Running;
            i.started_at_ms = Some(now_ms());
        });
    }

    /// Marks the job failed unless it already finished.
    pub fn mark_failed(&self, message: impl Into<String>) {
        let message = message.into();
        self.with_inner(|i| {
            if matches!(i.state, JobState::Completed | JobState::Failed) {
                return;
            }
            i.state = JobState::Failed;
            i.error = Some(message);
            i.finished_at_ms = Some(now_ms());
        });
    }

    /// Records the final summary: `completed` only if the run had no error.
    pub fn finish(&self, summary: &IngestSummary) {
        self.with_inner(|i| {
            i.mode = Some(summary.mode);
            i.rejected_samples = summary.rejected_samples.clone();
            i.error = summary.error.clone();
            i.state = if summary.is_success() {
                JobState::Completed
            } else {
                JobState::Failed
            };
            i.finished_at_ms = Some(now_ms());
        });
    }

    pub fn snapshot(&self) -> JobSnapshot {
        let p = &self.progress;
        self.with_inner(|i| JobSnapshot {
            job_id: self.id.clone(),
            collection: self.collection.clone(),
            state: i.state,
            on_error: self.on_error,
            mode: i.mode,
            records_read: p.records_read.load(Ordering::Relaxed),
            records_inserted: p.inserted.load(Ordering::Relaxed),
            records_updated: p.updated.load(Ordering::Relaxed),
            records_rejected: p.rejected.load(Ordering::Relaxed),
            current_line: p.current_line.load(Ordering::Relaxed),
            nodes_linked: p.nodes_linked.load(Ordering::Relaxed),
            created_at_ms: i.created_at_ms,
            started_at_ms: i.started_at_ms,
            finished_at_ms: i.finished_at_ms,
            error: i.error.clone(),
            rejected_samples: i.rejected_samples.clone(),
        })
    }

    fn finished_at(&self) -> Option<u64> {
        self.with_inner(|i| i.finished_at_ms)
    }
}

/// Process-local registry of ingestion jobs.
#[derive(Debug, Default)]
pub struct JobRegistry {
    jobs: DashMap<String, Arc<IngestJob>>,
    seq: AtomicU64,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new queued job for `collection`.
    pub fn create(&self, collection: &str, on_error: OnError) -> Arc<IngestJob> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let id = format!("job-{}-{}", now_ms(), seq);
        let job = Arc::new(IngestJob::new(id.clone(), collection.to_string(), on_error));
        self.jobs.insert(id, Arc::clone(&job));
        self.evict_finished();
        job
    }

    /// Looks up a job; it must belong to `collection`.
    pub fn get(&self, collection: &str, job_id: &str) -> Option<Arc<IngestJob>> {
        let job = self.jobs.get(job_id).map(|r| Arc::clone(r.value()))?;
        if job.collection == collection {
            Some(job)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    fn evict_finished(&self) {
        if self.jobs.len() <= MAX_RETAINED_JOBS {
            return;
        }
        let mut finished: Vec<(u64, String)> = self
            .jobs
            .iter()
            .filter_map(|e| e.value().finished_at().map(|t| (t, e.key().clone())))
            .collect();
        finished.sort();
        let excess = self.jobs.len() - MAX_RETAINED_JOBS;
        for (_, id) in finished.into_iter().take(excess) {
            self.jobs.remove(&id);
        }
    }
}

/// Validates a client-supplied server-side ingestion path.
///
/// * The path must be absolute and refer to an existing regular file.
/// * If `allowed_root` is set (env `RISKANN_INGEST_DIR`), the canonical path
///   must lie inside the canonical root; symlinks are resolved first, so
///   `..` or links cannot escape it.
///
/// Error messages never echo the path back to the client.
pub fn resolve_ingest_path(raw: &str, allowed_root: Option<&Path>) -> Result<PathBuf, EngineError> {
    if raw.trim().is_empty() {
        return Err(EngineError::invalid("path", "must not be empty"));
    }
    let p = Path::new(raw);
    if !p.is_absolute() {
        return Err(EngineError::invalid(
            "path",
            "must be an absolute path on the server",
        ));
    }
    let canonical = std::fs::canonicalize(p).map_err(|_| {
        EngineError::IngestFileUnavailable("bulk file not found or not accessible".into())
    })?;
    if let Some(root) = allowed_root {
        let root = std::fs::canonicalize(root).map_err(|_| {
            EngineError::Internal("configured ingestion directory is not accessible".into())
        })?;
        if !canonical.starts_with(&root) {
            return Err(EngineError::IngestPathNotAllowed);
        }
    }
    if !canonical.is_file() {
        return Err(EngineError::IngestFileUnavailable(
            "bulk file path is not a regular file".into(),
        ));
    }
    Ok(canonical)
}

/// Runs a JSONL ingestion job to completion on the current thread.
/// The job ends `completed` or `failed`; it is never left `running`.
pub fn run_ingest_job(
    collection: Arc<Collection>,
    job: Arc<IngestJob>,
    path: PathBuf,
    options: IngestOptions,
) {
    // Every log line emitted while the job runs (including engine logs)
    // carries the job id and collection name.
    let span = tracing::info_span!("ingest_job", job = %job.id, collection = %job.collection);
    let _entered = span.enter();
    job.mark_running();
    let file_bytes = std::fs::metadata(&path).map(|m| m.len()).ok();
    tracing::info!(on_error = ?options.on_error, file_bytes = ?file_bytes, "ingestion job started");
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "cannot open bulk file");
            job.mark_failed("could not open the bulk file");
            return;
        }
    };
    let source = JsonlRecords::new(BufReader::new(file));
    let summary = collection.ingest(source, &options, &job.progress);
    job.finish(&summary);
    match &summary.error {
        None => tracing::info!(
            inserted = summary.inserted,
            updated = summary.updated,
            rejected = summary.rejected,
            "ingestion job completed"
        ),
        Some(err) => tracing::warn!(
            inserted = summary.inserted,
            updated = summary.updated,
            rejected = summary.rejected,
            error = %err,
            "ingestion job failed"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn items(text: &str) -> Vec<SourceItem> {
        JsonlRecords::new(Cursor::new(text.as_bytes().to_vec())).collect()
    }

    #[test]
    fn parses_records_skips_blank_lines_and_reports_line_numbers() {
        let text = "{\"id\":\"a\",\"vector\":[1,2]}\n\n  \n{\"id\":\"b\",\"vector\":[3,4],\"extra\":1}\r\nnot json\n{\"id\":\"c\",\"vector\":[5,6]}";
        let got = items(text);
        assert_eq!(got.len(), 4);
        match &got[0] {
            SourceItem::Record { line, record } => {
                assert_eq!(*line, 1);
                assert_eq!(record.id, "a");
                assert_eq!(record.vector, vec![1.0, 2.0]);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(&got[1], SourceItem::Record { line: 4, .. }));
        assert!(matches!(&got[2], SourceItem::Invalid { line: 5, .. }));
        assert!(matches!(&got[3], SourceItem::Record { line: 6, .. }));
    }

    #[test]
    fn missing_fields_are_invalid() {
        let got = items("{\"id\":\"a\"}\n{\"vector\":[1]}\n{\"id\":5,\"vector\":[1]}\n");
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|i| matches!(i, SourceItem::Invalid { .. })));
    }

    #[test]
    fn job_state_transitions() {
        let reg = JobRegistry::new();
        let job = reg.create("c", OnError::Skip);
        assert_eq!(job.state(), JobState::Queued);
        job.mark_running();
        assert_eq!(job.state(), JobState::Running);
        let summary = IngestSummary::default();
        job.finish(&summary);
        assert_eq!(job.state(), JobState::Completed);
        // A finished job cannot be flipped to failed afterwards.
        job.mark_failed("late panic report");
        assert_eq!(job.state(), JobState::Completed);
        let snap = job.snapshot();
        assert!(snap.finished_at_ms.is_some());
        assert!(reg.get("c", &job.id).is_some());
        assert!(reg.get("other", &job.id).is_none());
    }

    #[test]
    fn failed_summary_marks_job_failed() {
        let reg = JobRegistry::new();
        let job = reg.create("c", OnError::FailFast);
        job.mark_running();
        let mut summary = IngestSummary::default();
        summary.fail("line 3: bad".into());
        job.finish(&summary);
        let snap = job.snapshot();
        assert_eq!(snap.state, JobState::Failed);
        assert_eq!(snap.error.as_deref(), Some("line 3: bad"));
    }

    #[test]
    fn ingest_path_policy() {
        assert!(matches!(
            resolve_ingest_path("relative/file.jsonl", None),
            Err(EngineError::InvalidArgument { .. })
        ));
        assert!(matches!(
            resolve_ingest_path("", None),
            Err(EngineError::InvalidArgument { .. })
        ));
        let missing = std::env::temp_dir().join("riskann-definitely-missing-file.jsonl");
        assert!(matches!(
            resolve_ingest_path(missing.to_str().unwrap(), None),
            Err(EngineError::IngestFileUnavailable(_))
        ));
    }
}
