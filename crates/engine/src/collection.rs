//! A single RiskANN collection.
//!
//! # Synchronisation contract
//!
//! Each collection owns one `std::sync::RwLock<IndexState>` that guards the
//! memory-mapped store, the id map and the entry point together.
//!
//! * Searches, exact searches, vector reads and `info()` take the **read**
//!   lock and run concurrently with each other.
//! * Upserts take the **write** lock for the whole insert (vector write,
//!   graph search, pruning, reverse edges, metadata persist), so a reader can
//!   never observe a partially written vector or adjacency list.
//! * An initial bulk build (ingestion into an empty collection) holds the
//!   write lock for the whole build; searches wait and then see either the
//!   empty collection or the complete index. Incremental ingestion takes the
//!   write lock per record, so searches interleave and observe a consistent
//!   prefix of the file.
//!
//! There is a single lock per collection, so lock ordering cannot deadlock.
//! Write throughput is serialised by design (correctness first).
//!
//! # Live slots
//!
//! Internal ids are allocated densely: ids `0..len()` are live, ids
//! `len()..capacity` are unused. Vector deletion is not implemented, so
//! there are no tombstones. Metadata is written only after the vector and
//! graph data it describes, so metadata never claims an id without data.
//!
//! # Persistence order (per upsert / checkpoint)
//!
//! 1. vector and adjacency writes into the mappings,
//! 2. `msync` of both mappings,
//! 3. `metadata.json` written to a temp file, `fsync`ed, renamed over the old
//!    file, and the directory `fsync`ed (Unix).
//!
//! Remaining failure windows (documented, not hidden): a crash between 1 and
//! 3 can leave adjacency entries pointing at ids that metadata does not list;
//! these dangling edges are removed when the collection is opened. An
//! in-place update of an existing id that was not yet checkpointed may be
//! observed as either the old or the new vector after a crash.

use crate::error::EngineError;
use crate::ingest::{
    IngestMode, IngestOptions, IngestProgress, IngestSummary, OnError, SourceItem,
};
use crate::limits::{
    DEFAULT_EF_SEARCH, DEFAULT_ENTRY_POINTS, MAX_ALPHA, MAX_COLLECTION_NAME, MAX_DIM,
    MAX_EF_SEARCH, MAX_ENTRY_POINTS, MAX_EXTERNAL_ID_BYTES, MAX_GRAPH_SNAPSHOT,
    MAX_INITIAL_CAPACITY, MAX_K, MAX_L_BUILD, MAX_R, MIN_ALPHA,
};
use riskann_core::distances::{DistanceMetric, squared_norm};
use riskann_core::vamana::{BeamSearchConfig, Candidate, SearchError, beam_search, robust_prune};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;
use storage::{DiskStore, MAX_CAPACITY, StorageError};

pub const CONFIG_FILE: &str = "config.json";
pub const METADATA_FILE: &str = "metadata.json";
/// Metadata/storage format version written by this build.
/// Version 0 = original layout (`id_map` in metadata, no version field).
pub const FORMAT_VERSION: u32 = 1;
/// Bulk-build progress is logged every this many records / linked nodes
/// (or every 10 % of the nodes when linking, whichever is larger).
const BULK_PROGRESS_EVERY: u64 = 50_000;
/// Incremental ingestion progress is logged every this many records.
const INCREMENTAL_PROGRESS_EVERY: u64 = 10_000;

/// Collection configuration, persisted in `config.json`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CollectionConfig {
    pub dim: usize,
    pub metric: DistanceMetric,
    /// Maximum out-degree of every graph node.
    pub r: usize,
    /// Robust-pruning parameter (`alpha >= 1`).
    pub alpha: f32,
    /// Construction beam width (Vamana's `L`). Absent in collections created
    /// before this field existed; those keep the original `2 * r`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l_build: Option<usize>,
    /// Maximum number of search entry points (the number used is
    /// `min(entry_points, ceil(sqrt(n)))`). Absent in collections created
    /// before this field existed; those keep a single entry point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_points: Option<usize>,
}

/// Default construction beam for new collections: `max(2r, 64)`, capped.
pub fn default_l_build(r: usize) -> usize {
    r.saturating_mul(2).clamp(64, MAX_L_BUILD)
}

impl CollectionConfig {
    pub fn new(dim: usize, metric: DistanceMetric, r: usize, alpha: f32) -> Self {
        Self {
            dim,
            metric,
            r,
            alpha,
            l_build: Some(default_l_build(r)),
            entry_points: Some(DEFAULT_ENTRY_POINTS),
        }
    }

    pub fn effective_l_build(&self) -> usize {
        self.l_build
            .unwrap_or_else(|| self.r.saturating_mul(2))
            .clamp(1, MAX_L_BUILD)
    }

    pub fn effective_entry_points(&self) -> usize {
        self.entry_points.unwrap_or(1).clamp(1, MAX_ENTRY_POINTS)
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if self.dim == 0 || self.dim > MAX_DIM {
            return Err(EngineError::invalid(
                "dim",
                format!("must be in 1..={MAX_DIM}"),
            ));
        }
        if self.r == 0 || self.r > MAX_R {
            return Err(EngineError::invalid("r", format!("must be in 1..={MAX_R}")));
        }
        if !self.alpha.is_finite() || self.alpha < MIN_ALPHA || self.alpha > MAX_ALPHA {
            return Err(EngineError::invalid(
                "alpha",
                format!("must be a finite number in [{MIN_ALPHA}, {MAX_ALPHA}]"),
            ));
        }
        if matches!(self.l_build, Some(l) if l == 0 || l > MAX_L_BUILD) {
            return Err(EngineError::invalid(
                "l_build",
                format!("must be in 1..={MAX_L_BUILD}"),
            ));
        }
        if matches!(self.entry_points, Some(e) if e == 0 || e > MAX_ENTRY_POINTS) {
            return Err(EngineError::invalid(
                "entry_points",
                format!("must be in 1..={MAX_ENTRY_POINTS}"),
            ));
        }
        Ok(())
    }
}

/// `metadata.json`. Reads both format 0 (`id_map`) and format 1
/// (`external_ids`, indexed by internal id); always writes format 1.
#[derive(Serialize, Deserialize, Default, Debug)]
struct MetadataFile {
    #[serde(default)]
    format_version: u32,
    #[serde(default)]
    medoid_id: Option<u32>,
    #[serde(default)]
    next_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    capacity: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    external_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id_map: Option<HashMap<String, u32>>,
    /// Absent before entry-point sets existed: then `[medoid_id]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    entry_points: Option<Vec<u32>>,
}

struct IndexState {
    store: DiskStore,
    /// `external_ids[i]` is the external id of internal id `i`.
    external_ids: Vec<String>,
    id_map: HashMap<String, u32>,
    /// Primary entry point; `None` iff the collection is empty.
    medoid: Option<u32>,
    /// Every search starts from all of these. `entry_points[0] == medoid`;
    /// empty iff the collection is empty.
    entry_points: Vec<u32>,
}

impl IndexState {
    fn len(&self) -> u32 {
        self.external_ids.len() as u32
    }
}

/// Approximate-search options.
#[derive(Debug, Clone, Copy)]
pub struct QueryOptions {
    pub k: usize,
    pub ef_search: usize,
    pub include_vectors: bool,
    /// Record the order in which graph nodes were expanded.
    pub trace: bool,
}

impl QueryOptions {
    /// `ef_search` defaults to `max(k, DEFAULT_EF_SEARCH)`.
    pub fn new(k: usize, ef_search: Option<usize>) -> Self {
        Self {
            k,
            ef_search: ef_search.unwrap_or_else(|| k.max(DEFAULT_EF_SEARCH)),
            include_vectors: false,
            trace: false,
        }
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        validate_k(self.k)?;
        if self.ef_search < self.k || self.ef_search > MAX_EF_SEARCH {
            return Err(EngineError::invalid(
                "ef_search",
                format!("must be >= k ({}) and <= {MAX_EF_SEARCH}", self.k),
            ));
        }
        Ok(())
    }
}

fn validate_k(k: usize) -> Result<(), EngineError> {
    if k == 0 || k > MAX_K {
        return Err(EngineError::invalid("k", format!("must be in 1..={MAX_K}")));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchHit {
    pub external_id: String,
    pub internal_id: u32,
    pub distance: f32,
    pub vector: Option<Vec<f32>>,
}

/// Per-query counters. `duration_us` is measured inside the engine: it
/// includes validation, lock acquisition, graph traversal and result
/// assembly, but not HTTP parsing or JSON serialisation.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct QueryStats {
    pub visited_nodes: usize,
    pub expanded_nodes: usize,
    pub distance_computations: usize,
    pub returned: usize,
    pub duration_us: u64,
}

#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub stats: QueryStats,
    /// Present only for approximate searches with `trace = true`.
    pub trace: Option<SearchTrace>,
}

/// How an approximate search walked the graph: the entry point and every
/// expanded node, in expansion order. Used by the console visualisation.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchTrace {
    /// Primary entry point.
    pub entry_point: u32,
    /// Every entry point the search started from.
    pub entry_points: Vec<u32>,
    pub expanded: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    pub internal_id: u32,
    pub id: String,
    pub vector: Vec<f32>,
}

/// Nodes `0..min(limit, count)` with their vectors and the edges between
/// them (edges to nodes outside the snapshot are omitted).
#[derive(Debug, Clone, Serialize)]
pub struct GraphSnapshot {
    pub collection: String,
    pub metric: DistanceMetric,
    pub dim: usize,
    pub r: usize,
    pub count: u32,
    pub entry_point: Option<u32>,
    pub entry_points: Vec<u32>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<[u32; 2]>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UpsertOutcome {
    pub internal_id: u32,
    /// `true` for a new external id, `false` for an update in place.
    pub created: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorRecord {
    pub id: String,
    pub internal_id: u32,
    pub vector: Vec<f32>,
}

/// Safe-to-expose collection summary (no filesystem paths).
#[derive(Debug, Clone, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dim: usize,
    pub metric: DistanceMetric,
    pub r: usize,
    pub alpha: f32,
    pub l_build: usize,
    pub count: u32,
    pub capacity: u32,
    pub entry_point: Option<u32>,
    pub entry_point_count: usize,
    pub format_version: u32,
    /// `"ready"`, or `"ingesting"` while an ingestion job is running.
    pub index_state: &'static str,
    pub disk_bytes: u64,
}

/// Validates a collection name: 1–64 characters from `[A-Za-z0-9_-]`.
/// This also guarantees that the name is a single safe path component.
pub fn validate_collection_name(name: &str) -> Result<(), EngineError> {
    if name.is_empty() || name.len() > MAX_COLLECTION_NAME {
        return Err(EngineError::InvalidCollectionName(format!(
            "must be 1..={MAX_COLLECTION_NAME} characters"
        )));
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(EngineError::InvalidCollectionName(
            "only ASCII letters, digits, '_' and '-' are allowed".into(),
        ));
    }
    Ok(())
}

fn validate_external_id(id: &str) -> Result<(), EngineError> {
    if id.is_empty() {
        return Err(EngineError::invalid("id", "must not be empty"));
    }
    if id.len() > MAX_EXTERNAL_ID_BYTES {
        return Err(EngineError::invalid(
            "id",
            format!("must be at most {MAX_EXTERNAL_ID_BYTES} bytes"),
        ));
    }
    Ok(())
}

/// Writes `bytes` to `path` atomically where the platform allows:
/// temp file, `fsync`, rename, then `fsync` of the directory on Unix.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| EngineError::Internal("invalid metadata path".into()))?;
    let tmp = path.with_file_name(format!("{file_name}.tmp"));
    {
        let mut f =
            File::create(&tmp).map_err(EngineError::io(format!("create {}", tmp.display())))?;
        f.write_all(bytes)
            .map_err(EngineError::io(format!("write {}", tmp.display())))?;
        f.sync_all()
            .map_err(EngineError::io(format!("fsync {}", tmp.display())))?;
    }
    std::fs::rename(tmp, path).map_err(EngineError::io(format!("rename {}", path.display())))?;
    #[cfg(unix)]
    {
        // Best effort: directory fsync is not supported on every filesystem.
        if let Some(dir) = path.parent().and_then(|p| File::open(p).ok()) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, what: &str) -> Result<T, EngineError> {
    let bytes = std::fs::read(path).map_err(EngineError::io(format!("read {}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| EngineError::Corrupt(format!("{what} is not valid: {e}")))
}

#[inline]
fn access_err(id: u32, e: StorageError) -> SearchError {
    SearchError::Access {
        id,
        message: e.to_string(),
    }
}

#[inline]
fn dist_to(
    store: &DiskStore,
    metric: DistanceMetric,
    id: u32,
    query: &[f32],
) -> Result<f32, SearchError> {
    let v = store.read_vector(id).map_err(|e| access_err(id, e))?;
    Ok(metric.calculate(v, query))
}

#[inline]
fn pair_dist(
    store: &DiskStore,
    metric: DistanceMetric,
    a: u32,
    b: u32,
) -> Result<f32, SearchError> {
    let va = store.read_vector(a).map_err(|e| access_err(a, e))?;
    let vb = store.read_vector(b).map_err(|e| access_err(b, e))?;
    Ok(metric.calculate(va, vb))
}

/// Decrements the active-ingestion counter on drop (also on panic).
struct IngestionGuard<'a>(&'a AtomicUsize);

impl Drop for IngestionGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct Collection {
    name: String,
    dir: PathBuf,
    config: CollectionConfig,
    state: RwLock<IndexState>,
    deleted: AtomicBool,
    active_ingestions: AtomicUsize,
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.name)
            .field("config", &self.config)
            .finish()
    }
}

impl Collection {
    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Creates a new collection in `dir` (which must not exist yet).
    ///
    /// Write order: data files, then `metadata.json`, then `config.json`.
    /// `config.json` marks a complete collection; a directory without it is
    /// ignored at startup. On failure the partially created directory is
    /// removed.
    pub fn create(
        dir: PathBuf,
        name: &str,
        config: CollectionConfig,
        capacity: u32,
    ) -> Result<Self, EngineError> {
        validate_collection_name(name)?;
        config.validate()?;
        if capacity == 0 || capacity > MAX_INITIAL_CAPACITY {
            return Err(EngineError::invalid(
                "initial_capacity",
                format!("must be in 1..={MAX_INITIAL_CAPACITY}"),
            ));
        }
        match std::fs::create_dir(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(EngineError::CollectionExists(format!(
                    "collection '{name}' already exists on disk"
                )));
            }
            Err(e) => return Err(EngineError::io(format!("create {}", dir.display()))(e)),
        }
        match Self::create_files(&dir, &config, capacity) {
            Ok(state) => Ok(Self {
                name: name.to_string(),
                dir,
                config,
                state: RwLock::new(state),
                deleted: AtomicBool::new(false),
                active_ingestions: AtomicUsize::new(0),
            }),
            Err(e) => {
                let _ = std::fs::remove_dir_all(dir);
                Err(e)
            }
        }
    }

    fn create_files(
        dir: &Path,
        config: &CollectionConfig,
        capacity: u32,
    ) -> Result<IndexState, EngineError> {
        let store = DiskStore::create_new(dir, config.dim, config.r, capacity)?;
        let state = IndexState {
            store,
            external_ids: Vec::new(),
            id_map: HashMap::new(),
            medoid: None,
            entry_points: Vec::new(),
        };
        write_metadata(dir, &state)?;
        let cfg = serde_json::to_vec_pretty(config)?;
        write_atomic(&dir.join(CONFIG_FILE), &cfg)?;
        Ok(state)
    }

    /// Opens an existing collection directory.
    ///
    /// Capacity is derived from the data files (never truncated). Fails with
    /// [`EngineError::Corrupt`] if the metadata and files disagree, instead
    /// of silently starting empty. Dangling, self or duplicate edges left by
    /// an interrupted write are removed and the cleaned lists written back.
    pub fn open(dir: PathBuf) -> Result<Self, EngineError> {
        let opened_at = Instant::now();
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                EngineError::InvalidCollectionName("directory name is not UTF-8".into())
            })?
            .to_string();
        validate_collection_name(&name)?;

        let config: CollectionConfig = read_json(&dir.join(CONFIG_FILE), CONFIG_FILE)?;
        config.validate()?;
        let meta_path = dir.join(METADATA_FILE);
        if !meta_path.exists() {
            return Err(EngineError::Corrupt(format!(
                "{METADATA_FILE} is missing for collection '{name}'"
            )));
        }
        let meta: MetadataFile = read_json(&meta_path, METADATA_FILE)?;
        if meta.format_version > FORMAT_VERSION {
            return Err(EngineError::Corrupt(format!(
                "format version {} is newer than supported version {FORMAT_VERSION}",
                meta.format_version
            )));
        }

        let count = meta.next_id;
        let external_ids = resolve_external_ids(&meta)?;
        let mut id_map = HashMap::with_capacity(external_ids.len());
        for (i, ext) in external_ids.iter().enumerate() {
            if ext.is_empty() {
                return Err(EngineError::Corrupt(format!(
                    "internal id {i} has an empty external id"
                )));
            }
            if id_map.insert(ext.clone(), i as u32).is_some() {
                return Err(EngineError::Corrupt(format!(
                    "external id '{ext}' appears twice"
                )));
            }
        }
        match (count, meta.medoid_id) {
            (0, Some(_)) => {
                return Err(EngineError::Corrupt(
                    "empty collection has an entry point".into(),
                ));
            }
            (n, None) if n > 0 => {
                return Err(EngineError::Corrupt(
                    "non-empty collection has no entry point".into(),
                ));
            }
            (n, Some(m)) if m >= n => {
                return Err(EngineError::Corrupt(format!(
                    "entry point {m} is not a live id (count {n})"
                )));
            }
            _ => {}
        }

        let entry_points: Vec<u32> = match &meta.entry_points {
            Some(eps) => eps.clone(),
            None => meta.medoid_id.into_iter().collect(),
        };
        if entry_points.first().copied() != meta.medoid_id {
            return Err(EngineError::Corrupt(
                "entry_points must start with medoid_id".into(),
            ));
        }
        for (i, &e) in entry_points.iter().enumerate() {
            if e >= count || entry_points[..i].contains(&e) {
                return Err(EngineError::Corrupt(format!("invalid entry point {e}")));
            }
        }

        let (store, report) = DiskStore::open_existing(&dir, config.dim, config.r, count)?;
        if report.repaired {
            tracing::warn!(
                collection = %name,
                vectors_capacity = report.vectors_capacity,
                graph_capacity = report.graph_capacity,
                "data files had different capacities (interrupted resize); grew the shorter file"
            );
        }
        if let Some(recorded) = meta.capacity.filter(|&c| c != store.capacity()) {
            tracing::info!(
                collection = %name,
                recorded,
                actual = store.capacity(),
                "metadata capacity differs from data files; using the file-derived capacity"
            );
        }
        let mut state = IndexState {
            store,
            external_ids,
            id_map,
            medoid: meta.medoid_id,
            entry_points,
        };
        let removed = sanitize_graph(&mut state)?;
        if removed > 0 {
            tracing::warn!(collection = %name, removed_edges = removed, "removed dangling/invalid edges on open");
            state.store.flush()?;
        }
        tracing::info!(
            collection = %name,
            vectors = count,
            capacity = state.store.capacity(),
            dim = config.dim,
            metric = config.metric.name(),
            r = config.r,
            l_build = config.effective_l_build(),
            format_version = meta.format_version,
            elapsed_ms = opened_at.elapsed().as_millis() as u64,
            "opened collection"
        );
        Ok(Self {
            name,
            dir,
            config,
            state: RwLock::new(state),
            deleted: AtomicBool::new(false),
            active_ingestions: AtomicUsize::new(0),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn is_deleted(&self) -> bool {
        self.deleted.load(Ordering::SeqCst)
    }

    /// Marks the collection deleted, then waits for in-flight writers to
    /// finish (by acquiring the write lock once). Running ingestion stops at
    /// its next record; later persists are refused.
    pub fn mark_deleted_and_wait(&self) {
        self.deleted.store(true, Ordering::SeqCst);
        let _guard = match self.state.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
    }

    fn read_state(&self) -> Result<RwLockReadGuard<'_, IndexState>, EngineError> {
        self.state.read().map_err(|_| {
            EngineError::Internal("collection lock poisoned by an earlier panic".into())
        })
    }

    fn write_state(&self) -> Result<RwLockWriteGuard<'_, IndexState>, EngineError> {
        self.state.write().map_err(|_| {
            EngineError::Internal("collection lock poisoned by an earlier panic".into())
        })
    }

    fn check_not_deleted(&self) -> Result<(), EngineError> {
        if self.is_deleted() {
            Err(EngineError::Cancelled(format!(
                "collection '{}' was deleted",
                self.name
            )))
        } else {
            Ok(())
        }
    }

    /// Flushes the mappings, then atomically rewrites metadata.
    fn persist_locked(&self, state: &IndexState) -> Result<(), EngineError> {
        self.check_not_deleted()?;
        state.store.flush()?;
        write_metadata(&self.dir, state)?;
        tracing::trace!(collection = %self.name, vectors = state.len(), "data flushed and metadata persisted");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Checks dimension, finiteness and (for cosine) a non-zero norm.
    pub fn validate_vector(&self, v: &[f32], field: &'static str) -> Result<(), EngineError> {
        if v.len() != self.config.dim {
            return Err(EngineError::DimensionMismatch {
                expected: self.config.dim,
                got: v.len(),
            });
        }
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(EngineError::invalid(
                field,
                format!("component {i} is NaN or infinite"),
            ));
        }
        if self.config.metric == DistanceMetric::Cosine && squared_norm(v) == 0.0 {
            return Err(EngineError::invalid(
                field,
                "zero vector is not allowed with the COSINE metric",
            ));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Reads
    // ------------------------------------------------------------------

    pub fn len(&self) -> Result<u32, EngineError> {
        Ok(self.read_state()?.len())
    }

    pub fn is_empty(&self) -> Result<bool, EngineError> {
        Ok(self.len()? == 0)
    }

    pub fn info(&self) -> Result<CollectionInfo, EngineError> {
        let state = self.read_state()?;
        Ok(CollectionInfo {
            name: self.name.clone(),
            dim: self.config.dim,
            metric: self.config.metric,
            r: self.config.r,
            alpha: self.config.alpha,
            l_build: self.config.effective_l_build(),
            count: state.len(),
            capacity: state.store.capacity(),
            entry_point: state.medoid,
            entry_point_count: state.entry_points.len(),
            format_version: FORMAT_VERSION,
            index_state: if self.active_ingestions.load(Ordering::SeqCst) > 0 {
                "ingesting"
            } else {
                "ready"
            },
            disk_bytes: state.store.disk_usage_bytes(),
        })
    }

    /// Returns the stored vector for an external id.
    pub fn get_vector(&self, external_id: &str) -> Result<VectorRecord, EngineError> {
        let state = self.read_state()?;
        let id = *state
            .id_map
            .get(external_id)
            .ok_or_else(|| EngineError::VectorNotFound(external_id.to_string()))?;
        let vector = state.store.read_vector(id)?.to_vec();
        Ok(VectorRecord {
            id: external_id.to_string(),
            internal_id: id,
            vector,
        })
    }

    /// Returns the adjacency list of an internal id (diagnostics/tests).
    pub fn neighbors(&self, internal_id: u32) -> Result<Vec<u32>, EngineError> {
        let state = self.read_state()?;
        if internal_id >= state.len() {
            return Err(EngineError::invalid("internal_id", "not a live id"));
        }
        Ok(state.store.read_edges(internal_id)?)
    }

    /// Checks graph invariants for every live node: degree `<= r`, no
    /// self-edges, no duplicates, every edge targets a live id.
    pub fn verify_graph(&self) -> Result<(), EngineError> {
        let state = self.read_state()?;
        let n = state.len();
        if state.entry_points.first().copied() != state.medoid {
            return Err(EngineError::Corrupt(
                "entry points do not start with the medoid".into(),
            ));
        }
        for (i, &e) in state.entry_points.iter().enumerate() {
            if e >= n || state.entry_points[..i].contains(&e) {
                return Err(EngineError::Corrupt(format!("invalid entry point {e}")));
            }
        }
        for id in 0..n {
            let edges = state.store.read_edges(id)?;
            if edges.len() > self.config.r {
                return Err(EngineError::Corrupt(format!(
                    "node {id} has degree {}",
                    edges.len()
                )));
            }
            for (i, &e) in edges.iter().enumerate() {
                if e >= n || e == id || edges[..i].contains(&e) {
                    return Err(EngineError::Corrupt(format!(
                        "node {id} has invalid edge {e}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn hits_from(
        &self,
        state: &IndexState,
        results: &[Candidate],
        include_vectors: bool,
    ) -> Result<Vec<SearchHit>, EngineError> {
        let mut hits = Vec::with_capacity(results.len());
        for c in results {
            let external_id = state
                .external_ids
                .get(c.id as usize)
                .ok_or_else(|| {
                    EngineError::Corrupt(format!("result id {} has no external id", c.id))
                })?
                .clone();
            let vector = if include_vectors {
                Some(state.store.read_vector(c.id)?.to_vec())
            } else {
                None
            };
            hits.push(SearchHit {
                external_id,
                internal_id: c.id,
                distance: c.distance,
                vector,
            });
        }
        Ok(hits)
    }

    /// Approximate k-NN search over the Vamana-inspired graph.
    ///
    /// Returns up to `k` live vectors in ascending `(distance, internal id)`
    /// order. An empty collection returns an empty result (not an error).
    /// Approximate search does not guarantee the exact nearest neighbours.
    pub fn search(
        &self,
        query: &[f32],
        options: &QueryOptions,
    ) -> Result<SearchResponse, EngineError> {
        let started = Instant::now();
        options.validate()?;
        self.validate_vector(query, "vector")?;
        let state = self.read_state()?;
        let n = state.len() as usize;
        let mut stats = QueryStats::default();
        let mut trace = None;
        let hits = if n == 0 {
            Vec::new()
        } else {
            let entry = state.medoid.ok_or_else(|| {
                EngineError::Corrupt("non-empty collection has no entry point".into())
            })?;
            let metric = self.config.metric;
            let store = &state.store;
            // Options were validated above (1 <= k <= ef_search), so the
            // beam search runs directly; `trace` decides whether the
            // expansion order is recorded.
            let config = BeamSearchConfig {
                ef: options.ef_search,
                num_nodes: n,
                collect_expanded: options.trace,
            };
            let mut out = beam_search(
                &state.entry_points,
                &config,
                |id| dist_to(store, metric, id, query),
                |id, buf| {
                    store
                        .read_edges_into(id, buf)
                        .map_err(|e| access_err(id, e))
                },
            )?;
            out.results.truncate(options.k);
            if options.trace {
                trace = Some(SearchTrace {
                    entry_point: entry,
                    entry_points: state.entry_points.clone(),
                    expanded: out.expanded.iter().map(|c| c.id).collect(),
                });
            }
            stats.visited_nodes = out.stats.visited_nodes;
            stats.expanded_nodes = out.stats.expanded_nodes;
            stats.distance_computations = out.stats.distance_computations;
            self.hits_from(&state, &out.results, options.include_vectors)?
        };
        drop(state);
        stats.returned = hits.len();
        stats.duration_us = started.elapsed().as_micros() as u64;
        tracing::debug!(
            collection = %self.name,
            mode = "approximate",
            k = options.k,
            ef_search = options.ef_search,
            visited = stats.visited_nodes,
            expanded = stats.expanded_nodes,
            returned = stats.returned,
            duration_us = stats.duration_us,
            "search executed"
        );
        Ok(SearchResponse { hits, stats, trace })
    }

    /// Exact brute-force k-NN: scans every live vector with the same metric
    /// and the same `(distance, id)` ordering as [`Collection::search`].
    ///
    /// This is the ground-truth oracle for Recall@K and diagnostics. It is
    /// never used as a fallback for approximate search. Cost is linear in
    /// the collection size.
    pub fn exact_search(
        &self,
        query: &[f32],
        k: usize,
        include_vectors: bool,
    ) -> Result<SearchResponse, EngineError> {
        let started = Instant::now();
        validate_k(k)?;
        self.validate_vector(query, "vector")?;
        let state = self.read_state()?;
        let n = state.len();
        let metric = self.config.metric;
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        for id in 0..n {
            let d = metric.calculate(state.store.read_vector(id)?, query);
            if !d.is_finite() {
                return Err(EngineError::Internal(format!(
                    "non-finite distance for node {id}"
                )));
            }
            let c = Candidate::new(id, d);
            if heap.len() < k {
                heap.push(c);
            } else if heap.peek().is_some_and(|w| c < *w) {
                heap.pop();
                heap.push(c);
            }
        }
        let results = heap.into_sorted_vec();
        let hits = self.hits_from(&state, &results, include_vectors)?;
        drop(state);
        let stats = QueryStats {
            visited_nodes: n as usize,
            expanded_nodes: 0,
            distance_computations: n as usize,
            returned: hits.len(),
            duration_us: started.elapsed().as_micros() as u64,
        };
        tracing::debug!(
            collection = %self.name,
            mode = "exact",
            k,
            scanned = stats.distance_computations,
            returned = stats.returned,
            duration_us = stats.duration_us,
            "search executed"
        );
        Ok(SearchResponse {
            hits,
            stats,
            trace: None,
        })
    }

    // ------------------------------------------------------------------
    // Writes
    // ------------------------------------------------------------------

    /// Snapshot of the first `limit` live nodes and the edges among them,
    /// for visualisation (`GET /collections/{name}/graph`).
    pub fn graph_snapshot(&self, limit: usize) -> Result<GraphSnapshot, EngineError> {
        if limit == 0 || limit > MAX_GRAPH_SNAPSHOT {
            return Err(EngineError::invalid(
                "limit",
                format!("must be in 1..={MAX_GRAPH_SNAPSHOT}"),
            ));
        }
        let state = self.read_state()?;
        let count = state.len();
        let m = (limit as u64).min(count as u64) as u32;
        let mut nodes = Vec::with_capacity(m as usize);
        let mut edges = Vec::new();
        for id in 0..m {
            let ext = state
                .external_ids
                .get(id as usize)
                .ok_or_else(|| EngineError::Corrupt(format!("node {id} has no external id")))?;
            nodes.push(GraphNode {
                internal_id: id,
                id: ext.clone(),
                vector: state.store.read_vector(id)?.to_vec(),
            });
            for e in state.store.read_edges(id)? {
                if e < m {
                    edges.push([id, e]);
                }
            }
        }
        Ok(GraphSnapshot {
            collection: self.name.clone(),
            metric: self.config.metric,
            dim: self.config.dim,
            r: self.config.r,
            count,
            entry_point: state.medoid,
            entry_points: state.entry_points.clone(),
            nodes,
            edges,
            truncated: m < count,
        })
    }

    /// Inserts a new vector or updates the vector of an existing external
    /// id (same internal id), links it into the graph and persists.
    pub fn upsert(&self, external_id: &str, vector: &[f32]) -> Result<UpsertOutcome, EngineError> {
        validate_external_id(external_id)?;
        self.validate_vector(vector, "vector")?;
        let started = Instant::now();
        let mut state = self.write_state()?;
        self.check_not_deleted()?;
        let outcome = self.insert_locked(&mut state, external_id, vector)?;
        self.persist_locked(&state)?;
        drop(state);
        tracing::debug!(
            collection = %self.name,
            internal_id = outcome.internal_id,
            created = outcome.created,
            elapsed_us = started.elapsed().as_micros() as u64,
            "upsert applied"
        );
        Ok(outcome)
    }

    /// Grows storage (doubling) so that `needed` slots fit.
    fn ensure_capacity(&self, state: &mut IndexState, needed: u64) -> Result<(), EngineError> {
        let cap = state.store.capacity() as u64;
        if needed <= cap {
            return Ok(());
        }
        if needed > MAX_CAPACITY as u64 {
            return Err(EngineError::CapacityExceeded(format!(
                "a collection holds at most {MAX_CAPACITY} vectors"
            )));
        }
        let target = cap.saturating_mul(2).max(needed).min(MAX_CAPACITY as u64) as u32;
        state.store.resize(target)?;
        tracing::info!(collection = %self.name, from = cap, to = target, "grew collection storage");
        Ok(())
    }

    /// Caller holds the write lock and has validated the input.
    fn insert_locked(
        &self,
        state: &mut IndexState,
        external_id: &str,
        vector: &[f32],
    ) -> Result<UpsertOutcome, EngineError> {
        if let Some(&id) = state.id_map.get(external_id) {
            state.store.write_vector(id, vector)?;
            self.link_node(state, id, vector)?;
            return Ok(UpsertOutcome {
                internal_id: id,
                created: false,
            });
        }
        let id = state.len();
        self.ensure_capacity(state, id as u64 + 1)?;
        state.store.write_vector(id, vector)?;
        state.store.write_edges(id, &[])?;
        // Link before publishing the id: the search below runs over ids
        // 0..id only; the new node becomes reachable via reverse edges.
        self.link_node(state, id, vector)?;
        state.external_ids.push(external_id.to_string());
        state.id_map.insert(external_id.to_string(), id);
        if state.medoid.is_none() {
            state.medoid = Some(id);
            state.entry_points = vec![id];
        } else if state.entry_points.len() < self.entry_point_target(state.len()) {
            // Collections built by single upserts: early vectors become extra
            // entry points until the target is reached. Ingestion jobs
            // re-select a well-spread set with farthest-point sampling.
            state.entry_points.push(id);
        }
        Ok(UpsertOutcome {
            internal_id: id,
            created: true,
        })
    }

    /// Vamana insertion of node `id` whose vector is `vector`:
    /// 1. beam search from the entry point with `L = l_build`;
    /// 2. RobustPrune over (expanded nodes ∪ final beam), excluding `id`;
    /// 3. write the forward edges;
    /// 4. add the reverse edge `nb -> id` to each new neighbour, re-pruning
    ///    a neighbour whose degree would exceed `r`.
    ///
    /// No-op while the collection has no entry point.
    fn link_node(
        &self,
        state: &mut IndexState,
        id: u32,
        vector: &[f32],
    ) -> Result<(), EngineError> {
        if state.entry_points.is_empty() {
            return Ok(());
        }
        let metric = self.config.metric;
        let r = self.config.r;
        let alpha = self.config.alpha;
        let num_nodes = state.len() as usize;

        let forward: Vec<u32> = {
            let store = &state.store;
            let cfg = BeamSearchConfig {
                ef: self.config.effective_l_build(),
                num_nodes,
                collect_expanded: true,
            };
            let out = beam_search(
                &state.entry_points,
                &cfg,
                |n| dist_to(store, metric, n, vector),
                |n, buf| store.read_edges_into(n, buf).map_err(|e| access_err(n, e)),
            )?;
            let mut pool = out.expanded;
            pool.extend_from_slice(&out.results);
            robust_prune(id, &pool, r, alpha, |a, b| pair_dist(store, metric, a, b))?
        };
        state.store.write_edges(id, &forward)?;

        for &nb in &forward {
            let updated: Option<Vec<u32>> = {
                let store = &state.store;
                let mut edges = store.read_edges(nb)?;
                if edges.contains(&id) {
                    None
                } else {
                    edges.push(id);
                    if edges.len() <= r {
                        Some(edges)
                    } else {
                        let base = store.read_vector(nb)?;
                        let mut pool = Vec::with_capacity(edges.len());
                        for &e in &edges {
                            pool.push(Candidate::new(
                                e,
                                metric.calculate(base, store.read_vector(e)?),
                            ));
                        }
                        Some(robust_prune(nb, &pool, r, alpha, |a, b| {
                            pair_dist(store, metric, a, b)
                        })?)
                    }
                }
            };
            if let Some(list) = updated {
                state.store.write_edges(nb, &list)?;
            }
        }
        Ok(())
    }

    /// Entry point for a bulk build: the live vector nearest (by the
    /// collection metric) to the centroid of all live vectors; ties go to
    /// the lower id.
    fn select_medoid(&self, state: &IndexState) -> Result<u32, EngineError> {
        let n = state.len();
        let dim = self.config.dim;
        let mut sum = vec![0.0f64; dim];
        for id in 0..n {
            for (s, &x) in sum.iter_mut().zip(state.store.read_vector(id)?) {
                *s += x as f64;
            }
        }
        let centroid: Vec<f32> = sum.iter().map(|s| (s / n.max(1) as f64) as f32).collect();
        let mut best = Candidate::new(0, f32::INFINITY);
        for id in 0..n {
            let c = Candidate::new(
                id,
                self.config
                    .metric
                    .calculate(&centroid, state.store.read_vector(id)?),
            );
            if c < best {
                best = c;
            }
        }
        Ok(best.id)
    }

    /// Entry points to use for `n` vectors: `min(entry_points, ceil(sqrt(n)))`.
    fn entry_point_target(&self, n: u32) -> usize {
        let sqrt = (n as f64).sqrt().ceil() as usize;
        self.config.effective_entry_points().min(sqrt.max(1))
    }

    /// Deterministic farthest-point sampling: start at the medoid, then keep
    /// adding the live vector farthest from every entry point chosen so far
    /// (ties go to the lower id). This spreads entry points across the data,
    /// so every cluster has a nearby place for a search to start.
    fn select_entry_points(&self, state: &IndexState) -> Result<Vec<u32>, EngineError> {
        let n = state.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let target = self.entry_point_target(n);
        let metric = self.config.metric;
        let medoid = self.select_medoid(state)?;
        let mut entries = vec![medoid];
        let mut nearest: Vec<f32> = Vec::with_capacity(n as usize);
        {
            let base = state.store.read_vector(medoid)?;
            for id in 0..n {
                nearest.push(metric.calculate(base, state.store.read_vector(id)?));
            }
        }
        while entries.len() < target {
            let mut best: Option<(u32, f32)> = None;
            for (id, &d) in nearest.iter().enumerate() {
                if best.is_none_or(|(_, bd)| d > bd) {
                    best = Some((id as u32, d));
                }
            }
            let Some((next, dist)) = best else { break };
            if dist <= 0.0 {
                break; // every remaining vector duplicates an entry point
            }
            entries.push(next);
            let v = state.store.read_vector(next)?;
            for (id, slot) in nearest.iter_mut().enumerate() {
                let d = metric.calculate(v, state.store.read_vector(id as u32)?);
                if d < *slot {
                    *slot = d;
                }
            }
        }
        Ok(entries)
    }

    // ------------------------------------------------------------------
    // Ingestion
    // ------------------------------------------------------------------

    /// Ingests records from `source` (see [`crate::ingest`] for semantics).
    /// Never panics on bad input; the returned summary says whether the run
    /// succeeded.
    pub fn ingest<I>(
        &self,
        source: I,
        options: &IngestOptions,
        progress: &IngestProgress,
    ) -> IngestSummary
    where
        I: IntoIterator<Item = SourceItem>,
    {
        self.active_ingestions.fetch_add(1, Ordering::SeqCst);
        let _guard = IngestionGuard(&self.active_ingestions);
        let mut summary = IngestSummary::default();
        let checkpoint_every = options.checkpoint_every.max(1);

        let mut state = match self.write_state() {
            Ok(s) => s,
            Err(e) => {
                summary.fail(client_safe(&e));
                return summary;
            }
        };
        let started = Instant::now();
        let mode = if state.len() == 0 {
            IngestMode::InitialBuild
        } else {
            IngestMode::Incremental
        };
        tracing::info!(
            collection = %self.name,
            mode = ?mode,
            on_error = ?options.on_error,
            existing_vectors = state.len(),
            "ingestion started"
        );
        summary.mode = mode;
        if mode == IngestMode::InitialBuild {
            self.initial_build(&mut state, source, options.on_error, progress, &mut summary);
        } else {
            drop(state);
            self.incremental_ingest(
                source,
                options.on_error,
                checkpoint_every,
                progress,
                &mut summary,
            );
        }
        let secs = started.elapsed().as_secs_f64();
        let applied = summary.inserted + summary.updated;
        tracing::info!(
            collection = %self.name,
            mode = ?mode,
            success = summary.is_success(),
            records_read = summary.records_read,
            inserted = summary.inserted,
            updated = summary.updated,
            rejected = summary.rejected,
            elapsed_secs = secs,
            records_per_sec = if secs > 0.0 { applied as f64 / secs } else { 0.0 },
            "ingestion finished"
        );
        summary
    }

    /// Validates one source item. Returns the record if it should be applied,
    /// `None` if it was rejected and ingestion continues, and `Err(())` if
    /// ingestion must stop (the summary already holds the reason).
    fn admit(
        &self,
        item: SourceItem,
        on_error: OnError,
        progress: &IngestProgress,
        summary: &mut IngestSummary,
    ) -> Result<Option<(u64, crate::ingest::IngestRecord)>, ()> {
        match item {
            SourceItem::Fatal { line, reason } => {
                summary.fail(format!("line {line}: {reason}"));
                Err(())
            }
            SourceItem::Invalid { line, reason } => {
                summary.records_read += 1;
                progress.records_read.fetch_add(1, Ordering::Relaxed);
                progress.current_line.store(line, Ordering::Relaxed);
                self.reject_or_stop(line, reason, on_error, progress, summary)
            }
            SourceItem::Record { line, record } => {
                summary.records_read += 1;
                progress.records_read.fetch_add(1, Ordering::Relaxed);
                progress.current_line.store(line, Ordering::Relaxed);
                let check = validate_external_id(&record.id)
                    .and_then(|_| self.validate_vector(&record.vector, "vector"));
                match check {
                    Ok(()) => Ok(Some((line, record))),
                    Err(e) => self.reject_or_stop(line, e.to_string(), on_error, progress, summary),
                }
            }
        }
    }

    fn reject_or_stop(
        &self,
        line: u64,
        reason: String,
        on_error: OnError,
        progress: &IngestProgress,
        summary: &mut IngestSummary,
    ) -> Result<Option<(u64, crate::ingest::IngestRecord)>, ()> {
        tracing::debug!(collection = %self.name, line, reason = %reason, "record rejected");
        summary.reject(progress, line, reason.clone());
        match on_error {
            OnError::Skip => Ok(None),
            OnError::FailFast => {
                summary.fail(format!("line {line}: {reason}"));
                Err(())
            }
        }
    }

    fn initial_build<I>(
        &self,
        state: &mut IndexState,
        source: I,
        on_error: OnError,
        progress: &IngestProgress,
        summary: &mut IngestSummary,
    ) where
        I: IntoIterator<Item = SourceItem>,
    {
        // Phase 1: validate and store vectors (not yet linked).
        for item in source {
            if self.is_deleted() {
                summary.fail("collection was deleted during ingestion".into());
                return;
            }
            let (line, record) = match self.admit(item, on_error, progress, summary) {
                Ok(Some(x)) => x,
                Ok(None) => continue,
                Err(()) => break,
            };
            let result = if let Some(&id) = state.id_map.get(&record.id) {
                state
                    .store
                    .write_vector(id, &record.vector)
                    .map(|_| false)
                    .map_err(EngineError::from)
            } else {
                let id = state.len();
                self.ensure_capacity(state, id as u64 + 1)
                    .and_then(|_| {
                        state
                            .store
                            .write_vector(id, &record.vector)
                            .map_err(EngineError::from)
                    })
                    .and_then(|_| state.store.write_edges(id, &[]).map_err(EngineError::from))
                    .map(|_| {
                        state.id_map.insert(record.id.clone(), id);
                        state.external_ids.push(record.id);
                        true
                    })
            };
            match result {
                Ok(true) => {
                    summary.inserted += 1;
                    progress.inserted.fetch_add(1, Ordering::Relaxed);
                }
                Ok(false) => {
                    summary.updated += 1;
                    progress.updated.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    tracing::error!(collection = %self.name, line, error = %e, "bulk build write failed");
                    summary.fail(format!("line {line}: {}", client_safe(&e)));
                    break;
                }
            }
            if summary.records_read % BULK_PROGRESS_EVERY == 0 {
                tracing::info!(
                    collection = %self.name,
                    records_read = summary.records_read,
                    vectors = state.len(),
                    "bulk build: loading vectors"
                );
            }
        }

        // Phase 2: choose the entry point and link every stored node, even
        // after an early stop, so that every published id is indexed.
        tracing::info!(
            collection = %self.name,
            vectors = state.len(),
            rejected = summary.rejected,
            "bulk build: vectors loaded, linking graph"
        );
        let linked = if state.len() > 0 {
            self.link_all(state, progress)
        } else {
            Ok(())
        };
        if let Err(e) = linked {
            tracing::error!(collection = %self.name, error = %e, "bulk graph build failed");
            summary.fail(client_safe(&e));
        }
        if let Err(e) = self.persist_locked(state) {
            tracing::error!(collection = %self.name, error = %e, "persisting bulk build failed");
            summary.fail(client_safe(&e));
        }
    }

    fn link_all(
        &self,
        state: &mut IndexState,
        progress: &IngestProgress,
    ) -> Result<(), EngineError> {
        let n = state.len();
        let entries = self.select_entry_points(state)?;
        let mut is_entry = vec![false; n as usize];
        for &e in &entries {
            state.store.write_edges(e, &[])?;
            is_entry[e as usize] = true;
        }
        state.medoid = entries.first().copied();
        progress
            .nodes_linked
            .store(entries.len() as u64, Ordering::Relaxed);
        let started = Instant::now();
        let report_every = (n as u64 / 10).max(BULK_PROGRESS_EVERY);
        tracing::info!(
            collection = %self.name,
            entry_point = ?state.medoid,
            entry_points = entries.len(),
            nodes = n,
            "bulk build: entry points selected"
        );
        state.entry_points = entries;
        let mut buf = vec![0.0f32; self.config.dim];
        for id in 0..n {
            if is_entry[id as usize] {
                continue;
            }
            if self.is_deleted() {
                return Err(EngineError::Cancelled(
                    "collection was deleted during ingestion".into(),
                ));
            }
            buf.copy_from_slice(state.store.read_vector(id)?);
            self.link_node(state, id, &buf)?;
            let linked = progress.nodes_linked.fetch_add(1, Ordering::Relaxed) + 1;
            if linked % report_every == 0 {
                tracing::info!(
                    collection = %self.name,
                    linked,
                    total = n,
                    percent = linked as f64 * 100.0 / n as f64,
                    elapsed_secs = started.elapsed().as_secs_f64(),
                    "bulk build: linking"
                );
            }
        }
        tracing::info!(
            collection = %self.name,
            nodes = n,
            elapsed_secs = started.elapsed().as_secs_f64(),
            "bulk build: graph linked"
        );
        Ok(())
    }

    fn incremental_ingest<I>(
        &self,
        source: I,
        on_error: OnError,
        checkpoint_every: usize,
        progress: &IngestProgress,
        summary: &mut IngestSummary,
    ) where
        I: IntoIterator<Item = SourceItem>,
    {
        let mut dirty = 0usize;
        for item in source {
            let (line, record) = match self.admit(item, on_error, progress, summary) {
                Ok(Some(x)) => x,
                Ok(None) => continue,
                Err(()) => break,
            };
            let mut state = match self.write_state() {
                Ok(s) => s,
                Err(e) => {
                    summary.fail(client_safe(&e));
                    return;
                }
            };
            if self.is_deleted() {
                summary.fail("collection was deleted during ingestion".into());
                return;
            }
            match self.insert_locked(&mut state, &record.id, &record.vector) {
                Ok(o) if o.created => {
                    summary.inserted += 1;
                    progress.inserted.fetch_add(1, Ordering::Relaxed);
                }
                Ok(_) => {
                    summary.updated += 1;
                    progress.updated.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    tracing::error!(collection = %self.name, line, error = %e, "incremental insert failed");
                    summary.fail(format!("line {line}: {}", client_safe(&e)));
                    break;
                }
            }
            dirty += 1;
            if dirty >= checkpoint_every {
                if let Err(e) = self.persist_locked(&state) {
                    summary.fail(client_safe(&e));
                    return;
                }
                tracing::debug!(collection = %self.name, vectors = state.len(), "ingestion checkpoint persisted");
                dirty = 0;
            }
            let applied = summary.inserted + summary.updated;
            if applied % INCREMENTAL_PROGRESS_EVERY == 0 {
                tracing::info!(
                    collection = %self.name,
                    applied,
                    rejected = summary.rejected,
                    current_line = line,
                    "incremental ingestion progress"
                );
            }
        }
        // New vectors may have opened up regions without a nearby entry point:
        // re-select the set (legacy single-entry collections are left as is).
        let refresh = summary.inserted > 0 && self.config.effective_entry_points() > 1;
        if dirty > 0 || refresh {
            match self.write_state() {
                Ok(mut state) => {
                    if refresh {
                        match self.select_entry_points(&state) {
                            Ok(entries) => {
                                state.medoid = entries.first().copied();
                                tracing::info!(
                                    collection = %self.name,
                                    entry_points = entries.len(),
                                    "entry points refreshed"
                                );
                                state.entry_points = entries;
                            }
                            Err(e) => summary.fail(client_safe(&e)),
                        }
                    }
                    if let Err(e) = self.persist_locked(&state) {
                        summary.fail(client_safe(&e));
                    }
                }
                Err(e) => summary.fail(client_safe(&e)),
            }
        }
    }
}

/// Client-safe text for an error raised during ingestion.
fn client_safe(e: &EngineError) -> String {
    if e.is_client_error() || matches!(e, EngineError::Cancelled(_)) {
        e.to_string()
    } else {
        "internal storage error (see server logs)".into()
    }
}

fn resolve_external_ids(meta: &MetadataFile) -> Result<Vec<String>, EngineError> {
    let count = meta.next_id as usize;
    if let Some(ids) = &meta.external_ids {
        if ids.len() != count {
            return Err(EngineError::Corrupt(format!(
                "metadata lists {} external ids but next_id is {count}",
                ids.len()
            )));
        }
        return Ok(ids.clone());
    }
    // Format 0: invert id_map; ids must be dense in 0..next_id.
    let map = meta.id_map.clone().unwrap_or_default();
    let mut slots: Vec<Option<String>> = vec![None; count];
    for (ext, id) in map {
        let slot = slots.get_mut(id as usize).ok_or_else(|| {
            EngineError::Corrupt(format!("id_map entry {id} is >= next_id {count}"))
        })?;
        if slot.is_some() {
            return Err(EngineError::Corrupt(format!(
                "internal id {id} is mapped twice"
            )));
        }
        *slot = Some(ext);
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            s.ok_or_else(|| EngineError::Corrupt(format!("internal id {i} has no external id")))
        })
        .collect()
}

/// Borrowed view used to write format-1 metadata without cloning the ids.
#[derive(Serialize)]
struct MetadataOut<'a> {
    format_version: u32,
    medoid_id: Option<u32>,
    entry_points: &'a [u32],
    next_id: u32,
    capacity: u32,
    external_ids: &'a [String],
}

fn write_metadata(dir: &Path, state: &IndexState) -> Result<(), EngineError> {
    let meta = MetadataOut {
        format_version: FORMAT_VERSION,
        medoid_id: state.medoid,
        entry_points: &state.entry_points,
        next_id: state.len(),
        capacity: state.store.capacity(),
        external_ids: &state.external_ids,
    };
    let bytes = serde_json::to_vec(&meta)?;
    write_atomic(&dir.join(METADATA_FILE), &bytes)
}

/// Removes edges that target non-live ids, self-edges and duplicates from
/// every live node. Returns the number of edges removed.
fn sanitize_graph(state: &mut IndexState) -> Result<usize, EngineError> {
    let n = state.len();
    let mut removed = 0usize;
    for id in 0..n {
        let edges = state.store.read_edges(id)?;
        let mut clean: Vec<u32> = Vec::with_capacity(edges.len());
        for &e in &edges {
            if e < n && e != id && !clean.contains(&e) {
                clean.push(e);
            }
        }
        if clean.len() != edges.len() {
            removed += edges.len() - clean.len();
            state.store.write_edges(id, &clean)?;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_names() {
        assert!(validate_collection_name("bulk_db").is_ok());
        assert!(validate_collection_name("A-1_b").is_ok());
        let long = "x".repeat(65);
        for bad in [
            "",
            "..",
            "a/b",
            "a\\b",
            "a b",
            "naïve",
            ".hidden",
            long.as_str(),
        ] {
            assert!(validate_collection_name(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn config_validation() {
        let ok = CollectionConfig::new(3, DistanceMetric::L2, 16, 1.2);
        assert!(ok.validate().is_ok());
        assert_eq!(ok.l_build, Some(64));
        let mut c = ok.clone();
        c.dim = 0;
        assert!(c.validate().is_err());
        let mut c = ok.clone();
        c.r = 0;
        assert!(c.validate().is_err());
        let mut c = ok.clone();
        c.alpha = f32::NAN;
        assert!(c.validate().is_err());
        let mut c = ok.clone();
        c.alpha = 0.5;
        assert!(c.validate().is_err());
        let mut c = ok.clone();
        c.l_build = Some(0);
        assert!(c.validate().is_err());
        let mut c = ok;
        c.entry_points = Some(0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn legacy_config_without_l_build_uses_two_r() {
        let c: CollectionConfig =
            serde_json::from_str(r#"{"dim":3,"metric":"L2","r":3,"alpha":1.2}"#).unwrap();
        assert_eq!(c.l_build, None);
        assert_eq!(c.effective_l_build(), 6);
        assert_eq!(c.entry_points, None);
        assert_eq!(c.effective_entry_points(), 1);
    }

    #[test]
    fn query_options_defaults_and_limits() {
        let o = QueryOptions::new(5, None);
        assert_eq!(o.ef_search, DEFAULT_EF_SEARCH);
        assert!(o.validate().is_ok());
        let o = QueryOptions::new(100, None);
        assert_eq!(o.ef_search, 100);
        assert!(QueryOptions::new(0, Some(10)).validate().is_err());
        assert!(QueryOptions::new(10, Some(9)).validate().is_err());
        assert!(
            QueryOptions::new(MAX_K + 1, Some(MAX_EF_SEARCH))
                .validate()
                .is_err()
        );
        assert!(
            QueryOptions::new(10, Some(MAX_EF_SEARCH + 1))
                .validate()
                .is_err()
        );
        assert!(QueryOptions::new(10, Some(10)).validate().is_ok());
    }

    #[test]
    fn legacy_metadata_is_inverted_densely() {
        let meta: MetadataFile =
            serde_json::from_str(r#"{"medoid_id":0,"id_map":{"b":1,"a":0,"c":2},"next_id":3}"#)
                .unwrap();
        assert_eq!(meta.format_version, 0);
        assert_eq!(resolve_external_ids(&meta).unwrap(), vec!["a", "b", "c"]);

        let gap: MetadataFile =
            serde_json::from_str(r#"{"medoid_id":0,"id_map":{"a":0,"c":2},"next_id":3}"#).unwrap();
        assert!(resolve_external_ids(&gap).is_err());
        let beyond: MetadataFile =
            serde_json::from_str(r#"{"medoid_id":0,"id_map":{"a":0,"c":5},"next_id":2}"#).unwrap();
        assert!(resolve_external_ids(&beyond).is_err());
    }
}
