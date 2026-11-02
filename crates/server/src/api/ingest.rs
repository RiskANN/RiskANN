//! Ingestion jobs.
//!
//! * `POST /collections/{name}/ingest` — start a JSONL ingestion job from a
//!   server-side file; returns `202 Accepted` with a `job_id`.
//! * `GET /collections/{name}/ingest/{job_id}` — job state and counters.
//!
//! Trust boundary: the path is read by the server process with its own
//! permissions. Set `RISKANN_INGEST_DIR` to confine ingestion to one
//! directory. Job status is kept in memory and lost on restart.

use crate::error::ApiError;
use crate::state::AppState;
use axum::Json;
use axum::extract::rejection::{JsonRejection, PathRejection};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use engine::collection::{Collection, validate_collection_name};
use engine::ingest::{
    DEFAULT_CHECKPOINT_EVERY, IngestJob, IngestOptions, JobSnapshot, OnError, resolve_ingest_path,
    run_ingest_job,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestReq {
    /// Absolute path of a JSONL file on the server.
    pub path: String,
    #[serde(default)]
    pub on_error: OnError,
    /// Incremental mode: persist metadata every N records (default 10000).
    #[serde(default)]
    pub checkpoint_every: Option<usize>,
}

/// Registers a job and runs it on the blocking pool. A watcher task marks
/// the job failed if the worker panics, so a job never stays `running`.
pub(crate) fn spawn_ingest(
    state: &Arc<AppState>,
    collection: Arc<Collection>,
    path: PathBuf,
    options: IngestOptions,
) -> Arc<IngestJob> {
    let job = state.db.jobs.create(collection.name(), options.on_error);
    let worker_job = Arc::clone(&job);
    let handle =
        tokio::task::spawn_blocking(move || run_ingest_job(collection, worker_job, path, options));
    let watch_job = Arc::clone(&job);
    tokio::spawn(async move {
        if let Err(e) = handle.await {
            tracing::error!(job = %watch_job.id, error = %e, "ingestion worker terminated abnormally");
            watch_job.mark_failed("ingestion task terminated unexpectedly");
        }
    });
    job
}

pub async fn start_ingest(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    payload: Result<Json<IngestReq>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let Path(name) = path?;
    let Json(req) = payload?;
    let collection = state.db.get(&name)?;
    if req.checkpoint_every == Some(0) {
        return Err(engine::EngineError::invalid("checkpoint_every", "must be at least 1").into());
    }
    let file = resolve_ingest_path(&req.path, state.config.ingest_root.as_deref())?;
    let options = IngestOptions {
        on_error: req.on_error,
        checkpoint_every: req.checkpoint_every.unwrap_or(DEFAULT_CHECKPOINT_EVERY),
    };
    let job = spawn_ingest(&state, collection, file, options);
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "job_id": job.id,
            "collection": name,
            "state": job.state(),
            "status_url": format!("/collections/{name}/ingest/{}", job.id),
        })),
    ))
}

pub async fn job_status(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let Path((name, job_id)) = path?;
    validate_collection_name(&name)?;
    let job = state.db.jobs.get(&name, &job_id).ok_or_else(|| {
        ApiError::not_found(
            "JOB_NOT_FOUND",
            format!("ingestion job '{job_id}' not found"),
        )
    })?;
    Ok(Json(job.snapshot()))
}
