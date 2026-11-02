//! Collection lifecycle: create (optionally with a bulk JSONL file), inspect,
//! list, delete.

use crate::api::blocking;
use crate::api::ingest::spawn_ingest;
use crate::error::ApiError;
use crate::state::AppState;
use axum::Json;
use axum::extract::rejection::{JsonRejection, PathRejection};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use engine::DistanceMetric;
use engine::collection::{CollectionConfig, default_l_build};
use engine::ingest::{IngestOptions, OnError, resolve_ingest_path};
use engine::limits::DEFAULT_ENTRY_POINTS;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

pub const DEFAULT_INITIAL_CAPACITY: u32 = 1_000;

fn default_capacity() -> u32 {
    DEFAULT_INITIAL_CAPACITY
}

/// `PUT /collections/{name}` body. Fields `dim`, `metric`, `r`, `alpha`,
/// `initial_capacity` and `bulk_file` keep their original meaning.
#[derive(Debug, Deserialize)]
pub struct CreateCollectionReq {
    pub dim: usize,
    pub metric: DistanceMetric,
    pub r: usize,
    pub alpha: f32,
    /// Initial number of slots; storage grows automatically afterwards.
    #[serde(default = "default_capacity")]
    pub initial_capacity: u32,
    /// Construction beam width; defaults to `max(2r, 64)`.
    #[serde(default)]
    pub l_build: Option<usize>,
    /// Maximum number of search entry points; defaults to 64.
    #[serde(default)]
    pub entry_points: Option<usize>,
    /// Optional server-side JSONL file ingested by a background job.
    #[serde(default)]
    pub bulk_file: Option<String>,
    /// Error policy for `bulk_file`: `fail_fast` (default) or `skip`.
    #[serde(default)]
    pub on_error: OnError,
}

pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    payload: Result<Json<CreateCollectionReq>, JsonRejection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let Path(name) = path?;
    let Json(req) = payload?;
    let config = CollectionConfig {
        dim: req.dim,
        metric: req.metric,
        r: req.r,
        alpha: req.alpha,
        l_build: Some(req.l_build.unwrap_or_else(|| default_l_build(req.r))),
        entry_points: Some(req.entry_points.unwrap_or(DEFAULT_ENTRY_POINTS)),
    };
    // Validate the bulk path *before* creating anything, so a rejected path
    // never leaves an empty collection behind.
    let bulk_path = match &req.bulk_file {
        Some(p) => Some(resolve_ingest_path(p, state.config.ingest_root.as_deref())?),
        None => None,
    };

    let st = Arc::clone(&state);
    let n = name.clone();
    let capacity = req.initial_capacity;
    let collection = blocking(move || st.db.create_collection(&n, config, capacity)).await?;
    let info = {
        let c = Arc::clone(&collection);
        blocking(move || c.info()).await?
    };

    match bulk_path {
        Some(p) => {
            let options = IngestOptions {
                on_error: req.on_error,
                ..IngestOptions::default()
            };
            let job = spawn_ingest(&state, collection, p, options);
            Ok((
                StatusCode::ACCEPTED,
                Json(json!({
                    "collection": info,
                    "job_id": job.id,
                    "status_url": format!("/collections/{name}/ingest/{}", job.id),
                })),
            ))
        }
        None => Ok((StatusCode::CREATED, Json(json!({"collection": info})))),
    }
}

/// `GET /collections/{name}`
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
) -> Result<Json<Value>, ApiError> {
    let Path(name) = path?;
    let c = state.db.get(&name)?;
    let info = blocking(move || c.info()).await?;
    Ok(Json(json!({"collection": info})))
}

/// `GET /collections`
pub async fn list_collections(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let st = Arc::clone(&state);
    let all = blocking(move || st.db.list()).await?;
    Ok(Json(json!({"collections": all})))
}

/// `DELETE /collections/{name}`
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
) -> Result<Json<Value>, ApiError> {
    let Path(name) = path?;
    let st = Arc::clone(&state);
    let n = name.clone();
    blocking(move || st.db.delete_collection(&n)).await?;
    Ok(Json(json!({"deleted": name})))
}
