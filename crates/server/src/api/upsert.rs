//! `POST /collections/{name}/upsert`

use crate::api::blocking;
use crate::error::ApiError;
use crate::state::AppState;
use axum::Json;
use axum::extract::rejection::{JsonRejection, PathRejection};
use axum::extract::{Path, State};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct UpsertReq {
    pub id: String,
    pub vector: Vec<f32>,
}

/// Inserts a new vector or replaces the vector of an existing external id
/// (the internal id is kept). Returns once the vector is linked into the
/// graph and metadata is persisted.
pub async fn upsert_vector(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    payload: Result<Json<UpsertReq>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Path(name) = path?;
    let Json(req) = payload?;
    let c = state.db.get(&name)?;
    let UpsertReq { id, vector } = req;
    let ext = id.clone();
    let outcome = blocking(move || c.upsert(&ext, &vector)).await?;
    Ok(Json(json!({
        "id": id,
        "internal_id": outcome.internal_id,
        "created": outcome.created,
    })))
}
