//! `GET /collections/{name}/vectors/{id}`

use crate::api::blocking;
use crate::error::ApiError;
use crate::state::AppState;
use axum::Json;
use axum::extract::rejection::PathRejection;
use axum::extract::{Path, State};
use engine::VectorRecord;
use std::sync::Arc;

pub async fn get_vector(
    State(state): State<Arc<AppState>>,
    path: Result<Path<(String, String)>, PathRejection>,
) -> Result<Json<VectorRecord>, ApiError> {
    let Path((name, id)) = path?;
    let c = state.db.get(&name)?;
    let record = blocking(move || c.get_vector(&id)).await?;
    Ok(Json(record))
}
