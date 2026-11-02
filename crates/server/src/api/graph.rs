//! `GET /collections/{name}/graph?limit=N`: nodes, vectors and edges for
//! visualisation (default 1000 nodes, at most 5000).

use crate::api::blocking;
use crate::error::ApiError;
use crate::state::AppState;
use axum::Json;
use axum::extract::rejection::{PathRejection, QueryRejection};
use axum::extract::{Path, Query, State};
use engine::collection::GraphSnapshot;
use engine::limits::DEFAULT_GRAPH_SNAPSHOT;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

pub async fn graph_snapshot(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    query: Result<Query<GraphQuery>, QueryRejection>,
) -> Result<Json<GraphSnapshot>, ApiError> {
    let Path(name) = path?;
    let Query(q) = query?;
    let c = state.db.get(&name)?;
    let limit = q.limit.unwrap_or(DEFAULT_GRAPH_SNAPSHOT);
    let snapshot = blocking(move || c.graph_snapshot(limit)).await?;
    Ok(Json(snapshot))
}
