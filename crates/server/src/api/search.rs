//! Nearest-neighbour search.
//!
//! * `POST /collections/{name}/search` — approximate (Vamana graph) search.
//!   This is the normal retrieval path; it never falls back to brute force.
//! * `POST /collections/{name}/search/exact` — explicitly named exact
//!   brute-force search, intended for diagnostics and ground truth. Its cost
//!   grows linearly with the collection size.
//!
//! Unknown request fields (for example `filter`) are rejected with 400
//! instead of being silently ignored: metadata filtering is not supported.

use crate::api::blocking;
use crate::error::ApiError;
use crate::state::AppState;
use axum::Json;
use axum::extract::rejection::{JsonRejection, PathRejection};
use axum::extract::{Path, State};
use engine::collection::{QueryOptions, SearchResponse, SearchTrace};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const DEFAULT_K: usize = 10;

fn default_k() -> usize {
    DEFAULT_K
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchReq {
    pub vector: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
    /// Exploration budget; defaults to `max(k, 64)`. Must be `>= k`.
    #[serde(default)]
    pub ef_search: Option<usize>,
    #[serde(default)]
    pub include_vectors: bool,
    #[serde(default = "default_true")]
    pub include_distances: bool,
    /// Return the entry point and the expansion order of the beam search.
    #[serde(default)]
    pub trace: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSearchReq {
    pub vector: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub include_vectors: bool,
    #[serde(default = "default_true")]
    pub include_distances: bool,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub id: String,
    pub internal_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

/// `search_duration_us` is measured inside the engine (validation, lock,
/// traversal, result assembly); it excludes HTTP parsing and serialisation.
#[derive(Debug, Serialize)]
pub struct SearchStatsBody {
    pub visited_nodes: usize,
    pub expanded_nodes: usize,
    pub distance_computations: usize,
    pub returned: usize,
    pub search_duration_us: u64,
}

#[derive(Debug, Serialize)]
pub struct SearchResponseBody {
    pub collection: String,
    pub metric: &'static str,
    /// `"approximate"` or `"exact"`.
    pub mode: &'static str,
    pub k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<usize>,
    pub results: Vec<SearchResultItem>,
    pub stats: SearchStatsBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<SearchTrace>,
}

struct BodyParts {
    collection: String,
    metric: &'static str,
    mode: &'static str,
    k: usize,
    ef_search: Option<usize>,
    include_distances: bool,
}

/// Logs queries whose engine time exceeds `RISKANN_SLOW_QUERY_MS`.
fn warn_if_slow(state: &AppState, p: &BodyParts, resp: &SearchResponse) {
    let threshold_ms = state.config.slow_query_ms;
    if threshold_ms > 0 && resp.stats.duration_us >= threshold_ms.saturating_mul(1000) {
        tracing::warn!(
            collection = %p.collection,
            mode = p.mode,
            k = p.k,
            ef_search = ?p.ef_search,
            visited = resp.stats.visited_nodes,
            distance_computations = resp.stats.distance_computations,
            duration_us = resp.stats.duration_us,
            threshold_ms,
            "slow query"
        );
    }
}

fn to_body(p: BodyParts, resp: SearchResponse) -> SearchResponseBody {
    let results = resp
        .hits
        .into_iter()
        .map(|h| SearchResultItem {
            id: h.external_id,
            internal_id: h.internal_id,
            distance: if p.include_distances {
                Some(h.distance)
            } else {
                None
            },
            vector: h.vector,
        })
        .collect();
    SearchResponseBody {
        collection: p.collection,
        metric: p.metric,
        mode: p.mode,
        k: p.k,
        ef_search: p.ef_search,
        results,
        stats: SearchStatsBody {
            visited_nodes: resp.stats.visited_nodes,
            expanded_nodes: resp.stats.expanded_nodes,
            distance_computations: resp.stats.distance_computations,
            returned: resp.stats.returned,
            search_duration_us: resp.stats.duration_us,
        },
        trace: resp.trace,
    }
}

pub async fn search(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    payload: Result<Json<SearchReq>, JsonRejection>,
) -> Result<Json<SearchResponseBody>, ApiError> {
    let Path(name) = path?;
    let Json(req) = payload?;
    let c = state.db.get(&name)?;
    let metric = c.config().metric.name();
    let mut options = QueryOptions::new(req.k, req.ef_search);
    options.include_vectors = req.include_vectors;
    options.trace = req.trace;
    let parts = BodyParts {
        collection: name,
        metric,
        mode: "approximate",
        k: req.k,
        ef_search: Some(options.ef_search),
        include_distances: req.include_distances,
    };
    let vector = req.vector;
    let resp = blocking(move || c.search(&vector, &options)).await?;
    warn_if_slow(&state, &parts, &resp);
    Ok(Json(to_body(parts, resp)))
}

pub async fn exact_search(
    State(state): State<Arc<AppState>>,
    path: Result<Path<String>, PathRejection>,
    payload: Result<Json<ExactSearchReq>, JsonRejection>,
) -> Result<Json<SearchResponseBody>, ApiError> {
    let Path(name) = path?;
    let Json(req) = payload?;
    let c = state.db.get(&name)?;
    let metric = c.config().metric.name();
    let parts = BodyParts {
        collection: name,
        metric,
        mode: "exact",
        k: req.k,
        ef_search: None,
        include_distances: req.include_distances,
    };
    let (vector, k, include_vectors) = (req.vector, req.k, req.include_vectors);
    let resp = blocking(move || c.exact_search(&vector, k, include_vectors)).await?;
    warn_if_slow(&state, &parts, &resp);
    Ok(Json(to_body(parts, resp)))
}
