//! RiskANN HTTP API.
//!
//! [`build_router`] assembles every route so that tests can drive the API
//! in-process (no socket) and `main.rs` only adds configuration, logging and
//! graceful shutdown.

pub mod api;
pub mod error;
pub mod logging;
pub mod state;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post, put};
use std::sync::Arc;

pub use error::ApiError;
pub use state::{AppState, ServerConfig};

pub fn build_router(state: Arc<AppState>) -> Router {
    let body_limit = state.config.max_body_bytes;
    Router::new()
        .route("/", get(api::ui::console))
        .route("/ui", get(api::ui::console))
        .route("/health", get(api::health::health))
        .route("/ready", get(api::health::ready))
        .route("/collections", get(api::collections::list_collections))
        .route(
            "/collections/:name",
            put(api::collections::create_collection)
                .get(api::collections::get_collection)
                .delete(api::collections::delete_collection),
        )
        .route(
            "/collections/:name/upsert",
            post(api::upsert::upsert_vector),
        )
        .route("/collections/:name/search", post(api::search::search))
        .route(
            "/collections/:name/search/exact",
            post(api::search::exact_search),
        )
        .route(
            "/collections/:name/vectors/:id",
            get(api::vectors::get_vector),
        )
        .route("/collections/:name/graph", get(api::graph::graph_snapshot))
        .route("/collections/:name/ingest", post(api::ingest::start_ingest))
        .route(
            "/collections/:name/ingest/:job_id",
            get(api::ingest::job_status),
        )
        .fallback(api::not_found)
        .layer(DefaultBodyLimit::max(body_limit))
        // Outermost layer: every request, including rejected and unknown
        // ones, gets a request id and exactly one log line.
        .layer(axum::middleware::from_fn(logging::log_requests))
        .with_state(state)
}
