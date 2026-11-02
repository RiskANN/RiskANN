//! Liveness and readiness.

use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde_json::{Value, json};
use std::sync::Arc;

/// `GET /health` — cheap liveness probe; touches no collection data.
pub async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

/// `GET /ready` — 200 once startup loading has completed, 503 after
/// graceful shutdown has begun.
pub async fn ready(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    let ready = state.is_ready();
    let failed: Vec<&str> = state
        .load_report
        .failed
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    let body = json!({
        "ready": ready,
        "collections_loaded": state.db.collections.len(),
        "collections_failed": failed,
    });
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body))
}
