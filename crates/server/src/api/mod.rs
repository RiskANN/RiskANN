//! HTTP handlers. Every handler that touches collection data runs the
//! (synchronous) engine call on Tokio's blocking pool, so mmap I/O, graph
//! traversal and fsync never stall the async runtime.

pub mod collections;
pub mod graph;
pub mod health;
pub mod ingest;
pub mod search;
pub mod ui;
pub mod upsert;
pub mod vectors;

use crate::error::ApiError;
use engine::EngineError;

/// Runs `f` on the blocking pool and maps engine errors to API errors.
pub(crate) async fn blocking<T, F>(f: F) -> Result<T, ApiError>
where
    F: FnOnce() -> Result<T, EngineError> + Send + 'static,
    T: Send + 'static,
{
    // Carry the request span onto the blocking thread so engine logs are
    // tagged with the request id.
    let span = tracing::Span::current();
    match tokio::task::spawn_blocking(move || {
        let _entered = span.enter();
        f()
    })
    .await
    {
        Ok(result) => result.map_err(ApiError::from),
        Err(e) => {
            tracing::error!(error = %e, "blocking task panicked or was cancelled");
            Err(ApiError::internal())
        }
    }
}

/// Fallback for unknown routes: same JSON envelope as every other error.
pub async fn not_found() -> ApiError {
    ApiError::not_found("ROUTE_NOT_FOUND", "no route matches this path")
}
