//! Logging and request tracing.
//!
//! * [`init`] installs a `tracing` subscriber: human-readable text (default)
//!   or one JSON object per line (`RISKANN_LOG_FORMAT=json`). Verbosity is
//!   controlled by `RUST_LOG` (default `info`).
//! * [`log_requests`] is an Axum middleware that assigns every request an id
//!   (reusing a valid incoming `x-request-id`), runs the request inside a
//!   `request` span, logs one line per request with status, error code and
//!   latency, and returns the id in the `x-request-id` response header.
//! * [`install_panic_hook`] routes panics through the log as well.
//!
//! Request and response bodies are never logged, so vectors and ids from
//! clients do not end up in log files.

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::Response;
use std::io::IsTerminal;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::Instrument;

pub const REQUEST_ID_HEADER: &str = "x-request-id";
pub const DEFAULT_LOG_FILTER: &str = "info";
const MAX_REQUEST_ID_LEN: usize = 64;

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Machine-readable error code attached to error responses so the request
/// log can report it (see `ApiError::into_response`).
#[derive(Debug, Clone, Copy)]
pub struct ErrorCode(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable lines; colours only when stdout is a terminal.
    Text,
    /// One JSON object per line, for log shippers.
    Json,
}

impl LogFormat {
    pub fn name(&self) -> &'static str {
        match self {
            LogFormat::Text => "text",
            LogFormat::Json => "json",
        }
    }

    /// Reads `RISKANN_LOG_FORMAT` (`text` or `json`, default `text`).
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("RISKANN_LOG_FORMAT") {
            Err(_) => Ok(LogFormat::Text),
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "" | "text" => Ok(LogFormat::Text),
                "json" => Ok(LogFormat::Json),
                other => Err(format!(
                    "RISKANN_LOG_FORMAT must be 'text' or 'json', got '{other}'"
                )),
            },
        }
    }
}

/// Installs the global subscriber. Call once, at startup.
pub fn init(format: LogFormat) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_LOG_FILTER));
    match format {
        LogFormat::Text => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_ansi(std::io::stdout().is_terminal())
            .init(),
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_current_span(true)
            .with_span_list(false)
            .init(),
    }
}

/// Logs panics (with location) through `tracing`, then runs the default hook.
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown".into());
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".into());
        tracing::error!(location = %location, panic = %payload, "panic");
        default_hook(info);
    }));
}

fn is_valid_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_REQUEST_ID_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// Request-logging middleware (see module docs).
///
/// Levels: `/health` and `/ready` log at `debug` (probes are frequent);
/// other requests log at `info` (2xx/3xx), `warn` (4xx) or `error` (5xx).
pub async fn log_requests(req: Request, next: Next) -> Response {
    let started = Instant::now();
    let request_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| is_valid_request_id(s))
        .map(str::to_owned)
        .unwrap_or_else(|| format!("req-{}", NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed)));
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let is_probe = path == "/health" || path == "/ready";

    let span =
        tracing::info_span!("request", request_id = %request_id, method = %method, path = %path);
    let mut response = next.run(req).instrument(span.clone()).await;

    let latency_ms = (started.elapsed().as_secs_f64() * 1e3 * 100.0).round() / 100.0;
    let status = response.status().as_u16();
    let error_code = response
        .extensions()
        .get::<ErrorCode>()
        .map(|c| c.0)
        .unwrap_or("-");
    span.in_scope(|| {
        if response.status().is_server_error() {
            tracing::error!(status, error_code, latency_ms, "request failed");
        } else if response.status().is_client_error() {
            tracing::warn!(status, error_code, latency_ms, "request rejected");
        } else if is_probe {
            tracing::debug!(status, latency_ms, "request completed");
        } else {
            tracing::info!(status, latency_ms, "request completed");
        }
    });

    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_validation() {
        assert!(is_valid_request_id("abc-123_x.y"));
        assert!(!is_valid_request_id(""));
        assert!(!is_valid_request_id("has space"));
        assert!(!is_valid_request_id("new\nline"));
        assert!(!is_valid_request_id(&"a".repeat(65)));
    }
}
