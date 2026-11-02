//! Uniform JSON error envelope:
//! `{"error": {"code": "SOME_CODE", "message": "human readable"}}`.
//!
//! Internal errors are logged with full detail and returned as a generic
//! `INTERNAL_ERROR` without file paths or internals.

use axum::Json;
use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use engine::EngineError;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub fn internal() -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            "internal server error",
        )
    }

    pub fn not_found(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, code, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_client_error() {
            // The request log records the code at `warn`; the message is
            // only logged at `debug` because it may echo request content.
            tracing::debug!(code = self.code, message = %self.message, "client error detail");
        }
        let body = json!({"error": {"code": self.code, "message": self.message}});
        let mut response = (self.status, Json(body)).into_response();
        response
            .extensions_mut()
            .insert(crate::logging::ErrorCode(self.code));
        response
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        use EngineError as E;
        let msg = e.to_string();
        match e {
            E::InvalidCollectionName(_) => {
                Self::new(StatusCode::BAD_REQUEST, "INVALID_COLLECTION_NAME", msg)
            }
            E::CollectionNotFound(_) => {
                Self::new(StatusCode::NOT_FOUND, "COLLECTION_NOT_FOUND", msg)
            }
            E::CollectionExists(_) => Self::new(StatusCode::CONFLICT, "COLLECTION_EXISTS", msg),
            E::InvalidArgument { .. } => {
                Self::new(StatusCode::BAD_REQUEST, "INVALID_ARGUMENT", msg)
            }
            E::DimensionMismatch { .. } => {
                Self::new(StatusCode::BAD_REQUEST, "DIMENSION_MISMATCH", msg)
            }
            E::VectorNotFound(_) => Self::new(StatusCode::NOT_FOUND, "VECTOR_NOT_FOUND", msg),
            E::CapacityExceeded(_) => Self::new(StatusCode::CONFLICT, "CAPACITY_EXCEEDED", msg),
            E::IngestPathNotAllowed => {
                Self::new(StatusCode::FORBIDDEN, "INGEST_PATH_NOT_ALLOWED", msg)
            }
            E::IngestFileUnavailable(_) => {
                Self::new(StatusCode::BAD_REQUEST, "INGEST_FILE_UNAVAILABLE", msg)
            }
            E::Cancelled(_) => Self::new(StatusCode::CONFLICT, "COLLECTION_DELETED", msg),
            E::Corrupt(_) | E::Storage(_) | E::Io { .. } | E::Internal(_) => {
                tracing::error!(error = %msg, "internal error while handling request");
                Self::internal()
            }
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(r: JsonRejection) -> Self {
        let status = r.status();
        if status == StatusCode::PAYLOAD_TOO_LARGE {
            Self::new(
                status,
                "PAYLOAD_TOO_LARGE",
                "request body exceeds the configured size limit",
            )
        } else if status == StatusCode::UNSUPPORTED_MEDIA_TYPE {
            Self::new(
                status,
                "UNSUPPORTED_MEDIA_TYPE",
                "expected request with `Content-Type: application/json`",
            )
        } else {
            Self::new(StatusCode::BAD_REQUEST, "MALFORMED_REQUEST", r.body_text())
        }
    }
}

impl From<QueryRejection> for ApiError {
    fn from(r: QueryRejection) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "MALFORMED_REQUEST", r.body_text())
    }
}

impl From<PathRejection> for ApiError {
    fn from(r: PathRejection) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "MALFORMED_REQUEST", r.body_text())
    }
}
