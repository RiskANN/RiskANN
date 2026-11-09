//! Engine error type.
//!
//! `Display` messages for client-caused errors are safe to return to API
//! clients. `Storage`, `Io`, `Corrupt` and `Internal` messages may contain
//! file paths or internal details: the server logs them and returns a generic
//! message instead.

use riskann_core::vamana::SearchError;
use std::fmt;
use std::io;
use storage::StorageError;

#[derive(Debug)]
pub enum EngineError {
    InvalidCollectionName(String),
    CollectionNotFound(String),
    CollectionExists(String),
    InvalidArgument {
        field: &'static str,
        message: String,
    },
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    VectorNotFound(String),
    CapacityExceeded(String),
    /// The ingestion path is outside the configured ingestion directory.
    IngestPathNotAllowed,
    /// The ingestion file does not exist or cannot be read (client-safe text).
    IngestFileUnavailable(String),
    /// The collection was deleted while an operation was running.
    Cancelled(String),
    Corrupt(String),
    Storage(StorageError),
    Io {
        context: String,
        source: io::Error,
    },
    Internal(String),
}

impl EngineError {
    pub fn invalid(field: &'static str, message: impl Into<String>) -> Self {
        EngineError::InvalidArgument {
            field,
            message: message.into(),
        }
    }

    pub fn io(context: impl Into<String>) -> impl FnOnce(io::Error) -> EngineError {
        let context = context.into();
        move |source| EngineError::Io { context, source }
    }

    /// True if the error was caused by the request rather than the server.
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            EngineError::InvalidCollectionName(_)
                | EngineError::CollectionNotFound(_)
                | EngineError::CollectionExists(_)
                | EngineError::InvalidArgument { .. }
                | EngineError::DimensionMismatch { .. }
                | EngineError::VectorNotFound(_)
                | EngineError::CapacityExceeded(_)
                | EngineError::IngestPathNotAllowed
                | EngineError::IngestFileUnavailable(_)
        )
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidCollectionName(m) => write!(f, "invalid collection name: {m}"),
            EngineError::CollectionNotFound(n) => write!(f, "collection '{n}' not found"),
            EngineError::CollectionExists(m) => write!(f, "{m}"),
            EngineError::InvalidArgument { field, message } => write!(f, "{field}: {message}"),
            EngineError::DimensionMismatch { expected, got } => write!(
                f,
                "vector has {got} components but the collection dimension is {expected}"
            ),
            EngineError::VectorNotFound(id) => write!(f, "vector with id '{id}' not found"),
            EngineError::CapacityExceeded(m) => write!(f, "capacity exceeded: {m}"),
            EngineError::IngestPathNotAllowed => {
                write!(
                    f,
                    "the bulk file path is outside the permitted ingestion directory"
                )
            }
            EngineError::IngestFileUnavailable(m) => write!(f, "{m}"),
            EngineError::Cancelled(m) => write!(f, "operation cancelled: {m}"),
            EngineError::Corrupt(m) => write!(f, "corrupt collection data: {m}"),
            EngineError::Storage(e) => write!(f, "storage error: {e}"),
            EngineError::Io { context, source } => write!(f, "I/O error ({context}): {source}"),
            EngineError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EngineError::Storage(e) => Some(e),
            EngineError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<StorageError> for EngineError {
    fn from(e: StorageError) -> Self {
        match e {
            StorageError::Corrupt(m) => EngineError::Corrupt(m),
            other => EngineError::Storage(other),
        }
    }
}

impl From<SearchError> for EngineError {
    fn from(e: SearchError) -> Self {
        match e {
            SearchError::InvalidParams(m) => EngineError::invalid("search", m),
            SearchError::EntryOutOfRange { .. } => EngineError::Corrupt(e.to_string()),
            SearchError::NonFiniteDistance { .. } => EngineError::Internal(e.to_string()),
            SearchError::Access { .. } => EngineError::Internal(e.to_string()),
        }
    }
}

impl From<serde_json::Error> for EngineError {
    fn from(e: serde_json::Error) -> Self {
        EngineError::Internal(format!("JSON serialization failed: {e}"))
    }
}
