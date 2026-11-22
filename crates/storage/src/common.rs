//! Shared types and constants for RiskANN storage backends.

use std::{fmt, io};

pub const VECTORS_FILE: &str = "vectors.bin";
pub const GRAPH_FILE: &str = "graph.bin";
/// Sentinel that terminates an adjacency list.
pub const EMPTY_EDGE: u32 = u32::MAX;
/// Largest supported capacity; `u32::MAX` is reserved for [`EMPTY_EDGE`].
pub const MAX_CAPACITY: u32 = u32::MAX - 1;

/// Typed storage errors. `Io` contexts may contain file paths; callers that
/// face untrusted clients must log them rather than return them.
#[derive(Debug)]
pub enum StorageError {
    Io { context: String, source: io::Error },
    InvalidArgument(String),
    OutOfBounds { node_id: u32, capacity: u32 },
    DimensionMismatch { expected: usize, got: usize },
    NonFinite { node_id: u32 },
    TooManyEdges { count: usize, max: usize },
    InvalidEdge { node_id: u32, edge: u32, reason: &'static str },
    Corrupt(String),
    Overflow(&'static str),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "I/O error ({context}): {source}"),
            Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Self::OutOfBounds { node_id, capacity } => write!(f, "node id {node_id} is out of bounds (capacity {capacity})"),
            Self::DimensionMismatch { expected, got } => write!(f, "vector has {got} components, expected {expected}"),
            Self::NonFinite { node_id } => write!(f, "vector for node {node_id} contains NaN or infinity"),
            Self::TooManyEdges { count, max } => write!(f, "{count} edges exceed the maximum degree {max}"),
            Self::InvalidEdge { node_id, edge, reason } => write!(f, "invalid edge {node_id} -> {edge}: {reason}"),
            Self::Corrupt(m) => write!(f, "corrupt or inconsistent storage: {m}"),
            Self::Overflow(what) => write!(f, "arithmetic overflow computing {what}"),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self { Self::Io { source, .. } => Some(source), _ => None }
    }
}

pub(crate) fn io_err(context: String) -> impl FnOnce(io::Error) -> StorageError {
    move |source| StorageError::Io { context, source }
}

/// Result of opening an existing store. A capacity mismatch is repaired by
/// growing the shorter file; storage is never truncated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenReport {
    pub vectors_capacity: u32,
    pub graph_capacity: u32,
    pub repaired: bool,
}
