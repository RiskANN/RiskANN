//! Memory-mapped persistent storage for RiskANN vectors and graph adjacency.

pub mod memory;
pub use memory::{
    DiskStore, EMPTY_EDGE, GRAPH_FILE, MAX_CAPACITY, OpenReport, StorageError, VECTORS_FILE,
};
