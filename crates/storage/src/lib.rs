//! Persistent storage for RiskANN vectors and graph adjacency.

pub mod common;
pub use common::{EMPTY_EDGE, GRAPH_FILE, MAX_CAPACITY, OpenReport, StorageError, VECTORS_FILE};

#[cfg(feature = "mmap")]
pub mod memory;
#[cfg(feature = "mmap")]
pub use memory::DiskStore;

#[cfg(feature = "disk")]
pub mod buffered;
#[cfg(feature = "disk")]
pub use buffered::BufferedDiskStore;

#[cfg(feature = "io-uring")]
pub mod uring;
#[cfg(feature = "io-uring")]
pub use uring::IoUringDiskStore;
