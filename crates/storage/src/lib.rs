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
pub use uring::{IoUringConfig, IoUringDiskStore};

#[cfg(all(feature = "direct-io", target_os = "linux"))]
pub mod direct_io;
#[cfg(all(feature = "direct-io", target_os = "linux"))]
pub mod buffered_direct;
#[cfg(all(feature = "direct-io", target_os = "linux"))]
pub use buffered_direct::DirectIoDiskStore;
#[cfg(all(feature = "direct-io", target_os = "linux"))]
pub use direct_io::{AlignedBuffer, DirectIoAlignment};

#[cfg(all(feature = "direct-io", not(target_os = "linux")))]
compile_error!("the direct-io feature currently requires Linux");

/// Common interface implemented by all storage backends.
pub mod backend;
pub use backend::StorageBackend;
