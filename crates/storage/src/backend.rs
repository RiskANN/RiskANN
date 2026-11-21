//! Common engine-facing interface implemented by every storage backend.

use std::borrow::Cow;
use std::path::Path;

use crate::{OpenReport, StorageError};

/// Synchronous storage contract used by the engine.
///
/// A backend is owned by one collection. The engine serializes mutations with
/// its collection write lock; implementations may still support concurrent
/// reads through shared references.
pub trait StorageBackend: Send + Sync + Sized {
    fn create_new(
        dir: &Path,
        dim: usize,
        r: usize,
        capacity: u32,
    ) -> Result<Self, StorageError>;

    fn open_existing(
        dir: &Path,
        dim: usize,
        r: usize,
        required_slots: u32,
    ) -> Result<(Self, OpenReport), StorageError>;

    /// Returns a borrowed slice for mmap and owned storage for file backends.
    fn read_vector_cow(&self, node_id: u32) -> Result<Cow<'_, [f32]>, StorageError>;
    fn write_vector(&mut self, node_id: u32, vector: &[f32]) -> Result<(), StorageError>;
    fn read_edges(&self, node_id: u32) -> Result<Vec<u32>, StorageError>;
    fn read_edges_into(&self, node_id: u32, out: &mut Vec<u32>) -> Result<(), StorageError>;
    fn write_edges(&mut self, node_id: u32, edges: &[u32]) -> Result<(), StorageError>;
    fn resize(&mut self, new_capacity: u32) -> Result<(), StorageError>;
    fn flush(&self) -> Result<(), StorageError>;
    fn dim(&self) -> usize;
    fn r(&self) -> usize;
    fn capacity(&self) -> u32;
    fn disk_usage_bytes(&self) -> u64;
}

#[cfg(feature = "mmap")]
impl StorageBackend for crate::memory::DiskStore {
    fn create_new(dir: &Path, dim: usize, r: usize, capacity: u32) -> Result<Self, StorageError> {
        Self::create_new(dir, dim, r, capacity)
    }
    fn open_existing(dir: &Path, dim: usize, r: usize, required_slots: u32) -> Result<(Self, OpenReport), StorageError> {
        Self::open_existing(dir, dim, r, required_slots)
    }
    fn read_vector_cow(&self, node_id: u32) -> Result<Cow<'_, [f32]>, StorageError> {
        self.read_vector(node_id).map(Cow::Borrowed)
    }
    fn write_vector(&mut self, node_id: u32, vector: &[f32]) -> Result<(), StorageError> {
        Self::write_vector(self, node_id, vector)
    }
    fn read_edges(&self, node_id: u32) -> Result<Vec<u32>, StorageError> {
        Self::read_edges(self, node_id)
    }
    fn read_edges_into(&self, node_id: u32, out: &mut Vec<u32>) -> Result<(), StorageError> {
        Self::read_edges_into(self, node_id, out)
    }
    fn write_edges(&mut self, node_id: u32, edges: &[u32]) -> Result<(), StorageError> {
        Self::write_edges(self, node_id, edges)
    }
    fn resize(&mut self, new_capacity: u32) -> Result<(), StorageError> { Self::resize(self, new_capacity) }
    fn flush(&self) -> Result<(), StorageError> { Self::flush(self) }
    fn dim(&self) -> usize { Self::dim(self) }
    fn r(&self) -> usize { Self::r(self) }
    fn capacity(&self) -> u32 { Self::capacity(self) }
    fn disk_usage_bytes(&self) -> u64 { Self::disk_usage_bytes(self) }
}

macro_rules! impl_owned_backend {
    ($ty:path) => {
        impl StorageBackend for $ty {
            fn create_new(dir: &Path, dim: usize, r: usize, capacity: u32) -> Result<Self, StorageError> {
                Self::create_new(dir, dim, r, capacity)
            }
            fn open_existing(dir: &Path, dim: usize, r: usize, required_slots: u32) -> Result<(Self, OpenReport), StorageError> {
                Self::open_existing(dir, dim, r, required_slots)
            }
            fn read_vector_cow(&self, node_id: u32) -> Result<Cow<'_, [f32]>, StorageError> {
                self.read_vector(node_id).map(Cow::Owned)
            }
            fn write_vector(&mut self, node_id: u32, vector: &[f32]) -> Result<(), StorageError> {
                Self::write_vector(self, node_id, vector)
            }
            fn read_edges(&self, node_id: u32) -> Result<Vec<u32>, StorageError> { Self::read_edges(self, node_id) }
            fn read_edges_into(&self, node_id: u32, out: &mut Vec<u32>) -> Result<(), StorageError> {
                Self::read_edges_into(self, node_id, out)
            }
            fn write_edges(&mut self, node_id: u32, edges: &[u32]) -> Result<(), StorageError> {
                Self::write_edges(self, node_id, edges)
            }
            fn resize(&mut self, new_capacity: u32) -> Result<(), StorageError> { Self::resize(self, new_capacity) }
            fn flush(&self) -> Result<(), StorageError> { Self::flush(self) }
            fn dim(&self) -> usize { Self::dim(self) }
            fn r(&self) -> usize { Self::r(self) }
            fn capacity(&self) -> u32 { Self::capacity(self) }
            fn disk_usage_bytes(&self) -> u64 { Self::disk_usage_bytes(self) }
        }
    };
}

#[cfg(feature = "disk")]
impl_owned_backend!(crate::buffered::BufferedDiskStore);
#[cfg(feature = "direct-io")]
#[cfg(target_os = "linux")]
impl_owned_backend!(crate::buffered_direct::DirectIoDiskStore);
#[cfg(feature = "io-uring")]
impl_owned_backend!(crate::uring::IoUringDiskStore);
