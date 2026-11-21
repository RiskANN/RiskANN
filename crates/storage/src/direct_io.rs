//! Linux direct-I/O alignment discovery and aligned memory primitives.
//!
//! This module deliberately does not change either storage backend. It provides
//! the shared building blocks that buffered `O_DIRECT` and io_uring `O_DIRECT`
//! paths can use later.

use crate::common::StorageError;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::os::fd::AsRawFd;
use std::ptr::NonNull;

/// Alignment constraints reported by Linux for a file opened on a filesystem.
///
/// `memory` applies to the address passed to the kernel. `offset` applies to
/// both the file offset and the I/O length. These values are not assumed to be
/// equal (nor is either one hard-coded to 4096).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectIoAlignment {
    memory: usize,
    offset: usize,
}

impl DirectIoAlignment {
    /// Discovers direct-I/O alignment for an already-open file using
    /// `statx(STATX_DIOALIGN)`.
    ///
    /// The file should be opened with the flags intended for actual direct I/O.
    /// Returns an error if the running kernel/filesystem does not report the
    /// required alignment rather than guessing a value.
    #[cfg(target_os = "linux")]
    pub fn discover(file: &File) -> Result<Self, StorageError> {
        let mut stat: libc::statx = unsafe { std::mem::zeroed() };
        // SAFETY: `stat` is a valid writable statx structure; AT_EMPTY_PATH
        // asks statx to inspect the already-open descriptor.
        let result = unsafe {
            libc::statx(
                file.as_raw_fd(),
                c"".as_ptr(),
                libc::AT_EMPTY_PATH | libc::AT_STATX_DONT_SYNC,
                libc::STATX_DIOALIGN,
                &mut stat,
            )
        };
        if result != 0 {
            return Err(StorageError::Io {
                context: "discover O_DIRECT alignment with statx(STATX_DIOALIGN)".into(),
                source: io::Error::last_os_error(),
            });
        }
        if stat.stx_mask & libc::STATX_DIOALIGN != libc::STATX_DIOALIGN {
            return Err(StorageError::InvalidArgument(
                "filesystem/kernel did not report STATX_DIOALIGN; direct I/O alignment cannot be determined safely".into(),
            ));
        }
        Self::new(stat.stx_dio_mem_align as usize, stat.stx_dio_offset_align as usize)
    }

    /// Constructs validated alignment requirements. Primarily useful for
    /// testing and for callers that obtain alignment from another trusted API.
    pub fn new(memory: usize, offset: usize) -> Result<Self, StorageError> {
        if memory == 0 || offset == 0 {
            return Err(StorageError::InvalidArgument(
                "direct-I/O alignment values must be non-zero".into(),
            ));
        }
        // Rust's allocation Layout requires a power-of-two alignment. Linux
        // direct-I/O memory alignments are expected to satisfy this; reject an
        // unusual value rather than silently allocating incorrectly.
        if !memory.is_power_of_two() || memory < std::mem::align_of::<usize>() {
            return Err(StorageError::InvalidArgument(format!(
                "unsupported direct-I/O memory alignment {memory}: it must be a power of two and at least {}",
                std::mem::align_of::<usize>()
            )));
        }
        Ok(Self { memory, offset })
    }

    pub fn memory(self) -> usize { self.memory }
    pub fn offset(self) -> usize { self.offset }

    /// Rounds a file range outward to the enclosing offset-aligned range.
    /// Returns `(aligned_offset, aligned_len)`.
    pub fn aligned_range(self, offset: u64, len: usize) -> Result<(u64, usize), StorageError> {
        if len == 0 {
            return Ok((offset, 0));
        }
        let alignment = self.offset as u64;
        let start = offset / alignment * alignment;
        let end = offset
            .checked_add(u64::try_from(len).map_err(|_| StorageError::Overflow("I/O range length"))?)
            .ok_or(StorageError::Overflow("I/O range end"))?;
        let remainder = end % alignment;
        let aligned_end = if remainder == 0 {
            end
        } else {
            end.checked_add(alignment - remainder)
                .ok_or(StorageError::Overflow("aligned I/O range end"))?
        };
        let aligned_len = usize::try_from(aligned_end - start)
            .map_err(|_| StorageError::Overflow("aligned I/O range length"))?;
        Ok((start, aligned_len))
    }
}

/// An owned, zero-initialized byte buffer whose address satisfies a requested
/// alignment. The allocation is stable until this value is dropped.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    pub fn new(len: usize, alignment: usize) -> Result<Self, StorageError> {
        if len == 0 {
            return Err(StorageError::InvalidArgument(
                "aligned buffer length must be non-zero".into(),
            ));
        }
        if !alignment.is_power_of_two() || alignment < std::mem::align_of::<usize>() {
            return Err(StorageError::InvalidArgument(format!(
                "buffer alignment {alignment} must be a power of two and at least {}",
                std::mem::align_of::<usize>()
            )));
        }
        let layout = Layout::from_size_align(len, alignment)
            .map_err(|_| StorageError::Overflow("aligned buffer layout"))?;
        // SAFETY: layout is non-zero and valid. The returned allocation is
        // owned by this value and released with the same layout in Drop.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| StorageError::Io {
                context: "allocate aligned I/O buffer".into(),
                source: io::Error::new(io::ErrorKind::OutOfMemory, "aligned allocation failed"),
            })?;
        Ok(Self { ptr, len, layout })
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn alignment(&self) -> usize { self.layout.align() }
    pub fn as_ptr(&self) -> *const u8 { self.ptr.as_ptr() }
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr.as_ptr() }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // SAFETY: the allocation is valid for `len` bytes for self's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: &mut self guarantees unique access to the allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

// Moving the owned allocation between threads is safe; access remains governed
// by Rust's exclusive/shared borrowing rules.
unsafe impl Send for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr/layout are exactly the pair returned/used at allocation.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buffer_has_requested_alignment_and_zeroes_memory() {
        for alignment in [8, 64, 4096] {
            let buffer = AlignedBuffer::new(8192, alignment).unwrap();
            assert_eq!((buffer.as_ptr() as usize) % alignment, 0);
            assert!(buffer.iter().all(|byte| *byte == 0));
            assert_eq!(buffer.len(), 8192);
        }
    }

    #[test]
    fn aligned_range_rounds_outward() {
        let alignment = DirectIoAlignment::new(4096, 512).unwrap();
        assert_eq!(alignment.aligned_range(513, 1000).unwrap(), (512, 1024));
        assert_eq!(alignment.aligned_range(1024, 512).unwrap(), (1024, 512));
        assert_eq!(alignment.aligned_range(1, 1).unwrap(), (0, 512));
        assert_eq!(alignment.aligned_range(7, 0).unwrap(), (7, 0));
    }

    #[test]
    fn alignment_rejects_invalid_values() {
        assert!(DirectIoAlignment::new(0, 512).is_err());
        assert!(DirectIoAlignment::new(3000, 512).is_err());
        assert!(DirectIoAlignment::new(4096, 0).is_err());
    }
}
