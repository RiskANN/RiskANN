//! io_uring-backed vector and adjacency storage backend.
//!
//! Uses io_uring positional read/write submissions. Single-record methods wait
//! for one completion; batch methods submit independent operations together
//! up to the configured queue depth and batch size. The public API remains synchronous.

use crate::common::{io_err, OpenReport, StorageError, EMPTY_EDGE, GRAPH_FILE, MAX_CAPACITY, VECTORS_FILE};
use std::collections::HashSet;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use io_uring::{opcode, squeue, types, IoUring};
use std::path::Path;

const F32_BYTES: usize = std::mem::size_of::<f32>();
const U32_BYTES: usize = std::mem::size_of::<u32>();
const DEFAULT_QUEUE_DEPTH: usize = 8;
const DEFAULT_BATCH_SIZE: usize = 8;

/// Runtime tuning for the io_uring backend.
///
/// `queue_depth` controls the maximum number of operations submitted in one
/// in-flight group. `batch_size` controls how many requests a batch API call
/// groups before moving to the next group. The effective submission size is
/// `min(queue_depth, batch_size)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoUringConfig {
    queue_depth: usize,
    batch_size: usize,
}

impl Default for IoUringConfig {
    fn default() -> Self {
        Self {
            queue_depth: DEFAULT_QUEUE_DEPTH,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

impl IoUringConfig {
    /// Creates a configuration. Both values must be greater than zero.
    pub fn new(queue_depth: usize, batch_size: usize) -> Result<Self, StorageError> {
        if queue_depth == 0 {
            return Err(StorageError::InvalidArgument("io_uring queue depth must be > 0".into()));
        }
        if batch_size == 0 {
            return Err(StorageError::InvalidArgument("io_uring batch size must be > 0".into()));
        }
        u32::try_from(queue_depth).map_err(|_| StorageError::Overflow("io_uring queue depth"))?;
        Ok(Self { queue_depth, batch_size })
    }

    pub fn queue_depth(&self) -> usize { self.queue_depth }
    pub fn batch_size(&self) -> usize { self.batch_size }

    fn submission_size(&self) -> usize {
        self.queue_depth.min(self.batch_size)
    }
}

fn strides(dim: usize, r: usize) -> Result<(usize, usize), StorageError> {
    if dim == 0 { return Err(StorageError::InvalidArgument("dim must be > 0".into())); }
    if r == 0 { return Err(StorageError::InvalidArgument("r must be > 0".into())); }
    let vs = dim.checked_mul(F32_BYTES).ok_or(StorageError::Overflow("vector record size"))?;
    let gs = r.checked_mul(U32_BYTES).ok_or(StorageError::Overflow("graph record size"))?;
    Ok((vs, gs))
}

fn validate_capacity(capacity: u32) -> Result<(), StorageError> {
    if capacity == 0 || capacity > MAX_CAPACITY {
        return Err(StorageError::InvalidArgument(format!("capacity must be in 1..={MAX_CAPACITY}")));
    }
    Ok(())
}

fn byte_len(capacity: u32, stride: usize) -> Result<u64, StorageError> {
    let bytes = (capacity as usize).checked_mul(stride).ok_or(StorageError::Overflow("file size"))?;
    u64::try_from(bytes).map_err(|_| StorageError::Overflow("file size"))
}

fn capacity_from_len(len: u64, stride: usize, name: &str) -> Result<u32, StorageError> {
    let stride64 = u64::try_from(stride).map_err(|_| StorageError::Overflow("record size"))?;
    if len == 0 { return Err(StorageError::Corrupt(format!("{name} is empty"))); }
    if len % stride64 != 0 {
        return Err(StorageError::Corrupt(format!("{name} length {len} is not a multiple of the record size {stride}")));
    }
    let cap = len / stride64;
    if cap > MAX_CAPACITY as u64 {
        return Err(StorageError::Corrupt(format!("{name} holds {cap} slots, more than the supported maximum")));
    }
    Ok(cap as u32)
}

pub struct IoUringDiskStore {
    ring: Mutex<IoUring>,
    vec_file: File,
    graph_file: File,
    dim: usize,
    r: usize,
    capacity: u32,
    vec_stride: usize,
    graph_stride: usize,
    config: IoUringConfig,
}

impl fmt::Debug for IoUringDiskStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoUringDiskStore")
            .field("dim", &self.dim)
            .field("r", &self.r)
            .field("capacity", &self.capacity)
            .field("queue_depth", &self.config.queue_depth)
            .field("batch_size", &self.config.batch_size)
            .finish()
    }
}

/// Submits one operation and waits for its completion.
///
/// The submitted entry contains a raw buffer pointer. Callers keep the buffer
/// borrowed and untouched until this function has consumed the completion.
fn submit_and_wait(ring: &mut IoUring, entry: squeue::Entry, context: &str) -> Result<i32, StorageError> {
    // SAFETY: The caller guarantees that any buffer referenced by `entry`
    // remains valid until the completion is consumed below. The ring is
    // exclusively borrowed for submission and completion processing.
    unsafe {
        ring.submission().push(&entry).map_err(|_| {
            io_err(format!("submit {context} to io_uring"))(
                io::Error::new(io::ErrorKind::WouldBlock, "io_uring submission queue is full"),
            )
        })?;
    }

    loop {
        match ring.submit_and_wait(1) {
            Ok(_) => break,
            // The SQE may already have been submitted when the wait is
            // interrupted. Keep its buffer alive and continue waiting.
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_err(format!("wait for {context} completion"))(error)),
        }
    }

    let result = ring.completion().next().map(|cqe| cqe.result()).ok_or_else(|| {
        io_err(format!("complete {context}"))(
            io::Error::new(io::ErrorKind::Other, "io_uring returned no completion entry"),
        )
    })?;
    Ok(result)
}

fn read_exact_at(
    ring: &mut IoUring,
    file: &File,
    mut buf: &mut [u8],
    mut offset: u64,
    context: &str,
) -> Result<(), StorageError> {
    while !buf.is_empty() {
        let len = buf.len().min(i32::MAX as usize);
        let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), buf.as_mut_ptr(), len as u32)
            .offset(offset)
            .build();
        let result = submit_and_wait(ring, entry, &format!("read {context}"))?;
        if result == 0 {
            return Err(io_err(format!("read {context}"))(io::Error::from(io::ErrorKind::UnexpectedEof)));
        }
        if result < 0 {
            let error = io::Error::from_raw_os_error(-result);
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(io_err(format!("read {context}"))(error));
        }
        let n = result as usize;
        offset = offset.checked_add(n as u64).ok_or(StorageError::Overflow("read offset"))?;
        buf = &mut buf[n..];
    }
    Ok(())
}

fn write_all_at(
    ring: &mut IoUring,
    file: &File,
    mut buf: &[u8],
    mut offset: u64,
    context: &str,
) -> Result<(), StorageError> {
    while !buf.is_empty() {
        let len = buf.len().min(i32::MAX as usize);
        let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), buf.as_ptr(), len as u32)
            .offset(offset)
            .build();
        let result = submit_and_wait(ring, entry, &format!("write {context}"))?;
        if result == 0 {
            return Err(io_err(format!("write {context}"))(io::Error::from(io::ErrorKind::WriteZero)));
        }
        if result < 0 {
            let error = io::Error::from_raw_os_error(-result);
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(io_err(format!("write {context}"))(error));
        }
        let n = result as usize;
        offset = offset.checked_add(n as u64).ok_or(StorageError::Overflow("write offset"))?;
        buf = &buf[n..];
    }
    Ok(())
}

/// One positional operation in a batch. The pointed-to memory must remain
/// valid and untouched until every completion for this batch has been reaped.
struct BatchRequest {
    ptr: *mut u8,
    len: usize,
    offset: u64,
}

/// Submits up to `queue_depth` independent operations and returns results in
/// request order. The caller owns and keeps all pointed-to buffers alive until
/// this function returns.
fn submit_batch_once(
    ring: &mut IoUring,
    file: &File,
    requests: &[BatchRequest],
    is_write: bool,
    queue_depth: usize,
    context: &str,
) -> Result<Vec<i32>, StorageError> {
    debug_assert!(!requests.is_empty());
    debug_assert!(requests.len() <= queue_depth);

    for (index, request) in requests.iter().enumerate() {
        let len = u32::try_from(request.len)
            .map_err(|_| StorageError::Overflow("batch I/O length"))?;
        let entry = (if is_write {
            opcode::Write::new(types::Fd(file.as_raw_fd()), request.ptr.cast_const(), len)
                .offset(request.offset)
                .build()
        } else {
            opcode::Read::new(types::Fd(file.as_raw_fd()), request.ptr, len)
                .offset(request.offset)
                .build()
        })
        .user_data(index as u64);

        // SAFETY: Every request buffer remains alive and untouched by the
        // caller until this function has consumed all corresponding CQEs.
        unsafe {
            ring.submission().push(&entry).map_err(|_| {
                io_err(format!("queue {context} batch"))(
                    io::Error::new(io::ErrorKind::WouldBlock, "io_uring submission queue is full"),
                )
            })?;
        }
    }

    loop {
        match ring.submit_and_wait(requests.len()) {
            Ok(_) => break,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_err(format!("submit {context} batch"))(error)),
        }
    }

    let mut results = vec![None; requests.len()];
    let mut completed = 0usize;
    {
        let mut cq = ring.completion();
        for cqe in &mut cq {
            let index = usize::try_from(cqe.user_data()).unwrap_or(usize::MAX);
            if index < results.len() && results[index].is_none() {
                results[index] = Some(cqe.result());
                completed += 1;
            }
        }
    }

    // submit_and_wait(n) waits for n completions. This loop is defensive in
    // case the completion queue was observed in more than one pass.
    while completed < requests.len() {
        loop {
            match ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(io_err(format!("wait for {context} batch"))(error)),
            }
        }
        let mut cq = ring.completion();
        for cqe in &mut cq {
            let index = usize::try_from(cqe.user_data()).unwrap_or(usize::MAX);
            if index < results.len() && results[index].is_none() {
                results[index] = Some(cqe.result());
                completed += 1;
            }
        }
    }

    results.into_iter().map(|result| {
        result.ok_or_else(|| io_err(format!("complete {context} batch"))(
            io::Error::new(io::ErrorKind::Other, "missing io_uring completion"),
        ))
    }).collect()
}

fn lock_ring(ring: &Mutex<IoUring>) -> Result<std::sync::MutexGuard<'_, IoUring>, StorageError> {
    ring.lock().map_err(|_| {
        io_err("lock io_uring instance".into())(
            io::Error::new(io::ErrorKind::Other, "io_uring mutex was poisoned"),
        )
    })
}

/// Fills a file range with a repeated byte using bounded-size writes.
fn fill_file_range(ring: &mut IoUring, file: &File, start: u64, len: u64, byte: u8, context: &str) -> Result<(), StorageError> {
    const CHUNK: usize = 64 * 1024;
    let buffer = vec![byte; CHUNK];
    let mut written = 0u64;
    while written < len {
        let n = usize::try_from((len - written).min(CHUNK as u64))
            .map_err(|_| StorageError::Overflow("fill chunk size"))?;
        write_all_at(ring, file, &buffer[..n], start.checked_add(written).ok_or(StorageError::Overflow("fill offset"))?, context)?;
        written += n as u64;
    }
    Ok(())
}

impl IoUringDiskStore {
    /// Creates the two storage files using the default io_uring configuration.
    /// Existing files are never overwritten.
    pub fn create_new(dir: &Path, dim: usize, r: usize, capacity: u32) -> Result<Self, StorageError> {
        Self::create_new_with_config(dir, dim, r, capacity, IoUringConfig::default())
    }

    /// Creates the two storage files using an explicit io_uring configuration.
    pub fn create_new_with_config(
        dir: &Path,
        dim: usize,
        r: usize,
        capacity: u32,
        config: IoUringConfig,
    ) -> Result<Self, StorageError> {
        let (vec_stride, graph_stride) = strides(dim, r)?;
        validate_capacity(capacity)?;
        let entries = u32::try_from(config.queue_depth)
            .map_err(|_| StorageError::Overflow("io_uring queue depth"))?;
        let ring = IoUring::new(entries).map_err(io_err("create io_uring instance".into()))?;
        let ring = Mutex::new(ring);
        let vec_len = byte_len(capacity, vec_stride)?;
        let graph_len = byte_len(capacity, graph_stride)?;
        let vec_path = dir.join(VECTORS_FILE);
        let graph_path = dir.join(GRAPH_FILE);

        let vec_file = OpenOptions::new().read(true).write(true).create_new(true)
            .open(&vec_path).map_err(io_err(format!("create {}", vec_path.display())))?;
        vec_file.set_len(vec_len).map_err(io_err(format!("size {}", vec_path.display())))?;
        let graph_file = OpenOptions::new().read(true).write(true).create_new(true)
            .open(&graph_path).map_err(io_err(format!("create {}", graph_path.display())))?;
        graph_file.set_len(graph_len).map_err(io_err(format!("size {}", graph_path.display())))?;
        {
            let mut ring_guard = lock_ring(&ring)?;
            fill_file_range(&mut ring_guard, &graph_file, 0, graph_len, 0xFF, GRAPH_FILE)?;
        }

        Ok(Self { ring, vec_file, graph_file, dim, r, capacity, vec_stride, graph_stride, config })
    }

    /// Opens existing files using the default io_uring configuration.
    /// Grows the shorter file if a previous resize was interrupted.
    pub fn open_existing(dir: &Path, dim: usize, r: usize, required_slots: u32) -> Result<(Self, OpenReport), StorageError> {
        Self::open_existing_with_config(dir, dim, r, required_slots, IoUringConfig::default())
    }

    /// Opens existing files using an explicit io_uring configuration.
    pub fn open_existing_with_config(
        dir: &Path,
        dim: usize,
        r: usize,
        required_slots: u32,
        config: IoUringConfig,
    ) -> Result<(Self, OpenReport), StorageError> {
        let (vec_stride, graph_stride) = strides(dim, r)?;
        let entries = u32::try_from(config.queue_depth)
            .map_err(|_| StorageError::Overflow("io_uring queue depth"))?;
        let ring = Mutex::new(IoUring::new(entries).map_err(io_err("create io_uring instance".into()))?);
        let vec_path = dir.join(VECTORS_FILE);
        let graph_path = dir.join(GRAPH_FILE);
        let vec_file = OpenOptions::new().read(true).write(true).open(&vec_path)
            .map_err(io_err(format!("open {}", vec_path.display())))?;
        let graph_file = OpenOptions::new().read(true).write(true).open(&graph_path)
            .map_err(io_err(format!("open {}", graph_path.display())))?;
        let vec_len = vec_file.metadata().map_err(io_err(format!("stat {}", vec_path.display())))?.len();
        let graph_len = graph_file.metadata().map_err(io_err(format!("stat {}", graph_path.display())))?.len();
        let cap_v = capacity_from_len(vec_len, vec_stride, VECTORS_FILE)?;
        let cap_g = capacity_from_len(graph_len, graph_stride, GRAPH_FILE)?;
        // A resize can be interrupted after one file has grown. Validate the
        // metadata requirement against the larger on-disk capacity, then grow
        // the shorter file to match it below.
        let capacity = cap_v.max(cap_g);
        if capacity < required_slots {
            return Err(StorageError::Corrupt(format!(
                "metadata references {required_slots} vectors but {VECTORS_FILE} holds {cap_v} slots and {GRAPH_FILE} holds {cap_g}"
            )));
        }
        let repaired = cap_v != cap_g;
        if cap_v < capacity {
            vec_file.set_len(byte_len(capacity, vec_stride)?)
                .map_err(io_err(format!("grow {}", vec_path.display())))?;
        }
        if cap_g < capacity {
            let new_len = byte_len(capacity, graph_stride)?;
            graph_file.set_len(new_len).map_err(io_err(format!("grow {}", graph_path.display())))?;
            let mut ring_guard = lock_ring(&ring)?;
            fill_file_range(&mut ring_guard, &graph_file, graph_len, new_len - graph_len, 0xFF, GRAPH_FILE)?;
        }
        Ok((Self { ring, vec_file, graph_file, dim, r, capacity, vec_stride, graph_stride, config }, OpenReport {
            vectors_capacity: cap_v, graph_capacity: cap_g, repaired,
        }))
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn r(&self) -> usize { self.r }
    pub fn capacity(&self) -> u32 { self.capacity }
    pub fn config(&self) -> IoUringConfig { self.config }
    pub fn queue_depth(&self) -> usize { self.config.queue_depth }
    pub fn batch_size(&self) -> usize { self.config.batch_size }

    /// Total logical bytes occupied by the two files.
    pub fn disk_usage_bytes(&self) -> u64 {
        byte_len(self.capacity, self.vec_stride).unwrap_or(0)
            .saturating_add(byte_len(self.capacity, self.graph_stride).unwrap_or(0))
    }

    fn slot_offset(&self, node_id: u32, stride: usize) -> Result<u64, StorageError> {
        if node_id >= self.capacity {
            return Err(StorageError::OutOfBounds { node_id, capacity: self.capacity });
        }
        let offset = (node_id as u64).checked_mul(stride as u64)
            .ok_or(StorageError::Overflow("slot offset"))?;
        Ok(offset)
    }

    /// Reads multiple vectors, allocating one result buffer per requested node.
    pub fn read_vectors_batch(&self, node_ids: &[u32]) -> Result<Vec<Vec<f32>>, StorageError> {
        let mut outputs = (0..node_ids.len())
            .map(|_| Vec::with_capacity(self.dim))
            .collect::<Vec<_>>();
        self.read_vectors_batch_into(node_ids, &mut outputs)?;
        Ok(outputs)
    }

    /// Reads multiple vectors into caller-owned buffers, resizing each to `dim`.
    /// This avoids per-call allocation when the caller reuses the output vectors.
    pub fn read_vectors_batch_into(
        &self,
        node_ids: &[u32],
        outputs: &mut [Vec<f32>],
    ) -> Result<(), StorageError> {
        if node_ids.len() != outputs.len() {
            return Err(StorageError::InvalidArgument(format!(
                "batch has {} node IDs but {} output buffers", node_ids.len(), outputs.len()
            )));
        }
        if node_ids.is_empty() {
            return Ok(());
        }

        let offsets = node_ids.iter()
            .map(|&node_id| self.slot_offset(node_id, self.vec_stride))
            .collect::<Result<Vec<_>, _>>()?;
        for output in outputs.iter_mut() {
            output.resize(self.dim, 0.0);
        }
        let mut ring = lock_ring(&self.ring)?;

        for base in (0..node_ids.len()).step_by(self.config.submission_size()) {
            let end = (base + self.config.submission_size()).min(node_ids.len());
            let requests = outputs[base..end].iter_mut().zip(&offsets[base..end])
                .map(|(output, &offset)| BatchRequest {
                    ptr: bytemuck::cast_slice_mut(output.as_mut_slice()).as_mut_ptr(),
                    len: self.vec_stride,
                    offset,
                })
                .collect::<Vec<_>>();
            let results = submit_batch_once(&mut ring, &self.vec_file, &requests, false, self.config.queue_depth, VECTORS_FILE)?;
            drop(requests);

            for (relative, result) in results.into_iter().enumerate() {
                let index = base + relative;
                if result < 0 {
                    let error = io::Error::from_raw_os_error(-result);
                    if error.kind() != io::ErrorKind::Interrupted {
                        return Err(io_err(format!("read {VECTORS_FILE}"))(error));
                    }
                    let bytes = bytemuck::cast_slice_mut(outputs[index].as_mut_slice());
                    read_exact_at(&mut ring, &self.vec_file, bytes, offsets[index], VECTORS_FILE)?;
                } else if result == 0 {
                    let bytes = bytemuck::cast_slice_mut(outputs[index].as_mut_slice());
                    read_exact_at(&mut ring, &self.vec_file, bytes, offsets[index], VECTORS_FILE)?;
                } else {
                    let completed = result as usize;
                    if completed > self.vec_stride {
                        return Err(io_err(format!("read {VECTORS_FILE}"))(
                            io::Error::new(io::ErrorKind::InvalidData, "io_uring read exceeded requested length"),
                        ));
                    }
                    if completed < self.vec_stride {
                        let bytes = bytemuck::cast_slice_mut(outputs[index].as_mut_slice());
                        read_exact_at(&mut ring, &self.vec_file, &mut bytes[completed..],
                            offsets[index] + completed as u64, VECTORS_FILE)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Writes multiple vectors using concurrent io_uring submissions.
    /// Duplicate node IDs are rejected to avoid nondeterministic write ordering.
    pub fn write_vectors_batch(&mut self, writes: &[(u32, &[f32])]) -> Result<(), StorageError> {
        if writes.is_empty() {
            return Ok(());
        }

        let mut offsets = Vec::with_capacity(writes.len());
        let mut seen = HashSet::with_capacity(writes.len());
        for &(node_id, vector) in writes {
            if !seen.insert(node_id) {
                return Err(StorageError::InvalidArgument(format!("duplicate node {node_id} in vector batch")));
            }
            if vector.len() != self.dim {
                return Err(StorageError::DimensionMismatch { expected: self.dim, got: vector.len() });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(StorageError::NonFinite { node_id });
            }
            offsets.push(self.slot_offset(node_id, self.vec_stride)?);
        }

        let mut ring = lock_ring(&self.ring)?;
        for base in (0..writes.len()).step_by(self.config.submission_size()) {
            let end = (base + self.config.submission_size()).min(writes.len());
            let requests = writes[base..end].iter().zip(&offsets[base..end])
                .map(|((_, vector), &offset)| BatchRequest {
                    ptr: bytemuck::cast_slice::<f32, u8>(vector).as_ptr() as *mut u8,
                    len: self.vec_stride,
                    offset,
                })
                .collect::<Vec<_>>();
            let results = submit_batch_once(&mut ring, &self.vec_file, &requests, true, self.config.queue_depth, VECTORS_FILE)?;
            drop(requests);

            for (relative, result) in results.into_iter().enumerate() {
                let index = base + relative;
                if result < 0 {
                    let error = io::Error::from_raw_os_error(-result);
                    if error.kind() != io::ErrorKind::Interrupted {
                        return Err(io_err(format!("write {VECTORS_FILE}"))(error));
                    }
                    write_all_at(&mut ring, &self.vec_file, bytemuck::cast_slice(writes[index].1), offsets[index], VECTORS_FILE)?;
                } else if result == 0 {
                    write_all_at(&mut ring, &self.vec_file, bytemuck::cast_slice(writes[index].1), offsets[index], VECTORS_FILE)?;
                } else {
                    let completed = result as usize;
                    if completed > self.vec_stride {
                        return Err(io_err(format!("write {VECTORS_FILE}"))(
                            io::Error::new(io::ErrorKind::InvalidData, "io_uring write exceeded requested length"),
                        ));
                    }
                    if completed < self.vec_stride {
                        let bytes = bytemuck::cast_slice(writes[index].1);
                        write_all_at(&mut ring, &self.vec_file, &bytes[completed..],
                            offsets[index] + completed as u64, VECTORS_FILE)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Reads multiple adjacency lists, allocating one result buffer per node.
    pub fn read_edges_batch(&self, node_ids: &[u32]) -> Result<Vec<Vec<u32>>, StorageError> {
        let mut outputs = (0..node_ids.len())
            .map(|_| Vec::with_capacity(self.r))
            .collect::<Vec<_>>();
        self.read_edges_batch_into(node_ids, &mut outputs)?;
        Ok(outputs)
    }

    /// Reads multiple adjacency lists into caller-owned buffers.
    /// Each output is replaced (cleared and refilled), rather than appended to.
    pub fn read_edges_batch_into(
        &self,
        node_ids: &[u32],
        outputs: &mut [Vec<u32>],
    ) -> Result<(), StorageError> {
        if node_ids.len() != outputs.len() {
            return Err(StorageError::InvalidArgument(format!(
                "batch has {} node IDs but {} output buffers", node_ids.len(), outputs.len()
            )));
        }
        if node_ids.is_empty() {
            return Ok(());
        }
        let offsets = node_ids.iter()
            .map(|&node_id| self.slot_offset(node_id, self.graph_stride))
            .collect::<Result<Vec<_>, _>>()?;
        for output in outputs.iter_mut() {
            output.clear();
            output.resize(self.r, EMPTY_EDGE);
        }
        let mut ring = lock_ring(&self.ring)?;

        for base in (0..node_ids.len()).step_by(self.config.submission_size()) {
            let end = (base + self.config.submission_size()).min(node_ids.len());
            let requests = outputs[base..end].iter_mut().zip(&offsets[base..end])
                .map(|(edges, &offset)| BatchRequest {
                    ptr: bytemuck::cast_slice_mut(edges.as_mut_slice()).as_mut_ptr(),
                    len: self.graph_stride,
                    offset,
                })
                .collect::<Vec<_>>();
            let results = submit_batch_once(&mut ring, &self.graph_file, &requests, false, self.config.queue_depth, GRAPH_FILE)?;
            drop(requests);

            for (relative, result) in results.into_iter().enumerate() {
                let index = base + relative;
                if result < 0 {
                    let error = io::Error::from_raw_os_error(-result);
                    if error.kind() != io::ErrorKind::Interrupted {
                        return Err(io_err(format!("read {GRAPH_FILE}"))(error));
                    }
                    let bytes = bytemuck::cast_slice_mut(outputs[index].as_mut_slice());
                    read_exact_at(&mut ring, &self.graph_file, bytes, offsets[index], GRAPH_FILE)?;
                } else if result == 0 {
                    let bytes = bytemuck::cast_slice_mut(outputs[index].as_mut_slice());
                    read_exact_at(&mut ring, &self.graph_file, bytes, offsets[index], GRAPH_FILE)?;
                } else {
                    let completed = result as usize;
                    if completed > self.graph_stride {
                        return Err(io_err(format!("read {GRAPH_FILE}"))(
                            io::Error::new(io::ErrorKind::InvalidData, "io_uring read exceeded requested length"),
                        ));
                    }
                    if completed < self.graph_stride {
                        let bytes = bytemuck::cast_slice_mut(outputs[index].as_mut_slice());
                        read_exact_at(&mut ring, &self.graph_file, &mut bytes[completed..],
                            offsets[index] + completed as u64, GRAPH_FILE)?;
                    }
                }
            }
        }
        for output in outputs.iter_mut() {
            let used = output.iter().position(|&edge| edge == EMPTY_EDGE).unwrap_or(output.len());
            output.truncate(used);
        }
        Ok(())
    }

    /// Writes multiple adjacency lists using concurrent io_uring submissions.
    /// Duplicate node IDs are rejected to avoid nondeterministic write ordering.
    pub fn write_edges_batch(&mut self, writes: &[(u32, &[u32])]) -> Result<(), StorageError> {
        if writes.is_empty() {
            return Ok(());
        }
        let mut offsets = Vec::with_capacity(writes.len());
        let mut padded = Vec::with_capacity(writes.len());
        let mut seen = HashSet::with_capacity(writes.len());
        for &(node_id, edges) in writes {
            if !seen.insert(node_id) {
                return Err(StorageError::InvalidArgument(format!("duplicate node {node_id} in edge batch")));
            }
            if edges.len() > self.r {
                return Err(StorageError::TooManyEdges { count: edges.len(), max: self.r });
            }
            for (edge_index, &edge) in edges.iter().enumerate() {
                if edge == EMPTY_EDGE || edge >= self.capacity {
                    return Err(StorageError::InvalidEdge { node_id, edge, reason: "target out of range" });
                }
                if edge == node_id {
                    return Err(StorageError::InvalidEdge { node_id, edge, reason: "self-edge" });
                }
                if edges[..edge_index].contains(&edge) {
                    return Err(StorageError::InvalidEdge { node_id, edge, reason: "duplicate edge" });
                }
            }
            offsets.push(self.slot_offset(node_id, self.graph_stride)?);
            let mut record = vec![EMPTY_EDGE; self.r];
            record[..edges.len()].copy_from_slice(edges);
            padded.push(record);
        }

        let mut ring = lock_ring(&self.ring)?;
        for base in (0..writes.len()).step_by(self.config.submission_size()) {
            let end = (base + self.config.submission_size()).min(writes.len());
            let requests = padded[base..end].iter()
                .zip(&offsets[base..end])
                .map(|(edges, &offset)| BatchRequest {
                    ptr: bytemuck::cast_slice::<u32, u8>(edges).as_ptr() as *mut u8,
                    len: self.graph_stride,
                    offset,
                })
                .collect::<Vec<_>>();
            let results = submit_batch_once(&mut ring, &self.graph_file, &requests, true, self.config.queue_depth, GRAPH_FILE)?;
            drop(requests);

            for (relative, result) in results.into_iter().enumerate() {
                let index = base + relative;
                if result < 0 {
                    let error = io::Error::from_raw_os_error(-result);
                    if error.kind() != io::ErrorKind::Interrupted {
                        return Err(io_err(format!("write {GRAPH_FILE}"))(error));
                    }
                    write_all_at(&mut ring, &self.graph_file, bytemuck::cast_slice(&padded[index]), offsets[index], GRAPH_FILE)?;
                } else if result == 0 {
                    write_all_at(&mut ring, &self.graph_file, bytemuck::cast_slice(&padded[index]), offsets[index], GRAPH_FILE)?;
                } else {
                    let completed = result as usize;
                    if completed > self.graph_stride {
                        return Err(io_err(format!("write {GRAPH_FILE}"))(
                            io::Error::new(io::ErrorKind::InvalidData, "io_uring write exceeded requested length"),
                        ));
                    }
                    if completed < self.graph_stride {
                        let bytes = bytemuck::cast_slice(&padded[index]);
                        write_all_at(&mut ring, &self.graph_file, &bytes[completed..],
                            offsets[index] + completed as u64, GRAPH_FILE)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Reads a vector into a reusable buffer, resizing it to `dim` if needed.
    pub fn read_vector_into(&self, node_id: u32, out: &mut Vec<f32>) -> Result<(), StorageError> {
        let offset = self.slot_offset(node_id, self.vec_stride)?;
        out.resize(self.dim, 0.0);
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(out.as_mut_slice());
        let mut ring = lock_ring(&self.ring)?;
        read_exact_at(&mut ring, &self.vec_file, bytes, offset, VECTORS_FILE)
    }

    /// Reads a vector into a newly allocated owned buffer.
    pub fn read_vector(&self, node_id: u32) -> Result<Vec<f32>, StorageError> {
        let mut out = Vec::with_capacity(self.dim);
        self.read_vector_into(node_id, &mut out)?;
        Ok(out)
    }

    pub fn write_vector(&mut self, node_id: u32, vector: &[f32]) -> Result<(), StorageError> {
        if vector.len() != self.dim {
            return Err(StorageError::DimensionMismatch { expected: self.dim, got: vector.len() });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(StorageError::NonFinite { node_id });
        }
        let offset = self.slot_offset(node_id, self.vec_stride)?;
        let mut ring = lock_ring(&self.ring)?;
        write_all_at(&mut ring, &self.vec_file, bytemuck::cast_slice(vector), offset, VECTORS_FILE)
    }

    /// Appends the adjacency list to `out` without clearing it.
    pub fn read_edges_into(&self, node_id: u32, out: &mut Vec<u32>) -> Result<(), StorageError> {
        let offset = self.slot_offset(node_id, self.graph_stride)?;
        let mut raw = vec![EMPTY_EDGE; self.r];
        let mut ring = lock_ring(&self.ring)?;
        read_exact_at(&mut ring, &self.graph_file, bytemuck::cast_slice_mut(raw.as_mut_slice()), offset, GRAPH_FILE)?;
        out.extend(raw.into_iter().take_while(|&edge| edge != EMPTY_EDGE));
        Ok(())
    }

    pub fn read_edges(&self, node_id: u32) -> Result<Vec<u32>, StorageError> {
        let mut out = Vec::with_capacity(self.r);
        self.read_edges_into(node_id, &mut out)?;
        Ok(out)
    }

    pub fn write_edges(&mut self, node_id: u32, edges: &[u32]) -> Result<(), StorageError> {
        if edges.len() > self.r {
            return Err(StorageError::TooManyEdges { count: edges.len(), max: self.r });
        }
        for (i, &edge) in edges.iter().enumerate() {
            if edge == EMPTY_EDGE || edge >= self.capacity {
                return Err(StorageError::InvalidEdge { node_id, edge, reason: "target out of range" });
            }
            if edge == node_id {
                return Err(StorageError::InvalidEdge { node_id, edge, reason: "self-edge" });
            }
            if edges[..i].contains(&edge) {
                return Err(StorageError::InvalidEdge { node_id, edge, reason: "duplicate edge" });
            }
        }
        let offset = self.slot_offset(node_id, self.graph_stride)?;
        let mut padded = vec![EMPTY_EDGE; self.r];
        padded[..edges.len()].copy_from_slice(edges);
        let mut ring = lock_ring(&self.ring)?;
        write_all_at(&mut ring, &self.graph_file, bytemuck::cast_slice(&padded), offset, GRAPH_FILE)
    }

    /// Persists pending file data and metadata to the storage device.
    pub fn flush(&self) -> Result<(), StorageError> {
        self.vec_file.sync_data().map_err(io_err(format!("sync {VECTORS_FILE}")))?;
        self.graph_file.sync_data().map_err(io_err(format!("sync {GRAPH_FILE}")))?;
        Ok(())
    }

    /// Grows both files; shrinking is deliberately unsupported.
    pub fn resize(&mut self, new_capacity: u32) -> Result<(), StorageError> {
        validate_capacity(new_capacity)?;
        if new_capacity <= self.capacity {
            return Err(StorageError::InvalidArgument(format!(
                "new capacity {new_capacity} must exceed current capacity {}", self.capacity
            )));
        }
        self.flush()?;
        let old_graph_len = byte_len(self.capacity, self.graph_stride)?;
        let new_vec_len = byte_len(new_capacity, self.vec_stride)?;
        let new_graph_len = byte_len(new_capacity, self.graph_stride)?;
        self.vec_file.set_len(new_vec_len).map_err(io_err(format!("grow {VECTORS_FILE}")))?;
        self.graph_file.set_len(new_graph_len).map_err(io_err(format!("grow {GRAPH_FILE}")))?;
        {
            let mut ring = lock_ring(&self.ring)?;
            fill_file_range(&mut ring, &self.graph_file, old_graph_len, new_graph_len - old_graph_len, 0xFF, GRAPH_FILE)?;
        }
        self.capacity = new_capacity;
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let p = std::env::temp_dir().join(format!(
                "riskann-storage-test-{}-{}-{}",
                std::process::id(),
                n,
                nanos
            ));
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn configurable_queue_depth_and_batch_size_work_independently() {
        let dir = TempDir::new();
        let config = IoUringConfig::new(4, 2).unwrap();
        let mut store = IoUringDiskStore::create_new_with_config(dir.path(), 2, 2, 12, config).unwrap();
        assert_eq!(store.queue_depth(), 4);
        assert_eq!(store.batch_size(), 2);
        assert_eq!(store.config(), config);

        let vectors = (0..10u32)
            .map(|id| (id, [id as f32, id as f32 + 0.5]))
            .collect::<Vec<_>>();
        let writes = vectors.iter().map(|(id, vector)| (*id, vector.as_slice())).collect::<Vec<_>>();
        store.write_vectors_batch(&writes).unwrap();
        let ids = (0..10u32).collect::<Vec<_>>();
        let outputs = store.read_vectors_batch(&ids).unwrap();
        for (id, output) in ids.iter().zip(outputs) {
            assert_eq!(output, vec![*id as f32, *id as f32 + 0.5]);
        }

        let reopened = IoUringDiskStore::open_existing_with_config(dir.path(), 2, 2, 10, config).unwrap().0;
        assert_eq!(reopened.config(), config);
    }

    #[test]
    fn configurable_batch_size_is_capped_by_queue_depth() {
        let dir = TempDir::new();
        let config = IoUringConfig::new(2, 5).unwrap();
        let mut store = IoUringDiskStore::create_new_with_config(dir.path(), 1, 1, 8, config).unwrap();
        let values = (0..8u32).map(|id| (id, [id as f32])).collect::<Vec<_>>();
        let writes = values.iter().map(|(id, value)| (*id, value.as_slice())).collect::<Vec<_>>();
        store.write_vectors_batch(&writes).unwrap();
        let ids = (0..8u32).collect::<Vec<_>>();
        let outputs = store.read_vectors_batch(&ids).unwrap();
        assert_eq!(outputs.into_iter().map(|v| v[0]).collect::<Vec<_>>(), (0..8).map(|v| v as f32).collect::<Vec<_>>());
    }

    #[test]
    fn io_uring_config_rejects_zero_values() {
        assert!(IoUringConfig::new(0, 1).is_err());
        assert!(IoUringConfig::new(1, 0).is_err());
    }

    #[test]
    fn create_write_reopen_preserves_vectors_and_edges() {
        let dir = TempDir::new();
        {
            let mut s = IoUringDiskStore::create_new(dir.path(), 3, 4, 10).unwrap();
            s.write_vector(0, &[1.0, 2.0, 3.0]).unwrap();
            s.write_vector(9, &[7.0, 8.0, 9.0]).unwrap();
            s.write_edges(0, &[9, 3]).unwrap();
            s.flush().unwrap();
        }
        let (s, report) = IoUringDiskStore::open_existing(dir.path(), 3, 4, 10).unwrap();
        assert!(!report.repaired);
        assert_eq!(s.capacity(), 10);
        assert_eq!(s.read_vector(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(s.read_vector(9).unwrap(), &[7.0, 8.0, 9.0]);
        assert_eq!(s.read_edges(0).unwrap(), vec![9, 3]);
    }

    #[test]
    fn reopen_never_truncates() {
        let dir = TempDir::new();
        {
            let mut s = IoUringDiskStore::create_new(dir.path(), 2, 2, 20_000).unwrap();
            s.write_vector(19_999, &[5.0, 6.0]).unwrap();
            s.flush().unwrap();
        }
        let before = std::fs::metadata(dir.path().join(VECTORS_FILE))
            .unwrap()
            .len();
        let (s, _) = IoUringDiskStore::open_existing(dir.path(), 2, 2, 1).unwrap();
        let after = std::fs::metadata(dir.path().join(VECTORS_FILE))
            .unwrap()
            .len();
        assert_eq!(before, after);
        assert_eq!(s.capacity(), 20_000);
        assert_eq!(s.read_vector(19_999).unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn first_and_last_slot_and_out_of_bounds() {
        let dir = TempDir::new();
        let mut s = IoUringDiskStore::create_new(dir.path(), 2, 2, 5).unwrap();
        s.write_vector(0, &[0.5, 0.25]).unwrap();
        s.write_vector(4, &[4.5, 4.25]).unwrap();
        assert_eq!(s.read_vector(4).unwrap(), &[4.5, 4.25]);
        assert!(matches!(
            s.write_vector(5, &[1.0, 1.0]),
            Err(StorageError::OutOfBounds {
                node_id: 5,
                capacity: 5
            })
        ));
        assert!(matches!(
            s.read_vector(5),
            Err(StorageError::OutOfBounds { .. })
        ));
        assert!(matches!(
            s.read_edges(u32::MAX),
            Err(StorageError::OutOfBounds { .. })
        ));
        assert!(matches!(
            s.write_edges(5, &[]),
            Err(StorageError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_wrong_dimension_and_non_finite() {
        let dir = TempDir::new();
        let mut s = IoUringDiskStore::create_new(dir.path(), 3, 2, 2).unwrap();
        assert!(matches!(
            s.write_vector(0, &[1.0, 2.0]),
            Err(StorageError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
        assert!(matches!(
            s.write_vector(0, &[1.0, f32::NAN, 2.0]),
            Err(StorageError::NonFinite { node_id: 0 })
        ));
        assert!(matches!(
            s.write_vector(0, &[1.0, f32::NEG_INFINITY, 2.0]),
            Err(StorageError::NonFinite { .. })
        ));
    }

    #[test]
    fn edges_empty_max_degree_and_invalid() {
        let dir = TempDir::new();
        let mut s = IoUringDiskStore::create_new(dir.path(), 1, 3, 10).unwrap();
        // Fresh slots are empty lists, not lists of zeros.
        assert!(s.read_edges(7).unwrap().is_empty());
        s.write_edges(1, &[2, 3, 4]).unwrap();
        assert_eq!(s.read_edges(1).unwrap(), vec![2, 3, 4]);
        s.write_edges(1, &[]).unwrap();
        assert!(s.read_edges(1).unwrap().is_empty());
        assert!(matches!(
            s.write_edges(1, &[2, 3, 4, 5]),
            Err(StorageError::TooManyEdges { count: 4, max: 3 })
        ));
        assert!(matches!(
            s.write_edges(1, &[1]),
            Err(StorageError::InvalidEdge {
                reason: "self-edge",
                ..
            })
        ));
        assert!(matches!(
            s.write_edges(1, &[2, 2]),
            Err(StorageError::InvalidEdge {
                reason: "duplicate edge",
                ..
            })
        ));
        assert!(matches!(
            s.write_edges(1, &[10]),
            Err(StorageError::InvalidEdge {
                reason: "target out of range",
                ..
            })
        ));
        assert!(matches!(
            s.write_edges(1, &[EMPTY_EDGE]),
            Err(StorageError::InvalidEdge { .. })
        ));
    }

    #[test]
    fn read_edges_into_appends() {
        let dir = TempDir::new();
        let mut s = IoUringDiskStore::create_new(dir.path(), 1, 2, 4).unwrap();
        s.write_edges(0, &[1, 2]).unwrap();
        let mut out = vec![99];
        s.read_edges_into(0, &mut out).unwrap();
        assert_eq!(out, vec![99, 1, 2]);
    }

    #[test]
    fn create_new_refuses_to_overwrite() {
        let dir = TempDir::new();
        let _s = IoUringDiskStore::create_new(dir.path(), 2, 2, 2).unwrap();
        assert!(matches!(
            IoUringDiskStore::create_new(dir.path(), 2, 2, 2),
            Err(StorageError::Io { .. })
        ));
    }

    #[test]
    fn invalid_shapes_rejected() {
        let dir = TempDir::new();
        assert!(IoUringDiskStore::create_new(dir.path(), 0, 2, 2).is_err());
        assert!(IoUringDiskStore::create_new(dir.path(), 2, 0, 2).is_err());
        assert!(IoUringDiskStore::create_new(dir.path(), 2, 2, 0).is_err());
        assert!(IoUringDiskStore::create_new(dir.path(), 2, 2, u32::MAX).is_err());
    }

    #[test]
    fn malformed_file_length_is_corrupt() {
        let dir = TempDir::new();
        {
            IoUringDiskStore::create_new(dir.path(), 3, 2, 4).unwrap();
        }
        let f = OpenOptions::new()
            .write(true)
            .open(dir.path().join(VECTORS_FILE))
            .unwrap();
        f.set_len(4 * 12 + 5).unwrap();
        drop(f);
        assert!(matches!(
            IoUringDiskStore::open_existing(dir.path(), 3, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn empty_file_is_corrupt() {
        let dir = TempDir::new();
        {
            IoUringDiskStore::create_new(dir.path(), 3, 2, 4).unwrap();
        }
        OpenOptions::new()
            .write(true)
            .open(dir.path().join(GRAPH_FILE))
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(matches!(
            IoUringDiskStore::open_existing(dir.path(), 3, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn wrong_shape_on_open_is_corrupt() {
        let dir = TempDir::new();
        {
            IoUringDiskStore::create_new(dir.path(), 3, 2, 4).unwrap(); // 48 bytes of vectors
        }
        // dim 5 -> 20-byte records; 48 is not a multiple.
        assert!(matches!(
            IoUringDiskStore::open_existing(dir.path(), 5, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn metadata_claiming_more_slots_than_files_is_corrupt() {
        let dir = TempDir::new();
        {
            IoUringDiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
        }
        assert!(matches!(
            IoUringDiskStore::open_existing(dir.path(), 2, 2, 5),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn missing_files_are_io_errors() {
        let dir = TempDir::new();
        assert!(matches!(
            IoUringDiskStore::open_existing(dir.path(), 2, 2, 0),
            Err(StorageError::Io { .. })
        ));
    }

    #[test]
    fn resize_preserves_data_and_initialises_new_slots() {
        let dir = TempDir::new();
        let mut s = IoUringDiskStore::create_new(dir.path(), 2, 3, 4).unwrap();
        s.write_vector(3, &[3.0, 3.5]).unwrap();
        s.write_edges(3, &[0, 1]).unwrap();
        s.resize(9).unwrap();
        assert_eq!(s.capacity(), 9);
        assert_eq!(s.read_vector(3).unwrap(), &[3.0, 3.5]);
        assert_eq!(s.read_edges(3).unwrap(), vec![0, 1]);
        assert!(s.read_edges(8).unwrap().is_empty());
        s.write_vector(8, &[8.0, 8.5]).unwrap();
        assert!(s.resize(9).is_err());
        assert!(s.resize(2).is_err());
        drop(s);
        let (s, report) = IoUringDiskStore::open_existing(dir.path(), 2, 3, 9).unwrap();
        assert!(!report.repaired);
        assert_eq!(s.capacity(), 9);
        assert_eq!(s.read_vector(8).unwrap(), &[8.0, 8.5]);
    }

    #[test]
    fn capacity_mismatch_is_repaired_by_growing() {
        let dir = TempDir::new();
        {
            let mut s = IoUringDiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
            s.write_edges(0, &[1]).unwrap();
            s.flush().unwrap();
        }
        // Simulate a crash after vectors.bin grew to 8 slots but before graph.bin did.
        OpenOptions::new()
            .write(true)
            .open(dir.path().join(VECTORS_FILE))
            .unwrap()
            .set_len(8 * 8)
            .unwrap();
        let (s, report) = IoUringDiskStore::open_existing(dir.path(), 2, 2, 8).unwrap();
        assert!(report.repaired);
        assert_eq!(report.vectors_capacity, 8);
        assert_eq!(report.graph_capacity, 4);
        assert_eq!(s.capacity(), 8);
        assert_eq!(s.read_edges(0).unwrap(), vec![1]);
        for id in 4..8 {
            assert!(s.read_edges(id).unwrap().is_empty());
        }
        let glen = std::fs::metadata(dir.path().join(GRAPH_FILE))
            .unwrap()
            .len();
        assert_eq!(glen, 8 * 8);
    }

    #[test]
    fn batch_vector_and_edge_operations_cover_multiple_queue_depths() {
        let dir = TempDir::new();
        let mut store = IoUringDiskStore::create_new(dir.path(), 2, 3, 20).unwrap();
        let vectors = (0..10u32)
            .map(|id| (id, vec![id as f32, id as f32 + 0.5]))
            .collect::<Vec<_>>();
        let vector_writes = vectors.iter().map(|(id, values)| (*id, values.as_slice())).collect::<Vec<_>>();
        store.write_vectors_batch(&vector_writes).unwrap();

        let edge_data = (0..10u32)
            .map(|id| (id, vec![(id + 1) % 20, (id + 2) % 20]))
            .collect::<Vec<_>>();
        let edge_writes = edge_data.iter().map(|(id, values)| (*id, values.as_slice())).collect::<Vec<_>>();
        store.write_edges_batch(&edge_writes).unwrap();

        let ids = (0..10u32).collect::<Vec<_>>();
        let read_vectors = store.read_vectors_batch(&ids).unwrap();
        for (index, vector) in read_vectors.iter().enumerate() {
            assert_eq!(vector, &vectors[index].1);
        }
        let read_edges = store.read_edges_batch(&ids).unwrap();
        for (index, edges) in read_edges.iter().enumerate() {
            assert_eq!(edges, &edge_data[index].1);
        }
    }

    #[test]
    fn batch_into_methods_reuse_output_allocations_and_replace_contents() {
        let dir = TempDir::new();
        let mut store = IoUringDiskStore::create_new(dir.path(), 2, 3, 5).unwrap();
        store.write_vector(1, &[4.0, 5.0]).unwrap();
        store.write_edges(1, &[2, 3]).unwrap();

        let ids = [1, 0];
        let mut vectors = vec![vec![99.0; 2], vec![88.0; 2]];
        store.read_vectors_batch_into(&ids, &mut vectors).unwrap();
        assert_eq!(vectors, vec![vec![4.0, 5.0], vec![0.0, 0.0]]);

        let mut edges = vec![vec![99, 98], vec![88]];
        store.read_edges_batch_into(&ids, &mut edges).unwrap();
        assert_eq!(edges, vec![vec![2, 3], vec![]]);

        assert!(matches!(
            store.read_vectors_batch_into(&ids, &mut vectors[..1]),
            Err(StorageError::InvalidArgument(_))
        ));
    }

    #[test]
    fn batch_writes_validate_all_requests_before_submitting() {
        let dir = TempDir::new();
        let mut store = IoUringDiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
        let good = [1.0, 2.0];
        let bad = [3.0];
        let writes = [(0, good.as_slice()), (1, bad.as_slice())];
        assert!(matches!(store.write_vectors_batch(&writes), Err(StorageError::DimensionMismatch { .. })));
        assert_eq!(store.read_vector(0).unwrap(), vec![0.0, 0.0]);

        let duplicate = [(0, good.as_slice()), (0, good.as_slice())];
        assert!(matches!(store.write_vectors_batch(&duplicate), Err(StorageError::InvalidArgument(_))));
    }

    #[test]
    fn empty_batches_are_noops() {
        let dir = TempDir::new();
        let mut store = IoUringDiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
        assert!(store.read_vectors_batch(&[]).unwrap().is_empty());
        assert!(store.read_edges_batch(&[]).unwrap().is_empty());
        store.write_vectors_batch(&[]).unwrap();
        store.write_edges_batch(&[]).unwrap();
    }

    #[test]
    fn io_uring_backend_round_trips_and_reuses_vector_buffer() {
        let dir = TempDir::new();
        {
            let mut store = IoUringDiskStore::create_new(dir.path(), 3, 4, 5).unwrap();
            store.write_vector(2, &[1.25, 2.5, 3.75]).unwrap();
            store.write_edges(2, &[0, 4]).unwrap();
            store.flush().unwrap();
        }
        let (store, report) = IoUringDiskStore::open_existing(dir.path(), 3, 4, 5).unwrap();
        assert!(!report.repaired);
        assert_eq!(store.read_vector(2).unwrap(), vec![1.25, 2.5, 3.75]);
        assert_eq!(store.read_edges(2).unwrap(), vec![0, 4]);
        let mut buffer = vec![99.0; 16];
        store.read_vector_into(2, &mut buffer).unwrap();
        assert_eq!(buffer, vec![1.25, 2.5, 3.75]);
        assert!(store.read_edges(4).unwrap().is_empty());
    }

    #[test]
    fn io_uring_backend_resize_initializes_new_edges_and_reopens() {
        let dir = TempDir::new();
        let mut store = IoUringDiskStore::create_new(dir.path(), 2, 3, 2).unwrap();
        store.write_vector(1, &[8.0, 9.0]).unwrap();
        store.resize(6).unwrap();
        assert_eq!(store.capacity(), 6);
        assert!(store.read_edges(5).unwrap().is_empty());
        drop(store);
        let (store, report) = IoUringDiskStore::open_existing(dir.path(), 2, 3, 6).unwrap();
        assert!(!report.repaired);
        assert_eq!(store.read_vector(1).unwrap(), vec![8.0, 9.0]);
    }

}
