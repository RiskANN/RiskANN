//! File-I/O vector and adjacency storage backend.
//!
//! Uses positional reads and writes (`FileExt::read_at` / `write_at`) and
//! never memory-maps either storage file. The on-disk layout is shared with
//! the mmap backend.

use crate::common::{io_err, OpenReport, StorageError, EMPTY_EDGE, GRAPH_FILE, MAX_CAPACITY, VECTORS_FILE};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

const F32_BYTES: usize = std::mem::size_of::<f32>();
const U32_BYTES: usize = std::mem::size_of::<u32>();

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

pub struct BufferedDiskStore {
    vec_file: File,
    graph_file: File,
    dim: usize,
    r: usize,
    capacity: u32,
    vec_stride: usize,
    graph_stride: usize,
}

impl fmt::Debug for BufferedDiskStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedDiskStore")
            .field("dim", &self.dim)
            .field("r", &self.r)
            .field("capacity", &self.capacity)
            .finish()
    }
}

fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64, context: &str) -> Result<(), StorageError> {
    while !buf.is_empty() {
        match file.read_at(buf, offset) {
            Ok(0) => return Err(io_err(format!("read {context}"))(io::Error::from(io::ErrorKind::UnexpectedEof))),
            Ok(n) => {
                offset = offset.checked_add(n as u64).ok_or(StorageError::Overflow("read offset"))?;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io_err(format!("read {context}"))(e)),
        }
    }
    Ok(())
}

fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64, context: &str) -> Result<(), StorageError> {
    while !buf.is_empty() {
        match file.write_at(buf, offset) {
            Ok(0) => return Err(io_err(format!("write {context}"))(io::Error::from(io::ErrorKind::WriteZero))),
            Ok(n) => {
                offset = offset.checked_add(n as u64).ok_or(StorageError::Overflow("write offset"))?;
                buf = &buf[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io_err(format!("write {context}"))(e)),
        }
    }
    Ok(())
}

/// Fills a file range with a repeated byte using bounded-size writes.
fn fill_file_range(file: &File, start: u64, len: u64, byte: u8, context: &str) -> Result<(), StorageError> {
    const CHUNK: usize = 64 * 1024;
    let buffer = vec![byte; CHUNK];
    let mut written = 0u64;
    while written < len {
        let n = usize::try_from((len - written).min(CHUNK as u64))
            .map_err(|_| StorageError::Overflow("fill chunk size"))?;
        write_all_at(file, &buffer[..n], start.checked_add(written).ok_or(StorageError::Overflow("fill offset"))?, context)?;
        written += n as u64;
    }
    Ok(())
}

impl BufferedDiskStore {
    /// Creates the two storage files. Existing files are never overwritten.
    pub fn create_new(dir: &Path, dim: usize, r: usize, capacity: u32) -> Result<Self, StorageError> {
        let (vec_stride, graph_stride) = strides(dim, r)?;
        validate_capacity(capacity)?;
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
        fill_file_range(&graph_file, 0, graph_len, 0xFF, GRAPH_FILE)?;

        Ok(Self { vec_file, graph_file, dim, r, capacity, vec_stride, graph_stride })
    }

    /// Opens existing files and grows the shorter one if a previous resize was interrupted.
    pub fn open_existing(dir: &Path, dim: usize, r: usize, required_slots: u32) -> Result<(Self, OpenReport), StorageError> {
        let (vec_stride, graph_stride) = strides(dim, r)?;
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
            fill_file_range(&graph_file, graph_len, new_len - graph_len, 0xFF, GRAPH_FILE)?;
        }
        Ok((Self { vec_file, graph_file, dim, r, capacity, vec_stride, graph_stride }, OpenReport {
            vectors_capacity: cap_v, graph_capacity: cap_g, repaired,
        }))
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn r(&self) -> usize { self.r }
    pub fn capacity(&self) -> u32 { self.capacity }

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

    /// Reads a vector into a reusable buffer, resizing it to `dim` if needed.
    pub fn read_vector_into(&self, node_id: u32, out: &mut Vec<f32>) -> Result<(), StorageError> {
        let offset = self.slot_offset(node_id, self.vec_stride)?;
        out.resize(self.dim, 0.0);
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(out.as_mut_slice());
        read_exact_at(&self.vec_file, bytes, offset, VECTORS_FILE)
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
        write_all_at(&self.vec_file, bytemuck::cast_slice(vector), offset, VECTORS_FILE)
    }

    /// Appends the adjacency list to `out` without clearing it.
    pub fn read_edges_into(&self, node_id: u32, out: &mut Vec<u32>) -> Result<(), StorageError> {
        let offset = self.slot_offset(node_id, self.graph_stride)?;
        let mut raw = vec![EMPTY_EDGE; self.r];
        read_exact_at(&self.graph_file, bytemuck::cast_slice_mut(raw.as_mut_slice()), offset, GRAPH_FILE)?;
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
        write_all_at(&self.graph_file, bytemuck::cast_slice(&padded), offset, GRAPH_FILE)
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
        fill_file_range(&self.graph_file, old_graph_len, new_graph_len - old_graph_len, 0xFF, GRAPH_FILE)?;
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
    fn create_write_reopen_preserves_vectors_and_edges() {
        let dir = TempDir::new();
        {
            let mut s = BufferedDiskStore::create_new(dir.path(), 3, 4, 10).unwrap();
            s.write_vector(0, &[1.0, 2.0, 3.0]).unwrap();
            s.write_vector(9, &[7.0, 8.0, 9.0]).unwrap();
            s.write_edges(0, &[9, 3]).unwrap();
            s.flush().unwrap();
        }
        let (s, report) = BufferedDiskStore::open_existing(dir.path(), 3, 4, 10).unwrap();
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
            let mut s = BufferedDiskStore::create_new(dir.path(), 2, 2, 20_000).unwrap();
            s.write_vector(19_999, &[5.0, 6.0]).unwrap();
            s.flush().unwrap();
        }
        let before = std::fs::metadata(dir.path().join(VECTORS_FILE))
            .unwrap()
            .len();
        let (s, _) = BufferedDiskStore::open_existing(dir.path(), 2, 2, 1).unwrap();
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
        let mut s = BufferedDiskStore::create_new(dir.path(), 2, 2, 5).unwrap();
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
        let mut s = BufferedDiskStore::create_new(dir.path(), 3, 2, 2).unwrap();
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
        let mut s = BufferedDiskStore::create_new(dir.path(), 1, 3, 10).unwrap();
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
        let mut s = BufferedDiskStore::create_new(dir.path(), 1, 2, 4).unwrap();
        s.write_edges(0, &[1, 2]).unwrap();
        let mut out = vec![99];
        s.read_edges_into(0, &mut out).unwrap();
        assert_eq!(out, vec![99, 1, 2]);
    }

    #[test]
    fn create_new_refuses_to_overwrite() {
        let dir = TempDir::new();
        let _s = BufferedDiskStore::create_new(dir.path(), 2, 2, 2).unwrap();
        assert!(matches!(
            BufferedDiskStore::create_new(dir.path(), 2, 2, 2),
            Err(StorageError::Io { .. })
        ));
    }

    #[test]
    fn invalid_shapes_rejected() {
        let dir = TempDir::new();
        assert!(BufferedDiskStore::create_new(dir.path(), 0, 2, 2).is_err());
        assert!(BufferedDiskStore::create_new(dir.path(), 2, 0, 2).is_err());
        assert!(BufferedDiskStore::create_new(dir.path(), 2, 2, 0).is_err());
        assert!(BufferedDiskStore::create_new(dir.path(), 2, 2, u32::MAX).is_err());
    }

    #[test]
    fn malformed_file_length_is_corrupt() {
        let dir = TempDir::new();
        {
            BufferedDiskStore::create_new(dir.path(), 3, 2, 4).unwrap();
        }
        let f = OpenOptions::new()
            .write(true)
            .open(dir.path().join(VECTORS_FILE))
            .unwrap();
        f.set_len(4 * 12 + 5).unwrap();
        drop(f);
        assert!(matches!(
            BufferedDiskStore::open_existing(dir.path(), 3, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn empty_file_is_corrupt() {
        let dir = TempDir::new();
        {
            BufferedDiskStore::create_new(dir.path(), 3, 2, 4).unwrap();
        }
        OpenOptions::new()
            .write(true)
            .open(dir.path().join(GRAPH_FILE))
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(matches!(
            BufferedDiskStore::open_existing(dir.path(), 3, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn wrong_shape_on_open_is_corrupt() {
        let dir = TempDir::new();
        {
            BufferedDiskStore::create_new(dir.path(), 3, 2, 4).unwrap(); // 48 bytes of vectors
        }
        // dim 5 -> 20-byte records; 48 is not a multiple.
        assert!(matches!(
            BufferedDiskStore::open_existing(dir.path(), 5, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn metadata_claiming_more_slots_than_files_is_corrupt() {
        let dir = TempDir::new();
        {
            BufferedDiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
        }
        assert!(matches!(
            BufferedDiskStore::open_existing(dir.path(), 2, 2, 5),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn missing_files_are_io_errors() {
        let dir = TempDir::new();
        assert!(matches!(
            BufferedDiskStore::open_existing(dir.path(), 2, 2, 0),
            Err(StorageError::Io { .. })
        ));
    }

    #[test]
    fn resize_preserves_data_and_initialises_new_slots() {
        let dir = TempDir::new();
        let mut s = BufferedDiskStore::create_new(dir.path(), 2, 3, 4).unwrap();
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
        let (s, report) = BufferedDiskStore::open_existing(dir.path(), 2, 3, 9).unwrap();
        assert!(!report.repaired);
        assert_eq!(s.capacity(), 9);
        assert_eq!(s.read_vector(8).unwrap(), &[8.0, 8.5]);
    }

    #[test]
    fn capacity_mismatch_is_repaired_by_growing() {
        let dir = TempDir::new();
        {
            let mut s = BufferedDiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
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
        let (s, report) = BufferedDiskStore::open_existing(dir.path(), 2, 2, 8).unwrap();
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
    fn buffered_backend_round_trips_and_reuses_vector_buffer() {
        let dir = TempDir::new();
        {
            let mut store = BufferedDiskStore::create_new(dir.path(), 3, 4, 5).unwrap();
            store.write_vector(2, &[1.25, 2.5, 3.75]).unwrap();
            store.write_edges(2, &[0, 4]).unwrap();
            store.flush().unwrap();
        }
        let (store, report) = BufferedDiskStore::open_existing(dir.path(), 3, 4, 5).unwrap();
        assert!(!report.repaired);
        assert_eq!(store.read_vector(2).unwrap(), vec![1.25, 2.5, 3.75]);
        assert_eq!(store.read_edges(2).unwrap(), vec![0, 4]);
        let mut buffer = vec![99.0; 16];
        store.read_vector_into(2, &mut buffer).unwrap();
        assert_eq!(buffer, vec![1.25, 2.5, 3.75]);
        assert!(store.read_edges(4).unwrap().is_empty());
    }

    #[test]
    fn buffered_backend_resize_initializes_new_edges_and_reopens() {
        let dir = TempDir::new();
        let mut store = BufferedDiskStore::create_new(dir.path(), 2, 3, 2).unwrap();
        store.write_vector(1, &[8.0, 9.0]).unwrap();
        store.resize(6).unwrap();
        assert_eq!(store.capacity(), 6);
        assert!(store.read_edges(5).unwrap().is_empty());
        drop(store);
        let (store, report) = BufferedDiskStore::open_existing(dir.path(), 2, 3, 6).unwrap();
        assert!(!report.repaired);
        assert_eq!(store.read_vector(1).unwrap(), vec![8.0, 9.0]);
    }

}
