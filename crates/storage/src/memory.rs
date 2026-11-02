//! Memory-mapped vector and adjacency files.
//!
//! # On-disk format (format version 1, unchanged from the original layout)
//!
//! * `vectors.bin` — `capacity` fixed-size slots of `dim` little-endian `f32`
//!   values (`dim * 4` bytes each). Slot `i` holds the vector of internal id
//!   `i`.
//! * `graph.bin` — `capacity` fixed-size slots of `r` `u32` neighbour ids
//!   (`r * 4` bytes each). A list ends at the first [`EMPTY_EDGE`]
//!   (`u32::MAX`); the remainder of the slot is padded with `EMPTY_EDGE`.
//!
//! Values use the host's native byte order (little-endian on all supported
//! targets). Capacity is **derived from the file lengths** on open; it is
//! never taken from a hard-coded constant, and opening a store never shrinks
//! a file.
//!
//! Which slots hold live vectors is decided by the engine (ids
//! `0..next_id` are live); this crate only guarantees bounds-checked access.
//! Freshly created or grown adjacency slots are filled with `EMPTY_EDGE`, so
//! an unused slot reads as "no neighbours" rather than "neighbour 0".
//!
//! # Safety model
//!
//! All reads and writes go through checked byte ranges and safe slices
//! (`bytemuck::try_cast_slice{,_mut}`). Writes require `&mut self`, so the
//! borrow checker guarantees no reader holds a slice into a mapping while it
//! is written or remapped. The only `unsafe` code is the call that creates a
//! mapping; see [`map_file`].

use memmap2::MmapMut;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::ops::Range;
use std::path::Path;

pub const VECTORS_FILE: &str = "vectors.bin";
pub const GRAPH_FILE: &str = "graph.bin";
/// Sentinel that terminates an adjacency list.
pub const EMPTY_EDGE: u32 = u32::MAX;
/// Largest supported capacity; `u32::MAX` is reserved for [`EMPTY_EDGE`].
pub const MAX_CAPACITY: u32 = u32::MAX - 1;

const F32_BYTES: usize = std::mem::size_of::<f32>();
const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Typed storage errors. `Io` contexts may contain file paths; callers that
/// face untrusted clients must log them rather than return them.
#[derive(Debug)]
pub enum StorageError {
    Io {
        context: String,
        source: io::Error,
    },
    InvalidArgument(String),
    OutOfBounds {
        node_id: u32,
        capacity: u32,
    },
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    NonFinite {
        node_id: u32,
    },
    TooManyEdges {
        count: usize,
        max: usize,
    },
    InvalidEdge {
        node_id: u32,
        edge: u32,
        reason: &'static str,
    },
    Corrupt(String),
    Overflow(&'static str),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Io { context, source } => write!(f, "I/O error ({context}): {source}"),
            StorageError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            StorageError::OutOfBounds { node_id, capacity } => {
                write!(
                    f,
                    "node id {node_id} is out of bounds (capacity {capacity})"
                )
            }
            StorageError::DimensionMismatch { expected, got } => {
                write!(f, "vector has {got} components, expected {expected}")
            }
            StorageError::NonFinite { node_id } => {
                write!(f, "vector for node {node_id} contains NaN or infinity")
            }
            StorageError::TooManyEdges { count, max } => {
                write!(f, "{count} edges exceed the maximum degree {max}")
            }
            StorageError::InvalidEdge {
                node_id,
                edge,
                reason,
            } => {
                write!(f, "invalid edge {node_id} -> {edge}: {reason}")
            }
            StorageError::Corrupt(m) => write!(f, "corrupt or inconsistent storage: {m}"),
            StorageError::Overflow(what) => write!(f, "arithmetic overflow computing {what}"),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StorageError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn io_err(context: String) -> impl FnOnce(io::Error) -> StorageError {
    move |source| StorageError::Io { context, source }
}

/// Result of [`DiskStore::open_existing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenReport {
    pub vectors_capacity: u32,
    pub graph_capacity: u32,
    /// The two files had different capacities (e.g. a crash during resize)
    /// and the shorter one was grown to match. Nothing was truncated.
    pub repaired: bool,
}

/// Creates a writable mapping of `file`.
fn map_file(file: &File, context: &str) -> Result<MmapMut, StorageError> {
    // SAFETY: `MmapMut::map_mut` is unsafe because the mapped file could be
    // truncated or modified by another process or handle while the mapping
    // is alive, which would make the mapped memory change underneath safe
    // Rust references (or raise SIGBUS). Invariants upheld here:
    // * The engine opens at most one `DiskStore` per collection directory per
    //   process, and only `DiskStore` touches `vectors.bin`/`graph.bin`.
    // * `DiskStore` never changes a file's length while a mapping of it is
    //   alive: `grow_mapping` drops the old mapping before `set_len`.
    // * All access goes through bounds-checked slices of `mmap.len()` bytes.
    // Modifying collection files externally while the server runs is
    // unsupported and documented as such.
    let mapped = unsafe { MmapMut::map_mut(file) };
    mapped.map_err(io_err(format!("mmap {context}")))
}

/// Replaces `map` with a mapping of `file` after resizing it to `new_len`.
/// The old mapping is dropped *before* the file is resized.
fn grow_mapping(
    file: &File,
    map: &mut MmapMut,
    new_len: u64,
    context: &str,
) -> Result<(), StorageError> {
    let placeholder = MmapMut::map_anon(1).map_err(io_err(format!("placeholder for {context}")))?;
    let old = std::mem::replace(map, placeholder);
    drop(old);
    if let Err(e) = file.set_len(new_len) {
        // Restore a mapping of the (unchanged) file. If even that fails the
        // 1-byte placeholder stays in place; every access is bounds-checked
        // against `map.len()` and therefore returns `Corrupt` rather than
        // touching invalid memory.
        *map = map_file(file, context)?;
        return Err(io_err(format!("resize {context}"))(e));
    }
    *map = map_file(file, context)?;
    Ok(())
}

fn strides(dim: usize, r: usize) -> Result<(usize, usize), StorageError> {
    if dim == 0 {
        return Err(StorageError::InvalidArgument("dim must be > 0".into()));
    }
    if r == 0 {
        return Err(StorageError::InvalidArgument("r must be > 0".into()));
    }
    let vs = dim
        .checked_mul(F32_BYTES)
        .ok_or(StorageError::Overflow("vector record size"))?;
    let gs = r
        .checked_mul(U32_BYTES)
        .ok_or(StorageError::Overflow("graph record size"))?;
    Ok((vs, gs))
}

fn validate_capacity(capacity: u32) -> Result<(), StorageError> {
    if capacity == 0 || capacity > MAX_CAPACITY {
        return Err(StorageError::InvalidArgument(format!(
            "capacity must be in 1..={MAX_CAPACITY}"
        )));
    }
    Ok(())
}

fn byte_len(capacity: u32, stride: usize) -> Result<u64, StorageError> {
    let bytes = (capacity as usize)
        .checked_mul(stride)
        .ok_or(StorageError::Overflow("file size"))?;
    u64::try_from(bytes).map_err(|_| StorageError::Overflow("file size"))
}

fn capacity_from_len(len: u64, stride: usize, name: &str) -> Result<u32, StorageError> {
    let stride64 = stride as u64;
    if len == 0 {
        return Err(StorageError::Corrupt(format!("{name} is empty")));
    }
    if len % stride64 != 0 {
        return Err(StorageError::Corrupt(format!(
            "{name} length {len} is not a multiple of the record size {stride}"
        )));
    }
    if usize::try_from(len).is_err() {
        return Err(StorageError::Corrupt(format!(
            "{name} is too large to map on this platform"
        )));
    }
    let cap = len / stride64;
    if cap > MAX_CAPACITY as u64 {
        return Err(StorageError::Corrupt(format!(
            "{name} holds {cap} slots, more than the supported maximum"
        )));
    }
    Ok(cap as u32)
}

/// Persistent, memory-mapped vector and adjacency storage for one collection.
pub struct DiskStore {
    vec_file: File,
    graph_file: File,
    vectors: MmapMut,
    graph: MmapMut,
    dim: usize,
    r: usize,
    capacity: u32,
    vec_stride: usize,
    graph_stride: usize,
}

impl fmt::Debug for DiskStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiskStore")
            .field("dim", &self.dim)
            .field("r", &self.r)
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl DiskStore {
    /// Creates `vectors.bin` and `graph.bin` in `dir`. Fails if either file
    /// already exists, so existing data can never be overwritten here.
    pub fn create_new(
        dir: &Path,
        dim: usize,
        r: usize,
        capacity: u32,
    ) -> Result<Self, StorageError> {
        let (vec_stride, graph_stride) = strides(dim, r)?;
        validate_capacity(capacity)?;
        let vec_len = byte_len(capacity, vec_stride)?;
        let graph_len = byte_len(capacity, graph_stride)?;

        let vec_path = dir.join(VECTORS_FILE);
        let graph_path = dir.join(GRAPH_FILE);
        let vec_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&vec_path)
            .map_err(io_err(format!("create {}", vec_path.display())))?;
        vec_file
            .set_len(vec_len)
            .map_err(io_err(format!("size {}", vec_path.display())))?;
        let graph_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&graph_path)
            .map_err(io_err(format!("create {}", graph_path.display())))?;
        graph_file
            .set_len(graph_len)
            .map_err(io_err(format!("size {}", graph_path.display())))?;

        let vectors = map_file(&vec_file, VECTORS_FILE)?;
        let mut graph = map_file(&graph_file, GRAPH_FILE)?;
        graph.fill(0xFF); // every slot starts as an empty adjacency list
        graph
            .flush()
            .map_err(io_err(format!("flush {GRAPH_FILE}")))?;

        Ok(Self {
            vec_file,
            graph_file,
            vectors,
            graph,
            dim,
            r,
            capacity,
            vec_stride,
            graph_stride,
        })
    }

    /// Opens existing files without truncating them.
    ///
    /// * Each file length must be a non-zero multiple of its record size.
    /// * Both files must hold at least `required_slots` slots (the number of
    ///   ids the caller's metadata claims); otherwise data is missing and
    ///   `Corrupt` is returned.
    /// * If the two capacities differ (possible after a crash during
    ///   [`DiskStore::resize`]), the shorter file is **grown** to match.
    pub fn open_existing(
        dir: &Path,
        dim: usize,
        r: usize,
        required_slots: u32,
    ) -> Result<(Self, OpenReport), StorageError> {
        let (vec_stride, graph_stride) = strides(dim, r)?;
        let vec_path = dir.join(VECTORS_FILE);
        let graph_path = dir.join(GRAPH_FILE);
        let vec_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&vec_path)
            .map_err(io_err(format!("open {}", vec_path.display())))?;
        let graph_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&graph_path)
            .map_err(io_err(format!("open {}", graph_path.display())))?;
        let vec_len = vec_file
            .metadata()
            .map_err(io_err(format!("stat {}", vec_path.display())))?
            .len();
        let graph_len = graph_file
            .metadata()
            .map_err(io_err(format!("stat {}", graph_path.display())))?
            .len();
        let cap_v = capacity_from_len(vec_len, vec_stride, VECTORS_FILE)?;
        let cap_g = capacity_from_len(graph_len, graph_stride, GRAPH_FILE)?;
        if cap_v < required_slots || cap_g < required_slots {
            return Err(StorageError::Corrupt(format!(
                "metadata references {required_slots} vectors but {VECTORS_FILE} holds {cap_v} \
                 slots and {GRAPH_FILE} holds {cap_g}"
            )));
        }

        let capacity = cap_v.max(cap_g);
        let mut repaired = false;
        if cap_v < capacity {
            vec_file
                .set_len(byte_len(capacity, vec_stride)?)
                .map_err(io_err(format!("grow {}", vec_path.display())))?;
            repaired = true;
        }
        let old_graph_bytes = graph_len as usize;
        if cap_g < capacity {
            graph_file
                .set_len(byte_len(capacity, graph_stride)?)
                .map_err(io_err(format!("grow {}", graph_path.display())))?;
            repaired = true;
        }

        let vectors = map_file(&vec_file, VECTORS_FILE)?;
        let mut graph = map_file(&graph_file, GRAPH_FILE)?;
        if cap_g < capacity {
            graph[old_graph_bytes..].fill(0xFF);
            graph
                .flush()
                .map_err(io_err(format!("flush {GRAPH_FILE}")))?;
        }

        let store = Self {
            vec_file,
            graph_file,
            vectors,
            graph,
            dim,
            r,
            capacity,
            vec_stride,
            graph_stride,
        };
        Ok((
            store,
            OpenReport {
                vectors_capacity: cap_v,
                graph_capacity: cap_g,
                repaired,
            },
        ))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn r(&self) -> usize {
        self.r
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Total bytes occupied by the two data files.
    pub fn disk_usage_bytes(&self) -> u64 {
        (self.vectors.len() as u64) + (self.graph.len() as u64)
    }

    fn slot(
        &self,
        node_id: u32,
        stride: usize,
        map_len: usize,
    ) -> Result<Range<usize>, StorageError> {
        if node_id >= self.capacity {
            return Err(StorageError::OutOfBounds {
                node_id,
                capacity: self.capacity,
            });
        }
        let start = (node_id as usize)
            .checked_mul(stride)
            .ok_or(StorageError::Overflow("slot offset"))?;
        let end = start
            .checked_add(stride)
            .ok_or(StorageError::Overflow("slot end"))?;
        if end > map_len {
            return Err(StorageError::Corrupt(format!(
                "slot {node_id} ends at byte {end} but the mapping has {map_len} bytes"
            )));
        }
        Ok(start..end)
    }

    /// Borrow the vector stored in slot `node_id`.
    pub fn read_vector(&self, node_id: u32) -> Result<&[f32], StorageError> {
        let range = self.slot(node_id, self.vec_stride, self.vectors.len())?;
        bytemuck::try_cast_slice(&self.vectors[range])
            .map_err(|e| StorageError::Corrupt(format!("unaligned vector slot {node_id}: {e:?}")))
    }

    /// Writes `vector` into slot `node_id`. Rejects wrong lengths and
    /// non-finite components.
    pub fn write_vector(&mut self, node_id: u32, vector: &[f32]) -> Result<(), StorageError> {
        if vector.len() != self.dim {
            return Err(StorageError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(StorageError::NonFinite { node_id });
        }
        let range = self.slot(node_id, self.vec_stride, self.vectors.len())?;
        let bytes: &[u8] = bytemuck::cast_slice(vector);
        self.vectors[range].copy_from_slice(bytes);
        Ok(())
    }

    /// Appends the adjacency list of `node_id` to `out` (does not clear it).
    pub fn read_edges_into(&self, node_id: u32, out: &mut Vec<u32>) -> Result<(), StorageError> {
        let range = self.slot(node_id, self.graph_stride, self.graph.len())?;
        let raw: &[u32] = bytemuck::try_cast_slice(&self.graph[range])
            .map_err(|e| StorageError::Corrupt(format!("unaligned graph slot {node_id}: {e:?}")))?;
        out.extend(raw.iter().copied().take_while(|&e| e != EMPTY_EDGE));
        Ok(())
    }

    /// Returns the adjacency list of `node_id`.
    pub fn read_edges(&self, node_id: u32) -> Result<Vec<u32>, StorageError> {
        let mut out = Vec::with_capacity(self.r);
        self.read_edges_into(node_id, &mut out)?;
        Ok(out)
    }

    /// Replaces the adjacency list of `node_id`. Rejects more than `r`
    /// edges, self-edges, duplicates and ids outside the capacity.
    pub fn write_edges(&mut self, node_id: u32, edges: &[u32]) -> Result<(), StorageError> {
        if edges.len() > self.r {
            return Err(StorageError::TooManyEdges {
                count: edges.len(),
                max: self.r,
            });
        }
        for (i, &e) in edges.iter().enumerate() {
            if e == EMPTY_EDGE || e >= self.capacity {
                return Err(StorageError::InvalidEdge {
                    node_id,
                    edge: e,
                    reason: "target out of range",
                });
            }
            if e == node_id {
                return Err(StorageError::InvalidEdge {
                    node_id,
                    edge: e,
                    reason: "self-edge",
                });
            }
            if edges[..i].contains(&e) {
                return Err(StorageError::InvalidEdge {
                    node_id,
                    edge: e,
                    reason: "duplicate edge",
                });
            }
        }
        let range = self.slot(node_id, self.graph_stride, self.graph.len())?;
        let dst: &mut [u32] = bytemuck::try_cast_slice_mut(&mut self.graph[range])
            .map_err(|e| StorageError::Corrupt(format!("unaligned graph slot {node_id}: {e:?}")))?;
        dst[..edges.len()].copy_from_slice(edges);
        dst[edges.len()..].fill(EMPTY_EDGE);
        Ok(())
    }

    /// Flushes both mappings to disk (`msync`).
    pub fn flush(&self) -> Result<(), StorageError> {
        self.vectors
            .flush()
            .map_err(io_err(format!("flush {VECTORS_FILE}")))?;
        self.graph
            .flush()
            .map_err(io_err(format!("flush {GRAPH_FILE}")))?;
        Ok(())
    }

    /// Grows both files to `new_capacity` slots and remaps them.
    ///
    /// Order: flush; grow `vectors.bin`; grow `graph.bin` and fill the new
    /// adjacency slots with `EMPTY_EDGE`; flush. `capacity()` changes only
    /// after both files have grown. A crash in between leaves two files of
    /// different lengths, which [`DiskStore::open_existing`] repairs by
    /// growing the shorter one. Shrinking is not supported.
    pub fn resize(&mut self, new_capacity: u32) -> Result<(), StorageError> {
        validate_capacity(new_capacity)?;
        if new_capacity <= self.capacity {
            return Err(StorageError::InvalidArgument(format!(
                "new capacity {new_capacity} must exceed current capacity {}",
                self.capacity
            )));
        }
        let vec_len = byte_len(new_capacity, self.vec_stride)?;
        let graph_len = byte_len(new_capacity, self.graph_stride)?;
        self.flush()?;

        grow_mapping(&self.vec_file, &mut self.vectors, vec_len, VECTORS_FILE)?;
        let old_graph_bytes = self.graph.len();
        grow_mapping(&self.graph_file, &mut self.graph, graph_len, GRAPH_FILE)?;
        if old_graph_bytes <= self.graph.len() {
            self.graph[old_graph_bytes..].fill(0xFF);
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
    fn create_write_reopen_preserves_vectors_and_edges() {
        let dir = TempDir::new();
        {
            let mut s = DiskStore::create_new(dir.path(), 3, 4, 10).unwrap();
            s.write_vector(0, &[1.0, 2.0, 3.0]).unwrap();
            s.write_vector(9, &[7.0, 8.0, 9.0]).unwrap();
            s.write_edges(0, &[9, 3]).unwrap();
            s.flush().unwrap();
        }
        let (s, report) = DiskStore::open_existing(dir.path(), 3, 4, 10).unwrap();
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
            let mut s = DiskStore::create_new(dir.path(), 2, 2, 20_000).unwrap();
            s.write_vector(19_999, &[5.0, 6.0]).unwrap();
            s.flush().unwrap();
        }
        let before = std::fs::metadata(dir.path().join(VECTORS_FILE))
            .unwrap()
            .len();
        let (s, _) = DiskStore::open_existing(dir.path(), 2, 2, 1).unwrap();
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
        let mut s = DiskStore::create_new(dir.path(), 2, 2, 5).unwrap();
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
        let mut s = DiskStore::create_new(dir.path(), 3, 2, 2).unwrap();
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
        let mut s = DiskStore::create_new(dir.path(), 1, 3, 10).unwrap();
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
        let mut s = DiskStore::create_new(dir.path(), 1, 2, 4).unwrap();
        s.write_edges(0, &[1, 2]).unwrap();
        let mut out = vec![99];
        s.read_edges_into(0, &mut out).unwrap();
        assert_eq!(out, vec![99, 1, 2]);
    }

    #[test]
    fn create_new_refuses_to_overwrite() {
        let dir = TempDir::new();
        let _s = DiskStore::create_new(dir.path(), 2, 2, 2).unwrap();
        assert!(matches!(
            DiskStore::create_new(dir.path(), 2, 2, 2),
            Err(StorageError::Io { .. })
        ));
    }

    #[test]
    fn invalid_shapes_rejected() {
        let dir = TempDir::new();
        assert!(DiskStore::create_new(dir.path(), 0, 2, 2).is_err());
        assert!(DiskStore::create_new(dir.path(), 2, 0, 2).is_err());
        assert!(DiskStore::create_new(dir.path(), 2, 2, 0).is_err());
        assert!(DiskStore::create_new(dir.path(), 2, 2, u32::MAX).is_err());
    }

    #[test]
    fn malformed_file_length_is_corrupt() {
        let dir = TempDir::new();
        {
            DiskStore::create_new(dir.path(), 3, 2, 4).unwrap();
        }
        let f = OpenOptions::new()
            .write(true)
            .open(dir.path().join(VECTORS_FILE))
            .unwrap();
        f.set_len(4 * 12 + 5).unwrap();
        drop(f);
        assert!(matches!(
            DiskStore::open_existing(dir.path(), 3, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn empty_file_is_corrupt() {
        let dir = TempDir::new();
        {
            DiskStore::create_new(dir.path(), 3, 2, 4).unwrap();
        }
        OpenOptions::new()
            .write(true)
            .open(dir.path().join(GRAPH_FILE))
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(matches!(
            DiskStore::open_existing(dir.path(), 3, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn wrong_shape_on_open_is_corrupt() {
        let dir = TempDir::new();
        {
            DiskStore::create_new(dir.path(), 3, 2, 4).unwrap(); // 48 bytes of vectors
        }
        // dim 5 -> 20-byte records; 48 is not a multiple.
        assert!(matches!(
            DiskStore::open_existing(dir.path(), 5, 2, 0),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn metadata_claiming_more_slots_than_files_is_corrupt() {
        let dir = TempDir::new();
        {
            DiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
        }
        assert!(matches!(
            DiskStore::open_existing(dir.path(), 2, 2, 5),
            Err(StorageError::Corrupt(_))
        ));
    }

    #[test]
    fn missing_files_are_io_errors() {
        let dir = TempDir::new();
        assert!(matches!(
            DiskStore::open_existing(dir.path(), 2, 2, 0),
            Err(StorageError::Io { .. })
        ));
    }

    #[test]
    fn resize_preserves_data_and_initialises_new_slots() {
        let dir = TempDir::new();
        let mut s = DiskStore::create_new(dir.path(), 2, 3, 4).unwrap();
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
        let (s, report) = DiskStore::open_existing(dir.path(), 2, 3, 9).unwrap();
        assert!(!report.repaired);
        assert_eq!(s.capacity(), 9);
        assert_eq!(s.read_vector(8).unwrap(), &[8.0, 8.5]);
    }

    #[test]
    fn capacity_mismatch_is_repaired_by_growing() {
        let dir = TempDir::new();
        {
            let mut s = DiskStore::create_new(dir.path(), 2, 2, 4).unwrap();
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
        let (s, report) = DiskStore::open_existing(dir.path(), 2, 2, 4).unwrap();
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
}
