//! Synchronous Linux O_DIRECT vector and adjacency storage backend.
//!
//! Arbitrary record ranges are served through aligned read/modify/write
//! bounce buffers, preserving the existing on-disk record layout.

use crate::common::{io_err, OpenReport, StorageError, EMPTY_EDGE, GRAPH_FILE, MAX_CAPACITY, VECTORS_FILE};
use crate::direct_io::{AlignedBuffer, DirectIoAlignment};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
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

pub struct DirectIoDiskStore {
    vec_file: File,
    graph_file: File,
    dim: usize,
    r: usize,
    capacity: u32,
    vec_stride: usize,
    graph_stride: usize,
    vec_alignment: DirectIoAlignment,
    graph_alignment: DirectIoAlignment,
}

impl fmt::Debug for DirectIoDiskStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DirectIoDiskStore")
            .field("dim", &self.dim)
            .field("r", &self.r)
            .field("capacity", &self.capacity)
            .finish()
    }
}

fn open_direct(path: &Path, create_new: bool) -> Result<File, StorageError> {
    let mut options = OpenOptions::new();
    options.read(true)
        .write(true)
        .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC);

    if create_new {
        options.create_new(true);
    }

    options.open(path)
        .map_err(io_err(format!("open O_DIRECT {}", path.display())))
}

/// Reads an arbitrary logical range using an aligned O_DIRECT request.
/// Bytes beyond EOF are zero-filled; callers only consume bytes within the
/// logical file length.

fn direct_read_exact_at(
    file: &File,
    alignment: DirectIoAlignment,
    out: &mut [u8],
    offset: u64,
    context: &str,
) -> Result<(), StorageError> {
    if out.is_empty() {
        return Ok(());
    }

    let (aligned_offset, aligned_len) =
        alignment.aligned_range(offset, out.len())?;

    let mut bounce =
        AlignedBuffer::new(aligned_len, alignment.memory())?;

    let mut done = 0usize;

    while done < aligned_len {
        let remaining = aligned_len - done;

        match file.read_at(
            &mut bounce[done..],
            aligned_offset + done as u64,
        ) {
            Ok(0) => break,

            Ok(n) => {
                done += n;

                if n < remaining {
                    break;
                }
            }

            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,

            Err(e) => {
                return Err(io_err(format!("read {context}"))(e));
            }
        }
    }

    let within = usize::try_from(offset - aligned_offset)
        .map_err(|_| StorageError::Overflow("aligned read offset"))?;

    let end = within
        .checked_add(out.len())
        .ok_or(StorageError::Overflow("aligned read range"))?;

    if done < end {
        let file_len = file
            .metadata()
            .map_err(io_err(format!("metadata {context}")))?
            .len();
        eprintln!(
            "direct read diagnostic: context={context}, \
             file_len={file_len}, offset={offset}, \
             out_len={}, aligned_offset={aligned_offset}, \
             aligned_len={aligned_len}, bytes_read={done}, \
             required={end}",
            out.len(),
        );

        return Err(io_err(format!("read {context}"))(
            io::Error::from(io::ErrorKind::UnexpectedEof),
        ));
    }

    out.copy_from_slice(&bounce[within..end]);

    Ok(())
}

/// Writes an arbitrary logical range with an aligned read/modify/write.
/// The temporary aligned extension is truncated back to the original logical
/// length after the direct write, so the existing file format is unchanged.

fn direct_write_all_at(
    file: &File,
    alignment: DirectIoAlignment,
    input: &[u8],
    offset: u64,
    context: &str,
) -> Result<(), StorageError> {
    if input.is_empty() {
        return Ok(());
    }

    let (aligned_offset, aligned_len) =
        alignment.aligned_range(offset, input.len())?;

    let original_len = file
        .metadata()
        .map_err(io_err(format!("stat {context}")))?
        .len();

    let logical_end = offset
        .checked_add(input.len() as u64)
        .ok_or(StorageError::Overflow("logical write end"))?;

    let final_len = original_len.max(logical_end);

    let aligned_end = aligned_offset
        .checked_add(aligned_len as u64)
        .ok_or(StorageError::Overflow("aligned write end"))?;

    let mut bounce =
        AlignedBuffer::new(aligned_len, alignment.memory())?;

    // Read existing bytes to preserve data outside the requested range.
    // The buffer is zero-initialized, so bytes beyond EOF remain zero.
    let mut done = 0usize;

    while done < aligned_len {
        let remaining = aligned_len - done;

        match file.read_at(
            &mut bounce[done..],
            aligned_offset + done as u64,
        ) {
            Ok(0) => break,

            Ok(n) => {
                done += n;
                if n < remaining {
                    break;
                }
            }

            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,

            Err(e) => {
                return Err(io_err(format!("read before write {context}"))(e));
            }
        }
    }

    let existing = usize::try_from(
        original_len.saturating_sub(aligned_offset),
    )
    .unwrap_or(usize::MAX)
    .min(aligned_len);

    if done < existing {
        return Err(io_err(format!(
            "read existing bytes before write {context}"
        ))(io::Error::from(io::ErrorKind::UnexpectedEof)));
    }

    let within = usize::try_from(offset - aligned_offset)
        .map_err(|_| StorageError::Overflow("aligned write offset"))?;

    let end = within
        .checked_add(input.len())
        .ok_or(StorageError::Overflow("aligned write range"))?;

    bounce[within..end].copy_from_slice(input);

    // Extend to an aligned size before issuing O_DIRECT.
    if aligned_end > original_len {
        file.set_len(aligned_end)
            .map_err(io_err(format!("extend for O_DIRECT write {context}")))?;
    }
    let result = loop {
        match file.write_at(&bounce, aligned_offset) {
            Ok(n) if n == aligned_len => break Ok(()),

            Ok(_) => {
                break Err(io_err(format!("write {context}"))(
                    io::Error::new(
                        io::ErrorKind::WriteZero,
                        "O_DIRECT returned a partial aligned write",
                    ),
                ));
            }

            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,

            Err(e) => {
                break Err(io_err(format!("write {context}"))(e));
            }
        }
    };
    // Restore the logical length, preserving any newly written data.
    if aligned_end > original_len {
        let truncate = file
            .set_len(final_len)
            .map_err(io_err(format!(
                "restore logical length after O_DIRECT write {context}"
            )));

        match (result, truncate) {
            (Err(error), _) => return Err(error),
            (Ok(()), Err(error)) => return Err(error),
            (Ok(()), Ok(())) => {}
        }
    } else {
        result?;
    }

    Ok(())
}

fn fill_file_range(
    file: &File,
    alignment: DirectIoAlignment,
    start: u64,
    len: u64,
    byte: u8,
    context: &str,
) -> Result<(), StorageError> {
    const CHUNK: usize = 64 * 1024;
    let buffer = vec![byte; CHUNK];
    let mut written = 0u64;
    while written < len {
        let n = usize::try_from((len - written).min(CHUNK as u64))
            .map_err(|_| StorageError::Overflow("fill chunk size"))?;
        direct_write_all_at(file, alignment, &buffer[..n],
            start.checked_add(written).ok_or(StorageError::Overflow("fill offset"))?, context)?;
        written += n as u64;
    }
    Ok(())
}

impl DirectIoDiskStore {
    /// Creates the two storage files. Existing files are never overwritten.
    
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

        let vec_file = open_direct(&vec_path, true)?;
        let graph_file = open_direct(&graph_path, true)?;

        // Establish the full logical vector-file length.
        // Newly extended regions read as zero.
        vec_file
            .set_len(vec_len)
            .map_err(io_err(format!("initialize {VECTORS_FILE}")))?;

        let vec_alignment = DirectIoAlignment::discover(&vec_file)?;
        let graph_alignment = DirectIoAlignment::discover(&graph_file)?;

        fill_file_range(
            &graph_file,
            graph_alignment,
            0,
            graph_len,
            0xFF,
            GRAPH_FILE,
        )?;

        Ok(Self {
            vec_file,
            graph_file,
            dim,
            r,
            capacity,
            vec_stride,
            graph_stride,
            vec_alignment,
            graph_alignment,
        })
    }

    
    /// Opens existing files and grows the shorter one if a previous resize was interrupted.
    pub fn open_existing(
        dir: &Path,
        dim: usize,
        r: usize,
        required_slots: u32,
    ) -> Result<(Self, OpenReport), StorageError> {
        let (vec_stride, graph_stride) = strides(dim, r)?;
        let vec_path = dir.join(VECTORS_FILE);
        let graph_path = dir.join(GRAPH_FILE);

        let vec_file = open_direct(&vec_path, false)?;
        let graph_file = open_direct(&graph_path, false)?;

        let vec_alignment = DirectIoAlignment::discover(&vec_file)?;
        let graph_alignment = DirectIoAlignment::discover(&graph_file)?;

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
            vec_file
                .set_len(byte_len(capacity, vec_stride)?)
                .map_err(io_err(format!("grow {}", vec_path.display())))?;
        }

        if cap_g < capacity {
            let new_len = byte_len(capacity, graph_stride)?;
            graph_file
                .set_len(new_len)
                .map_err(io_err(format!("grow {}", graph_path.display())))?;

            fill_file_range(
                &graph_file,
                graph_alignment,
                graph_len,
                new_len - graph_len,
                0xFF,
                GRAPH_FILE,
            )?;
        }

        Ok((
            Self {
                vec_file,
                graph_file,
                dim,
                r,
                capacity,
                vec_stride,
                graph_stride,
                vec_alignment,
                graph_alignment,
            },
            OpenReport {
                vectors_capacity: cap_v,
                graph_capacity: cap_g,
                repaired,
            },
        ))
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
        direct_read_exact_at(&self.vec_file, self.vec_alignment, bytes, offset, VECTORS_FILE)
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
        direct_write_all_at(&self.vec_file, self.vec_alignment, bytemuck::cast_slice(vector), offset, VECTORS_FILE)
    }

    /// Appends the adjacency list to `out` without clearing it.
    pub fn read_edges_into(&self, node_id: u32, out: &mut Vec<u32>) -> Result<(), StorageError> {
        let offset = self.slot_offset(node_id, self.graph_stride)?;
        let mut raw = vec![EMPTY_EDGE; self.r];
        direct_read_exact_at(&self.graph_file, self.graph_alignment, bytemuck::cast_slice_mut(raw.as_mut_slice()), offset, GRAPH_FILE)?;
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
        direct_write_all_at(&self.graph_file, self.graph_alignment, bytemuck::cast_slice(&padded), offset, GRAPH_FILE)
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
        fill_file_range(&self.graph_file, self.graph_alignment, old_graph_len, new_graph_len - old_graph_len, 0xFF, GRAPH_FILE)?;
        self.capacity = new_capacity;
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let path = std::env::temp_dir().join(format!("riskann-direct-{unique}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn direct_backend_round_trips_vectors_and_edges() {
        let dir = test_dir();

        let mut store =
            DirectIoDiskStore::create_new(&dir, 4, 4, 8).unwrap();

        let vector = vec![1.0, 2.0, 3.0, 4.0];
        store.write_vector(2, &vector).unwrap();
        store.write_edges(2, &[1, 3, 5]).unwrap();
        store.flush().unwrap();

        assert_eq!(store.read_vector(2).unwrap(), vector);
        assert_eq!(store.read_edges(2).unwrap(), vec![1, 3, 5]);

        drop(store);

        let (store, report) =
            DirectIoDiskStore::open_existing(&dir, 4, 4, 8).unwrap();

        assert!(!report.repaired);
        assert_eq!(store.read_vector(2).unwrap(), vector);
        assert_eq!(store.read_edges(2).unwrap(), vec![1, 3, 5]);

        drop(store);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
