use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;
use std::io::Result;

pub struct DiskStore {
    vectors_mmap: MmapMut,
    graph_mmap: MmapMut,
    pub dim: usize,
    pub r: usize,
}

impl DiskStore {
    pub fn new(dir: &Path, dim: usize, r: usize, capacity: u32) -> Result<Self> {
        let vec_path = dir.join("vectors.bin");
        let graph_path = dir.join("graph.bin");

        let vec_size = (capacity as usize) * dim * 4;
        let graph_size = (capacity as usize) * r * 4;

        let vec_file = OpenOptions::new().read(true).write(true).create(true).open(&vec_path)?;
        vec_file.set_len(vec_size as u64)?;

        let graph_file = OpenOptions::new().read(true).write(true).create(true).open(&graph_path)?;
        graph_file.set_len(graph_size as u64)?;

        let vectors_mmap = unsafe { MmapMut::map_mut(&vec_file)? };
        let graph_mmap = unsafe { MmapMut::map_mut(&graph_file)? };

        Ok(Self {vectors_mmap, graph_mmap, dim, r:r})
    }


    pub fn write_vector(&self, node_id: u32, vector: &[f32]) {
        let start = (node_id as usize) * self.dim * 4;
        let bytes: &[u8] = bytemuck::cast_slice(vector);

        // q_q
        unsafe {
            let ptr = self.vectors_mmap.as_ptr().add(start) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        }
    }

    pub fn write_edges(&self, node_id: u32, edges: &[u32]) {
        let mut padded = vec![u32::MAX; self.r];
        padded[..edges.len()].copy_from_slice(edges);

        let start = (node_id as usize) * self.r * 4;
        let bytes: &[u8] = bytemuck::cast_slice(&padded);

        unsafe {
            let ptr = self.graph_mmap.as_ptr().add(start) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        }
    }

    pub fn read_vector(&self, node_id: u32) -> &[f32] {
        let start = (node_id as usize) * self.dim * 4;
        let end = start + (self.dim * 4);
        bytemuck::cast_slice(&self.vectors_mmap[start..end])
    }

    pub fn read_edges(&self, node_id: u32) -> Vec<u32> {
        let start = (node_id as usize) * self.r * 4;
        let end = start + (self.r * 4);
        let raw_slice: &[u32] = bytemuck::cast_slice(&self.graph_mmap[start..end]);

        raw_slice
            .iter()
            .copied()
            .take_while(|&id| id != u32::MAX)
            .collect()
    }

}
