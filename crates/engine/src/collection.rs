use core::vamana::{greedy_search, robust_prune, Candidate};
pub use core::distances::DistanceMetric;
use storage::DiskStore;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;
use dashmap::DashMap;
use std::sync::Arc;
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde::{Deserialize, Serialize};

const NUM_LOCKS: usize = 1024;

#[derive(Deserialize)]
struct BulkRecord {
    id: String,
    vector: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CollectionConfig {
    pub dim: usize,
    pub metric: DistanceMetric,
    pub r: usize,
    pub alpha: f32,
}

#[derive(Serialize, Deserialize, Default)]
struct CollectionMetadata {
    medoid_id: Option<u32>,
    id_map: HashMap<String, u32>,
    next_id: u32,
}

pub struct DBInfo {
    pub collections: DashMap<String, Arc<Collection>>,
    pub data_loc: String,
}

impl DBInfo {
    pub async fn init(data_loc: String) -> Self {
        let collections = DashMap::new();
        let base_path = PathBuf::from(&data_loc);

        if let Ok(entries) = std::fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                // NEW: Only attempt to load if it's a directory AND has a config.json
                if path.is_dir() && path.join("config.json").exists() {
                    let name = path.file_name().unwrap().to_string_lossy().to_string();
                    let collection = Collection::load(path).await;
                    collections.insert(name.clone(), Arc::new(collection));
                    println!("Loaded existing collection: {}", name);
                } else if path.is_dir() {
                    println!("Skipping corrupted or empty directory: {:?}", path);
                }
            }
        }
        Self { collections, data_loc }
    }
}

pub struct Collection {
    pub path: PathBuf,
    pub config: CollectionConfig,
    pub medoid_id: RwLock<Option<u32>>,
    pub store: DiskStore,
    pub id_map: RwLock<HashMap<String, u32>>, // map external id to internal id 
    pub next_id: std::sync::atomic::AtomicU32,
    locks: Vec<tokio::sync::Mutex<()>>
}

impl Collection {
   
    fn init_locks() -> Vec<tokio::sync::Mutex<()>> {
        let mut locks = Vec::with_capacity(NUM_LOCKS);
        for _ in 0..NUM_LOCKS { locks.push(tokio::sync::Mutex::new(())); }
        locks
    }

    async fn lock_node(&self, id: u32) -> tokio::sync::MutexGuard<'_, ()> {
        self.locks[(id as usize) % NUM_LOCKS].lock().await
    }

    // for persistence
    async fn save_metadata(&self) {
        let meta = CollectionMetadata {
            medoid_id: *self.medoid_id.read().await,
            id_map: self.id_map.read().await.clone(),
            next_id: self.next_id.load(std::sync::atomic::Ordering::SeqCst),
        };
        
        let file_path = self.path.join("metadata.json");
        let temp_path = self.path.join("metadata.json.tmp");
        
        if let Ok(data) = serde_json::to_string(&meta) {
            let _ = std::fs::write(&temp_path, data);
            let _ = std::fs::rename(temp_path, file_path); 
        }
    }

    pub async fn create(path: PathBuf, config: CollectionConfig, capacity: u32) -> Self {
        let store = DiskStore::new(&path, config.dim, config.r, capacity).expect("storage loading failed");
        
        let config_path = path.join("config.json");
        std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let collection = Self {
            path,
            config,
            medoid_id: RwLock::new(None),
            store,
            id_map: RwLock::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicU32::new(0),
            locks: Self::init_locks(),
        };
        collection.save_metadata().await;
        collection
    }

    pub async fn load(path: PathBuf) -> Self {
        let config_path = path.join("config.json");
        let config_bytes = std::fs::read(&config_path).expect("Missing config.json");
        let config: CollectionConfig = serde_json::from_slice(&config_bytes).unwrap();

        let meta_path = path.join("metadata.json");
        let meta: CollectionMetadata = if meta_path.exists() {
            let meta_bytes = std::fs::read(&meta_path).unwrap();
            serde_json::from_slice(&meta_bytes).unwrap_or_default()
        } else {
            CollectionMetadata::default()
        };

        //10k for now, plz change
        let store = DiskStore::new(&path, config.dim, config.r, 10000).unwrap();

        Self {
            path,
            config,
            medoid_id: RwLock::new(meta.medoid_id),
            store,
            id_map: RwLock::new(meta.id_map),
            next_id: std::sync::atomic::AtomicU32::new(meta.next_id),
            locks: Self::init_locks(),
        }
    }

    fn build_candidate_pool(&self, node: u32, edges: &[u32], store: &storage::DiskStore) -> Vec<Candidate> {
            let vec = store.read_vector(node).to_vec();

            edges.iter().map(|&id| Candidate {
                id,
                distance: self.config.metric.calculate(&vec, store.read_vector(id)),
            }).collect()
    }

    pub async fn upsert(&self, internal_id: u32, vector: &[f32]) {
        
        {
            let _lock = self.lock_node(internal_id).await;
            self.store.write_vector(internal_id, vector);
        }

        let medoid = {
            let mut medoid_guard = self.medoid_id.write().await;
            match *medoid_guard {
                None => {
                    *medoid_guard = Some(internal_id);
                    let _lock = self.lock_node(internal_id).await;
                    self.store.write_edges(internal_id, &[]);
                    
                    drop(medoid_guard); 
                    self.save_metadata().await;
                    return;
                }
                Some(m) => m,
            }
        }; 

        let beam_l = self.config.r * 2; 
        let metric = self.config.metric;
        let distance = |a: &[f32], b: &[f32]| metric.calculate(a,b);

        let candidates = greedy_search(
            medoid, vector, beam_l,
            |node, q| distance(self.store.read_vector(node), q),
            |node| self.store.read_edges(node),
        );

        let forward_edges = robust_prune(
            &candidates, self.config.r, self.config.alpha,
            |a, b| distance(self.store.read_vector(a), self.store.read_vector(b)),
        );

        {
            let _lock = self.lock_node(internal_id).await;
            self.store.write_edges(internal_id, &forward_edges);
        }

        for &neighbor in &forward_edges {
            let _neighbor_lock = self.lock_node(neighbor).await;
            
            let mut neighbor_edges = self.store.read_edges(neighbor);
            if !neighbor_edges.contains(&internal_id) {
                neighbor_edges.push(internal_id);
                if neighbor_edges.len() > self.config.r {
                    let pool = self.build_candidate_pool(neighbor, &neighbor_edges, &self.store);
                    let re_pruned = robust_prune(
                        &pool, self.config.r, self.config.alpha,
                        |a, b| metric.calculate(self.store.read_vector(a), self.store.read_vector(b)),
                    );
                    self.store.write_edges(neighbor, &re_pruned);
                } else {
                    self.store.write_edges(neighbor, &neighbor_edges);
                }
            }
        }

        self.save_metadata().await; 
    }
    
    pub async fn bulk_insert(&self, file_path: &str) {
        let file = File::open(file_path).expect("Failed to open bulk file");
        let reader = BufReader::new(file);

        let medoid_exists = self.medoid_id.read().await.is_some();

        if medoid_exists {
            for line_result in reader.lines() {
                let line = line_result.unwrap();
                let record: BulkRecord = serde_json::from_str(&line).unwrap();

                let internal_id = {
                    let mut map = self.id_map.write().await;
                    *map.entry(record.id).or_insert_with(|| {
                        self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                    })
                };
                self.upsert(internal_id, &record.vector).await;
            }
        } else {
            let mut centroid = vec![0.0; self.config.dim];
            let mut count = 0;

            for line_result in reader.lines() {
                let line = line_result.unwrap();
                let record: BulkRecord = serde_json::from_str(&line).unwrap();

                let internal_id = {
                    let mut map = self.id_map.write().await;
                    *map.entry(record.id).or_insert_with(|| {
                        self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                    })
                };

                {
                    let _lock = self.lock_node(internal_id).await;
                    self.store.write_vector(internal_id, &record.vector);
                }

                for (i, &val) in record.vector.iter().enumerate() {
                    centroid[i] += val;
                }
                count += 1;
            }

            if count == 0 { return; }
            for val in centroid.iter_mut() { *val /= count as f32; }

            let mut best_medoid = 0;
            let mut best_dist = f32::MAX;
            let metric = self.config.metric;

            for id in 0..count {
                let vec = self.store.read_vector(id);
                let dist = metric.calculate(&centroid, vec);
                if dist < best_dist {
                    best_dist = dist;
                    best_medoid = id;
                }
            }

            {
                let mut medoid_guard = self.medoid_id.write().await;
                *medoid_guard = Some(best_medoid);
                
                let _lock = self.lock_node(best_medoid).await;
                self.store.write_edges(best_medoid, &[]);
            }

            for id in 0..count {
                if id == best_medoid { continue; }

                let vector = self.store.read_vector(id).to_vec();
                self.upsert(id, &vector).await;
            }
        }

        self.save_metadata().await;
    }
}
