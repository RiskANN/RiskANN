//! Registry of collections under one data directory.

use crate::collection::{
    CONFIG_FILE, Collection, CollectionConfig, CollectionInfo, validate_collection_name,
};
use crate::error::EngineError;
use crate::ingest::JobRegistry;
use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Outcome of loading the data directory at startup.
#[derive(Debug, Default, Clone)]
pub struct LoadReport {
    pub loaded: Vec<String>,
    /// `(directory name, error)` for collections that failed to open. They
    /// stay on disk untouched and are not served.
    pub failed: Vec<(String, String)>,
    /// Directories ignored because they are not collections.
    pub skipped: Vec<String>,
}

pub struct DBInfo {
    pub collections: DashMap<String, Arc<Collection>>,
    pub data_loc: PathBuf,
    pub jobs: JobRegistry,
    /// Serialises create/delete so name checks and directory changes are atomic.
    admin: Mutex<()>,
}

impl std::fmt::Debug for DBInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DBInfo")
            .field("data_loc", &self.data_loc)
            .field("collections", &self.collections.len())
            .finish()
    }
}

impl DBInfo {
    /// Opens (creating if needed) `data_loc` and loads every collection in it.
    /// A collection that fails to open is reported and skipped; it is never
    /// overwritten or re-initialised.
    pub fn open(data_loc: impl Into<PathBuf>) -> Result<(Self, LoadReport), EngineError> {
        let data_loc = data_loc.into();
        std::fs::create_dir_all(&data_loc)
            .map_err(EngineError::io(format!("create {}", data_loc.display())))?;
        let mut report = LoadReport::default();
        let collections = DashMap::new();

        let mut entries: Vec<PathBuf> = std::fs::read_dir(&data_loc)
            .map_err(EngineError::io(format!("read {}", data_loc.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        entries.sort();

        for path in entries {
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            if validate_collection_name(&name).is_err() || !path.join(CONFIG_FILE).exists() {
                tracing::info!(dir = %path.display(), "skipping directory that is not a collection");
                report.skipped.push(name);
                continue;
            }
            match Collection::open(path.clone()) {
                Ok(c) => {
                    tracing::debug!(collection = %name, "registered collection");
                    collections.insert(name.clone(), Arc::new(c));
                    report.loaded.push(name);
                }
                Err(e) => {
                    tracing::error!(
                        collection = %name,
                        error = %e,
                        "failed to load collection; it is left untouched on disk and will not be served"
                    );
                    report.failed.push((name, e.to_string()));
                }
            }
        }
        Ok((
            Self {
                collections,
                data_loc,
                jobs: JobRegistry::new(),
                admin: Mutex::new(()),
            },
            report,
        ))
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_loc
    }

    fn admin_lock(&self) -> std::sync::MutexGuard<'_, ()> {
        match self.admin.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    pub fn get(&self, name: &str) -> Result<Arc<Collection>, EngineError> {
        validate_collection_name(name)?;
        self.collections
            .get(name)
            .map(|r| Arc::clone(r.value()))
            .ok_or_else(|| EngineError::CollectionNotFound(name.to_string()))
    }

    /// Creates a collection. Fails with `CollectionExists` if the name is
    /// loaded *or* a directory of that name already exists on disk (for
    /// example a collection that failed to load), so data is never replaced.
    pub fn create_collection(
        &self,
        name: &str,
        config: CollectionConfig,
        initial_capacity: u32,
    ) -> Result<Arc<Collection>, EngineError> {
        validate_collection_name(name)?;
        let _guard = self.admin_lock();
        if self.collections.contains_key(name) {
            return Err(EngineError::CollectionExists(format!(
                "collection '{name}' already exists"
            )));
        }
        let dir = self.data_loc.join(name);
        if dir.exists() {
            return Err(EngineError::CollectionExists(format!(
                "a directory for '{name}' already exists in the data directory but is not loaded"
            )));
        }
        let (dim, metric, r, alpha) = (config.dim, config.metric, config.r, config.alpha);
        let l_build = config.effective_l_build();
        let collection = Arc::new(Collection::create(dir, name, config, initial_capacity)?);
        self.collections
            .insert(name.to_string(), Arc::clone(&collection));
        tracing::info!(
            collection = %name,
            dim,
            metric = metric.name(),
            r,
            alpha,
            l_build,
            initial_capacity,
            "created collection"
        );
        Ok(collection)
    }

    /// Unregisters the collection, waits for in-flight writes, and removes
    /// its directory.
    pub fn delete_collection(&self, name: &str) -> Result<(), EngineError> {
        validate_collection_name(name)?;
        let _guard = self.admin_lock();
        let (_, collection) = self
            .collections
            .remove(name)
            .ok_or_else(|| EngineError::CollectionNotFound(name.to_string()))?;
        tracing::info!(collection = %name, "deleting collection: waiting for in-flight writes");
        collection.mark_deleted_and_wait();
        std::fs::remove_dir_all(collection.dir()).map_err(EngineError::io(format!(
            "remove {}",
            collection.dir().display()
        )))?;
        tracing::info!(collection = %name, "deleted collection");
        Ok(())
    }

    /// Summaries of all loaded collections, sorted by name.
    pub fn list(&self) -> Result<Vec<CollectionInfo>, EngineError> {
        let mut all: Vec<Arc<Collection>> = self
            .collections
            .iter()
            .map(|r| Arc::clone(r.value()))
            .collect();
        all.sort_by(|a, b| a.name().cmp(b.name()));
        all.iter().map(|c| c.info()).collect()
    }
}
