//! Server configuration and shared state.

use engine::{DBInfo, LoadReport};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

pub const DEFAULT_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Searches slower than this (engine time) are logged at `warn`.
pub const DEFAULT_SLOW_QUERY_MS: u64 = 250;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Directory that holds one sub-directory per collection.
    pub data_loc: PathBuf,
    /// If set, ingestion paths must resolve inside this directory.
    pub ingest_root: Option<PathBuf>,
    /// Maximum accepted request body size in bytes.
    pub max_body_bytes: usize,
    /// Slow-query log threshold in milliseconds; `0` disables it.
    pub slow_query_ms: u64,
}

impl ServerConfig {
    /// Defaults for a given data directory (used by tests and `from_env`).
    pub fn for_data_dir(data_loc: impl Into<PathBuf>) -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 3000,
            data_loc: data_loc.into(),
            ingest_root: None,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            slow_query_ms: DEFAULT_SLOW_QUERY_MS,
        }
    }

    /// Reads `HOST`, `PORT`, `DATA_LOC`, `RISKANN_INGEST_DIR`,
    /// `RISKANN_MAX_BODY_BYTES` and `RISKANN_SLOW_QUERY_MS`.
    pub fn from_env() -> Result<Self, String> {
        let mut cfg =
            Self::for_data_dir(std::env::var("DATA_LOC").unwrap_or_else(|_| "/tmp/db_data".into()));
        if let Ok(h) = std::env::var("HOST") {
            cfg.host = h;
        }
        if let Ok(p) = std::env::var("PORT") {
            cfg.port = p
                .parse()
                .map_err(|_| format!("PORT must be a number in 0..=65535, got '{p}'"))?;
        }
        if let Some(dir) = std::env::var("RISKANN_INGEST_DIR")
            .ok()
            .filter(|d| !d.trim().is_empty())
        {
            cfg.ingest_root = Some(PathBuf::from(dir));
        }
        if let Ok(b) = std::env::var("RISKANN_MAX_BODY_BYTES") {
            cfg.max_body_bytes = b.parse().map_err(|_| {
                format!("RISKANN_MAX_BODY_BYTES must be a positive integer, got '{b}'")
            })?;
        }
        if let Ok(ms) = std::env::var("RISKANN_SLOW_QUERY_MS") {
            cfg.slow_query_ms = ms.parse().map_err(|_| {
                format!("RISKANN_SLOW_QUERY_MS must be a non-negative integer, got '{ms}'")
            })?;
        }
        Ok(cfg)
    }
}

pub struct AppState {
    pub db: DBInfo,
    pub config: ServerConfig,
    pub load_report: LoadReport,
    ready: AtomicBool,
}

impl AppState {
    /// Collections are loaded before the state is built, so a new state is
    /// ready. Readiness is withdrawn when graceful shutdown begins.
    pub fn new(db: DBInfo, config: ServerConfig, load_report: LoadReport) -> Self {
        Self {
            db,
            config,
            load_report,
            ready: AtomicBool::new(true),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::SeqCst);
    }
}
