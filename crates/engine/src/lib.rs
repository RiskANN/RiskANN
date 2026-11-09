//! RiskANN engine: collection lifecycle, external-ID mapping, Vamana-inspired
//! index maintenance, approximate and exact retrieval, JSONL ingestion jobs,
//! and deterministic datasets for tests and benchmarks.
//!
//! All engine operations are synchronous. Callers on an async runtime should
//! run them on a blocking thread (the server uses `spawn_blocking`).

pub mod collection;
pub mod dataset;
pub mod db;
pub mod error;
pub mod ingest;

pub use collection::{
    Collection, CollectionConfig, CollectionInfo, GraphNode, GraphSnapshot, QueryOptions,
    QueryStats, SearchHit, SearchResponse, SearchTrace, UpsertOutcome, VectorRecord,
};
pub use db::{DBInfo, LoadReport};
pub use error::EngineError;
pub use ingest::{
    IngestJob, IngestOptions, IngestProgress, IngestSummary, JobRegistry, JobSnapshot, JobState,
    OnError,
};
pub use riskann_core::distances::DistanceMetric;

/// Limits enforced on every request. Documented in the README.
pub mod limits {
    /// Largest `k` accepted by search.
    pub const MAX_K: usize = 1_000;
    /// Default `ef_search` when the client omits it: `max(k, DEFAULT_EF_SEARCH)`.
    pub const DEFAULT_EF_SEARCH: usize = 64;
    /// Largest `ef_search` accepted by search.
    pub const MAX_EF_SEARCH: usize = 4_096;
    /// Largest vector dimension.
    pub const MAX_DIM: usize = 16_384;
    /// Largest graph degree `r`.
    pub const MAX_R: usize = 256;
    /// Largest construction beam `l_build`.
    pub const MAX_L_BUILD: usize = 1_024;
    /// Accepted range for the pruning parameter `alpha`.
    pub const MIN_ALPHA: f32 = 1.0;
    pub const MAX_ALPHA: f32 = 10.0;
    /// Largest `initial_capacity` accepted at creation (storage grows later).
    pub const MAX_INITIAL_CAPACITY: u32 = 50_000_000;
    /// Largest external id, in bytes.
    pub const MAX_EXTERNAL_ID_BYTES: usize = 512;
    /// Longest accepted collection name.
    pub const MAX_COLLECTION_NAME: usize = 64;
    /// Default and maximum number of search entry points per collection.
    /// The number actually used is `min(entry_points, ceil(sqrt(n)))`.
    pub const DEFAULT_ENTRY_POINTS: usize = 64;
    pub const MAX_ENTRY_POINTS: usize = 1_024;
    /// Default and maximum number of nodes in a graph snapshot.
    pub const DEFAULT_GRAPH_SNAPSHOT: usize = 1_000;
    pub const MAX_GRAPH_SNAPSHOT: usize = 5_000;
    /// Longest accepted JSONL line, in bytes.
    pub const MAX_JSONL_LINE_BYTES: usize = 16 * 1024 * 1024;
}
