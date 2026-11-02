//! Core algorithms for RiskANN, an experimental Rust-based, Vamana-inspired
//! approximate nearest-neighbour vector database.
//!
//! This crate is storage-agnostic: graph traversal and pruning access vectors
//! and adjacency lists only through caller-supplied closures.

pub mod distances;
pub mod vamana;

pub use distances::{DistanceError, DistanceMetric};
pub use vamana::{
    BeamSearchConfig, BeamSearchOutput, Candidate, SearchError, SearchStats, beam_search,
    robust_prune, vamana_search,
};
