//! Engine-level integration tests. Every test uses its own temporary
//! directory; nothing depends on `/tmp/db_data` or other local state.

use engine::collection::{Collection, CollectionConfig, QueryOptions, SearchResponse};
use engine::dataset::{self, DatasetSpec, Distribution};
use engine::ingest::{
    IngestMode, IngestOptions, IngestProgress, IngestRecord, IngestSummary, JobState, OnError,
    records_source, run_ingest_job,
};
use engine::{DBInfo, DistanceMetric, EngineError};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
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
            "riskann-engine-test-{}-{}-{}",
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

fn create(
    dir: &TempDir,
    name: &str,
    dim: usize,
    metric: DistanceMetric,
    r: usize,
    cap: u32,
) -> Collection {
    Collection::create(
        dir.path().join(name),
        name,
        CollectionConfig::new(dim, metric, r, 1.2),
        cap,
    )
    .unwrap()
}

fn ext_ids(resp: &SearchResponse) -> Vec<String> {
    resp.hits.iter().map(|h| h.external_id.clone()).collect()
}

fn internal_ids(resp: &SearchResponse) -> Vec<u32> {
    resp.hits.iter().map(|h| h.internal_id).collect()
}

fn bulk(c: &Collection, vectors: &[Vec<f32>], offset: usize) -> IngestSummary {
    let recs = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| IngestRecord {
            id: format!("v{}", i + offset),
            vector: v.clone(),
        })
        .collect();
    c.ingest(
        records_source(recs),
        &IngestOptions::default(),
        &IngestProgress::default(),
    )
}

fn assert_sorted_unique(resp: &SearchResponse) {
    for w in resp.hits.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        assert!(
            a.distance < b.distance || (a.distance == b.distance && a.internal_id < b.internal_id),
            "not ordered: {:?}",
            resp.hits
        );
    }
    let set: HashSet<u32> = internal_ids(resp).into_iter().collect();
    assert_eq!(set.len(), resp.hits.len(), "duplicate results");
}

/// Mean Recall@K of ANN search against exact search on the same collection.
fn recall_at_k(c: &Collection, queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
    let n = c.len().unwrap() as usize;
    let mut total = 0.0;
    for q in queries {
        let ann = c.search(q, &QueryOptions::new(k, Some(ef))).unwrap();
        let exact = c.exact_search(q, k, false).unwrap();
        assert_sorted_unique(&ann);
        let truth: HashSet<u32> = internal_ids(&exact).into_iter().collect();
        let hit = internal_ids(&ann)
            .iter()
            .filter(|id| truth.contains(id))
            .count();
        total += hit as f64 / k.min(n) as f64;
    }
    total / queries.len() as f64
}

fn spec(n: usize, dim: usize) -> DatasetSpec {
    DatasetSpec {
        n,
        dim,
        seed: 42,
        distribution: Distribution::Clustered { clusters: 32 },
    }
}

// ---------------------------------------------------------------------
// Basic retrieval
// ---------------------------------------------------------------------

#[test]
fn empty_collection_returns_empty_results() {
    let dir = TempDir::new();
    let c = create(&dir, "empty", 3, DistanceMetric::L2, 8, 10);
    let r = c
        .search(&[0.0, 0.0, 0.0], &QueryOptions::new(5, None))
        .unwrap();
    assert!(r.hits.is_empty());
    assert_eq!(r.stats.returned, 0);
    assert!(
        c.exact_search(&[0.0, 0.0, 0.0], 5, false)
            .unwrap()
            .hits
            .is_empty()
    );
}

#[test]
fn single_vector_collection() {
    let dir = TempDir::new();
    let c = create(&dir, "one", 3, DistanceMetric::L2, 8, 10);
    let o = c.upsert("a", &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!((o.internal_id, o.created), (0, true));
    let r = c
        .search(&[0.0, 0.0, 0.0], &QueryOptions::new(5, None))
        .unwrap();
    assert_eq!(ext_ids(&r), vec!["a"]);
    assert!((r.hits[0].distance - 14.0f32.sqrt()).abs() < 1e-6);
    assert!(r.hits[0].vector.is_none());
}

#[test]
fn known_small_dataset_matches_hand_computed_order() {
    let dir = TempDir::new();
    let c = create(&dir, "line", 2, DistanceMetric::L2, 16, 8);
    for (id, x) in [("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 5.0), ("e", 10.0)] {
        c.upsert(id, &[x, 0.0]).unwrap();
    }
    let q = [1.9, 0.0];
    let expected = ["c", "b", "a", "d", "e"];
    let exact = c.exact_search(&q, 5, false).unwrap();
    assert_eq!(ext_ids(&exact), expected);
    let want = [0.1f32, 0.9, 1.9, 3.1, 8.1];
    for (h, w) in exact.hits.iter().zip(want) {
        assert!((h.distance - w).abs() < 1e-5, "{} vs {}", h.distance, w);
    }
    let ann = c.search(&q, &QueryOptions::new(5, Some(8))).unwrap();
    assert_eq!(ext_ids(&ann), expected);
    let top2 = c.search(&q, &QueryOptions::new(2, Some(8))).unwrap();
    assert_eq!(ext_ids(&top2), vec!["c", "b"]);
    c.verify_graph().unwrap();
}

#[test]
fn equal_distances_are_ordered_by_internal_id_and_stable() {
    let dir = TempDir::new();
    let c = create(&dir, "ties", 2, DistanceMetric::L2, 8, 8);
    for (id, v) in [
        ("p", [1.0, 0.0]),
        ("q", [0.0, 1.0]),
        ("r", [-1.0, 0.0]),
        ("s", [0.0, -1.0]),
    ] {
        c.upsert(id, &v).unwrap();
    }
    let exact = c.exact_search(&[0.0, 0.0], 4, false).unwrap();
    assert_eq!(ext_ids(&exact), vec!["p", "q", "r", "s"]);
    let first = c
        .search(&[0.0, 0.0], &QueryOptions::new(4, Some(4)))
        .unwrap();
    assert_eq!(ext_ids(&first), vec!["p", "q", "r", "s"]);
    for _ in 0..20 {
        let again = c
            .search(&[0.0, 0.0], &QueryOptions::new(4, Some(4)))
            .unwrap();
        assert_eq!(ext_ids(&again), ext_ids(&first));
    }
}

#[test]
fn include_vectors_returns_stored_values() {
    let dir = TempDir::new();
    let c = create(&dir, "vecs", 2, DistanceMetric::L2, 8, 8);
    c.upsert("a", &[0.5, 0.25]).unwrap();
    let mut opts = QueryOptions::new(1, None);
    opts.include_vectors = true;
    let r = c.search(&[0.0, 0.0], &opts).unwrap();
    assert_eq!(r.hits[0].vector.as_deref(), Some(&[0.5f32, 0.25][..]));
    assert_eq!(c.get_vector("a").unwrap().vector, vec![0.5, 0.25]);
    assert!(matches!(
        c.get_vector("zzz"),
        Err(EngineError::VectorNotFound(_))
    ));
}

// ---------------------------------------------------------------------
// IDs and validation
// ---------------------------------------------------------------------

#[test]
fn upsert_existing_id_keeps_internal_id() {
    let dir = TempDir::new();
    let c = create(&dir, "ids", 2, DistanceMetric::L2, 8, 8);
    assert_eq!(c.upsert("a", &[0.0, 0.0]).unwrap().internal_id, 0);
    assert_eq!(c.upsert("b", &[1.0, 1.0]).unwrap().internal_id, 1);
    let o = c.upsert("a", &[5.0, 5.0]).unwrap();
    assert_eq!((o.internal_id, o.created), (0, false));
    assert_eq!(c.len().unwrap(), 2);
    assert_eq!(c.get_vector("a").unwrap().vector, vec![5.0, 5.0]);
    let r = c.search(&[5.0, 5.0], &QueryOptions::new(1, None)).unwrap();
    assert_eq!(ext_ids(&r), vec!["a"]);
    c.verify_graph().unwrap();
}

#[test]
fn new_ids_are_unique_and_dense() {
    let dir = TempDir::new();
    let c = create(&dir, "dense", 2, DistanceMetric::L2, 8, 128);
    let recs: Vec<IngestRecord> = (0..100)
        .map(|i| IngestRecord {
            id: format!("id-{i}"),
            vector: vec![i as f32, (i * 7 % 13) as f32],
        })
        .collect();
    let s = c.ingest(
        records_source(recs),
        &IngestOptions::default(),
        &IngestProgress::default(),
    );
    assert!(s.is_success(), "{:?}", s.error);
    assert_eq!(s.inserted, 100);
    let mut seen = HashSet::new();
    for i in 0..100 {
        let rec = c.get_vector(&format!("id-{i}")).unwrap();
        assert!(rec.internal_id < 100);
        assert!(seen.insert(rec.internal_id));
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let dir = TempDir::new();
    let c = create(&dir, "val", 3, DistanceMetric::L2, 8, 8);
    assert!(matches!(
        c.upsert("a", &[1.0, 2.0]),
        Err(EngineError::DimensionMismatch {
            expected: 3,
            got: 2
        })
    ));
    assert!(matches!(
        c.upsert("a", &[1.0, f32::NAN, 2.0]),
        Err(EngineError::InvalidArgument { .. })
    ));
    assert!(matches!(
        c.upsert("a", &[1.0, f32::INFINITY, 2.0]),
        Err(EngineError::InvalidArgument { .. })
    ));
    assert!(matches!(
        c.upsert("", &[1.0, 2.0, 3.0]),
        Err(EngineError::InvalidArgument { .. })
    ));
    let long_id = "x".repeat(10_000);
    assert!(matches!(
        c.upsert(&long_id, &[1.0, 2.0, 3.0]),
        Err(EngineError::InvalidArgument { .. })
    ));
    c.upsert("ok", &[1.0, 2.0, 3.0]).unwrap();
    assert!(matches!(
        c.search(&[1.0, 2.0], &QueryOptions::new(1, None)),
        Err(EngineError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        c.search(&[1.0, f32::NAN, 2.0], &QueryOptions::new(1, None)),
        Err(EngineError::InvalidArgument { .. })
    ));
    assert!(matches!(
        c.search(&[1.0, 2.0, 3.0], &QueryOptions::new(0, Some(10))),
        Err(EngineError::InvalidArgument { field: "k", .. })
    ));
    assert!(matches!(
        c.search(&[1.0, 2.0, 3.0], &QueryOptions::new(10, Some(5))),
        Err(EngineError::InvalidArgument {
            field: "ef_search",
            ..
        })
    ));
    assert!(matches!(
        c.exact_search(&[1.0, 2.0, 3.0], 0, false),
        Err(EngineError::InvalidArgument { field: "k", .. })
    ));
    // Nothing invalid was stored.
    assert_eq!(c.len().unwrap(), 1);

    let cos = create(&dir, "cos", 2, DistanceMetric::Cosine, 8, 8);
    assert!(matches!(
        cos.upsert("z", &[0.0, 0.0]),
        Err(EngineError::InvalidArgument { .. })
    ));
    cos.upsert("a", &[1.0, 0.0]).unwrap();
    assert!(matches!(
        cos.search(&[0.0, 0.0], &QueryOptions::new(1, None)),
        Err(EngineError::InvalidArgument { .. })
    ));
}

#[test]
fn invalid_collection_configs_are_rejected() {
    let dir = TempDir::new();
    let mk =
        |cfg: CollectionConfig, cap: u32| Collection::create(dir.path().join("x"), "x", cfg, cap);
    assert!(mk(CollectionConfig::new(0, DistanceMetric::L2, 8, 1.2), 10).is_err());
    assert!(mk(CollectionConfig::new(3, DistanceMetric::L2, 0, 1.2), 10).is_err());
    assert!(
        mk(
            CollectionConfig::new(3, DistanceMetric::L2, 8, f32::NAN),
            10
        )
        .is_err()
    );
    assert!(mk(CollectionConfig::new(3, DistanceMetric::L2, 8, 1.2), 0).is_err());
    assert!(
        !dir.path().join("x").exists(),
        "failed create left a directory behind"
    );
    assert!(
        Collection::create(
            dir.path().join("bad"),
            "../bad",
            CollectionConfig::new(3, DistanceMetric::L2, 8, 1.2),
            10
        )
        .is_err()
    );
}

// ---------------------------------------------------------------------
// Recall against the exact baseline
// ---------------------------------------------------------------------

#[test]
fn ann_recall_bulk_build_l2() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(2000, 16), 50);
    let c = create(&dir, "rl2", 16, DistanceMetric::L2, 16, 100);
    let s = bulk(&c, &data.base, 0);
    assert!(s.is_success(), "{:?}", s.error);
    assert_eq!(s.mode, IngestMode::InitialBuild);
    assert_eq!(s.inserted, 2000);
    c.verify_graph().unwrap();
    let recall = recall_at_k(&c, &data.queries, 10, 64);
    assert!(recall >= 0.95, "Recall@10 at ef=64 was {recall}");
}

#[test]
fn ann_recall_bulk_build_cosine() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(2000, 16), 50);
    let c = create(&dir, "rcos", 16, DistanceMetric::Cosine, 16, 100);
    assert!(bulk(&c, &data.base, 0).is_success());
    c.verify_graph().unwrap();
    let recall = recall_at_k(&c, &data.queries, 10, 64);
    assert!(recall >= 0.95, "Recall@10 at ef=64 was {recall}");
}

#[test]
fn ann_recall_incremental_build() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(1000, 16), 50);
    let c = create(&dir, "rinc", 16, DistanceMetric::L2, 16, 16);
    c.upsert("v0", &data.base[0]).unwrap();
    let s = bulk(&c, &data.base[1..], 1);
    assert!(s.is_success(), "{:?}", s.error);
    assert_eq!(s.mode, IngestMode::Incremental);
    assert_eq!(c.len().unwrap(), 1000);
    c.verify_graph().unwrap();
    let recall = recall_at_k(&c, &data.queries, 10, 64);
    assert!(recall >= 0.95, "Recall@10 at ef=64 was {recall}");
}

#[test]
fn ann_results_are_sorted_unique_and_deterministic() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(500, 8), 10);
    let c = create(&dir, "det", 8, DistanceMetric::L2, 12, 600);
    assert!(bulk(&c, &data.base, 0).is_success());
    for q in &data.queries {
        let a = c.search(q, &QueryOptions::new(10, Some(32))).unwrap();
        assert_eq!(a.hits.len(), 10);
        assert_sorted_unique(&a);
        assert!(a.stats.visited_nodes >= a.stats.expanded_nodes);
        assert_eq!(a.stats.visited_nodes, a.stats.distance_computations);
        let b = c.search(q, &QueryOptions::new(10, Some(32))).unwrap();
        assert_eq!(internal_ids(&a), internal_ids(&b));
    }
}

#[test]
fn identical_input_builds_identical_graphs() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(300, 8), 0);
    let a = create(&dir, "ga", 8, DistanceMetric::L2, 8, 400);
    let b = create(&dir, "gb", 8, DistanceMetric::L2, 8, 400);
    assert!(bulk(&a, &data.base, 0).is_success());
    assert!(bulk(&b, &data.base, 0).is_success());
    for id in 0..300 {
        assert_eq!(a.neighbors(id).unwrap(), b.neighbors(id).unwrap());
    }
}

// ---------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------

#[test]
fn search_results_survive_restart() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(300, 8), 10);
    let path = dir.path().join("persist");
    let before: Vec<Vec<String>> = {
        let c = Collection::create(
            path.clone(),
            "persist",
            CollectionConfig::new(8, DistanceMetric::L2, 12, 1.2),
            400,
        )
        .unwrap();
        assert!(bulk(&c, &data.base, 0).is_success());
        data.queries
            .iter()
            .map(|q| ext_ids(&c.search(q, &QueryOptions::new(5, Some(32))).unwrap()))
            .collect()
    };
    let c = Collection::open(path.clone()).unwrap();
    assert_eq!(c.len().unwrap(), 300);
    c.verify_graph().unwrap();
    for (q, want) in data.queries.iter().zip(&before) {
        assert_eq!(
            &ext_ids(&c.search(q, &QueryOptions::new(5, Some(32))).unwrap()),
            want
        );
    }
    assert_eq!(c.get_vector("v17").unwrap().vector, data.base[17]);
    // Writes after reopen keep working and persist.
    let o = c.upsert("new", &[0.0; 8]).unwrap();
    assert_eq!(o.internal_id, 300);
    drop(c);
    let c = Collection::open(path).unwrap();
    assert_eq!(c.len().unwrap(), 301);
    assert_eq!(c.get_vector("new").unwrap().internal_id, 300);
}

#[test]
fn storage_grows_beyond_initial_capacity_and_survives_restart() {
    let dir = TempDir::new();
    let path = dir.path().join("grow");
    {
        let c = Collection::create(
            path.clone(),
            "grow",
            CollectionConfig::new(2, DistanceMetric::L2, 4, 1.2),
            2,
        )
        .unwrap();
        for i in 0..40 {
            c.upsert(&format!("g{i}"), &[i as f32, 0.5 * i as f32])
                .unwrap();
        }
        let info = c.info().unwrap();
        assert_eq!(info.count, 40);
        assert!(info.capacity >= 40);
        c.verify_graph().unwrap();
    }
    let c = Collection::open(path).unwrap();
    assert_eq!(c.len().unwrap(), 40);
    assert_eq!(c.get_vector("g39").unwrap().vector, vec![39.0, 19.5]);
    let r = c
        .search(&[39.0, 19.5], &QueryOptions::new(1, Some(16)))
        .unwrap();
    assert_eq!(ext_ids(&r), vec!["g39"]);
}

#[test]
fn reopen_of_large_collection_does_not_truncate() {
    // Regression test for the former hard-coded 10_000-slot reload.
    let dir = TempDir::new();
    let path = dir.path().join("big");
    let vec_len_before;
    {
        let c = Collection::create(
            path.clone(),
            "big",
            CollectionConfig::new(2, DistanceMetric::L2, 4, 1.2),
            20_000,
        )
        .unwrap();
        c.upsert("last", &[1.0, 1.0]).unwrap();
        vec_len_before = std::fs::metadata(path.join("vectors.bin")).unwrap().len();
    }
    let c = Collection::open(path.clone()).unwrap();
    assert_eq!(c.info().unwrap().capacity, 20_000);
    assert_eq!(
        std::fs::metadata(path.join("vectors.bin")).unwrap().len(),
        vec_len_before
    );
}

#[test]
fn dangling_edges_from_interrupted_writes_are_removed_on_open() {
    let dir = TempDir::new();
    let path = dir.path().join("dangle");
    {
        let c = Collection::create(
            path.clone(),
            "dangle",
            CollectionConfig::new(2, DistanceMetric::L2, 4, 1.2),
            16,
        )
        .unwrap();
        for (i, v) in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].iter().enumerate() {
            c.upsert(&format!("d{i}"), v).unwrap();
        }
    }
    {
        // Simulate a crash after a reverse edge to unpublished id 7 was written.
        let (mut s, _) = storage::DiskStore::open_existing(&path, 2, 4, 3).unwrap();
        s.write_edges(0, &[1, 7]).unwrap();
        s.flush().unwrap();
    }
    let c = Collection::open(path).unwrap();
    c.verify_graph().unwrap();
    assert_eq!(c.neighbors(0).unwrap(), vec![1]);
}

#[test]
fn inconsistent_or_missing_metadata_is_an_error_not_an_empty_index() {
    let dir = TempDir::new();
    let path = dir.path().join("bad");
    {
        let c = Collection::create(
            path.clone(),
            "bad",
            CollectionConfig::new(2, DistanceMetric::L2, 4, 1.2),
            4,
        )
        .unwrap();
        c.upsert("a", &[1.0, 1.0]).unwrap();
    }
    let meta = path.join("metadata.json");
    let good = std::fs::read(&meta).unwrap();

    // next_id beyond the data files.
    let ids: Vec<String> = (0..10).map(|i| format!("x{i}")).collect();
    let text =
        serde_json::json!({"format_version":1,"medoid_id":0,"next_id":10,"external_ids":ids});
    std::fs::write(&meta, text.to_string()).unwrap();
    assert!(matches!(
        Collection::open(path.clone()),
        Err(EngineError::Corrupt(_))
    ));

    // Malformed JSON.
    std::fs::write(&meta, b"{not json").unwrap();
    assert!(matches!(
        Collection::open(path.clone()),
        Err(EngineError::Corrupt(_))
    ));

    // Entry point that is not live.
    std::fs::write(
        &meta,
        br#"{"format_version":1,"medoid_id":5,"next_id":1,"external_ids":["a"]}"#,
    )
    .unwrap();
    assert!(matches!(
        Collection::open(path.clone()),
        Err(EngineError::Corrupt(_))
    ));

    // Unsupported future format.
    std::fs::write(
        &meta,
        br#"{"format_version":99,"medoid_id":0,"next_id":1,"external_ids":["a"]}"#,
    )
    .unwrap();
    assert!(matches!(
        Collection::open(path.clone()),
        Err(EngineError::Corrupt(_))
    ));

    // Missing metadata.
    std::fs::remove_file(&meta).unwrap();
    assert!(matches!(
        Collection::open(path.clone()),
        Err(EngineError::Corrupt(_))
    ));

    // Restoring the original metadata makes the collection usable again:
    // nothing was overwritten by the failed opens.
    std::fs::write(meta, good).unwrap();
    let c = Collection::open(path).unwrap();
    assert_eq!(c.get_vector("a").unwrap().vector, vec![1.0, 1.0]);
}

#[test]
fn legacy_format_zero_collection_opens_and_upgrades() {
    let dir = TempDir::new();
    let path = dir.path().join("legacy");
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(
        path.join("config.json"),
        br#"{"dim":2,"metric":"L2","r":2,"alpha":1.2}"#,
    )
    .unwrap();
    std::fs::write(
        path.join("metadata.json"),
        br#"{"medoid_id":0,"id_map":{"b":1,"a":0,"c":2},"next_id":3}"#,
    )
    .unwrap();
    // Legacy files: 10 slots; unused graph slots are zero-filled (old behaviour).
    let mut vbytes = Vec::new();
    for i in 0..10 {
        let v = if i < 3 { [i as f32, 0.0] } else { [0.0, 0.0] };
        for x in v {
            vbytes.extend_from_slice(&x.to_ne_bytes());
        }
    }
    std::fs::write(path.join("vectors.bin"), vbytes).unwrap();
    let edges: [[u32; 2]; 3] = [[1, u32::MAX], [0, 2], [1, u32::MAX]];
    let mut gbytes = Vec::new();
    for i in 0..10 {
        let e = edges.get(i).copied().unwrap_or([0, 0]);
        for x in e {
            gbytes.extend_from_slice(&x.to_ne_bytes());
        }
    }
    std::fs::write(path.join("graph.bin"), gbytes).unwrap();

    let c = Collection::open(path.clone()).unwrap();
    assert_eq!(c.config().effective_l_build(), 4); // legacy: 2 * r
    assert_eq!(c.len().unwrap(), 3);
    c.verify_graph().unwrap();
    let r = c
        .search(&[2.0, 0.0], &QueryOptions::new(3, Some(3)))
        .unwrap();
    assert_eq!(ext_ids(&r), vec!["c", "b", "a"]);
    // A new vector lands in a legacy zero-filled slot; it must not inherit
    // phantom edges to node 0.
    c.upsert("d", &[3.0, 0.0]).unwrap();
    c.verify_graph().unwrap();
    drop(c);
    let meta = std::fs::read_to_string(path.join("metadata.json")).unwrap();
    assert!(meta.contains("\"format_version\":1"));
    assert!(meta.contains("external_ids"));
    let c = Collection::open(path).unwrap();
    assert_eq!(c.get_vector("d").unwrap().internal_id, 3);
}

// ---------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------

fn write_jsonl(dir: &TempDir, name: &str, lines: &[String]) -> PathBuf {
    let p = dir.path().join(name);
    std::fs::write(&p, lines.join("\n")).unwrap();
    p
}

fn db(dir: &TempDir) -> DBInfo {
    DBInfo::open(dir.path().join("data")).unwrap().0
}

#[test]
fn jsonl_job_completes_and_data_is_searchable() {
    let dir = TempDir::new();
    let db = db(&dir);
    let c = db
        .create_collection(
            "jobs",
            CollectionConfig::new(3, DistanceMetric::L2, 8, 1.2),
            16,
        )
        .unwrap();
    let lines: Vec<String> = (0..200)
        .map(|i| format!("{{\"id\":\"r{i}\",\"vector\":[{},{},{}]}}", i, i % 7, i % 3))
        .collect();
    let file = write_jsonl(&dir, "ok.jsonl", &lines);
    let job = db.jobs.create("jobs", OnError::FailFast);
    assert_eq!(job.state(), JobState::Queued);
    run_ingest_job(
        Arc::clone(&c),
        Arc::clone(&job),
        file,
        IngestOptions::default(),
    );
    let snap = job.snapshot();
    assert_eq!(snap.state, JobState::Completed, "{:?}", snap.error);
    assert_eq!(snap.records_read, 200);
    assert_eq!(snap.records_inserted, 200);
    assert_eq!(snap.records_rejected, 0);
    assert_eq!(snap.nodes_linked, 200);
    assert_eq!(c.len().unwrap(), 200);
    let r = c
        .search(&[150.0, 3.0, 0.0], &QueryOptions::new(1, Some(32)))
        .unwrap();
    assert_eq!(ext_ids(&r), vec!["r150"]);
    assert_eq!(c.info().unwrap().index_state, "ready");
}

#[test]
fn jsonl_fail_fast_reports_line_and_keeps_earlier_records() {
    let dir = TempDir::new();
    let db = db(&dir);
    let c = db
        .create_collection(
            "ff",
            CollectionConfig::new(2, DistanceMetric::L2, 8, 1.2),
            16,
        )
        .unwrap();
    let lines = vec![
        r#"{"id":"a","vector":[1,2]}"#.to_string(),
        r#"{"id":"b","vector":[3,4]}"#.to_string(),
        r#"{"id":"c","vector":[5,6,7]}"#.to_string(),
        r#"{"id":"d","vector":[8,9]}"#.to_string(),
    ];
    let file = write_jsonl(&dir, "ff.jsonl", &lines);
    let job = db.jobs.create("ff", OnError::FailFast);
    run_ingest_job(
        Arc::clone(&c),
        Arc::clone(&job),
        file,
        IngestOptions::default(),
    );
    let snap = job.snapshot();
    assert_eq!(snap.state, JobState::Failed);
    assert!(
        snap.error.as_deref().unwrap_or("").contains("line 3"),
        "{:?}",
        snap.error
    );
    assert_eq!(snap.records_inserted, 2);
    assert_eq!(c.len().unwrap(), 2);
    c.verify_graph().unwrap();
    assert!(matches!(
        c.get_vector("d"),
        Err(EngineError::VectorNotFound(_))
    ));
}

#[test]
fn jsonl_skip_mode_counts_rejections_and_updates() {
    let dir = TempDir::new();
    let db = db(&dir);
    let c = db
        .create_collection(
            "skip",
            CollectionConfig::new(2, DistanceMetric::L2, 8, 1.2),
            16,
        )
        .unwrap();
    let lines = vec![
        r#"{"id":"a","vector":[1,2]}"#.to_string(),
        "garbage".to_string(),
        r#"{"id":"b","vector":[3,4]}"#.to_string(),
        r#"{"id":"c","vector":[5]}"#.to_string(),
        r#"{"id":"a","vector":[9,9]}"#.to_string(),
    ];
    let file = write_jsonl(&dir, "skip.jsonl", &lines);
    let job = db.jobs.create("skip", OnError::Skip);
    let opts = IngestOptions {
        on_error: OnError::Skip,
        ..IngestOptions::default()
    };
    run_ingest_job(Arc::clone(&c), Arc::clone(&job), file, opts);
    let snap = job.snapshot();
    assert_eq!(snap.state, JobState::Completed, "{:?}", snap.error);
    assert_eq!(snap.records_read, 5);
    assert_eq!(snap.records_inserted, 2);
    assert_eq!(snap.records_updated, 1);
    assert_eq!(snap.records_rejected, 2);
    let lines_rejected: Vec<u64> = snap.rejected_samples.iter().map(|r| r.line).collect();
    assert_eq!(lines_rejected, vec![2, 4]);
    assert_eq!(c.get_vector("a").unwrap().vector, vec![9.0, 9.0]);
    c.verify_graph().unwrap();
}

#[test]
fn jsonl_missing_file_fails_job() {
    let dir = TempDir::new();
    let db = db(&dir);
    let c = db
        .create_collection(
            "miss",
            CollectionConfig::new(2, DistanceMetric::L2, 8, 1.2),
            16,
        )
        .unwrap();
    let job = db.jobs.create("miss", OnError::FailFast);
    run_ingest_job(
        c,
        Arc::clone(&job),
        dir.path().join("nope.jsonl"),
        IngestOptions::default(),
    );
    assert_eq!(job.snapshot().state, JobState::Failed);
}

// ---------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------

#[test]
fn concurrent_searches_and_upserts_are_consistent() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(800, 8), 20);
    let c = Arc::new(create(&dir, "conc", 8, DistanceMetric::L2, 12, 64));
    assert!(bulk(&c, &data.base[..500], 0).is_success());
    let queries = Arc::new(data.queries.clone());
    let writer = {
        let c = Arc::clone(&c);
        let rest: Vec<Vec<f32>> = data.base[500..].to_vec();
        std::thread::spawn(move || {
            for (i, v) in rest.iter().enumerate() {
                c.upsert(&format!("v{}", 500 + i), v).unwrap();
            }
        })
    };
    let readers: Vec<_> = (0..4)
        .map(|t| {
            let c = Arc::clone(&c);
            let queries = Arc::clone(&queries);
            std::thread::spawn(move || {
                for i in 0..100 {
                    let q = &queries[(i + t) % queries.len()];
                    let r = c.search(q, &QueryOptions::new(10, Some(32))).unwrap();
                    assert_eq!(r.hits.len(), 10);
                    assert_sorted_unique(&r);
                    for h in &r.hits {
                        assert_eq!(h.external_id, format!("v{}", h.internal_id));
                    }
                }
            })
        })
        .collect();
    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }
    assert_eq!(c.len().unwrap(), 800);
    c.verify_graph().unwrap();
}

// ---------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------

#[test]
fn registry_create_delete_reload_and_safety() {
    let dir = TempDir::new();
    let data_dir = dir.path().join("data");
    {
        let (db, report) = DBInfo::open(&data_dir).unwrap();
        assert!(report.loaded.is_empty());
        let cfg = CollectionConfig::new(2, DistanceMetric::L2, 4, 1.2);
        let c = db.create_collection("keep", cfg.clone(), 8).unwrap();
        c.upsert("a", &[1.0, 2.0]).unwrap();
        assert!(matches!(
            db.create_collection("keep", cfg.clone(), 8),
            Err(EngineError::CollectionExists(_))
        ));
        assert!(matches!(
            db.create_collection("../evil", cfg.clone(), 8),
            Err(EngineError::InvalidCollectionName(_))
        ));
        db.create_collection("gone", cfg.clone(), 8).unwrap();
        db.delete_collection("gone").unwrap();
        assert!(!data_dir.join("gone").exists());
        assert!(matches!(
            db.delete_collection("gone"),
            Err(EngineError::CollectionNotFound(_))
        ));
        assert!(matches!(
            db.get("gone"),
            Err(EngineError::CollectionNotFound(_))
        ));
        // A directory that exists on disk but is not loaded is never replaced.
        std::fs::create_dir_all(data_dir.join("orphan")).unwrap();
        assert!(matches!(
            db.create_collection("orphan", cfg, 8),
            Err(EngineError::CollectionExists(_))
        ));
        let names: Vec<String> = db.list().unwrap().into_iter().map(|i| i.name).collect();
        assert_eq!(names, vec!["keep"]);
    }
    // Corrupt one collection on disk: it is reported, not served, not touched.
    let bad = data_dir.join("broken");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("config.json"), b"{}").unwrap();
    let (db, report) = DBInfo::open(data_dir).unwrap();
    assert_eq!(report.loaded, vec!["keep"]);
    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].0, "broken");
    assert!(report.skipped.contains(&"orphan".to_string()));
    assert!(bad.join("config.json").exists());
    assert_eq!(
        db.get("keep").unwrap().get_vector("a").unwrap().vector,
        vec![1.0, 2.0]
    );
}

#[test]
fn search_trace_and_graph_snapshot() {
    let dir = TempDir::new();
    let data = dataset::generate(&spec(300, 4), 5);
    let c = create(&dir, "viz", 4, DistanceMetric::L2, 8, 400);
    assert!(bulk(&c, &data.base, 0).is_success());
    let entry = c.info().unwrap().entry_point.unwrap();
    for q in &data.queries {
        let plain = c.search(q, &QueryOptions::new(5, Some(32))).unwrap();
        assert!(plain.trace.is_none());
        let mut opts = QueryOptions::new(5, Some(32));
        opts.trace = true;
        let traced = c.search(q, &opts).unwrap();
        // Tracing must not change the results.
        assert_eq!(internal_ids(&traced), internal_ids(&plain));
        let trace = traced.trace.unwrap();
        assert_eq!(trace.entry_point, entry);
        assert!(trace.entry_points.contains(&trace.expanded[0]));
        assert_eq!(trace.entry_points[0], entry);
        assert_eq!(trace.expanded.len(), traced.stats.expanded_nodes);
    }
    let snap = c.graph_snapshot(1000).unwrap();
    assert_eq!(snap.nodes.len(), 300);
    assert!(!snap.truncated);
    assert_eq!(snap.entry_point, Some(entry));
    for [a, b] in &snap.edges {
        assert!(*a < 300 && *b < 300 && a != b);
    }
    let small = c.graph_snapshot(10).unwrap();
    assert_eq!(small.nodes.len(), 10);
    assert!(small.truncated);
    assert!(small.edges.iter().all(|[a, b]| *a < 10 && *b < 10));
    assert!(c.graph_snapshot(0).is_err());
    assert!(c.graph_snapshot(1_000_000).is_err());
    assert!(
        c.exact_search(&data.queries[0], 3, false)
            .unwrap()
            .trace
            .is_none()
    );
}

#[test]
fn clustered_l2_data_stays_navigable_with_multiple_entry_points() {
    // Regression test for a measured failure: on 10 000 x 64-d L2 data in 32
    // tight clusters, a single entry point gave Recall@10 = 0.20 at
    // ef_search 32 because whole clusters were unreachable from it.
    let dir = TempDir::new();
    let spec = DatasetSpec {
        n: 4000,
        dim: 32,
        seed: 42,
        distribution: Distribution::Clustered { clusters: 32 },
    };
    let data = dataset::generate(&spec, 100);

    let multi = create(&dir, "multi", 32, DistanceMetric::L2, 16, 4000);
    assert!(bulk(&multi, &data.base, 0).is_success());
    assert_eq!(multi.info().unwrap().entry_point_count, 64);
    multi.verify_graph().unwrap();
    let recall_multi = recall_at_k(&multi, &data.queries, 10, 32);

    let mut cfg = CollectionConfig::new(32, DistanceMetric::L2, 16, 1.2);
    cfg.entry_points = Some(1);
    let single = Collection::create(dir.path().join("single"), "single", cfg, 4000).unwrap();
    assert!(bulk(&single, &data.base, 0).is_success());
    assert_eq!(single.info().unwrap().entry_point_count, 1);
    let recall_single = recall_at_k(&single, &data.queries, 10, 32);

    assert!(
        recall_multi >= 0.95,
        "multi-entry Recall@10 at ef 32 was {recall_multi}"
    );
    assert!(
        recall_multi > recall_single,
        "multi {recall_multi} should beat single {recall_single}"
    );

    // The entry-point set survives a restart.
    drop(multi);
    let reopened = Collection::open(dir.path().join("multi")).unwrap();
    assert_eq!(reopened.info().unwrap().entry_point_count, 64);
    reopened.verify_graph().unwrap();
}
