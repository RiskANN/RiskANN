//! RiskANN benchmark and Recall@K harness.
//!
//! ```text
//! riskann-bench run   [options]            one configuration + ef_search sweep
//! riskann-bench suite [--profile quick|standard|full] [--out DIR]
//! riskann-bench gen-jsonl --n N --dim D --out FILE [--seed S] [--distribution D]
//! ```
//!
//! Everything is generated from fixed seeds, so a run is reproducible on the
//! same machine and build. Numbers from different machines, build profiles,
//! dimensions or data distributions are not directly comparable.
//!
//! Measurement notes (also written into every result file):
//! * Latency is measured around `Collection::search` / `exact_search`
//!   (engine level: validation, read lock, traversal, result assembly). It
//!   excludes HTTP and JSON.
//! * Warm-up: every query is run `--warmup` times (default 1) per `ef_search`
//!   before the measured pass, so results reflect a warm page cache.
//! * `qps_single` = measured queries / total measured latency on one thread.
//!   `qps_concurrent` (only with `--threads > 1`) = queries / wall time with
//!   that many threads sharing one collection.
//! * Recall@K = |ANN top-K ∩ exact top-K| / min(K, n), averaged over queries.
//! * `peak_rss_kb` is the process high-water mark (Linux `VmHWM`); it
//!   includes the in-memory dataset and touched mmap pages.

use engine::DistanceMetric;
use engine::collection::{Collection, CollectionConfig, QueryOptions, default_l_build};
use engine::dataset::{self, DatasetSpec, Distribution};
use engine::ingest::{IngestOptions, IngestProgress, IngestRecord, records_source};
use engine::limits::DEFAULT_ENTRY_POINTS;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

type BoxError = Box<dyn std::error::Error>;

const USAGE: &str = "\
usage:
  riskann-bench run [--n 10000] [--dim 64] [--metric l2|cosine] [--distribution clustered|gaussian|uniform|clustered:N]
                    [--seed 42] [--queries 200] [--k 10] [--ef 16,32,64,128,256] [--r 32] [--alpha 1.2]
                    [--l-build 64] [--entry-points 64] [--build-mode bulk|incremental] [--threads 1] [--warmup 1]
                    [--out bench_results] [--work-dir DIR] [--label NAME]
  riskann-bench suite [--profile quick|standard|full] [--out bench_results] [--threads 1]
  riskann-bench gen-jsonl --n N --dim D --out FILE [--seed 42] [--distribution clustered] [--id-prefix v]";

struct Args {
    values: HashMap<String, String>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self, BoxError> {
        let mut values = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let key = raw[i]
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument '{}'", raw[i]))?;
            let value = raw
                .get(i + 1)
                .ok_or_else(|| format!("missing value for --{key}"))?;
            values.insert(key.to_string(), value.clone());
            i += 2;
        }
        Ok(Self { values })
    }

    fn check_known(&self, known: &[&str]) -> Result<(), BoxError> {
        for k in self.values.keys() {
            if !known.contains(&k.as_str()) {
                return Err(format!("unknown option --{k}").into());
            }
        }
        Ok(())
    }

    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, BoxError> {
        match self.values.get(key) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("invalid value '{v}' for --{key}").into()),
        }
    }

    fn get_str(&self, key: &str, default: &str) -> String {
        self.values
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }
}

#[derive(Debug, Clone)]
struct RunConfig {
    label: String,
    n: usize,
    dim: usize,
    metric: DistanceMetric,
    distribution: Distribution,
    seed: u64,
    queries: usize,
    k: usize,
    efs: Vec<usize>,
    r: usize,
    alpha: f32,
    l_build: usize,
    entry_points: usize,
    incremental: bool,
    threads: usize,
    warmup: usize,
    out: PathBuf,
    work_dir: PathBuf,
}

const RUN_KEYS: &[&str] = &[
    "n",
    "dim",
    "metric",
    "distribution",
    "seed",
    "queries",
    "k",
    "ef",
    "r",
    "alpha",
    "l-build",
    "entry-points",
    "build-mode",
    "threads",
    "warmup",
    "out",
    "work-dir",
    "label",
];

impl RunConfig {
    fn from_args(a: &Args) -> Result<Self, BoxError> {
        a.check_known(RUN_KEYS)?;
        let n: usize = a.get("n", 10_000)?;
        let dim: usize = a.get("dim", 64)?;
        let metric = DistanceMetric::parse(&a.get_str("metric", "l2"))
            .ok_or("--metric must be l2 or cosine")?;
        let distribution = Distribution::parse(&a.get_str("distribution", "clustered"))
            .ok_or("--distribution must be uniform, gaussian, clustered or clustered:N")?;
        let r: usize = a.get("r", 32)?;
        let efs: Vec<usize> = a
            .get_str("ef", "16,32,64,128,256")
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<_, _>>()
            .map_err(|_| "--ef must be a comma-separated list of integers")?;
        let build_mode = a.get_str("build-mode", "bulk");
        let incremental = match build_mode.as_str() {
            "bulk" => false,
            "incremental" => true,
            _ => return Err("--build-mode must be bulk or incremental".into()),
        };
        let seed: u64 = a.get("seed", 42)?;
        let entry_points: usize = a.get("entry-points", DEFAULT_ENTRY_POINTS)?;
        let label = a.get_str(
            "label",
            &format!(
                "n{n}_d{dim}_{}_{}_r{r}_e{entry_points}_{}",
                metric.name().to_lowercase(),
                distribution.to_string().replace(':', "-"),
                build_mode
            ),
        );
        let work_dir = match a.values.get("work-dir") {
            Some(p) => PathBuf::from(p),
            None => std::env::temp_dir().join(format!("riskann-bench-{}", std::process::id())),
        };
        let cfg = RunConfig {
            label,
            n,
            dim,
            metric,
            distribution,
            seed,
            queries: a.get("queries", 200)?,
            k: a.get("k", 10)?,
            efs,
            r,
            alpha: a.get("alpha", 1.2f32)?,
            l_build: a.get("l-build", default_l_build(r))?,
            entry_points,
            incremental,
            threads: a.get("threads", 1usize)?.max(1),
            warmup: a.get("warmup", 1)?,
            out: PathBuf::from(a.get_str("out", "bench_results")),
            work_dir,
        };
        if cfg.n == 0 || cfg.queries == 0 || cfg.k == 0 {
            return Err("--n, --queries and --k must be positive".into());
        }
        Ok(cfg)
    }
}

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    // Nearest-rank method.
    let rank = ((p / 100.0) * sorted_us.len() as f64).ceil() as usize;
    sorted_us[rank.clamp(1, sorted_us.len()) - 1]
}

struct LatencySummary {
    p50: f64,
    p95: f64,
    p99: f64,
    mean: f64,
    qps: f64,
}

fn summarize(mut us: Vec<f64>) -> LatencySummary {
    us.sort_by(|a, b| a.total_cmp(b));
    let total: f64 = us.iter().sum();
    LatencySummary {
        p50: percentile(&us, 50.0),
        p95: percentile(&us, 95.0),
        p99: percentile(&us, 99.0),
        mean: if us.is_empty() {
            0.0
        } else {
            total / us.len() as f64
        },
        qps: if total > 0.0 {
            us.len() as f64 / (total / 1e6)
        } else {
            0.0
        },
    }
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn machine_info() -> Value {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    let mem_total_kb = read_status_kb("/proc/meminfo", "MemTotal:");
    json!({
        "cpu_model": cpu,
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "mem_total_kb": mem_total_kb,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "riskann_version": env!("CARGO_PKG_VERSION"),
    })
}

fn read_status_kb(file: &str, key: &str) -> Option<u64> {
    let text = std::fs::read_to_string(file).ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn dir_size(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

fn records(base: &[Vec<f32>], offset: usize) -> Vec<IngestRecord> {
    base.iter()
        .enumerate()
        .map(|(i, v)| IngestRecord {
            id: format!("v{}", i + offset),
            vector: v.clone(),
        })
        .collect()
}

fn run(cfg: &RunConfig, machine: &Value) -> Result<Vec<Value>, BoxError> {
    if cfg!(debug_assertions) {
        eprintln!("warning: debug build; use `cargo run --release` for meaningful numbers");
    }
    eprintln!(
        "[{}] generating {} base + {} query vectors (dim {})",
        cfg.label, cfg.n, cfg.queries, cfg.dim
    );
    let data = dataset::generate(
        &DatasetSpec {
            n: cfg.n,
            dim: cfg.dim,
            seed: cfg.seed,
            distribution: cfg.distribution,
        },
        cfg.queries,
    );

    let coll_dir = cfg.work_dir.join(&cfg.label);
    if coll_dir.exists() {
        std::fs::remove_dir_all(&coll_dir)?;
    }
    std::fs::create_dir_all(&cfg.work_dir)?;
    let config = CollectionConfig {
        dim: cfg.dim,
        metric: cfg.metric,
        r: cfg.r,
        alpha: cfg.alpha,
        l_build: Some(cfg.l_build),
        entry_points: Some(cfg.entry_points),
    };
    let initial_capacity = u32::try_from(cfg.n).map_err(|_| "--n too large")?;

    // ---- build ----
    eprintln!(
        "[{}] building index ({})",
        cfg.label,
        if cfg.incremental {
            "incremental"
        } else {
            "bulk"
        }
    );
    let build_start = Instant::now();
    let collection = Collection::create(coll_dir.clone(), "bench", config, initial_capacity)?;
    let summary = if cfg.incremental {
        collection.upsert("v0", &data.base[0])?;
        collection.ingest(
            records_source(records(&data.base[1..], 1)),
            &IngestOptions::default(),
            &IngestProgress::default(),
        )
    } else {
        collection.ingest(
            records_source(records(&data.base, 0)),
            &IngestOptions::default(),
            &IngestProgress::default(),
        )
    };
    let build_secs = build_start.elapsed().as_secs_f64();
    if let Some(e) = &summary.error {
        return Err(format!("build failed: {e}").into());
    }
    collection.verify_graph()?;
    let n_live = collection.len()? as usize;
    let mut degree_sum = 0usize;
    for id in 0..n_live as u32 {
        degree_sum += collection.neighbors(id)?.len();
    }
    let avg_degree = degree_sum as f64 / n_live.max(1) as f64;
    eprintln!(
        "[{}] built in {:.2}s, avg degree {:.1}",
        cfg.label, build_secs, avg_degree
    );

    // ---- exact baseline / ground truth ----
    let mut truth: Vec<HashSet<u32>> = Vec::with_capacity(cfg.queries);
    let mut exact_us = Vec::with_capacity(cfg.queries);
    for q in &data.queries {
        let t = Instant::now();
        let r = collection.exact_search(q, cfg.k, false)?;
        exact_us.push(micros(t.elapsed()));
        truth.push(r.hits.iter().map(|h| h.internal_id).collect());
    }
    let exact = summarize(exact_us);
    let denom = cfg.k.min(n_live).max(1) as f64;

    // ---- ANN sweep ----
    let collection = Arc::new(collection);
    let mut rows = Vec::new();
    let mut sweep = Vec::new();
    for &ef in &cfg.efs {
        if ef < cfg.k {
            eprintln!("[{}] skipping ef_search={ef} < k={}", cfg.label, cfg.k);
            continue;
        }
        let opts = QueryOptions::new(cfg.k, Some(ef));
        for _ in 0..cfg.warmup {
            for q in &data.queries {
                collection.search(q, &opts)?;
            }
        }
        let mut lat = Vec::with_capacity(cfg.queries);
        let (mut recall_sum, mut visited, mut expanded, mut dists) =
            (0.0f64, 0usize, 0usize, 0usize);
        for (q, t_set) in data.queries.iter().zip(&truth) {
            let t = Instant::now();
            let r = collection.search(q, &opts)?;
            lat.push(micros(t.elapsed()));
            let hit = r
                .hits
                .iter()
                .filter(|h| t_set.contains(&h.internal_id))
                .count();
            recall_sum += hit as f64 / denom;
            visited += r.stats.visited_nodes;
            expanded += r.stats.expanded_nodes;
            dists += r.stats.distance_computations;
        }
        let nq = cfg.queries as f64;
        let s = summarize(lat);
        let qps_concurrent = if cfg.threads > 1 {
            Some(concurrent_qps(
                &collection,
                &data.queries,
                opts,
                cfg.threads,
            )?)
        } else {
            None
        };
        let recall = recall_sum / nq;
        eprintln!(
            "[{}] ef={ef:<4} recall@{}={recall:.4} p50={:.1}us p99={:.1}us",
            cfg.label, cfg.k, s.p50, s.p99
        );
        sweep.push(json!({
            "ef_search": ef,
            "recall_at_k": recall,
            "latency_us": {"p50": s.p50, "p95": s.p95, "p99": s.p99, "mean": s.mean},
            "qps_single_thread": s.qps,
            "qps_concurrent": qps_concurrent,
            "avg_visited_nodes": visited as f64 / nq,
            "avg_expanded_nodes": expanded as f64 / nq,
            "avg_distance_computations": dists as f64 / nq,
        }));
        rows.push((
            ef,
            recall,
            s,
            qps_concurrent,
            visited as f64 / nq,
            expanded as f64 / nq,
            dists as f64 / nq,
        ));
    }

    // ---- storage, reload, memory ----
    let disk_bytes = dir_size(&coll_dir);
    let probe: Vec<Vec<u32>> = data
        .queries
        .iter()
        .take(10)
        .map(|q| {
            collection
                .search(q, &QueryOptions::new(cfg.k, Some(cfg.k.max(64))))
                .map(|r| r.hits.iter().map(|h| h.internal_id).collect())
        })
        .collect::<Result<_, _>>()?;
    drop(collection);
    let reload_start = Instant::now();
    let reopened = Collection::open(coll_dir.clone())?;
    let reload_ms = reload_start.elapsed().as_secs_f64() * 1e3;
    let mut reload_consistent = true;
    for (q, want) in data.queries.iter().take(10).zip(&probe) {
        let got: Vec<u32> = reopened
            .search(q, &QueryOptions::new(cfg.k, Some(cfg.k.max(64))))?
            .hits
            .iter()
            .map(|h| h.internal_id)
            .collect();
        reload_consistent &= &got == want;
    }
    drop(reopened);
    let peak_rss_kb = read_status_kb("/proc/self/status", "VmHWM:");
    std::fs::remove_dir_all(coll_dir).ok();

    let result = json!({
        "label": cfg.label,
        "machine": machine,
        "dataset": {
            "n": cfg.n, "dim": cfg.dim, "distribution": cfg.distribution.to_string(),
            "seed": cfg.seed, "queries": cfg.queries, "queries_held_out": true,
        },
        "index": {
            "metric": cfg.metric.name(), "r": cfg.r, "alpha": cfg.alpha, "l_build": cfg.l_build, "entry_points": cfg.entry_points,
            "build_mode": if cfg.incremental { "incremental" } else { "bulk" },
        },
        "k": cfg.k,
        "warmup_passes": cfg.warmup,
        "threads": cfg.threads,
        "build_secs": build_secs,
        "avg_out_degree": avg_degree,
        "exact_baseline": {"latency_us": {"p50": exact.p50, "p95": exact.p95, "p99": exact.p99, "mean": exact.mean}, "qps_single_thread": exact.qps},
        "sweep": sweep,
        "disk_bytes": disk_bytes,
        "reload_ms": reload_ms,
        "reload_results_identical": reload_consistent,
        "peak_rss_kb": peak_rss_kb,
        "notes": "Latency measured around engine search calls (no HTTP). Recall@K = |ANN ∩ exact| / min(K, n).",
    });

    std::fs::create_dir_all(&cfg.out)?;
    let json_path = cfg.out.join(format!("{}.json", cfg.label));
    std::fs::write(&json_path, serde_json::to_string_pretty(&result)?)?;
    append_csv(
        cfg,
        machine,
        build_secs,
        &exact,
        disk_bytes,
        reload_ms,
        peak_rss_kb,
        &rows,
    )?;
    print_table(
        cfg,
        build_secs,
        &exact,
        disk_bytes,
        reload_ms,
        reload_consistent,
        &rows,
    );
    eprintln!("[{}] wrote {}", cfg.label, json_path.display());
    Ok(vec![result])
}

fn concurrent_qps(
    collection: &Arc<Collection>,
    queries: &[Vec<f32>],
    opts: QueryOptions,
    threads: usize,
) -> Result<f64, BoxError> {
    // Each thread runs the full query set once.
    let start = Instant::now();
    let outcome: Result<(), String> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let c = Arc::clone(collection);
                s.spawn(move || -> Result<(), String> {
                    for q in queries {
                        c.search(q, &opts).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })
            })
            .collect();
        for h in handles {
            h.join()
                .map_err(|_| "benchmark thread panicked".to_string())??;
        }
        Ok(())
    });
    outcome?;
    let secs = start.elapsed().as_secs_f64();
    Ok((threads * queries.len()) as f64 / secs.max(1e-9))
}

type Row = (usize, f64, LatencySummary, Option<f64>, f64, f64, f64);

#[allow(clippy::too_many_arguments)]
fn append_csv(
    cfg: &RunConfig,
    machine: &Value,
    build_secs: f64,
    exact: &LatencySummary,
    disk_bytes: u64,
    reload_ms: f64,
    peak_rss_kb: Option<u64>,
    rows: &[Row],
) -> Result<(), BoxError> {
    let path = cfg.out.join("summary.csv");
    let new = !path.exists();
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(
            f,
            "label,n,dim,metric,distribution,r,alpha,l_build,build_mode,k,ef_search,queries,threads,\
             build_secs,recall_at_k,p50_us,p95_us,p99_us,mean_us,qps_single,qps_concurrent,\
             avg_visited,avg_expanded,avg_distance_computations,exact_p50_us,exact_qps,disk_bytes,\
             reload_ms,peak_rss_kb,build_profile,cpu_model"
        )?;
    }
    let cpu = machine["cpu_model"]
        .as_str()
        .unwrap_or("unknown")
        .replace(',', " ");
    let profile = machine["build_profile"].as_str().unwrap_or("unknown");
    for (ef, recall, s, qc, vis, exp, dc) in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.5},{:.2},{:.2},{:.2},{:.2},{:.1},{},{:.1},{:.1},{:.1},{:.2},{:.1},{},{:.2},{},{},{}",
            cfg.label,
            cfg.n,
            cfg.dim,
            cfg.metric.name(),
            cfg.distribution,
            cfg.r,
            cfg.alpha,
            cfg.l_build,
            if cfg.incremental {
                "incremental"
            } else {
                "bulk"
            },
            cfg.k,
            ef,
            cfg.queries,
            cfg.threads,
            build_secs,
            recall,
            s.p50,
            s.p95,
            s.p99,
            s.mean,
            s.qps,
            qc.map(|q| format!("{q:.1}")).unwrap_or_default(),
            vis,
            exp,
            dc,
            exact.p50,
            exact.qps,
            disk_bytes,
            reload_ms,
            peak_rss_kb.map(|k| k.to_string()).unwrap_or_default(),
            profile,
            cpu
        )?;
    }
    Ok(())
}

fn print_table(
    cfg: &RunConfig,
    build_secs: f64,
    exact: &LatencySummary,
    disk_bytes: u64,
    reload_ms: f64,
    reload_consistent: bool,
    rows: &[Row],
) {
    println!();
    println!(
        "### {} — n={} dim={} metric={} data={} r={} alpha={} l_build={} entry_points={} k={} queries={}",
        cfg.label,
        cfg.n,
        cfg.dim,
        cfg.metric.name(),
        cfg.distribution,
        cfg.r,
        cfg.alpha,
        cfg.l_build,
        cfg.entry_points,
        cfg.k,
        cfg.queries
    );
    println!(
        "build {build_secs:.2}s | disk {:.1} MiB | reload {reload_ms:.1} ms (results identical: {reload_consistent}) | exact baseline p50 {:.1} us, {:.0} QPS",
        disk_bytes as f64 / (1024.0 * 1024.0),
        exact.p50,
        exact.qps
    );
    println!();
    println!(
        "| ef_search | Recall@{} | p50 µs | p95 µs | p99 µs | QPS (1 thread) | visited | dist. comps |",
        cfg.k
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (ef, recall, s, _, vis, _, dc) in rows {
        println!(
            "| {ef} | {recall:.4} | {:.1} | {:.1} | {:.1} | {:.0} | {vis:.1} | {dc:.1} |",
            s.p50, s.p95, s.p99, s.qps
        );
    }
    println!();
}

fn suite(a: &Args, machine: &Value) -> Result<(), BoxError> {
    a.check_known(&["profile", "out", "threads", "work-dir"])?;
    let profile = a.get_str("profile", "quick");
    // (n, dim, metric, r)
    let mut plan: Vec<(usize, usize, &str, usize)> = vec![
        (1_000, 3, "l2", 16),
        (1_000, 64, "l2", 32),
        (10_000, 64, "l2", 32),
        (10_000, 64, "cosine", 32),
    ];
    let efs = match profile.as_str() {
        "quick" => "10,16,32,64,128",
        "standard" | "full" => {
            plan.push((10_000, 128, "l2", 32));
            plan.push((50_000, 128, "l2", 32));
            plan.push((50_000, 128, "cosine", 32));
            if profile == "full" {
                plan.push((100_000, 128, "l2", 32));
            }
            "10,16,32,64,128,256"
        }
        _ => return Err("--profile must be quick, standard or full".into()),
    };
    for (n, dim, metric, r) in plan {
        let mut values: HashMap<String, String> = HashMap::new();
        values.insert("n".into(), n.to_string());
        values.insert("dim".into(), dim.to_string());
        values.insert("metric".into(), metric.into());
        values.insert("r".into(), r.to_string());
        values.insert("ef".into(), efs.into());
        values.insert("queries".into(), "200".into());
        for key in ["out", "threads", "work-dir"] {
            if let Some(v) = a.values.get(key) {
                values.insert(key.into(), v.clone());
            }
        }
        let cfg = RunConfig::from_args(&Args { values })?;
        run(&cfg, machine)?;
    }
    Ok(())
}

fn gen_jsonl(a: &Args) -> Result<(), BoxError> {
    a.check_known(&["n", "dim", "out", "seed", "distribution", "id-prefix"])?;
    let n: usize = a.get("n", 0)?;
    let dim: usize = a.get("dim", 0)?;
    if n == 0 || dim == 0 {
        return Err("--n and --dim are required and must be positive".into());
    }
    let out = a.values.get("out").ok_or("--out is required")?;
    let distribution = Distribution::parse(&a.get_str("distribution", "clustered"))
        .ok_or("invalid --distribution")?;
    let prefix = a.get_str("id-prefix", "v");
    let data = dataset::generate(
        &DatasetSpec {
            n,
            dim,
            seed: a.get("seed", 42)?,
            distribution,
        },
        0,
    );
    // A typed struct keeps f32 formatting short (serde_json::Value would
    // widen every component to f64 and print e.g. 0.10000000149011612).
    #[derive(Serialize)]
    struct Line<'a> {
        id: String,
        vector: &'a [f32],
    }
    let mut w = BufWriter::new(File::create(out)?);
    for (i, v) in data.base.iter().enumerate() {
        let line = serde_json::to_string(&Line {
            id: format!("{prefix}{i}"),
            vector: v,
        })?;
        writeln!(w, "{line}")?;
    }
    w.flush()?;
    eprintln!("wrote {n} records of dimension {dim} to {out}");
    Ok(())
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = raw.split_first() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let result = Args::parse(rest).and_then(|args| {
        let machine = machine_info();
        match cmd.as_str() {
            "run" => RunConfig::from_args(&args).and_then(|cfg| run(&cfg, &machine).map(|_| ())),
            "suite" => suite(&args, &machine),
            "gen-jsonl" => gen_jsonl(&args),
            "help" | "--help" | "-h" => {
                println!("{USAGE}");
                Ok(())
            }
            other => Err(format!("unknown command '{other}'\n{USAGE}").into()),
        }
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
