# Changelog

All notable changes to RiskANN are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries describe what the code does; measured results are recorded only after they have been produced.

## [Unreleased] — Phase III: retrieval, evaluation and observability

### Verification status of this entry

First real run: 2 November 2026, AMD Ryzen 7 7735HS laptop (14 GiB RAM, Linux), `rustc 1.99.1`.

| Check | Command | Result on 2 Nov | After the entry-point fix below |
|---|---|---|---|
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` | 5 lint errors (`useless_vec`, `manual_clamp`), fixed; then clean | *to be re-run* |
| Tests | `cargo test --workspace` | **106 passed, 0 failed** | *to be re-run* (109 tests) |
| Release build and server | `cargo build --release --workspace` | Built; server starts and serves `/ui` | *to be re-run* |
| Benchmarks | `riskann-bench suite --profile quick` | Recorded in README §25.7; revealed the recall failure below | *to be re-run* |

### Measured, diagnosed and fixed: recall collapse on clustered L2 data

**Symptom.** The first benchmark gave Recall@10 = 0.048 / 0.197 / 0.510 at `ef_search` 10 / 32 / 128 on 10 000 × 64-d L2 data in 32 clusters. The same data under cosine reached 0.948 / 1.000 / 1.000.

**Root cause.** With a single entry point, only 7 882 of 10 000 nodes were reachable. In 64-d, the points inside each Gaussian cluster are almost equidistant, so RobustPrune keeps same-cluster neighbours in all `r` slots and drops cross-cluster edges; clusters become islands. Search, storage and ID mapping were correct.

**Diagnosis method.** The JavaScript port of the engine (crates/server/ui) reproduced the Rust recall to three decimals. Construction variants were then measured on the same data (Recall@10 at `ef_search` 10 / 32 / 128):

| Variant | Reachable | Recall@10 |
|---|---:|---|
| Single pass, α 1.2 (Phase III as measured) | 79 % | 0.048 / 0.197 / 0.510 |
| Two passes, α 1.2 | 49 % | 0.039 / 0.129 / 0.219 |
| Vamana α 1.0 then 1.2 | 95 % | 0.447 / 0.501 / 0.547 |
| Random initial graph + two passes | 92 % | 0.401 / 0.460 / 0.519 |
| 16 entry points (farthest-point sampling) | 90 % | 0.501 / 0.603 / 0.754 |
| **64 entry points (chosen fix)** | **99.9 %** | **0.949 / 0.996 / 1.000** |

**No regressions elsewhere** (1 entry point → 64, Recall@10 at `ef_search` 10 / 32 / 128):

| Dataset | 1 entry point | 64 entry points |
|---|---|---|
| Cosine 10k × 64 | 0.948 / 1.000 / 1.000 | 0.952 / 1.000 / 1.000 |
| L2 1k × 3 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| L2 Gaussian 3k × 64 | 0.677 / 0.874 / 0.980 | 0.679 / 0.877 / 0.980 |
| L2 uniform 3k × 8 | 0.982 / 1.000 / 1.000 | 0.982 / 1.000 / 1.000 |
| L2 4k × 32, 32 clusters | 0.379 / 0.719 / 0.950 | 0.968 / 0.999 / 1.000 |

**Change.**
- `beam_search` and `vamana_search` take a list of entry points.
- Collections keep an entry-point set chosen by deterministic farthest-point sampling, of size `min(entry_points, ⌈√n⌉)`. `entry_points` is a new collection setting, default 64.
- The set is persisted in `metadata.json`, validated on open, and re-selected after incremental ingestion jobs. Search traces and graph snapshots include it.
- `riskann-bench --entry-points`.

**Compatibility.** Collections created earlier keep one entry point. Recreate them to benefit from the fix.

**Tests.** Two core tests (several entry points, validation) and an engine regression test (4 000 × 32-d, 32 clusters) were added.

### Lint fixes after the first Clippy run
- `useless_vec` in four core test fixtures, and `manual_clamp` in `default_l_build`.
- Pre-emptively, for lints Clippy had not yet reached: borrowed paths passed by value, and arrays instead of `vec!` in tests.

### Audit findings fixed

| # | Problem in the Phase-II code | Consequence | Fix |
|---|---|---|---|
| 1 | `Collection::load` re-opened storage with a hard-coded 10 000-slot `set_len` | Collections larger than 10 000 vectors were **truncated on restart** | Separate `DiskStore::create_new` / `open_existing`; capacity derived from file lengths; files are never shrunk |
| 2 | `DiskStore::write_*` wrote through `&self` using raw pointers without bounds checks | Undefined behaviour; possible out-of-bounds writes | Writes take `&mut self` and use checked ranges and safe slices; the only `unsafe` is mmap creation, with a safety argument |
| 3 | No synchronisation between search reads and upsert writes | Torn vectors or adjacency lists | One `RwLock` per collection |
| 4 | `greedy_search` used `partial_cmp().unwrap()` and re-sorted a `Vec` on each step | Panic on NaN; O(n²) frontier | Min-heap frontier and max-heap result set with a total `(distance, id)` order |
| 5 | `robust_prune` did not exclude the node itself | Self-edges after updating an existing id | Self, duplicate and non-finite candidates are discarded |
| 6 | Bulk build iterated `0..count` where `count` included duplicate ids in the file | Linking unallocated slots | Ids are allocated densely; duplicates are updates |
| 7 | Bulk ingestion `unwrap()`ed every file open, read and parse | A bad line killed the task silently after `202 Accepted` | Streaming reader with typed errors, line numbers, `fail_fast`/`skip`, job status |
| 8 | Metadata write errors were ignored (`let _ =`) | Silent data loss | Errors propagate; atomic temp-file + fsync + rename + directory fsync |
| 9 | The crate named `core` shadowed Rust's built-in `core` | Fragile macro expansion, unusable integration tests | Package renamed `riskann-core` (directory unchanged) |
| 10 | Unused adjacency slots were zero-filled | An unused slot read as "edges to node 0" | New and grown slots are filled with `u32::MAX` |
| 11 | Collection names were used as paths without validation | Path traversal (`../`) | Names restricted to `[A-Za-z0-9_-]{1,64}` |
| 12 | Construction beam fixed at `2r` | Low recall with small `r` (the demo used `r = 3`) | Configurable `l_build`, default `max(2r, 64)`; legacy collections keep `2r` |

### Added
- **Approximate search** `POST /collections/{name}/search` with `k`, `ef_search`, `include_distances` and `include_vectors`; deterministic ordering; per-query statistics.
- **Exact baseline** `POST /collections/{name}/search/exact`, plus `Collection::exact_search` as the Recall@K oracle.
- **Collection inspection**: `GET /collections`, `GET /collections/{name}`, `GET /collections/{name}/vectors/{id}`.
- **Health and readiness**: `GET /health`, `GET /ready` (returns `503` once shutdown has started).
- **Ingestion jobs**: `POST /collections/{name}/ingest` and `GET /collections/{name}/ingest/{job_id}`.
  - `queued → running → completed | failed` states and live counters.
  - Line-numbered rejection samples; `bulk_file` at creation now returns a `job_id`.
- **Storage growth**: storage doubles automatically; an interrupted resize is repaired on open.
- **Metadata format v1**: `external_ids`, `capacity` and `format_version`. Legacy format 0 is still read.
- **Dangling-edge cleanup** when a collection is opened.
- **Validation** of dimension, finite values, cosine zero vectors, `k`, `ef_search`, `dim`, `r`, `alpha`, `l_build`, capacity, external id length, collection names and ingestion paths (`RISKANN_INGEST_DIR`).
- **Uniform JSON error envelope** with stable error codes; internal details stay in the logs.
- **Structured logging (tracing)**:
  - Text or JSON output (`RISKANN_LOG_FORMAT`), filtered by `RUST_LOG`.
  - A per-request id (`x-request-id`) propagated into engine logs, and one access-log line per request.
  - Slow-query warnings (`RISKANN_SLOW_QUERY_MS`) and ingestion progress logs.
  - Startup configuration summary and a panic hook.
- **Request body limit** (`RISKANN_MAX_BODY_BYTES`) and graceful shutdown on Ctrl-C/SIGTERM.
- **`riskann-bench`**: deterministic datasets, Recall@K sweep over `ef_search`, latency percentiles, QPS, build/reload time, disk size, peak RSS; JSON and CSV output; `gen-jsonl` generator.
- **RiskANN Console** at `/` and `/ui`: graph explorer with animated beam search, construction stepper, recall measurement, ingestion monitor, storage slot viewer, in-page and live API tests, architecture walk-through, request inspector. Falls back to a labelled in-browser simulation when no server is reachable.
- **Visualisation endpoints**: `GET /collections/{name}/graph?limit=N`, and a `trace` option on search returning the expansion order.
- **109 tests** (106 before the entry-point fix): unit tests in `core`, `storage`, `engine` and `server`, plus engine and HTTP integration tests.
- **Documentation**: detailed `README.md`, `docs/DESIGN.md` and this changelog.

### Changed
- **Distances** are implemented in `riskann-core` (L2 = Euclidean, cosine = `1 − cos`, `f64` accumulation) instead of `anndists`.
- **Engine API is synchronous**; the server runs it on Tokio's blocking pool.
- **Response bodies**:
  - `PUT /collections/{name}` now returns a JSON body describing the collection.
  - Upsert returns `{id, internal_id, created}`.
  - Delete returns `{deleted}`.
  - Status codes are unchanged.
- **`initial_capacity`** is now optional (default 1000).

### Removed
- The `anndists` dependency.
- `greedy_search` (replaced by `beam_search` / `vamana_search`).
- `Collection::bulk_insert` (replaced by `Collection::ingest`).

### Known limitations
- Vectors cannot be deleted individually.
- The graph build is a single pass, and the entry point is fixed after the first build.
- Writes are serialised per collection. HTTP upserts persist metadata on every call (O(n) ids).
- Job status is process-local.
- There is no authentication.

See README §28 for the full list.

## [Phase II] — collection management and ingestion prototype
- Rust workspace with `core`, `storage`, `engine` and `server` crates.
- `PUT`/`DELETE /collections/{name}` and `POST /collections/{name}/upsert`.
- JSONL bulk ingestion scheduled in a background task (`202 Accepted`).
- Memory-mapped `vectors.bin`/`graph.bin`; Vamana-inspired greedy search and robust pruning.
