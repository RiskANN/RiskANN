# RiskANN

**RiskANN is an experimental Rust-based, Vamana-inspired approximate nearest-neighbour vector database that combines graph-based retrieval with persistent memory-mapped vector and adjacency storage.**

| | |
|---|---|
| Team | OSDBMS-V-2026-T305, Department of CSE, Graphic Era (Deemed to be University), Dehradun |
| Course | Project-Based Learning, OS / DBMS domain, Academic Session 2026–27 |
| Repository | <https://github.com/RiskANN/RiskANN> |
| License | MIT |
| Language | Rust (edition 2024, Rust ≥ 1.85) |

> **Status: experimental, not production-ready.**
> - Approximate search, an exact brute-force baseline, persistent storage, ingestion jobs, structured logging and a benchmark harness are implemented.
> - Benchmark results measured on an AMD Ryzen 7 7735HS laptop are recorded in §25.7, including a recall failure they exposed and the fix that followed.
> - The Phase-I target of 95 % Recall@K is a goal, **not a claim**.
> - The [verification record](#3-status-and-verification-record) lists the build and test results. Fill it in before relying on this revision.

Related documents: [`CHANGELOG.md`](CHANGELOG.md) (what changed and why, including the audit of the Phase-II code) · [`docs/DESIGN.md`](docs/DESIGN.md) (design decisions and invariants).

---

## Table of contents

1. [Overview](#1-overview)
2. [Project context](#2-project-context)
3. [Status and verification record](#3-status-and-verification-record)
4. [Features](#4-features)
5. [Concepts primer](#5-concepts-primer)
6. [Architecture](#6-architecture)
7. [Request lifecycles](#7-request-lifecycles)
8. [Workspace structure](#8-workspace-structure)
9. [Technology stack](#9-technology-stack)
10. [Getting started](#10-getting-started)
11. [Configuration](#11-configuration)
12. [Logging and observability](#12-logging-and-observability)
13. [API conventions](#13-api-conventions)
14. [API reference: collections](#14-api-reference-collections)
15. [API reference: vectors](#15-api-reference-vectors)
16. [API reference: search](#16-api-reference-search)
17. [API reference: ingestion jobs](#17-api-reference-ingestion-jobs)
18. [API reference: health and readiness](#18-api-reference-health-and-readiness)
19. [Error reference](#19-error-reference)
20. [Distance semantics](#20-distance-semantics)
21. [Tuning: `k`, `ef_search`, `r`, `alpha`, `l_build`](#21-tuning-k-ef_search-r-alpha-l_build)
22. [On-disk storage format](#22-on-disk-storage-format)
23. [Persistence, crash behaviour and concurrency](#23-persistence-crash-behaviour-and-concurrency)
24. [Testing guide](#24-testing-guide)
25. [Benchmarks (`riskann-bench`)](#25-benchmarks-riskann-bench)
26. [Recall@K methodology and reporting](#26-recallk-methodology-and-reporting)
27. [Evaluation demo walkthrough](#27-evaluation-demo-walkthrough)
28. [Limitations and roadmap](#28-limitations-and-roadmap)
29. [Troubleshooting](#29-troubleshooting)
30. [FAQ](#30-faq)
31. [Security and trust boundaries](#31-security-and-trust-boundaries)
32. [Contributing and coding conventions](#32-contributing-and-coding-conventions)
33. [Glossary](#33-glossary)
34. [References](#34-references)
35. [License](#35-license)

---

## 1. Overview

**Problem.** Vector-search applications, such as semantic search, recommendation and retrieval-augmented generation, need to find the vectors most similar to a query vector. An exact scan compares the query with every stored vector, so its cost grows linearly with the collection. Graph-based approximate indexes answer much faster by walking a navigable graph, but in-memory implementations keep the whole index in RAM.

**What RiskANN does.**
- **Storage.** Vectors and graph adjacency lists live in memory-mapped binary files, so the operating system pages them in and out.
- **Index.** A Vamana-style proximity graph is built over the vectors, following the graph used by DiskANN (Subramanya et al., NeurIPS 2019).
- **Retrieval.** Queries run a bounded best-first beam search from a set of entry points spread across the data.
- **Interfaces.**
  - An HTTP/JSON API for collections, vectors, search and bulk ingestion.
  - An exact brute-force search used as ground truth.
  - A benchmark tool that measures recall, latency and build cost.

**What RiskANN is not.**
- It is not a full DiskANN implementation: there is no product quantisation, no SSD-optimised layout and no beam I/O.
- It is not a distributed or multi-tenant database.
- It is not production-ready.
- Memory mapping by itself does **not** guarantee a small RAM footprint (§28).

## 2. Project context

### 2.1 Team and mentor

| Member | Roll no. | Planned role (Phase-I plan) |
|---|---|---|
| Utkarsh Pant | 24021473 | Storage / index: vector storage, Vamana construction, persistence, integration |
| Kartikay Srivastava | 240211311 | Retrieval: query traversal, candidate management, top-k results, Recall@K |
| Mrigansh Chaudhary | 24021776 | Optimisation: I/O, `O_DIRECT`, SIMD and benchmarking exploration |

Mentor: (Prof.) Dr. Sumit Pundir. The roles above are the **Phase-I plan**; confirm actual responsibilities against the commit history.

### 2.2 Phases

| Phase | Scope | State |
|---|---|---|
| I | Requirements: collections, ids, graph/metadata design, evaluation goals (Recall@K, latency, memory, indexing time) | Done |
| II | Workspace split into crates; create/delete/upsert routes; JSONL bulk ingestion; mmap persistence; graph visualisation | Done (prototype) |
| III | Retrieval API, exact baseline, Recall@K, benchmarks, storage-safety fixes, ingestion status, tests, logging, documentation | **Implemented in this revision**; verification and benchmark results to be recorded (§3) |

### 2.3 Course concepts reflected in the code

| Course | Concept | Where |
|---|---|---|
| Operating Systems | Memory-mapped files (`mmap`), page cache, `msync`/`fsync`, atomic `rename`, file growth (`ftruncate`) | `crates/storage/src/memory.rs`, `write_atomic` in `crates/engine/src/collection.rs` |
| Operating Systems | Reader–writer locks, thread pools (Tokio blocking pool), signals (SIGTERM/Ctrl-C), graceful shutdown | `collection.rs`, `crates/server/src/api/mod.rs`, `main.rs` |
| DBMS | Collections as tables, catalogue/metadata, keys (external ↔ internal ids), upsert semantics | `collection.rs`, `db.rs` |
| DBMS | Durability via write ordering (data → sync → metadata), recovery checks on open, concurrency control | §23, `Collection::open` |
| DBMS | Index structure (proximity graph) and query processing (k-NN), exact versus approximate plans | `crates/core/src/vamana.rs`, `Collection::search` / `exact_search` |
| Systems programming | Ownership preventing aliasing during remap, an audited `unsafe` block, typed errors, deterministic testing | `storage`, `error.rs`, tests |

## 3. Status and verification record

### 3.1 Feature status

| Implemented, with automated tests | Implemented, not yet benchmarked | Planned / future work |
|---|---|---|
| Collection create / inspect / list / delete | ANN recall, latency and QPS at 10k–100k vectors (harness exists, no results recorded) | Deleting individual vectors (tombstones) |
| Upsert by external id (internal id kept on update) | Concurrent QPS (`--threads`) | Metadata filtering |
| Approximate search, exact search | Peak memory and reload time | Durable job status across restarts |
| Input validation for every field | | Authentication / authorisation / TLS |
| Persistent storage, reload without truncation, automatic growth, legacy format read | | Two-pass Vamana build, entry-point recomputation |
| Ingestion jobs with status, counters and line-numbered errors | | SIMD, quantisation, SSD-specific I/O (only after profiling) |
| Request ids, access log, slow-query log, JSON logs | | Metrics endpoint (Prometheus) |
| Health/readiness, body limit, graceful shutdown | | |
| Deterministic datasets, Recall@K and benchmark harness | | |

### 3.2 Verification record

Recorded on 2 November 2026 on an AMD Ryzen 7 7735HS laptop (14 GiB RAM, Linux), `rustc 1.99.1`.

| Check | Command | Revision measured on 2 Nov | Current revision (multiple entry points) |
|---|---|---|---|
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` | Clean, after the lint fixes in `CHANGELOG.md` | *to be re-run* |
| Tests | `cargo test --workspace` | **106 passed, 0 failed** | *to be re-run* (109 tests) |
| Release build and server | `cargo build --release --workspace` | Built; server starts, loads data and serves `/ui` | *to be re-run* |
| Benchmarks | `riskann-bench suite --profile quick` | Recorded in §25.7; exposed the clustered-L2 recall failure | *to be re-run* |
| Live API self-test | Console → Run tests → live server | *not yet reported* | *to be re-run* |

The fix for the recall failure was checked against the console's JavaScript port of the engine, which reproduced the measured Rust numbers to three decimals before the change. Replace the "to be re-run" cells with the results of the current revision.

## 4. Features

**Index and retrieval**
- **Vamana-inspired proximity graph.** Every node has at most `r` out-edges, chosen by RobustPrune with parameter `alpha` (α ≥ 1) so that edges point in diverse directions.
- **Two build modes.**
  - Incremental insertion links each new vector immediately (§5.3).
  - Bulk loading into an empty collection first chooses a medoid near the centroid, then links every node.
- **Approximate k-NN search.** A bounded best-first beam search with exploration budget `ef_search`. It is deterministic: equal distances are ordered by the lower internal id.
- **Exact k-NN search.** A separately named brute-force path using the same metric and ordering. It serves as the ground-truth oracle and is never a hidden fallback.
- **Metrics.** Euclidean (`L2`) and cosine distance (`COSINE`), both implemented and unit-tested in `riskann-core`.

**Data management**
- **Collections.** Each collection has its own dimension, metric and graph parameters.
- **Stable ids.** External string ids map to dense internal ids; an upsert keeps the internal id.
- **Growing storage.** Storage doubles automatically, and capacity is always derived from the files, so restart never truncates data.
- **Durable metadata.** Metadata is written with temp-file, fsync and rename. Corrupt or inconsistent collections are reported, never silently re-initialised.
- **Streaming ingestion jobs.** JSONL files are streamed with `fail_fast` or `skip` policies, line-numbered errors, live counters and explicit `completed` / `failed` states.

**Operations**
- **Structured logs.** Output is text or JSON with a per-request id that also appears in engine logs. Ingestion logs its progress, slow queries are flagged, the startup configuration is summarised and panics are logged.
- **One JSON error envelope** with stable codes. File paths and internals never reach clients.
- **Server hygiene.** Request body limit, health and readiness probes, graceful shutdown.

**Evaluation**
- **`riskann-bench`.** Deterministic synthetic data, an `ef_search` sweep, Recall@K, latency p50/p95/p99, QPS, build and reload time, disk size, peak RSS, and JSON + CSV output.

## 5. Concepts primer

This section explains the ideas the code implements. Readers who know Vamana/HNSW can skip to §6.

### 5.1 Approximate nearest-neighbour (ANN) search

Given a query vector `q` and a set of vectors `X`, the **k nearest neighbours** are the `k` vectors with the smallest distance to `q`.

| Approach | Accuracy | Cost per query |
|---|---|---|
| Exact (brute force) | Always exact | `n · dim` operations |
| Approximate (graph index) | Usually a high fraction of the true neighbours (**recall**) | Visits a small subset of the vectors |

### 5.2 The proximity graph

Each vector is a node, and each node stores up to `r` directed edges to other nodes. The graph is built so that a **greedy walk** that always moves towards the query reaches the query's neighbourhood in a few hops. Two properties make that possible:

- **Short edges** connect a node to its close neighbours, which gives local accuracy.
- **Diverse edges**, kept by RobustPrune, leave each node in different directions, which gives fast navigation.

```text
RobustPrune(p, candidates, r, α):
  sort candidates by distance to p (ties → lower id)
  for each candidate c, nearest first:
      if some already-selected s has α · d(s, c) ≤ d(p, c):
          drop c          # c is "behind" s as seen from p; s already covers it
      else:
          select c
      stop when r edges are selected
```

With α = 1 only strictly "uncovered" candidates survive. With α > 1 (e.g. 1.2) a few more long edges survive, which keeps the graph navigable.

### 5.3 Building the graph

**Inserting one vector `v` with internal id `i`:**
1. Run a beam search for `v` from the entry points with beam width `l_build`.
2. The candidate pool is every node expanded during that search plus the final beam.
3. `edges(i) = RobustPrune(i, pool, r, α)`.
4. For each new neighbour `j`, add the reverse edge `j → i`. If `j` now has more than `r` edges, re-prune `j`'s list.

**Bulk build (empty collection):**
1. Store all vectors.
2. Choose the entry points: the medoid (the stored vector nearest the centroid), then further vectors by farthest-point sampling, up to `min(64, ⌈√n⌉)` (§5.6).
3. Insert (link) every other vector in id order using the steps above.

### 5.4 Searching the graph (beam search)

```text
frontier  ← min-heap {entry points} # nearest unexpanded first
results   ← best ef entry points    # max-heap of at most ef; top = worst
visited   ← {entry points}
loop:
    c ← pop nearest from frontier (stop if frontier empty)
    if |results| = ef and c is worse than worst(results): stop
    for each neighbour u of c not yet visited:
        mark u visited; compute d(u, q)
        if |results| < ef or u better than worst(results):
            push u to frontier and results; drop worst if |results| > ef
return the best k of results, ascending by (distance, id)
```

`ef_search` is the size of `results`. A larger value explores more nodes, so recall rises along with latency.

### 5.5 Recall@K

`Recall@K = |approximate top-K ∩ exact top-K| / K`, averaged over many queries. For example, 0.95 means that on average 95 % of the true 10 nearest neighbours were returned when K = 10. RiskANN computes it against its own exact search (§26).

### 5.6 Why several entry points

A greedy walk only finds what it can reach going "downhill" from where it starts. Data made of many well-separated clusters breaks that assumption.

- **In high dimensions, every other cluster looks equally far away.** From inside the wrong cluster, no neighbour is noticeably closer to the query, so the walk has no direction to follow.
- **Pruning cannot make room for long edges.** Points inside one cluster are almost equidistant from each other, so RobustPrune finds nothing to prune; all `r` edge slots fill with same-cluster neighbours, and clusters become islands.
- **One entry point therefore reaches only part of the graph.** In the first benchmark run (§25.7), only 79 % of a 10 000-vector collection was reachable from its single entry point, and Recall@10 was 0.20 at `ef_search` 32.

RiskANN therefore starts every search from a **set** of entry points chosen by **farthest-point sampling**:
1. Start with the medoid.
2. Repeatedly add the vector that is farthest from every entry point chosen so far (ties go to the lower id).

This gives every cluster a nearby starting point. The set size is `min(entry_points, ⌈√n⌉)`, with `entry_points` = 64 by default. It costs one distance computation per entry point per query, and is persisted in `metadata.json`.

## 6. Architecture

```text
 Client (curl / application)
        │  HTTP + JSON
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │ server   (Axum 0.7 on Tokio)                             │
 │  logging middleware: request id, span, access log        │
 │  body limit · JSON extraction · validation · error codes │
 │  handlers → spawn_blocking (span propagated)             │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ engine                                                   │
 │  DBInfo: registry of collections + ingestion job registry│
 │  Collection: RwLock<IndexState>                          │
 │    IndexState = DiskStore + external_ids + id_map + medoid│
 │  upsert/link · search · exact_search · ingest · persist  │
 └──────────────┬──────────────────────────┬────────────────┘
                ▼                          ▼
 ┌────────────────────────┐   ┌──────────────────────────────┐
 │ riskann-core           │   │ storage                      │
 │ distances (L2, cosine) │   │ DiskStore                    │
 │ beam_search            │   │  vectors.bin  (mmap)         │
 │ vamana_search          │   │  graph.bin    (mmap)         │
 │ robust_prune           │   │  checked reads/writes, resize│
 └────────────────────────┘   └──────────────────────────────┘
```

| Crate | Responsibility | Knows about |
|---|---|---|
| `riskann-core` | Distance functions, candidate ordering, beam search, RobustPrune | Nothing else: storage is accessed only through closures |
| `storage` | Two memory-mapped files per collection; bounds-checked vector/edge I/O; growth | Files only |
| `engine` | Collections, id mapping, graph maintenance, search, persistence, ingestion jobs, datasets, benchmarks | `riskann-core`, `storage` |
| `server` | HTTP routes, request/response types, error codes, logging, configuration | `engine` |

**Threading model.**
- **Async side.** Tokio runs the HTTP server.
- **Engine side.** Engine calls are synchronous: they touch mmap pages, hold locks and call `fsync`. Handlers therefore run them with `tokio::task::spawn_blocking`, which keeps the async worker threads responsive.
- **Ingestion.** Each ingestion job runs on its own blocking thread.
- **Locking.** Searches share a collection's read lock; writes take the write lock.

## 7. Request lifecycles

### 7.1 Search (`POST /collections/{name}/search`)

1. **Middleware.** `log_requests` assigns a request id and opens a `request` span.
2. **Body.** The body limit is checked. The JSON body is parsed into `SearchReq`; unknown fields are rejected (`MALFORMED_REQUEST`).
3. **Lookup.** The collection name is validated and looked up (`COLLECTION_NOT_FOUND`).
4. **Options.** `QueryOptions` are built, with `ef_search` defaulting to `max(k, 64)`.
5. **Blocking thread.** The engine work runs on a blocking thread inside the request span:
   1. Validate `k`, `ef_search`, the vector dimension and finiteness, and for COSINE that the vector is non-zero.
   2. Take the collection **read** lock.
   3. Run the beam search from the persisted entry point; map internal ids to external ids.
   4. Release the lock and log `search executed` at `debug`.
6. **Response.** The handler logs `slow query` at `warn` if the engine time exceeds the threshold, then serialises the response.
7. **Access log.** The middleware logs `request completed` with status and latency and sets the `x-request-id` response header.

### 7.2 Upsert (`POST /collections/{name}/upsert`)

1. Validate the id (1–512 bytes) and the vector.
2. Take the collection **write** lock.
3. **Existing id:** overwrite its vector in place and re-link the node. **New id:**
   1. Grow the storage if it is full.
   2. Write the vector and an empty edge list.
   3. Link the node (§5.3).
   4. Publish the id. The first vector of an empty collection becomes the entry point.
4. Persist: `msync` both files, then atomically rewrite `metadata.json`.
5. Release the lock, log `upsert applied` at `debug`, and return `{id, internal_id, created}`.

### 7.3 Ingestion job (`POST /collections/{name}/ingest`)

1. Validate the path (absolute, an existing regular file, inside `RISKANN_INGEST_DIR` if that is set).
2. Register the job (`queued`) and start a blocking worker. A watcher marks the job `failed` if the worker panics.
3. The worker marks the job `running`, opens an `ingest_job` span, and streams the file line by line.
4. **Empty collection → initial build.**
   1. Under the write lock, store all vectors.
   2. Choose the medoid and link every node.
   3. Persist once at the end.
5. **Non-empty collection → incremental.** Each record is inserted under the write lock. A checkpoint is persisted every `checkpoint_every` records and at the end.
6. The job ends `completed` or `failed`, with counters and the first 20 rejected lines.

## 8. Workspace structure

```text
RiskANN/
├── Cargo.toml                 workspace manifest and shared dependency versions
├── Cargo.lock                 locked dependency versions
├── README.md                  this document
├── CHANGELOG.md               change log, audit findings, verification record
├── LICENSE                    MIT
├── docs/
│   ├── DESIGN.md              design decisions and invariants (15 questions)
│   └── README.md              pointer to this file
└── crates/
    ├── core/                  package `riskann-core` (lib name `riskann_core`)
    │   └── src/
    │       ├── lib.rs         re-exports
    │       ├── distances.rs   DistanceMetric, L2, cosine, checked variants (+ tests)
    │       └── vamana.rs      Candidate, beam_search, vamana_search, robust_prune (+ tests)
    ├── storage/
    │   └── src/
    │       ├── lib.rs
    │       └── memory.rs      DiskStore: create_new / open_existing / resize, checked I/O (+ tests)
    ├── engine/
    │   ├── src/
    │   │   ├── lib.rs         re-exports and the `limits` module
    │   │   ├── collection.rs  Collection: config, metadata, upsert/link, search, exact search, ingest
    │   │   ├── db.rs          DBInfo: load data directory, create/delete/list collections
    │   │   ├── ingest.rs      JSONL reader, ingestion options, progress, job registry, path policy
    │   │   ├── dataset.rs     SplitMix64 generator and synthetic datasets
    │   │   ├── error.rs       EngineError
    │   │   └── bin/riskann_bench.rs   benchmark, Recall@K and JSONL generator CLI
    │   └── tests/engine.rs    engine integration tests
    └── server/
        ├── src/
        │   ├── main.rs        binary: config, logging init, startup load, graceful shutdown
        │   ├── lib.rs         build_router (all routes and layers)
        │   ├── state.rs       ServerConfig (env), AppState (readiness)
        │   ├── error.rs       ApiError and the JSON error envelope
        │   ├── logging.rs     log formats, request-id middleware, panic hook
        │   └── api/           handlers: collections, upsert, search, vectors, ingest, health
        └── tests/api.rs       HTTP integration tests (in-process router, no socket)
```

The core package was renamed from `core` to `riskann-core`; its directory is unchanged. A crate named `core` shadows Rust's built-in `core` crate in every crate that depends on it.

## 9. Technology stack

| Concern | Choice | Why |
|---|---|---|
| Language | Rust 2024 | Memory safety around raw file mappings; zero-cost abstractions |
| HTTP | Axum 0.7 | Typed extractors, Tower middleware |
| Runtime | Tokio | Async I/O, blocking pool, signals |
| Serialisation | Serde, serde_json | Typed request/response and metadata |
| Storage | memmap2, bytemuck | Memory-mapped files; safe `[u8]` ↔ `[f32]/[u32]` casts with alignment checks |
| Concurrent maps | DashMap | Collection and job registries |
| Logging | tracing, tracing-subscriber (env-filter, json) | Structured events, spans, `RUST_LOG` filtering |
| Configuration | Environment variables, dotenvy | `.env` support |
| Tests | Rust test harness, tower `ServiceExt` | In-process HTTP tests |

There is no random-number crate. Datasets use a SplitMix64 implementation in `dataset.rs`, so the generated data is identical on every platform and dependency version. `anndists` was removed in favour of explicit, tested distance functions (see `CHANGELOG.md`).

## 10. Getting started

### 10.1 Prerequisites

| Requirement | Notes |
|---|---|
| Rust ≥ 1.85 | Install with [rustup](https://rustup.rs/); `rustup update` if older |
| Linux or macOS | Linux is the primary platform. Windows is untested: deleting a collection there can fail while its files are mapped. |
| Internet access on the first build | Cargo downloads crates from crates.io and updates `Cargo.lock` |
| `curl` (optional) | For the examples; `jq` helps read JSON output |

### 10.2 Build

```bash
git clone https://github.com/RiskANN/RiskANN.git
cd RiskANN
rustc --version                       # >= 1.85
cargo build --release --workspace     # binaries in target/release/
```

### 10.3 First run

```bash
DATA_LOC=./data cargo run --release -p server
```

Expected startup log (text format; timestamps and values illustrative):

```text
2026-11-02T10:00:00.000001Z  INFO server: starting RiskANN server version="1.0.0" host=0.0.0.0 port=3000 data_dir=./data ingest_dir=None max_body_bytes=8388608 slow_query_ms=250 log_format="text"
2026-11-02T10:00:00.004210Z  INFO server: startup loading finished loaded=[] failed=[] skipped=[] elapsed_ms=0
2026-11-02T10:00:00.004300Z  WARN server: RISKANN_INGEST_DIR is not set: ingestion accepts any absolute file path readable by the server process
2026-11-02T10:00:00.004512Z  INFO server: RiskANN server listening address=0.0.0.0:3000
```

### 10.4 Smoke test (second terminal)

```bash
curl -s localhost:3000/health                      # {"status":"ok"}
curl -s -X PUT localhost:3000/collections/demo -H 'Content-Type: application/json' \
  -d '{"dim":3,"metric":"L2","r":16,"alpha":1.2}'
curl -s -X POST localhost:3000/collections/demo/upsert -H 'Content-Type: application/json' \
  -d '{"id":"a","vector":[0.1,0.2,0.3]}'
curl -s -X POST localhost:3000/collections/demo/search -H 'Content-Type: application/json' \
  -d '{"vector":[0.1,0.2,0.3],"k":1}'
```

The last command returns `"results":[{"id":"a","internal_id":0,"distance":0.0}]`.

### 10.5 Open the console

With the server running, open **<http://localhost:3000/ui>** (or just `http://localhost:3000/`). The RiskANN Console is a single page compiled into the server binary (`crates/server/ui/index.html`). It uses only the public HTTP API, so everything it shows is real server behaviour.

| Section | What you can do |
|---|---|
| Explore the graph | See every vector and edge (2-D data directly, higher dimensions as a PCA projection). Click to place a query and watch the beam search step by step: entry point, expanded nodes, nodes reached, final results against the exact answer. Shows Recall@k, the work done, and the work saved compared with exact search. The animation is a replay of the same algorithm on the downloaded graph, checked against the server's own `trace`. |
| How the graph is built | Step through Vamana insertion on a small 2-D set: beam search, candidate pool, RobustPrune's kept and pruned candidates (with the neighbour that "covers" each pruned one), reverse edges and re-pruning. Incremental and bulk modes; adjustable `r`, `alpha` and `l_build`. Runs locally in the page. |
| Measure recall | Sweep `ef_search` against a server collection and chart Recall@k, nodes reached and the server-reported engine time. |
| Ingest a file | Start a JSONL ingestion job from a server-side path and follow its state machine, counters and rejected lines live. |
| Storage files | Slot map of `vectors.bin` and `graph.bin` (live, free, entry point). Click a slot to see its byte offsets, float values, hex bytes and its padded `u32` neighbour list, plus `config.json` and `metadata.json`. |
| Run tests | Run the in-page algorithm tests, and run the API self-test **against the live server**: it creates a temporary collection, exercises every endpoint and error code, then deletes it. |
| Architecture | Clickable crate diagram, and an animated walk-through of the search, upsert and ingestion request paths. |
| Requests and logs | Every request the console made, with status, latency, `x-request-id`, and the full request and response JSON. |

When the page cannot reach a RiskANN server, for example when opened as a standalone file, it switches to an **in-browser simulation**: a JavaScript port of the same algorithms behind the same API, labelled as such on screen. Timings shown in simulation are JavaScript timings, not RiskANN performance. When both are available, a button switches between them.

## 11. Configuration

### 11.1 Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address. Use `127.0.0.1` to accept local connections only. |
| `PORT` | `3000` | Bind port (0–65535). |
| `DATA_LOC` | `/tmp/db_data` | Data directory: one sub-directory per collection. Created if missing. |
| `RISKANN_INGEST_DIR` | *(unset)* | If set, every ingestion file must resolve (after following symlinks and `..`) inside this directory. |
| `RISKANN_MAX_BODY_BYTES` | `8388608` | Maximum request body (bytes); larger bodies get `413`. |
| `RISKANN_SLOW_QUERY_MS` | `250` | Searches whose engine time is ≥ this many ms are logged at `warn`. `0` disables. |
| `RISKANN_LOG_FORMAT` | `text` | `text` (human-readable) or `json` (one object per line). |
| `RUST_LOG` | `info` | Log filter (§12.3). |

Invalid values (for example `PORT=abc`) stop the server at startup with an error message. `/tmp/db_data` is kept for compatibility with the Phase-II demo; many systems clear `/tmp` on reboot, so use a persistent `DATA_LOC` for real data.

### 11.2 `.env` file

Values in a `.env` file in the working directory are loaded at startup; real environment variables take precedence.

```dotenv
HOST=127.0.0.1
PORT=3000
DATA_LOC=/var/lib/riskann
RISKANN_INGEST_DIR=/var/lib/riskann-import
RISKANN_SLOW_QUERY_MS=100
RISKANN_LOG_FORMAT=json
RUST_LOG=info,engine=debug
```

### 11.3 Limits

The limits are defined in `crates/engine/src/lib.rs`, module `limits`, and are enforced on every request.

| Limit | Value | Error when violated |
|---|---|---|
| `k` | 1 – 1 000 (default 10) | `INVALID_ARGUMENT` |
| `ef_search` | `k` – 4 096 (default `max(k, 64)`) | `INVALID_ARGUMENT` |
| `dim` | 1 – 16 384 | `INVALID_ARGUMENT` |
| `r` | 1 – 256 | `INVALID_ARGUMENT` |
| `alpha` | 1.0 – 10.0, finite | `INVALID_ARGUMENT` |
| `l_build` | 1 – 1 024 (default `max(2r, 64)`) | `INVALID_ARGUMENT` |
| `initial_capacity` | 1 – 50 000 000 (default 1 000) | `INVALID_ARGUMENT` |
| Total vectors per collection | 2³² − 2 | `CAPACITY_EXCEEDED` |
| External id | 1 – 512 bytes | `INVALID_ARGUMENT` |
| Collection name | 1 – 64 characters from `[A-Za-z0-9_-]` | `INVALID_COLLECTION_NAME` |
| JSONL line | ≤ 16 MiB | Job fails (`failed`) |
| Request body | `RISKANN_MAX_BODY_BYTES` | `PAYLOAD_TOO_LARGE` (413) |

## 12. Logging and observability

RiskANN logs through the [`tracing`](https://docs.rs/tracing) ecosystem. Log events are **structured**: each has a level, a target (the Rust module), a message and typed fields, and they run inside **spans** that carry context such as the request id or the ingestion job id.

### 12.1 Where logs go

Logs are written to **standard output**. To keep them in a file:

```bash
cargo run --release -p server > riskann.log 2>&1        # file only
cargo run --release -p server 2>&1 | tee riskann.log    # file and terminal
```

In text format, colours are enabled only when stdout is a terminal, so log files contain no escape codes.

### 12.2 Formats

**Text** (default, `RISKANN_LOG_FORMAT=text`), illustrative:

```text
2026-11-02T10:05:12.417230Z  INFO request{request_id=req-7 method=POST path=/collections/demo/search}: server::logging: request completed status=200 latency_ms=0.84
```

The line reads as: timestamp (UTC) · level · span with its fields · target · message · event fields.

**JSON** (`RISKANN_LOG_FORMAT=json`), one object per line, illustrative:

```json
{"timestamp":"2026-11-02T10:05:12.417230Z","level":"INFO","fields":{"message":"request completed","status":200,"latency_ms":0.84},"target":"server::logging","span":{"request_id":"req-7","method":"POST","path":"/collections/demo/search","name":"request"}}
```

Use JSON when logs are shipped to a collector or analysed with `jq`.

### 12.3 Choosing verbosity with `RUST_LOG`

| Goal | `RUST_LOG` |
|---|---|
| Normal operation (default) | `info` |
| Quiet: only problems | `warn` |
| See every search and upsert with its statistics | `info,engine=debug` |
| Also see client-error messages and health-probe requests | `info,engine=debug,server=debug` |
| See every metadata persist | `info,engine::collection=trace` |
| Everything, including dependency internals | `debug` (very verbose) |

The filter syntax is `level` or `target=level`, comma-separated; the most specific target wins.

### 12.4 Levels used

| Level | Used for |
|---|---|
| `error` | 5xx responses, storage/I/O failures, failed collection loads, panics, failed bulk builds |
| `warn` | 4xx responses, slow queries, failed ingestion jobs, repaired files or dangling edges on open, missing `RISKANN_INGEST_DIR` |
| `info` | Startup/shutdown, one access-log line per request, collection create/open/delete, storage growth, ingestion start/progress/finish |
| `debug` | Per-search and per-upsert statistics, client-error messages, rejected JSONL lines, checkpoints, health-probe requests, collection registration |
| `trace` | Every metadata persist |

### 12.5 Request ids and spans

- **One id per request.** Every HTTP request runs inside a `request` span with `request_id`, `method` and `path`.
  - A valid incoming `x-request-id` header (1–64 characters from `[A-Za-z0-9._-]`) is reused.
  - Otherwise the server generates `req-<n>`.
- **Returned to the client.** The id is sent back in the **`x-request-id` response header**, on errors too.
- **Propagated into the engine.** The span is carried onto the blocking thread, so engine events (`search executed`, `upsert applied`, …) for that request carry the same `request_id`.
- **Ingestion jobs** run in their own `ingest_job` span with `job` and `collection` fields; a job outlives the request that started it.

To follow one request end to end:

```bash
curl -si -X POST localhost:3000/collections/demo/search -H 'Content-Type: application/json' \
     -H 'x-request-id: demo-query-1' -d '{"vector":[0.1,0.2,0.3],"k":3}' | grep -i x-request-id
grep 'demo-query-1' riskann.log                                      # text logs
jq -c 'select(.span.request_id == "demo-query-1")' riskann.log       # JSON logs
```

### 12.6 Event catalogue

The **target** is the Rust module that emits the event; `server` is the binary's `main.rs`.

**Server lifecycle**

| Message | Level | Target | Fields |
|---|---|---|---|
| `starting RiskANN server` | info | `server` | `version, host, port, data_dir, ingest_dir, max_body_bytes, slow_query_ms, log_format` |
| `startup loading finished` | info | `server` | `loaded, failed, skipped, elapsed_ms` |
| `RISKANN_INGEST_DIR is not set: …` | warn | `server` | — |
| `RiskANN server listening` | info | `server` | `address` |
| `shutdown signal received; readiness withdrawn, draining in-flight requests` | info | `server` | — |
| `HTTP server stopped; waiting for running ingestion jobs to finish` | info | `server` | — |
| `server failed` | error | `server` | `error` |
| `panic` | error | `server::logging` | `location, panic` |

**Requests**

| Message | Level | Target | Fields |
|---|---|---|---|
| `request completed` | info (debug for `/health`, `/ready`) | `server::logging` | `status, latency_ms` + span |
| `request rejected` | warn | `server::logging` | `status, error_code, latency_ms` + span |
| `request failed` | error | `server::logging` | `status, error_code, latency_ms` + span |
| `client error detail` | debug | `server::error` | `code, message` |
| `internal error while handling request` | error | `server::error` | `error` (full internal detail, may include paths) |
| `slow query` | warn | `server::api::search` | `collection, mode, k, ef_search, visited, distance_computations, duration_us, threshold_ms` |
| `blocking task panicked or was cancelled` | error | `server::api` | `error` |

**Collections and queries**

| Message | Level | Target | Fields |
|---|---|---|---|
| `created collection` | info | `engine::db` | `collection, dim, metric, r, alpha, l_build, initial_capacity` |
| `opened collection` | info | `engine::collection` | `collection, vectors, capacity, dim, metric, r, l_build, format_version, elapsed_ms` |
| `registered collection` | debug | `engine::db` | `collection` |
| `skipping directory that is not a collection` | info | `engine::db` | `dir` |
| `failed to load collection; it is left untouched on disk and will not be served` | error | `engine::db` | `collection, error` |
| `deleting collection: waiting for in-flight writes` / `deleted collection` | info | `engine::db` | `collection` |
| `data files had different capacities (interrupted resize); grew the shorter file` | warn | `engine::collection` | `collection, vectors_capacity, graph_capacity` |
| `metadata capacity differs from data files; using the file-derived capacity` | info | `engine::collection` | `collection, recorded, actual` |
| `removed dangling/invalid edges on open` | warn | `engine::collection` | `collection, removed_edges` |
| `grew collection storage` | info | `engine::collection` | `collection, from, to` |
| `search executed` | debug | `engine::collection` | approximate: `collection, mode, k, ef_search, visited, expanded, returned, duration_us`; exact: `collection, mode, k, scanned, returned, duration_us` |
| `upsert applied` | debug | `engine::collection` | `collection, internal_id, created, elapsed_us` |
| `data flushed and metadata persisted` | trace | `engine::collection` | `collection, vectors` |

**Ingestion**

| Message | Level | Target | Fields |
|---|---|---|---|
| `ingestion job started` | info | `engine::ingest` | `on_error, file_bytes` + job span |
| `cannot open bulk file` | warn | `engine::ingest` | `path, error` |
| `ingestion started` | info | `engine::collection` | `collection, mode, on_error, existing_vectors` |
| `record rejected` | debug | `engine::collection` | `collection, line, reason` |
| `bulk build: loading vectors` (every 50 000 records) | info | `engine::collection` | `collection, records_read, vectors` |
| `bulk build: vectors loaded, linking graph` | info | `engine::collection` | `collection, vectors, rejected` |
| `bulk build: entry points selected` | info | `engine::collection` | `collection, entry_point, entry_points, nodes` |
| `entry points refreshed` (end of incremental ingestion) | info | `engine::collection` | `collection, entry_points` |
| `bulk build: linking` (every max(10 %, 50 000) nodes) | info | `engine::collection` | `collection, linked, total, percent, elapsed_secs` |
| `bulk build: graph linked` | info | `engine::collection` | `collection, nodes, elapsed_secs` |
| `incremental ingestion progress` (every 10 000 records) | info | `engine::collection` | `collection, applied, rejected, current_line` |
| `ingestion checkpoint persisted` | debug | `engine::collection` | `collection, vectors` |
| `ingestion finished` | info | `engine::collection` | `collection, mode, success, records_read, inserted, updated, rejected, elapsed_secs, records_per_sec` |
| `ingestion job completed` / `ingestion job failed` | info / warn | `engine::ingest` | `inserted, updated, rejected` (+ `error`) |
| `bulk build write failed`, `bulk graph build failed`, `persisting bulk build failed`, `incremental insert failed` | error | `engine::collection` | `collection, error` (+ `line`) |
| `ingestion worker terminated abnormally` | error | `server::api::ingest` | `job, error` |

### 12.7 Example: what a bulk ingestion looks like

Illustrative text output with `RUST_LOG=info`:

```text
INFO request{request_id=req-1 method=PUT path=/collections/bulk_db}: engine::db: created collection collection=bulk_db dim=3 metric="L2" r=16 alpha=1.2 l_build=64 initial_capacity=1000
INFO request{request_id=req-1 method=PUT path=/collections/bulk_db}: server::logging: request completed status=202 latency_ms=3.1
INFO ingest_job{job=job-1767348000000-1 collection=bulk_db}: engine::ingest: ingestion job started on_error=FailFast file_bytes=Some(52311)
INFO ingest_job{job=job-1767348000000-1 collection=bulk_db}: engine::collection: ingestion started collection=bulk_db mode=InitialBuild on_error=FailFast existing_vectors=0
INFO ingest_job{…}: engine::collection: bulk build: vectors loaded, linking graph collection=bulk_db vectors=1000 rejected=0
INFO ingest_job{…}: engine::collection: grew collection storage collection=bulk_db from=1000 to=2000        (only if needed)
INFO ingest_job{…}: engine::collection: bulk build: entry points selected collection=bulk_db entry_point=Some(417) entry_points=32 nodes=1000
INFO ingest_job{…}: engine::collection: bulk build: graph linked collection=bulk_db nodes=1000 elapsed_secs=0.21
INFO ingest_job{…}: engine::collection: ingestion finished collection=bulk_db mode=InitialBuild success=true records_read=1000 inserted=1000 updated=0 rejected=0 elapsed_secs=0.23 records_per_sec=4347.8
INFO ingest_job{…}: engine::ingest: ingestion job completed inserted=1000 updated=0 rejected=0
```

### 12.8 Debugging recipes

| Question | What to do |
|---|---|
| Why did my request fail? | Read `error.code` in the response and its `x-request-id`, then search the logs for that id. With `RUST_LOG=info,server=debug` the `client error detail` event shows the full message. |
| Which queries are slow? | Watch for `slow query` warnings; lower `RISKANN_SLOW_QUERY_MS` to catch more. `visited` and `distance_computations` show how much of the graph was explored. Reduce `ef_search` or compare against `/search/exact`. |
| Is my ingestion still running? | `GET …/ingest/{job_id}` gives live counters. Progress lines appear every 50 000 records (bulk) or 10 000 (incremental). |
| Why is a collection missing after restart? | Look for `failed to load collection` at startup and check `collections_failed` in `GET /ready`. The data is left untouched; see §29. |
| Did a crash leave inconsistencies? | On open, `removed dangling/invalid edges on open` and `data files had different capacities` show what was repaired. |
| Is something panicking? | Search for `panic` events; they include the source location. |

### 12.9 What is never logged

Request and response bodies (vectors and ids) are never logged, so stored data does not leak into log files. Two exceptions are worth knowing:

- Rejected JSONL lines are logged only as a reason (e.g. `vector has 2 components …`), at `debug`.
- Internal errors can include **server-side file paths**. These appear in logs only, never in responses.

Treat log files as internal operational data.

### 12.10 Metrics without logs

| Source | What it gives |
|---|---|
| Every search response | `stats` with `visited_nodes`, `expanded_nodes`, `distance_computations`, `returned` and `search_duration_us` |
| `GET /collections/{name}` | `count`, `capacity`, `disk_bytes` and `index_state` |
| `GET /ready` | Loaded and failed collections |

A Prometheus-style metrics endpoint is future work.

## 13. API conventions

- **Base URL:** `http://HOST:PORT` (default `http://localhost:3000`).
- **Request bodies:** JSON, with header `Content-Type: application/json`. Without it the server returns `415`.
- **Responses:** JSON, plus an `x-request-id` header.
- **Errors:** always `{"error": {"code": "…", "message": "…"}}` (§19).
- **Collection names:** `[A-Za-z0-9_-]{1,64}`. Put them directly in the path.
- **External ids** are arbitrary strings of 1–512 bytes. In paths (`/vectors/{id}`), percent-encode special characters.
- **Numbers:** JSON numbers are converted to 32-bit floats. Values beyond the `f32` range become infinite and are rejected.
- **Distances:** always "smaller is closer" (§20).

### Endpoint summary

| Method | Path | Purpose | Success |
|---|---|---|---|
| `GET` | `/health` | Liveness | 200 |
| `GET` | `/ready` | Readiness | 200 / 503 |
| `GET` | `/collections` | List collections | 200 |
| `PUT` | `/collections/{name}` | Create collection (optionally bulk-load) | 201 / 202 |
| `GET` | `/collections/{name}` | Inspect collection | 200 |
| `DELETE` | `/collections/{name}` | Delete collection and its data | 200 |
| `POST` | `/collections/{name}/upsert` | Insert or update a vector | 200 |
| `GET` | `/collections/{name}/vectors/{id}` | Read one vector | 200 |
| `POST` | `/collections/{name}/search` | Approximate k-NN search | 200 |
| `POST` | `/collections/{name}/search/exact` | Exact k-NN search (diagnostic) | 200 |
| `POST` | `/collections/{name}/ingest` | Start a JSONL ingestion job | 202 |
| `GET` | `/collections/{name}/ingest/{job_id}` | Job status | 200 |
| `GET` | `/collections/{name}/graph?limit=N` | Nodes, vectors and edges for visualisation | 200 |
| `GET` | `/` or `/ui` | RiskANN Console (web UI, §10.5) | 200 |

## 14. API reference: collections

### 14.1 `PUT /collections/{name}`: create

**Request body**

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `dim` | integer | yes | — | Vector dimension (1–16384) |
| `metric` | string | yes | — | `"L2"` or `"COSINE"` (`"l2"`, `"cosine"` also accepted) |
| `r` | integer | yes | — | Maximum out-degree per node (1–256) |
| `alpha` | number | yes | — | RobustPrune parameter (1.0–10.0) |
| `initial_capacity` | integer | no | 1000 | Initial slots; storage grows automatically |
| `l_build` | integer | no | `max(2r, 64)` | Construction beam width |
| `entry_points` | integer | no | 64 | Maximum number of search entry points (1–1024); the number used is `min(entry_points, ⌈√n⌉)` (§5.6) |
| `bulk_file` | string | no | — | Absolute server-side JSONL path; starts an ingestion job |
| `on_error` | string | no | `"fail_fast"` | `"fail_fast"` or `"skip"`, applies to `bulk_file` |

**Example**

```bash
curl -i -X PUT http://localhost:3000/collections/demo \
  -H 'Content-Type: application/json' \
  -d '{"dim": 3, "metric": "L2", "r": 16, "alpha": 1.2, "initial_capacity": 1000}'
```

**Response `201 Created`**

```json
{"collection": {"name": "demo", "dim": 3, "metric": "L2", "r": 16, "alpha": 1.2, "l_build": 64,
                "count": 0, "capacity": 1000, "entry_point": null, "entry_point_count": 0, "format_version": 1,
                "index_state": "ready", "disk_bytes": 76000}}
```

**Response `202 Accepted`** (with `bulk_file`) contains the same `collection` object plus:

```json
{"job_id": "job-1767348000000-1", "status_url": "/collections/demo/ingest/job-1767348000000-1"}
```

The ingestion path is validated **before** the collection is created, so a rejected path never leaves an empty collection behind.

**Collection object fields** (also returned by `GET`)

| Field | Meaning |
|---|---|
| `name` | Collection name |
| `dim`, `metric`, `r`, `alpha` | Configuration |
| `l_build` | Effective construction beam (legacy collections: `2r`) |
| `count` | Number of stored vectors |
| `capacity` | Allocated slots (grows automatically) |
| `entry_point` | Primary entry point (the medoid); `null` if empty |
| `entry_point_count` | How many entry points every search starts from (§5.6) |
| `format_version` | On-disk format written by this build (1) |
| `index_state` | `"ready"`, or `"ingesting"` while an ingestion job runs |
| `disk_bytes` | Size of `vectors.bin` + `graph.bin` |

**Errors:** `400 MALFORMED_REQUEST` (missing field, unknown metric) · `400 INVALID_ARGUMENT` (out-of-range parameter, relative `bulk_file`) · `400 INVALID_COLLECTION_NAME` · `400 INGEST_FILE_UNAVAILABLE` · `403 INGEST_PATH_NOT_ALLOWED` · `409 COLLECTION_EXISTS`.

### 14.2 `GET /collections/{name}`: inspect

`200` → `{"collection": { …collection object… }}`. Errors: `404 COLLECTION_NOT_FOUND`, `400 INVALID_COLLECTION_NAME`.

### 14.3 `GET /collections`: list

`200` → `{"collections": [ {…}, {…} ]}`, sorted by name. Collections that failed to load are not listed (see `GET /ready`).

### 14.4 `DELETE /collections/{name}`: delete

```bash
curl -i -X DELETE http://localhost:3000/collections/demo
```

`200` → `{"deleted": "demo"}`.

Deletion happens in three steps:
1. The collection is unregistered, so new requests get `404`.
2. In-flight writes complete, and a running ingestion job stops at its next record and reports `failed`.
3. The collection directory is removed.

Errors: `404 COLLECTION_NOT_FOUND`, `500 INTERNAL_ERROR` if the directory cannot be removed.

## 15. API reference: vectors

### 15.1 `POST /collections/{name}/upsert`: insert or update

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | External id, 1–512 bytes |
| `vector` | array of numbers | yes | Exactly `dim` finite values; non-zero for COSINE |

```bash
curl -i -X POST http://localhost:3000/collections/demo/upsert \
  -H 'Content-Type: application/json' \
  -d '{"id": "item-001", "vector": [0.12, 0.34, 0.56]}'
```

`200` → `{"id": "item-001", "internal_id": 0, "created": true}`

| Case | Behaviour |
|---|---|
| New id | Next dense internal id; `created: true` |
| Existing id | Vector replaced in place, same `internal_id`, node re-linked; `created: false` |

The call returns after the vector is linked into the graph and the data and metadata are persisted.

Each upsert syncs to disk and rewrites the id list, so many single upserts are slow. Use ingestion jobs for bulk data (§17).

**Errors:** `400 DIMENSION_MISMATCH` · `400 INVALID_ARGUMENT` (NaN/∞, zero vector with COSINE, empty or too long id) · `400 MALFORMED_REQUEST` · `404 COLLECTION_NOT_FOUND` · `409 CAPACITY_EXCEEDED` · `409 COLLECTION_DELETED`.

### 15.2 `GET /collections/{name}/vectors/{id}`: read one vector

`200` → `{"id": "item-001", "internal_id": 0, "vector": [0.12, 0.34, 0.56]}`. Errors: `404 VECTOR_NOT_FOUND`, `404 COLLECTION_NOT_FOUND`.

## 16. API reference: search

### 16.1 `POST /collections/{name}/search`: approximate

| Field | Type | Default | Meaning |
|---|---|---|---|
| `vector` | array | required | Query; length `dim`; finite values; non-zero for COSINE |
| `k` | integer | 10 | Results to return (1–1000) |
| `ef_search` | integer | `max(k, 64)` | Exploration budget, `k ≤ ef_search ≤ 4096` |
| `include_distances` | boolean | `true` | Include `distance` in each result |
| `include_vectors` | boolean | `false` | Include each stored vector; adds `dim` numbers per result to the payload |
| `trace` | boolean | `false` | Also return `"trace": {"entry_point": …, "entry_points": [ … ], "expanded": [ … ]}`: the entry points and the internal ids of the nodes the beam search expanded, in order. The console uses it to animate the search. Tracing does not change the results. |

Any other field, including `filter`, is rejected with `400 MALFORMED_REQUEST`. Metadata filtering is not supported, and is never silently ignored.

```bash
curl -s -X POST http://localhost:3000/collections/demo/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.10, 0.30, 0.50], "k": 5, "ef_search": 64, "include_distances": true, "include_vectors": false}'
```

**Response `200`** (schema; values illustrative)

```json
{
  "collection": "demo",
  "metric": "L2",
  "mode": "approximate",
  "k": 5,
  "ef_search": 64,
  "results": [
    {"id": "item-004", "internal_id": 3, "distance": 0.0},
    {"id": "item-001", "internal_id": 0, "distance": 0.0748}
  ],
  "stats": {"visited_nodes": 5, "expanded_nodes": 5, "distance_computations": 5, "returned": 2, "search_duration_us": 41}
}
```

**Result ordering.** Results are unique and in ascending distance, with ties ordered by lower `internal_id`. The same query on an unchanged index returns the same order. There are fewer than `k` results if the collection has fewer vectors; an empty collection returns `[]` with `200`.

| `stats` field | Meaning |
|---|---|
| `visited_nodes` | Distinct nodes reached (distance computed), including the entry point |
| `expanded_nodes` | Nodes whose neighbour list was read |
| `distance_computations` | Query-to-node distance evaluations; equals `visited_nodes` |
| `returned` | Number of results |
| `search_duration_us` | Engine time in µs: validation, read lock, traversal and result assembly. Excludes HTTP and JSON. |

**Errors:** `400 DIMENSION_MISMATCH` · `400 INVALID_ARGUMENT` (`k`, `ef_search`, non-finite, zero cosine vector) · `400 MALFORMED_REQUEST` · `404 COLLECTION_NOT_FOUND` · `415 UNSUPPORTED_MEDIA_TYPE`.

### 16.2 `POST /collections/{name}/search/exact`: exact (diagnostic)

The request has the same fields **without** `ef_search`. The response has the same shape, with `"mode": "exact"`, no `ef_search`, and `distance_computations` equal to the collection size.

Exact search scans every vector, so its cost is O(n · dim). Use it to check results and to measure recall, not as the normal query path.

```bash
curl -s -X POST http://localhost:3000/collections/demo/search/exact \
  -H 'Content-Type: application/json' -d '{"vector": [0.10, 0.30, 0.50], "k": 5}'
```

### 16.3 `GET /collections/{name}/graph?limit=N`: graph snapshot

Returns the first `N` nodes (default 1 000, at most 5 000), each with its internal id, external id and vector, plus every edge between them. This is the data behind the console's graph view; it is a diagnostic endpoint, not a bulk export.

```json
{"collection": "demo", "metric": "L2", "dim": 3, "r": 16, "count": 5, "entry_point": 3, "entry_points": [3, 0, 4],
 "nodes": [{"internal_id": 0, "id": "item-001", "vector": [0.12, 0.34, 0.56]}, …],
 "edges": [[0, 3], [3, 0], …], "truncated": false}
```

| Field | Meaning |
|---|---|
| `edges` | `[from, to]` pairs of internal ids. Edges to nodes outside the snapshot are omitted. |
| `truncated` | `true` when the collection has more nodes than `limit` |

Errors: `400 INVALID_ARGUMENT` (`limit` outside 1–5000), `400 MALFORMED_REQUEST` (non-numeric `limit`), `404 COLLECTION_NOT_FOUND`.

## 17. API reference: ingestion jobs

### 17.1 JSONL file format

```jsonl
{"id":"item-001","vector":[0.12,0.34,0.56]}
{"id":"item-002","vector":[0.78,0.90,0.12]}
```

- One JSON object per line with `id` (string) and `vector` (numbers). Extra fields are ignored.
- Blank lines are skipped. Windows line endings (`\r\n`) are accepted.
- Lines may be at most 16 MiB; a longer line stops the job.
- A duplicate `id` within a file or already in the collection is an **update**.
- The file must be readable by the server process, and the path is a path **on the server**, not on the client.

### 17.2 `POST /collections/{name}/ingest`: start a job

| Field | Type | Default | Meaning |
|---|---|---|---|
| `path` | string | required | Absolute server-side path |
| `on_error` | string | `"fail_fast"` | `"fail_fast"` or `"skip"` |
| `checkpoint_every` | integer | 10000 | Incremental mode: persist every N records (≥ 1) |

```bash
curl -i -X POST http://localhost:3000/collections/demo/ingest \
  -H 'Content-Type: application/json' -d '{"path": "/data/import/vectors.jsonl", "on_error": "skip"}'
```

`202` → `{"job_id": "job-1767348000000-1", "collection": "demo", "state": "queued", "status_url": "/collections/demo/ingest/job-1767348000000-1"}`

**Errors:** `400 INVALID_ARGUMENT` (relative path, `checkpoint_every: 0`) · `400 INGEST_FILE_UNAVAILABLE` (missing, unreadable, not a regular file) · `403 INGEST_PATH_NOT_ALLOWED` · `404 COLLECTION_NOT_FOUND`.

### 17.3 `GET /collections/{name}/ingest/{job_id}`: job status

```json
{"job_id": "job-1767348000000-1", "collection": "demo", "state": "completed", "on_error": "skip",
 "mode": "initial_build", "records_read": 1000, "records_inserted": 998, "records_updated": 0,
 "records_rejected": 2, "current_line": 1000, "nodes_linked": 998,
 "created_at_ms": 1767348000000, "started_at_ms": 1767348000002, "finished_at_ms": 1767348000950,
 "error": null,
 "rejected_samples": [{"line": 17, "reason": "vector has 2 components but the collection dimension is 3"}]}
```

| Field | Meaning |
|---|---|
| `state` | `queued` → `running` → `completed` or `failed` |
| `mode` | `initial_build` (collection was empty) or `incremental`; `null` until the job finishes |
| `records_read` | Non-blank lines processed |
| `records_inserted` / `records_updated` / `records_rejected` | Outcome counters (live while running) |
| `current_line` | Last line number processed |
| `nodes_linked` | Initial build only: nodes linked into the graph so far |
| `*_at_ms` | Unix timestamps in milliseconds |
| `error` | Why the job failed (client-safe, e.g. `line 3: vector has …`) |
| `rejected_samples` | First 20 rejected lines with line numbers |

`404 JOB_NOT_FOUND` is returned for an unknown id, or an id belonging to another collection. Job status is held in memory: it is lost on restart, and only the most recent 1 000 jobs are kept.

### 17.4 Error policy

| Policy | First invalid line | Earlier records | Final state |
|---|---|---|---|
| `fail_fast` | Stops the job | Kept and persisted (no rollback) | `failed`, `error` names the line |
| `skip` | Counted in `records_rejected`; ingestion continues | Kept | `completed` (unless an I/O error occurs) |

Unreadable files, read errors and over-long lines always fail the job.

### 17.5 Modes and visibility to searches

| Mode | When | Behaviour | What searches see meanwhile |
|---|---|---|---|
| `initial_build` | Collection is empty | Load all vectors → choose entry point → link every node → persist once | Searches wait for the write lock, then see the complete index |
| `incremental` | Collection has data | Each record fully inserted under the write lock; checkpoint every `checkpoint_every` records and at the end | A consistent prefix of the file; searches interleave with ingestion |

`GET /collections/{name}` reports `"index_state": "ingesting"` while a job runs.

## 18. API reference: health and readiness

| Endpoint | Response | Notes |
|---|---|---|
| `GET /health` | `200 {"status":"ok"}` | Cheap liveness check; touches no data; logged at `debug` |
| `GET /ready` | `200 {"ready":true,"collections_loaded":2,"collections_failed":["broken"]}` | `503` with `"ready":false` once graceful shutdown has started |

The server binds its port only after all collections are loaded.

## 19. Error reference

Every error response:

```json
{"error": {"code": "DIMENSION_MISMATCH", "message": "vector has 2 components but the collection dimension is 3"}}
```

| Status | Code | Typical cause | What to do |
|---|---|---|---|
| 400 | `MALFORMED_REQUEST` | Invalid JSON, missing/unknown field, wrong type, unknown metric, negative number | Fix the request body; the message names the field |
| 400 | `INVALID_ARGUMENT` | Out-of-range `k`/`ef_search`/`dim`/`r`/`alpha`/`l_build`/capacity, NaN/∞, zero cosine vector, empty id, relative ingest path | Respect the limits in §11.3 |
| 400 | `DIMENSION_MISMATCH` | Vector length ≠ collection `dim` | Check `GET /collections/{name}` for `dim` |
| 400 | `INVALID_COLLECTION_NAME` | Name with `.`, `/`, spaces, non-ASCII, or > 64 chars | Use `[A-Za-z0-9_-]` |
| 400 | `INGEST_FILE_UNAVAILABLE` | File missing, unreadable, or a directory | Check the path *on the server* and its permissions |
| 403 | `INGEST_PATH_NOT_ALLOWED` | Path outside `RISKANN_INGEST_DIR` | Move the file into that directory |
| 404 | `COLLECTION_NOT_FOUND` | No such collection (or it failed to load) | `GET /collections`, `GET /ready` |
| 404 | `VECTOR_NOT_FOUND` | No such external id | — |
| 404 | `JOB_NOT_FOUND` | Unknown job id, other collection, or server restarted | — |
| 404 | `ROUTE_NOT_FOUND` | Unknown path | See §13 |
| 409 | `COLLECTION_EXISTS` | Name loaded, or a directory with that name exists in `DATA_LOC` | Choose another name, or inspect the leftover directory |
| 409 | `COLLECTION_DELETED` | Collection deleted while the request ran | — |
| 409 | `CAPACITY_EXCEEDED` | More than 2³² − 2 vectors | — |
| 413 | `PAYLOAD_TOO_LARGE` | Body > `RISKANN_MAX_BODY_BYTES` | Raise the limit or use ingestion |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Missing `Content-Type: application/json` | Add the header |
| 500 | `INTERNAL_ERROR` | Storage/I/O failure, corruption, panic | Look up the `x-request-id` in the server log |

A wrong HTTP method on an existing path returns Axum's default `405` with an empty body.

## 20. Distance semantics

| Metric | Formula returned | Range | Notes |
|---|---|---|---|
| `L2` | `sqrt(Σ (aᵢ − bᵢ)²)` | `[0, ∞)` | Euclidean, **not** squared |
| `COSINE` | `1 − (a·b) / (‖a‖ ‖b‖)` | `[0, 2]` | Cosine **distance**: 0 = same direction, 1 = orthogonal, 2 = opposite |

- **Smaller is always closer.** The same functions are used for graph construction, approximate search and exact search, so all three agree on ranking.
- **Precision.** Sums are accumulated in `f64` and the result is stored as `f32`.
- **Rejected inputs.** NaN and ±∞ components are rejected at insert and query time. For COSINE, zero vectors are rejected because the cosine is undefined.
- **Overflow.** A distance that overflows (beyond `f32::MAX`) is an error, never a ranked result.
- **Tie-break.** Equal distances are ordered by the lower internal id.

## 21. Tuning: `k`, `ef_search`, `r`, `alpha`, `l_build`

| Parameter | Set at | Effect of increasing it | Cost |
|---|---|---|---|
| `k` | query | More results | Slightly more work; `ef_search ≥ k` is required |
| `ef_search` | query | Explores more nodes, so recall usually rises | Latency and distance computations rise |
| `r` | creation | Denser graph, so recall rises at equal `ef_search` | Larger `graph.bin`; slower build and query |
| `alpha` | creation | Keeps more long-range edges (better navigation) | More edges kept per node, up to `r` |
| `l_build` | creation | Better neighbour candidates during build, so a better graph | Slower build |
| `entry_points` | creation | More starting points spread across the data, so clustered data stays reachable | About one extra distance computation per entry point per query |

**Practical procedure**
1. Start with `r = 16–32`, `alpha = 1.2` and the default `l_build`.
2. Build the index and run `riskann-bench` (or the exact endpoint) on representative queries.
3. Pick the smallest `ef_search` that reaches the recall you need. Recall is never guaranteed to be 1.0 for approximate search.
4. If recall plateaus too low, increase `r` or `l_build` and rebuild.

The Phase-II demo used `r = 3`. Such a small degree, especially with the old construction beam of `2r = 6`, gives low recall. New collections default to `l_build = max(2r, 64)`; still prefer `r ≥ 8`, and measure.

## 22. On-disk storage format

```text
$DATA_LOC/
└── demo/
    ├── config.json      collection configuration
    ├── metadata.json    ids, entry point, counts (format version 1)
    ├── vectors.bin      capacity × dim × f32
    └── graph.bin        capacity × r × u32
```

| File | Content | Example |
|---|---|---|
| `config.json` | `dim`, `metric`, `r`, `alpha`, `l_build` | `{"dim":3,"metric":"L2","r":16,"alpha":1.2,"l_build":64,"entry_points":64}` |
| `metadata.json` | `format_version`, `medoid_id` (primary entry point), `entry_points` (all entry points, medoid first), `next_id` (vector count), `capacity`, `external_ids` (index = internal id) | `{"format_version":1,"medoid_id":0,"entry_points":[0,1],"next_id":2,"capacity":1000,"external_ids":["a","b"]}` |
| `vectors.bin` | Slot `i` = bytes `[i·dim·4, (i+1)·dim·4)`, `dim` native-endian `f32` | — |
| `graph.bin` | Slot `i` = `r` native-endian `u32` neighbour ids; the list ends at the first `u32::MAX`; the rest is padded with `u32::MAX` | — |

**Invariants**
- **Live ids.** Internal ids `0 … next_id−1` are live; slots up to `capacity` are free. Capacity is always `file length / record size`, never a constant.
- **Growth.** When full, both files double (vectors first, then graph), and new adjacency slots are filled with `u32::MAX`.
- **Write order.** `config.json` is written last when a collection is created; a directory without it is not treated as a collection.
- **Byte order.** Native, i.e. little-endian on x86-64 and ARM64. Files are not portable to big-endian machines.

**Legacy format 0** (Phase-II collections: `id_map` in metadata, no `format_version`) is read transparently.
- Its construction beam stays `2r`.
- The first write rewrites `metadata.json` in format 1, which older binaries cannot read. Back up the directory before upgrading if you need to roll back.

**Never edit these files while the server runs.** The memory maps assume exclusive ownership.

## 23. Persistence, crash behaviour and concurrency

### 23.1 Write order

Every upsert, ingestion checkpoint and bulk-build completion follows the same order:

1. Write the vector and adjacency data into the mappings.
2. `msync` both files.
3. Write `metadata.json.tmp`, `fsync` it, rename it over `metadata.json`, and `fsync` the directory (Unix, best effort).

Metadata therefore never claims an id whose data was not synced first.

### 23.2 Checks on open

| Condition | Result |
|---|---|
| File length not a multiple of its record size, or zero | Error: collection not served |
| Files hold fewer slots than `next_id` | Error |
| `external_ids` count ≠ `next_id`, duplicate ids, entry point not live | Error |
| Missing or unparsable `metadata.json` / `config.json` | Error |
| `format_version` newer than supported | Error |
| The two files hold different slot counts (interrupted resize) | Shorter file **grown**; warning logged |
| Edges pointing to non-live ids, self-edges, duplicates | Removed; warning logged |

A collection that fails these checks is reported in the logs and in `/ready`. It is **left untouched**: never re-initialised, never overwritten. `PUT` with the same name returns `409` while the directory exists.

### 23.3 What a crash can lose (honest list)

| Situation | After restart |
|---|---|
| Crash during an initial bulk build | Collection is **empty**; re-run the ingestion |
| Crash during incremental ingestion | Records after the last checkpoint are missing |
| Crash during an in-place update that was not yet persisted | The vector may be the old or the new value |
| Crash during resize | Repaired on open (shorter file grown) |
| Server restart | Ingestion job status is gone (collections are not) |

Full crash consistency (a write-ahead log) is **not** implemented or claimed.

### 23.4 Concurrency

| Operation | Lock taken | Runs concurrently with |
|---|---|---|
| Search, exact search, `GET` vector or collection | Read | Other reads |
| Upsert, one incremental-ingestion record | Write | Nothing in the same collection |
| Initial bulk build | Write, for the whole build | Nothing in the same collection; searches wait |
| Delete | Waits for the write lock once | — |

Operations on different collections are independent. Readers can never observe a partially written vector or adjacency list. Each collection has a single lock, so lock ordering cannot deadlock.

### 23.5 Shutdown

Ctrl-C or SIGTERM triggers this sequence:
1. `/ready` starts returning `503`.
2. The server stops accepting connections and finishes in-flight requests.
3. The runtime waits for running ingestion jobs before the process exits.

`kill -9` skips all of this; see §23.3 for what that can lose.

## 24. Testing guide

### 24.1 Running tests

```bash
cargo test --workspace                          # everything
cargo test -p riskann-core                      # one crate
cargo test -p engine --test engine              # engine integration tests only
cargo test -p server --test api                 # HTTP tests only
cargo test -p engine ann_recall                 # tests whose name contains "ann_recall"
cargo test --workspace -- --nocapture           # show println output
cargo test --release -p engine --test engine    # faster for the recall tests
```

- **Isolation.** Every test creates its own temporary directory under the system temp dir and removes it afterwards. Nothing depends on `/tmp/db_data`, a running server or network access.
- **Logs.** Tests do not install a log subscriber, so log events are not printed during tests.

### 24.2 Test inventory (109 tests)

| File | Tests | Covers |
|---|---:|---|
| `crates/core/src/distances.rs` | 11 | L2/cosine known values, Euclidean-not-squared, symmetry, identical/orthogonal/opposite vectors, zero-vector cosine, dimension mismatch, NaN/∞, overflow boundary, metric names |
| `crates/core/src/vamana.rs` | 26 | Candidate ordering and ties, NaN-safe ordering, empty/single/linear/cyclic/disconnected graphs, duplicate neighbours, `k = 1`, `k` > nodes, `ef = k`, invalid parameters, deterministic tie-break, out-of-range edges, brute-force equivalence, expansion order, callback errors, multiple entry points and their validation, RobustPrune (occlusion, diversity, self/duplicate removal, α effect, invalid parameters) |
| `crates/storage/src/memory.rs` | 15 | Create/reopen, no truncation on reopen, first/last/out-of-bounds slots, wrong dimension, non-finite values, empty/max-degree/invalid edges, refusal to overwrite, invalid shapes, malformed/empty files, metadata-vs-file mismatch, missing files, resize, interrupted-resize repair |
| `crates/engine/src/collection.rs` | 5 | Collection names, config validation, legacy `l_build`, query defaults and limits, legacy metadata inversion |
| `crates/engine/src/ingest.rs` | 5 | JSONL parsing and line numbers, missing fields, job state transitions, failed jobs, path policy |
| `crates/engine/src/dataset.rs` | 4 | SplitMix64 reference values, determinism, Gaussian moments, distribution parsing |
| `crates/engine/tests/engine.rs` | 28 | Empty/one-vector collections, hand-computed order, ties, `include_vectors`, id stability, validation, **Recall@10 ≥ 0.95** (L2 and cosine bulk, incremental), sorted/unique/deterministic results, identical builds, restart, growth beyond capacity, no truncation, dangling-edge repair, corrupt metadata, legacy format, ingestion (success, `fail_fast`, `skip`, missing file), concurrent search + upsert, registry, search trace and graph snapshot, **clustered-L2 regression** (4 000 × 32-d, 32 clusters: Recall@10 ≥ 0.95 at `ef_search` 32 with entry points, and better than with one) |
| `crates/server/src/logging.rs` | 1 | Request-id validation |
| `crates/server/tests/api.rs` | 14 | Health/ready; create → upsert → search → exact → inspect → list → get vector → delete; every search validation error; payload options; empty collection; creation errors; upsert errors; ingestion lifecycle; `bulk_file`; failed jobs; path policy; 413/404; restart; `x-request-id`; console page, graph snapshot and search trace |

The recall thresholds in the tests are regression guards on small datasets, not benchmark claims. The clustered-L2 test reproduces, at a smaller scale, the failure found in the first benchmark run.

### 24.3 Quality gates before committing

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## 25. Benchmarks (`riskann-bench`)

`riskann-bench` lives in `crates/engine/src/bin/riskann_bench.rs` and calls the engine directly (no HTTP).

### 25.1 Commands

```bash
# Always use --release for numbers you report.
cargo run --release -p engine --bin riskann-bench -- help

# One configuration with an ef_search sweep
cargo run --release -p engine --bin riskann-bench -- run \
  --n 10000 --dim 64 --metric l2 --distribution clustered --seed 42 \
  --queries 200 --k 10 --ef 16,32,64,128,256 --r 32 --alpha 1.2

# Predefined matrices
cargo run --release -p engine --bin riskann-bench -- suite --profile quick      # minutes
cargo run --release -p engine --bin riskann-bench -- suite --profile standard   # longer
cargo run --release -p engine --bin riskann-bench -- suite --profile full       # longest

# Deterministic JSONL file for HTTP demos
cargo run --release -p engine --bin riskann-bench -- gen-jsonl --n 1000 --dim 3 --out /tmp/data.jsonl
```

### 25.2 `run` options

| Option | Default | Meaning |
|---|---|---|
| `--n` | 10000 | Base vectors |
| `--dim` | 64 | Dimension |
| `--metric` | `l2` | `l2` or `cosine` |
| `--distribution` | `clustered` | `uniform` ([-1,1)), `gaussian` (N(0,1)), `clustered` (32 Gaussian clusters), `clustered:N` |
| `--seed` | 42 | Dataset seed |
| `--queries` | 200 | Held-out queries (drawn independently of the base vectors) |
| `--k` | 10 | Neighbours per query |
| `--ef` | `16,32,64,128,256` | `ef_search` values to sweep (values < k are skipped) |
| `--r` | 32 | Graph degree |
| `--alpha` | 1.2 | Pruning parameter |
| `--l-build` | `max(2r, 64)` | Construction beam |
| `--entry-points` | 64 | Maximum number of entry points (`1` reproduces the single-entry behaviour) |
| `--build-mode` | `bulk` | `bulk` (initial build) or `incremental` (first vector upserted, rest ingested incrementally) |
| `--threads` | 1 | If > 1, also measure concurrent QPS with this many threads |
| `--warmup` | 1 | Unmeasured passes over the queries before each measured pass |
| `--out` | `bench_results` | Output directory |
| `--work-dir` | system temp | Where the temporary collection is built (deleted afterwards) |
| `--label` | derived, e.g. `n10000_d64_l2_clustered-32_r32_e64_bulk` | File name for results |

### 25.3 `suite` profiles

| Profile | Configurations (n × dim, metric) | `ef_search` sweep |
|---|---|---|
| `quick` | 1k × 3 L2 (r=16), 1k × 64 L2, 10k × 64 L2, 10k × 64 cosine (r=32) | 10, 16, 32, 64, 128 |
| `standard` | quick + 10k × 128 L2, 50k × 128 L2, 50k × 128 cosine | 10 … 256 |
| `full` | standard + 100k × 128 L2 | 10 … 256 |

All suite runs use 200 queries, k = 10, the clustered distribution and seed 42.

### 25.4 What one run does

1. **Generate** the base vectors and queries (not timed).
2. **Build** the index (timed; includes the final fsync), then verify the graph invariants and the average out-degree.
3. **Ground truth.** Run exact search for every query; this is also the exact-baseline latency.
4. **Sweep.** For each `ef_search`: warm up, then run the measured pass. Record Recall@K, latency percentiles, QPS, and mean visited nodes, expanded nodes and distance computations.
5. **Storage and reload.** Measure the on-disk size, close the collection, reopen it (timed), and check that 10 probe queries return identical results.
6. **Memory.** Read peak RSS (`VmHWM`, Linux).
7. **Write** the JSON and CSV output and print a Markdown table.

### 25.5 Output files

**`bench_results/<label>.json`**, one per configuration:

| Key | Content |
|---|---|
| `machine` | `cpu_model`, `logical_cpus`, `mem_total_kb`, `os`, `arch`, `build_profile`, `riskann_version` |
| `dataset` | `n`, `dim`, `distribution`, `seed`, `queries`, `queries_held_out` |
| `index` | `metric`, `r`, `alpha`, `l_build`, `build_mode` |
| `k`, `warmup_passes`, `threads` | Run parameters |
| `build_secs`, `avg_out_degree` | Build cost and graph density |
| `exact_baseline` | `latency_us{p50,p95,p99,mean}`, `qps_single_thread` |
| `sweep[]` | Per `ef_search`: `recall_at_k`, `latency_us{…}`, `qps_single_thread`, `qps_concurrent`, `avg_visited_nodes`, `avg_expanded_nodes`, `avg_distance_computations` |
| `disk_bytes` | Size of the collection directory (data + JSON files) |
| `reload_ms`, `reload_results_identical` | Reopen time and consistency check |
| `peak_rss_kb` | Process high-water mark (includes the in-memory dataset) |

**`bench_results/summary.csv`**: one row per (configuration, `ef_search`), appended across runs. Columns:

`label, n, dim, metric, distribution, r, alpha, l_build, build_mode, k, ef_search, queries, threads, build_secs, recall_at_k, p50_us, p95_us, p99_us, mean_us, qps_single, qps_concurrent, avg_visited, avg_expanded, avg_distance_computations, exact_p50_us, exact_qps, disk_bytes, reload_ms, peak_rss_kb, build_profile, cpu_model`

### 25.6 Measurement rules

- **Engine-level latency.** Latency is timed around `Collection::search` / `exact_search`; it excludes HTTP and JSON. HTTP adds overhead not shown here.
- **QPS definitions.** `qps_single` = queries ÷ total measured latency on one thread. `qps_concurrent` = (threads × queries) ÷ wall time.
- **Warm cache.** Results reflect a warm page cache after warm-up. Cold-cache behaviour is not measured.
- **Comparability.** Numbers from different machines, build profiles, dimensions, distributions or cache states are **not** comparable.
- **Synthetic data.** It is not a substitute for real embeddings; repeat with real data before drawing conclusions.

### 25.7 Recorded results

Machine: AMD Ryzen 7 7735HS, 14 GiB RAM, Linux, `rustc 1.99.1`, release build, 2 November 2026.
Command: `riskann-bench suite --profile quick`.
Settings: 200 held-out queries, k = 10, seed 42, clustered data (32 clusters), α = 1.2, `l_build` = 64.

**These runs used the revision with a single entry point per collection.** The 10k × 64-d L2 rows show the failure described below; re-run the suite on the current revision and add its results here.

| Configuration | Build | Disk | Reload (results identical) | Exact search p50 | Exact QPS |
|---|---:|---:|---:|---:|---:|
| 1 000 × 3, L2, r = 16 | 0.04 s | 0.1 MiB | 0.3 ms (yes) | 20.5 µs | 48 555 |
| 1 000 × 64, L2, r = 32 | 0.19 s | 0.4 MiB | 0.4 ms (yes) | 41.1 µs | 23 368 |
| 10 000 × 64, L2, r = 32 | 5.14 s | 3.7 MiB | 3.4 ms (yes) | 334.7 µs | 2 938 |
| 10 000 × 64, cosine, r = 32 | 4.19 s | 3.7 MiB | 3.4 ms (yes) | 860.4 µs | 1 116 |

| Configuration | ef_search | Recall@10 | p50 µs | p99 µs | QPS (1 thread) | Nodes reached |
|---|---:|---:|---:|---:|---:|---:|
| 1 000 × 3, L2 | 10 | 0.9995 | 9.6 | 17.8 | 100 879 | 79 |
| | 32 | 1.0000 | 14.9 | 23.7 | 65 858 | 133 |
| | 128 | 1.0000 | 41.0 | 47.0 | 24 957 | 299 |
| 1 000 × 64, L2 | 10 | 1.0000 | 14.0 | 21.9 | 70 380 | 118 |
| | 32 | 1.0000 | 20.9 | 31.0 | 47 212 | 173 |
| | 128 | 1.0000 | 59.9 | 76.8 | 16 566 | 484 |
| 10 000 × 64, L2 | 10 | **0.0480** | 18.1 | 33.8 | 52 709 | 179 |
| | 32 | **0.1970** | 32.3 | 106.7 | 24 619 | 334 |
| | 128 | **0.5100** | 92.5 | 208.4 | 10 295 | 606 |
| 10 000 × 64, cosine | 10 | 0.9480 | 40.2 | 62.4 | 25 013 | 257 |
| | 32 | 1.0000 | 62.2 | 91.7 | 15 534 | 349 |
| | 128 | 1.0000 | 86.7 | 135.2 | 11 362 | 414 |

The `ef_search` values 16 and 64 were also measured and are in `bench_results/summary.csv`.

**Reading the results**
- **Where the index works, ANN is much faster than exact search.** On the cosine collection at `ef_search` 32, recall is 1.0 at 62 µs median, about 14 × faster than the 860 µs exact scan, while computing distances to 3.5 % of the vectors.
- **The 10 000 × 64-d L2 configuration is a failure, not a trade-off.** Recall stays far below 1 even at `ef_search` 128.

### Finding: recall collapse on clustered L2 data, and the fix

- **Cause.** Only **7 882 of 10 000 nodes (79 %) were reachable** from the single entry point, so whole clusters could not be found at any `ef_search`. Under L2 in 64-d, the 32 clusters are almost equidistant from each other and the points inside one cluster are almost equidistant from each other. RobustPrune therefore fills every node's `r` slots with same-cluster neighbours (§5.6). The data, graph code and storage were correct; construction quality was the problem.
- **Diagnosis.** The JavaScript port of the engine reproduced the measured recall exactly (0.048 / 0.197 / 0.510).
- **Alternatives tried.** A second build pass, Vamana's α = 1 then α = 1.2 passes, and a random initial graph all left recall at or below 0.55.
- **Fix.** Search and construction start from a set of entry points chosen by farthest-point sampling (§5.6).

Measured with the same port on the same configuration (not yet the Rust engine):

| ef_search | 1 entry point | 64 entry points |
|---:|---:|---:|
| 10 | 0.048 | 0.949 |
| 32 | 0.197 | 0.996 |
| 128 | 0.510 | 1.000 |

Reachability rose to 99.9 % while fewer nodes were visited per query (317 instead of 334 at `ef_search` 32). Other datasets were unchanged or slightly better (`CHANGELOG.md`). The Rust test suite now pins this with a regression test (§24.2).

## 26. Recall@K methodology and reporting

For each held-out query:

1. **Exact top-K:** `exact_search` (brute force, same metric, same tie-break).
2. **Approximate top-K:** `search` at a given `ef_search`.
3. **Recall:** `|ANN ∩ exact| / min(K, n)`, comparing internal ids.
4. **Average** over all queries, and repeat for each `ef_search`.

Exact search has recall 1.0 by definition, but costs O(n · dim) per query. ANN trades a **measured** loss of recall for exploring far fewer nodes, and the `avg_distance_computations` column shows how many.

When reporting results, include all of the following:
- the dataset (`n`, `dim`, distribution, seed) and the number of queries;
- the metric and the `k`, `r`, `alpha`, `l_build` values;
- the build mode and every `ef_search` value;
- the machine (CPU, RAM, OS), `rustc -V` and the build profile;
- the exact command.

Report a recall target (e.g. 95 %) only for the configuration that achieved it.

## 27. Evaluation demo walkthrough

A complete script for showing the project. Run it from the repository root with two terminals.

**Terminal 1: server with readable logs**

```bash
cargo build --release --workspace
RUST_LOG=info,engine=debug DATA_LOC=./data ./target/release/server 2>&1 | tee riskann.log
```

**Terminal 2: requests**

```bash
# 1. Health
curl -s localhost:3000/health; echo

# 2. Generate 1 000 deterministic 3-d vectors
./target/release/riskann-bench gen-jsonl --n 1000 --dim 3 --out /tmp/data.jsonl
head -2 /tmp/data.jsonl

# 3. Create a collection that bulk-loads the file (returns 202 + job_id)
curl -s -X PUT localhost:3000/collections/bulk_db -H 'Content-Type: application/json' \
  -d '{"dim":3,"metric":"L2","r":16,"alpha":1.2,"initial_capacity":1000,"bulk_file":"/tmp/data.jsonl"}' | tee /tmp/create.json; echo
JOB=$(sed -E 's/.*"job_id":"([^"]+)".*/\1/' /tmp/create.json)

# 4. Job status → "state":"completed","records_inserted":1000
curl -s localhost:3000/collections/bulk_db/ingest/$JOB; echo

# 5. Approximate search (with an explicit request id to find it in the log)
curl -si -X POST localhost:3000/collections/bulk_db/search -H 'Content-Type: application/json' \
  -H 'x-request-id: demo-1' -d '{"vector":[0.5,0.5,0.5],"k":5,"ef_search":64}'; echo

# 6. Exact search → compare the ids with step 5
curl -s -X POST localhost:3000/collections/bulk_db/search/exact -H 'Content-Type: application/json' \
  -d '{"vector":[0.5,0.5,0.5],"k":5}'; echo

# 7. Validation and the error envelope
curl -s -X POST localhost:3000/collections/bulk_db/search -H 'Content-Type: application/json' \
  -d '{"vector":[0.5,0.5],"k":5}'; echo          # DIMENSION_MISMATCH

# 8. Upsert a new vector and find it
curl -s -X POST localhost:3000/collections/bulk_db/upsert -H 'Content-Type: application/json' \
  -d '{"id":"new-point","vector":[100,100,100]}'; echo
curl -s -X POST localhost:3000/collections/bulk_db/search -H 'Content-Type: application/json' \
  -d '{"vector":[100,100,100],"k":1}'; echo

# 9. Show the log trail of request demo-1
grep demo-1 riskann.log
```

**Persistence.** Stop the server with Ctrl-C and watch the shutdown logs. Start it again, check that `opened collection … vectors=1001` appears in the log, and repeat steps 5–6: the results are identical.

**Measurement.** Run `cargo run --release -p engine --bin riskann-bench -- suite --profile quick` and show the printed Recall@K versus `ef_search` tables.

### Console walkthrough (for a live demonstration)

1. Start the server and open `http://localhost:3000/ui`. The header shows **Live: RiskANN server**.
2. **Explore the graph → Create and load.** This creates `demo_2d` with 400 clustered points. Click near a cluster: the amber path is the beam search, the teal rings the results, the dashed violet rings the exact answer. Lower `ef_search` to 3–5 and click again to see recall drop.
3. **How the graph is built → Play.** Narrate one insertion: search, pool, prune decisions, reverse edges.
4. **Measure recall → Run measurement** on `demo_2d` (or a larger collection). The chart is the recall/effort trade-off.
5. **Ingest a file.** Use a file made with `riskann-bench gen-jsonl`, then add a malformed line to show line-numbered rejection under `skip` and `fail_fast`.
6. **Storage files.** Click the entry-point slot and a free slot to explain the binary format.
7. **Run tests → Run API tests on the live server.** All checks hit the real server.
8. **Requests and logs.** Copy a request id and `grep` it in the server log to show the same request on both sides.

## 28. Limitations and roadmap

### Limitations

| Area | Limitation |
|---|---|
| Deletion | Individual vectors cannot be deleted, only whole collections. Updates leave old in-edges, so many repeated updates can degrade graph quality. |
| Graph build | Single pass with the configured α (Vamana proper uses two passes); the entry point is fixed after the first build. |
| Writes | Serialised per collection. Each HTTP upsert syncs and rewrites the id list (O(n)), so bulk loads should use ingestion jobs. |
| Initial bulk build | Holds the write lock until it finishes; searches on that collection wait. |
| Memory | The id map and id list are in RAM. Memory mapping lets the OS page vectors, but hot pages stay cached; no bounded-RAM or billion-scale claims are made. |
| Durability | Ordered writes, not a write-ahead log; the crash windows are listed in §23.3. |
| Jobs | Status is in memory and lost on restart. |
| Security | No authentication, authorisation or TLS. |
| Platforms | Linux-first; Windows untested; data files use native byte order. |
| Evidence | No benchmark results recorded yet; verification record pending (§3.2). |

### Roadmap

The order is deliberate:

1. Record the verification and benchmark results (§3.2, §25.7).
2. Delete individual vectors with tombstones, skipped by search and repaired in the graph.
3. A durable job log, and an append-only id log to remove the O(n) metadata rewrite.
4. Two-pass bulk build and entry-point recomputation; compare recall against the current build.
5. Profiling-driven optimisation: SIMD distances, prefetching, parallel bulk build.
6. A metrics endpoint, authentication, then filtering.

## 29. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `package requires rustc 1.85` / `edition 2024 is unstable` | Old toolchain | `rustup update stable` |
| `Address already in use` at startup | Port taken | `PORT=3001`, or stop the other process |
| `415 UNSUPPORTED_MEDIA_TYPE` | Missing JSON header | Add `-H 'Content-Type: application/json'` |
| `400 DIMENSION_MISMATCH` | Wrong vector length | Check `dim` with `GET /collections/{name}` |
| `400 INGEST_FILE_UNAVAILABLE` | Path does not exist **on the server**, is relative, or is unreadable | Use an absolute server path; check permissions with `ls -l` |
| `403 INGEST_PATH_NOT_ALLOWED` | `RISKANN_INGEST_DIR` is set and the file is elsewhere | Move the file into that directory |
| Job `failed` with `line N: …` | Malformed record at line N (`fail_fast`) | `sed -n 'Np' file.jsonl`; fix it, or use `"on_error":"skip"` |
| `409 COLLECTION_EXISTS`, but the name is not listed | A leftover or unloadable directory with that name in `DATA_LOC` | Check the startup log for `failed to load collection`; inspect, back up or remove the directory manually |
| A collection disappeared after restart | It failed integrity checks (logged at `error`, listed in `/ready`) | Read the error message; restore `metadata.json` from a backup. The data files are untouched. |
| All data gone after reboot | `DATA_LOC` defaulted to `/tmp/db_data` | Set a persistent `DATA_LOC` |
| Searches block for a long time | An initial bulk build holds the write lock | Wait for the job; ingest into a non-empty collection for interleaving |
| Low recall | Small `r`, small `ef_search`, or a narrow construction beam | Raise `ef_search` first; then `r` or `l_build` (§21); measure with `/search/exact` |
| Upserts are slow | Each upsert syncs to disk | Use ingestion jobs for bulk data |
| No log output | `RUST_LOG` too strict, or output redirected | `RUST_LOG=info`; check where stdout goes |
| Garbled characters in a log file | Colour codes from a terminal capture | Logs written to a file have no colours; use `RISKANN_LOG_FORMAT=json` for tools |
| `cargo fmt -- --check` fails | Formatting differences | Run `cargo fmt --all` |
| Recall tests slow in debug mode | Unoptimised build | `cargo test --release -p engine --test engine` |

## 30. FAQ

**Is this DiskANN?**
No. RiskANN implements the Vamana graph and beam search used by DiskANN, with memory-mapped storage. It has none of DiskANN's compressed in-memory vectors, SSD sector layout or asynchronous beam I/O.

**Are search results exact?**
`/search` is approximate; `/search/exact` is exact. Recall depends on the data and the parameters; measure it with `riskann-bench`.

**Why Euclidean and not squared L2?**
Results are reported in natural units. Squaring would not change the ranking but would change how α behaves during pruning; one convention is used everywhere.

**Does everything have to fit in RAM?**
Vectors and edges are paged by the OS through memory mapping. The id map and id list are in RAM. Performance is best when the working set is cached.

**Can I see new data without restarting?**
Yes. An upsert is searchable when the call returns. Ingested data is visible per record (incremental) or when the build finishes (initial build).

**Is it safe to search while ingesting?**
Yes. Readers never see partial writes (§23.4).

**How many vectors can a collection hold?**
Up to 2³² − 2 by format; in practice it is limited by disk, RAM for the ids, and build time.

**Is there a client library?**
No. Any HTTP client works; the API is plain JSON (§13–§18).

**Why is there a separate exact endpoint?**
It provides ground truth for checking and measuring the approximate index. It is never used silently in place of approximate search.

## 31. Security and trust boundaries

- **No authentication.** Bind to `127.0.0.1` or put the server behind an authenticating reverse proxy. Do not expose it to untrusted networks.
- **Server-side file reads.** `bulk_file` and `/ingest` make the server read files with its own permissions.
  - Set `RISKANN_INGEST_DIR` to confine ingestion to one directory; symlinks and `..` are resolved before the check.
  - Without it, a warning is logged at startup and any readable absolute path is accepted.
- **Path safety.** Collection names cannot contain path separators or dots, so they cannot escape `DATA_LOC`.
- **Error responses** never contain file paths or internal error text. Logs can contain server paths, so treat log files as internal.
- **Resource limits.** The request body is limited by `RISKANN_MAX_BODY_BYTES`, and `k` and `ef_search` are capped, so one query cannot exhaust memory.
- **File ownership.** Collection files must not be modified by other processes while the server runs.

## 32. Contributing and coding conventions

- **Layering.**
  - The server never touches storage or graph internals; everything goes through `engine`.
  - `riskann-core` stays storage-agnostic.
- **Errors.**
  - No `unwrap`/`expect` in non-test code; return typed errors (`StorageError`, `EngineError`, `ApiError`).
  - Client-caused errors map to 4xx with a stable code.
  - Internal details are logged, not returned.
- **Unsafe code.** Every `unsafe` block needs a `SAFETY:` comment stating the invariant. Currently only mmap creation is unsafe.
- **Logging.**
  - Use `tracing` macros with structured fields (`collection = %name`), not formatted strings.
  - Never log vectors or request bodies.
  - Choose the level using §12.4.
- **Determinism.** No randomness in build or search; ties are broken by internal id.
- **Tests.** Every behaviour change needs a test that uses a temporary directory, never fixed paths. Run `cargo fmt`, `clippy -D warnings` and `cargo test --workspace` before pushing.
- **Docs.** Update this README, `docs/DESIGN.md` and `CHANGELOG.md` together with the code. Do not document planned features as implemented.

**Adding an endpoint**
1. Add the engine method, with validation and a test in `crates/engine`.
2. Add the handler in `crates/server/src/api/`, running the engine call through `blocking(...)`.
3. Register the route in `build_router` (`crates/server/src/lib.rs`).
4. Add an HTTP test in `crates/server/tests/api.rs`.
5. Document it in §13–§19.

## 33. Glossary

| Term | Meaning |
|---|---|
| ANN | Approximate nearest neighbour |
| k-NN | The k nearest neighbours of a query |
| Recall@K | Fraction of the true top-K neighbours returned by approximate search |
| Vamana | Graph construction algorithm from DiskANN: greedy search + RobustPrune with α |
| RobustPrune | Neighbour selection that drops candidates "covered" by closer selected neighbours |
| α (alpha) | Pruning slack; larger keeps more long edges |
| r | Maximum out-degree of a node |
| `l_build` | Beam width used while inserting (Vamana's L during construction) |
| `ef_search` | Beam width (result-list size) during a query |
| Medoid / entry point | Node where every search starts |
| Beam search | Best-first graph search keeping the `ef` best candidates |
| Visited / expanded node | Distance computed / neighbour list read |
| Upsert | Insert or update by key |
| Internal id | Dense `u32` slot index; external id = client string |
| mmap | Memory-mapped file: file contents accessed as memory |
| msync / fsync | Flush mapped / file data to stable storage |
| Span | Logging context (e.g. one request) attached to all events inside it |

## 34. References

1. S. J. Subramanya, F. Devvrit, R. Kadekodi, R. Krishnaswamy, H. V. Simhadri. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS 2019.
2. A. Singh, S. J. Subramanya, R. Krishnaswamy, H. V. Simhadri. *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search.* arXiv:2105.09613, 2021.
3. Y. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 2018.
4. G. L. Steele Jr., D. Lea, C. H. Flood. *Fast Splittable Pseudorandom Number Generators* (SplitMix64). OOPSLA 2014.
5. Rust documentation: <https://doc.rust-lang.org> · Axum: <https://docs.rs/axum> · Tokio: <https://docs.rs/tokio> · tracing: <https://docs.rs/tracing> · memmap2: <https://docs.rs/memmap2>.
6. RiskANN source: <https://github.com/RiskANN/RiskANN>.

## 35. License

MIT. See [`LICENSE`](LICENSE).
