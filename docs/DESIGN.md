# RiskANN design notes

These answers describe the code in this repository. Section numbers in the [README](../README.md) give the user-facing view.

## 1. Is L2 Euclidean or squared Euclidean?
**Euclidean** (`sqrt` of the sum of squared differences), accumulated in `f64` (`crates/core/src/distances.rs::l2_distance`). Pruning uses the same value, so α is applied to Euclidean distances.

## 2. What does cosine return?
**Cosine distance** `1 − cos(a, b)`, clamped to `[0, 2]`.
- If either vector has zero norm the function returns `1.0`.
- The engine rejects zero vectors for COSINE collections on insert and on query, so that fallback is never reached through the API.

## 3. Frontier ordering
A min-heap `BinaryHeap<Reverse<Candidate>>`. `Candidate` has a total order: `(distance via f32::total_cmp, id)`. The nearest unexpanded candidate is expanded first.

## 4. Bounded result-set ordering
A max-heap `BinaryHeap<Candidate>` holding at most `ef` entries; its top is the worst retained candidate.
- A reached neighbour is admitted only if the set is not full or it beats the worst entry.
- Admission pushes the neighbour to both heaps and evicts the worst entry when the set grows past `ef`.

## 5. Stopping conditions
The search stops when the frontier is empty, or when the result set is full and the popped frontier candidate is worse than the worst retained result.
- Evicted candidates are always worse than the current worst, and the worst only improves over time.
- So this is equivalent to Vamana's GreedySearch with `L = ef`, which stops once every candidate in the list has been expanded.
- Each node is evaluated at most once (bitset visited set), so the search terminates on any graph, including cyclic and disconnected ones.

## 6. Equal distances
Equal distances are ordered by **lower internal id**, in approximate search, exact search and pruning alike. Repeated queries on an unchanged index return identical ordering. There is no randomness anywhere in build or search.

## 7. Defaults and maxima

| Parameter | Default | Range |
|---|---|---|
| `k` | 10 | 1 – 1000 |
| `ef_search` | `max(k, 64)` | must satisfy `k ≤ ef_search ≤ 4096` |

`ef_search < k` is rejected with 400; it is never silently raised.

## 8. Empty collection
Search and exact search return `200` with `"results": []`.

## 9. Entry points

Every search starts from a **set** of entry points (`entry_points` in `metadata.json`; `entry_points[0]` = `medoid_id`).

| How the data arrives | Entry points |
|---|---|
| First upsert into an empty collection | That vector |
| Further single upserts | Each new vector is added while the set is smaller than `min(entry_points, ⌈√n⌉)` |
| Bulk ingestion into an empty collection | The medoid (live vector nearest the `f64` centroid), then farthest-point sampling up to the target. Entry points are not linked as nodes themselves; they gain edges as reverse edges. |
| Incremental ingestion job | The set is re-selected by farthest-point sampling when the job ends (skipped for legacy single-entry collections) |

**Defaults and legacy collections.** `entry_points` defaults to 64 for new collections. Collections created before this setting existed have no `entry_points` in `config.json` and keep exactly one entry point.

**Validation on open.** The set must be unique, contain only live ids, and start with the medoid.

**Why.** The first benchmark showed a single entry point leaving 21 % of a clustered 64-d L2 collection unreachable; see README §5.6 and §25.7.

## 10. Uninitialised versus live slots
Internal ids are allocated densely.
- Ids `< next_id` are live; ids `next_id..capacity` are free.
- Free adjacency slots are filled with `u32::MAX` (empty list); legacy zero-filled slots are overwritten when allocated.
- A new id is published (added to `external_ids` / `id_map`) only after its vector and edges are written. Metadata is persisted after the data.
- Search treats any neighbour `>= next_id` as invalid and skips it.
- Deletion of single vectors is not implemented, so there are no tombstones.

## 11. Synchronisation
Each collection has one `std::sync::RwLock<IndexState>` covering the store, id maps and entry point.
- **Readers**: search, exact search, vector reads, info and graph verification.
- **Exclusive**: upserts, ingestion steps, and deletion (which waits once for in-flight writers).
- Mmap writes require `&mut DiskStore`, so the borrow checker prevents any slice from being held across a write or a remap.
- The server runs engine calls on `spawn_blocking`, so no lock is held across an `.await` or during HTTP serialisation.

Collection create and delete are serialised by a registry mutex.

## 12. Process stops during ingestion

| Mode | After a crash or kill |
|---|---|
| Initial build | Metadata is written only at the end, so the collection reopens **empty**. Vector bytes already written are unreferenced and are overwritten by later inserts. |
| Incremental | Records after the last checkpoint (`checkpoint_every`, default 10 000) are lost. Dangling edges they created are removed on the next open. |

Job status is in memory only. On graceful shutdown the Tokio runtime waits for running blocking ingestion tasks before exiting.

## 13. Metadata and file lengths disagree

| Condition | Result |
|---|---|
| A file length is not a multiple of its record size | `Corrupt` error |
| A file is empty | `Corrupt` error |
| A file holds fewer slots than `next_id` | `Corrupt` error |
| `external_ids` length ≠ `next_id`, duplicate ids, or a non-live entry point | `Corrupt` error |
| The two files hold different numbers of slots | The shorter file is **grown**, never truncated (interrupted resize) |
| `capacity` recorded in metadata differs from the files | Informational only: logged; the file-derived value is used |

A collection that fails to open is skipped at startup, reported in `/ready`, and left untouched.

## 14. Durability across restart

| Item | Durable? |
|---|---|
| `config.json` | Yes |
| `metadata.json` | Yes (atomic temp-file rename, fsync of file and directory) |
| Vector and graph files | Yes (`msync` before metadata) |
| Ingestion job status | **No**, process-local |

## 15. Measured benchmarks
None are recorded in this repository yet. The harness is `riskann-bench` (README §18); results must be reported together with the machine, the build profile, the command and the generated `bench_results/*.json`.

## Construction parameters (`l_build`)
New collections default to `l_build = max(2r, 64)`, capped at 1024. Collections created before this field existed have no `l_build` in `config.json` and keep the original `2r`. With the Phase-II demo setting `r = 3`, `2r = 6` is a very narrow construction beam, which is why the default was widened. Measure recall with `riskann-bench` (`--r`, `--l-build`) before choosing parameters.

## Observability
Logging uses `tracing`. Design rules:
- **One access-log line per request.** The request-id middleware (`crates/server/src/logging.rs`) is the outermost layer, so even rejected requests and unknown routes are logged and get an `x-request-id`.
- **Request context reaches the engine.** The request span is entered on the blocking thread that runs engine work, so engine events inherit `request_id`.
- **Jobs have their own context.** Ingestion jobs run in an `ingest_job` span, because they outlive the request that started them.
- **No bodies in logs.** Vectors and ids are never logged. Error responses carry a code in their extensions so the access log can report it without logging the message; the message is logged only at `debug`.
- **Levels:** `error` = server-side failure; `warn` = client error, slow query, repair on open; `info` = lifecycle and progress; `debug` = per-operation statistics; `trace` = every persist.

The full event catalogue is in README §12.6.
