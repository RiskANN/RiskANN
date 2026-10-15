# RiskANN

**A Rust-based, Vamana-inspired Approximate Nearest Neighbour (ANN) Vector Index.**

RiskANN is an experimental vector indexing project built in Rust. It explores graph-based approximate nearest-neighbour indexing, disk-backed vector storage, and an asynchronous HTTP API through a modular architecture.

The goal is to build a lightweight foundation for vector-based applications such as semantic retrieval, similarity search, recommendation systems, and AI-powered search.

> **Project Status:** Work in progress. The current implementation includes graph construction algorithms, distance metrics, persistent collection storage, vector upsert, bulk ingestion, and collection management APIs. A complete HTTP nearest-neighbour search API is not yet implemented.

---

## Features

- **Vamana-Inspired Graph Indexing:** Implements greedy graph traversal and robust pruning algorithms.
- **Distance Metrics:** Supports L2 and cosine distance through the `anndists` crate.
- **Vector Collections:** Create and delete named collections with configurable parameters.
- **Vector Upsert:** Insert or update vectors using external string identifiers.
- **Memory-Mapped Storage:** Stores vectors and graph adjacency data in binary files.
- **Metadata Persistence:** Saves collection configuration, ID mappings, and medoid metadata.
- **Bulk Ingestion:** Supports importing vectors from JSON Lines (JSONL) files.
- **Asynchronous HTTP Server:** Uses Axum and Tokio to expose REST endpoints.
- **Modular Rust Workspace:** Separates algorithms, storage, engine logic, and API handling into independent crates.

## Architecture

RiskANN follows a modular architecture designed to separate core algorithms from storage, collection management, and HTTP handling.

```text
RiskANN/
├── Cargo.toml
├── Cargo.lock
├── crates/
│   ├── core/
│   │   └── src/
│   │       ├── distances.rs
│   │       ├── vamana.rs
│   │       └── lib.rs
│   │
│   ├── storage/
│   │   └── src/
│   │       ├── memory.rs
│   │       └── lib.rs
│   │
│   ├── engine/
│   │   └── src/
│   │       ├── collection.rs
│   │       └── lib.rs
│   │
│   └── server/
│       └── src/
│           ├── api/
│           │   ├── collections.rs
│           │   ├── upsert.rs
│           │   └── mod.rs
│           └── main.rs
│
└── docs/
    └── README.md
```

### Core Components

| Component | Responsibility |
|---|---|
| `core` | Distance metrics, candidate management, greedy graph traversal, and robust pruning. |
| `storage` | Memory-mapped vector and graph storage using binary files. |
| `engine` | Collection lifecycle, vector insertion, graph updates, ID mapping, and metadata persistence. |
| `server` | HTTP endpoints implemented using Axum and Tokio. |

## Technology Stack

- **Language:** Rust (Edition 2024)
- **HTTP Framework:** Axum
- **Asynchronous Runtime:** Tokio
- **Serialization:** Serde and Serde JSON
- **Vector Distance Calculations:** `anndists`
- **Storage:** Memory-mapped files using `memmap2`
- **Binary Data Handling:** `bytemuck`
- **Concurrent Data Structures:** `DashMap`
- **Configuration:** Environment variables and `dotenvy`
- **Build System:** Cargo workspace

---

## Getting Started

### Prerequisites

Install Rust and Cargo using [rustup](https://rustup.rs/).

Verify the installation:

```bash
rustc --version
cargo --version
```

### Clone the Repository

```bash
git clone https://github.com/RiskANN/RiskANN.git
cd RiskANN
```

### Build the Project

Build all workspace crates:

```bash
cargo build
```

For an optimized build:

```bash
cargo build --release
```

### Run the Server

```bash
cargo run -p server
```

By default, the server listens on:

```text
http://localhost:3000
```

The default bind address is `0.0.0.0:3000`.

### Run with Custom Configuration

You can override the server configuration through environment variables:

```bash
HOST=127.0.0.1 PORT=3000 DATA_LOC=./data cargo run -p server
```

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Address on which the server listens. |
| `PORT` | `3000` | HTTP server port. |
| `DATA_LOC` | `/tmp/db_data` | Directory used to store collection data. |

The server attempts to load environment variables from a `.env` file in its working directory.

---

## HTTP API

The current server exposes three endpoints for collection management and vector insertion.

### 1. Create a Collection

**Endpoint:** `PUT /collections/{name}`

Creates a new vector collection with the specified configuration.

Example:

```bash
curl -i -X PUT http://localhost:3000/collections/demo \
  -H 'Content-Type: application/json' \
  -d '{
    "dim": 3,
    "metric": "L2",
    "r": 16,
    "alpha": 1.2,
    "initial_capacity": 1000
  }'
```

Parameters:

| Field | Type | Description |
|---|---|---|
| `dim` | Integer | Dimension of each vector. |
| `metric` | String | `L2` or `COSINE`. |
| `r` | Integer | Graph degree and pruning parameter. |
| `alpha` | Number | Robust-pruning parameter. |
| `initial_capacity` | Integer | Initial number of vector slots. |
| `bulk_file` | String (optional) | Server-side path to a JSONL file for bulk ingestion. |

Expected responses:

- `201 Created` — collection created.
- `202 Accepted` — bulk ingestion scheduled.
- `409 Conflict` — collection already exists.

### 2. Upsert a Vector

**Endpoint:** `POST /collections/{name}/upsert`

Inserts a vector into an existing collection or updates the vector associated with an existing external ID.

Example:

```bash
curl -i -X POST http://localhost:3000/collections/demo/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "item-001",
    "vector": [0.12, 0.34, 0.56]
  }'
```

Request fields:

- `id`: Unique external identifier for the vector.
- `vector`: Array of floating-point values whose length matches the collection dimension.

Expected responses:

- `200 OK` — vector upsert completed.
- `400 Bad Request` — vector dimension mismatch.
- `404 Not Found` — collection does not exist.

### 3. Delete a Collection

**Endpoint:** `DELETE /collections/{name}`

Deletes a collection and its associated data directory.

Example:

```bash
curl -i -X DELETE http://localhost:3000/collections/demo
```

Expected responses:

- `200 OK` — collection deleted.
- `404 Not Found` — collection does not exist.

### API Summary

| Method | Endpoint | Description |
|---|---|---|
| PUT | `/collections/{name}` | Create a collection. |
| POST | `/collections/{name}/upsert` | Insert or update a vector. |
| DELETE | `/collections/{name}` | Delete a collection. |

**Note:** The current HTTP API does not yet provide nearest-neighbour search, collection listing, vector retrieval, or bulk-ingestion status endpoints.

---

## Bulk Vector Ingestion

RiskANN supports bulk ingestion from a JSON Lines file. Each line represents one vector record.

Example `vectors.jsonl`:

```jsonl
{"id":"item-001","vector":[0.12,0.34,0.56]}
{"id":"item-002","vector":[0.78,0.90,0.12]}
{"id":"item-003","vector":[0.45,0.67,0.89]}
```

To start bulk ingestion, provide the file path in the collection creation request:

```json
{
  "dim": 3,
  "metric": "L2",
  "r": 16,
  "alpha": 1.2,
  "initial_capacity": 1000,
  "bulk_file": "/absolute/path/to/vectors.jsonl"
}
```

The file must be accessible to the server process. The API returns `202 Accepted` after scheduling the ingestion task; it does not currently expose an endpoint to check ingestion progress.

---

## Storage Design

Each collection is stored in its own directory under `DATA_LOC`.

```text
data/
└── demo/
    ├── config.json
    ├── metadata.json
    ├── vectors.bin
    └── graph.bin
```

- **`config.json`** — collection configuration, including vector dimension, distance metric, and graph parameters.
- **`metadata.json`** — collection metadata, including the medoid and external-to-internal ID mapping.
- **`vectors.bin`** — memory-mapped binary vector storage.
- **`graph.bin`** — memory-mapped graph adjacency storage.

The current storage implementation allocates fixed-capacity files. Dynamic storage growth and robust capacity management are not yet implemented.

---

## Development

### Format the Code

```bash
cargo fmt --all
```

### Check the Workspace

```bash
cargo check --workspace
```

### Run Tests

```bash
cargo test --workspace
```

These commands are provided for development and validation. The presence of a command does not imply that comprehensive tests have already been implemented.

### Build an Optimized Binary

```bash
cargo build --release -p server
```

---

## Current Limitations and Roadmap

RiskANN is an evolving project. The following areas remain important development opportunities:

- [ ] Implement an HTTP nearest-neighbour search endpoint.
- [ ] Add collection listing and vector retrieval APIs.
- [ ] Add bulk-ingestion progress and error reporting.
- [ ] Implement dynamic storage capacity growth.
- [ ] Improve input validation and malformed-file handling.
- [ ] Strengthen persistence, crash recovery, and concurrency guarantees.
- [ ] Add automated unit and integration tests.
- [ ] Benchmark indexing speed, memory usage, and query latency.
- [ ] Add authentication, authorization, and production deployment safeguards.
- [ ] Improve monitoring, logging, and operational diagnostics.

The roadmap is indicative and does not imply that these features are already available.

---

## Contributing

Contributions, bug reports, and technical suggestions are welcome.

When contributing:

1. Fork the repository.
2. Create a branch for your changes.
3. Follow Rust formatting conventions.
4. Test your changes where possible.
5. Submit a pull request with a clear description of the implementation.

For bug reports, include reproduction steps, expected behavior, actual behavior, and relevant environment details.

---

## License

RiskANN is intended to be distributed under the **MIT License**.

See the [`LICENSE`](LICENSE) file in the repository root for the complete license terms.

If the root-level `LICENSE` file has not yet been committed, add it before treating the repository as officially MIT-licensed. Confirm the copyright holder and year before committing the license.

---

## Repository

GitHub: https://github.com/RiskANN/RiskANN

**RiskANN — Building the foundations for efficient graph-based vector indexing in Rust.**
