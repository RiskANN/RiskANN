//! HTTP API integration tests. The router is exercised in-process (no
//! socket) with a fresh temporary data directory per test.

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use engine::DBInfo;
use serde_json::{Value, json};
use server::{AppState, ServerConfig, build_router};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tower::ServiceExt;

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
            "riskann-api-test-{}-{}-{}",
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

fn app_with(dir: &TempDir, tweak: impl FnOnce(&mut ServerConfig)) -> Router {
    let mut cfg = ServerConfig::for_data_dir(dir.path().join("data"));
    tweak(&mut cfg);
    let (db, report) = DBInfo::open(cfg.data_loc.clone()).unwrap();
    build_router(Arc::new(AppState::new(db, cfg, report)))
}

fn app(dir: &TempDir) -> Router {
    app_with(dir, |_| {})
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, value)
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(v) => {
            builder = builder.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    send(app, builder.body(body).unwrap()).await
}

async fn raw(
    app: &Router,
    method: &str,
    uri: &str,
    content_type: Option<&str>,
    body: &str,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(ct) = content_type {
        builder = builder.header("content-type", ct);
    }
    send(app, builder.body(Body::from(body.to_string())).unwrap()).await
}

fn assert_error(v: &Value, code: &str) {
    assert_eq!(v["error"]["code"], code, "unexpected body: {v}");
    assert!(v["error"]["message"].is_string(), "missing message: {v}");
}

async fn create(app: &Router, name: &str, dim: usize) -> (StatusCode, Value) {
    call(
        app,
        "PUT",
        &format!("/collections/{name}"),
        Some(json!({"dim": dim, "metric": "L2", "r": 8, "alpha": 1.2, "initial_capacity": 16})),
    )
    .await
}

async fn seed_demo(app: &Router) {
    assert_eq!(create(app, "demo", 3).await.0, StatusCode::CREATED);
    let points = [
        ("item-001", [0.12, 0.34, 0.56]),
        ("item-002", [0.78, 0.90, 0.12]),
        ("item-003", [0.45, 0.67, 0.89]),
        ("item-004", [0.10, 0.30, 0.50]),
        ("item-005", [0.90, 0.10, 0.20]),
    ];
    for (id, v) in points {
        let (s, body) = call(
            app,
            "POST",
            "/collections/demo/upsert",
            Some(json!({"id": id, "vector": v})),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{body}");
        assert_eq!(body["created"], true);
    }
}

async fn wait_for_job(app: &Router, url: &str) -> Value {
    for _ in 0..500 {
        let (s, v) = call(app, "GET", url, None).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        if v["state"] == "completed" || v["state"] == "failed" {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("job did not finish: {url}");
}

fn write_jsonl(dir: &TempDir, name: &str, lines: &[String]) -> PathBuf {
    let p = dir.path().join(name);
    std::fs::write(&p, lines.join("\n")).unwrap();
    p
}

#[tokio::test]
async fn health_and_ready() {
    let dir = TempDir::new();
    let app = app(&dir);
    let (s, v) = call(&app, "GET", "/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["status"], "ok");
    let (s, v) = call(&app, "GET", "/ready", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["ready"], true);
}

#[tokio::test]
async fn create_upsert_search_inspect_delete_flow() {
    let dir = TempDir::new();
    let app = app(&dir);
    seed_demo(&app).await;

    // Updating an existing id keeps its internal id.
    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/upsert",
        Some(json!({"id": "item-005", "vector": [0.9, 0.1, 0.25]})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["created"], false);
    assert_eq!(v["internal_id"], 4);

    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/search",
        Some(json!({"vector": [0.10, 0.30, 0.50], "k": 2, "ef_search": 16})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["collection"], "demo");
    assert_eq!(v["metric"], "L2");
    assert_eq!(v["mode"], "approximate");
    assert_eq!(v["ef_search"], 16);
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["id"], "item-004");
    assert_eq!(results[1]["id"], "item-001");
    assert!(results[0]["distance"].as_f64().unwrap() < 1e-6);
    assert!(results[0]["distance"].as_f64() <= results[1]["distance"].as_f64());
    assert!(results[0].get("vector").is_none());
    for key in [
        "visited_nodes",
        "expanded_nodes",
        "distance_computations",
        "returned",
        "search_duration_us",
    ] {
        assert!(v["stats"][key].is_u64(), "missing stats.{key}: {v}");
    }

    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/search/exact",
        Some(json!({"vector": [0.10, 0.30, 0.50], "k": 2})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["mode"], "exact");
    assert_eq!(v["results"][0]["id"], "item-004");
    assert_eq!(v["stats"]["distance_computations"], 5);

    let (s, v) = call(&app, "GET", "/collections/demo", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["collection"]["count"], 5);
    assert_eq!(v["collection"]["dim"], 3);
    assert_eq!(v["collection"]["index_state"], "ready");
    assert!(
        !v.to_string().contains(dir.path().to_str().unwrap()),
        "path leaked: {v}"
    );

    let (s, v) = call(&app, "GET", "/collections", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["collections"][0]["name"], "demo");

    let (s, v) = call(&app, "GET", "/collections/demo/vectors/item-003", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["id"], "item-003");
    assert_eq!(v["vector"].as_array().unwrap().len(), 3);

    let (s, _) = call(&app, "DELETE", "/collections/demo", None).await;
    assert_eq!(s, StatusCode::OK);
    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/search",
        Some(json!({"vector": [0, 0, 0]})),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "COLLECTION_NOT_FOUND");
    let (s, v) = call(&app, "DELETE", "/collections/demo", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "COLLECTION_NOT_FOUND");
}

#[tokio::test]
async fn search_request_validation() {
    let dir = TempDir::new();
    let app = app(&dir);
    seed_demo(&app).await;
    let url = "/collections/demo/search";

    let cases = [
        (json!({"vector": [0.1, 0.2]}), "DIMENSION_MISMATCH"),
        (
            json!({"vector": [0.1, 0.2, 0.3], "k": 0}),
            "INVALID_ARGUMENT",
        ),
        (
            json!({"vector": [0.1, 0.2, 0.3], "k": 1001}),
            "INVALID_ARGUMENT",
        ),
        (
            json!({"vector": [0.1, 0.2, 0.3], "k": 10, "ef_search": 5}),
            "INVALID_ARGUMENT",
        ),
        (
            json!({"vector": [0.1, 0.2, 0.3], "ef_search": 100000}),
            "INVALID_ARGUMENT",
        ),
        (
            json!({"vector": [0.1, 0.2, 0.3], "filter": {"tag": "x"}}),
            "MALFORMED_REQUEST",
        ),
        (json!({"k": 3}), "MALFORMED_REQUEST"),
        (
            json!({"vector": [0.1, 0.2, 0.3], "k": -1}),
            "MALFORMED_REQUEST",
        ),
    ];
    for (body, code) in cases {
        let (s, v) = call(&app, "POST", url, Some(body.clone())).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{body} -> {v}");
        assert_error(&v, code);
    }

    // 1e39 overflows f32 to +inf; it must be rejected, never searched.
    let (s, v) = raw(
        &app,
        "POST",
        url,
        Some("application/json"),
        r#"{"vector":[1e39,0,0]}"#,
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"]["code"].is_string());

    let (s, v) = raw(&app, "POST", url, Some("application/json"), "{not json").await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "MALFORMED_REQUEST");

    let (s, v) = raw(&app, "POST", url, None, r#"{"vector":[0,0,0]}"#).await;
    assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_error(&v, "UNSUPPORTED_MEDIA_TYPE");

    let (s, v) = call(
        &app,
        "POST",
        "/collections/nope/search",
        Some(json!({"vector": [0, 0, 0]})),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "COLLECTION_NOT_FOUND");
}

#[tokio::test]
async fn result_payload_options_and_empty_collection() {
    let dir = TempDir::new();
    let app = app(&dir);
    assert_eq!(create(&app, "empty", 3).await.0, StatusCode::CREATED);
    let (s, v) = call(
        &app,
        "POST",
        "/collections/empty/search",
        Some(json!({"vector": [1, 2, 3], "k": 5})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["results"], json!([]));

    seed_demo(&app).await;
    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/search",
        Some(json!({"vector": [0.1, 0.3, 0.5], "k": 1, "include_distances": false, "include_vectors": true})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let hit = &v["results"][0];
    assert!(hit.get("distance").is_none(), "{v}");
    assert_eq!(hit["vector"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn collection_creation_errors() {
    let dir = TempDir::new();
    let app = app(&dir);
    assert_eq!(create(&app, "dup", 3).await.0, StatusCode::CREATED);
    let (s, v) = create(&app, "dup", 3).await;
    assert_eq!(s, StatusCode::CONFLICT);
    assert_error(&v, "COLLECTION_EXISTS");

    let (s, v) = create(&app, "bad.name", 3).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "INVALID_COLLECTION_NAME");

    let (s, v) = create(&app, "zero", 0).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "INVALID_ARGUMENT");

    let (s, v) = call(
        &app,
        "PUT",
        "/collections/metric",
        Some(json!({"dim": 3, "metric": "IP", "r": 8, "alpha": 1.2})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "MALFORMED_REQUEST");

    let (s, v) = call(
        &app,
        "PUT",
        "/collections/alpha",
        Some(json!({"dim": 3, "metric": "COSINE", "r": 8, "alpha": 0.5})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "INVALID_ARGUMENT");
}

#[tokio::test]
async fn upsert_and_vector_errors() {
    let dir = TempDir::new();
    let app = app(&dir);
    seed_demo(&app).await;
    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/upsert",
        Some(json!({"id": "x", "vector": [1, 2]})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "DIMENSION_MISMATCH");
    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/upsert",
        Some(json!({"id": "", "vector": [1, 2, 3]})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "INVALID_ARGUMENT");
    let (s, v) = call(
        &app,
        "POST",
        "/collections/none/upsert",
        Some(json!({"id": "a", "vector": [1, 2, 3]})),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "COLLECTION_NOT_FOUND");
    let (s, v) = call(&app, "GET", "/collections/demo/vectors/missing", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "VECTOR_NOT_FOUND");
}

#[tokio::test]
async fn ingestion_job_lifecycle_and_retrieval() {
    let dir = TempDir::new();
    let app = app(&dir);
    assert_eq!(create(&app, "bulk", 3).await.0, StatusCode::CREATED);
    let lines: Vec<String> = (0..50)
        .map(|i| {
            json!({"id": format!("r{i}"), "vector": [i as f32, (i % 5) as f32, 1.0]}).to_string()
        })
        .collect();
    let file = write_jsonl(&dir, "data.jsonl", &lines);

    let (s, v) = call(
        &app,
        "POST",
        "/collections/bulk/ingest",
        Some(json!({"path": file.to_str().unwrap()})),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    let url = v["status_url"].as_str().unwrap().to_string();
    let job = wait_for_job(&app, &url).await;
    assert_eq!(job["state"], "completed", "{job}");
    assert_eq!(job["records_inserted"], 50);
    assert_eq!(job["records_rejected"], 0);
    assert_eq!(job["mode"], "initial_build");

    let (s, v) = call(
        &app,
        "POST",
        "/collections/bulk/search",
        Some(json!({"vector": [42.0, 2.0, 1.0], "k": 1})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["results"][0]["id"], "r42");

    // A second job into the non-empty collection runs incrementally.
    let more: Vec<String> = (50..60)
        .map(|i| json!({"id": format!("r{i}"), "vector": [i as f32, 0.0, 1.0]}).to_string())
        .collect();
    let file2 = write_jsonl(&dir, "more.jsonl", &more);
    let (_, v) = call(
        &app,
        "POST",
        "/collections/bulk/ingest",
        Some(json!({"path": file2.to_str().unwrap()})),
    )
    .await;
    let job = wait_for_job(&app, v["status_url"].as_str().unwrap()).await;
    assert_eq!(job["state"], "completed");
    assert_eq!(job["mode"], "incremental");
    let (_, v) = call(&app, "GET", "/collections/bulk", None).await;
    assert_eq!(v["collection"]["count"], 60);
}

#[tokio::test]
async fn create_with_bulk_file_returns_job() {
    let dir = TempDir::new();
    let app = app(&dir);
    let lines: Vec<String> = (0..20)
        .map(|i| json!({"id": format!("b{i}"), "vector": [i as f32, 1.0, 2.0]}).to_string())
        .collect();
    let file = write_jsonl(&dir, "bulk.jsonl", &lines);
    let (s, v) = call(
        &app,
        "PUT",
        "/collections/bulk_db",
        Some(
            json!({"dim": 3, "metric": "L2", "r": 3, "alpha": 1.2, "initial_capacity": 10000,
                    "bulk_file": file.to_str().unwrap()}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    assert!(v["job_id"].is_string());
    let job = wait_for_job(&app, v["status_url"].as_str().unwrap()).await;
    assert_eq!(job["state"], "completed", "{job}");
    assert_eq!(job["records_inserted"], 20);
}

#[tokio::test]
async fn failed_job_reports_line_number() {
    let dir = TempDir::new();
    let app = app(&dir);
    assert_eq!(create(&app, "ff", 2).await.0, StatusCode::CREATED);
    let lines = vec![
        r#"{"id":"a","vector":[1,2]}"#.to_string(),
        r#"{"id":"b","vector":[1]}"#.to_string(),
    ];
    let file = write_jsonl(&dir, "bad.jsonl", &lines);
    let (_, v) = call(
        &app,
        "POST",
        "/collections/ff/ingest",
        Some(json!({"path": file.to_str().unwrap()})),
    )
    .await;
    let job = wait_for_job(&app, v["status_url"].as_str().unwrap()).await;
    assert_eq!(job["state"], "failed");
    assert!(job["error"].as_str().unwrap().contains("line 2"), "{job}");
    assert_eq!(job["rejected_samples"][0]["line"], 2);
    assert!(
        !job.to_string().contains(dir.path().to_str().unwrap()),
        "path leaked: {job}"
    );
}

#[tokio::test]
async fn ingestion_path_policy() {
    let dir = TempDir::new();
    let allowed = dir.path().join("allowed");
    std::fs::create_dir_all(&allowed).unwrap();
    let inside = allowed.join("in.jsonl");
    std::fs::write(&inside, r#"{"id":"a","vector":[1,2,3]}"#).unwrap();
    let outside = write_jsonl(
        &dir,
        "out.jsonl",
        &[r#"{"id":"a","vector":[1,2,3]}"#.to_string()],
    );
    let root = allowed.clone();
    let app = app_with(&dir, move |c| c.ingest_root = Some(root));
    assert_eq!(create(&app, "p", 3).await.0, StatusCode::CREATED);

    let post = |path: String| {
        let app = app.clone();
        async move {
            call(
                &app,
                "POST",
                "/collections/p/ingest",
                Some(json!({"path": path})),
            )
            .await
        }
    };
    let (s, v) = post("relative.jsonl".into()).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "INVALID_ARGUMENT");
    let (s, v) = post(allowed.join("missing.jsonl").to_string_lossy().into_owned()).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "INGEST_FILE_UNAVAILABLE");
    let (s, v) = post(outside.to_string_lossy().into_owned()).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_error(&v, "INGEST_PATH_NOT_ALLOWED");
    let escape = allowed.join("..").join("out.jsonl");
    let (s, v) = post(escape.to_string_lossy().into_owned()).await;
    assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
    let (s, v) = post(inside.to_string_lossy().into_owned()).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");

    let (s, v) = call(&app, "GET", "/collections/p/ingest/job-0-0", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "JOB_NOT_FOUND");

    // A rejected bulk_file must not leave a collection behind.
    let (s, _) = call(
        &app,
        "PUT",
        "/collections/never",
        Some(json!({"dim": 3, "metric": "L2", "r": 8, "alpha": 1.2, "bulk_file": outside.to_str().unwrap()})),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    let (s, _) = call(&app, "GET", "/collections/never", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn oversized_body_and_unknown_route() {
    let dir = TempDir::new();
    let app = app_with(&dir, |c| c.max_body_bytes = 256);
    assert_eq!(create(&app, "small", 3).await.0, StatusCode::CREATED);
    let big = json!({"id": "x", "vector": [1.0, 2.0, 3.0], "padding": "a".repeat(2048)});
    let (s, v) = call(&app, "POST", "/collections/small/upsert", Some(big)).await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
    assert_error(&v, "PAYLOAD_TOO_LARGE");

    let (s, v) = call(&app, "GET", "/does/not/exist", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_error(&v, "ROUTE_NOT_FOUND");
}

#[tokio::test]
async fn data_survives_server_restart() {
    let dir = TempDir::new();
    {
        let app = app(&dir);
        seed_demo(&app).await;
    }
    let app = app(&dir);
    let (s, v) = call(&app, "GET", "/ready", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["collections_loaded"], 1);
    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/search",
        Some(json!({"vector": [0.10, 0.30, 0.50], "k": 1})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["results"][0]["id"], "item-004");
}

#[tokio::test]
async fn every_response_carries_a_request_id() {
    let dir = TempDir::new();
    let app = app(&dir);

    // Generated id on success.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let id = resp
        .headers()
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(id.starts_with("req-"), "{id}");

    // A valid incoming id is echoed back, also on errors.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/collections/missing")
                .header("x-request-id", "client-abc_1.2")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        resp.headers().get("x-request-id").unwrap(),
        "client-abc_1.2"
    );

    // An invalid incoming id is replaced, not reflected.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("x-request-id", "bad id with spaces")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let id = resp
        .headers()
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(id.starts_with("req-"), "{id}");
}

#[tokio::test]
async fn console_graph_snapshot_and_search_trace() {
    let dir = TempDir::new();
    let app = app(&dir);
    let resp = app
        .clone()
        .oneshot(Request::builder().uri("/ui").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(ct.starts_with("text/html"), "{ct}");
    let html = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&html).contains("RiskANN Console"));

    seed_demo(&app).await;
    let (s, v) = call(&app, "GET", "/collections/demo/graph?limit=100", None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["nodes"].as_array().unwrap().len(), 5);
    assert_eq!(v["count"], 5);
    assert_eq!(v["truncated"], false);
    assert!(v["entry_point"].is_u64());
    let (s, v) = call(&app, "GET", "/collections/demo/graph?limit=0", None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "INVALID_ARGUMENT");
    let (s, v) = call(&app, "GET", "/collections/demo/graph?limit=abc", None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_error(&v, "MALFORMED_REQUEST");

    let (s, v) = call(
        &app,
        "POST",
        "/collections/demo/search",
        Some(json!({"vector": [0.1, 0.3, 0.5], "k": 2, "trace": true})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let expanded = v["trace"]["expanded"].as_array().unwrap();
    assert!(
        v["trace"]["entry_points"]
            .as_array()
            .unwrap()
            .contains(&expanded[0])
    );
    assert_eq!(
        expanded.len() as u64,
        v["stats"]["expanded_nodes"].as_u64().unwrap()
    );
    let (_, v) = call(
        &app,
        "POST",
        "/collections/demo/search",
        Some(json!({"vector": [0.1, 0.3, 0.5]})),
    )
    .await;
    assert!(v.get("trace").is_none());
}
