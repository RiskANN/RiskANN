use axum::{ routing, Router};
use engine::collection::DBInfo;
use std::sync::Arc;
use tokio::net::TcpListener;

mod api;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    let data_loc = std::env::var("DATA_LOC").unwrap_or_else(|_| "/tmp/db_data".to_string());
    let host  = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());

    std::fs::create_dir_all(&data_loc).unwrap();

    let shared_state = Arc::new(DBInfo::init(data_loc).await);
    let app = Router::new()
        .route("/collections/:name", routing::put(api::collections::create_collection))
        .route("/collections/:name", routing::delete(api::collections::delete_collection))
        .route("/collections/:name/upsert", routing::post(api::upsert::upsert_vector))
        .with_state(shared_state);

    let address = format!("{host}:{port}");
    let listener = TcpListener::bind(address).await.unwrap();

    println!("{}",format!("Running on: {host}:{port}"));
    axum::serve(listener, app).await.unwrap();
}
