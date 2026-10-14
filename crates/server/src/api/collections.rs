use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use engine::collection::{Collection, CollectionConfig, DistanceMetric, DBInfo};
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc};

#[derive(Deserialize)]
pub struct CreateCollectionReq {
    pub dim: usize,
    pub metric: DistanceMetric,
    pub r: usize,
    pub alpha: f32,
    pub initial_capacity: u32,
    pub bulk_file: Option<String>, 
}

pub async fn create_collection( Path(name): Path<String>, State(state): State<Arc<DBInfo>>, Json(payload): Json<CreateCollectionReq> ) 
                                                                                                           -> Result<StatusCode, StatusCode> {
    if state.collections.contains_key(&name) {
        return Err(StatusCode::CONFLICT);
    }

    let mut collection_dir = PathBuf::from(&state.data_loc);
    collection_dir.push(&name);

    let _ = std::fs::create_dir_all(&collection_dir);

    let config = CollectionConfig {
        dim: payload.dim,
        metric: payload.metric,
        r: payload.r,
        alpha: payload.alpha,
    };

    let collection = Collection::create(collection_dir, config, payload.initial_capacity).await;
    let collection_arc = Arc::new(collection);

    state.collections.insert(name.clone(), collection_arc.clone());

    if let Some(uri) = payload.bulk_file {
        tokio::spawn(async move { // Background process, to prevent blocking
            collection_arc.bulk_insert(&uri).await;
        });
        return Ok(StatusCode::ACCEPTED);
    }

    Ok(StatusCode::CREATED)
}

pub async fn delete_collection( Path(name): Path<String>, State(state): State<Arc<DBInfo>>) -> StatusCode {
    if let Some((_, _collection)) = state.collections.remove(&name) {
        let mut collection_dir = PathBuf::from(&state.data_loc);
        collection_dir.push(&name);
        let _ = std::fs::remove_dir_all(&collection_dir);
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}
