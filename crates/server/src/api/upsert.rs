use axum::{http::StatusCode, Json};
use axum::extract::{Path, State};
use engine::collection::DBInfo;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct UpsertReq {
    pub id: String,
    pub vector: Vec<f32>,
}

pub async fn upsert_vector( Path(name): Path<String>, State(state): State<Arc<DBInfo>>, Json(payload): Json<UpsertReq> ) 
                                                                                                       -> Result<StatusCode, (StatusCode, String)> {
    
    let collection = {
        let Some(coll_ref) = state.collections.get(&name) else {
            return Err((StatusCode::NOT_FOUND, "Collection not found".to_string()));
        };
        coll_ref.clone()
    }; 

    if payload.vector.len() != collection.config.dim {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Wrong dimension of data [{}]. Collection config: {}", payload.vector.len(), collection.config.dim),
        ));
    }

    let internal_id = {
        let mut map = collection.id_map.write().await;
        
        if let Some(&existing_id) = map.get(&payload.id) {
            existing_id
        } else {
            let new_id = collection.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            map.insert(payload.id.clone(), new_id);
            new_id
        }
    };

    collection.upsert(internal_id, &payload.vector).await;
    Ok(StatusCode::OK)
}
