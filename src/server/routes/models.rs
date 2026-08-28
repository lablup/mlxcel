// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Models endpoint.
//!
//! This route reports server-visible model identity only; model loading and
//! capability policy stay in the loading and provider layers.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::types::{ModelInfo, ModelsResponse};

/// GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    // `capabilities` says what each id may be sent to (#1452). The primary
    // entry is the interesting one: when `-m` is itself an embedding or
    // reranker checkpoint its id is the only id on the list, and before this
    // field a client had no way to tell that sending it to
    // `/v1/chat/completions` would 501.
    let mut primary: Vec<&'static str> = Vec::new();
    if !state.model_provider.is_chat_unavailable()
        && !state.config.embedding_serving_mode.blocks_generation()
    {
        primary.push("completion");
    }
    if state
        .embedding_model
        .as_ref()
        .is_some_and(|e| e.model_id() == state.display_model_id())
    {
        primary.push("embedding");
    }
    if state
        .rerank_model
        .as_ref()
        .is_some_and(|r| r.model_id() == state.display_model_id())
    {
        primary.push("rerank");
    }
    let mut models = vec![ModelInfo {
        id: state.display_model_id().to_string(),
        object: "model".to_string(),
        created: state.model_provider.created_at(),
        owned_by: "user".to_string(),
        capabilities: primary,
    }];

    // A separately loaded embedding model (`--embedding-model`) is listed
    // next to the chat model; when `-m` itself is the embedding checkpoint
    // the two ids coincide and the entry above already covers it.
    if let Some(embedding) = state.embedding_model.as_ref()
        && embedding.model_id() != state.display_model_id()
    {
        models.push(ModelInfo {
            id: embedding.model_id().to_string(),
            object: "model".to_string(),
            created: embedding.created_at(),
            owned_by: "user".to_string(),
            capabilities: vec!["embedding"],
        });
    }

    // Same rule for a separately loaded reranker (`--reranker-model`): it is
    // its own served id unless `-m` is the reranker checkpoint itself.
    if let Some(reranker) = state.rerank_model.as_ref()
        && !models.iter().any(|model| model.id == reranker.model_id())
    {
        models.push(ModelInfo {
            id: reranker.model_id().to_string(),
            object: "model".to_string(),
            created: reranker.created_at(),
            owned_by: "user".to_string(),
            capabilities: vec!["rerank"],
        });
    }

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}
