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

//! Models endpoint (llama-server b10621 compatible, issue #1438).
//!
//! b10621's single-model `GET /models` / `GET /v1/models` answer carries two
//! blocks: the OpenAI-compatible `data` array whose entries have `id`,
//! `aliases`, `tags`, `object`, `created`, `owned_by: "llamacpp"` and a
//! `meta` block of checkpoint facts, and an Ollama-compatible `models` array
//! (`name`, `model`, `capabilities`, `details`, ...). mlxcel mirrors both;
//! its own additions (the `capabilities` routing hints, and extra entries
//! for a separately loaded embedder or reranker) are additive keys and
//! entries a b10621 client ignores.

use axum::{Json, extract::State};

use crate::server::AppState;
use crate::server::model_meta::{model_facts, vocab_type_code};

/// The b10621 OpenAI-compat model object (`get_res_model_info`).
fn b10621_model_info(state: &AppState, created: i64) -> serde_json::Value {
    let mut facts = model_facts(&state.model_path);
    facts.vocab_type = vocab_type_code(&state.tokenizer);
    // mlxcel routing hints (#1452), additive next to the b10621 keys: which
    // routes this id may be sent to. The primary entry matters most: when
    // `-m` is itself an embedding or reranker checkpoint its id is the only
    // one listed, and without this a client cannot tell that chat would 501.
    let mut capabilities: Vec<&'static str> = Vec::new();
    if !state.model_provider.is_chat_unavailable()
        && !state.config.embedding_serving_mode.blocks_generation()
    {
        capabilities.push("completion");
    }
    if state
        .embedding_model
        .as_ref()
        .is_some_and(|e| e.model_id() == state.display_model_id())
    {
        capabilities.push("embedding");
    }
    if state
        .rerank_model
        .as_ref()
        .is_some_and(|r| r.model_id() == state.display_model_id())
    {
        capabilities.push("rerank");
    }
    serde_json::json!({
        "id": state.display_model_id(),
        "aliases": state.config.model_aliases,
        "tags": state.config.model_tags,
        "object": "model",
        "created": created,
        // b10621's literal string; clients match on it (#1438 cleared the
        // former "user" divergence).
        "owned_by": "llamacpp",
        "meta": {
            "vocab_type": facts.vocab_type,
            "n_vocab": facts.n_vocab,
            // The per-slot window a request is actually bounded by; 0 means
            // the checkpoint's own trained context applies unclamped.
            "n_ctx": state.config.context_size,
            "n_ctx_train": facts.n_ctx_train,
            "n_embd": facts.n_embd,
            "n_params": facts.n_params,
            "size": facts.size,
            "ftype": facts.ftype,
        },
        "capabilities": capabilities,
    })
}

/// The b10621 Ollama-compat block (`get_res_models`'s `models` array entry).
fn b10621_ollama_entry(state: &AppState) -> serde_json::Value {
    let multimodal =
        state.media_support.image || state.media_support.video || state.media_support.audio;
    let capabilities = if multimodal {
        serde_json::json!(["completion", "multimodal"])
    } else {
        serde_json::json!(["completion"])
    };
    serde_json::json!({
        "name": state.display_model_id(),
        "model": state.display_model_id(),
        "modified_at": "",
        "size": "",
        // b10621's own comment: a dummy value, model file hashes are not
        // managed.
        "digest": "",
        "type": "model",
        "description": "",
        "tags": [""],
        "capabilities": capabilities,
        "parameters": "",
        "details": {
            "parent_model": "",
            // The checkpoint's real storage format; b10621 reports its own
            // ("gguf") here for the same reason.
            "format": "safetensors",
            "family": "",
            "families": [""],
            "parameter_size": "",
            "quantization_level": ""
        }
    })
}

/// GET /models, GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    // b10621 stamps the current time on every call rather than a load time.
    let created = chrono::Utc::now().timestamp();
    let mut data = vec![b10621_model_info(&state, created)];

    // mlxcel extension entries: a separately loaded embedding model
    // (`--embedding-model`) and reranker (`--reranker-model`) are listed next
    // to the chat model; when `-m` is itself that checkpoint the ids coincide
    // and the primary entry covers it.
    if let Some(embedding) = state.embedding_model.as_ref()
        && embedding.model_id() != state.display_model_id()
    {
        data.push(serde_json::json!({
            "id": embedding.model_id(),
            "object": "model",
            "created": embedding.created_at(),
            "owned_by": "llamacpp",
            "capabilities": ["embedding"],
        }));
    }
    if let Some(reranker) = state.rerank_model.as_ref()
        && data.iter().all(|model| model["id"] != reranker.model_id())
    {
        data.push(serde_json::json!({
            "id": reranker.model_id(),
            "object": "model",
            "created": reranker.created_at(),
            "owned_by": "llamacpp",
            "capabilities": ["rerank"],
        }));
    }

    Json(serde_json::json!({
        "models": [b10621_ollama_entry(&state)],
        "object": "list",
        "data": data,
    }))
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod models_tests;
