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

//! Tests for the b10621 `/v1/models` shape (issue #1438): the OpenAI `data`
//! block with `aliases`, `tags`, `owned_by: "llamacpp"` and the `meta` facts,
//! plus the Ollama-compat `models` block.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

use crate::server::config::ServerConfig;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, create_app};
use crate::tokenizer::MlxcelTokenizer;

fn app_with(config: ServerConfig) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("models-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn get_models(app: Router, path: &str) -> serde_json::Value {
    let request = Request::builder()
        .method(Method::GET)
        .uri(path)
        .body(Body::empty())
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json")
}

#[tokio::test]
async fn the_data_entry_carries_the_b10621_key_set() {
    let body = get_models(
        app_with(ServerConfig {
            model_aliases: vec!["alpha".into(), "beta".into()],
            model_tags: vec!["prod".into(), "chat".into()],
            context_size: 4096,
            ..Default::default()
        }),
        "/v1/models",
    )
    .await;
    let entry = &body["data"][0];
    for key in [
        "id", "aliases", "tags", "object", "created", "owned_by", "meta",
    ] {
        assert!(entry.get(key).is_some(), "missing {key}: {entry}");
    }
    assert_eq!(entry["owned_by"], "llamacpp");
    assert_eq!(entry["aliases"], serde_json::json!(["alpha", "beta"]));
    assert_eq!(entry["tags"], serde_json::json!(["prod", "chat"]));
    let meta = entry["meta"].as_object().expect("meta object");
    for key in [
        "vocab_type",
        "n_vocab",
        "n_ctx",
        "n_ctx_train",
        "n_embd",
        "n_params",
        "size",
        "ftype",
    ] {
        assert!(meta.contains_key(key), "missing meta.{key}: {meta:?}");
    }
    assert_eq!(meta["n_ctx"], 4096);
}

#[tokio::test]
async fn the_ollama_block_mirrors_b10621() {
    let body = get_models(app_with(ServerConfig::default()), "/models").await;
    let entry = &body["models"][0];
    assert_eq!(entry["model"], entry["name"]);
    assert_eq!(entry["type"], "model");
    assert_eq!(entry["capabilities"], serde_json::json!(["completion"]));
    assert_eq!(entry["details"]["format"], "safetensors");
    assert_eq!(body["object"], "list");
}

#[tokio::test]
async fn both_spellings_answer_the_same_shape() {
    let a = get_models(app_with(ServerConfig::default()), "/models").await;
    let b = get_models(app_with(ServerConfig::default()), "/v1/models").await;
    // `created` is stamped per call; compare everything else.
    let strip = |mut v: serde_json::Value| {
        v["data"][0]["created"] = 0.into();
        v
    };
    assert_eq!(strip(a), strip(b));
}
