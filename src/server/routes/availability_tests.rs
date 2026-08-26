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

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::embeddings::{EmbedOptions, EmbedReply, ImageInput};
use crate::rerank::{RerankItem, RerankScores, RerankerKind};
use crate::server::audio_model::{AudioModelKind, AudioModelProvider};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};
use crate::server::rerank_model::{RerankError, RerankModelProvider};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

struct SideEmbeddingProvider;

impl EmbeddingModelProvider for SideEmbeddingProvider {
    fn embed_texts(
        &self,
        _texts: Vec<String>,
        _opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        unreachable!("availability preflight must prevent embedding dispatch")
    }

    fn embed_tokens(
        &self,
        _token_rows: Vec<Vec<u32>>,
        _opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        unreachable!("availability preflight must prevent embedding dispatch")
    }

    fn embed_image(
        &self,
        _image: ImageInput,
        _opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        unreachable!("availability preflight must prevent embedding dispatch")
    }

    fn model_id(&self) -> &str {
        "side-embedding"
    }

    fn created_at(&self) -> i64 {
        0
    }

    fn dim(&self) -> usize {
        1
    }

    fn multi_vector(&self) -> bool {
        false
    }

    fn vocab_size(&self) -> usize {
        1
    }

    fn max_length(&self) -> usize {
        1
    }
}

struct SideRerankProvider;

impl RerankModelProvider for SideRerankProvider {
    fn rerank(
        &self,
        _query: RerankItem,
        _documents: Vec<RerankItem>,
        _instruction: Option<String>,
    ) -> Result<RerankScores, RerankError> {
        unreachable!("availability preflight must prevent rerank dispatch")
    }

    fn model_id(&self) -> &str {
        "side-reranker"
    }

    fn created_at(&self) -> i64 {
        0
    }

    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn max_length(&self) -> usize {
        1
    }
}

struct SideAudioProvider;

impl AudioModelProvider for SideAudioProvider {
    fn supports(&self, kind: AudioModelKind) -> bool {
        kind == AudioModelKind::Stt
    }
}

fn state_with(provider: ModelProvider) -> AppState {
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
}

async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("route responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or_else(|err| {
            panic!(
                "status {status}, invalid JSON body {body:?}: {err}",
                body = String::from_utf8_lossy(&bytes)
            )
        })
    };
    (status, body)
}

#[tokio::test]
async fn embedding_only_server_returns_501_on_all_generation_routes() {
    let cases = [
        (
            "/v1/chat/completions",
            json!({"model": "route-test-model", "messages": [{"role": "user", "content": "hello"}]}),
        ),
        (
            "/v1/completions",
            json!({"model": "route-test-model", "prompt": "hello"}),
        ),
        (
            "/v1/responses",
            json!({"model": "route-test-model", "input": "hello"}),
        ),
        (
            "/v1/messages",
            json!({
                "model": "route-test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1
            }),
        ),
        ("/completion", json!({"prompt": "hello", "n_predict": 1})),
    ];

    for (path, body) in cases {
        let state = state_with(ModelProvider::chat_unavailable_for_route_tests())
            .with_embedding_model(Some(Arc::new(SideEmbeddingProvider)));
        let (status, body) = post(create_app(state), path, body).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {body}");
        assert_eq!(body["error"]["type"], "not_implemented", "{path}: {body}");
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("/v1/embeddings"), "{path}: {body}");
        assert!(message.contains("--embedding-model"), "{path}: {body}");
    }
}

#[tokio::test]
async fn rerank_only_server_names_the_served_route_and_flag() {
    let state = state_with(ModelProvider::chat_unavailable_for_route_tests())
        .with_rerank_model(Some(Arc::new(SideRerankProvider)));
    let (status, body) = post(
        create_app(state),
        "/v1/chat/completions",
        json!({"model": "route-test-model", "messages": [{"role": "user", "content": "hello"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("/v1/rerank"), "{body}");
    assert!(message.contains("--reranker-model"), "{body}");
}

#[tokio::test]
async fn audio_only_server_names_the_served_routes() {
    let state = state_with(ModelProvider::chat_unavailable_for_route_tests())
        .with_audio_model(Some(Arc::new(SideAudioProvider)));
    let (status, body) = post(
        create_app(state),
        "/v1/chat/completions",
        json!({"model": "route-test-model", "messages": [{"role": "user", "content": "hello"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("/v1/audio/transcriptions"), "{body}");
    assert!(message.contains("/v1/audio/translations"), "{body}");
}

#[tokio::test]
async fn exited_chat_worker_returns_503_without_channel_details() {
    let state = state_with(ModelProvider::exited_chat_worker_for_route_tests());
    let (status, body) = post(
        create_app(state),
        "/v1/chat/completions",
        json!({"model": "route-test-model", "messages": [{"role": "user", "content": "hello"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["type"], "server_error");
    assert!(!body.to_string().contains("closed channel"), "{body}");
    assert!(
        body.to_string().contains("chat worker has exited"),
        "{body}"
    );
}
