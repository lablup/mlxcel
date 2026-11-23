// Copyright 2025-2026 Lablup Inc.
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

//! The per-request document cap on `/v1/rerank` (issue #3296).

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{DEFAULT_MAX_RERANK_DOCUMENTS, parse_max_documents};
use crate::rerank::{RerankItem, RerankScores, RerankerKind};
use crate::server::rerank_model::{RerankError, RerankModelProvider};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

/// Scores every document 0.5 and counts the calls that reach it.
struct CountingProvider {
    calls: AtomicUsize,
}

impl RerankModelProvider for CountingProvider {
    fn rerank(
        &self,
        _query: RerankItem,
        documents: Vec<RerankItem>,
        _instruction: Option<String>,
    ) -> Result<RerankScores, RerankError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(RerankScores {
            scores: vec![0.5; documents.len()],
            prompt_tokens: 1,
        })
    }

    fn model_id(&self) -> &str {
        "counting-reranker"
    }

    fn created_at(&self) -> i64 {
        0
    }

    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeVl
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn max_length(&self) -> usize {
        512
    }

    fn shutdown_and_wait(&self, _timeout: std::time::Duration) -> anyhow::Result<()> {
        Ok(())
    }
}

fn app_with(provider: Arc<CountingProvider>) -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let model_provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = model_provider.batch_metrics().clone();
    let provider: Arc<dyn RerankModelProvider> = provider;
    let state = AppState::new(
        model_provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_rerank_model(Some(provider));
    create_app(state)
}

async fn post(app: axum::Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/rerank")
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
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn text_documents(count: usize) -> Vec<Value> {
    (0..count)
        .map(|index| json!(format!("doc {index}")))
        .collect()
}

fn counting_provider() -> Arc<CountingProvider> {
    Arc::new(CountingProvider {
        calls: AtomicUsize::new(0),
    })
}

#[tokio::test]
async fn documents_over_the_cap_are_a_400() {
    let provider = counting_provider();
    let documents = text_documents(DEFAULT_MAX_RERANK_DOCUMENTS + 1);
    let (status, body) = post(
        app_with(Arc::clone(&provider)),
        json!({"query": "alpha", "documents": documents}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["message"],
        "`documents` holds 1025 entries; at most 1024 are allowed per request (set by \
         MLXCEL_RERANK_MAX_DOCUMENTS)"
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn documents_at_the_cap_are_scored() {
    let provider = counting_provider();
    let documents = text_documents(DEFAULT_MAX_RERANK_DOCUMENTS);
    let (status, body) = post(
        app_with(Arc::clone(&provider)),
        json!({"query": "alpha", "documents": documents, "top_n": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn the_cap_is_checked_before_any_image_fetch() {
    // The one image document points at this listener, so a fetch shows up
    // as a connection.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("address");
    let connections = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&connections);
    let accept = tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            drop(socket);
        }
    });
    let mut documents = vec![json!({"image": format!("http://{address}/doc.png")})];
    documents.extend(text_documents(DEFAULT_MAX_RERANK_DOCUMENTS));
    let (status, body) = post(
        app_with(counting_provider()),
        json!({"query": "alpha", "documents": documents}),
    )
    .await;
    accept.abort();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        connections.load(Ordering::SeqCst),
        0,
        "an image was fetched"
    );
}

#[test]
fn the_env_override_must_be_a_positive_integer() {
    assert_eq!(parse_max_documents(None), DEFAULT_MAX_RERANK_DOCUMENTS);
    assert_eq!(parse_max_documents(Some("16")), 16);
    assert_eq!(parse_max_documents(Some(" 4096 ")), 4096);
    for invalid in ["0", "-1", "lots", ""] {
        assert_eq!(
            parse_max_documents(Some(invalid)),
            DEFAULT_MAX_RERANK_DOCUMENTS,
            "{invalid:?}"
        );
    }
}
