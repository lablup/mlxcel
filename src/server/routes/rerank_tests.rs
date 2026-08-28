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

//! `/v1/rerank` route tests over the stub reranker, driven through the real
//! router so the request flow (JSON body, validation, worker dispatch,
//! response shape) is exercised end to end.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{NO_RERANKER_MODEL_MESSAGE, rerank_error_response};
use crate::rerank::stub::stub_loaded_reranker;
use crate::rerank::{RerankItem, RerankScores, RerankerKind};
use crate::server::rerank_model::{RerankError, RerankModelProvider};
use crate::server::rerank_worker::RerankWorkerProvider;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const STUB_MODEL_ID: &str = "stub-reranker";

/// A 1x1 PNG as a data URI, so an image item can reach the route without a
/// file or a network fetch.
const TINY_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

struct NonFiniteRerankProvider;

impl RerankModelProvider for NonFiniteRerankProvider {
    fn rerank(
        &self,
        _query: RerankItem,
        documents: Vec<RerankItem>,
        _instruction: Option<String>,
    ) -> Result<RerankScores, RerankError> {
        Ok(RerankScores {
            scores: vec![f32::NAN; documents.len()],
            prompt_tokens: 1,
        })
    }

    fn model_id(&self) -> &str {
        "non-finite-reranker"
    }

    fn created_at(&self) -> i64 {
        0
    }

    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn max_length(&self) -> usize {
        32
    }
}

fn stub_provider(kind: RerankerKind, supports_images: bool) -> Arc<dyn RerankModelProvider> {
    Arc::new(
        RerankWorkerProvider::from_loader(
            STUB_MODEL_ID.to_string(),
            8,
            Duration::from_secs(30),
            move || Ok(stub_loaded_reranker(kind, supports_images)),
        )
        .expect("stub rerank worker spawns"),
    )
}

fn app_with(provider: Option<Arc<dyn RerankModelProvider>>) -> axum::Router {
    app_with_config(provider, ServerConfig::default())
}

fn app_with_config(
    provider: Option<Arc<dyn RerankModelProvider>>,
    config: ServerConfig,
) -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let model_provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = model_provider.batch_metrics().clone();
    let state = AppState::new(
        model_provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_rerank_model(provider);
    create_app(state)
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
    let value: Value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body")
    };
    (status, value)
}

/// The `index` of every result, in response order.
fn indices(body: &Value) -> Vec<usize> {
    body["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|entry| entry["index"].as_u64().expect("index") as usize)
        .collect()
}

fn text_provider() -> Arc<dyn RerankModelProvider> {
    stub_provider(RerankerKind::GenerativeText, false)
}

#[tokio::test]
async fn results_sorted_desc_ties_by_index() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({
            "query": "alpha beta",
            "documents": ["gamma", "alpha", "beta", "alpha beta"],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], STUB_MODEL_ID);
    // 1.0 for the full overlap, 0.5 for each half, 0.0 for the miss; the two
    // ties must come back in ascending index order.
    assert_eq!(indices(&body), vec![3, 1, 2, 0]);
    let scores: Vec<f64> = body["results"]
        .as_array()
        .expect("results")
        .iter()
        .map(|entry| entry["relevance_score"].as_f64().expect("score"))
        .collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "scores must be non-increasing, got {scores:?}"
    );
    assert!(
        body["results"][0]["document"].is_null(),
        "no echo by default"
    );
    assert_eq!(
        body["usage"]["prompt_tokens"],
        body["usage"]["total_tokens"]
    );
    assert!(body["usage"]["prompt_tokens"].as_u64().expect("tokens") > 0);
}

#[tokio::test]
async fn top_n_truncates() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({
            "query": "alpha beta",
            "documents": ["gamma", "alpha", "beta", "alpha beta"],
            "top_n": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(indices(&body), vec![3, 1]);
}

#[tokio::test]
async fn return_documents_echoes_items() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({
            "query": "alpha",
            "documents": ["alpha", {"text": "beta"}],
            "return_documents": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(indices(&body), vec![0, 1]);
    assert_eq!(
        body["results"][0]["document"], "alpha",
        "the string form is echoed as a string"
    );
    assert_eq!(
        body["results"][1]["document"],
        json!({"text": "beta"}),
        "the object form is echoed as an object"
    );
}

#[tokio::test]
async fn empty_documents_is_400() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"query": "alpha", "documents": []}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    // b10621's own wording since #1452; the message used to be mlxcel's.
    assert_eq!(
        body["error"]["message"], "\"documents\" must be a non-empty string array",
        "{body}"
    );
}

#[tokio::test]
async fn blank_query_and_blank_document_are_400() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app.clone(),
        "/v1/rerank",
        json!({"query": "   ", "documents": ["alpha"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("`query`"),
        "{body}"
    );

    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"query": "alpha", "documents": ["alpha", ""]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("documents[1]"),
        "{body}"
    );
}

#[tokio::test]
async fn top_n_zero_is_400() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"query": "alpha", "documents": ["alpha"], "top_n": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("`top_n` must be at least 1"),
        "{body}"
    );
}

#[tokio::test]
async fn image_item_on_text_reranker_is_400() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({
            "query": "alpha",
            "documents": [{"image": TINY_PNG}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("does not accept images"),
        "{body}"
    );
}

#[tokio::test]
async fn image_documents_reach_a_multimodal_reranker() {
    let app = app_with(Some(stub_provider(RerankerKind::GenerativeVl, true)));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({
            "query": "alpha",
            "documents": [
                {"image": TINY_PNG},
                {"image_url": {"url": TINY_PNG}},
                "alpha",
            ],
            "return_documents": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The text document matches the query exactly (1.0) and the two image
    // documents take the stub's fixed 0.5, tie-broken by index.
    assert_eq!(indices(&body), vec![2, 0, 1]);
    assert_eq!(
        body["results"][1]["document"],
        json!({"image": TINY_PNG}),
        "the image item is echoed verbatim"
    );
}

#[tokio::test]
async fn instruction_on_sequence_classifier_is_400() {
    let app = app_with(Some(stub_provider(RerankerKind::SequenceClassifier, false)));
    let (status, body) = post(
        app.clone(),
        "/v1/rerank",
        json!({
            "query": "alpha",
            "documents": ["alpha"],
            "instruction": "Find relevant passages",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("only supported by the generative rerankers"),
        "{body}"
    );

    // A blank instruction is not an instruction and must not be rejected.
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"query": "alpha", "documents": ["alpha"], "instruction": "  "}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn instruction_is_accepted_by_a_generative_reranker() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({
            "query": "alpha",
            "documents": ["alpha"],
            "instruction": "Find relevant passages",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn no_reranker_is_501() {
    let app = app_with(None);
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"query": "alpha", "documents": ["alpha"]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["message"], NO_RERANKER_MODEL_MESSAGE);
    assert_eq!(body["error"]["type"], "not_implemented");
}

#[tokio::test]
async fn model_mismatch_is_400() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"model": "other", "documents": ["alpha"], "query": "alpha"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("is not the served reranker"),
        "{body}"
    );
}

#[tokio::test]
async fn malformed_body_is_400() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(app, "/v1/rerank", json!({"documents": ["alpha"]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    // A missing `query` is one of the three shapes b10621 names itself, so the
    // body carries upstream's sentence rather than serde's (#1452).
    assert_eq!(
        body["error"]["message"], "\"query\" must be provided",
        "{body}"
    );
}

#[tokio::test]
async fn a_body_that_is_not_an_object_is_still_a_serde_400() {
    // The b10621-worded checks only cover the three shapes upstream names; a
    // body that is not an object at all falls through to serde, which is where
    // mlxcel's own richer item forms are validated too.
    let app = app_with(Some(text_provider()));
    let (status, body) = post(app, "/v1/rerank", json!([1, 2, 3])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("invalid request body"),
        "{body}"
    );
}

#[tokio::test]
async fn all_rerank_aliases_are_mounted() {
    for path in ["/rerank", "/reranking", "/v1/rerank", "/v1/reranking"] {
        // No worker is needed to prove route registration. The existing
        // handler's structured 501 distinguishes a mounted alias from Axum's
        // 404 while keeping this test hardware-independent.
        let app = app_with(None);
        let (status, body) =
            post(app, path, json!({"query": "alpha", "documents": ["alpha"]})).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {body}");
        assert!(body["error"]["message"].is_string(), "{path}: {body}");
    }
}

#[tokio::test]
async fn v1_health_alias_is_public_when_api_key_auth_is_enabled() {
    let app = app_with_config(
        None,
        ServerConfig {
            api_keys: crate::server::resolve_api_keys(&["secret".to_string()], &[])
                .expect("valid key set"),
            ..Default::default()
        },
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/health")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("route responds");

    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    assert_ne!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_reranker_is_listed_in_v1_models() {
    let app = app_with(Some(text_provider()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("route responds");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let body: Value = serde_json::from_slice(&bytes).expect("json body");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|entry| entry["id"].as_str().expect("id"))
        .collect();
    assert!(ids.contains(&STUB_MODEL_ID), "{ids:?}");
}

#[test]
fn provider_errors_map_to_the_shared_status_codes() {
    assert_eq!(
        rerank_error_response(RerankError::QueueFull).status,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        rerank_error_response(RerankError::Timeout).status,
        StatusCode::GATEWAY_TIMEOUT
    );
    assert_eq!(
        rerank_error_response(RerankError::InvalidInput("bad".into())).status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        rerank_error_response(RerankError::Internal("boom".into())).status,
        StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn too_many_images_are_rejected_before_resolution() {
    let documents: Vec<Value> = (0..17).map(|_| json!({"image": TINY_PNG})).collect();
    let app = app_with(Some(stub_provider(RerankerKind::GenerativeVl, true)));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"query": "alpha", "documents": documents}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("Too many image inputs"),
        "{body}"
    );
}

#[tokio::test]
async fn non_finite_scores_return_500() {
    let app = app_with(Some(Arc::new(NonFiniteRerankProvider)));
    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({"query": "alpha", "documents": ["alpha"]}),
    )
    .await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert_eq!(body["error"]["type"], "server_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("invalid numeric result"),
        "{body}"
    );
}
