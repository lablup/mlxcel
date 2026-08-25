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
use crate::rerank::RerankerKind;
use crate::rerank::stub::stub_loaded_reranker;
use crate::server::rerank_model::{RerankError, RerankModelProvider};
use crate::server::rerank_worker::RerankWorkerProvider;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const STUB_MODEL_ID: &str = "stub-reranker";

/// A 1x1 PNG as a data URI, so an image item can reach the route without a
/// file or a network fetch.
const TINY_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

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
    let (options_tx, _options_rx) = mpsc::channel();
    let model_provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = model_provider.batch_metrics().clone();
    let state = AppState::new(
        model_provider,
        ServerConfig::default(),
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
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("`documents` must not be empty"),
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
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("invalid request body"),
        "{body}"
    );
}

#[tokio::test]
async fn the_unversioned_alias_is_mounted() {
    let app = app_with(Some(text_provider()));
    let (status, body) = post(
        app,
        "/rerank",
        json!({"query": "alpha", "documents": ["alpha"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(indices(&body), vec![0]);
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
