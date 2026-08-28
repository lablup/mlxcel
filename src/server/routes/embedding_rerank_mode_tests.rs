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

//! b10621 embedding and reranking mode, envelope and capability tests (#1452).
//!
//! Three things a client can observe and could not before: the rerank envelope
//! (`object`, and b10621's TEI array for a `texts` request), the resolved
//! capability block on `/props` and `/v1/models`, and the generation refusal a
//! `--embeddings` server answers.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::rerank::RerankerKind;
use crate::rerank::stub::stub_loaded_reranker;
use crate::server::config::EmbeddingServingMode;
use crate::server::embedding_model::EmbeddingModelProvider;
use crate::server::embedding_worker::EmbeddingWorkerProvider;
use crate::server::rerank_model::RerankModelProvider;
use crate::server::rerank_worker::RerankWorkerProvider;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const EMBEDDING_ID: &str = "stub-embedder";
const RERANK_ID: &str = "stub-reranker";

fn embedding_provider() -> Arc<dyn EmbeddingModelProvider> {
    Arc::new(
        EmbeddingWorkerProvider::from_loader(
            EMBEDDING_ID.to_string(),
            4,
            8,
            Duration::from_secs(30),
            || Ok(crate::embeddings::stub::stub_loaded_model(false)),
        )
        .expect("stub embedding worker spawns"),
    )
}

fn rerank_provider() -> Arc<dyn RerankModelProvider> {
    Arc::new(
        RerankWorkerProvider::from_loader(
            RERANK_ID.to_string(),
            8,
            Duration::from_secs(30),
            || {
                Ok(stub_loaded_reranker(
                    RerankerKind::SequenceClassifier,
                    false,
                ))
            },
        )
        .expect("stub rerank worker spawns"),
    )
}

/// A server with the given side models and serving mode.
fn app(embedding: bool, rerank: bool, mode: EmbeddingServingMode) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let model_provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = model_provider.batch_metrics().clone();
    let config = ServerConfig {
        embedding_serving_mode: mode,
        enable_props_endpoint: true,
        ..ServerConfig::default()
    };
    let state = AppState::new(
        model_provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_embedding_model(embedding.then(embedding_provider))
    .with_rerank_model(rerank.then(rerank_provider));
    create_app(state)
}

async fn send(app: Router, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let builder = Request::builder().method(method).uri(path);
    let request = match body {
        Some(value) => builder
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(value.to_string())),
        None => builder.body(Body::empty()),
    }
    .expect("request builds");
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn post(app: Router, path: &str, body: Value) -> (StatusCode, Value) {
    send(app, Method::POST, path, Some(body)).await
}

async fn get(app: Router, path: &str) -> (StatusCode, Value) {
    send(app, Method::GET, path, None).await
}

fn cohere_body() -> Value {
    json!({"query": "q", "documents": ["a", "b"]})
}

// ── rerank envelope ─────────────────────────────────────────────────────────

#[tokio::test]
async fn the_jina_envelope_carries_the_object_key() {
    // b10621 emits `"object": "list"`; mlxcel omitted it, which was the
    // recorded divergence on all four rerank routes.
    for path in ["/rerank", "/reranking", "/v1/rerank", "/v1/reranking"] {
        let (status, body) = post(
            app(false, true, EmbeddingServingMode::Any),
            path,
            cohere_body(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{path}: {body}");
        assert_eq!(body["object"], "list", "{path}: {body}");
        assert!(body["results"].is_array(), "{path}: {body}");
        assert!(body["usage"]["prompt_tokens"].as_u64().is_some(), "{path}");
        assert!(body["results"][0]["relevance_score"].is_number(), "{path}");
    }
}

#[tokio::test]
async fn a_texts_request_gets_b10621s_tei_array() {
    // The document-list spelling decides the response TYPE, not just a key
    // name: `texts` answers a bare array of {index, score}.
    let (status, body) = post(
        app(false, true, EmbeddingServingMode::Any),
        "/rerank",
        json!({"query": "q", "texts": ["a", "b"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body.as_array().expect("a bare array");
    assert_eq!(results.len(), 2, "{body}");
    assert!(results[0]["score"].is_number(), "{body}");
    assert!(
        results[0].get("relevance_score").is_none(),
        "the TEI shape uses `score`, not `relevance_score`: {body}"
    );
    assert!(
        results[0].get("text").is_none(),
        "no echo without return_text"
    );
}

#[tokio::test]
async fn return_text_echoes_under_the_tei_spelling() {
    let (_, body) = post(
        app(false, true, EmbeddingServingMode::Any),
        "/rerank",
        json!({"query": "q", "texts": ["a", "b"], "return_text": true}),
    )
    .await;
    let results = body.as_array().expect("a bare array");
    assert!(results.iter().all(|r| r["text"].is_string()), "{body}");
}

#[tokio::test]
async fn return_documents_still_echoes_under_the_cohere_spelling() {
    // The Cohere surface mlxcel already served must keep working; the TEI
    // support is additive.
    let (_, body) = post(
        app(false, true, EmbeddingServingMode::Any),
        "/v1/rerank",
        json!({"query": "q", "documents": ["a", "b"], "return_documents": true}),
    )
    .await;
    assert_eq!(body["object"], "list");
    assert!(
        body["results"]
            .as_array()
            .expect("results")
            .iter()
            .all(|r| r["document"].is_string()),
        "{body}"
    );
}

#[tokio::test]
async fn a_body_carrying_both_spellings_stays_on_the_jina_envelope() {
    // `documents` present is the Jina form even when `texts` is there too, so
    // a client that sends both is not silently switched to the other shape.
    let (_, body) = post(
        app(false, true, EmbeddingServingMode::Any),
        "/rerank",
        json!({"query": "q", "documents": ["a"], "texts": ["b"]}),
    )
    .await;
    assert_eq!(body["object"], "list", "{body}");
}

#[tokio::test]
async fn the_rerank_501_opens_with_the_upstream_sentence() {
    let (status, body) = post(
        app(false, false, EmbeddingServingMode::Any),
        "/rerank",
        cohere_body(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    let message = body["error"]["message"].as_str().expect("a message");
    assert!(
        message.starts_with("This server does not support reranking. Start it with `--reranking`"),
        "{message}"
    );
}

#[tokio::test]
async fn the_embedding_501_opens_with_the_upstream_sentence() {
    for path in ["/embedding", "/embeddings", "/v1/embeddings"] {
        let (status, body) = post(
            app(false, false, EmbeddingServingMode::Any),
            path,
            json!({"input": "hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {body}");
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(
            message.starts_with(
                "This server does not support embeddings. Start it with `--embeddings`"
            ),
            "{path}: {message}"
        );
    }
}

// ── serving mode ────────────────────────────────────────────────────────────

#[tokio::test]
async fn an_embedding_only_server_refuses_generation_and_names_the_flag() {
    // The chat worker is loaded and healthy in this harness, so the refusal
    // can only come from the mode flag.
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model": "route-test-model", "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (
            "/v1/completions",
            json!({"model": "route-test-model", "prompt": "hi"}),
        ),
        ("/completion", json!({"prompt": "hi"})),
    ] {
        let (status, response) = post(
            app(true, false, EmbeddingServingMode::EmbeddingOnly),
            path,
            body,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {response}");
        let message = response["error"]["message"].as_str().expect("a message");
        assert!(message.contains("--embeddings"), "{path}: {message}");
        assert!(
            message.contains("generation is disabled"),
            "{path}: {message}"
        );
    }
}

#[tokio::test]
async fn a_rerank_only_server_names_its_own_flag() {
    let (status, body) = post(
        app(false, true, EmbeddingServingMode::RerankOnly),
        "/completion",
        json!({"prompt": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    let message = body["error"]["message"].as_str().expect("a message");
    assert!(message.contains("--reranking"), "{message}");
    assert!(message.contains("reranking routes"), "{message}");
}

#[tokio::test]
async fn the_side_model_routes_still_serve_in_a_restricted_mode() {
    // Restricting generation must not restrict the route the mode exists for.
    let (status, body) = post(
        app(false, true, EmbeddingServingMode::RerankOnly),
        "/rerank",
        cohere_body(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn an_unrestricted_server_still_generates() {
    let (status, body) = post(
        app(true, true, EmbeddingServingMode::Any),
        "/completion",
        json!({"prompt": "hi", "n_predict": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

// ── capability reporting ────────────────────────────────────────────────────

#[tokio::test]
async fn props_reports_the_resolved_side_model_capability() {
    let (status, body) = get(app(true, true, EmbeddingServingMode::Any), "/props").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let caps = &body["capabilities"];
    assert_eq!(caps["generation"], true, "{body}");
    assert!(caps.get("serving_mode").is_none(), "{body}");
    assert_eq!(caps["embedding"]["model"], EMBEDDING_ID, "{body}");
    assert!(caps["embedding"]["dim"].as_u64().is_some(), "{body}");
    assert!(caps["embedding"]["pooling"].is_string(), "{body}");
    assert!(caps["embedding"]["embd_normalize"].is_number(), "{body}");
    assert_eq!(caps["reranking"]["model"], RERANK_ID, "{body}");
    assert_eq!(caps["reranking"]["kind"], "sequence_classifier", "{body}");
}

#[tokio::test]
async fn props_reports_a_restricted_server_as_restricted() {
    let (_, body) = get(
        app(true, false, EmbeddingServingMode::EmbeddingOnly),
        "/props",
    )
    .await;
    let caps = &body["capabilities"];
    assert_eq!(caps["generation"], false, "{body}");
    assert_eq!(caps["serving_mode"], "--embeddings", "{body}");
    assert!(caps.get("reranking").is_none(), "{body}");
}

#[tokio::test]
async fn props_omits_the_side_model_blocks_when_none_is_loaded() {
    let (_, body) = get(app(false, false, EmbeddingServingMode::Any), "/props").await;
    let caps = &body["capabilities"];
    assert!(caps.get("embedding").is_none(), "{body}");
    assert!(caps.get("reranking").is_none(), "{body}");
    assert_eq!(caps["generation"], true, "{body}");
}

#[tokio::test]
async fn models_labels_each_entry_with_what_it_can_be_asked_for() {
    let (status, body) = get(app(true, true, EmbeddingServingMode::Any), "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entries = body["data"].as_array().expect("data array");
    let by_id = |id: &str| {
        entries
            .iter()
            .find(|e| e["id"] == id)
            .unwrap_or_else(|| panic!("{id} is listed: {body}"))
            .clone()
    };
    assert_eq!(
        by_id("route-test-model")["capabilities"],
        json!(["completion"])
    );
    assert_eq!(by_id(EMBEDDING_ID)["capabilities"], json!(["embedding"]));
    assert_eq!(by_id(RERANK_ID)["capabilities"], json!(["rerank"]));
}

#[tokio::test]
async fn a_restricted_server_does_not_advertise_completion() {
    let (_, body) = get(
        app(true, false, EmbeddingServingMode::EmbeddingOnly),
        "/v1/models",
    )
    .await;
    let primary = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .find(|e| e["id"] == "route-test-model")
        .expect("primary entry")
        .clone();
    assert!(
        primary.get("capabilities").is_none() || primary["capabilities"] == json!([]),
        "generation is off, so the primary id must not advertise completion: {body}"
    );
}

// ── upstream-worded shape errors (#1452) ────────────────────────────────────

#[tokio::test]
async fn a_rerank_body_missing_its_query_uses_the_upstream_wording() {
    for (body, expected) in [
        (json!({"documents": ["a"]}), "\"query\" must be provided"),
        (
            json!({"query": 7, "documents": ["a"]}),
            "\"query\" must be a string",
        ),
        (
            json!({"query": "q"}),
            "\"documents\" must be a non-empty string array",
        ),
        (
            json!({"query": "q", "documents": []}),
            "\"documents\" must be a non-empty string array",
        ),
        (
            json!({"query": "q", "texts": []}),
            "\"documents\" must be a non-empty string array",
        ),
    ] {
        let (status, response) =
            post(app(false, true, EmbeddingServingMode::Any), "/rerank", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert_eq!(response["error"]["message"], expected, "{response}");
    }
}

#[tokio::test]
async fn an_embedding_body_with_neither_input_nor_content_uses_the_upstream_wording() {
    for path in ["/embedding", "/embeddings", "/v1/embeddings"] {
        let (status, body) = post(
            app(true, false, EmbeddingServingMode::Any),
            path,
            json!({"model": "stub-embedder"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        assert_eq!(
            body["error"]["message"], "\"input\" or \"content\" must be provided",
            "{path}: {body}"
        );
    }
}

#[tokio::test]
async fn the_richer_mlxcel_item_forms_still_pass_the_shape_check() {
    // mlxcel accepts an object query carrying an image, which upstream has no
    // equivalent for; the b10621-worded checks must not refuse it.
    let (status, body) = post(
        app(false, true, EmbeddingServingMode::Any),
        "/rerank",
        json!({"query": {"text": "q"}, "documents": [{"text": "a"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn a_body_carrying_both_document_spellings_is_served_not_refused() {
    // serde reads `texts` as an alias of `documents` and would reject a body
    // carrying both as a duplicate field; upstream reads one and ignores the
    // other, so the redundant key is dropped before deserializing.
    let (status, body) = post(
        app(false, true, EmbeddingServingMode::Any),
        "/rerank",
        json!({"query": "q", "documents": ["a", "b"], "texts": ["c"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["results"].as_array().expect("results").len(),
        2,
        "{body}"
    );
}

// ── readiness in a restricted mode (#1452) ──────────────────────────────────

#[tokio::test]
async fn a_restricted_server_reports_healthy_once_its_worker_is_up() {
    // Without this the mode is unusable behind a container probe: there is no
    // chat worker to be "loaded", so /health would answer 503 for the life of
    // the process.
    for (mode, embedding, rerank) in [
        (EmbeddingServingMode::EmbeddingOnly, true, false),
        (EmbeddingServingMode::RerankOnly, false, true),
    ] {
        let (status, body) = get(app(embedding, rerank, mode), "/health").await;
        assert_eq!(status, StatusCode::OK, "{mode:?}: {body}");
        assert_eq!(body["status"], "ok", "{mode:?}: {body}");
    }
}

#[tokio::test]
async fn an_unrestricted_server_answers_the_b10621_health_body() {
    // #1440 aligned /health with b10621: the ready body is exactly
    // {"status": "ok"} for restricted and unrestricted servers alike; the
    // former rich payload lives on GET /slots and GET /metrics now.
    let (status, body) = get(app(true, true, EmbeddingServingMode::Any), "/health").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, serde_json::json!({ "status": "ok" }), "{body}");
}

#[tokio::test]
async fn props_reports_the_normalization_an_unqualified_request_gets() {
    // Not the checkpoint's own answer: a server started with
    // `--embd-normalize 1` serves L1 to a request that names nothing, so
    // reporting the checkpoint's 2 would describe a server that does not exist.
    let (options_tx, _options_rx) = mpsc::channel();
    let model_provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = model_provider.batch_metrics().clone();
    let config = ServerConfig {
        enable_props_endpoint: true,
        embd_normalize: Some(crate::embeddings::EmbdNormalize::TAXICAB),
        ..ServerConfig::default()
    };
    let state = AppState::new(
        model_provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_embedding_model(Some(embedding_provider()));
    let request = Request::builder()
        .method(Method::GET)
        .uri("/props")
        .body(Body::empty())
        .expect("request builds");
    let response = create_app(state)
        .oneshot(request)
        .await
        .expect("router responds");
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let body: Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(
        body["capabilities"]["embedding"]["embd_normalize"], 1,
        "{body}"
    );
}
