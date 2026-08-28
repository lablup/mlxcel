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

//! `POST /infill` route tests (#1442).
//!
//! The native completion response echoes the prompt it was served, so the
//! assembled FIM prompt is observable from the HTTP body without a model.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

/// A vocabulary carrying the CodeLlama-style FIM triple and nothing optional.
fn fim_tokenizer() -> MlxcelTokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<PRE>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<SUF>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "<MID>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {"<PRE>": 0, "<SUF>": 1, "<MID>": 2, "a": 10, "b": 11},
            "merges": []
        }
    }"#;
    MlxcelTokenizer::HuggingFace(
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("stub tokenizer builds"),
    )
}

fn app_with(tokenizer: MlxcelTokenizer, spm_infill: bool) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let config = ServerConfig {
        spm_infill,
        ..ServerConfig::default()
    };
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        tokenizer,
        PathBuf::from("infill-route-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn post(app: Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/infill")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

#[tokio::test]
async fn a_model_without_fim_tokens_is_refused_with_the_upstream_message() {
    let (status, body) = post(
        app_with(MlxcelTokenizer::stub(), false),
        serde_json::json!({"input_prefix": "a", "input_suffix": "b"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(
        body["error"]["message"],
        "Infill is not supported by this model: prefix token is missing. suffix token is \
         missing. middle token is missing. "
    );
}

#[tokio::test]
async fn the_capability_gate_runs_before_request_validation() {
    // An incapable model answers 501 even for a body that would also fail
    // validation, which is the order upstream checks them in.
    let (status, _) = post(
        app_with(MlxcelTokenizer::stub(), false),
        serde_json::json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn a_missing_prefix_is_a_400_on_a_capable_model() {
    let (status, body) = post(app_with(fim_tokenizer(), false), serde_json::json!({})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["message"], "\"input_prefix\" is required");

    let (status, body) = post(
        app_with(fim_tokenizer(), false),
        serde_json::json!({"input_prefix": "a"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["message"], "\"input_suffix\" is required");
}

#[tokio::test]
async fn a_capable_model_answers_the_native_completion_shape() {
    let (status, body) = post(
        app_with(fim_tokenizer(), false),
        serde_json::json!({"input_prefix": "a", "input_suffix": "b", "n_predict": 4}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("content").is_some(), "{body}");
    assert!(body.get("tokens_predicted").is_some(), "{body}");
    assert!(body.get("timings").is_some(), "{body}");
}

#[tokio::test]
async fn the_served_prompt_carries_the_default_prefix_suffix_middle_ordering() {
    let (_, body) = post(
        app_with(fim_tokenizer(), false),
        serde_json::json!({"input_prefix": "a", "input_suffix": "b", "n_predict": 4}),
    )
    .await;
    assert_eq!(body["prompt"], "<PRE>a<SUF>b<MID>");
}

#[tokio::test]
async fn spm_infill_changes_the_prompt_the_server_actually_sends() {
    let (_, body) = post(
        app_with(fim_tokenizer(), true),
        serde_json::json!({"input_prefix": "a", "input_suffix": "b", "n_predict": 4}),
    )
    .await;
    assert_eq!(
        body["prompt"], "<SUF>b<PRE>a<MID>",
        "--spm-infill must reach the served prompt, not just the config"
    );
}

#[tokio::test]
async fn the_completion_half_of_the_body_is_still_honored() {
    // `n_predict` is a `/completion` field that survives the FIM rewrite, so a
    // request cannot silently lose its generation settings on this route.
    let (_, body) = post(
        app_with(fim_tokenizer(), false),
        serde_json::json!({
            "input_prefix": "a",
            "input_suffix": "b",
            "n_predict": 7,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(body["generation_settings"]["n_predict"], 7);
    assert_eq!(body["generation_settings"]["temperature"], 0.0);
}

#[tokio::test]
async fn a_marker_in_the_prefix_is_refused_with_an_actionable_diagnostic() {
    let (status, body) = post(
        app_with(fim_tokenizer(), false),
        serde_json::json!({"input_prefix": "a<MID>", "input_suffix": "b"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().expect("a message");
    assert!(message.contains("<MID>"), "{message}");
    assert!(message.contains("POST /completion"), "{message}");
}

#[tokio::test]
async fn the_capability_refusal_carries_upstreams_error_type() {
    // b10621 answers the FIM capability failure with ERROR_TYPE_NOT_SUPPORTED,
    // whose wire spelling is `not_supported_error`, not mlxcel's own
    // `not_implemented`.
    let (status, body) = post(
        app_with(MlxcelTokenizer::stub(), false),
        serde_json::json!({"input_prefix": "a", "input_suffix": "b"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["type"], "not_supported_error");
}
