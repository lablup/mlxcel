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

//! #3265: a per-request image geometry field (`min_pixels`, `max_pixels`,
//! `resize_shape`, `resized_height`, `resized_width`) on a request that carries
//! an image must be refused with HTTP 400 naming the field on every route that
//! prepares images, not silently dropped.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::server::state::ModelMediaSupport;
use crate::tokenizer::MlxcelTokenizer;

const IMAGE_DATA_URL: &str = "data:image/png;base64,aGVsbG8=";

/// A recording app whose model declares image support, so requests get past
/// the modality gate to the per-request image checks.
fn image_app() -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_media_support(ModelMediaSupport {
        image: true,
        ..ModelMediaSupport::default()
    });
    create_app(state)
}

async fn post(path: &str, body: Value) -> (StatusCode, String) {
    let response = image_app()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("route test request"),
        )
        .await
        .expect("route test response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("route test body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn with_field(mut body: Value, field: Option<(&str, Value)>) -> Value {
    if let Some((key, value)) = field {
        body.as_object_mut().unwrap().insert(key.to_string(), value);
    }
    body
}

fn chat_body(field: Option<(&str, Value)>) -> Value {
    with_field(
        json!({
            "model": "route-test-model",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_DATA_URL}},
                {"type": "text", "text": "Describe."}
            ]}]
        }),
        field,
    )
}

fn responses_body(field: Option<(&str, Value)>) -> Value {
    with_field(
        json!({
            "model": "route-test-model",
            "max_output_tokens": 1,
            "store": false,
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": IMAGE_DATA_URL},
                {"type": "input_text", "text": "Describe."}
            ]}]
        }),
        field,
    )
}

fn messages_body(field: Option<(&str, Value)>) -> Value {
    with_field(
        json!({
            "model": "route-test-model",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                {"type": "text", "text": "Describe."}
            ]}]
        }),
        field,
    )
}

async fn assert_refused(path: &str, body: Value, field: &str) {
    let (status, text) = post(path, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {text}");
    assert!(
        text.contains(field),
        "{path}: error must name {field}: {text}"
    );
}

/// The control: the same requests without the field are served, so the 400s
/// below come from the field and not from the harness.
#[tokio::test]
async fn image_requests_without_geometry_fields_are_served() {
    for (path, body) in [
        ("/v1/chat/completions", chat_body(None)),
        ("/v1/responses", responses_body(None)),
        ("/v1/messages", messages_body(None)),
    ] {
        let (status, text) = post(path, body).await;
        assert_eq!(status, StatusCode::OK, "{path}: {text}");
    }
}

#[tokio::test]
async fn chat_completions_refuses_image_geometry_fields() {
    let min_pixels = || Some(("min_pixels", json!(200704)));
    assert_refused(
        "/v1/chat/completions",
        chat_body(min_pixels()),
        "min_pixels",
    )
    .await;
    assert_refused(
        "/v1/chat/completions",
        chat_body(Some(("extra_body", json!({"min_pixels": 200704})))),
        "min_pixels",
    )
    .await;
}

#[tokio::test]
async fn responses_refuses_image_geometry_fields() {
    assert_refused(
        "/v1/responses",
        responses_body(Some(("min_pixels", json!(200704)))),
        "min_pixels",
    )
    .await;
    assert_refused(
        "/v1/responses",
        responses_body(Some(("extra_body", json!({"min_pixels": 200704})))),
        "min_pixels",
    )
    .await;
}

#[tokio::test]
async fn messages_refuses_image_geometry_fields() {
    assert_refused(
        "/v1/messages",
        messages_body(Some(("min_pixels", json!(200704)))),
        "min_pixels",
    )
    .await;
    assert_refused(
        "/v1/messages",
        messages_body(Some(("extra_body", json!({"min_pixels": 200704})))),
        "min_pixels",
    )
    .await;
}
