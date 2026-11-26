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

//! Every audio-carrying route refuses an Inkling 2 checkpoint with the one
//! shared message (issue #3376).
//!
//! The app is built with the `ModelMediaSupport` that
//! `detect_model_media_support` returns for an Inkling 2 config, over a
//! provider that would happily generate: before the change each of these
//! requests either reached the worker (a late 500 after the clip was fetched)
//! or, on `/v1/messages`, was silently answered without the audio.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::media::INKLING2_AUDIO_REFUSAL;
use crate::server::state::ModelMediaSupport;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

fn inkling2_support() -> ModelMediaSupport {
    ModelMediaSupport {
        image: true,
        audio: false,
        video_frames_fallback: true,
        audio_refusal: Some(INKLING2_AUDIO_REFUSAL),
        ..ModelMediaSupport::default()
    }
}

fn inkling2_app() -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::media_expanding_for_route_tests(
        options_tx, 10,
    ));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub_all_byte_fallback(),
        PathBuf::from("inkling-2-test-model"),
        batch_metrics,
    )
    .with_media_support(inkling2_support());
    create_app(state)
}

async fn send(request: Request<Body>) -> (StatusCode, serde_json::Value) {
    let response = inkling2_app()
        .oneshot(request)
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

async fn post_json(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    send(
        Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds"),
    )
    .await
}

fn assert_refused(status: StatusCode, body: &serde_json::Value) {
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["type"], "not_supported_error", "{body}");
    assert_eq!(body["error"]["message"], INKLING2_AUDIO_REFUSAL, "{body}");
}

fn input_audio_part() -> serde_json::Value {
    serde_json::json!({"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}})
}

#[tokio::test]
async fn chat_input_audio_is_refused() {
    let (status, body) = post_json(
        "/v1/chat/completions",
        serde_json::json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"}, input_audio_part()]}]}),
    )
    .await;
    assert_refused(status, &body);
}

#[tokio::test]
async fn chat_audio_url_is_refused_not_a_body_shape_error() {
    let (status, body) = post_json(
        "/v1/chat/completions",
        serde_json::json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "audio_url", "audio_url": {"url": "file:///missing.wav"}}]}]}),
    )
    .await;
    assert_refused(status, &body);
}

#[tokio::test]
async fn chat_audio_with_an_image_still_refuses_on_audio() {
    let (status, body) = post_json(
        "/v1/chat/completions",
        serde_json::json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            input_audio_part()]}]}),
    )
    .await;
    assert_refused(status, &body);
}

#[tokio::test]
async fn responses_input_audio_is_refused() {
    let (status, body) = post_json(
        "/v1/responses",
        serde_json::json!({"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "hi"},
            {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}]}]}),
    )
    .await;
    assert_refused(status, &body);
}

#[tokio::test]
async fn messages_audio_block_is_refused_instead_of_dropped() {
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "aGVsbG8="}}]}]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["type"], "not_supported_error", "{body}");
    assert_eq!(body["error"]["message"], INKLING2_AUDIO_REFUSAL, "{body}");
}

#[tokio::test]
async fn count_tokens_audio_block_is_refused() {
    let (status, body) = post_json(
        "/v1/messages/count_tokens",
        serde_json::json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"}, {"type": "audio", "source": {}}]}]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["message"], INKLING2_AUDIO_REFUSAL, "{body}");
}

#[tokio::test]
async fn prompt_inspection_audio_is_refused() {
    let (status, body) = post_json(
        "/apply-template",
        serde_json::json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"}, input_audio_part()]}]}),
    )
    .await;
    assert_refused(status, &body);
}

#[tokio::test]
async fn transcription_names_the_inkling2_reason() {
    const BOUNDARY: &str = "----inkling2AudioBoundary";
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; \
             filename=\"x.wav\"\r\n\r\nhello\r\n--{BOUNDARY}--\r\n"
        )
        .as_bytes(),
    );
    let (status, body) = send(
        Request::builder()
            .method(Method::POST)
            .uri("/v1/audio/transcriptions")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body))
            .expect("request builds"),
    )
    .await;
    assert_refused(status, &body);
}
