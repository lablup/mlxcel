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

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::DefaultBodyLimit;
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::server::{
    AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, ServerGenerateOptions, create_app,
};
use crate::tokenizer::MlxcelTokenizer;

struct Harness {
    app: Router,
    options: mpsc::Receiver<ServerGenerateOptions>,
    prompts: mpsc::Receiver<String>,
}

impl Harness {
    fn assert_dispatched(&self, prompt: Option<&str>) {
        assert_eq!(
            self.options.try_iter().count(),
            usize::from(prompt.is_some())
        );
        assert_eq!(
            self.prompts.try_iter().collect::<Vec<_>>(),
            prompt.into_iter().collect::<Vec<_>>()
        );
    }
}

fn harness() -> Harness {
    harness_with_body_limit(None)
}

fn harness_with_body_limit(limit: Option<usize>) -> Harness {
    let (options_tx, options) = mpsc::channel();
    let (prompt_tx, prompts) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    // Finish even an accidental dispatch of invalid input, so it fails promptly.
    handle.token("ok");
    handle.finish();
    let provider = Arc::new(provider);
    let metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub_all_byte_fallback(),
        PathBuf::from("completion-prompt-test-model"),
        metrics,
    );
    let app = if let Some(limit) = limit {
        // Exercise the real handler with a small limit instead of allocating
        // a body above the production multimodal upload cap.
        Router::new()
            .route("/v1/completions", axum::routing::post(super::completions))
            .layer(DefaultBodyLimit::max(limit))
            .with_state(state)
    } else {
        create_app(state)
    };
    Harness {
        app,
        options,
        prompts,
    }
}

async fn post(app: &Router, prompt: Value, stream: bool) -> (StatusCode, String, String) {
    let body = json!({
        "model": "completion-prompt-test-model",
        "prompt": prompt,
        "max_tokens": 4,
        "stream": stream,
        "stream_options": {"include_usage": true},
    });
    post_raw(app, body.to_string(), Some("application/json")).await
}

async fn post_raw(
    app: &Router,
    body: String,
    content_type: Option<&str>,
) -> (StatusCode, String, String) {
    let mut request = Request::post("/v1/completions");
    if let Some(content_type) = content_type {
        request = request.header(header::CONTENT_TYPE, content_type);
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body)).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let content_type = response.headers()[header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .to_string();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (
        status,
        content_type,
        String::from_utf8(bytes.to_vec()).unwrap(),
    )
}

#[tokio::test]
async fn completion_prompt_string_and_singleton_array_reach_the_same_generation_paths() {
    for text in ["Hello", "  안녕하세요\n\"quoted\" ", ""] {
        for prompt in [json!(text), json!([text])] {
            for stream in [false, true] {
                let h = harness();
                let (status, content_type, body) = post(&h.app, prompt.clone(), stream).await;
                assert_eq!(status, StatusCode::OK, "{prompt}, stream={stream}: {body}");
                let events: Vec<Value> = if stream {
                    assert!(content_type.starts_with("text/event-stream"));
                    assert!(body.contains("data: [DONE]"), "{body}");
                    body.lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter(|data| *data != "[DONE]")
                        .map(|data| serde_json::from_str(data).unwrap())
                        .collect()
                } else {
                    assert!(content_type.starts_with("application/json"));
                    vec![serde_json::from_str(&body).unwrap()]
                };
                let output: String = events
                    .iter()
                    .filter_map(|event| event["choices"][0]["text"].as_str())
                    .collect();
                assert_eq!(output, "ok", "{body}");
                let usage = events.iter().find_map(|event| event.get("usage")).unwrap();
                assert_eq!(usage["prompt_tokens"], 1, "{body}");
                assert_eq!(usage["completion_tokens"], 1, "{body}");
                h.assert_dispatched(Some(text));
            }
        }
    }
}

#[tokio::test]
async fn completion_prompt_arrays_are_rejected_before_generation_when_not_singletons() {
    for (prompt, message) in [
        (json!([]), "empty prompt arrays"),
        (json!(["first", "second"]), "multiple prompts"),
    ] {
        for stream in [false, true] {
            let h = harness();
            let (status, content_type, body) = post(&h.app, prompt.clone(), stream).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{prompt}: {body}");
            assert!(content_type.starts_with("application/json"), "{body}");
            let error: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(error["error"]["type"], "invalid_request_error");
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(message),
                "{body}"
            );
            h.assert_dispatched(None);
        }
    }
}

#[tokio::test]
async fn completion_prompt_singleton_array_preserves_whitespace_only_rejection() {
    for prompt in [json!(" \n\t"), json!([" \n\t"])] {
        for stream in [false, true] {
            let h = harness();
            let (status, content_type, body) = post(&h.app, prompt.clone(), stream).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{prompt}: {body}");
            assert!(content_type.starts_with("application/json"));
            let error: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(error["error"]["type"], "invalid_request_error");
            h.assert_dispatched(None);
        }
    }
}

#[tokio::test]
async fn completion_prompt_unsupported_json_types_never_reach_the_worker() {
    for prompt in [
        json!(null),
        json!(123),
        json!([123]),
        json!([[123]]),
        json!(["hello", 123]),
        json!({"text": "hello"}),
    ] {
        let h = harness();
        let (status, content_type, body) = post(&h.app, prompt, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(content_type.starts_with("application/json"));
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
        h.assert_dispatched(None);
    }
}

#[tokio::test]
async fn completion_prompt_json_rejections_preserve_http_status() {
    let valid = r#"{"model":"m","prompt":"hello"}"#;
    for (body, content_type, limit, expected_status) in [
        ("{", Some("application/json"), None, StatusCode::BAD_REQUEST),
        (
            r#"{"model":"m"}"#,
            Some("application/json"),
            None,
            StatusCode::BAD_REQUEST,
        ),
        (valid, None, None, StatusCode::UNSUPPORTED_MEDIA_TYPE),
        (
            valid,
            Some("text/plain"),
            None,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
        ),
        (
            valid,
            Some("application/json"),
            Some(8),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
    ] {
        let h = harness_with_body_limit(limit);
        let (status, content_type, body) = post_raw(&h.app, body.to_string(), content_type).await;
        assert_eq!(status, expected_status, "{body}");
        assert!(content_type.starts_with("application/json"));
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
        h.assert_dispatched(None);
    }
}
