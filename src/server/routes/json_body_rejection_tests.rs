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

//! Request bodies that the typed `Json` extractor rejects, on the chat-shaped
//! generation routes.
//!
//! `/v1/completions` answers these with a JSON `invalid_request_error` (see
//! `completion_prompt_tests.rs`). These tests hold `/v1/chat/completions`,
//! `/v1/responses`, `/v1/messages` and `/v1/messages/count_tokens`, and the
//! aliases without `/v1`, to the same contract, each in its own error envelope.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::DefaultBodyLimit;
use axum::http::{Request, StatusCode, header};
use axum::routing::post;
use serde_json::Value;
use tower::ServiceExt;

use crate::server::{
    AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, ServerGenerateOptions, create_app,
};
use crate::tokenizer::MlxcelTokenizer;

const MODEL: &str = "json-rejection-test-model";
const JSON: Option<&str> = Some("application/json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Chat,
    Responses,
    Messages,
    CountTokens,
}

impl Route {
    fn valid_body(self) -> String {
        match self {
            // The same body parses as an OpenAI chat and an Anthropic request.
            Self::Chat | Self::Messages | Self::CountTokens => format!(
                r#"{{"model":"{MODEL}","max_tokens":4,"messages":[{{"role":"user","content":"hi"}}]}}"#
            ),
            Self::Responses => format!(r#"{{"model":"{MODEL}","input":"hi"}}"#),
        }
    }

    fn is_anthropic(self) -> bool {
        matches!(self, Self::Messages | Self::CountTokens)
    }
}

/// Every single-node route whose handler extracts one of the typed chat-shaped
/// request bodies, including the aliases mounted on the same handlers.
const ROUTES: [(&str, Route); 8] = [
    ("/v1/chat/completions", Route::Chat),
    ("/chat/completions", Route::Chat),
    ("/v1/responses", Route::Responses),
    ("/responses", Route::Responses),
    ("/v1/messages", Route::Messages),
    ("/messages", Route::Messages),
    ("/v1/messages/count_tokens", Route::CountTokens),
    ("/messages/count_tokens", Route::CountTokens),
];

struct Case {
    name: &'static str,
    body: Option<&'static str>,
    content_type: Option<&'static str>,
    status: StatusCode,
    /// A schema error names the offending field.
    mentions_model: bool,
}

/// Before this change, axum answered all of these in `text/plain`, with 422
/// for the three schema errors.
const CASES: [Case; 7] = [
    Case {
        name: "syntax error",
        body: Some("{"),
        content_type: JSON,
        status: StatusCode::BAD_REQUEST,
        mentions_model: false,
    },
    Case {
        name: "empty body",
        body: Some(""),
        content_type: JSON,
        status: StatusCode::BAD_REQUEST,
        mentions_model: false,
    },
    Case {
        name: "wrong field type",
        body: Some(r#"{"model":5}"#),
        content_type: JSON,
        status: StatusCode::BAD_REQUEST,
        mentions_model: true,
    },
    Case {
        name: "wrong field type with stream",
        body: Some(r#"{"model":5,"stream":true}"#),
        content_type: JSON,
        status: StatusCode::BAD_REQUEST,
        mentions_model: true,
    },
    Case {
        name: "missing model",
        body: Some("{}"),
        content_type: JSON,
        status: StatusCode::BAD_REQUEST,
        mentions_model: true,
    },
    // `None` sends the route's valid body: only the content type is wrong.
    Case {
        name: "no content type",
        body: None,
        content_type: None,
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        mentions_model: false,
    },
    Case {
        name: "text/plain",
        body: None,
        content_type: Some("text/plain"),
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        mentions_model: false,
    },
];

struct Harness {
    app: Router,
    options: mpsc::Receiver<ServerGenerateOptions>,
    prompts: mpsc::Receiver<String>,
}

impl Harness {
    fn dispatch_count(&self) -> usize {
        let options = self.options.try_iter().count();
        assert_eq!(self.prompts.try_iter().count(), options);
        options
    }
}

fn harness_with(app: impl FnOnce(AppState) -> Router) -> Harness {
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
        PathBuf::from(MODEL),
        metrics,
    );
    Harness {
        app: app(state),
        options,
        prompts,
    }
}

fn harness() -> Harness {
    harness_with(create_app)
}

/// The four handlers on a bare router with a small body limit, so a 413 does
/// not need a body above the production multimodal upload cap.
fn harness_with_body_limit(limit: usize) -> Harness {
    harness_with(|state| {
        Router::new()
            .route("/v1/chat/completions", post(super::chat_completions))
            .route("/v1/responses", post(super::create_response))
            .route("/v1/messages", post(super::anthropic_messages))
            .route(
                "/v1/messages/count_tokens",
                post(super::anthropic_count_tokens),
            )
            .layer(DefaultBodyLimit::max(limit))
            .with_state(state)
    })
}

#[derive(Debug, PartialEq, Eq)]
struct Answer {
    status: StatusCode,
    content_type: String,
    body: String,
}

impl Answer {
    fn json(&self) -> Value {
        assert!(
            self.content_type.starts_with("application/json"),
            "{} {}: {}",
            self.status,
            self.content_type,
            self.body
        );
        serde_json::from_str(&self.body).unwrap()
    }
}

async fn post_raw(app: &Router, path: &str, body: String, content_type: Option<&str>) -> Answer {
    let mut request = Request::post(path);
    if let Some(content_type) = content_type {
        request = request.header(header::CONTENT_TYPE, content_type);
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body)).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|value| value.to_str().unwrap().to_string())
        .unwrap_or_default();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    Answer {
        status,
        content_type,
        body: String::from_utf8(bytes.to_vec()).unwrap(),
    }
}

async fn post_case(app: &Router, path: &str, route: Route, case: &Case) -> Answer {
    let body = case.body.map_or_else(|| route.valid_body(), str::to_string);
    post_raw(app, path, body, case.content_type).await
}

/// Assert the error envelope the route's clients parse: OpenAI's
/// `{"error": {...}}`, or Anthropic's `{"type": "error", "error": {...}}`.
fn assert_error_envelope(route: Route, error: &Value, context: &str) -> String {
    assert_eq!(
        error["error"]["type"], "invalid_request_error",
        "{context}: {error}"
    );
    if route.is_anthropic() {
        assert_eq!(error["type"], "error", "{context}: {error}");
    } else {
        assert!(error.get("type").is_none(), "{context}: {error}");
    }
    let message = error["error"]["message"].as_str().unwrap_or_default();
    assert!(!message.is_empty(), "{context}: {error}");
    message.to_string()
}

#[tokio::test]
async fn json_body_rejection_valid_bodies_reach_each_handler() {
    // Control for the dispatch counts below: a valid body generates once on
    // the generation routes and is counted without generating on count_tokens.
    for (path, route) in ROUTES {
        let h = harness();
        let answer = post_raw(&h.app, path, route.valid_body(), JSON).await;
        assert_eq!(answer.status, StatusCode::OK, "{path}: {}", answer.body);
        let body = answer.json();
        if route == Route::CountTokens {
            assert!(body["input_tokens"].is_u64(), "{path}: {body}");
            assert_eq!(h.dispatch_count(), 0, "{path}");
        } else {
            assert_eq!(h.dispatch_count(), 1, "{path}");
        }
    }
}

#[tokio::test]
async fn json_body_rejection_answers_json_in_each_route_envelope() {
    for (path, route) in ROUTES {
        for case in &CASES {
            let h = harness();
            let context = format!("{path}, {}", case.name);
            let answer = post_case(&h.app, path, route, case).await;
            assert_eq!(answer.status, case.status, "{context}: {}", answer.body);
            // `answer.json()` also asserts the content type, so a streaming
            // body that fails to parse never opens an SSE response.
            let message = assert_error_envelope(route, &answer.json(), &context);
            if case.mentions_model {
                assert!(message.contains("model"), "{context}: {message}");
            }
            assert_eq!(h.dispatch_count(), 0, "{context}: reached the model");
        }
    }
}

#[tokio::test]
async fn json_body_rejection_count_tokens_answers_like_messages() {
    // One body gets one answer from either Anthropic route (issue #2337).
    for (messages, count_tokens) in [
        ("/v1/messages", "/v1/messages/count_tokens"),
        ("/messages", "/messages/count_tokens"),
    ] {
        for case in &CASES {
            let h = harness();
            let expected = post_case(&h.app, messages, Route::Messages, case).await;
            let actual = post_case(&h.app, count_tokens, Route::CountTokens, case).await;
            assert_eq!(actual, expected, "{count_tokens}, {}", case.name);
        }
    }
}

#[tokio::test]
async fn json_body_rejection_keeps_413_for_oversized_bodies() {
    // The handler's own answer. `create_app` also wraps every route in
    // `openai_payload_too_large` (app.rs), which rewrites any 413 to the
    // OpenAI envelope, so on the full server the Anthropic routes still
    // answer 413 in the OpenAI envelope. This change leaves that layer alone.
    for (path, route) in ROUTES
        .into_iter()
        .filter(|(path, _)| path.starts_with("/v1/"))
    {
        let h = harness_with_body_limit(8);
        let answer = post_raw(&h.app, path, route.valid_body(), JSON).await;
        assert_eq!(
            answer.status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{path}: {}",
            answer.body
        );
        assert_error_envelope(route, &answer.json(), path);
        assert_eq!(h.dispatch_count(), 0, "{path}: reached the model");
    }
}

#[tokio::test]
async fn json_body_rejection_precedes_the_chat_availability_check() {
    // Unchanged order: the extractor runs before the handler body, so a body
    // that does not parse is a 400 even on a server that cannot generate. A
    // body that parses still gets the 501 (`availability_tests.rs`).
    for (path, route) in ROUTES {
        let provider = Arc::new(ModelProvider::chat_unavailable_for_route_tests());
        let metrics = provider.batch_metrics().clone();
        let app = create_app(AppState::new(
            provider,
            ServerConfig::default(),
            ChatTemplateProcessor::with_template("ok".to_string()),
            MlxcelTokenizer::stub(),
            PathBuf::from(MODEL),
            metrics,
        ));
        let answer = post_raw(&app, path, "{".to_string(), JSON).await;
        assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{path}");
        assert_error_envelope(route, &answer.json(), path);
    }
}
