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

//! `POST /v1/responses/{id}/cancel` through the real router against a
//! scripted model. A stored response is always a finished result, so cancel
//! must refuse it without modifying it; only an in-flight stream can still be
//! cancelled.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig, StoredResponse};
use crate::server::types::responses_response::ResponseStatus;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

struct Harness {
    app: Router,
    store: Arc<ResponsesStore>,
    handle: ScriptedStreamHandle,
    _prompts: mpsc::Receiver<String>,
    _options: mpsc::Receiver<crate::server::ServerGenerateOptions>,
}

fn harness() -> Harness {
    harness_with(ServerConfig::default(), MlxcelTokenizer::stub())
}

fn harness_with(config: ServerConfig, tokenizer: MlxcelTokenizer) -> Harness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let store = Arc::new(ResponsesStore::new(ResponsesStoreConfig::default()));
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        tokenizer,
        PathBuf::from("cancel-test-model"),
        batch_metrics,
    )
    .with_responses_store(Some(store.clone()));
    Harness {
        app: create_app(state),
        store,
        handle,
        _prompts: prompt_rx,
        _options: options_rx,
    }
}

impl Harness {
    async fn send(&self, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
            .expect("request builds");
        let response = self
            .app
            .clone()
            .oneshot(request)
            .await
            .expect("router responds");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body collects");
        (
            status,
            serde_json::from_slice(&bytes).expect("response is JSON"),
        )
    }

    /// Post one streaming request and collect its whole SSE body.
    async fn send_raw_stream(&self, body: Value) -> (StatusCode, String) {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/responses")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds");
        let response = self
            .app
            .clone()
            .oneshot(request)
            .await
            .expect("router responds");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body collects");
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Run one non-streaming generation and return the stored response id.
    async fn create_completed(&self) -> String {
        self.handle.token("Hello.");
        self.handle.finish();
        let (status, body) = self
            .send(
                Method::POST,
                "/v1/responses",
                Some(json!({"model": "cancel-test-model", "input": "hi"})),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "completed", "{body}");
        body["id"].as_str().expect("response id").to_string()
    }

    async fn cancel(&self, id: &str) -> (StatusCode, Value) {
        self.send(Method::POST, &format!("/v1/responses/{id}/cancel"), None)
            .await
    }

    async fn retrieve(&self, id: &str) -> Value {
        let (status, body) = self
            .send(Method::GET, &format!("/v1/responses/{id}"), None)
            .await;
        assert_eq!(status, StatusCode::OK, "{id}: {body}");
        body
    }
}

#[tokio::test]
async fn cancel_refuses_a_finished_stored_response_and_leaves_it_unchanged() {
    let h = harness();
    let completed = h.create_completed().await;

    // An `incomplete` result is just as finished as a `completed` one.
    let incomplete = "resp_incomplete".to_string();
    let mut copy = h.store.get(&completed).expect("response is stored");
    copy.response.id = incomplete.clone();
    copy.response.status = ResponseStatus::Incomplete;
    h.store.insert(
        incomplete.clone(),
        StoredResponse {
            response: copy.response,
            input_items: copy.input_items,
        },
    );

    for id in [&completed, &incomplete] {
        let before = h.retrieve(id).await;
        // A repeated cancel must keep refusing and keep the entry intact.
        for _ in 0..2 {
            let (status, body) = h.cancel(id).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{id}: {body}");
            assert_eq!(
                body["error"]["type"], "invalid_request_error",
                "{id}: {body}"
            );
            assert_eq!(
                h.retrieve(id).await,
                before,
                "{id}: stored response changed"
            );
        }
    }
}

#[tokio::test]
async fn cancel_of_a_stored_response_leaves_its_in_flight_token_alone() {
    // The stream persists its result before it unregisters the in-flight
    // token, so both can exist for the same id at once.
    let h = harness();
    let id = h.create_completed().await;
    let token = h.store.register_in_flight(id.clone());

    let (status, body) = h.cancel(&id).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        !token.load(Ordering::Relaxed),
        "a finished response must not flip the in-flight token"
    );
}

#[tokio::test]
async fn cancel_of_an_in_flight_stream_still_flips_its_token() {
    let h = harness();
    let token = h.store.register_in_flight("resp_live".to_string());

    let (status, body) = h.cancel("resp_live").await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body,
        json!({"id": "resp_live", "object": "response", "status": "cancelled"})
    );
    assert!(
        token.load(Ordering::Relaxed),
        "the stream must be told to stop"
    );
}

#[tokio::test]
async fn cancel_of_an_unknown_id_is_not_found() {
    let h = harness();

    let (status, body) = h.cancel("resp_missing").await;

    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["type"], "not_found", "{body}");
}

#[tokio::test]
async fn cancelling_a_live_stream_records_the_response_as_cancelled() {
    let h = harness();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "cancel-test-model", "input": "hi", "stream": true}).to_string(),
        ))
        .expect("request builds");
    let mut body = h
        .app
        .clone()
        .oneshot(request)
        .await
        .expect("router responds")
        .into_body();

    // Wait for the first token so the stream is in flight.
    h.handle.token("Hel");
    let mut text = String::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while !text.contains("Hel") {
            let frame = body.frame().await.expect("stream open").expect("frame ok");
            if let Some(data) = frame.data_ref() {
                text.push_str(&String::from_utf8_lossy(data));
            }
        }
    })
    .await
    .expect("first token arrives");
    let id = text
        .split("\"id\":\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .expect("response.created carries the id")
        .to_string();

    let (status, _) = h.cancel(&id).await;
    assert_eq!(status, StatusCode::OK);
    h.handle.finish();
    let rest = axum::body::to_bytes(body, 4 * 1024 * 1024)
        .await
        .expect("stream ends");
    text.push_str(&String::from_utf8_lossy(&rest));

    // Both the terminal event and the stored copy agree with the cancel reply.
    let (name, terminal) = terminal_event(&text);
    assert_eq!(name, "response.cancelled", "{text}");
    assert_eq!(terminal["type"], "response.cancelled", "{terminal}");
    assert_eq!(terminal["response"]["status"], "cancelled", "{terminal}");
    assert!(terminal["response"]["completed_at"].is_null(), "{terminal}");
    assert_eq!(h.retrieve(&id).await["status"], "cancelled");
}

/// The last SSE frame of `text`, as its `event:` name and parsed `data:`.
fn terminal_event(text: &str) -> (String, Value) {
    let frame = text
        .split("\n\n")
        .filter(|frame| frame.contains("data: "))
        .last()
        .expect("stream carries an event");
    let field = |prefix: &str| {
        frame
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("frame has `{prefix}`: {frame}"))
    };
    (
        field("event: ").to_string(),
        serde_json::from_str(field("data: ")).expect("event data is JSON"),
    )
}

/// Open a stream and read until the first token has arrived; returns the
/// response id, the body still open, and the text read so far.
async fn open_stream(h: &Harness) -> (String, Body, String) {
    open_stream_with(
        h,
        json!({"model": "cancel-test-model", "input": "hi", "stream": true}),
    )
    .await
}

async fn open_stream_with(h: &Harness, request: Value) -> (String, Body, String) {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(request.to_string()))
        .expect("request builds");
    let mut body = h
        .app
        .clone()
        .oneshot(request)
        .await
        .expect("router responds")
        .into_body();
    h.handle.token("Hel");
    let mut text = String::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while !text.contains("Hel") {
            let frame = body.frame().await.expect("stream open").expect("frame ok");
            if let Some(data) = frame.data_ref() {
                text.push_str(&String::from_utf8_lossy(data));
            }
        }
    })
    .await
    .expect("first token arrives");
    let id = text
        .split("\"id\":\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .expect("response.created carries the id")
        .to_string();
    (id, body, text)
}

#[tokio::test]
async fn an_uncancelled_stream_still_ends_completed() {
    let h = harness();
    let (id, body, mut text) = open_stream(&h).await;
    h.handle.finish();
    let rest = axum::body::to_bytes(body, 4 * 1024 * 1024)
        .await
        .expect("stream ends");
    text.push_str(&String::from_utf8_lossy(&rest));

    let (name, terminal) = terminal_event(&text);
    assert_eq!(name, "response.completed", "{text}");
    assert_eq!(terminal["response"]["status"], "completed", "{terminal}");
    assert_eq!(h.retrieve(&id).await["status"], "completed");
}

/// A client that hangs up once generation has finished cancelled nothing.
/// Only the post-generation sends can observe the closed channel here, so the
/// sender must not flip the token for them.
#[tokio::test]
async fn a_disconnect_after_generation_finished_stays_completed() {
    let h = harness();
    let (id, body, _text) = open_stream(&h).await;
    // Drop the client while generation is live but idle, then finish without
    // any further token: no send fails until generation has already ended.
    drop(body);
    h.handle.finish();

    let stored = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            if let Some(stored) = h.store.get(&id) {
                return stored;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("response is stored");
    assert_eq!(stored.response.status, ResponseStatus::Completed);
}

/// A cancel during queueing or prefill ends generation with an error before
/// the first token. The client was told `cancelled`, so the stream and the
/// stored copy must say so, not `server_error` and 404.
#[tokio::test]
async fn a_cancel_that_ends_generation_with_an_error_is_recorded_cancelled() {
    let h = harness();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "cancel-test-model", "input": "hi", "stream": true}).to_string(),
        ))
        .expect("request builds");
    let mut body = h
        .app
        .clone()
        .oneshot(request)
        .await
        .expect("router responds")
        .into_body();
    let mut text = String::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while !text.contains("response.in_progress") {
            let frame = body.frame().await.expect("stream open").expect("frame ok");
            if let Some(data) = frame.data_ref() {
                text.push_str(&String::from_utf8_lossy(data));
            }
        }
    })
    .await
    .expect("stream opens");
    let id = text
        .split("\"id\":\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .expect("response.created carries the id")
        .to_string();

    let (status, _) = h.cancel(&id).await;
    assert_eq!(status, StatusCode::OK);
    h.handle.fail("Request cancelled: client disconnected");
    let rest = axum::body::to_bytes(body, 4 * 1024 * 1024)
        .await
        .expect("stream ends");
    text.push_str(&String::from_utf8_lossy(&rest));

    let (name, terminal) = terminal_event(&text);
    assert_eq!(name, "response.cancelled", "{text}");
    assert_eq!(terminal["response"]["status"], "cancelled", "{terminal}");
    assert!(terminal["response"]["completed_at"].is_null(), "{terminal}");
    assert_eq!(h.retrieve(&id).await["status"], "cancelled");
}

/// A generation error nobody cancelled stays an error and stores nothing.
#[tokio::test]
async fn an_uncancelled_generation_error_stays_an_error() {
    let h = harness();
    h.handle.fail("boom");
    let (status, body) = h
        .send_raw_stream(json!({"model": "cancel-test-model", "input": "hi", "stream": true}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (name, terminal) = terminal_event(&body);
    assert_eq!(name, "response.error", "{body}");
    assert_eq!(terminal["code"], "server_error", "{terminal}");
    assert_eq!(h.store.len(), 0);
}

/// `response.cancelled` for a limit-stopped `context_management` reply must
/// not carry `incomplete_details`; the override drops the incomplete reason.
#[tokio::test]
async fn a_cancelled_stream_carries_no_incomplete_details() {
    let h = harness_with(
        ServerConfig {
            context_size: 512,
            ..ServerConfig::default()
        },
        MlxcelTokenizer::stub_all_byte_fallback(),
    );
    let (id, body, mut text) = open_stream_with(
        &h,
        json!({
            "model": "cancel-test-model",
            "input": "hi",
            "stream": true,
            "context_management": [{"type": "compaction", "compact_threshold": 10_000}],
        }),
    )
    .await;
    let (status, _) = h.cancel(&id).await;
    assert_eq!(status, StatusCode::OK);
    h.handle.finish_length();
    let rest = axum::body::to_bytes(body, 4 * 1024 * 1024)
        .await
        .expect("stream ends");
    text.push_str(&String::from_utf8_lossy(&rest));

    let (_, terminal) = terminal_event(&text);
    assert_eq!(terminal["response"]["status"], "cancelled", "{terminal}");
    assert!(
        terminal["response"]["incomplete_details"].is_null(),
        "{terminal}"
    );
}
