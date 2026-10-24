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
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let store = Arc::new(ResponsesStore::new(ResponsesStoreConfig::default()));
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub(),
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
