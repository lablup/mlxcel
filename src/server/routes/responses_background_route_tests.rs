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

//! `background: true` on `POST /v1/responses` through the real router
//! against a scripted model (issue #2898): create answers `queued`, polling
//! follows the job to a terminal status, and no exit leaves a job active.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use axum::response::Response;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{BackgroundJob, spawn_background_job, spawn_background_job_with};
use crate::server::config::ReasoningBudgetOverride;
use crate::server::conversation_store::{ConversationStore, ConversationStoreConfig};
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::request_options::resolve_server_max_tokens_with_live;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig};
use crate::server::responses_translator::responses_request_to_chat;
use crate::server::thinking_budget::resolve_request_budget;
use crate::server::types::ResponseStatus;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";
const WAIT: Duration = Duration::from_secs(10);

struct Harness {
    app: Router,
    state: AppState,
    store: Arc<ResponsesStore>,
    handle: ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
    _options: mpsc::Receiver<crate::server::ServerGenerateOptions>,
}

fn harness() -> Harness {
    harness_with(ResponsesStoreConfig::default())
}

fn harness_with(store: ResponsesStoreConfig) -> Harness {
    harness_full(
        Some(store),
        ServerConfig::default(),
        MlxcelTokenizer::stub(),
    )
}

fn harness_full(
    store_config: Option<ResponsesStoreConfig>,
    config: ServerConfig,
    tokenizer: MlxcelTokenizer,
) -> Harness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let store = store_config.map(|config| Arc::new(ResponsesStore::new(config)));
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        tokenizer,
        PathBuf::from("background-test-model"),
        batch_metrics,
    )
    .with_responses_store(store.clone())
    .with_conversation_store(Some(Arc::new(ConversationStore::new(
        ConversationStoreConfig::default(),
    ))));
    Harness {
        app: create_app(state.clone()),
        state,
        // A harness without a store only exercises the refusal.
        store: store.unwrap_or_else(|| Arc::new(ResponsesStore::new(Default::default()))),
        handle,
        prompts: prompt_rx,
        _options: options_rx,
    }
}

fn background_body() -> Value {
    json!({"model": "background-test-model", "input": "hi", "background": true})
}

impl Harness {
    /// Send one request and return the response with its body unread.
    async fn raw(&self, method: Method, path: &str, body: Option<Value>) -> Response {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
            .expect("request builds");
        let app = self.app.clone();
        app.oneshot(request).await.expect("router responds")
    }

    async fn send(&self, method: Method, path: &str, body: Option<Value>) -> (StatusCode, Value) {
        let response = self.raw(method, path, body).await;
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body collects");
        (
            status,
            serde_json::from_slice(&bytes).expect("response is JSON"),
        )
    }

    async fn create(&self, body: Value) -> (StatusCode, Value) {
        self.send(Method::POST, "/v1/responses", Some(body)).await
    }
    /// Create one background response and return its id.
    async fn create_background(&self, body: Value) -> String {
        let (status, body) = self.create(body).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "queued", "{body}");
        body["id"].as_str().expect("response id").to_string()
    }

    async fn get(&self, id: &str) -> (StatusCode, Value) {
        self.send(Method::GET, &format!("/v1/responses/{id}"), None)
            .await
    }

    /// Poll `GET /v1/responses/{id}` until it reports `status`.
    async fn poll_until(&self, id: &str, status: &str) -> Value {
        tokio::time::timeout(WAIT, async {
            loop {
                let (code, body) = self.get(id).await;
                assert_eq!(code, StatusCode::OK, "{id}: {body}");
                if body["status"] == status {
                    return body;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("response {id} never reached {status}"))
    }

    /// Wait for the job to end, then return what a poll reads.
    async fn terminal(&self, id: &str) -> Value {
        self.store
            .wait_background_terminal(id, WAIT)
            .await
            .unwrap_or_else(|| panic!("response {id} never finished"));
        self.get(id).await.1
    }

    /// The job spec the create handler builds for `body`.
    fn job(&self, body: Value) -> BackgroundJob {
        let request = serde_json::from_value(body).expect("request parses");
        let translated = responses_request_to_chat(
            &request,
            self.state.responses_store.as_ref(),
            self.state.conversation_store.as_ref(),
        )
        .expect("request translates");
        let live = self.state.live();
        let max_tokens = resolve_server_max_tokens_with_live(
            &self.state.config,
            &live,
            translated.chat_request.params.max_tokens,
        );
        let budget = resolve_request_budget(None, live.reasoning_budget, max_tokens)
            .expect("default budget resolves");
        BackgroundJob {
            state: self.state.clone(),
            live,
            request,
            translated,
            response_id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
            created_at: 0.0,
            priority: Default::default(),
            budget_override: ReasoningBudgetOverride::Explicit(budget),
            structured: None,
            compact_threshold: None,
        }
    }
}

/// `output` without the random item ids.
fn output_without_ids(response: &Value) -> Value {
    let mut output = response["output"].clone();
    for item in output.as_array_mut().expect("output is an array") {
        item.as_object_mut()
            .expect("item is an object")
            .remove("id");
    }
    output
}

#[tokio::test]
async fn create_answers_queued_and_polling_follows_the_job_to_completed() {
    let h = harness();
    // Nothing is scripted yet: create must not wait for generation.
    let (status, created) = h.create(background_body()).await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(created["status"], "queued", "{created}");
    assert_eq!(created["background"], true, "{created}");
    assert_eq!(created["output"], json!([]), "{created}");
    assert_eq!(created["usage"]["total_tokens"], 0, "{created}");
    let id = created["id"].as_str().expect("response id").to_string();
    assert_eq!(h.get(&id).await.1["status"], "queued");
    assert_eq!(h.store.active_background_jobs(), 1);
    h.handle.token("Hello.");
    let running = h.poll_until(&id, "in_progress").await;
    assert_eq!(running["output"], json!([]), "{running}");
    assert_eq!(running["usage"]["total_tokens"], 0, "{running}");

    h.handle.finish();
    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["background"], true, "{done}");
    assert_eq!(done["output_text"], "Hello.", "{done}");
    assert_eq!(h.store.active_background_jobs(), 0);

    // The same script in the foreground produces the same output, and the
    // foreground object carries no `background` key at all.
    for body in [
        json!({"model": "background-test-model", "input": "hi"}),
        json!({"model": "background-test-model", "input": "hi", "background": false}),
    ] {
        h.handle.token("Hello.");
        h.handle.finish();
        let (status, foreground) = h.create(body).await;
        assert_eq!(status, StatusCode::OK, "{foreground}");
        assert_eq!(output_without_ids(&done), output_without_ids(&foreground));
        assert!(foreground.get("background").is_none(), "{foreground}");
        let stored = h.get(foreground["id"].as_str().expect("id")).await.1;
        assert!(stored.get("background").is_none(), "{stored}");
    }
}

#[tokio::test]
async fn a_job_at_its_output_limit_ends_incomplete() {
    let h = harness_full(
        Some(ResponsesStoreConfig::default()),
        ServerConfig {
            context_size: 512,
            ..ServerConfig::default()
        },
        MlxcelTokenizer::stub_all_byte_fallback(),
    );
    let mut body = background_body();
    body["context_management"] = json!([{"type": "compaction", "compact_threshold": 10_000}]);
    let id = h.create_background(body).await;
    h.handle.token("Hel");
    h.handle.finish_length();

    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "incomplete", "{done}");
    assert_eq!(
        done["incomplete_details"]["reason"], "max_output_tokens",
        "{done}"
    );
    assert_eq!(done["output_text"], "Hel", "{done}");
}

#[tokio::test]
async fn a_failed_generation_ends_failed_with_an_error() {
    let h = harness();
    let id = h.create_background(background_body()).await;
    h.handle.fail("boom");

    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "failed", "{done}");
    assert_eq!(done["error"]["code"], "server_error", "{done}");
    assert!(
        done["error"]["message"].to_string().contains("boom"),
        "{done}"
    );
    assert_eq!(done["output"], json!([]), "{done}");
    assert_eq!(h.store.active_background_jobs(), 0);
}

#[tokio::test]
async fn validation_errors_return_400_and_create_no_job() {
    let h = harness();
    let invalid = [
        json!({"stream": true}),
        json!({"input": "  "}),
        json!({"xtc_probability": 2.0}),
        json!({"previous_response_id": "resp_missing"}),
        json!({"context_management": [{"type": "unknown"}]}),
    ];
    for patch in invalid {
        let mut body = background_body();
        for (key, value) in patch.as_object().expect("patch is an object") {
            body[key] = value.clone();
        }
        let (status, reply) = h.create(body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {reply}");
        assert_eq!(reply["error"]["type"], "invalid_request_error", "{reply}");
    }
    assert_eq!(h.store.active_background_jobs(), 0);
    assert_eq!(h.store.len(), 0);

    // Background mode off: no response store, or a job limit of zero.
    let no_store = harness_full(None, ServerConfig::default(), MlxcelTokenizer::stub());
    let no_jobs = harness_with(ResponsesStoreConfig {
        background_max_jobs: 0,
        ..ResponsesStoreConfig::default()
    });
    // A zero entry or byte budget disables background mode too, and answers
    // 400 before the queue probe could turn it into a 503.
    let no_entries = harness_with(ResponsesStoreConfig {
        max_entries: 0,
        ..ResponsesStoreConfig::default()
    });
    let no_bytes = harness_with(ResponsesStoreConfig {
        max_bytes: 0,
        ..ResponsesStoreConfig::default()
    });
    for h in [&no_store, &no_jobs, &no_entries, &no_bytes] {
        let (status, reply) = h.create(background_body()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(reply["error"]["type"], "invalid_request_error", "{reply}");
        assert_eq!(h.store.active_background_jobs(), 0);
    }
}

#[tokio::test]
async fn the_job_limit_returns_429_and_admits_nothing() {
    let h = harness_with(ResponsesStoreConfig {
        background_max_jobs: 1,
        ..ResponsesStoreConfig::default()
    });
    let first = h.create_background(background_body()).await;

    let (status, reply) = h.create(background_body()).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{reply}");
    assert_eq!(reply["error"]["code"], "rate_limit_exceeded", "{reply}");
    assert_eq!(h.store.active_background_jobs(), 1);

    // The slot frees once the first job ends.
    h.handle.finish();
    assert_eq!(h.terminal(&first).await["status"], "completed");
    let second = h.create_background(background_body()).await;
    h.handle.finish();
    assert_eq!(h.terminal(&second).await["status"], "completed");
}

#[tokio::test]
async fn dropping_the_create_response_does_not_cancel_the_job() {
    let h = harness();
    // The client goes away without reading the body.
    let body = Some(background_body());
    drop(h.raw(Method::POST, "/v1/responses", body).await);

    h.handle.token("Hello.");
    h.handle.finish();
    tokio::time::timeout(WAIT, async {
        while h.store.active_background_jobs() != 0 || h.store.len() != 1 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the job finishes and is stored");
    let flag = h.handle.cancellation_flag(0).expect("one generation ran");
    assert!(!flag.load(Ordering::Relaxed), "nothing cancelled the job");
}

#[tokio::test]
async fn a_worker_that_exited_ends_the_job_failed() {
    let h = harness();
    h.state
        .model_provider
        .shutdown_and_wait(WAIT)
        .expect("worker exits");
    let id = h.create_background(background_body()).await;

    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "failed", "{done}");
    assert_eq!(done["error"]["code"], "server_error", "{done}");
    assert!(
        done.to_string().contains("chat worker has exited"),
        "{done}"
    );
    assert_eq!(h.store.active_background_jobs(), 0);
}

#[tokio::test]
async fn a_driver_panic_ends_the_job_failed() {
    let h = harness();
    let queued = spawn_background_job_with(h.job(background_body()), |_run, _cancelled, _sink| {
        panic!("driver panic under test")
    })
    .expect("job is admitted");
    assert_eq!(h.store.active_background_jobs(), 1);

    let done = h.terminal(&queued.id).await;
    assert_eq!(done["error"]["code"], "server_error", "{done}");
    assert_eq!(h.store.active_background_jobs(), 0);
}

#[tokio::test]
async fn a_driver_panic_after_the_terminal_keeps_the_recorded_result() {
    let h = harness();
    let store = Arc::clone(&h.store);
    let job = h.job(background_body());
    let id = job.response_id.clone();
    let queued_id = id.clone();
    let queued = spawn_background_job_with(job, move |_run, _c, _sink| {
        // The terminal is recorded, then the driver panics.
        let mut terminal = store
            .background_snapshot(&queued_id)
            .expect("the queued object is readable");
        terminal.status = ResponseStatus::Completed;
        store.finish_background(&queued_id, terminal);
        panic!("driver panic after the terminal")
    })
    .expect("job is admitted");
    assert_eq!(queued.id, id);

    let done = h.terminal(&queued.id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert!(done["error"].is_null(), "{done}");
}

#[tokio::test]
async fn a_cancel_keeps_working_through_the_in_flight_registry() {
    let h = harness();
    // Cancelled while queued: the job task has not run yet on this
    // single-threaded runtime, so generation is skipped.
    let queued = spawn_background_job(h.job(background_body())).expect("job is admitted");
    assert!(h.store.cancel_in_flight(&queued.id));
    let done = h.terminal(&queued.id).await;
    assert_eq!(done["status"], "cancelled", "{done}");
    assert_eq!(done["output"], json!([]), "{done}");
    assert!(
        h.handle.cancellation_flag(0).is_none(),
        "nothing dispatched"
    );

    // Cancelled mid-generation through the route.
    let id = h.create_background(background_body()).await;
    h.handle.token("Hel");
    h.poll_until(&id, "in_progress").await;
    let cancel = format!("/v1/responses/{id}/cancel");
    let (_, reply) = h.send(Method::POST, &cancel, None).await;
    assert_eq!(reply["status"], "cancelled", "{reply}");
    h.handle.finish();
    assert_eq!(h.terminal(&id).await["status"], "cancelled");
    assert_eq!(h.store.active_background_jobs(), 0);
}

#[tokio::test]
async fn a_store_false_result_is_readable_until_it_expires() {
    let h = harness_with(ResponsesStoreConfig {
        background_ephemeral_ttl: Duration::from_millis(200),
        ..ResponsesStoreConfig::default()
    });
    let mut body = background_body();
    body["store"] = json!(false);
    let id = h.create_background(body).await;
    h.handle.token("Hello.");
    h.handle.finish();

    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["output_text"], "Hello.", "{done}");
    // Readable by polling only: it cannot seed a chained request.
    let mut chained = background_body();
    chained["previous_response_id"] = json!(id);
    assert_eq!(h.create(chained).await.0, StatusCode::BAD_REQUEST);

    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(h.get(&id).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn previous_response_id_waits_for_the_job_and_delete_refuses_an_active_job() {
    let h = harness();
    let mut body = background_body();
    body["conversation"] = json!("conv_background");
    let id = h.create_background(body).await;
    let chained = json!({
        "model": "background-test-model",
        "input": "and then?",
        "previous_response_id": id,
    });

    let (status, reply) = h.create(chained.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert_eq!(
        reply["error"]["message"],
        format!("previous response '{id}' has not finished (status: queued)")
    );
    let path = format!("/v1/responses/{id}");
    let (status, _) = h.send(Method::DELETE, &path, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(h.get(&id).await.1["status"], "queued");

    h.handle.token("First answer.");
    h.handle.finish();
    assert_eq!(h.terminal(&id).await["status"], "completed");
    // The turn reached its conversation at the terminal state.
    let conversations = h.state.conversation_store.as_ref().expect("store");
    let transcript = conversations.get("conv_background").expect("conversation");
    assert_eq!(transcript.items.len(), 2);

    // After the terminal state the response chains like a foreground one.
    while h.prompts.try_recv().is_ok() {}
    h.handle.token("Second answer.");
    h.handle.finish();
    assert_eq!(h.create(chained).await.0, StatusCode::OK);
    let prompt = h.prompts.recv().expect("chained prompt");
    assert!(prompt.contains("First answer."), "{prompt}");

    let (_, reply) = h.send(Method::DELETE, &path, None).await;
    assert_eq!(reply["deleted"], true, "{reply}");
}
