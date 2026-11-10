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

//! [`drive_response_run`] against a recording sink on the scripted model
//! (issue #2896): the run delivers the same events a foreground SSE stream
//! carries, and calls the sink's lifecycle hooks in the contract's order.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{ResponseEventSink, drive_response_run, prepare_response_run};
use crate::server::config::ReasoningBudgetOverride;
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::request_options::resolve_server_max_tokens_with_live;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig};
use crate::server::responses_translator::responses_request_to_chat;
use crate::server::thinking_budget::resolve_request_budget;
use crate::server::types::responses_request::CreateResponseRequest;
use crate::server::types::responses_response::{ResponseObject, ResponseStatus};
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

/// One sink call, in the order the driver made it.
#[derive(Debug, Clone, PartialEq)]
enum Call {
    /// An event: its SSE name and `sequence_number`.
    Emit(String, u64),
    Started,
    RecordTerminal(ResponseStatus),
    FinishRegistration,
    GenerationFinished,
}

#[derive(Default)]
struct RecordingSink {
    calls: Mutex<Vec<Call>>,
}

impl RecordingSink {
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn push(&self, call: Call) {
        self.calls.lock().expect("calls lock").push(call);
    }
}

impl ResponseEventSink for RecordingSink {
    fn emit(&self, event: &ResponseStreamEvent) {
        let payload = serde_json::to_value(event).expect("event serializes");
        let sequence_number = payload["sequence_number"]
            .as_u64()
            .expect("every event carries a sequence_number");
        self.push(Call::Emit(event.event_name().to_string(), sequence_number));
    }

    fn on_started(&self) {
        self.push(Call::Started);
    }

    fn record_terminal(&self, response: &ResponseObject) {
        self.push(Call::RecordTerminal(response.status));
    }

    fn finish_registration(&self) -> bool {
        self.push(Call::FinishRegistration);
        false
    }

    fn generation_finished(&self) {
        self.push(Call::GenerationFinished);
    }
}

struct Harness {
    state: AppState,
    handle: ScriptedStreamHandle,
    _options: mpsc::Receiver<crate::server::ServerGenerateOptions>,
}

fn harness() -> Harness {
    let (options_tx, options_rx) = mpsc::channel();
    let (provider, handle) = ModelProvider::scripted_streaming_for_route_tests(options_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("run-test-model"),
        batch_metrics,
    )
    .with_responses_store(Some(Arc::new(ResponsesStore::new(
        ResponsesStoreConfig::default(),
    ))));
    Harness {
        state,
        handle,
        _options: options_rx,
    }
}

fn request_body() -> Value {
    json!({"model": "run-test-model", "input": "hi", "stream": true})
}

impl Harness {
    fn script(&self, tokens: &[&str]) {
        for token in tokens {
            self.handle.token(token);
        }
        self.handle.finish();
    }

    /// The `(event name, sequence_number)` pairs of one foreground SSE
    /// stream, read off the wire.
    async fn sse_events(&self) -> Vec<Call> {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/responses")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(request_body().to_string()))
            .expect("request builds");
        let response = create_app(self.state.clone())
            .oneshot(request)
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body collects");
        let body = String::from_utf8_lossy(&bytes).into_owned();
        let mut events = Vec::new();
        let mut name = None;
        for line in body.lines() {
            if let Some(event) = line.strip_prefix("event: ") {
                name = Some(event.to_string());
            } else if let Some(data) = line.strip_prefix("data: ") {
                let payload: Value = serde_json::from_str(data).expect("data is JSON");
                let sequence_number = payload["sequence_number"]
                    .as_u64()
                    .expect("frame carries a sequence_number");
                events.push(Call::Emit(
                    name.take().expect("data follows an event header"),
                    sequence_number,
                ));
            }
        }
        events
    }

    /// Prepare one streamed request the way the handler does and drive it
    /// to the end into a recording sink.
    async fn drive(&self) -> Vec<Call> {
        let request: CreateResponseRequest =
            serde_json::from_value(request_body()).expect("request parses");
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
        let run = prepare_response_run(
            self.state.clone(),
            live,
            request,
            translated,
            format!("resp_{}", uuid::Uuid::new_v4().simple()),
            chrono::Utc::now().timestamp() as f64,
            Default::default(),
            ReasoningBudgetOverride::Explicit(budget),
            None,
            None,
        )
        .await
        .unwrap_or_else(|err| panic!("run prepares: {}", err.error.message));
        let sink = Arc::new(RecordingSink::default());
        let sink_for_run: Arc<dyn ResponseEventSink> = sink.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        tokio::task::spawn_blocking(move || drive_response_run(run, cancelled, sink_for_run))
            .await
            .expect("driver task joins");
        sink.calls()
    }
}

fn emitted(calls: &[Call]) -> Vec<Call> {
    calls
        .iter()
        .filter(|call| matches!(call, Call::Emit(..)))
        .cloned()
        .collect()
}

#[tokio::test]
async fn drive_response_run_delivers_events_to_a_non_sse_sink() {
    let harness = harness();
    let tokens = ["Hello", ",", " world"];

    harness.script(&tokens);
    let on_the_wire = harness.sse_events().await;
    harness.script(&tokens);
    let calls = harness.drive().await;

    let names: Vec<&str> = on_the_wire
        .iter()
        .map(|call| match call {
            Call::Emit(name, _) => name.as_str(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(names.first(), Some(&"response.created"), "{on_the_wire:?}");
    assert_eq!(names.last(), Some(&"response.completed"), "{on_the_wire:?}");
    assert_eq!(
        names
            .iter()
            .filter(|name| **name == "response.output_text.delta")
            .count(),
        tokens.len(),
        "{on_the_wire:?}"
    );
    assert_eq!(emitted(&calls), on_the_wire);
}

#[tokio::test]
async fn record_terminal_runs_once_before_the_terminal_event() {
    let harness = harness();
    harness.script(&["Hello", " there"]);
    let calls = harness.drive().await;

    let recorded: Vec<usize> = calls
        .iter()
        .enumerate()
        .filter(|(_, call)| matches!(call, Call::RecordTerminal(_)))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(recorded.len(), 1, "{calls:?}");
    assert_eq!(
        calls[recorded[0]],
        Call::RecordTerminal(ResponseStatus::Completed)
    );
    let terminal = calls
        .iter()
        .position(|call| matches!(call, Call::Emit(name, _) if name == "response.completed"))
        .expect("terminal event emitted");
    assert!(recorded[0] < terminal, "{calls:?}");
    assert_eq!(
        terminal,
        calls.len() - 1,
        "terminal event is last: {calls:?}"
    );
    // The registration is retired before the status is decided, after
    // generation is over.
    let finished = calls
        .iter()
        .position(|call| *call == Call::GenerationFinished)
        .expect("generation_finished called");
    let retired = calls
        .iter()
        .position(|call| *call == Call::FinishRegistration)
        .expect("finish_registration called");
    assert!(finished < retired && retired < recorded[0], "{calls:?}");
}

#[tokio::test]
async fn on_started_fires_once_across_several_tokens() {
    let harness = harness();
    harness.script(&["one", " two", " three", " four"]);
    let calls = harness.drive().await;

    let started: Vec<usize> = calls
        .iter()
        .enumerate()
        .filter(|(_, call)| **call == Call::Started)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(started.len(), 1, "{calls:?}");
    let first_delta = calls
        .iter()
        .position(
            |call| matches!(call, Call::Emit(name, _) if name == "response.output_text.delta"),
        )
        .expect("a delta was emitted");
    assert!(started[0] < first_delta, "{calls:?}");
}
