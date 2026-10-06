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

//! Route-level reasoning re-injection on `/v1/responses` and `/v1/messages`
//! (issue #2118): the real handlers, streaming and non-streaming, against a
//! scripted model and the Jamba-Reasoning template. Each test fails with the
//! store disabled (`MLXCEL_REASONING_ECHO_MAX_BYTES=0`).

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::JAMBA_TEMPLATE;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::model_provider::ModelProvider;
use crate::server::prompt_cache::PromptCacheStore;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig};
use crate::server::{AppState, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

struct Harness {
    app: axum::Router,
    handle: crate::server::model_provider::ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
    _options: mpsc::Receiver<crate::server::ServerGenerateOptions>,
}

fn harness() -> Harness {
    harness_with_responses_store(false)
}

fn harness_with_responses_store(responses_store: bool) -> Harness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(JAMBA_TEMPLATE.to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_prompt_cache(Some(Arc::new(PromptCacheStore::new())))
    .with_responses_store(
        responses_store.then(|| Arc::new(ResponsesStore::new(ResponsesStoreConfig::default()))),
    );
    Harness {
        app: create_app(state),
        handle,
        prompts: prompt_rx,
        _options: options_rx,
    }
}

impl Harness {
    /// Post `body` to `path` with a scripted generation of `tokens`; return the
    /// rendered prompt the model saw and the response body.
    async fn post(&self, path: &str, body: Value, tokens: &[&str]) -> (String, String) {
        for token in tokens {
            self.handle.token(token);
        }
        self.handle.finish();
        let response = self
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body collects")
            .to_bytes();
        assert_eq!(
            status,
            StatusCode::OK,
            "{path}: {}",
            String::from_utf8_lossy(&bytes)
        );
        let prompt = self.prompts.recv().expect("generation dispatched");
        (prompt, String::from_utf8_lossy(&bytes).into_owned())
    }
}

const TURN1_TOKENS: &[&str] = &["<think>", "the capital", "</think>", "\n\n", "Paris."];

/// The turn-1 history boundary that a content-only turn 2 must still start
/// with: the thinking instruction stays on the earlier user turn only when the
/// following assistant turn carries reasoning.
const GENERATED_TURN1: &str = "<|im_start|>user\nTHINK_PREFIX\nq1<|im_end|>\n";

fn responses_body(stream: bool, input: Value) -> Value {
    json!({
        "model": "route-test-model",
        "stream": stream,
        "store": false,
        "input": input,
    })
}

/// A Responses client that echoes only the assistant `message` item (no
/// `reasoning` item) gets the reasoning the server returned re-injected, so
/// turn 1 renders as generated. Both the non-streaming `split_reasoning` reply
/// and the streamed reasoning envelope feed the record.
#[tokio::test]
async fn responses_content_only_follow_up_keeps_the_generated_rendering() {
    for stream in [false, true] {
        let h = harness();
        let (prompt1, body1) = h
            .post(
                "/v1/responses",
                responses_body(
                    stream,
                    json!([{"type": "message", "role": "user", "content": "q1"}]),
                ),
                TURN1_TOKENS,
            )
            .await;
        assert!(prompt1.starts_with(GENERATED_TURN1), "{prompt1:?}");
        assert!(
            body1.contains("the capital"),
            "reasoning item surfaced: {body1}"
        );

        let (prompt2, _) = h
            .post(
                "/v1/responses",
                responses_body(
                    stream,
                    json!([
                        {"type": "message", "role": "user", "content": "q1"},
                        {"type": "message", "role": "assistant", "content": "Paris."},
                        {"type": "message", "role": "user", "content": "q2"},
                    ]),
                ),
                &["ok"],
            )
            .await;
        assert!(
            prompt2.starts_with(GENERATED_TURN1),
            "stream={stream}: the earlier turn must render as generated: {prompt2:?}"
        );
    }
}

/// A `previous_response_id` chain replays the stored reply's message item but
/// never its reasoning item, so every chained turn is content-only history;
/// the replayed turn is filled like an echoed one.
#[tokio::test]
async fn responses_previous_response_id_chain_keeps_the_generated_rendering() {
    let h = harness_with_responses_store(true);
    let (_, body1) = h
        .post(
            "/v1/responses",
            json!({"model": "route-test-model", "input": "q1"}),
            TURN1_TOKENS,
        )
        .await;
    let id = serde_json::from_str::<Value>(&body1).expect("json")["id"]
        .as_str()
        .expect("response id")
        .to_string();

    let (prompt2, _) = h
        .post(
            "/v1/responses",
            json!({"model": "route-test-model", "input": "q2", "previous_response_id": id}),
            &["ok"],
        )
        .await;
    assert!(prompt2.starts_with(GENERATED_TURN1), "{prompt2:?}");
}

fn messages_body(stream: bool, messages: Value) -> Value {
    json!({
        "model": "route-test-model",
        "stream": stream,
        "max_tokens": 256,
        "thinking": {"type": "enabled", "budget_tokens": 128},
        "messages": messages,
    })
}

/// An Anthropic client with extended thinking that echoes only the assistant
/// text gets the `thinking` block the server returned re-injected.
#[tokio::test]
async fn messages_content_only_follow_up_keeps_the_generated_rendering() {
    for stream in [false, true] {
        let h = harness();
        let (prompt1, body1) = h
            .post(
                "/v1/messages",
                messages_body(stream, json!([{"role": "user", "content": "q1"}])),
                TURN1_TOKENS,
            )
            .await;
        assert!(prompt1.starts_with(GENERATED_TURN1), "{prompt1:?}");
        assert!(
            body1.contains("the capital"),
            "thinking block surfaced: {body1}"
        );

        let (prompt2, _) = h
            .post(
                "/v1/messages",
                messages_body(
                    stream,
                    json!([
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "Paris."},
                        {"role": "user", "content": "q2"},
                    ]),
                ),
                &["ok"],
            )
            .await;
        assert!(
            prompt2.starts_with(GENERATED_TURN1),
            "stream={stream}: the earlier turn must render as generated: {prompt2:?}"
        );
    }
}

/// Without extended thinking the client never receives the reasoning, so
/// nothing is recorded and the follow-up keeps the content-only rendering.
#[tokio::test]
async fn messages_without_thinking_records_nothing() {
    let h = harness();
    let mut body = messages_body(false, json!([{"role": "user", "content": "q1"}]));
    body.as_object_mut().expect("object").remove("thinking");
    let (_, body1) = h.post("/v1/messages", body, TURN1_TOKENS).await;
    assert!(!body1.contains("the capital"), "{body1}");

    let mut body = messages_body(
        false,
        json!([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "q2"},
        ]),
    );
    body.as_object_mut().expect("object").remove("thinking");
    let (prompt2, _) = h.post("/v1/messages", body, &["ok"]).await;
    assert!(
        prompt2.starts_with("<|im_start|>user\nq1<|im_end|>\n"),
        "{prompt2:?}"
    );
}

/// Clients replay `response.output` items without a `type` and with
/// `output_text` parts on the assistant turn; both shapes must be accepted.
#[tokio::test]
async fn responses_accepts_untyped_items_and_output_text_parts() {
    let h = harness();
    let (prompt, _) = h
        .post(
            "/v1/responses",
            responses_body(
                false,
                json!([
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "Paris."}]},
                    {"role": "user", "content": "And Germany?"},
                ]),
            ),
            &["Berlin."],
        )
        .await;
    assert!(
        prompt.contains("<|im_start|>assistant\nParis.<|im_end|>"),
        "{prompt:?}"
    );
}
