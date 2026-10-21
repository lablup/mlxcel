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

//! Automatic compaction on `/v1/responses` (issue #2397, phase B).

//! Validation runs model-free; the route tests drive the real router against
//! a scripted model and a one-token-per-byte tokenizer, so every token count
//! is a byte count of the rendered prompt.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::parse_context_management;
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::responses_compaction::SUMMARY_INSTRUCTION;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig};
use crate::server::{
    AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, ServerGenerateOptions, create_app,
};
use crate::tokenizer::MlxcelTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";
const MODEL: &str = "auto-compact-test-model";
/// Marks every turn compaction should summarize away.
const OLD: &str = "OLDTURN";
const SUMMARY: &str = "HANDOFF-SUMMARY";
const CONTEXT: usize = 4096;

// -- Validation ----------------------------------------------------------------

#[test]
fn context_management_absent_null_or_empty_disables_compaction() {
    assert_eq!(parse_context_management(None).ok(), Some(None));
    assert_eq!(
        parse_context_management(Some(&Value::Null)).ok(),
        Some(None)
    );
    assert_eq!(parse_context_management(Some(&json!([]))).ok(), Some(None));
    assert_eq!(
        parse_context_management(Some(&json!([
            {"type": "compaction", "compact_threshold": 24000}
        ])))
        .ok(),
        Some(Some(24000))
    );
}

#[test]
fn malformed_context_management_is_invalid_request() {
    for bad in [
        json!({"type": "compaction", "compact_threshold": 10}),
        json!([{"type": "truncation", "compact_threshold": 10}]),
        json!([{"compact_threshold": 10}]),
        json!([{"type": "compaction"}]),
        json!([{"type": "compaction", "compact_threshold": 0}]),
        json!([{"type": "compaction", "compact_threshold": -5}]),
        json!([{"type": "compaction", "compact_threshold": 1.5}]),
        json!([{"type": "compaction", "compact_threshold": "10"}]),
        json!([{"type": "compaction", "compact_threshold": 100_000_001u64}]),
        json!([
            {"type": "compaction", "compact_threshold": 10},
            {"type": "compaction", "compact_threshold": 10},
        ]),
    ] {
        let err = parse_context_management(Some(&bad)).expect_err(&bad.to_string());
        assert_eq!(err.status, StatusCode::BAD_REQUEST, "{bad}");
        assert_eq!(err.error.error_type, "invalid_request_error", "{bad}");
    }
}

// -- Route harness -------------------------------------------------------------

struct Harness {
    app: Router,
    handle: ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
    options: mpsc::Receiver<ServerGenerateOptions>,
}

fn harness(context_size: usize) -> Harness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig {
            context_size,
            ..ServerConfig::default()
        },
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub_all_byte_fallback(),
        PathBuf::from(MODEL),
        batch_metrics,
    )
    .with_responses_store(Some(Arc::new(ResponsesStore::new(
        ResponsesStoreConfig::default(),
    ))));
    Harness {
        app: create_app(state),
        handle,
        prompts: prompt_rx,
        options: options_rx,
    }
}

/// One scripted generation: its tokens, and whether it stops at `length`.
struct Script<'a>(&'a [&'a str], bool);

impl Harness {
    /// POST `body` with the generations it will run scripted in order.
    async fn post_raw(&self, body: &Value, scripts: &[Script<'_>]) -> (StatusCode, String) {
        for Script(tokens, length) in scripts {
            for token in *tokens {
                self.handle.token(token);
            }
            if *length {
                self.handle.finish_length();
            } else {
                self.handle.finish();
            }
        }
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

    async fn post(&self, body: &Value, scripts: &[Script<'_>]) -> (StatusCode, Value) {
        let (status, text) = self.post_raw(body, scripts).await;
        (status, serde_json::from_str(&text).unwrap_or_default())
    }

    /// Every rendered prompt dispatched so far.
    fn drain_prompts(&self) -> Vec<String> {
        self.prompts.try_iter().collect()
    }
}

fn history() -> Value {
    let filler = "x".repeat(300);
    json!([
        {"type": "message", "role": "system", "content": "SYS"},
        {"type": "message", "role": "user", "content": format!("{OLD} q1 {filler}")},
        {"type": "message", "role": "assistant", "content": format!("{OLD} a1 {filler}")},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": format!("{OLD} result {filler}")},
        {"type": "message", "role": "user", "content": "latest question"},
    ])
}

fn body(input: Value, threshold: Option<u64>) -> Value {
    let mut body = json!({"model": MODEL, "input": input, "store": true});
    if let Some(threshold) = threshold {
        body["context_management"] =
            json!([{"type": "compaction", "compact_threshold": threshold}]);
    }
    body
}

/// `(event name, payload)` of every SSE frame.
fn sse_events(text: &str) -> Vec<(String, Value)> {
    text.split("\n\n")
        .filter_map(|frame| {
            let mut name = None;
            let mut data = None;
            for line in frame.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    name = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data = serde_json::from_str(rest.trim()).ok();
                }
            }
            Some((name?, data?))
        })
        .collect()
}

// -- Route tests ---------------------------------------------------------------

#[tokio::test]
async fn auto_compaction_puts_the_capsule_first_and_generates_from_the_summary() {
    let h = harness(CONTEXT);
    let (status, response) = h
        .post(
            &body(history(), Some(200)),
            &[Script(&[SUMMARY], false), Script(&["answer"], false)],
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let output = response["output"].as_array().expect("output array");
    assert_eq!(output[0]["type"], "compaction", "{output:?}");
    assert!(
        output[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("cmp_"))
    );
    assert_eq!(output[1]["type"], "message");
    assert_eq!(response["output_text"], "answer");
    assert_eq!(response["status"], "completed");

    let prompts = h.drain_prompts();
    assert_eq!(prompts.len(), 2, "one summary pass, one generation");
    assert!(prompts[0].contains(OLD) && prompts[0].contains(&SUMMARY_INSTRUCTION[..40]));
    for kept in ["SYS", SUMMARY, "latest question"] {
        assert!(prompts[1].contains(kept), "{kept} missing: {}", prompts[1]);
    }
    assert!(!prompts[1].contains(OLD), "{}", prompts[1]);
}

#[tokio::test]
async fn a_chained_turn_replays_the_summary_without_compacting_again() {
    let h = harness(CONTEXT);
    let (status, first) = h
        .post(
            &body(history(), Some(200)),
            &[Script(&[SUMMARY], false), Script(&["ok"], false)],
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(h.drain_prompts().len(), 2);

    let mut next = body(json!("next step"), Some(200));
    next["previous_response_id"] = first["id"].clone();
    let (status, second) = h.post(&next, &[Script(&["done"], false)]).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let prompts = h.drain_prompts();
    assert_eq!(prompts.len(), 1, "below threshold: no second summary pass");
    for kept in ["SYS", SUMMARY, "latest question", "ok", "next step"] {
        assert!(prompts[0].contains(kept), "{kept} missing: {}", prompts[0]);
    }
    assert!(!prompts[0].contains(OLD), "{}", prompts[0]);
    assert_eq!(
        prompts[0].matches("<|system|>SYS").count(),
        1,
        "{}",
        prompts[0]
    );
    assert_eq!(second["output"][0]["type"], "message", "{second}");
}

#[tokio::test]
async fn without_context_management_nothing_changes() {
    let h = harness(CONTEXT);
    let mut request = body(history(), None);
    request["max_output_tokens"] = json!(64);
    let (status, response) = h.post(&request, &[Script(&["answer"], true)]).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let prompts = h.drain_prompts();
    assert_eq!(prompts.len(), 1, "no summary pass");
    assert!(prompts[0].contains(OLD));
    assert_eq!(response["output"][0]["type"], "message");
    // A `length` finish keeps today's shape outside `context_management`.
    assert_eq!(response["status"], "completed");
    assert!(response["incomplete_details"].is_null());
    let options = h.options.try_recv().expect("generation options");
    assert_eq!(options.max_tokens, 64, "no output cap");
}

#[tokio::test]
async fn output_is_capped_at_the_remaining_context_and_reports_incomplete() {
    let context = 512;
    let h = harness(context);
    let mut request = body(json!("short question"), Some(10_000));
    request["max_output_tokens"] = json!(100_000);
    let (status, response) = h.post(&request, &[Script(&["partial"], true)]).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let prompts = h.drain_prompts();
    assert_eq!(prompts.len(), 1, "below threshold: no summary pass");
    let options = h.options.try_recv().expect("generation options");
    let prompt_tokens = prompts[0].len();
    assert!(
        options.max_tokens <= context - prompt_tokens
            && options.max_tokens + 2 >= context - prompt_tokens,
        "max_tokens {} for a {prompt_tokens}-token prompt in {context}",
        options.max_tokens
    );
    assert_eq!(response["status"], "incomplete", "{response}");
    assert_eq!(
        response["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(response["output_text"], "partial");
}

#[tokio::test]
async fn streamed_auto_compaction_reports_progress_before_the_capsule() {
    let h = harness(CONTEXT);
    let mut request = body(history(), Some(200));
    request["stream"] = json!(true);
    let (status, text) = h
        .post_raw(
            &request,
            &[Script(&[SUMMARY], false), Script(&["an", "swer"], false)],
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let events = sse_events(&text);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        &names[..6],
        [
            "response.created",
            "response.in_progress",
            "mlx.compaction.started",
            "mlx.compaction.completed",
            "response.output_item.added",
            "response.output_item.done",
        ],
        "{names:?}"
    );
    assert_eq!(names.last(), Some(&"response.completed"), "{names:?}");
    let sequence: Vec<u64> = events
        .iter()
        .map(|(_, data)| data["sequence_number"].as_u64().expect("sequence_number"))
        .collect();
    assert!(
        sequence.windows(2).all(|w| w[1] == w[0] + 1),
        "{sequence:?}"
    );

    let (_, started) = &events[2];
    let (_, completed) = &events[3];
    let response_id = events[0].1["response"]["id"].clone();
    assert_eq!(started["response_id"], response_id);
    assert_eq!(completed["response_id"], response_id);
    let before = completed["input_tokens_before"].as_u64().expect("before");
    let after = completed["input_tokens_after"].as_u64().expect("after");
    assert_eq!(started["input_tokens"].as_u64(), Some(before));
    assert!(before >= 200 && after < before, "{before} -> {after}");

    let (_, added) = &events[4];
    assert_eq!(added["output_index"], 0);
    assert_eq!(added["item"]["type"], "compaction");
    let capsule_id = added["item"]["id"].clone();
    let message = events
        .iter()
        .find(|(name, data)| {
            name == "response.output_item.added" && data["item"]["type"] == "message"
        })
        .expect("message item");
    assert_eq!(
        message.1["output_index"], 1,
        "generated items follow the capsule"
    );

    let (_, done) = events.last().expect("final event");
    let output = done["response"]["output"].as_array().expect("output");
    assert_eq!(output[0]["id"], capsule_id, "{output:?}");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(done["response"]["output_text"], "answer");

    // Counts are byte counts of the rendered prompt here; the generation ran
    // on the context `input_tokens_after` describes.
    let prompts = h.drain_prompts();
    assert_eq!(prompts.len(), 2);
    assert!(
        (after as usize).abs_diff(prompts[1].len()) <= 1,
        "{after} vs {}",
        prompts[1].len()
    );
}

#[tokio::test]
async fn compaction_failures_return_context_length_exceeded() {
    // One oversized turn: nothing to split, and it does not fit.
    let h = harness(256);
    for stream in [false, true] {
        let mut request = body(json!("y".repeat(400)), Some(100));
        request["stream"] = json!(stream);
        let (status, response) = h.post(&request, &[]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert_eq!(response["error"]["code"], "context_length_exceeded");
        assert_eq!(response["error"]["type"], "exceed_context_size_error");
    }
    assert!(h.drain_prompts().is_empty(), "nothing generated");
}

#[tokio::test]
async fn nothing_to_split_but_fitting_generates_uncompacted() {
    let h = harness(CONTEXT);
    let (status, response) = h
        .post(
            &body(json!("only turn"), Some(5)),
            &[Script(&["fine"], false)],
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["output"][0]["type"], "message", "{response}");
    assert_eq!(h.drain_prompts().len(), 1, "no summary pass");
}

#[tokio::test]
async fn malformed_context_management_is_refused_before_generation() {
    let h = harness(CONTEXT);
    let mut request = body(history(), None);
    request["context_management"] = json!([{"type": "compaction", "compact_threshold": 0}]);
    let (status, response) = h.post(&request, &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["type"], "invalid_request_error");
    assert!(h.drain_prompts().is_empty());
}

#[tokio::test]
async fn streamed_tool_requests_list_the_capsule_once_and_length_ends_incomplete() {
    let h = harness(CONTEXT);
    let mut request = body(history(), Some(200));
    request["stream"] = json!(true);
    // Declared tools turn on tool-call streaming, whose final object is
    // built from the streamed items (#2396).
    request["tools"] = json!([{
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object", "properties": {}},
    }]);
    let (status, text) = h
        .post_raw(
            &request,
            &[Script(&[SUMMARY], false), Script(&["cut off"], true)],
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let events = sse_events(&text);
    let (name, last) = events.last().expect("final event");
    assert_eq!(name, "response.incomplete", "{text}");
    assert_eq!(last["response"]["status"], "incomplete");
    assert_eq!(
        last["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    let output = last["response"]["output"].as_array().expect("output");
    let capsules = output.iter().filter(|i| i["type"] == "compaction").count();
    assert_eq!(capsules, 1, "{output:?}");
    assert_eq!(output[0]["type"], "compaction", "{output:?}");
}
