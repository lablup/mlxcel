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

//! Primed open thinking on `/v1/responses` and `/v1/messages` (issue #2123).
//!
//! The real Jamba-Reasoning template ends the generation prompt with
//! `<|im_start|>assistant\n<think>\n`, so the model emits its trace with no
//! `<think>` opener and only the `</think>` close. The streaming arms used to
//! build an unprimed filter and streamed that trace as answer text; no arm set
//! `thinking_enter_block_on_start`, so the reasoning budget never counted it.

use serde_json::{Value, json};

use super::{Harness, JAMBA_TEMPLATE, harness_with, messages_body, responses_body};

/// Turn-1 generation for a primed prompt: the trace, the close marker, then
/// the answer, with no opener.
const PRIMED_TURN1: &[&str] = &["the capital", "</think>", "\n\n", "Paris."];
const PRIMED_TURN2: &[&str] = &["second trace", "</think>", "\n\n", "Rome."];

/// The Jamba test template with the real checkpoint's primed generation prompt.
fn primed_template() -> String {
    let unprimed = r"{{- '<|im_start|>assistant\n' }}{%- endif -%}";
    let primed = r"{{- '<|im_start|>assistant\n<think>\n' }}{%- endif -%}";
    assert!(JAMBA_TEMPLATE.contains(unprimed), "template shape changed");
    JAMBA_TEMPLATE.replace(unprimed, primed)
}

fn primed_harness() -> Harness {
    harness_with(&primed_template(), false)
}

/// Every JSON payload of an SSE body, in order.
fn sse_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str(data.trim()).ok())
        .collect()
}

/// Concatenated `delta` strings of the Responses events of type `ty`.
fn responses_deltas(events: &[Value], ty: &str) -> String {
    events
        .iter()
        .filter(|e| e["type"] == ty)
        .filter_map(|e| e["delta"].as_str())
        .collect()
}

/// Concatenated Anthropic `content_block_delta` payloads of `delta.type == ty`
/// read from `field`.
fn messages_deltas(events: &[Value], ty: &str, field: &str) -> String {
    events
        .iter()
        .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == ty)
        .filter_map(|e| e["delta"][field].as_str())
        .collect()
}

impl Harness {
    /// Post and also return whether the scheduler was told the prompt left
    /// generation inside an open thinking block.
    async fn post_primed(
        &self,
        path: &str,
        body: Value,
        tokens: &[&str],
    ) -> (String, String, bool) {
        let (prompt, body) = self.post(path, body, tokens).await;
        let options = self.options.recv().expect("options dispatched");
        (prompt, body, options.thinking_enter_block_on_start)
    }
}

/// Streaming `/v1/responses` routes the primed trace into a `reasoning` item
/// and leaves only the answer in `output_text`, as the non-streaming arm does.
#[tokio::test]
async fn responses_stream_routes_primed_reasoning_to_reasoning_events() {
    let h = primed_harness();
    let (prompt, body, enter_block) = h
        .post_primed(
            "/v1/responses",
            responses_body(true, json!("q1")),
            PRIMED_TURN1,
        )
        .await;
    assert!(prompt.ends_with("<think>\n"), "{prompt:?}");

    let events = sse_events(&body);
    let reasoning = responses_deltas(&events, "response.reasoning_text.delta");
    let text = responses_deltas(&events, "response.output_text.delta");
    assert_eq!(reasoning.trim(), "the capital", "{body}");
    assert_eq!(text.trim(), "Paris.", "{body}");
    let completed = events
        .iter()
        .find(|e| e["type"] == "response.completed")
        .expect("completed event");
    assert_eq!(completed["response"]["output"][0]["type"], "reasoning");
    assert_eq!(completed["response"]["output"][1]["type"], "message");
    assert!(enter_block, "the reasoning budget must count primed tokens");
}

/// Streaming `/v1/messages` with extended thinking opens a `thinking` block
/// for the primed trace and keeps the `text` block to the answer.
#[tokio::test]
async fn messages_stream_routes_primed_reasoning_to_a_thinking_block() {
    let h = primed_harness();
    let (prompt, body, enter_block) = h
        .post_primed(
            "/v1/messages",
            messages_body(true, json!([{"role": "user", "content": "q1"}])),
            PRIMED_TURN1,
        )
        .await;
    assert!(prompt.ends_with("<think>\n"), "{prompt:?}");

    let events = sse_events(&body);
    let starts: Vec<&str> = events
        .iter()
        .filter(|e| e["type"] == "content_block_start")
        .filter_map(|e| e["content_block"]["type"].as_str())
        .collect();
    assert_eq!(starts, ["thinking", "text"], "{body}");
    let thinking = messages_deltas(&events, "thinking_delta", "thinking");
    let text = messages_deltas(&events, "text_delta", "text");
    assert_eq!(thinking.trim(), "the capital", "{body}");
    assert_eq!(text.trim(), "Paris.", "{body}");
    assert!(enter_block, "the reasoning budget must count primed tokens");
}

/// The non-streaming arms already split the trace; they also have to arm the
/// scheduler's budget accounting for a primed prompt.
#[tokio::test]
async fn non_streaming_arms_arm_the_budget_for_a_primed_prompt() {
    let h = primed_harness();
    let (_, body, enter_block) = h
        .post_primed(
            "/v1/responses",
            responses_body(false, json!("q1")),
            PRIMED_TURN1,
        )
        .await;
    assert!(enter_block, "/v1/responses non-streaming");
    let response: Value = serde_json::from_str(&body).expect("json");
    assert_eq!(
        response["output_text"].as_str().map(str::trim),
        Some("Paris.")
    );

    let (_, body, enter_block) = h
        .post_primed(
            "/v1/messages",
            messages_body(false, json!([{"role": "user", "content": "q1"}])),
            PRIMED_TURN1,
        )
        .await;
    assert!(enter_block, "/v1/messages non-streaming");
    let message: Value = serde_json::from_str(&body).expect("json");
    assert_eq!(message["content"][0]["type"], "thinking", "{body}");
}

/// An unprimed prompt leaves the budget flag off on both endpoints.
#[tokio::test]
async fn unprimed_prompt_leaves_the_budget_flag_off() {
    let h = harness_with(JAMBA_TEMPLATE, false);
    for stream in [false, true] {
        let tokens = ["<think>", "t", "</think>", "Paris."];
        let (_, _, enter_block) = h
            .post_primed(
                "/v1/responses",
                responses_body(stream, json!("q1")),
                &tokens,
            )
            .await;
        assert!(!enter_block, "/v1/responses stream={stream}");
        let (_, _, enter_block) = h
            .post_primed(
                "/v1/messages",
                messages_body(stream, json!([{"role": "user", "content": "q1"}])),
                &tokens,
            )
            .await;
        assert!(!enter_block, "/v1/messages stream={stream}");
    }
}

/// Turn 3 of a content-only conversation; both earlier user turns render as
/// generated (they keep the thinking instruction) only when both streamed
/// turns recorded non-empty reasoning (#2121). This is what keeps the prompt
/// prefix stable across turns, so the prompt cache can hit.
fn third_turn_prefix() -> &'static str {
    "<|im_start|>user\nTHINK_PREFIX\nq1<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>\n<|im_start|>user\nTHINK_PREFIX\nq2<|im_end|>\n"
}

fn responses_turn(history: &[(&str, &str)], next: &str) -> Value {
    let mut input: Vec<Value> = Vec::new();
    for (q, a) in history {
        input.push(json!({"type": "message", "role": "user", "content": q}));
        input.push(json!({"type": "message", "role": "assistant", "content": a}));
    }
    input.push(json!({"type": "message", "role": "user", "content": next}));
    responses_body(true, Value::Array(input))
}

fn messages_turn(history: &[(&str, &str)], next: &str) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    for (q, a) in history {
        messages.push(json!({"role": "user", "content": q}));
        messages.push(json!({"role": "assistant", "content": a}));
    }
    messages.push(json!({"role": "user", "content": next}));
    messages_body(true, Value::Array(messages))
}

#[tokio::test]
async fn responses_primed_stream_records_reasoning_for_a_three_turn_chat() {
    let h = primed_harness();
    h.post("/v1/responses", responses_turn(&[], "q1"), PRIMED_TURN1)
        .await;
    let (prompt2, _) = h
        .post(
            "/v1/responses",
            responses_turn(&[("q1", "Paris.")], "q2"),
            PRIMED_TURN2,
        )
        .await;
    assert!(prompt2.starts_with(super::GENERATED_TURN1), "{prompt2:?}");
    let (prompt3, _) = h
        .post(
            "/v1/responses",
            responses_turn(&[("q1", "Paris."), ("q2", "Rome.")], "q3"),
            &["ok"],
        )
        .await;
    assert!(prompt3.starts_with(third_turn_prefix()), "{prompt3:?}");
}

#[tokio::test]
async fn messages_primed_stream_records_thinking_for_a_three_turn_chat() {
    let h = primed_harness();
    h.post("/v1/messages", messages_turn(&[], "q1"), PRIMED_TURN1)
        .await;
    let (prompt2, _) = h
        .post(
            "/v1/messages",
            messages_turn(&[("q1", "Paris.")], "q2"),
            PRIMED_TURN2,
        )
        .await;
    assert!(prompt2.starts_with(super::GENERATED_TURN1), "{prompt2:?}");
    let (prompt3, _) = h
        .post(
            "/v1/messages",
            messages_turn(&[("q1", "Paris."), ("q2", "Rome.")], "q3"),
            &["ok"],
        )
        .await;
    assert!(prompt3.starts_with(third_turn_prefix()), "{prompt3:?}");
}
