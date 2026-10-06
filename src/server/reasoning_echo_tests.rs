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

//! Reasoning re-injection for content-only history (issue #2110).

use std::borrow::Cow;

use super::{ReasoningEchoScope, ReasoningEchoStore};
use crate::server::chat_request::prepare_chat_request_with_cache;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::types::ChatCompletionRequest;
use crate::tokenizer::ThinkingMarkers;

/// The user/assistant branches of the shipped Jamba-Reasoning template, with
/// the instruction shortened to `THINK_PREFIX` (same reduction as the #2089
/// tests in `chat_request_reasoning_content_tests.rs`).
pub(super) const JAMBA_TEMPLATE: &str = r#"{%- set thinking_prefix = 'THINK_PREFIX\n' -%}
{%- for message in messages %}
{%- if message.role == "user" %}
{%- set prefix = '' %}
{%- if '<think>' not in message.content %}
{%- if loop.last %}{%- set prefix = thinking_prefix %}{%- endif %}
{%- if not loop.last %}
{%- if loop.nextitem.role == 'assistant' and loop.nextitem.content.startswith('<think>') or loop.nextitem.reasoning_content is defined and loop.nextitem.reasoning_content is not none %}
{%- set prefix = thinking_prefix %}
{%- endif %}
{%- endif %}
{%- endif %}
{{- '<|im_start|>user\n' + prefix + message.content + '<|im_end|>\n' }}
{%- elif message.role == "assistant" %}
{{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif -%}"#;

fn request(messages: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({ "model": "jamba", "messages": messages }))
        .expect("valid chat request")
}

fn scope() -> ReasoningEchoScope {
    ReasoningEchoScope {
        model_id: "m".to_string(),
        template_sig: "sig".to_string(),
        session_key: "__mlxcel_anon__".to_string(),
    }
}

fn store() -> ReasoningEchoStore {
    ReasoningEchoStore::new(1 << 20, 64)
}

/// Turn 1 as the client sent it, and turn 2 echoing only `content`.
fn turn1() -> ChatCompletionRequest {
    request(serde_json::json!([{"role": "user", "content": "q1"}]))
}

fn turn2_content_only() -> ChatCompletionRequest {
    request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "q2"},
    ]))
}

async fn render(processor: &ChatTemplateProcessor, req: &ChatCompletionRequest) -> String {
    prepare_chat_request_with_cache(
        processor,
        req,
        None,
        true,
        true,
        false,
        &ThinkingMarkers::default(),
    )
    .await
    .expect("render succeeds")
    .prompt
}

fn reasoning_at(req: &ChatCompletionRequest, idx: usize) -> Option<&str> {
    req.messages[idx].reasoning.as_deref()
}

#[tokio::test]
async fn stored_content_only_turn_renders_identically_to_the_echoed_form() {
    let store = store();
    store.record(&scope(), &turn1().messages, "Paris.", "trace");

    let content_only = turn2_content_only();
    let filled = store.fill(&scope(), &content_only);
    assert!(
        matches!(filled, Cow::Owned(_)),
        "stored turn must be filled"
    );
    assert_eq!(reasoning_at(&filled, 1), Some("trace"));

    let echoed = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris.", "reasoning_content": "trace"},
        {"role": "user", "content": "q2"},
    ]));
    let processor = ChatTemplateProcessor::with_template(JAMBA_TEMPLATE.to_string());
    let filled_prompt = render(&processor, &filled).await;
    assert_eq!(filled_prompt, render(&processor, &echoed).await);
    assert!(
        filled_prompt.starts_with("<|im_start|>user\nTHINK_PREFIX\nq1<|im_end|>\n"),
        "the earlier user turn keeps its instruction: {filled_prompt:?}"
    );
}

#[test]
fn unstored_turn_is_unchanged() {
    let store = store();
    store.record(&scope(), &turn1().messages, "Rome.", "other trace");
    let req = turn2_content_only();
    assert!(matches!(store.fill(&scope(), &req), Cow::Borrowed(_)));
}

#[test]
fn echoed_reasoning_wins_over_the_stored_trace() {
    let store = store();
    store.record(&scope(), &turn1().messages, "Paris.", "stored");
    let req = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris.", "reasoning_content": "client"},
        {"role": "user", "content": "q2"},
    ]));
    let filled = store.fill(&scope(), &req);
    assert!(matches!(filled, Cow::Borrowed(_)));
    assert_eq!(reasoning_at(&filled, 1), Some("client"));
}

#[test]
fn edited_assistant_content_is_not_filled() {
    let store = store();
    store.record(&scope(), &turn1().messages, "Paris.", "trace");
    let req = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris, France."},
        {"role": "user", "content": "q2"},
    ]));
    assert!(matches!(store.fill(&scope(), &req), Cow::Borrowed(_)));
}

/// A streamed reply carries the blank line after the close marker in
/// `content`; the non-streaming reply and most clients trim it. Either echo
/// finds the entry.
#[test]
fn surrounding_whitespace_does_not_change_the_key() {
    let store = store();
    store.record(&scope(), &turn1().messages, "\n\nParis.", "trace");
    let req = turn2_content_only();
    let filled = store.fill(&scope(), &req);
    assert_eq!(reasoning_at(&filled, 1), Some("trace"));
}

#[test]
fn edited_earlier_message_is_not_filled() {
    let store = store();
    store.record(&scope(), &turn1().messages, "Paris.", "trace");
    let req = request(serde_json::json!([
        {"role": "user", "content": "q1 (edited)"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "q2"},
    ]));
    assert!(matches!(store.fill(&scope(), &req), Cow::Borrowed(_)));
}

#[test]
fn another_session_template_or_model_never_sees_the_trace() {
    let store = store();
    store.record(&scope(), &turn1().messages, "Paris.", "trace");
    let req = turn2_content_only();
    let mut other_session = scope();
    other_session.session_key = "alice".to_string();
    let mut other_template = scope();
    other_template.template_sig = "sig-with-enable_thinking-false".to_string();
    let mut other_model = scope();
    other_model.model_id = "other".to_string();
    for other in [other_session, other_template, other_model] {
        assert!(
            matches!(store.fill(&other, &req), Cow::Borrowed(_)),
            "{other:?} must not see the trace"
        );
    }
}

#[test]
fn tool_call_and_inline_think_turns_are_never_filled() {
    let store = store();
    store.record(
        &scope(),
        &turn1().messages,
        "<think>x</think>Paris.",
        "trace",
    );
    let inline = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>x</think>Paris."},
        {"role": "user", "content": "q2"},
    ]));
    assert!(matches!(store.fill(&scope(), &inline), Cow::Borrowed(_)));

    store.record(&scope(), &turn1().messages, "Paris.", "trace");
    let with_tools = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris.", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "f", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]));
    assert!(matches!(
        store.fill(&scope(), &with_tools),
        Cow::Borrowed(_)
    ));
}

#[test]
fn prefill_continuations_and_blank_reasoning_are_not_recorded() {
    let store = store();
    let prefill = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Par"},
    ]));
    store.record(&scope(), &prefill.messages, "is.", "trace");
    store.record(&scope(), &turn1().messages, "Paris.", "\n");
    assert_eq!(store.len(), 0);
}

/// A reply `max_tokens` cut off inside its thinking block has empty
/// `content`; the client echoes an empty assistant turn. It is filled only for
/// the conversation that produced it.
#[test]
fn reasoning_only_reply_is_filled_only_for_its_own_conversation() {
    let store = store();
    store.record(&scope(), &turn1().messages, "", "truncated trace");
    let empty_echo = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "q2"},
    ]));
    let filled = store.fill(&scope(), &empty_echo);
    assert_eq!(reasoning_at(&filled, 1), Some("truncated trace"));

    let other_conversation = request(serde_json::json!([
        {"role": "user", "content": "another question"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "q2"},
    ]));
    assert!(matches!(
        store.fill(&scope(), &other_conversation),
        Cow::Borrowed(_)
    ));
}

/// Turn 3 looks every earlier assistant turn up under the conversation the
/// client sent, so turn 1's re-injected trace does not disturb the key of
/// turn 2's entry.
#[test]
fn every_earlier_turn_is_filled_on_turn_three() {
    let store = store();
    store.record(&scope(), &turn1().messages, "Paris.", "trace one");
    store.record(
        &scope(),
        &turn2_content_only().messages,
        "Berlin.",
        "trace two",
    );
    let turn3 = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "Berlin."},
        {"role": "user", "content": "q3"},
    ]));
    let filled = store.fill(&scope(), &turn3);
    assert_eq!(reasoning_at(&filled, 1), Some("trace one"));
    assert_eq!(reasoning_at(&filled, 3), Some("trace two"));
}

fn single_turn(q: &str) -> ChatCompletionRequest {
    request(serde_json::json!([{"role": "user", "content": q}]))
}

fn follow_up(q: &str, a: &str) -> ChatCompletionRequest {
    request(serde_json::json!([
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
        {"role": "user", "content": "next"},
    ]))
}

#[test]
fn entry_cap_evicts_least_recently_used_and_hits_refresh() {
    let store = ReasoningEchoStore::new(1 << 20, 3);
    for q in ["a", "b", "c"] {
        store.record(&scope(), &single_turn(q).messages, q, "trace");
    }
    // A hit on "a" makes "b" the least recently used.
    assert!(matches!(
        store.fill(&scope(), &follow_up("a", "a")),
        Cow::Owned(_)
    ));
    store.record(&scope(), &single_turn("d").messages, "d", "trace");
    assert_eq!(store.len(), 3);
    assert!(matches!(
        store.fill(&scope(), &follow_up("b", "b")),
        Cow::Borrowed(_)
    ));
    for q in ["a", "c", "d"] {
        assert!(
            matches!(store.fill(&scope(), &follow_up(q, q)), Cow::Owned(_)),
            "{q} should survive"
        );
    }
}

#[test]
fn byte_budget_bounds_the_store() {
    let trace = "x".repeat(200);
    let cost = trace.len() + super::ENTRY_OVERHEAD_BYTES;
    let store = ReasoningEchoStore::new(cost * 8, 64);
    for i in 0..20 {
        let q = format!("q{i}");
        store.record(&scope(), &single_turn(&q).messages, &q, &trace);
    }
    assert_eq!(store.len(), 8);
    assert!(store.bytes() <= cost * 8);
    // The newest survive, the oldest are gone.
    assert!(matches!(
        store.fill(&scope(), &follow_up("q19", "q19")),
        Cow::Owned(_)
    ));
    assert!(matches!(
        store.fill(&scope(), &follow_up("q0", "q0")),
        Cow::Borrowed(_)
    ));
}

#[test]
fn oversized_trace_and_disabled_store_keep_nothing() {
    let store = ReasoningEchoStore::new(8 * 1024, 64);
    store.record(&scope(), &turn1().messages, "Paris.", &"x".repeat(2048));
    assert_eq!(store.len(), 0, "a trace above a budget eighth is refused");

    let disabled = ReasoningEchoStore::new(0, 64);
    disabled.record(&scope(), &turn1().messages, "Paris.", "trace");
    assert_eq!(disabled.len(), 0);
    assert!(!disabled.enabled());
}

#[path = "reasoning_echo_route_tests.rs"]
mod route;

#[path = "reasoning_echo_endpoint_tests.rs"]
mod endpoint;

#[path = "reasoning_echo_warmup_tests.rs"]
mod warmup;
