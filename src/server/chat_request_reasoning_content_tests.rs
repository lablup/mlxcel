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

//! Echoed `reasoning_content` and the AI21 Jamba-Reasoning template (#2089).
//!
//! Jamba-Reasoning-3B's template adds a thinking instruction to the last user
//! turn, and keeps it on an earlier user turn only when the assistant reply
//! that follows carries `reasoning_content` (or opens with `<think>`). The
//! server used to forward an echoed trace under `reasoning` alone, so this
//! template never saw it: the earlier user turn re-rendered without the
//! instruction, the turn-1 history boundary stopped prefixing turn 2, and every
//! follow-up missed the prompt cache.

use super::{build_raw_json_messages, prepare_chat_request_with_cache};
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::types::ChatCompletionRequest;
use crate::tokenizer::ThinkingMarkers;

/// The user/assistant branches of the shipped Jamba-Reasoning template
/// (`chat_template.jinja` of `AI21-Jamba-Reasoning-3B`), with the instruction
/// shortened to `THINK_PREFIX`. The prefix rule is copied verbatim.
fn jamba_reasoning_template() -> String {
    r#"{%- set thinking_prefix = 'THINK_PREFIX\n' -%}
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
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n' }}{%- endif -%}"#
        .to_string()
}

fn request(messages: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({ "model": "jamba", "messages": messages }))
        .expect("valid chat request")
}

async fn render(
    processor: &ChatTemplateProcessor,
    req: &ChatCompletionRequest,
) -> (String, Option<String>) {
    let prepared = prepare_chat_request_with_cache(
        processor,
        req,
        None,
        true,
        true,
        false,
        &ThinkingMarkers::default(),
    )
    .await
    .expect("render succeeds");
    (prepared.prompt, prepared.history_prompt)
}

#[test]
fn echoed_reasoning_content_is_forwarded_under_both_spellings() {
    let req = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris.", "reasoning_content": "trace"},
        {"role": "user", "content": "q2"},
    ]));
    let raw = build_raw_json_messages(&req);
    let assistant = &raw.as_array().expect("array")[1];
    assert_eq!(assistant["reasoning_content"], "trace");
    assert_eq!(assistant["reasoning"], "trace");
}

#[tokio::test]
async fn jamba_history_boundary_prefixes_the_next_turn_when_reasoning_is_echoed() {
    let processor = ChatTemplateProcessor::with_template(jamba_reasoning_template());
    let turn1 = request(serde_json::json!([{"role": "user", "content": "q1"}]));
    let (_, history1) = render(&processor, &turn1).await;
    let history1 = history1.expect("prompt cache on: history boundary rendered");
    assert_eq!(history1, "<|im_start|>user\nTHINK_PREFIX\nq1<|im_end|>\n");

    let turn2 = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris.", "reasoning_content": "trace"},
        {"role": "user", "content": "q2"},
    ]));
    let (prompt2, _) = render(&processor, &turn2).await;
    assert!(
        prompt2.starts_with(&history1),
        "turn-1 boundary must prefix turn 2 when the client echoes reasoning_content;\n\
         boundary: {history1:?}\nturn 2: {prompt2:?}"
    );
}

/// Pins the renderer's side of the limit: a request whose assistant turn
/// carries only `content` gets an earlier user turn without the instruction.
/// That rewrite is the template's own rule. The chat routes avoid it by
/// re-injecting the trace they generated before rendering (issue #2110, see
/// `reasoning_echo_tests.rs`); this renderer sees whatever the route hands it.
#[tokio::test]
async fn jamba_content_only_echo_rewrites_the_earlier_user_turn() {
    let processor = ChatTemplateProcessor::with_template(jamba_reasoning_template());
    let turn1 = request(serde_json::json!([{"role": "user", "content": "q1"}]));
    let (_, history1) = render(&processor, &turn1).await;
    let history1 = history1.expect("history boundary rendered");

    let turn2 = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "q2"},
    ]));
    let (prompt2, _) = render(&processor, &turn2).await;
    assert!(prompt2.starts_with("<|im_start|>user\nq1<|im_end|>\n"));
    assert!(!prompt2.starts_with(&history1));
}

/// Templates that accept both spellings read them as alternatives, so the
/// second key never renders the trace twice.
#[tokio::test]
async fn template_reading_either_spelling_renders_the_trace_once() {
    let processor = ChatTemplateProcessor::with_template(
        "{%- for m in messages %}{{ m.role }}:{{ m.get('reasoning') or m.get('reasoning_content') or '' }}|{{ m.content }}\n{%- endfor %}"
            .to_string(),
    );
    let req = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "TRACE"},
        {"role": "user", "content": "q2"},
    ]));
    let (prompt, _) = render(&processor, &req).await;
    assert_eq!(prompt.matches("TRACE").count(), 1, "{prompt:?}");
}

/// Pins the `finish_reason=length` branch the investigation measured: when
/// `max_tokens` ends the decode inside the primed `<think>` block, the output is
/// all reasoning and `content` is empty, the same as every primed-thinking
/// family. When `</think>` is reached, the answer lands in `content`.
#[test]
fn jamba_primed_think_splits_on_close_and_stays_reasoning_when_truncated() {
    use crate::server::routes::chat::{extract_reasoning_content, is_prompt_primed_open_thinking};
    let markers = ThinkingMarkers {
        think_start: Some("<think>".to_string()),
        think_end: Some("</think>".to_string()),
        think_start_tokens: Some(vec![541]),
        think_end_tokens: Some(vec![542]),
        ..ThinkingMarkers::default()
    };
    let prompt = "<|im_start|>user\nTHINK_PREFIX\nq1<|im_end|>\n<|im_start|>assistant\n<think>\n";
    assert!(is_prompt_primed_open_thinking(&markers, prompt));

    let truncated = "The user asks for the capital. It is Paris. So we";
    assert_eq!(
        extract_reasoning_content(truncated, true).as_deref(),
        Some(truncated)
    );

    let closed = "The capital is Paris.\n</think>\n\nParis.";
    let reasoning = extract_reasoning_content(closed, true).expect("reasoning captured");
    assert!(reasoning.contains("The capital is Paris."));
    assert!(!reasoning.contains("</think>"));
}
