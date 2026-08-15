// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Regression tests for issue #1164: a chat template that rejects one of its
//! own kwargs must fail the request, and the OpenAI-standard top-level
//! `reasoning_effort` field must reach the template.
//!
//! The fixture is the chat template shipped with
//! [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
//! (Apache-2.0), pinned by digest the same way
//! [`super::muse_glimmer_template_tests`] pins its own. It is the first shipped
//! checkpoint template in this tree that *validates* a caller-supplied kwarg:
//!
//! ```jinja
//! {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
//! {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
//!     {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ ' ...') }}
//! {%- endif %}
//! ```
//!
//! Note the accepted set is `{xhigh, medium, low}` while OpenAI's
//! `reasoning_effort` vocabulary is `{minimal, low, medium, high}`, so `high` is
//! valid OpenAI and invalid here. That mismatch is what makes the silent
//! fallback reachable from an ordinary OpenAI-compatible client.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::chat_request::prepare_chat_request;
use super::chat_template::{ChatTemplateProcessor, template_rejection_message};
use super::chat_template_kwargs::ChatTemplateKwargs;
use super::types::{ChatCompletionRequest, Message, MessageContent, Role, SamplingParams};

const QWEN3_8_TEMPLATE: &str = include_str!("../../tests/fixtures/qwen3_8/chat_template.jinja");
const QWEN3_8_TEMPLATE_SHA256: &str =
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";

/// A template with no `reasoning_effort` reference at all: the control for
/// "a checkpoint that ignores the field must be unaffected".
const PLAIN_TEMPLATE: &str = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}";

/// A template that fails for an *engine* reason rather than a deliberate
/// refusal: the filter does not exist, so rendering raises
/// `ErrorKind::UnknownFilter` with no [`super::chat_template::TemplateRejection`]
/// source. This is the arm that must keep falling back, and it is exactly the
/// case a coarser discriminator (matching on `ErrorKind::InvalidOperation`)
/// would have turned into a 400.
const ENGINE_FAILURE_TEMPLATE: &str =
    "{% for m in messages %}{{ m.content | mlxcel_unknown_filter_probe }}{% endfor %}";

fn qwen3_8() -> ChatTemplateProcessor {
    let mut processor = ChatTemplateProcessor::with_template(QWEN3_8_TEMPLATE.to_string());
    // `with_template` leaves `default_enable_thinking` at the conservative
    // `false` used for template-string construction, but the server derives it
    // from the tokenizer's think markers in `startup::resolve_chat_template`
    // and Qwen3.8 is a thinking checkpoint, so it is `true` in production. The
    // template wraps its entire reasoning-effort block in
    // `{%- if enable_thinking is undefined or enable_thinking is true %}`, so
    // leaving the default at `false` would skip the branch under test and every
    // effort value would render identically.
    processor.set_default_enable_thinking(true);
    processor
}

fn kwargs(pairs: &[(&str, Value)]) -> ChatTemplateKwargs {
    let mut map = Map::new();
    for (key, value) in pairs {
        map.insert((*key).to_string(), value.clone());
    }
    ChatTemplateKwargs::from_json_object(map)
}

fn user_hi() -> Vec<Message> {
    vec![Message {
        role: Role::User,
        content: MessageContent::Text("hi".to_string()),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning: None,
    }]
}

/// A request carrying `messages: [user "hi"]` and nothing else set.
fn request(messages: Vec<Message>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "qwen3.8-27b-4bit".to_string(),
        messages,
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        chat_template_kwargs: None,
        extra_body: None,
        prompt_cache_key: None,
        user: None,
        reasoning_effort: None,
        extra_body_fields: Map::new(),
        response_format: None,
        params: SamplingParams::default(),
    }
}

fn with_kwargs(mut req: ChatCompletionRequest, pairs: &[(&str, Value)]) -> ChatCompletionRequest {
    let mut map = Map::new();
    for (key, value) in pairs {
        map.insert((*key).to_string(), value.clone());
    }
    req.chat_template_kwargs = Some(map);
    req
}

async fn prompt_for(
    processor: &ChatTemplateProcessor,
    req: &ChatCompletionRequest,
) -> anyhow::Result<String> {
    prepare_chat_request(processor, req, None)
        .await
        .map(|prepared| prepared.prompt)
}

// ---------------------------------------------------------------------------
// Fixture pin
// ---------------------------------------------------------------------------

#[test]
fn qwen3_8_template_fixture_matches_pinned_digest() {
    let digest = Sha256::digest(QWEN3_8_TEMPLATE.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex, QWEN3_8_TEMPLATE_SHA256,
        "the vendored Qwen3.8 chat template changed; re-check the accepted \
         reasoning_effort set before updating this digest"
    );
    assert!(
        QWEN3_8_TEMPLATE.contains("reasoning_effort"),
        "fixture must be the reasoning-effort-validating template"
    );
}

// ---------------------------------------------------------------------------
// Defect 1: template rejection vs engine failure
// ---------------------------------------------------------------------------

/// The rendered prompt must actually move with the effort value, otherwise the
/// rest of this file would be asserting against a template that ignores it.
/// This is the in-process form of the `prompt_tokens` column measured against
/// the running server in issue #1164.
#[test]
fn accepted_reasoning_effort_values_change_the_rendered_prompt() {
    let processor = qwen3_8();
    let messages = json!([{"role": "user", "content": "hi"}]);

    let render = |effort: Option<&str>| -> String {
        let kw = match effort {
            Some(v) => kwargs(&[("reasoning_effort", json!(v))]),
            None => ChatTemplateKwargs::new(),
        };
        processor
            .apply_raw_with_kwargs(&messages, None, &kw)
            .expect("accepted reasoning_effort must render")
    };

    let unset = render(None);
    let xhigh = render(Some("xhigh"));
    let medium = render(Some("medium"));
    let low = render(Some("low"));

    // Unset resolves to the template's own `default('xhigh')`.
    assert_eq!(unset, xhigh, "unset must resolve to the xhigh default");
    // The template injects a reasoning-instruction system message whose length
    // is a direct function of the resolved effort: xhigh > low > medium, with
    // medium injecting none at all.
    assert!(
        xhigh.len() > low.len(),
        "xhigh prompt ({}) must be longer than low ({})",
        xhigh.len(),
        low.len()
    );
    assert!(
        low.len() > medium.len(),
        "low prompt ({}) must be longer than medium ({})",
        low.len(),
        medium.len()
    );
    assert!(xhigh.contains("Reasoning effort is set to xhigh"));
    assert!(low.contains("Reasoning effort is set to low"));
    assert!(!medium.contains("Reasoning effort is set to"));
}

#[test]
fn rejected_reasoning_effort_is_recognised_as_a_template_rejection() {
    let processor = qwen3_8();
    let messages = json!([{"role": "user", "content": "hi"}]);

    // `high` is OpenAI-valid and Qwen3.8-invalid. `HIGH` covers the casing
    // variant measured in the issue.
    for value in ["high", "HIGH", "minimal", ""] {
        let err = processor
            .apply_raw_with_kwargs(
                &messages,
                None,
                &kwargs(&[("reasoning_effort", json!(value))]),
            )
            .expect_err("template must refuse an unsupported reasoning effort");
        let message = template_rejection_message(&err)
            .unwrap_or_else(|| panic!("`{value}` must be recognised as a template rejection"));
        assert!(
            message.contains("Supported types are xhigh (default), medium, and low."),
            "rejection message must name the accepted set, got: {message}"
        );
    }
}

/// The other half of the discriminator: an engine-side failure carries
/// `ErrorKind` values that overlap a deliberate refusal, so keying on the kind
/// would misclassify it. It must not be reported as a rejection.
#[test]
fn engine_failures_are_not_template_rejections() {
    let messages = json!([{"role": "user", "content": "hi"}]);

    let unknown_filter = ChatTemplateProcessor::with_template(ENGINE_FAILURE_TEMPLATE.to_string())
        .apply_raw(&messages, None)
        .expect_err("unknown filter must fail to render");
    assert!(
        template_rejection_message(&unknown_filter).is_none(),
        "an unknown filter is an engine failure, not a template rejection"
    );

    let malformed = ChatTemplateProcessor::with_template("{% for m in messages %}".to_string())
        .apply_raw(&messages, None)
        .expect_err("unclosed block must fail to parse");
    assert!(
        template_rejection_message(&malformed).is_none(),
        "a malformed template is an engine failure, not a template rejection"
    );
}

#[tokio::test]
async fn template_rejection_fails_the_request_instead_of_returning_a_fallback_prompt() {
    let processor = qwen3_8();
    let req = with_kwargs(request(user_hi()), &[("reasoning_effort", json!("high"))]);

    let err = prompt_for(&processor, &req)
        .await
        .expect_err("a rejected kwarg must fail the request, not degrade the prompt");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("chat template rejected this request"),
        "error must say the template refused the request, got: {rendered}"
    );
    assert!(
        rendered.contains("Supported types are xhigh (default), medium, and low."),
        "error must carry the template's own message, got: {rendered}"
    );
}

/// Same rejection through the raw/multimodal render arm. A message carrying a
/// prior-turn `reasoning` field routes preparation through
/// `apply_raw_with_kwargs`, which has its own structurally identical fallback
/// site; fixing one arm and not the other would leave the defect reachable.
#[tokio::test]
async fn template_rejection_also_fails_on_the_raw_render_path() {
    let processor = qwen3_8();
    let messages = vec![
        Message {
            role: Role::User,
            content: MessageContent::Text("hi".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
        },
        Message {
            role: Role::Assistant,
            content: MessageContent::Text("hello".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: Some("prior thinking".to_string()),
        },
        Message {
            role: Role::User,
            content: MessageContent::Text("again".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
        },
    ];
    let req = with_kwargs(request(messages), &[("reasoning_effort", json!("high"))]);

    let err = prompt_for(&processor, &req)
        .await
        .expect_err("the raw render path must reject too");
    assert!(
        format!("{err:#}").contains("chat template rejected this request"),
        "raw path must produce the same rejection"
    );
}

/// The conservative half of the change: a template mlxcel genuinely cannot
/// render still degrades to the plain prompt rather than failing the request.
#[tokio::test]
async fn engine_render_failure_still_falls_back() {
    let processor = ChatTemplateProcessor::with_template(ENGINE_FAILURE_TEMPLATE.to_string());
    let prompt = prompt_for(&processor, &request(user_hi()))
        .await
        .expect("an engine failure must still serve the request from the fallback prompt");
    assert_eq!(prompt, "User: hi\n\nAssistant: ");
}

// ---------------------------------------------------------------------------
// Defect 2: top-level `reasoning_effort`
// ---------------------------------------------------------------------------

#[tokio::test]
async fn top_level_reasoning_effort_reaches_the_template() {
    let processor = qwen3_8();

    let mut req = request(user_hi());
    req.reasoning_effort = Some("low".to_string());
    let mapped = prompt_for(&processor, &req).await.expect("low must render");

    let via_kwargs = prompt_for(
        &processor,
        &with_kwargs(request(user_hi()), &[("reasoning_effort", json!("low"))]),
    )
    .await
    .expect("low must render");

    assert_eq!(
        mapped, via_kwargs,
        "the top-level field must produce the same prompt as the explicit kwarg"
    );

    let unset = prompt_for(&processor, &request(user_hi()))
        .await
        .expect("unset must render");
    assert_ne!(
        mapped, unset,
        "the top-level field must actually change the prompt"
    );
}

#[tokio::test]
async fn chat_template_kwargs_outrank_the_top_level_field() {
    let processor = qwen3_8();

    let mut req = with_kwargs(request(user_hi()), &[("reasoning_effort", json!("low"))]);
    req.reasoning_effort = Some("medium".to_string());
    let resolved = prompt_for(&processor, &req).await.expect("must render");

    let expected = prompt_for(
        &processor,
        &with_kwargs(request(user_hi()), &[("reasoning_effort", json!("low"))]),
    )
    .await
    .expect("must render");

    assert_eq!(
        resolved, expected,
        "an explicit chat_template_kwargs.reasoning_effort must win"
    );
}

/// No value translation. `high` is the single most likely thing an
/// OpenAI-compatible client sends, and remapping it to the template's `xhigh`
/// would silently pick a reasoning budget the caller did not ask for.
#[tokio::test]
async fn top_level_high_is_rejected_rather_than_translated() {
    let processor = qwen3_8();
    let mut req = request(user_hi());
    req.reasoning_effort = Some("high".to_string());

    let err = prompt_for(&processor, &req)
        .await
        .expect_err("`high` must not be silently remapped to `xhigh`");
    let rendered = format!("{err:#}");
    assert!(rendered.contains("Supported types are xhigh (default), medium, and low."));

    // Prove the remap did not happen by another route: the xhigh prompt renders
    // fine, so an accidental translation would have produced Ok.
    let xhigh = prompt_for(
        &processor,
        &with_kwargs(request(user_hi()), &[("reasoning_effort", json!("xhigh"))]),
    )
    .await
    .expect("xhigh must render");
    assert!(xhigh.contains("Reasoning effort is set to xhigh"));
}

/// A checkpoint whose template never reads `reasoning_effort` must be
/// completely unaffected: no injected kwarg, no new rejection path.
#[tokio::test]
async fn template_without_reasoning_effort_is_unaffected() {
    let processor = ChatTemplateProcessor::with_template(PLAIN_TEMPLATE.to_string());

    let baseline = prompt_for(&processor, &request(user_hi()))
        .await
        .expect("plain template must render");

    for value in ["low", "high", "xhigh", "nonsense"] {
        let mut req = request(user_hi());
        req.reasoning_effort = Some(value.to_string());
        let prompt = prompt_for(&processor, &req).await.unwrap_or_else(|e| {
            panic!("`{value}` must not fail a template that ignores it: {e:#}")
        });
        assert_eq!(
            prompt, baseline,
            "`{value}` must not change the prompt of a template that ignores the kwarg"
        );
    }
}

/// The OpenAI Python SDK merges `extra_body={...}` into the request root and
/// other callers send a nested `extra_body` object. Both spellings resolve the
/// same way `prompt_cache_key` and `user` already do.
#[tokio::test]
async fn extra_body_spellings_of_reasoning_effort_resolve() {
    let processor = qwen3_8();
    let expected = prompt_for(
        &processor,
        &with_kwargs(request(user_hi()), &[("reasoning_effort", json!("low"))]),
    )
    .await
    .expect("must render");

    let mut flattened = request(user_hi());
    flattened
        .extra_body_fields
        .insert("reasoning_effort".to_string(), json!("low"));
    assert_eq!(
        prompt_for(&processor, &flattened)
            .await
            .expect("must render"),
        expected,
        "flattened OpenAI-SDK extra_body must resolve"
    );

    let mut nested = request(user_hi());
    let mut body = Map::new();
    body.insert("reasoning_effort".to_string(), json!("low"));
    nested.extra_body = Some(body);
    assert_eq!(
        prompt_for(&processor, &nested).await.expect("must render"),
        expected,
        "nested extra_body must resolve"
    );
}

/// The template gates its reasoning-effort validation on `enable_thinking`, so
/// with thinking off it never inspects the value and there is nothing to
/// refuse. Pinned because it is the template's own semantics, not mlxcel's: the
/// 400 follows the template's decision rather than pre-validating the value
/// against a vocabulary mlxcel keeps on the side.
#[tokio::test]
async fn effort_is_inert_when_the_template_has_thinking_disabled() {
    let processor = qwen3_8();
    let mut req = with_kwargs(
        request(user_hi()),
        &[
            ("reasoning_effort", json!("high")),
            ("enable_thinking", json!(false)),
        ],
    );
    req.reasoning_effort = None;
    let prompt = prompt_for(&processor, &req)
        .await
        .expect("thinking off means the template never validates the effort");
    assert!(!prompt.contains("Reasoning effort is set to"));
}

#[test]
fn reasoning_effort_is_not_a_reserved_context_key() {
    // The reserved-key guard in `build_template_context` drops keys that would
    // let a client replace the conversation or the tool list. `reasoning_effort`
    // is not one of them, so it must survive into the Jinja context; the render
    // below is the proof, since the template only branches on a value it saw.
    let processor = qwen3_8();
    let rendered = processor
        .apply_raw_with_kwargs(
            &json!([{"role": "user", "content": "hi"}]),
            None,
            &kwargs(&[("reasoning_effort", json!("low"))]),
        )
        .expect("must render");
    assert!(rendered.contains("Reasoning effort is set to low"));
}
