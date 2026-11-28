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

use serde_json::json;

use super::{apply_to_chat_body, cli_template_kwargs, level_value, parse_reasoning_effort};
use crate::server::chat_template::{ChatMessage, ChatTemplateProcessor};

/// The effort macro of Inkling 2's `chat_template.jinja`, verbatim, followed by
/// a one-turn body.
const INKLING2_EFFORT_TEMPLATE: &str = r#"{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}
{%- macro emit_thinking_effort() -%}
    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
    {%- if eff is string -%}
        {%- set key = eff | trim -%}
        {%- if key not in effort_map -%}
            {{- raise_exception("Unknown reasoning_effort: " ~ eff) -}}
        {%- endif -%}
        {%- set num = effort_map[key] -%}
    {%- else -%}
        {%- set num = eff | float -%}
    {%- endif -%}
    {%- if num < 0.0 or num > 0.99 -%}
        {{- raise_exception("reasoning_effort must be in [0.0, 0.99]") -}}
    {%- endif -%}
    {{- "<|message_system|><|content_text|>Thinking effort level: " -}}
    {%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}
    {{- "<|end_message|>" -}}
{%- endmacro -%}
{{- emit_thinking_effort() -}}
{%- for m in messages -%}<|message_user|>{{ m.content }}{%- endfor -%}"#;

/// A ChatML template that reads neither a reasoning level nor `enable_thinking`.
const QWEN3_STYLE_TEMPLATE: &str = "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor -%}<|im_start|>assistant\n";

/// A template that, like Qwen3's, reads `enable_thinking` and no level.
const THINKING_SWITCH_TEMPLATE: &str = "{{ messages[0].content }}|{{ 'on' if enable_thinking is not defined or enable_thinking else 'off' }}";

fn render(template: &str, level: Option<&str>) -> anyhow::Result<String> {
    let processor = ChatTemplateProcessor::with_template(template.to_string());
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: "hi".to_string(),
    }];
    let kwargs = match level {
        Some(level) => cli_template_kwargs(&processor, &parse_reasoning_effort(level).unwrap())?,
        None => Default::default(),
    };
    processor.apply_with_kwargs(&messages, None, &kwargs)
}

#[test]
fn inkling2_none_renders_effort_zero() {
    let prompt = render(INKLING2_EFFORT_TEMPLATE, Some("none")).unwrap();
    assert!(
        prompt.contains("Thinking effort level: 0<|end_message|>"),
        "{prompt}"
    );
}

#[test]
fn inkling2_numeric_level_renders_the_number() {
    let prompt = render(INKLING2_EFFORT_TEMPLATE, Some("0.5")).unwrap();
    assert!(
        prompt.contains("Thinking effort level: 0.5<|end_message|>"),
        "{prompt}"
    );
}

#[test]
fn inkling2_named_level_renders_its_mapped_number() {
    let prompt = render(INKLING2_EFFORT_TEMPLATE, Some(" XHigh ")).unwrap();
    assert!(
        prompt.contains("Thinking effort level: 0.99<|end_message|>"),
        "{prompt}"
    );
}

#[test]
fn inkling2_without_the_flag_keeps_the_template_default() {
    let prompt = render(INKLING2_EFFORT_TEMPLATE, None).unwrap();
    assert!(
        prompt.contains("Thinking effort level: 0.9<|end_message|>"),
        "{prompt}"
    );
}

#[test]
fn inkling2_invalid_level_surfaces_the_template_message() {
    let err = render(INKLING2_EFFORT_TEMPLATE, Some("bogus")).unwrap_err();
    assert!(
        format!("{err:#}").contains("Unknown reasoning_effort: bogus"),
        "{err:#}"
    );
    let err = render(INKLING2_EFFORT_TEMPLATE, Some("2")).unwrap_err();
    assert!(
        format!("{err:#}").contains("reasoning_effort must be in [0.0, 0.99]"),
        "{err:#}"
    );
}

#[test]
fn a_template_that_ignores_the_level_renders_byte_identically() {
    let without = render(QWEN3_STYLE_TEMPLATE, None).unwrap();
    for level in ["high", "0.5"] {
        assert_eq!(render(QWEN3_STYLE_TEMPLATE, Some(level)).unwrap(), without);
    }
}

#[test]
fn level_value_is_a_number_only_for_finite_numbers() {
    assert_eq!(level_value("0.5"), json!(0.5));
    assert_eq!(level_value("high"), json!("high"));
    assert_eq!(level_value("nan"), json!("nan"));
    assert_eq!(level_value("inf"), json!("inf"));
}

#[test]
fn parse_rejects_only_empty_values() {
    assert_eq!(parse_reasoning_effort(" Low ").unwrap(), "low");
    assert!(parse_reasoning_effort("  ").is_err());
}

#[test]
fn chat_body_carries_the_field_and_the_typed_kwarg() {
    let mut body = json!({"model": "m", "messages": []});
    apply_to_chat_body(&mut body, "0.5");
    assert_eq!(body["reasoning_effort"], json!("0.5"));
    assert_eq!(body["chat_template_kwargs"]["reasoning_effort"], json!(0.5));
}

#[test]
fn a_disabled_level_switches_thinking_off_as_the_server_does() {
    assert_eq!(
        render(THINKING_SWITCH_TEMPLATE, Some("none")).unwrap(),
        "hi|off"
    );
    assert_eq!(
        render(THINKING_SWITCH_TEMPLATE, Some("high")).unwrap(),
        "hi|on"
    );
}
