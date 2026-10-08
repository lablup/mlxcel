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

//! Raw-message content adaptation without losing non-content fields.
//!

use std::path::PathBuf;

use serde_json::{Value, json};

use super::{ChatMessage, ChatTemplateProcessor};

const JSON_OBSERVER: &str = concat!(
    "{% for m in messages %}",
    "{% if m.content is not string %}{{ m | tojson }}{{ '\\n' }}{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}GENERATE{% endif %}",
);

fn checkpoint(template: &str) -> ChatTemplateProcessor {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"chat_template": template}).to_string(),
    )
    .unwrap();
    ChatTemplateProcessor::from_model_path(dir.path())
        .unwrap()
        .unwrap()
}

fn granite() -> ChatTemplateProcessor {
    ChatTemplateProcessor::from_model_path(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/granite_vision_3_2"),
    )
    .unwrap()
    .unwrap()
}

fn observed_messages(rendered: &str) -> Vec<Value> {
    rendered
        .strip_suffix("GENERATE")
        .unwrap_or(rendered)
        .lines()
        .map(|line| serde_json::from_str(line).expect("template emits one JSON message per line"))
        .collect()
}

#[test]
fn granite_raw_mixed_image_history_preserves_all_string_roles() {
    let processor = granite();
    let raw = json!([
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": [
            {"type": "text", "text": "IMAGE_QUESTION"},
            {"type": "image"},
        ]},
        {"role": "assistant", "content": "ANSWER"},
        {"role": "user", "content": "FOLLOWUP"},
    ]);
    let original = raw.clone();
    let history = concat!(
        "<|system|>\nSYSTEM\n",
        "<|user|>\n<image>\nIMAGE_QUESTION\n",
        "<|assistant|>\nANSWER<|end_of_text|>",
        "<|user|>\nFOLLOWUP\n",
    );
    assert_eq!(
        processor.apply_raw(&raw, None).unwrap(),
        format!("{history}<|assistant|>\n")
    );
    assert_eq!(
        processor
            .apply_raw_history_with_kwargs(&raw, None, &Default::default())
            .unwrap(),
        history
    );
    assert_eq!(
        raw, original,
        "normalization must not mutate caller-owned JSON"
    );
}

#[test]
fn raw_and_ordinary_text_share_generation_and_history_representation() {
    let processor = granite();
    let messages = [
        ChatMessage {
            role: "system".into(),
            content: "SYSTEM".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: "QUESTION".into(),
        },
        ChatMessage {
            role: "assistant".into(),
            content: "ANSWER".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: "FOLLOWUP".into(),
        },
    ];
    let raw = serde_json::to_value(&messages).unwrap();
    assert_eq!(
        processor.apply_raw(&raw, None).unwrap(),
        processor.apply(&messages, None).unwrap()
    );
    assert_eq!(
        processor
            .apply_raw_history_with_kwargs(&raw, None, &Default::default())
            .unwrap(),
        processor
            .apply_history_with_kwargs(&messages, None, &Default::default())
            .unwrap()
    );
}

#[test]
fn raw_string_conversion_preserves_tool_reasoning_and_extension_metadata() {
    let processor = checkpoint(JSON_OBSERVER);
    let raw = json!([
        {
            "role": "assistant",
            "content": "ANSWER",
            "name": "speaker",
            "reasoning": "TRACE",
            "reasoning_content": "TRACE",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "weather", "arguments": {"city": "Seoul"}},
            }],
            "extension": {"nested": [null, false, 17, "kept"]},
        },
        {
            "role": "tool",
            "content": "",
            "tool_call_id": "call-1",
            "name": "weather",
        },
    ]);
    let original = raw.clone();
    let rows = observed_messages(&processor.apply_raw(&raw, None).unwrap());
    assert_eq!(rows.len(), 2);
    let mut expected_assistant = raw[0].clone();
    expected_assistant["content"] = json!([{"type": "text", "text": "ANSWER"}]);
    let mut expected_tool = raw[1].clone();
    expected_tool["content"] = json!([{"type": "text", "text": ""}]);
    assert_eq!(rows, [expected_assistant, expected_tool]);
    assert_eq!(raw, original);
}

#[test]
fn raw_nonstring_content_is_not_flattened_coerced_or_reordered() {
    let processor = checkpoint(JSON_OBSERVER);
    let raw = json!([
        {"role": "assistant", "content": null, "tool_calls": []},
        {"role": "assistant", "name": "missing-content"},
        {"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "image", "extension": {"position": 1}},
            {"type": "text", "text": "after"},
        ]},
        {"role": "assistant", "content": []},
        {"role": "assistant", "content": false},
        {"role": "assistant", "content": 17},
    ]);
    let generation = processor.apply_raw(&raw, None).unwrap();
    let history = processor
        .apply_raw_history_with_kwargs(&raw, None, &Default::default())
        .unwrap();
    assert_eq!(observed_messages(&generation), *raw.as_array().unwrap());
    assert_eq!(observed_messages(&history), *raw.as_array().unwrap());
    assert_eq!(generation, format!("{history}GENERATE"));
}

#[test]
fn raw_normalization_remains_disabled_for_string_templates_and_explicit_overrides() {
    let raw = json!([{"role": "user", "content": "QUESTION", "name": "speaker"}]);
    let string_compatible = checkpoint("{{ messages | tojson }}");
    assert_eq!(
        serde_json::from_str::<Value>(&string_compatible.apply_raw(&raw, None).unwrap()).unwrap(),
        raw
    );

    // Operator-supplied templates keep their explicit content contract.
    let explicit = ChatTemplateProcessor::with_template(JSON_OBSERVER.into());
    assert_eq!(explicit.apply_raw(&raw, None).unwrap(), "GENERATE");
    assert_eq!(
        explicit
            .apply_raw_history_with_kwargs(&raw, None, &Default::default())
            .unwrap(),
        ""
    );
}

#[test]
fn mixed_raw_normalization_wraps_only_strings_and_preserves_other_values() {
    let processor = checkpoint(JSON_OBSERVER);
    let raw = json!([
        {"role": "assistant", "content": null, "tool_calls": []},
        {"role": "assistant", "name": "missing-content"},
        {"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "image", "extension": {"position": 1}},
            {"type": "text", "text": "after"},
        ]},
        {"role": "assistant", "content": []},
        {"role": "assistant", "content": false},
        {"role": "assistant", "content": 17},
        {"role": "user", "content": "FOLLOWUP"},
    ]);
    let original = raw.clone();
    let mut expected = raw.clone();
    expected[6]["content"] = json!([{"type": "text", "text": "FOLLOWUP"}]);

    let generation = processor.apply_raw(&raw, None).unwrap();
    let history = processor
        .apply_raw_history_with_kwargs(&raw, None, &Default::default())
        .unwrap();
    assert_eq!(
        observed_messages(&generation),
        *expected.as_array().unwrap()
    );
    assert_eq!(observed_messages(&history), *expected.as_array().unwrap());
    assert_eq!(generation, format!("{history}GENERATE"));
    assert_eq!(raw, original, "caller-owned JSON must remain unchanged");
}
