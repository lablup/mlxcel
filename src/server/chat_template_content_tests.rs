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

//! Checkpoint-selected templates that require typed text content (#2135).

use std::path::PathBuf;

use super::{ChatMessage, ChatTemplateProcessor};
use serde_json::{Value, json};

const QUESTION: &str = "What is the capital of France?";
const GRANITE_SYSTEM: &str = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n";
const GRANITE_3: &str = include_str!("../../tests/fixtures/granite_vision_3_2/chat_template.jinja");
const GRANITE_3_CONFIG: &str =
    include_str!("../../tests/fixtures/granite_vision_3_2/tokenizer_config.json");

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
    }
}

fn granite_3() -> ChatTemplateProcessor {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/granite_vision_3_2");
    ChatTemplateProcessor::from_model_path(&path)
        .unwrap()
        .unwrap()
}

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

#[test]
fn granite_3_from_model_path_preserves_the_exact_user_prompt() {
    // The standalone typed template must win over the competing string one,
    // then receive typed text. This assertion fails against unmodified main.
    let processor = granite_3();
    let actual = processor.apply(&[message("user", QUESTION)], None).unwrap();
    assert_eq!(
        actual,
        format!("{GRANITE_SYSTEM}<|user|>\n{QUESTION}\n<|assistant|>\n")
    );
}

#[test]
fn granite_3_wraps_system_assistant_and_history_content() {
    let processor = granite_3();
    let messages = [
        message("system", "Answer concisely."),
        message("user", QUESTION),
        message("assistant", "Paris."),
        message("user", "And Germany?"),
    ];
    let expected = concat!(
        "<|system|>\nAnswer concisely.\n",
        "<|user|>\nWhat is the capital of France?\n",
        "<|assistant|>\nParis.<|end_of_text|>",
        "<|user|>\nAnd Germany?\n"
    );
    assert_eq!(
        processor.apply(&messages, None).unwrap(),
        format!("{expected}<|assistant|>\n")
    );
    assert_eq!(
        processor
            .apply_history_with_kwargs(&messages, None, &Default::default())
            .unwrap(),
        expected
    );
    let mut no_generation_prompt = processor.clone();
    no_generation_prompt.set_add_generation_prompt(false);
    assert_eq!(
        no_generation_prompt.apply(&messages, None).unwrap(),
        expected
    );
}

#[test]
fn empty_text_remains_one_empty_text_part_in_every_role() {
    let processor = granite_3();
    let messages = [
        message("system", ""),
        message("user", ""),
        message("assistant", ""),
    ];
    assert_eq!(
        processor
            .apply_history_with_kwargs(&messages, None, &Default::default())
            .unwrap(),
        "<|system|>\n\n<|user|>\n\n<|assistant|>\n<|end_of_text|>"
    );

    let count_parts = checkpoint(
        "{% for m in messages %}{% if m.content is not string %}{{ m.content | length }}:{% for p in m.content %}{{ p.type }}={{ p.text }}{% endfor %}{% endif %}{% endfor %}",
    );
    assert_eq!(
        count_parts.apply(&[message("user", "")], None).unwrap(),
        "1:text="
    );
}

#[test]
fn granite_3_raw_image_is_unchanged_and_strings_are_normalized() {
    let processor = granite_3();
    let raw = json!([{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": QUESTION}],
    }]);
    assert_eq!(
        processor.apply_raw(&raw, None).unwrap(),
        format!("{GRANITE_SYSTEM}<|user|>\n<image>\n{QUESTION}\n<|assistant|>\n")
    );
    assert_eq!(
        processor
            .apply_raw_history_with_kwargs(&raw, None, &Default::default())
            .unwrap(),
        format!("{GRANITE_SYSTEM}<|user|>\n<image>\n{QUESTION}\n")
    );
    // Raw strings use the same normalization as ordinary messages.
    let raw_string = json!([{"role": "user", "content": QUESTION}]);
    assert_eq!(
        processor.apply_raw(&raw_string, None).unwrap(),
        format!("{GRANITE_SYSTEM}<|user|>\n{QUESTION}\n<|assistant|>\n")
    );
}

#[test]
fn raw_message_metadata_survives_on_a_wrapping_processor() {
    let processor = checkpoint(
        "{% for m in messages %}{{ m.role }}|{{ m.name | default('') }}|{{ m.tool_call_id | default('') }}|{{ m.reasoning | default('') }}|{% if m.tool_calls is defined %}{{ m.tool_calls[0].function.name }}{% endif %}|{% for p in m.content | selectattr('type', 'equalto', 'text') %}{{ p.text }}{% endfor %}{% endfor %}",
    );
    let raw = json!([{
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
        "name": "speaker",
        "tool_call_id": "call-1",
        "reasoning": "reason",
        "tool_calls": [{"function": {"name": "weather"}}],
    }]);
    assert_eq!(
        processor.apply_raw(&raw, None).unwrap(),
        "assistant|speaker|call-1|reason|weather|answer"
    );
}

#[test]
fn granite_3_config_only_string_template_is_not_wrapped() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), GRANITE_3_CONFIG).unwrap();
    let processor = ChatTemplateProcessor::from_model_path(dir.path())
        .unwrap()
        .unwrap();
    assert_eq!(
        processor.apply(&[message("user", QUESTION)], None).unwrap(),
        format!("{GRANITE_SYSTEM}<|user|>\n{QUESTION}\n<|assistant|>\n")
    );
}

#[test]
fn granite_4_accepts_both_shapes_without_wrapping() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/granite_4_vision");
    let processor = ChatTemplateProcessor::from_model_path(&path)
        .unwrap()
        .unwrap();
    let messages = [
        message("system", "Answer concisely."),
        message("user", QUESTION),
        message("assistant", "Paris."),
    ];
    let raw: Vec<Value> = messages
        .iter()
        .map(|m| json!({"role": m.role, "content": [{"type": "text", "text": m.content}]}))
        .collect();
    // The checkpoint's render_content macro preserves eight indentation
    // spaces for strings, but trims them for typed text parts. Both forms
    // retain their original prompt contract; they need not render identically.
    let expected_string = concat!(
        "<|start_of_role|>system<|end_of_role|>Answer concisely.<|end_of_text|>\n",
        "<|start_of_role|>user<|end_of_role|>        What is the capital of France?<|end_of_text|>\n",
        "<|start_of_role|>assistant<|end_of_role|>        Paris.<|end_of_text|>\n",
        "<|start_of_role|>assistant<|end_of_role|>"
    );
    let expected_typed = concat!(
        "<|start_of_role|>system<|end_of_role|>Answer concisely.<|end_of_text|>\n",
        "<|start_of_role|>user<|end_of_role|>What is the capital of France?<|end_of_text|>\n",
        "<|start_of_role|>assistant<|end_of_role|>Paris.<|end_of_text|>\n",
        "<|start_of_role|>assistant<|end_of_role|>"
    );
    let original = ChatTemplateProcessor::with_template(
        std::fs::read_to_string(path.join("chat_template.jinja")).unwrap(),
    );
    // Custom-template and raw rendering keep the pre-fix content contract.
    assert_eq!(original.apply(&messages, None).unwrap(), expected_string);
    assert_eq!(
        original.apply_raw(&json!(raw), None).unwrap(),
        expected_typed
    );
    assert_eq!(processor.apply(&messages, None).unwrap(), expected_string);
    assert_eq!(
        processor.apply_raw(&json!(messages), None).unwrap(),
        expected_string
    );
    assert_eq!(
        processor.apply_raw(&json!(raw), None).unwrap(),
        expected_typed
    );
}

#[test]
fn default_and_template_override_do_not_opt_into_wrapping() {
    let default = ChatTemplateProcessor::default();
    assert!(
        default
            .apply(&[message("user", QUESTION)], None)
            .unwrap()
            .contains(QUESTION)
    );

    let explicit = ChatTemplateProcessor::with_template(GRANITE_3.into());
    assert!(
        !explicit
            .apply(&[message("user", QUESTION)], None)
            .unwrap()
            .contains(QUESTION)
    );
}

#[test]
fn probe_uses_the_selected_processors_special_tokens_and_preprocessing() {
    let dir = tempfile::tempdir().unwrap();
    let template = concat!(
        "{% if bos_token != '<bos>' or eos_token != '<eos>' %}",
        "{{ raise_exception('missing checkpoint tokens') }}{% endif %}",
        "{% generation %}{% for p in messages[0].content | selectattr('type', 'equalto', 'text') %}",
        "{{ bos_token }}{{ p.text }}{{ eos_token }}{% endfor %}{% endgeneration %}"
    );
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({
            "chat_template": template,
            "bos_token": {"content": "<bos>"},
            "eos_token": "<eos>",
        })
        .to_string(),
    )
    .unwrap();
    let processor = ChatTemplateProcessor::from_model_path(dir.path())
        .unwrap()
        .unwrap();
    assert_eq!(
        processor.apply(&[message("user", QUESTION)], None).unwrap(),
        format!("<bos>{QUESTION}<eos>")
    );
    assert_eq!(
        processor.template_compile_count(),
        1,
        "probes reuse the compiled template"
    );
}

#[test]
fn either_probe_render_error_leaves_wrapping_disabled() {
    for (template, string_succeeds) in [
        (
            "{% if messages[0].content is string %}{{ raise_exception('string rejected') }}{% else %}{{ messages[0].content[0].text }}{% endif %}",
            false,
        ),
        (
            "{% if messages[0].content is string %}{{ messages[0].content }}{% else %}{{ raise_exception('typed rejected') }}{% endif %}",
            true,
        ),
        (
            "{% if messages[0].content is not string %}{{ raise_exception('typed rejected') }}{% endif %}",
            true,
        ),
        (
            "{% if messages[0].content is string %}{{ unavailable_probe_function() }}{% else %}{{ messages[0].content[0].text }}{% endif %}",
            false,
        ),
    ] {
        let processor = checkpoint(template);
        assert_eq!(
            processor.apply(&[message("user", QUESTION)], None).is_ok(),
            string_succeeds,
            "{template}"
        );
    }
}

#[test]
fn parse_errors_and_missing_typed_markers_do_not_enable_wrapping() {
    let malformed = checkpoint("{% for m in messages %}");
    assert!(malformed.apply(&[message("user", QUESTION)], None).is_err());

    let drops_both = checkpoint("fixed prompt");
    assert_eq!(
        drops_both
            .apply(&[message("user", QUESTION)], None)
            .unwrap(),
        "fixed prompt"
    );
}

#[test]
fn a_string_accepting_template_keeps_the_original_content_shape() {
    let processor = checkpoint(
        "{% for m in messages %}{% if m.content is string %}string:{{ m.content }}{% else %}typed:{{ m.content[0].text }}{% endif %}{% endfor %}",
    );
    assert_eq!(
        processor.apply(&[message("user", QUESTION)], None).unwrap(),
        format!("string:{QUESTION}")
    );
}
