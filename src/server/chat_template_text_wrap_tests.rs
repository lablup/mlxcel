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

use serde_json::json;

use super::{ChatMessage, ChatTemplateProcessor};

const QUESTION: &str = "What is the capital of France?";
const GRANITE_SYSTEM: &str = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n";
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
        .expect("fixture must load")
        .expect("fixture must contain a template")
}

fn checkpoint(template: &str) -> ChatTemplateProcessor {
    let directory = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        directory.path().join("tokenizer_config.json"),
        json!({"chat_template": template}).to_string(),
    )
    .expect("write tokenizer config");
    ChatTemplateProcessor::from_model_path(directory.path())
        .expect("checkpoint must load")
        .expect("checkpoint must contain a template")
}

#[test]
fn granite_3_from_model_path_preserves_the_exact_user_prompt() {
    let processor = granite_3();
    assert!(processor.wrap_string_content);
    assert_eq!(
        processor
            .apply(&[message("user", QUESTION)], None)
            .expect("render"),
        format!("{GRANITE_SYSTEM}<|user|>\n{QUESTION}\n<|assistant|>\n")
    );
}

#[test]
fn granite_3_wraps_system_assistant_and_empty_content() {
    let processor = granite_3();
    let history = [
        message("system", "Answer concisely."),
        message("user", QUESTION),
        message("assistant", "Paris."),
        message("user", ""),
    ];
    let expected = concat!(
        "<|system|>\nAnswer concisely.\n",
        "<|user|>\nWhat is the capital of France?\n",
        "<|assistant|>\nParis.<|end_of_text|>",
        "<|user|>\n\n"
    );
    assert_eq!(
        processor
            .apply_history_with_kwargs(&history, None, &Default::default())
            .expect("history render"),
        expected
    );
    assert_eq!(
        processor.apply(&history, None).expect("generation render"),
        format!("{expected}<|assistant|>\n")
    );
}

#[test]
fn granite_3_raw_rendering_keeps_its_original_contract() {
    let processor = granite_3();
    let typed = json!([{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": QUESTION}],
    }]);
    assert_eq!(
        processor.apply_raw(&typed, None).expect("typed render"),
        format!("{GRANITE_SYSTEM}<|user|>\n<image>\n{QUESTION}\n<|assistant|>\n")
    );

    let raw_string = json!([{"role": "user", "content": QUESTION}]);
    let rendered = processor
        .apply_raw(&raw_string, None)
        .expect("raw string render");
    assert!(!rendered.contains(QUESTION));
}

#[test]
fn string_compatible_templates_keep_string_content() {
    let processor = checkpoint(
        "{% for m in messages %}{% if m.content is string %}string:{{ m.content }}{% else %}typed:{{ m.content[0].text }}{% endif %}{% endfor %}",
    );
    assert!(!processor.wrap_string_content);
    assert_eq!(
        processor
            .apply(&[message("user", QUESTION)], None)
            .expect("render"),
        format!("string:{QUESTION}")
    );
}

#[test]
fn granite_3_config_only_string_template_is_not_wrapped() {
    let directory = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        directory.path().join("tokenizer_config.json"),
        GRANITE_3_CONFIG,
    )
    .expect("write tokenizer config");
    let processor = ChatTemplateProcessor::from_model_path(directory.path())
        .expect("checkpoint must load")
        .expect("checkpoint must contain a template");
    assert!(!processor.wrap_string_content);
    assert!(
        processor
            .apply(&[message("user", QUESTION)], None)
            .expect("render")
            .contains(QUESTION)
    );
}

#[test]
fn granite_4_accepts_both_shapes_without_wrapping() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/granite_4_vision");
    let processor = ChatTemplateProcessor::from_model_path(&path)
        .expect("fixture must load")
        .expect("fixture must contain a template");
    assert!(!processor.wrap_string_content);
    assert!(
        processor
            .apply(&[message("user", QUESTION)], None)
            .expect("string render")
            .contains(QUESTION)
    );
    assert!(
        processor
            .apply_raw(
                &json!([{
                    "role": "user",
                    "content": [{"type": "text", "text": QUESTION}],
                }]),
                None,
            )
            .expect("typed render")
            .contains(QUESTION)
    );
}

#[test]
fn explicit_template_constructors_do_not_enable_wrapping() {
    let template = include_str!("../../tests/fixtures/granite_vision_3_2/chat_template.jinja");
    let processor = ChatTemplateProcessor::with_template(template.into());
    assert!(!processor.wrap_string_content);
    assert!(
        !processor
            .apply(&[message("user", QUESTION)], None)
            .expect("render")
            .contains(QUESTION)
    );
    assert!(!ChatTemplateProcessor::default().wrap_string_content);
}

#[test]
fn probe_render_errors_leave_wrapping_disabled() {
    let processor = checkpoint(concat!(
        "{% if messages[0].content is string %}",
        "{{ raise_exception('string rejected') }}",
        "{% else %}{{ messages[0].content[0].text }}{% endif %}"
    ));
    assert!(!processor.wrap_string_content);
    assert!(processor.apply(&[message("user", QUESTION)], None).is_err());
}
