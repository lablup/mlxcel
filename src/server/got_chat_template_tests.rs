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

//! GOT-OCR 2.0's builtin serving template.
//!
//! The checkpoint ships no `chat_template` in any of its three declared
//! sources, so without a builtin the server falls to the generic
//! `User:` / `Assistant:` prompt, which this model never saw. These gates check
//! that the builtin is selected for the family and that it renders the exact
//! conversation `modeling_GOT.py::chat` builds, using a temporary directory
//! rather than a downloaded checkpoint so they run in CI.

use std::path::Path;

use super::chat_template::{ChatMessage, ChatTemplateProcessor};
use crate::multimodal::got_ocr_prompt::{GOT_ASSISTANT_ROLE, GOT_SEP, GOT_SYSTEM, GOT_USER_ROLE};

/// A minimal GOT `config.json`. `builtin_chat_template` resolves the family
/// through `get_model_type`, so the directory only has to be detectable.
const GOT_CONFIG: &str = r#"{
    "architectures": ["GOTQwenForCausalLM"],
    "model_type": "GOT",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "intermediate_size": 2816,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "vocab_size": 151860,
    "tie_word_embeddings": true,
    "image_token_len": 256,
    "im_start_token": 151857,
    "im_end_token": 151858,
    "im_patch_token": 151859
}"#;

fn got_model_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), GOT_CONFIG).expect("write config");
    dir
}

fn processor(path: &Path) -> ChatTemplateProcessor {
    ChatTemplateProcessor::from_model_path(path)
        .expect("template lookup")
        .expect("the GOT builtin must be selected for a checkpoint that ships none")
}

fn user(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: text.to_string(),
    }]
}

/// A one-turn render is byte-identical to the conversation upstream assembles,
/// minus the image block the runtime splices in afterwards.
#[test]
fn got_builtin_template_renders_the_reference_conversation() {
    let dir = got_model_dir();
    let rendered = processor(dir.path())
        .apply(&user("OCR: "), None)
        .expect("render");
    assert_eq!(
        rendered,
        format!("{GOT_SYSTEM}{GOT_SEP}{GOT_USER_ROLE}OCR: {GOT_SEP}{GOT_ASSISTANT_ROLE}")
    );
    // The eight spaces of the reference's source indentation and the absence of
    // a newline after the system turn's separator are both load-bearing.
    assert!(rendered.starts_with("<|im_start|>system\n        You should follow"));
    assert!(rendered.contains("detail.<|im_end|><|im_start|>user\n"));
}

/// A client system message is dropped and the fixed system turn is emitted
/// exactly once.
///
/// The vision features are scattered into a prompt whose prefix the model saw
/// on every training example. Substituting a caller's system text there
/// degrades OCR accuracy with nothing to point at, so the template does not
/// render it.
#[test]
fn got_builtin_template_replaces_a_client_system_message() {
    let dir = got_model_dir();
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a pirate.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "OCR: ".to_string(),
        },
    ];
    let rendered = processor(dir.path())
        .apply(&messages, None)
        .expect("render");
    assert!(!rendered.contains("pirate"));
    assert_eq!(rendered.matches("<|im_start|>system").count(), 1);
    assert_eq!(
        rendered,
        format!("{GOT_SYSTEM}{GOT_SEP}{GOT_USER_ROLE}OCR: {GOT_SEP}{GOT_ASSISTANT_ROLE}")
    );
}

/// A prior assistant turn is rendered as history and the generation prompt
/// still opens a fresh assistant turn.
#[test]
fn got_builtin_template_renders_multi_turn_history() {
    let dir = got_model_dir();
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "OCR: ".to_string(),
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "hello".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "OCR with format: ".to_string(),
        },
    ];
    let rendered = processor(dir.path())
        .apply(&messages, None)
        .expect("render");
    assert_eq!(
        rendered,
        format!(
            "{GOT_SYSTEM}{GOT_SEP}\
             {GOT_USER_ROLE}OCR: {GOT_SEP}\
             {GOT_ASSISTANT_ROLE}hello{GOT_SEP}\
             {GOT_USER_ROLE}OCR with format: {GOT_SEP}\
             {GOT_ASSISTANT_ROLE}"
        )
    );
}

/// The builtin never shadows a template a checkpoint actually ships.
#[test]
fn a_shipped_template_still_wins_over_the_builtin() {
    let dir = got_model_dir();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"chat_template": "SHIPPED:{{ messages[0]['content'] }}"}"#,
    )
    .expect("write tokenizer config");
    let rendered = processor(dir.path())
        .apply(&user("OCR: "), None)
        .expect("render");
    assert_eq!(rendered.trim(), "SHIPPED:OCR:");
}

/// A non-GOT checkpoint is unaffected: no builtin, so the lookup still reports
/// that the directory ships no template.
#[test]
fn the_builtin_is_scoped_to_the_got_family() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen2", "hidden_size": 1024, "num_hidden_layers": 24}"#,
    )
    .expect("write config");
    let resolved = ChatTemplateProcessor::from_model_path(dir.path()).expect("template lookup");
    assert!(
        resolved.is_none(),
        "a plain Qwen2 directory must not pick up the GOT builtin"
    );
}
