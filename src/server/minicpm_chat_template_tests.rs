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

//! MiniCPM-V 4.6 and MiniCPM-o server prompts on a template-less checkpoint
//! (issue #2547).
//!
//! The model store's `minicpm-v-4.6-bf16` copy ships no chat template. Without
//! a builtin, the server rendered the generic `User: ...\n\nAssistant: ` prompt
//! and the image expansion then wrapped that in a ChatML turn of its own, so the
//! server prompt carried both framings (88 tokens) while `mlxcel generate`
//! (bare question, ChatML wrap only) carried one (82). These gates resolve the
//! template the way `mlxcel-server` does and check that the expanded prompt is
//! the one the CLI builds, using a temporary directory rather than a checkpoint.

use std::path::Path;

use super::chat_template::{ChatMessage, ChatTemplateProcessor};
use super::startup::resolve_chat_template;
use crate::multimodal::minicpmo_prompt::ensure_minicpmo_image_placeholders_with_sizes;
use crate::multimodal::minicpmv46_prompt::expand_minicpmv46_image_placeholders;

const QUESTION: &str = "Describe the image.";
/// The 448x448 fixture image's feature count on `minicpm-v-4.6-bf16`.
const FEATURES: usize = 64;

fn model_dir(model_type: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        format!(r#"{{"model_type": "{model_type}"}}"#),
    )
    .expect("write config");
    dir
}

/// The processor `mlxcel-server` renders with when no `--chat-template` is set.
fn server_processor(path: &Path) -> ChatTemplateProcessor {
    resolve_chat_template(None, None, path).expect("template resolution")
}

fn user(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
    }
}

#[test]
fn template_less_minicpmv46_server_prompt_matches_the_cli_prompt() {
    let dir = model_dir("minicpmv4_6");
    let rendered = server_processor(dir.path())
        .apply(&[user(QUESTION)], None)
        .expect("render");
    let server = expand_minicpmv46_image_placeholders(&rendered, &[FEATURES]).expect("expand");
    // `mlxcel generate` finds no template and hands the bare question over.
    let cli = expand_minicpmv46_image_placeholders(QUESTION, &[FEATURES]).expect("expand");

    let block = format!(
        "<image_id>0</image_id><image>{}</image>",
        "<|image_pad|>".repeat(FEATURES)
    );
    assert_eq!(
        server,
        format!("<|im_start|>user\n{block}\n{QUESTION}<|im_end|>\n<|im_start|>assistant\n")
    );
    assert_eq!(server, cli);
}

#[test]
fn template_less_minicpmo_server_prompt_matches_the_cli_prompt() {
    let dir = model_dir("minicpmo");
    let rendered = server_processor(dir.path())
        .apply(&[user(QUESTION)], None)
        .expect("render");
    let server =
        ensure_minicpmo_image_placeholders_with_sizes(&rendered, &[FEATURES]).expect("expand");
    let cli = ensure_minicpmo_image_placeholders_with_sizes(QUESTION, &[FEATURES]).expect("expand");
    assert!(!server.contains("User:"), "double framing: {server:?}");
    assert_eq!(server, cli);
}

#[test]
fn minicpm_builtin_renders_system_and_assistant_turns_as_chatml() {
    let dir = model_dir("minicpmv4_6");
    let processor = ChatTemplateProcessor::from_model_path(dir.path())
        .expect("template lookup")
        .expect("the MiniCPM builtin must be selected for a checkpoint that ships none");
    let rendered = processor
        .apply(
            &[
                ChatMessage {
                    role: "system".to_string(),
                    content: "S".to_string(),
                },
                user("Q1"),
                ChatMessage {
                    role: "assistant".to_string(),
                    content: "A1".to_string(),
                },
                user("Q2"),
            ],
            None,
        )
        .expect("render");
    assert_eq!(
        rendered,
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ1<|im_end|>\n\
         <|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n\
         <|im_start|>assistant\n"
    );
}

#[test]
fn a_shipped_minicpm_template_is_not_shadowed_by_the_builtin() {
    let dir = model_dir("minicpmv4_6");
    std::fs::write(
        dir.path().join("chat_template.jinja"),
        "{% for m in messages %}[{{ m.content }}]{% endfor %}",
    )
    .expect("write template");
    let rendered = server_processor(dir.path())
        .apply(&[user(QUESTION)], None)
        .expect("render");
    assert_eq!(rendered, format!("[{QUESTION}]"));
}
