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

//! Unlimited-OCR's builtin serving template.
//!
//! `mlx-community/Unlimited-OCR-4bit` ships DeepSeek-OCR's
//! `chat_template.jinja`, which puts a space after every message. mlx-vlm's
//! `UnlimitedOCRProcessor` renders its own template instead (no space after
//! the last message), and so does `baidu/Unlimited-OCR` (`format_messages`
//! strips the prompt). These gates check that the builtin replaces the shipped
//! template for this family only, using temporary directories so they run in
//! CI without a checkpoint.

use std::path::Path;

use super::chat_template::{ChatMessage, ChatTemplateProcessor};

/// `chat_template.jinja` as shipped by `mlx-community/Unlimited-OCR-4bit`
/// (revision `945968c0`) and `mlx-community/DeepSeek-OCR-4bit` (revision
/// `4958b6b0`), byte for byte.
const SHIPPED_DEEPSEEK_OCR_TEMPLATE: &str = "{% for message in messages %}{% if message['role'] == 'user' %}{% elif message['role'] == 'assistant' %}{% endif %}{{message['content']}} {% endfor %}{% if add_generation_prompt %}{% endif %}";

fn model_dir(model_type: &str, shipped_template: Option<&str>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        format!(r#"{{"model_type": "{model_type}"}}"#),
    )
    .expect("write config");
    if let Some(template) = shipped_template {
        std::fs::write(dir.path().join("chat_template.jinja"), template).expect("write template");
    }
    dir
}

fn processor(path: &Path) -> ChatTemplateProcessor {
    ChatTemplateProcessor::from_model_path(path)
        .expect("template lookup")
        .expect("a template is selected")
}

fn user(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: text.to_string(),
    }]
}

/// The shipped DeepSeek-OCR template renders `Free OCR. `; Unlimited-OCR's
/// reference renders `Free OCR.`. The trailing space is token 223, the one
/// token by which the prompt was longer than mlx-vlm's (898 vs 897 on a
/// 1920x1280 page).
#[test]
fn unlimited_ocr_replaces_the_shipped_deepseek_ocr_template() {
    let dir = model_dir("unlimited-ocr", Some(SHIPPED_DEEPSEEK_OCR_TEMPLATE));
    let p = processor(dir.path());
    assert_eq!(
        p.apply(&user("Free OCR."), None).expect("render"),
        "Free OCR."
    );
    // A server request with an image part renders the same text; the page
    // reaches the prompt as image tokens after BOS.
    let typed = p
        .apply_raw(
            &serde_json::json!([{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "Free OCR."},
                ],
            }]),
            None,
        )
        .expect("render");
    assert_eq!(typed, "Free OCR.");
}

/// The builtin is mlx-vlm's `UnlimitedOCRProcessor.default_chat_template`:
/// message texts joined by one space, nothing after the last one, no role
/// markers and no generation prompt. The `unlimited_ocr` spelling of the
/// model type routes to the same family and so gets the same template.
#[test]
fn unlimited_ocr_builtin_joins_messages_with_one_space() {
    let dir = model_dir("unlimited_ocr", None);
    let messages = ["Free OCR.", "Northern Network", "Again."]
        .iter()
        .zip(["user", "assistant", "user"])
        .map(|(text, role)| ChatMessage {
            role: role.to_string(),
            content: text.to_string(),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        processor(dir.path())
            .apply(&messages, None)
            .expect("render"),
        "Free OCR. Northern Network Again."
    );
}

/// DeepSeek-OCR keeps the template its checkpoint ships, trailing space
/// included. The override is scoped to Unlimited-OCR; DeepSeek-OCR's prompt is
/// not changed here.
#[test]
fn deepseek_ocr_keeps_its_shipped_template() {
    let dir = model_dir("deepseekocr", Some(SHIPPED_DEEPSEEK_OCR_TEMPLATE));
    assert_eq!(
        processor(dir.path())
            .apply(&user("Free OCR."), None)
            .expect("render"),
        "Free OCR. "
    );
}
