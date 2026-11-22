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

//! PLaMo 2.1 VL's builtin serving template (issue #1380): selected for a
//! `plamo2vl` checkpoint that ships no template, and rendering exactly the text
//! the processor's `_apply_ja01_template` builds, so the server prompt equals
//! the CLI prompt once the runtime places the image markers.

use super::chat_template::{ChatMessage, ChatTemplateProcessor};
use crate::multimodal::plamo2_vl_prompt::{build_plamo2_vl_prompt, format_plamo2_vl_prompt};

fn processor() -> (tempfile::TempDir, ChatTemplateProcessor) {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "plamo2vl", "text_config": {}, "vision_config": {}}"#,
    )
    .expect("write config");
    let processor = ChatTemplateProcessor::from_model_path(dir.path())
        .expect("template lookup")
        .expect("the PLaMo 2.1 VL builtin must be selected");
    (dir, processor)
}

fn message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
    }
}

#[test]
fn plamo2_vl_builtin_template_renders_the_instruction_template() {
    let (_dir, processor) = processor();
    let rendered = processor
        .apply(&[message("user", "この画像の色は何色ですか。")], None)
        .expect("render");
    assert_eq!(
        rendered,
        format_plamo2_vl_prompt("この画像の色は何色ですか。", 0)
    );
    // The runtime places the image marker where the CLI's bare text gets it.
    assert_eq!(
        build_plamo2_vl_prompt(&rendered, 1).0,
        build_plamo2_vl_prompt("この画像の色は何色ですか。", 1).0
    );
}

#[test]
fn plamo2_vl_builtin_template_joins_messages_with_newlines() {
    let (_dir, processor) = processor();
    let rendered = processor
        .apply(&[message("system", "S"), message("user", "Q")], None)
        .expect("render");
    assert_eq!(rendered, format_plamo2_vl_prompt("S\nQ", 0));
}
