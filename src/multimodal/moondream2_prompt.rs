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

//! Prompt shaping for Moondream2 query/caption flows.
//!
//! Moondream2 does not ship a chat template; it frames the user text as a
//! `Question: ... Answer:` turn (or a caption request for an empty prompt). The
//! BOS token and projected image tokens are prepended by the VL wrapper, so for
//! the image path only the framed text tokens are produced here. Text-only
//! prompts additionally lead with the BOS id so the model still sees a prefix.

/// The moondream2 tokenizer (GPT-2/CodeGen) uses `<|endoftext|>` (id 50256) as
/// its begin-of-text token. Moondream3 uses id 0 with a different tokenizer;
/// the moondream2 port must not reuse that value.
pub const MOONDREAM2_BOS_ID: i32 = 50256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moondream2PromptMode {
    Caption,
    Query,
}

#[derive(Debug)]
pub struct PreparedMoondream2Prompt {
    pub tokens: Vec<i32>,
    pub mode: Moondream2PromptMode,
}

fn query_text(prompt: &str) -> String {
    format!("\n\nQuestion: {}\n\nAnswer:", prompt.trim())
}

const CAPTION_TEXT: &str = "\n\nQuestion: Describe this image.\n\nAnswer:";

pub fn prepare_moondream2_prompt_tokens<E>(
    prompt: &str,
    image_count: usize,
    mut encode: E,
) -> Result<PreparedMoondream2Prompt, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if image_count > 1 {
        return Err(format!(
            "Moondream2 currently supports at most one image, got {}",
            image_count
        ));
    }

    if image_count == 0 {
        // Text-only: [BOS] + framed question (the VL wrapper adds no prefix).
        if prompt.trim().is_empty() {
            return Err("Moondream2 text-only query requires a non-empty prompt".to_string());
        }
        let mut tokens = vec![MOONDREAM2_BOS_ID];
        tokens.extend(encode(&query_text(prompt), false));
        return Ok(PreparedMoondream2Prompt {
            tokens,
            mode: Moondream2PromptMode::Query,
        });
    }

    // Image path: the wrapper prepends BOS + image tokens, so emit text only.
    if prompt.trim().is_empty() {
        return Ok(PreparedMoondream2Prompt {
            tokens: encode(CAPTION_TEXT, false),
            mode: Moondream2PromptMode::Caption,
        });
    }

    Ok(PreparedMoondream2Prompt {
        tokens: encode(&query_text(prompt), false),
        mode: Moondream2PromptMode::Query,
    })
}

#[cfg(test)]
#[path = "moondream2_prompt_tests.rs"]
mod tests;
