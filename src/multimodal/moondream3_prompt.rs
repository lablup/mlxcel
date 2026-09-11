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

//! Prompt shaping for Moondream3 query/caption flows.

pub const MOONDREAM3_BOS_ID: i32 = 0;
pub const MOONDREAM3_QUERY_PREFIX: &[i32] = &[1, 15381, 2];
pub const MOONDREAM3_QUERY_SUFFIX: &[i32] = &[3];
pub const MOONDREAM3_CAPTION_NORMAL: &[i32] = &[1, 32708, 2, 6382, 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moondream3PromptMode {
    Caption,
    Query,
}

#[derive(Debug)]
pub struct PreparedMoondream3Prompt {
    pub tokens: Vec<i32>,
    pub mode: Moondream3PromptMode,
}

pub fn prepare_moondream3_prompt_tokens<E>(
    prompt: &str,
    image_count: usize,
    mut encode: E,
) -> Result<PreparedMoondream3Prompt, String>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if image_count > 1 {
        return Err(format!(
            "Moondream3 currently supports at most one image, got {}",
            image_count
        ));
    }

    if image_count == 0 {
        // Text-only: [BOS=0] + prefix + question + suffix
        if prompt.trim().is_empty() {
            return Err("Moondream3 text-only query requires a non-empty prompt".to_string());
        }
        let mut tokens = vec![MOONDREAM3_BOS_ID];
        tokens.extend(MOONDREAM3_QUERY_PREFIX);
        tokens.extend(encode(prompt, false));
        tokens.extend(MOONDREAM3_QUERY_SUFFIX);
        return Ok(PreparedMoondream3Prompt {
            tokens,
            mode: Moondream3PromptMode::Query,
        });
    }

    if prompt.trim().is_empty() {
        return Ok(PreparedMoondream3Prompt {
            tokens: MOONDREAM3_CAPTION_NORMAL.to_vec(),
            mode: Moondream3PromptMode::Caption,
        });
    }

    let mut tokens = MOONDREAM3_QUERY_PREFIX.to_vec();
    tokens.extend(encode(prompt, false));
    tokens.extend(MOONDREAM3_QUERY_SUFFIX);

    Ok(PreparedMoondream3Prompt {
        tokens,
        mode: Moondream3PromptMode::Query,
    })
}
