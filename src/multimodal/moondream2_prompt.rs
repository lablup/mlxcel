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
//! The `vikhyatk/moondream2` checkpoints that use the unified
//! `model.text.blocks.*` weight layout (2025 revisions) come in two prompt
//! generations, and the weights only produce coherent text when paired with
//! the matching tokenizer and template ids:
//!
//! - **Starmie templates** (revision 2025-06-21 and later): the weights are
//!   trained against the `moondream/starmie-v1` tokenizer, whose ids 0..=20
//!   are control tokens (`<|endoftext|>` = 0, `<|md_reserved_*|>` = 1..). The
//!   shipped `config.py` frames a query as `[1, 15381, 2] + question + [3]`
//!   (`<|md_reserved_0|> query <|md_reserved_1|> ... <|md_reserved_2|>`) with
//!   bos = eos = 0. This is the same contract the in-tree Moondream3 port
//!   uses; see [`crate::moondream3_prompt`].
//! - **Legacy Question/Answer** (revisions 2025-01-09 .. 2025-04-14): the
//!   weights are trained against the GPT-2/CodeGen tokenizer shipped in the
//!   checkpoint (`<|endoftext|>` = 50256 doubles as bos/eos) and the
//!   templates are plain text: `"\n\nQuestion: {q}\n\nAnswer:"`.
//!
//! The official repository never removed the legacy GPT-2 `tokenizer.json` /
//! `tokenizer_config.json`, so a 2025-06-21 snapshot still carries tokenizer
//! files that do NOT match its weights. Feeding GPT-2 ids to starmie-era
//! weights makes the model emit its true EOS (id 0) followed by degenerate
//! loops, which the GPT-2 vocabulary then decodes as `!NCJNCJ...`-style
//! garbage. [`detect_moondream2_prompt_style`] tells the two generations
//! apart, and `src/tokenizer/mod.rs::load_tokenizer` resolves the actual
//! starmie tokenizer for the starmie era.
//!
//! The BOS token and projected image tokens are prepended by the VL wrapper,
//! so for the image path only the framed text tokens are produced here.
//! Text-only prompts additionally lead with the BOS id so the model still
//! sees a prefix, mirroring `moondream.py::query`.

use std::path::Path;

/// Begin/end-of-text id for the legacy (GPT-2/CodeGen tokenizer) revisions,
/// where `<|endoftext|>` is id 50256.
pub const MOONDREAM2_LEGACY_BOS_ID: i32 = 50256;

/// Begin/end-of-text id for starmie-era revisions (`<|endoftext|>` is id 0 in
/// `moondream/starmie-v1`), identical to Moondream3.
pub const MOONDREAM2_STARMIE_BOS_ID: i32 = 0;

/// Starmie query prefix: `<|md_reserved_0|> query <|md_reserved_1|>`.
pub const MOONDREAM2_STARMIE_QUERY_PREFIX: &[i32] = &[1, 15381, 2];

/// Starmie query suffix (answer marker): `<|md_reserved_2|>`.
pub const MOONDREAM2_STARMIE_QUERY_SUFFIX: &[i32] = &[3];

/// Starmie normal-length caption template:
/// `<|md_reserved_0|> describe <|md_reserved_1|> normal <|md_reserved_2|>`.
pub const MOONDREAM2_STARMIE_CAPTION_NORMAL: &[i32] = &[1, 32708, 2, 6382, 3];

/// Which prompt/tokenizer generation a moondream2 checkpoint belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moondream2PromptStyle {
    /// 2025-06-21+ revisions: `moondream/starmie-v1` tokenizer, control-token
    /// templates, bos = eos = 0.
    StarmieTemplates,
    /// 2025-01-09 .. 2025-04-14 revisions: GPT-2/CodeGen tokenizer shipped in
    /// the checkpoint, plain-text Question/Answer framing, bos = eos = 50256.
    LegacyQuestionAnswer,
}

/// Detect which prompt generation a moondream2 checkpoint directory uses.
///
/// The authoritative discriminator ships with every `trust_remote_code`
/// snapshot: `moondream.py` names the tokenizer repository the weights were
/// trained against (`Tokenizer.from_pretrained("moondream/starmie-v1")` from
/// revision 2025-06-21 onwards, `"vikhyatk/moondream2"` before). The stale
/// GPT-2 `tokenizer.json` cannot be trusted for this because the official
/// repository kept it alongside starmie-era weights.
///
/// When `moondream.py` is absent (e.g. a pruned conversion), fall back to
/// sniffing `tokenizer.json` itself: only the starmie vocabulary defines the
/// `<|md_reserved_0|>` control token. A directory with neither file defaults
/// to the legacy style, matching the tokenizer that `load_tokenizer` would
/// fail to find anyway.
pub fn detect_moondream2_prompt_style(model_path: &Path) -> Moondream2PromptStyle {
    if let Ok(source) = std::fs::read_to_string(model_path.join("moondream.py")) {
        return if source.contains("moondream/starmie-v1") {
            Moondream2PromptStyle::StarmieTemplates
        } else {
            Moondream2PromptStyle::LegacyQuestionAnswer
        };
    }
    if let Ok(tokenizer_json) = std::fs::read_to_string(model_path.join("tokenizer.json"))
        && tokenizer_json.contains("<|md_reserved_0|>")
    {
        return Moondream2PromptStyle::StarmieTemplates;
    }
    Moondream2PromptStyle::LegacyQuestionAnswer
}

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

fn legacy_query_text(prompt: &str) -> String {
    format!("\n\nQuestion: {}\n\nAnswer:", prompt.trim())
}

const LEGACY_CAPTION_TEXT: &str = "\n\nQuestion: Describe this image.\n\nAnswer:";

pub fn prepare_moondream2_prompt_tokens<E>(
    prompt: &str,
    image_count: usize,
    style: Moondream2PromptStyle,
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

    match style {
        Moondream2PromptStyle::StarmieTemplates => {
            if image_count == 0 {
                // Text-only: [BOS=0] + prefix + question + suffix, mirroring
                // `moondream.py::query` without an encoded image.
                if prompt.trim().is_empty() {
                    return Err(
                        "Moondream2 text-only query requires a non-empty prompt".to_string()
                    );
                }
                let mut tokens = vec![MOONDREAM2_STARMIE_BOS_ID];
                tokens.extend(MOONDREAM2_STARMIE_QUERY_PREFIX);
                tokens.extend(encode(prompt, false));
                tokens.extend(MOONDREAM2_STARMIE_QUERY_SUFFIX);
                return Ok(PreparedMoondream2Prompt {
                    tokens,
                    mode: Moondream2PromptMode::Query,
                });
            }

            // Image path: the wrapper prepends BOS + image tokens.
            if prompt.trim().is_empty() {
                return Ok(PreparedMoondream2Prompt {
                    tokens: MOONDREAM2_STARMIE_CAPTION_NORMAL.to_vec(),
                    mode: Moondream2PromptMode::Caption,
                });
            }

            let mut tokens = MOONDREAM2_STARMIE_QUERY_PREFIX.to_vec();
            tokens.extend(encode(prompt, false));
            tokens.extend(MOONDREAM2_STARMIE_QUERY_SUFFIX);
            Ok(PreparedMoondream2Prompt {
                tokens,
                mode: Moondream2PromptMode::Query,
            })
        }
        Moondream2PromptStyle::LegacyQuestionAnswer => {
            if image_count == 0 {
                // Text-only: [BOS] + framed question (the VL wrapper adds no
                // prefix).
                if prompt.trim().is_empty() {
                    return Err(
                        "Moondream2 text-only query requires a non-empty prompt".to_string()
                    );
                }
                let mut tokens = vec![MOONDREAM2_LEGACY_BOS_ID];
                tokens.extend(encode(&legacy_query_text(prompt), false));
                return Ok(PreparedMoondream2Prompt {
                    tokens,
                    mode: Moondream2PromptMode::Query,
                });
            }

            // Image path: the wrapper prepends BOS + image tokens, so emit
            // text only.
            if prompt.trim().is_empty() {
                return Ok(PreparedMoondream2Prompt {
                    tokens: encode(LEGACY_CAPTION_TEXT, false),
                    mode: Moondream2PromptMode::Caption,
                });
            }

            Ok(PreparedMoondream2Prompt {
                tokens: encode(&legacy_query_text(prompt), false),
                mode: Moondream2PromptMode::Query,
            })
        }
    }
}

#[cfg(test)]
#[path = "moondream2_prompt_tests.rs"]
mod tests;
