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

//! FLUX.2 prompt conditioning.
//!
//! A port of `Flux2Tokenizer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/tokenizer.py>
//! (one user turn through the checkpoint's chat template with
//! `add_generation_prompt=True` and `enable_thinking=False`, right-padded with
//! the pad token) and of `encode_prompt`, `_pick_bucket` and
//! `prepare_text_ids` in `flux2/prompt.py`: the padded length is the smallest
//! bucket of `(32, 64, 128, 256, 512)` that holds the prompt, capped at
//! `max_sequence_length`, and the conditioning is the Qwen3 hidden states of
//! layers 9, 18 and 27 stacked into `[B, T, 3 * H]`.
//!
//! Used by: Flux2, Bonsai.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use crate::imagegen::qwen3_text_encoder::Qwen3TextEncoder;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::chat_template_kwargs::ChatTemplateKwargs;

/// Padded prompt lengths tried in order.
pub const DEFAULT_SEQ_LEN_BUCKETS: [usize; 5] = [32, 64, 128, 256, 512];
/// Hidden-state indices (0 = embeddings) stacked into the conditioning.
pub const HIDDEN_STATE_LAYERS: [usize; 3] = [9, 18, 27];
/// Default and largest `max_sequence_length`.
pub const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 512;
/// The negative prompt classifier-free guidance pairs with every prompt.
pub const NEGATIVE_PROMPT: &str = " ";

/// Upstream `_pick_bucket`: the smallest bucket `>= true_len` among those
/// `<= cap`, else `cap`.
#[must_use]
pub fn pick_bucket(true_len: usize, buckets: &[usize], cap: usize) -> usize {
    let mut eligible: Vec<usize> = buckets.iter().copied().filter(|b| *b <= cap).collect();
    eligible.sort_unstable();
    eligible.into_iter().find(|b| *b >= true_len).unwrap_or(cap)
}

/// Token ids and the matching attention mask of one padded prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedPrompt {
    pub input_ids: Vec<i32>,
    /// 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<i32>,
}

/// The checkpoint's `tokenizer/` plus its chat template.
pub struct Flux2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    template: ChatTemplateProcessor,
    pad_id: i32,
}

impl Flux2Tokenizer {
    /// Load `<tokenizer_dir>/tokenizer.json` and its chat template.
    pub fn load(tokenizer_dir: &Path) -> Result<Self> {
        let path = tokenizer_dir.join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("cannot load {}: {e}", path.display()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("cannot reset tokenizer truncation: {e}"))?;
        let template =
            ChatTemplateProcessor::from_model_path(tokenizer_dir)?.with_context(|| {
                format!(
                    "FLUX.2 tokenizer {} has no chat template",
                    tokenizer_dir.display()
                )
            })?;
        let pad_token = read_pad_token(tokenizer_dir).unwrap_or_else(|| "<|endoftext|>".into());
        let pad_id = tokenizer
            .token_to_id(&pad_token)
            .with_context(|| format!("pad token {pad_token:?} is not in the vocabulary"))?;
        Ok(Self {
            tokenizer,
            template,
            pad_id: i32::try_from(pad_id)?,
        })
    }

    /// The chat-formatted prompt upstream's `_format_prompts` produces.
    pub fn format_prompt(&self, prompt: &str) -> Result<String> {
        let messages = json!([{ "role": "user", "content": prompt }]);
        let kwargs = ChatTemplateKwargs::from_json_object(
            json!({ "enable_thinking": false })
                .as_object()
                .cloned()
                .unwrap_or_default(),
        );
        let mut template = self.template.clone();
        template.set_add_generation_prompt(true);
        template.apply_raw_with_kwargs(&messages, None, &kwargs)
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("cannot tokenize the prompt: {e}"))?;
        encoding
            .get_ids()
            .iter()
            .map(|&id| i32::try_from(id).map_err(anyhow::Error::from))
            .collect()
    }

    /// Unpadded token count of the formatted prompt.
    pub fn count_tokens(&self, prompt: &str) -> Result<usize> {
        Ok(self.encode(&self.format_prompt(prompt)?)?.len())
    }

    /// Format, tokenize, truncate to `max_length` and right-pad to it.
    pub fn tokenize(&self, prompt: &str, max_length: usize) -> Result<TokenizedPrompt> {
        if max_length == 0 {
            bail!("max_sequence_length must be positive");
        }
        let mut ids = self.encode(&self.format_prompt(prompt)?)?;
        ids.truncate(max_length);
        let real = ids.len();
        ids.resize(max_length, self.pad_id);
        let mut attention_mask = vec![1; real];
        attention_mask.resize(max_length, 0);
        Ok(TokenizedPrompt {
            input_ids: ids,
            attention_mask,
        })
    }
}

fn read_pad_token(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    match config.get("pad_token")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => map.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

/// `[batch, seq_len, 4]` int32 text position ids `(0, 0, 0, i)`.
#[must_use]
pub fn prepare_text_ids(batch: i32, seq_len: i32) -> UniquePtr<MlxArray> {
    let mut coords = Vec::with_capacity((seq_len.max(0) * 4) as usize);
    for i in 0..seq_len {
        coords.extend_from_slice(&[0, 0, 0, i]);
    }
    let row = mlxcel_core::from_slice_i32(&coords, &[1, seq_len, 4]);
    mlxcel_core::broadcast_to(&row, &[batch, seq_len, 4])
}

/// Prompt conditioning and its position ids.
pub struct PromptEmbeds {
    /// `[1, T, 3 * H]` in the text encoder's dtype.
    pub embeds: UniquePtr<MlxArray>,
    /// `[1, T, 4]` int32.
    pub text_ids: UniquePtr<MlxArray>,
}

/// Upstream `encode_prompt`: bucket, tokenize, and stack the hidden states.
///
/// `bucketed = false` pads to `max_sequence_length` (the edit default).
pub fn encode_prompt(
    tokenizer: &Flux2Tokenizer,
    encoder: &Qwen3TextEncoder,
    prompt: &str,
    max_sequence_length: usize,
    bucketed: bool,
) -> Result<PromptEmbeds> {
    if max_sequence_length == 0 || max_sequence_length > DEFAULT_MAX_SEQUENCE_LENGTH {
        bail!(
            "max_sequence_length must be between 1 and {DEFAULT_MAX_SEQUENCE_LENGTH}, \
             got {max_sequence_length}"
        );
    }
    let true_len = if bucketed {
        tokenizer.count_tokens(prompt)?
    } else {
        max_sequence_length
    };
    let length = pick_bucket(true_len, &DEFAULT_SEQ_LEN_BUCKETS, max_sequence_length);
    let tokens = tokenizer.tokenize(prompt, length)?;
    let len = i32::try_from(length)?;
    let ids = mlxcel_core::from_slice_i32(&tokens.input_ids, &[1, len]);
    let mask = mlxcel_core::from_slice_i32(&tokens.attention_mask, &[1, len]);
    let embeds = encoder.stacked_prompt_embeds(&ids, Some(&mask), &HIDDEN_STATE_LAYERS)?;
    mlxcel_core::eval(&embeds);
    let text_ids = prepare_text_ids(1, len);
    Ok(PromptEmbeds { embeds, text_ids })
}

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;
