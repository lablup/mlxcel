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

//! Z-Image prompt conditioning.
//!
//! A port of `ZImagePipeline._format_prompt` / `_encode_prompt` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/z_image/pipeline.py>
//! (`#L42-L78`): the prompt is rendered as one user turn with the checkpoint's
//! chat template (`add_generation_prompt`, `enable_thinking=True`), tokenized,
//! right-padded to 512 tokens, run through the shared Qwen3 conditioner, and
//! the penultimate hidden state (`hidden_states[-2]`, the second-to-last
//! decoder layer before the final norm) is trimmed to the valid tokens.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use crate::imagegen::qwen3_text_encoder::Qwen3TextEncoder;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// Prompts are padded (and truncated) to this many tokens before encoding.
pub const MAX_PROMPT_TOKENS: usize = 512;

/// Tokenizer and chat template of `<model>/tokenizer/`.
pub struct PromptFormatter {
    tokenizer: MlxcelTokenizer,
    template: ChatTemplateProcessor,
    pad_token_id: u32,
}

/// `pad_token` of `tokenizer_config.json`, as a string or `{content: ...}`.
fn pad_token(dir: &Path) -> Result<String> {
    let path = dir.join("tokenizer_config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let config: Value =
        serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))?;
    match config.get("pad_token") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Object(o)) => o
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
            .context("pad_token has no content"),
        _ => bail!("{} declares no pad_token", path.display()),
    }
}

impl PromptFormatter {
    /// Load `<tokenizer_dir>/tokenizer.json` and its chat template.
    pub fn load(tokenizer_dir: &Path) -> Result<Self> {
        let tokenizer = load_tokenizer(tokenizer_dir)
            .with_context(|| format!("cannot load tokenizer from {}", tokenizer_dir.display()))?;
        let template = ChatTemplateProcessor::from_model_path(tokenizer_dir)?
            .with_context(|| format!("{} ships no chat template", tokenizer_dir.display()))?;
        let pad = pad_token(tokenizer_dir)?;
        let pad_token_id = tokenizer
            .token_to_id(&pad)
            .with_context(|| format!("pad token {pad:?} is not in the vocabulary"))?;
        Ok(Self {
            tokenizer,
            template,
            pad_token_id,
        })
    }

    /// `apply_chat_template([user prompt], add_generation_prompt=True,
    /// enable_thinking=True)`.
    pub fn format(&self, prompt: &str) -> Result<String> {
        let mut kwargs = ChatTemplateKwargs::new();
        kwargs.set("enable_thinking", Value::Bool(true));
        let messages = json!([{ "role": "user", "content": prompt }]);
        self.template
            .apply_raw_with_kwargs(&messages, None, &kwargs)
    }

    /// Token ids of the formatted prompt, untruncated (upstream's
    /// `count_prompt_tokens`).
    pub fn token_ids(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(&self.format(prompt)?, true)
    }

    /// `(input_ids, attention_mask, valid_len)`, padded on the right to
    /// [`MAX_PROMPT_TOKENS`] with the pad token and truncated to it.
    pub fn padded(&self, prompt: &str) -> Result<(Vec<i32>, Vec<i32>, usize)> {
        let mut ids = self.token_ids(prompt)?;
        ids.truncate(MAX_PROMPT_TOKENS);
        let valid = ids.len();
        if valid == 0 {
            bail!("the formatted Z-Image prompt has no tokens");
        }
        let mut mask = vec![1_i32; valid];
        ids.resize(MAX_PROMPT_TOKENS, self.pad_token_id);
        mask.resize(MAX_PROMPT_TOKENS, 0);
        let ids = ids
            .into_iter()
            .map(|id| i32::try_from(id).context("token id exceeds i32"))
            .collect::<Result<Vec<_>>>()?;
        Ok((ids, mask, valid))
    }
}

/// Encode `prompt` to caption features `[1, valid_len, hidden]` in the
/// encoder's dtype.
pub fn encode_prompt(
    formatter: &PromptFormatter,
    encoder: &Qwen3TextEncoder,
    prompt: &str,
) -> Result<UniquePtr<MlxArray>> {
    let (ids, mask, valid) = formatter.padded(prompt)?;
    let len = ids.len() as i32;
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, len]);
    let attention_mask = mlxcel_core::from_slice_i32(&mask, &[1, len]);
    let layers = encoder.num_layers();
    if layers < 2 {
        bail!("the Z-Image text encoder needs at least two layers, got {layers}");
    }
    let mut states = encoder.hidden_states(&input_ids, Some(&attention_mask))?;
    // `hidden_states[-2]` of the `num_layers + 1` entries.
    let penultimate = states.swap_remove(layers - 1);
    let hidden = mlxcel_core::array_shape(&penultimate)[2];
    let trimmed = mlxcel_core::slice(&penultimate, &[0, 0, 0], &[1, valid as i32, hidden]);
    mlxcel_core::eval(&trimmed);
    Ok(trimmed)
}

#[cfg(test)]
#[path = "prompt_tests.rs"]
mod tests;
