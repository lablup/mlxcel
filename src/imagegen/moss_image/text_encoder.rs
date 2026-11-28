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

//! The MOSS-Image prompt conditioner: the Qwen3.5-9B language backbone, whose
//! final-norm hidden states are the transformer's `context`.
//!
//! The reference runs transformers' `Qwen3_5Model` on the chat-templated
//! prompt (one prompt at a time, no padding, `enable_thinking=True`,
//! generation prompt appended) and takes `last_hidden_state`. Only the
//! language path is used; the checkpoint's vision tower is skipped by
//! mlxcel's Qwen3.5 loader.
//!
//! The template is rendered here rather than through a Jinja engine: with one
//! user message and the generation prompt, `tokenizer/chat_template.jinja`
//! reduces to [`prompt_text`] exactly.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::MAX_PROMPT_TOKENS;
use crate::models::qwen3_5::Qwen35Model;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// `tokenizer/chat_template.jinja` for one user message, `add_generation_prompt`
/// and `enable_thinking=True` (the reference's arguments).
#[must_use]
pub fn prompt_text(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        prompt.trim()
    )
}

/// The Qwen3.5-9B conditioner.
pub struct TextEncoder {
    model: Qwen35Model,
    tokenizer: MlxcelTokenizer,
    max_tokens: usize,
}

impl TextEncoder {
    /// Load `<root>/text_encoder` (the language backbone) and `<root>/tokenizer`.
    ///
    /// `text_dim` is the width the transformer's `text_embedder` expects; a
    /// conditioner of another width is refused here rather than failing inside
    /// the first matmul.
    pub fn load(text_encoder_dir: &Path, tokenizer_dir: &Path, text_dim: i32) -> Result<Self> {
        // The exported MLX tier nests its quantization block under
        // `moss_image_mlx.quantization`; the language loader reads
        // `text_config.quantization`, so hand it over explicitly.
        let exported = std::fs::read_to_string(text_encoder_dir.join("config.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|v| v.get("moss_image_mlx")?.get("quantization").cloned());
        let (model, config) = Qwen35Model::load_text_backbone(text_encoder_dir, exported.as_ref())
            .map_err(|e| anyhow!("MOSS-Image text encoder: {e}"))?;
        ensure!(
            i32::try_from(config.hidden_size).ok() == Some(text_dim),
            "MOSS-Image text encoder hidden_size {} does not match the transformer's text_dim {text_dim}",
            config.hidden_size
        );
        let tokenizer = load_tokenizer(tokenizer_dir).context("MOSS-Image tokenizer")?;
        Ok(Self {
            model,
            tokenizer,
            max_tokens: MAX_PROMPT_TOKENS,
        })
    }

    /// Token ids of a prompt, truncated to the conditioner's window.
    pub fn token_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        let ids = self.tokenizer.encode(&prompt_text(prompt), true)?;
        let mut ids = ids
            .into_iter()
            .map(|id| i32::try_from(id).context("MOSS-Image token id out of range"))
            .collect::<Result<Vec<i32>>>()?;
        ids.truncate(self.max_tokens);
        Ok(ids)
    }

    /// `[1, L, 4096]` final-norm hidden states for one prompt, unpadded.
    pub fn encode(&self, prompt: &str) -> Result<UniquePtr<MlxArray>> {
        let ids = self.token_ids(prompt)?;
        ensure!(!ids.is_empty(), "MOSS-Image prompt tokenized to nothing");
        let ids = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
        Ok(self.model.forward_prefill_hidden(&ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_template_matches_the_reference_render() {
        assert_eq!(
            prompt_text("  a fox  "),
            "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }
}
