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

//! ERNIE-Image text conditioner on the in-tree Ministral 3 decoder.
//!
//! A port of `ErnieImageTextEncoder` and `ErnieImageTokenizer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/text_encoder.py>.
//! Upstream re-implements a Ministral 3 decoder; this port runs
//! [`crate::models::ministral3::Ministral3Model`] instead, through its
//! [`Ministral3Model::forward_hidden_states`] entry point. Conditioning runs all
//! but the last decoder layer and skips the final norm (`#L176-L192`); the
//! prompt enhancer (see [`super::prompt_enhancer`]) runs every layer, the norm
//! and the tied head.
//!
//! The text encoder is a `Mistral3Model` VLM checkpoint; only its
//! `language_model.*` tower is loaded. Upstream's encoder layer omits the
//! Llama 4 attention scale the in-tree layer applies, but that scale is exactly
//! 1 below `original_max_position_embeddings` (16384) and prompts are
//! truncated at [`MAX_TEXT_TOKENS`] (2048), so the in-tree layer computes the
//! same thing.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use super::config::MAX_TEXT_TOKENS;
use crate::models::ministral3::{Ministral3Model, ModelArgs};
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// Prefix of the text tower inside a `Mistral3Model` VLM checkpoint.
const VLM_TEXT_PREFIX: &str = "language_model.";

/// Load the Ministral 3 decoder of `dir` (a `Ministral3ForCausalLM`, or a
/// `Mistral3Model` VLM whose `text_config` and `language_model.*` tensors are
/// used), with the project's text-weight dtype policy.
///
/// Used by: [`ErnieTextEncoder::load`] and
/// [`super::prompt_enhancer::ErniePromptEnhancer::load`].
pub fn load_ministral3(dir: &Path) -> Result<(Ministral3Model, ModelArgs)> {
    let config_path = dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("cannot read {}", config_path.display()))?;
    let config: Value = serde_json::from_str(&text)
        .with_context(|| format!("cannot parse {}", config_path.display()))?;
    let text_config = config.get("text_config").cloned().unwrap_or(config);
    let args: ModelArgs = serde_json::from_value(text_config)
        .with_context(|| format!("{} is not a Ministral 3 config", config_path.display()))?;
    let weights = crate::models::load_text_weights(dir, None)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load weights from {}", dir.display()))?;
    let prefix = if weights.contains_key(&format!("{VLM_TEXT_PREFIX}model.embed_tokens.weight")) {
        VLM_TEXT_PREFIX
    } else {
        ""
    };
    let weights = text_tower_weights(weights, prefix, args.tie_word_embeddings);
    let model = Ministral3Model::from_weights_with_prefix(&weights, &args, prefix)
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "cannot build the Ministral 3 decoder from {}",
                dir.display()
            )
        })?;
    Ok((model, args))
}

/// Keep only the decoder tensors: drops the VLM's vision tower and projector
/// and, with tied embeddings, an `lm_head.weight` upstream never reads (the
/// `pe/` checkpoint ships one; upstream's enhancer uses `embed_tokens`).
fn text_tower_weights(weights: WeightMap, prefix: &str, tied: bool) -> WeightMap {
    let model_prefix = format!("{prefix}model.");
    let head_prefix = format!("{prefix}lm_head.");
    weights
        .into_iter()
        .filter(|(key, _)| {
            key.starts_with(&model_prefix) || (!tied && key.starts_with(&head_prefix))
        })
        .collect()
}

/// `text_encoder/` and `tokenizer/` of an ERNIE-Image checkpoint.
pub struct ErnieTextEncoder {
    tokenizer: MlxcelTokenizer,
    model: Ministral3Model,
    hidden_size: i32,
}

impl ErnieTextEncoder {
    /// Load `<root>/text_encoder/` and `<root>/tokenizer/`.
    pub fn load(root: &Path) -> Result<Self> {
        let tokenizer_dir = root.join("tokenizer");
        let tokenizer = load_tokenizer(&tokenizer_dir)
            .with_context(|| format!("cannot load tokenizer from {}", tokenizer_dir.display()))?;
        let (model, args) =
            load_ministral3(&root.join("text_encoder")).context("ERNIE-Image text encoder")?;
        Self::new(tokenizer, model, args.hidden_size as i32)
    }

    /// Assemble from loaded parts; the decoder needs at least two layers.
    pub fn new(
        tokenizer: MlxcelTokenizer,
        model: Ministral3Model,
        hidden_size: i32,
    ) -> Result<Self> {
        if model.layers.len() < 2 {
            bail!(
                "the ERNIE-Image text encoder needs at least two layers, got {}",
                model.layers.len()
            );
        }
        Ok(Self {
            tokenizer,
            model,
            hidden_size,
        })
    }

    #[must_use]
    pub fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    /// Upstream `ErnieImageTokenizer.encode`: special tokens added (BOS),
    /// truncated to [`MAX_TEXT_TOKENS`].
    pub fn token_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        let mut ids = self.tokenizer.encode(prompt, true)?;
        ids.truncate(MAX_TEXT_TOKENS);
        ids.into_iter()
            .map(|id| i32::try_from(id).context("token id exceeds i32"))
            .collect()
    }

    /// Conditioning of `prompt`: `[1, tokens, hidden]`.
    pub fn encode(&self, prompt: &str) -> Result<UniquePtr<MlxArray>> {
        let ids = self.token_ids(prompt)?;
        if ids.is_empty() {
            bail!("the ERNIE-Image prompt encodes to no tokens");
        }
        self.encode_ids(&ids)
    }

    /// Conditioning of a token sequence: every decoder layer but the last,
    /// no final norm.
    pub fn encode_ids(&self, ids: &[i32]) -> Result<UniquePtr<MlxArray>> {
        conditioning_hidden_states(&self.model, ids)
    }
}

/// `ErnieImageTextEncoder(input_ids)` with `normalize=False`: the hidden states
/// after `layers[:-1]`, without the final norm, on fresh caches.
///
/// Used by: [`ErnieTextEncoder::encode_ids`] and the conditioning tests.
pub fn conditioning_hidden_states(
    model: &Ministral3Model,
    ids: &[i32],
) -> Result<UniquePtr<MlxArray>> {
    let input = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
    let mut caches = model.make_caches();
    let layers = model.layers.len().saturating_sub(1);
    let hidden = model
        .forward_hidden_states(&input, &mut caches, Some(layers), false)
        .map_err(anyhow::Error::msg)?;
    mlxcel_core::try_eval(&hidden).context("ERNIE-Image prompt encoding failed")?;
    Ok(hidden)
}

#[cfg(test)]
#[path = "text_encoder_tests.rs"]
mod tests;
