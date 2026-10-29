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

//! Qwen3 hidden-state conditioner for image-generation text encoders.
//!
//! FLUX.2, Z-Image, Ideogram 4 and Bonsai condition their transformers on
//! intermediate hidden states of a Qwen3 language model rather than on its
//! logits. This is a port of `Qwen3TextEncoder` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/qwen/text_encoder.py>
//! (imported there by `z_image`, `ideogram4` and `bonsai`), built on
//! [`Qwen3Model`] instead of a second copy of the decoder.
//!
//! [`Qwen3TextEncoder::hidden_states`] returns `num_layers + 1` arrays: index 0
//! is the token embeddings and index `i >= 1` is the output of decoder layer `i`
//! (1-based) before the final norm, run under a causal plus key-padding
//! additive mask with positions `0..T` and standard RoPE. Attention runs in
//! the weights' dtype. Upstream (`qwen/attention.py`) runs SDPA in f32;
//! [`Qwen3Model::set_attention_f32`] reproduces that, but measured on the
//! real checkpoints it does not bring the conditioning closer to upstream's
//! (#2666), so it stays off.
//! [`Qwen3TextEncoder::stacked_prompt_embeds`] stacks chosen entries into the
//! `[B, T, n * H]` layout FLUX.2 uses.
//!
//! Used by: Flux2, ZImage, Ideogram4, Bonsai.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

use crate::models::qwen3::{ModelArgs, Qwen3Model};

/// Additive mask value for a blocked position.
///
/// Finite on purpose. Upstream uses `-inf`, which is identical for every row
/// that has at least one visible key (the softmax weight underflows to exactly
/// zero either way) but turns a row with none into NaN, and a NaN row at a
/// padded position then poisons the valid positions through the next layer's
/// `0 * NaN`. `-30000` is exact in f16, bf16 and f32 and still underflows
/// `exp`; an all-blocked row becomes a uniform average instead of NaN.
const MASK_FILL: f32 = -30000.0;

/// Which tensors of a checkpoint belong to the language model, and the key the
/// [`Qwen3Model`] loader expects for them.
///
/// Accepts the layouts the conditioner checkpoints ship: `model.layers.N...`,
/// `model.language_model.layers.N...` (transformers Qwen3-VL) and
/// `language_model.model.layers.N...` (mlx-vlm conversions). `lm_head.*` and
/// every non-text tensor (vision tower, projector, ...) are dropped.
fn text_key(key: &str) -> Option<String> {
    let mut rest = key;
    loop {
        if let Some(stripped) = rest.strip_prefix("model.") {
            rest = stripped;
        } else if let Some(stripped) = rest.strip_prefix("language_model.") {
            rest = stripped;
        } else {
            break;
        }
    }
    ["embed_tokens.", "layers.", "norm."]
        .iter()
        .any(|p| rest.starts_with(p))
        .then(|| format!("model.{rest}"))
}

/// Keep the text tensors of `weights`, renamed for [`Qwen3Model::from_weights`].
fn normalize_weights(weights: WeightMap) -> WeightMap {
    weights
        .into_iter()
        .filter_map(|(key, value)| text_key(&key).map(|k| (k, value)))
        .collect()
}

/// The Qwen3 language-model fields of a checkpoint `config.json`.
///
/// Top-level for a plain Qwen3 checkpoint, nested under `text_config` for a
/// Qwen3-VL one. Quantization metadata may sit on the outer object.
fn language_config(config: &Value) -> Result<Value> {
    let mut inner = match config.get("text_config") {
        Some(text) if text.is_object() => text.clone(),
        _ => config.clone(),
    };
    let object = inner
        .as_object_mut()
        .context("config.json must be a JSON object")?;
    object
        .entry("model_type")
        .or_insert_with(|| Value::String("qwen3".to_string()));
    for key in ["quantization", "quantization_config"] {
        if !object.contains_key(key)
            && let Some(outer) = config.get(key)
        {
            object.insert(key.to_string(), outer.clone());
        }
    }
    Ok(inner)
}

/// Qwen3 language model exposed as a multi-layer hidden-state extractor.
pub struct Qwen3TextEncoder {
    model: Qwen3Model,
    hidden_size: i32,
}

impl Qwen3TextEncoder {
    /// Load a text-encoder directory (`config.json` plus safetensors).
    ///
    /// bf16 weights are converted per the load-time dtype policy
    /// (`bf16_to_f16_at_load`); MLX affine `quantization` is honored.
    pub fn load(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("cannot read {}", config_path.display()))?;
        let config: Value = serde_json::from_str(&crate::models::sanitize_config_json(&text))
            .with_context(|| format!("cannot parse {}", config_path.display()))?;
        let mut weights = mlxcel_core::weights::load_weights_from_dir(dir)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("cannot load text encoder weights from {}", dir.display()))?;
        let quantized = config.get("quantization").is_some()
            || config
                .get("text_config")
                .and_then(|t| t.get("quantization"))
                .is_some();
        if crate::models::bf16_to_f16_at_load(quantized, Some(&config)) {
            let _ = crate::models::convert_bf16_weights(&mut weights);
        }
        Self::from_weights(&config, weights)
    }

    /// Build from an in-memory weight map, in any of the layouts [`text_key`]
    /// accepts. The caller owns the dtype: nothing is converted here, so a
    /// family that dequantizes first (Ideogram 4's FP8) hands over its result.
    pub fn from_weights(config: &Value, weights: WeightMap) -> Result<Self> {
        let language = language_config(config)?;
        let mut args: ModelArgs = serde_json::from_value(language)
            .context("config.json is not a Qwen3 language-model config")?;
        // Standard RoPE: positions are 0..T on every axis, so M-RoPE sections add
        // nothing for text, and the tables must not depend on `rope_scaling`.
        args.rope_scaling = None;
        // The hidden-state path never reads a head; do not require one.
        args.tie_word_embeddings = true;
        let weights = normalize_weights(weights);
        let hidden_size = args.hidden_size as i32;
        // Upstream mlx-vlm `models/flux2/qwen/attention.py:148-162` (git
        // 73d366ec) casts q, k and v to f32 for SDPA and the output back to
        // the query dtype. `set_attention_f32(true)` reproduces that, with two
        // value-exact differences: K/V enter the fresh per-call cache already
        // in f32 (upstream casts after the cache fetch; the cast is lossless
        // and the cache is discarded), and SDPA handles grouped KV heads
        // natively instead of upstream's `_repeat_kv`. It is not enabled: on
        // GB10 against the CUDA oracle it leaves the FLUX.2, Z-Image and
        // Ideogram 4 conditioning as far from upstream's as before (#2666),
        // while the end-to-end runs move by bf16-noise amounts that take
        // Bonsai and Ideogram 4 past their parity gates.
        let model = Qwen3Model::from_weights(&weights, &args)
            .map_err(anyhow::Error::msg)
            .context("cannot build the Qwen3 text encoder")?;
        Ok(Self { model, hidden_size })
    }

    /// Run attention in f32 like upstream (`true`) or in the activation
    /// dtype (`false`, the default). Lets a test measure what the f32
    /// attention changes.
    #[cfg(test)]
    pub(crate) fn set_attention_f32(&mut self, on: bool) {
        self.model.set_attention_f32(on);
    }

    /// Number of decoder layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Width `H` of one hidden state.
    #[must_use]
    pub fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    /// Additive `[B, 1, T, T]` mask: causal, plus `FILL` on keys whose
    /// `attention_mask` entry is not 1.
    fn attention_bias(
        attention_mask: Option<&MlxArray>,
        batch: i32,
        len: i32,
    ) -> UniquePtr<MlxArray> {
        let positions = mlxcel_core::arange_i32(0, len, 1);
        let query = mlxcel_core::reshape(&positions, &[len, 1]);
        let key = mlxcel_core::reshape(&positions, &[1, len]);
        let allowed = mlxcel_core::less_equal(&key, &query);
        let causal = mlxcel_core::where_cond(
            &allowed,
            &mlxcel_core::zeros(&[len, len], dtype::FLOAT32),
            &mlxcel_core::full_f32(&[len, len], MASK_FILL, dtype::FLOAT32),
        );
        let causal = mlxcel_core::reshape(&causal, &[1, 1, len, len]);
        let Some(mask) = attention_mask else {
            return mlxcel_core::broadcast_to(&causal, &[batch, 1, len, len]);
        };
        let visible = mlxcel_core::equal(
            &mlxcel_core::astype(mask, dtype::INT32),
            &mlxcel_core::full_f32(&[1], 1.0, dtype::INT32),
        );
        let padding = mlxcel_core::where_cond(
            &visible,
            &mlxcel_core::zeros(&[batch, len], dtype::FLOAT32),
            &mlxcel_core::full_f32(&[batch, len], MASK_FILL, dtype::FLOAT32),
        );
        let padding = mlxcel_core::reshape(&padding, &[batch, 1, 1, len]);
        mlxcel_core::add(&causal, &padding)
    }

    /// Embeddings and every decoder layer's output (before the final norm).
    ///
    /// `input_ids` is `[B, T]`. `attention_mask` is `[B, T]` with 1 for real
    /// tokens (anything else is padding); `None` means all real.
    pub fn hidden_states(
        &self,
        input_ids: &MlxArray,
        attention_mask: Option<&MlxArray>,
    ) -> Result<Vec<UniquePtr<MlxArray>>> {
        let shape = mlxcel_core::array_shape(input_ids);
        if shape.len() != 2 {
            bail!("input_ids must be [B, T], got {shape:?}");
        }
        if let Some(mask) = attention_mask {
            let mask_shape = mlxcel_core::array_shape(mask);
            if mask_shape != shape {
                bail!("attention_mask {mask_shape:?} must match input_ids {shape:?}");
            }
        }
        let bias = Self::attention_bias(attention_mask, shape[0], shape[1]);
        Ok(self.model.forward_all_hidden(input_ids, Some(&bias)))
    }

    /// Stack the hidden states at `layers` into `[B, T, layers.len() * H]`.
    ///
    /// Entries are indexed as in [`Self::hidden_states`] (0 = embeddings), and
    /// concatenated in the order given: the FLUX.2 layout, `stack(axis=1)`
    /// followed by a `[B, n, T, H] -> [B, T, n * H]` reshape.
    pub fn stacked_prompt_embeds(
        &self,
        input_ids: &MlxArray,
        attention_mask: Option<&MlxArray>,
        layers: &[usize],
    ) -> Result<UniquePtr<MlxArray>> {
        if layers.is_empty() {
            bail!("at least one hidden-state layer is required");
        }
        let states = self.hidden_states(input_ids, attention_mask)?;
        if let Some(bad) = layers.iter().find(|&&l| l >= states.len()) {
            bail!(
                "hidden-state layer {bad} is out of range for a {}-layer model (valid: 0..={})",
                self.num_layers(),
                self.num_layers()
            );
        }
        let chosen: Vec<UniquePtr<MlxArray>> = layers
            .iter()
            .map(|&l| mlxcel_core::expand_dims(&states[l], 1))
            .collect();
        let stacked =
            mlxcel_core::concatenate_many(&chosen.iter().map(|a| &**a).collect::<Vec<_>>(), 1);
        let shape = mlxcel_core::array_shape(&stacked);
        let (b, n, t, h) = (shape[0], shape[1], shape[2], shape[3]);
        let swapped = mlxcel_core::transpose_axes(&stacked, &[0, 2, 1, 3]);
        Ok(mlxcel_core::reshape(&swapped, &[b, t, n * h]))
    }
}

#[cfg(test)]
#[path = "qwen3_text_encoder_tests.rs"]
mod tests;
