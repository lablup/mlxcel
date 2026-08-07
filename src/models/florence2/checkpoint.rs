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

//! Florence-2 checkpoint plumbing: whole-model config parsing and the
//! weight-key normalization that has to happen before either half of the
//! model is built.
//!
//! Kept apart from `model.rs` because these two run once at load time and
//! have nothing to do with the forward path.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::weights::WeightMap;

use crate::vision::encoders::florence2_davit;

use super::Florence2VisionConfig;

/// Weight-key prefix of the language model inside a Florence-2 checkpoint.
pub(crate) const FLORENCE2_TEXT_PREFIX: &str = "language_model.";

/// Both halves of a Florence-2 `config.json` plus the fusion-stage fields.
#[derive(Debug, Clone)]
pub struct Florence2Config {
    pub text: Florence2TextConfig,
    pub vision: Florence2VisionConfig,
    /// Placeholder id the processor may emit for the image. Florence-2 does
    /// not actually splice image features into placeholder slots, so any
    /// occurrence is dropped before the prompt is embedded. Upstream defaults
    /// this to the text vocabulary size (51289), one past the last real id.
    pub image_token_id: i32,
}

impl Florence2Config {
    /// Parse a full Florence-2 `config.json` (`model_type: florence2`).
    pub fn from_model_config(config: &Value) -> Result<Self> {
        let text = Florence2TextConfig::from_model_config(config)?;
        let vision = Florence2VisionConfig::from_model_config(config)?;
        let image_token_id = config
            .get("image_token_id")
            .or_else(|| config.get("image_token_index"))
            .and_then(Value::as_i64)
            .map(|v| v as i32)
            .unwrap_or(text.vocab_size);
        Ok(Self {
            text,
            vision,
            image_token_id,
        })
    }
}

/// Normalize a full Florence-2 checkpoint's weight keys.
///
/// Three passes, in order:
///
/// 1. The DaViT conv channels-last remap and `position_ids` drop from
///    [`florence2_davit::sanitize`]. Both of its key guards are specific
///    enough (`convs`/`proj.weight`, `blocks`/`dw.weight`) that running them
///    over the whole checkpoint leaves `language_model.*` untouched, and both
///    are idempotent, so this is a no-op on the `mlx-community` exports that
///    already ship channels-last conv weights.
/// 2. Drop `final_logits_bias`. HuggingFace BART carries it as a registered
///    buffer of zeros; it is not a parameter and there is no LM-head bias in
///    this implementation.
/// 3. BART shared-embedding naming. The encoder and decoder token tables are
///    tied to `model.shared` and exports vary in which of the three they
///    materialize, so fill `model.shared.weight` from an `embed_tokens` copy
///    when only the latter is present.
pub fn sanitize(weights: WeightMap) -> WeightMap {
    let weights = florence2_davit::sanitize(weights);

    let mut out = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        if key.contains("final_logits_bias") {
            continue;
        }
        out.insert(key, value);
    }

    let shared = format!("{FLORENCE2_TEXT_PREFIX}model.shared.weight");
    if !out.contains_key(&shared) {
        let source = [
            format!("{FLORENCE2_TEXT_PREFIX}model.encoder.embed_tokens.weight"),
            format!("{FLORENCE2_TEXT_PREFIX}model.decoder.embed_tokens.weight"),
        ]
        .into_iter()
        .find_map(|key| out.get(&key).map(|w| mlxcel_core::copy(w)));
        if let Some(tensor) = source {
            out.insert(shared, tensor);
        }
    }
    out
}

/// Upper bound accepted for `encoder_layers` / `decoder_layers`. Real
/// Florence-2 exports ship 6 (base) or 12 (large). The cap exists because
/// [`encoder::Florence2Encoder::from_weights`] and
/// [`decoder::Florence2Decoder::from_weights`] size a `Vec` from this field
/// before they look up a single weight, so a negative value out of a hostile
/// `config.json` becomes `usize::MAX` and aborts with a capacity overflow.
const MAX_LAYERS: i32 = 256;

/// Upper bound accepted for `max_position_embeddings`. Florence-2 ships 1024.
///
/// The field bounds two separate allocations: the position-table slice the
/// encoder and decoder take (validated against the loaded table at load time)
/// and the O(n^2) host buffer [`layers::additive_causal_mask`] fills for a
/// multi-token decoder call. An unbounded value out of a hostile `config.json`
/// is an out-of-memory abort rather than an error return.
const MAX_POSITION_EMBEDDINGS: i32 = 65_536;

/// BART encoder-decoder shape parameters, parsed from the `text_config`
/// object of a Florence-2 `config.json` (`model_type: florence2_language`).
#[derive(Debug, Clone)]
pub struct Florence2TextConfig {
    pub d_model: i32,
    pub encoder_layers: i32,
    pub decoder_layers: i32,
    pub encoder_attention_heads: i32,
    pub decoder_attention_heads: i32,
    pub encoder_ffn_dim: i32,
    pub decoder_ffn_dim: i32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub scale_embedding: bool,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    pub decoder_start_token_id: i32,
}

fn config_i32(config: &Value, key: &str) -> Option<i32> {
    config.get(key).and_then(Value::as_i64).map(|v| v as i32)
}

impl Florence2TextConfig {
    /// Parse from the `text_config` sub-object itself.
    pub fn from_text_config(config: &Value) -> Result<Self> {
        let require = |key: &str| -> Result<i32> {
            config_i32(config, key)
                .ok_or_else(|| anyhow!("Florence-2 text_config missing field: {key}"))
        };
        let d_model = require("d_model")?;
        let parsed = Self {
            d_model,
            encoder_layers: require("encoder_layers")?,
            decoder_layers: require("decoder_layers")?,
            encoder_attention_heads: require("encoder_attention_heads")?,
            decoder_attention_heads: require("decoder_attention_heads")?,
            encoder_ffn_dim: config_i32(config, "encoder_ffn_dim").unwrap_or(4 * d_model),
            decoder_ffn_dim: config_i32(config, "decoder_ffn_dim").unwrap_or(4 * d_model),
            vocab_size: require("vocab_size")?,
            max_position_embeddings: config_i32(config, "max_position_embeddings").unwrap_or(1024),
            scale_embedding: config
                .get("scale_embedding")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            pad_token_id: config_i32(config, "pad_token_id").unwrap_or(1),
            bos_token_id: config_i32(config, "bos_token_id").unwrap_or(0),
            eos_token_id: config_i32(config, "eos_token_id").unwrap_or(2),
            decoder_start_token_id: config_i32(config, "decoder_start_token_id").unwrap_or(2),
        };
        // Every field below becomes a shape, an index, a divisor, or a `Vec`
        // capacity somewhere downstream. MLX throws on an out-of-range shape
        // and the throw crosses an FFI boundary that cannot carry it, so a bad
        // value aborts the process rather than surfacing as an error; these
        // checks are what keep an untrusted `config.json` from getting that
        // far. They also have to run *before* the divisibility check below,
        // which would itself divide by zero on `encoder_attention_heads: 0`.
        if parsed.d_model < 1 {
            return Err(anyhow!(
                "Florence-2 text_config d_model must be positive, got {}",
                parsed.d_model
            ));
        }
        if parsed.encoder_attention_heads < 1 || parsed.decoder_attention_heads < 1 {
            return Err(anyhow!(
                "Florence-2 text_config attention heads must be positive, got {} enc / {} dec",
                parsed.encoder_attention_heads,
                parsed.decoder_attention_heads
            ));
        }
        if !(1..=MAX_LAYERS).contains(&parsed.encoder_layers)
            || !(1..=MAX_LAYERS).contains(&parsed.decoder_layers)
        {
            return Err(anyhow!(
                "Florence-2 text_config layer counts {} enc / {} dec outside 1..={MAX_LAYERS}",
                parsed.encoder_layers,
                parsed.decoder_layers
            ));
        }
        if parsed.encoder_ffn_dim < 1 || parsed.decoder_ffn_dim < 1 {
            return Err(anyhow!(
                "Florence-2 text_config ffn dims must be positive, got {} enc / {} dec",
                parsed.encoder_ffn_dim,
                parsed.decoder_ffn_dim
            ));
        }
        if parsed.vocab_size < 1 {
            return Err(anyhow!(
                "Florence-2 text_config vocab_size must be positive, got {}",
                parsed.vocab_size
            ));
        }
        if !(1..=MAX_POSITION_EMBEDDINGS).contains(&parsed.max_position_embeddings) {
            return Err(anyhow!(
                "Florence-2 text_config max_position_embeddings {} outside 1..={MAX_POSITION_EMBEDDINGS}",
                parsed.max_position_embeddings
            ));
        }
        // Both of these reach the shared embedding gather as literal token ids
        // (`generate_greedy` seeds the decoder with `decoder_start_token_id`,
        // `shift_tokens_right` substitutes `pad_token_id`), so an
        // out-of-vocabulary or negative value is an out-of-range gather.
        // `bos_token_id` and `eos_token_id` are deliberately left unchecked:
        // they are only ever compared, never used as an index, and some
        // exports legitimately carry sentinel values there.
        if !(0..parsed.vocab_size).contains(&parsed.pad_token_id) {
            return Err(anyhow!(
                "Florence-2 text_config pad_token_id {} outside 0..vocab_size {}",
                parsed.pad_token_id,
                parsed.vocab_size
            ));
        }
        if !(0..parsed.vocab_size).contains(&parsed.decoder_start_token_id) {
            return Err(anyhow!(
                "Florence-2 text_config decoder_start_token_id {} outside 0..vocab_size {}",
                parsed.decoder_start_token_id,
                parsed.vocab_size
            ));
        }
        if parsed.d_model % parsed.encoder_attention_heads != 0
            || parsed.d_model % parsed.decoder_attention_heads != 0
        {
            return Err(anyhow!(
                "Florence-2 d_model {} not divisible by attention heads ({} enc / {} dec)",
                parsed.d_model,
                parsed.encoder_attention_heads,
                parsed.decoder_attention_heads
            ));
        }
        Ok(parsed)
    }

    /// Parse from a full Florence-2 `config.json` (`model_type: florence2`),
    /// descending into its `text_config` sub-object. A bare text config (no
    /// `text_config` key) is also accepted so the sub-object can be passed
    /// directly.
    pub fn from_model_config(config: &Value) -> Result<Self> {
        match config.get("text_config") {
            Some(text) => Self::from_text_config(text),
            None => Self::from_text_config(config),
        }
    }
}
