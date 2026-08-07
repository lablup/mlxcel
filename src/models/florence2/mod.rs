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

//! Florence-2 BART-style seq2seq engine and text core.
//!
//! Florence-2 pairs a DaViT vision tower with a BART encoder-decoder
//! language model. This module implements the reusable seq2seq half: a
//! bidirectional text encoder, a causal decoder with encoder
//! cross-attention, and the dual KV cache (per-step self-attention history
//! plus one-shot cross-attention K/V) that the decode loop needs.
//!
//! [`Florence2Model`] is the assembled vision-language model: it drives the
//! DaViT tower, projects its features into the text embedding space, and
//! feeds the fused image + prompt sequence into the [`Florence2TextModel`]
//! API exposed here through [`Florence2TextModel::encode_embeds_with_mask`].
//! The task-prompt processor and runtime registration build on top of it.
//!
//! Structure mirrors `crate::models::whisper`, mlxcel's other
//! encoder-decoder family, adapted from audio-to-text to token-to-token and
//! from pre-norm to BART post-norm blocks.
//!
//! References:
//! - https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/language.py
//! - https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/config.py

mod checkpoint;
mod coords;
mod decoder;
mod encoder;
mod fusion;
pub(crate) mod layers;
mod model;
mod parse;
mod postprocess;
mod processor;
mod scan;
mod tasks;

/// DaViT vision backbone, re-exported from the shared vision-encoder tree so
/// the Florence-2 family directory exposes both halves of the model.
pub use crate::vision::encoders::florence2_davit::{
    FLORENCE2_VISION_PREFIX, Florence2DaViT, Florence2VisionConfig,
};

pub use checkpoint::{Florence2Config, Florence2TextConfig, sanitize};
pub use coords::{
    FLORENCE2_LOC_TOKEN_BASE, Florence2BoundingBox, Florence2ImageSize, Florence2Polygon,
    Florence2QuadBox, florence2_loc_token_id,
};
pub use model::Florence2Model;
pub use postprocess::{Florence2PostProcessingType, Florence2TaskResult};
pub use processor::{Florence2Output, Florence2Processor};
pub use tasks::Florence2Task;

use std::path::Path;

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::layers::{Embedding, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use decoder::Florence2Decoder;
use encoder::Florence2Encoder;
use layers::Florence2LayerCache;

/// Decode-loop state for one sequence: per-layer dual KV caches plus the
/// running token offset (the absolute position of the next decoder token).
pub struct Florence2SeqCache {
    layers: Vec<Florence2LayerCache>,
    offset: i32,
}

impl Florence2SeqCache {
    fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| Florence2LayerCache::default())
                .collect(),
            offset: 0,
        }
    }

    /// Number of decoder tokens already consumed by this cache.
    pub fn offset(&self) -> i32 {
        self.offset
    }
}

/// HuggingFace BART `shift_tokens_right`: the teacher-forcing decoder input
/// is the label sequence shifted one position right, with
/// `decoder_start_token_id` in front and any `-100` label padding replaced
/// by `pad_token_id`. Florence-2 uses `decoder_start_token_id = 2` (its EOS
/// token), per the checkpoint config.
pub fn shift_tokens_right(
    input_ids: &[i32],
    pad_token_id: i32,
    decoder_start_token_id: i32,
) -> Vec<i32> {
    let mut shifted = Vec::with_capacity(input_ids.len());
    shifted.push(decoder_start_token_id);
    shifted.extend(input_ids.iter().take(input_ids.len().saturating_sub(1)));
    for tok in shifted.iter_mut() {
        if *tok == -100 {
            *tok = pad_token_id;
        }
    }
    shifted
}

/// Loaded Florence-2 text core: shared token embedding, BART encoder,
/// causal decoder with cross-attention, and the LM head. Holds MLX weight
/// handles, so the owning provider serializes access.
pub struct Florence2TextModel {
    config: Florence2TextConfig,
    dtype: i32,
    shared: Embedding,
    encoder: Florence2Encoder,
    decoder: Florence2Decoder,
    lm_head: Option<Linear>,
}

impl Florence2TextModel {
    /// Load the text core from a Florence-2 checkpoint directory
    /// (`config.json` + safetensors). Only `language_model.*` tensors are
    /// read; the vision tower and projection weights are left on disk for
    /// the vision-fusion loader.
    pub fn load(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("Failed to read {config_path:?}: {e}"))?;
        let config_str = super::sanitize_config_json(&config_str);
        let config: Value = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse Florence-2 config: {e}"))?;
        let text_config = Florence2TextConfig::from_model_config(&config)?;

        let mut weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_path, |k| {
            k.starts_with("language_model.")
        })
        .map_err(|e| anyhow!("Failed to load Florence-2 weights: {e}"))?;
        // Apple Silicon precision policy: bf16 -> f16 for non-quantized weights.
        let _ = super::convert_bf16_weights(&mut weights);

        Self::from_weights(&weights, text_config, "language_model.")
            .map_err(|e| anyhow!("Failed to build Florence-2 text model: {e}"))
    }

    /// Build the text core from an already-loaded [`WeightMap`].
    ///
    /// `prefix` is the key prefix in front of `model.` / `lm_head.`
    /// (`"language_model."` inside a full Florence-2 checkpoint, `""` for a
    /// bare BART export). The vision-fusion loader reuses this entry point
    /// with the full-checkpoint weight map so the checkpoint is only read
    /// once.
    pub fn from_weights(
        weights: &WeightMap,
        config: Florence2TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let shared = Embedding::from_weights(weights, &format!("{prefix}model.shared"))?;
        let dtype = mlxcel_core::array_dtype(&shared.weight);

        let encoder =
            Florence2Encoder::from_weights(weights, &format!("{prefix}model.encoder"), &config)?;
        let decoder = Florence2Decoder::from_weights(
            weights,
            &format!("{prefix}model.decoder"),
            &config,
            dtype,
        )?;

        // `tie_word_embeddings` is true for Florence-2, but checkpoints ship
        // a materialized `lm_head.weight` as well; use it when present and
        // fall back to the tied shared embedding otherwise.
        let lm_head = if weights.contains_key(&format!("{prefix}lm_head.weight")) {
            Some(Linear::from_weights(weights, &format!("{prefix}lm_head"))?)
        } else {
            None
        };

        Ok(Self {
            config,
            dtype,
            shared,
            encoder,
            decoder,
            lm_head,
        })
    }

    /// Parsed text configuration.
    pub fn config(&self) -> &Florence2TextConfig {
        &self.config
    }

    /// MLX dtype the weights are held in (f16 after the bf16 conversion).
    pub fn dtype(&self) -> i32 {
        self.dtype
    }

    /// Look up `[batch, seq]` token ids in the shared embedding table,
    /// applying the BART `embed_scale` (identity for Florence-2, whose
    /// config sets `scale_embedding: false`). The vision-fusion path uses
    /// this to embed the text half before splicing in image features.
    pub fn embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embeds = self.shared.forward(input_ids);
        if self.config.scale_embedding {
            mlxcel_core::multiply_scalar(&embeds, (self.config.d_model as f32).sqrt())
        } else {
            embeds
        }
    }

    /// Encode `[batch, seq]` token ids into encoder hidden states
    /// `[batch, seq, d_model]`.
    ///
    /// Precondition: `seq` must not exceed
    /// [`Florence2TextConfig::max_position_embeddings`]; the learned position
    /// table has no rows past that bound. [`Self::generate_greedy`] checks
    /// this; callers embedding longer sequences must bound them first.
    pub fn encode_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embeds = self.embed_tokens(input_ids);
        self.encoder.forward(&embeds, None)
    }

    /// Encode pre-computed `[batch, seq, d_model]` input embeddings. This is
    /// the entry point the vision-fusion path drives with concatenated
    /// image + text embeddings.
    ///
    /// Precondition: `seq` must not exceed
    /// [`Florence2TextConfig::max_position_embeddings`], as in
    /// [`Self::encode_tokens`].
    pub fn encode_embeds(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        self.encoder.forward(inputs_embeds, None)
    }

    /// [`Self::encode_embeds`] with an additive attention mask
    /// (`[batch, 1, 1, seq]`, `0` for a real key and `-inf` for a padded
    /// one). [`Florence2Model`] builds the mask from the joint image + prompt
    /// attention mask; `None` is equivalent to [`Self::encode_embeds`].
    pub fn encode_embeds_with_mask(
        &self,
        inputs_embeds: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.encoder.forward(inputs_embeds, mask)
    }

    /// Fresh decode-loop cache sized to the decoder depth.
    pub fn make_cache(&self) -> Florence2SeqCache {
        Florence2SeqCache::new(self.decoder.num_layers())
    }

    /// Run the decoder over `[batch, seq]` token ids attending to
    /// `encoder_hidden_states`, advancing `cache` by `seq` positions.
    /// Returns logits `[batch, seq, vocab_size]`.
    ///
    /// The first call against a fresh cache projects and stores the
    /// cross-attention K/V from the encoder output; subsequent calls reuse
    /// them and only append self-attention K/V, so an incremental decode
    /// step costs O(1) encoder-side work.
    pub fn decode(
        &self,
        decoder_input_ids: &MlxArray,
        encoder_hidden_states: &MlxArray,
        cache: &mut Florence2SeqCache,
    ) -> UniquePtr<MlxArray> {
        let embeds = self.embed_tokens(decoder_input_ids);
        let hidden = self.decoder.forward(
            &embeds,
            encoder_hidden_states,
            cache.offset,
            &mut cache.layers,
        );
        cache.offset += mlxcel_core::array_shape(decoder_input_ids)[1];
        match &self.lm_head {
            Some(head) => head.forward(&hidden),
            None => self.shared.as_linear(&hidden),
        }
    }

    /// Text-only greedy round trip: encode `input_ids`, seed the decoder
    /// with `decoder_start_token_id`, and decode until EOS or
    /// `max_new_tokens`. Returns the generated ids (EOS excluded).
    ///
    /// This is the reference decode loop for the engine; task-prompted
    /// generation through the vision path replaces the encoder input but
    /// drives the same [`Self::decode`] / [`Florence2SeqCache`] machinery.
    pub fn generate_greedy(&self, input_ids: &[i32], max_new_tokens: usize) -> Result<Vec<i32>> {
        if input_ids.is_empty() {
            return Err(anyhow!("Florence-2 encoder input is empty"));
        }
        if input_ids.len() > self.config.max_position_embeddings as usize {
            return Err(anyhow!(
                "Florence-2 encoder input length {} exceeds max_position_embeddings {}",
                input_ids.len(),
                self.config.max_position_embeddings
            ));
        }
        let prompt = mlxcel_core::from_slice_i32(input_ids, &[1, input_ids.len() as i32]);
        let encoder_hidden = self.encode_tokens(&prompt);

        let mut cache = self.make_cache();
        let mut generated = Vec::new();
        let mut next = self.config.decoder_start_token_id;
        for _ in 0..max_new_tokens {
            if cache.offset() >= self.config.max_position_embeddings {
                break;
            }
            let tok = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
            let logits = self.decode(&tok, &encoder_hidden, &mut cache);
            next = argmax_last_position(&logits)?;
            if next == self.config.eos_token_id {
                break;
            }
            generated.push(next);
        }
        Ok(generated)
    }
}

/// Argmax over the vocabulary at the last position of `[batch, seq, vocab]`
/// logits. This is the single point in the greedy loop where the lazy MLX
/// graph is forced, routed through the fallible [`mlxcel_core::try_eval`]
/// boundary so an MLX failure surfaces as `Err` instead of aborting the
/// process with an uncaught C++ exception.
pub(crate) fn argmax_last_position(logits: &MlxArray) -> Result<i32> {
    let shape = mlxcel_core::array_shape(logits);
    let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
    let last = mlxcel_core::astype(&last, mlxcel_core::dtype::FLOAT32);
    let idx = mlxcel_core::argmax(&last, -1, false);
    mlxcel_core::try_eval(&idx).map_err(|e| anyhow!("Florence-2 logits evaluation failed: {e}"))?;
    Ok(mlxcel_core::item_i32(&idx))
}

#[cfg(test)]
#[path = "florence2_tests.rs"]
mod florence2_tests;
