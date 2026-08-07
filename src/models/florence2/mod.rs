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
mod render;
mod runtime;
mod scan;
mod tasks;

/// DaViT vision backbone, re-exported from the shared vision-encoder tree so
/// the Florence-2 family directory exposes both halves of the model.
pub use crate::vision::encoders::florence2_davit::{
    FLORENCE2_VISION_PREFIX, Florence2DaViT, Florence2VisionConfig,
};

pub(crate) use checkpoint::reject_unsupported_quantized_tensors;
pub use checkpoint::{Florence2Config, Florence2Quantization, Florence2TextConfig, sanitize};
pub use coords::{
    FLORENCE2_LOC_TOKEN_BASE, Florence2BoundingBox, Florence2ImageSize, Florence2Polygon,
    Florence2QuadBox, florence2_loc_token_id,
};
pub use model::Florence2Model;
pub use postprocess::{Florence2PostProcessingType, Florence2TaskResult};
pub use processor::{Florence2Output, Florence2Processor};
pub use render::{render_task_result, structured_task_json};
pub use runtime::{Florence2RunOutput, Florence2VlmModel, parse_task_prompt};
pub use tasks::Florence2Task;

use std::path::Path;

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::layers::{UnifiedEmbedding, UnifiedLinear};
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
    shared: UnifiedEmbedding,
    encoder: Florence2Encoder,
    decoder: Florence2Decoder,
    lm_head: Option<UnifiedLinear>,
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

        let weights = mlxcel_core::weights::load_weights_from_dir_filtered(model_path, |k| {
            k.starts_with("language_model.")
        })
        .map_err(|e| anyhow!("Failed to load Florence-2 weights: {e}"))?;
        // Same three steps the whole-model loader runs, in the same order, so
        // that loading the text half on its own is not a weaker path than
        // loading it as part of `Florence2Model`. `sanitize` matters here for
        // the shared-embedding fill: BART ties the encoder and decoder token
        // tables to `model.shared` and exports vary in which of the three they
        // materialize, so without it a checkpoint carrying only `embed_tokens`
        // fails to find `model.shared`. The refusal matters because every
        // LayerNorm in this stack is read as a raw weight and handed to
        // `fast::layer_norm`, so a packed one aborts the process here exactly
        // as it would through the fused loader.
        let mut weights = sanitize(weights);
        reject_unsupported_quantized_tensors(&weights).map_err(|e| anyhow!("{e}"))?;
        // Apple Silicon precision policy: bf16 -> f16, but only for a dense
        // export. A quantized one keeps its scales and biases at the stored
        // width; they are dequantization operands rather than activations, so
        // rounding them changes every weight the stack reconstructs.
        if !Florence2Quantization::config_is_quantized(&config) {
            let _ = super::convert_bf16_weights(&mut weights);
        }

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
        let shared_key = format!("{prefix}model.shared");
        let shared = UnifiedEmbedding::from_weights(
            weights,
            &shared_key,
            config.quantization.group_size,
            config.quantization.bits,
        )
        .map_err(|e| format!("Florence-2 {e}"))?;
        layers::embedding_table_rows(
            &shared,
            &shared_key,
            config.vocab_size as i64,
            "vocab_size",
            config.d_model,
            "d_model",
        )?;
        // The activation dtype has to come from the scales on the quantized
        // arm: `weight` there is the packed `uint32` plane, and taking its
        // dtype would make the decoder build its causal mask (and the fusion
        // path cast its image features) to an integer type.
        let dtype = match shared.quantized() {
            Some(quantized) => mlxcel_core::array_dtype(&quantized.scales),
            None => mlxcel_core::array_dtype(shared.weight()),
        };

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
        let lm_head_key = format!("{prefix}lm_head");
        let lm_head = match weights.get(&format!("{lm_head_key}.weight")) {
            Some(head_weight) => {
                // The head's output width is what bounds the next token id.
                // `generate_greedy` takes `argmax_last_position` over these
                // logits and feeds the result straight back into
                // `embed_tokens`, which is a gather into `shared`, and the
                // guard above only proves `shared` has *at least*
                // `vocab_size` rows. A head wider than the token table would
                // let the argmax return an id past the table's last row, and
                // MLX does not range-check a positive gather index: that
                // lookup reads past the end of the buffer instead of faulting.
                // Requiring the two to agree is what closes the one-sided
                // guard. Dimension 0 is the row axis on both arms, since
                // quantization packs the input axis only.
                let shape = mlxcel_core::array_shape(head_weight);
                let [rows, _] = shape.as_slice() else {
                    return Err(format!(
                        "Florence-2 {lm_head_key}.weight must be a 2-D [vocab_size, d_model] \
                         matrix, got shape {shape:?}"
                    ));
                };
                if *rows != config.vocab_size {
                    return Err(format!(
                        "Florence-2 {lm_head_key}.weight has {rows} output rows but config \
                         vocab_size is {}; the argmax over these logits indexes the shared token \
                         table, whose gather does not range-check a positive index",
                        config.vocab_size
                    ));
                }
                Some(
                    UnifiedLinear::from_weights(
                        weights,
                        &lm_head_key,
                        config.quantization.group_size,
                        config.quantization.bits,
                    )
                    .map_err(|e| format!("Florence-2 {e}"))?,
                )
            }
            None => None,
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
    /// Precondition, memory safety: `seq` must not exceed
    /// [`Florence2TextConfig::max_position_embeddings`]; the learned position
    /// table has no rows past that bound. The encoder reads its position rows
    /// with a gather into that table, and MLX does not range-check a positive
    /// gather index, so violating this does not fault or error. It reads past
    /// the end of the table (and, on a quantized checkpoint, past the packed
    /// weight, scales, and biases planes) and the result reaches the hidden
    /// states. [`Self::generate_greedy`] and [`Florence2Model::encode_fused`]
    /// check this; every other caller must bound `seq` itself.
    pub fn encode_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embeds = self.embed_tokens(input_ids);
        self.encoder.forward(&embeds, None)
    }

    /// Encode pre-computed `[batch, seq, d_model]` input embeddings. This is
    /// the entry point the vision-fusion path drives with concatenated
    /// image + text embeddings.
    ///
    /// Precondition, memory safety: `seq` must not exceed
    /// [`Florence2TextConfig::max_position_embeddings`], and for the same
    /// reason as in [`Self::encode_tokens`]: the position lookup behind it is
    /// an unchecked gather, so a longer sequence reads past the table rather
    /// than failing.
    pub fn encode_embeds(&self, inputs_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        self.encoder.forward(inputs_embeds, None)
    }

    /// [`Self::encode_embeds`] with an additive attention mask
    /// (`[batch, 1, 1, seq]`, `0` for a real key and `-inf` for a padded
    /// one). [`Florence2Model`] builds the mask from the joint image + prompt
    /// attention mask; `None` is equivalent to [`Self::encode_embeds`].
    ///
    /// Carries the same memory-safety precondition on `seq` as
    /// [`Self::encode_embeds`]. The mask bounds which keys are attended, not
    /// which position rows are gathered, so it does not substitute for the
    /// length check.
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
    ///
    /// Precondition, memory safety: `cache.offset() + seq` must not exceed
    /// [`Florence2TextConfig::max_position_embeddings`]. The decoder gathers
    /// rows `[POSITION_OFFSET + offset, POSITION_OFFSET + offset + seq)` of
    /// the learned position table, so the running offset counts against the
    /// same bound the sequence length does. MLX does not range-check a
    /// positive gather index, so exceeding it reads past the table instead of
    /// faulting. [`Self::generate_greedy`] and
    /// [`Florence2Model::generate_greedy`] hold the bound by breaking their
    /// loop once `cache.offset()` reaches it.
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
