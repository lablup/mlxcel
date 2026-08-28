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

//! Nemotron-3-Embed: the Ministral 3 backbone run bidirectionally, mean
//! pooled.
//!
//! The checkpoint declares `model_type: ministral3`, `architectures:
//! ["Ministral3Model"]` and `is_causal: false`, which is the flag detection
//! keys on. The layers, the Llama 4 attention scaling and the RoPE schedule
//! are exactly [`crate::models::ministral3`]'s; three things differ from the
//! generator:
//!
//! - Every layer sees `create_bidirectional_padding_mask` instead of a causal
//!   mask. The published checkpoints carry `sliding_window: null` and no
//!   `layer_types`, so every layer is full attention; a checkpoint that did
//!   declare `sliding_attention` layers gets
//!   `create_bidirectional_window_mask` for those, which is the bidirectional
//!   analogue of the generator's sliding overlay.
//! - The per-position Llama 4 attention scale is computed at offset 0 for the
//!   whole input, since the embedder runs one prefill and never decodes. This
//!   is the same [`crate::models::ministral3::get_llama4_attn_scale`] the
//!   generator calls, so a sequence of length `L` gets identical scales on
//!   both paths.
//! - No head is applied. `tie_word_embeddings` is true on both published
//!   checkpoints, so `Ministral3Model::from_weights` leaves `lm_head` as
//!   `None` and the forward pass here stops at the final norm.
//!
//! Each call builds fresh caches through `Ministral3Model::make_caches`, so
//! every prefill starts at offset 0 and a right-padded batch reproduces the
//! unpadded single row.
//!
//! Prompt prefixes (`query: ` / `passage: `) are caller-side and documented in
//! `docs/embeddings.md`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::{create_bidirectional_padding_mask, create_bidirectional_window_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::limits::config_normalize_flag;
use crate::embeddings::loader::load_embedding_weights;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::models::embedding_sanitize::sanitize_decoder_embedding_weights;
use crate::models::ministral3::{Ministral3Model, ModelArgs, get_llama4_attn_scale};

/// Nemotron-3-Embed: bidirectional Ministral 3 layers, mean pooling.
pub struct Ministral3EmbeddingModel {
    model: Ministral3Model,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

impl Ministral3EmbeddingModel {
    /// Load a Nemotron-3-Embed checkpoint from `model_dir`.
    ///
    /// Both the NVIDIA bf16 original and the `mlx-community` 8-bit conversion
    /// save the inner `Ministral3Model`, so the backbone roots arrive bare
    /// (`embed_tokens.weight`, `layers.0.…`, `norm.weight`) and are prefixed
    /// with `model.` before the Ministral 3 constructor runs.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let mut args: ModelArgs = serde_json::from_value(config.clone())
            .context("failed to parse the Nemotron-3-Embed config.json as a Ministral 3 config")?;
        // The embedder never applies a head; the tied flag keeps
        // `Ministral3Model::from_weights` from looking for an `lm_head` the
        // sanitize pass has just dropped.
        args.tie_word_embeddings = true;

        let mut weights = load_embedding_weights(model_dir, config)?;
        let dense_folders = sanitize_decoder_embedding_weights(&mut weights);
        if dense_folders > 0 {
            bail!(
                "{} carries {dense_folders} sentence-transformers Dense module(s). \
                 Nemotron-3-Embed applies no post-pooling projection, so loading it would \
                 silently drop a trained layer; this checkpoint variant is not supported",
                model_dir.display()
            );
        }

        let pooling = resolve_pooling_mode(model_dir, PoolingMode::Mean)?;
        Self::from_weights(&weights, &args, pooling, config_normalize_flag(config))
    }

    /// Build the model from an already sanitized weight map.
    ///
    /// Split from [`Self::load`] so the mask, the attention scale and the
    /// pooling can be exercised on a synthetic checkpoint-free model.
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        pooling: PoolingMode,
        normalize: bool,
    ) -> Result<Self> {
        let model = Ministral3Model::from_weights(weights, args)
            .map_err(|e| anyhow::anyhow!("Nemotron-3-Embed: {e}"))?;
        Ok(Self {
            model,
            pooling,
            normalize,
            embedding_dim: args.hidden_size,
        })
    }

    /// Run the bidirectional backbone and the final norm, returning the
    /// `[B, L, hidden_size]` hidden states before pooling.
    ///
    /// `attention_mask` is the `[B, L]` int32 padding mask (`1` = real token).
    pub(crate) fn forward_hidden(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.model.embed_tokens.forward(input_ids);
        let seq_len = mlxcel_core::array_shape(&h)[1];

        // Offset 0: one prefill per call, so position `i` is token `i`.
        let attn_scale = match &self.model.rope_params {
            Some(params) => get_llama4_attn_scale(
                seq_len,
                0,
                params.llama_4_scaling_beta,
                params.original_max_position_embeddings,
            ),
            None => vec![1.0; seq_len as usize],
        };

        let full_mask = create_bidirectional_padding_mask(attention_mask);
        let has_sliding = self.model.layers.iter().any(|layer| layer.use_sliding);
        let window_mask = match self.model.sliding_window.filter(|_| has_sliding) {
            Some(window) if window >= 1 => Some(create_bidirectional_window_mask(
                attention_mask,
                window as i32,
            )),
            _ => None,
        };

        let mut caches = self.model.make_caches();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let mask: &MlxArray = if layer.use_sliding {
                window_mask.as_deref().unwrap_or(&full_mask)
            } else {
                &full_mask
            };
            h = layer.forward(&h, &attn_scale, caches[i].as_interface(), Some(mask));
        }
        self.model.norm.forward(&h)
    }
}

impl EmbeddingModel for Ministral3EmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("Nemotron-3-Embed is a text-only embedder and does not accept images");
        }

        let hidden = self.forward_hidden(batch.input_ids, batch.attention_mask);
        let pooled = pool(&hidden, batch.attention_mask, self.pooling);

        Ok(EmbeddingOutput {
            embeddings: pooled,
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::Mean
    }

    fn pooling(&self) -> PoolingMode {
        self.pooling
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
#[path = "ministral3_embedding_tests.rs"]
mod ministral3_embedding_tests;
