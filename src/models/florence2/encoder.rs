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

//! Florence-2 bidirectional text encoder: learned absolute position
//! embeddings (with the BART table offset of 2), an embedding LayerNorm, and
//! N post-norm self-attention blocks. No causal mask; every position attends
//! to every other.
//!
//! The encoder consumes pre-computed input embeddings rather than token ids
//! so the vision-fusion path can feed a concatenated image + text embedding
//! sequence through the same code (the token embedding itself lives on
//! [`super::Florence2TextModel`], which owns the shared embedding table).
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/language.py
//! (Florence2Encoder)

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Florence2TextConfig;
use super::layers::{Florence2EncoderLayer, embedding_table_rows, layer_norm};

/// BART reserves the first two rows of the learned position table (a legacy
/// of fairseq's padding-offset scheme); real positions start at row 2.
pub(crate) const POSITION_OFFSET: i32 = 2;

pub(crate) struct Florence2Encoder {
    embed_positions: UnifiedEmbedding,
    layernorm_embedding: LayerNorm,
    layers: Vec<Florence2EncoderLayer>,
}

impl Florence2Encoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Florence2TextConfig,
    ) -> Result<Self, String> {
        // `Florence2Model` bounds both the fused encoder input length and the
        // decode offset against `config.max_position_embeddings`, and those
        // guards are only sound if the loaded table actually covers that
        // bound. Without this check a `config.json` that inflates
        // `max_position_embeddings` (or `d_model`) passes them and reaches MLX
        // as an out-of-range gather, which reads past the end of the table
        // rather than faulting. Sum in i64 so the bound cannot overflow on a
        // hostile value.
        let key = format!("{prefix}.embed_positions");
        let embed_positions = UnifiedEmbedding::from_weights(
            weights,
            &key,
            config.quantization.group_size,
            config.quantization.bits,
        )
        .map_err(|e| format!("Florence-2 {e}"))?;
        let required_rows = POSITION_OFFSET as i64 + config.max_position_embeddings as i64;
        embedding_table_rows(
            &embed_positions,
            &key,
            required_rows,
            "max_position_embeddings + POSITION_OFFSET",
            config.d_model,
            "d_model",
        )?;
        let layernorm_embedding = layer_norm(weights, &format!("{prefix}.layernorm_embedding"))?;

        let mut layers = Vec::with_capacity(config.encoder_layers as usize);
        for i in 0..config.encoder_layers {
            layers.push(Florence2EncoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config.encoder_attention_heads,
                config.quantization,
            )?);
        }

        Ok(Self {
            embed_positions,
            layernorm_embedding,
            layers,
        })
    }

    /// Encode `[batch, seq, d_model]` input embeddings into encoder hidden
    /// states of the same shape.
    ///
    /// `mask` is the additive attention mask (`[batch, 1, 1, seq]`, `0` for a
    /// real key and `-inf` for a padded one) applied by every block's
    /// self-attention. Unpadded input passes `None`.
    pub(crate) fn forward(
        &self,
        inputs_embeds: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let seq = mlxcel_core::array_shape(inputs_embeds)[1];
        // Rows `[POSITION_OFFSET, POSITION_OFFSET + seq)` of the learned
        // table, gathered rather than sliced so the quantized and dense arms
        // are the same call. On the dense arm the gather returns exactly the
        // rows the previous slice returned.
        let positions = mlxcel_core::arange_i32(POSITION_OFFSET, POSITION_OFFSET + seq, 1);
        let pos = self.embed_positions.forward(&positions);
        let x = mlxcel_core::add(inputs_embeds, &pos);
        let mut x = self.layernorm_embedding.forward(&x);
        for layer in &self.layers {
            x = layer.forward(&x, mask);
        }
        x
    }
}
