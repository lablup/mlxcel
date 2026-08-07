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

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Florence2TextConfig;
use super::layers::{Florence2EncoderLayer, layer_norm};

/// BART reserves the first two rows of the learned position table (a legacy
/// of fairseq's padding-offset scheme); real positions start at row 2.
pub(crate) const POSITION_OFFSET: i32 = 2;

pub(crate) struct Florence2Encoder {
    embed_positions: UniquePtr<MlxArray>,
    layernorm_embedding: LayerNorm,
    layers: Vec<Florence2EncoderLayer>,
    d_model: i32,
}

impl Florence2Encoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Florence2TextConfig,
    ) -> Result<Self, String> {
        let embed_positions = weights
            .get(&format!("{prefix}.embed_positions.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| {
                format!("Florence-2 weight not found: {prefix}.embed_positions.weight")
            })?;
        // `Florence2Model` bounds both the fused encoder input length and the
        // decode offset against `config.max_position_embeddings`, and those
        // guards are only sound if the loaded table actually covers that
        // bound. Without this check a `config.json` that inflates
        // `max_position_embeddings` (or `d_model`) passes them and reaches MLX
        // as an out-of-range slice, which throws across the cxx bridge and
        // aborts the process instead of returning `Err`. Sum in i64 so the
        // comparison cannot overflow on a hostile value.
        let position_shape = mlxcel_core::array_shape(&embed_positions);
        let required_rows = POSITION_OFFSET as i64 + config.max_position_embeddings as i64;
        if position_shape.len() != 2
            || position_shape[1] != config.d_model
            || (position_shape[0] as i64) < required_rows
        {
            return Err(format!(
                "Florence-2 {prefix}.embed_positions.weight: expected at least [{required_rows}, {}] (max_position_embeddings {} + POSITION_OFFSET {POSITION_OFFSET}), got {position_shape:?}",
                config.d_model, config.max_position_embeddings
            ));
        }
        let layernorm_embedding = layer_norm(weights, &format!("{prefix}.layernorm_embedding"))?;

        let mut layers = Vec::with_capacity(config.encoder_layers as usize);
        for i in 0..config.encoder_layers {
            layers.push(Florence2EncoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config.encoder_attention_heads,
            )?);
        }

        Ok(Self {
            embed_positions,
            layernorm_embedding,
            layers,
            d_model: config.d_model,
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
        let pos = mlxcel_core::slice(
            &self.embed_positions,
            &[POSITION_OFFSET, 0],
            &[POSITION_OFFSET + seq, self.d_model],
        );
        let x = mlxcel_core::add(inputs_embeds, &pos);
        let mut x = self.layernorm_embedding.forward(&x);
        for layer in &self.layers {
            x = layer.forward(&x, mask);
        }
        x
    }
}
