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

//! Florence-2 causal text decoder: learned absolute position embeddings
//! indexed by the running decode offset, an embedding LayerNorm, and N
//! post-norm blocks with causal self-attention plus cross-attention to the
//! encoder output. Returns hidden states; the owning model applies the
//! LM head.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/language.py
//! (Florence2Decoder)

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Florence2TextConfig;
use super::encoder::POSITION_OFFSET;
use super::layers::{
    Florence2DecoderLayer, Florence2LayerCache, additive_causal_mask, embedding_table_rows,
    layer_norm,
};

pub(crate) struct Florence2Decoder {
    embed_positions: UnifiedEmbedding,
    layernorm_embedding: LayerNorm,
    layers: Vec<Florence2DecoderLayer>,
    dtype: i32,
}

impl Florence2Decoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Florence2TextConfig,
        dtype: i32,
    ) -> Result<Self, String> {
        // Same bound as the encoder table, and for the same reason: the
        // decode offset is checked against `config.max_position_embeddings`,
        // so that check is only sound if the loaded table covers it.
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

        let mut layers = Vec::with_capacity(config.decoder_layers as usize);
        for i in 0..config.decoder_layers {
            layers.push(Florence2DecoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config.decoder_attention_heads,
                config.quantization,
            )?);
        }

        Ok(Self {
            embed_positions,
            layernorm_embedding,
            layers,
            dtype,
        })
    }

    /// Number of decoder layers (the per-layer cache vector is sized to this).
    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Run one decode call over `[batch, seq, d_model]` input embeddings
    /// attending to encoder output `xa`. `offset` is the number of decoder
    /// tokens already cached (the absolute position of the first token in
    /// this call). Returns hidden states `[batch, seq, d_model]`.
    pub(crate) fn forward(
        &self,
        inputs_embeds: &MlxArray,
        xa: &MlxArray,
        offset: i32,
        caches: &mut [Florence2LayerCache],
    ) -> UniquePtr<MlxArray> {
        debug_assert_eq!(caches.len(), self.layers.len());
        let seq = mlxcel_core::array_shape(inputs_embeds)[1];
        // Rows `[POSITION_OFFSET + offset, ... + seq)`, gathered so the dense
        // and quantized tables go through one call. See the encoder.
        let positions =
            mlxcel_core::arange_i32(POSITION_OFFSET + offset, POSITION_OFFSET + offset + seq, 1);
        let pos = self.embed_positions.forward(&positions);
        let x = mlxcel_core::add(inputs_embeds, &pos);
        let mut x = self.layernorm_embedding.forward(&x);

        // A causal mask is only needed when this call carries more than one
        // token; a single new token attends to the whole cached history.
        let mask = if seq > 1 {
            Some(additive_causal_mask(seq, offset, self.dtype))
        } else {
            None
        };
        let mask_ref = mask.as_deref();

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            x = layer.forward(&x, xa, mask_ref, cache);
        }
        x
    }
}
