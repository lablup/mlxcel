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

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Florence2TextConfig;
use super::encoder::POSITION_OFFSET;
use super::layers::{Florence2DecoderLayer, Florence2LayerCache, additive_causal_mask, layer_norm};

pub(crate) struct Florence2Decoder {
    embed_positions: UniquePtr<MlxArray>,
    layernorm_embedding: LayerNorm,
    layers: Vec<Florence2DecoderLayer>,
    d_model: i32,
    dtype: i32,
}

impl Florence2Decoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Florence2TextConfig,
        dtype: i32,
    ) -> Result<Self, String> {
        let embed_positions = weights
            .get(&format!("{prefix}.embed_positions.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| {
                format!("Florence-2 weight not found: {prefix}.embed_positions.weight")
            })?;
        let layernorm_embedding = layer_norm(weights, &format!("{prefix}.layernorm_embedding"))?;

        let mut layers = Vec::with_capacity(config.decoder_layers as usize);
        for i in 0..config.decoder_layers {
            layers.push(Florence2DecoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config.decoder_attention_heads,
            )?);
        }

        Ok(Self {
            embed_positions,
            layernorm_embedding,
            layers,
            d_model: config.d_model,
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
        let pos = mlxcel_core::slice(
            &self.embed_positions,
            &[POSITION_OFFSET + offset, 0],
            &[POSITION_OFFSET + offset + seq, self.d_model],
        );
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
