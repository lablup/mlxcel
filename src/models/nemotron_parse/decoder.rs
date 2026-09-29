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

//! Nemotron-Parse text decoder: HF `MBartDecoderLayer` blocks, which are
//! pre-norm (`x = x + Sublayer(LN(x))`), behind an embedding LayerNorm and
//! followed by a final `layer_norm`. Unlike Florence-2's BART decoder there
//! is no learned position table at all; order comes only from the causal
//! mask.
//!
//! The attention sublayers and the dual (self + one-shot cross) KV cache are
//! the Florence-2 seq2seq pieces, reused as-is.
//!
//! Reference: `NemotronParseDecoder` in `hf_nemotron_parse_modeling.py` and
//! `transformers.models.mbart.modeling_mbart.MBartDecoderLayer`.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::florence2::Florence2Quantization;
use crate::models::florence2::layers::{
    Florence2Attention, Florence2LayerCache, additive_causal_mask, layer_norm,
};

use super::config::NemotronParseTextConfig;

pub(crate) struct NemotronParseDecoderLayer {
    self_attn: Florence2Attention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: Florence2Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    final_layer_norm: LayerNorm,
}

impl NemotronParseDecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        n_head: i32,
        q: Florence2Quantization,
    ) -> Result<Self, String> {
        let lin = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), q.group_size, q.bits)
                .map_err(|e| format!("Nemotron-Parse {e}"))
        };
        Ok(Self {
            self_attn: Florence2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                n_head,
                q,
            )?,
            self_attn_layer_norm: layer_norm(weights, &format!("{prefix}.self_attn_layer_norm"))?,
            encoder_attn: Florence2Attention::from_weights(
                weights,
                &format!("{prefix}.encoder_attn"),
                n_head,
                q,
            )?,
            encoder_attn_layer_norm: layer_norm(
                weights,
                &format!("{prefix}.encoder_attn_layer_norm"),
            )?,
            fc1: lin("fc1")?,
            fc2: lin("fc2")?,
            final_layer_norm: layer_norm(weights, &format!("{prefix}.final_layer_norm"))?,
        })
    }

    /// Pre-norm order: each sublayer reads a normalized copy and adds its
    /// output back onto the un-normalized residual stream.
    fn forward(
        &self,
        x: &MlxArray,
        xa: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut Florence2LayerCache,
    ) -> UniquePtr<MlxArray> {
        let h = self.self_attn_layer_norm.forward(x);
        let h = self.self_attn.self_attention(&h, mask, &mut cache.self_kv);
        let x = mlxcel_core::add(x, &h);

        let h = self.encoder_attn_layer_norm.forward(&x);
        let h = self
            .encoder_attn
            .cross_attention(&h, xa, &mut cache.cross_kv);
        let x = mlxcel_core::add(&x, &h);

        let h = self.final_layer_norm.forward(&x);
        let h = mlxcel_core::gelu(&self.fc1.forward(&h));
        let h = self.fc2.forward(&h);
        mlxcel_core::add(&x, &h)
    }
}

pub(crate) struct NemotronParseDecoder {
    layernorm_embedding: LayerNorm,
    layers: Vec<NemotronParseDecoderLayer>,
    layer_norm: LayerNorm,
}

impl NemotronParseDecoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronParseTextConfig,
        quantization: Florence2Quantization,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            layers.push(NemotronParseDecoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config.decoder_attention_heads,
                quantization,
            )?);
        }
        Ok(Self {
            layernorm_embedding: layer_norm(weights, &format!("{prefix}.layernorm_embedding"))?,
            layers,
            layer_norm: layer_norm(weights, &format!("{prefix}.layer_norm"))?,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Run one decoder call over already-scaled `[B, T, d_model]` token
    /// embeddings. `offset` is the number of positions already cached; a
    /// multi-token call gets a causal mask, a single-token step attends to
    /// the whole cached history. Returns the final-normed hidden states.
    pub(crate) fn forward(
        &self,
        inputs_embeds: &MlxArray,
        xa: &MlxArray,
        offset: i32,
        caches: &mut [Florence2LayerCache],
    ) -> UniquePtr<MlxArray> {
        debug_assert_eq!(caches.len(), self.layers.len());
        let seq = mlxcel_core::array_shape(inputs_embeds)[1];
        let mut x = self.layernorm_embedding.forward(inputs_embeds);
        let mask = (seq > 1)
            .then(|| additive_causal_mask(seq, offset, mlxcel_core::array_dtype(inputs_embeds)));
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            x = layer.forward(&x, xa, mask.as_deref(), cache);
        }
        self.layer_norm.forward(&x)
    }
}
