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

//! Whisper text decoder: token + learned positional embeddings, N blocks with
//! masked self-attention and cross-attention to the encoder output, a final
//! LayerNorm, and tied output logits.

use mlxcel_core::layers::{Embedding, LayerNorm};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::WhisperDims;
use super::layers::{KvCache, ResidualAttentionBlock, additive_causal_mask};

pub(crate) struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: UniquePtr<MlxArray>,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    n_state: i32,
    dtype: i32,
}

impl TextDecoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        dims: &WhisperDims,
        dtype: i32,
    ) -> Result<Self, String> {
        let token_embedding = Embedding::from_weights(weights, "decoder.token_embedding")?;
        let positional_embedding = weights
            .get("decoder.positional_embedding")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Whisper weight not found: decoder.positional_embedding".to_string())?;

        let mut blocks = Vec::with_capacity(dims.n_text_layer as usize);
        for i in 0..dims.n_text_layer {
            blocks.push(ResidualAttentionBlock::from_weights(
                weights,
                &format!("decoder.blocks.{i}"),
                dims.n_text_head,
                true,
            )?);
        }

        let ln_weight = weights
            .get("decoder.ln.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Whisper weight not found: decoder.ln.weight".to_string())?;
        let ln_bias = weights.get("decoder.ln.bias").map(|w| mlxcel_core::copy(w));

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln: LayerNorm::new(ln_weight, ln_bias, 1e-5),
            n_state: dims.n_text_state,
            dtype,
        })
    }

    /// Number of decoder layers (cache vectors are sized to this).
    pub(crate) fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Run one decode call over `tokens` (`[batch, seq]`) attending to encoder
    /// features `xa`. `offset` is the number of tokens already cached (i.e. the
    /// position of the first token in this call). Returns logits
    /// `[batch, seq, n_vocab]`.
    pub(crate) fn forward(
        &self,
        tokens: &MlxArray,
        xa: &MlxArray,
        offset: i32,
        self_caches: &mut [Option<KvCache>],
        cross_caches: &mut [Option<KvCache>],
    ) -> UniquePtr<MlxArray> {
        let seq = mlxcel_core::array_shape(tokens)[1];

        let emb = self.token_embedding.forward(tokens);
        let pos = mlxcel_core::slice(
            &self.positional_embedding,
            &[offset, 0],
            &[offset + seq, self.n_state],
        );
        let mut x = mlxcel_core::add(&emb, &pos);

        // A causal mask is only needed for a multi-token prefill; a single new
        // token attends to the whole cached history without masking.
        let mask = if seq > 1 {
            Some(additive_causal_mask(seq, self.dtype))
        } else {
            None
        };
        let mask_ref = mask.as_deref();

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                Some(xa),
                mask_ref,
                &mut self_caches[i],
                &mut cross_caches[i],
            );
        }

        let x = self.ln.forward(&x);
        self.token_embedding.as_linear(&x)
    }
}
