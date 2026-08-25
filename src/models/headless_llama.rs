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

//! A Llama decoder assembled without a generation head.
//!
//! [`crate::models::Llama3Model`] always loads an `lm_head`, tied or untied,
//! and a checkpoint exported for retrieval can ship neither: ColIdefics3
//! declares `tie_word_embeddings: false` and carries no head tensor at all,
//! so the generator's constructor fails on it. The blocks, the final norm
//! and the embedding table are the same public types the generator uses, so
//! this is an assembly difference rather than a second implementation of the
//! architecture, and [`HeadlessLlama::forward_hidden`] is the generator's
//! forward pass truncated at the final norm.
//!
//! Used by: `crate::models::colidefics3`.

use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::llama3::{ModelArgs, TransformerBlock};

/// A Llama backbone: embedding table, transformer blocks, final norm.
pub(crate) struct HeadlessLlama {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
}

impl HeadlessLlama {
    /// Build from weights keyed the way the generator expects
    /// (`model.embed_tokens.*`, `model.layers.{i}.*`, `model.norm.weight`).
    pub(crate) fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // The rope table is identical for every layer, so it is resolved once
        // here the way `Llama3Model::from_weights` does.
        let rope = args.rope_scaling_kind();
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for index in 0..args.num_hidden_layers {
            layers.push(TransformerBlock::from_weights_with_rope(
                weights, args, index, &rope,
            )?);
        }

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Weight not found: model.norm.weight".to_string())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, args.rms_norm_eps),
        })
    }

    /// Token embeddings, for a caller that merges vision features into them
    /// before running the stack.
    pub(crate) fn embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// One empty `KVCache` per layer.
    pub(crate) fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// The generator's forward pass up to and including the final norm.
    pub(crate) fn forward_hidden(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed_tokens.forward(input_ids),
        };
        for (index, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[index], mask);
        }
        self.norm.forward(&h)
    }
}
