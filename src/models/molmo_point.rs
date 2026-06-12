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

//! Molmo-Point text model implementation using mlxcel-core
//!
//! The language model is identical to Molmo2: fused QKV, per-head QK norm,
//! pre-norm, SwiGLU MLP, dual embedding, RoPE. Molmo-Point also has an
//! ExtendedLmHead that outputs to vocab_size + additional_vocab_size.
//!
//! This module re-exports the Molmo2 text model components and adds the
//! extended LM head needed for point prediction.
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/language.py

use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::molmo2::{Molmo2Embedding, Molmo2TextConfig, Molmo2TransformerBlock};

/// Extended LM head that outputs to vocab_size + additional_vocab_size.
///
/// Pre-fuses the base output_embeddings and new_output_embeddings during
/// construction, then computes logits via matrix multiplication.
pub struct ExtendedLmHead {
    fused: UniquePtr<MlxArray>, // [vocab_size + additional_vocab_size, hidden_size]
}

impl ExtendedLmHead {
    pub fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        // hidden_states @ fused.T
        let fused_t = mlxcel_core::transpose_axes(&self.fused, &[1, 0]);
        mlxcel_core::matmul(hidden_states, &fused_t)
    }

    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let output_embeddings = weights
            .get(&format!("{prefix}.output_embeddings"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.output_embeddings"))?;
        let new_output_embeddings = weights
            .get(&format!("{prefix}.new_output_embeddings"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.new_output_embeddings"))?;
        let fused = mlxcel_core::concatenate(&output_embeddings, &new_output_embeddings, 0);
        Ok(Self { fused })
    }
}

/// Molmo-Point Transformer model (same blocks as Molmo2, with extended LM head).
pub struct MolmoPointTransformer {
    pub wte: Molmo2Embedding,
    pub blocks: Vec<Molmo2TransformerBlock>,
    pub ln_f: mlxcel_core::layers::RMSNorm,
}

impl MolmoPointTransformer {
    /// Forward pass returning (post_ln_output, pre_ln_output).
    /// pre_ln output is needed by the point predictor.
    pub fn forward_with_pre_ln(
        &self,
        input_ids: Option<&MlxArray>,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            let ids = input_ids.unwrap();
            // Replace -1 tokens with 0 for safe embedding lookup
            let ids_i32 = mlxcel_core::astype(ids, mlxcel_core::dtype::INT32);
            let neg_one = mlxcel_core::from_slice_i32(&[-1], &[1]);
            let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
            let is_neg = mlxcel_core::equal(&ids_i32, &neg_one);
            let safe_ids = mlxcel_core::where_cond(&is_neg, &zero, &ids_i32);
            self.wte.forward(&safe_ids)
        };

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &mut caches[i], mask);
        }

        let pre_ln = mlxcel_core::copy(&h);
        let post_ln = self.ln_f.forward(&h);
        (post_ln, pre_ln)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &Molmo2TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let wte = Molmo2Embedding::from_weights(weights, &format!("{prefix}.wte"))?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = Molmo2TransformerBlock::from_weights(weights, config, i, prefix)?;
            blocks.push(block);
        }

        let ln_f_weight = get_weight_copy(weights, &format!("{prefix}.ln_f.weight"))?;
        let ln_f = mlxcel_core::layers::RMSNorm::new(ln_f_weight, config.layer_norm_eps);

        Ok(Self { wte, blocks, ln_f })
    }
}

/// Molmo-Point language model wrapper (transformer + extended LM head).
pub struct MolmoPointLanguageModel {
    pub model: MolmoPointTransformer,
    pub lm_head: ExtendedLmHead,
}

impl MolmoPointLanguageModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (h, _) = self
            .model
            .forward_with_pre_ln(Some(input_ids), None, caches, mask);
        self.lm_head.forward(&h)
    }

    pub fn forward_with_embeddings(
        &self,
        _input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (h, _) = self
            .model
            .forward_with_pre_ln(None, input_embeddings, caches, mask);
        self.lm_head.forward(&h)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.model.blocks.len())
            .map(|_| KVCache::new())
            .collect()
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &Molmo2TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let model =
            MolmoPointTransformer::from_weights(weights, config, &format!("{prefix}.model"))?;

        let lm_head_prefix = if prefix.ends_with(".model") {
            let base = prefix.strip_suffix(".model").unwrap();
            format!("{base}.lm_head")
        } else {
            format!("{prefix}.lm_head")
        };

        let lm_head = ExtendedLmHead::from_weights(weights, &lm_head_prefix)?;

        Ok(Self { model, lm_head })
    }
}

// Helper.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}
