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

//! [`LagunaDFlashDecoderLayer`]: pre-norm attention plus post-norm SwiGLU
//! MLP. Unlike the Qwen 3.5 DFlash layer, the context rows pass through this
//! layer's `input_layernorm` before their K/V projection (vLLM
//! `DFlashLagunaModel::_project_context_kv`).

use crate::drafter::dflash::DFlashMlp;
use crate::ffi::{self, MlxArray};
use crate::layers::RMSNorm;
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::attention::LagunaDFlashAttention;
use super::cache::LagunaDFlashContextCache;
use super::config::LagunaDFlashConfig;

/// One drafter transformer block.
pub struct LagunaDFlashDecoderLayer {
    pub self_attn: LagunaDFlashAttention,
    pub mlp: DFlashMlp,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl LagunaDFlashDecoderLayer {
    /// `x`: proposal residual stream `[B, L, H]`; `x_ctx`: projected context
    /// `[B, T, H]` (already through `hidden_norm`).
    pub fn forward(
        &self,
        x: &MlxArray,
        x_ctx: &MlxArray,
        cache: &mut LagunaDFlashContextCache,
    ) -> UniquePtr<MlxArray> {
        let x_normed = self.input_layernorm.forward(x);
        let ctx_normed = self.input_layernorm.forward(x_ctx);
        let attn_out = self.self_attn.forward(&x_normed, &ctx_normed, cache);
        let h = ffi::add(x, &attn_out);
        let h_normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&h_normed);
        ffi::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &LagunaDFlashConfig,
        window: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let self_attn = LagunaDFlashAttention::from_weights(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            window,
            group_size,
            bits,
        )?;
        let mlp = DFlashMlp::from_weights(weights, &format!("{prefix}.mlp"), group_size, bits)?;
        let norm = |leaf: &str| -> Result<RMSNorm, String> {
            let key = format!("{prefix}.{leaf}.weight");
            let w = weights
                .get(&key)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            Ok(RMSNorm::new(w, config.rms_norm_eps))
        };
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
        })
    }
}
