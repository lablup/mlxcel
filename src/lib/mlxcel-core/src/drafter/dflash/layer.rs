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

//! `DFlashDecoderLayer` — one DFlash drafter transformer block (pre-norm
//! attention + post-norm MLP, both wrapped in residual connections).

use crate::cache::KVCache;
use crate::ffi::{self, MlxArray};
use crate::layers::RMSNorm;
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::attention::DFlashAttention;
use super::config::DFlashConfig;
use super::mlp::DFlashMlp;

/// DFlash decoder layer = pre-norm attention + post-norm MLP, with
/// residual connections wrapping each sub-block. Mirrors upstream
/// `DFlashDecoderLayer.__call__`:
///
/// ```python
/// h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
/// return h + self.mlp(self.post_attention_layernorm(h))
/// ```
pub struct DFlashDecoderLayer {
    pub self_attn: DFlashAttention,
    pub mlp: DFlashMlp,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DFlashDecoderLayer {
    /// Forward for one decoder layer.
    pub fn forward(
        &self,
        x: &MlxArray,
        x_ctx: &MlxArray,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        // h = x + self_attn(input_layernorm(x), x_ctx, cache)
        let x_normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&x_normed, x_ctx, cache);
        let h = ffi::add(x, &attn_out);

        // h = h + mlp(post_attention_layernorm(h))
        let h_normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&h_normed);
        ffi::add(&h, &mlp_out)
    }

    /// Load one decoder layer's weights from the map.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DFlashConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let self_attn = DFlashAttention::from_weights(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            group_size,
            bits,
        )?;
        let mlp = DFlashMlp::from_weights(weights, &format!("{prefix}.mlp"), group_size, bits)?;

        let input_layernorm_w = weights
            .get(&format!("{prefix}.input_layernorm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.input_layernorm.weight"))?;
        let post_attention_layernorm_w = weights
            .get(&format!("{prefix}.post_attention_layernorm.weight"))
            .map(|w| ffi::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.post_attention_layernorm.weight"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_layernorm_w, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_attention_layernorm_w, config.rms_norm_eps),
        })
    }
}
