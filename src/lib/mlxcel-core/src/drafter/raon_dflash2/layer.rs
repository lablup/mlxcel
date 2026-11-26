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

//! One DFlash2 decoder layer.
//!
//! ```text
//! residual = x                        (layer 0)     | x + residual (later layers)
//! x        = input_layernorm(residual)
//! d        = attention_conv.deltas(x)
//! x        = conv(attn(conv(x, d[0])), d[1])
//! residual = x + residual
//! x        = post_attention_layernorm(residual)
//! d        = mlp_conv.deltas(x)
//! x        = conv(down(silu(gate(x')) * up(x')), d[1]),  x' = conv(x, d[0])
//! ```
//!
//! The running residual is returned beside the branch output; the model
//! adds them once more before the final norm.

use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::attention::DFlash2Attention;
use super::cache::ContextWindow;
use super::config::RaonDFlash2Config;
use super::conv::GroupedConv;

pub struct DFlash2Layer {
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pub attn: DFlash2Attention,
    attention_conv: GroupedConv,
    mlp_conv: GroupedConv,
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl DFlash2Layer {
    /// # Errors
    /// Returns a message naming a missing tensor.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &RaonDFlash2Config,
    ) -> Result<Self, String> {
        let norm = |leaf: &str| -> Result<RMSNorm, String> {
            let key = format!("{prefix}.{leaf}.weight");
            let w = weights
                .get(&key)
                .ok_or_else(|| format!("Raon DFlash2: missing weight {key}"))?;
            Ok(RMSNorm::new(ffi::copy(w), config.rms_norm_eps))
        };
        let linear =
            |leaf: &str| UnifiedLinear::from_weights(weights, &format!("{prefix}.{leaf}"), 64, 4);
        let conv = |leaf: &str| {
            GroupedConv::from_weights(
                weights,
                &format!("{prefix}.{leaf}"),
                config.conv_taps,
                config.conv_group_size,
                config.hidden_size,
            )
        };
        Ok(Self {
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            attn: DFlash2Attention::from_weights(weights, &format!("{prefix}.self_attn"), config)?,
            attention_conv: conv("attention_conv")?,
            mlp_conv: conv("mlp_conv")?,
            gate_proj: linear("mlp.gate_proj")?,
            up_proj: linear("mlp.up_proj")?,
            down_proj: linear("mlp.down_proj")?,
        })
    }

    /// One layer over block rows `x` (`[n, hidden]`, positions `pos..`).
    /// `residual` is `None` for the first layer. Returns `(branch, residual)`.
    pub fn forward(
        &self,
        x: &MlxArray,
        residual: Option<&MlxArray>,
        pos: i32,
        block_size: i32,
        window: &ContextWindow,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let residual = match residual {
            Some(r) => ffi::add(x, r),
            None => ffi::copy(x),
        };
        let h = self.input_layernorm.forward(&residual);
        let d = self.attention_conv.deltas(&h);
        let h = self
            .attention_conv
            .apply(&h, &self.attention_conv.stage(&d, 0), 0, block_size);
        let h = self.attn.forward(&h, pos, window);
        let h = self
            .attention_conv
            .apply(&h, &self.attention_conv.stage(&d, 1), 1, block_size);
        let residual = ffi::add(&h, &residual);

        let h = self.post_attention_layernorm.forward(&residual);
        let d = self.mlp_conv.deltas(&h);
        let h = self
            .mlp_conv
            .apply(&h, &self.mlp_conv.stage(&d, 0), 0, block_size);
        let act = ffi::multiply(
            &crate::utils::silu(&self.gate_proj.forward(&h)),
            &self.up_proj.forward(&h),
        );
        let h = self.down_proj.forward(&act);
        let h = self
            .mlp_conv
            .apply(&h, &self.mlp_conv.stage(&d, 1), 1, block_size);
        (h, residual)
    }
}
