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

//! Youtu-VL VL patch merger.
//!
//! Mirrors upstream's `VLPatchMerger`: an RMSNorm followed by a two-layer
//! GELU MLP that takes encoder output `[total_tokens, hidden_size]` and
//! emits `[total_tokens / merge_unit, out_hidden_size]` ready to slot into
//! the language-model hidden space.
//!
//! Weight layout expected (matches upstream):
//!   `merger.ln_q.weight`              `[context_dim]`
//!   `merger.mlp.0.{weight,bias}`      `Linear(context_dim * M^2, context_dim * M^2)`
//!   `merger.mlp.2.{weight,bias}`      `Linear(context_dim * M^2, out_hidden_size)`
//!   (`mlp.1` is the GELU activation, no parameters.)

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub(super) struct VisionRMSNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl VisionRMSNorm {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        eps: f32,
    ) -> Result<Self, String> {
        let weight_key = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
        Ok(Self { weight, eps })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::rms_norm(x, &self.weight, self.eps)
    }
}

pub(super) struct PatchMerger {
    ln_q: VisionRMSNorm,
    mlp_0: UnifiedLinear,
    mlp_2: UnifiedLinear,
    merged_dim: i32,
}

impl PatchMerger {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        context_dim: usize,
        spatial_merge_size: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let merged_dim = (context_dim * spatial_merge_size * spatial_merge_size) as i32;
        let ln_q = VisionRMSNorm::from_weights(weights, &format!("{}.ln_q", prefix), 1e-6)?;
        let mlp_0 = UnifiedLinear::from_weights(weights, &format!("{}.mlp.0", prefix), gs, bits)?;
        let mlp_2 = UnifiedLinear::from_weights(weights, &format!("{}.mlp.2", prefix), gs, bits)?;
        Ok(Self {
            ln_q,
            mlp_0,
            mlp_2,
            merged_dim,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.ln_q.forward(x);
        let h = mlxcel_core::reshape(&h, &[-1, self.merged_dim]);
        let h = self.mlp_0.forward(&h);
        let h = mlxcel_core::utils::gelu_approx(&h);
        self.mlp_2.forward(&h)
    }
}
