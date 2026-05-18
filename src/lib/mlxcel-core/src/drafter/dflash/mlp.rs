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

//! SwiGLU MLP block reused by every DFlash drafter layer.

use crate::ffi::MlxArray;
use crate::layers::UnifiedLinear;
use crate::weights::WeightMap;
use cxx::UniquePtr;

/// SwiGLU MLP block reused by every drafter layer.
///
/// Mirrors upstream `Qwen3MLP(config.hidden_size, config.intermediate_size)`
/// (`gate_proj`, `up_proj`, `down_proj`, all bias-free). The non-quantized
/// path uses the fused compiled FP MLP (`compiled_swiglu_mlp_fp16`), so
/// performance matches the in-tree Qwen 3 MLP at `src/models/qwen3.rs`.
///
/// The Qwen 3 MLP lives in the `mlxcel` binary crate (not in `mlxcel-core`)
/// and is therefore not reachable from this module. The cheapest fix is a
/// thin in-crate equivalent — three [`UnifiedLinear`]s plus the same fused
/// compile graph. This keeps the drafter self-contained and removes a
/// would-be cross-crate dependency. If a future refactor moves `Qwen3MLP`
/// down into `mlxcel-core`, this struct can be replaced wholesale.
pub struct DFlashMlp {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl DFlashMlp {
    /// Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Non-quantized path: fused compiled FP MLP (single compile graph).
        if let Some(result) = crate::layers::compiled_swiglu_mlp_fp16(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ) {
            return result;
        }

        // Quantized path: separate projections + compiled SwiGLU activation.
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = crate::ffi::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    /// Load `{prefix}.gate_proj`, `{prefix}.up_proj`, `{prefix}.down_proj`
    /// from the weight map. Auto-detects quantization (via `.scales` key
    /// per projection).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let gate_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.gate_proj"), group_size, bits)?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.up_proj"), group_size, bits)?;
        let down_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.down_proj"), group_size, bits)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}
