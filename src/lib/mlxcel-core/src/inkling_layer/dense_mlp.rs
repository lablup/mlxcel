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

use crate::layers::UnifiedLinear;
use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr};

use super::{InklingFeedForward, InklingLayerSpec, weight};

/// Dense SwiGLU plane used by every Inkling MTP block.
pub struct InklingDenseMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    global_scale: UniquePtr<MlxArray>,
}

impl InklingDenseMlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        spec: &InklingLayerSpec,
    ) -> Result<Self, String> {
        if let Some(gate) = weights.get(&format!("{prefix}.gate_proj.weight")) {
            let shape = crate::array_shape(gate);
            if shape.len() == 2 && shape[0] != spec.dense_intermediate_size as i32 {
                return Err(format!(
                    "{prefix}.gate_proj.weight: config dense width {} disagrees with {shape:?}",
                    spec.dense_intermediate_size
                ));
            }
        }
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                spec.quantization_group_size,
                spec.quantization_bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                spec.quantization_group_size,
                spec.quantization_bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                spec.quantization_group_size,
                spec.quantization_bits,
            )?,
            global_scale: weight(weights, &format!("{prefix}.global_scale"))?,
        })
    }
}

impl InklingFeedForward for InklingDenseMlp {
    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(input);
        let up = self.up_proj.forward(input);
        let activated = crate::compiled_swiglu_activation(&gate, &up);
        let down = self.down_proj.forward(&activated);
        crate::multiply(&down, &self.global_scale)
    }
}
