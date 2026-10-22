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

//! SAM-local MLX `nn.Linear` arithmetic.

use mlxcel_core::layers::UnifiedLinear as CoreUnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Preserve upstream MLX's fused bias epilogue for dense SAM projections.
///
/// `mlx.nn.Linear` uses `addmm(bias, input, weight.T)`. The shared dense
/// linear layer adds its bias after the matrix multiplication, which rounds
/// twice for bf16/f16 outputs. Quantized and runtime-LoRA projections keep the
/// shared path because their composition rules differ from a plain `addmm`.
pub(crate) struct UnifiedLinear(CoreUnifiedLinear);

impl UnifiedLinear {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        CoreUnifiedLinear::from_weights(weights, prefix, group_size, bits).map(Self)
    }

    pub(crate) fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.0 {
            CoreUnifiedLinear::Regular(linear)
                if linear.bias.is_some() && !linear.has_active_runtime_lora() =>
            {
                let weight = mlxcel_core::transpose(&linear.weight);
                mlxcel_core::addmm(
                    linear.bias.as_deref().expect("bias presence checked"),
                    input,
                    &weight,
                    1.0,
                    1.0,
                )
            }
            _ => self.0.forward(input),
        }
    }

    #[cfg(test)]
    pub(crate) fn dense_weight_dtype(&self) -> Option<i32> {
        match &self.0 {
            CoreUnifiedLinear::Regular(linear) => Some(mlxcel_core::array_dtype(&linear.weight)),
            CoreUnifiedLinear::Quantized { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use mlxcel_core::layers::Linear;

    use super::*;

    #[test]
    fn biased_bf16_projection_matches_mlx_addmm_rounding() {
        let x = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(
                &[-0.447_265_63, 0.84375, 1.140625, -0.089_355_47],
                &[1, 4],
            ),
            mlxcel_core::dtype::BFLOAT16,
        );
        let weight = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(
                &[
                    0.80078125,
                    -0.1640625,
                    0.267_578_13,
                    -1.3125,
                    0.76171875,
                    -0.73046875,
                    0.97265625,
                    -0.57421875,
                    1.015625,
                    -1.8203125,
                    2.765625,
                    0.6484375,
                ],
                &[3, 4],
            ),
            mlxcel_core::dtype::BFLOAT16,
        );
        let bias = mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&[1.3671875, -0.70703125, 0.82421875], &[3]),
            mlxcel_core::dtype::BFLOAT16,
        );
        let layer = UnifiedLinear(CoreUnifiedLinear::Regular(Linear::new(weight, Some(bias))));
        let actual = layer.forward(&x);
        assert_eq!(
            mlxcel_core::utils::array_to_vec_f32(&actual),
            vec![1.296875, -0.50390625, 1.9296875]
        );
    }
}
