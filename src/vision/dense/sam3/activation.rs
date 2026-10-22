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

use mlxcel_core::{MlxArray, UniquePtr};

/// Exact operation order used by `mlx.nn.gelu` in the pinned SAM 3 runtime.
pub(crate) fn gelu(x: &MlxArray) -> UniquePtr<MlxArray> {
    let normalized = mlxcel_core::divide_scalar(x, std::f32::consts::SQRT_2);
    let cdf = mlxcel_core::add(
        &mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(x)),
        &mlxcel_core::erf(&normalized),
    );
    mlxcel_core::divide_scalar(&mlxcel_core::multiply(x, &cdf), 2.0)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires MLX BF16 execution"]
    fn matches_authentic_sam3_layer_zero_bf16_values() {
        const INPUT: [f32; 8] = [
            2.765625, 2.3125, 1.3359375, 1.2578125, 1.578125, 1.6484375, 1.515625, 1.8203125,
        ];
        const EXPECTED: [f32; 8] = [
            2.765625, 2.296875, 1.21875, 1.1328125, 1.4921875, 1.5625, 1.421875, 1.765625,
        ];

        let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
        let input = mlxcel_core::from_slice_f32(&INPUT, &[INPUT.len() as i32]);
        let input = mlxcel_core::astype(&input, mlxcel_core::dtype::BFLOAT16);
        let actual = super::gelu(&input);
        assert_eq!(mlxcel_core::utils::array_to_vec_f32(&actual), EXPECTED);
    }
}
