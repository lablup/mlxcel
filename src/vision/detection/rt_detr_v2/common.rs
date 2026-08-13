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

//! Small shared helpers for the RT-DETRv2 modules.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Copy a weight by exact key, erroring with the missing key name.
///
/// RT-DETRv2 weights are dense bf16 (no quantization), so a plain `copy` of the
/// MLX-managed lazy array is the right load primitive — there is no
/// quantized-vs-regular branch to detect.
pub fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("RT-DETRv2 weight not found: {key}"))
}

/// Like [`copy_weight`] but optional; returns `None` if the key is absent.
pub fn copy_weight_opt(weights: &WeightMap, key: &str) -> Option<UniquePtr<MlxArray>> {
    weights.get(key).map(|w| mlxcel_core::copy(w))
}

/// Cast an array to float32.
///
/// Used to normalize the input image to f32 before the vision tower. Note that
/// this does not pin the *graph* to f32: MLX settles each op on its operand
/// dtypes, so the first conv against a bf16 weight brings the graph back to
/// bf16 for a bf16 checkpoint. Anything reading model outputs back to the host
/// must convert by dtype rather than assuming a width.
pub fn to_f32(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32)
}
