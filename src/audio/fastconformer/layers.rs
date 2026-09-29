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

//! Weight-loading helpers and small layers shared by the FastConformer port.
//!
//! The released VoiceChat safetensors keep PyTorch layouts (Conv2d
//! `[out, in, kh, kw]`, Conv1d `[out, in, k]`) and mlx-vlm converts them at
//! load time. The gates here transpose to MLX layout (`[out, kh, kw, in]`,
//! `[out, k, in]`) only when the trailing axis is not already the input-channel
//! axis, so an export that was sanitized beforehand loads unchanged.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("FastConformer weight not found: {key}"))
}

pub(crate) fn maybe_copy_weight(weights: &WeightMap, key: &str) -> Option<UniquePtr<MlxArray>> {
    weights.get(key).map(|w| mlxcel_core::copy(w))
}

/// Load a Conv2d kernel in MLX layout `[out, kh, kw, in_per_group]`.
pub(crate) fn conv2d_weight(
    weights: &WeightMap,
    key: &str,
    in_per_group: i32,
) -> Result<UniquePtr<MlxArray>, String> {
    let w = copy_weight(weights, key)?;
    let shape = mlxcel_core::array_shape(&w);
    if shape.len() != 4 {
        return Err(format!("{key}: expected a 4D conv kernel, got {shape:?}"));
    }
    if shape[3] == in_per_group {
        return Ok(w);
    }
    if shape[1] != in_per_group {
        return Err(format!(
            "{key}: kernel {shape:?} has no input-channel axis of size {in_per_group}"
        ));
    }
    Ok(mlxcel_core::transpose_axes(&w, &[0, 2, 3, 1]))
}

/// Load a Conv1d kernel in MLX layout `[out, k, in_per_group]`.
pub(crate) fn conv1d_weight(
    weights: &WeightMap,
    key: &str,
    in_per_group: i32,
) -> Result<UniquePtr<MlxArray>, String> {
    let w = copy_weight(weights, key)?;
    let shape = mlxcel_core::array_shape(&w);
    if shape.len() != 3 {
        return Err(format!("{key}: expected a 3D conv kernel, got {shape:?}"));
    }
    if shape[2] == in_per_group {
        return Ok(w);
    }
    if shape[1] != in_per_group {
        return Err(format!(
            "{key}: kernel {shape:?} has no input-channel axis of size {in_per_group}"
        ));
    }
    Ok(mlxcel_core::transpose_axes(&w, &[0, 2, 1]))
}

/// `nn.LayerNorm(d)` (eps 1e-5, affine) through `mx.fast.layer_norm`.
pub(crate) struct LayerNorm {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl LayerNorm {
    pub(crate) const EPS: f32 = 1e-5;

    pub(crate) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            bias: maybe_copy_weight(weights, &format!("{prefix}.bias")),
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let weight_ptr: *const MlxArray = &*self.weight;
        let bias_ptr: *const MlxArray = match &self.bias {
            Some(b) => &**b,
            None => std::ptr::null(),
        };
        // SAFETY: `weight_ptr` points at an array owned by `self` for the
        // duration of the call; `bias_ptr` is either owned by `self` or null,
        // which the bridge treats as "no bias".
        unsafe { mlxcel_core::fast_layer_norm(x, weight_ptr, bias_ptr, Self::EPS) }
    }
}

/// `nn.glu(x, axis=-1)`: first half times sigmoid of the second half.
pub(crate) fn glu_last_axis(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last = shape.len() - 1;
    let half = shape[last] / 2;
    let starts = vec![0; shape.len()];
    let mut mid = shape.clone();
    mid[last] = half;
    let mut starts_b = starts.clone();
    starts_b[last] = half;
    let a = mlxcel_core::slice(x, &starts, &mid);
    let b = mlxcel_core::slice(x, &starts_b, &shape);
    mlxcel_core::multiply(&a, &mlxcel_core::sigmoid(&b))
}
