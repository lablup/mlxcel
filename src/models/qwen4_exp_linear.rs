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

//! Decode-equivalent reductions for Qwen4-Exp's quantized gate projections.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::{MlxArray, UniquePtr};

/// Match mlx-vlm's `_row_invariant_quantized_projection`: each prefill token
/// uses the singleton quantized matrix-vector kernel rather than a batched QMM.
pub(super) fn row_invariant_projection(
    linear: &UnifiedLinear,
    x: &MlxArray,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let UnifiedLinear::Quantized { weight, bias } = linear else {
        return linear.forward(x);
    };
    if shape.len() != 3
        || shape[1] <= 1
        || weight.bits == 1
        || weight.global_scale.is_some()
        || !weight.runtime_loras.is_empty()
    {
        return linear.forward(x);
    }
    let x = mlxcel_core::contiguous(x, false);
    let x = mlxcel_core::reshape(&x, &[shape[0], shape[1], 1, 1, shape[2]]);
    let w = mlxcel_core::expand_dims(&weight.weight, 0);
    let scales = mlxcel_core::expand_dims(&weight.scales, 0);
    let biases = weight
        .biases
        .as_ref()
        .map(|b| mlxcel_core::expand_dims(b, 0));
    let indices = mlxcel_core::zeros(&[shape[0], shape[1], 1], mlxcel_core::dtype::INT32);
    // SAFETY: nullable pointers reference arrays alive for the entire FFI call.
    // The loader validates packed dimensions and quantization parameters.
    let out = unsafe {
        mlxcel_core::gather_qmm(
            &x,
            &w,
            &scales,
            biases
                .as_ref()
                .map_or(std::ptr::null(), |b| b.as_ref().unwrap() as *const MlxArray),
            std::ptr::null(),
            indices.as_ref().unwrap() as *const MlxArray,
            true,
            weight.group_size,
            weight.bits,
            false,
            &weight.mode,
            std::ptr::null(),
        )
    };
    let width = mlxcel_core::array_shape(&weight.weight)[0];
    let out = mlxcel_core::reshape(&out, &[shape[0], shape[1], width]);
    match bias {
        Some(bias) => mlxcel_core::add(&out, bias),
        None => out,
    }
}

/// Fuse the four GatedDeltaNet decode projections exactly as mlx-vlm does.
/// Prefill still uses the original matrices and its row-invariant gate paths.
pub(super) fn fused_decode_projection(linears: &[&UnifiedLinear]) -> Option<UnifiedLinear> {
    use mlxcel_core::layers::QuantizedWeight;
    let mut parts = Vec::with_capacity(linears.len());
    for linear in linears {
        let UnifiedLinear::Quantized { weight, bias: None } = linear else {
            return None;
        };
        if weight.mode != "affine"
            || weight.global_scale.is_some()
            || !weight.runtime_loras.is_empty()
            || weight.biases.is_none()
            || weight.bits == 1
        {
            return None;
        }
        parts.push(weight);
    }
    let first = *parts.first()?;
    if parts.iter().any(|weight| {
        weight.bits != first.bits
            || weight.group_size != first.group_size
            || mlxcel_core::array_dtype(&weight.scales) != mlxcel_core::array_dtype(&first.scales)
            || mlxcel_core::array_dtype(weight.biases.as_ref().unwrap())
                != mlxcel_core::array_dtype(&first.scales)
    }) {
        return None;
    }
    let join = |arrays: Vec<&MlxArray>| {
        let mut result = mlxcel_core::copy(arrays[0]);
        for array in &arrays[1..] {
            result = mlxcel_core::concatenate(&result, array, 0);
        }
        mlxcel_core::eval(&result);
        result
    };
    let weight = join(parts.iter().map(|w| w.weight.as_ref().unwrap()).collect());
    let scales = join(parts.iter().map(|w| w.scales.as_ref().unwrap()).collect());
    let biases = join(
        parts
            .iter()
            .map(|w| w.biases.as_ref().unwrap().as_ref().unwrap())
            .collect(),
    );
    Some(UnifiedLinear::new(
        QuantizedWeight::new(weight, scales, biases, first.group_size, first.bits),
        None,
    ))
}
