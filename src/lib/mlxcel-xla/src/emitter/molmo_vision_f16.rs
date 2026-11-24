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

//! Molmo v1 operators at eager MLX's F16 rounding boundaries (#3390).
//!
//! The Molmo-7B checkpoint stores its vision tower in F16, so the eager tower
//! runs on F16 tensors: each MLX operator computes in f32 internally and
//! stores an F16 result. The graph keeps the f32 carrier of its public ABI but
//! rounds to F16 wherever eager materializes an F16 tensor.
//!
//! Without these roundings the f32 graph drifted away from eager in the CLIP
//! tower's outlier channel 650 (|x| up to about 200): on the pinned fixture the
//! selected layers 14 and 21 missed eager by up to 2.8 on elements of 13 to 47
//! (8 and 4 elements over the gate's `VISION` bound), far more than one F16
//! step at that size, so the difference is accumulated over blocks rather than
//! one output rounding. This was never a CUDA regression: the #870 merge
//! commit already failed layer 21 with eager on CUDA (it was qualified with
//! eager on ROCm). Every boundary below was measured to matter: dropping any
//! one of them fails the gate again.
//!
//! Each helper names the MLX kernel whose rounding it reproduces.

use super::builder::{Builder, Val};
use super::molmo_vision_ops::scalar_broadcast;

/// Round an f32 carrier to the nearest F16 value (ties to even), keeping f32
/// storage. Values beyond the F16 range become infinite, as they do in eager.
pub(super) fn round_f16(builder: &mut Builder, value: &Val) -> Val {
    assert_eq!(value.ty.elt, "f32", "F16 rounding expects an f32 carrier");
    let half = builder.convert(value, "f16");
    builder.convert(&half, "f32")
}

/// MLX `quantized_matmul` then `add(product, bias)` on F16 tensors
/// (`quantized_linear_forward` in `mlx_cxx_bridge.cpp`): the product is stored
/// as F16, and the separate bias add stores a second F16 result.
pub(super) fn linear_f16(
    builder: &mut Builder,
    value: &Val,
    weight: &Val,
    bias: Option<&Val>,
) -> Val {
    let product = builder.linear_seq(value, weight);
    let product = round_f16(builder, &product);
    let Some(bias) = bias else {
        return product;
    };
    let bias = builder.broadcast(bias, &[1], product.ty.shape.clone());
    let output = builder.add(&product, &bias);
    round_f16(builder, &output)
}

/// MLX `fast::layer_norm` on an F16 row (`mlx/backend/cuda/layer_norm.cu`):
/// mean and centered variance reduce in f32, the normalized value is cast to
/// F16, and `w * norm + b` runs as one F16 fused multiply-add, so it rounds
/// once. Rounding the product and the sum separately measured further from
/// eager on the pinned fixture (selected layer 21 max_abs 2.81 against 2.50).
pub(super) fn layer_norm_f16(
    builder: &mut Builder,
    value: &Val,
    weight: &Val,
    bias: &Val,
    epsilon: f32,
) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let zero = builder.const_f32(0.0);
    let width_scalar = builder.const_f32(width as f32);
    let width_rows = builder.broadcast(&width_scalar, &[], vec![rows]);
    let sum = builder.reduce_add(value, 1, &zero);
    let mean = builder.divide(&sum, &width_rows);
    let mean = builder.broadcast(&mean, &[0], vec![rows, width]);
    let centered = builder.subtract(value, &mean);
    let squared = builder.multiply(&centered, &centered);
    let squared_sum = builder.reduce_add(&squared, 1, &zero);
    let variance = builder.divide(&squared_sum, &width_rows);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![rows]);
    let variance = builder.add(&variance, &epsilon);
    let inv_std = builder.rsqrt(&variance);
    let inv_std = builder.broadcast(&inv_std, &[0], vec![rows, width]);
    let normalized = builder.multiply(&centered, &inv_std);
    let normalized = round_f16(builder, &normalized);
    let weight = builder.broadcast(weight, &[1], vec![rows, width]);
    let bias = builder.broadcast(bias, &[1], vec![rows, width]);
    let scaled = builder.multiply(&normalized, &weight);
    let output = builder.add(&scaled, &bias);
    round_f16(builder, &output)
}

/// `quick_gelu` as the eager encoder evaluates it on F16
/// (`mlxcel_core::utils::gelu_sigmoid`): the coefficient 1.702 is an F16
/// constant (1.7021484375), and `coef * x`, `sigmoid` and `x * sigmoid` each
/// store an F16 result.
pub(super) fn quick_gelu_f16(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let coefficient = scalar_broadcast(builder, 1.702, shape.clone());
    let coefficient = round_f16(builder, &coefficient);
    let scaled = builder.multiply(&coefficient, value);
    let scaled = round_f16(builder, &scaled);
    let negative = builder.negate(&scaled);
    let exponential = builder.exponential(&negative);
    let one = scalar_broadcast(builder, 1.0, shape);
    let denominator = builder.add(&one, &exponential);
    let sigmoid = builder.divide(&one, &denominator);
    let sigmoid = round_f16(builder, &sigmoid);
    let output = builder.multiply(value, &sigmoid);
    round_f16(builder, &output)
}

#[cfg(test)]
#[path = "molmo_vision_f16_tests.rs"]
pub(super) mod tests;
