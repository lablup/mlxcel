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

//! Qwen3-VL vision operators at eager MLX's BF16 rounding boundaries (#3291).
//!
//! A BF16 checkpoint runs the eager vision tower on BF16 tensors: every MLX
//! operator accumulates in f32 internally and stores a BF16 result. The graph
//! keeps the f32 carrier of its public ABI but rounds each value to BF16 at
//! the points where eager materializes a BF16 tensor, so the two backends see
//! the same intermediate grid instead of an f32 tower that drifts away from
//! eager's accumulated BF16 error over 24 blocks.
//!
//! Each operator below names the MLX kernel whose rounding it reproduces.

use super::builder::{Builder, Val};

/// Round an f32 carrier to the nearest BF16 value (ties to even), keeping f32
/// storage.
///
/// Integer round-to-nearest-even on the f32 bit pattern instead of an
/// `f32 -> bf16 -> f32` convert pair, so the same graph lowers on the Metal
/// target, which has no BF16 type (#612). NaN passes through unchanged: the
/// integer carry would otherwise turn CUDA's canonical `0x7fffffff` NaN into
/// `-0.0`.
pub(super) fn round_bf16(builder: &mut Builder, value: &Val) -> Val {
    assert_eq!(value.ty.elt, "f32", "BF16 rounding expects an f32 carrier");
    let shape = value.ty.shape.clone();
    let bits = builder.bitcast_convert(value, "ui32");
    let sixteen = builder.const_splat_u32(16, shape.clone());
    let one = builder.const_splat_u32(1, shape.clone());
    let half_ulp = builder.const_splat_u32(0x7fff, shape.clone());
    let high_half = builder.const_splat_u32(0xffff_0000, shape);
    let kept_lsb = builder.shift_right_logical(&bits, &sixteen);
    let kept_lsb = builder.and(&kept_lsb, &one);
    let biased = builder.add(&bits, &half_ulp);
    let biased = builder.add(&biased, &kept_lsb);
    let truncated = builder.and(&biased, &high_half);
    let rounded = builder.bitcast_convert(&truncated, "f32");
    let is_nan = builder.compare("NE", value, value, "FLOAT");
    builder.select(&is_nan, value, &rounded)
}

fn bias_2d(builder: &mut Builder, value: &Val, bias: &Val) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let bias = builder.broadcast(bias, &[1], vec![rows, width]);
    builder.add(value, &bias)
}

/// MLX `Linear` on BF16 tensors: `matmul` stores a BF16 product (f32
/// accumulation inside the GEMM), then `add(product, bias)` stores a second
/// BF16 result. Both roundings are observable.
pub(super) fn linear_bf16(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val) -> Val {
    let product = builder.linear_seq(value, weight);
    let product = round_bf16(builder, &product);
    let output = bias_2d(builder, &product, bias);
    round_bf16(builder, &output)
}

/// MLX `fast::layer_norm` on a BF16 row (`mlx/backend/cuda/layer_norm.cu`):
/// mean and centered variance reduce in f32, the normalized value is cast to
/// BF16, and `w * norm + b` runs in BF16 arithmetic, rounding the product and
/// the sum separately.
pub(super) fn layer_norm_bf16(
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
    let normalized = round_bf16(builder, &normalized);
    let weight = builder.broadcast(weight, &[1], vec![rows, width]);
    let bias = builder.broadcast(bias, &[1], vec![rows, width]);
    let scaled = builder.multiply(&normalized, &weight);
    let scaled = round_bf16(builder, &scaled);
    let shifted = builder.add(&scaled, &bias);
    round_bf16(builder, &shifted)
}

/// The merger's exact GELU. The bridge's `gelu` promotes a BF16 input to f32
/// through its f32 constants, evaluates `x * (0.5 * (1 + erf(x / sqrt(2))))`,
/// and casts the result back once.
pub(super) fn exact_gelu_bf16(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let sqrt_two = builder.const_f32(std::f32::consts::SQRT_2);
    let sqrt_two = builder.broadcast(&sqrt_two, &[], shape);
    let scaled = builder.divide(value, &sqrt_two);
    let erf = builder.erf(&scaled);
    let cdf = builder.add(&one, &erf);
    let cdf = builder.multiply(&half, &cdf);
    let output = builder.multiply(value, &cdf);
    round_bf16(builder, &output)
}

/// The block MLP's `gelu_pytorch_tanh` (`GeluActivation::TanhF32`): evaluated
/// in f32 and cast back to BF16 once.
pub(super) fn tanh_gelu_bf16(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let coefficient = builder.const_f32(0.044_715);
    let coefficient = builder.broadcast(&coefficient, &[], shape.clone());
    let scale = builder.const_f32(0.797_884_6);
    let scale = builder.broadcast(&scale, &[], shape);
    let squared = builder.multiply(value, value);
    let cubed = builder.multiply(&squared, value);
    let nonlinear = builder.multiply(&coefficient, &cubed);
    let inner = builder.add(value, &nonlinear);
    let scaled = builder.multiply(&scale, &inner);
    let tanh = builder.tanh(&scaled);
    let cdf = builder.add(&one, &tanh);
    let half_value = builder.multiply(value, &half);
    let output = builder.multiply(&half_value, &cdf);
    round_bf16(builder, &output)
}

fn rotate_half(builder: &mut Builder, value: &Val) -> Val {
    let tokens = value.ty.shape[0];
    let heads = value.ty.shape[1];
    let width = value.ty.shape[2];
    let half = width / 2;
    let first = builder.slice(value, &[(0, tokens), (0, heads), (0, half)]);
    let second = builder.slice(value, &[(0, tokens), (0, heads), (half, width)]);
    let second = builder.negate(&second);
    builder.concatenate(&second, &first, 2)
}

/// `apply_rotary_pos_emb_vision`: the BF16 query or key is widened to f32,
/// rotated against f32 frequencies, and cast back to BF16.
fn apply_vision_rope_bf16(builder: &mut Builder, value: &Val, freqs: &Val) -> Val {
    let tokens = value.ty.shape[0];
    let heads = value.ty.shape[1];
    let half = freqs.ty.shape[1];
    let cos = builder.cosine(freqs);
    let sin = builder.sine(freqs);
    let cos = builder.concatenate(&cos, &cos, 1);
    let sin = builder.concatenate(&sin, &sin, 1);
    let cos = builder.broadcast(&cos, &[0, 2], vec![tokens, heads, half * 2]);
    let sin = builder.broadcast(&sin, &[0, 2], vec![tokens, heads, half * 2]);
    let rotated = rotate_half(builder, value);
    let direct = builder.multiply(value, &cos);
    let rotated = builder.multiply(&rotated, &sin);
    let output = builder.add(&direct, &rotated);
    round_bf16(builder, &output)
}

/// Projection weights of one attention block, in checkpoint order.
pub(super) struct AttentionWeights {
    pub(super) qkv_weight: Val,
    pub(super) qkv_bias: Val,
    pub(super) proj_weight: Val,
    pub(super) proj_bias: Val,
}

/// Packed vision self-attention as MLX's fused CUDA SDPA (cuDNN flash
/// attention) computes it on BF16 query, key, and value: scores and the
/// softmax statistics stay in f32, the unnormalized probabilities are cast to
/// BF16 for the value GEMM, the f32 context is divided by the f32 row sum, and
/// the output is stored as BF16. The flash kernel's per-tile running maximum
/// is not reproduced; the row maximum stands in for it.
pub(super) fn attention_bf16(
    builder: &mut Builder,
    hidden: &Val,
    weights: &AttentionWeights,
    heads: usize,
    freqs: &Val,
    attention_bias: &Val,
) -> Val {
    let tokens = hidden.ty.shape[0];
    let width = hidden.ty.shape[1];
    let head_dim = width / heads;
    let qkv = linear_bf16(builder, hidden, &weights.qkv_weight, &weights.qkv_bias);
    let qkv = builder.reshape(&qkv, vec![tokens, 3, heads, head_dim]);
    let slice = |builder: &mut Builder, index: usize| {
        let part = builder.slice(
            &qkv,
            &[(0, tokens), (index, index + 1), (0, heads), (0, head_dim)],
        );
        builder.reshape(&part, vec![tokens, heads, head_dim])
    };
    let q = slice(builder, 0);
    let k = slice(builder, 1);
    let v = slice(builder, 2);
    let q = apply_vision_rope_bf16(builder, &q, freqs);
    let k = apply_vision_rope_bf16(builder, &k, freqs);
    let q = builder.transpose(&q, &[1, 0, 2]);
    let k = builder.transpose(&k, &[1, 0, 2]);
    let v = builder.transpose(&v, &[1, 0, 2]);
    let score_shape = vec![heads, tokens, tokens];
    let scores = builder.dot_general(&q, &k, &[0], &[0], &[2], &[2], score_shape.clone());
    let scale = builder.const_f32((head_dim as f32).powf(-0.5));
    let scale = builder.broadcast(&scale, &[], score_shape.clone());
    let scores = builder.multiply(&scores, &scale);
    let attention_bias = builder.broadcast(attention_bias, &[1, 2], score_shape.clone());
    let scores = builder.add(&scores, &attention_bias);
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(&scores, 2, &negative_infinity);
    let maximum = builder.broadcast(&maximum, &[0, 1], score_shape.clone());
    let shifted = builder.subtract(&scores, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let denominator = builder.reduce_add(&exponentials, 2, &zero);
    let probabilities = round_bf16(builder, &exponentials);
    let context_shape = vec![heads, tokens, head_dim];
    let context = builder.dot_general(
        &probabilities,
        &v,
        &[0],
        &[0],
        &[2],
        &[1],
        context_shape.clone(),
    );
    let denominator = builder.broadcast(&denominator, &[0, 1], context_shape);
    let context = builder.divide(&context, &denominator);
    let context = round_bf16(builder, &context);
    let context = builder.transpose(&context, &[1, 0, 2]);
    let context = builder.reshape(&context, vec![tokens, width]);
    linear_bf16(builder, &context, &weights.proj_weight, &weights.proj_bias)
}

/// A residual `add` of two BF16 tensors stores a BF16 sum.
pub(super) fn add_bf16(builder: &mut Builder, left: &Val, right: &Val) -> Val {
    let sum = builder.add(left, right);
    round_bf16(builder, &sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::builder::{Precision, Ty};
    use crate::float::round_bf16_f32;

    /// The integer schedule `round_bf16` emits, evaluated on the host with
    /// StableHLO's wrapping `ui32` add.
    fn graph_schedule(value: f32) -> f32 {
        let bits = value.to_bits();
        let kept_lsb = (bits >> 16) & 1;
        let biased = bits.wrapping_add(0x7fff).wrapping_add(kept_lsb);
        let rounded = f32::from_bits(biased & 0xffff_0000);
        if value.is_nan() { value } else { rounded }
    }

    #[test]
    fn integer_schedule_matches_host_bf16_rounding() {
        let mut values = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            1.0e-40,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            // Exact ties: the kept LSB decides the direction.
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f81_8000),
            f32::from_bits(0xbf80_8000),
            2656.0,
            22.898,
        ];
        let mut state = 0x9e37_79b9_u32;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = f32::from_bits(state);
            if value.is_finite() {
                values.push(value);
            }
        }
        for value in values {
            assert_eq!(
                graph_schedule(value).to_bits(),
                round_bf16_f32(value).to_bits(),
                "{value:e}"
            );
        }
        // CUDA's canonical NaN would carry into the sign bit without the guard.
        let canonical_nan = f32::from_bits(0x7fff_ffff);
        assert!(graph_schedule(canonical_nan).is_nan());
        assert_eq!(
            f32::from_bits((0x7fff_ffff_u32.wrapping_add(0x8000)) & 0xffff_0000),
            -0.0
        );
    }

    #[test]
    fn round_bf16_emits_the_integer_schedule_without_a_bf16_type() {
        let mut builder = Builder::new();
        let value = Builder::arg(0, Ty::f32(vec![2, 8]));
        let rounded = round_bf16(&mut builder, &value);
        let body = builder.body();
        assert_eq!(rounded.ty.render(), "tensor<2x8xf32>");
        assert!(!body.contains("bf16"), "Metal cannot lower a BF16 type");
        assert!(
            body.contains(
                "stablehlo.bitcast_convert %arg0 : (tensor<2x8xf32>) -> tensor<2x8xui32>"
            )
        );
        assert!(body.contains("dense<32767> : tensor<2x8xui32>"));
        assert!(body.contains("dense<4294901760> : tensor<2x8xui32>"));
        assert!(body.contains("stablehlo.shift_right_logical"));
        assert_eq!(body.matches("stablehlo.add").count(), 2);
        assert_eq!(body.matches("stablehlo.and").count(), 2);
        assert!(body.contains("stablehlo.compare NE, %arg0, %arg0, FLOAT"));
        assert!(body.contains("stablehlo.select"));
    }

    fn rounds(body: &str) -> usize {
        body.matches("stablehlo.bitcast_convert").count() / 2
    }

    #[test]
    fn linear_rounds_the_product_before_the_bias() {
        let mut builder = Builder::new().with_precision(Precision::Bf16);
        let value = Builder::arg(0, Ty::f32(vec![4, 8]));
        let weight = Builder::arg(1, Ty::f32(vec![6, 8]));
        let bias = Builder::arg(2, Ty::f32(vec![6]));
        linear_bf16(&mut builder, &value, &weight, &bias);
        let body = builder.body();
        assert_eq!(rounds(body), 2);
        let dot = body.find("stablehlo.dot_general").unwrap();
        let first_round = body.find("stablehlo.bitcast_convert").unwrap();
        let bias_broadcast = body.find("%arg2").unwrap();
        assert!(dot < first_round && first_round < bias_broadcast);
        assert!(body.contains("tensor<4x8xbf16>"), "BF16 GEMM inputs");
    }

    #[test]
    fn layer_norm_rounds_normalized_scaled_and_shifted_values() {
        let mut builder = Builder::new();
        let value = Builder::arg(0, Ty::f32(vec![2, 8]));
        let weight = Builder::arg(1, Ty::f32(vec![8]));
        let bias = Builder::arg(2, Ty::f32(vec![8]));
        layer_norm_bf16(&mut builder, &value, &weight, &bias, 1.0e-6);
        let body = builder.body();
        assert_eq!(rounds(body), 3);
        let rsqrt = body.find("stablehlo.rsqrt").unwrap();
        let first_round = body.find("stablehlo.bitcast_convert").unwrap();
        let weight = body.find("%arg1").unwrap();
        assert!(rsqrt < first_round && first_round < weight);
    }

    #[test]
    fn attention_rounds_probabilities_before_the_value_gemm_and_normalizes_after() {
        let mut builder = Builder::new().with_precision(Precision::Bf16);
        let hidden = Builder::arg(0, Ty::f32(vec![4, 8]));
        let weights = AttentionWeights {
            qkv_weight: Builder::arg(1, Ty::f32(vec![24, 8])),
            qkv_bias: Builder::arg(2, Ty::f32(vec![24])),
            proj_weight: Builder::arg(3, Ty::f32(vec![8, 8])),
            proj_bias: Builder::arg(4, Ty::f32(vec![8])),
        };
        let freqs = Builder::arg(5, Ty::f32(vec![4, 2]));
        let bias = Builder::arg(6, Ty::f32(vec![4, 4]));
        let output = attention_bf16(&mut builder, &hidden, &weights, 2, &freqs, &bias);
        let body = builder.body();
        assert_eq!(output.ty.render(), "tensor<4x8xf32>");
        // qkv (2) + q/k RoPE (2) + probabilities + context + projection (2).
        assert_eq!(rounds(body), 8);
        let exp = body.find("stablehlo.exponential").unwrap();
        let after_exp = &body[exp..];
        let probability_round = after_exp.find("stablehlo.bitcast_convert").unwrap();
        let context_dot = after_exp.find("stablehlo.dot_general").unwrap();
        let normalize = after_exp.find("stablehlo.divide").unwrap();
        assert!(probability_round < context_dot && context_dot < normalize);
    }
}
