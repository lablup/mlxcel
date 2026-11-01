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

//! Bit-exact bf16 carrier of the Gemma3n sparse `compiled_gelu_topk` (#2795).
//!
//! mlx-lm's `gelu_topk` keeps every intermediate in the input dtype: the f32
//! multiplier is cast to the stddev dtype, the zero and the `nn.gelu_approx`
//! constants are weak Python scalars, and the cube is one `power` op. The
//! references below spell that graph out op by op from mlxcel-core primitives.
//!
//! MLX's CUDA compiled kernels contract a bf16 `a + b * c` into one
//! single-rounded FMA, which the CPU backend and uncompiled ops do not. mlx-lm's
//! compiled `gelu_topk` has two such sites (`mean + std * multiplier` and
//! `x + 0.044715 * x**3`), so on CUDA it equals the per-op tape with those two
//! sites fused, and on CPU it equals the plain per-op tape. The eager kernel is
//! the same compiled graph, so it must equal one of the two bit for bit.

use super::*;

/// sqrt(2) * erfinv(2 * 0.95 - 1), Gemma3n's `activation_sparsity = 0.95`.
const SPARSITY_095_MULTIPLIER: f32 = 1.644_853_6;

/// Deterministic pseudo-normal values: Box-Muller over a fixed 64-bit LCG.
fn pseudo_normal(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 24 high bits -> (0, 1); the +0.5 keeps ln() finite.
        ((state >> 40) as f64 + 0.5) / (1u64 << 24) as f64
    };
    (0..len)
        .map(|_| {
            let (u1, u2) = (uniform(), uniform());
            let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            normal as f32 * scale
        })
        .collect()
}

/// `a + b * c` rounded once into `a`'s dtype, the way an FMA rounds. A bf16
/// product is exact in f32, so only the f32 add can round before the final
/// cast, and a double-rounding tie there is vanishingly rare.
fn fused_multiply_add(a: &MlxArray, b: &MlxArray, c: &MlxArray) -> UniquePtr<MlxArray> {
    let f32_of = |v: &MlxArray| astype(v, dtype::FLOAT32);
    let wide = add(&f32_of(a), &multiply(&f32_of(b), &f32_of(c)));
    astype(&wide, array_dtype(a))
}

/// mlx-lm `gelu_topk` in the input dtype, op for op. With `fma_sites`, the two
/// mul+add sites that MLX's CUDA kernel compiler contracts round once.
fn gelu_topk_input_dtype_tape(
    x: &MlxArray,
    std_multiplier: f32,
    fma_sites: bool,
) -> UniquePtr<MlxArray> {
    let dt = array_dtype(x);
    let scalar = |value: f32| full_f32(&[1], value, dt);
    let mean = mean_axis_native_dtype(x, -1, true);
    let stddev = std_axis(x, -1, true, 0);
    let multiplier = scalar(std_multiplier);
    let cutoff = if fma_sites {
        fused_multiply_add(&mean, &stddev, &multiplier)
    } else {
        add(&mean, &multiply(&stddev, &multiplier))
    };
    let z = maximum(&scalar(0.0), &subtract(x, &cutoff));
    let cube = power(&z, &scalar(3.0));
    let cubic = scalar(0.044_715);
    let poly = if fma_sites {
        fused_multiply_add(&z, &cubic, &cube)
    } else {
        add(&z, &multiply(&cubic, &cube))
    };
    let inner = multiply(&scalar(0.797_884_6), &poly);
    let half_z = multiply(&scalar(0.5), &z);
    multiply(&half_z, &add(&scalar(1.0), &tanh(&inner)))
}

/// The pre-#2795 eager carrier: the strong f32 multiplier and zero promote the
/// cutoff and the whole GELU to f32, cubed as `x*x*x`, cast back at the end.
fn gelu_topk_f32_carrier(x: &MlxArray, std_multiplier: f32) -> UniquePtr<MlxArray> {
    let f32_scalar = |value: f32| full_f32(&[1], value, dtype::FLOAT32);
    let mean = mean_axis_native_dtype(x, -1, true);
    let stddev = std_axis(x, -1, true, 0);
    let cutoff = add(&mean, &multiply(&stddev, &f32_scalar(std_multiplier)));
    let z = maximum(&subtract(x, &cutoff), &f32_scalar(0.0));
    let cube = multiply(&multiply(&z, &z), &z);
    let poly = add(&z, &multiply(&f32_scalar(0.044_715), &cube));
    let inner = multiply(&f32_scalar(0.797_884_6), &poly);
    let scale = multiply(&f32_scalar(0.5), &add(&f32_scalar(1.0), &tanh(&inner)));
    astype(&multiply(&z, &scale), array_dtype(x))
}

/// Element count whose f32 bit patterns differ.
fn bit_mismatches(a: &MlxArray, b: &MlxArray) -> usize {
    let a = crate::utils::array_to_vec_f32(&astype(a, dtype::FLOAT32));
    let b = crate::utils::array_to_vec_f32(&astype(b, dtype::FLOAT32));
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
fn test_compiled_gelu_topk_bf16_is_bit_exact_with_the_mlx_lm_carrier() {
    for (width, seed) in [(2048usize, 0x2795u64), (8192, 0x5972)] {
        let rows = 4usize;
        let values = pseudo_normal(rows * width, seed, 1.7);
        let x = astype(
            &from_slice_f32(&values, &[rows as i32, width as i32]),
            dtype::BFLOAT16,
        );
        let out = compiled_gelu_topk(&x, SPARSITY_095_MULTIPLIER);
        let per_op = gelu_topk_input_dtype_tape(&x, SPARSITY_095_MULTIPLIER, false);
        let fused = gelu_topk_input_dtype_tape(&x, SPARSITY_095_MULTIPLIER, true);
        let f32_carrier = gelu_topk_f32_carrier(&x, SPARSITY_095_MULTIPLIER);
        eval(&out);
        eval(&per_op);
        eval(&fused);
        eval(&f32_carrier);
        assert_eq!(array_dtype(&out), dtype::BFLOAT16);

        let to_per_op = bit_mismatches(&out, &per_op);
        let to_fused = bit_mismatches(&out, &fused);
        let to_f32 = bit_mismatches(&out, &f32_carrier);
        eprintln!(
            "compiled_gelu_topk bf16 [{rows}, {width}]: mismatches per-op tape {to_per_op}, \
             fma-site tape {to_fused}, f32 carrier {to_f32} (of {})",
            rows * width
        );
        assert!(
            to_per_op == 0 || to_fused == 0,
            "width {width}: compiled_gelu_topk is not the mlx-lm bf16 carrier \
             (per-op {to_per_op}, fma-site {to_fused} mismatches)"
        );
        // The references must be able to tell the carriers apart, or the exact
        // match above proves nothing.
        assert!(
            bit_mismatches(&per_op, &f32_carrier) > 0 && bit_mismatches(&fused, &f32_carrier) > 0,
            "width {width}: the f32-carrier reference is indistinguishable from the bf16 tape"
        );
        assert!(
            to_f32 > 0,
            "width {width}: kernel still uses the f32 carrier"
        );
    }
}
