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

//! Tests for the explicit GELU selector (#2423).
//!
//! The private tanh copies that `GeluActivation` replaced are reproduced here
//! verbatim as oracles, so folding a call site onto the shared helper is proven
//! bit-equal rather than "close": a fold that changed numerics would be a
//! behavior change for that family and needs its own measurement.

use super::{
    GeluActivation, array_to_vec_f32, compiled_gelu_tanh_f32, gelu_approx, gelu_sigmoid, gelu_tanh,
    gelu_tanh_f32,
};
use crate::{dtype, ffi};
use cxx::UniquePtr;

/// The f32-evaluated copy that lived in `vision/encoders/siglip.rs`,
/// `molmo2.rs` and `qwen3_vl.rs` before #2423.
fn legacy_pytorch_tanh_f32(x: &ffi::MlxArray) -> UniquePtr<ffi::MlxArray> {
    let output_dtype = ffi::array_dtype(x);
    let x = ffi::astype(x, dtype::FLOAT32);
    let half = ffi::full_f32(&[1], 0.5, dtype::FLOAT32);
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
    let sqrt_two_over_pi = ffi::full_f32(&[1], 0.797_884_6, dtype::FLOAT32);
    let cubic_coefficient = ffi::full_f32(&[1], 0.044_715, dtype::FLOAT32);
    let squared = ffi::multiply(&x, &x);
    let cubed = ffi::multiply(&squared, &x);
    let cubic = ffi::multiply(&cubic_coefficient, &cubed);
    let inner = ffi::multiply(&sqrt_two_over_pi, &ffi::add(&x, &cubic));
    let cdf = ffi::multiply(&half, &ffi::add(&one, &ffi::tanh(&inner)));
    let activated = ffi::multiply(&x, &cdf);
    if output_dtype == dtype::FLOAT32 {
        activated
    } else {
        ffi::astype(&activated, output_dtype)
    }
}

/// The input-dtype copy that lived in `vision/encoders/kimi_vl.rs` (and, for
/// f32 inputs, `models/kokoro/ops.rs`) before #2423.
fn legacy_tanh_input_dtype(x: &ffi::MlxArray) -> UniquePtr<ffi::MlxArray> {
    let scalar = |a: &ffi::MlxArray, v: f32| {
        let s = ffi::full_f32(&[1], v, ffi::array_dtype(a));
        ffi::multiply(a, &s)
    };
    let x2 = ffi::multiply(x, x);
    let x3 = ffi::multiply(&x2, x);
    let inner = ffi::add(x, &scalar(&x3, 0.044_715));
    let t = ffi::tanh(&scalar(&inner, 0.797_884_6));
    let one = ffi::full_f32(&[1], 1.0, ffi::array_dtype(&t));
    let g = ffi::add(&t, &one);
    ffi::multiply(&scalar(x, 0.5), &g)
}

/// A spread that covers the GELU knee, both tails, and magnitudes far past
/// anything a projection produces, deterministic so failures reproduce.
fn sample_inputs() -> Vec<f32> {
    let mut v: Vec<f32> = (-400..=400).map(|i| i as f32 * 0.025).collect();
    v.extend([-1.0e4, -300.0, -60.0, -12.5, 12.5, 60.0, 300.0, 1.0e4]);
    v
}

fn as_dtype(values: &[f32], target: i32) -> UniquePtr<ffi::MlxArray> {
    let x = ffi::from_slice_f32(values, &[values.len() as i32]);
    ffi::astype(&x, target)
}

fn bits_equal(a: &ffi::MlxArray, b: &ffi::MlxArray) -> bool {
    let eq = ffi::array_equal(a, b, true);
    ffi::eval(&eq);
    ffi::item_bool(&eq)
}

#[test]
fn tanh_f32_matches_the_folded_siglip_molmo2_qwen3_vl_copy_bit_for_bit() {
    for target in [dtype::FLOAT32, dtype::BFLOAT16, dtype::FLOAT16] {
        let x = as_dtype(&sample_inputs(), target);
        let folded = GeluActivation::TanhF32.apply(&x);
        let legacy = legacy_pytorch_tanh_f32(&x);
        assert_eq!(ffi::array_dtype(&folded), target);
        assert!(
            bits_equal(&folded, &legacy),
            "TanhF32 diverged from the private f32 copy it replaced for dtype {target}"
        );
    }
}

/// The compiled kernel the #2423 switched families use evaluates the same f32
/// formula as the eager helper; only FMA contraction inside the fused kernel
/// may move a lane. Bound the difference to a few f32 ulps for f32 output and
/// to one ulp of the narrow dtype after the cast back. The magnitudes here (up
/// to 1e4) exceed anything a vision projection produces (max |x| about 170).
#[test]
fn compiled_tanh_f32_matches_the_eager_helper_to_f32_rounding() {
    // (dtype, relative bound covering the allowed ulps, subnormal floor)
    let bounds = [
        (dtype::FLOAT32, 4.0 * f32::EPSILON, 4.0 * f32::from_bits(1)),
        (dtype::BFLOAT16, 2.0_f32.powi(-7), 0.0),
        (dtype::FLOAT16, 2.0_f32.powi(-10), 2.0_f32.powi(-24)),
    ];
    let inputs = sample_inputs();
    for (target, relative, floor) in bounds {
        let x = as_dtype(&inputs, target);
        let compiled = compiled_gelu_tanh_f32(&x);
        assert_eq!(ffi::array_dtype(&compiled), target);
        assert!(bits_equal(
            &compiled,
            &GeluActivation::TanhF32Compiled.apply(&x)
        ));
        let eager = gelu_tanh_f32(&x);
        let got = array_to_vec_f32(&ffi::astype(&compiled, dtype::FLOAT32));
        let want = array_to_vec_f32(&ffi::astype(&eager, dtype::FLOAT32));
        assert_eq!(got.len(), inputs.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.is_finite(), "dtype {target} lane {i}: compiled {g}");
            let bound = (relative * w.abs()).max(floor);
            assert!(
                (g - w).abs() <= bound,
                "dtype {target} lane {i} (x = {}): compiled {g} vs eager {w}",
                inputs[i]
            );
        }
    }
}

/// Tower-shaped bf16 input: one SigLIP MLP row block ([64, 4304]) of
/// N(0, 4^2) values with sparse outliers out to +-170, the largest
/// pre-activation magnitude measured on the gemma-3-4b-it-4bit tower. Built
/// on the host with a fixed LCG so it does not touch the global MLX RNG.
fn tower_like_bf16() -> (UniquePtr<ffi::MlxArray>, usize) {
    let (rows, cols) = (64_usize, 4304_usize);
    let mut state = 0x2423_u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let mut values = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        let (u1, u2) = (uniform(), uniform());
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let v = if i % 997 == 0 {
            (10.0 + 160.0 * u1) * if u2 < 0.5 { -1.0 } else { 1.0 }
        } else {
            4.0 * normal
        };
        values.push(v as f32);
    }
    let x = ffi::from_slice_f32(&values, &[rows as i32, cols as i32]);
    (ffi::astype(&x, dtype::BFLOAT16), rows * cols)
}

/// Tower-scale check of the same bound, with the size of the disagreement
/// printed for the record (`--nocapture`).
#[test]
fn compiled_tanh_f32_matches_the_eager_helper_on_tower_scale_bf16() {
    let (x, lanes) = tower_like_bf16();
    let compiled = compiled_gelu_tanh_f32(&x);
    let eager = gelu_tanh_f32(&x);
    let got = array_to_vec_f32(&compiled);
    let want = array_to_vec_f32(&eager);
    assert_eq!(got.len(), lanes);
    let (mut differing, mut max_abs, mut max_rel) = (0_usize, 0.0_f32, 0.0_f32);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g.is_finite(), "lane {i}: compiled {g}");
        let diff = (g - w).abs();
        if diff > 0.0 {
            differing += 1;
            max_abs = max_abs.max(diff);
            max_rel = max_rel.max(diff / w.abs());
        }
        assert!(
            diff <= 2.0_f32.powi(-7) * w.abs(),
            "lane {i}: compiled {g} vs eager {w}"
        );
    }
    eprintln!(
        "compiled vs eager TanhF32, bf16 [64, 4304]: {differing}/{lanes} lanes differ, \
         max abs diff {max_abs:e}, max rel diff {max_rel:e}"
    );
}

#[test]
fn tanh_matches_the_folded_input_dtype_copy_bit_for_bit() {
    // f32 covers Kokoro (which runs f32) and Kimi; bf16 / f16 cover Kimi K3
    // and LocateAnything, which run in the checkpoint dtype.
    for target in [dtype::FLOAT32, dtype::BFLOAT16, dtype::FLOAT16] {
        let x = as_dtype(&sample_inputs(), target);
        let folded = GeluActivation::Tanh.apply(&x);
        let legacy = legacy_tanh_input_dtype(&x);
        assert_eq!(ffi::array_dtype(&folded), target);
        assert!(
            bits_equal(&folded, &legacy),
            "Tanh diverged from the private input-dtype copy it replaced for dtype {target}"
        );
    }
}

#[test]
fn selector_variants_compute_their_named_formulas() {
    let values = [-3.0_f32, -1.0, 0.0, 1.0, 3.0];
    let x = ffi::from_slice_f32(&values, &[5]);
    // Hugging Face `gelu_pytorch_tanh` and exact `gelu` at the same points
    // (f64 references, rounded).
    let tanh_ref = [-0.003_637_392_f32, -0.158_808, 0.0, 0.841_192, 2.996_362_7];
    let erf_ref = [
        -0.004_049_694_f32,
        -0.158_655_25,
        0.0,
        0.841_344_8,
        2.995_950_3,
    ];
    let fast_ref: Vec<f32> = values
        .iter()
        .map(|&v| v / (1.0 + (-1.702 * v).exp()))
        .collect();

    let cases: [(GeluActivation, &[f32]); 5] = [
        (GeluActivation::Erf, &erf_ref),
        (GeluActivation::Tanh, &tanh_ref),
        (GeluActivation::TanhF32, &tanh_ref),
        (GeluActivation::TanhF32Compiled, &tanh_ref),
        (GeluActivation::Fast, &fast_ref),
    ];
    for (kind, expected) in cases {
        let out = kind.apply(&x);
        ffi::eval(&out);
        let got = array_to_vec_f32(&out);
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert!(
                (g - e).abs() <= 2.0e-6,
                "{kind:?} at {i}: got {g}, expected {e}"
            );
        }
    }
    // The default is erf, which is what `gelu_approx` has always computed.
    assert_eq!(GeluActivation::default(), GeluActivation::Erf);
    let erf = gelu_approx(&x);
    let selected = GeluActivation::default().apply(&x);
    assert!(bits_equal(&erf, &selected));
    let fast = gelu_sigmoid(&x);
    assert!(bits_equal(&fast, &GeluActivation::Fast.apply(&x)));
}

/// The comment that made `gelu_approx` erf blamed bf16 overflow inside the
/// `x*x*x` tanh graph. bf16 shares f32's exponent range, so the cube of any
/// finite projection output stays finite; tanh saturates and the result is
/// finite for every finite input. f16 is the dtype that can overflow the cube
/// (|x| > ~40), and tanh's saturation still keeps the product finite there
/// except where the cube itself is +-inf: those lanes must still come out as
/// +-x or 0, never NaN.
#[test]
fn tanh_in_input_dtype_never_produces_nan_for_finite_input() {
    for target in [dtype::BFLOAT16, dtype::FLOAT16] {
        let x = as_dtype(&sample_inputs(), target);
        let out = gelu_tanh(&x);
        let nan = ffi::any_all(&ffi::isnan(&out));
        ffi::eval(&nan);
        assert!(
            !ffi::item_bool(&nan),
            "tanh GELU produced NaN for dtype {target}"
        );
    }
}

#[test]
fn hugging_face_names_map_to_the_act2fn_formula() {
    for (name, expected) in [
        ("gelu", Some(GeluActivation::Erf)),
        ("gelu_python", Some(GeluActivation::Erf)),
        ("gelu_pytorch_tanh", Some(GeluActivation::Tanh)),
        ("gelu_new", Some(GeluActivation::Tanh)),
        ("gelu_fast", Some(GeluActivation::Tanh)),
        ("GELU_PYTORCH_TANH", Some(GeluActivation::Tanh)),
        ("quick_gelu", Some(GeluActivation::Fast)),
        ("silu", None),
    ] {
        assert_eq!(GeluActivation::from_hf_name(name), expected, "{name}");
    }
    assert_eq!(GeluActivation::Tanh.in_f32(), GeluActivation::TanhF32);
    assert_eq!(GeluActivation::Erf.in_f32(), GeluActivation::Erf);
    assert_eq!(GeluActivation::Fast.in_f32(), GeluActivation::Fast);
    for tanh in [GeluActivation::Tanh, GeluActivation::TanhF32] {
        assert_eq!(tanh.in_f32_compiled(), GeluActivation::TanhF32Compiled);
    }
    assert_eq!(GeluActivation::Erf.in_f32_compiled(), GeluActivation::Erf);
    assert_eq!(GeluActivation::Fast.in_f32_compiled(), GeluActivation::Fast);
}
