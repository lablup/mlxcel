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

use super::{GeluActivation, array_to_vec_f32, gelu_approx, gelu_sigmoid, gelu_tanh};
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

    let cases: [(GeluActivation, &[f32]); 4] = [
        (GeluActivation::Erf, &erf_ref),
        (GeluActivation::Tanh, &tanh_ref),
        (GeluActivation::TanhF32, &tanh_ref),
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
}
