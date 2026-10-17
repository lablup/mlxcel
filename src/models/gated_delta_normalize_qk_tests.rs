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

//! `normalize_qk` against the reference l2norm (#2361).
//!
//! The reference GDN/KDA q/k normalization is
//! `l2norm(x) = x / sqrt(sum(x^2) + 1e-6)`. The fused `rms_norm` adds its eps
//! to `mean(x^2)`, so the eps has to be `1e-6 / head_k_dim`; the flat `1e-6`
//! the call sites used to pass is `head_k_dim` times too large. The two only
//! disagree where `sum(x^2)` is near the eps, so the discriminating input is a
//! small-norm row.

use super::{QK_L2NORM_EPS, normalize_qk, normalize_qk_rms_eps, scaled_fast_rms_norm_no_weight};
use mlxcel_core::utils::array_to_vec_f32;

const HEAD_DIM: usize = 128;

/// Deterministic values in `[-scale, scale]`, kept away from zero so a
/// per-element relative error is well defined.
fn filler(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32;
            let magnitude = 0.25 + 0.75 * u;
            let sign = if (state >> 7) & 1 == 0 { 1.0 } else { -1.0 };
            sign * magnitude * scale
        })
        .collect()
}

/// Row-wise `scale * x / sqrt(sum(x^2) + eps)` in f64 over rows of `HEAD_DIM`.
fn reference_l2norm(x: &[f32], scale: f64, eps: f64) -> Vec<f64> {
    x.chunks(HEAD_DIM)
        .flat_map(|row| {
            let sum: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let denom = (sum + eps).sqrt();
            row.iter().map(move |&v| scale * f64::from(v) / denom)
        })
        .collect()
}

fn max_rel_err(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (f64::from(a) - e).abs() / e.abs())
        .fold(0.0, f64::max)
}

#[test]
fn normalize_qk_rms_eps_is_l2norm_eps_over_head_dim() {
    for d in [64usize, 128, 256] {
        let inv_scale = (d as f32).powf(-0.5);
        let rms_eps = normalize_qk_rms_eps(inv_scale, QK_L2NORM_EPS);
        let expected = QK_L2NORM_EPS / d as f32;
        assert_eq!(
            rms_eps.to_bits(),
            expected.to_bits(),
            "head_dim {d}: rms eps {rms_eps:e} != 1e-6 / {d} = {expected:e}"
        );
    }
}

/// The discriminating check: on a small-norm q/k (entries around `1e-5`,
/// `head_dim = 128`, so `sum(x^2)` is about `4e-9`, far below the eps),
/// `normalize_qk` matches the hand-computed reference within `1e-6` relative
/// error and the old flat-eps call does not.
#[test]
fn normalize_qk_matches_reference_l2norm_on_small_norm_rows() {
    let rows = 4usize;
    let shape = [1, 2, 2, HEAD_DIM as i32];
    let q_data = filler(rows * HEAD_DIM, 7, 1e-5);
    let k_data = filler(rows * HEAD_DIM, 11, 1e-5);
    let q = mlxcel_core::from_slice_f32(&q_data, &shape);
    let k = mlxcel_core::from_slice_f32(&k_data, &shape);
    let inv_scale = (HEAD_DIM as f32).powf(-0.5);
    let eps = f64::from(QK_L2NORM_EPS);

    let q_ref = reference_l2norm(&q_data, f64::from(inv_scale), eps);
    let k_ref = reference_l2norm(&k_data, 1.0, eps);

    let (q_new, k_new) = normalize_qk(&q, &k, inv_scale, QK_L2NORM_EPS);
    let q_err = max_rel_err(&array_to_vec_f32(&q_new), &q_ref);
    let k_err = max_rel_err(&array_to_vec_f32(&k_new), &k_ref);
    eprintln!("normalize_qk small-norm rel err: q={q_err:e} k={k_err:e}");
    assert!(q_err <= 1e-6, "q: normalize_qk rel err {q_err:e} > 1e-6");
    assert!(k_err <= 1e-6, "k: normalize_qk rel err {k_err:e} > 1e-6");

    // The pre-#2361 call: the l2norm eps handed straight to rms_norm.
    let q_old = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, QK_L2NORM_EPS);
    let k_old = scaled_fast_rms_norm_no_weight(&k, inv_scale, QK_L2NORM_EPS);
    let q_old_err = max_rel_err(&array_to_vec_f32(&q_old), &q_ref);
    let k_old_err = max_rel_err(&array_to_vec_f32(&k_old), &k_ref);
    eprintln!("flat-eps small-norm rel err: q={q_old_err:e} k={k_old_err:e}");
    assert!(
        q_old_err > 0.5 && k_old_err > 0.5,
        "the flat-eps formula should be far off on small-norm rows: q={q_old_err:e} k={k_old_err:e}"
    );
}

/// On ordinary rows `sum(x^2)` dwarfs either eps, so the fix leaves them
/// where they were. This is why most tokens are unaffected.
#[test]
fn normalize_qk_matches_flat_eps_on_ordinary_rows() {
    let shape = [1, 1, 2, HEAD_DIM as i32];
    let q_data = filler(2 * HEAD_DIM, 3, 1.0);
    let k_data = filler(2 * HEAD_DIM, 5, 1.0);
    let q = mlxcel_core::from_slice_f32(&q_data, &shape);
    let k = mlxcel_core::from_slice_f32(&k_data, &shape);
    let inv_scale = (HEAD_DIM as f32).powf(-0.5);
    let eps = f64::from(QK_L2NORM_EPS);

    let (q_new, k_new) = normalize_qk(&q, &k, inv_scale, QK_L2NORM_EPS);
    let q_ref = reference_l2norm(&q_data, f64::from(inv_scale), eps);
    let k_ref = reference_l2norm(&k_data, 1.0, eps);
    assert!(max_rel_err(&array_to_vec_f32(&q_new), &q_ref) <= 1e-6);
    assert!(max_rel_err(&array_to_vec_f32(&k_new), &k_ref) <= 1e-6);

    let k_old = scaled_fast_rms_norm_no_weight(&k, inv_scale, QK_L2NORM_EPS);
    let drift = max_rel_err(&array_to_vec_f32(&k_old), &k_ref);
    assert!(
        drift <= 1e-5,
        "flat eps drifted {drift:e} on an ordinary row"
    );
}
