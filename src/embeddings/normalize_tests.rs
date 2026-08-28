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

//! `--embd-normalize` numerical tests (#1452).
//!
//! Every mode is checked against the closed-form value computed here in Rust,
//! not against a recorded output, so a kernel change that shifts the numbers
//! fails rather than being blessed.

use super::*;

/// Run one mode over a `[rows, D]` matrix and read the result back.
fn normalized(rows: &[Vec<f32>], kind: EmbdNormalize) -> Vec<Vec<f32>> {
    let width = rows[0].len();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let input = mlxcel_core::from_slice_f32(&flat, &[rows.len() as i32, width as i32]);
    let out = apply_embd_normalize(&input, kind);
    mlxcel_core::try_eval(&out).expect("normalization evaluates");
    let values = mlxcel_core::utils::array_to_vec_f32(&out);
    values.chunks(width).map(<[f32]>::to_vec).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: width");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5 * e.abs().max(1.0),
            "{what}: component {i} is {a}, expected {e}"
        );
    }
}

const SAMPLE: [f32; 4] = [3.0, -4.0, 0.0, 12.0];

#[test]
fn the_domain_accepts_minus_one_and_above() {
    for value in [-1, 0, 1, 2, 3, 7, i32::MAX] {
        assert_eq!(EmbdNormalize::new(value).expect("in domain").value(), value);
    }
    let err = EmbdNormalize::new(-2).expect_err("out of domain");
    assert!(err.contains("--embd-normalize -2"), "{err}");
    assert!(err.contains("p-norm"), "{err}");
}

#[test]
fn the_default_is_euclidean_and_the_model_flag_maps_onto_the_domain() {
    assert_eq!(EmbdNormalize::default(), EmbdNormalize::EUCLIDEAN);
    assert_eq!(EmbdNormalize::EUCLIDEAN.value(), 2);
    assert_eq!(
        EmbdNormalize::from_model_flag(true),
        EmbdNormalize::EUCLIDEAN
    );
    assert_eq!(EmbdNormalize::from_model_flag(false), EmbdNormalize::NONE);
    assert!(EmbdNormalize::NONE.is_none());
    assert!(!EmbdNormalize::EUCLIDEAN.is_none());
}

#[test]
fn none_leaves_the_vector_untouched() {
    let out = normalized(&[SAMPLE.to_vec()], EmbdNormalize::NONE);
    assert_close(&out[0], &SAMPLE, "none");
}

#[test]
fn euclidean_produces_a_unit_vector() {
    let out = normalized(&[SAMPLE.to_vec()], EmbdNormalize::EUCLIDEAN);
    let norm: f32 = SAMPLE.iter().map(|v| v * v).sum::<f32>().sqrt();
    let expected: Vec<f32> = SAMPLE.iter().map(|v| v / norm).collect();
    assert_close(&out[0], &expected, "euclidean");
    let recovered: f32 = out[0].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((recovered - 1.0).abs() < 1e-5, "L2 norm is {recovered}");
}

#[test]
fn taxicab_produces_a_unit_l1_vector() {
    let out = normalized(&[SAMPLE.to_vec()], EmbdNormalize::TAXICAB);
    let norm: f32 = SAMPLE.iter().map(|v| v.abs()).sum();
    let expected: Vec<f32> = SAMPLE.iter().map(|v| v / norm).collect();
    assert_close(&out[0], &expected, "taxicab");
    let recovered: f32 = out[0].iter().map(|v| v.abs()).sum();
    assert!((recovered - 1.0).abs() < 1e-5, "L1 norm is {recovered}");
}

#[test]
fn a_p_norm_above_two_uses_that_p() {
    for p in [3, 5] {
        let kind = EmbdNormalize::new(p).expect("in domain");
        let out = normalized(&[SAMPLE.to_vec()], kind);
        let norm = SAMPLE
            .iter()
            .map(|v| v.abs().powi(p))
            .sum::<f32>()
            .powf(1.0 / p as f32);
        let expected: Vec<f32> = SAMPLE.iter().map(|v| v / norm).collect();
        assert_close(&out[0], &expected, &format!("p={p}"));
    }
}

#[test]
fn max_absolute_rescales_into_the_int16_range() {
    let out = normalized(&[SAMPLE.to_vec()], EmbdNormalize::MAX_ABS_INT16);
    let divisor = SAMPLE.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 32760.0;
    let expected: Vec<f32> = SAMPLE.iter().map(|v| v / divisor).collect();
    assert_close(&out[0], &expected, "max-abs");
    // The largest component lands exactly on the int16 bound, which is the
    // point of the mode.
    let largest = out[0].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!((largest - 32760.0).abs() < 1e-1, "largest is {largest}");
}

#[test]
fn a_zero_vector_normalizes_to_zeros_rather_than_nan() {
    // Upstream's `norm = sum > 0 ? 1/sum : 0`. A NaN here would reach the
    // response body and the route's finite-value guard would answer 500 for an
    // input b10621 serves.
    for kind in [
        EmbdNormalize::MAX_ABS_INT16,
        EmbdNormalize::TAXICAB,
        EmbdNormalize::EUCLIDEAN,
        EmbdNormalize::new(3).expect("in domain"),
    ] {
        let out = normalized(&[vec![0.0; 4]], kind);
        for (i, v) in out[0].iter().enumerate() {
            assert!(
                v.is_finite() && *v == 0.0,
                "{kind}: component {i} is {v}, expected 0"
            );
        }
    }
}

#[test]
fn rows_are_normalized_independently() {
    let rows = vec![SAMPLE.to_vec(), vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 4]];
    let out = normalized(&rows, EmbdNormalize::EUCLIDEAN);
    assert_eq!(out.len(), 3);
    assert_close(&out[1], &[1.0, 0.0, 0.0, 0.0], "unit row");
    assert_close(&out[2], &[0.0; 4], "zero row");
    let norm: f32 = SAMPLE.iter().map(|v| v * v).sum::<f32>().sqrt();
    let expected: Vec<f32> = SAMPLE.iter().map(|v| v / norm).collect();
    assert_close(&out[0], &expected, "first row");
}

#[test]
fn euclidean_matches_the_pooling_kernel_it_delegates_to() {
    // The default path must stay byte-identical to what shipped before this
    // change, so the delegation is asserted rather than assumed.
    let input = mlxcel_core::from_slice_f32(&SAMPLE, &[1, 4]);
    let through_kind = apply_embd_normalize(&input, EmbdNormalize::EUCLIDEAN);
    let through_kernel = crate::embeddings::normalize_l2(&input);
    mlxcel_core::try_eval(&through_kind).expect("evaluates");
    mlxcel_core::try_eval(&through_kernel).expect("evaluates");
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&through_kind),
        mlxcel_core::utils::array_to_vec_f32(&through_kernel)
    );
}
