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

use super::*;

pub(crate) fn host(x: &MlxArray) -> Vec<f32> {
    let x = mlxcel_core::astype(x, dtype::FLOAT32);
    mlxcel_core::eval(&x);
    mlxcel_core::array_to_raw_bytes(&x)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn prompt_template_matches_upstream() {
    assert_eq!(
        format_prompt(Some("  a red panda ")),
        "<role>HUMAN</role> Generate an image: a red panda\n<role>ASSISTANT</role>\n<IMAGE1>"
    );
    assert_eq!(
        format_prompt(None),
        "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>"
    );
}

#[test]
fn text_attends_text_and_queries_attend_everything() {
    let mask = query_attention_mask_values(2, 2);
    #[rustfmt::skip]
    let want = [
        true, true, false, false,
        true, true, false, false,
        true, true, true, true,
        true, true, true, true,
    ];
    assert_eq!(mask, want);
    assert_eq!(
        mlxcel_core::array_shape(&query_attention_mask(3, 4)),
        vec![1, 1, 7, 7]
    );
}

#[test]
fn gelu_forms_follow_their_formulas() {
    let xs = [-3.0_f32, -0.5, 0.0, 0.7, 2.5];
    let x = mlxcel_core::from_slice_f32(&xs, &[5]);
    let approx = host(&gelu_approx(&x));
    let exact = host(&gelu(&x));
    for (i, &v) in xs.iter().enumerate() {
        let v = f64::from(v);
        let a = 0.5
            * v
            * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh());
        assert!((f64::from(approx[i]) - a).abs() < 1e-5, "{approx:?}");
        // erf(x / sqrt 2) via the identity with the normal CDF.
        let e = v * 0.5 * (1.0 + libm_erf(v / std::f64::consts::SQRT_2));
        assert!((f64::from(exact[i]) - e).abs() < 1e-5, "{exact:?}");
    }
}

/// Abramowitz-Stegun 7.1.26 is too loose; use a series that is exact enough here.
fn libm_erf(x: f64) -> f64 {
    let mut sum = 0.0;
    let mut term = x;
    for n in 0..60 {
        sum += term / f64::from(2 * n + 1);
        term *= -x * x / f64::from(n + 1);
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

#[test]
fn dense_adds_its_bias_once() {
    let w = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let b = mlxcel_core::from_slice_f32(&[0.5, -1.0, 2.0], &[3]);
    let x = mlxcel_core::from_slice_f32(&[1.0, -1.0], &[1, 2]);
    let biased = Dense::from_parts(&w, Some(b)).unwrap();
    assert_eq!(host(&biased.forward(&x)), vec![-0.5, -2.0, 1.0]);
    let plain = Dense::from_parts(&w, None).unwrap();
    assert_eq!(host(&plain.forward(&x)), vec![-1.0, -1.0, -1.0]);
}
