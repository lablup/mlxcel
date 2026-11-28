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

use super::*;
use crate::emitter::builder::Ty;

/// Number of f32 -> F16 narrowing converts, one per rounding.
pub(in crate::emitter) fn f16_roundings(body: &str) -> usize {
    body.lines()
        .filter(|line| line.contains("stablehlo.convert") && line.trim_end().ends_with("xf16>"))
        .count()
}

fn rows(width: usize) -> Val {
    Builder::arg(0, Ty::f32(vec![2, width]))
}

#[test]
fn round_f16_is_a_narrowing_convert_pair() {
    let mut builder = Builder::new();
    let rounded = round_f16(&mut builder, &rows(3));
    let body = builder.body();
    assert_eq!(f16_roundings(body), 1);
    assert!(body.contains("(tensor<2x3xf16>) -> tensor<2x3xf32>"));
    assert_eq!(rounded.ty.elt, "f32");
}

#[test]
fn linear_rounds_the_product_and_the_bias_add() {
    let weight = Builder::arg(1, Ty::f32(vec![4, 3]));
    let bias = Builder::arg(2, Ty::f32(vec![4]));
    let mut builder = Builder::new();
    let _ = linear_f16(&mut builder, &rows(3), &weight, Some(&bias));
    assert_eq!(f16_roundings(builder.body()), 2);
    let mut builder = Builder::new();
    let _ = linear_f16(&mut builder, &rows(3), &weight, None);
    assert_eq!(f16_roundings(builder.body()), 1);
}

#[test]
fn layer_norm_rounds_the_normalized_value_and_one_fused_affine() {
    let weight = Builder::arg(1, Ty::f32(vec![3]));
    let bias = Builder::arg(2, Ty::f32(vec![3]));
    let mut builder = Builder::new();
    let _ = layer_norm_f16(&mut builder, &rows(3), &weight, &bias, 1e-5);
    assert_eq!(f16_roundings(builder.body()), 2);
}

#[test]
fn quick_gelu_rounds_coefficient_scale_sigmoid_and_product() {
    let mut builder = Builder::new();
    let _ = quick_gelu_f16(&mut builder, &rows(3));
    let body = builder.body();
    assert_eq!(f16_roundings(body), 4);
    assert!(body.contains(&crate::emitter::builder::f32_hex(1.702)));
    assert!(!body.contains("stablehlo.tanh"));
}
