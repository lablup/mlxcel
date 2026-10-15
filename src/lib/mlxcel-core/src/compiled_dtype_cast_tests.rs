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

//! Compiled kernels agree with eager evaluation (issue #1792).
//!
//! The Metal overlay of MLX's compiled-kernel generator casts an op's
//! mismatched inputs so mixed bf16/f32 arithmetic deduces one template type.
//! Keyed on the output type, the cast turned comparison operands into `bool`,
//! so a compiled `less(0.2, 0.7)` was `false`, `isnan` passed NaN through and
//! the compiled min-p filter returned its input unfiltered. Each graph below
//! runs through [`ffi::compile_dtype_probe_for_test`] eagerly, compiled and
//! compiled shapeless; both compiled results must equal the eager one.
//!
//!   cargo test -p mlxcel-core --lib compiled_dtype_cast_tests

use super::*;

const SELECT_LESS: i32 = 0;
const GREATER_SCALED: i32 = 1;
const COMPARISON_CODES: i32 = 2;
const ISNAN_FILL: i32 = 3;
const ISINF_FILL: i32 = 4;
const MIN_P: i32 = 5;
const MIXED_ARITHMETIC: i32 = 6;

const EAGER: i32 = 0;
const COMPILED: [(i32, &str); 2] = [(1, "compiled"), (2, "compiled shapeless")];

const A: [f32; 5] = [0.2, -0.5, 0.7, 0.0, 3.0];
const B: [f32; 5] = [0.7, 0.3, 0.2, 1.0, 3.0];

fn array(data: &[f32], shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let a = from_slice_f32(data, shape);
    if dt == dtype::FLOAT32 {
        a
    } else {
        astype(&a, dt)
    }
}

/// The output as f32 values plus its dtype.
fn run(graph: i32, a: &MlxArray, b: &MlxArray, mode: i32) -> (Vec<f32>, i32) {
    let out = compile_dtype_probe_for_test(graph, a, b, mode)
        .unwrap_or_else(|e| panic!("graph {graph} mode {mode}: {e}"));
    let out_dtype = array_dtype(&out);
    let f = astype(&out, dtype::FLOAT32);
    try_eval(&f).unwrap_or_else(|e| panic!("graph {graph} mode {mode}: {e}"));
    let values = array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (values, out_dtype)
}

/// Run `graph` eagerly and both compiled ways, assert each compiled result
/// matches the eager one within `tol` (NaN matches NaN), and return the eager
/// values.
fn assert_compiled_matches_eager(
    label: &str,
    graph: i32,
    a: &MlxArray,
    b: &MlxArray,
    tol: f32,
) -> Vec<f32> {
    let (eager, eager_dtype) = run(graph, a, b, EAGER);
    for (mode, name) in COMPILED {
        let (compiled, compiled_dtype) = run(graph, a, b, mode);
        assert_eq!(compiled_dtype, eager_dtype, "{label}: {name} output dtype");
        let same = compiled.len() == eager.len()
            && compiled.iter().zip(&eager).all(|(&c, &e)| {
                (c.is_nan() && e.is_nan()) || c == e || (c - e).abs() <= tol * e.abs().max(1.0)
            });
        assert!(same, "{label}: {name} {compiled:?} != eager {eager:?}");
    }
    eager
}

#[test]
fn compiled_same_dtype_f32_comparisons_match_eager() {
    let a = array(&A, &[5], dtype::FLOAT32);
    let b = array(&B, &[5], dtype::FLOAT32);

    // The eager oracle itself, against host arithmetic.
    let selected = assert_compiled_matches_eager("f32 where(a < b)", SELECT_LESS, &a, &b, 0.0);
    let host: Vec<f32> = A
        .iter()
        .zip(&B)
        .map(|(&x, &y)| if x < y { x } else { y })
        .collect();
    assert_eq!(selected, host, "eager where(a < b, a, b)");

    let greater = assert_compiled_matches_eager("f32 2a > b", GREATER_SCALED, &a, &b, 0.0);
    let host: Vec<f32> = A
        .iter()
        .zip(&B)
        .map(|(&x, &y)| f32::from(u8::from(2.0 * x > y)))
        .collect();
    assert_eq!(greater, host, "eager 2a > b");

    let codes = assert_compiled_matches_eager("f32 comparisons", COMPARISON_CODES, &a, &b, 0.0);
    let host: Vec<f32> = A
        .iter()
        .zip(&B)
        .map(|(&x, &y)| {
            let bits = [x < y, x <= y, x > y, x >= y, x == y, x != y];
            bits.iter()
                .enumerate()
                .map(|(i, &on)| if on { (1 << i) as f32 } else { 0.0 })
                .sum()
        })
        .collect();
    assert_eq!(codes, host, "eager comparison codes");
}

#[test]
fn compiled_mixed_f16_f32_comparisons_match_eager() {
    for (a_dtype, b_dtype) in [
        (dtype::FLOAT16, dtype::FLOAT32),
        (dtype::FLOAT32, dtype::FLOAT16),
        (dtype::BFLOAT16, dtype::FLOAT32),
    ] {
        let a = array(&A, &[5], a_dtype);
        let b = array(&B, &[5], b_dtype);
        for graph in [SELECT_LESS, GREATER_SCALED, COMPARISON_CODES] {
            let label = format!("dtypes ({a_dtype}, {b_dtype}) graph {graph}");
            assert_compiled_matches_eager(&label, graph, &a, &b, 0.0);
        }
    }
}

#[test]
fn compiled_isnan_and_isinf_match_eager() {
    let data = [1.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -2.5, 0.0];
    for dt in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        let x = array(&data, &[6], dt);
        let filled =
            assert_compiled_matches_eager("where(isnan(x), 7, x)", ISNAN_FILL, &x, &x, 0.0);
        assert!(
            !filled.iter().any(|v| v.is_nan()) && filled[1] == 7.0,
            "dtype {dt}: eager isnan fill {filled:?}"
        );
        let filled =
            assert_compiled_matches_eager("where(isinf(x), 9, x)", ISINF_FILL, &x, &x, 0.0);
        assert!(
            filled[2] == 9.0 && filled[3] == 9.0,
            "dtype {dt}: eager isinf fill {filled:?}"
        );
    }
}

#[test]
fn compiled_min_p_filter_matches_eager() {
    // Per row, min-p keeps logits within ln(0.3) of the row max: 3.0 and 2.0
    // in the first row, 0.5 and 0.0 in the second.
    let logits = [
        2.0, 1.0, 0.0, -1.0, 3.0, 0.5, -3.0, 0.5, 0.0, -2.0, -1.0, -4.0,
    ];
    let x = array(&logits, &[2, 6], dtype::FLOAT32);
    let filtered = assert_compiled_matches_eager("min-p", MIN_P, &x, &x, 0.0);
    let kept: Vec<f32> = filtered.iter().copied().filter(|&v| v > f32::MIN).collect();
    assert_eq!(kept, [2.0, 3.0, 0.5, 0.0], "eager min-p {filtered:?}");
}

#[test]
fn compiled_mixed_bf16_f32_arithmetic_matches_eager() {
    // The case the Metal overlay's input cast exists for. Rounding may differ
    // by a bf16 ulp where a fused kernel contracts a multiply-add.
    let a = array(&A, &[5], dtype::BFLOAT16);
    let b = array(&B, &[5], dtype::FLOAT32);
    assert_compiled_matches_eager("bf16 / f32 chain", MIXED_ARITHMETIC, &a, &b, 1e-2);
    let a = array(&A, &[5], dtype::FLOAT32);
    let b = array(&B, &[5], dtype::BFLOAT16);
    assert_compiled_matches_eager("f32 / bf16 chain", MIXED_ARITHMETIC, &a, &b, 1e-2);
}
