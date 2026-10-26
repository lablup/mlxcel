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

//! Compiled kernels agree with eager evaluation (issues #1792, #2333).
//!
//! mlxcel used to overlay MLX's Metal compiled-kernel generator with a cast of
//! every op input whose type differed from the op's output type. For a
//! comparison that cast the operands to `bool`, so a compiled
//! `less(0.2, 0.7)` was `false`, `isnan` passed NaN through and the compiled
//! min-p filter returned its input unfiltered (#1792). For `Imag` it cast the
//! complex64 input to `float`, dropping the imaginary part, so a compiled
//! `imag()` returned 0 (#2333). The overlay is retired and Metal now uses
//! upstream emission. Each graph below runs through
//! [`ffi::compile_dtype_probe_for_test`] eagerly, compiled and compiled
//! shapeless; both compiled results must equal the eager one.
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
const REAL_OF_FUSED_COMPLEX: i32 = 7;
const IMAG_OF_FUSED_COMPLEX: i32 = 8;
const REAL_OF_COMPLEX_INPUT: i32 = 9;
const IMAG_OF_COMPLEX_INPUT: i32 = 10;
const ABS_OF_FUSED_COMPLEX: i32 = 11;

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
    // Parity guard for the mixed-dtype arithmetic the retired overlay cast
    // was added for (1923da32). MLX ops promote through `AsType`, so these
    // fused ops see matching input dtypes under upstream emission; the test
    // pins that compiled output still equals eager. Rounding may differ by a
    // bf16 ulp where a fused kernel contracts a multiply-add.
    let a = array(&A, &[5], dtype::BFLOAT16);
    let b = array(&B, &[5], dtype::FLOAT32);
    assert_compiled_matches_eager("bf16 / f32 chain", MIXED_ARITHMETIC, &a, &b, 1e-2);
    let a = array(&A, &[5], dtype::FLOAT32);
    let b = array(&B, &[5], dtype::BFLOAT16);
    assert_compiled_matches_eager("f32 / bf16 chain", MIXED_ARITHMETIC, &a, &b, 1e-2);
}

/// Run the in-graph and the complex64-input variant of a `real` / `imag`
/// graph on z = A + i*B, check the output is float32, and assert compiled
/// equals eager and eager equals `host`.
fn assert_complex_part_matches(arm: &str, graphs: [i32; 2], host: &[f32]) {
    // Every B is nonzero, so an `imag()` that drops the imaginary part
    // (returns 0) cannot match eager.
    assert!(B.iter().all(|&y| y != 0.0));
    let a = array(&A, &[5], dtype::FLOAT32);
    let b = array(&B, &[5], dtype::FLOAT32);
    for (graph, where_z) in graphs.into_iter().zip(["in graph", "complex64 input"]) {
        let label = format!("{arm}(z) * 2 + a, z {where_z}");
        let (_, out_dtype) = run(graph, &a, &b, EAGER);
        assert_eq!(out_dtype, dtype::FLOAT32, "{label}: eager output dtype");
        let eager = assert_compiled_matches_eager(&label, graph, &a, &b, 0.0);
        assert_eq!(eager, host, "{label}: eager");
    }
}

#[test]
fn compiled_real_of_complex64_matches_eager() {
    let host: Vec<f32> = A.iter().map(|&x| x * 2.0 + x).collect();
    assert_complex_part_matches(
        "real",
        [REAL_OF_FUSED_COMPLEX, REAL_OF_COMPLEX_INPUT],
        &host,
    );
}

#[test]
fn compiled_imag_of_complex64_matches_eager() {
    let host: Vec<f32> = A.iter().zip(&B).map(|(&x, &y)| y * 2.0 + x).collect();
    assert_complex_part_matches(
        "imag",
        [IMAG_OF_FUSED_COMPLEX, IMAG_OF_COMPLEX_INPUT],
        &host,
    );
}

#[test]
fn compiled_abs_of_complex64_matches_eager() {
    // MLX's `Abs` node keeps the complex64 dtype and casts to float32 after,
    // so a fused kernel must return |z| as a complex64 value, not a float.
    let a = array(&A, &[5], dtype::FLOAT32);
    let b = array(&B, &[5], dtype::FLOAT32);
    let label = "abs(z) * 2 + a, z in graph";
    let (_, out_dtype) = run(ABS_OF_FUSED_COMPLEX, &a, &b, EAGER);
    assert_eq!(out_dtype, dtype::FLOAT32, "{label}: eager output dtype");
    let eager = assert_compiled_matches_eager(label, ABS_OF_FUSED_COMPLEX, &a, &b, 1e-6);
    for ((&got, &x), &y) in eager.iter().zip(&A).zip(&B) {
        let want = x.hypot(y) * 2.0 + x;
        assert!(
            (got - want).abs() <= 1e-5,
            "{label}: eager {got} vs host {want}"
        );
    }
}
