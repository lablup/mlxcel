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

//! The CUDA `QuantizedMatmul` batch fold (issue #2690).
//!
//! Batched decode hands a shared 2D quantized weight `x = [B, 1, K]`. The
//! overlay in `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/quantized/
//! quantized.cpp` folds that batch into `[B, K]` rows, so both shapes take the
//! same kernel over the same rows and must agree bit for bit. This guards the
//! fold's correctness (row order, and an `x` that is a row-contiguous view at
//! a non-zero offset, which the folded row view has to keep), not its speed:
//! on sm_121 the unfolded per-batch dispatch happens to give identical bits
//! too, so the test also passes with `MLXCEL_QMM_FOLD_BATCH=0`. The speed is
//! measured by `examples/qmm_fold_batch_probe.rs`. A dequantized reference bounds
//! both shapes against a wrong answer that happens to match itself.
//!
//!   cargo test -p mlxcel-core --features cuda --lib qmm_fold_batch_tests

use super::*;

const GROUP: i32 = 64;
const BITS: i32 = 4;

fn randn(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let key = random_key(seed);
    // SAFETY: `key` lives across the call.
    let x = unsafe { random_normal(shape, dtype::FLOAT32, key.as_ref().unwrap() as *const _) };
    astype(&x, dtype::BFLOAT16)
}

struct Quantized {
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
}

fn quantize(n: i32, k: i32) -> Quantized {
    let w = multiply_scalar(&randn(&[n, k], 3), 0.02);
    let q = quantize_weights(&w, GROUP, BITS);
    Quantized {
        w: quantized_weights_w(&q),
        scales: quantized_weights_scales(&q),
        biases: quantized_weights_biases(&q),
    }
}

fn qmm(x: &MlxArray, q: &Quantized) -> UniquePtr<MlxArray> {
    // SAFETY: every pointer is owned by `q` for the whole call.
    unsafe {
        quantized_matmul(
            x,
            &q.w,
            &q.scales,
            q.biases.as_ref().unwrap() as *const _,
            true,
            GROUP,
            BITS,
            "affine",
        )
    }
}

fn reference(x: &MlxArray, q: &Quantized) -> UniquePtr<MlxArray> {
    // SAFETY: every pointer is owned by `q` for the whole call.
    let w = unsafe {
        dequantize(
            &q.w,
            &q.scales,
            q.biases.as_ref().unwrap() as *const _,
            GROUP,
            BITS,
            "affine",
        )
    };
    let x = astype(x, dtype::FLOAT32);
    let w = astype(&w, dtype::FLOAT32);
    matmul(&x, &transpose(&w))
}

fn check(b: i32, n: i32, k: i32, offset_view: bool) {
    let q = quantize(n, k);
    // With `offset_view`, x is rows 1..=b of a [b + 1, 1, k] array: row
    // contiguous, but starting one row into its buffer.
    let x3 = if offset_view {
        let base = randn(&[b + 1, 1, k], 5);
        eval(&base);
        slice(&base, &[1, 0, 0], &[b + 1, 1, k])
    } else {
        randn(&[b, 1, k], 5)
    };
    let x2 = reshape(&x3, &[b, k]);
    let y3 = reshape(&qmm(&x3, &q), &[b, n]);
    let y2 = qmm(&x2, &q);
    let same = array_equal(&y3, &y2, false);
    eval(&same);
    assert!(
        item_bool(&same),
        "[B, 1, K] and [B, K] disagree at B={b} N={n} K={k} offset_view={offset_view}"
    );
    let close = allclose(
        &astype(&y2, dtype::FLOAT32),
        &reshape(&reference(&x2, &q), &[b, n]),
        2e-2,
        2e-2,
    );
    eval(&close);
    assert!(
        item_bool(&close),
        "quantized matmul strays from the dequantized reference at B={b} N={n} K={k}"
    );
}

#[test]
fn batch_of_rows_matches_folded_rows() {
    if !default_device_is_gpu() {
        eprintln!("skipping: no GPU default device");
        return;
    }
    // N = 1024 is a multiple of 128 (qmm_sm80 eligible from 8 rows); N = 704
    // is not, so B >= 8 takes qmm_naive there.
    for (n, k) in [(1024, 2048), (704, 2816)] {
        for b in [1, 2, 4, 8, 16] {
            check(b, n, k, false);
        }
        check(8, n, k, true);
    }
}
