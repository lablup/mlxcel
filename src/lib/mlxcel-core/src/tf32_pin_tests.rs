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

//! Sentinel for the test-process TF32 pin (issues #1065, #1088, #1259).
//!
//! `pin_full_precision_f32_matmuls_for_tests` in `lib.rs` sets
//! `MLX_ENABLE_TF32=0` before MLX latches it, so the suite's f32
//! algorithm-equivalence tests compare full-f32 arithmetic. Most of those
//! tests only notice a lost pin on some backends: on CUDA the four
//! `mlxcel-core` tests #1065 named stay green under TF32 because their
//! shapes go through MLX's gemv and SDPA kernels, never cuBLAS. This test
//! reaches the GEMM that TF32 governs on both CUDA (cuBLAS
//! `CUBLAS_COMPUTE_32F_FAST_TF32`) and Apple GPU generation 17 (NAX), and
//! compares it with an f64 host reference, so removing or breaking the pin
//! turns it red on either backend instead of only on whichever hardware
//! happens to route a given fixture through the reduced-precision kernel.

use crate::mla::testkit::{Rng, to_vec_f32};
use crate::{eval, from_slice_f32, matmul};

/// M > 1 with an untransposed right operand keeps MLX off its gemv path
/// (`can_use_gemv` takes only `M == 1 && b_transposed` or
/// `N == 1 && !a_transposed`), so this is a real GEMM.
const M: usize = 16;
const K: usize = 512;
const N: usize = 16;

/// Full f32 accumulation over K = 512 inputs in [-1, 1) lands within about
/// 1e-5 of the f64 reference. TF32 rounds each input to a 10-bit mantissa,
/// which puts the error near 1e-3 at this K, two orders above the bound.
const F32_ABS_TOL: f64 = 1e-4;

#[test]
fn f32_gemm_runs_at_full_precision_in_the_test_process() {
    let mut rng = Rng::new(0x1065);
    let a = rng.vec(M * K, 1.0);
    let b = rng.vec(K * N, 1.0);

    let a_arr = from_slice_f32(&a, &[M as i32, K as i32]);
    let b_arr = from_slice_f32(&b, &[K as i32, N as i32]);
    let out = matmul(&a_arr, &b_arr);
    eval(&out);
    let got = to_vec_f32(&out);

    let mut worst = 0.0_f64;
    for i in 0..M {
        for j in 0..N {
            let want: f64 = (0..K)
                .map(|k| a[i * K + k] as f64 * b[k * N + j] as f64)
                .sum();
            worst = worst.max((got[i * N + j] as f64 - want).abs());
        }
    }
    assert!(
        worst < F32_ABS_TOL,
        "f32 GEMM differs from the f64 reference by {worst} (bound {F32_ABS_TOL}); \
         MLX_ENABLE_TF32 is {:?}. A TF32-class error here means the test-process \
         pin in lib.rs is not in effect, and every f32 parity test in this binary \
         is measuring TF32 instead of f32.",
        std::env::var("MLX_ENABLE_TF32").ok()
    );
}
