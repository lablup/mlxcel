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

//! An f32 matmul on the CPU stream of a ROCm build is right every time (issue
//! #2072, `patches-rocm/LOCAL_FIXES.md` item 27).
//!
//! On an APU the ROCm allocator hands out fine-grained device memory for every
//! array, CPU-stream arrays included, and the CPU stream's f32 matmul is an
//! OpenBLAS `cblas_sgemm` that writes its result straight into that memory.
//! With OpenBLAS's default thread count (32 on the gfx1151 host) some output
//! columns came back wrong, differently from call to call: this was the
//! intermittent `qmm 2880x2880 M=1` failure of `rocm_mxfp4_quant`, whose GPU
//! result was right and whose CPU reference was not. The allocator now drops
//! OpenBLAS to one thread on its first fine-grained allocation.
//!
//! The case is the gpt-oss decode projection, `[1, 2880] x [2880, 2880]^T`,
//! repeated, against an f64 host reference. With the allocator change reverted
//! this test failed in 5 of 5 runs, within the first 8 calls, on an otherwise
//! idle gfx1151 host. The wrong columns were each the last column of one
//! OpenBLAS thread's share of the output (shares of 93 columns, so 650,
//! 1022, 1859, ...), the column whose 64-byte line the next thread also
//! writes. With one thread each call takes about 1.4 s, because the CPU reads
//! fine-grained memory slowly, which is what bounds the call count.
//!
//! ```sh
//! cargo test --features rocm --test rocm_cpu_blas_finegrained -- --test-threads=1
//! ```
//!
//! The test moves the process-global default device, so it holds
//! `streams::lock_default_device` for its whole body. Skips on any other
//! backend.

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const N: usize = 2880;
const K: usize = 2880;
const CALLS: usize = 16;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn eval_ok(label: &str, a: &MlxArray) {
    if let Err(err) = mlxcel_core::try_eval(a) {
        panic!("{label}: evaluation failed: {err}");
    }
}

/// Standard normal f32 samples, generated on the GPU (MLX's CPU generator is
/// slow at this size); the CPU stream reads the same evaluated bytes.
fn normal(shape: &[i32]) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    let out = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    eval_ok("random input", &out);
    out
}

fn host_f32(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::array_to_raw_bytes(a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn cpu_stream_f32_matmul_is_exact_on_every_call() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(2072);
    let w = normal(&[N as i32, K as i32]);
    let x = normal(&[1, K as i32]);
    let (wv, xv) = (host_f32(&w), host_f32(&x));
    let want: Vec<f64> = (0..N)
        .map(|c| {
            let row = &wv[c * K..(c + 1) * K];
            row.iter()
                .zip(&xv)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum()
        })
        .collect();
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "reference is all zeros");

    let _cpu = DefaultDeviceGuard::cpu();
    for call in 0..CALLS {
        let y = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&w));
        eval_ok("CPU matmul", &y);
        let got = host_f32(&y);
        assert_eq!(got.len(), N, "call {call}: output length");
        let wrong: Vec<usize> = (0..N)
            .filter(|&c| (got[c] as f64 - want[c]).abs() / scale > 1e-4)
            .collect();
        assert!(
            wrong.is_empty(),
            "call {call}: {} of {N} output columns wrong, first {:?}: got {} want {}",
            wrong.len(),
            &wrong[..wrong.len().min(8)],
            got[wrong[0]],
            want[wrong[0]]
        );
    }
}
