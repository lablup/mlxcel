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

//! Reference parity for CUDA `qmv` at Gemma 4 LM-head vocabulary widths (`N >= 262144`).
//!
//! Issue #2958 investigated intermittent `<unused6226>` (token id `262143`, the
//! last row of Gemma 4's `N = 262144` quantized LM head) emissions during 4-bit
//! CUDA decode. While the emission itself came from `bfloat16` `promote_types`
//! demotion inside `categorical_inverse_cdf`, no test previously held CUDA's
//! `M = 1` `quantized_matmul` (`qmv_kernel` and `qmv_rowblock_kernel`) against
//! an `f32` `dequantize` + `matmul` CPU reference at `N = 262144` or at an `N`
//! leaving partial warp/row-block tails (`N = 262147`).
//!
//! Covers:
//! * `M = 1` in both `[1, K]` and `[1, 1, K]` decode layouts (asserting byte
//!   identity between them),
//! * `N = 262144` (exact Gemma 4 vocabulary size) and `N = 262147` (not
//!   divisible by `rows_per_block = 8` or `8 * rows_per_warp` for `R in 1..=4`),
//! * `K = 512` (`512 % (WARP_SIZE * elems_per_thread) == 0`, `has_residue_k = false`)
//!   and `K = 576` (`576 % 512 == 64`, `has_residue_k = true`, `576 % 64 == 0`),
//! * `float16` and `bfloat16` activations and affine 4-bit scales/biases,
//! * full-row relative error (`< 2e-2`) plus an explicit check on the final 64
//!   rows (`N - 64 .. N`, including `N - 1`),
//! * separate-process runs with `MLXCEL_QMV_ROWS_PER_WARP` unset, `2`, and `4`
//!   (the environment variable is latched once per process on first call).

#![cfg(feature = "cuda")]

use std::process::Command;
use std::sync::Mutex;

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const TAIL_ROWS: i32 = 64;
/// Relative error ceiling of a 4-bit half/bf16 GEMV against the f32 CPU reference
/// (same tolerance as `tests/rocm_qmm_batched_rows.rs`).
const TOLERANCE: f32 = 2e-2;
const SUBPROCESS_SENTINEL: &str = "MLXCEL_QMV_PARITY_SUBPROCESS";

static GPU: Mutex<()> = Mutex::new(());

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn on_cuda() -> bool {
    gpu_backend_kind() == GpuBackendKind::Cuda
}

/// Deterministic pseudo-random values in `[-0.5, 0.5)` (same generator as
/// `tests/cuda_qmm_determinism.rs`).
fn fixed_f32(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut state = seed;
    let values: Vec<f32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    let out = mlxcel_core::from_slice_f32(&values, shape);
    mlxcel_core::eval(&out);
    out
}

fn max_abs(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&f));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn assert_close_with_scale(name: &str, got: &MlxArray, reference: &MlxArray, scale: f32) {
    let diff = mlxcel_core::subtract(&mlxcel_core::astype(got, dtype::FLOAT32), reference);
    let rel = max_abs(&diff) / scale.max(1e-6);
    // Written so a NaN or infinity fails too.
    assert!(
        rel < TOLERANCE,
        "{name}: relative error {rel} against the f32 reference exceeds {TOLERANCE}"
    );
}

struct Weight {
    packed: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
}

impl Weight {
    fn from_f32(w_f32: &MlxArray, dt: i32) -> Self {
        let q = mlxcel_core::quantize_weights_with_mode(
            &mlxcel_core::astype(w_f32, dt),
            GROUP_SIZE,
            BITS,
            "affine",
        );
        let packed = mlxcel_core::quantized_weights_w(&q);
        let scales = mlxcel_core::quantized_weights_scales(&q);
        let biases = mlxcel_core::quantized_weights_biases(&q);
        mlxcel_core::eval(&packed);
        mlxcel_core::eval(&scales);
        mlxcel_core::eval(&biases);
        Self {
            packed,
            scales,
            biases,
        }
    }

    fn gpu(&self, x: &MlxArray, expected_dt: i32, n: i32) -> UniquePtr<MlxArray> {
        let _gpu = DefaultDeviceGuard::gpu();
        // SAFETY: `self.biases` is a live affine bias array owned by `self`.
        let out = unsafe {
            mlxcel_core::quantized_matmul(
                x,
                &self.packed,
                &self.scales,
                &*self.biases as *const MlxArray,
                true,
                GROUP_SIZE,
                BITS,
                "affine",
            )
        };
        let flat = mlxcel_core::reshape(&out, &[1, n]);
        if let Err(err) = mlxcel_core::try_eval(&flat) {
            panic!("quantized_matmul on the GPU failed: {err}");
        }
        assert_eq!(
            mlxcel_core::array_dtype(&flat),
            expected_dt,
            "quantized_matmul output dtype drifted from activation dtype"
        );
        flat
    }

    /// `dequantize` + `matmul` in `f32` on the CPU stream (`tests/rocm_qmm_batched_rows.rs`).
    fn reference(&self, x2: &MlxArray) -> UniquePtr<MlxArray> {
        let _cpu = DefaultDeviceGuard::cpu();
        // SAFETY: `self.biases` is a live affine bias array owned by `self`.
        let w = unsafe {
            mlxcel_core::dequantize(
                &self.packed,
                &self.scales,
                &*self.biases as *const MlxArray,
                GROUP_SIZE,
                BITS,
                "affine",
            )
        };
        let out = mlxcel_core::matmul(
            &mlxcel_core::astype(x2, dtype::FLOAT32),
            &mlxcel_core::transpose(&mlxcel_core::astype(&w, dtype::FLOAT32)),
        );
        mlxcel_core::eval(&out);
        out
    }
}

fn check_shape(w_f32: &MlxArray, dt: i32, dt_label: &str, n: i32, k: i32) {
    let rpw = std::env::var("MLXCEL_QMV_ROWS_PER_WARP").unwrap_or_else(|_| "unset".to_string());
    let name = format!("{dt_label} M=1 N={n} K={k} rows_per_warp={rpw}");

    let w = Weight::from_f32(w_f32, dt);
    let x_f32 = fixed_f32(&[1, k], 0x2958_1000 ^ (n as u64) ^ ((k as u64) << 20));
    let x2 = mlxcel_core::astype(&x_f32, dt);
    let x3 = mlxcel_core::reshape(&x2, &[1, 1, k]);
    mlxcel_core::eval(&x2);
    mlxcel_core::eval(&x3);

    let reference = w.reference(&x2);
    let scale = max_abs(&reference);

    let got_2d = w.gpu(&x2, dt, n);
    let got_3d = w.gpu(&x3, dt, n);

    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&got_2d),
        mlxcel_core::array_to_raw_bytes(&got_3d),
        "{name}: [1, 1, K] decode activation differs from [1, K]"
    );

    assert_close_with_scale(&format!("{name} all rows"), &got_2d, &reference, scale);

    // Explicitly check the final 64 rows (including row N - 1, e.g. 262143 for
    // Gemma 4's 262144-row LM head and 262146 for the unaligned 262147 case).
    let tail_start = n - TAIL_ROWS;
    let got_tail = mlxcel_core::slice(&got_2d, &[0, tail_start], &[1, n]);
    let ref_tail = mlxcel_core::slice(&reference, &[0, tail_start], &[1, n]);
    mlxcel_core::eval(&got_tail);
    mlxcel_core::eval(&ref_tail);
    assert_close_with_scale(
        &format!("{name} tail[{tail_start}..{n}]"),
        &got_tail,
        &ref_tail,
        scale,
    );

    let got_tail_f32 = mlxcel_core::astype(&got_tail, dtype::FLOAT32);
    mlxcel_core::eval(&got_tail_f32);
    let tail_bytes = mlxcel_core::array_to_raw_bytes(&got_tail_f32);
    assert_eq!(tail_bytes.len(), (TAIL_ROWS as usize) * 4);
    for (offset, chunk) in tail_bytes.chunks_exact(4).enumerate() {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            val.is_finite(),
            "{name}: non-finite output {val} at row {}",
            tail_start as usize + offset
        );
    }
}

#[test]
fn qmv_decode_shapes_match_f32_reference() {
    let _gpu = gpu_guard();
    if !on_cuda() {
        eprintln!("skip: not a CUDA build/device");
        return;
    }

    // K = 512 has no residue against WARP_SIZE * elems_per_thread (32 * 16 = 512);
    // K = 576 leaves a 64-element k-residue while staying divisible by GROUP_SIZE = 64.
    for n in [262_144, 262_147] {
        for k in [512, 576] {
            let w_f32 = fixed_f32(&[n, k], 0x2958_0001 ^ (n as u64) ^ ((k as u64) << 16));
            check_shape(&w_f32, dtype::FLOAT16, "f16", n, k);
            check_shape(&w_f32, dtype::BFLOAT16, "bf16", n, k);
        }
    }
    mlxcel_core::synchronize_default();
}

/// `MLXCEL_QMV_ROWS_PER_WARP` is read once per process in `qmv.cu`, so row-blocked
/// arms (`2` and `4`) and the stock `unset` arm must each run in a fresh process.
#[test]
fn qmv_rows_per_warp_arms_match_f32_reference_in_fresh_processes() {
    if std::env::var_os(SUBPROCESS_SENTINEL).is_some() {
        return;
    }
    let _gpu = gpu_guard();
    if !on_cuda() {
        eprintln!("skip: not a CUDA build/device");
        return;
    }

    let exe = std::env::current_exe().expect("test binary path");
    for arm in [None, Some("2"), Some("4")] {
        let mut cmd = Command::new(&exe);
        cmd.env(SUBPROCESS_SENTINEL, "1")
            .arg("--exact")
            .arg("qmv_decode_shapes_match_f32_reference")
            .arg("--test-threads=1")
            .arg("--nocapture");
        match arm {
            None => {
                cmd.env_remove("MLXCEL_QMV_ROWS_PER_WARP");
            }
            Some(v) => {
                cmd.env("MLXCEL_QMV_ROWS_PER_WARP", v);
            }
        }
        let output = cmd.output().expect("spawn qmv parity subprocess");
        assert!(
            output.status.success(),
            "qmv parity subprocess failed for MLXCEL_QMV_ROWS_PER_WARP={arm:?}:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
