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

//! A decode batch's `[B, 1, K]` activation against an unbatched quantized
//! weight is one product of `B` rows (issue #2156).
//!
//! The batched server decode feeds every projection `[B, 1, K]`. The ROCm
//! overlay's `QuantizedMatmul::eval_gpu` used to read that as `B` separate
//! one-row products and launch the batched qmv kernel, which streams the whole
//! weight once per batch element; a batch-4 Meta-Llama-3.1-8B 4-bit decode
//! step cost 158 ms against 31 ms at batch 1 on gfx1151. It now folds the
//! batch into the row count when the weight is unbatched and the activation
//! row contiguous, which is how the Metal backend reads it.
//!
//! The fold makes `[B, 1, K]`, `[B, K]` and `[1, B, K]` the same GEMM, so the
//! test asserts their output bytes are equal and that each is within the
//! half-precision tolerance of an f32 CPU reference. Before the fix the
//! `[B, 1, K]` result came from a different kernel: for the f16 4096 x 4096
//! case at `B = 4` it differed from `[B, K]` by up to 3.9e-3 (measured with
//! `examples/qmm_batch_rows_probe.rs` on gfx1151), so the byte comparison
//! fails without it. A strided `[B, 1, K]` view (not row contiguous) keeps the
//! batched path and is checked against the reference only.
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --release --features rocm --test rocm_qmm_batched_rows -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
/// Relative error of a half-precision GEMM against the f32 reference.
const TOLERANCE: f32 = 2e-2;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn pseudo_random(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&f));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

struct Weight {
    packed: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
}

impl Weight {
    fn new(n: i32, k: i32, dt: i32) -> Self {
        let w = mlxcel_core::from_slice_f32(&pseudo_random((n * k) as usize, 0x2156), &[n, k]);
        let q = mlxcel_core::quantize_weights_with_mode(
            &mlxcel_core::astype(&w, dt),
            GROUP_SIZE,
            BITS,
            "affine",
        );
        Self {
            packed: mlxcel_core::quantized_weights_w(&q),
            scales: mlxcel_core::quantized_weights_scales(&q),
            biases: mlxcel_core::quantized_weights_biases(&q),
        }
    }

    /// `quantized_matmul` on the GPU, flattened to `[rows, n]`.
    fn gpu(&self, x: &MlxArray, rows: i32, n: i32) -> UniquePtr<MlxArray> {
        let _gpu = DefaultDeviceGuard::gpu();
        // SAFETY: every reference is a live array owned by `self` or the caller.
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
        let out = mlxcel_core::reshape(&out, &[rows, n]);
        if let Err(err) = mlxcel_core::try_eval(&out) {
            panic!("quantized_matmul on the GPU failed: {err}");
        }
        out
    }

    /// `dequantize` + `matmul` in f32 on the CPU stream.
    fn reference(&self, x2: &MlxArray) -> UniquePtr<MlxArray> {
        let _cpu = DefaultDeviceGuard::cpu();
        // SAFETY: `self.biases` is a live array owned by `self`.
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

fn assert_close(name: &str, got: &MlxArray, reference: &MlxArray) {
    let scale = max_abs(reference).max(1e-6);
    let diff = mlxcel_core::subtract(&mlxcel_core::astype(got, dtype::FLOAT32), reference);
    let rel = max_abs(&diff) / scale;
    // Written so a NaN fails too.
    assert!(
        rel < TOLERANCE,
        "{name}: relative error {rel} against the f32 reference"
    );
}

fn check(dt: i32, label: &str, batch: i32, n: i32, k: i32) {
    let name = format!("{label} B={batch} N={n} K={k}");
    let w = Weight::new(n, k, dt);
    let x_f32 = mlxcel_core::from_slice_f32(
        &pseudo_random((batch * k) as usize, 0x2156 ^ batch as u32),
        &[batch, k],
    );
    let x2 = mlxcel_core::astype(&x_f32, dt);
    let x3 = mlxcel_core::reshape(&x2, &[batch, 1, k]);
    let x1b = mlxcel_core::reshape(&x2, &[1, batch, k]);
    mlxcel_core::eval(&x3);
    mlxcel_core::eval(&x1b);
    let reference = w.reference(&x2);

    let rows = w.gpu(&x2, batch, n);
    let decode = w.gpu(&x3, batch, n);
    let single = w.gpu(&x1b, batch, n);
    assert_close(&format!("{name} [B, K]"), &rows, &reference);
    let rows_bytes = mlxcel_core::array_to_raw_bytes(&rows);
    assert!(
        mlxcel_core::array_to_raw_bytes(&decode) == rows_bytes,
        "{name}: [B, 1, K] differs from [B, K]"
    );
    assert!(
        mlxcel_core::array_to_raw_bytes(&single) == rows_bytes,
        "{name}: [1, B, K] differs from [B, K]"
    );

    // Every other row of a [B, 2, K] activation: a [B, 1, K] view with a gap
    // between rows, which must not be folded.
    let wide_f32 = mlxcel_core::from_slice_f32(
        &pseudo_random((batch * 2 * k) as usize, 0x2156 ^ (batch as u32) << 8),
        &[batch, 2, k],
    );
    let wide = mlxcel_core::astype(&wide_f32, dt);
    let strided = mlxcel_core::slice(&wide, &[0, 0, 0], &[batch, 1, k]);
    mlxcel_core::eval(&strided);
    let strided_rows = mlxcel_core::reshape(&mlxcel_core::copy(&strided), &[batch, k]);
    let got = w.gpu(&strided, batch, n);
    assert_close(
        &format!("{name} strided"),
        &got,
        &w.reference(&strided_rows),
    );
}

#[test]
fn decode_batch_rows_are_one_product() {
    if !on_rocm() {
        eprintln!("skip: not a ROCm build/device");
        return;
    }
    // The f16 4096 x 4096 batch-4 case is the one measured to differ before
    // the fold; the others cover batch 2 and 8 and bf16 (whose folded rows
    // take the fused WMMA route) at a size the CPU reference finishes fast.
    check(dtype::FLOAT16, "f16", 4, 4096, 4096);
    for batch in [2, 8] {
        check(dtype::FLOAT16, "f16", batch, 1024, 4096);
    }
    for batch in [2, 4, 8] {
        check(dtype::BFLOAT16, "bf16", batch, 1024, 4096);
    }
}
