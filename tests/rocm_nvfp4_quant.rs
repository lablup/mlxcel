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

//! NVFP4 runs natively and correctly on ROCm (issue #2617).
//!
//! The capability table (`hardware.rs`) reports NVFP4 as native on ROCm, so
//! MLX-native NVFP4 planes load as-is and ModelOpt NVFP4 checkpoints are
//! transcoded to NVFP4 instead of converted to affine 4-bit. These tests hold
//! that claim to numbers: every route `QuantizedMatmul` and `GatherQMM` take
//! for NVFP4 (E2M1 values, one E4M3 scale per 16, an optional per-expert
//! float32 global scale) must match `x @ dequantize(w).T` computed in f32 on
//! the CPU, where MLX's own `dequantize` is the ground truth.
//!
//! - `quantized_matmul`: `nvfp4_qmv_kernel` at one row and at 6, 16 and 33
//!   rows (the multi-row arm, below the gfx1151 crossover), and
//!   `nvfp4_dequantize_kernel` + GEMM at 64 and 300 rows, for f32, f16 and
//!   bf16; a batched weight; a non-transposed weight; and a global scale
//!   applied after the matmul, as the dense loader does.
//! - `gather_qmm`: unsorted (decode) and sorted (prefill, eight rows a block)
//!   index layouts, with and without the per-expert global scale, and an
//!   activation with three rows per expert.
//!
//! Each case prints its maximum absolute error and that error relative to
//! the reference's largest magnitude. Every result is evaluated through
//! `try_eval`, so a HIP failure fails the test with its status. Each test
//! holds `streams::lock_default_device` for its whole body. Skips on any
//! other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_nvfp4_quant -- --test-threads=1 --nocapture
//! ```

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{
    GpuBackendKind, QuantMode, QuantModeSupport, gpu_backend_kind, quant_mode_support,
};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const GROUP_SIZE: i32 = 16;
const BITS: i32 = 4;
const MODE: &str = "nvfp4";
/// E4M3 max times E2M1 max: MLX dequantizes with `global_scale / (448 * 6)`.
const GLOBAL_SCALE_DENOM: f32 = 448.0 * 6.0;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn eval_ok(label: &str, a: &MlxArray) {
    if let Err(err) = mlxcel_core::try_eval(a) {
        panic!("{label}: evaluation failed: {err}");
    }
}

/// Standard normal samples of `shape` in `dt`, generated on the GPU.
fn normal(shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    // SAFETY: null is the bridge's documented value for "no PRNG key".
    let f32 = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    let out = mlxcel_core::astype(&f32, dt);
    eval_ok("random input", &out);
    out
}

fn cpu_eval(label: &str, a: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    eval_ok(label, &a);
    a
}

/// NVFP4 quantization of `w` on the GPU, optionally relative to a one-element
/// f32 global scale: `(packed, scales)`.
fn quantize(
    w: &MlxArray,
    global_scale: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let _guard = DefaultDeviceGuard::gpu();
    let q = match global_scale {
        Some(gs) => mlxcel_core::quantize_weights_with_global_scale(w, GROUP_SIZE, BITS, MODE, gs),
        None => mlxcel_core::quantize_weights_with_mode(w, GROUP_SIZE, BITS, MODE),
    };
    assert!(!mlxcel_core::quantized_weights_has_biases(&q));
    let packed = mlxcel_core::quantized_weights_w(&q);
    let scales = mlxcel_core::quantized_weights_scales(&q);
    eval_ok("nvfp4 quantize (packed)", &packed);
    eval_ok("nvfp4 quantize (scales)", &scales);
    assert_eq!(
        mlxcel_core::array_dtype(&scales),
        dtype::UINT8,
        "E4M3 scale bytes"
    );
    (packed, scales)
}

/// f32 dequantization on the CPU, without a global scale.
fn dequantize_cpu(packed: &MlxArray, scales: &MlxArray) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    // SAFETY: null is the bridge's documented value for absent biases.
    let dense = unsafe {
        mlxcel_core::dequantize(packed, scales, std::ptr::null(), GROUP_SIZE, BITS, MODE)
    };
    cpu_eval(
        "CPU nvfp4 dequantize",
        mlxcel_core::astype(&dense, dtype::FLOAT32),
    )
}

/// `(max |got - want|, that / max |want|)`, on the CPU in f32.
fn errors(got: &MlxArray, want: &MlxArray) -> (f32, f32) {
    let _guard = DefaultDeviceGuard::cpu();
    let got = mlxcel_core::astype(got, dtype::FLOAT32);
    let want = mlxcel_core::astype(want, dtype::FLOAT32);
    let diff = cpu_eval(
        "error metric",
        mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(&got, &want))),
    );
    let scale = cpu_eval(
        "error metric",
        mlxcel_core::max_all(&mlxcel_core::abs(&want)),
    );
    let (diff, scale) = (mlxcel_core::item_f32(&diff), mlxcel_core::item_f32(&scale));
    assert!(scale > 0.0, "reference is all zeros");
    (diff, diff / scale)
}

/// Relative tolerance against the f32 reference. The weights are exact in
/// every float type, so the error is output rounding plus summation order:
/// far under these bounds when the kernel is right, far above them when it
/// decodes a wrong nibble, scale or global scale.
fn tolerance(dt: i32) -> f32 {
    match dt {
        dtype::FLOAT32 => 1e-4,
        dtype::FLOAT16 => 5e-3,
        dtype::BFLOAT16 => 2e-2,
        _ => unreachable!("untested dtype {dt}"),
    }
}

fn dtype_name(dt: i32) -> &'static str {
    match dt {
        dtype::FLOAT32 => "f32",
        dtype::FLOAT16 => "f16",
        dtype::BFLOAT16 => "bf16",
        _ => "?",
    }
}

fn check(label: &str, dt: i32, got: &MlxArray, want: &MlxArray) {
    assert_eq!(
        mlxcel_core::array_shape(got),
        mlxcel_core::array_shape(want),
        "{label}: shape"
    );
    let (abs_err, rel_err) = errors(got, want);
    eprintln!("{label}: max abs {abs_err:.3e}, relative {rel_err:.3e}");
    assert!(
        rel_err <= tolerance(dt),
        "{label}: relative error {rel_err} exceeds {}",
        tolerance(dt)
    );
}

/// `x @ dense.T` (or `x @ dense` when `transpose` is false) in f32 on the CPU.
fn dense_reference(x: &MlxArray, dense: &MlxArray, transpose: bool) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    let xf = mlxcel_core::astype(x, dtype::FLOAT32);
    let rhs = if transpose {
        mlxcel_core::swap_axes(dense, -1, -2)
    } else {
        mlxcel_core::copy(dense)
    };
    cpu_eval("reference", mlxcel_core::matmul(&xf, &rhs))
}

fn qmm_gpu(
    label: &str,
    x: &MlxArray,
    packed: &MlxArray,
    scales: &MlxArray,
    transpose: bool,
) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    // SAFETY: null is the bridge's documented value for absent biases.
    let y = unsafe {
        mlxcel_core::quantized_matmul(
            x,
            packed,
            scales,
            std::ptr::null(),
            transpose,
            GROUP_SIZE,
            BITS,
            MODE,
        )
    };
    eval_ok(label, &y);
    y
}

#[test]
fn capability_table_reports_nvfp4_native_on_rocm() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    assert_eq!(
        quant_mode_support(QuantMode::Nvfp4),
        QuantModeSupport::Native
    );
}

/// `quantized_matmul` with a transposed NVFP4 weight (as `QuantizedLinear`
/// calls it) at every row count: 1 (decode), 6 and 16 (speculative-decode
/// verify widths) and 33 (a partial eight-row block) on `nvfp4_qmv_kernel`,
/// and 64 and 300 on the dequantize + GEMM route (the gfx1151 crossover is
/// 36 rows at these sizes, 24 above 2048).
#[test]
fn quantized_matmul_matches_dequantized_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(2617);
    for (out_features, in_features) in [(256, 512), (512, 272), (2048, 2048)] {
        let w = normal(&[out_features, in_features], dtype::FLOAT32);
        let (packed, scales) = quantize(&w, None);
        let dense = dequantize_cpu(&packed, &scales);
        for dt in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
            for m in [1, 6, 16, 33, 64, 300] {
                let label = format!("qmm {out_features}x{in_features} M={m} {}", dtype_name(dt));
                let x = normal(&[1, m, in_features], dt);
                let got = qmm_gpu(&label, &x, &packed, &scales, true);
                assert_eq!(mlxcel_core::array_dtype(&got), dt, "{label}: dtype");
                check(&label, dt, &got, &dense_reference(&x, &dense, true));
            }
        }
    }
}

/// A batched NVFP4 weight `[2, N, K]` against a batched activation (the
/// batched qmv arm at 3 rows, dequantize + batched GEMM at 64), and a
/// non-transposed weight (always dequantize + GEMM).
#[test]
fn quantized_matmul_batched_and_untransposed_match_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(26171);
    let (n, k) = (384, 512);
    let w = normal(&[2, n, k], dtype::FLOAT32);
    let (packed, scales) = quantize(&w, None);
    let dense = dequantize_cpu(&packed, &scales);
    for dt in [dtype::FLOAT32, dtype::BFLOAT16] {
        for m in [3, 64] {
            let label = format!("qmm batched [2,{n},{k}] M={m} {}", dtype_name(dt));
            let x = normal(&[2, m, k], dt);
            let got = qmm_gpu(&label, &x, &packed, &scales, true);
            check(&label, dt, &got, &dense_reference(&x, &dense, true));
        }
    }

    // Untransposed: w is [K, N] quantized along N, y = x @ dequantize(w).
    let w = normal(&[k, n], dtype::FLOAT32);
    let (packed, scales) = quantize(&w, None);
    let dense = dequantize_cpu(&packed, &scales);
    for dt in [dtype::FLOAT32, dtype::BFLOAT16] {
        for m in [1, 40] {
            let label = format!("qmm untransposed {k}x{n} M={m} {}", dtype_name(dt));
            let x = normal(&[m, k], dt);
            let got = qmm_gpu(&label, &x, &packed, &scales, false);
            check(&label, dt, &got, &dense_reference(&x, &dense, false));
        }
    }
}

/// A one-element f32 global scale encoded into the block scales at quantize
/// time and applied after the matmul as `global_scale / (448 * 6)`, which is
/// how the dense loader runs an NVFP4 layer with a global scale
/// (`quantized_linear_forward_global_scale`).
#[test]
fn quantized_matmul_with_global_scale_matches_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(26172);
    let (n, k) = (512, 1024);
    let w = normal(&[n, k], dtype::FLOAT32);
    let amax = {
        let _guard = DefaultDeviceGuard::cpu();
        let a = cpu_eval("amax", mlxcel_core::max_all(&mlxcel_core::abs(&w)));
        mlxcel_core::item_f32(&a)
    };
    let gs_value = amax * 1.5;
    let gs = mlxcel_core::from_slice_f32(&[gs_value], &[1]);
    let (packed, scales) = quantize(&w, Some(&gs));
    let factor = mlxcel_core::from_slice_f32(&[gs_value / GLOBAL_SCALE_DENOM], &[1]);
    let dense = {
        let _guard = DefaultDeviceGuard::cpu();
        cpu_eval(
            "scaled dense",
            mlxcel_core::multiply(&dequantize_cpu(&packed, &scales), &factor),
        )
    };
    for dt in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        for m in [1, 8, 64] {
            let label = format!("qmm global scale {n}x{k} M={m} {}", dtype_name(dt));
            let x = normal(&[1, m, k], dt);
            let got = {
                let _guard = DefaultDeviceGuard::gpu();
                // SAFETY: `factor` outlives the call; null means no biases and
                // no linear bias.
                let y = unsafe {
                    mlxcel_core::quantized_linear_forward_global_scale(
                        &x,
                        &packed,
                        &scales,
                        std::ptr::null(),
                        &*factor as *const MlxArray,
                        std::ptr::null(),
                        GROUP_SIZE,
                        BITS,
                        MODE,
                    )
                };
                eval_ok(&label, &y);
                y
            };
            check(&label, dt, &got, &dense_reference(&x, &dense, true));
        }
    }
}

#[path = "rocm_nvfp4_quant/gather.rs"]
mod gather;
