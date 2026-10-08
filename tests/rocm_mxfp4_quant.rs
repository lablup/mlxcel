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

//! mxfp4 runs natively and correctly on ROCm (issue #1808).
//!
//! The backend quantization capability table (`hardware.rs`, issue #1806)
//! reports mxfp4 as native on ROCm, so mxfp4 checkpoints such as
//! gpt-oss-20b-MXFP4-Q4 load as-is and run the ROCm `quantize`,
//! `quantized_matmul` and `gather_qmm` kernels. These tests hold that claim to
//! numbers, one per failure the issue was filed for:
//!
//! - GPU `quantize` in mxfp4 at 4096x4096 failed with `hipLaunchKernel(...)
//!   failed: invalid configuration argument` (an `arg_reduce` launch whose
//!   1-D grid exceeded AMD's 2^32 - 1 thread limit, `patches-rocm/
//!   LOCAL_FIXES.md` item 11). It must now succeed and match the CPU
//!   quantizer bit for bit.
//! - `quantized_matmul` in mxfp4 hung at 256x512 with M = 1, and on the
//!   unfixed fork faulted in `qmv_warp_shared_kernel` (E8M0 scale dispatch,
//!   item 8). It must now finish and match a dequantized f32 reference on the
//!   CPU, for the qmv (M = 1) and qmm (M > 1) paths and the gpt-oss shape.
//! - `gather_qmm` in mxfp4 reached an affine-only kernel instantiation (item
//!   10). It must match a per-expert dequantized reference, sorted and
//!   unsorted, as `SwitchLinear` calls it for MoE experts.
//! - bf16 mxfp4 `gather_qmm` at gpt-oss-20b's decode shape ran the per-row
//!   `gather_qmv_kernel` (item 10) at about 1.39 ms per call. It now takes
//!   the warp-shared gather kernel's mxfp4 word path (issue #2178) and must
//!   stay as accurate as the per-row kernel, which
//!   `MLX_ROCM_GATHER_QMV_USE_WARP=0` still selects.
//!
//! Every result is evaluated through `try_eval`, which surfaces a ROCm GPU
//! failure as an error (issue #1804), so a regression fails the test with the
//! HIP status instead of producing silent garbage. A hang still blocks the
//! test, so run it under a timeout when bisecting.
//!
//! Every test moves the process-global default device, so each holds
//! `streams::lock_default_device` for its whole body. Skips on any other
//! backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_mxfp4_quant -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{
    GpuBackendKind, QuantMode, QuantModeSupport, gpu_backend_kind, quant_mode_support,
};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const GROUP_SIZE: i32 = 32;
const BITS: i32 = 4;
const MODE: &str = "mxfp4";

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// Evaluate `a`, failing the test with the backend's error if the GPU work
/// behind it failed.
fn eval_ok(label: &str, a: &MlxArray) {
    if let Err(err) = mlxcel_core::try_eval(a) {
        panic!("{label}: evaluation failed: {err}");
    }
}

fn device(gpu: bool) -> DefaultDeviceGuard {
    if gpu {
        DefaultDeviceGuard::gpu()
    } else {
        DefaultDeviceGuard::cpu()
    }
}

/// Standard normal samples of `shape` in `dt`, generated on the GPU (MLX's
/// CPU generator needs about 16 s for 16M samples on this host). Every
/// comparison below reads the same evaluated bytes on both devices.
fn normal(shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    let f32 = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    let out = mlxcel_core::astype(&f32, dt);
    eval_ok("random input", &out);
    out
}

/// mxfp4 quantization of `w` on the chosen device: `(packed, scales)`.
fn quantize_on(gpu: bool, w: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let _guard = device(gpu);
    let q = mlxcel_core::quantize_weights_with_mode(w, GROUP_SIZE, BITS, MODE);
    assert!(
        !mlxcel_core::quantized_weights_has_biases(&q),
        "mxfp4 must not produce affine biases"
    );
    let packed = mlxcel_core::quantized_weights_w(&q);
    let scales = mlxcel_core::quantized_weights_scales(&q);
    let side = if gpu { "GPU" } else { "CPU" };
    eval_ok(&format!("{side} mxfp4 quantize (packed)"), &packed);
    eval_ok(&format!("{side} mxfp4 quantize (scales)"), &scales);
    (packed, scales)
}

/// f32 dequantization of an mxfp4 pair, on the CPU.
fn dequantize_cpu(packed: &MlxArray, scales: &MlxArray) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    let dense = unsafe {
        mlxcel_core::dequantize(packed, scales, std::ptr::null(), GROUP_SIZE, BITS, MODE)
    };
    let dense = mlxcel_core::astype(&dense, dtype::FLOAT32);
    eval_ok("CPU mxfp4 dequantize", &dense);
    dense
}

/// `max |got - want| / max |want|`, computed on the CPU in f32.
fn relative_error(got: &MlxArray, want: &MlxArray) -> f32 {
    let _guard = DefaultDeviceGuard::cpu();
    let got = mlxcel_core::astype(got, dtype::FLOAT32);
    let want = mlxcel_core::astype(want, dtype::FLOAT32);
    let diff = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(&got, &want)));
    let scale = mlxcel_core::max_all(&mlxcel_core::abs(&want));
    eval_ok("error metric", &diff);
    eval_ok("error metric", &scale);
    let (diff, scale) = (mlxcel_core::item_f32(&diff), mlxcel_core::item_f32(&scale));
    assert!(scale > 0.0, "reference is all zeros");
    diff / scale
}

/// Relative tolerance of a GPU result in `dt` against an f32 reference. The
/// mxfp4 weights are exact in every float type, so the error is the output
/// rounding plus accumulation order: well under these bounds when the kernel
/// is right, and far above them when it decodes the wrong scale or nibble.
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

fn raw(a: &MlxArray) -> Vec<u8> {
    mlxcel_core::array_to_raw_bytes(a)
}

/// The capability table must keep reporting mxfp4 as native on ROCm; the
/// numeric tests below are the evidence that claim rests on. If they ever
/// fail for good, the entry has to become `ConvertTo(Affine)` instead.
#[test]
fn capability_table_reports_mxfp4_native_on_rocm() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    assert_eq!(
        quant_mode_support(QuantMode::Mxfp4),
        QuantModeSupport::Native
    );
}

/// Rows `[start, stop)` of a 2-D array, evaluated on the CPU.
fn rows(a: &MlxArray, start: i32, stop: i32) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    let cols = mlxcel_core::array_shape(a)[1];
    let out = mlxcel_core::copy(&mlxcel_core::slice(a, &[start, 0], &[stop, cols]));
    eval_ok("row slice", &out);
    out
}

/// GPU mxfp4 `quantize` at the issue's failing 4096x4096 shape (and the
/// 256x512 shape that always worked) matches the CPU quantizer bit for bit.
///
/// Quantization is independent per row group, so the CPU reference is taken
/// over row bands of the input rather than the whole matrix (MLX's CPU
/// quantizer needs about 90 s for 4096x4096 on this host): the first and last
/// 64 rows at 4096x4096, which catch a failed launch and a grid that stops
/// short of the tail, and every row at 256x512.
#[test]
fn gpu_quantize_matches_cpu_bitwise() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(1808);
    for (n_rows, n_cols) in [(256, 512), (4096, 4096)] {
        for dt in [dtype::FLOAT32, dtype::BFLOAT16] {
            let label = format!("{n_rows}x{n_cols} {}", dtype_name(dt));
            let w = normal(&[n_rows, n_cols], dt);
            let (gpu_packed, gpu_scales) = quantize_on(true, &w);
            assert_eq!(
                mlxcel_core::array_shape(&gpu_packed),
                vec![n_rows, n_cols * BITS / 32],
                "{label}: packed shape"
            );
            assert_eq!(
                mlxcel_core::array_shape(&gpu_scales),
                vec![n_rows, n_cols / GROUP_SIZE],
                "{label}: scales shape"
            );
            let band = n_rows.min(64);
            let bands = if band == n_rows {
                vec![(0, n_rows)]
            } else {
                vec![(0, band), (n_rows - band, n_rows)]
            };
            for (start, stop) in bands {
                let (cpu_packed, cpu_scales) = quantize_on(false, &rows(&w, start, stop));
                assert!(
                    raw(&rows(&gpu_packed, start, stop)) == raw(&cpu_packed),
                    "{label}: GPU packed weights in rows {start}..{stop} differ from the CPU quantizer"
                );
                assert!(
                    raw(&rows(&gpu_scales, start, stop)) == raw(&cpu_scales),
                    "{label}: GPU E8M0 scales in rows {start}..{stop} differ from the CPU quantizer"
                );
            }
        }
    }
}

/// mxfp4 `quantized_matmul` (transposed weight, as `QuantizedLinear` calls it)
/// on the GPU matches `x @ dequantize(w).T` computed in f32 on the CPU. Covers
/// the issue's hanging case (256x512, M = 1), its transpose, and gpt-oss-20b's
/// 2880x2880 projection, each at M = 1 (qmv) and M = 8, 64 (qmm).
#[test]
fn quantized_matmul_matches_dequantized_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(18081);
    for (out_features, in_features) in [(256, 512), (512, 256), (2880, 2880)] {
        let w = normal(&[out_features, in_features], dtype::FLOAT32);
        let (packed, scales) = quantize_on(true, &w);
        let dense = dequantize_cpu(&packed, &scales);
        for dt in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
            for m in [1, 8, 64] {
                let label = format!("qmm {out_features}x{in_features} M={m} {}", dtype_name(dt));
                let x = normal(&[1, m, in_features], dt);
                let got = {
                    let _guard = DefaultDeviceGuard::gpu();
                    let y = unsafe {
                        mlxcel_core::quantized_matmul(
                            &x,
                            &packed,
                            &scales,
                            std::ptr::null(),
                            true,
                            GROUP_SIZE,
                            BITS,
                            MODE,
                        )
                    };
                    eval_ok(&label, &y);
                    y
                };
                assert_eq!(
                    mlxcel_core::array_shape(&got),
                    vec![1, m, out_features],
                    "{label}: shape"
                );
                assert_eq!(mlxcel_core::array_dtype(&got), dt, "{label}: dtype");
                let want = {
                    let _guard = DefaultDeviceGuard::cpu();
                    let xf = mlxcel_core::astype(&x, dtype::FLOAT32);
                    let y = mlxcel_core::matmul(&xf, &mlxcel_core::transpose(&dense));
                    eval_ok(&format!("{label} reference"), &y);
                    y
                };
                let err = relative_error(&got, &want);
                assert!(
                    err <= tolerance(dt),
                    "{label}: relative error {err} exceeds {}",
                    tolerance(dt)
                );
            }
        }
    }
}

/// Per-expert dense reference for `gather_qmm`: row `i` of `x` (shape
/// `[R, 1, K]`) times expert `indices[i]` of `dense` (`[E, N, K]`), on the CPU.
fn gather_reference(x: &MlxArray, dense: &MlxArray, indices: &[i32]) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    let idx = mlxcel_core::from_slice_i32(indices, &[indices.len() as i32]);
    let picked = mlxcel_core::take(dense, &idx, 0);
    let xf = mlxcel_core::astype(x, dtype::FLOAT32);
    let y = mlxcel_core::matmul(&xf, &mlxcel_core::swap_axes(&picked, -1, -2));
    eval_ok("gather reference", &y);
    y
}

/// mxfp4 `gather_qmm` on the GPU matches a per-expert dequantized reference,
/// in both shapes `SwitchLinear::forward` passes: unsorted (`x` as
/// `[T, 1, 1, K]`, indices `[T, top_k]`) and sorted (`x` as `[T*top_k, 1, K]`,
/// indices sorted by expert). The reduction width is gpt-oss-20b's (2880, 90
/// groups of 32) with its top 4 routing; the output width and expert count are
/// cut to 256 and 16 to keep the CPU reference small. The full gpt-oss shape
/// runs end to end on the real checkpoint.
#[test]
fn gather_qmm_matches_per_expert_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    mlxcel_core::random_seed(18082);
    let (experts, n, k, top_k) = (16, 256, 2880, 4);
    let w = normal(&[experts, n, k], dtype::FLOAT32);
    let (packed, scales) = quantize_on(true, &w);
    let dense = dequantize_cpu(&packed, &scales);

    for dt in [dtype::FLOAT32, dtype::BFLOAT16] {
        for tokens in [1, 8] {
            // Deterministic, repeated and out-of-order expert choices.
            let indices: Vec<i32> = (0..tokens * top_k).map(|i| (i * 7 + 3) % experts).collect();
            let x = normal(&[tokens, 1, 1, k], dt);

            for sorted in [false, true] {
                let label = format!(
                    "gather_qmm T={tokens} top_k={top_k} sorted={sorted} {}",
                    dtype_name(dt)
                );
                let rows = tokens * top_k;
                // Sorted: one row per (token, slot), ordered by expert, the
                // way SwitchGLU's sorted path flattens and reorders.
                let (order, x_in, idx_shape): (Vec<i32>, UniquePtr<MlxArray>, Vec<i32>) = if sorted
                {
                    let mut order: Vec<i32> = (0..rows).collect();
                    order.sort_by_key(|&r| indices[r as usize]);
                    let _guard = DefaultDeviceGuard::cpu();
                    let token_of: Vec<i32> = order.iter().map(|r| r / top_k).collect();
                    let tok = mlxcel_core::from_slice_i32(&token_of, &[rows]);
                    let flat = mlxcel_core::reshape(&x, &[tokens, 1, k]);
                    let x_sorted = mlxcel_core::take(&flat, &tok, 0);
                    eval_ok("sorted input", &x_sorted);
                    (order, x_sorted, vec![rows])
                } else {
                    (
                        (0..rows).collect(),
                        mlxcel_core::reshape(&x, &[tokens, 1, 1, k]),
                        vec![tokens, top_k],
                    )
                };
                let expert_of: Vec<i32> = order.iter().map(|&r| indices[r as usize]).collect();
                let rhs = {
                    let flat = if sorted { &expert_of } else { &indices };
                    let _guard = DefaultDeviceGuard::cpu();
                    let a = mlxcel_core::from_slice_i32(flat, &idx_shape);
                    let a = mlxcel_core::astype(&a, dtype::UINT32);
                    eval_ok("indices", &a);
                    a
                };

                let got = {
                    let _guard = DefaultDeviceGuard::gpu();
                    let y = unsafe {
                        mlxcel_core::gather_qmm(
                            &x_in,
                            &packed,
                            &scales,
                            std::ptr::null(),
                            std::ptr::null(),
                            &*rhs as *const MlxArray,
                            true,
                            GROUP_SIZE,
                            BITS,
                            sorted,
                            MODE,
                        )
                    };
                    eval_ok(&label, &y);
                    y
                };
                let got = {
                    let _guard = DefaultDeviceGuard::cpu();
                    let flat = mlxcel_core::reshape(&got, &[rows, 1, n]);
                    eval_ok("reshape", &flat);
                    flat
                };

                // Reference rows in the same order as `got`.
                let ref_x = {
                    let _guard = DefaultDeviceGuard::cpu();
                    let token_of: Vec<i32> = order.iter().map(|r| r / top_k).collect();
                    let tok = mlxcel_core::from_slice_i32(&token_of, &[rows]);
                    let flat = mlxcel_core::reshape(&x_in, &[-1, 1, k]);
                    let flat = if sorted {
                        flat
                    } else {
                        mlxcel_core::take(&flat, &tok, 0)
                    };
                    eval_ok("reference input", &flat);
                    flat
                };
                let want = gather_reference(&ref_x, &dense, &expert_of);
                let err = relative_error(&got, &want);
                assert!(
                    err <= tolerance(dt),
                    "{label}: relative error {err} exceeds {}",
                    tolerance(dt)
                );
            }
        }
    }
}

const GATHER_WARP_ENV: &str = "MLX_ROCM_GATHER_QMV_USE_WARP";

/// Sets `MLX_ROCM_GATHER_QMV_USE_WARP` (`None` removes it) and restores the
/// value it found when dropped. `GatherQMM::eval_gpu` reads the variable on
/// every call.
struct GatherWarpEnv(Option<std::ffi::OsString>);

impl GatherWarpEnv {
    fn new() -> Self {
        Self(std::env::var_os(GATHER_WARP_ENV))
    }

    fn set(&self, value: Option<&str>) {
        // SAFETY: `set_var` and `remove_var` mutate the process-global
        // environment. The caller holds `lock_default_device` while it sets
        // the variable and while MLX reads it during evaluation, and nothing
        // else in this binary touches the environment.
        unsafe {
            match value {
                Some(v) => std::env::set_var(GATHER_WARP_ENV, v),
                None => std::env::remove_var(GATHER_WARP_ENV),
            }
        }
    }
}

impl Drop for GatherWarpEnv {
    fn drop(&mut self) {
        let prev = self.0.take();
        self.set(prev.as_deref().and_then(|v| v.to_str()));
    }
}

/// `||got - want|| / ||want||`, computed on the CPU in f32.
fn relative_l2(got: &MlxArray, want: &MlxArray) -> f32 {
    let _guard = DefaultDeviceGuard::cpu();
    let got = mlxcel_core::astype(got, dtype::FLOAT32);
    let want = mlxcel_core::astype(want, dtype::FLOAT32);
    let diff = mlxcel_core::sum_all(&mlxcel_core::square(&mlxcel_core::subtract(&got, &want)));
    let norm = mlxcel_core::sum_all(&mlxcel_core::square(&want));
    eval_ok("error metric", &diff);
    eval_ok("error metric", &norm);
    let (diff, norm) = (mlxcel_core::item_f32(&diff), mlxcel_core::item_f32(&norm));
    assert!(norm > 0.0, "reference is all zeros");
    (diff / norm).sqrt()
}

/// mxfp4 `gather_qmm` on the GPU, returned as `[rows, 1, n]` in output order.
fn gather_qmm_gpu(
    label: &str,
    x: &MlxArray,
    packed: &MlxArray,
    scales: &MlxArray,
    rhs: &MlxArray,
    sorted: bool,
    n: i32,
) -> UniquePtr<MlxArray> {
    let y = {
        let _guard = DefaultDeviceGuard::gpu();
        // SAFETY: `rhs` outlives the call, and null is the bridge's documented
        // value for absent biases and lhs indices.
        let y = unsafe {
            mlxcel_core::gather_qmm(
                x,
                packed,
                scales,
                std::ptr::null(),
                std::ptr::null(),
                rhs as *const MlxArray,
                true,
                GROUP_SIZE,
                BITS,
                sorted,
                MODE,
            )
        };
        eval_ok(label, &y);
        y
    };
    let _guard = DefaultDeviceGuard::cpu();
    let flat = mlxcel_core::reshape(&y, &[-1, 1, n]);
    eval_ok("reshape", &flat);
    flat
}

/// bf16 mxfp4 decode-shaped `gather_qmm` takes the warp-shared gather kernel's
/// mxfp4 word path (issue #2178) and stays as accurate as the per-row
/// `gather_qmv_kernel` it replaces (`MLX_ROCM_GATHER_QMV_USE_WARP=0`).
///
/// gpt-oss-20b's expert layer: 32 experts, top 4, `K = 2880` (two
/// `shared_x` chunks, 2048 and 832), at `N = 2880` and at `N = 516`, whose
/// last column block is partial. Unsorted with 1 token (a decode step) and 8
/// tokens, and sorted with 16 tokens (`B = 64`, `B / E = 2`, so the sorted
/// call misses the expert-batched gate and reaches this arm). Each case
/// checks that the default path's relative L2 error against the dequantized
/// f32 reference is within 1.05 times the per-row path's and under 2e-2, that
/// two default runs agree bit for bit, and that over the `N = 2880` cases
/// the default and per-row outputs differ somewhere, which proves the
/// dispatch reached another kernel.
#[test]
fn mxfp4_warp_shared_gather_qmv_matches_per_row_and_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let env = GatherWarpEnv::new();
    mlxcel_core::random_seed(2178);
    let (experts, k, top_k) = (32, 2880, 4);
    let dt = dtype::BFLOAT16;
    let mut differing_at_full_width = 0usize;

    for n in [2880, 516] {
        let (packed, scales) = {
            let w = normal(&[experts, n, k], dtype::FLOAT32);
            quantize_on(true, &w)
        };
        let dense = dequantize_cpu(&packed, &scales);

        for (tokens, sorted) in [(1, false), (8, false), (16, true)] {
            let label = format!("mxfp4 gather_qmm N={n} T={tokens} sorted={sorted}");
            let rows = tokens * top_k;
            let indices: Vec<i32> = (0..rows).map(|i| (i * 7 + 3) % experts).collect();
            let x = normal(&[tokens, 1, 1, k], dt);
            let mut order: Vec<i32> = (0..rows).collect();
            if sorted {
                order.sort_by_key(|&r| indices[r as usize]);
            }
            let expert_of: Vec<i32> = order.iter().map(|&r| indices[r as usize]).collect();
            // One input row per output row, in output order: the sorted call's
            // input, and the reference's input for both layouts.
            let x_rows = {
                let _guard = DefaultDeviceGuard::cpu();
                let token_of: Vec<i32> = order.iter().map(|r| r / top_k).collect();
                let tok = mlxcel_core::from_slice_i32(&token_of, &[rows]);
                let flat = mlxcel_core::reshape(&x, &[tokens, 1, k]);
                let picked = mlxcel_core::take(&flat, &tok, 0);
                eval_ok("input rows", &picked);
                picked
            };
            let (x_in, rhs) = {
                let _guard = DefaultDeviceGuard::cpu();
                let (x_in, flat, shape) = if sorted {
                    (mlxcel_core::copy(&x_rows), &expert_of, vec![rows])
                } else {
                    (mlxcel_core::copy(&x), &indices, vec![tokens, top_k])
                };
                let rhs =
                    mlxcel_core::astype(&mlxcel_core::from_slice_i32(flat, &shape), dtype::UINT32);
                eval_ok("inputs", &x_in);
                eval_ok("indices", &rhs);
                (x_in, rhs)
            };

            let run = |what: &str| {
                gather_qmm_gpu(
                    &format!("{label} {what}"),
                    &x_in,
                    &packed,
                    &scales,
                    &rhs,
                    sorted,
                    n,
                )
            };
            env.set(None);
            let fast = run("default");
            let again = run("default rerun");
            env.set(Some("0"));
            let per_row = run("per-row");
            env.set(None);

            // The reference in row chunks keeps the gathered f32 experts small.
            let want = {
                let parts: Vec<UniquePtr<MlxArray>> = (0..rows)
                    .step_by(8)
                    .map(|s| {
                        let e = (s + 8).min(rows);
                        let xr = {
                            let _guard = DefaultDeviceGuard::cpu();
                            let part = mlxcel_core::copy(&mlxcel_core::slice(
                                &x_rows,
                                &[s, 0, 0],
                                &[e, 1, k],
                            ));
                            eval_ok("reference rows", &part);
                            part
                        };
                        gather_reference(&xr, &dense, &expert_of[s as usize..e as usize])
                    })
                    .collect();
                let _guard = DefaultDeviceGuard::cpu();
                let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
                let all = mlxcel_core::concatenate_many(&refs, 0);
                eval_ok("reference", &all);
                all
            };

            assert!(
                raw(&fast) == raw(&again),
                "{label}: two default runs differ"
            );
            let err_fast = relative_l2(&fast, &want);
            let err_per_row = relative_l2(&per_row, &want);
            eprintln!("{label}: default {err_fast:.4e}, per-row {err_per_row:.4e}");
            assert!(
                err_fast < 2e-2,
                "{label}: default relative error {err_fast} exceeds 2e-2"
            );
            assert!(
                err_fast <= err_per_row * 1.05,
                "{label}: default relative error {err_fast} is above 1.05 times the per-row \
                 path's {err_per_row}"
            );
            let (fast_bytes, per_row_bytes) = (raw(&fast), raw(&per_row));
            let differing = fast_bytes
                .chunks_exact(2)
                .zip(per_row_bytes.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            eprintln!("{label}: {differing} bf16 outputs differ from the per-row path");
            if n == k {
                differing_at_full_width += differing;
            }
        }
    }
    // The two kernels sum each output in another order, but over 2880 terms
    // in f32 the results rarely straddle a bf16 rounding boundary: measured,
    // the 1-token case alone can round to identical outputs. Over the three
    // full-width cases (288 rows of 2880 outputs) some must differ; none
    // differing means the default call never left the per-row kernel.
    assert!(
        differing_at_full_width > 0,
        "the default calls at N = K = {k} did not leave the per-row kernel"
    );
}
