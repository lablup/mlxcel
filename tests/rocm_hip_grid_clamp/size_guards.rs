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

//! Empty and 64-bit sizes in the ROCm norm, gemv and GatherQMM launchers
//! (issue #2633, `LOCAL_FIXES.md` item 65).
//!
//! * Norms: `row_grid(0, ...)` returned `dim3(1, 0, 1)`, an invalid launch,
//!   and nothing checked for zero rows, so `rms_norm` and `layer_norm` of an
//!   empty input and their VJPs failed. They now return early, with `dw` and
//!   `db` filled with zeros (the sum over no rows). The VJPs shaped the
//!   weight-gradient temporary and the reduction plan with
//!   `static_cast<int>(n_rows)`; both types hold only 32-bit dimensions, so
//!   past `INT_MAX` rows the VJP now throws a named error instead.
//! * gemv: `gemv_impl` read row `row` at `mat + row * cols` and the batched
//!   kernels wrote at `out + batch_idx * rows`, both in `int`; they are now
//!   64-bit, as #2624 did for the gather kernels.
//! * GatherQMM: `B = out.size() / M / N` came before the only empty-output
//!   check, so `M == 0` or `N == 0` divided by zero (SIGFPE). The check now
//!   comes first, for every mode.
//!
//! The `#[ignore]`d tests need about 5 GiB each, or, for the row-count
//! guard, a 4 GiB contiguous copy on a build without the guard; run them
//! with `--ignored`.

use super::*;

const EPS: f32 = 1e-5;
const AXIS: i32 = 8;

fn f32_vec(values: &[f32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, &[values.len() as i32])
}

fn weight() -> UniquePtr<MlxArray> {
    f32_vec(&[0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 0.25, 3.0])
}

fn bias() -> UniquePtr<MlxArray> {
    f32_vec(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
}

/// Evaluates `out` on the GPU and checks its shape, and that every element
/// is zero when `zero` is set.
fn assert_evaluates(what: &str, out: &MlxArray, shape: &[i32], zero: bool) {
    if let Err(e) = mlxcel_core::try_eval(out) {
        panic!("{what}: GPU evaluation failed: {e}");
    }
    assert_eq!(mlxcel_core::array_shape(out), shape, "{what}: shape");
    if zero {
        let zeros = mlxcel_core::zeros(shape, dtype::FLOAT32);
        let same = mlxcel_core::allclose(out, &zeros, 0.0, 0.0);
        assert!(
            mlxcel_core::item_bool(&same),
            "{what}: expected zeros, got {:?}",
            mlxcel_core::array_to_raw_bytes(&mlxcel_core::astype(out, dtype::FLOAT32))
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
                .collect::<Vec<_>>()
        );
    }
}

// ---- (a) norms on zero rows ----------------------------------------------------

#[test]
fn norms_of_zero_rows_return_empty_and_zero_gradients() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let empty = [0, AXIS];
    let x = mlxcel_core::zeros(&empty, dtype::FLOAT32);
    let g = mlxcel_core::zeros(&empty, dtype::FLOAT32);
    let (w, b) = (weight(), bias());

    let rms = mlxcel_core::fast_rms_norm(&x, &w, EPS);
    assert_evaluates("rms_norm", &rms, &empty, false);
    for (wrt, shape, zero) in [(0, &empty[..], false), (1, &[AXIS][..], true)] {
        let grad =
            mlxcel_core::fast_rms_norm_vjp_for_test(&x, &w, &g, EPS, wrt).expect("the vjp builds");
        assert_evaluates(&format!("rms_norm vjp wrt {wrt}"), &grad, shape, zero);
    }

    // SAFETY: `w` and `b` are live arrays for the whole call.
    let ln = unsafe { mlxcel_core::fast_layer_norm(&x, &*w, &*b, EPS) };
    assert_evaluates("layer_norm", &ln, &empty, false);
    for (wrt, shape, zero) in [
        (0, &empty[..], false),
        (1, &[AXIS][..], true),
        (2, &[AXIS][..], true),
    ] {
        // SAFETY: `w` is a live array for the whole call.
        let grad = unsafe { mlxcel_core::fast_layer_norm_vjp_for_test(&x, &*w, &b, &g, EPS, wrt) }
            .expect("the vjp builds");
        assert_evaluates(&format!("layer_norm vjp wrt {wrt}"), &grad, shape, zero);
    }
    // No weight: the bias gradient alone.
    let grad =
        unsafe { mlxcel_core::fast_layer_norm_vjp_for_test(&x, std::ptr::null(), &b, &g, EPS, 2) }
            .expect("the vjp builds");
    assert_evaluates("layer_norm vjp wrt 2 without weight", &grad, &[AXIS], true);
}

/// `2 * (2^30 + 1)` rows of one element, past `INT_MAX`. The inputs are
/// broadcast scalars, so with the guard the VJP throws before it copies
/// anything; without it the VJP first makes a 4 GiB contiguous copy of each
/// and then sizes `gw_temp` from the wrapped row count.
#[test]
#[ignore = "without the guard, copies two 4 GiB f16 inputs; run with --ignored on a device with ~10 GiB free"]
fn norm_vjp_past_int_max_rows_throws() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let shape = [2, (1 << 30) + 1, 1];
    let one = mlxcel_core::astype(&f32_vec(&[1.0]), dtype::FLOAT16);
    let x = mlxcel_core::broadcast_to(&one, &shape);
    let g = mlxcel_core::broadcast_to(&one, &shape);
    let w = mlxcel_core::astype(&f32_vec(&[2.0]), dtype::FLOAT16);
    let b = mlxcel_core::astype(&f32_vec(&[0.5]), dtype::FLOAT16);
    let rms = mlxcel_core::fast_rms_norm_vjp_for_test(&x, &w, &g, EPS, 1).expect("the vjp builds");
    // SAFETY: `w` is a live array for the whole call.
    let ln = unsafe { mlxcel_core::fast_layer_norm_vjp_for_test(&x, &*w, &b, &g, EPS, 2) }
        .expect("the vjp builds");
    for (what, grad) in [("rms_norm vjp dw", rms), ("layer_norm vjp db", ln)] {
        match mlxcel_core::try_eval(&grad) {
            Ok(()) => panic!("{what}: evaluated past INT_MAX rows instead of throwing"),
            Err(e) => assert!(
                e.what().contains("INT_MAX"),
                "{what}: unexpected error: {}",
                e.what()
            ),
        }
    }
}

// ---- (b) gemv past 2^31 elements ---------------------------------------------

/// `[len]` f16 of `i % modulus`, built on the GPU.
fn cycle(len: i32, modulus: f32) -> UniquePtr<MlxArray> {
    let i = mlxcel_core::arange_f32(0.0, len as f32, 1.0);
    let m = mlxcel_core::remainder(&i, &f32_vec(&[modulus]));
    mlxcel_core::astype(&m, dtype::FLOAT16)
}

/// `[rows, cols]` f16 with element `[r, c] = r % 13 + c % 5`, materialized on
/// the GPU (no host copy of the large matrix).
fn row_col_matrix(rows: i32, cols: i32) -> UniquePtr<MlxArray> {
    let r = mlxcel_core::reshape(&cycle(rows, 13.0), &[rows, 1]);
    let c = mlxcel_core::reshape(&cycle(cols, 5.0), &[1, cols]);
    let m = mlxcel_core::add(&r, &c);
    mlxcel_core::try_eval(&m).expect("the matrix builds");
    m
}

/// `sum(c % 5 for c in 0..cols)`.
fn col_cycle_sum(cols: i32) -> f32 {
    (0..cols).map(|c| (c % 5) as f32).sum()
}

fn host_f32(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, dtype::FLOAT32);
    mlxcel_core::try_eval(&a).expect("reads back");
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

/// `v @ W.T` with `W` of `65_537 x 32_768` (2^31 + 2^15 elements, 4 GiB of
/// f16): the M == 1 gemv reads row 65_536 at element 2^31.
#[test]
#[ignore = "allocates a 4 GiB f16 matrix; run with --ignored on a device with ~6 GiB free"]
fn gemv_past_i32_matrix_elements_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let (rows, cols) = (65_537, 32_768);
    let w = row_col_matrix(rows, cols);
    // v = 1/64 everywhere, so out[r] = (cols * (r % 13) + sum(c % 5)) / 64.
    let v = mlxcel_core::astype(
        &mlxcel_core::broadcast_to(&f32_vec(&[1.0 / 64.0]), &[1, cols]),
        dtype::FLOAT16,
    );
    let out = mlxcel_core::matmul(&v, &mlxcel_core::transpose(&w));
    if let Err(e) = mlxcel_core::try_eval(&out) {
        panic!("gemv: GPU launch failed: {e}");
    }
    assert_eq!(mlxcel_core::array_shape(&out), [1, rows]);
    let got = host_f32(&out);
    let tail = col_cycle_sum(cols);
    for r in [0, 1, 65_535, 65_536, rows - 1] {
        let want = (cols as f32 * (r % 13) as f32 + tail) / 64.0;
        let g = got[r as usize];
        assert!(
            (g - want).abs() <= 4.0,
            "gemv row {r}: got {g}, expected {want}"
        );
    }
}

/// `A @ v[b]` for `65_535` batches with `A` (`32_800 x 32`) broadcast over
/// the batch: the batched gemv writes batch `b` at `b * 32_800`, past
/// `INT_MAX` from batch 65_473 on. The output is 2^31 + 2^21 f16 elements.
#[test]
#[ignore = "allocates a 4.3 GB f16 output; run with --ignored on a device with ~6 GiB free"]
fn batched_gemv_past_i32_output_elements_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let (batch, rows, cols) = (65_535, 32_800, 32);
    let a = mlxcel_core::broadcast_to(&row_col_matrix(rows, cols), &[batch, rows, cols]);
    // v[b, c] = (b % 7 + 1) / 64.
    let scale = mlxcel_core::multiply(
        &mlxcel_core::add(
            &cycle(batch, 7.0),
            &mlxcel_core::astype(&f32_vec(&[1.0]), dtype::FLOAT16),
        ),
        &mlxcel_core::astype(&f32_vec(&[1.0 / 64.0]), dtype::FLOAT16),
    );
    let v = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&scale, &[batch, 1, 1]),
        &[batch, cols, 1],
    );
    let v = mlxcel_core::contiguous(&v, false);
    let out = mlxcel_core::matmul(&a, &v);
    if let Err(e) = mlxcel_core::try_eval(&out) {
        panic!("batched gemv: GPU launch failed: {e}");
    }
    assert_eq!(mlxcel_core::array_shape(&out), [batch, rows, 1]);
    let tail = col_cycle_sum(cols);
    for b in [0, 65_472, 65_473, batch - 1] {
        let got = host_f32(&mlxcel_core::slice(&out, &[b, 0, 0], &[b + 1, rows, 1]));
        let s = (b % 7 + 1) as f32 / 64.0;
        for r in [0, 1, 12, rows - 1] {
            let want = s * (cols as f32 * (r % 13) as f32 + tail);
            let g = got[r as usize];
            assert!(
                (g - want).abs() <= 0.05,
                "batched gemv batch {b} row {r}: got {g}, expected {want}"
            );
        }
    }
}

// ---- (c) GatherQMM with an empty output -----------------------------------------

/// `gather_qmm` over 2 experts with `m` rows per batch entry and `n` outputs,
/// from zero-filled packed weights (a valid all-zero quantization).
fn empty_gather(mode: &'static str, group_size: i32, m: i32, n: i32) -> UniquePtr<MlxArray> {
    let (experts, k, batch) = (2, 64, 4);
    let bits = 4;
    let w = mlxcel_core::zeros(&[experts, n, k * bits / 32], dtype::UINT32);
    let (scales, biases) = if mode == "affine" {
        (
            mlxcel_core::zeros(&[experts, n, k / group_size], dtype::FLOAT16),
            Some(mlxcel_core::zeros(
                &[experts, n, k / group_size],
                dtype::FLOAT16,
            )),
        )
    } else {
        (
            mlxcel_core::zeros(&[experts, n, k / group_size], dtype::UINT8),
            None,
        )
    };
    let x = mlxcel_core::zeros(&[batch, m, k], dtype::FLOAT16);
    let rhs = mlxcel_core::astype(
        &mlxcel_core::from_slice_i32(&[0, 1, 1, 0], &[batch]),
        dtype::UINT32,
    );
    let biases_ptr = biases
        .as_ref()
        .map_or(std::ptr::null(), |b| &**b as *const MlxArray);
    // SAFETY: every array outlives the call; null pointers select no biases,
    // the implicit lhs indices and no global scale.
    unsafe {
        mlxcel_core::gather_qmm(
            &x,
            &w,
            &scales,
            biases_ptr,
            std::ptr::null(),
            &*rhs,
            true,
            group_size,
            bits,
            false,
            mode,
            std::ptr::null(),
        )
    }
}

fn assert_empty_gather(mode: &'static str, group_size: i32) {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    for (m, n) in [(0, 64), (1, 0)] {
        let out = empty_gather(mode, group_size, m, n);
        let what = format!("{mode} gather_qmm M = {m}, N = {n}");
        assert_evaluates(&what, &out, &[4, m, n], false);
    }
}

#[test]
fn gather_qmm_affine_with_empty_output_returns_empty() {
    assert_empty_gather("affine", 64);
}

#[test]
fn gather_qmm_nvfp4_with_empty_output_returns_empty() {
    assert_empty_gather("nvfp4", 16);
}
