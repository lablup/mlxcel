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

//! A reduce-type `SliceUpdate` (Sum, Prod, Max, Min) applies every element of
//! a large update on ROCm (issue #2050).
//!
//! The ROCm launch caps `slice_update_op_kernel` at 65535 blocks of 256
//! threads, each thread covering `NWORK` elements, where `NWORK` is 4, 2 or 1
//! by what divides the innermost update dimension. Before the kernel became
//! grid-stride, everything past 65535 x 256 x `NWORK` elements (16,776,960,
//! 33,553,920 or 67,107,840) kept the input values with no error. Each case
//! here sits just above one of those limits and compares the GPU result with
//! the same op on the CPU device, exactly, on int32 data. The above-threshold
//! cases fail with the kernel change reverted (checked on gfx1151).
//!
//! The arrays are large: the NWORK=4 case holds four arrays of about 67M
//! int32 elements (source, update, and the two results), about 1.1 GB, so run
//! the binary on its own and serially:
//!
//! ```sh
//! cargo test --features rocm --test rocm_slice_update_reduce -- --test-threads=1
//! ```
//!
//! Every test moves the process-global default device, so each holds
//! `streams::lock_default_device` for its whole body.
//!
//! Skips on any other backend.

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// `reduce` codes of `mlxcel_core::slice_update_reduce`.
const SUM: i32 = 0;
const PROD: i32 = 1;
const MAX: i32 = 2;
const MIN: i32 = 3;

/// 65535 blocks x 256 threads: the most `NWORK`-element chunks the clamped
/// grid covers in one pass.
const CLAMPED_THREADS: i64 = 65535 * 256;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// `[start, stop)` as an int32 array, built and evaluated on the GPU.
fn arange(start: i32, stop: i32) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    let out = mlxcel_core::arange_i32(start, stop, 1);
    mlxcel_core::eval(&out);
    out
}

fn reduce_on(
    gpu: bool,
    src: &MlxArray,
    update: &MlxArray,
    starts: &[i32],
    stops: &[i32],
    reduce: i32,
) -> UniquePtr<MlxArray> {
    let _guard = if gpu {
        DefaultDeviceGuard::gpu()
    } else {
        DefaultDeviceGuard::cpu()
    };
    let out = mlxcel_core::slice_update_reduce(src, update, starts, stops, reduce)
        .expect("slice_update_reduce builds the graph");
    mlxcel_core::eval(&out);
    out
}

/// First index where two int32 arrays differ, and how many differ. Only run
/// after `array_equal` has already failed, to make the failure readable.
fn first_mismatch(got: &MlxArray, want: &MlxArray) -> (usize, i32, i32, usize) {
    let got = mlxcel_core::array_to_raw_bytes(got);
    let want = mlxcel_core::array_to_raw_bytes(want);
    assert_eq!(got.len(), want.len(), "result sizes differ");
    let as_i32 = |b: &[u8]| i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let mut first = None;
    let mut count = 0usize;
    for (i, (g, w)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        if g != w {
            count += 1;
            first.get_or_insert((i, as_i32(g), as_i32(w)));
        }
    }
    let (i, g, w) = first.expect("array_equal reported a difference");
    (i, g, w, count)
}

/// Run the reduce slice update on the GPU and on the CPU and require the two
/// results to be identical.
fn assert_gpu_matches_cpu(
    label: &str,
    src: &MlxArray,
    update: &MlxArray,
    starts: &[i32],
    stops: &[i32],
    reduce: i32,
) {
    // The GPU op runs first on the same `src` the CPU reference then reads:
    // `SliceUpdate::eval_gpu` must not write into a source the caller still
    // holds (issue #2052, `tests/rocm_slice_update_source.rs`).
    let gpu = reduce_on(true, src, update, starts, stops, reduce);
    let cpu = reduce_on(false, src, update, starts, stops, reduce);
    let equal = {
        let _guard = DefaultDeviceGuard::cpu();
        let eq = mlxcel_core::array_equal(&gpu, &cpu, false);
        mlxcel_core::item_bool(&eq)
    };
    if !equal {
        let (index, got, want, count) = first_mismatch(&gpu, &cpu);
        panic!(
            "{label}: GPU differs from CPU at {count} of {} elements; first at flat index {index}: GPU {got}, CPU {want}",
            mlxcel_core::array_size(&cpu)
        );
    }
}

/// Sum over a contiguous 1-D update of `n` elements written at offset 3 of a
/// slightly larger source (a full-size slice would take MLX's elementwise
/// shortcut instead of `SliceUpdate`). `n % 4` picks the kernel's `NWORK`.
fn contiguous_sum(n: i32, nwork: i64) {
    assert!(
        i64::from(n) > CLAMPED_THREADS * nwork,
        "the case must exceed the clamped grid's reach for NWORK={nwork}"
    );
    let src = arange(0, n + 8);
    let update = arange(1, n + 1);
    assert_gpu_matches_cpu(
        &format!("contiguous Sum, {n} elements, NWORK={nwork}"),
        &src,
        &update,
        &[3],
        &[n + 3],
        SUM,
    );
}

#[test]
fn contiguous_sum_nwork1_above_clamp() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    // Odd: NWORK=1.
    contiguous_sum(16_777_217, 1);
}

#[test]
fn contiguous_sum_nwork2_above_clamp() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    // 2 x an odd number: NWORK=2.
    contiguous_sum(33_554_434, 2);
}

#[test]
fn contiguous_sum_nwork4_above_clamp() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    // 4 x an odd number: NWORK=4.
    contiguous_sum(67_108_868, 4);
}

/// Max into columns 1..4098 of a `[4099, 4099]` output: the output rows are
/// strided (non-contiguous), so each chunk's output index comes from
/// `elem_to_loc`, and the innermost dimension 4097 is odd (NWORK=1).
#[test]
fn strided_max_above_clamp() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    const SIDE: i32 = 4099;
    const COLS: i32 = SIDE - 2;
    let n = SIDE * COLS;
    assert!(i64::from(n) > CLAMPED_THREADS);
    // Negative source, positive update: every applied element changes.
    let src = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::reshape(&arange(-SIDE * SIDE, 0), &[SIDE, SIDE]);
        mlxcel_core::eval(&out);
        out
    };
    let update = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::reshape(&arange(1, n + 1), &[SIDE, COLS]);
        mlxcel_core::eval(&out);
        out
    };
    assert_gpu_matches_cpu(
        &format!("strided Max, {n} elements, NWORK=1"),
        &src,
        &update,
        &[0, 1],
        &[SIDE, SIDE - 1],
        MAX,
    );
}

/// Sum through a transposed (non-row-contiguous) `[4099, 4097]` update into
/// columns 1..4098 of a `[4099, 4099]` output: both the output and the update
/// index come from `elem_to_loc` at the top of every chunk (NWORK=1).
#[test]
fn transposed_update_sum_above_clamp() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    const SIDE: i32 = 4099;
    const COLS: i32 = SIDE - 2;
    let n = SIDE * COLS;
    assert!(i64::from(n) > CLAMPED_THREADS);
    let src = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::reshape(&arange(0, SIDE * SIDE), &[SIDE, SIDE]);
        mlxcel_core::eval(&out);
        out
    };
    // `[COLS, SIDE]` transposed to `[SIDE, COLS]`: a strided view, not a copy.
    let update = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::transpose(&mlxcel_core::reshape(&arange(1, n + 1), &[COLS, SIDE]));
        mlxcel_core::eval(&out);
        out
    };
    assert_gpu_matches_cpu(
        &format!("transposed-update Sum, {n} elements, NWORK=1"),
        &src,
        &update,
        &[0, 1],
        &[SIDE, SIDE - 1],
        SUM,
    );
}

/// Sum with a scalar update broadcast over 16,777,217 elements: the update
/// index stays 0 on every grid-stride iteration.
#[test]
fn scalar_sum_above_clamp() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let n: i32 = 16_777_217;
    assert!(i64::from(n) > CLAMPED_THREADS);
    let src = arange(0, n + 8);
    let update = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::full_f32(&[], 5.0, dtype::INT32);
        mlxcel_core::eval(&out);
        out
    };
    assert_gpu_matches_cpu(
        &format!("scalar Sum, {n} elements, NWORK=1"),
        &src,
        &update,
        &[3],
        &[n + 3],
        SUM,
    );
}

/// Below the clamp, where one pass of the grid already covered the update:
/// every reduce op, on a strided 2-D slice (NWORK=4, 2 and 1 by the
/// innermost width), behaves exactly as before.
#[test]
fn every_reduce_below_clamp() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let src = {
        let _guard = DefaultDeviceGuard::gpu();
        // Values 1..=3 with sign changes, so Prod stays small and Max and Min
        // both keep some source and some update values.
        let base = mlxcel_core::reshape(&arange(0, 64 * 64), &[64, 64]);
        let three = mlxcel_core::full_f32(&[], 3.0, dtype::INT32);
        let two = mlxcel_core::full_f32(&[], 2.0, dtype::INT32);
        let out = mlxcel_core::subtract(&mlxcel_core::remainder(&base, &three), &two);
        mlxcel_core::eval(&out);
        out
    };
    for (width, nwork) in [(8, 4), (6, 2), (5, 1)] {
        let update = {
            let _guard = DefaultDeviceGuard::gpu();
            let base = mlxcel_core::reshape(&arange(0, 60 * width), &[60, width]);
            let five = mlxcel_core::full_f32(&[], 5.0, dtype::INT32);
            let two = mlxcel_core::full_f32(&[], 2.0, dtype::INT32);
            let out = mlxcel_core::subtract(&mlxcel_core::remainder(&base, &five), &two);
            mlxcel_core::eval(&out);
            out
        };
        for (reduce, name) in [(SUM, "Sum"), (PROD, "Prod"), (MAX, "Max"), (MIN, "Min")] {
            assert_gpu_matches_cpu(
                &format!("strided {name} below the clamp, width {width}, NWORK={nwork}"),
                &src,
                &update,
                &[2, 7],
                &[62, 7 + width],
                reduce,
            );
        }
    }
}

/// A `reduce` code outside 0..=3 is an error naming the valid range, not a
/// silent None-reduce update.
#[test]
fn unknown_reduce_code_is_an_error() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let src = arange(0, 16);
    let update = arange(0, 4);
    let msg = mlxcel_core::slice_update_reduce(&src, &update, &[2], &[6], 4)
        .err()
        .expect("reduce code 4 must be rejected")
        .to_string();
    assert!(
        msg.contains("0 (Sum), 1 (Prod), 2 (Max) or 3 (Min)") && msg.contains("got 4"),
        "the error must name the valid range, got: {msg}"
    );
}
