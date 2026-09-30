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

//! A GPU `slice_update` leaves a source array the caller still holds
//! unchanged on ROCm (issue #2052).
//!
//! `SliceUpdate::eval_gpu` and `DynamicSliceUpdate::eval_gpu` may reuse the
//! source's buffer for the output ("donation") only when nothing else can read
//! the source afterwards. The ROCm overlay checked only that the data buffer
//! had one owner, so a source array still referenced by the caller (here, a
//! test variable; in mlxcel, for example a cache snapshot) was donated and the
//! update was written into it. Upstream's `array::is_donatable` also requires
//! the array itself to have one reference, which is what Metal and CUDA reach
//! through `copy_gpu`.
//!
//! Each case builds a small evaluated source, keeps it, runs one GPU update,
//! and checks both the output and that the source still holds its original
//! values. The cases cover the None reduce (the path every mlxcel KV cache
//! write takes through `ffi::slice_update`), the Sum and Max reduce kernels
//! (through the test-only `slice_update_reduce` bridge), and
//! `DynamicSliceUpdate` (through the test-only `slice_update_dynamic` bridge,
//! whose start clamping is checked too).
//! A last case drops the source before evaluating, the case donation exists
//! for, and checks the output. With the old donation check every held-source
//! case fails on gfx1151.
//!
//! ```sh
//! cargo test --features rocm --test rocm_slice_update_source -- --test-threads=1
//! ```
//!
//! Every test moves the process-global default device, so each holds
//! `streams::lock_default_device` for its whole body. Skips on any other
//! backend.

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr};

/// `reduce` codes of `mlxcel_core::slice_update_reduce`.
const SUM: i32 = 0;
const MAX: i32 = 2;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// `0..n` as an int32 array of `shape`. The array owns a fresh contiguous
/// buffer with one owner, which is what makes it a donation candidate.
fn source(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<i32> = (0..n).collect();
    let out = mlxcel_core::from_slice_i32(&data, shape);
    mlxcel_core::eval(&out);
    out
}

fn values(a: &MlxArray) -> Vec<i32> {
    mlxcel_core::array_to_raw_bytes(a)
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn update(values: &[i32], shape: &[i32]) -> UniquePtr<MlxArray> {
    let out = mlxcel_core::from_slice_i32(values, shape);
    mlxcel_core::eval(&out);
    out
}

fn assert_source_intact(label: &str, src: &MlxArray, original: &[i32]) {
    let now = values(src);
    let changed: Vec<usize> = now
        .iter()
        .zip(original)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        changed.is_empty(),
        "{label}: the GPU update wrote into the source array the caller still holds; \
         {} of {} elements changed, first at flat index {} ({} -> {})",
        changed.len(),
        original.len(),
        changed[0],
        original[changed[0]],
        now[changed[0]]
    );
}

#[test]
fn none_reduce_leaves_held_source_unchanged() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    // A KV-cache-shaped write: [batch, heads, slots, head_dim], one new row
    // at slot 5.
    let shape = [1, 2, 8, 4];
    let src = source(&shape);
    let original = values(&src);
    let upd = update(&[-1; 8], &[1, 2, 1, 4]);
    let out = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::slice_update(&src, &upd, &[0, 0, 5, 0], &[1, 2, 6, 4]);
        mlxcel_core::eval(&out);
        out
    };
    let mut want = original.clone();
    for head in 0..2 {
        for d in 0..4 {
            want[head * 32 + 5 * 4 + d] = -1;
        }
    }
    assert_eq!(values(&out), want, "slice_update output");
    assert_source_intact("slice_update (None)", &src, &original);
}

fn reduce_case(label: &str, reduce: i32, upd_values: &[i32], want_slice: &[i32]) {
    let src = source(&[16]);
    let original = values(&src);
    let upd = update(upd_values, &[4]);
    let out = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::slice_update_reduce(&src, &upd, &[3], &[7], reduce)
            .expect("slice_update_reduce builds the graph");
        mlxcel_core::eval(&out);
        out
    };
    let mut want = original.clone();
    want[3..7].copy_from_slice(want_slice);
    assert_eq!(values(&out), want, "{label} output");
    assert_source_intact(label, &src, &original);
}

#[test]
fn sum_reduce_leaves_held_source_unchanged() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    reduce_case("slice_update_add", SUM, &[1, 1, 1, 1], &[4, 5, 6, 7]);
}

#[test]
fn max_reduce_leaves_held_source_unchanged() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    reduce_case("slice_update_max", MAX, &[0, 10, 0, 10], &[3, 10, 5, 10]);
}

#[test]
fn dynamic_slice_update_leaves_held_source_unchanged() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let shape = [1, 2, 8, 4];
    let src = source(&shape);
    let original = values(&src);
    let upd = update(&[-1; 8], &[1, 2, 1, 4]);
    let start = update(&[5], &[1]);
    let out = {
        let _guard = DefaultDeviceGuard::gpu();
        let out = mlxcel_core::slice_update_dynamic(&src, &upd, &start, &[2])
            .expect("slice_update_dynamic builds the graph");
        mlxcel_core::eval(&out);
        out
    };
    let mut want = original.clone();
    for head in 0..2 {
        for d in 0..4 {
            want[head * 32 + 5 * 4 + d] = -1;
        }
    }
    assert_eq!(values(&out), want, "DynamicSliceUpdate output");
    assert_source_intact("DynamicSliceUpdate", &src, &original);
}

/// The bridge clamps a dynamic start to where the update fits, so an
/// out-of-range offset writes the last slot rather than past the buffer.
#[test]
fn dynamic_slice_update_clamps_an_out_of_range_start() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let src = source(&[16]);
    let upd = update(&[-1, -2], &[2]);
    let starts = [(1000, 14), (-5, 0)];
    for (start, landed) in starts {
        let start_arr = update(&[start], &[1]);
        let out = {
            let _guard = DefaultDeviceGuard::gpu();
            let out = mlxcel_core::slice_update_dynamic(&src, &upd, &start_arr, &[0])
                .expect("slice_update_dynamic builds the graph");
            mlxcel_core::eval(&out);
            out
        };
        let mut want: Vec<i32> = (0..16).collect();
        want[landed] = -1;
        want[landed + 1] = -2;
        assert_eq!(values(&out), want, "start {start} lands at {landed}");
    }
}

/// The donation the check still allows: the source is dropped before the
/// update is evaluated, so the output may take its buffer. Only the output
/// is observable, and it must be right.
#[test]
fn dropped_source_update_is_correct() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let shape = [1, 2, 8, 4];
    let upd = update(&[-1; 8], &[1, 2, 1, 4]);
    let (none_out, sum_out, dyn_out) = {
        let _guard = DefaultDeviceGuard::gpu();
        let none_out =
            mlxcel_core::slice_update(&source(&shape), &upd, &[0, 0, 5, 0], &[1, 2, 6, 4]);
        let sum_out = mlxcel_core::slice_update_reduce(
            &source(&shape),
            &upd,
            &[0, 0, 5, 0],
            &[1, 2, 6, 4],
            SUM,
        )
        .expect("slice_update_reduce builds the graph");
        let start = update(&[5], &[1]);
        let dyn_out = mlxcel_core::slice_update_dynamic(&source(&shape), &upd, &start, &[2])
            .expect("slice_update_dynamic builds the graph");
        mlxcel_core::eval(&none_out);
        mlxcel_core::eval(&sum_out);
        mlxcel_core::eval(&dyn_out);
        (none_out, sum_out, dyn_out)
    };
    let base: Vec<i32> = (0..64).collect();
    let mut assigned = base.clone();
    let mut summed = base.clone();
    for head in 0..2 {
        for d in 0..4 {
            let i = head * 32 + 5 * 4 + d;
            assigned[i] = -1;
            summed[i] -= 1;
        }
    }
    assert_eq!(values(&none_out), assigned, "slice_update output");
    assert_eq!(values(&sum_out), summed, "slice_update_add output");
    assert_eq!(values(&dyn_out), assigned, "DynamicSliceUpdate output");
}
