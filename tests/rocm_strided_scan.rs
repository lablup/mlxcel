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

//! A scan along a non-last axis launches the right number of blocks on ROCm
//! (issue #1809, `patches-rocm/LOCAL_FIXES.md` item 22).
//!
//! `Scan::eval_gpu` sizes the `strided_scan` grid with `get_2d_grid_dims(shape,
//! strides, axis_size * stride)`. The ROCm overlay carried an older copy of that
//! helper which dropped a dimension from the divisor only when the dimension
//! divided it whole, and never divided the grid by what was left. For
//! `[1, 48, 24, 24]` scanned along axis 2 (the SSD `segsum` of
//! `granite-4.0-h-tiny` over a 24-token chunk) it returned 576 blocks instead
//! of 48, and the extra blocks read and wrote past the end of the array: the
//! correctness matrix's width-8 trace of that checkpoint, and of
//! `NVIDIA-Nemotron-3-Nano-30B-A3B`, ended in a GPU memory fault inside
//! `strided_scan`.
//!
//! Each case compares the GPU scan with the CPU stream exactly. The inputs are
//! small integers stored as f32, so every partial sum is exact in f32 and the
//! two devices must agree bit for bit whatever their summation order. The
//! second case is sized so the old grid overshoots by 16x on a 25 MB array,
//! which faults rather than landing in mapped memory; it fails with the
//! header change reverted (checked on gfx1151).
//!
//! ```sh
//! cargo test --features rocm --test rocm_strided_scan -- --test-threads=1
//! ```
//!
//! Every test moves the process-global default device, so each holds
//! `streams::lock_default_device` for its whole body. Skips on any other
//! backend.

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr};

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// Small integers in `[-3, 3]` as f32, so every prefix sum is exact.
fn small_ints(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 7 - 3) as f32).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn cumsum_on(
    gpu: bool,
    x: &MlxArray,
    axis: i32,
    reverse: bool,
    inclusive: bool,
) -> UniquePtr<MlxArray> {
    let _guard = if gpu {
        DefaultDeviceGuard::gpu()
    } else {
        DefaultDeviceGuard::cpu()
    };
    let out = mlxcel_core::cumsum(x, axis, reverse, inclusive);
    mlxcel_core::eval(&out);
    out
}

fn assert_scan_matches_cpu(shape: &[i32], axis: i32) {
    let x = small_ints(shape);
    for (reverse, inclusive) in [(false, true), (true, true), (false, false), (true, false)] {
        let cpu = cumsum_on(false, &x, axis, reverse, inclusive);
        let gpu = cumsum_on(true, &x, axis, reverse, inclusive);
        let equal = {
            let _guard = DefaultDeviceGuard::cpu();
            let eq = mlxcel_core::array_equal(&gpu, &cpu, false);
            mlxcel_core::item_bool(&eq)
        };
        assert!(
            equal,
            "cumsum of {shape:?} along axis {axis} (reverse={reverse}, inclusive={inclusive}) differs between GPU and CPU"
        );
    }
}

/// The SSD `segsum` shape that faulted in the matrix: 48 SSM heads, a 24-token
/// chunk, scanned along the second-to-last axis (stride 24).
#[test]
fn ssd_segsum_shape_matches_cpu() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_scan_matches_cpu(&[1, 48, 24, 24], 2);
}

/// A leading dimension that divides part of the divisor (64 of 1024) and a
/// next one that does not divide the rest (96 against 16). The old helper
/// returned 98304 blocks for the 6144 this needs.
#[test]
fn partially_dividing_shape_matches_cpu() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_scan_matches_cpu(&[64, 96, 32, 32], 2);
}

/// Shapes the old helper already sized correctly, kept so the fix is not
/// traded for a regression on them: whole-dimension divisors, and a stride
/// that is not a multiple of the 32-wide tile.
#[test]
fn evenly_dividing_shapes_still_match_cpu() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_scan_matches_cpu(&[1, 48, 8, 8], 2);
    assert_scan_matches_cpu(&[1, 48, 16, 16], 2);
    assert_scan_matches_cpu(&[3, 257, 45], 1);
    assert_scan_matches_cpu(&[1, 8, 48], 1);
}
