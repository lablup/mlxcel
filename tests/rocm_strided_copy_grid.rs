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

//! Strided copies of 2^32 elements or more on ROCm (issue #2184,
//! `patches-rocm/LOCAL_FIXES.md` item 42).
//!
//! The overlay's strided copy kernels (`copy_g_byval`, `copy_gg_byval`,
//! `copy_gg_dynamic_nd`, `copy_gg_dynamic`) launched one thread per element.
//! HIP refuses a launch of 2^32 threads or more (`hipErrorInvalidConfiguration`),
//! so every strided copy that large failed, and the dynamic kernels sized their
//! grid by the whole destination rather than by the copied region, so even a
//! one-block `DynamicSliceUpdate` into such a slab failed. The grid is now
//! capped at 65535 blocks of 256 threads, as `copy_contiguous` already was,
//! the kernels loop over it past the cap, and the dynamic launch is sized by
//! the copied shape.
//!
//! Each op below picks its kernel through `copy_gpu_inplace`:
//! `contiguous` of a strided view is `copy_g_byval`, `concatenate` copies each
//! input into a slice of the output with `copy_gg_byval`, and
//! `slice_update_dynamic` writes through `copy_gg_dynamic` (4-D update) or
//! `copy_gg_dynamic_nd` (an update that collapses to one dimension).
//!
//! The inputs are `as_strided` views of a small buffer with every stride 1,
//! so element `[r, j, k, l]` is `buf[r + j + k + l]`: a different value per
//! position that the host can recompute for any row without materializing
//! the input, and dimensions that `collapse_contiguous_dims` cannot merge.
//! Values are small integers, exact in f16.
//!
//! The fast tests copy 1300 rows (42.6M elements, about 2.5 passes of the
//! capped grid) and compare every element with a reference computed on the
//! host; with the loop cut to one pass they fail. The `#[ignore]`d ones build
//! 131,073 rows of `[32, 8, 128]` (2^32 + 32,768 elements, the shape of the
//! #2153 paged slab), about 8 GiB of f16 each, one at a time, and check rows
//! on both sides of 2^31 and 2^32 and the last ones. Before the fix every one
//! of them failed in `try_eval` with `hipErrorInvalidConfiguration` ("invalid
//! configuration argument").
//!
//! ```sh
//! cargo test --features rocm --test rocm_strided_copy_grid -- --test-threads=1
//! cargo test --features rocm --test rocm_strided_copy_grid -- --ignored --test-threads=1
//! ```
//!
//! Every test moves the process-global default device, so each holds
//! `streams::lock_default_device` for its whole body. Skips on any other
//! backend.

#![cfg(feature = "rocm")]

use mlxcel_core::dtype;
use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr};

/// One row is one block of the paged KV slab: `[PAGE, HKV, DIM]`.
const INNER: [i32; 3] = [32, 8, 128];
const ROW: usize = 32 * 8 * 128;
/// Rows in the fast tests: 1300 * 32768 = 42.6M elements, about 2.5 passes of
/// a 65535 x 256 grid.
const FAST_ROWS: i32 = 1300;
/// Rows in the large tests: 2^17 + 1 rows of 2^15 elements, 2^32 + 2^15.
const BIG_ROWS: i32 = 131_073;
/// Values cycle below 2048, the last integer range f16 holds exactly.
const MODULUS: usize = 2039;

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn shape(rows: i32) -> [i32; 4] {
    [rows, INNER[0], INNER[1], INNER[2]]
}

/// Value of the strided view at row `r`, flat inner index `i`.
fn view_value(r: usize, i: usize) -> f32 {
    let (j, k, l) = (i / 1024, (i / 128) % 8, i % 128);
    ((r + j + k + l + 1) % MODULUS) as f32
}

/// `[rows, 32, 8, 128]` f16 whose element `[r, j, k, l]` is `buf[r + j + k + l]`
/// and `buf[i] = (i + 1) % MODULUS`: no storage beyond `rows + 165` elements.
fn strided_view(rows: i32) -> UniquePtr<MlxArray> {
    let len = rows as usize + INNER.iter().map(|d| *d as usize - 1).sum::<usize>() + 1;
    let data: Vec<f32> = (0..len).map(|i| ((i + 1) % MODULUS) as f32).collect();
    let buf = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&data, &[len as i32]),
        dtype::FLOAT16,
    );
    mlxcel_core::as_strided(&buf, &shape(rows), &[1, 1, 1, 1], 0)
}

/// One contiguous `[1, 32, 8, 128]` f16 block with values unlike the view's.
fn block() -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..ROW).map(|i| -(((i * 7) % 1000) as f32) - 1.0).collect();
    mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&data, &shape(1)),
        dtype::FLOAT16,
    )
}

fn block_value(i: usize) -> f32 {
    -(((i * 7) % 1000) as f32) - 1.0
}

/// A `[rows, 32, 8, 128]` f16 zero source with one element of storage.
fn zero_source(rows: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::broadcast_to(&mlxcel_core::zeros(&[1], dtype::FLOAT16), &shape(rows))
}

fn dynamic_update(src: &MlxArray, update: &MlxArray, first_row: i32) -> UniquePtr<MlxArray> {
    let start = mlxcel_core::from_slice_i32(&[first_row], &[1]);
    mlxcel_core::slice_update_dynamic(src, update, &start, &[0]).expect("valid update")
}

fn concatenate(a: &MlxArray, b: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::concatenate(a, b, 0)
}

/// Builds `op` on the GPU or the CPU stream and evaluates it there.
/// Builds `op` on the GPU and evaluates it there.
fn eval_on_gpu(op: &dyn Fn() -> UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    let out = op();
    mlxcel_core::try_eval(&out).expect("the copy launches");
    out
}

/// Evaluates `op` on the GPU and compares every element with `expected(r, i)`
/// (row `r`, flat inner index `i`) computed on the host. The reference goes up
/// as f32 and is rounded to f16 on the GPU, which is exact for these values;
/// comparing there keeps the test off the slow host reads of fine-grained
/// device memory that a CPU-stream reference costs.
fn assert_gpu_matches_host(
    what: &str,
    rows: i32,
    op: &dyn Fn() -> UniquePtr<MlxArray>,
    expected: impl Fn(usize, usize) -> f32,
) {
    let host: Vec<f32> = (0..rows as usize)
        .flat_map(|r| (0..ROW).map(move |i| (r, i)))
        .map(|(r, i)| expected(r, i))
        .collect();
    let out = eval_on_gpu(op);
    let _guard = DefaultDeviceGuard::gpu();
    let reference = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&host, &shape(rows)),
        dtype::FLOAT16,
    );
    let eq = mlxcel_core::array_equal(&out, &reference, false);
    assert!(
        mlxcel_core::item_bool(&eq),
        "{what}: GPU differs from the host reference"
    );
}

/// Row `r` of a `[rows, 32, 8, 128]` result, as f32 on the host.
fn row(out: &MlxArray, r: i32) -> Vec<f32> {
    let _guard = DefaultDeviceGuard::gpu();
    let one = mlxcel_core::slice(out, &[r, 0, 0, 0], &[r + 1, INNER[0], INNER[1], INNER[2]]);
    let as_f32 = mlxcel_core::astype(&one, dtype::FLOAT32);
    mlxcel_core::try_eval(&as_f32).expect("row reads back");
    mlxcel_core::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

fn assert_row(out: &MlxArray, r: i32, expected: impl Fn(usize) -> f32, what: &str) {
    let got = row(out, r);
    assert_eq!(got.len(), ROW);
    if let Some(i) = (0..ROW).find(|&i| got[i] != expected(i)) {
        panic!(
            "{what}: row {r} element {i} is {} (expected {})",
            got[i],
            expected(i)
        );
    }
}

/// Rows of the large result that straddle element 2^31 and element 2^32.
const BOUNDARY_ROWS: [i32; 4] = [65_535, 65_536, 131_071, 131_072];

// ---- fast: the grid-stride loop at sizes past the grid cap ------------------

#[test]
fn strided_input_copy_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_gpu_matches_host(
        "contiguous(strided view)",
        FAST_ROWS,
        &|| mlxcel_core::contiguous(&strided_view(FAST_ROWS), false),
        view_value,
    );
}

#[test]
fn concatenate_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let last = FAST_ROWS as usize;
    assert_gpu_matches_host(
        "concatenate(strided view, block)",
        FAST_ROWS + 1,
        &|| concatenate(&strided_view(FAST_ROWS), &block()),
        |r, i| {
            if r == last {
                block_value(i)
            } else {
                view_value(r, i)
            }
        },
    );
}

#[test]
fn dynamic_update_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_gpu_matches_host(
        "slice_update_dynamic(zeros, strided view, 1)",
        FAST_ROWS + 1,
        &|| dynamic_update(&zero_source(FAST_ROWS + 1), &strided_view(FAST_ROWS), 1),
        |r, i| if r == 0 { 0.0 } else { view_value(r - 1, i) },
    );
    let middle = FAST_ROWS / 2;
    assert_gpu_matches_host(
        "slice_update_dynamic(zeros, block, middle)",
        FAST_ROWS,
        &|| dynamic_update(&zero_source(FAST_ROWS), &block(), middle),
        |r, i| {
            if r == middle as usize {
                block_value(i)
            } else {
                0.0
            }
        },
    );
}

// ---- large: more than 2^32 elements -----------------------------------------

#[test]
#[ignore = "allocates an 8 GiB f16 array; run with --ignored on a device with ~10 GiB free"]
fn strided_input_copy_past_u32_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let out = eval_on_gpu(&|| mlxcel_core::contiguous(&strided_view(BIG_ROWS), false));
    for r in [0].into_iter().chain(BOUNDARY_ROWS) {
        assert_row(&out, r, |i| view_value(r as usize, i), "contiguous");
    }
}

#[test]
#[ignore = "allocates an 8 GiB f16 array; run with --ignored on a device with ~10 GiB free"]
fn concatenate_past_u32_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let out = eval_on_gpu(&|| concatenate(&strided_view(BIG_ROWS), &block()));
    for r in [0].into_iter().chain(BOUNDARY_ROWS) {
        assert_row(&out, r, |i| view_value(r as usize, i), "concatenate");
    }
    assert_row(&out, BIG_ROWS, block_value, "concatenate tail");
}

#[test]
#[ignore = "allocates an 8 GiB f16 array; run with --ignored on a device with ~10 GiB free"]
fn dynamic_update_past_u32_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let out =
        eval_on_gpu(&|| dynamic_update(&zero_source(BIG_ROWS + 1), &strided_view(BIG_ROWS), 1));
    assert_row(&out, 0, |_| 0.0, "dynamic update, row before the update");
    for r in BOUNDARY_ROWS.into_iter().chain([BIG_ROWS]) {
        assert_row(&out, r, |i| view_value(r as usize - 1, i), "dynamic update");
    }
}

/// A one-block write into a slab past 2^32 elements, the paged-cache pattern:
/// the old launch was sized by the destination, not by the block.
#[test]
#[ignore = "allocates an 8 GiB f16 array; run with --ignored on a device with ~10 GiB free"]
fn dynamic_one_block_update_into_a_slab_past_u32_elements() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let last = BIG_ROWS - 1;
    let out = eval_on_gpu(&|| dynamic_update(&zero_source(BIG_ROWS), &block(), last));
    assert_row(&out, last, block_value, "one-block update");
    assert_row(&out, last - 1, |_| 0.0, "row before the block");
    assert_row(&out, 0, |_| 0.0, "first row");
}
