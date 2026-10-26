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

//! HIP launchers that size a grid from the element count (issue #2234,
//! `patches-rocm/LOCAL_FIXES.md` item 46).
//!
//! The overlay's elementwise, reduce, arange, indexing, sort, quantize and
//! transpose-copy launchers computed the block count from the element count
//! with no cap and their kernels ran one index per thread. HIP refuses a
//! launch of 2^32 threads or more (`hipErrorInvalidConfiguration`), grid
//! dimensions y and z top out at 65535 blocks, and an `int num_blocks`
//! overflows earlier, so every op that large failed. The launchers now
//! compute the block count with `rocm::copy_grid_blocks` (64-bit, clamped to
//! [1, 65535], from `copy/copy.hpp`), and the kernels walk their elements
//! with a grid-stride loop over an `int64_t` counter, the pattern of #2184.
//!
//! The ops below pick their launchers through the bridge: `add`, `negative`
//! and `where_cond` of an `as_strided` view take the general elementwise
//! kernels (`binary_g`, `unary_g`, `ternary_g`, the last two with their
//! capped row grid), `sum_axis` over a leading axis takes
//! `col_reduce_small`, `arange_u32` takes `arange_kernel`, `take` along a
//! middle axis takes `gather_axis_kernel`, the bridge's `scatter` routes to
//! `scatter_general_kernel`, `argsort` initializes its index array with
//! `iota_kernel`, and `contiguous` of a column-contiguous 2-D view takes the
//! `copy_col_row` tile kernel whose y grid the fix caps.
//!
//! The elementwise inputs are `as_strided` views of a small buffer with every
//! stride 1, so element `[r, j, k, l]` is `buf[r + j + k + l]`: a different
//! value per position that the host can recompute for any row without
//! materializing the input, and dimensions that `collapse_contiguous_dims`
//! cannot merge. Values are small integers, exact in f16.
//!
//! The fast tests run sizes past the 65535-block cap and compare every
//! element with a reference computed on the host: 68.8M elements for the
//! elementwise ops (about 4.1 passes of a 65535 x 256 grid, and 67200 blocks
//! of the row grid, over the 65535-block y limit), 21.3M for the gather and
//! scatter, 67.1M for the column reduce, 2^24 + 1 for arange, argsort and
//! the transpose copy. The row-grid and tile-grid tests fail without the fix
//! with `hipErrorInvalidConfiguration`: `num_blocks_y` exceeded 65535 blocks
//! of grid y, which HIP refuses outright. Before the fix, the `#[ignore]`d
//! tests failed the same way in `try_eval` at 2^32 elements or more.
//!
//! ```sh
//! cargo test --features rocm --test rocm_hip_grid_clamp -- --test-threads=1
//! cargo test --features rocm --test rocm_hip_grid_clamp -- --ignored --test-threads=1
//! ```
//!
//! The `gather_qmv` module (issue #2620, `LOCAL_FIXES.md` item 61) runs an
//! unsorted `gather_qmm` whose batch of 65537 rows sits in grid z, through
//! each gather QMV kernel the dispatch picks, and compares every element with
//! a host reference; its `#[ignore]`d test checks a `gather_qmv_kernel`
//! output of more than 2^31 elements on both sides of element 2^31. Those
//! grids launched before the fix as well: measured with ROCm 7.15 on
//! gfx1151, `hipLaunchKernel` runs grids of 65537 and more blocks in y and z
//! although `hipDeviceProp_t::maxGridSize` reports 65535 for both, so the
//! limit that applies there is 2^32 - 1 threads per dimension. That runtime
//! did not reproduce the y-grid refusal described above for item 46; the
//! item 46 caps stay in place. The fast tests stay as a guard for the
//! batch-in-z launch.
//!
//! The per-row softmax, logsumexp, rms_norm and layer_norm launchers of issue
//! #2619 have their own tests in `rocm_hip_grid_clamp/row_grid.rs`.
//! The layer_norm bias gradient of issue #2632 (`LOCAL_FIXES.md` item 64) is
//! checked in `rocm_hip_grid_clamp/layer_norm_bias_grad.rs` at small sizes and
//! in `row_grid.rs` past the row-grid cap.
//!
//! Every test moves the process-global default device, so each holds
//! `streams::lock_default_device` for its whole body. Skips on any other
//! backend.

#![cfg(feature = "rocm")]

use mlxcel_core::dtype;
use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr};

#[path = "rocm_hip_grid_clamp/gather_qmv.rs"]
mod gather_qmv;
#[path = "rocm_hip_grid_clamp/layer_norm_bias_grad.rs"]
mod layer_norm_bias_grad;
#[path = "rocm_hip_grid_clamp/row_grid.rs"]
mod row_grid;

/// One row is one block of the paged KV slab: `[PAGE, HKV, DIM]`.
const INNER: [i32; 3] = [32, 8, 128];
const ROW: usize = 32 * 8 * 128;
/// Rows in the fast tests: 2100 * 32768 = 68.8M elements, about 4.1 passes of
/// a 65535 x 256 grid, and 67200 blocks of the elementwise row grid, past the
/// 65535-block y limit.
const FAST_ROWS: i32 = 2100;
/// Rows in the large tests: 2^17 + 1 rows of 2^15 elements, 2^32 + 2^15.
const BIG_ROWS: i32 = 131_073;
/// Values cycle below 2048, the last integer range f16 holds exactly.
const MODULUS: usize = 2039;
/// Rows of the large result that straddle element 2^31 and element 2^32.
const BOUNDARY_ROWS: [i32; 5] = [0, 65_535, 65_536, 131_071, 131_072];

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

/// Value of the second view, with values unlike the first one.
fn view_b_value(r: usize, i: usize) -> f32 {
    let (j, k, l) = (i / 1024, (i / 128) % 8, i % 128);
    (((r + j + k + l) * 13 + 5) % 1000) as f32
}

fn cond_value(r: usize, i: usize) -> bool {
    let (j, k, l) = (i / 1024, (i / 128) % 8, i % 128);
    // Matches the buffer: element i of `strided_cond` is true when i is odd.
    !((r + j + k + l) * 7).is_multiple_of(2)
}

/// An f16 (or other-dtype) array from a flat generator, as one 1-D buffer.
fn typed_from(flat: impl Fn(usize) -> f32, len: usize, dtype: i32) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..len).map(&flat).collect();
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(&data, &[len as i32]), dtype)
}

/// `[rows, 32, 8, 128]` f16 whose element `[r, j, k, l]` is `buf[r + j + k + l]`:
/// no storage beyond `rows + 165` elements.
fn strided_view(rows: i32) -> UniquePtr<MlxArray> {
    let buf = typed_from(
        |i| ((i + 1) % MODULUS) as f32,
        rows as usize + 165,
        dtype::FLOAT16,
    );
    mlxcel_core::as_strided(&buf, &shape(rows), &[1, 1, 1, 1], 0)
}

fn strided_view_b(rows: i32) -> UniquePtr<MlxArray> {
    let buf = typed_from(
        |i| ((i * 13 + 5) % 1000) as f32,
        rows as usize + 165,
        dtype::FLOAT16,
    );
    mlxcel_core::as_strided(&buf, &shape(rows), &[1, 1, 1, 1], 0)
}

/// A `[rows, 32, 8, 128]` bool view with the same stride-1 layout.
fn strided_cond(rows: i32) -> UniquePtr<MlxArray> {
    let buf = typed_from(|i| ((i * 7) % 2) as f32, rows as usize + 165, dtype::BOOL);
    mlxcel_core::as_strided(&buf, &shape(rows), &[1, 1, 1, 1], 0)
}

/// Builds `op` on the GPU and evaluates it there.
fn eval_on_gpu(op: &dyn Fn() -> UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    let out = op();
    mlxcel_core::try_eval(&out).expect("the launches");
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

/// Row `r` of a result, as f32 on the host.
fn row(out: &MlxArray, starts: &[i32], stops: &[i32], len: usize) -> Vec<f32> {
    let _guard = DefaultDeviceGuard::gpu();
    let one = mlxcel_core::slice(out, starts, stops);
    let as_f32 = mlxcel_core::astype(&one, dtype::FLOAT32);
    mlxcel_core::try_eval(&as_f32).expect("row reads back");
    mlxcel_core::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .take(len)
        .collect()
}

/// Asserts the sampled rows of a large result against `expected` computed on
/// the host, one panic message per mismatching element.
fn assert_rows(
    what: &str,
    out: &MlxArray,
    slices: &[(&str, Vec<i32>, Vec<i32>, usize)],
    expected: impl Fn(&[usize]) -> f32,
) {
    for (name, starts, stops, len) in slices {
        let got = row(out, starts, stops, *len);
        assert_eq!(got.len(), *len, "{what}: {name} read back short");
        let idx: Vec<usize> = starts.iter().map(|&s| s as usize).collect();
        for (i, value) in got.iter().enumerate() {
            let mut flat = idx.clone();
            let last = flat.len() - 1;
            flat[last] += i;
            let want = expected(&flat);
            assert!(
                *value == want,
                "{what}: {name} element {i} is {value} (expected {want})"
            );
        }
    }
}

/// A zero slab of `[d0, d1, d2]` f16, a middle-axis index vector and the
/// matching update, for the gather and scatter axis tests.
struct Axis {
    d0: i32,
    d1: i32,
    d2: i32,
    l: i32,
    idx: Vec<i32>,
}

impl Axis {
    fn new(d0: i32, d1: i32, d2: i32, l: i32, offset: i32) -> Self {
        let idx = (0..l).map(|i| (i * 97 + offset) % d1).collect();
        Axis { d0, d1, d2, l, idx }
    }

    fn src(&self) -> UniquePtr<MlxArray> {
        let flat = typed_from(
            |i| ((i * 3 + 1) % 2000) as f32,
            (self.d0 * self.d1 * self.d2) as usize,
            dtype::FLOAT16,
        );
        mlxcel_core::reshape(&flat, &[self.d0, self.d1, self.d2])
    }

    fn indices(&self) -> UniquePtr<MlxArray> {
        let data = self.idx.clone();
        mlxcel_core::from_slice_i32(&data, &[self.l])
    }
}

// ---- fast: sizes past the 65535-block cap -----------------------------------

#[test]
fn binary_add_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_gpu_matches_host(
        "add(strided view, strided view b)",
        FAST_ROWS,
        &|| mlxcel_core::add(&strided_view(FAST_ROWS), &strided_view_b(FAST_ROWS)),
        |r, i| view_value(r, i) + view_b_value(r, i),
    );
}

#[test]
fn unary_negative_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_gpu_matches_host(
        "negative(strided view)",
        FAST_ROWS,
        &|| mlxcel_core::negative(&strided_view(FAST_ROWS)),
        |r, i| -view_value(r, i),
    );
}

#[test]
fn ternary_where_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    assert_gpu_matches_host(
        "where_cond(cond, view, view b)",
        FAST_ROWS,
        &|| {
            mlxcel_core::where_cond(
                &strided_cond(FAST_ROWS),
                &strided_view(FAST_ROWS),
                &strided_view_b(FAST_ROWS),
            )
        },
        |r, i| {
            if cond_value(r, i) {
                view_value(r, i)
            } else {
                view_b_value(r, i)
            }
        },
    );
}

#[test]
fn arange_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let n: i32 = (1 << 24) + 1;
    let out = eval_on_gpu(&|| mlxcel_core::arange_u32(n));
    let reference = typed_from(|i| i as f32, n as usize, dtype::UINT32);
    let eq = mlxcel_core::array_equal(&out, &reference, false);
    assert!(mlxcel_core::item_bool(&eq), "arange_u32 differs from 0..n");
}

/// `take` along a middle axis of a contiguous source: `gather_axis_kernel`
/// with `idx_size_pre * idx_size_axis * idx_size_post` indices, past the cap.
#[test]
fn gather_axis_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let axis = Axis::new(1300, 512, 64, 256, 0);
    let src = axis.src();
    let out = eval_on_gpu(&|| mlxcel_core::take(&src, &axis.indices(), 1));
    let _guard = DefaultDeviceGuard::gpu();
    let reference = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &(0..(axis.d0 * axis.l * axis.d2))
                .map(|i| {
                    let (a, rem) = (i / (axis.l * axis.d2), i % (axis.l * axis.d2));
                    let (b, rest) = (rem / axis.d2, rem % axis.d2);
                    let b = axis.idx[b as usize];
                    // The modulo stays in integer math: the flat index is
                    // above 2^24, so an f32 cast before it would round.
                    ((((a * axis.d1 + b) * axis.d2 + rest) * 3 + 1) % 2000) as f32
                })
                .collect::<Vec<_>>(),
            &[axis.d0, axis.l, axis.d2],
        ),
        dtype::FLOAT16,
    );
    let eq = mlxcel_core::array_equal(&out, &reference, false);
    assert!(mlxcel_core::item_bool(&eq), "gather axis differs from host");
}

/// `scatter` of a slab at disjoint element offsets: the bridge's `scatter`
/// routes to `scatter_general_kernel` with one index per update element, past
/// the cap. The fork's scatter writes `out[idx[i] + j] = updates[i, j]` with
/// bounds, so disjoint offsets keep every row's host reference independent.
#[test]
fn scatter_general_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let (n, l): (usize, usize) = (1 << 10, 1 << 14);
    let out_len = (l - 1) * n + n;
    let idx_data: Vec<i32> = (0..l).map(|i| (i * n) as i32).collect();
    let a = mlxcel_core::zeros(&[out_len as i32], dtype::FLOAT16);
    let upd = mlxcel_core::reshape(
        &typed_from(|i| ((i * 7 + 3) % 2000) as f32, l * n, dtype::FLOAT16),
        &[l as i32, n as i32],
    );
    let idx = mlxcel_core::from_slice_i32(&idx_data, &[l as i32]);
    let out = eval_on_gpu(&|| mlxcel_core::scatter(&a, &idx, &upd, 0));
    let _guard = DefaultDeviceGuard::gpu();
    let mut host: Vec<f32> = vec![0.0; out_len];
    for (i, &off) in idx_data.iter().enumerate() {
        for j in 0..n {
            host[off as usize + j] = (((i * n + j) * 7 + 3) % 2000) as f32;
        }
    }
    let reference = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&host, &[out_len as i32]),
        dtype::FLOAT16,
    );
    let eq = mlxcel_core::array_equal(&out, &reference, false);
    assert!(
        mlxcel_core::item_bool(&eq),
        "scatter general differs from host"
    );
}

/// `sum_axis` over a leading axis of size 2: `col_reduce_small` with 67.1M
/// outputs, past the cap.
#[test]
fn col_reduce_small_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let n: i32 = 1 << 26;
    let host_row = |i: usize| ((i % MODULUS) + 1) as f32;
    let src = mlxcel_core::reshape(
        &typed_from(host_row, 2 * n as usize, dtype::FLOAT16),
        &[2, n],
    );
    let out = eval_on_gpu(&|| mlxcel_core::sum_axis(&src, 0, false));
    let host: Vec<f32> = (0..n as usize)
        .map(|i| host_row(i) + host_row(i + n as usize))
        .collect();
    let reference = typed_from(|i| host[i], n as usize, dtype::FLOAT16);
    let eq = mlxcel_core::array_equal(&out, &reference, false);
    assert!(mlxcel_core::item_bool(&eq), "col reduce differs from host");
}

/// `argsort` of more than 2^24 elements: `gpu_sort` initializes its index
/// array with `iota_kernel`, whose grid the fix caps.
#[test]
fn argsort_iota_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let n: i32 = (1 << 24) + 1;
    let a = typed_from(|i| ((i * 37) % 4093) as f32, n as usize, dtype::FLOAT32);
    let out = eval_on_gpu(&|| mlxcel_core::argsort(&a, 0));
    let mut order: Vec<i32> = (0..n).collect();
    order.sort_by_key(|&i| ((i as usize * 37) % 4093) as i32);
    let reference = mlxcel_core::astype(&mlxcel_core::from_slice_i32(&order, &[n]), dtype::UINT32);
    let eq = mlxcel_core::array_equal(&out, &reference, false);
    assert!(mlxcel_core::item_bool(&eq), "argsort differs from host");
}

/// `contiguous` of a column-contiguous 2-D view: the `copy_col_row` tile
/// kernel with a 2-D tile grid whose y the fix caps at 65535 blocks.
#[test]
fn copy_col_row_past_the_grid_cap_matches_host() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let (rows, cols): (i32, i32) = (16, 1 << 20);
    let buf = typed_from(
        |i| ((i * 11 + 7) % 2000) as f32,
        rows as usize * cols as usize,
        dtype::FLOAT16,
    );
    let col_view = mlxcel_core::as_strided(&buf, &[rows, cols], &[1, rows as i64], 0);
    let out = eval_on_gpu(&|| mlxcel_core::contiguous(&col_view, false));
    let _guard = DefaultDeviceGuard::gpu();
    let host: Vec<f32> = (0..rows as usize)
        .flat_map(|r| (0..cols as usize).map(move |c| (r, c)))
        .map(|(r, c)| (((r + c * rows as usize) * 11 + 7) % 2000) as f32)
        .collect();
    let reference = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&host, &[rows, cols]),
        dtype::FLOAT16,
    );
    let eq = mlxcel_core::array_equal(&out, &reference, false);
    assert!(
        mlxcel_core::item_bool(&eq),
        "col-row copy differs from host"
    );
}

// ---- large: more than 2^32 elements -----------------------------------------

#[test]
#[ignore = "allocates an 8 GiB f16 array; run with --ignored on a device with ~10 GiB free"]
fn binary_add_past_u32_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let out = mlxcel_core::add(&strided_view(BIG_ROWS), &strided_view_b(BIG_ROWS));
    mlxcel_core::try_eval(&out).expect("the add launches");
    let slices: Vec<(&str, Vec<i32>, Vec<i32>, usize)> = BOUNDARY_ROWS
        .iter()
        .map(|&r| {
            (
                &*Box::leak(format!("row {r}").into_boxed_str()),
                vec![r, 0, 0, 0],
                vec![r + 1, INNER[0], INNER[1], INNER[2]],
                ROW,
            )
        })
        .collect();
    assert_rows("add", &out, &slices, |f| {
        view_value(f[0], f[1] * 1024 + f[2] * 128 + f[3])
            + view_b_value(f[0], f[1] * 1024 + f[2] * 128 + f[3])
    });
}

#[test]
#[ignore = "allocates an 8 GiB f16 array; run with --ignored on a device with ~10 GiB free"]
fn unary_negative_past_u32_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let out = mlxcel_core::negative(&strided_view(BIG_ROWS));
    mlxcel_core::try_eval(&out).expect("the negative launches");
    let slices: Vec<(&str, Vec<i32>, Vec<i32>, usize)> = BOUNDARY_ROWS
        .iter()
        .map(|&r| {
            (
                &*Box::leak(format!("row {r}").into_boxed_str()),
                vec![r, 0, 0, 0],
                vec![r + 1, INNER[0], INNER[1], INNER[2]],
                ROW,
            )
        })
        .collect();
    assert_rows("negative", &out, &slices, |f| {
        -view_value(f[0], f[1] * 1024 + f[2] * 128 + f[3])
    });
}

#[test]
#[ignore = "allocates an 8 GiB f16 array; run with --ignored on a device with ~10 GiB free"]
fn ternary_where_past_u32_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let out = mlxcel_core::where_cond(
        &strided_cond(BIG_ROWS),
        &strided_view(BIG_ROWS),
        &strided_view_b(BIG_ROWS),
    );
    mlxcel_core::try_eval(&out).expect("the where launches");
    let slices: Vec<(&str, Vec<i32>, Vec<i32>, usize)> = BOUNDARY_ROWS
        .iter()
        .map(|&r| {
            (
                &*Box::leak(format!("row {r}").into_boxed_str()),
                vec![r, 0, 0, 0],
                vec![r + 1, INNER[0], INNER[1], INNER[2]],
                ROW,
            )
        })
        .collect();
    assert_rows("where", &out, &slices, |f| {
        let i = f[1] * 1024 + f[2] * 128 + f[3];
        if cond_value(f[0], i) {
            view_value(f[0], i)
        } else {
            view_b_value(f[0], i)
        }
    });
}

/// `take` along a middle axis of 2^32 or more elements: 1300 * 65536 * 64,
/// about 10.9 GiB of f16.
#[test]
#[ignore = "allocates a ~10.9 GiB f16 array; run with --ignored on a device with ~12 GiB free"]
fn gather_axis_past_u32_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let axis = Axis::new(1300, 512, 64, 65_536, 0);
    let src = axis.src();
    let out = mlxcel_core::take(&src, &axis.indices(), 1);
    mlxcel_core::try_eval(&out).expect("the gather launches");
    let mut spots: Vec<(&str, Vec<i32>, Vec<i32>, usize)> = Vec::new();
    for a in [0, axis.d0 - 1] {
        for b in [0, axis.l / 2, axis.l - 1] {
            spots.push((
                &*Box::leak(format!("[{a}, {b}]").into_boxed_str()),
                vec![a, b, 0],
                vec![a + 1, b + 1, axis.d2],
                axis.d2 as usize,
            ));
        }
    }
    assert_rows("gather", &out, &spots, |f| {
        let b = axis.idx[f[1]] as usize;
        ((((f[0] * axis.d1 as usize + b) * axis.d2 as usize + f[2]) * 3 + 1) % 2000) as f32
    });
}
