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

//! Per-row launchers that put one block per row in grid x (issue #2619,
//! `LOCAL_FIXES.md` item 62, the ROCm side of ml-explore/mlx#4717).
//!
//! The softmax, logsumexp, rms_norm and layer_norm launchers (forward and
//! VJP) launched `dim3(n_rows)` blocks with `n_rows` an `int`. HIP refuses a
//! launch of 2^32 threads or more, so with 256 threads a block the launch
//! failed from 2^24 rows on ("invalid configuration argument" on gfx1151),
//! with 512 from 2^23 and with 1024 from 2^22. The launchers now cap grid x
//! at `UINT32_MAX / BLOCK_DIM` and spill the rest into grid y; each kernel
//! recovers its 64-bit row from both and returns past `n_rows`.
//!
//! The fast tests run 2^24 + 1 rows of 4 f32 (256 MiB an input) through the
//! 256-thread launchers and compare every element on the GPU with a
//! reference computed on the host in f64. A CPU-stream reference was tried
//! first: it took about five minutes a test on gfx1151 (host reads of device
//! memory), and its f32 sum of 2^24 rows for the layer_norm weight gradient
//! was 2.7e5 off the GPU's, which is within 1.7e3 of the f64 sum. The `#[ignore]`d tests drive the 512
//! and 1024-thread softmax arms, whose smallest failing inputs hold about
//! 8.6G elements.

use mlxcel_core::dtype;
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr};

use super::on_rocm;

/// 2^24 + 1 rows: one block past the 256-thread cap of 2^32 threads.
const ROWS: i32 = (1 << 24) + 1;
const AXIS: usize = 4;
const EPS: f32 = 1e-5;
const W: [f32; AXIS] = [0.5, 1.0, 1.5, 2.0];
const B: [f32; AXIS] = [0.1, -0.2, 0.3, -0.4];
/// Salt of the cotangent, so it differs from the input.
const G_SALT: usize = 41;

/// Element `[r, j]` of an input: a 97-value cycle over `7r + 13j`, a multiple
/// of 1/16 in [-3, 3], exact in f32.
fn val(r: usize, j: usize, salt: usize) -> f64 {
    ((r * 7 + j * 13 + salt) % 97) as f64 / 16.0 - 3.0
}

fn row_of(r: usize, salt: usize) -> [f64; AXIS] {
    std::array::from_fn(|j| val(r, j, salt))
}

/// `[rows, axis]` f32 of [`val`].
fn input(rows: i32, axis: usize, salt: usize) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..rows as usize * axis)
        .map(|i| val(i / axis, i % axis, salt) as f32)
        .collect();
    mlxcel_core::from_slice_f32(&data, &[rows, axis as i32])
}

fn vector(values: &[f32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, &[values.len() as i32])
}

/// `[ROWS, AXIS]` reference rows from `f`, flattened to f32.
fn per_row(f: impl Fn(usize) -> [f64; AXIS]) -> Vec<f32> {
    (0..ROWS as usize)
        .flat_map(|r| f(r).map(|v| v as f32))
        .collect()
}

/// Evaluates `op` on the GPU and asserts every element is within `atol + rtol
/// * |expected|` of `expected`, compared on the GPU.
fn assert_gpu_matches(
    what: &str,
    op: &dyn Fn() -> UniquePtr<MlxArray>,
    expected: &[f32],
    shape: &[i32],
    rtol: f64,
    atol: f64,
) {
    let _guard = DefaultDeviceGuard::gpu();
    let out = op();
    if let Err(e) = mlxcel_core::try_eval(&out) {
        panic!("{what}: GPU launch failed: {e}");
    }
    assert_eq!(mlxcel_core::array_shape(&out), shape, "{what}: shape");
    let want = mlxcel_core::from_slice_f32(expected, shape);
    let close = mlxcel_core::allclose(&out, &want, rtol, atol);
    if !mlxcel_core::item_bool(&close) {
        let diff = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(&out, &want)));
        let detail = if expected.len() <= 16 {
            let got: Vec<f32> = mlxcel_core::array_to_raw_bytes(&out)
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
                .collect();
            format!(" (got {got:?}, expected {expected:?})")
        } else {
            String::new()
        };
        panic!(
            "{what}: GPU differs from the host reference, max |diff| {}{detail}",
            mlxcel_core::item_f32(&diff)
        );
    }
}

/// `1 / sqrt(mean(x^2) + eps)` of one row.
fn rms_scale(x: &[f64; AXIS]) -> f64 {
    1.0 / (x.iter().map(|v| v * v).sum::<f64>() / AXIS as f64 + EPS as f64).sqrt()
}

/// The centered row and `1 / sqrt(var + eps)`.
fn centered(x: &[f64; AXIS]) -> ([f64; AXIS], f64) {
    let mean = x.iter().sum::<f64>() / AXIS as f64;
    let xc = x.map(|v| v - mean);
    let var = xc.iter().map(|v| v * v).sum::<f64>() / AXIS as f64;
    (xc, 1.0 / (var + EPS as f64).sqrt())
}

/// Sum over every row of `f(r)`, the shape of a weight gradient.
fn column_sums(f: impl Fn(usize) -> [f64; AXIS]) -> Vec<f32> {
    let mut acc = [0.0_f64; AXIS];
    for r in 0..ROWS as usize {
        for (a, v) in acc.iter_mut().zip(f(r)) {
            *a += v;
        }
    }
    acc.iter().map(|&v| v as f32).collect()
}

macro_rules! rocm_only {
    () => {
        if !on_rocm() {
            eprintln!("skipping: not running on a ROCm device");
            return;
        }
    };
}

const SHAPE: [i32; 2] = [ROWS, AXIS as i32];

// ---- fast: 2^24 + 1 rows through the 256-thread launchers --------------------

#[test]
fn softmax_past_the_row_grid_cap_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    let x = input(ROWS, AXIS, 0);
    let expected = per_row(|r| {
        let x = row_of(r, 0);
        let max = x.iter().copied().fold(f64::MIN, f64::max);
        let e = x.map(|v| (v - max).exp());
        let sum: f64 = e.iter().sum();
        e.map(|v| v / sum)
    });
    let op = || mlxcel_core::softmax(&x, -1);
    assert_gpu_matches("softmax", &op, &expected, &SHAPE, 1e-5, 1e-6);
}

#[test]
fn logsumexp_past_the_row_grid_cap_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    let x = input(ROWS, AXIS, 0);
    let expected: Vec<f32> = (0..ROWS as usize)
        .map(|r| {
            let x = row_of(r, 0);
            let max = x.iter().copied().fold(f64::MIN, f64::max);
            (max + x.iter().map(|v| (v - max).exp()).sum::<f64>().ln()) as f32
        })
        .collect();
    let op = || mlxcel_core::logsumexp_axis(&x, -1, false);
    assert_gpu_matches("logsumexp", &op, &expected, &[ROWS], 1e-5, 1e-5);
}

fn rms_norm_reference(salt: usize) -> Vec<f32> {
    per_row(|r| {
        let x = row_of(r, salt);
        let n = rms_scale(&x);
        std::array::from_fn(|j| W[j] as f64 * x[j] * n)
    })
}

#[test]
fn rms_norm_past_the_row_grid_cap_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    let x = input(ROWS, AXIS, 0);
    let w = vector(&W);
    let op = || mlxcel_core::fast_rms_norm(&x, &w, EPS);
    assert_gpu_matches("rms_norm", &op, &rms_norm_reference(0), &SHAPE, 1e-5, 1e-5);
}

/// Rows of a `[ROWS, 8]` input sliced to their first 4 columns: last axis
/// contiguous, rows not packed, so the launcher takes the strided kernel.
#[test]
fn rms_norm_strided_past_the_row_grid_cap_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    let wide = input(ROWS, 2 * AXIS, 5);
    let w = vector(&W);
    let op = || {
        let x = mlxcel_core::slice(&wide, &[0, 0], &[ROWS, AXIS as i32]);
        mlxcel_core::fast_rms_norm(&x, &w, EPS)
    };
    let expected = rms_norm_reference(5);
    assert_gpu_matches("strided rms_norm", &op, &expected, &SHAPE, 1e-5, 1e-5);
}

#[test]
fn layer_norm_past_the_row_grid_cap_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    let x = input(ROWS, AXIS, 0);
    let (w, b) = (vector(&W), vector(&B));
    let expected = per_row(|r| {
        let (xc, n) = centered(&row_of(r, 0));
        std::array::from_fn(|j| xc[j] * n * W[j] as f64 + B[j] as f64)
    });
    // SAFETY: `w` and `b` are live arrays for the whole call.
    let op = || unsafe { mlxcel_core::fast_layer_norm(&x, &*w, &*b, EPS) };
    assert_gpu_matches("layer_norm", &op, &expected, &SHAPE, 1e-5, 1e-5);
}

/// The weight gradient sums 2^24 + 1 rows in f32 on the GPU, about 4e-4 off
/// the f64 sum on gfx1151, so it gets a relative tolerance of 1e-3. At that
/// tolerance one missing row would not show in it; the per-row `dx`, which
/// the same blocks write, is what checks every row ran.
#[test]
fn rms_norm_vjp_past_the_row_grid_cap_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    let (x, g, w) = (input(ROWS, AXIS, 0), input(ROWS, AXIS, G_SALT), vector(&W));
    let wf = W.map(f64::from);
    let dx = per_row(|r| {
        let (x, g) = (row_of(r, 0), row_of(r, G_SALT));
        let n = rms_scale(&x);
        let mean_wgx = (0..AXIS).map(|j| wf[j] * g[j] * x[j]).sum::<f64>() / AXIS as f64;
        std::array::from_fn(|j| n * wf[j] * g[j] - x[j] * mean_wgx * n * n * n)
    });
    let dw = column_sums(|r| {
        let (x, g) = (row_of(r, 0), row_of(r, G_SALT));
        let n = rms_scale(&x);
        std::array::from_fn(|j| g[j] * x[j] * n)
    });
    let vjp = |wrt| {
        mlxcel_core::fast_rms_norm_vjp_for_test(&x, &w, &g, EPS, wrt).expect("the vjp builds")
    };
    assert_gpu_matches("rms_norm vjp dx", &|| vjp(0), &dx, &SHAPE, 1e-4, 1e-4);
    assert_gpu_matches(
        "rms_norm vjp dw",
        &|| vjp(1),
        &dw,
        &[AXIS as i32],
        1e-3,
        1.0,
    );
}

/// Offset of the layer_norm cotangent (issue #2632): [`val`] sums to about
/// zero over each 97-row cycle, so a bias gradient of unwritten (zeroed)
/// memory would pass for its column sums. With the offset they are about
/// `G_OFFSET * ROWS`.
const G_OFFSET: f64 = 0.5;

/// Row `r` of the layer_norm cotangent.
fn cotangent_row(r: usize) -> [f64; AXIS] {
    row_of(r, G_SALT).map(|v| v + G_OFFSET)
}

#[test]
fn layer_norm_vjp_past_the_row_grid_cap_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    let x = input(ROWS, AXIS, 0);
    let g = mlxcel_core::from_slice_f32(&per_row(cotangent_row), &[ROWS, AXIS as i32]);
    let (w, b) = (vector(&W), vector(&B));
    let wf = W.map(f64::from);
    let dx = per_row(|r| {
        let ((xc, n), g) = (centered(&row_of(r, 0)), cotangent_row(r));
        let mean_wg = (0..AXIS).map(|j| wf[j] * g[j]).sum::<f64>() / AXIS as f64;
        let mean_wgxc = (0..AXIS).map(|j| wf[j] * g[j] * xc[j]).sum::<f64>() / AXIS as f64;
        std::array::from_fn(|j| n * (wf[j] * g[j] - mean_wg) - xc[j] * mean_wgxc * n * n * n)
    });
    let dw = column_sums(|r| {
        let ((xc, n), g) = (centered(&row_of(r, 0)), cotangent_row(r));
        std::array::from_fn(|j| g[j] * xc[j] * n)
    });
    // Issue #2632: the bias gradient is the column sum of the cotangent.
    let db = column_sums(cotangent_row);
    let vjp = |wrt| {
        // SAFETY: `w` is a live array for the whole call.
        unsafe { mlxcel_core::fast_layer_norm_vjp_for_test(&x, &*w, &b, &g, EPS, wrt) }
            .expect("the vjp builds")
    };
    assert_gpu_matches("layer_norm vjp dx", &|| vjp(0), &dx, &SHAPE, 1e-4, 1e-4);
    assert_gpu_matches(
        "layer_norm vjp dw",
        &|| vjp(1),
        &dw,
        &[AXIS as i32],
        1e-3,
        1.0,
    );
    assert_gpu_matches(
        "layer_norm vjp db",
        &|| vjp(2),
        &db,
        &[AXIS as i32],
        1e-3,
        1.0,
    );
}

// ---- large: the 512 and 1024-thread softmax arms -----------------------------

/// Softmax of `rows` copies of one `axis`-long f16 row, sampled at the first
/// row, both sides of the grid-x cap and the last row. Every row's softmax is
/// the same, and an unprocessed row would still hold its input (the output
/// shares the contiguous copy's buffer), which is never a softmax of it.
fn assert_softmax_arm(rows: i32, axis: i32, block: i64) {
    let pattern: Vec<f32> = (0..axis).map(|j| (j % 7) as f32).collect();
    let max = 6.0_f32;
    let denom: f32 = pattern.iter().map(|v| (v - max).exp()).sum();
    let expected: Vec<f32> = pattern.iter().map(|v| (v - max).exp() / denom).collect();
    let _guard = DefaultDeviceGuard::gpu();
    let row = mlxcel_core::astype(&vector(&pattern), dtype::FLOAT16);
    let x = mlxcel_core::broadcast_to(&mlxcel_core::reshape(&row, &[1, axis]), &[rows, axis]);
    let out = mlxcel_core::softmax(&x, -1);
    if let Err(e) = mlxcel_core::try_eval(&out) {
        panic!("softmax ({block}-thread arm) launch failed: {e}");
    }
    let cap = (u32::MAX as i64 / block) as i32;
    let want = mlxcel_core::astype(
        &mlxcel_core::astype(&vector(&expected), dtype::FLOAT16),
        dtype::FLOAT32,
    );
    for r in [0, cap - 1, cap, rows - 1] {
        let got = mlxcel_core::astype(
            &mlxcel_core::reshape(&mlxcel_core::slice(&out, &[r, 0], &[r + 1, axis]), &[axis]),
            dtype::FLOAT32,
        );
        let close = mlxcel_core::allclose(&got, &want, 1e-2, 1e-3);
        assert!(
            mlxcel_core::item_bool(&close),
            "softmax ({block}-thread arm): row {r} differs from the host reference"
        );
    }
}

#[test]
#[ignore = "allocates a ~17 GiB f16 array; run with --ignored on a device with ~20 GiB free"]
fn softmax_512_thread_arm_past_the_row_grid_cap() {
    rocm_only!();
    let _lock = lock_default_device();
    // 2^23 + 1 rows of 1025: the smallest axis the 512-thread arm takes.
    assert_softmax_arm((1 << 23) + 1, 1025, 512);
}

#[test]
#[ignore = "allocates a ~17 GiB f16 array; run with --ignored on a device with ~20 GiB free"]
fn softmax_1024_thread_arm_past_the_row_grid_cap() {
    rocm_only!();
    let _lock = lock_default_device();
    // 2^22 + 1 rows of 2049: the smallest axis the 1024-thread arm takes.
    assert_softmax_arm((1 << 22) + 1, 2049, 1024);
}
