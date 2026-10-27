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

//! The bias gradient of `fast::layer_norm` on ROCm (issue #2632,
//! `LOCAL_FIXES.md` item 64).
//!
//! `LayerNormVJP::eval_gpu` allocated the bias gradient and never wrote it,
//! so `d/db` was whatever the allocator handed back. It is now the column sum
//! of the cotangent over every axis but the last, the fallback in
//! `mlx/fast.cpp`, computed before the VJP kernel can overwrite a donated
//! cotangent. These tests compare it with that sum taken on the host in f64,
//! for f32, f16 and bf16, with and without a weight, on a 2-D input whose
//! row count takes the small column-reduce kernel and on a 3-D input whose
//! row count takes the looped one. The case past the row-grid cap is
//! `row_grid::layer_norm_vjp_past_the_row_grid_cap_matches_host`.
//!
//! The donated-cotangent tests (issue #2663, `LOCAL_FIXES.md` item 68) reach
//! the reason `gb` is reduced first. A caller-held contiguous cotangent is
//! never donatable, so they pass a transposed one: `check_input` copies a
//! cotangent that is not row-contiguous and marks the copy donatable
//! (`donate_g |= g_copied`). With a held contiguous `x` the copy is donated
//! into `gx` (`g_in_gx`); with a transposed `x` as well, the copy of `x` takes
//! `gx` and, given a weight, the cotangent copy is donated into the
//! weight-gradient temporary (`g_in_gw`). In both the VJP kernel overwrites
//! the cotangent, so a column reduce enqueued after it sums `gx` or the
//! per-row weight gradient instead. `gx`, `gw` and `gb` are compared with a
//! host f64 reference.

use mlxcel_core::dtype;
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr};

use super::on_rocm;

const EPS: f32 = 1e-5;

/// Element `i` of an input: a 97-value cycle, a multiple of 1/16 in [-3, 3],
/// exact in f32, f16 and bf16.
fn val(i: usize, salt: usize) -> f64 {
    ((i * 7 + salt) % 97) as f64 / 16.0 - 3.0
}

fn typed(shape: &[i32], salt: usize, dt: i32) -> UniquePtr<MlxArray> {
    let len: usize = shape.iter().map(|&d| d as usize).product();
    let data: Vec<f32> = (0..len).map(|i| val(i, salt) as f32).collect();
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(&data, shape), dt)
}

/// Sum of the cotangent over every axis but the last, in f64.
fn column_sums(shape: &[i32], salt: usize) -> Vec<f32> {
    let axis = *shape.last().expect("non-empty shape") as usize;
    let len: usize = shape.iter().map(|&d| d as usize).product();
    let mut acc = vec![0.0_f64; axis];
    for i in 0..len {
        acc[i % axis] += val(i, salt);
    }
    acc.iter().map(|&v| v as f32).collect()
}

/// `(rtol, atol)` of the comparison: the GPU sums in the input dtype.
fn tolerance(dt: i32) -> (f64, f64) {
    match dt {
        dtype::FLOAT32 => (1e-5, 1e-4),
        dtype::FLOAT16 => (2e-3, 0.25),
        _ => (1e-2, 1.0),
    }
}

fn dtype_name(dt: i32) -> &'static str {
    match dt {
        dtype::FLOAT32 => "f32",
        dtype::FLOAT16 => "f16",
        _ => "bf16",
    }
}

/// Checks `d/db` of `layer_norm(x, weight?, bias)` against the host column
/// sums of the cotangent.
fn assert_bias_grad(shape: &[i32], dt: i32, with_weight: bool) {
    let axis = *shape.last().expect("non-empty shape");
    let what = format!(
        "layer_norm vjp db, {} {shape:?} {}",
        dtype_name(dt),
        if with_weight {
            "with weight"
        } else {
            "without weight"
        }
    );
    let _guard = DefaultDeviceGuard::gpu();
    let x = typed(shape, 0, dt);
    let g = typed(shape, 41, dt);
    let w = typed(&[axis], 3, dt);
    let b = typed(&[axis], 5, dt);
    let weight: *const MlxArray = if with_weight { &*w } else { std::ptr::null() };
    // SAFETY: `weight` is null or points to `w`, live for the whole call.
    let db = unsafe { mlxcel_core::fast_layer_norm_vjp_for_test(&x, weight, &b, &g, EPS, 2) }
        .expect("the vjp builds");
    if let Err(e) = mlxcel_core::try_eval(&db) {
        panic!("{what}: GPU launch failed: {e}");
    }
    assert_eq!(mlxcel_core::array_shape(&db), [axis], "{what}: shape");
    assert_eq!(mlxcel_core::array_dtype(&db), dt, "{what}: dtype");
    let expected = column_sums(shape, 41);
    let got: Vec<f32> = mlxcel_core::array_to_raw_bytes(&mlxcel_core::astype(&db, dtype::FLOAT32))
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect();
    let (rtol, atol) = tolerance(dt);
    for (j, (&got, &want)) in got.iter().zip(&expected).enumerate() {
        let tol = atol + rtol * f64::from(want.abs());
        assert!(
            f64::from(got - want).abs() <= tol && got.is_finite(),
            "{what}: element {j} is {got}, expected {want} (got {got:?}, expected {expected:?})"
        );
    }
}

/// Inputs of the donated-cotangent tests, as `[rows, axis]` values.
struct Donated {
    rows: usize,
    axis: usize,
    x: Vec<f64>,
    w: Vec<f64>,
    g: Vec<f64>,
}

impl Donated {
    /// `x[r, c]` and `g[r, c]` come from the column-major index `c * rows + r`,
    /// the flat index of the `[axis, rows]` buffer a transposed view reads.
    fn new(rows: usize, axis: usize) -> Self {
        let at = |salt| {
            (0..rows * axis)
                .map(|i| val((i % axis) * rows + i / axis, salt))
                .collect::<Vec<_>>()
        };
        Self {
            rows,
            axis,
            x: at(0),
            w: (0..axis).map(|i| val(i, 3)).collect(),
            g: at(41),
        }
    }

    /// `[rows, axis]` array of `values`: row-contiguous, or a transposed view
    /// of an `[axis, rows]` buffer.
    fn array(&self, values: &[f64], transposed: bool, dt: i32) -> UniquePtr<MlxArray> {
        let (rows, axis) = (self.rows as i32, self.axis as i32);
        if transposed {
            let col_major: Vec<f32> = (0..self.rows * self.axis)
                .map(|i| values[(i % self.rows) * self.axis + i / self.rows] as f32)
                .collect();
            let buf =
                mlxcel_core::astype(&mlxcel_core::from_slice_f32(&col_major, &[axis, rows]), dt);
            mlxcel_core::transpose(&buf)
        } else {
            let row_major: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            mlxcel_core::astype(&mlxcel_core::from_slice_f32(&row_major, &[rows, axis]), dt)
        }
    }

    /// `(gx, gw, gb)` of `layer_norm(x, weight?, bias)` in f64.
    fn reference(&self, with_weight: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (rows, axis) = (self.rows, self.axis);
        let n = axis as f64;
        let mut gx = vec![0.0; rows * axis];
        let mut gw = vec![0.0; axis];
        let mut gb = vec![0.0; axis];
        for r in 0..rows {
            let x = &self.x[r * axis..(r + 1) * axis];
            let g = &self.g[r * axis..(r + 1) * axis];
            let mean = x.iter().sum::<f64>() / n;
            let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            let inv = 1.0 / (var + f64::from(EPS)).sqrt();
            let xhat: Vec<f64> = x.iter().map(|v| (v - mean) * inv).collect();
            let wg: Vec<f64> = (0..axis)
                .map(|c| if with_weight { self.w[c] * g[c] } else { g[c] })
                .collect();
            let mean_wg = wg.iter().sum::<f64>() / n;
            let mean_wg_xhat = (0..axis).map(|c| wg[c] * xhat[c]).sum::<f64>() / n;
            for c in 0..axis {
                gx[r * axis + c] = inv * (wg[c] - mean_wg - xhat[c] * mean_wg_xhat);
                gw[c] += g[c] * xhat[c];
                gb[c] += g[c];
            }
        }
        (gx, gw, gb)
    }
}

fn host_f32(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::array_to_raw_bytes(&mlxcel_core::astype(a, dtype::FLOAT32))
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

/// `(rtol, atol)` of a gradient whose largest reference magnitude is `scale`:
/// the GPU computes in the input dtype and rounds the result to it.
fn grad_tolerance(dt: i32, scale: f64) -> (f64, f64) {
    match dt {
        dtype::FLOAT32 => (1e-4, 1e-4 * scale.max(1.0)),
        dtype::FLOAT16 => (4e-3, 4e-3 * scale.max(1.0)),
        _ => (2e-2, 2e-2 * scale.max(1.0)),
    }
}

/// Checks `gx`, `gw` (with a weight) and `gb` of a layer norm whose
/// cotangent is a transposed `[rows, axis]` view, so that the VJP donates its
/// contiguous copy into `gx`, or, with `x_transposed` and a weight, into the
/// weight-gradient temporary.
fn assert_donated_cotangent(
    rows: usize,
    axis: usize,
    dt: i32,
    with_weight: bool,
    x_transposed: bool,
) {
    let what = format!(
        "layer_norm vjp, donated cotangent, {} [{rows}, {axis}] {}, {} x",
        dtype_name(dt),
        if with_weight {
            "with weight"
        } else {
            "without weight"
        },
        if x_transposed {
            "transposed"
        } else {
            "held contiguous"
        },
    );
    let _guard = DefaultDeviceGuard::gpu();
    let case = Donated::new(rows, axis);
    let x = case.array(&case.x, x_transposed, dt);
    let g = case.array(&case.g, true, dt);
    let w = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &case.w.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &[axis as i32],
        ),
        dt,
    );
    let b = typed(&[axis as i32], 5, dt);
    let weight: *const MlxArray = if with_weight { &*w } else { std::ptr::null() };
    let (gx_want, gw_want, gb_want) = case.reference(with_weight);
    let mut checks = vec![(0, "gx", gx_want), (2, "gb", gb_want)];
    if with_weight {
        checks.push((1, "gw", gw_want));
    }
    for (wrt, name, want) in checks {
        // SAFETY: `weight` is null or points to `w`, live for the whole call.
        let grad =
            unsafe { mlxcel_core::fast_layer_norm_vjp_for_test(&x, weight, &b, &g, EPS, wrt) }
                .expect("the vjp builds");
        if let Err(e) = mlxcel_core::try_eval(&grad) {
            panic!("{what}: {name}: GPU launch failed: {e}");
        }
        assert_eq!(mlxcel_core::array_dtype(&grad), dt, "{what}: {name} dtype");
        let got = host_f32(&grad);
        assert_eq!(got.len(), want.len(), "{what}: {name} size");
        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let (rtol, atol) = grad_tolerance(dt, scale);
        for (j, (&got, &want)) in got.iter().zip(&want).enumerate() {
            let tol = atol + rtol * want.abs();
            assert!(
                got.is_finite() && (f64::from(got) - want).abs() <= tol,
                "{what}: {name} element {j} is {got}, expected {want} (tolerance {tol})"
            );
        }
    }
}

macro_rules! rocm_only {
    () => {
        if !on_rocm() {
            eprintln!("skipping: not running on a ROCm device");
            return;
        }
    };
}

const DTYPES: [i32; 3] = [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16];

/// 24 rows of 8: the column reduce takes `col_reduce_small`.
const SMALL: [i32; 2] = [24, 8];
/// 4 x 37 rows of 33: 148 rows take `col_reduce_looped`, and the 3-D input
/// checks the reduction runs over every leading axis.
const LOOPED: [i32; 3] = [4, 37, 33];

#[test]
fn layer_norm_vjp_bias_grad_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    for dt in DTYPES {
        assert_bias_grad(&SMALL, dt, true);
        assert_bias_grad(&LOOPED, dt, true);
    }
}

#[test]
fn layer_norm_vjp_bias_grad_without_weight_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    for dt in DTYPES {
        assert_bias_grad(&SMALL, dt, false);
        assert_bias_grad(&LOOPED, dt, false);
    }
}

/// 24 rows of 8 take `col_reduce_small`; 148 rows of 33 take
/// `col_reduce_looped`.
const DONATED: [(usize, usize); 2] = [(24, 8), (148, 33)];

#[test]
fn layer_norm_vjp_with_cotangent_donated_into_gx_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    for dt in DTYPES {
        for (rows, axis) in DONATED {
            assert_donated_cotangent(rows, axis, dt, true, false);
            assert_donated_cotangent(rows, axis, dt, false, false);
        }
    }
}

#[test]
fn layer_norm_vjp_with_cotangent_donated_into_gw_matches_host() {
    rocm_only!();
    let _lock = lock_default_device();
    for dt in DTYPES {
        for (rows, axis) in DONATED {
            assert_donated_cotangent(rows, axis, dt, true, true);
            // No weight: no weight-gradient temporary, so `gx` takes the copy
            // of `x` and the cotangent copy is only a temporary.
            assert_donated_cotangent(rows, axis, dt, false, true);
        }
    }
}
