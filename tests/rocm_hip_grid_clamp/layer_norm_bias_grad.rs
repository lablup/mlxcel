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
