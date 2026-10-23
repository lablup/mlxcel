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

//! MoGe3 view-plane, focal/shift recovery, and pinhole geometry.

use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::MAX_RUNTIME_PIXELS;
use super::policy::Policy;

fn scalar(value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&[value], &[1])
}
fn sum_last_two(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let prefix = &shape[..shape.len() - 2];
    let mut flat = prefix.to_vec();
    flat.push(shape[shape.len() - 2] * shape[shape.len() - 1]);
    mlxcel_core::sum_axis(&mlxcel_core::reshape(x, &flat), -1, false)
}
fn max_scalar(x: &MlxArray, value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::maximum(x, &scalar(value))
}

fn view_plane_axes(
    width: usize,
    height: usize,
    aspect: Option<f64>,
    policy: Policy,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let [u0, u1, v0, v1] = policy.endpoints(width, height, aspect);
    let u = mlxcel_core::linspace(u0, u1, width as i32, mlxcel_core::dtype::FLOAT32);
    let v = mlxcel_core::linspace(v0, v1, height as i32, mlxcel_core::dtype::FLOAT32);
    (u, v)
}

pub(super) fn normalized_view_plane_uv(
    width: usize,
    height: usize,
    aspect: Option<f32>,
) -> Result<UniquePtr<MlxArray>> {
    normalized_view_plane_uv_with_policy(width, height, aspect.map(f64::from), Policy::Standard)
}

pub(super) fn normalized_view_plane_uv_with_policy(
    width: usize,
    height: usize,
    aspect: Option<f64>,
    policy: Policy,
) -> Result<UniquePtr<MlxArray>> {
    ensure!(
        width > 0 && height > 0,
        "MoGe3 view plane dimensions must be positive"
    );
    ensure!(
        width <= i32::MAX as usize && height <= i32::MAX as usize,
        "MoGe3 view plane dimensions must fit int32"
    );
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 view plane size overflow"))?;
    ensure!(
        pixels <= MAX_RUNTIME_PIXELS,
        "MoGe3 view plane exceeds {MAX_RUNTIME_PIXELS} pixels"
    );
    if let Some(value) = aspect {
        ensure!(
            value.is_finite() && value > 0.0,
            "MoGe3 view plane aspect must be finite and positive"
        );
    }
    let (u, v) = view_plane_axes(width, height, aspect, policy);
    let u = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&u, &[1, width as i32]),
        &[height as i32, width as i32],
    );
    let v = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&v, &[height as i32, 1]),
        &[height as i32, width as i32],
    );
    Ok(mlxcel_core::stack_owned(&[u, v], -1))
}

struct LmState {
    shift: UniquePtr<MlxArray>,
    lambda: UniquePtr<MlxArray>,
    done: UniquePtr<MlxArray>,
    cost: UniquePtr<MlxArray>,
    gradient: UniquePtr<MlxArray>,
    hessian: UniquePtr<MlxArray>,
    focal: UniquePtr<MlxArray>,
}

fn lm_evaluate(
    shift: &MlxArray,
    uv: &MlxArray,
    xy: &MlxArray,
    z: &MlxArray,
    fixed: Option<&MlxArray>,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let shifted = mlxcel_core::add(z, shift);
    let inv = mlxcel_core::divide(&scalar(1.0), &shifted);
    let inv3 = mlxcel_core::expand_dims(&inv, -1);
    let p = mlxcel_core::multiply(xy, &inv3);
    let q = mlxcel_core::negative(&mlxcel_core::multiply(&p, &inv3));
    let (f, df) = if let Some(fixed) = fixed {
        (mlxcel_core::copy(fixed), mlxcel_core::zeros_like(fixed))
    } else {
        let a = sum_last_two(&mlxcel_core::multiply(&p, uv));
        let b = max_scalar(&sum_last_two(&mlxcel_core::multiply(&p, &p)), 1e-12);
        let da = sum_last_two(&mlxcel_core::multiply(&q, uv));
        let db = mlxcel_core::multiply(&scalar(2.0), &sum_last_two(&mlxcel_core::multiply(&p, &q)));
        let f = mlxcel_core::divide(&a, &b);
        let df = mlxcel_core::divide(
            &mlxcel_core::subtract(
                &mlxcel_core::multiply(&da, &b),
                &mlxcel_core::multiply(&a, &db),
            ),
            &mlxcel_core::multiply(&b, &b),
        );
        (f, df)
    };
    let fb = mlxcel_core::reshape(&f, &[-1, 1, 1]);
    let dfb = mlxcel_core::reshape(&df, &[-1, 1, 1]);
    let residual = mlxcel_core::subtract(&mlxcel_core::multiply(&fb, &p), uv);
    let jacobian = mlxcel_core::add(
        &mlxcel_core::multiply(&dfb, &p),
        &mlxcel_core::multiply(&fb, &q),
    );
    let cost = sum_last_two(&mlxcel_core::multiply(&residual, &residual));
    let gradient = sum_last_two(&mlxcel_core::multiply(&jacobian, &residual));
    let hessian = sum_last_two(&mlxcel_core::multiply(&jacobian, &jacobian));
    (cost, gradient, hessian, f)
}

fn solve_focal_shift(
    uv: &MlxArray,
    xyz: &MlxArray,
    fixed: Option<&MlxArray>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(xyz);
    let batch = shape[0];
    let xy = mlxcel_core::slice(xyz, &[0, 0, 0], &[batch, shape[1], 2]);
    let z = mlxcel_core::reshape(
        &mlxcel_core::slice(xyz, &[0, 0, 2], &[batch, shape[1], 3]),
        &[batch, shape[1]],
    );
    let shift = mlxcel_core::zeros(&[batch, 1], mlxcel_core::dtype::FLOAT32);
    let lambda = mlxcel_core::full_f32(&[batch], 1e-3, mlxcel_core::dtype::FLOAT32);
    let done = mlxcel_core::zeros(&[batch], mlxcel_core::dtype::BOOL);
    let (cost, gradient, hessian, focal) = lm_evaluate(&shift, uv, &xy, &z, fixed);
    let mut state = LmState {
        shift,
        lambda,
        done,
        cost,
        gradient,
        hessian,
        focal,
    };
    for _ in 0..30 {
        let denom = max_scalar(
            &mlxcel_core::multiply(
                &state.hessian,
                &mlxcel_core::add(&scalar(1.0), &state.lambda),
            ),
            1e-30,
        );
        let step = mlxcel_core::negative(&mlxcel_core::divide(&state.gradient, &denom));
        let trial = mlxcel_core::add(&state.shift, &mlxcel_core::reshape(&step, &[batch, 1]));
        let (cost_t, g_t, h_t, f_t) = lm_evaluate(&trial, uv, &xy, &z, fixed);
        let accept = mlxcel_core::logical_and(
            &mlxcel_core::less(&cost_t, &state.cost),
            &mlxcel_core::logical_not(&state.done),
        );
        let decrease = mlxcel_core::subtract(&state.cost, &cost_t);
        let converged = mlxcel_core::logical_and(
            &accept,
            &mlxcel_core::less_equal(&decrease, &mlxcel_core::multiply(&scalar(1e-3), &cost_t)),
        );
        let old_shift = mlxcel_core::reshape(&state.shift, &[batch]);
        let tiny = mlxcel_core::less(
            &mlxcel_core::abs(&step),
            &mlxcel_core::multiply(
                &scalar(1e-8),
                &max_scalar(&mlxcel_core::abs(&old_shift), 1.0),
            ),
        );
        let accept2 = mlxcel_core::reshape(&accept, &[batch, 1]);
        state.shift = mlxcel_core::where_cond(&accept2, &trial, &state.shift);
        state.cost = mlxcel_core::where_cond(&accept, &cost_t, &state.cost);
        state.gradient = mlxcel_core::where_cond(&accept, &g_t, &state.gradient);
        state.hessian = mlxcel_core::where_cond(&accept, &h_t, &state.hessian);
        state.focal = mlxcel_core::where_cond(&accept, &f_t, &state.focal);
        let adjusted = mlxcel_core::where_cond(
            &accept,
            &mlxcel_core::divide(&state.lambda, &scalar(3.0)),
            &mlxcel_core::multiply(&state.lambda, &scalar(5.0)),
        );
        state.lambda = mlxcel_core::where_cond(&state.done, &state.lambda, &adjusted);
        state.done =
            mlxcel_core::logical_or(&state.done, &mlxcel_core::logical_or(&converged, &tiny));
    }
    (state.focal, mlxcel_core::reshape(&state.shift, &[batch]))
}

fn nearest_indices(input: usize, output: usize) -> Vec<i32> {
    (0..output)
        .map(|i| ((i as f32 * input as f32 / output as f32).floor() as usize).min(input - 1) as i32)
        .collect()
}

pub(super) fn recover_focal_shift(
    points: &MlxArray,
    mask: Option<&MlxArray>,
    focal: Option<&MlxArray>,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    recover_focal_shift_with_policy(points, mask, focal, Policy::Standard)
}

pub(super) fn recover_focal_shift_with_policy(
    points: &MlxArray,
    mask: Option<&MlxArray>,
    focal: Option<&MlxArray>,
    policy: Policy,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    let shape = mlxcel_core::array_shape(points);
    ensure!(
        shape.len() == 4 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0 && shape[3] == 3,
        "MoGe3 focal recovery expects [B,H,W,3]"
    );
    ensure!(
        matches!(
            mlxcel_core::array_dtype(points),
            mlxcel_core::dtype::FLOAT16
                | mlxcel_core::dtype::BFLOAT16
                | mlxcel_core::dtype::FLOAT32
        ),
        "MoGe3 focal recovery points must be floating point"
    );
    let (b, h, w) = (shape[0], shape[1] as usize, shape[2] as usize);
    if let Some(mask) = mask {
        ensure!(
            mlxcel_core::array_shape(mask) == [b, shape[1], shape[2]]
                && mlxcel_core::array_dtype(mask) == mlxcel_core::dtype::BOOL,
            "MoGe3 focal recovery mask must be bool [B,H,W]"
        );
    }
    if let Some(focal) = focal {
        ensure!(
            mlxcel_core::array_shape(focal) == [b]
                && matches!(
                    mlxcel_core::array_dtype(focal),
                    mlxcel_core::dtype::FLOAT16
                        | mlxcel_core::dtype::BFLOAT16
                        | mlxcel_core::dtype::FLOAT32
                ),
            "MoGe3 fixed focal must be floating point [B]"
        );
        let valid = mlxcel_core::logical_and(
            &mlxcel_core::isfinite(focal),
            &mlxcel_core::greater(focal, &scalar(0.0)),
        );
        ensure!(
            mlxcel_core::item_bool(&mlxcel_core::all_all(&valid)),
            "MoGe3 fixed focal must be finite and positive"
        );
    }
    let oh = 64;
    let ow = 64;
    let ii = mlxcel_core::from_slice_i32(&nearest_indices(h, oh), &[oh as i32]);
    let jj = mlxcel_core::from_slice_i32(&nearest_indices(w, ow), &[ow as i32]);
    let sampled = mlxcel_core::take(&mlxcel_core::take(points, &ii, 1), &jj, 2);
    let xyz = mlxcel_core::reshape(
        &mlxcel_core::astype(&sampled, mlxcel_core::dtype::FLOAT32),
        &[b, (oh * ow) as i32, 3],
    );
    let (u, v) = view_plane_axes(w, h, None, policy);
    let u = mlxcel_core::take(&u, &jj, 0);
    let v = mlxcel_core::take(&v, &ii, 0);
    let u = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&u, &[1, ow as i32]),
        &[oh as i32, ow as i32],
    );
    let v = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&v, &[oh as i32, 1]),
        &[oh as i32, ow as i32],
    );
    let uv = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(
            &mlxcel_core::stack_owned(&[u, v], -1),
            &[1, (oh * ow) as i32, 2],
        ),
        &[b, (oh * ow) as i32, 2],
    );
    let (uv, xyz, valid_count) = if let Some(mask) = mask {
        let valid = mlxcel_core::reshape(
            &mlxcel_core::take(&mlxcel_core::take(mask, &ii, 1), &jj, 2),
            &[b, (oh * ow) as i32, 1],
        );
        let count = mlxcel_core::sum_axis(
            &mlxcel_core::astype(&valid, mlxcel_core::dtype::INT32),
            -2,
            false,
        );
        let uv = mlxcel_core::where_cond(&valid, &uv, &mlxcel_core::zeros_like(&uv));
        let fallback = mlxcel_core::broadcast_to(
            &mlxcel_core::from_slice_f32(&[0.0, 0.0, 1.0], &[1, 1, 3]),
            &[b, (oh * ow) as i32, 3],
        );
        let xyz = mlxcel_core::where_cond(&valid, &xyz, &fallback);
        (uv, xyz, count)
    } else {
        (
            uv,
            xyz,
            mlxcel_core::full_f32(&[b, 1], (oh * ow) as f32, mlxcel_core::dtype::FLOAT32),
        )
    };
    let fixed = focal
        .map(|v| mlxcel_core::reshape(&mlxcel_core::astype(v, mlxcel_core::dtype::FLOAT32), &[b]));
    let (mut f, mut shift) = if policy == Policy::Objects && fixed.is_none() {
        // Match upstream's eager initializer and one compiled boundary per LM
        // step. Sampling and the fewer-than-two-valid-points fallback stay here.
        let solved = mlxcel_core::compiled_objects_focal_shift(&uv, &xyz)?;
        ensure!(
            mlxcel_core::array_shape(&solved) == [b, 2]
                && mlxcel_core::array_dtype(&solved) == mlxcel_core::dtype::FLOAT32,
            "Objects focal solver must return F32 [B,2]"
        );
        (
            mlxcel_core::reshape(&mlxcel_core::slice(&solved, &[0, 0], &[b, 1]), &[b]),
            mlxcel_core::reshape(&mlxcel_core::slice(&solved, &[0, 1], &[b, 2]), &[b]),
        )
    } else {
        solve_focal_shift(&uv, &xyz, fixed.as_deref())
    };
    let degenerate = mlxcel_core::less(&mlxcel_core::reshape(&valid_count, &[b]), &scalar(2.0));
    shift = mlxcel_core::where_cond(&degenerate, &mlxcel_core::zeros_like(&shift), &shift);
    if focal.is_none() {
        f = mlxcel_core::where_cond(&degenerate, &mlxcel_core::ones_like(&f), &f);
    }
    Ok((f, shift))
}

pub(super) fn intrinsics_from_focal(focal: &MlxArray, aspect: f32) -> UniquePtr<MlxArray> {
    let fx = mlxcel_core::multiply(
        focal,
        &scalar((1.0 + aspect * aspect).sqrt() / (2.0 * aspect)),
    );
    let fy = mlxcel_core::multiply(focal, &scalar((1.0 + aspect * aspect).sqrt() / 2.0));
    let shape = mlxcel_core::array_shape(focal);
    let b = shape[0];
    let zero = mlxcel_core::zeros(&[b], mlxcel_core::dtype::FLOAT32);
    let half = mlxcel_core::full_f32(&[b], 0.5, mlxcel_core::dtype::FLOAT32);
    let one = mlxcel_core::ones(&[b], mlxcel_core::dtype::FLOAT32);
    let r0 = mlxcel_core::stack_owned(
        &[fx, mlxcel_core::copy(&zero), mlxcel_core::copy(&half)],
        -1,
    );
    let r1 = mlxcel_core::stack_owned(&[mlxcel_core::copy(&zero), fy, half], -1);
    let r2 = mlxcel_core::stack_owned(
        &[
            zero,
            mlxcel_core::zeros(&[b], mlxcel_core::dtype::FLOAT32),
            one,
        ],
        -1,
    );
    mlxcel_core::stack_owned(&[r0, r1, r2], -2)
}

/// Project depth using the canonical diagonal intrinsics produced above.
///
/// This helper is intentionally crate-private because arbitrary 3x3 camera
/// matrices require the general inverse projection used by upstream.
pub(super) fn depth_map_to_point_map(
    depth: &MlxArray,
    intrinsics: &MlxArray,
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(depth);
    ensure!(
        shape.len() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
        "MoGe3 depth must be [B,H,W]"
    );
    let (b, h, w) = (shape[0], shape[1], shape[2]);
    let kshape = mlxcel_core::array_shape(intrinsics);
    ensure!(kshape == [b, 3, 3], "MoGe3 intrinsics must be [B,3,3]");
    let u = mlxcel_core::divide(
        &mlxcel_core::add(&mlxcel_core::arange_f32(0.0, w as f32, 1.0), &scalar(0.5)),
        &scalar(w as f32),
    );
    let v = mlxcel_core::divide(
        &mlxcel_core::add(&mlxcel_core::arange_f32(0.0, h as f32, 1.0), &scalar(0.5)),
        &scalar(h as f32),
    );
    let u = mlxcel_core::broadcast_to(&mlxcel_core::reshape(&u, &[1, 1, w]), &[b, h, w]);
    let v = mlxcel_core::broadcast_to(&mlxcel_core::reshape(&v, &[1, h, 1]), &[b, h, w]);
    let elem = |row: i32, col: i32| {
        mlxcel_core::reshape(
            &mlxcel_core::slice(intrinsics, &[0, row, col], &[b, row + 1, col + 1]),
            &[b, 1, 1],
        )
    };
    let x = mlxcel_core::divide(&mlxcel_core::subtract(&u, &elem(0, 2)), &elem(0, 0));
    let y = mlxcel_core::divide(&mlxcel_core::subtract(&v, &elem(1, 2)), &elem(1, 1));
    Ok(mlxcel_core::stack_owned(
        &[
            mlxcel_core::multiply(&x, depth),
            mlxcel_core::multiply(&y, depth),
            mlxcel_core::copy(depth),
        ],
        -1,
    ))
}

#[cfg(test)]
#[path = "geometry_solver_tests.rs"]
mod solver_tests;
