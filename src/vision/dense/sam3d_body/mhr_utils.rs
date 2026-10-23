// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Exact rotation and continuous-pose conversions from the pinned MHR model.
use super::layers::Array;
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray};
#[path = "mhr_rotations.rs"]
mod rotations;
pub use rotations::{euler_xyz_to_rotmat, quat_to_rotmat, rotmat_to_quat};
#[path = "mhr_indices.rs"]
mod indices;
pub use indices::{compact_body, compact_hand};

pub(super) fn scalar(x: f32) -> Array {
    mx::from_slice_f32(&[x], &[])
}
pub(super) fn add_scalar(x: &MlxArray, v: f32) -> Array {
    mx::add(x, &scalar(v))
}
pub(super) fn one_minus(x: &MlxArray) -> Array {
    mx::subtract(&scalar(1.), x)
}
pub(super) fn last(x: &MlxArray, start: i32, end: i32) -> Array {
    let mut stops = mx::array_shape(x);
    let mut starts = vec![0; stops.len()];
    let axis = stops.len() - 1;
    starts[axis] = start;
    stops[axis] = end;
    mx::slice(x, &starts, &stops)
}
pub(super) fn component(x: &MlxArray, i: i32) -> Array {
    mx::squeeze_axis(&last(x, i, i + 1), -1)
}
pub(super) fn entry(x: &MlxArray, i: i32, j: i32) -> Array {
    let mut stops = mx::array_shape(x);
    let mut starts = vec![0; stops.len()];
    let n = stops.len();
    starts[n - 2] = i;
    starts[n - 1] = j;
    stops[n - 2] = i + 1;
    stops[n - 1] = j + 1;
    mx::reshape(&mx::slice(x, &starts, &stops), &stops[..n - 2])
}
pub(super) fn einsum(expression: &str, args: &[&MlxArray]) -> Array {
    let pointers: Vec<*const MlxArray> = args.iter().map(|a| *a as *const MlxArray).collect();
    // SAFETY: every operand is borrowed and remains alive throughout the call.
    unsafe { mx::einsum(expression, &pointers) }
}
pub(super) fn check_last(x: &MlxArray, size: i32) -> Result<()> {
    let s = mx::array_shape(x);
    ensure!(
        s.last() == Some(&size)
            && s.iter().all(|&d| d > 0)
            && mx::array_dtype(x) == mx::dtype::FLOAT32,
        "MHR expects nonempty F32 [...,{size}], got {s:?}"
    );
    Ok(())
}
pub(super) fn check_matrix(x: &MlxArray) -> Result<()> {
    let s = mx::array_shape(x);
    ensure!(
        s.ends_with(&[3, 3])
            && s.iter().all(|&d| d > 0)
            && mx::array_dtype(x) == mx::dtype::FLOAT32,
        "MHR expects F32 [...,3,3], got {s:?}"
    );
    Ok(())
}
fn cross(a: &MlxArray, b: &MlxArray) -> Array {
    mx::stack_owned(
        &(0..3)
            .map(|i| {
                let j = (i + 1) % 3;
                let k = (i + 2) % 3;
                mx::subtract(
                    &mx::multiply(&component(a, j), &component(b, k)),
                    &mx::multiply(&component(a, k), &component(b, j)),
                )
            })
            .collect::<Vec<_>>(),
        -1,
    )
}
pub fn rot6d_to_rotmat(x: &MlxArray) -> Result<Array> {
    check_last(x, 6)?;
    let x1 = last(x, 0, 3);
    let x2 = last(x, 3, 6);
    let x1 = mx::divide(&x1, &add_scalar(&mx::linalg_norm(&x1, -1, true), 1e-8));
    let z = cross(&x1, &x2);
    let z = mx::divide(&z, &add_scalar(&mx::linalg_norm(&z, -1, true), 1e-8));
    let y = cross(&z, &x1);
    Ok(mx::stack_owned(&[x1, y, z], -1))
}
fn euler_parts(r: &MlxArray) -> [Array; 3] {
    let sy = mx::sqrt(&mx::add(
        &mx::power(&entry(r, 0, 0), &scalar(2.)),
        &mx::power(&entry(r, 1, 0), &scalar(2.)),
    ));
    let singular = mx::astype(&mx::less(&sy, &scalar(1e-6)), mx::dtype::FLOAT32);
    let nonsingular = one_minus(&singular);
    let ex = mx::add(
        &mx::multiply(&mx::arctan2(&entry(r, 2, 1), &entry(r, 2, 2)), &nonsingular),
        &mx::multiply(
            &mx::arctan2(&mx::negative(&entry(r, 1, 2)), &entry(r, 1, 1)),
            &singular,
        ),
    );
    let ey = mx::arctan2(&mx::negative(&entry(r, 2, 0)), &sy);
    let ez = mx::multiply(&mx::arctan2(&entry(r, 1, 0), &entry(r, 0, 0)), &nonsingular);
    [ex, ey, ez]
}
pub fn rotmat_to_euler_zyx(r: &MlxArray) -> Result<Array> {
    check_matrix(r)?;
    let [x, y, z] = euler_parts(r);
    Ok(mx::stack_owned(&[z, y, x], -1))
}
pub fn batch_xyz_from_6d(x: &MlxArray) -> Result<Array> {
    Ok(mx::stack_owned(&euler_parts(&*rot6d_to_rotmat(x)?), -1))
}
pub(super) fn sincos_to_angle(x: &MlxArray) -> Array {
    mx::arctan2(&component(x, 0), &component(x, 1))
}
