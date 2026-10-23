// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Quaternion and Euler conversions; preserve pinned arithmetic association.
use super::{Array, add_scalar, check_last, check_matrix, component, entry, one_minus, scalar};
use anyhow::Result;
use mlxcel_core::{self as mx, MlxArray};

pub fn quat_to_rotmat(q: &MlxArray) -> Result<Array> {
    check_last(q, 4)?;
    let x = component(q, 0);
    let y = component(q, 1);
    let z = component(q, 2);
    let w = component(q, 3);
    let x2 = mx::multiply(&x, &x);
    let y2 = mx::multiply(&y, &y);
    let z2 = mx::multiply(&z, &z);
    let xy = mx::multiply(&x, &y);
    let xz = mx::multiply(&x, &z);
    let yz = mx::multiply(&y, &z);
    let wx = mx::multiply(&w, &x);
    let wy = mx::multiply(&w, &y);
    let wz = mx::multiply(&w, &z);
    let twice = |a: Array| mx::multiply_scalar(&a, 2.);
    let row0 = mx::stack_owned(
        &[
            one_minus(&twice(mx::add(&y2, &z2))),
            twice(mx::subtract(&xy, &wz)),
            twice(mx::add(&xz, &wy)),
        ],
        -1,
    );
    let row1 = mx::stack_owned(
        &[
            twice(mx::add(&xy, &wz)),
            one_minus(&twice(mx::add(&x2, &z2))),
            twice(mx::subtract(&yz, &wx)),
        ],
        -1,
    );
    let row2 = mx::stack_owned(
        &[
            twice(mx::subtract(&xz, &wy)),
            twice(mx::add(&yz, &wx)),
            one_minus(&twice(mx::add(&x2, &y2))),
        ],
        -1,
    );
    Ok(mx::stack_owned(&[row0, row1, row2], -2))
}
pub fn euler_xyz_to_rotmat(a: &MlxArray) -> Result<Array> {
    check_last(a, 3)?;
    let cx = mx::cos(&component(a, 0));
    let sx = mx::sin(&component(a, 0));
    let cy = mx::cos(&component(a, 1));
    let sy = mx::sin(&component(a, 1));
    let cz = mx::cos(&component(a, 2));
    let sz = mx::sin(&component(a, 2));
    let row0 = mx::stack_owned(
        &[
            mx::multiply(&cz, &cy),
            mx::subtract(
                &mx::multiply(&mx::multiply(&cz, &sy), &sx),
                &mx::multiply(&sz, &cx),
            ),
            mx::add(
                &mx::multiply(&mx::multiply(&cz, &sy), &cx),
                &mx::multiply(&sz, &sx),
            ),
        ],
        -1,
    );
    let row1 = mx::stack_owned(
        &[
            mx::multiply(&sz, &cy),
            mx::add(
                &mx::multiply(&mx::multiply(&sz, &sy), &sx),
                &mx::multiply(&cz, &cx),
            ),
            mx::subtract(
                &mx::multiply(&mx::multiply(&sz, &sy), &cx),
                &mx::multiply(&cz, &sx),
            ),
        ],
        -1,
    );
    let row2 = mx::stack_owned(
        &[
            mx::negative(&sy),
            mx::multiply(&cy, &sx),
            mx::multiply(&cy, &cx),
        ],
        -1,
    );
    Ok(mx::stack_owned(&[row0, row1, row2], -2))
}
pub fn rotmat_to_quat(r: &MlxArray) -> Result<Array> {
    check_matrix(r)?;
    let mut output_shape = mx::array_shape(r);
    output_shape.truncate(output_shape.len() - 2);
    output_shape.push(4);
    let r = mx::reshape(r, &[-1, 3, 3]);
    let r00 = entry(&r, 0, 0);
    let r01 = entry(&r, 0, 1);
    let r02 = entry(&r, 0, 2);
    let r10 = entry(&r, 1, 0);
    let r11 = entry(&r, 1, 1);
    let r12 = entry(&r, 1, 2);
    let r20 = entry(&r, 2, 0);
    let r21 = entry(&r, 2, 1);
    let r22 = entry(&r, 2, 2);
    let trace = mx::add(&mx::add(&r00, &r11), &r22);
    let scale = |x: Array| mx::multiply_scalar(&mx::sqrt(&mx::maximum(&x, &scalar(1e-10))), 2.);
    let s0 = scale(add_scalar(&trace, 1.));
    let d0 = add_scalar(&s0, 1e-10);
    let q0 = mx::stack_owned(
        &[
            mx::divide(&mx::subtract(&r21, &r12), &d0),
            mx::divide(&mx::subtract(&r02, &r20), &d0),
            mx::divide(&mx::subtract(&r10, &r01), &d0),
            mx::multiply_scalar(&s0, 0.25),
        ],
        -1,
    );
    let s1 = scale(mx::subtract(
        &mx::subtract(&add_scalar(&r00, 1.), &r11),
        &r22,
    ));
    let d1 = add_scalar(&s1, 1e-10);
    let q1 = mx::stack_owned(
        &[
            mx::multiply_scalar(&s1, 0.25),
            mx::divide(&mx::add(&r01, &r10), &d1),
            mx::divide(&mx::add(&r02, &r20), &d1),
            mx::divide(&mx::subtract(&r21, &r12), &d1),
        ],
        -1,
    );
    let s2 = scale(mx::subtract(&mx::add(&one_minus(&r00), &r11), &r22));
    let d2 = add_scalar(&s2, 1e-10);
    let q2 = mx::stack_owned(
        &[
            mx::divide(&mx::add(&r01, &r10), &d2),
            mx::multiply_scalar(&s2, 0.25),
            mx::divide(&mx::add(&r12, &r21), &d2),
            mx::divide(&mx::subtract(&r02, &r20), &d2),
        ],
        -1,
    );
    let s3 = scale(mx::add(&mx::subtract(&one_minus(&r00), &r11), &r22));
    let d3 = add_scalar(&s3, 1e-10);
    let q3 = mx::stack_owned(
        &[
            mx::divide(&mx::add(&r02, &r20), &d3),
            mx::divide(&mx::add(&r12, &r21), &d3),
            mx::multiply_scalar(&s3, 0.25),
            mx::divide(&mx::subtract(&r10, &r01), &d3),
        ],
        -1,
    );
    let best = mx::argmax(&mx::stack_owned(&[trace, r00, r11, r22], -1), -1, false);
    let mut result = q0;
    for (index, candidate) in [q1, q2, q3].into_iter().enumerate() {
        let condition = mx::equal(&best, &mx::from_slice_i32(&[index as i32 + 1], &[]));
        result = mx::where_cond(&mx::expand_dims(&condition, -1), &candidate, &result);
    }
    Ok(mx::reshape(&result, &output_shape))
}
