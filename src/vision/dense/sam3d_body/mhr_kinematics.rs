// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Parent-ordered MHR forward kinematics.
use super::super::{
    layers::Array,
    mhr_utils::{check_last, einsum, euler_xyz_to_rotmat, last, quat_to_rotmat, rotmat_to_quat},
};
use super::weights::{parents_valid, reserve};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray};
pub(super) struct Skeleton {
    pub state: Array,
    pub position: Array,
    pub rotation: Array,
    pub scale: Array,
}
fn joint(x: &MlxArray, index: i32) -> Array {
    let mut end = mx::array_shape(x);
    let mut start = vec![0; end.len()];
    start[1] = index;
    end[1] = index + 1;
    mx::squeeze_axis(&mx::slice(x, &start, &end), 1)
}
pub(super) fn forward(
    dofs: &MlxArray,
    offsets: &MlxArray,
    prerotations: &MlxArray,
    parents: &[i32],
) -> Result<Skeleton> {
    parents_valid(parents)?;
    let joints = i32::try_from(parents.len())?;
    let width = joints
        .checked_mul(7)
        .ok_or_else(|| anyhow::anyhow!("MHR joint DOF overflow"))?;
    check_last(dofs, width)?;
    let shape = mx::array_shape(dofs);
    ensure!(
        shape.len() == 2
            && mx::array_shape(offsets) == [joints, 3]
            && mx::array_shape(prerotations) == [joints, 4],
        "MHR FK dimensions disagree"
    );
    check_last(offsets, 3)?;
    check_last(prerotations, 4)?;
    let jd = mx::reshape(dofs, &[shape[0], joints, 7]);
    let local_rotation = euler_xyz_to_rotmat(&last(&jd, 3, 6))?;
    let prerot = quat_to_rotmat(prerotations)?;
    let local_rotation = einsum("jpq,bjqr->bjpr", &[&prerot, &local_rotation]);
    let translation = mx::add(&mx::expand_dims(offsets, 0), &last(&jd, 0, 3));
    let scale = mx::exp(&mx::multiply_scalar(
        &last(&jd, 6, 7),
        std::f32::consts::LN_2,
    ));
    let mut positions: Vec<Array> = reserve(parents.len())?;
    let mut rotations: Vec<Array> = reserve(parents.len())?;
    let mut scales: Vec<Array> = reserve(parents.len())?;
    for (j, &parent) in parents.iter().enumerate() {
        let lr = joint(&local_rotation, j as i32);
        let lt = joint(&translation, j as i32);
        let ls = joint(&scale, j as i32);
        let (position, rotation, scale) = if parent == -1 {
            (lt, lr, ls)
        } else {
            let p = parent as usize;
            let displacement = einsum("bij,bj->bi", &[&rotations[p], &lt]);
            (
                mx::add(&positions[p], &mx::multiply(&scales[p], &displacement)),
                einsum("bij,bjk->bik", &[&rotations[p], &lr]),
                mx::multiply(&scales[p], &ls),
            )
        };
        positions.push(position);
        rotations.push(rotation);
        scales.push(scale);
    }
    let position = mx::stack_owned(&positions, 1);
    let rotation = mx::stack_owned(&rotations, 1);
    let scale = mx::stack_owned(&scales, 1);
    let quaternion = rotmat_to_quat(&rotation)?;
    let state = mx::concatenate_many(&[&position, &quaternion, &scale], -1);
    Ok(Skeleton {
        state,
        position,
        rotation,
        scale,
    })
}
