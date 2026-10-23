// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Continuous body/hand layout, including the noncontiguous finger ordering.
use super::{Array, batch_xyz_from_6d, check_last, last, sincos_to_angle};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray};
const THREE: [[usize; 3]; 23] = [
    [0, 2, 4],
    [6, 8, 10],
    [12, 13, 14],
    [15, 16, 17],
    [18, 19, 20],
    [21, 22, 23],
    [24, 25, 26],
    [27, 28, 29],
    [34, 35, 36],
    [37, 38, 39],
    [44, 45, 46],
    [53, 54, 55],
    [64, 65, 66],
    [85, 69, 73],
    [86, 70, 79],
    [87, 71, 82],
    [88, 72, 76],
    [91, 92, 93],
    [112, 96, 100],
    [113, 97, 106],
    [114, 98, 109],
    [115, 99, 103],
    [130, 131, 132],
];
const ONE: [usize; 58] = [
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60,
    61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107,
    108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123,
];
const HAND: [usize; 16] = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1];
pub fn compact_body(x: &MlxArray) -> Result<Array> {
    check_last(x, 260)?;
    ensure!(mx::array_ndim(x) == 2, "MHR body pose must be [B,260]");
    let mut parts = Vec::with_capacity(133);
    let mut order = [0_i32; 133];
    let mut offset = 0;
    for triple in THREE {
        let angles = batch_xyz_from_6d(&last(x, offset, offset + 6))?;
        for (axis, index) in triple.into_iter().enumerate() {
            order[index] = parts.len() as i32;
            parts.push(last(&angles, axis as i32, axis as i32 + 1));
        }
        offset += 6;
    }
    for index in ONE {
        order[index] = parts.len() as i32;
        parts.push(mx::expand_dims(
            &sincos_to_angle(&last(x, offset, offset + 2)),
            -1,
        ));
        offset += 2;
    }
    for destination in &mut order[124..130] {
        *destination = parts.len() as i32;
        parts.push(last(x, offset, offset + 1));
        offset += 1;
    }
    let refs = parts.iter().map(|v| &**v).collect::<Vec<_>>();
    Ok(mx::take(
        &mx::concatenate_many(&refs, 1),
        &mx::from_slice_i32(&order, &[133]),
        1,
    ))
}
pub fn compact_hand(x: &MlxArray) -> Result<Array> {
    check_last(x, 54)?;
    ensure!(mx::array_ndim(x) == 2, "MHR hand pose must be [B,54]");
    let mut parts = Vec::with_capacity(16);
    let mut offset = 0;
    for dof in HAND {
        let part = if dof == 3 {
            batch_xyz_from_6d(&last(x, offset, offset + 6))?
        } else {
            let a = mx::expand_dims(&sincos_to_angle(&last(x, offset, offset + 2)), -1);
            if dof == 1 {
                a
            } else {
                let b = mx::expand_dims(&sincos_to_angle(&last(x, offset + 2, offset + 4)), -1);
                mx::concatenate(&a, &b, 1)
            }
        };
        parts.push(part);
        offset += (dof * 2) as i32;
    }
    let refs = parts.iter().map(|v| &**v).collect::<Vec<_>>();
    Ok(mx::concatenate_many(&refs, 1))
}
