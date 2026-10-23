// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Small shape-preserving operations for the Body graph.
use super::layers::Array;
use mlxcel_core::{self as mx, MlxArray};
pub fn cat(arrays: &[&MlxArray], axis: i32) -> Array {
    mx::concatenate_many(arrays, axis)
}
pub fn slice_axis(x: &MlxArray, axis: usize, start: i32, end: i32) -> Array {
    let mut ends = mx::array_shape(x);
    let mut starts = vec![0; ends.len()];
    starts[axis] = start;
    ends[axis] = end;
    mx::slice(x, &starts, &ends)
}
pub fn scalar_add(x: &MlxArray, v: f32) -> Array {
    mx::add(x, &mx::full_f32(&[], v, mx::array_dtype(x)))
}
pub fn evaluate(arrays: &[&MlxArray]) {
    let p: Vec<*const MlxArray> = arrays.iter().map(|&a| a as *const _).collect();
    // SAFETY: every array remains alive until the synchronous evaluation returns.
    unsafe { mx::eval_all(&p) }
}
