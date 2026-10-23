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

//! Sparse pooling and nearest upsampling for the MoGe3 refiner.

use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

use super::sparse::{
    SparseShape, coords_to_keys, i32_scalar, is_float, validate_coords, validate_features,
};

const CELL_MEMBERS: i32 = 8;

pub struct PoolResult {
    pub features: UniquePtr<MlxArray>,
    pub coords: UniquePtr<MlxArray>,
    pub shape: SparseShape,
    pub parent_indices: UniquePtr<MlxArray>,
}

pub fn sparse_pool2x_mean(
    feats: &MlxArray,
    coords: &MlxArray,
    shape: SparseShape,
) -> Result<PoolResult> {
    let fs = mlxcel_core::array_shape(feats);
    let rows_count = validate_coords(coords, shape)?;
    let channels = validate_features(feats, rows_count)?;
    let out_shape = shape.pooled();
    let (order, first, counts, parent, out_coords, segments) = {
        let divisor = mlxcel_core::from_slice_i32(&[1, 2, 2, 2], &[1, 4]);
        let tcoords = mlxcel_core::floor_divide(coords, &divisor);
        let keys = coords_to_keys(&tcoords, out_shape);
        let order = mlxcel_core::argsort(&keys, 0);
        let sorted = mlxcel_core::take(&keys, &order, 0);
        let tail = mlxcel_core::slice(&sorted, &[1], &[fs[0]]);
        let head = mlxcel_core::slice(&sorted, &[0], &[fs[0] - 1]);
        let is_new = mlxcel_core::concatenate(
            &mlxcel_core::from_slice_i32(&[1], &[1]),
            &mlxcel_core::astype(
                &mlxcel_core::not_equal(&tail, &head),
                mlxcel_core::dtype::INT32,
            ),
            0,
        );
        let seg_sorted = mlxcel_core::subtract(
            &mlxcel_core::cumsum(&is_new, 0, false, true),
            &i32_scalar(1),
        );
        let num = mlxcel_core::sum_all(&is_new);
        mlxcel_core::eval(&num);
        let segments = mlxcel_core::item_i32(&num);
        ensure!(segments > 0, "MoGe3 sparse pool produced no segments");
        let parent = mlxcel_core::scatter_add(
            &mlxcel_core::zeros(&[fs[0]], mlxcel_core::dtype::INT32),
            &order,
            &mlxcel_core::reshape(&seg_sorted, &[fs[0], 1]),
            0,
        );
        let first = mlxcel_core::scatter_min(
            &mlxcel_core::full_f32(&[segments], fs[0] as f32, mlxcel_core::dtype::INT32),
            &seg_sorted,
            &mlxcel_core::reshape(&mlxcel_core::arange_i32(0, fs[0], 1), &[fs[0], 1]),
            0,
        );
        let ends = mlxcel_core::concatenate(
            &mlxcel_core::slice(&first, &[1], &[segments]),
            &i32_scalar(fs[0]),
            0,
        );
        let counts = mlxcel_core::subtract(&ends, &first);
        let max_count = mlxcel_core::max_all(&counts);
        ensure!(
            mlxcel_core::item_i32(&max_count) <= CELL_MEMBERS,
            "MoGe3 sparse pool cell contains more than {CELL_MEMBERS} voxels"
        );
        let first_rows = mlxcel_core::take(
            &order,
            &mlxcel_core::astype(&first, mlxcel_core::dtype::UINT32),
            0,
        );
        let out_coords = mlxcel_core::take(&tcoords, &first_rows, 0);
        // Resolve topology before dropping sorted keys, segment flags and scratch.
        for array in [&order, &first, &counts, &parent, &out_coords] {
            mlxcel_core::eval(array);
        }
        (order, first, counts, parent, out_coords, segments)
    };
    let rows_per_chunk = super::sparse_budget::chunk_rows(
        channels,
        channels,
        mlxcel_core::array_itemsize(feats),
        CELL_MEMBERS as usize,
    )? as i32;
    let slots = mlxcel_core::arange_i32(0, CELL_MEMBERS, 1);
    let mut chunks = Vec::new();
    chunks.try_reserve_exact((segments as usize).div_ceil(rows_per_chunk as usize))?;
    let mut start = 0;
    while start < segments {
        let end = (start + rows_per_chunk).min(segments);
        let chunk_first = mlxcel_core::slice(&first, &[start], &[end]);
        let chunk_counts = mlxcel_core::slice(&counts, &[start], &[end]);
        let rows = mlxcel_core::add(
            &mlxcel_core::reshape(&chunk_first, &[end - start, 1]),
            &mlxcel_core::reshape(&slots, &[1, CELL_MEMBERS]),
        );
        let rows = mlxcel_core::minimum(&rows, &i32_scalar(fs[0] - 1));
        let rows = mlxcel_core::take(&order, &rows, 0);
        let valid = mlxcel_core::less(
            &mlxcel_core::reshape(&slots, &[1, CELL_MEMBERS]),
            &mlxcel_core::reshape(&chunk_counts, &[end - start, 1]),
        );
        let gathered = mlxcel_core::take(feats, &rows, 0);
        let gathered = mlxcel_core::where_cond(
            &mlxcel_core::reshape(&valid, &[end - start, CELL_MEMBERS, 1]),
            &gathered,
            &mlxcel_core::zeros(&[1], mlxcel_core::array_dtype(feats)),
        );
        let summed = mlxcel_core::sum_axis(&gathered, 1, false);
        let chunk = mlxcel_core::divide(
            &summed,
            &mlxcel_core::astype(
                &mlxcel_core::reshape(&chunk_counts, &[end - start, 1]),
                mlxcel_core::array_dtype(feats),
            ),
        );
        mlxcel_core::eval(&chunk);
        chunks.push(chunk);
        start = end;
    }
    drop(order);
    drop(first);
    drop(counts);
    drop(slots);
    let output = if chunks.len() == 1 {
        chunks.pop().expect("one sparse pool chunk")
    } else {
        let mut refs = Vec::new();
        refs.try_reserve_exact(chunks.len())?;
        refs.extend(chunks.iter().map(|v| v.as_ref().expect("MLX array")));
        mlxcel_core::concatenate_many(&refs, 0)
    };
    mlxcel_core::eval(&output);
    Ok(PoolResult {
        features: output,
        coords: out_coords,
        shape: out_shape,
        parent_indices: parent,
    })
}

pub fn sparse_upsample2x_nearest(
    feats: &MlxArray,
    parent_indices: &MlxArray,
    target_coords: &MlxArray,
    target_shape: SparseShape,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>, SparseShape)> {
    let fs = mlxcel_core::array_shape(feats);
    let ps = mlxcel_core::array_shape(parent_indices);
    let target_rows = validate_coords(target_coords, target_shape)?;
    ensure!(
        fs.len() == 2
            && fs[0] > 0
            && fs[1] > 0
            && is_float(mlxcel_core::array_dtype(feats))
            && ps == [target_rows as i32]
            && mlxcel_core::array_dtype(parent_indices) == mlxcel_core::dtype::INT32,
        "MoGe3 sparse upsample shapes mismatch"
    );
    let valid = mlxcel_core::logical_and(
        &mlxcel_core::greater_equal(parent_indices, &i32_scalar(0)),
        &mlxcel_core::less(parent_indices, &i32_scalar(fs[0])),
    );
    ensure!(
        mlxcel_core::item_bool(&mlxcel_core::all_all(&valid)),
        "MoGe3 sparse upsample parent index is out of bounds"
    );
    Ok((
        mlxcel_core::take(feats, parent_indices, 0),
        mlxcel_core::copy(target_coords),
        target_shape,
    ))
}
