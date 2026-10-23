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

//! Pure-MLX sparse 3D operations used by MoGe3.
//!
//! Coordinates are `(batch, row, column, depth)` in ascending raster order.
//! No operation materializes a dense 3D volume.

// Used by: MoGe3. SAM 3D Objects has a separate upstream sparse module.

use anyhow::{Result, ensure};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub use super::sparse_pool::{PoolResult, sparse_pool2x_mean, sparse_upsample2x_nearest};
pub use super::sparse_shape::SparseShape;
use super::weights;

pub(super) const GATHER_CHUNK_BYTES: usize = 16 << 20;
pub(super) const MAX_SPARSE_NEIGHBOR_BYTES: usize = 256 << 20;
const NEIGHBOR_CHUNK_BYTES: usize = 16 << 20;
const NEIGHBOR_BUILD_BYTES_PER_ROW: usize = 1536;
const NEIGHBOR_CHUNK_ROWS: usize = NEIGHBOR_CHUNK_BYTES / NEIGHBOR_BUILD_BYTES_PER_ROW;

pub(super) fn is_float(dtype: i32) -> bool {
    matches!(
        dtype,
        mlxcel_core::dtype::FLOAT16 | mlxcel_core::dtype::BFLOAT16 | mlxcel_core::dtype::FLOAT32
    )
}

pub(super) fn validate_coords(coords: &MlxArray, shape: SparseShape) -> Result<usize> {
    let s = mlxcel_core::array_shape(coords);
    ensure!(
        s.len() == 2 && s[0] > 0 && s[1] == 4,
        "MoGe3 sparse coords must be non-empty [M,4]"
    );
    ensure!(
        mlxcel_core::array_dtype(coords) == mlxcel_core::dtype::INT32,
        "MoGe3 sparse coords must be int32"
    );
    let limits = [shape.batch(), shape.height(), shape.width(), shape.depth()];
    for (column, limit) in limits.into_iter().enumerate() {
        let values = coord_column(coords, column as i32);
        let valid = mlxcel_core::logical_and(
            &mlxcel_core::greater_equal(&values, &i32_scalar(0)),
            &mlxcel_core::less(&values, &i32_scalar(limit as i32)),
        );
        ensure!(
            mlxcel_core::item_bool(&mlxcel_core::all_all(&valid)),
            "MoGe3 sparse coordinate column {column} is out of bounds"
        );
    }
    if s[0] > 1 {
        let keys = coords_to_keys(coords, shape);
        let ascending = mlxcel_core::greater(
            &mlxcel_core::slice(&keys, &[1], &[s[0]]),
            &mlxcel_core::slice(&keys, &[0], &[s[0] - 1]),
        );
        ensure!(
            mlxcel_core::item_bool(&mlxcel_core::all_all(&ascending)),
            "MoGe3 sparse coords must be unique and strictly raster-sorted"
        );
    }
    Ok(s[0] as usize)
}

pub(super) fn validate_features(feats: &MlxArray, rows: usize) -> Result<usize> {
    let shape = mlxcel_core::array_shape(feats);
    ensure!(
        shape.len() == 2 && shape[0] == rows as i32 && shape[1] > 0,
        "MoGe3 sparse features must be non-empty [M,C]"
    );
    ensure!(
        is_float(mlxcel_core::array_dtype(feats)),
        "MoGe3 sparse features must be floating point"
    );
    Ok(shape[1] as usize)
}

pub(super) fn neighbor_map_bytes(rows: usize) -> Result<usize> {
    rows.checked_mul(27)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>()))
        .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse neighbor budget overflow"))
}

pub(super) fn neighbor_construction_peak_bytes(
    rows: usize,
    channels: usize,
    retained_bytes: usize,
) -> Result<usize> {
    let map = neighbor_map_bytes(rows)?;
    let features = rows
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse feature budget overflow"))?;
    let coords_and_keys = rows
        .checked_mul(16 + 80)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse coordinate budget overflow"))?;
    let chunk = rows
        .min(NEIGHBOR_CHUNK_ROWS)
        .checked_mul(NEIGHBOR_BUILD_BYTES_PER_ROW)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse lookup budget overflow"))?;
    retained_bytes
        .checked_add(features)
        .and_then(|value| value.checked_add(coords_and_keys))
        // `retained_bytes` includes the final map. Concatenating evaluated
        // chunks briefly retains one additional complete map.
        .and_then(|value| value.checked_add(map))
        .and_then(|value| value.checked_add(chunk))
        .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse construction budget overflow"))
}
fn i64_scalar(v: i64) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i64(&[v], &[1])
}
pub(super) fn i32_scalar(v: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(&[v], &[1])
}
pub(crate) fn coord_column(coords: &MlxArray, index: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(coords);
    mlxcel_core::reshape(
        &mlxcel_core::slice(coords, &[0, index], &[s[0], index + 1]),
        &[s[0]],
    )
}

pub(super) fn coords_to_keys(coords: &MlxArray, shape: SparseShape) -> UniquePtr<MlxArray> {
    let c = mlxcel_core::astype(coords, mlxcel_core::dtype::INT64);
    let b = coord_column(&c, 0);
    let i = coord_column(&c, 1);
    let j = coord_column(&c, 2);
    let z = coord_column(&c, 3);
    let key = mlxcel_core::add(&mlxcel_core::multiply(&b, &i64_scalar(shape.height())), &i);
    let key = mlxcel_core::add(&mlxcel_core::multiply(&key, &i64_scalar(shape.width())), &j);
    mlxcel_core::add(&mlxcel_core::multiply(&key, &i64_scalar(shape.depth())), &z)
}

pub fn submanifold_conv3d_neighbor_map(
    coords: &MlxArray,
    shape: SparseShape,
) -> Result<UniquePtr<MlxArray>> {
    let coord_shape = mlxcel_core::array_shape(coords);
    ensure!(
        coord_shape.len() == 2 && coord_shape[0] > 0 && coord_shape[1] == 4,
        "MoGe3 sparse coords must be non-empty [M,4]"
    );
    ensure!(
        mlxcel_core::array_dtype(coords) == mlxcel_core::dtype::INT32,
        "MoGe3 sparse coords must be int32"
    );
    let rows = coord_shape[0] as usize;
    let map_bytes = neighbor_map_bytes(rows)?;
    ensure!(
        map_bytes <= MAX_SPARSE_NEIGHBOR_BYTES,
        "MoGe3 sparse neighbor map requires {map_bytes} bytes"
    );
    let standalone_peak = neighbor_construction_peak_bytes(rows, 0, map_bytes)?;
    ensure!(
        standalone_peak <= super::config::MAX_SPARSE_WORKING_BYTES,
        "MoGe3 sparse neighbor construction requires {standalone_peak} bytes"
    );
    ensure!(
        validate_coords(coords, shape)? == rows,
        "MoGe3 sparse coordinate row count changed during validation"
    );
    let keys = coords_to_keys(coords, shape);
    mlxcel_core::eval(&keys);
    let mut offsets = Vec::with_capacity(81);
    for di in -1..=1 {
        for dj in -1..=1 {
            for dz in -1..=1 {
                offsets.extend_from_slice(&[di, dj, dz])
            }
        }
    }
    let offsets = mlxcel_core::from_slice_i32(&offsets, &[27, 3]);
    let di = coord_column(&offsets, 0);
    let dj = coord_column(&offsets, 1);
    let dz = coord_column(&offsets, 2);
    let dio = mlxcel_core::astype(&di, mlxcel_core::dtype::INT64);
    let djo = mlxcel_core::astype(&dj, mlxcel_core::dtype::INT64);
    let dzo = mlxcel_core::astype(&dz, mlxcel_core::dtype::INT64);
    let ko = mlxcel_core::add(
        &mlxcel_core::multiply(
            &mlxcel_core::add(
                &mlxcel_core::multiply(&dio, &i64_scalar(shape.width())),
                &djo,
            ),
            &i64_scalar(shape.depth()),
        ),
        &dzo,
    );
    let mut chunks = Vec::with_capacity(rows.div_ceil(NEIGHBOR_CHUNK_ROWS));
    for start in (0..rows).step_by(NEIGHBOR_CHUNK_ROWS) {
        let end = (start + NEIGHBOR_CHUNK_ROWS).min(rows);
        let chunk_rows = (end - start) as i32;
        let chunk_coords = mlxcel_core::slice(coords, &[start as i32, 0], &[end as i32, 4]);
        let chunk_keys = mlxcel_core::slice(&keys, &[start as i32], &[end as i32]);
        let nkeys = mlxcel_core::add(
            &mlxcel_core::reshape(&chunk_keys, &[chunk_rows, 1]),
            &mlxcel_core::reshape(&ko, &[1, 27]),
        );
        let bounds = |coordinate: UniquePtr<MlxArray>, delta: &MlxArray, size: i64| {
            let c = mlxcel_core::reshape(&coordinate, &[chunk_rows, 1]);
            let d = mlxcel_core::reshape(delta, &[1, 27]);
            let lower = mlxcel_core::logical_or(
                &mlxcel_core::greater_equal(&d, &i32_scalar(0)),
                &mlxcel_core::greater(&c, &i32_scalar(0)),
            );
            let upper = mlxcel_core::logical_or(
                &mlxcel_core::less_equal(&d, &i32_scalar(0)),
                &mlxcel_core::less(&c, &i32_scalar((size - 1) as i32)),
            );
            mlxcel_core::logical_and(&lower, &upper)
        };
        let valid = mlxcel_core::logical_and(
            &bounds(coord_column(&chunk_coords, 1), &di, shape.height()),
            &mlxcel_core::logical_and(
                &bounds(coord_column(&chunk_coords, 2), &dj, shape.width()),
                &bounds(coord_column(&chunk_coords, 3), &dz, shape.depth()),
            ),
        );
        let flat = mlxcel_core::reshape(&nkeys, &[chunk_rows * 27]);
        let idx = mlxcel_core::searchsorted(&keys, &flat, false);
        let idx = mlxcel_core::minimum(
            &idx,
            &mlxcel_core::from_slice_u32(&[(rows - 1) as u32], &[1]),
        );
        let idx = mlxcel_core::reshape(&idx, &[chunk_rows, 27]);
        let found = mlxcel_core::logical_and(
            &mlxcel_core::equal(&mlxcel_core::take(&keys, &idx, 0), &nkeys),
            &valid,
        );
        let idx = mlxcel_core::astype(&idx, mlxcel_core::dtype::INT32);
        let chunk = mlxcel_core::where_cond(
            &found,
            &idx,
            &mlxcel_core::full_f32(&[1], -1.0, mlxcel_core::dtype::INT32),
        );
        mlxcel_core::eval(&chunk);
        chunks.push(chunk);
    }
    let references: Vec<&MlxArray> = chunks
        .iter()
        .map(|chunk| chunk.as_ref().expect("evaluated neighbor chunk"))
        .collect();
    let map = mlxcel_core::concatenate_many(&references, 0);
    mlxcel_core::eval(&map);
    Ok(map)
}

pub struct SubmanifoldConv3d {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
}
impl SubmanifoldConv3d {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let weight = weights::array(weights, &format!("{prefix}.weight"))?;
        let bias = weights::array(weights, &format!("{prefix}.bias"))?;
        ensure!(
            mlxcel_core::array_shape(&weight) == [out_channels as i32, 3, 3, 3, in_channels as i32],
            "MoGe3 sparse convolution {prefix} weight shape mismatch"
        );
        ensure!(
            mlxcel_core::array_shape(&bias) == [out_channels as i32],
            "MoGe3 sparse convolution {prefix} bias shape mismatch"
        );
        Self::new(weight, bias)
    }
    pub fn new(weight: UniquePtr<MlxArray>, bias: UniquePtr<MlxArray>) -> Result<Self> {
        let shape = mlxcel_core::array_shape(&weight);
        ensure!(
            shape.len() == 5 && shape[0] > 0 && shape[1..4] == [3, 3, 3] && shape[4] > 0,
            "MoGe3 sparse convolution weight must be [Co,3,3,3,Ci]"
        );
        ensure!(
            mlxcel_core::array_shape(&bias) == [shape[0]],
            "MoGe3 sparse convolution bias must be [Co]"
        );
        ensure!(
            is_float(mlxcel_core::array_dtype(&weight))
                && mlxcel_core::array_dtype(&weight) == mlxcel_core::array_dtype(&bias),
            "MoGe3 sparse convolution weight and bias must have one floating dtype"
        );
        Ok(Self { weight, bias })
    }
    pub fn forward(
        &self,
        feats: &MlxArray,
        coords: &MlxArray,
        shape: SparseShape,
        neighbor: Option<&MlxArray>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let fs = mlxcel_core::array_shape(feats);
        let ws = mlxcel_core::array_shape(&self.weight);
        let rows = validate_coords(coords, shape)?;
        let channels = validate_features(feats, rows)?;
        ensure!(
            channels == ws[4] as usize,
            "MoGe3 sparse feature/channel shape mismatch"
        );
        ensure!(
            mlxcel_core::array_dtype(feats) == mlxcel_core::array_dtype(&self.weight),
            "MoGe3 sparse feature and convolution dtypes must match"
        );
        let neighbors = match neighbor {
            Some(v) => {
                ensure!(
                    mlxcel_core::array_shape(v) == [fs[0], 27]
                        && mlxcel_core::array_dtype(v) == mlxcel_core::dtype::INT32,
                    "MoGe3 sparse neighbor map must be int32 [M,27]"
                );
                let valid = mlxcel_core::logical_and(
                    &mlxcel_core::greater_equal(v, &i32_scalar(-1)),
                    &mlxcel_core::less(v, &i32_scalar(fs[0])),
                );
                ensure!(
                    mlxcel_core::item_bool(&mlxcel_core::all_all(&valid)),
                    "MoGe3 sparse neighbor map contains an invalid row"
                );
                mlxcel_core::copy(v)
            }
            None => submanifold_conv3d_neighbor_map(coords, shape)?,
        };
        let padded = mlxcel_core::concatenate(
            &mlxcel_core::copy(feats),
            &mlxcel_core::zeros(&[1, fs[1]], mlxcel_core::array_dtype(feats)),
            0,
        );
        let idx = mlxcel_core::where_cond(
            &mlxcel_core::greater_equal(&neighbors, &i32_scalar(0)),
            &neighbors,
            &i32_scalar(fs[0]),
        );
        let weight =
            mlxcel_core::transpose(&mlxcel_core::reshape(&self.weight, &[ws[0], 27 * fs[1]]));
        let bytes = mlxcel_core::array_itemsize(feats);
        let rows = (GATHER_CHUNK_BYTES / (27 * fs[1] as usize * bytes)).max(1) as i32;
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < fs[0] {
            let end = (start + rows).min(fs[0]);
            let index = mlxcel_core::slice(&idx, &[start, 0], &[end, 27]);
            let gathered = mlxcel_core::take(&padded, &index, 0);
            let columns = mlxcel_core::reshape(&gathered, &[end - start, 27 * fs[1]]);
            outputs.push(mlxcel_core::addmm(&self.bias, &columns, &weight, 1.0, 1.0));
            start = end
        }
        let output = if outputs.len() == 1 {
            outputs.pop().expect("one sparse chunk")
        } else {
            let refs: Vec<&MlxArray> = outputs
                .iter()
                .map(|v| v.as_ref().expect("MLX array"))
                .collect();
            mlxcel_core::concatenate_many(&refs, 0)
        };
        Ok((output, neighbors))
    }
}
