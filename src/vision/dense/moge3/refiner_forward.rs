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

//! Refiner execution with bounded, recomputable integer neighbor maps.

use super::budget::{self, Limits, Request};
use super::*;

type Array = UniquePtr<MlxArray>;
struct State {
    maps: Vec<Option<Array>>,
    parents: Vec<Option<Array>>,
    skips: Vec<Option<Skip>>,
    skip_bytes: Vec<usize>,
    evictions: usize,
    recomputations: usize,
}
fn empty_slots<T>(count: usize) -> Result<Vec<Option<T>>> {
    let mut slots = Vec::new();
    slots.try_reserve_exact(count)?;
    slots.resize_with(count, || None);
    Ok(slots)
}
impl State {
    fn new(levels: usize) -> Result<Self> {
        let mut skip_bytes = Vec::new();
        skip_bytes.try_reserve_exact(levels - 1)?;
        skip_bytes.resize(levels - 1, 0);
        Ok(Self {
            maps: empty_slots(levels)?,
            parents: empty_slots(levels - 1)?,
            skips: empty_slots(levels - 1)?,
            skip_bytes,
            evictions: 0,
            recomputations: 0,
        })
    }
    fn retained_skips(&self) -> Result<usize> {
        budget::sum(self.skip_bytes.iter().copied())
    }
    fn make_room(
        &mut self,
        feats: &MlxArray,
        coords: &MlxArray,
        request: Request,
        limits: Limits,
    ) -> Result<()> {
        let mut cached = Vec::new();
        cached.try_reserve_exact(self.maps.len())?;
        for map in &self.maps {
            cached.push(match map {
                Some(map) => neighbor_map_bytes(mlxcel_core::array_shape(map)[0] as usize)?,
                None => 0,
            });
        }
        let plan = budget::plan(&cached, &request, limits)?;
        if !plan.evicted.is_empty() {
            // Dropping a Rust handle alone does not free a map retained by
            // an unevaluated MLX graph. Resolve every surviving dependent.
            mlxcel_core::eval(feats);
            mlxcel_core::eval(coords);
            for skip in self.skips.iter().flatten() {
                mlxcel_core::eval(&skip.features);
                mlxcel_core::eval(&skip.coords);
            }
            for parent in self.parents.iter().flatten() {
                mlxcel_core::eval(parent);
            }
            for &level in &plan.evicted {
                self.maps[level] = None;
            }
            self.evictions += plan.evicted.len();
            tracing::debug!(phase=request.phase, level=request.level, rows=request.rows,
                channels=request.channels, skip_bytes=request.skips, current_map_bytes=request.current_map,
                workspace_bytes=request.workspace, peak_bytes=plan.peak, map_bytes=plan.neighbors,
                evicted_levels=?plan.evicted, "MoGe3 neighbor cache eviction");
        }
        Ok(())
    }
    fn prepare_map(
        &mut self,
        feats: &MlxArray,
        coords: &MlxArray,
        level: usize,
        phase: &'static str,
        limits: Limits,
    ) -> Result<()> {
        let fs = mlxcel_core::array_shape(feats);
        let (rows, channels) = (fs[0] as usize, fs[1] as usize);
        let cached = self.maps[level].is_some();
        let workspace = if cached {
            budget::sum([
                budget::feature_bytes(rows, channels)?,
                budget::feature_bytes(rows, 4)?,
            ])?
        } else {
            neighbor_construction_peak_bytes(rows, channels, 0)?
        };
        self.make_room(
            feats,
            coords,
            Request {
                phase,
                level,
                rows,
                channels,
                skips: self.retained_skips()?,
                current_map: if cached { 0 } else { neighbor_map_bytes(rows)? },
                workspace,
                protected: cached.then_some(level),
            },
            limits,
        )
    }
}

impl Sparse3dUnet {
    pub fn forward(
        &self,
        features: &MlxArray,
        coords: &MlxArray,
        shape: SparseShape,
        encoder: &MlxArray,
    ) -> Result<Array> {
        self.forward_with_limits(features, coords, shape, encoder, Limits::default())
            .map(|(output, _, _)| output)
    }
    pub(super) fn forward_with_limits(
        &self,
        features: &MlxArray,
        coords: &MlxArray,
        mut shape: SparseShape,
        encoder: &MlxArray,
        limits: Limits,
    ) -> Result<(Array, usize, usize)> {
        let es = mlxcel_core::array_shape(encoder);
        ensure!(
            es.len() == 4
                && shape.batch() == es[0] as i64
                && shape.height() == es[1] as i64 * self.encoder_downsample as i64
                && shape.width() == es[2] as i64 * self.encoder_downsample as i64
                && es[3] == self.encoder_channels as i32
                && mlxcel_core::array_dtype(encoder) == mlxcel_core::dtype::FLOAT32,
            "MoGe3 refiner and encoder resolutions are inconsistent"
        );
        let input_shape = mlxcel_core::array_shape(features);
        ensure!(
            input_shape.len() == 2
                && input_shape[0] > 0
                && input_shape[1] == self.input_channels as i32
                && mlxcel_core::array_dtype(features) == mlxcel_core::dtype::FLOAT32,
            "MoGe3 refiner features must be non-empty [M,{}]",
            self.input_channels
        );
        ensure!(
            mlxcel_core::array_shape(coords) == [input_shape[0], 4]
                && mlxcel_core::array_dtype(coords) == mlxcel_core::dtype::INT32,
            "MoGe3 refiner coords must be int32 [M,4]"
        );
        let input_rows = input_shape[0] as usize;
        let input_bytes = input_rows
            .checked_mul(self.channels[0])
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse input budget overflow"))?;
        ensure!(
            input_bytes <= MAX_SPARSE_WORKING_BYTES,
            "MoGe3 sparse input projection requires {input_bytes} bytes"
        );
        let mut feats = self.input.forward(features);
        let mut coords = mlxcel_core::copy(coords);
        let levels = self.down_stages.len();
        let transitions = self.downsample.len();
        let mut state = State::new(levels)?;
        for i in 0..levels {
            let rows = mlxcel_core::array_shape(&coords)[0] as usize;
            let working = budget::feature_bytes(rows, self.channels[i])?;
            ensure!(
                working <= limits.working,
                "MoGe3 sparse feature budget exceeded: phase=down level={i} rows={rows} channels={} feature_bytes={working} working_limit={}",
                self.channels[i],
                limits.working
            );
            if !self.down_stages[i].is_empty() {
                state.prepare_map(&feats, &coords, i, "down-map", limits)?;
            }
            let mut map = None;
            for block in &self.down_stages[i] {
                let (next, next_map) = block.forward(&feats, &coords, shape, map.as_deref())?;
                feats = next;
                map = Some(next_map);
            }
            state.maps[i] = map;
            if i < transitions {
                let skip_bytes = budget::skip_bytes(rows, self.channels[i])?;
                let skips = budget::sum([state.retained_skips()?, skip_bytes])?;
                state.make_room(
                    &feats,
                    &coords,
                    Request {
                        phase: "down-skip",
                        level: i,
                        rows,
                        channels: self.channels[i],
                        skips,
                        current_map: 0,
                        workspace: 0,
                        protected: None,
                    },
                    limits,
                )?;
                state.skip_bytes[i] = skip_bytes;
                state.skips[i] = Some(Skip {
                    features: mlxcel_core::copy(&feats),
                    coords: mlxcel_core::copy(&coords),
                    shape,
                });
                let pooled = sparse_pool2x_mean(&feats, &coords, shape)?;
                feats = self.downsample[i].linear.forward(&pooled.features);
                coords = pooled.coords;
                shape = pooled.shape;
                state.parents[i] = Some(pooled.parent_indices);
            }
        }
        let cs = mlxcel_core::array_shape(&coords);
        {
            let b = mlxcel_core::astype(
                &super::super::sparse::coord_column(&coords, 0),
                mlxcel_core::dtype::INT64,
            );
            let row = mlxcel_core::astype(
                &super::super::sparse::coord_column(&coords, 1),
                mlxcel_core::dtype::INT64,
            );
            let col = mlxcel_core::astype(
                &super::super::sparse::coord_column(&coords, 2),
                mlxcel_core::dtype::INT64,
            );
            let index = mlxcel_core::add(
                &mlxcel_core::multiply(
                    &mlxcel_core::add(
                        &mlxcel_core::multiply(
                            &b,
                            &mlxcel_core::from_slice_i64(&[es[1] as i64], &[1]),
                        ),
                        &row,
                    ),
                    &mlxcel_core::from_slice_i64(&[es[2] as i64], &[1]),
                ),
                &col,
            );
            let enc_flat = mlxcel_core::reshape(encoder, &[es[0] * es[1] * es[2], es[3]]);
            let enc = self
                .encoder_fuse
                .forward(&mlxcel_core::take(&enc_flat, &index, 0));
            let fused = mlxcel_core::concatenate(&feats, &enc, -1);
            feats = self
                .fuse2
                .forward(&mlxcel_core::utils::silu(&self.fuse1.forward(&fused)));
            mlxcel_core::eval(&feats);
        }
        if !self.bottleneck.is_empty() {
            state.prepare_map(&feats, &coords, levels - 1, "bottleneck-map", limits)?;
        }
        let mut map = state.maps[levels - 1].take();
        for block in &self.bottleneck {
            let (next, next_map) = block.forward(&feats, &coords, shape, map.as_deref())?;
            feats = next;
            map = Some(next_map)
        }
        // No later stage uses the bottleneck map. Evaluate its last dependent
        // before releasing it and before planning the first up stage.
        mlxcel_core::eval(&feats);
        drop(map);
        for i in 0..transitions {
            let level = levels - 2 - i;
            {
                let skip = state.skips[level].take().expect("MoGe3 refiner skip");
                let parent = state.parents[level].take().expect("MoGe3 parent map");
                let projected = self.upsample[i].linear.forward(&feats);
                let (up, up_coords, up_shape) =
                    sparse_upsample2x_nearest(&projected, &parent, &skip.coords, skip.shape)?;
                coords = up_coords;
                shape = up_shape;
                feats = mlxcel_core::add(&up, &skip.features);
                // Resolve parent/skip dependencies before subtracting their
                // storage from the next construction's mandatory live set.
                mlxcel_core::eval(&feats);
                mlxcel_core::eval(&coords);
            }
            state.skip_bytes[level] = 0;
            if !self.up_stages[i].is_empty() {
                if state.maps[level].is_none() {
                    state.recomputations += 1;
                }
                state.prepare_map(&feats, &coords, level, "up-map", limits)?;
            }
            let mut map = state.maps[level].take();
            for block in &self.up_stages[i] {
                let (next, next_map) = block.forward(&feats, &coords, shape, map.as_deref())?;
                feats = next;
                map = Some(next_map);
            }
            mlxcel_core::eval(&feats);
            drop(map);
        }
        ensure!(cs[0] > 0, "MoGe3 bottleneck coords must be non-empty");
        Ok((
            self.output.forward(&feats),
            state.evictions,
            state.recomputations,
        ))
    }
}
