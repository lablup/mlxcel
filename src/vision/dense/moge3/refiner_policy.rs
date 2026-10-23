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

//! Conservative checkpoint/input admission, distinct from the cache target.
//!
//! This bounds refiner tensor phases, not total process/device memory: model
//! weights, caller-owned tensors, and backend allocator scratch are external.
use super::super::sparse_budget as phases;
use super::{
    MAX_SPARSE_WORKING_BYTES, SparseShape, neighbor_construction_peak_bytes, neighbor_map_bytes,
};
use anyhow::{Result, ensure};

// Finite support envelope above the released Objects capture's conservative
// 5,598,663,168-byte phase bound (see objects-refiner-admission.json). Depth
// quantization resolution is NOT grid depth: the latter depends on logz span.
// Each run still receives its smaller checked input/config-derived allowance.
pub(super) const SUPPORTED_PHASE_BYTES: u64 = 8 << 30;
pub(super) struct Admission {
    pub rows: Vec<usize>,
    pub working: usize,
}
pub(super) fn admission(
    rows: usize,
    shape: SparseShape,
    channels: &[usize],
    encoder: usize,
) -> Result<Admission> {
    ensure!(
        rows > 0
            && (2..=8).contains(&channels.len())
            && channels.iter().all(|&c| c > 0 && c <= 4096)
            && encoder > 0,
        "MoGe3 invalid refiner admission dimensions"
    );
    let initial = neighbor_construction_peak_bytes(rows, channels[0], neighbor_map_bytes(rows)?)?;
    ensure!(
        initial <= MAX_SPARSE_WORKING_BYTES,
        "MoGe3 initial sparse grid exceeds admission budget: rows={rows} channels={} peak={initial} limit={MAX_SPARSE_WORKING_BYTES}",
        channels[0]
    );
    let mut grid = [
        usize::try_from(shape.batch())?,
        usize::try_from(shape.height())?,
        usize::try_from(shape.width())?,
        usize::try_from(shape.depth())?,
    ];
    let volume = grid.iter().try_fold(1usize, |a, &b| phases::bytes(a, b))?;
    ensure!(rows <= volume, "MoGe3 sparse rows exceed grid volume");
    let mut bounds = Vec::new();
    bounds.try_reserve_exact(channels.len())?;
    let mut skips = Vec::new();
    skips.try_reserve_exact(channels.len())?;
    let mut n = rows;
    let mut retained = 0;
    let mut peak = MAX_SPARSE_WORKING_BYTES;
    let levels = channels.len();
    for (level, &channels) in channels.iter().enumerate() {
        n = n.min(grid.iter().try_fold(1usize, |a, &b| phases::bytes(a, b))?);
        bounds.push(n);
        skips.push(retained);
        let map = neighbor_map_bytes(n)?;
        let construction = neighbor_construction_peak_bytes(n, channels, map)?;
        let residual = phases::sum([phases::residual(n, channels)?, map])?;
        peak = peak.max(phases::sum([retained, construction.max(residual)])?);
        if level + 1 < levels {
            peak = peak.max(phases::sum([retained, phases::pool(n, channels)?])?);
            retained = phases::sum([retained, super::budget::skip_bytes(n, channels)?])?;
        }
        for dim in &mut grid[1..] {
            *dim = dim.div_ceil(2);
        }
    }
    for level in 0..channels.len() - 1 {
        peak = peak.max(phases::sum([
            skips[level],
            phases::projection(
                bounds[level],
                bounds[level + 1],
                channels[level],
                channels[level + 1],
            )?,
        ])?);
    }
    let last = channels.len() - 1;
    peak = peak.max(phases::sum([
        skips[last],
        phases::fusion(bounds[last], channels[last], encoder)?,
    ])?);
    for level in (0..last).rev() {
        peak = peak.max(phases::sum([
            skips[level + 1],
            phases::upsample(
                bounds[level + 1],
                channels[level + 1],
                bounds[level],
                channels[level],
            )?,
        ])?);
    }
    ensure!(
        (peak as u128) <= SUPPORTED_PHASE_BYTES as u128,
        "MoGe3 configured refiner exceeds supported phase budget: rows={rows} shape={shape:?} channels={channels:?} row_bounds={bounds:?} required={peak} supported_limit={SUPPORTED_PHASE_BYTES}"
    );
    Ok(Admission {
        rows: bounds,
        working: peak,
    })
}
