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

//! Checked live-state planning for the recomputable neighbor-map cache.

use anyhow::{Result, ensure};

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub working: usize,
    pub neighbors: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            working: super::MAX_SPARSE_WORKING_BYTES,
            neighbors: super::MAX_SPARSE_NEIGHBOR_BYTES,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Request {
    pub phase: &'static str,
    pub level: usize,
    pub rows: usize,
    pub channels: usize,
    pub skips: usize,
    /// New final map, excluding maps already in the cache.
    pub current_map: usize,
    /// Feature/coordinate/key/chunk storage, including the concatenation copy.
    pub workspace: usize,
    /// An existing current-level map cannot be evicted while being used.
    pub protected: Option<usize>,
}
pub(super) struct Plan {
    pub evicted: Vec<usize>,
    pub peak: usize,
    pub neighbors: usize,
}

pub(super) fn sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse live-state budget overflow"))
    })
}
pub(super) fn feature_bytes(rows: usize, channels: usize) -> Result<usize> {
    rows.checked_mul(channels)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("MoGe3 sparse feature budget overflow"))
}
pub(super) fn skip_bytes(rows: usize, channels: usize) -> Result<usize> {
    sum([feature_bytes(rows, channels)?, feature_bytes(rows, 5)?])
}

/// Evict in level order (largest, finest maps first). Only exact integer
/// lookup tables are discarded; features, skip tensors and parent indices
/// are mandatory and may never be dropped to satisfy a budget.
pub(super) fn plan(cached: &[usize], request: &Request, limits: Limits) -> Result<Plan> {
    ensure!(
        request.protected.is_none_or(|level| level < cached.len()),
        "MoGe3 invalid protected cache level"
    );
    let mut neighbors = sum(cached.iter().copied().chain([request.current_map]))?;
    let mut peak = sum([neighbors, request.skips, request.workspace])?;
    let mut evicted = Vec::new();
    evicted.try_reserve_exact(cached.len())?;
    for (level, &bytes) in cached.iter().enumerate() {
        // 512 MiB remains the cache-retention target even when mandatory
        // checkpoint/input phases need the separately admitted larger bound.
        if peak <= limits.working.min(super::MAX_SPARSE_WORKING_BYTES)
            && neighbors <= limits.neighbors
        {
            break;
        }
        if bytes != 0 && request.protected != Some(level) {
            neighbors -= bytes;
            peak -= bytes;
            evicted.push(level);
        }
    }
    ensure!(
        peak <= limits.working && neighbors <= limits.neighbors,
        "MoGe3 sparse budget exceeded: phase={} level={} rows={} channels={} skips={} cached_maps={} current_map={} workspace={} peak={} working_limit={} map_bytes={} map_limit={} evictable_levels={:?}",
        request.phase,
        request.level,
        request.rows,
        request.channels,
        request.skips,
        sum(cached.iter().copied())?,
        request.current_map,
        request.workspace,
        peak,
        limits.working,
        neighbors,
        limits.neighbors,
        evicted
    );
    Ok(Plan {
        evicted,
        peak,
        neighbors,
    })
}
