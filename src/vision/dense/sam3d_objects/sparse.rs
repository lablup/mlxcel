// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! SAM 3D Objects sparse grid topology. Coordinate order is preserved.
use anyhow::{Result, ensure};
use mlxcel_core as mx;
use mlxcel_core::{MlxArray, UniquePtr};
use std::{
    cell::OnceCell,
    collections::HashMap,
    sync::{Arc, Mutex},
};
#[path = "sparse_ops.rs"]
mod ops;
pub use ops::{SparseConv, SparseGroupNorm, pool, window_attention};
pub type Array = UniquePtr<MlxArray>;
pub(super) type WindowBuckets = Vec<(usize, Vec<Vec<i32>>)>;
type WindowEntry = ((i32, i32), Arc<WindowBuckets>);
type WindowCache = Arc<Mutex<Vec<WindowEntry>>>;
fn reserve<T>(n: usize) -> Result<Vec<T>> {
    ensure!(
        n.checked_mul(std::mem::size_of::<T>())
            .is_some_and(|v| v <= isize::MAX as usize),
        "sparse window allocation exceeds addressable size"
    );
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|e| anyhow::anyhow!("reserve sparse windows: {e}"))?;
    Ok(result)
}

pub struct Grid {
    pub coords: Array,
    pub resolution: i32,
    host: Arc<Vec<[i32; 4]>>,
    neighbors: OnceCell<Array>,
    windows: WindowCache,
}
impl Clone for Grid {
    fn clone(&self) -> Self {
        Self {
            coords: mx::copy(&self.coords),
            resolution: self.resolution,
            host: Arc::clone(&self.host),
            neighbors: OnceCell::new(),
            windows: Arc::clone(&self.windows),
        }
    }
}
impl Grid {
    #[must_use = "sparse grid validation errors must be handled"]
    pub fn new(coords: Array, resolution: i32) -> Result<Self> {
        ensure!(
            resolution > 0 && resolution <= 1_048_576,
            "sparse resolution must be in 1..=1048576"
        );
        let shape = mx::array_shape(&coords);
        ensure!(
            shape.len() == 2
                && shape[1] == 4
                && shape[0] >= 0
                && mx::array_dtype(&coords) == mx::dtype::INT32,
            "sparse coordinates must be int32 [N,4]"
        );
        let values = host_i32(&coords);
        let host: Vec<[i32; 4]> = values
            .chunks_exact(4)
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect();
        ensure!(
            host.iter()
                .all(|p| p[0] >= 0 && p[1..].iter().all(|&v| v >= 0 && v < resolution)),
            "sparse coordinate outside grid"
        );
        let mut unique = host.clone();
        unique.sort_unstable();
        ensure!(
            unique.windows(2).all(|w| w[0] != w[1]),
            "duplicate sparse coordinate"
        );
        Ok(Self {
            coords,
            resolution,
            host: Arc::new(host),
            neighbors: OnceCell::new(),
            windows: Arc::new(Mutex::new(Vec::new())),
        })
    }
    pub fn len(&self) -> usize {
        self.host.len()
    }
    pub fn is_empty(&self) -> bool {
        self.host.is_empty()
    }
    pub fn host_coords(&self) -> &[[i32; 4]] {
        &self.host
    }
    pub fn neighbors(&self) -> Result<&MlxArray> {
        if self.neighbors.get().is_none() {
            ensure!(
                self.len().checked_mul(27 * 4).is_some_and(|b| b <= 1 << 30),
                "sparse neighbor map exceeds 1 GiB"
            );
            let lookup: HashMap<_, _> = self
                .host
                .iter()
                .enumerate()
                .map(|(i, &p)| (p, i as i32))
                .collect();
            let mut indices = Vec::with_capacity(self.len() * 27);
            for p in self.host.iter() {
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            let q = [p[0], p[1] + x, p[2] + y, p[3] + z];
                            indices.push(*lookup.get(&q).unwrap_or(&(self.len() as i32)));
                        }
                    }
                }
            }
            let value = mx::from_slice_i32(&indices, &[self.len() as i32, 27]);
            let _ = self.neighbors.set(value);
        }
        self.neighbors
            .get()
            .and_then(|v| v.as_ref())
            .ok_or_else(|| anyhow::anyhow!("missing neighbor map"))
    }
    pub fn downsample(&self) -> Result<(Grid, Array)> {
        let parents: Vec<_> = self
            .host
            .iter()
            .map(|p| [p[0], p[1] / 2, p[2] / 2, p[3] / 2])
            .collect();
        let mut unique = parents.clone();
        unique.sort_unstable();
        unique.dedup();
        let indices = parents
            .iter()
            .map(|p| unique.binary_search(p).map(|i| i as i32))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| anyhow::anyhow!("missing sparse parent"))?;
        let flat: Vec<_> = unique.iter().flatten().copied().collect();
        Ok((
            Grid::new(
                mx::from_slice_i32(&flat, &[unique.len() as i32, 4]),
                (self.resolution + 1) / 2,
            )?,
            mx::from_slice_i32(&indices, &[self.len() as i32]),
        ))
    }
    pub fn subdivide(&self) -> Result<Grid> {
        let resolution = self
            .resolution
            .checked_mul(2)
            .filter(|&r| r <= 1_048_576)
            .ok_or_else(|| {
                anyhow::anyhow!("subdivided sparse resolution exceeds supported range")
            })?;
        let count = self
            .len()
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("sparse subdivision overflow"))?;
        ensure!(
            count <= i32::MAX as usize && count.checked_mul(16).is_some_and(|b| b <= 1 << 30),
            "sparse subdivision exceeds coordinate budget"
        );
        let mut values = reserve::<i32>(
            count
                .checked_mul(4)
                .ok_or_else(|| anyhow::anyhow!("subdivision coordinate count overflow"))?,
        )?;
        for p in self.host.iter() {
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        values.extend([p[0], p[1] * 2 + x, p[2] * 2 + y, p[3] * 2 + z]);
                    }
                }
            }
        }
        Grid::new(mx::from_slice_i32(&values, &[count as i32, 4]), resolution)
    }
    pub(super) fn windows(&self, size: i32, shift: i32) -> Result<Arc<WindowBuckets>> {
        ensure!(
            size > 0 && shift >= 0 && shift <= self.resolution,
            "invalid sparse attention window"
        );
        let key = (size, shift);
        {
            let cache = self
                .windows
                .lock()
                .map_err(|_| anyhow::anyhow!("sparse window cache poisoned"))?;
            if let Some((_, value)) = cache.iter().find(|(k, _)| *k == key) {
                return Ok(Arc::clone(value));
            }
        }
        let mut order = reserve::<([i32; 4], i32)>(self.len())?;
        order.extend(self.host.iter().enumerate().map(|(i, p)| {
            (
                [
                    p[0],
                    (p[1] + shift) / size,
                    (p[2] + shift) / size,
                    (p[3] + shift) / size,
                ],
                i as i32,
            )
        }));
        // Explicit original-row tie breaker preserves the pinned stable order.
        order.sort_unstable();
        let count =
            usize::from(!order.is_empty()) + order.windows(2).filter(|w| w[0].0 != w[1].0).count();
        let mut ranges = reserve::<(usize, usize, usize)>(count)?;
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && order[end].0 == order[start].0 {
                end += 1;
            }
            ranges.push((
                (end - start)
                    .checked_next_power_of_two()
                    .ok_or_else(|| anyhow::anyhow!("sparse window width overflow"))?,
                start,
                end,
            ));
            start = end;
        }
        ranges.sort_unstable();
        let count = usize::from(!ranges.is_empty())
            + ranges.windows(2).filter(|w| w[0].0 != w[1].0).count();
        let mut buckets = reserve::<(usize, Vec<Vec<i32>>)>(count)?;
        let mut start = 0;
        while start < ranges.len() {
            let width = ranges[start].0;
            let mut end = start + 1;
            while end < ranges.len() && ranges[end].0 == width {
                end += 1;
            }
            let mut windows = reserve::<Vec<i32>>(end - start)?;
            for &(_, first, last) in &ranges[start..end] {
                let mut rows = reserve::<i32>(last - first)?;
                rows.extend(order[first..last].iter().map(|v| v.1));
                windows.push(rows);
            }
            buckets.push((width, windows));
            start = end;
        }
        let value = Arc::new(buckets);
        let mut cache = self
            .windows
            .lock()
            .map_err(|_| anyhow::anyhow!("sparse window cache poisoned"))?;
        if let Some((_, value)) = cache.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(value));
        }
        cache
            .try_reserve_exact(1)
            .map_err(|e| anyhow::anyhow!("reserve sparse window cache: {e}"))?;
        cache.push((key, Arc::clone(&value)));
        Ok(value)
    }
    /// Bytes retained by this grid, including cached window partitions.
    pub(super) fn resident_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        let mut add = |count: usize, width: usize| -> Result<()> {
            bytes = count
                .checked_mul(width)
                .and_then(|n| bytes.checked_add(n))
                .ok_or_else(|| anyhow::anyhow!("sparse resident-byte overflow"))?;
            Ok(())
        };
        add(self.len(), 16)?;
        add(self.host.capacity(), 16)?;
        if self.neighbors.get().is_some() {
            add(self.len(), 27 * 4)?;
        }
        let cache = self
            .windows
            .lock()
            .map_err(|_| anyhow::anyhow!("sparse window cache poisoned"))?;
        add(cache.capacity(), std::mem::size_of::<WindowEntry>())?;
        for (_, buckets) in cache.iter() {
            add(
                buckets.capacity(),
                std::mem::size_of::<(usize, Vec<Vec<i32>>)>(),
            )?;
            for (_, windows) in buckets.iter() {
                add(windows.capacity(), std::mem::size_of::<Vec<i32>>())?;
                for rows in windows {
                    add(rows.capacity(), std::mem::size_of::<i32>())?;
                }
            }
        }
        Ok(bytes)
    }
}
pub(super) fn host_i32(a: &MlxArray) -> Vec<i32> {
    mx::eval(a);
    mx::array_to_raw_bytes(a)
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
pub(super) fn concat(values: &[Array], axis: i32) -> Result<Array> {
    ensure!(!values.is_empty(), "empty sparse concatenation");
    let refs = values
        .iter()
        .map(|v| {
            v.as_ref()
                .ok_or_else(|| anyhow::anyhow!("null sparse array"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(mx::concatenate_many(&refs, axis))
}
pub(super) fn weight(weights: &mlxcel_core::weights::WeightMap, key: &str) -> Result<Array> {
    weights
        .get(key)
        .map(|v| mx::copy(v))
        .ok_or_else(|| anyhow::anyhow!("missing sparse weight {key}"))
}
pub(super) fn validate(x: &MlxArray, rows: usize) -> Result<i32> {
    let s = mx::array_shape(x);
    ensure!(
        s.len() == 2 && s[0] == rows as i32 && s[1] > 0,
        "sparse features must be [N,C]"
    );
    ensure!(
        matches!(
            mx::array_dtype(x),
            mx::dtype::FLOAT32 | mx::dtype::FLOAT16 | mx::dtype::BFLOAT16
        ),
        "sparse features must be floating point"
    );
    Ok(s[1])
}
#[cfg(test)]
#[path = "sparse_tests.rs"]
mod tests;
