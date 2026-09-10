// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! MoonViT3D position embedding (Kimi K3).
//!
//! The tower holds a learnable `[init_h, init_w, dim]` grid
//! (`vision_tower.patch_embed.pos_emb.weight`, 64x64x1024 on the published
//! checkpoint) that is **bilinearly** resampled to each image's `(h, w)` patch
//! grid with half-pixel sampling, in f32:
//!
//! ```text
//! src = (i + 0.5) * in / out - 0.5, clipped to [0, in - 1]
//! i0 = floor(src);  i1 = min(i0 + 1, in - 1);  frac = src - i0
//! out[i] = (1 - frac) * grid[i0] + frac * grid[i1]      (rows, then columns)
//! ```
//!
//! This is where MoonViT3D departs from MoonViT (`kimi_vl_pos_emb.rs`), whose
//! grid is resampled bicubically; the two are not interchangeable.
//!
//! Multi-frame inputs add a fixed sincos time table on top of the spatial
//! grid: `time[k] = concat(sin(k * omega), cos(k * omega))` with
//! `omega = 1 / 10000^(arange(dim / 2) / (dim / 2))`. That table is the one
//! `kimi_vl_pos_emb::temporal_sinusoid` already builds, so it is reused. The
//! time term is added only when `t > 1`: a single frame gets the plain 2D
//! grid, so an image is not shifted by the frame-0 constant
//! (`time[0] = concat(0, 1)` is not zero).
//!
//! Used by: `encoders::moonvit3d::PatchEmbed` (Kimi K3).

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::encoders::kimi_vl::pos_emb::temporal_sinusoid;

/// The `(t, h, w)` patch grid of one media item handed to MoonViT3D.
///
/// `t` is the frame count (1 for an image), `(h, w)` the per-frame patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoonViT3DGrid {
    pub t: i32,
    pub h: i32,
    pub w: i32,
}

impl MoonViT3DGrid {
    /// A still image: `t = 1`.
    #[inline]
    pub fn image(h: i32, w: i32) -> Self {
        Self { t: 1, h, w }
    }

    /// Encoder tokens (pre-merge): `t * h * w`.
    #[inline]
    pub fn token_count(&self) -> i32 {
        self.t * self.h * self.w
    }

    /// Merged tokens after the `(kh, kw)` spatial merge: `(h / kh) * (w / kw)`.
    /// The temporal mean-pool makes this independent of `t`.
    #[inline]
    pub fn merged_count(&self, merge: (i32, i32)) -> i32 {
        (self.h / merge.0) * (self.w / merge.1)
    }
}

/// The `[out, in]` bilinear interpolation matrix for one axis, half-pixel
/// sampling with the source index clipped to `[0, in - 1]`.
///
/// Each row holds at most two non-zero taps that sum to one, so a constant
/// grid resamples to the same constant and a same-size resample is the
/// identity.
pub(crate) fn bilinear_matrix(in_size: i32, out_size: i32) -> Vec<f32> {
    let mut m = vec![0.0f32; (out_size * in_size) as usize];
    let scale = in_size as f32 / out_size as f32;
    let last = (in_size - 1) as f32;
    for o in 0..out_size {
        let src = ((o as f32 + 0.5) * scale - 0.5).clamp(0.0, last);
        let i0 = src.floor();
        let frac = src - i0;
        let i0 = i0 as i32;
        let i1 = (i0 + 1).min(in_size - 1);
        m[(o * in_size + i0) as usize] += 1.0 - frac;
        m[(o * in_size + i1) as usize] += frac;
    }
    m
}

/// Separable bilinear resample of a `[in_h, in_w, dim]` grid to
/// `[out_h, out_w, dim]`, computed in f32 (rows first, then columns).
pub(crate) fn bilinear_resample(
    grid: &MlxArray,
    in_h: i32,
    in_w: i32,
    dim: i32,
    out_h: i32,
    out_w: i32,
) -> UniquePtr<MlxArray> {
    let grid = mlxcel_core::astype(grid, mlxcel_core::dtype::FLOAT32);
    let w_h = mlxcel_core::from_slice_f32(&bilinear_matrix(in_h, out_h), &[out_h, in_h]);
    let w_w = mlxcel_core::from_slice_f32(&bilinear_matrix(in_w, out_w), &[out_w, in_w]);

    // Rows: [out_h, in_h] @ [in_h, in_w * dim] -> [out_h, in_w, dim].
    let rows = mlxcel_core::reshape(&grid, &[in_h, in_w * dim]);
    let rows = mlxcel_core::matmul(&w_h, &rows);
    let rows = mlxcel_core::reshape(&rows, &[out_h, in_w, dim]);

    // Columns: move in_w to the front, contract, move back.
    let cols = mlxcel_core::transpose_axes(&rows, &[1, 0, 2]); // [in_w, out_h, dim]
    let cols = mlxcel_core::reshape(&cols, &[in_w, out_h * dim]);
    let out = mlxcel_core::matmul(&w_w, &cols); // [out_w, out_h * dim]
    let out = mlxcel_core::reshape(&out, &[out_w, out_h, dim]);
    mlxcel_core::transpose_axes(&out, &[1, 0, 2]) // [out_h, out_w, dim]
}

/// The fixed sincos time table `[t, dim]`:
/// `time[k] = concat(sin(k * omega), cos(k * omega))`,
/// `omega[j] = 1 / 10000^(j / (dim / 2))` for `j in 0..dim/2`.
pub(crate) fn time_embedding(t: i32, dim: i32) -> UniquePtr<MlxArray> {
    temporal_sinusoid(t, dim)
}

/// Learnable 2D grid with bilinear resampling and the sincos time term.
pub(crate) struct DividedFixedPosEmb {
    weight: UniquePtr<MlxArray>,
    init_h: i32,
    init_w: i32,
    dim: i32,
}

impl DividedFixedPosEmb {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        init_h: i32,
        init_w: i32,
        dim: i32,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.weight");
        let weight = weights
            .get(&key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        let shape = mlxcel_core::array_shape(&weight);
        if shape != [init_h, init_w, dim] {
            return Err(format!(
                "{key}: expected shape [{init_h}, {init_w}, {dim}], got {shape:?}"
            ));
        }
        Ok(Self {
            weight,
            init_h,
            init_w,
            dim,
        })
    }

    /// Construct directly from an in-memory grid (unit tests).
    #[cfg(test)]
    pub(crate) fn from_array(
        weight: UniquePtr<MlxArray>,
        init_h: i32,
        init_w: i32,
        dim: i32,
    ) -> Self {
        Self {
            weight,
            init_h,
            init_w,
            dim,
        }
    }

    /// The `[h, w, dim]` f32 spatial table for one grid.
    pub(crate) fn spatial(&self, h: i32, w: i32) -> UniquePtr<MlxArray> {
        if h == self.init_h && w == self.init_w {
            return mlxcel_core::astype(&self.weight, mlxcel_core::dtype::FLOAT32);
        }
        bilinear_resample(&self.weight, self.init_h, self.init_w, self.dim, h, w)
    }

    /// The `[t * h * w, dim]` f32 position term for one media item
    /// (frame-major): the spatial table alone for `t == 1`, the spatial table
    /// tiled `t` times plus `time[k]` on frame `k` for `t > 1`.
    pub(crate) fn pos_for(&self, grid: MoonViT3DGrid) -> UniquePtr<MlxArray> {
        let MoonViT3DGrid { t, h, w } = grid;
        let spatial = mlxcel_core::reshape(&self.spatial(h, w), &[h * w, self.dim]);
        if t <= 1 {
            return spatial;
        }
        let spatial = mlxcel_core::tile(&spatial, &[t, 1]); // [t*h*w, dim]
        let time = time_embedding(t, self.dim); // [t, dim]
        let time = mlxcel_core::repeat(&time, h * w, 0); // [t*h*w, dim]
        mlxcel_core::add(&spatial, &time)
    }
}
