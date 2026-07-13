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

//! MoonViT learnable 2D interpolated position embedding.
//!
//! Port of `Learnable2DInterpPosEmb` from upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/vision.py.
//!
//! The module holds a learned `[init_h, init_w, dim]` grid. For a native
//! resolution whose patch grid equals the learned grid it flattens the grid
//! directly (the upstream fast path); otherwise it bicubically resamples the
//! grid to the target `[h, w]` before flattening and adding it to the patch
//! embeddings.
//!
//! `mlxcel_core` exposes no `bicubic_interpolate` op (upstream calls a custom
//! Metal kernel), so we implement the resample as a separable
//! interpolation-matrix contraction: `out = W_h · grid · W_wᵀ`, where `W_h`
//! and `W_w` are precomputed cubic-convolution weight matrices. This matches
//! PyTorch's non-antialiased bicubic (`align_corners=False`, cubic `a = -0.75`,
//! edge-replicated boundaries) — the exact mode upstream's
//! `bicubic_interpolate(..., size=shape)` uses by default. The weights sum to
//! one per output position, so a constant grid resamples to the same constant,
//! and a same-size resample is the identity.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::KimiMediaGrid;

/// Build the temporal position-embedding table for `t` frames, shape
/// `[t, dim]`.
///
/// A closed-form 1D sinusoid over the frame index (Kimi-VL 2.5); it is computed
/// on the fly and is never a checkpoint weight. With `half = dim / 2`:
/// - `pos[f] = f` for `f in 0..t`,
/// - `freq[j] = exp(-ln(10000) * j / half)` for `j in 0..half`,
/// - `ang[f, j] = pos[f] * freq[j]`,
/// - `table = concat([sin(ang), cos(ang)], axis = 1)`.
///
/// Row 0 is therefore `sin(0) = 0` across columns `0..half` and `cos(0) = 1`
/// across columns `half..2*half`. When `dim` is odd the sin/cos halves cover
/// `dim - 1` columns, so a single zero column is appended to reach `dim`.
pub(crate) fn temporal_sinusoid(t: i32, dim: i32) -> UniquePtr<MlxArray> {
    let half = dim / 2;
    // pos = [0, 1, ..., t-1]; shape [t].
    let pos = mlxcel_core::arange_f32(0.0, t as f32, 1.0);
    // freq[j] = exp(-ln(10000) * j / half); shape [half].
    let j = mlxcel_core::arange_f32(0.0, half as f32, 1.0);
    let ln_theta = 10_000.0f32.ln();
    let exponent = mlxcel_core::multiply_scalar(&j, -ln_theta / half as f32);
    let freq = mlxcel_core::exp(&exponent);
    // ang[f, j] = pos[f] * freq[j]; shape [t, half].
    let ang = mlxcel_core::outer(&pos, &freq);
    let sin = mlxcel_core::sin(&ang);
    let cos = mlxcel_core::cos(&ang);
    let table = mlxcel_core::concatenate(&sin, &cos, 1); // [t, 2*half]
    if 2 * half < dim {
        // Odd embed_dim: pad the missing trailing column(s) with zeros.
        let pad = mlxcel_core::zeros(&[t, dim - 2 * half], mlxcel_core::dtype::FLOAT32);
        mlxcel_core::concatenate(&table, &pad, 1)
    } else {
        table
    }
}

/// Learned per-patch position grid with on-the-fly bicubic resampling.
pub(crate) struct Learnable2DInterpPosEmb {
    weight: UniquePtr<MlxArray>,
    init_h: i32,
    init_w: i32,
    dim: i32,
}

impl Learnable2DInterpPosEmb {
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
        Ok(Self {
            weight,
            init_h,
            init_w,
            dim,
        })
    }

    /// Construct directly from an in-memory grid (used by unit tests).
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

    /// Return the flattened position embedding for one `(h, w)` grid, shape
    /// `[h * w, dim]`.
    pub(crate) fn pos_for(&self, h: i32, w: i32) -> UniquePtr<MlxArray> {
        if h == self.init_h && w == self.init_w {
            return mlxcel_core::reshape(&self.weight, &[h * w, self.dim]);
        }
        let resampled = bicubic_resample(&self.weight, self.init_h, self.init_w, self.dim, h, w);
        mlxcel_core::reshape(&resampled, &[h * w, self.dim])
    }

    /// Build the per-item position term for one media grid, shape
    /// `[token_count, dim]`.
    ///
    /// For an image this is the flattened spatial table `[h*w, dim]`. For a
    /// video the spatial table is tiled `t` times along axis 0 (frame-major),
    /// and the temporal sinusoid `[t, dim]` (row `f` repeated `h*w` times) is
    /// added on top, giving `[t*h*w, dim]`.
    fn pos_for_media(&self, grid: &KimiMediaGrid) -> UniquePtr<MlxArray> {
        match *grid {
            KimiMediaGrid::Image { h, w } => self.pos_for(h, w),
            KimiMediaGrid::Video { t, h, w } => {
                let spatial = self.pos_for(h, w); // [h*w, dim]
                // Tile the per-frame spatial table t times along axis 0.
                let spatial = mlxcel_core::tile(&spatial, &[t, 1]); // [t*h*w, dim]
                // Temporal table [t, dim], each row repeated h*w times.
                let temporal = temporal_sinusoid(t, self.dim); // [t, dim]
                let temporal = mlxcel_core::repeat(&temporal, h * w, 0); // [t*h*w, dim]
                mlxcel_core::add(&spatial, &temporal)
            }
        }
    }

    /// Add the concatenated per-image position embeddings to `x`
    /// (`[total_tokens, dim]`), matching upstream `Learnable2DInterpPosEmb.__call__`.
    ///
    /// Image-only convenience over [`add_to_media`](Self::add_to_media); reused
    /// by other encoders (e.g. LFM2-VL) that share this position embedding.
    pub(crate) fn add_to(&self, x: &MlxArray, shapes: &[(i32, i32)]) -> UniquePtr<MlxArray> {
        let media_grids: Vec<KimiMediaGrid> = shapes
            .iter()
            .map(|&(h, w)| KimiMediaGrid::Image { h, w })
            .collect();
        self.add_to_media(x, &media_grids)
    }

    /// Add the concatenated per-item position embeddings to `x`
    /// (`[total_tokens, dim]`) for a mixed list of image and video grids,
    /// extending the upstream image behaviour with the temporal sinusoid for
    /// video clips.
    pub(crate) fn add_to_media(
        &self,
        x: &MlxArray,
        media_grids: &[KimiMediaGrid],
    ) -> UniquePtr<MlxArray> {
        let mut pos: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(media_grids.len());
        for grid in media_grids {
            pos.push(self.pos_for_media(grid));
        }
        let concatenated = if pos.len() == 1 {
            pos.into_iter().next().unwrap()
        } else {
            let mut iter = pos.into_iter();
            let first = iter.next().unwrap();
            iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
        };
        let concatenated = mlxcel_core::astype(&concatenated, mlxcel_core::array_dtype(x));
        mlxcel_core::add(x, &concatenated)
    }
}

/// Keys' cubic-convolution kernel, PyTorch ATen upsample coefficient `a = -0.75`.
fn cubic(x: f32) -> f32 {
    const A: f32 = -0.75;
    let x = x.abs();
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

/// Build the `[out_size, in_size]` cubic interpolation matrix for one axis
/// (PyTorch `align_corners=False`, edge-replicated boundaries).
fn interp_matrix(in_size: i32, out_size: i32) -> Vec<f32> {
    let scale = in_size as f32 / out_size as f32;
    let mut m = vec![0.0f32; (out_size * in_size) as usize];
    for o in 0..out_size {
        let src = (o as f32 + 0.5) * scale - 0.5;
        let base = src.floor();
        let t = src - base;
        let base = base as i32;
        // Four taps at base-1, base, base+1, base+2 with cubic weights.
        for k in -1i32..=2 {
            let w = cubic(t - k as f32);
            let mut idx = base + k;
            // Edge replicate (clamp) — matches ATen's boundary handling.
            if idx < 0 {
                idx = 0;
            } else if idx >= in_size {
                idx = in_size - 1;
            }
            m[(o * in_size + idx) as usize] += w;
        }
    }
    m
}

/// Separable bicubic resample of a `[in_h, in_w, dim]` grid to `[out_h, out_w, dim]`.
fn bicubic_resample(
    grid: &MlxArray,
    in_h: i32,
    in_w: i32,
    dim: i32,
    out_h: i32,
    out_w: i32,
) -> UniquePtr<MlxArray> {
    let w_h = mlxcel_core::from_slice_f32(&interp_matrix(in_h, out_h), &[out_h, in_h]);
    let w_w = mlxcel_core::from_slice_f32(&interp_matrix(in_w, out_w), &[out_w, in_w]);

    // Height contraction: [out_h, in_h] @ [in_h, in_w*dim] -> [out_h, in_w, dim].
    let grid2 = mlxcel_core::reshape(grid, &[in_h, in_w * dim]);
    let tmp = mlxcel_core::matmul(&w_h, &grid2);
    let tmp = mlxcel_core::reshape(&tmp, &[out_h, in_w, dim]);

    // Width contraction: move in_w to the front, contract, move back.
    let tmp = mlxcel_core::transpose_axes(&tmp, &[1, 0, 2]); // [in_w, out_h, dim]
    let tmp = mlxcel_core::reshape(&tmp, &[in_w, out_h * dim]);
    let out = mlxcel_core::matmul(&w_w, &tmp); // [out_w, out_h*dim]
    let out = mlxcel_core::reshape(&out, &[out_w, out_h, dim]);
    mlxcel_core::transpose_axes(&out, &[1, 0, 2]) // [out_h, out_w, dim]
}
