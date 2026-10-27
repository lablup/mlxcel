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

//! MoonViT learnable 2D interpolated position embedding.
//!
//! Port of `Learnable2DInterpPosEmb` from upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/vision.py.
//!
//! The module holds a learned `[init_h, init_w, dim]` grid. For a native
//! resolution whose patch grid equals the learned grid it flattens the grid
//! directly (the upstream fast path); otherwise it resamples the grid to the
//! target `[h, w]` before flattening and adding it to the patch embeddings.
//!
//! Two resample modes exist ([`PosEmbResample`]):
//!
//! - `Bicubic` (MoonViT, the default). Upstream calls a custom Metal
//!   `bicubic_interpolate` kernel; here the resample is a separable
//!   interpolation-matrix contraction, `out = W_h · grid · W_wᵀ`, where `W_h`
//!   and `W_w` are precomputed cubic-convolution weight matrices. This matches
//!   PyTorch's non-antialiased bicubic (`align_corners=False`, cubic
//!   `a = -0.75`, edge-replicated boundaries), the mode upstream's
//!   `bicubic_interpolate(..., size=shape)` uses by default.
//! - `BilinearAntialias` (LFM2-VL, whose tower is SigLIP2). SigLIP2 resizes its
//!   grid with `F.interpolate(mode="bilinear", antialias=True,
//!   align_corners=False)`, which mlx-vlm mirrors since Blaizzy/mlx-vlm#2266;
//!   the matrices come from the shared
//!   [`resample_matrix::interp_matrix`](super::super::resample_matrix::interp_matrix)
//!   with the triangle kernel, and the contraction runs in float32 before
//!   casting back to the table dtype.
//!
//! The weights sum to one per output position in both modes, so a constant grid
//! resamples to the same constant, and a same-size resample is the identity.

use mlxcel_core::layers::{QuantizedTensorShapes, UnifiedEmbedding, validate_quantized_packing};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::resample_matrix::{self, ResampleKernel};
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
///
/// Used by: `encoders::kimi_vl` (MoonViT video position term) and
/// `encoders::moonvit3d_pos_emb::time_embedding` (Kimi K3, `t > 1`).
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

/// How [`Learnable2DInterpPosEmb`] resizes its learned grid to an image's
/// patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PosEmbResample {
    /// PyTorch non-antialiased bicubic with edge replication (MoonViT).
    #[default]
    Bicubic,
    /// PyTorch bilinear with `antialias=True`, computed in float32 (SigLIP2,
    /// used by LFM2-VL).
    BilinearAntialias,
}

/// Learned per-patch position grid with on-the-fly resampling.
pub(crate) struct Learnable2DInterpPosEmb {
    weight: UniquePtr<MlxArray>,
    init_h: i32,
    init_w: i32,
    dim: i32,
    resample: PosEmbResample,
}

// Used by: encoders::lfm2_vl, encoders::kimi_vl
impl Learnable2DInterpPosEmb {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        init_h: i32,
        init_w: i32,
        dim: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.weight");
        let stored = weights
            .get(&key)
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        let stored_shape = mlxcel_core::array_shape(stored);
        let flat_rows = init_h
            .checked_mul(init_w)
            .ok_or_else(|| format!("{key} grid dimensions [{init_h}, {init_w}] overflow i32"))?;
        let leading_matches = match stored_shape.as_slice() {
            [rows, _] => *rows == flat_rows,
            [height, width, _] => *height == init_h && *width == init_w,
            _ => false,
        };
        if init_h <= 0 || init_w <= 0 || dim <= 0 || !leading_matches {
            return Err(format!(
                "{key} shape mismatch: expected [{flat_rows}, {dim}] or [{init_h}, {init_w}, {dim}], got {stored_shape:?}"
            ));
        }
        let logical_shape = if stored_shape.len() == 2 {
            vec![flat_rows, dim]
        } else {
            vec![init_h, init_w, dim]
        };
        let (effective_group_size, effective_bits) =
            crate::vision::weights::resolve_param_quantization(
                weights,
                &key,
                &logical_shape,
                Some((group_size, bits)),
            )?
            .unwrap_or((group_size, bits));
        let table =
            UnifiedEmbedding::from_weights(weights, prefix, effective_group_size, effective_bits)?;
        if let Some(quantized) = table.quantized() {
            if mlxcel_core::array_dtype(&quantized.weight) != mlxcel_core::dtype::UINT32 {
                return Err(format!("{key} must have uint32 packed weights"));
            }
            let scales_shape = mlxcel_core::array_shape(&quantized.scales);
            let biases_shape = quantized
                .biases
                .as_ref()
                .map(|biases| mlxcel_core::array_shape(biases));
            validate_quantized_packing(
                &key,
                &QuantizedTensorShapes {
                    weight: &stored_shape,
                    scales: &scales_shape,
                    biases: biases_shape.as_deref(),
                },
                dim as usize,
                quantized.group_size,
                quantized.bits,
                &quantized.mode,
            )?;
        } else if stored_shape[stored_shape.len() - 1] != dim {
            return Err(format!(
                "{key} shape mismatch: expected last dimension {dim}, got {stored_shape:?}"
            ));
        }
        let weight = table.dequantized_weight();
        let dense_shape = mlxcel_core::array_shape(&weight);
        if dense_shape != logical_shape {
            return Err(format!(
                "{key} dequantized to shape {dense_shape:?}, expected {logical_shape:?}"
            ));
        }
        let weight = mlxcel_core::reshape(&weight, &[init_h, init_w, dim]);
        Ok(Self {
            weight,
            init_h,
            init_w,
            dim,
            resample: PosEmbResample::Bicubic,
        })
    }

    /// Select the resample mode (the default is [`PosEmbResample::Bicubic`]).
    #[must_use]
    pub(crate) fn with_resample(mut self, resample: PosEmbResample) -> Self {
        self.resample = resample;
        self
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
            resample: PosEmbResample::Bicubic,
        }
    }

    /// Return the flattened position embedding for one `(h, w)` grid, shape
    /// `[h * w, dim]`.
    pub(crate) fn pos_for(&self, h: i32, w: i32) -> UniquePtr<MlxArray> {
        if h == self.init_h && w == self.init_w {
            return mlxcel_core::reshape(&self.weight, &[h * w, self.dim]);
        }
        let resampled = match self.resample {
            PosEmbResample::Bicubic => {
                bicubic_resample(&self.weight, self.init_h, self.init_w, self.dim, h, w)
            }
            PosEmbResample::BilinearAntialias => {
                let table_dtype = mlxcel_core::array_dtype(&self.weight);
                let grid = mlxcel_core::astype(&self.weight, mlxcel_core::dtype::FLOAT32);
                let w_h =
                    resample_matrix::interp_matrix(self.init_h, h, ResampleKernel::Triangle, true);
                let w_w =
                    resample_matrix::interp_matrix(self.init_w, w, ResampleKernel::Triangle, true);
                let out = resample_matrix::separable_resample(
                    &grid,
                    self.init_h,
                    self.init_w,
                    self.dim,
                    h,
                    w,
                    &w_h,
                    &w_w,
                );
                mlxcel_core::astype(&out, table_dtype)
            }
        };
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

/// Build the `[out_size, in_size]` cubic interpolation matrix for one axis
/// (PyTorch `align_corners=False`, edge-replicated boundaries, taps not
/// renormalized). This differs from the shared
/// [`resample_matrix::interp_matrix`] at the borders, where that one drops and
/// renormalizes out-of-range taps, so MoonViT keeps its own.
pub(crate) fn bicubic_interp_matrix(in_size: i32, out_size: i32) -> Vec<f32> {
    let scale = in_size as f32 / out_size as f32;
    let mut m = vec![0.0f32; (out_size * in_size) as usize];
    for o in 0..out_size {
        let src = (o as f32 + 0.5) * scale - 0.5;
        let base = src.floor();
        let t = src - base;
        let base = base as i32;
        // Four taps at base-1, base, base+1, base+2 with cubic weights.
        for k in -1i32..=2 {
            let w = resample_matrix::cubic(t - k as f32);
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
    let w_h = bicubic_interp_matrix(in_h, out_h);
    let w_w = bicubic_interp_matrix(in_w, out_w);
    resample_matrix::separable_resample(grid, in_h, in_w, dim, out_h, out_w, &w_h, &w_w)
}

#[cfg(test)]
#[path = "kimi_vl_pos_emb_quant_tests.rs"]
mod quantized_layout_tests;
