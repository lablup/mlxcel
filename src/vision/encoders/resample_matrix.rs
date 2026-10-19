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

//! Separable interpolation matrices for resampling learned position grids.
//!
//! `mlxcel_core` has no `interpolate` op, so the vision towers that resize a
//! learned `[h, w, dim]` position table do it as two matrix contractions,
//! `out = W_h · grid · W_wᵀ`, with per-axis weight matrices built here on the
//! host. Every matrix row sums to one, so a constant grid resamples to the same
//! constant and a same-size resample is the identity.
//!
//! [`interp_matrix`] is PyTorch's antialiased formulation
//! (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`,
//! `_compute_indices_min_size_weights_aa`) with a pluggable kernel: when
//! downsampling with `antialias`, the kernel support stretches by the scale
//! factor and the taps of each output row are renormalized. Upsampling (or
//! downsampling without `antialias`) uses the kernel's base support, truncated
//! at the borders and renormalized.
//!
//! Used by: DeepSeek-OCR SAM / CLIP (`deepseekocr_sam`, cubic), LFM2-VL
//! (`kimi_vl::pos_emb` in `BilinearAntialias` mode, triangle), and the
//! MoonViT bicubic grid (`kimi_vl::pos_emb`, which keeps its own
//! edge-replicated matrix and shares the contraction).

use mlxcel_core::{MlxArray, UniquePtr};

/// The 1-D interpolation kernel an [`interp_matrix`] row is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResampleKernel {
    /// Keys' cubic convolution with `a = -0.75` (support 2).
    Cubic,
    /// The linear "tent" kernel `max(0, 1 - |x|)` (support 1), which is what
    /// PyTorch's `bilinear` mode evaluates.
    Triangle,
}

impl ResampleKernel {
    /// Half-width of the kernel at scale 1.
    fn support(self) -> f32 {
        match self {
            Self::Cubic => 2.0,
            Self::Triangle => 1.0,
        }
    }

    fn weight(self, x: f32) -> f32 {
        match self {
            Self::Cubic => cubic(x),
            Self::Triangle => {
                let x = x.abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            }
        }
    }
}

/// Keys' cubic-convolution kernel, PyTorch ATen upsample coefficient `a = -0.75`.
pub(crate) fn cubic(x: f32) -> f32 {
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

/// `[out_size, in_size]` interpolation matrix for one axis with
/// `align_corners=False` sampling.
///
/// With `antialias` and downsampling (`scale = in / out > 1`) the kernel
/// support is stretched by `scale` and the taps renormalized, which is
/// PyTorch's `interpolate(..., antialias=True)`. Otherwise the kernel keeps its
/// base support; taps falling outside the input are dropped and the remaining
/// ones renormalized.
pub(crate) fn interp_matrix(
    in_size: i32,
    out_size: i32,
    kernel: ResampleKernel,
    antialias: bool,
) -> Vec<f32> {
    let scale = in_size as f32 / out_size as f32;
    let filterscale = if antialias { scale.max(1.0) } else { 1.0 };
    let support = kernel.support() * filterscale;
    let inv = 1.0 / filterscale;
    let mut m = vec![0.0f32; (out_size * in_size) as usize];
    for o in 0..out_size {
        let center = (o as f32 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as i32).max(0);
        let xmax = ((center + support + 0.5).floor() as i32).min(in_size);
        let mut taps: Vec<(i32, f32)> = Vec::new();
        let mut total = 0.0f32;
        for idx in xmin..xmax {
            let w = kernel.weight((idx as f32 + 0.5 - center) * inv);
            taps.push((idx, w));
            total += w;
        }
        if total == 0.0 {
            // Degenerate (out of range): clamp to nearest source pixel.
            let idx = center.floor().clamp(0.0, (in_size - 1) as f32) as i32;
            m[(o * in_size + idx) as usize] += 1.0;
            continue;
        }
        for (idx, w) in taps {
            m[(o * in_size + idx) as usize] += w / total;
        }
    }
    m
}

/// Separable resample of a `[in_h, in_w, dim]` grid to `[out_h, out_w, dim]`
/// with the given `[out_h, in_h]` and `[out_w, in_w]` row-major weight
/// matrices: rows first, then columns. The matrices are float32, so the result
/// is float32 for a float32 or narrower grid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn separable_resample(
    grid: &MlxArray,
    in_h: i32,
    in_w: i32,
    dim: i32,
    out_h: i32,
    out_w: i32,
    w_h: &[f32],
    w_w: &[f32],
) -> UniquePtr<MlxArray> {
    let w_h = mlxcel_core::from_slice_f32(w_h, &[out_h, in_h]);
    let w_w = mlxcel_core::from_slice_f32(w_w, &[out_w, in_w]);

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

#[cfg(test)]
#[path = "resample_matrix_tests.rs"]
mod tests;
