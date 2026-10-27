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

//! Tiled VAE decode.
//!
//! A port of `TilingConfig` and `decode_image_tiled` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/tiling.py>,
//! with the Blaizzy/mlx-vlm#1381 blend: each tile is weighted by a raised-cosine
//! ramp over the overlap on the sides that have a neighbour, the weights are
//! accumulated alongside the pixels, and the sum is divided by the accumulated
//! weight, so overlaps neither double-count nor shift color. Blending runs on
//! the host in f32, as upstream's NumPy loop does.
//!
//! Used by: Flux2, Bonsai.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Upstream `TilingConfig` (sizes in pixels, overlap in latent cells).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilingConfig {
    /// Tiling is on only when this is `Some(n)` with `n > 1`.
    pub tiles_per_dim: Option<u32>,
    pub tile_size: i32,
    pub overlap: i32,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            tiles_per_dim: Some(8),
            tile_size: 128,
            overlap: 8,
        }
    }
}

impl TilingConfig {
    /// Whether this configuration tiles at all.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.tiles_per_dim.is_some_and(|n| n > 1)
    }

    /// Upstream `_resolve_tiling` for an explicit override or the `auto` mode:
    /// tiled when `max(height, width) >= 2 * tile_size`.
    #[must_use]
    pub fn resolve(height: u32, width: u32, override_tiled: Option<bool>) -> Option<Self> {
        let config = Self::default();
        match override_tiled {
            Some(true) => Some(config),
            Some(false) => None,
            None => {
                (i64::from(height.max(width)) >= 2 * i64::from(config.tile_size)).then_some(config)
            }
        }
    }
}

/// `0.5 - 0.5 * cos(t * pi)` for `t = linspace(0, 1, n)` in f32.
fn cos_ramp(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => (0..n)
            .map(|i| {
                // NumPy's linspace steps in f64 and casts to f32.
                let t = (i as f64 / (n - 1) as f64) as f32;
                0.5 - 0.5 * (t * std::f32::consts::PI).cos()
            })
            .collect(),
    }
}

/// Decode `latent` (`[1, C, H, W]`) tile by tile with `decode_fn` and blend.
///
/// `decode_fn` maps a `[1, C, h, w]` latent tile to `[1, 3, h * scale, w *
/// scale]` pixels. `tile_size` is in pixels, `tile_overlap` in pixels; a batch
/// larger than one, or a latent that fits one tile, is decoded whole. Returns
/// `[1, 3, H * scale, W * scale]` f32.
pub fn decode_image_tiled(
    latent: &MlxArray,
    decode_fn: &dyn Fn(&MlxArray) -> Result<UniquePtr<MlxArray>>,
    tile_size: (i32, i32),
    tile_overlap: (i32, i32),
    spatial_scale: i32,
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(latent);
    if shape.len() != 4 {
        bail!("decode_image_tiled expects [B, C, H, W], got {shape:?}");
    }
    let (batch, channels, height_lat, width_lat) = (shape[0], shape[1], shape[2], shape[3]);
    let scale = spatial_scale.max(1);
    let (tile_h, tile_w) = tile_size;
    let (overlap_h, overlap_w) = tile_overlap;
    let latent_tile_h = (tile_h / scale).max(1);
    let latent_tile_w = (tile_w / scale).max(1);
    if batch != 1 || (height_lat <= latent_tile_h && width_lat <= latent_tile_w) {
        return decode_fn(latent);
    }

    let height_out = (height_lat * scale) as usize;
    let width_out = (width_lat * scale) as usize;
    let latent_overlap_h = (overlap_h / scale).min(latent_tile_h - 1).max(0);
    let latent_overlap_w = (overlap_w / scale).min(latent_tile_w - 1).max(0);
    let stride_h = (latent_tile_h - latent_overlap_h).max(1);
    let stride_w = (latent_tile_w - latent_overlap_w).max(1);
    let ramp_h = cos_ramp(overlap_h.max(0) as usize);
    let ramp_w = cos_ramp(overlap_w.max(0) as usize);

    let mut out: Option<(Vec<f32>, usize)> = None;
    let mut count = vec![0.0_f32; height_out * width_out];

    for y_lat in (0..height_lat).step_by(stride_h as usize) {
        let y_end = (y_lat + latent_tile_h).min(height_lat);
        for x_lat in (0..width_lat).step_by(stride_w as usize) {
            let x_end = (x_lat + latent_tile_w).min(width_lat);
            if (y_lat > 0 && y_end - y_lat <= latent_overlap_h)
                || (x_lat > 0 && x_end - x_lat <= latent_overlap_w)
            {
                continue;
            }
            let tile =
                mlxcel_core::slice(latent, &[0, 0, y_lat, x_lat], &[1, channels, y_end, x_end]);
            let decoded = mlxcel_core::astype(&*decode_fn(&tile)?, dtype::FLOAT32);
            let ds = mlxcel_core::array_shape(&decoded);
            if ds.len() != 4 || ds[0] != 1 {
                bail!("tile decode returned shape {ds:?}");
            }
            let (tc, th, tw) = (ds[1] as usize, ds[2] as usize, ds[3] as usize);
            let pixels = mlxcel_core::utils::array_to_vec_f32(&decoded);

            let y_out = (y_lat * scale) as usize;
            let x_out = (x_lat * scale) as usize;
            let eff_h = (((y_end - y_lat) * scale) as usize)
                .min(th)
                .min(height_out - y_out);
            let eff_w = (((x_end - x_lat) * scale) as usize)
                .min(tw)
                .min(width_out - x_out);

            let (acc, acc_c) =
                out.get_or_insert_with(|| (vec![0.0; height_out * width_out * tc], tc));
            if *acc_c != tc {
                bail!("tile decode changed channel count from {acc_c} to {tc}");
            }

            let ov_h = (overlap_h.max(0) as usize).min(eff_h.saturating_sub(1));
            let ov_w = (overlap_w.max(0) as usize).min(eff_w.saturating_sub(1));
            let mut wh = vec![1.0_f32; eff_h];
            let mut ww = vec![1.0_f32; eff_w];
            if ov_h > 0 {
                if y_lat > 0 {
                    wh[..ov_h].copy_from_slice(&ramp_h[..ov_h]);
                }
                if y_end < height_lat {
                    for (i, w) in wh[eff_h - ov_h..].iter_mut().enumerate() {
                        *w = 1.0 - ramp_h[i];
                    }
                }
            }
            if ov_w > 0 {
                if x_lat > 0 {
                    ww[..ov_w].copy_from_slice(&ramp_w[..ov_w]);
                }
                if x_end < width_lat {
                    for (i, w) in ww[eff_w - ov_w..].iter_mut().enumerate() {
                        *w = 1.0 - ramp_w[i];
                    }
                }
            }

            for y in 0..eff_h {
                for x in 0..eff_w {
                    let weight = wh[y] * ww[x];
                    let dst = (y_out + y) * width_out + (x_out + x);
                    for c in 0..tc {
                        acc[dst * tc + c] += pixels[(c * th + y) * tw + x] * weight;
                    }
                    count[dst] += weight;
                }
            }
        }
    }

    let Some((acc, tc)) = out else {
        bail!("tiled decode produced no tile");
    };
    // `out / clip(count, 1e-6)`, then HWC -> CHW.
    let mut chw = vec![0.0_f32; acc.len()];
    for p in 0..height_out * width_out {
        let denom = count[p].max(1e-6);
        for c in 0..tc {
            chw[c * height_out * width_out + p] = acc[p * tc + c] / denom;
        }
    }
    Ok(mlxcel_core::from_slice_f32(
        &chw,
        &[1, tc as i32, height_out as i32, width_out as i32],
    ))
}

#[cfg(test)]
#[path = "tiling_tests.rs"]
mod tests;
