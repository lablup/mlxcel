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

//! AnyRes (LLaVA-Next style) multi-tile image preprocessing.
//!
//! Shared tiling machinery for `image_grid_pinpoints`-driven VLMs: pick the
//! best grid resolution for the image, aspect-preserving-resize + symmetric
//! zero-pad into it, split into `image_size`-square tiles row-major, and prepend
//! a base tile (the whole image squashed to one square). This module also owns
//! the token-space unpad math so feature packing and prompt-token counting can
//! never disagree.
//!
//! Used by: Granite Vision (`granite_vision`), and (planned) Granite 4 Vision.
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite_vision/vision.py

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-image tiling result the model wrapper needs for feature packing and the
/// prompt expander needs for the image-token count. They must be computed from
/// the same numbers, so both consume this struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnyResTileInfo {
    pub orig_h: i32,
    pub orig_w: i32,
    pub n_tiles_h: i32,
    pub n_tiles_w: i32,
    /// `1 + n_tiles_h * n_tiles_w` (tile 0 is the base tile).
    pub num_tiles: i32,
}

/// Pick the grid resolution `(best_h, best_w)` from `pinpoints` (`(h, w)` pairs)
/// that maximizes the effectively-used area for an image of `(orig_h, orig_w)`,
/// breaking ties by minimal wasted area (first candidate wins a full tie).
/// Mirrors the LLaVA-Next `select_best_resolution`.
pub fn select_best_resolution(orig_h: i32, orig_w: i32, pinpoints: &[(i32, i32)]) -> (i32, i32) {
    let (ow, oh) = (orig_w as f64, orig_h as f64);
    let mut best: Option<(i32, i32)> = None;
    let mut best_effective = -1i64;
    let mut best_wasted = i64::MAX;
    for &(cand_h, cand_w) in pinpoints {
        let scale = (cand_w as f64 / ow).min(cand_h as f64 / oh);
        let down_w = (ow * scale) as i64; // truncation, matching int()
        let down_h = (oh * scale) as i64;
        let effective = (down_w * down_h).min((orig_w as i64) * (orig_h as i64));
        let wasted = (cand_h as i64) * (cand_w as i64) - effective;
        if effective > best_effective || (effective == best_effective && wasted < best_wasted) {
            best_effective = effective;
            best_wasted = wasted;
            best = Some((cand_h, cand_w));
        }
    }
    best.unwrap_or((orig_h.max(1), orig_w.max(1)))
}

/// Token-space unpad dims `(H, W)` after packing the grid tiles back into a
/// `(gh, gw)` feature grid (`gh = side * n_tiles_h`, `gw = side * n_tiles_w`).
/// Undoes the pixel-space symmetric padding; the `round(v, 7)` then truncate
/// rule is part of the checkpoint contract. Shared by packing and token
/// counting so they cannot drift.
pub fn unpadded_token_hw(
    orig_h: i32,
    orig_w: i32,
    n_tiles_h: i32,
    n_tiles_w: i32,
    side: i32,
) -> (i32, i32) {
    let gh = side * n_tiles_h;
    let gw = side * n_tiles_w;
    let (oh, ow) = (orig_h as f64, orig_w as f64);
    let round7_trunc = |v: f64| -> i32 { ((v * 1e7).round() / 1e7).trunc() as i32 };

    if ow / oh > gw as f64 / gh as f64 {
        let new_h = round7_trunc(oh * (gw as f64 / ow));
        let pad = (gh - new_h) / 2;
        ((gh - 2 * pad).max(1), gw)
    } else {
        let new_w = round7_trunc(ow * (gh as f64 / oh));
        let pad = (gw - new_w) / 2;
        (gh, (gw - 2 * pad).max(1))
    }
}

/// Total merged image-token count for one image: `base_tokens + H * (W + 1)`
/// (the `+1` per row is the appended `image_newline` column). Must use the same
/// integer math as feature packing.
pub fn num_image_tokens(info: &AnyResTileInfo, side: i32, base_tokens: i32) -> i32 {
    let (h, w) = unpadded_token_hw(
        info.orig_h,
        info.orig_w,
        info.n_tiles_h,
        info.n_tiles_w,
        side,
    );
    base_tokens + h * (w + 1)
}

/// AnyRes tiling processor: `image_size`-square tiles with SigLIP-style
/// mean/std normalization, channels-last output.
pub struct AnyResProcessor {
    /// `(h, w)` grid candidates.
    pub grid_pinpoints: Vec<(i32, i32)>,
    pub image_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl AnyResProcessor {
    pub fn new(grid_pinpoints: Vec<(i32, i32)>, image_size: u32) -> Self {
        Self {
            grid_pinpoints,
            image_size,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    /// Compute the tile layout for an image without decoding pixels. Kept
    /// separate so the prompt expander can size image-token runs cheaply.
    pub fn tile_info(&self, orig_h: i32, orig_w: i32) -> AnyResTileInfo {
        let (best_h, best_w) = select_best_resolution(orig_h, orig_w, &self.grid_pinpoints);
        let sz = self.image_size as i32;
        let n_tiles_h = (best_h / sz).max(1);
        let n_tiles_w = (best_w / sz).max(1);
        AnyResTileInfo {
            orig_h,
            orig_w,
            n_tiles_h,
            n_tiles_w,
            num_tiles: 1 + n_tiles_h * n_tiles_w,
        }
    }

    /// Push one `image_size`-square tile's normalized `(y, x, c)` pixels onto
    /// `out`, reading from `canvas` at pixel offset `(x0, y0)`.
    fn push_tile(&self, canvas: &image::RgbImage, x0: u32, y0: u32, out: &mut Vec<f32>) {
        let sz = self.image_size;
        let (cw, ch) = (canvas.width(), canvas.height());
        for y in 0..sz {
            for x in 0..sz {
                let (px, py) = (x0 + x, y0 + y);
                let pixel = if px < cw && py < ch {
                    *canvas.get_pixel(px, py)
                } else {
                    image::Rgb([0u8, 0, 0])
                };
                for c in 0..3 {
                    let v = pixel[c] as f32 / 255.0;
                    out.push((v - self.mean[c]) / self.std[c]);
                }
            }
        }
    }

    /// Preprocess a batch of images into channels-last tile pixels
    /// `[sum_i num_tiles_i, image_size, image_size, 3]` (base tile first per
    /// image, then grid tiles row-major) plus per-image tile info.
    pub fn preprocess_with_tiles(
        &self,
        images: &[DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<AnyResTileInfo>) {
        let sz = self.image_size;
        let mut data: Vec<f32> = Vec::new();
        let mut infos: Vec<AnyResTileInfo> = Vec::with_capacity(images.len());

        for image in images {
            let (orig_w, orig_h) = (image.width() as i32, image.height() as i32);
            let info = self.tile_info(orig_h, orig_w);
            infos.push(info);

            // Base tile: the whole image squashed to one square (not aspect-preserving).
            let base = image.resize_exact(sz, sz, FilterType::CatmullRom).to_rgb8();
            self.push_tile(&base, 0, 0, &mut data);

            // Grid tiles: aspect-preserving resize into (best_h, best_w) then
            // symmetric zero-pad, split row-major into image_size squares.
            let best_h = (info.n_tiles_h * sz as i32) as u32;
            let best_w = (info.n_tiles_w * sz as i32) as u32;
            let scale = (best_w as f64 / orig_w as f64).min(best_h as f64 / orig_h as f64);
            let new_w = ((orig_w as f64 * scale).round() as u32).clamp(1, best_w);
            let new_h = ((orig_h as f64 * scale).round() as u32).clamp(1, best_h);
            let resized = image
                .resize_exact(new_w, new_h, FilterType::CatmullRom)
                .to_rgb8();

            // Zero-padded canvas (black -> normalizes to -1), resized image centered.
            let mut canvas = image::RgbImage::from_pixel(best_w, best_h, image::Rgb([0, 0, 0]));
            let left = (best_w - new_w) / 2;
            let top = (best_h - new_h) / 2;
            for y in 0..new_h {
                for x in 0..new_w {
                    canvas.put_pixel(left + x, top + y, *resized.get_pixel(x, y));
                }
            }

            for tr in 0..info.n_tiles_h as u32 {
                for tc in 0..info.n_tiles_w as u32 {
                    self.push_tile(&canvas, tc * sz, tr * sz, &mut data);
                }
            }
        }

        let total_tiles: i32 = infos.iter().map(|i| i.num_tiles).sum();
        let pixel_values =
            mlxcel_core::from_slice_f32(&data, &[total_tiles, sz as i32, sz as i32, 3]);
        (pixel_values, infos)
    }
}

#[cfg(test)]
#[path = "anyres_tests.rs"]
mod tests;
