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

//! DeepSeek-VL2 image processor.
//!
//! Produces one padded-to-square global view per image plus a grid of local
//! tiles chosen from `candidate_resolutions` by a best-fit rule (maximise the
//! effective covered area, tie-break on least wasted area). Cropping is enabled
//! only when the request carries at most two images; otherwise every image gets
//! the `(1, 1)` grid (a single local tile). Both the global view and each local
//! tile are `image_size x image_size`; pixels are channels-last, scaled to
//! `[0, 1]` then normalised `(x - 0.5) / 0.5` with a mean-colour pad.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseek_vl_v2/processing_deepseek_vl_v2.py`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseek_vl_v2/processing_deepseek_vl_v2.py>).

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct DeepSeekVl2Processor {
    /// `[W, H]` tile targets; `candidate_resolutions[0].0` sets `image_size`.
    pub candidate_resolutions: Vec<(i32, i32)>,
    /// Global-view / local-tile square size (384 for the released checkpoints).
    pub image_size: i32,
    pub patch_size: i32,
    pub downsample_ratio: i32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// Per-batch preprocessing output.
pub struct DeepSeekVl2Preprocessed {
    /// Global views, channels-last `(n_images, image_size, image_size, 3)`.
    pub global: UniquePtr<MlxArray>,
    /// All local tiles across all images, channels-last
    /// `(total_tiles, image_size, image_size, 3)`. Always present: every image
    /// contributes at least one local tile.
    pub tiles: UniquePtr<MlxArray>,
    /// Per image: `(tw, th)` tile-grid counts (width count first).
    pub crops: Vec<(i32, i32)>,
    /// Per image: the number of flat `<image>` placeholder tokens it expands to.
    pub placeholder_counts: Vec<i32>,
}

impl DeepSeekVl2Processor {
    /// Build from `candidate_resolutions` (with `image_size = first[0]`).
    pub fn new(
        candidate_resolutions: Vec<(i32, i32)>,
        patch_size: i32,
        downsample_ratio: i32,
    ) -> Self {
        let image_size = candidate_resolutions
            .first()
            .map(|&(w, _)| w)
            .unwrap_or(384);
        Self {
            candidate_resolutions,
            image_size,
            patch_size,
            downsample_ratio,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    /// Per-axis query count `h' = w' = ceil((image_size / patch_size) / ds)`; the
    /// mosaic geometry (14 for a 384 view at patch 14, ds 2).
    fn num_queries(&self) -> i32 {
        let per_side = self.image_size / self.patch_size;
        (per_side + self.downsample_ratio - 1) / self.downsample_ratio
    }

    /// Placeholder-token count for a `(tw, th)` tile grid: global mosaic
    /// `h'*(w'+1)` + view separator `1` + local mosaic `(th*h')*(tw*w'+1)`.
    pub fn placeholder_count(&self, tw: i32, th: i32) -> i32 {
        let hq = self.num_queries();
        hq * (hq + 1) + 1 + (th * hq) * (tw * hq + 1)
    }

    /// Best-fit candidate resolution for an original `(w, h)`: maximise effective
    /// covered area, tie-break on least wasted area. Returns `(best_W, best_H)`.
    fn select_best_resolution(&self, w: u32, h: u32) -> (i32, i32) {
        let (w0, h0) = (w as f64, h as f64);
        let orig_area = (w as i64) * (h as i64);
        let mut best = (self.image_size, self.image_size);
        let mut best_effective = -1i64;
        let mut best_wasted = i64::MAX;
        for &(cw, ch) in &self.candidate_resolutions {
            let scale = (cw as f64 / w0).min(ch as f64 / h0);
            let rw = (w0 * scale).floor() as i64;
            let rh = (h0 * scale).floor() as i64;
            let effective = (rw * rh).min(orig_area);
            let wasted = (cw as i64) * (ch as i64) - effective;
            if effective > best_effective || (effective == best_effective && wasted < best_wasted) {
                best_effective = effective;
                best_wasted = wasted;
                best = (cw, ch);
            }
        }
        best
    }

    /// Resize preserving aspect to fit `(tw, th)`, then centre-pad with the mean
    /// colour (grey 127) to exactly `tw x th`.
    fn pad_to(&self, rgb: &image::RgbImage, tw: i32, th: i32) -> image::RgbImage {
        let (w, h) = (rgb.width(), rgb.height());
        let scale = (tw as f64 / w as f64).min(th as f64 / h as f64);
        let new_w = ((w as f64 * scale).round() as u32).max(1);
        let new_h = ((h as f64 * scale).round() as u32).max(1);
        let resized = image::imageops::resize(rgb, new_w, new_h, FilterType::CatmullRom);
        let grey = image::Rgb([
            (self.mean[0] * 255.0).round() as u8,
            (self.mean[1] * 255.0).round() as u8,
            (self.mean[2] * 255.0).round() as u8,
        ]);
        let mut canvas = image::RgbImage::from_pixel(tw as u32, th as u32, grey);
        let ox = ((tw as u32).saturating_sub(new_w)) / 2;
        let oy = ((th as u32).saturating_sub(new_h)) / 2;
        image::imageops::overlay(&mut canvas, &resized, ox as i64, oy as i64);
        canvas
    }

    fn append_hwc(&self, tile: &image::RgbImage, out: &mut Vec<f32>) {
        for y in 0..tile.height() {
            for x in 0..tile.width() {
                let p = tile.get_pixel(x, y);
                for c in 0..3 {
                    let v = p[c] as f32 / 255.0;
                    out.push((v - self.mean[c]) / self.std[c]);
                }
            }
        }
    }

    pub fn preprocess(&self, images: &[image::DynamicImage]) -> DeepSeekVl2Preprocessed {
        let sz = self.image_size;
        // Cropping (multi-tile grids) is only allowed with at most two images.
        let crop_enabled = images.len() <= 2;

        let mut global_px: Vec<f32> = Vec::new();
        let mut tile_px: Vec<f32> = Vec::new();
        let mut crops: Vec<(i32, i32)> = Vec::with_capacity(images.len());
        let mut counts: Vec<i32> = Vec::with_capacity(images.len());
        let mut total_tiles = 0i32;

        for image in images {
            let rgb = image.to_rgb8();
            let (w, h) = (rgb.width(), rgb.height());

            let (best_w, best_h) = if crop_enabled {
                self.select_best_resolution(w, h)
            } else {
                (sz, sz)
            };
            let (tw, th) = (best_w / sz, best_h / sz);

            // Global thumbnail: pad-to-square into image_size.
            let global = self.pad_to(&rgb, sz, sz);
            self.append_hwc(&global, &mut global_px);

            // Local views: pad into (best_w, best_h), crop image_size tiles
            // row-major (row outer, col inner).
            let padded = self.pad_to(&rgb, best_w, best_h);
            for row in 0..th {
                for col in 0..tw {
                    let view = image::imageops::crop_imm(
                        &padded,
                        (col * sz) as u32,
                        (row * sz) as u32,
                        sz as u32,
                        sz as u32,
                    )
                    .to_image();
                    self.append_hwc(&view, &mut tile_px);
                    total_tiles += 1;
                }
            }

            crops.push((tw, th));
            counts.push(self.placeholder_count(tw, th));
        }

        let global = mlxcel_core::from_slice_f32(&global_px, &[images.len() as i32, sz, sz, 3]);
        let global = mlxcel_core::astype(&global, mlxcel_core::dtype::BFLOAT16);
        let tiles = mlxcel_core::from_slice_f32(&tile_px, &[total_tiles, sz, sz, 3]);
        let tiles = mlxcel_core::astype(&tiles, mlxcel_core::dtype::BFLOAT16);

        DeepSeekVl2Preprocessed {
            global,
            tiles,
            crops,
            placeholder_counts: counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proc() -> DeepSeekVl2Processor {
        DeepSeekVl2Processor::new(
            vec![
                (384, 384),
                (384, 768),
                (768, 384),
                (384, 1152),
                (1152, 384),
                (768, 768),
                (1152, 768),
                (768, 1152),
                (1152, 1152),
            ],
            14,
            2,
        )
    }

    #[test]
    fn num_queries_is_14_for_384_patch14_ds2() {
        assert_eq!(proc().num_queries(), 14);
    }

    #[test]
    fn placeholder_counts_match_reference_formula() {
        let p = proc();
        // 1x1 grid: global 14*15 + sep 1 + local 14*15 = 210 + 1 + 210 = 421.
        assert_eq!(p.placeholder_count(1, 1), 421);
        // 2x1 grid: 210 + 1 + (1*14)*(2*14+1) = 210 + 1 + 14*29 = 617.
        assert_eq!(p.placeholder_count(2, 1), 617);
        // 1x2 grid: 210 + 1 + (2*14)*(1*14+1) = 210 + 1 + 28*15 = 631.
        assert_eq!(p.placeholder_count(1, 2), 631);
    }

    #[test]
    fn best_resolution_small_square_picks_single_tile() {
        // A square image no larger than one tile (<= 384) is fully covered by
        // every candidate, so the min-wasted tie-break selects (384, 384).
        assert_eq!(proc().select_best_resolution(350, 350), (384, 384));
    }

    #[test]
    fn best_resolution_wide_picks_landscape() {
        // A 1500x400 image is wide: the landscape candidate covers more area.
        let (w, _h) = proc().select_best_resolution(1500, 400);
        assert!(
            w >= 768,
            "wide image should pick a wider-than-tall grid, got w={w}"
        );
    }

    #[test]
    fn best_resolution_prefers_larger_effective_area() {
        // A large near-square image should not collapse to a single tile: the
        // biggest candidate preserves the most original detail (max effective).
        let (w, h) = proc().select_best_resolution(1100, 1100);
        assert_eq!((w, h), (1152, 1152));
    }
}
