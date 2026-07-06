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

//! DeepSeek-OCR image processor.
//!
//! Produces one padded-to-square global view per image and, when the image is
//! larger than a tile, a dynamic grid of `tile_size` tiles (closest-aspect-ratio
//! selection, same family as `internvl`). Pixels are channels-last, scaled to
//! `[0, 1]` then normalized `(x - 0.5) / 0.5`. Also computes the flat
//! `<image>`-placeholder count each image expands to.
//!
//! Two variants share this code. [`DeepSeekOcrVariant::V1`] (DeepSeek-OCR) lays
//! the projected features out as a 2D mosaic with a per-row `image_newline`
//! column and only tiles images larger than a tile. [`DeepSeekOcrVariant::V2`]
//! (DeepSeek-OCR 2) emits flat feature runs (no newline column) and tiles every
//! image by default.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseekocr/processing_deepseekocr.py`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/processing_deepseekocr.py>).

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// `ceil((side / patch_size) / downsample_ratio)`, the per-axis query count that
/// drives the mosaic geometry (16 for the 1024 global view, 10 for a 640 tile,
/// 12 for a 768 tile).
fn num_queries(side: i32) -> i32 {
    const PATCH: i32 = 16;
    const DS: i32 = 4;
    ((side / PATCH) + DS - 1) / DS
}

/// Feature-layout family: `V1` (mosaic + newline column, tile only when large)
/// vs `V2` (flat runs, always tile).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekOcrVariant {
    V1,
    V2,
}

pub struct DeepSeekOcrProcessor {
    /// Global-view square size (1024 for OCR, resolution knob for OCR-2).
    pub base_size: i32,
    /// Tile size for the dynamic grid (640 OCR, 768 OCR-2).
    pub tile_size: i32,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub variant: DeepSeekOcrVariant,
    /// When false, skip the tile grid entirely (global view only). V2 exposes
    /// this as the "cropping disabled" mode; V1 always leaves it true.
    pub crop_enabled: bool,
}

/// Per-batch preprocessing output.
pub struct DeepSeekOcrPreprocessed {
    /// Global views, channels-last `(n_images, base, base, 3)`.
    pub global: UniquePtr<MlxArray>,
    /// All tiles across all images, channels-last `(total_tiles, tile, tile, 3)`;
    /// `None` when no image needed cropping.
    pub tiles: Option<UniquePtr<MlxArray>>,
    /// Per image: `(width_crop_num, height_crop_num)` of the tile grid.
    pub crops: Vec<(i32, i32)>,
    /// Per image: number of tiles emitted (0 = global view only). Distinguishes
    /// V2's `(1, 1)` single-tile case from a genuinely untiled image.
    pub tiles_per_image: Vec<i32>,
    /// Per image: the number of flat `<image>` placeholder tokens it expands to.
    pub placeholder_counts: Vec<i32>,
}

impl Default for DeepSeekOcrProcessor {
    fn default() -> Self {
        Self {
            base_size: 1024,
            tile_size: 640,
            min_tiles: 2,
            max_tiles: 9,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            variant: DeepSeekOcrVariant::V1,
            crop_enabled: true,
        }
    }
}

impl DeepSeekOcrProcessor {
    /// DeepSeek-OCR 2 preset: 768 tiles, grid product 1..6, always tiling, flat
    /// (no-newline) feature runs.
    pub fn v2() -> Self {
        Self {
            base_size: 1024,
            tile_size: 768,
            min_tiles: 1,
            max_tiles: 6,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            variant: DeepSeekOcrVariant::V2,
            crop_enabled: true,
        }
    }
}

impl DeepSeekOcrProcessor {
    /// Placeholder-token count for a `(w_crop, h_crop)` grid producing `n_tiles`
    /// tiles (`n_tiles == 0` means global view only).
    ///
    /// V1: global mosaic `(nb+1)*nb` (newline column) + view separator `1`, plus
    /// when tiled `(nt*w + 1) * (nt*h)`. V2: flat global `nb*nb` + separator `1`,
    /// plus when tiled `nt*nt * n_tiles` (no newline column).
    pub fn placeholder_count(&self, w_crop: i32, h_crop: i32, n_tiles: i32) -> i32 {
        let nb = num_queries(self.base_size);
        let nt = num_queries(self.tile_size);
        match self.variant {
            DeepSeekOcrVariant::V1 => {
                let mut n = (nb + 1) * nb + 1;
                if n_tiles > 0 {
                    n += (nt * w_crop + 1) * (nt * h_crop);
                }
                n
            }
            DeepSeekOcrVariant::V2 => {
                let mut n = nb * nb + 1;
                if n_tiles > 0 {
                    n += nt * nt * n_tiles;
                }
                n
            }
        }
    }

    /// `find_closest_aspect_ratio` (with the area tiebreak) over grids whose
    /// product is in `[min_tiles, max_tiles]`. Returns `(cols, rows)`.
    fn closest_grid(&self, w: u32, h: u32) -> (i32, i32) {
        let aspect = w as f64 / h as f64;
        let area = (w as f64) * (h as f64);
        let ts = self.tile_size as f64;
        let mut best = (1i32, 1i32);
        let mut best_diff = f64::INFINITY;
        let mut ratios: Vec<(i32, i32)> = Vec::new();
        for n in self.min_tiles..=self.max_tiles {
            for i in 1..=n {
                for j in 1..=n {
                    let p = i * j;
                    if p >= self.min_tiles && p <= self.max_tiles {
                        ratios.push((i as i32, j as i32));
                    }
                }
            }
        }
        ratios.sort_by_key(|&(i, j)| i * j);
        for &(i, j) in &ratios {
            let target = i as f64 / j as f64;
            let diff = (aspect - target).abs();
            if diff < best_diff {
                best_diff = diff;
                best = (i, j);
            } else if diff == best_diff && area > 0.5 * ts * ts * (i as f64) * (j as f64) {
                best = (i, j);
            }
        }
        best
    }

    /// `ImageOps.pad`: resize preserving aspect to fit `size`, then center-pad
    /// with the mean colour (grey 127) to exactly `size x size`.
    fn pad_to_square(&self, rgb: &image::RgbImage, size: i32) -> image::RgbImage {
        let (w, h) = (rgb.width(), rgb.height());
        let scale = (size as f64 / w as f64).min(size as f64 / h as f64);
        let new_w = ((w as f64 * scale).round() as u32).max(1);
        let new_h = ((h as f64 * scale).round() as u32).max(1);
        let resized = image::imageops::resize(rgb, new_w, new_h, FilterType::CatmullRom);
        let grey = image::Rgb([
            (self.mean[0] * 255.0).round() as u8,
            (self.mean[1] * 255.0).round() as u8,
            (self.mean[2] * 255.0).round() as u8,
        ]);
        let mut canvas = image::RgbImage::from_pixel(size as u32, size as u32, grey);
        let ox = ((size as u32).saturating_sub(new_w)) / 2;
        let oy = ((size as u32).saturating_sub(new_h)) / 2;
        image::imageops::overlay(&mut canvas, &resized, ox as i64, oy as i64);
        canvas
    }

    /// Tile an image into the `(cols, rows)` grid of `tile_size` crops.
    fn tiles_of(&self, rgb: &image::RgbImage, cols: i32, rows: i32) -> Vec<image::RgbImage> {
        let (tw, th) = (
            (self.tile_size * cols) as u32,
            (self.tile_size * rows) as u32,
        );
        let resized = image::imageops::resize(rgb, tw, th, FilterType::CatmullRom);
        let t = self.tile_size as u32;
        let mut out = Vec::with_capacity((cols * rows) as usize);
        for i in 0..(cols * rows) {
            let col = (i % cols) as u32;
            let row = (i / cols) as u32;
            let view = image::imageops::crop_imm(&resized, col * t, row * t, t, t);
            out.push(view.to_image());
        }
        out
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

    pub fn preprocess(&self, images: &[image::DynamicImage]) -> DeepSeekOcrPreprocessed {
        let mut global_px: Vec<f32> = Vec::new();
        let mut tile_px: Vec<f32> = Vec::new();
        let mut crops: Vec<(i32, i32)> = Vec::with_capacity(images.len());
        let mut tiles_per_image: Vec<i32> = Vec::with_capacity(images.len());
        let mut counts: Vec<i32> = Vec::with_capacity(images.len());
        let mut total_tiles = 0i32;

        for image in images {
            let rgb = image.to_rgb8();
            let (w, h) = (rgb.width(), rgb.height());
            // V1 crops only images larger than a tile; V2 tiles every image
            // (unless cropping is disabled), so `(1, 1)` there means one tile.
            let (cols, rows, n_tiles) = match self.variant {
                DeepSeekOcrVariant::V1 => {
                    if w as i32 <= self.tile_size && h as i32 <= self.tile_size {
                        (1, 1, 0)
                    } else {
                        let (c, r) = self.closest_grid(w, h);
                        (c, r, c * r)
                    }
                }
                DeepSeekOcrVariant::V2 => {
                    if !self.crop_enabled {
                        (1, 1, 0)
                    } else {
                        let (c, r) = self.closest_grid(w, h);
                        (c, r, c * r)
                    }
                }
            };

            let global = self.pad_to_square(&rgb, self.base_size);
            self.append_hwc(&global, &mut global_px);

            if n_tiles > 0 {
                for tile in self.tiles_of(&rgb, cols, rows) {
                    self.append_hwc(&tile, &mut tile_px);
                    total_tiles += 1;
                }
            }
            crops.push((cols, rows));
            tiles_per_image.push(n_tiles);
            counts.push(self.placeholder_count(cols, rows, n_tiles));
        }

        let global = mlxcel_core::from_slice_f32(
            &global_px,
            &[images.len() as i32, self.base_size, self.base_size, 3],
        );
        let global = mlxcel_core::astype(&global, mlxcel_core::dtype::BFLOAT16);
        let tiles = if total_tiles > 0 {
            let t = mlxcel_core::from_slice_f32(
                &tile_px,
                &[total_tiles, self.tile_size, self.tile_size, 3],
            );
            Some(mlxcel_core::astype(&t, mlxcel_core::dtype::BFLOAT16))
        } else {
            None
        };

        DeepSeekOcrPreprocessed {
            global,
            tiles,
            crops,
            tiles_per_image,
            placeholder_counts: counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_counts_match_reference_formula() {
        let p = DeepSeekOcrProcessor::default();
        // No tiles: (16+1)*16 + 1 = 273.
        assert_eq!(p.placeholder_count(1, 1, 0), 273);
        // 800x600 picks grid (3, 2): 273 + (10*3+1)*(10*2) = 273 + 31*20 = 893.
        assert_eq!(p.placeholder_count(3, 2, 6), 893);
    }

    #[test]
    fn closest_grid_800x600_is_3x2() {
        let p = DeepSeekOcrProcessor::default();
        assert_eq!(p.closest_grid(800, 600), (3, 2));
    }

    #[test]
    fn v2_placeholder_counts_are_flat() {
        let p = DeepSeekOcrProcessor::v2();
        // Cropping disabled (global only): 16*16 + 1 = 257.
        assert_eq!(p.placeholder_count(1, 1, 0), 257);
        // 1 tile (near-square default): 257 + 12*12*1 = 401.
        assert_eq!(p.placeholder_count(1, 1, 1), 401);
        // 6 tiles: 257 + 144*6 = 1121.
        assert_eq!(p.placeholder_count(3, 2, 6), 1121);
    }

    #[test]
    fn v2_grid_bounds_allow_single_tile() {
        let p = DeepSeekOcrProcessor::v2();
        // A square image picks a 1x1 grid (product 1 is now allowed).
        assert_eq!(p.closest_grid(1000, 1000), (1, 1));
    }
}
