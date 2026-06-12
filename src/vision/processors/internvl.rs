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

//! InternVL (`internvl_chat`) image processor.
//!
//! Faithful port of
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/internvl_chat/processor.py.
//!
//! Dynamic tiling:
//! 1. Pick the target aspect ratio (i, j) with `min_dynamic_patch <= i*j <=
//!    max_dynamic_patch` whose ratio is closest to the image's aspect ratio.
//! 2. Resize to `(i*image_size, j*image_size)` (BICUBIC) and crop into
//!    `i*j` non-overlapping `image_size x image_size` tiles.
//! 3. When `use_thumbnail` is set and the image was split into more than one
//!    tile, append a single full-image thumbnail tile.
//!
//! Each tile is rescaled to `[0, 1]`, normalized with ImageNet mean/std, and
//! emitted in channels-first `[C, H, W]` layout.
//!
//! Output: a flattened `[total_tiles, 3, image_size, image_size]` tensor plus
//! the per-image tile counts (used by the runtime to expand `<IMG_CONTEXT>`
//! placeholders by `num_image_token * tiles`).
//!
//! Used by: InternVL (internvl_chat) VLM.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// ImageNet normalization constants (from the Python processor).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct InternVLProcessor {
    pub image_size: usize,
    pub min_dynamic_patch: usize,
    pub max_dynamic_patch: usize,
    pub use_thumbnail: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl InternVLProcessor {
    pub fn new(
        image_size: usize,
        min_dynamic_patch: usize,
        max_dynamic_patch: usize,
        use_thumbnail: bool,
    ) -> Self {
        Self {
            image_size,
            min_dynamic_patch,
            max_dynamic_patch,
            use_thumbnail,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        }
    }

    /// Enumerate candidate `(cols, rows)` aspect ratios with
    /// `min <= cols*rows <= max`, sorted by tile count (the Python set is
    /// later sorted by `x[0] * x[1]`).
    fn target_ratios(&self) -> Vec<(usize, usize)> {
        let mut ratios: Vec<(usize, usize)> = Vec::new();
        for n in self.min_dynamic_patch..=self.max_dynamic_patch {
            for i in 1..=n {
                for j in 1..=n {
                    let prod = i * j;
                    if prod >= self.min_dynamic_patch && prod <= self.max_dynamic_patch {
                        ratios.push((i, j));
                    }
                }
            }
        }
        ratios.sort_unstable();
        ratios.dedup();
        ratios.sort_by_key(|(i, j)| i * j);
        ratios
    }

    /// Port of `find_closest_aspect_ratio`. Returns `(cols, rows)`.
    fn find_closest_aspect_ratio(
        &self,
        aspect_ratio: f64,
        width: u32,
        height: u32,
    ) -> (usize, usize) {
        let ratios = self.target_ratios();
        let image_size = self.image_size as f64;
        let area = width as f64 * height as f64;

        let mut best_ratio_diff = f64::INFINITY;
        let mut best_ratio = (1usize, 1usize);
        for &(i, j) in &ratios {
            let target_aspect_ratio = i as f64 / j as f64;
            let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();
            if ratio_diff < best_ratio_diff {
                best_ratio_diff = ratio_diff;
                best_ratio = (i, j);
            } else if ratio_diff == best_ratio_diff {
                // Tie-break: prefer the ratio whose target area is closer to
                // the original image area.
                let target_area = image_size * image_size * i as f64 * j as f64;
                let best_area = image_size * image_size * best_ratio.0 as f64 * best_ratio.1 as f64;
                if (area - target_area).abs() < (area - best_area).abs() {
                    best_ratio = (i, j);
                }
            }
        }
        best_ratio
    }

    /// Split a single image into tiles (PIL-equivalent crops), optionally
    /// appending a thumbnail. Returns the list of `image_size x image_size`
    /// RGB tiles.
    fn dynamic_tiles(&self, image: &image::DynamicImage) -> Vec<image::RgbImage> {
        let rgb = image.to_rgb8();
        let (orig_w, orig_h) = (rgb.width(), rgb.height());
        if orig_w == 0 || orig_h == 0 {
            return Vec::new();
        }

        let aspect_ratio = orig_w as f64 / orig_h as f64;
        let (cols, rows) = self.find_closest_aspect_ratio(aspect_ratio, orig_w, orig_h);

        let target_w = (self.image_size * cols) as u32;
        let target_h = (self.image_size * rows) as u32;
        let blocks = cols * rows;

        // Resize whole image to the tiled canvas (BICUBIC ~ CatmullRom).
        let resized = image::DynamicImage::ImageRgb8(rgb.clone()).resize_exact(
            target_w,
            target_h,
            FilterType::CatmullRom,
        );
        let resized = resized.to_rgb8();

        let tile = self.image_size as u32;
        let mut tiles: Vec<image::RgbImage> = Vec::with_capacity(blocks + 1);
        for i in 0..blocks {
            let row_idx = (i / cols) as u32;
            let col_idx = (i % cols) as u32;
            let left = col_idx * tile;
            let top = row_idx * tile;
            // image::imageops::crop_imm yields a view; copy it to an owned tile.
            let view = image::imageops::crop_imm(&resized, left, top, tile, tile);
            tiles.push(view.to_image());
        }

        // Thumbnail tile (only when the image was actually split).
        if self.use_thumbnail && blocks > 1 {
            let thumb = image::DynamicImage::ImageRgb8(rgb)
                .resize_exact(tile, tile, FilterType::CatmullRom)
                .to_rgb8();
            tiles.push(thumb);
        }

        tiles
    }

    /// Normalize a tile into channels-first `[C, H, W]` f32 values appended to
    /// `out`.
    fn append_normalized_chw(&self, tile: &image::RgbImage, out: &mut Vec<f32>) {
        let h = self.image_size;
        let w = self.image_size;
        // Channels-first: write all of channel 0, then 1, then 2.
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let pixel = tile.get_pixel(x as u32, y as u32);
                    let val = pixel[c] as f32 / 255.0;
                    out.push((val - self.mean[c]) / self.std[c]);
                }
            }
        }
    }

    /// Preprocess a batch of images. Returns the flattened pixel tensor
    /// `[total_tiles, 3, image_size, image_size]` and the per-image tile
    /// counts (in input order).
    pub fn preprocess_with_tiles(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<usize>) {
        let mut all_pixels: Vec<f32> = Vec::new();
        let mut tiles_per_image: Vec<usize> = Vec::with_capacity(images.len());
        let mut total_tiles = 0usize;

        for image in images {
            let tiles = self.dynamic_tiles(image);
            tiles_per_image.push(tiles.len());
            total_tiles += tiles.len();
            for tile in &tiles {
                self.append_normalized_chw(tile, &mut all_pixels);
            }
        }

        let pixel_values = mlxcel_core::from_slice_f32(
            &all_pixels,
            &[
                total_tiles as i32,
                3,
                self.image_size as i32,
                self.image_size as i32,
            ],
        );
        (pixel_values, tiles_per_image)
    }
}

impl ImageProcessor for InternVLProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_tiles(images);
        pixel_values
    }
}

#[cfg(test)]
#[path = "internvl_tests.rs"]
mod tests;
