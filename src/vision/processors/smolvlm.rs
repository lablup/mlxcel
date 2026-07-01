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

//! SmolVLM / Idefics3 image processor.
//!
//! Faithful port of the SmolVLM image path (HuggingFace
//! `SmolVLMImageProcessor`, which reuses the Idefics3 splitting scheme):
//!
//! 1. Each image is normalized to SigLIP's `[-1, 1]` range (mean `0.5`, std
//!    `0.5` per channel).
//! 2. When `do_image_splitting` is set and the image is larger than a single
//!    tile, the image is resized onto a `rows x cols` canvas of
//!    `image_size x image_size` tiles and cropped into `rows*cols` sub-image
//!    tiles, followed by a single global-image tile (the whole frame resized to
//!    one tile). Upstream emits the split tiles first and the global tile last,
//!    matching the `<row_i_col_j>` runs followed by the `<global-img>` run in
//!    the prompt; this processor preserves that order.
//! 3. When splitting is disabled (or the image already fits one tile), only the
//!    single global tile is emitted.
//!
//! Output: a flattened `[total_tiles, 3, image_size, image_size]` tensor plus
//! the per-image tile counts (used by the runtime to expand the `<image>`
//! placeholders by `num_image_token * tiles`). The tile grid geometry
//! (`rows x cols`) is a real-checkpoint-validation concern deferred to the
//! orchestrator; the per-request token accounting stays internally consistent
//! because both the pixel tensor and the prompt expansion are driven by the
//! same tile counts.
//!
//! Used by: SmolVLM (`smolvlm`) VLM.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// SigLIP normalization constants used by SmolVLM (rescale to `[-1, 1]`).
const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

pub struct SmolVLMProcessor {
    /// Side length of a single square tile (the vision tower's `image_size`,
    /// e.g. 384). Both split tiles and the global tile use this size.
    pub image_size: usize,
    /// Whether to split large images into a tile grid plus a global tile.
    pub do_image_splitting: bool,
    /// Upper bound on tiles per side when splitting (derived from the
    /// processor's outer `longest_edge` budget). Keeps a pathological aspect
    /// ratio from exploding the tile count.
    pub max_splits_per_side: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl SmolVLMProcessor {
    pub fn new(image_size: usize, do_image_splitting: bool, max_splits_per_side: usize) -> Self {
        Self {
            image_size,
            do_image_splitting,
            max_splits_per_side: max_splits_per_side.max(1),
            mean: SIGLIP_MEAN,
            std: SIGLIP_STD,
        }
    }

    /// Compute the `(cols, rows)` tile grid for a single image. Returns
    /// `(1, 1)` when splitting is disabled or the image fits one tile.
    fn tile_grid(&self, width: u32, height: u32) -> (usize, usize) {
        if !self.do_image_splitting || width == 0 || height == 0 {
            return (1, 1);
        }
        let tile = self.image_size as u32;
        if width <= tile && height <= tile {
            return (1, 1);
        }
        let cols = width.div_ceil(tile).max(1) as usize;
        let rows = height.div_ceil(tile).max(1) as usize;
        (
            cols.min(self.max_splits_per_side),
            rows.min(self.max_splits_per_side),
        )
    }

    /// Split a single image into `image_size x image_size` RGB tiles. When the
    /// grid is larger than 1x1 the split tiles come first and a single
    /// global-image tile is appended last (matching the upstream prompt order).
    fn tiles_for_image(&self, image: &image::DynamicImage) -> Vec<image::RgbImage> {
        let rgb = image.to_rgb8();
        let (orig_w, orig_h) = (rgb.width(), rgb.height());
        let tile = self.image_size as u32;
        if orig_w == 0 || orig_h == 0 {
            return Vec::new();
        }

        let (cols, rows) = self.tile_grid(orig_w, orig_h);

        // Single global tile only.
        if cols == 1 && rows == 1 {
            let global = image::DynamicImage::ImageRgb8(rgb)
                .resize_exact(tile, tile, FilterType::CatmullRom)
                .to_rgb8();
            return vec![global];
        }

        // Resize onto the tiled canvas, then crop into cols*rows tiles.
        let canvas_w = tile * cols as u32;
        let canvas_h = tile * rows as u32;
        let resized = image::DynamicImage::ImageRgb8(rgb.clone())
            .resize_exact(canvas_w, canvas_h, FilterType::CatmullRom)
            .to_rgb8();

        let mut tiles: Vec<image::RgbImage> = Vec::with_capacity(cols * rows + 1);
        for row in 0..rows as u32 {
            for col in 0..cols as u32 {
                let view = image::imageops::crop_imm(&resized, col * tile, row * tile, tile, tile);
                tiles.push(view.to_image());
            }
        }

        // Global thumbnail tile (whole frame resized to one tile), appended last.
        let global = image::DynamicImage::ImageRgb8(rgb)
            .resize_exact(tile, tile, FilterType::CatmullRom)
            .to_rgb8();
        tiles.push(global);

        tiles
    }

    /// Normalize a tile into channels-first `[C, H, W]` f32 values appended to
    /// `out` (rescale to `[0, 1]` then SigLIP mean/std).
    fn append_normalized_chw(&self, tile: &image::RgbImage, out: &mut Vec<f32>) {
        let h = self.image_size;
        let w = self.image_size;
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
    /// `[total_tiles, 3, image_size, image_size]` and the per-image tile counts
    /// (in input order).
    pub fn preprocess_with_tiles(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<usize>) {
        let mut all_pixels: Vec<f32> = Vec::new();
        let mut tiles_per_image: Vec<usize> = Vec::with_capacity(images.len());
        let mut total_tiles = 0usize;

        for image in images {
            let tiles = self.tiles_for_image(image);
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

impl ImageProcessor for SmolVLMProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_tiles(images);
        pixel_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, value: u8) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb([value, value, value]),
        ))
    }

    #[test]
    fn single_global_tile_when_splitting_disabled() {
        let proc = SmolVLMProcessor::new(32, false, 4);
        let img = solid(200, 100, 255);
        let (pixels, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(tiles, vec![1]);
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 3, 32, 32]);
    }

    #[test]
    fn siglip_normalization_maps_white_to_one() {
        // White pixel: 255/255 = 1.0 -> (1.0 - 0.5) / 0.5 = 1.0.
        let proc = SmolVLMProcessor::new(4, false, 4);
        let img = solid(4, 4, 255);
        let (pixels, _) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        let first = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
        mlxcel_core::eval(&first);
        assert!((mlxcel_core::item_f32(&first) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn splitting_emits_grid_plus_global_tile() {
        // 96x64 image, tile 32 -> ceil(96/32)=3 cols, ceil(64/32)=2 rows -> 6
        // split tiles + 1 global tile = 7.
        let proc = SmolVLMProcessor::new(32, true, 4);
        let img = solid(96, 64, 128);
        let (pixels, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(tiles, vec![7]);
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![7, 3, 32, 32]);
    }

    #[test]
    fn small_image_stays_single_tile_even_when_splitting_enabled() {
        let proc = SmolVLMProcessor::new(64, true, 4);
        let img = solid(50, 40, 10);
        let (_pixels, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(tiles, vec![1]);
    }

    #[test]
    fn max_splits_per_side_caps_grid() {
        // Very wide image but max_splits_per_side = 2 caps cols at 2.
        let proc = SmolVLMProcessor::new(16, true, 2);
        let img = solid(200, 16, 200);
        let (_pixels, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        // cols capped at 2, rows = 1 -> 2 split tiles + 1 global = 3.
        assert_eq!(tiles, vec![3]);
    }
}
