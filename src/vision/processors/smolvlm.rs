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
//! 1. Each image is normalized with the checkpoint's `image_mean` / `image_std`.
//! 2. When `do_image_splitting` is set and the image is larger than one tile,
//!    the longest edge is resized to the processor `size.longest_edge` budget,
//!    dimensions are rounded up to exact `max_image_size.longest_edge` tile
//!    multiples in aspect-preserving order, and the image is cropped into a
//!    row-major grid followed by a single global thumbnail tile.
//! 3. When splitting is disabled or the image fits one tile, the image is
//!    aspect-preserved into one square tile and padded.
//!
//! Output: a flattened `[total_tiles, 3, tile_size, tile_size]` tensor plus the
//! per-image tile layouts used by the prompt expander. Split layouts have
//! `rows > 0`, `cols > 0`, and `rows * cols + 1` tiles with the global tile
//! last; single-tile layouts are encoded as `rows = cols = 0`.
//!
//! Used by: SmolVLM (`smolvlm`) VLM.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// SigLIP normalization constants used by SmolVLM when the checkpoint does not
/// provide `image_mean` / `image_std`.
pub const DEFAULT_SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const DEFAULT_SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];
const MAX_RESIZED_EDGE: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileLayout {
    pub rows: usize,
    pub cols: usize,
}

impl TileLayout {
    pub const fn single() -> Self {
        Self { rows: 0, cols: 0 }
    }

    pub const fn split(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub const fn is_split(self) -> bool {
        self.rows > 0 && self.cols > 0
    }

    pub const fn total_tiles(self) -> usize {
        if self.is_split() {
            self.rows * self.cols + 1
        } else {
            1
        }
    }

    pub const fn checked_total_tiles(self) -> Option<usize> {
        if self.is_split() {
            match self.rows.checked_mul(self.cols) {
                Some(grid) => grid.checked_add(1),
                None => None,
            }
        } else {
            Some(1)
        }
    }
}

pub struct SmolVLMProcessor {
    /// Side length of a single square tile (`max_image_size.longest_edge`, equal
    /// to the vision tower's `image_size`, e.g. 364/384/512).
    pub tile_size: usize,
    /// Outer longest-edge resize budget (`size.longest_edge`, e.g. 1456/1536).
    pub longest_edge: usize,
    /// Whether to split large images into a tile grid plus a global tile.
    pub do_image_splitting: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl SmolVLMProcessor {
    pub fn new(
        tile_size: usize,
        do_image_splitting: bool,
        longest_edge: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Self {
        Self {
            tile_size: tile_size.max(1),
            longest_edge: longest_edge.max(tile_size.max(1)),
            do_image_splitting,
            mean,
            std,
        }
    }

    /// Backwards-compatible constructor for tests and callers that want the
    /// checkpoint defaults.
    pub fn with_defaults(tile_size: usize, do_image_splitting: bool) -> Self {
        Self::new(
            tile_size,
            do_image_splitting,
            tile_size.max(1) * 4,
            DEFAULT_SIGLIP_MEAN,
            DEFAULT_SIGLIP_STD,
        )
    }

    fn resize_longest_edge_even(&self, width: u32, height: u32, longest_edge: u32) -> (u32, u32) {
        let width = width.max(1);
        let height = height.max(1);
        let longest_edge = longest_edge.max(1);
        if width >= height {
            let new_width = longest_edge;
            let mut new_height =
                ((longest_edge as u64 * height as u64) / width as u64).max(1) as u32;
            new_height += new_height % 2;
            (new_width, new_height.max(1))
        } else {
            let new_height = longest_edge;
            let mut new_width =
                ((longest_edge as u64 * width as u64) / height as u64).max(1) as u32;
            new_width += new_width % 2;
            (new_width.max(1), new_height)
        }
    }

    fn round_up_to_tile(value: u32, tile: u32) -> u32 {
        value.div_ceil(tile).max(1) * tile
    }

    /// Return `(width, height)` after the upstream split resize chain.
    fn split_canvas_size(&self, width: u32, height: u32) -> (u32, u32) {
        let tile = self.tile_size as u32;
        let longest = (self.longest_edge as u32).min(MAX_RESIZED_EDGE);
        let (resized_w, resized_h) = self.resize_longest_edge_even(width, height, longest);

        if resized_w >= resized_h {
            let canvas_w = Self::round_up_to_tile(resized_w, tile);
            let projected_h =
                ((canvas_w as u64 * resized_h as u64) / resized_w.max(1) as u64).max(1) as u32;
            let canvas_h = Self::round_up_to_tile(projected_h, tile);
            (canvas_w, canvas_h)
        } else {
            let canvas_h = Self::round_up_to_tile(resized_h, tile);
            let projected_w =
                ((canvas_h as u64 * resized_w as u64) / resized_h.max(1) as u64).max(1) as u32;
            let canvas_w = Self::round_up_to_tile(projected_w, tile);
            (canvas_w, canvas_h)
        }
    }

    /// Compute the tile layout for one image. Returns `rows = cols = 0` for the
    /// single-tile path, otherwise the split grid dimensions.
    pub fn tile_layout(&self, width: u32, height: u32) -> TileLayout {
        if !self.do_image_splitting || width == 0 || height == 0 {
            return TileLayout::single();
        }
        let tile = self.tile_size as u32;
        if width <= tile && height <= tile {
            return TileLayout::single();
        }
        let (canvas_w, canvas_h) = self.split_canvas_size(width, height);
        if canvas_w <= tile && canvas_h <= tile {
            TileLayout::single()
        } else {
            TileLayout::split((canvas_h / tile) as usize, (canvas_w / tile) as usize)
        }
    }

    fn single_tile(&self, rgb: &image::RgbImage) -> image::RgbImage {
        let tile = self.tile_size as u32;
        let (width, height) = (rgb.width().max(1), rgb.height().max(1));
        let (out_w, out_h) = if width >= height {
            let out_w = tile;
            let out_h =
                ((tile as u64 * height as u64 + (width as u64 / 2)) / width as u64).max(1) as u32;
            (out_w, out_h.max(1).min(tile))
        } else {
            let out_h = tile;
            let out_w =
                ((tile as u64 * width as u64 + (height as u64 / 2)) / height as u64).max(1) as u32;
            (out_w.max(1).min(tile), out_h)
        };
        let resized = image::DynamicImage::ImageRgb8(rgb.clone())
            .resize_exact(out_w, out_h, FilterType::Triangle)
            .to_rgb8();
        let mut canvas = image::RgbImage::from_pixel(tile, tile, image::Rgb([0, 0, 0]));
        let x0 = (tile - out_w) / 2;
        let y0 = (tile - out_h) / 2;
        image::imageops::replace(&mut canvas, &resized, i64::from(x0), i64::from(y0));
        canvas
    }

    /// Split a single image into square RGB tiles and return those tiles plus
    /// the layout. Split tiles are row-major and the global tile is last.
    fn tiles_for_image(&self, image: &image::DynamicImage) -> (Vec<image::RgbImage>, TileLayout) {
        let rgb = image.to_rgb8();
        let (orig_w, orig_h) = (rgb.width(), rgb.height());
        if orig_w == 0 || orig_h == 0 {
            let tile = self.tile_size as u32;
            let blank = image::RgbImage::from_pixel(tile, tile, image::Rgb([0, 0, 0]));
            return (vec![blank], TileLayout::single());
        }

        let layout = self.tile_layout(orig_w, orig_h);
        if !layout.is_split() {
            return (vec![self.single_tile(&rgb)], layout);
        }

        let tile = self.tile_size as u32;
        let canvas_w = tile * layout.cols as u32;
        let canvas_h = tile * layout.rows as u32;
        let resized = image::DynamicImage::ImageRgb8(rgb.clone())
            .resize_exact(canvas_w, canvas_h, FilterType::Triangle)
            .to_rgb8();

        let mut tiles: Vec<image::RgbImage> = Vec::with_capacity(layout.total_tiles());
        for row in 0..layout.rows as u32 {
            for col in 0..layout.cols as u32 {
                let view = image::imageops::crop_imm(&resized, col * tile, row * tile, tile, tile);
                tiles.push(view.to_image());
            }
        }

        let global = image::DynamicImage::ImageRgb8(rgb)
            .resize_exact(tile, tile, FilterType::Triangle)
            .to_rgb8();
        tiles.push(global);

        (tiles, layout)
    }

    /// Normalize a tile into channels-first `[C, H, W]` f32 values appended to
    /// `out` (rescale to `[0, 1]` then checkpoint mean/std).
    fn append_normalized_chw(&self, tile: &image::RgbImage, out: &mut Vec<f32>) {
        let h = self.tile_size;
        let w = self.tile_size;
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
    /// `[total_tiles, 3, tile_size, tile_size]` and per-image tile layouts.
    pub fn preprocess_with_tiles(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<TileLayout>) {
        let mut all_pixels: Vec<f32> = Vec::new();
        let mut layouts: Vec<TileLayout> = Vec::with_capacity(images.len());
        let mut total_tiles = 0usize;

        for image in images {
            let (tiles, layout) = self.tiles_for_image(image);
            layouts.push(layout);
            total_tiles = total_tiles.saturating_add(tiles.len());
            for tile in &tiles {
                self.append_normalized_chw(tile, &mut all_pixels);
            }
        }

        let pixel_values = mlxcel_core::from_slice_f32(
            &all_pixels,
            &[
                total_tiles as i32,
                3,
                self.tile_size as i32,
                self.tile_size as i32,
            ],
        );
        (pixel_values, layouts)
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

    fn quadrant_image() -> image::DynamicImage {
        let mut img = image::RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let value = match (y < 2, x < 2) {
                    (true, true) => 32,
                    (true, false) => 96,
                    (false, true) => 160,
                    (false, false) => 224,
                };
                img.put_pixel(x, y, image::Rgb([value, value, value]));
            }
        }
        image::DynamicImage::ImageRgb8(img)
    }

    fn normalized_first_channel(pixels: &MlxArray, tile_index: i32, y: i32, x: i32) -> f32 {
        let value = mlxcel_core::slice(
            pixels,
            &[tile_index, 0, y, x],
            &[tile_index + 1, 1, y + 1, x + 1],
        );
        mlxcel_core::eval(&value);
        mlxcel_core::item_f32(&value)
    }

    fn denormalize(value: f32) -> u8 {
        (((value * 0.5 + 0.5) * 255.0).round() as i32).clamp(0, 255) as u8
    }

    #[test]
    fn single_global_tile_when_splitting_disabled() {
        let proc = SmolVLMProcessor::with_defaults(32, false);
        let img = solid(200, 100, 255);
        let (pixels, layouts) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(layouts, vec![TileLayout::single()]);
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 3, 32, 32]);
    }

    #[test]
    fn siglip_normalization_maps_white_to_one() {
        // White pixel: 255/255 = 1.0 -> (1.0 - 0.5) / 0.5 = 1.0.
        let proc = SmolVLMProcessor::with_defaults(4, false);
        let img = solid(4, 4, 255);
        let (pixels, _) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        let first = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
        mlxcel_core::eval(&first);
        assert!((mlxcel_core::item_f32(&first) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn resize_chain_matches_reference_geometry() {
        let proc = SmolVLMProcessor::new(364, true, 1456, DEFAULT_SIGLIP_MEAN, DEFAULT_SIGLIP_STD);
        assert_eq!(proc.tile_layout(800, 600), TileLayout::split(3, 4));
        assert_eq!(proc.tile_layout(300, 200), TileLayout::single());
        assert_eq!(proc.tile_layout(1456, 1456), TileLayout::split(4, 4));

        let img = solid(800, 600, 128);
        let (pixels, layouts) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(layouts, vec![TileLayout::split(3, 4)]);
        assert_eq!(layouts[0].total_tiles(), 13);
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![13, 3, 364, 364]);
    }

    #[test]
    fn tiles_are_row_major_with_global_last() {
        let proc = SmolVLMProcessor::new(2, true, 4, DEFAULT_SIGLIP_MEAN, DEFAULT_SIGLIP_STD);
        let img = quadrant_image();
        let (pixels, layouts) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(layouts, vec![TileLayout::split(2, 2)]);
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![5, 3, 2, 2]);

        let observed: Vec<u8> = (0..4)
            .map(|tile| denormalize(normalized_first_channel(&pixels, tile, 0, 0)))
            .collect();
        assert_eq!(observed, vec![32, 96, 160, 224]);

        // The global tile is a resize of the whole image and therefore differs
        // from the bottom-right split tile that immediately precedes it.
        let split_last = denormalize(normalized_first_channel(&pixels, 3, 0, 0));
        let global_first = denormalize(normalized_first_channel(&pixels, 4, 0, 0));
        assert_ne!(split_last, global_first);
    }

    #[test]
    fn small_image_stays_single_tile_even_when_splitting_enabled() {
        let proc = SmolVLMProcessor::new(64, true, 256, DEFAULT_SIGLIP_MEAN, DEFAULT_SIGLIP_STD);
        let img = solid(50, 40, 10);
        let (_pixels, layouts) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(layouts, vec![TileLayout::single()]);
    }

    #[test]
    fn zero_dimension_image_preserves_single_tile_invariant() {
        let proc = SmolVLMProcessor::with_defaults(8, true);
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(0, 8));
        let (pixels, layouts) = proc.preprocess_with_tiles(std::slice::from_ref(&img));

        assert_eq!(layouts, vec![TileLayout::single()]);
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 3, 8, 8]);
    }
}
