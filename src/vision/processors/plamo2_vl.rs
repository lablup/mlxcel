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

//! PLaMo 2.1 VL (`plamo2vl`) dynamic-tile image processor (issue #1380).
//!
//! Port of the Eagle2-style tiler in the checkpoint's
//! `processing_plamo2_vl.py` (mirrored by mlx-vlm's `plamo2vl/processing.py`):
//!
//! 1. Candidate grids `(cols, rows)` with `1 <= cols*rows <= max_num`, visited
//!    in ascending tile count. The grid maximizing the ratio factor
//!    `min(ar, cols/rows) / max(ar, cols/rows)` wins; on a tie the later
//!    (larger) grid wins only when `w*h > 0.5 * S^2 * cols * rows`. This is
//!    not InternVL's absolute-difference rule.
//! 2. The image is resized to `(cols*S, rows*S)` (bicubic) and cropped
//!    row-major; a full-image `S x S` thumbnail is appended when more than one
//!    crop was produced.
//! 3. Context budget: starting at `max_num = 24`, while
//!    `sum_i (2 + tiles_i * num_image_token) >= tile_context_length` the cap
//!    drops by 2 (by 1 once it is below 3) and every image is retiled. With one
//!    image this ends at no more than 4 tiles (2918 tokens).
//!
//! Each tile becomes channels-first `[3, S, S]` f32,
//! `(pixel * rescale - mean) / std` (mean = std = 0.5 on shipped checkpoints).
//!
//! Used by: `vision::plamo2_vl::Plamo2VlModel`, `multimodal::vlm_runtime`.

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::plamo2_vl_config::Plamo2VlProcessorConfig;

/// Upstream's fixed tile cap before the context budget shrinks it.
pub const PLAMO2_VL_MAX_TILES: usize = 24;

#[derive(Debug, Clone)]
pub struct Plamo2VlProcessor {
    /// Tile edge `S` (384 on shipped checkpoints), already rounded up to the
    /// downsample stride when `downsample_ratio != 1`.
    pub image_size: usize,
    pub patch_size: usize,
    pub downsample_ratio: f32,
    pub tile_context_length: usize,
    pub max_tiles: usize,
    pub rescale_factor: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Plamo2VlProcessor {
    pub fn new(config: &Plamo2VlProcessorConfig) -> Self {
        let mut image_size = config.size.width;
        let patch_size = config.patch_size.max(1);
        if config.downsample_ratio != 1.0 && config.downsample_ratio > 0.0 {
            // Upstream rounds the tile up to `patch * (1 / ratio)`; tiles are
            // then already stride-aligned, so its per-tile zero pad is a no-op.
            let stride = patch_size * (1.0 / config.downsample_ratio) as usize;
            if stride > 0 {
                image_size = image_size.div_ceil(stride) * stride;
            }
        }
        Self {
            image_size,
            patch_size,
            downsample_ratio: config.downsample_ratio,
            tile_context_length: config.tile_context_length,
            max_tiles: PLAMO2_VL_MAX_TILES,
            rescale_factor: config.rescale_factor,
            mean: config.image_mean,
            std: config.image_std,
        }
    }

    /// Image-feature tokens per tile: `(S / patch * downsample_ratio)^2`, 729
    /// for the 384px / patch-14 tower.
    pub fn num_image_token(&self) -> usize {
        let side = (self.image_size / self.patch_size) as f32 * self.downsample_ratio;
        (side * side) as usize
    }

    /// Candidate `(cols, rows)` grids for `max_num`, in ascending tile count
    /// (ties ordered by `(cols, rows)` so the walk is deterministic).
    fn target_ratios(max_num: usize) -> Vec<(usize, usize)> {
        let mut ratios: Vec<(usize, usize)> = (1..=max_num)
            .flat_map(|i| (1..=max_num).map(move |j| (i, j)))
            .filter(|(i, j)| i * j <= max_num)
            .collect();
        ratios.sort_by_key(|&(i, j)| (i * j, i, j));
        ratios
    }

    /// Port of upstream `_find_closest_aspect_ratio` (ratio-factor rule).
    pub(crate) fn grid_for(&self, width: u32, height: u32, max_num: usize) -> (usize, usize) {
        let aspect_ratio = f64::from(width) / f64::from(height);
        let area = f64::from(width) * f64::from(height);
        let size = self.image_size as f64;
        let mut best_factor = f64::NEG_INFINITY;
        let mut best = (1, 1);
        for (cols, rows) in Self::target_ratios(max_num.max(1)) {
            let target = cols as f64 / rows as f64;
            let factor = aspect_ratio.min(target) / aspect_ratio.max(target);
            if factor > best_factor {
                best_factor = factor;
                best = (cols, rows);
            } else if factor == best_factor && area > 0.5 * size * size * (cols * rows) as f64 {
                best = (cols, rows);
            }
        }
        best
    }

    /// Tiles one image produces under `max_num`: the grid's crops plus the
    /// thumbnail when the grid has more than one cell.
    fn tile_count(&self, width: u32, height: u32, max_num: usize) -> usize {
        let (cols, rows) = self.grid_for(width, height, max_num);
        let blocks = cols * rows;
        if blocks > 1 { blocks + 1 } else { blocks }
    }

    /// The context-budget shrink loop. Returns the final cap and the tile
    /// count per image (in input order).
    pub(crate) fn tile_budget(&self, sizes: &[(u32, u32)]) -> (usize, Vec<usize>) {
        let per_tile = self.num_image_token();
        let mut max_num = self.max_tiles.max(1);
        loop {
            let counts: Vec<usize> = sizes
                .iter()
                .map(|&(w, h)| self.tile_count(w, h, max_num))
                .collect();
            let total: usize = counts.iter().map(|tiles| 2 + tiles * per_tile).sum();
            if total < self.tile_context_length || max_num == 1 {
                return (max_num, counts);
            }
            max_num -= if max_num >= 3 { 2 } else { 1 };
        }
    }

    /// Crop one image into its grid tiles (+ thumbnail).
    fn tiles(&self, rgb: &image::RgbImage, max_num: usize) -> Vec<image::RgbImage> {
        let (width, height) = rgb.dimensions();
        let (cols, rows) = self.grid_for(width, height, max_num);
        let tile = self.image_size as u32;
        let resized = image::imageops::resize(
            rgb,
            tile * cols as u32,
            tile * rows as u32,
            FilterType::CatmullRom,
        );
        let mut tiles = Vec::with_capacity(cols * rows + 1);
        for index in 0..cols * rows {
            let left = (index % cols) as u32 * tile;
            let top = (index / cols) as u32 * tile;
            tiles.push(image::imageops::crop_imm(&resized, left, top, tile, tile).to_image());
        }
        if tiles.len() > 1 {
            tiles.push(image::imageops::resize(
                rgb,
                tile,
                tile,
                FilterType::CatmullRom,
            ));
        }
        tiles
    }

    /// Append one tile as normalized channels-first values.
    fn append_normalized_chw(&self, tile: &image::RgbImage, out: &mut Vec<f32>) {
        for c in 0..3 {
            for pixel in tile.pixels() {
                let value = f32::from(pixel[c]) * self.rescale_factor;
                out.push((value - self.mean[c]) / self.std[c]);
            }
        }
    }

    /// Tile and normalize every image under one shared context budget.
    /// Returns `[total_tiles, 3, S, S]` and the tile count per image.
    pub fn preprocess_with_tiles(
        &self,
        images: &[image::DynamicImage],
    ) -> Result<(UniquePtr<MlxArray>, Vec<usize>), String> {
        if images.is_empty() {
            return Err("PLaMo 2.1 VL preprocessing needs at least one image".to_string());
        }
        let rgbs: Vec<image::RgbImage> = images.iter().map(|img| img.to_rgb8()).collect();
        let sizes: Vec<(u32, u32)> = rgbs.iter().map(|img| img.dimensions()).collect();
        if let Some((w, h)) = sizes.iter().find(|(w, h)| *w == 0 || *h == 0) {
            return Err(format!("PLaMo 2.1 VL cannot tile an empty {w}x{h} image"));
        }
        let (max_num, counts) = self.tile_budget(&sizes);

        let side = self.image_size;
        let total: usize = counts.iter().sum();
        let mut pixels = Vec::with_capacity(total * 3 * side * side);
        for (rgb, &expected) in rgbs.iter().zip(&counts) {
            let tiles = self.tiles(rgb, max_num);
            debug_assert_eq!(tiles.len(), expected);
            for tile in &tiles {
                self.append_normalized_chw(tile, &mut pixels);
            }
        }
        let pixel_values =
            mlxcel_core::from_slice_f32(&pixels, &[total as i32, 3, side as i32, side as i32]);
        Ok((pixel_values, counts))
    }
}

#[cfg(test)]
#[path = "plamo2_vl_tests.rs"]
mod tests;
