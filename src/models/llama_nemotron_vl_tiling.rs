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

//! Dynamic image tiling for Llama-Nemotron-VL-Embed.
//!
//! This is InternVL's `dynamic_preprocess` with the two differences the
//! checkpoint's `processing_llama_nemotron_vl.py` introduces, which is why it
//! does not reuse [`crate::vision::processors::internvl::InternVLProcessor`]
//! (changing that type would change the InternVL VLM's behaviour):
//!
//! 1. `find_closest_aspect_ratio` maximizes `min(area_ratio, 0.6) *
//!    min(target / actual, actual / target)` instead of minimizing the aspect
//!    difference alone, so a grid that covers more of the original area wins
//!    ties the ratio-only rule would decide differently.
//! 2. Normalization is SigLIP's (mean and std `0.5`), not ImageNet's, and the
//!    tiles come out channels-last because
//!    [`crate::vision::encoders::siglip::SigLipVisionModel`] convolves
//!    `[B, H, W, C]`.
//!
//! The tile budget comes from `processor_config.json` (`image_size` 512,
//! `max_input_tiles` 6, `use_thumbnail` true): an image is split into 1 to 6
//! `512x512` crops plus, whenever it was actually split, one full-image
//! thumbnail tile.

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// SigLIP normalization (`norm_type: "siglip"` in `processor_config.json`).
pub(crate) const SIGLIP_MEAN: f32 = 0.5;
/// SigLIP normalization standard deviation.
pub(crate) const SIGLIP_STD: f32 = 0.5;

/// The area cap the reference scoring function applies: covering more than
/// 60% of the original area stops earning a higher score, so past that point
/// the aspect match alone decides.
const AREA_SATURATION: f64 = 0.6;

/// Tiling parameters read from `processor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NemotronTiling {
    /// Square tile edge in pixels (`image_size`).
    pub(crate) image_size: usize,
    /// Smallest number of tiles a grid may use (`min_dynamic_patch`).
    pub(crate) min_tiles: usize,
    /// Largest number of tiles a grid may use (`max_input_tiles`).
    pub(crate) max_tiles: usize,
    /// Append a full-image thumbnail tile when the image was split.
    pub(crate) use_thumbnail: bool,
}

impl Default for NemotronTiling {
    fn default() -> Self {
        Self {
            image_size: 512,
            min_tiles: 1,
            max_tiles: 6,
            use_thumbnail: true,
        }
    }
}

impl NemotronTiling {
    /// Candidate `(cols, rows)` grids with `min_tiles <= cols * rows <=
    /// max_tiles`, ordered by tile count then by `cols` and `rows`.
    ///
    /// The reference sorts a Python set by `cols * rows` alone; ordering the
    /// remaining freedom explicitly makes the tie-break in
    /// [`Self::closest_aspect_ratio`] deterministic across runs.
    fn candidate_grids(&self) -> Vec<(usize, usize)> {
        let mut grids: Vec<(usize, usize)> = Vec::new();
        for n in self.min_tiles..=self.max_tiles {
            for cols in 1..=n {
                for rows in 1..=n {
                    let tiles = cols * rows;
                    if tiles >= self.min_tiles && tiles <= self.max_tiles {
                        grids.push((cols, rows));
                    }
                }
            }
        }
        grids.sort_unstable_by_key(|&(cols, rows)| (cols * rows, cols, rows));
        grids.dedup();
        grids
    }

    /// Port of the checkpoint's `find_closest_aspect_ratio`: pick the grid
    /// with the highest `min(area_ratio, 0.6) * min(target / actual, actual /
    /// target)`, keeping the first winner on a tie.
    pub(crate) fn closest_aspect_ratio(&self, width: u32, height: u32) -> (usize, usize) {
        let aspect = width as f64 / height as f64;
        let area = width as f64 * height as f64;
        let tile_area = (self.image_size * self.image_size) as f64;

        let mut best = (1usize, 1usize);
        let mut best_score = f64::NEG_INFINITY;
        for (cols, rows) in self.candidate_grids() {
            let target = cols as f64 / rows as f64;
            let area_ratio = (cols * rows) as f64 * tile_area / area;
            let score = area_ratio.min(AREA_SATURATION) * (target / aspect).min(aspect / target);
            if score > best_score {
                best_score = score;
                best = (cols, rows);
            }
        }
        best
    }

    /// Split one image into `image_size` square tiles, appending the
    /// thumbnail when the grid held more than one tile.
    pub(crate) fn tiles(&self, image: &image::DynamicImage) -> Vec<image::RgbImage> {
        let rgb = image.to_rgb8();
        let (width, height) = (rgb.width(), rgb.height());
        if width == 0 || height == 0 {
            return Vec::new();
        }

        let (cols, rows) = self.closest_aspect_ratio(width, height);
        let edge = self.image_size as u32;
        let resized = image::DynamicImage::ImageRgb8(rgb.clone())
            .resize_exact(
                edge * cols as u32,
                edge * rows as u32,
                FilterType::CatmullRom,
            )
            .to_rgb8();

        let blocks = cols * rows;
        let mut tiles: Vec<image::RgbImage> = Vec::with_capacity(blocks + 1);
        for index in 0..blocks {
            let left = (index % cols) as u32 * edge;
            let top = (index / cols) as u32 * edge;
            tiles.push(image::imageops::crop_imm(&resized, left, top, edge, edge).to_image());
        }
        if self.use_thumbnail && blocks > 1 {
            tiles.push(
                image::DynamicImage::ImageRgb8(rgb)
                    .resize_exact(edge, edge, FilterType::CatmullRom)
                    .to_rgb8(),
            );
        }
        tiles
    }

    /// Preprocess images into a channels-last `[total_tiles, H, W, 3]` f32
    /// tensor plus the per-image tile counts in input order.
    pub(crate) fn preprocess(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<usize>) {
        let edge = self.image_size;
        let mut pixels: Vec<f32> = Vec::new();
        let mut counts: Vec<usize> = Vec::with_capacity(images.len());
        let mut total = 0usize;

        for image in images {
            let tiles = self.tiles(image);
            counts.push(tiles.len());
            total += tiles.len();
            for tile in &tiles {
                for y in 0..edge {
                    for x in 0..edge {
                        let pixel = tile.get_pixel(x as u32, y as u32);
                        for channel in 0..3 {
                            let value = pixel[channel] as f32 / 255.0;
                            pixels.push((value - SIGLIP_MEAN) / SIGLIP_STD);
                        }
                    }
                }
            }
        }

        let tensor =
            mlxcel_core::from_slice_f32(&pixels, &[total as i32, edge as i32, edge as i32, 3]);
        (tensor, counts)
    }
}
