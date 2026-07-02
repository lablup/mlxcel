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

//! Idefics2 (`idefics2`) image processor.
//!
//! Port of the Idefics2 `do_image_splitting=False` image path (HuggingFace
//! `Idefics2ImageProcessor`): each image is resized aspect-preserving to fit
//! within `[shortest_edge, longest_edge]`, cropped to whole `patch_size` tiles,
//! rescaled to `[0, 1]`, then SigLIP-normalized (mean `0.5`, std `0.5` per
//! channel). The image is fed as a single tile whose `(grid_h, grid_w)` patch
//! grid maps into the vision tower's fixed position table via bucketized ids
//! (see `crate::vision::idefics2::bucketize_position_ids`).
//!
//! Output: a `[num_images, 3, H, W]` channels-first tensor (with `H`, `W` whole
//! multiples of `patch_size`) plus one tile per image. Idefics2's optional
//! `do_image_splitting` (4 crops plus a global tile) is a resolution refinement
//! on the same per-tile math and is a documented follow-up.
//!
//! Used by: Idefics2 (`idefics2`) VLM.

use super::ImageProcessor;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use mlxcel_core::{MlxArray, UniquePtr};

/// SigLIP normalization constants used by Idefics2 (rescale to `[-1, 1]`).
const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];
/// Idefics2 default `size.shortest_edge`.
const DEFAULT_SHORTEST_EDGE: usize = 378;

pub struct Idefics2Processor {
    /// `size.longest_edge` (= the vision tower `image_size`, 980).
    pub longest_edge: usize,
    /// `size.shortest_edge` (378).
    pub shortest_edge: usize,
    /// Vision patch size (14); tiles are cropped to whole patches.
    pub patch_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Idefics2Processor {
    pub fn new(image_size: usize, patch_size: usize) -> Self {
        Self {
            longest_edge: image_size.max(1),
            shortest_edge: DEFAULT_SHORTEST_EDGE,
            patch_size: patch_size.max(1),
            mean: SIGLIP_MEAN,
            std: SIGLIP_STD,
        }
    }

    /// Aspect-preserving target `(height, width)` for an image, cropped down to
    /// whole `patch_size` tiles. Scales the longest side down to `longest_edge`
    /// when oversized, then up so the shortest side reaches `shortest_edge`.
    fn target_size(&self, w: u32, h: u32) -> (u32, u32) {
        let (wf, hf) = (w.max(1) as f64, h.max(1) as f64);
        let longest = self.longest_edge as f64;
        let shortest = self.shortest_edge as f64;

        let scale = (longest / wf.max(hf)).min(1.0);
        let (mut w2, mut h2) = (wf * scale, hf * scale);
        if w2.min(h2) < shortest {
            let s2 = shortest / w2.min(h2);
            w2 *= s2;
            h2 *= s2;
        }
        let p = self.patch_size as u32;
        let crop = |v: f64| -> u32 { ((v.round() as u32) / p * p).max(p) };
        (crop(h2), crop(w2))
    }

    /// Resize one image to `(out_h, out_w)` and append its channels-first
    /// `[C, H, W]` normalized f32 values to `out`.
    fn append_normalized_chw(
        &self,
        image: &DynamicImage,
        out_h: u32,
        out_w: u32,
        out: &mut Vec<f32>,
    ) {
        let tile = image
            .resize_exact(out_w, out_h, FilterType::CatmullRom)
            .to_rgb8();
        for c in 0..3 {
            for y in 0..out_h {
                for x in 0..out_w {
                    let pixel = tile.get_pixel(x, y);
                    let val = pixel[c] as f32 / 255.0;
                    out.push((val - self.mean[c]) / self.std[c]);
                }
            }
        }
    }

    /// Preprocess a batch of images into `[num_images, 3, H, W]` plus the
    /// per-image tile counts (always `1` for the single-tile path). All images
    /// in a request share the first image's aspect-preserved target size so the
    /// batch tensor is rectangular and a single grid drives the vision tower;
    /// single-image requests (the common case) use their own target exactly.
    pub fn preprocess_with_tiles(
        &self,
        images: &[DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<usize>) {
        if images.is_empty() {
            return (mlxcel_core::from_slice_f32(&[], &[0, 3, 0, 0]), Vec::new());
        }

        let (first_w, first_h) = images[0].dimensions();
        let (out_h, out_w) = self.target_size(first_w, first_h);

        let mut all_pixels: Vec<f32> = Vec::new();
        let mut tiles_per_image: Vec<usize> = Vec::with_capacity(images.len());
        for image in images {
            self.append_normalized_chw(image, out_h, out_w, &mut all_pixels);
            tiles_per_image.push(1);
        }

        let pixel_values = mlxcel_core::from_slice_f32(
            &all_pixels,
            &[images.len() as i32, 3, out_h as i32, out_w as i32],
        );
        (pixel_values, tiles_per_image)
    }
}

impl ImageProcessor for Idefics2Processor {
    fn preprocess(&self, images: &[DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_tiles(images);
        pixel_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, value: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            width,
            height,
            image::Rgb([value, value, value]),
        ))
    }

    #[test]
    fn aspect_preserving_tile_cropped_to_whole_patches() {
        // 640x480 fits within [378, 980], so no scale; cropped to multiples of 14.
        let proc = Idefics2Processor::new(980, 14);
        let img = solid(640, 480, 255);
        let (pixels, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        assert_eq!(tiles, vec![1]);
        // 480 -> 476 (34*14), 640 -> 630 (45*14).
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 3, 476, 630]);
    }

    #[test]
    fn small_image_scaled_up_to_shortest_edge() {
        let proc = Idefics2Processor::new(980, 14);
        let img = solid(100, 100, 0);
        let (pixels, _) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        // shortest edge 378 -> 378 // 14 * 14 = 378 (27*14).
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 3, 378, 378]);
    }

    #[test]
    fn large_image_scaled_down_to_longest_edge() {
        let proc = Idefics2Processor::new(980, 14);
        let img = solid(2000, 1000, 128);
        let (pixels, _) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        // 2000 -> 980 (70*14), 1000 -> 490 (35*14).
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 3, 490, 980]);
    }

    #[test]
    fn siglip_normalization_maps_white_to_one() {
        let proc = Idefics2Processor::new(980, 14);
        let img = solid(28, 28, 255);
        let (pixels, _) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
        let first = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
        mlxcel_core::eval(&first);
        assert!((mlxcel_core::item_f32(&first) - 1.0).abs() < 1e-6);
    }
}
