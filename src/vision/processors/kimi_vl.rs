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

//! Kimi-VL / Kimi-VL 2.5 (MoonViT) image processor.
//!
//! Faithful port of the image path of `KimiVLImageProcessor` from upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/processing_kimi_vl.py.
//!
//! MoonViT is native-resolution: an image is
//! 1. downscaled (bicubic) only when its patch count `(w/p)*(h/p)` exceeds
//!    `in_token_limit`, preserving aspect ratio;
//! 2. centre-cropped so both sides are multiples of `merge * patch` (so the
//!    patch grid divides evenly by the spatial merge);
//! 3. rescaled to `[0, 1]`, CLIP-normalized, and patchified into
//!    `[num_patches, C, p, p]` (channels-first per patch).
//!
//! Output: the per-image patches concatenated into
//! `[total_patches, C, p, p]` plus the `(grid_h, grid_w)` patch grid of each
//! image (used by the runtime to expand `<|media_pad|>` placeholders by
//! `grid_h * grid_w / (merge*merge)` and by MoonViT to build its 2D rope /
//! block-diagonal attention).
//!
//! Scope: image path only. The Kimi-VL 2.5 video patch-embedding path is a
//! separate follow-up.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// CLIP / OpenAI dataset normalization constants (from the Python processor).
const OPENAI_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const OPENAI_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// MoonViT native-resolution image processor.
pub struct KimiVLProcessor {
    pub patch_size: usize,
    pub merge_kernel_size: [usize; 2],
    pub in_token_limit: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl KimiVLProcessor {
    pub fn new(patch_size: usize, merge_kernel_size: [usize; 2], in_token_limit: usize) -> Self {
        Self {
            patch_size,
            merge_kernel_size,
            in_token_limit,
            mean: OPENAI_MEAN,
            std: OPENAI_STD,
        }
    }

    /// Upstream defaults: patch 14, 2x2 merge, 4096-token budget.
    pub fn default_config() -> Self {
        Self::new(14, [2, 2], 4096)
    }

    pub fn with_norm(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = mean;
        self.std = std;
        self
    }

    /// Rescale + centre-crop one image to a MoonViT-valid size (both sides a
    /// multiple of `merge * patch`). Returns the cropped RGB image.
    fn rescale(&self, image: &image::DynamicImage) -> image::RgbImage {
        let rgb = image.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let p = self.patch_size;

        // Downscale only if the patch count exceeds the budget (aspect-ratio
        // preserving).
        let mut cur = image::DynamicImage::ImageRgb8(rgb);
        let (mut cw, mut ch) = (w, h);
        let patches = (w / p) * (h / p);
        if patches > self.in_token_limit {
            let scale = (self.in_token_limit as f64 / patches as f64).sqrt();
            cw = ((w as f64 * scale) as usize).max(1);
            ch = ((h as f64 * scale) as usize).max(1);
            cur = cur.resize_exact(cw as u32, ch as u32, FilterType::CatmullRom);
        }

        // Crop both sides down to a multiple of merge*patch (centre crop). Keep
        // at least one merged block so degenerate tiny inputs stay valid.
        let crop_w = self.merge_kernel_size[1] * p;
        let crop_h = self.merge_kernel_size[0] * p;
        let cw = cw.max(crop_w);
        let ch = ch.max(crop_h);
        if cw != cur.width() as usize || ch != cur.height() as usize {
            cur = cur.resize_exact(cw as u32, ch as u32, FilterType::CatmullRom);
        }
        let new_w = cw - cw % crop_w;
        let new_h = ch - ch % crop_h;
        let left = ((cw - new_w) / 2) as u32;
        let top = ((ch - new_h) / 2) as u32;

        let rgb = cur.to_rgb8();
        let cropped = image::imageops::crop_imm(&rgb, left, top, new_w as u32, new_h as u32);
        cropped.to_image()
    }

    /// Normalize + patchify one cropped image into `[num_patches, C, p, p]`
    /// (channels-first per patch, patches in row-major grid order). Appends the
    /// f32 patch values to `out` and returns the `(grid_h, grid_w)` grid.
    fn patchify(&self, tile: &image::RgbImage, out: &mut Vec<f32>) -> (i32, i32) {
        let p = self.patch_size;
        let (w, h) = (tile.width() as usize, tile.height() as usize);
        let gh = h / p;
        let gw = w / p;

        let norm = |c: usize, y: usize, x: usize| -> f32 {
            let px = tile.get_pixel(x as u32, y as u32);
            (px[c] as f32 / 255.0 - self.mean[c]) / self.std[c]
        };

        // Patch order: row-major over (gh, gw); within a patch: [C, p, p].
        for row in 0..gh {
            for col in 0..gw {
                for c in 0..3 {
                    for py in 0..p {
                        for px in 0..p {
                            out.push(norm(c, row * p + py, col * p + px));
                        }
                    }
                }
            }
        }
        (gh as i32, gw as i32)
    }

    /// Preprocess a batch of images. Returns the flattened patch tensor
    /// `[total_patches, C, p, p]` and the per-image `(grid_h, grid_w)` grids.
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32)>) {
        let p = self.patch_size as i32;
        let mut all: Vec<f32> = Vec::new();
        let mut grids: Vec<(i32, i32)> = Vec::with_capacity(images.len());
        let mut total = 0i32;

        for image in images {
            let tile = self.rescale(image);
            let (gh, gw) = self.patchify(&tile, &mut all);
            grids.push((gh, gw));
            total += gh * gw;
        }

        let pixel_values = mlxcel_core::from_slice_f32(&all, &[total, 3, p, p]);
        (pixel_values, grids)
    }
}

impl ImageProcessor for KimiVLProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            w,
            h,
            image::Rgb([128, 128, 128]),
        ))
    }

    #[test]
    fn crops_to_merge_patch_multiple_and_grids() {
        // patch 14, merge 2 -> crop unit 28. A 60x60 image crops to 56x56 ->
        // grid (4, 4) = 16 patches.
        let proc = KimiVLProcessor::new(14, [2, 2], 4096);
        let (pixels, grids) = proc.preprocess_with_grid(&[gray(60, 60)]);
        assert_eq!(grids, vec![(4, 4)]);
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![16, 3, 14, 14]);
    }

    #[test]
    fn downscales_when_over_token_limit() {
        // A large image with an artificially small token budget must shrink so
        // (gw*gh) stays at/under the limit's neighbourhood.
        let proc = KimiVLProcessor::new(14, [2, 2], 16);
        let (_pixels, grids) = proc.preprocess_with_grid(&[gray(560, 560)]);
        let (gh, gw) = grids[0];
        assert!(
            (gh * gw) as usize <= 4 * 16,
            "patch count {} should be reduced toward the 16-token budget",
            gh * gw
        );
    }

    #[test]
    fn clip_normalization_is_applied() {
        // A constant gray (128/255) image normalizes to (128/255 - mean)/std.
        let proc = KimiVLProcessor::new(14, [2, 2], 4096);
        let (pixels, _) = proc.preprocess_with_grid(&[gray(28, 28)]);
        mlxcel_core::eval(&pixels);
        // First value is channel 0 of the first patch.
        let first = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
        let expected = (128.0f32 / 255.0 - OPENAI_MEAN[0]) / OPENAI_STD[0];
        assert!(
            (mlxcel_core::item_f32(&first) - expected).abs() < 1e-4,
            "CLIP normalization mismatch"
        );
    }
}
