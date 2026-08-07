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

//! LocateAnything (MoonViT) native-resolution image processor.
//!
//! Faithful port of `LocateAnythingImageProcessor` from upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/locateanything/image_processing_locateanything.py.
//!
//! The geometry deliberately differs from the Kimi-VL MoonViT processor
//! ([`super::kimi_vl::KimiVLProcessor`]), so the two are not shared:
//!
//! - Kimi-VL centre-**crops** each side down to a multiple of `merge * patch`.
//! - LocateAnything **resizes up** to the next multiple
//!   (`ceil(side / (merge * patch)) * merge * patch`), so no pixels are
//!   discarded and a small image is enlarged rather than rejected.
//! - Normalization is `mean = std = 0.5` (a plain `[-1, 1]` rescale) rather
//!   than the CLIP/OpenAI statistics.
//!
//! Both share the downscale-only-when-over-budget step and the
//! `[num_patches, C, p, p]` patch layout that MoonViT consumes.
//!
//! Output: the per-image patches concatenated into `[total_patches, C, p, p]`
//! plus the `(grid_h, grid_w)` patch grid of each image. The runtime uses the
//! grid to size each `<IMG_CONTEXT>` run (`grid_h * grid_w / (merge_h*merge_w)`)
//! and MoonViT uses it for the 2D rope, the interpolated position embedding,
//! and the block-diagonal cross-image attention.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// Upstream `LOCATEANYTHING_IMAGE_MEAN` / `LOCATEANYTHING_IMAGE_STD`.
const LOCATEANYTHING_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const LOCATEANYTHING_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Upstream refuses a grid whose side reaches 512 patches ("Exceed pos emb").
/// We clamp instead of failing, because the CLI/server image path has no
/// user-visible way to renegotiate the resolution mid-request.
const MAX_GRID_SIDE: usize = 511;

/// LocateAnything native-resolution image processor.
#[derive(Debug, Clone)]
pub struct LocateAnythingProcessor {
    pub patch_size: usize,
    /// `[merge_h, merge_w]` from `vision_config.merge_kernel_size`.
    pub merge_kernel_size: [usize; 2],
    /// Maximum pre-merge patch count per image (`in_token_limit`, 25600
    /// upstream).
    pub in_token_limit: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl LocateAnythingProcessor {
    pub fn new(patch_size: usize, merge_kernel_size: [usize; 2], in_token_limit: usize) -> Self {
        Self {
            patch_size: patch_size.max(1),
            merge_kernel_size: [merge_kernel_size[0].max(1), merge_kernel_size[1].max(1)],
            in_token_limit: in_token_limit.max(1),
            mean: LOCATEANYTHING_MEAN,
            std: LOCATEANYTHING_STD,
        }
    }

    /// Upstream defaults: patch 14, 2x2 merge, 25600-patch budget.
    pub fn default_config() -> Self {
        Self::new(14, [2, 2], 25_600)
    }

    /// Resize one image to a MoonViT-valid size: downscale first when the patch
    /// count exceeds `in_token_limit`, then grow each side up to the next
    /// multiple of `merge * patch`.
    fn rescale(&self, image: &image::DynamicImage) -> image::RgbImage {
        let p = self.patch_size;
        let rgb = image.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let mut cur = image::DynamicImage::ImageRgb8(rgb);

        // Step 1: aspect-preserving downscale, only when over the budget.
        let (mut cw, mut ch) = (w, h);
        let patches = (w / p) * (h / p);
        if patches > self.in_token_limit {
            let scale = (self.in_token_limit as f64 / patches as f64).sqrt();
            cw = ((w as f64 * scale) as usize).max(1);
            ch = ((h as f64 * scale) as usize).max(1);
        }

        // Step 2: round each side up to a multiple of merge * patch.
        let pad_w = self.merge_kernel_size[1] * p;
        let pad_h = self.merge_kernel_size[0] * p;
        let mut target_w = cw.div_ceil(pad_w).max(1) * pad_w;
        let mut target_h = ch.div_ceil(pad_h).max(1) * pad_h;

        // Step 3: keep the patch grid inside the rotary/position-embedding
        // envelope. Upstream raises here; clamping keeps the request alive and
        // only engages for extreme aspect ratios that the token budget alone
        // cannot bound.
        let max_w = MAX_GRID_SIDE / self.merge_kernel_size[1] * pad_w;
        let max_h = MAX_GRID_SIDE / self.merge_kernel_size[0] * pad_h;
        target_w = target_w.min(max_w);
        target_h = target_h.min(max_h);

        if (target_w, target_h) != (cur.width() as usize, cur.height() as usize) {
            cur = cur.resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom);
        }
        cur.to_rgb8()
    }

    /// Normalize + patchify one resized image into `[num_patches, C, p, p]`
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

    /// Number of merged tokens one `(grid_h, grid_w)` patch grid contributes.
    ///
    /// Delegates to the shared prompt-side helper so the processor and the
    /// `<IMG_CONTEXT>` run builder can never disagree on the count.
    #[inline]
    pub fn merged_token_count(&self, grid: (i32, i32)) -> usize {
        crate::multimodal::locateanything_prompt::merged_token_count(grid, self.merge_kernel_size)
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

impl ImageProcessor for LocateAnythingProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}

#[cfg(test)]
#[path = "locateanything_tests.rs"]
mod tests;
