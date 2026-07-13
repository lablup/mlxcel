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

//! Pixtral / Mistral3 dynamic aspect-ratio image preprocessor.
//!
//! Port of the resize and normalization rules from
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/pixtral/image_processing_pixtral.py
//! (`get_resize_output_image_size`, `PixtralImageProcessor`) and the row-token
//! layout in
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/pixtral/processing_pixtral.py
//! (`PixtralProcessor.__call__`).
//!
//! Unlike the fixed-square SigLIP path these models were trained on, Pixtral and
//! Mistral3 preserve aspect ratio: each image is downscaled so its longest side
//! fits `longest_edge`, then rounded up to a whole number of merged patches. The
//! per-image patch grid drives the encoder's 2D-RoPE position grid and the
//! row-structured `[IMG] / [IMG_BREAK] / [IMG_END]` token expansion, so all three
//! must agree on the same `(rows, cols)`.
//!
//! Round-up factor decision: the upstream resize helper rounds to `patch_size`,
//! but `PixtralProcessor.__call__` calls the image processor with
//! `patch_size = patch_size * spatial_merge_size`, and the token layout divides
//! by that same product (`num_h = h // (patch_size * spatial_merge_size)`). For
//! Pixtral (`spatial_merge_size = 1`) the product equals `patch_size`; for
//! Mistral3 (`spatial_merge_size = 2`) it is `patch_size * 2`. Rounding to
//! `patch_size` alone would let a partial merged row survive the floor division
//! and desync the token count from the emitted features, so this processor
//! rounds each side up to `patch_size * spatial_merge_size` for both families.
//!
//! Used by: Pixtral VLM, Mistral3 VLM

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

// The reference `image_mean` / `image_std` carry more decimal digits than f32
// represents; keep them verbatim from the checkpoint's preprocessor_config so
// the values are diffable against the upstream processor.
/// CLIP normalization mean, shared by Pixtral and Mistral3.
#[allow(clippy::excessive_precision)]
pub const PIXTRAL_IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// CLIP normalization std, shared by Pixtral and Mistral3.
#[allow(clippy::excessive_precision)]
pub const PIXTRAL_IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// One preprocessed image plus the patch/token geometry the encoder, connector
/// and token expansion all key off.
pub struct PixtralPreprocessed {
    /// Pixel tensor in channels-first `[1, C, H, W]` layout (f32).
    pub pixel_values: UniquePtr<MlxArray>,
    /// Pre-merge patch rows (`target_h / patch_size`). Drives the encoder grid.
    pub patches_h: usize,
    /// Pre-merge patch cols (`target_w / patch_size`).
    pub patches_w: usize,
    /// Post-merge token rows (`target_h / (patch_size * spatial_merge_size)`).
    pub tokens_h: usize,
    /// Post-merge token cols (`target_w / (patch_size * spatial_merge_size)`).
    pub tokens_w: usize,
}

/// Pixtral / Mistral3 dynamic aspect-ratio processor.
#[derive(Debug, Clone)]
pub struct PixtralProcessor {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub longest_edge: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl PixtralProcessor {
    pub fn new(patch_size: usize, spatial_merge_size: usize, longest_edge: usize) -> Self {
        Self {
            patch_size,
            spatial_merge_size: spatial_merge_size.max(1),
            longest_edge,
            mean: PIXTRAL_IMAGE_MEAN,
            std: PIXTRAL_IMAGE_STD,
        }
    }

    /// The whole-patch round-up factor (`patch_size * spatial_merge_size`).
    #[inline]
    pub fn factor(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    /// Resized `(target_h, target_w)` in pixels for an original `(h, w)`.
    ///
    /// Downscale-only: the longest side is scaled to fit `longest_edge`, never
    /// upscaled, then each side is rounded up to a whole multiple of
    /// [`Self::factor`]. Mirrors `get_resize_output_image_size` with
    /// `patch_size = patch_size * spatial_merge_size`.
    pub fn target_size(&self, h: usize, w: usize) -> (usize, usize) {
        let factor = self.factor();
        let max_edge = self.longest_edge as f64;
        let ratio = (h as f64 / max_edge).max(w as f64 / max_edge);

        let (mut hh, mut ww) = (h, w);
        if ratio > 1.0 {
            hh = (h as f64 / ratio).floor() as usize;
            ww = (w as f64 / ratio).floor() as usize;
        }

        // (dim - 1) // factor + 1 rounds up to whole merged patches; guard the
        // 1px case so the subtraction cannot underflow.
        let num_h = hh.saturating_sub(1) / factor + 1;
        let num_w = ww.saturating_sub(1) / factor + 1;
        (num_h * factor, num_w * factor)
    }

    /// Pre-merge patch grid `(patches_h, patches_w)` for a resized image. This
    /// is what the encoder derives from the pixel shape and what the Mistral3
    /// PatchMerger unfolds.
    pub fn patch_grid(&self, target_h: usize, target_w: usize) -> (usize, usize) {
        (target_h / self.patch_size, target_w / self.patch_size)
    }

    /// Post-merge token grid `(tokens_h, tokens_w)`; one `[IMG]` token per cell.
    pub fn token_grid(&self, target_h: usize, target_w: usize) -> (usize, usize) {
        let factor = self.factor();
        (target_h / factor, target_w / factor)
    }

    /// Resize (bicubic), rescale by 1/255 and CLIP-normalize a single image,
    /// returning the pixel tensor and its patch/token geometry.
    pub fn preprocess_one(&self, img: &image::DynamicImage) -> PixtralPreprocessed {
        let (orig_h, orig_w) = (img.height() as usize, img.width() as usize);
        let (target_h, target_w) = self.target_size(orig_h, orig_w);
        let (patches_h, patches_w) = self.patch_grid(target_h, target_w);
        let (tokens_h, tokens_w) = self.token_grid(target_h, target_w);

        let pixel_values = self.pixels_for_size(img, target_h, target_w);

        PixtralPreprocessed {
            pixel_values,
            patches_h,
            patches_w,
            tokens_h,
            tokens_w,
        }
    }

    /// Resize/normalize `img` to an exact `(target_h, target_w)` and emit a
    /// `[1, C, H, W]` f32 tensor. The reference resamples with PIL BICUBIC
    /// (`resample = 3`); `CatmullRom` is the `image` crate's cubic filter.
    fn pixels_for_size(
        &self,
        img: &image::DynamicImage,
        target_h: usize,
        target_w: usize,
    ) -> UniquePtr<MlxArray> {
        let channels = 3usize;
        let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        let mut data = vec![0.0f32; channels * target_h * target_w];
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..channels {
                    let val = pixel[c] as f32 / 255.0;
                    let normalized = (val - self.mean[c]) / self.std[c];
                    // [C, H, W] with a leading batch of 1.
                    let idx = c * target_h * target_w + y * target_w + x;
                    data[idx] = normalized;
                }
            }
        }

        mlxcel_core::from_slice_f32(
            &data,
            &[1, channels as i32, target_h as i32, target_w as i32],
        )
    }
}

impl ImageProcessor for PixtralProcessor {
    /// Trait entry point. Dynamic aspect-ratio inference runs through the
    /// dedicated Pixtral runtime path ([`crate::vlm_runtime`]) which calls
    /// [`Self::preprocess_one`] per image and packs features individually, so
    /// this batched stack is only exercised for the common single-image case
    /// (and same-size batches). Each image is still resized to its own
    /// aspect-preserving size; a mixed-size batch cannot be stacked into one
    /// dense tensor and must use the per-image runtime path.
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let first = self.preprocess_one(&images[0]);
        if images.len() == 1 {
            return first.pixel_values;
        }

        let mut stacked = mlxcel_core::copy(&first.pixel_values);
        for img in &images[1..] {
            let next = self.preprocess_one(img);
            stacked = mlxcel_core::concatenate(&stacked, &next.pixel_values, 0);
        }
        stacked
    }
}

/// Row-structured layout descriptor carried on a Pixtral/Mistral3
/// [`crate::vision::VisionModule`]. Its presence routes the model to the
/// dynamic aspect-ratio runtime path instead of the fixed-square LLaVA merge.
#[derive(Debug, Clone)]
pub struct PixtralLayout {
    /// Processor holding the resize/normalize policy and patch geometry.
    pub processor: PixtralProcessor,
    /// `[IMG]` placeholder id (replaced by vision features on merge).
    pub image_token_id: i32,
    /// `[IMG_BREAK]` id emitted between patch rows (kept as a text embedding).
    pub image_break_token_id: i32,
    /// `[IMG_END]` id emitted after the last patch row.
    pub image_end_token_id: i32,
}

#[cfg(test)]
#[path = "pixtral_tests.rs"]
mod tests;
