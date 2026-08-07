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

//! Falcon-OCR image processor.
//!
//! Falcon-OCR has no vision encoder, so "preprocessing" ends at a matrix of
//! flattened 16x16 RGB patches that the decoder's `img_projector` turns into
//! token embeddings directly.
//!
//! Two resize stages run back to back, mirroring the checkpoint's
//! `processing_falcon_ocr.py`:
//!
//! 1. `resize_image_if_necessary` clamps the image so both sides land inside
//!    `[min_dimension, max_dimension]`, preserving aspect ratio.
//! 2. `smart_resize` snaps each side to a multiple of the patch size and then
//!    rescales so the patch count stays within `[min_pixels, max_pixels]`.
//!
//! Both stages use bicubic resampling (PIL's `Image.resize` default and the
//! processor's explicit `resample=BICUBIC`), which maps to `CatmullRom` here.
//! A resize whose target equals the source is skipped, matching PIL's own
//! short-circuit, so an image that is already patch-aligned reaches the model
//! bit-identical to the reference.

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// Falcon-OCR normalizes with mean 0.5 / std 0.5, i.e. maps `[0, 255]` onto
/// `[-1, 1]`.
const IMAGE_MEAN: f32 = 0.5;
const IMAGE_STD: f32 = 0.5;

#[derive(Debug, Clone)]
pub struct FalconOcrProcessor {
    pub spatial_patch_size: u32,
    pub channel_size: usize,
    /// Shortest side accepted before stage 1 rescales.
    pub min_dimension: u32,
    /// Longest side accepted before stage 1 rescales.
    pub max_dimension: u32,
    /// Stage-2 lower bound in pixels (`56 * 56`).
    pub min_pixels: u32,
    /// Stage-2 upper bound in pixels (`28 * 28 * 1280`).
    pub max_pixels: u32,
}

impl Default for FalconOcrProcessor {
    fn default() -> Self {
        Self {
            spatial_patch_size: 16,
            channel_size: 3,
            min_dimension: 64,
            max_dimension: 1024,
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
        }
    }
}

impl FalconOcrProcessor {
    pub fn new(spatial_patch_size: u32, channel_size: usize) -> Self {
        Self {
            spatial_patch_size: spatial_patch_size.max(1),
            channel_size,
            ..Self::default()
        }
    }

    /// Stage 1: `resize_image_if_necessary`.
    ///
    /// Returns the `(width, height)` the image should be resized to, or `None`
    /// when both sides already sit inside the accepted band.
    pub fn clamp_dimensions(&self, width: u32, height: u32) -> Option<(u32, u32)> {
        let (lo, hi) = (self.min_dimension, self.max_dimension);
        if (lo..=hi).contains(&width) && (lo..=hi).contains(&height) {
            return None;
        }
        if width == 0 || height == 0 {
            return None;
        }
        let aspect = width as f64 / height as f64;
        let is_vertical = width < height;
        let target = if width < lo || height < lo { lo } else { hi };

        let (mut new_w, mut new_h) = if is_vertical {
            (target, (target as f64 / aspect) as u32)
        } else {
            ((target as f64 * aspect) as u32, target)
        };
        if new_w > hi {
            new_w = hi;
            new_h = (hi as f64 / aspect) as u32;
        }
        if new_h > hi {
            new_h = hi;
            new_w = (hi as f64 * aspect) as u32;
        }
        Some((new_w.max(1), new_h.max(1)))
    }

    /// Stage 2: `smart_resize`. Returns `(height, width)`.
    pub fn smart_resize(&self, height: u32, width: u32) -> (u32, u32) {
        let factor = self.spatial_patch_size;
        let mut h_bar = round_half_even(height as f64 / factor as f64) * factor;
        let mut w_bar = round_half_even(width as f64 / factor as f64) * factor;
        let pixels = height as f64 * width as f64;

        if (h_bar as u64) * (w_bar as u64) > self.max_pixels as u64 {
            let beta = (pixels / self.max_pixels as f64).sqrt();
            h_bar = (height as f64 / beta / factor as f64).floor() as u32 * factor;
            w_bar = (width as f64 / beta / factor as f64).floor() as u32 * factor;
        } else if (h_bar as u64) * (w_bar as u64) < self.min_pixels as u64 {
            let beta = (self.min_pixels as f64 / pixels).sqrt();
            h_bar = (height as f64 * beta / factor as f64).ceil() as u32 * factor;
            w_bar = (width as f64 * beta / factor as f64).ceil() as u32 * factor;
        }
        (h_bar.max(factor), w_bar.max(factor))
    }

    /// Patch grid `(rows, cols)` an image will occupy.
    pub fn grid_for(&self, image: &image::DynamicImage) -> (i32, i32) {
        let (h, w) = self.resolved_size(image);
        let p = self.spatial_patch_size;
        ((h / p) as i32, (w / p) as i32)
    }

    fn resolved_size(&self, image: &image::DynamicImage) -> (u32, u32) {
        let (w0, h0) = (image.width(), image.height());
        let (w1, h1) = self.clamp_dimensions(w0, h0).unwrap_or((w0, h0));
        self.smart_resize(h1, w1)
    }

    /// Preprocess into `[total_patches, patch_dim]` plus per-image
    /// `(rows, cols)` grids.
    ///
    /// The patch vector layout is `(patch_row, patch_col, channel)`, matching
    /// the reference einops pattern
    /// `"n (t pt) (h ph) (w pw) c -> n (t h w) (pt ph pw c)"` at
    /// `temporal_patch_size == 1`.
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32)>) {
        let (values, grids) = self.preprocess_values_with_grid(images);
        let patch_dim =
            (self.spatial_patch_size * self.spatial_patch_size) as i32 * self.channel_size as i32;
        let rows = if patch_dim > 0 {
            values.len() as i32 / patch_dim
        } else {
            0
        };
        (
            mlxcel_core::from_slice_f32(&values, &[rows, patch_dim]),
            grids,
        )
    }

    /// Host-side twin of [`Self::preprocess_with_grid`].
    pub fn preprocess_values_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (Vec<f32>, Vec<(i32, i32)>) {
        let p = self.spatial_patch_size as usize;
        let channels = self.channel_size.max(1);
        let mut out = Vec::new();
        let mut grids = Vec::with_capacity(images.len());

        for image in images {
            let (w0, h0) = (image.width(), image.height());
            // Stage 1 runs in the source color mode, exactly as the reference
            // resizes before `convert_to_rgb`.
            let clamped = match self.clamp_dimensions(w0, h0) {
                Some((w1, h1)) => resize_exact_if_needed(image, w1, h1),
                None => image.clone(),
            };
            let as_rgb = image::DynamicImage::ImageRgb8(clamped.to_rgb8());
            let (h_bar, w_bar) = self.smart_resize(as_rgb.height(), as_rgb.width());
            let resized = resize_exact_if_needed(&as_rgb, w_bar, h_bar);
            let rgb = resized.to_rgb8();

            let (rows, cols) = (
                (h_bar / self.spatial_patch_size) as usize,
                (w_bar / self.spatial_patch_size) as usize,
            );
            grids.push((rows as i32, cols as i32));
            out.reserve(rows * cols * p * p * channels);

            for pr in 0..rows {
                for pc in 0..cols {
                    for dy in 0..p {
                        for dx in 0..p {
                            let px = rgb.get_pixel((pc * p + dx) as u32, (pr * p + dy) as u32);
                            for c in 0..channels {
                                let raw = px.0.get(c).copied().unwrap_or(0) as f32 / 255.0;
                                out.push((raw - IMAGE_MEAN) / IMAGE_STD);
                            }
                        }
                    }
                }
            }
        }
        (out, grids)
    }
}

/// Bicubic resize that short-circuits when the target already matches, the way
/// PIL's `Image.resize` does. Skipping the identity pass keeps a patch-aligned
/// document byte-identical to the reference pipeline.
fn resize_exact_if_needed(
    image: &image::DynamicImage,
    width: u32,
    height: u32,
) -> image::DynamicImage {
    if image.width() == width && image.height() == height {
        return image.clone();
    }
    image.resize_exact(width, height, FilterType::CatmullRom)
}

/// Python's `round`: half-to-even, not half-away-from-zero.
///
/// It matters at every `height % 16 == 8`, where Rust's `f64::round` would pick
/// the other patch count and change the image token budget.
fn round_half_even(x: f64) -> u32 {
    let floor = x.floor();
    let diff = x - floor;
    // On an exact tie the even neighbour wins, which is `floor` when `floor` is
    // already even and `floor + 1` otherwise.
    let round_up = diff > 0.5 || (diff == 0.5 && (floor as i64) % 2 != 0);
    let n = if round_up { floor + 1.0 } else { floor };
    n.max(0.0) as u32
}

#[cfg(test)]
#[path = "falcon_ocr_tests.rs"]
mod tests;
