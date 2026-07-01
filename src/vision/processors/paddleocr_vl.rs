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

//! PaddleOCR-VL image processor (OCR-specific dynamic resolution).
//!
//! 1. `smart_resize`: round each side to `patch_size * merge_size` and clamp the
//!    pixel budget between `min_pixels` and `max_pixels` (reference formula).
//! 2. Convert to RGB, rescale to `[0, 1]`, normalize with SigLIP mean/std (0.5).
//! 3. Patchify to `[grid_t*grid_h*grid_w, C*patch*patch]` in `(C, h, w)` order
//!    (temporal_patch_size = 1, no frame duplication) for the Conv2d-as-Linear
//!    patch embedding.
//!
//! Used by: PaddleOCR-VL
//! Reference: mlx-vlm `paddleocr_vl/processing_paddleocr_vl.py`.

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct PaddleOcrVlProcessor {
    pub patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// Round half to even (matches Python's built-in `round`).
fn py_round(x: f64) -> f64 {
    let floor = x.floor();
    // `f64::round` rounds halves away from zero; detect the exact-half case
    // (without a float `==`) and pick the even neighbor to match the reference.
    if (x - floor - 0.5).abs() < 1e-9 {
        let is_even = (floor as i64) % 2 == 0;
        if is_even { floor } else { floor + 1.0 }
    } else {
        x.round()
    }
}

impl PaddleOcrVlProcessor {
    pub fn new(patch_size: usize, merge_size: usize) -> Self {
        Self {
            patch_size,
            merge_size,
            min_pixels: 147_384,
            max_pixels: 2_822_400,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    pub fn with_bounds(mut self, min_pixels: usize, max_pixels: usize) -> Self {
        self.min_pixels = min_pixels;
        self.max_pixels = max_pixels;
        self
    }

    /// Reference `smart_resize`: returns `(resized_height, resized_width)`.
    pub fn smart_resize(&self, height: u32, width: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.merge_size) as f64;
        let mut h = height as f64;
        let mut w = width as f64;

        if h < factor {
            w = py_round((w * factor) / h);
            h = factor;
        }
        if w < factor {
            h = py_round((h * factor) / w);
            w = factor;
        }

        let mut h_bar = py_round(h / factor) * factor;
        let mut w_bar = py_round(w / factor) * factor;

        let max_pixels = self.max_pixels as f64;
        let min_pixels = self.min_pixels as f64;
        if h_bar * w_bar > max_pixels {
            let beta = ((h * w) / max_pixels).sqrt();
            h_bar = (h / beta / factor).floor() * factor;
            w_bar = (w / beta / factor).floor() * factor;
        } else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels / (h * w)).sqrt();
            h_bar = (h * beta / factor).ceil() * factor;
            w_bar = (w * beta / factor).ceil() * factor;
        }

        (h_bar.max(factor) as u32, w_bar.max(factor) as u32)
    }

    /// Compute `(temporal=1, grid_h, grid_w)` per image.
    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                let gh = h as i32 / self.patch_size as i32;
                let gw = w as i32 / self.patch_size as i32;
                (1i32, gh, gw)
            })
            .collect()
    }

    /// Preprocess images. Returns `(pixel_values [total, C*patch*patch], grid_thw)`.
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let grid_thw = self.compute_grid_thw(images);
        let in_channels = 3usize;
        let patch = self.patch_size;
        let features_per_patch = in_channels * patch * patch;
        let mut all: Vec<f32> = Vec::new();

        for (idx, img) in images.iter().enumerate() {
            let (_t, gh, gw) = grid_thw[idx];
            let target_h = gh as u32 * patch as u32;
            let target_w = gw as u32 * patch as u32;

            let resized = img.resize_exact(target_w, target_h, FilterType::CatmullRom);
            let rgb = resized.to_rgb8();
            let h = target_h as usize;
            let w = target_w as usize;

            let mut normalized = vec![0f32; in_channels * h * w];
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..in_channels {
                        let val = pixel[c] as f32 / 255.0;
                        normalized[c * h * w + y * w + x] = (val - self.mean[c]) / self.std[c];
                    }
                }
            }

            let total_patches = (gh * gw) as usize;
            for patch_idx in 0..total_patches {
                let py = patch_idx / gw as usize;
                let px = patch_idx % gw as usize;
                let y0 = py * patch;
                let x0 = px * patch;
                for c in 0..in_channels {
                    for dy in 0..patch {
                        for dx in 0..patch {
                            all.push(normalized[c * h * w + (y0 + dy) * w + (x0 + dx)]);
                        }
                    }
                }
            }
        }

        let total_rows: usize = grid_thw.iter().map(|&(t, h, w)| (t * h * w) as usize).sum();
        let pixel_values =
            mlxcel_core::from_slice_f32(&all, &[total_rows as i32, features_per_patch as i32]);
        (pixel_values, grid_thw)
    }
}
