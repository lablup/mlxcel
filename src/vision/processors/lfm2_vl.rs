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

//! LFM2-VL image processor (smart resize + patch packing).
//!
//! Port of the LFM2-VL bounded single-resize path (`do_image_splitting` tiling
//! is out of scope). Each image is smart-resized so its patch count lands in
//! `[min_image_tokens, max_image_tokens]` after downsampling, bilinearly
//! resampled, rescaled to `[0, 1]`, SigLIP-normalized (mean = std = 0.5), then
//! packed into per-patch vectors `(row_in_patch, col_in_patch, channel)` with
//! channel varying fastest. Every image is processed at its own patch count; the
//! output concatenates all images' patches with a per-image `(h, w)` grid list
//! (the KimiVL native-resolution pattern).
//!
//! Used by: LFM2-VL (`lfm2_vl` / `lfm2-vl`) VLM.

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

const SIGLIP_MEAN: f32 = 0.5;
const SIGLIP_STD: f32 = 0.5;

pub struct Lfm2VlProcessor {
    pub patch_size: usize,        // encoder_patch_size (16)
    pub downsample_factor: usize, // f (2)
    pub min_image_tokens: usize,  // 64
    pub max_image_tokens: usize,  // 256
}

impl Lfm2VlProcessor {
    pub fn new(
        patch_size: usize,
        downsample_factor: usize,
        min_image_tokens: usize,
        max_image_tokens: usize,
    ) -> Self {
        Self {
            patch_size: patch_size.max(1),
            downsample_factor: downsample_factor.max(1),
            min_image_tokens: min_image_tokens.max(1),
            max_image_tokens: max_image_tokens.max(1),
        }
    }

    /// Smart resize: return `(h1, w1)` both divisible by `total = P * f`, with the
    /// downsampled token count clamped to `[min_image_tokens, max_image_tokens]`.
    fn smart_resize(&self, h: u32, w: u32) -> (u32, u32) {
        let total = (self.patch_size * self.downsample_factor) as f64; // 32
        let p2f2 =
            (self.patch_size * self.patch_size * self.downsample_factor * self.downsample_factor)
                as f64;
        let min_pixels = self.min_image_tokens as f64 * p2f2;
        let max_pixels = self.max_image_tokens as f64 * p2f2;
        let (hf, wf) = (h as f64, w as f64);

        let round_to = |v: f64| -> f64 { (v / total).round() * total };
        let floor_to = |v: f64| -> f64 { (v / total).floor() * total };
        let ceil_to = |v: f64| -> f64 { (v / total).ceil() * total };

        let mut h1 = round_to(hf).max(total);
        let mut w1 = round_to(wf).max(total);
        if h1 * w1 > max_pixels {
            let beta = (hf * wf / max_pixels).sqrt();
            h1 = floor_to(hf / beta).max(total);
            w1 = floor_to(wf / beta).max(total);
        } else if h1 * w1 < min_pixels {
            let beta = (min_pixels / (hf * wf)).sqrt();
            h1 = ceil_to(hf * beta);
            w1 = ceil_to(wf * beta);
        }
        (h1 as u32, w1 as u32)
    }

    /// Resize + normalize + pack one image into `h_i*w_i` patch vectors of length
    /// `P*P*3`, appended to `out`. Returns the patch grid `(h_i, w_i)`.
    fn pack_image(&self, image: &DynamicImage, out: &mut Vec<f32>) -> (i32, i32) {
        let (w, h) = (image.width(), image.height());
        let (h1, w1) = self.smart_resize(h, w);
        let img = image.resize_exact(w1, h1, FilterType::Triangle).to_rgb8();

        let p = self.patch_size as u32;
        let (grid_h, grid_w) = (h1 / p, w1 / p);
        for gr in 0..grid_h {
            for gc in 0..grid_w {
                for py in 0..p {
                    for px in 0..p {
                        let pixel = img.get_pixel(gc * p + px, gr * p + py);
                        for c in 0..3 {
                            let v = pixel[c] as f32 / 255.0;
                            out.push((v - SIGLIP_MEAN) / SIGLIP_STD);
                        }
                    }
                }
            }
        }
        (grid_h as i32, grid_w as i32)
    }

    /// Preprocess a batch of images. Returns the concatenated packed patches
    /// `[1, sum_i(h_i*w_i), P*P*3]` and the per-image `(h_i, w_i)` grids.
    pub fn preprocess_with_grid(
        &self,
        images: &[DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32)>) {
        let patch_dim = (self.patch_size * self.patch_size * 3) as i32;
        let mut data: Vec<f32> = Vec::new();
        let mut grids: Vec<(i32, i32)> = Vec::with_capacity(images.len());
        for image in images {
            grids.push(self.pack_image(image, &mut data));
        }
        let total_patches = grids.iter().map(|(h, w)| h * w).sum::<i32>();
        let pixel_values =
            mlxcel_core::from_slice_f32(&data, &[1, total_patches.max(0), patch_dim]);
        (pixel_values, grids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            w,
            h,
            image::Rgb([255, 255, 255]),
        ))
    }

    #[test]
    fn smart_resize_divisible_by_total_and_clamped() {
        let p = Lfm2VlProcessor::new(16, 2, 64, 256);
        // Small image scaled up so token count >= min (64).
        let (h1, w1) = p.smart_resize(20, 20);
        assert_eq!(h1 % 32, 0);
        assert_eq!(w1 % 32, 0);
        let tokens = (h1 / 32) * (w1 / 32);
        assert!(tokens >= 64, "tokens {tokens} below min");
        // Large image scaled down so token count <= max (256).
        let (h2, w2) = p.smart_resize(4000, 4000);
        assert_eq!(h2 % 32, 0);
        assert_eq!(w2 % 32, 0);
        assert!((h2 / 32) * (w2 / 32) <= 256, "over max tokens");
    }

    #[test]
    fn packs_patch_dim_and_grid() {
        let p = Lfm2VlProcessor::new(16, 2, 64, 256);
        let (pixels, grids) = p.preprocess_with_grid(std::slice::from_ref(&solid(64, 64)));
        assert_eq!(grids.len(), 1);
        let (gh, gw) = grids[0];
        let shape = mlxcel_core::array_shape(&pixels);
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], gh * gw);
        assert_eq!(shape[2], 16 * 16 * 3);
    }

    #[test]
    fn white_pixel_normalizes_to_one() {
        let p = Lfm2VlProcessor::new(16, 2, 64, 256);
        let (pixels, _) = p.preprocess_with_grid(std::slice::from_ref(&solid(64, 64)));
        let first = mlxcel_core::slice(&pixels, &[0, 0, 0], &[1, 1, 1]);
        mlxcel_core::eval(&first);
        assert!((mlxcel_core::item_f32(&first) - 1.0).abs() < 1e-6);
    }
}
