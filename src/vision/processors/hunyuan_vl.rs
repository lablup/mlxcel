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

//! Hunyuan-VL image processor.
//!
//! Smart resize to multiples of `patch_size * merge_size` = 32 inside
//! `[min_pixels, max_pixels]`, bilinear resize, CLIP normalization, then plain
//! raster-ordered 768-wide patch rows (`3 * 16 * 16`, each flattening
//! `(channel, pixel_row, pixel_col)`); `grid_thw = (1, gh, gw)` per image. The
//! per-image placeholder count is `mh * (mw + 1) + 2` (`mh = gh / merge`),
//! covering the merger's newline column and its begin / end rows.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/hunyuan_vl/processing_hunyuan_vl.py>.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct HunyuanVlProcessor {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for HunyuanVlProcessor {
    fn default() -> Self {
        Self {
            patch_size: 16,
            spatial_merge_size: 2,
            min_pixels: 512 * 512,
            max_pixels: 2048 * 2048,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
        }
    }
}

impl HunyuanVlProcessor {
    fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u32;
        let mut h = ((orig_h as f64 / factor as f64).round() as u32).max(1) * factor;
        let mut w = ((orig_w as f64 / factor as f64).round() as u32).max(1) * factor;

        if (h * w) as usize > self.max_pixels {
            let beta = ((orig_h as f64 * orig_w as f64) / self.max_pixels as f64).sqrt();
            h = ((orig_h as f64 / beta / factor as f64).floor() as u32).max(1) * factor;
            w = ((orig_w as f64 / beta / factor as f64).floor() as u32).max(1) * factor;
        } else if ((h * w) as usize) < self.min_pixels {
            let beta = (self.min_pixels as f64 / (orig_h as f64 * orig_w as f64)).sqrt();
            h = ((orig_h as f64 * beta / factor as f64).ceil() as u32).max(1) * factor;
            w = ((orig_w as f64 * beta / factor as f64).ceil() as u32).max(1) * factor;
        }
        (h, w)
    }

    /// Per image `(t = 1, h_patches, w_patches)`.
    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                (
                    1i32,
                    (h as usize / self.patch_size) as i32,
                    (w as usize / self.patch_size) as i32,
                )
            })
            .collect()
    }

    /// Prompt placeholder count for a grid: `mh * (mw + 1) + 2`.
    pub fn placeholder_count(&self, gh: i32, gw: i32) -> i32 {
        let m = self.spatial_merge_size as i32;
        let (mh, mw) = (gh / m, gw / m);
        mh * (mw + 1) + 2
    }

    /// Preprocess to `(total_patches, 3 * p * p)` raster-ordered rows plus grids.
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let grid_thw = self.compute_grid_thw(images);
        let p = self.patch_size;
        let row_width = 3 * p * p;
        let mut all_rows: Vec<f32> = Vec::new();

        for (img_idx, img) in images.iter().enumerate() {
            let (_t, gh, gw) = grid_thw[img_idx];
            let (gh, gw) = (gh as usize, gw as usize);
            let (target_h, target_w) = (gh * p, gw * p);

            let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::Triangle);
            let rgb = resized.to_rgb8();

            let mut planes = vec![0f32; 3 * target_h * target_w];
            for y in 0..target_h {
                for x in 0..target_w {
                    let px = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let v = px[c] as f32 / 255.0;
                        planes[c * target_h * target_w + y * target_w + x] =
                            (v - self.mean[c]) / self.std[c];
                    }
                }
            }

            // Raster patch order; each row flattens (channel, py, px).
            for py in 0..gh {
                for px_col in 0..gw {
                    let y0 = py * p;
                    let x0 = px_col * p;
                    for c in 0..3 {
                        let plane = &planes[c * target_h * target_w..];
                        for dy in 0..p {
                            let base = (y0 + dy) * target_w + x0;
                            all_rows.extend_from_slice(&plane[base..base + p]);
                        }
                    }
                }
            }
        }

        let total_rows: usize = grid_thw.iter().map(|&(t, h, w)| (t * h * w) as usize).sum();
        let pixel_values =
            mlxcel_core::from_slice_f32(&all_rows, &[total_rows as i32, row_width as i32]);
        (pixel_values, grid_thw)
    }
}

impl ImageProcessor for HunyuanVlProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_and_rows_and_count() {
        let proc = HunyuanVlProcessor::default();
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            640,
            480,
            image::Rgb([50, 100, 150]),
        ));
        let (pixels, grid) = proc.preprocess_with_grid(std::slice::from_ref(&img));
        let (t, gh, gw) = grid[0];
        assert_eq!(t, 1);
        assert_eq!(gh % 2, 0);
        assert_eq!(gw % 2, 0);
        // Below min_pixels (512^2), the image scales UP.
        assert!((gh * 16) * (gw * 16) >= (512 * 512));
        assert_eq!(mlxcel_core::array_shape(&pixels), vec![gh * gw, 768]);
        assert_eq!(
            proc.placeholder_count(gh, gw),
            (gh / 2) * (gw / 2 + 1) + 2,
            "count must include the newline column and begin/end rows"
        );
    }
}
