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

//! ERNIE-4.5-VL image processor.
//!
//! Qwen2-VL-family smart resize (factor `patch_size * spatial_merge_size` = 28,
//! CLIP normalization, bicubic), but the patch rows differ in two ways from
//! [`super::qwen2_vl::Qwen2VLProcessor`]: rows are 588 wide
//! (`3 * 14 * 14`, no temporal frame duplication; `grid_thw = (1, gh, gw)`
//! per image) and they are emitted in **merge-window order**, row-major over
//! 2x2 windows and then the intra-window patches row-major. The resampler's
//! spatial fold concatenates 4 consecutive rows as one merge window, so this
//! order is load-bearing.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/processing_ernie4_5_moe_vl.py>.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Ernie45VlProcessor {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for Ernie45VlProcessor {
    fn default() -> Self {
        Self {
            patch_size: 14,
            spatial_merge_size: 2,
            min_pixels: 56 * 56,        // 3136
            max_pixels: 28 * 28 * 1280, // 1003520
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
        }
    }
}

impl Ernie45VlProcessor {
    /// Target size rounded to multiples of `patch_size * spatial_merge_size`,
    /// clamped into `[min_pixels, max_pixels]`. Returns `(height, width)`.
    fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u32;

        let mut h = ((orig_h as f64 / factor as f64).round() as u32).max(1) * factor;
        let mut w = ((orig_w as f64 / factor as f64).round() as u32).max(1) * factor;

        if (h * w) as usize > self.max_pixels {
            let scale = (self.max_pixels as f64 / (h * w) as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
        }
        if ((h * w) as usize) < self.min_pixels {
            let scale = (self.min_pixels as f64 / (h * w) as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
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

    /// Preprocess to `(total_patches, 3 * p * p)` rows in merge-window order
    /// plus `grid_thw`.
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let grid_thw = self.compute_grid_thw(images);
        let p = self.patch_size;
        let merge = self.spatial_merge_size;
        let row_width = 3 * p * p;
        let mut all_rows: Vec<f32> = Vec::new();

        for (img_idx, img) in images.iter().enumerate() {
            let (_t, gh, gw) = grid_thw[img_idx];
            let (gh, gw) = (gh as usize, gw as usize);
            let (target_h, target_w) = (gh * p, gw * p);

            let resized =
                img.resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom);
            let rgb = resized.to_rgb8();

            // Normalize once into channel-major planes.
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

            // Merge-window row order: (window_row, window_col, intra_row,
            // intra_col); each row flattens (channel, pixel_row, pixel_col).
            for wr in 0..gh / merge {
                for wc in 0..gw / merge {
                    for ir in 0..merge {
                        for ic in 0..merge {
                            let py = wr * merge + ir;
                            let px_col = wc * merge + ic;
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
            }
        }

        let total_rows: usize = grid_thw.iter().map(|&(t, h, w)| (t * h * w) as usize).sum();
        let pixel_values =
            mlxcel_core::from_slice_f32(&all_rows, &[total_rows as i32, row_width as i32]);
        (pixel_values, grid_thw)
    }
}

impl ImageProcessor for Ernie45VlProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_multiple_of_merge_and_rows_are_588_wide() {
        let proc = Ernie45VlProcessor::default();
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            300,
            200,
            image::Rgb([100, 150, 200]),
        ));
        let (pixels, grid) = proc.preprocess_with_grid(std::slice::from_ref(&img));
        let (t, h, w) = grid[0];
        assert_eq!(t, 1);
        assert_eq!(h % 2, 0);
        assert_eq!(w % 2, 0);
        assert_eq!(
            mlxcel_core::array_shape(&pixels),
            vec![h * w, 588],
            "rows must be (patches, 3*14*14) with no temporal duplication"
        );
    }

    #[test]
    fn merge_window_row_order_is_load_bearing() {
        // A 56x56 image -> 4x4 patch grid -> 4 merge windows of 4 rows each.
        // Paint each 28x28 merge-window block a distinct value; then every 4
        // consecutive output rows must be constant within one window and the
        // window order must be row-major over windows.
        let mut rgb = image::RgbImage::new(56, 56);
        for y in 0..56 {
            for x in 0..56 {
                let wr = y / 28;
                let wc = x / 28;
                let v = (wr * 2 + wc) as u8 * 60 + 30;
                rgb.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let img = image::DynamicImage::ImageRgb8(rgb);
        // Uniform mean/std so a solid gray block yields constant rows (the
        // default per-channel CLIP constants would vary within a row).
        let proc = Ernie45VlProcessor {
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            ..Ernie45VlProcessor::default()
        };
        let (pixels, grid) = proc.preprocess_with_grid(std::slice::from_ref(&img));
        assert_eq!(grid[0], (1, 4, 4));

        mlxcel_core::eval(&pixels);
        // Row r belongs to window r / 4; all 4 rows of a window share the same
        // (normalized) constant value, and windows come in order 0, 1, 2, 3.
        let mut window_means = Vec::new();
        for window in 0..4 {
            let rows = mlxcel_core::slice(&pixels, &[window * 4, 0], &[window * 4 + 4, 588]);
            let mean = mlxcel_core::mean_axis(&rows, 1, false);
            let mean = mlxcel_core::mean_axis(&mean, 0, false);
            mlxcel_core::eval(&mean);
            window_means.push(mlxcel_core::item_f32(&mean));
            // Within-window variance must be ~0 (solid color block).
            let sq = mlxcel_core::multiply(&rows, &rows);
            let mean_sq = mlxcel_core::mean_axis(&sq, 1, false);
            let mean_sq = mlxcel_core::mean_axis(&mean_sq, 0, false);
            mlxcel_core::eval(&mean_sq);
            let var = mlxcel_core::item_f32(&mean_sq)
                - window_means[window as usize] * window_means[window as usize];
            assert!(
                var.abs() < 1e-3,
                "window {window} rows must be one solid block, var {var}"
            );
        }
        // Values 30, 90, 150, 210 normalized are strictly increasing.
        for i in 1..4 {
            assert!(
                window_means[i] > window_means[i - 1],
                "windows must be emitted row-major: {window_means:?}"
            );
        }
    }
}
