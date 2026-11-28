// Copyright 2025-2026 Lablup Inc.
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
    /// Port of transformers' `hunyuan_vl` `smart_resize`. Returns
    /// `(height, width)`, both multiples of `patch_size * spatial_merge_size`.
    fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u64;
        let (min_pixels, max_pixels) = (self.min_pixels as u64, self.max_pixels as u64);
        let area = |h: u64, w: u64| h.saturating_mul(w);
        // Python's `round` in the reference rounds half to even, and the
        // reference puts no floor on it: a thin edge rounding to 0 drops the
        // area under `min_pixels`, which then rescales the original size.
        let round_to =
            |edge: u32| (f64::from(edge) / factor as f64).round_ties_even() as u64 * factor;
        let mut h = round_to(orig_h);
        let mut w = round_to(orig_w);

        // Out of bounds the reference scales the original size. A zero edge
        // (the reference raises on it) is rescaled as one pixel so `beta`
        // stays finite; u64 saturating arithmetic keeps a huge edge from
        // wrapping.
        let (orig_h, orig_w) = (orig_h.max(1), orig_w.max(1));
        let orig_area = area(u64::from(orig_h), u64::from(orig_w)) as f64;
        let (orig_h, orig_w) = (f64::from(orig_h), f64::from(orig_w));
        let factor_f = factor as f64;
        let snap = |steps: f64| (steps as u64).max(1).saturating_mul(factor);
        let pixels = area(h, w);
        if pixels > max_pixels {
            let beta = (orig_area / max_pixels as f64).sqrt();
            h = snap((orig_h / beta / factor_f).floor());
            w = snap((orig_w / beta / factor_f).floor());
        } else if pixels < min_pixels {
            let beta = (min_pixels as f64 / orig_area).sqrt();
            h = snap((orig_h * beta / factor_f).ceil());
            w = snap((orig_w * beta / factor_f).ceil());
        }
        // Deviation: with `min_pixels == 0` an edge rounding to 0 passes both
        // checks and the reference returns a zero edge; floor it to `factor`.
        (
            u32::try_from(h.max(factor)).unwrap_or(u32::MAX),
            u32::try_from(w.max(factor)).unwrap_or(u32::MAX),
        )
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

    /// Expected values are the reference `smart_resize` (transformers 5.24.1
    /// `hunyuan_vl` and mlx-vlm, which agree), whose Python `round` breaks a
    /// tie towards the even neighbour.
    #[test]
    fn smart_resize_breaks_ties_to_even_like_python_round() {
        let proc = HunyuanVlProcessor::default();
        // 720 / 32 = 22.5 -> 22; 1680 / 32 = 52.5 -> 52.
        assert_eq!(proc.smart_resize(720, 1280), (704, 1280));
        assert_eq!(proc.smart_resize(1050, 1680), (1056, 1664));
        // Non-ties, an odd tie (1200 / 32 = 37.5 -> 38) and both clamps.
        assert_eq!(proc.smart_resize(1080, 1920), (1088, 1920));
        assert_eq!(proc.smart_resize(1200, 1920), (1216, 1920));
        assert_eq!(proc.smart_resize(3024, 4032), (1760, 2336));
        assert_eq!(proc.smart_resize(336, 336), (512, 512));
    }

    /// A zero edge must not turn `beta` infinite and overflow the other edge
    /// (a debug panic, a wrapped size in release); it is rescaled as one
    /// pixel. `(0, 10)`: `beta = sqrt(262144 / 10) = 161.9`, so
    /// `ceil(161.9 / 32) = 6` and `ceil(1619 / 32) = 51` give 192x1632.
    #[test]
    fn smart_resize_zero_edge_stays_bounded() {
        let p = HunyuanVlProcessor::default();
        assert_eq!(p.smart_resize(0, 10), (192, 1632));
        assert_eq!(p.smart_resize(10, 0), (1632, 192));
        assert_eq!(p.smart_resize(0, 0), (512, 512));
    }

    /// A thin edge rounds to 0 in the reference's first step, so the area
    /// falls under `min_pixels` and the original size is rescaled. Expected
    /// values are transformers 5.16.1 `hunyuan_vl` `smart_resize`. With the
    /// HunyuanOCR bounds `(10, 1000, 32, 262144, 4194304)` gives 64x5120,
    /// as before (32x992 was under `min_pixels` too). With `min_pixels =
    /// 4096`, `beta = sqrt(4096 / 10000) = 0.64`, `ceil(6.4 / 32) = 1` and
    /// `ceil(640 / 32) = 20` give 32x640, where a one-factor floor on the
    /// first step kept 32x992 (31744 pixels, over the bound).
    #[test]
    fn smart_resize_thin_edge_rescales_like_the_reference() {
        let p = HunyuanVlProcessor::default();
        assert_eq!(p.smart_resize(10, 1000), (64, 5120));
        assert_eq!(p.smart_resize(1000, 10), (5120, 64));
        let small = HunyuanVlProcessor {
            min_pixels: 4096,
            ..HunyuanVlProcessor::default()
        };
        assert_eq!(small.smart_resize(10, 1000), (32, 640));
        assert_eq!(small.smart_resize(1000, 10), (640, 32));
        let zero = HunyuanVlProcessor {
            min_pixels: 0,
            ..HunyuanVlProcessor::default()
        };
        assert_eq!(zero.smart_resize(10, 1000), (32, 992));
    }

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
