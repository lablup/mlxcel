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

//! MiniMax-M3-VL image processor.
//!
//! A faithful port of the checkpoint's `image_processor.py`
//! (`MiniMaxM3VLImageProcessor`), which is Qwen2-VL-style dynamic-resolution
//! preprocessing:
//! 1. `smart_resize` to dimensions that are multiples of
//!    `factor = patch_size * merge_size = 28`, bounded by `min_pixels` /
//!    `max_pixels` (672x672),
//! 2. CLIP mean/std normalization,
//! 3. temporal padding: repeat the last frame to a multiple of
//!    `temporal_patch_size` (a single image becomes 2 identical frames),
//! 4. patchify into the exact `(grid_t, grid_h/m, grid_w/m, m, m, C, temporal,
//!    patch_h, patch_w)` order, emitting `pixel_values` of shape
//!    `[num_patches, C * temporal * patch * patch]` (= `[num_patches, 1176]`)
//!    plus `image_grid_thw`.
//!
//! The patch sequence order groups each 2x2 spatial-merge cell contiguously and
//! the per-row feature order is `[channel, temporal, patch_h, patch_w]`, which
//! is what the tower's patch embedding and patch-merge fold both assume.

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

const MAX_RATIO: f64 = 200.0;

pub struct MiniMaxM3Processor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for MiniMaxM3Processor {
    fn default() -> Self {
        Self {
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            min_pixels: 4 * 28 * 28, // 3136
            max_pixels: 451_584,     // 672 * 672
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
        }
    }
}

fn round_by_factor(n: f64, factor: f64) -> f64 {
    (n / factor).round() * factor
}
fn ceil_by_factor(n: f64, factor: f64) -> f64 {
    (n / factor).ceil() * factor
}
fn floor_by_factor(n: f64, factor: f64) -> f64 {
    (n / factor).floor() * factor
}

impl MiniMaxM3Processor {
    fn factor(&self) -> f64 {
        (self.patch_size * self.spatial_merge_size) as f64
    }

    /// Port of `smart_resize`. Returns `(height, width)`, both multiples of
    /// `factor`. Aspect ratios beyond `MAX_RATIO` are clamped to `MAX_RATIO`
    /// rather than raising, so a degenerate input still yields a valid grid.
    pub fn smart_resize(&self, height: u32, width: u32) -> (u32, u32) {
        let factor = self.factor();
        let mut h = height as f64;
        let mut w = width as f64;

        let ratio = h.max(w) / h.min(w).max(1.0);
        if ratio > MAX_RATIO {
            // Clamp the long side so the ratio equals MAX_RATIO.
            if h > w {
                h = w * MAX_RATIO;
            } else {
                w = h * MAX_RATIO;
            }
        }

        let mut h_bar = factor.max(round_by_factor(h, factor));
        let mut w_bar = factor.max(round_by_factor(w, factor));

        if h_bar * w_bar > self.max_pixels as f64 {
            let beta = (h * w / self.max_pixels as f64).sqrt();
            h_bar = floor_by_factor(h / beta, factor).max(factor);
            w_bar = floor_by_factor(w / beta, factor).max(factor);
        } else if h_bar * w_bar < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (h * w)).sqrt();
            h_bar = ceil_by_factor(h * beta, factor);
            w_bar = ceil_by_factor(w * beta, factor);
        }

        (h_bar as u32, w_bar as u32)
    }

    /// Per-image `(temporal, grid_h, grid_w)` in post-resize patch units.
    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                let grid_h = h as i32 / self.patch_size as i32;
                let grid_w = w as i32 / self.patch_size as i32;
                (1i32, grid_h, grid_w)
            })
            .collect()
    }

    /// Preprocess images into `(pixel_values, grid_thw)`.
    ///
    /// `pixel_values` has shape `[sum(t * grid_h * grid_w), 1176]` with patch
    /// rows in merge-grouped order and per-row feature order
    /// `[channel, temporal, patch_h, patch_w]`.
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let grid_thw = self.compute_grid_thw(images);
        let patch = self.patch_size;
        let merge = self.spatial_merge_size;
        let in_channels = 3usize;
        let features_per_row = in_channels * self.temporal_patch_size * patch * patch;

        let mut all_patches: Vec<f32> = Vec::new();

        for (img_idx, img) in images.iter().enumerate() {
            let (_, grid_h, grid_w) = grid_thw[img_idx];
            let grid_h = grid_h as usize;
            let grid_w = grid_w as usize;
            let target_h = (grid_h * patch) as u32;
            let target_w = (grid_w * patch) as u32;

            let resized = img.resize_exact(target_w, target_h, FilterType::Lanczos3);
            let rgb = resized.to_rgb8();
            let h = target_h as usize;
            let w = target_w as usize;

            // Channels-first normalized buffer: normalized[c*h*w + y*w + x].
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

            // Merge-grouped patch order: for each 2x2 merge cell, emit its
            // `merge^2` sub-patches contiguously. Per-row features run
            // [channel, temporal, patch_h, patch_w]; the temporal frames are
            // identical copies (single-image temporal padding).
            let hm = grid_h / merge;
            let wm = grid_w / merge;
            for hh in 0..hm {
                for ww in 0..wm {
                    for mh in 0..merge {
                        for mw in 0..merge {
                            let py = hh * merge + mh;
                            let px = ww * merge + mw;
                            let y0 = py * patch;
                            let x0 = px * patch;
                            for c in 0..in_channels {
                                for _tp in 0..self.temporal_patch_size {
                                    for dy in 0..patch {
                                        for dx in 0..patch {
                                            let y = y0 + dy;
                                            let x = x0 + dx;
                                            all_patches.push(normalized[c * h * w + y * w + x]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let total_rows: usize = grid_thw
            .iter()
            .map(|&(t, gh, gw)| (t * gh * gw) as usize)
            .sum();

        let pixel_values = mlxcel_core::from_slice_f32(
            &all_patches,
            &[total_rows as i32, features_per_row as i32],
        );

        (pixel_values, grid_thw)
    }
}

impl ImageProcessor for MiniMaxM3Processor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            w,
            h,
            image::Rgb([120, 60, 200]),
        ))
    }

    #[test]
    fn smart_resize_aligns_to_factor_and_bounds() {
        let p = MiniMaxM3Processor::default();
        let factor = (p.patch_size * p.spatial_merge_size) as u32; // 28
        for &(h, w) in &[(100u32, 100u32), (37, 512), (1024, 300), (13, 640)] {
            let (rh, rw) = p.smart_resize(h, w);
            assert_eq!(rh % factor, 0, "height {rh} not multiple of {factor}");
            assert_eq!(rw % factor, 0, "width {rw} not multiple of {factor}");
            let pixels = (rh as usize) * (rw as usize);
            assert!(pixels >= p.min_pixels, "pixels {pixels} below min");
            assert!(pixels <= p.max_pixels, "pixels {pixels} above max");
        }
    }

    #[test]
    fn grid_thw_matches_resized_patch_grid() {
        let p = MiniMaxM3Processor::default();
        let img = solid_image(200, 100); // w=200, h=100
        let grid = p.compute_grid_thw(&[img]);
        assert_eq!(grid.len(), 1);
        let (t, gh, gw) = grid[0];
        assert_eq!(t, 1);
        let (rh, rw) = p.smart_resize(100, 200);
        assert_eq!(gh, rh as i32 / p.patch_size as i32);
        assert_eq!(gw, rw as i32 / p.patch_size as i32);
        // grid dims are even (multiple of merge) so the fold-by-merge^2 is exact.
        assert_eq!(gh % p.spatial_merge_size as i32, 0);
        assert_eq!(gw % p.spatial_merge_size as i32, 0);
    }

    #[test]
    fn preprocess_emits_1176_dim_rows_and_grid_prod_patches() {
        let p = MiniMaxM3Processor::default();
        let img = solid_image(140, 84); // small, non-square
        let (pixel_values, grid) = p.preprocess_with_grid(&[img]);
        mlxcel_core::eval(&pixel_values);
        let shape = mlxcel_core::array_shape(&pixel_values);
        let (t, gh, gw) = grid[0];
        let expected_rows = (t * gh * gw) as i32;
        assert_eq!(shape, vec![expected_rows, 1176]);
    }

    #[test]
    fn placeholder_count_equals_grid_prod_over_merge_squared() {
        // The vision tower emits grid.prod()/merge^2 merged tokens, which must
        // equal the placeholder expansion count.
        let p = MiniMaxM3Processor::default();
        let img = solid_image(280, 196);
        let grid = p.compute_grid_thw(&[img]);
        let (t, gh, gw) = grid[0];
        let merge = p.spatial_merge_size as i32;
        let merged_tokens = t * (gh / merge) * (gw / merge);
        let grid_prod_over_merge2 = (t * gh * gw) / (merge * merge);
        assert_eq!(merged_tokens, grid_prod_over_merge2);
    }
}
