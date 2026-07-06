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

//! dots.ocr dynamic-resolution image processor.
//!
//! Smart-resizes each image to a multiple of `patch_size * merge_size`, applies
//! CLIP normalization, and flattens to patch rows `(sum t*h*w, C*pH*pW)` with
//! `temporal_patch_size = 1` (one row per spatial patch). Patch rows are emitted
//! in merge-block-grouped order: blocks row-major over `(hb, wb)`, and within a
//! block the order `(0,0), (0,1), (1,0), (1,1)`, so row `r` of patch `(py, px)`
//! is `((hb*(w/2)+wb)*2+mh)*2+mw`. This matches the encoder's `rot_pos_emb`
//! ordering and the merger's consecutive-4 grouping.
//!
//! Reference: mlx-vlm `mlx_vlm/models/dots_ocr/` image processor.

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct DotsOcrProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for DotsOcrProcessor {
    fn default() -> Self {
        Self {
            patch_size: 14,
            temporal_patch_size: 1,
            merge_size: 2,
            min_pixels: 3136,     // 4 * 28 * 28
            max_pixels: 11289600, // dots.ocr default (differs from Qwen2-VL)
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
        }
    }
}

impl DotsOcrProcessor {
    /// Round each side to a multiple of `patch_size * merge_size` and clamp the
    /// area into `[min_pixels, max_pixels]`. Returns `(height, width)` in pixels.
    fn smart_resize(&self, h: u32, w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.merge_size) as f64; // 28
        let round = |x: f64| ((x / factor).round().max(1.0) * factor) as u32;
        let mut rh = round(h as f64);
        let mut rw = round(w as f64);
        let pixels = (rh as usize) * (rw as usize);
        if pixels > self.max_pixels {
            let scale = (self.max_pixels as f64 / pixels as f64).sqrt();
            let floor = |x: f64| (((x / factor).floor()).max(1.0) * factor) as u32;
            rh = floor(rh as f64 * scale);
            rw = floor(rw as f64 * scale);
        } else if pixels < self.min_pixels {
            let scale = (self.min_pixels as f64 / pixels as f64).sqrt();
            let ceil = |x: f64| (((x / factor).ceil()).max(1.0) * factor) as u32;
            rh = ceil(rh as f64 * scale);
            rw = ceil(rw as f64 * scale);
        }
        (rh, rw)
    }

    /// `(temporal=1, h_patches, w_patches)` per image.
    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                (
                    1,
                    h as i32 / self.patch_size as i32,
                    w as i32 / self.patch_size as i32,
                )
            })
            .collect()
    }

    /// Returns `(pixel_values (total_rows, C*pH*pW), grid_thw)`.
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let grid_thw = self.compute_grid_thw(images);
        let p = self.patch_size;
        let features_per_row = 3 * p * p;
        let merge = self.merge_size;
        let mut all: Vec<f32> = Vec::new();

        for (idx, img) in images.iter().enumerate() {
            let (_, hp, wp) = grid_thw[idx];
            let (hp, wp) = (hp as usize, wp as usize);
            let (th, tw) = ((hp * p) as u32, (wp * p) as u32);
            let resized = img.resize_exact(tw, th, FilterType::Lanczos3);
            let rgb = resized.to_rgb8();
            let (h, w) = (th as usize, tw as usize);

            let mut norm = vec![0f32; 3 * h * w];
            for y in 0..h {
                for x in 0..w {
                    let px = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        norm[c * h * w + y * w + x] =
                            (px[c] as f32 / 255.0 - self.mean[c]) / self.std[c];
                    }
                }
            }

            // Merge-block-grouped emission: blocks over (hb, wb), then (mh, mw).
            for hb in 0..hp / merge {
                for wb in 0..wp / merge {
                    for mh in 0..merge {
                        for mw in 0..merge {
                            let py = hb * merge + mh;
                            let pxx = wb * merge + mw;
                            let (ys, xs) = (py * p, pxx * p);
                            // Feature order (c, dy, dx).
                            for c in 0..3 {
                                for dy in 0..p {
                                    for dx in 0..p {
                                        all.push(norm[c * h * w + (ys + dy) * w + (xs + dx)]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let total_rows: usize = grid_thw.iter().map(|&(t, h, w)| (t * h * w) as usize).sum();
        let pixel_values =
            mlxcel_core::from_slice_f32(&all, &[total_rows as i32, features_per_row as i32]);
        (pixel_values, grid_thw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rounds_to_factor_and_clamps() {
        let p = DotsOcrProcessor::default();
        // In-range, rounds to multiples of 28.
        let (h, w) = p.smart_resize(224, 224);
        assert_eq!((h % 28, w % 28), (0, 0));
        assert_eq!((h, w), (224, 224));
        // Below min_pixels (3136 = 56*56): scaled up to at least one factor cell.
        let (h, w) = p.smart_resize(10, 10);
        assert!((h as usize) * (w as usize) >= p.min_pixels);
        assert_eq!((h % 28, w % 28), (0, 0));
        // Above max_pixels: scaled down under the cap.
        let (h, w) = p.smart_resize(8000, 8000);
        assert!((h as usize) * (w as usize) <= p.max_pixels);
        assert_eq!((h % 28, w % 28), (0, 0));
    }

    #[test]
    fn merge_block_row_index_matches_invariant() {
        // Reproduce the emission order for a 4x6 patch grid (h=4, w=6, merge=2)
        // and assert each patch lands at r = ((hb*(w/2)+wb)*2+mh)*2+mw.
        let (hp, wp, merge) = (4usize, 6usize, 2usize);
        let mut r = 0usize;
        for hb in 0..hp / merge {
            for wb in 0..wp / merge {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let expected = ((hb * (wp / 2) + wb) * 2 + mh) * 2 + mw;
                        assert_eq!(r, expected, "hb={hb} wb={wb} mh={mh} mw={mw}");
                        r += 1;
                    }
                }
            }
        }
        assert_eq!(r, hp * wp);
    }
}
