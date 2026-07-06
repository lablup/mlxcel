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

//! FastVLM image processor: pad-to-square, bicubic resize to a fixed square,
//! rescale to `[0, 1]`, per-channel normalize, and emit channels-first
//! `(B, C, image_size, image_size)`. Defaults (`image_mean = 0`, `image_std = 1`)
//! make normalization a no-op; the loader overrides them from
//! `preprocessor_config.json` when present.
//!
//! Reference: mlx-vlm `mlx_vlm/models/fastvlm/` (`image_aspect_ratio = "pad"`).

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct FastvlmProcessor {
    pub image_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for FastvlmProcessor {
    fn default() -> Self {
        Self {
            image_size: 1024,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }
}

impl ImageProcessor for FastvlmProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let batch = images.len();
        let size = self.image_size;
        let channels = 3usize;
        let mut data = vec![0.0f32; batch * channels * size * size];

        for (img_idx, img) in images.iter().enumerate() {
            let rgb = img.to_rgb8();
            let (w, h) = (rgb.width(), rgb.height());

            // Pad to square (side = max), original centered, fill 0.
            let side = w.max(h);
            let mut square = image::RgbImage::from_pixel(side, side, image::Rgb([0, 0, 0]));
            let ox = ((side - w) / 2) as i64;
            let oy = ((side - h) / 2) as i64;
            image::imageops::overlay(&mut square, &rgb, ox, oy);

            // Bicubic resize to (size, size).
            let resized =
                image::imageops::resize(&square, size as u32, size as u32, FilterType::CatmullRom);

            for y in 0..size {
                for x in 0..size {
                    let px = resized.get_pixel(x as u32, y as u32);
                    for c in 0..channels {
                        let v = px[c] as f32 / 255.0;
                        let v = (v - self.mean[c]) / self.std[c];
                        // (B, C, H, W) layout.
                        let idx = img_idx * channels * size * size + c * size * size + y * size + x;
                        data[idx] = v;
                    }
                }
            }
        }

        mlxcel_core::from_slice_f32(
            &data,
            &[batch as i32, channels as i32, size as i32, size as i32],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_channels_first_square_batch() {
        let proc = FastvlmProcessor::default();
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            640,
            480,
            image::Rgb([10, 20, 30]),
        ));
        let out = proc.preprocess(std::slice::from_ref(&img));
        assert_eq!(mlxcel_core::array_shape(&out), vec![1, 3, 1024, 1024]);
    }

    #[test]
    fn default_normalization_is_rescale_only() {
        // A solid mid-gray image with default mean 0 / std 1 maps to 128/255.
        let proc = FastvlmProcessor::default();
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            32,
            32,
            image::Rgb([128, 128, 128]),
        ));
        let out = proc.preprocess(std::slice::from_ref(&img));
        let center = mlxcel_core::slice(&out, &[0, 0, 512, 512], &[1, 1, 513, 513]);
        mlxcel_core::eval(&center);
        let v = mlxcel_core::item_f32(&center);
        assert!((v - 128.0 / 255.0).abs() < 1e-3, "expected ~0.502, got {v}");
    }
}
