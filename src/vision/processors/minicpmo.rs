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

//! MiniCPM-o image preprocessing.
//!
//! The upstream processor keeps per-image `tgt_size` metadata because the
//! vision tower uses dynamic positional bucketing rather than a fixed grid.

use image::imageops::FilterType;

pub struct MiniCPMOImageInput {
    pub pixel_values: Vec<f32>,
    pub pixel_values_shape: [i32; 4],
    pub spatial_shape: (i32, i32),
}

pub struct MiniCPMOProcessor {
    pub patch_size: usize,
    pub scale_resolution: usize,
    pub image_feature_size: usize,
    resize_multiple_h: usize,
    resize_multiple_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl MiniCPMOProcessor {
    pub fn new(patch_size: usize, scale_resolution: usize, image_feature_size: usize) -> Self {
        Self::new_with_resize_multiples(
            patch_size,
            scale_resolution,
            image_feature_size,
            patch_size,
            patch_size,
        )
    }

    /// Create a processor whose resized dimensions are rounded to caller
    /// supplied multiples.
    ///
    /// Used by: MiniCPM-O (`patch_size`) and MiniCPM-V 4.6
    /// (`patch_size * VitMerger * Merger`) preprocessing paths.
    pub fn new_with_resize_multiples(
        patch_size: usize,
        scale_resolution: usize,
        image_feature_size: usize,
        resize_multiple_h: usize,
        resize_multiple_w: usize,
    ) -> Self {
        Self {
            patch_size,
            scale_resolution,
            image_feature_size,
            resize_multiple_h: resize_multiple_h.max(1),
            resize_multiple_w: resize_multiple_w.max(1),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    fn ensure_divide(&self, length: usize, divisor: usize) -> usize {
        ((length.max(divisor) + divisor / 2) / divisor) * divisor
    }

    pub(crate) fn find_best_resize(&self, width: usize, height: usize) -> (usize, usize) {
        let mut resized_w = width;
        let mut resized_h = height;

        if width * height > self.scale_resolution * self.scale_resolution || width * height == 0 {
            let ratio = if height == 0 {
                1.0
            } else {
                width as f32 / height as f32
            };
            resized_h = (self.scale_resolution as f32 / ratio.max(1e-6).sqrt()).round() as usize;
            resized_w = (resized_h as f32 * ratio).round() as usize;
        }

        (
            self.ensure_divide(resized_w.max(self.patch_size), self.resize_multiple_w),
            self.ensure_divide(resized_h.max(self.patch_size), self.resize_multiple_h),
        )
    }

    fn preprocess_single(&self, image: &image::DynamicImage) -> MiniCPMOImageInput {
        let rgb = image.to_rgb8();
        let (resized_w, resized_h) =
            self.find_best_resize(rgb.width() as usize, rgb.height() as usize);
        let resized =
            image.resize_exact(resized_w as u32, resized_h as u32, FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        let mut pixel_values = vec![0.0f32; resized_w * resized_h * 3];
        for y in 0..resized_h {
            for x in 0..resized_w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let value = pixel[c] as f32 / 255.0;
                    let normalized = (value - self.mean[c]) / self.std[c];
                    pixel_values[(y * resized_w + x) * 3 + c] = normalized;
                }
            }
        }

        MiniCPMOImageInput {
            pixel_values,
            pixel_values_shape: [1, resized_h as i32, resized_w as i32, 3],
            spatial_shape: (
                (resized_h / self.patch_size) as i32,
                (resized_w / self.patch_size) as i32,
            ),
        }
    }

    pub fn preprocess(&self, images: &[image::DynamicImage]) -> Vec<MiniCPMOImageInput> {
        images
            .iter()
            .map(|image| self.preprocess_single(image))
            .collect()
    }
}

#[cfg(test)]
#[path = "minicpmo_tests.rs"]
mod tests;
