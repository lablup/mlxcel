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

//! SigLIP image preprocessor
//!
//! Preprocessing pipeline:
//! 1. Resize to (image_size, image_size) using bilinear interpolation
//! 2. Convert to f32, scale by 1/255
//! 3. Normalize: (pixel - mean) / std with SigLIP defaults mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
//! 4. Layout: [B, C, H, W] format for the vision module to transpose as needed

use super::ImageProcessor;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct SigLipProcessor {
    pub image_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub do_normalize: bool,
}

impl SigLipProcessor {
    pub fn new(image_size: usize) -> Self {
        Self {
            image_size,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            do_normalize: true,
        }
    }

    /// Create a processor that only rescales to [0, 1] without normalization
    pub fn new_rescale_only(image_size: usize) -> Self {
        Self {
            image_size,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            do_normalize: false,
        }
    }
}

impl ImageProcessor for SigLipProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let batch_size = images.len();
        let size = self.image_size;
        let channels = 3usize;

        // Allocate buffer: [B, C, H, W]
        let total = batch_size * channels * size * size;
        let mut data = vec![0.0f32; total];

        for (img_idx, img) in images.iter().enumerate() {
            // Resize to target size using bilinear interpolation
            let resized = img.resize_exact(size as u32, size as u32, FilterType::Triangle);
            let rgb = resized.to_rgb8();

            // Convert to f32, normalize
            for y in 0..size {
                for x in 0..size {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..channels {
                        let val = pixel[c] as f32 / 255.0;
                        let normalized = if self.do_normalize {
                            (val - self.mean[c]) / self.std[c]
                        } else {
                            val
                        };
                        // [B, C, H, W] layout
                        let idx = img_idx * channels * size * size + c * size * size + y * size + x;
                        data[idx] = normalized;
                    }
                }
            }
        }

        mlxcel_core::from_slice_f32(
            &data,
            &[batch_size as i32, channels as i32, size as i32, size as i32],
        )
    }
}
