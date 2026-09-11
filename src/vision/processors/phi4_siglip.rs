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

//! Phi4-SigLIP NaFlex-style image processor.
//!
//! Used by: Phi4-SigLIP VLM

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Phi4SigLipImageInput {
    pub pixel_values: UniquePtr<MlxArray>, // [1, num_patches, patch_dim]
    pub spatial_shape: (i32, i32),         // (h_patches, w_patches)
}

pub struct Phi4SigLipProcessor {
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub patch_size: usize,
    pub min_num_patches: usize,
    pub max_num_patches: usize,
}

impl Phi4SigLipProcessor {
    pub fn new(patch_size: usize, min_num_patches: usize, max_num_patches: usize) -> Self {
        Self {
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            patch_size,
            min_num_patches,
            max_num_patches,
        }
    }

    pub fn preprocess(&self, images: &[DynamicImage]) -> Vec<Phi4SigLipImageInput> {
        images
            .iter()
            .map(|image| self.preprocess_single(image))
            .collect()
    }

    fn preprocess_single(&self, image: &DynamicImage) -> Phi4SigLipImageInput {
        let rgb = image.to_rgb8();
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);
        let num_patches = ((height / self.patch_size) * (width / self.patch_size)).max(1);
        let bounded_patch_count = num_patches.clamp(self.min_num_patches, self.max_num_patches);
        let (target_h, target_w) =
            get_image_size_for_max_num_patches(height, width, self.patch_size, bounded_patch_count);

        let resized = image.resize_exact(target_w as u32, target_h as u32, FilterType::Triangle);
        let resized = resized.to_rgb8();
        let h_patches = target_h / self.patch_size;
        let w_patches = target_w / self.patch_size;
        let patch_dim = self.patch_size * self.patch_size * 3;
        let num_patches = h_patches * w_patches;

        let mut data = vec![0.0f32; num_patches * patch_dim];
        for patch_h in 0..h_patches {
            for patch_w in 0..w_patches {
                let patch_idx = patch_h * w_patches + patch_w;
                let patch_offset = patch_idx * patch_dim;

                for inner_h in 0..self.patch_size {
                    for inner_w in 0..self.patch_size {
                        let src_x = patch_w * self.patch_size + inner_w;
                        let src_y = patch_h * self.patch_size + inner_h;
                        let pixel = resized.get_pixel(src_x as u32, src_y as u32);
                        let flat_patch_index = (inner_h * self.patch_size + inner_w) * 3;
                        for channel in 0..3 {
                            let value = pixel[channel] as f32 / 255.0;
                            data[patch_offset + flat_patch_index + channel] =
                                (value - self.image_mean[channel]) / self.image_std[channel];
                        }
                    }
                }
            }
        }

        Phi4SigLipImageInput {
            pixel_values: mlxcel_core::from_slice_f32(
                &data,
                &[1, num_patches as i32, patch_dim as i32],
            ),
            spatial_shape: (h_patches as i32, w_patches as i32),
        }
    }
}

pub(crate) fn get_image_size_for_max_num_patches(
    image_height: usize,
    image_width: usize,
    patch_size: usize,
    max_num_patches: usize,
) -> (usize, usize) {
    let aspect_ratio = image_width as f32 / image_height.max(1) as f32;
    let mut max_height_patches = (max_num_patches as f32 / aspect_ratio).sqrt().floor() as usize;
    let mut max_width_patches = (max_height_patches as f32 * aspect_ratio).floor() as usize;

    while max_height_patches * max_width_patches > max_num_patches {
        if max_height_patches > max_width_patches {
            max_height_patches -= 1;
        } else {
            max_width_patches -= 1;
        }
    }

    let max_height_patches = max_height_patches.max(1);
    let max_width_patches = max_width_patches.max(1);
    (
        max_height_patches * patch_size,
        max_width_patches * patch_size,
    )
}

#[cfg(test)]
#[path = "phi4_siglip_tests.rs"]
mod tests;
