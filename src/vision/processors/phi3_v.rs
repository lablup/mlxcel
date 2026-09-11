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

//! Phi-3V HD dynamic resolution image processor
//!
//! Preprocessing pipeline:
//! 1. HD resize: find optimal tile layout preserving aspect ratio
//! 2. Pad to 336-multiple
//! 3. Create global (336x336) + sub-tiles (split into 336x336 tiles)
//! 4. CLIP normalize: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
//! 5. Output: [num_tiles+1, C, H, W] per image + image_sizes
//!
//! Used by: Phi-3.5 Vision VLM

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

const IMG_SIZE: usize = 336;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

pub struct Phi3VProcessor {
    pub num_crops: usize,
    pub num_img_tokens: usize,
}

impl Phi3VProcessor {
    pub fn new(num_crops: usize) -> Self {
        Self {
            num_crops,
            num_img_tokens: 144,
        }
    }

    /// Calculate the number of image tokens for a given image
    pub fn calc_num_image_tokens(&self, width: u32, height: u32) -> usize {
        let (hd_w, hd_h) = calc_hd_transform_size(width as usize, height as usize, self.num_crops);
        let num_h_tiles = hd_h / IMG_SIZE;
        let num_w_tiles = hd_w / IMG_SIZE;
        // (tiles + 1 global) * 144 + 1 glb_GN + (num_h_tiles + 1) * 12 sub_GN separators
        (num_h_tiles * num_w_tiles + 1) * self.num_img_tokens + 1 + (num_h_tiles + 1) * 12
    }

    /// Preprocess images for Phi-3V
    /// Returns (pixel_values, image_sizes)
    /// pixel_values: [B, num_tiles+1, C, H, W] flattened for single batch
    /// image_sizes: Vec<(hd_h, hd_w)> per image
    pub fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(usize, usize)>) {
        let mut all_pixel_values = Vec::new();
        let mut all_image_sizes = Vec::new();

        for img in images {
            let (pixel_values, image_size) = self.process_single_image(img);
            all_pixel_values.push(pixel_values);
            all_image_sizes.push(image_size);
        }

        // For single image (most common case), just return directly
        // For multiple images, we'd need padding — currently support single image
        assert!(!all_pixel_values.is_empty(), "No images to process");

        let pv = if all_pixel_values.len() == 1 {
            all_pixel_values.into_iter().next().unwrap()
        } else {
            // Pad to max tiles and stack
            let max_tiles = all_pixel_values
                .iter()
                .map(|pv| mlxcel_core::array_shape(pv)[0])
                .max()
                .unwrap();

            let mut first = all_pixel_values.remove(0);
            let shape = mlxcel_core::array_shape(&first);
            if shape[0] < max_tiles {
                let pad_shape = [max_tiles - shape[0], shape[1], shape[2], shape[3]];
                let zeros = mlxcel_core::zeros(&pad_shape, mlxcel_core::dtype::FLOAT32);
                first = mlxcel_core::concatenate(&first, &zeros, 0);
            }
            let mut stacked =
                mlxcel_core::reshape(&first, &[1, max_tiles, shape[1], shape[2], shape[3]]);

            for pv in all_pixel_values {
                let pv_shape = mlxcel_core::array_shape(&pv);
                let mut padded = pv;
                if pv_shape[0] < max_tiles {
                    let pad_shape = [
                        max_tiles - pv_shape[0],
                        pv_shape[1],
                        pv_shape[2],
                        pv_shape[3],
                    ];
                    let zeros = mlxcel_core::zeros(&pad_shape, mlxcel_core::dtype::FLOAT32);
                    padded = mlxcel_core::concatenate(&padded, &zeros, 0);
                }
                let reshaped = mlxcel_core::reshape(
                    &padded,
                    &[1, max_tiles, pv_shape[1], pv_shape[2], pv_shape[3]],
                );
                stacked = mlxcel_core::concatenate(&stacked, &reshaped, 0);
            }
            stacked
        };

        (pv, all_image_sizes)
    }

    /// Process a single image
    /// Returns (pixel_values [num_tiles+1, C, H, W], image_size (hd_h, hd_w))
    fn process_single_image(&self, img: &DynamicImage) -> (UniquePtr<MlxArray>, (usize, usize)) {
        let rgb = img.to_rgb8();
        let (orig_w, orig_h) = (rgb.width() as usize, rgb.height() as usize);

        // HD transform: resize preserving aspect ratio
        let (hd_w, hd_h) = calc_hd_transform_size(orig_w, orig_h, self.num_crops);
        let hd_image = img.resize_exact(hd_w as u32, hd_h as u32, FilterType::CatmullRom);
        let hd_rgb = hd_image.to_rgb8();

        // Pad to 336-multiple (should already be after calc_hd_transform_size, but be safe)
        let padded_w = hd_w.div_ceil(IMG_SIZE) * IMG_SIZE;
        let padded_h = hd_h.div_ceil(IMG_SIZE) * IMG_SIZE;

        // Create global image (resize to 336x336)
        let global_image =
            img.resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, FilterType::CatmullRom);
        let global_rgb = global_image.to_rgb8();

        // Count tiles
        let num_h_tiles = padded_h / IMG_SIZE;
        let num_w_tiles = padded_w / IMG_SIZE;
        let num_tiles = num_h_tiles * num_w_tiles;
        let total_tiles = num_tiles + 1; // global + sub-tiles

        // Allocate [total_tiles, C, H, W]
        let channels = 3usize;
        let total = total_tiles * channels * IMG_SIZE * IMG_SIZE;
        let mut data = vec![0.0f32; total];

        // Process global image (index 0)
        normalize_image_to_buffer(&global_rgb, &mut data, 0, IMG_SIZE, channels);

        // Process sub-tiles
        for h in 0..num_h_tiles {
            for w in 0..num_w_tiles {
                let tile_idx = 1 + h * num_w_tiles + w;
                let x_start = w * IMG_SIZE;
                let y_start = h * IMG_SIZE;

                // Extract tile from HD image (with padding for edges)
                let tile_offset = tile_idx * channels * IMG_SIZE * IMG_SIZE;
                for y in 0..IMG_SIZE {
                    for x in 0..IMG_SIZE {
                        let src_x = x_start + x;
                        let src_y = y_start + y;
                        for c in 0..channels {
                            let val = if src_x < hd_w && src_y < hd_h {
                                hd_rgb.get_pixel(src_x as u32, src_y as u32)[c] as f32 / 255.0
                            } else {
                                0.0 // padding
                            };
                            let normalized = (val - CLIP_MEAN[c]) / CLIP_STD[c];
                            let idx = tile_offset + c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x;
                            data[idx] = normalized;
                        }
                    }
                }
            }
        }

        let pixel_values = mlxcel_core::from_slice_f32(
            &data,
            &[
                total_tiles as i32,
                channels as i32,
                IMG_SIZE as i32,
                IMG_SIZE as i32,
            ],
        );

        (pixel_values, (padded_h, padded_w))
    }
}

/// Normalize a single image into the buffer at the given tile offset
fn normalize_image_to_buffer(
    rgb: &image::RgbImage,
    data: &mut [f32],
    tile_idx: usize,
    size: usize,
    channels: usize,
) {
    let offset = tile_idx * channels * size * size;
    for y in 0..size {
        for x in 0..size {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..channels {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - CLIP_MEAN[c]) / CLIP_STD[c];
                let idx = offset + c * size * size + y * size + x;
                data[idx] = normalized;
            }
        }
    }
}

/// Calculate HD transform size preserving aspect ratio
/// Returns (target_width, target_height) both divisible by 336
fn calc_hd_transform_size(width: usize, height: usize, hd_num: usize) -> (usize, usize) {
    let (w, h, transposed) = if width < height {
        (height, width, true)
    } else {
        (width, height, false)
    };

    let ratio = w as f64 / h as f64;
    let mut scale = 1usize;
    while scale * ((scale as f64 / ratio).ceil() as usize) <= hd_num {
        scale += 1;
    }
    scale -= 1;
    if scale == 0 {
        scale = 1;
    }

    let new_w = scale * IMG_SIZE;
    let new_h = (new_w as f64 / ratio) as usize;

    // Pad to 336-multiple
    let padded_w = new_w.div_ceil(IMG_SIZE) * IMG_SIZE;
    let padded_h = new_h.div_ceil(IMG_SIZE) * IMG_SIZE;

    if transposed {
        (padded_h, padded_w)
    } else {
        (padded_w, padded_h)
    }
}
