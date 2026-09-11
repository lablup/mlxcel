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

//! Moondream3 image preprocessing.
//!
//! The upstream model builds a global crop plus overlapping local crops. The
//! Rust port keeps the same crop-selection rules and normalization policy so
//! the vision tower sees the expected `[num_crops, 3, 378, 378]` tensor.

use image::imageops::{FilterType, resize};
use image::{DynamicImage, Rgb, RgbImage};

use super::ImageProcessor;

#[derive(Debug, Clone)]
pub struct Moondream3ImageInput {
    pub pixel_values: Vec<f32>,
    pub pixel_values_shape: Vec<i32>,
    /// Tiling grid (h_tiles, w_tiles) used for crop reconstruction.
    pub tiling: (usize, usize),
}

#[derive(Debug, Clone, Copy)]
pub struct Moondream3Processor {
    pub crop_size: usize,
    pub patch_size: usize,
    pub max_crops: usize,
    pub overlap_margin: usize,
}

impl Moondream3Processor {
    pub fn new(
        crop_size: usize,
        patch_size: usize,
        max_crops: usize,
        overlap_margin: usize,
    ) -> Self {
        Self {
            crop_size,
            patch_size,
            max_crops,
            overlap_margin,
        }
    }

    pub(crate) fn select_tiling(&self, height: usize, width: usize) -> (usize, usize) {
        if height <= self.crop_size || width <= self.crop_size {
            return (1, 1);
        }

        let min_h = height.div_ceil(self.crop_size);
        let min_w = width.div_ceil(self.crop_size);

        if min_h * min_w > self.max_crops {
            let ratio = (self.max_crops as f64 / (min_h * min_w) as f64).sqrt();
            return (
                ((min_h as f64 * ratio).floor() as usize).max(1),
                ((min_w as f64 * ratio).floor() as usize).max(1),
            );
        }

        let mut h_tiles = ((self.max_crops as f64 * height as f64 / width as f64)
            .sqrt()
            .floor() as usize)
            .max(min_h);
        let mut w_tiles = ((self.max_crops as f64 * width as f64 / height as f64)
            .sqrt()
            .floor() as usize)
            .max(min_w);

        if h_tiles * w_tiles > self.max_crops {
            if w_tiles > h_tiles {
                w_tiles = (self.max_crops / h_tiles).max(1);
            } else {
                h_tiles = (self.max_crops / w_tiles).max(1);
            }
        }

        (h_tiles.max(1), w_tiles.max(1))
    }

    fn overlap_crop_image(&self, image: &RgbImage) -> (Vec<RgbImage>, (usize, usize)) {
        let original_h = image.height() as usize;
        let original_w = image.width() as usize;
        let margin_pixels = self.patch_size * self.overlap_margin;
        let total_margin_pixels = margin_pixels * 2;
        let crop_patches = self.crop_size / self.patch_size;
        let crop_window_patches = crop_patches.saturating_sub(self.overlap_margin * 2);
        let crop_window_size = crop_window_patches * self.patch_size;

        let tiling = self.select_tiling(
            original_h.saturating_sub(total_margin_pixels),
            original_w.saturating_sub(total_margin_pixels),
        );

        let target_h = tiling.0 * crop_window_size + total_margin_pixels;
        let target_w = tiling.1 * crop_window_size + total_margin_pixels;

        let resized = resize(
            image,
            target_w as u32,
            target_h as u32,
            FilterType::Lanczos3,
        );
        let global = resize(
            image,
            self.crop_size as u32,
            self.crop_size as u32,
            FilterType::Lanczos3,
        );

        let mut crops = Vec::with_capacity(1 + tiling.0 * tiling.1);
        crops.push(global);

        for tile_y in 0..tiling.0 {
            for tile_x in 0..tiling.1 {
                let y0 = tile_y * crop_window_size;
                let x0 = tile_x * crop_window_size;
                let y_end = (y0 + self.crop_size).min(resized.height() as usize);
                let x_end = (x0 + self.crop_size).min(resized.width() as usize);

                let mut crop = RgbImage::from_pixel(
                    self.crop_size as u32,
                    self.crop_size as u32,
                    Rgb([0, 0, 0]),
                );
                for src_y in y0..y_end {
                    for src_x in x0..x_end {
                        let pixel = resized.get_pixel(src_x as u32, src_y as u32);
                        crop.put_pixel((src_x - x0) as u32, (src_y - y0) as u32, *pixel);
                    }
                }
                crops.push(crop);
            }
        }

        (crops, tiling)
    }

    pub fn preprocess_image(&self, image: &DynamicImage) -> Moondream3ImageInput {
        let rgb = image.to_rgb8();
        let (crops, tiling) = self.overlap_crop_image(&rgb);
        let crop_count = crops.len();
        let mut pixel_values = vec![0.0f32; crop_count * 3 * self.crop_size * self.crop_size];

        for (crop_idx, crop) in crops.iter().enumerate() {
            for y in 0..self.crop_size {
                for x in 0..self.crop_size {
                    let pixel = crop.get_pixel(x as u32, y as u32);
                    for channel in 0..3 {
                        let value = (pixel[channel] as f32 / 255.0 - 0.5) / 0.5;
                        let offset = crop_idx * 3 * self.crop_size * self.crop_size
                            + channel * self.crop_size * self.crop_size
                            + y * self.crop_size
                            + x;
                        pixel_values[offset] = value;
                    }
                }
            }
        }

        Moondream3ImageInput {
            pixel_values,
            pixel_values_shape: vec![
                crop_count as i32,
                3,
                self.crop_size as i32,
                self.crop_size as i32,
            ],
            tiling,
        }
    }
}

impl ImageProcessor for Moondream3Processor {
    fn preprocess(&self, images: &[DynamicImage]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
        let processed = images
            .first()
            .map(|image| self.preprocess_image(image))
            .unwrap_or(Moondream3ImageInput {
                pixel_values: Vec::new(),
                pixel_values_shape: vec![0, 3, self.crop_size as i32, self.crop_size as i32],
                tiling: (1, 1),
            });
        mlxcel_core::from_slice_f32(&processed.pixel_values, &processed.pixel_values_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::Moondream3Processor;
    use image::{DynamicImage, Rgb, RgbImage};

    fn processor() -> Moondream3Processor {
        Moondream3Processor::new(378, 14, 12, 4)
    }

    #[test]
    fn select_tiling_keeps_small_images_single_crop() {
        assert_eq!(processor().select_tiling(300, 320), (1, 1));
    }

    #[test]
    fn select_tiling_scales_large_images_under_max_crop_budget() {
        let tiling = processor().select_tiling(1600, 1200);
        assert!(tiling.0 >= 1);
        assert!(tiling.1 >= 1);
        assert!(tiling.0 * tiling.1 <= 12);
    }

    #[test]
    fn preprocess_image_produces_global_and_local_crops() {
        let mut image = RgbImage::new(1024, 768);
        for y in 0..image.height() {
            for x in 0..image.width() {
                image.put_pixel(x, y, Rgb([(x % 255) as u8, (y % 255) as u8, 127]));
            }
        }

        let processed = processor().preprocess_image(&DynamicImage::ImageRgb8(image));
        assert_eq!(processed.pixel_values_shape[1..], [3, 378, 378]);
        assert!(processed.pixel_values_shape[0] > 1);
        assert_eq!(
            processed.pixel_values.len(),
            processed.pixel_values_shape.iter().product::<i32>() as usize
        );
    }
}
