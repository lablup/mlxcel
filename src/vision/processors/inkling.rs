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

//! Inkling's exact 40x40 image tiler and CLIP normalizer.

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::ImageProcessor;

const TILE_SIZE: usize = 40;
const TEMPORAL_SIZE: usize = 2;
const CHANNELS: usize = 3;

fn default_mean() -> [f32; CHANNELS] {
    [0.481_454_66, 0.457_827_5, 0.408_210_73]
}

fn default_std() -> [f32; CHANNELS] {
    [0.268_629_54, 0.261_302_6, 0.275_777_1]
}

fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

fn default_max_upscaled_edge() -> Option<usize> {
    Some(2048)
}

#[derive(Debug, Clone, Deserialize)]
pub struct InklingImageProcessorConfig {
    #[serde(default = "default_mean")]
    pub image_mean: [f32; CHANNELS],
    #[serde(default = "default_std")]
    pub image_std: [f32; CHANNELS],
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
    #[serde(default)]
    pub rescale_image_frac: Option<f64>,
    #[serde(default = "default_max_upscaled_edge")]
    pub rescale_image_max_upscaled_long_edge: Option<usize>,
}

impl Default for InklingImageProcessorConfig {
    fn default() -> Self {
        Self {
            image_mean: default_mean(),
            image_std: default_std(),
            rescale_factor: default_rescale_factor(),
            rescale_image_frac: None,
            rescale_image_max_upscaled_long_edge: default_max_upscaled_edge(),
        }
    }
}

impl InklingImageProcessorConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.rescale_factor.is_finite() || self.rescale_factor <= 0.0 {
            return Err("Inkling rescale_factor must be finite and positive".into());
        }
        if let Some(frac) = self.rescale_image_frac
            && (!frac.is_finite() || frac <= 0.0)
        {
            return Err("Inkling rescale_image_frac must be finite and positive".into());
        }
        if self
            .image_mean
            .iter()
            .chain(&self.image_std)
            .any(|value| !value.is_finite())
            || self.image_std.iter().any(|value| *value <= 0.0)
        {
            return Err("Inkling image mean/std must be finite and std must be positive".into());
        }
        Ok(())
    }
}

pub struct InklingProcessedImages {
    pub pixel_values: UniquePtr<MlxArray>,
    pub tiles_per_image: Vec<usize>,
}

pub struct InklingImageProcessor {
    config: InklingImageProcessorConfig,
}

impl InklingImageProcessor {
    pub fn new(config: InklingImageProcessorConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { config })
    }

    fn maybe_resize(&self, image: &DynamicImage) -> DynamicImage {
        let Some(frac) = self.config.rescale_image_frac else {
            return image.clone();
        };
        let long_edge = image.width().max(image.height()) as f64;
        let mut target_long_edge = long_edge * frac;
        if let Some(maximum) = self.config.rescale_image_max_upscaled_long_edge {
            target_long_edge = target_long_edge.min(maximum.max(long_edge as usize) as f64);
        }
        let target_long_edge = target_long_edge.max(1.0);
        let ratio = target_long_edge / long_edge;
        let width = (image.width() as f64 * ratio + 0.5).floor().max(1.0) as u32;
        let height = (image.height() as f64 * ratio + 0.5).floor().max(1.0) as u32;
        image.resize_exact(width, height, FilterType::Lanczos3)
    }

    pub fn preprocess_with_counts(
        &self,
        images: &[DynamicImage],
    ) -> Result<InklingProcessedImages, String> {
        if images.is_empty() {
            return Err("Inkling image preprocessing requires at least one image".into());
        }
        let mut tiles_per_image = Vec::with_capacity(images.len());
        let mut values = Vec::new();
        for image in images {
            let image = self.maybe_resize(image).to_rgb8();
            let height = image.height() as usize;
            let width = image.width() as usize;
            if height == 0 || width == 0 {
                return Err("Inkling images must have positive width and height".into());
            }
            let rows = height.div_ceil(TILE_SIZE);
            let columns = width / TILE_SIZE + 1;
            let tile_count = rows
                .checked_mul(columns)
                .ok_or_else(|| "Inkling tile count overflowed".to_string())?;
            tiles_per_image.push(tile_count);
            let value_count = tile_count
                .checked_mul(TEMPORAL_SIZE * TILE_SIZE * TILE_SIZE * CHANNELS)
                .ok_or_else(|| "Inkling tile buffer size overflowed".to_string())?;
            values
                .try_reserve(value_count)
                .map_err(|error| format!("Failed to reserve Inkling tile buffer: {error}"))?;

            for row in 0..rows {
                for column in 0..columns {
                    let mut tile = vec![-1.0_f32; TILE_SIZE * TILE_SIZE * CHANNELS];
                    let tile_height = (height - row * TILE_SIZE).min(TILE_SIZE);
                    let tile_width = width.saturating_sub(column * TILE_SIZE).min(TILE_SIZE);
                    for y in 0..tile_height {
                        for x in 0..tile_width {
                            let pixel = image.get_pixel(
                                (column * TILE_SIZE + x) as u32,
                                (row * TILE_SIZE + y) as u32,
                            );
                            let base = (y * TILE_SIZE + x) * CHANNELS;
                            for channel in 0..CHANNELS {
                                tile[base + channel] = pixel[channel] as f32;
                            }
                        }
                    }
                    for (index, value) in tile.iter_mut().enumerate() {
                        let channel = index % CHANNELS;
                        *value = (*value * self.config.rescale_factor
                            - self.config.image_mean[channel])
                            / self.config.image_std[channel];
                    }
                    values.extend_from_slice(&tile);
                    values.extend_from_slice(&tile);
                }
            }
        }
        let tile_count = tiles_per_image
            .iter()
            .try_fold(0usize, |total, &count| total.checked_add(count));
        let tile_count = tile_count.ok_or_else(|| "Inkling tile count overflowed".to_string())?;
        let tile_count = i32::try_from(tile_count)
            .map_err(|_| "Inkling tile count exceeds the MLX i32 shape limit".to_string())?;
        Ok(InklingProcessedImages {
            pixel_values: mlxcel_core::from_slice_f32(&values, &[tile_count, 2, 40, 40, 3]),
            tiles_per_image,
        })
    }
}

impl Default for InklingImageProcessor {
    fn default() -> Self {
        Self::new(InklingImageProcessorConfig::default())
            .expect("the built-in Inkling image processor config is valid")
    }
}

impl ImageProcessor for InklingImageProcessor {
    fn preprocess(&self, images: &[DynamicImage]) -> UniquePtr<MlxArray> {
        self.preprocess_with_counts(images)
            .expect("Inkling image preprocessing failed")
            .pixel_values
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, RgbImage};
    use mlxcel_core::utils::array_to_vec_f32;

    use super::*;

    #[test]
    fn tiler_includes_exact_width_trailing_column() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(80, 100));
        let output = InklingImageProcessor::default()
            .preprocess_with_counts(&[image])
            .unwrap();
        assert_eq!(output.tiles_per_image, vec![9]);
        assert_eq!(
            mlxcel_core::array_shape(&output.pixel_values),
            vec![9, 2, 40, 40, 3]
        );
        mlxcel_core::eval(&output.pixel_values);
        let values = array_to_vec_f32(&output.pixel_values);
        let tile_len = TEMPORAL_SIZE * TILE_SIZE * TILE_SIZE * CHANNELS;
        let trailing = &values[2 * tile_len..3 * tile_len];
        for (index, value) in trailing.iter().enumerate() {
            let channel = index % CHANNELS;
            let expected =
                (-default_rescale_factor() - default_mean()[channel]) / default_std()[channel];
            assert!((value - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn each_image_reports_its_own_tile_count() {
        let output = InklingImageProcessor::default()
            .preprocess_with_counts(&[
                DynamicImage::ImageRgb8(RgbImage::new(39, 40)),
                DynamicImage::ImageRgb8(RgbImage::new(40, 41)),
            ])
            .unwrap();
        assert_eq!(output.tiles_per_image, vec![1, 4]);
    }
}
