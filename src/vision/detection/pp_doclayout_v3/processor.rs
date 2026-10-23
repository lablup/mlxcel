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

//! PP-DocLayoutV3 image preprocessing.
//!
//! Hugging Face's processor sends CPU `u8` images through PyTorch's native
//! bicubic kernel with `align_corners=false` and `antialias=false`. That path
//! is separable, uses the OpenCV-compatible cubic coefficient `a=-0.75`,
//! quantizes each pass with signed 16-bit weights, and rounds back to `u8`
//! between the horizontal and vertical passes. A generic floating-point or
//! Pillow resize is observably different on document edges.

use std::io::Read;
use std::path::Path;

use image::{DynamicImage, GenericImageView, RgbImage};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

const DEFAULT_IMAGE_SIZE: usize = 800;
const DEFAULT_RESCALE: f64 = 1.0 / 255.0;
const MAX_SOURCE_PIXELS: u64 = 32 * 1024 * 1024;
const MAX_IMAGE_SIZE: usize = 1024;
const MAX_RESIZE_VALUES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub height: usize,
    pub width: usize,
    pub rescale_factor: f64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            height: DEFAULT_IMAGE_SIZE,
            width: DEFAULT_IMAGE_SIZE,
            rescale_factor: DEFAULT_RESCALE,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawSize {
    height: usize,
    width: usize,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct RawPreprocessor {
    do_resize: bool,
    size: Option<RawSize>,
    resample: u8,
    do_rescale: bool,
    rescale_factor: f64,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

impl Default for RawPreprocessor {
    fn default() -> Self {
        Self {
            do_resize: true,
            size: None,
            resample: 3,
            do_rescale: true,
            rescale_factor: DEFAULT_RESCALE,
            do_normalize: true,
            image_mean: vec![0.0; 3],
            image_std: vec![1.0; 3],
        }
    }
}

impl ProcessorConfig {
    fn validate(&self) -> Result<(), String> {
        if self.height == 0
            || self.width == 0
            || self.height > MAX_IMAGE_SIZE
            || self.width > MAX_IMAGE_SIZE
        {
            return Err(format!(
                "PP-DocLayoutV3 processor size must be in 1..={MAX_IMAGE_SIZE}, got {}x{}",
                self.height, self.width
            ));
        }
        if !self.rescale_factor.is_finite() || self.rescale_factor <= 0.0 {
            return Err(format!(
                "PP-DocLayoutV3 rescale_factor must be finite and positive, got {}",
                self.rescale_factor
            ));
        }
        if self.image_mean.iter().any(|value| !value.is_finite())
            || self
                .image_std
                .iter()
                .any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(
                "PP-DocLayoutV3 normalization values must be finite with nonzero std".into(),
            );
        }
        Ok(())
    }

    pub fn from_pretrained<P: AsRef<Path>>(directory: P) -> Result<Self, String> {
        let path = directory.as_ref().join("preprocessor_config.json");
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let raw: RawPreprocessor = serde_json::from_str(&text)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        if !raw.do_resize || raw.resample != 3 || !raw.do_rescale || !raw.do_normalize {
            return Err(format!(
                "{} must enable bicubic resize (resample 3), rescaling, and normalization",
                path.display()
            ));
        }
        let size = raw
            .size
            .ok_or_else(|| format!("{} must declare size.height and size.width", path.display()))?;
        if size.height == 0
            || size.width == 0
            || size.height > MAX_IMAGE_SIZE
            || size.width > MAX_IMAGE_SIZE
        {
            return Err(format!(
                "PP-DocLayoutV3 processor size must be in 1..={MAX_IMAGE_SIZE}, got {}x{}",
                size.height, size.width
            ));
        }
        let image_mean = three_finite("image_mean", raw.image_mean)?;
        let image_std = three_finite("image_std", raw.image_std)?;
        if image_std.contains(&0.0) {
            return Err("PP-DocLayoutV3 image_std entries must be nonzero".into());
        }
        if !raw.rescale_factor.is_finite() || raw.rescale_factor <= 0.0 {
            return Err(format!(
                "PP-DocLayoutV3 rescale_factor must be finite and positive, got {}",
                raw.rescale_factor
            ));
        }
        let config = Self {
            height: size.height,
            width: size.width,
            rescale_factor: raw.rescale_factor,
            image_mean,
            image_std,
        };
        config.validate()?;
        Ok(config)
    }
}

fn three_finite(name: &str, values: Vec<f32>) -> Result<[f32; 3], String> {
    let values: [f32; 3] = values.try_into().map_err(|values: Vec<f32>| {
        format!(
            "PP-DocLayoutV3 {name} must have 3 entries, got {}",
            values.len()
        )
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "PP-DocLayoutV3 {name} must contain only finite values"
        ));
    }
    Ok(values)
}

pub struct ProcessedImage {
    pub pixel_values: UniquePtr<MlxArray>,
    pub original_size: (u32, u32),
}

pub struct PpDocLayoutV3Processor {
    config: ProcessorConfig,
}

impl PpDocLayoutV3Processor {
    pub fn new(config: ProcessorConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn from_pretrained<P: AsRef<Path>>(directory: P) -> Result<Self, String> {
        Self::new(ProcessorConfig::from_pretrained(directory)?)
    }

    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    pub fn process_path<P: AsRef<Path>>(&self, path: P) -> Result<ProcessedImage, String> {
        let path = path.as_ref();
        let limits = crate::current_image_input_limits();
        let file = std::fs::File::open(path)
            .map_err(|error| format!("failed to open image {}: {error}", path.display()))?;
        let mut bytes = Vec::new();
        file.take(limits.max_payload_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("failed to read image {}: {error}", path.display()))?;
        if bytes.len() > limits.max_payload_bytes {
            return Err(format!(
                "image {} exceeds the {} byte payload limit",
                path.display(),
                limits.max_payload_bytes
            ));
        }
        let mut images = crate::decode_image_payloads_with_limits(&[bytes], limits)
            .map_err(|error| format!("failed to decode image {}: {error}", path.display()))?;
        let image = images
            .pop()
            .ok_or_else(|| format!("image decoder returned no image for {}", path.display()))?;
        self.process_image(&image)
    }

    pub fn process_image(&self, image: &DynamicImage) -> Result<ProcessedImage, String> {
        let (width, height) = image.dimensions();
        let source_pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| "PP-DocLayoutV3 source pixel count overflow".to_string())?;
        if source_pixels == 0 || source_pixels > MAX_SOURCE_PIXELS {
            return Err(format!(
                "PP-DocLayoutV3 source has {source_pixels} pixels; limit is {MAX_SOURCE_PIXELS}"
            ));
        }
        checked_values(height as usize, self.config.width, "horizontal resize")?;
        checked_values(self.config.height, self.config.width, "processor output")?;

        let rgb = image.to_rgb8();
        let resized = resize_u8_bicubic(&rgb, self.config.width, self.config.height)?;
        // Transformers fuses rescaling into normalization by first computing
        // `1.0 / rescale_factor` as a Python f64, then casting that product to
        // the f32 normalization tensor. For the released checkpoints this is
        // exactly 255.0; multiplying pixels by the factor rounded to f32 is one
        // ulp different for 126 of the 256 possible channel values.
        let inverse_rescale = (1.0 / self.config.rescale_factor) as f32;
        let fused_mean = self.config.image_mean.map(|value| value * inverse_rescale);
        let fused_std = self.config.image_std.map(|value| value * inverse_rescale);
        let mut values = Vec::with_capacity(self.config.height * self.config.width * 3);
        for pixel in resized.pixels() {
            for channel in 0..3 {
                values.push((f32::from(pixel[channel]) - fused_mean[channel]) / fused_std[channel]);
            }
        }
        let pixel_values = mlxcel_core::from_slice_f32(
            &values,
            &[1, self.config.height as i32, self.config.width as i32, 3],
        );
        Ok(ProcessedImage {
            pixel_values,
            original_size: (width, height),
        })
    }
}

pub(super) fn checked_values(height: usize, width: usize, context: &str) -> Result<usize, String> {
    let values = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| format!("PP-DocLayoutV3 {context} size overflow"))?;
    if values > MAX_RESIZE_VALUES {
        return Err(format!(
            "PP-DocLayoutV3 {context} needs {values} values; limit is {MAX_RESIZE_VALUES}"
        ));
    }
    Ok(values)
}

#[derive(Clone)]
struct Coefficients {
    start: usize,
    count: usize,
    weights: [i16; 4],
}

fn resize_u8_bicubic(source: &RgbImage, width: usize, height: usize) -> Result<RgbImage, String> {
    if source.width() as usize == width && source.height() as usize == height {
        return Ok(source.clone());
    }
    let (horizontal, horizontal_precision) = coefficient_table(source.width() as usize, width)?;
    let (vertical, vertical_precision) = coefficient_table(source.height() as usize, height)?;
    let intermediate_len = checked_values(source.height() as usize, width, "resize intermediate")?;
    let mut intermediate = vec![0_u8; intermediate_len];
    let source_bytes = source.as_raw();
    for y in 0..source.height() as usize {
        for (x, coefficients) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let value = interpolate(coefficients, horizontal_precision, |source_x| {
                    source_bytes[(y * source.width() as usize + source_x) * 3 + channel]
                });
                intermediate[(y * width + x) * 3 + channel] = value;
            }
        }
    }
    let output_len = checked_values(height, width, "resize output")?;
    let mut output = vec![0_u8; output_len];
    for (y, coefficients) in vertical.iter().enumerate() {
        for x in 0..width {
            for channel in 0..3 {
                output[(y * width + x) * 3 + channel] =
                    interpolate(coefficients, vertical_precision, |source_y| {
                        intermediate[(source_y * width + x) * 3 + channel]
                    });
            }
        }
    }
    RgbImage::from_raw(width as u32, height as u32, output)
        .ok_or_else(|| "failed to construct PP-DocLayoutV3 resized image".to_string())
}

fn coefficient_table(input: usize, output: usize) -> Result<(Vec<Coefficients>, u32), String> {
    if input == 0 || output == 0 {
        return Err("PP-DocLayoutV3 resize dimensions must be positive".into());
    }
    let scale = input as f64 / output as f64;
    let mut rows = Vec::with_capacity(output);
    let mut maximum = 0.0_f64;
    for destination in 0..output {
        let source = scale * (destination as f64 + 0.5) - 0.5;
        let input_index = (source as f32).floor() as isize;
        let lambda = (source - input_index as f64).clamp(0.0, 1.0);
        let unbound_min = input_index - 1;
        let unbound_max = input_index + 3;
        let start = unbound_min.max(0) as usize;
        let count = (unbound_max.min(input as isize) - start as isize).clamp(0, 4) as usize;
        let mut weights = [0.0_f64; 4];
        let mut weight_index = 0_usize;
        for offset in 0..4 {
            let weight = cubic_filter(offset as f64 - 1.0 - lambda);
            if unbound_min + offset as isize <= 0 {
                weight_index = 0;
            } else if unbound_min + offset as isize >= input as isize - 1 {
                weight_index = count - 1;
            }
            weights[weight_index] += weight;
            maximum = maximum.max(weights[weight_index]);
            weight_index += 1;
        }
        rows.push((start, count, weights));
    }
    let precision = (0..22)
        .find(|&precision| (0.5 + maximum * f64::from(1_u32 << (precision + 1))) as i32 >= 1 << 15)
        .unwrap_or(22);
    let factor = f64::from(1_u32 << precision);
    let converted = rows
        .into_iter()
        .map(|(start, count, weights)| Coefficients {
            start,
            count,
            weights: weights.map(|weight| {
                let scaled = weight * factor;
                if scaled < 0.0 {
                    (scaled - 0.5) as i16
                } else {
                    (scaled + 0.5) as i16
                }
            }),
        })
        .collect();
    Ok((converted, precision))
}

fn cubic_filter(value: f64) -> f64 {
    let value = value.abs();
    let a = -0.75;
    if value < 1.0 {
        ((a + 2.0) * value - (a + 3.0)) * value * value + 1.0
    } else if value < 2.0 {
        ((a * value - 5.0 * a) * value + 8.0 * a) * value - 4.0 * a
    } else {
        0.0
    }
}

fn interpolate<F>(coefficients: &Coefficients, precision: u32, mut sample: F) -> u8
where
    F: FnMut(usize) -> u8,
{
    let mut value = 1_i64 << (precision - 1);
    for offset in 0..coefficients.count {
        value += i64::from(sample(coefficients.start + offset))
            * i64::from(coefficients.weights[offset]);
    }
    (value >> precision).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize_preserves_bytes() {
        let source = RgbImage::from_raw(2, 2, (0_u8..12).collect()).unwrap();
        assert_eq!(resize_u8_bicubic(&source, 2, 2).unwrap(), source);
    }

    #[test]
    fn uint8_bicubic_matches_pinned_torch_small_case() {
        let source = RgbImage::from_raw(2, 3, (0_u8..18).collect()).unwrap();
        let actual = resize_u8_bicubic(&source, 5, 4).unwrap();
        assert_eq!(
            actual.as_raw(),
            &[
                0, 1, 2, 0, 1, 2, 2, 3, 4, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 5, 6, 7, 6, 7, 8, 6,
                7, 8, 9, 10, 11, 9, 10, 11, 11, 12, 13, 12, 13, 14, 12, 13, 14, 12, 13, 14, 12, 13,
                14, 14, 15, 16, 15, 16, 17, 15, 16, 17
            ]
        );
    }

    #[test]
    fn fused_rescale_matches_transformers_division_order_exactly() {
        let config = ProcessorConfig::default();
        let inverse_rescale = (1.0 / config.rescale_factor) as f32;
        assert_eq!(inverse_rescale.to_bits(), 255.0_f32.to_bits());
        for pixel in 0_u8..=u8::MAX {
            let actual = (f32::from(pixel) - config.image_mean[0] * inverse_rescale)
                / (config.image_std[0] * inverse_rescale);
            assert_eq!(actual.to_bits(), (f32::from(pixel) / 255.0).to_bits());
        }
    }

    #[test]
    fn config_rejects_non_bicubic_processor() {
        let raw: RawPreprocessor = serde_json::from_str(r#"{"resample":2}"#).unwrap();
        assert_eq!(raw.resample, 2);
    }
}
