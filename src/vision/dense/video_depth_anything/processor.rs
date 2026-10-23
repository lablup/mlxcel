// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact VDA aspect-ratio sizing and OpenCV-style float bicubic preprocessing.

use anyhow::{Result, ensure};
use image::{DynamicImage, GenericImageView};
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const MAX_PREPROCESSED_PIXELS: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct VideoDepthProcessor {
    pub input_size: usize,
    pub ensure_multiple_of: usize,
}

impl Default for VideoDepthProcessor {
    fn default() -> Self {
        Self {
            input_size: 518,
            ensure_multiple_of: 14,
        }
    }
}

fn round_ties_even(value: f64) -> usize {
    value.round_ties_even().max(0.0) as usize
}

impl VideoDepthProcessor {
    pub fn from_path(path: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct ProcessorConfig {
            #[serde(default = "default_input_size")]
            input_size: usize,
            #[serde(default = "default_multiple")]
            ensure_multiple_of: usize,
        }
        fn default_input_size() -> usize {
            518
        }
        fn default_multiple() -> usize {
            14
        }
        let config_path = path.join("preprocessor_config.json");
        if !config_path.is_file() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&config_path)?;
        let config: ProcessorConfig = serde_json::from_str(&raw)?;
        ensure!(
            config.input_size > 0 && config.input_size <= 4096,
            "VDA processor input_size must be between 1 and 4096"
        );
        ensure!(
            config.ensure_multiple_of > 0 && config.ensure_multiple_of <= 128,
            "VDA processor multiple must be between 1 and 128"
        );
        Ok(Self {
            input_size: config.input_size,
            ensure_multiple_of: config.ensure_multiple_of,
        })
    }

    pub fn target_size(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        ensure!(
            height > 0 && width > 0,
            "VDA source dimensions must be positive"
        );
        ensure!(
            self.input_size > 0 && self.ensure_multiple_of > 0,
            "VDA processor sizes must be positive"
        );
        let ratio = height.max(width) as f64 / height.min(width) as f64;
        let mut target = self.input_size;
        if ratio > 1.78 {
            target = (target as f64 * 1.777 / ratio) as usize;
            target = round_ties_even(target as f64 / self.ensure_multiple_of as f64)
                * self.ensure_multiple_of;
        }
        ensure!(
            target > 0,
            "VDA aspect-ratio adjustment produced an empty target"
        );
        let scale = (target as f64 / height as f64).max(target as f64 / width as f64);
        let new_h = (round_ties_even(height as f64 * scale / self.ensure_multiple_of as f64)
            * self.ensure_multiple_of)
            .max(target);
        let new_w = (round_ties_even(width as f64 * scale / self.ensure_multiple_of as f64)
            * self.ensure_multiple_of)
            .max(target);
        let pixels = new_h
            .checked_mul(new_w)
            .ok_or_else(|| anyhow::anyhow!("VDA target pixel count overflowed"))?;
        ensure!(
            pixels <= MAX_PREPROCESSED_PIXELS,
            "VDA preprocessed frame exceeds {MAX_PREPROCESSED_PIXELS} pixels"
        );
        Ok((new_h, new_w))
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<UniquePtr<MlxArray>> {
        let (width, height) = image.dimensions();
        ensure!(
            width > 0 && height > 0,
            "VDA source image must not be empty"
        );
        let source_pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| anyhow::anyhow!("VDA source pixel count overflowed"))?;
        ensure!(
            source_pixels <= 32 * 1024 * 1024,
            "VDA source frame exceeds 32 Mi pixels"
        );
        let rgb = image.to_rgb8();
        let (new_h, new_w) = self.target_size(height as usize, width as usize)?;
        let input: Vec<f32> = rgb
            .as_raw()
            .iter()
            .map(|&value| value as f32 / 255.0)
            .collect();
        let mut resized =
            resize_cubic_f32(&input, height as usize, width as usize, new_h, new_w, 3);
        for pixel in resized.chunks_exact_mut(3) {
            for channel in 0..3 {
                pixel[channel] = (pixel[channel] - MEAN[channel]) / STD[channel];
            }
        }
        Ok(mlxcel_core::from_slice_f32(
            &resized,
            &[new_h as i32, new_w as i32, 3],
        ))
    }
}

fn cubic_coefficients(fraction: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let x = fraction;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let y = 1.0 - x;
    let c2 = ((A + 2.0) * y - (A + 3.0)) * y * y + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

fn axis_table(input: usize, output: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = input as f64 / output as f64;
    (0..output)
        .map(|destination| {
            let source = (destination as f64 + 0.5) * scale - 0.5;
            let base = source.floor() as isize;
            let indices = [-1isize, 0, 1, 2]
                .map(|offset| (base + offset).clamp(0, input as isize - 1) as usize);
            (indices, cubic_coefficients((source - base as f64) as f32))
        })
        .collect()
}

fn resize_cubic_f32(
    input: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    channels: usize,
) -> Vec<f32> {
    if (in_h, in_w) == (out_h, out_w) {
        return input.to_vec();
    }
    let horizontal = axis_table(in_w, out_w);
    let vertical = axis_table(in_h, out_h);
    let mut rows = vec![0.0f32; in_h * out_w * channels];
    for y in 0..in_h {
        for (x, (indices, coeffs)) in horizontal.iter().enumerate() {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for tap in 0..4 {
                    sum += input[(y * in_w + indices[tap]) * channels + c] * coeffs[tap];
                }
                rows[(y * out_w + x) * channels + c] = sum;
            }
        }
    }
    let mut output = vec![0.0f32; out_h * out_w * channels];
    for (y, (indices, coeffs)) in vertical.iter().enumerate() {
        for x in 0..out_w {
            for c in 0..channels {
                let mut sum = 0.0f32;
                for tap in 0..4 {
                    sum += rows[(indices[tap] * out_w + x) * channels + c] * coeffs[tap];
                }
                output[(y * out_w + x) * channels + c] = sum;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn published_target_sizes_match_python() {
        let p = VideoDepthProcessor::default();
        assert_eq!(p.target_size(1080, 1920).unwrap(), (518, 924));
        assert_eq!(p.target_size(1920, 1080).unwrap(), (924, 518));
        assert_eq!(p.target_size(480, 640).unwrap(), (518, 686));
        assert_eq!(p.target_size(100, 400).unwrap(), (224, 896));
    }
    #[test]
    fn python_rounding_is_ties_to_even() {
        assert_eq!(round_ties_even(2.5), 2);
        assert_eq!(round_ties_even(3.5), 4);
    }

    #[test]
    fn cubic_float_resize_matches_opencv_literal() {
        let input: Vec<f32> = (0..6).map(|value| value as f32 / 5.0).collect();
        let actual = resize_cubic_f32(&input, 2, 3, 4, 5, 1);
        let expected = [
            -0.08248132,
            -0.00008124532,
            0.13671875,
            0.27351886,
            0.35591877,
            0.116737455,
            0.19913752,
            0.3359375,
            0.47273764,
            0.5551375,
            0.44486243,
            0.52726257,
            0.66406244,
            0.8008626,
            0.88326234,
            0.64408123,
            0.7264813,
            0.86328125,
            1.0000813,
            1.0824811,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
}
