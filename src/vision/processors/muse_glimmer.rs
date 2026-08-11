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

//! Muse Glimmer image processor.
//!
//! The checkpoint processor resizes on the `patch_size * merge_size` grid,
//! converts to RGB, applies Lanczos resize, normalizes with `(x / 255 - 0.5) /
//! 0.5`, and flattens one static image into spatial patch rows whose feature
//! dimension contains two identical temporal frames.

use std::path::Path;

use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::ImageProcessor;
use crate::models::MuseGlimmerVisionConfig;

pub const DEFAULT_MAX_IMAGE_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
pub struct MuseGlimmerImageProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub max_image_tokens: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale_factor: f32,
    pub do_convert_rgb: bool,
    pub do_resize: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
}

impl MuseGlimmerImageProcessor {
    pub fn from_vision_config(config: &MuseGlimmerVisionConfig) -> Self {
        Self {
            patch_size: config.patch_size,
            temporal_patch_size: config.patch_temporal,
            merge_size: config.merge_size,
            max_image_tokens: DEFAULT_MAX_IMAGE_TOKENS,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            rescale_factor: 1.0 / 255.0,
            do_convert_rgb: true,
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
        }
    }

    pub fn from_model_dir(
        model_path: &Path,
        config: &MuseGlimmerVisionConfig,
    ) -> Result<Self, String> {
        let mut processor = Self::from_vision_config(config);
        let path = model_path.join("processor_config.json");
        if !path.exists() {
            return Ok(processor);
        }
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read processor_config.json: {e}"))?;
        let parsed: RawProcessorConfig = serde_json::from_str(&raw)
            .map_err(|e| format!("Failed to parse processor_config.json: {e}"))?;
        if let Some(image) = parsed.image_processor {
            processor.apply_raw(image)?;
        }
        Ok(processor)
    }

    fn apply_raw(&mut self, raw: RawImageProcessor) -> Result<(), String> {
        self.patch_size = raw.patch_size.unwrap_or(self.patch_size);
        self.temporal_patch_size = raw.temporal_patch_size.unwrap_or(self.temporal_patch_size);
        self.merge_size = raw.merge_size.unwrap_or(self.merge_size);
        self.max_image_tokens = raw.max_image_tokens.unwrap_or(self.max_image_tokens);
        self.mean = vec3(raw.image_mean, self.mean, "image_mean")?;
        self.std = vec3(raw.image_std, self.std, "image_std")?;
        self.rescale_factor = raw.rescale_factor.unwrap_or(self.rescale_factor);
        self.do_convert_rgb = raw.do_convert_rgb.unwrap_or(self.do_convert_rgb);
        self.do_resize = raw.do_resize.unwrap_or(self.do_resize);
        self.do_rescale = raw.do_rescale.unwrap_or(self.do_rescale);
        self.do_normalize = raw.do_normalize.unwrap_or(self.do_normalize);
        if raw.resample.is_some_and(|value| value != 1) {
            return Err("Muse Glimmer processor requires Lanczos resample=1".to_string());
        }
        if self.patch_size == 0 || self.temporal_patch_size == 0 || self.merge_size == 0 {
            return Err(
                "Muse Glimmer processor patch and merge sizes must be non-zero".to_string(),
            );
        }
        if self.max_image_tokens == 0 {
            return Err("Muse Glimmer max_image_tokens must be non-zero".to_string());
        }
        if self.std.contains(&0.0) {
            return Err("Muse Glimmer image_std entries must be non-zero".to_string());
        }
        Ok(())
    }

    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    pub fn min_pixels(&self) -> usize {
        self.factor() * self.factor()
    }

    pub fn max_pixels(&self) -> usize {
        self.max_image_tokens * self.factor() * self.factor()
    }

    pub fn smart_resize(&self, height: u32, width: u32) -> (u32, u32) {
        smart_resize(height, width, self.factor() as u32, self.max_image_tokens)
    }

    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|image| {
                let (height, width) = self.smart_resize(image.height(), image.width());
                (
                    1,
                    (height as usize / self.patch_size) as i32,
                    (width as usize / self.patch_size) as i32,
                )
            })
            .collect()
    }

    pub fn visual_token_count(&self, grid: (i32, i32, i32)) -> Result<usize, String> {
        let count = merged_visual_token_count(grid, self.merge_size)?;
        if count > self.max_image_tokens {
            return Err(format!(
                "Muse Glimmer image emits {count} visual tokens, above cap {}",
                self.max_image_tokens
            ));
        }
        Ok(count)
    }

    pub fn preprocess_values_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (Vec<f32>, Vec<(i32, i32, i32)>) {
        let grid_thw = self.compute_grid_thw(images);
        let patch_dim = self.temporal_patch_size * 3 * self.patch_size * self.patch_size;
        let total_patches: usize = grid_thw
            .iter()
            .map(|&(_, h, w)| h as usize * w as usize)
            .sum();
        let mut all = Vec::with_capacity(total_patches * patch_dim);

        for (image, &(_, h_patches, w_patches)) in images.iter().zip(&grid_thw) {
            let target_h = h_patches as usize * self.patch_size;
            let target_w = w_patches as usize * self.patch_size;
            let resized = if self.do_resize {
                image.resize_exact(target_w as u32, target_h as u32, FilterType::Lanczos3)
            } else {
                image.clone()
            };
            let rgb = if self.do_convert_rgb {
                resized.to_rgb8()
            } else {
                image::DynamicImage::ImageRgb8(resized.to_rgb8()).to_rgb8()
            };
            let mut chw = vec![0f32; 3 * target_h * target_w];
            for y in 0..target_h {
                for x in 0..target_w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let mut value = pixel[c] as f32;
                        if self.do_rescale {
                            value *= self.rescale_factor;
                        }
                        if self.do_normalize {
                            value = (value - self.mean[c]) / self.std[c];
                        }
                        chw[c * target_h * target_w + y * target_w + x] = value;
                    }
                }
            }
            for py in 0..h_patches as usize {
                for px in 0..w_patches as usize {
                    let y0 = py * self.patch_size;
                    let x0 = px * self.patch_size;
                    for _ in 0..self.temporal_patch_size {
                        for c in 0..3 {
                            for dy in 0..self.patch_size {
                                for dx in 0..self.patch_size {
                                    let y = y0 + dy;
                                    let x = x0 + dx;
                                    all.push(chw[c * target_h * target_w + y * target_w + x]);
                                }
                            }
                        }
                    }
                }
            }
        }
        (all, grid_thw)
    }

    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let (values, grid_thw) = self.preprocess_values_with_grid(images);
        let patch_dim = (self.temporal_patch_size * 3 * self.patch_size * self.patch_size) as i32;
        let rows = (values.len() as i32) / patch_dim;
        (
            mlxcel_core::from_slice_f32(&values, &[rows, patch_dim]),
            grid_thw,
        )
    }
}

pub fn smart_resize(height: u32, width: u32, factor: u32, max_tokens: usize) -> (u32, u32) {
    let factor = factor.max(1);
    let ratio = height.max(1) as f64 / width.max(1) as f64;
    let ideal_h = height.max(1) as f64 / factor as f64;
    let ideal_w = width.max(1) as f64 / factor as f64;
    let mut grid_h = ideal_h.round().max(1.0) as usize;
    let mut grid_w = ideal_w.round().max(1.0) as usize;
    if grid_h * grid_w > max_tokens {
        let scale = (max_tokens as f64 / (ideal_h * ideal_w)).sqrt();
        let h_scaled = (ideal_h * scale).max(1.0);
        let w_scaled = (ideal_w * scale).max(1.0);
        let mut candidates = Vec::new();
        for h in [h_scaled.floor(), h_scaled.ceil()] {
            for w in [w_scaled.floor(), w_scaled.ceil()] {
                let h = h.max(1.0) as usize;
                let w = w.max(1.0) as usize;
                if h * w <= max_tokens {
                    candidates.push((h, w));
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        let best = candidates
            .into_iter()
            .min_by(|(ah, aw), (bh, bw)| {
                let a = ((*ah as f64 / *aw as f64) - ratio).abs();
                let b = ((*bh as f64 / *bw as f64) - ratio).abs();
                a.total_cmp(&b)
            })
            .unwrap_or_else(|| {
                let h = h_scaled.floor().max(1.0) as usize;
                let w = (max_tokens / h).max(1);
                (h, w)
            });
        grid_h = best.0;
        grid_w = best.1;
    }
    (grid_h as u32 * factor, grid_w as u32 * factor)
}

pub fn merged_visual_token_count(
    grid: (i32, i32, i32),
    merge_size: usize,
) -> Result<usize, String> {
    let (t, h, w) = grid;
    if t <= 0 || h <= 0 || w <= 0 {
        return Err(format!("Muse Glimmer grid must be positive, got {grid:?}"));
    }
    let merge = merge_size as i32;
    if merge <= 0 || h % merge != 0 || w % merge != 0 {
        return Err(format!(
            "Muse Glimmer grid {grid:?} is not divisible by merge_size {merge_size}"
        ));
    }
    Ok((t as usize) * (h as usize / merge_size) * (w as usize / merge_size))
}

fn vec3(value: Option<Vec<f32>>, fallback: [f32; 3], label: &str) -> Result<[f32; 3], String> {
    match value {
        Some(v) if v.len() == 3 => Ok([v[0], v[1], v[2]]),
        Some(v) => Err(format!(
            "Muse Glimmer {label} must have 3 entries, got {}",
            v.len()
        )),
        None => Ok(fallback),
    }
}

#[derive(Debug, Deserialize)]
struct RawProcessorConfig {
    image_processor: Option<RawImageProcessor>,
}

#[derive(Debug, Deserialize)]
struct RawImageProcessor {
    patch_size: Option<usize>,
    temporal_patch_size: Option<usize>,
    merge_size: Option<usize>,
    max_image_tokens: Option<usize>,
    image_mean: Option<Vec<f32>>,
    image_std: Option<Vec<f32>>,
    rescale_factor: Option<f32>,
    resample: Option<i32>,
    do_convert_rgb: Option<bool>,
    do_resize: Option<bool>,
    do_rescale: Option<bool>,
    do_normalize: Option<bool>,
}

impl ImageProcessor for MuseGlimmerImageProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        self.preprocess_with_grid(images).0
    }
}

#[cfg(test)]
#[path = "muse_glimmer_tests.rs"]
mod tests;
