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

//! Nemotron H Nano Omni image preprocessor.
//!
//! Faithful Rust port of
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/image_processing_nemotron_h_nano_omni.py.
//! Produces channel-first `[1, 3, H, W]` float32 tensors normalized with the
//! checkpoint-supplied mean/std, plus the pre-downsample patch grid that
//! the multimodal projector uses to compute per-image token counts.
//!
//! The dynamic-resolution math mirrors NVIDIA's reference and vLLM's
//! tiler: each image is fit into a `(target_w_patches, target_h_patches)`
//! grid bounded by `min_num_patches` and `max_num_patches`, divisible by
//! `1 / downsample_ratio`. Resize uses bicubic resampling to match the
//! Python reference.
//!
//! Used by: Nemotron H Nano Omni VLM

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// Preprocessed output for a single image.
///
/// `pixel_values` is the channel-first tensor consumed by the RADIO
/// vision tower. `patch_grid` is `(patch_h, patch_w)` BEFORE the
/// projector's pixel-shuffle downsample; `num_tokens` is the post-
/// downsample count emitted into the text token stream (one slot per
/// surviving patch after `pixel_shuffle(scale=downsample_ratio)`).
pub struct NemotronHNanoOmniImageInput {
    pub pixel_values: UniquePtr<MlxArray>,
    pub patch_grid: (usize, usize),
    pub num_tokens: usize,
}

/// Configuration mirroring upstream `NemotronHNanoOmniImageProcessor.__init__`.
#[derive(Debug, Clone)]
pub struct NemotronHNanoOmniProcessorConfig {
    pub norm_mean: [f32; 3],
    pub norm_std: [f32; 3],
    pub patch_size: usize,
    pub downsample_ratio: f32,
    pub min_num_patches: usize,
    pub max_num_patches: usize,
    pub max_model_len: usize,
}

impl Default for NemotronHNanoOmniProcessorConfig {
    fn default() -> Self {
        // Released checkpoint: `image_processor.preprocessor_config.json`.
        // Values rounded to f32 precision (~7 significant digits) since
        // they are stored as JSON floats in the upstream config.
        Self {
            norm_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            norm_std: [0.268_629_54, 0.261_302_57, 0.275_777_1],
            patch_size: 16,
            downsample_ratio: 0.5,
            min_num_patches: 1024,
            max_num_patches: 13312,
            max_model_len: 16384,
        }
    }
}

/// Top-level image preprocessor.
pub struct NemotronHNanoOmniImageProcessor {
    pub config: NemotronHNanoOmniProcessorConfig,
}

impl NemotronHNanoOmniImageProcessor {
    pub fn new(config: NemotronHNanoOmniProcessorConfig) -> Self {
        Self { config }
    }

    /// Convenience constructor matching the upstream `from_pretrained`
    /// defaults — no `preprocessor_config.json` overrides applied.
    pub fn with_defaults() -> Self {
        Self::new(NemotronHNanoOmniProcessorConfig::default())
    }

    fn downsample_factor(&self) -> usize {
        // upstream: `int(round(1.0 / downsample_ratio))`. Defaults to 2.
        let raw = (1.0 / self.config.downsample_ratio.max(f32::EPSILON)).round() as i64;
        raw.max(1) as usize
    }

    /// Preprocess a batch of images. Returns one
    /// [`NemotronHNanoOmniImageInput`] per input image, in input order.
    pub fn preprocess_batch(&self, images: &[DynamicImage]) -> Vec<NemotronHNanoOmniImageInput> {
        if images.is_empty() {
            return Vec::new();
        }

        // Per-image budget mirrors upstream `_preprocess`:
        //   tokens_available = max_model_len - 4
        //   budget = tokens_available * downsample_factor**2
        //   budget = max(budget, min_num_patches * len(images))
        //   per_image_budget = max(min(budget, max_num_patches), min_num_patches)
        let downsample_factor = self.downsample_factor();
        let tokens_available = self.config.max_model_len.saturating_sub(4);
        let mut budget = tokens_available.saturating_mul(downsample_factor * downsample_factor);
        let min_total = self.config.min_num_patches.saturating_mul(images.len());
        if budget < min_total {
            budget = min_total;
        }
        let max_budget = if self.config.max_num_patches > 0 {
            self.config.max_num_patches
        } else {
            usize::MAX
        };
        let per_image_budget = budget.min(max_budget).max(self.config.min_num_patches);

        images
            .iter()
            .map(|image| self.preprocess_single(image, per_image_budget))
            .collect()
    }

    fn preprocess_single(
        &self,
        image: &DynamicImage,
        tokens_for_media: usize,
    ) -> NemotronHNanoOmniImageInput {
        let rgb = image.to_rgb8();
        let orig_h = rgb.height() as usize;
        let orig_w = rgb.width() as usize;
        let (target_w_patches, target_h_patches) =
            self.compute_target_patches(orig_h, orig_w, tokens_for_media);

        let target_h = target_h_patches * self.config.patch_size;
        let target_w = target_w_patches * self.config.patch_size;

        let resized = if orig_h == target_h && orig_w == target_w {
            rgb
        } else {
            // PIL bicubic resampling for parity with upstream.
            DynamicImage::ImageRgb8(rgb)
                .resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom)
                .to_rgb8()
        };

        // Channel-first packing, normalize to `(arr / 255 - mean) / std`.
        let channels = 3usize;
        let mut data = vec![0.0f32; channels * target_h * target_w];
        let inv_255 = 1.0_f32 / 255.0;
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..channels {
                    let value = pixel[c] as f32 * inv_255;
                    let dst = c * target_h * target_w + y * target_w + x;
                    data[dst] = (value - self.config.norm_mean[c]) / self.config.norm_std[c];
                }
            }
        }

        let downsample_factor = self.downsample_factor();
        let num_tokens =
            (target_w_patches * target_h_patches) / (downsample_factor * downsample_factor);

        let pixel_values = mlxcel_core::from_slice_f32(
            &data,
            &[1, channels as i32, target_h as i32, target_w as i32],
        );

        NemotronHNanoOmniImageInput {
            pixel_values,
            patch_grid: (target_h_patches, target_w_patches),
            num_tokens,
        }
    }

    /// Mirror of upstream `_compute_target_patches`. Returns
    /// `(target_w_patches, target_h_patches)` for parity with the
    /// upstream tuple ordering.
    pub fn compute_target_patches(
        &self,
        orig_h: usize,
        orig_w: usize,
        tokens_available: usize,
    ) -> (usize, usize) {
        let patch = self.config.patch_size as f64;
        // Upstream uses `round(orig_h / patch + 0.5)` for "ceil-ish" rounding.
        let closest_patch_h = ((orig_h as f64 / patch + 0.5).round()) as usize;
        let closest_patch_w = ((orig_w as f64 / patch + 0.5).round()) as usize;
        let patches = (closest_patch_h * closest_patch_w).max(1);

        let factor = ((tokens_available as f64) / (patches as f64))
            .sqrt()
            .min(1.0);
        let mut target_h = (factor * (closest_patch_h as f64)).floor() as usize;
        let mut target_w = (factor * (closest_patch_w as f64)).floor() as usize;
        target_h = target_h.max(1);
        target_w = target_w.max(1);

        if tokens_available > self.config.min_num_patches
            && target_h * target_w < self.config.min_num_patches
        {
            let denom = (target_h * target_w).max(1);
            let up = ((self.config.min_num_patches as f64) / (denom as f64)).sqrt();
            target_h = ((up * (target_h as f64)).ceil()) as usize;
            target_w = ((up * (target_w as f64)).ceil()) as usize;
        }

        let divisor = self.downsample_factor();
        // Pad / shrink to a multiple of `divisor`, preferring growth when
        // it stays inside `tokens_available`.
        if divisor > 0 {
            let rem_h = target_h % divisor;
            if rem_h != 0 {
                let inc_h = divisor - rem_h;
                if (target_h + inc_h) * target_w <= tokens_available {
                    target_h += inc_h;
                } else {
                    target_h = target_h.saturating_sub(rem_h);
                    target_h = target_h.max(divisor);
                }
            }
            let rem_w = target_w % divisor;
            if rem_w != 0 {
                let inc_w = divisor - rem_w;
                if target_h * (target_w + inc_w) <= tokens_available {
                    target_w += inc_w;
                } else {
                    target_w = target_w.saturating_sub(rem_w);
                    target_w = target_w.max(divisor);
                }
            }
        }

        (target_w.max(1), target_h.max(1))
    }
}

impl super::ImageProcessor for NemotronHNanoOmniImageProcessor {
    /// Single-image fast path conforming to the generic `ImageProcessor`
    /// trait. Preserves the channel-first `[1, 3, H, W]` layout that the
    /// vision tower expects; callers that need per-image metadata
    /// (`patch_grid`, `num_tokens`) should call [`Self::preprocess`]
    /// directly instead.
    fn preprocess(&self, images: &[DynamicImage]) -> UniquePtr<MlxArray> {
        let processed = self.preprocess_batch(images);
        if processed.is_empty() {
            // Defensive: empty batch still needs to satisfy the trait
            // contract. Returns a `[0, 3, 0, 0]` placeholder to mirror
            // PIL's behavior; downstream callers always ignore this when
            // there are no images.
            return mlxcel_core::zeros(&[0, 3, 0, 0], mlxcel_core::dtype::FLOAT32);
        }
        // The trait contract returns a single tensor; for callers that
        // reach this path we hand back the first image's tensor. The
        // batched VLM path uses [`Self::preprocess`] directly.
        mlxcel_core::copy(processed[0].pixel_values.as_ref().unwrap())
    }
}

#[cfg(test)]
#[path = "nemotron_h_nano_omni_tests.rs"]
mod tests;
