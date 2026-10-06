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

//! Youtu-VL image processor.
//!
//! Mirrors the contract that upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/vision.py
//! (VisionModel.__call__) expects:
//! - `pixel_values`: `[total_patches, patch_size**2 * channels]` flattened
//!   patches, normalized with SigLIP statistics.
//! - `spatial_shapes`: `[N, 2]` of `(h_patches, w_patches)` per image (no
//!   temporal dimension — Youtu-VL is image-only at this scope).
//!
//! Why this is not the standard SigLIP processor: that processor emits
//! `[B, C, H, W]` because the existing SigLIP encoder uses a `Conv2d` patch
//! embedding. Youtu-VL uses a `Linear` patch embedding over flattened
//! patches and feeds them with `spatial_shapes` to a windowed-attention
//! tower, so the channel layout is fundamentally different. We therefore
//! own a small purpose-built processor instead of forcing reuse with
//! conditional code paths.
//!
//! Why this is not the Qwen2-VL processor either: Qwen2-VL duplicates each
//! spatial patch `temporal_patch_size` times and prepends a temporal axis to
//! mimic image-as-video; Youtu-VL's vision tower does not consume a
//! temporal index, so we emit one row per spatial patch and skip the
//! duplicate.
//!
//! Used by: `vision::youtu_vl::YoutuVLModel`.

use super::ImageProcessor;
use image::{DynamicImage, imageops::FilterType};
use mlxcel_core::{MlxArray, UniquePtr};
use thiserror::Error;

/// Per-image patch cap of the checkpoint's documented entry point.
///
/// Source: `YoutuVLProcessor.__call__(..., max_image_patches: int=36864, ...)`
/// in the checkpoint's `processing_youtu_vl.py` (line 53), which forwards it as
/// `max_num_patches` to the image processor on every call. It is a constant
/// because 36864 exists only as that Python default argument: no JSON file in
/// the checkpoint carries it. `preprocessor_config.json`'s `max_num_patches:
/// 256` is the bare `Siglip2ImageProcessorFast` default that this call
/// overrides, and `vision_config.num_patches` bounds nothing here (the tower
/// uses 2D RoPE, not a learned position table).
pub const DEFAULT_MAX_PATCHES_PER_IMAGE: usize = 36864;

/// Step by which the reference walks its resize scale down until the grid
/// fits under the cap (`get_image_size_for_patches`).
const SCALE_STEP: f64 = 0.02;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum YoutuVLPreprocessError {
    #[error("invalid Youtu-VL processor config: {0}")]
    InvalidConfig(&'static str),
    #[error(
        "image {image_index} would produce {patches} patches, exceeding the configured per-image cap {max_patches}"
    )]
    TooManyPatches {
        image_index: usize,
        patches: usize,
        max_patches: usize,
    },
    #[error(
        "image {image_index} has a {h_patches}x{w_patches} patch grid that is not divisible by spatial_merge_size {spatial_merge_size}"
    )]
    UnalignedPatchGrid {
        image_index: usize,
        h_patches: usize,
        w_patches: usize,
        spatial_merge_size: usize,
    },
    #[error("Youtu-VL patch tensor allocation would overflow usize arithmetic")]
    AllocationOverflow,
    #[error("Youtu-VL patch tensor dimension {dimension} exceeds i32::MAX")]
    DimensionTooLarge { dimension: usize },
}

pub struct YoutuVLProcessor {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    /// Cap on flattened patches per image. `smart_resize` shrinks the image
    /// until the grid fits, and `try_preprocess_with_spatial` enforces it
    /// again before allocating the `[total_patches, patch_size**2 * channels]`
    /// tensor. Defaults to [`DEFAULT_MAX_PATCHES_PER_IMAGE`].
    pub max_patches_per_image: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl YoutuVLProcessor {
    /// SigLIP2 default normalization (mean=std=0.5 across all channels).
    pub fn new(patch_size: usize, spatial_merge_size: usize) -> Self {
        Self {
            patch_size,
            spatial_merge_size,
            max_patches_per_image: DEFAULT_MAX_PATCHES_PER_IMAGE,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    pub fn with_norm(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = mean;
        self.std = std;
        self
    }

    pub fn with_max_patches_per_image(mut self, max_patches: usize) -> Self {
        self.max_patches_per_image = max_patches;
        self
    }

    fn validate_config(&self) -> Result<(), YoutuVLPreprocessError> {
        if self.patch_size == 0 {
            return Err(YoutuVLPreprocessError::InvalidConfig(
                "patch_size must be greater than zero",
            ));
        }
        if self.spatial_merge_size == 0 {
            return Err(YoutuVLPreprocessError::InvalidConfig(
                "spatial_merge_size must be greater than zero",
            ));
        }
        if self.max_patches_per_image == 0 {
            return Err(YoutuVLPreprocessError::InvalidConfig(
                "max_patches_per_image must be greater than zero",
            ));
        }
        let resize_factor = self
            .patch_size
            .checked_mul(self.spatial_merge_size)
            .ok_or(YoutuVLPreprocessError::AllocationOverflow)?;
        if resize_factor > u32::MAX as usize {
            return Err(YoutuVLPreprocessError::DimensionTooLarge {
                dimension: resize_factor,
            });
        }
        if self
            .std
            .iter()
            .any(|v| !v.is_finite() || v.abs() < f32::EPSILON)
        {
            return Err(YoutuVLPreprocessError::InvalidConfig(
                "normalization std values must be finite and non-zero",
            ));
        }
        if self.mean.iter().any(|v| !v.is_finite()) {
            return Err(YoutuVLPreprocessError::InvalidConfig(
                "normalization mean values must be finite",
            ));
        }
        Ok(())
    }

    fn resize_factor(&self) -> u32 {
        self.patch_size
            .saturating_mul(self.spatial_merge_size)
            .max(1) as u32
    }

    /// Target `(h, w)` in pixels, a port of `get_image_size_for_patches` in
    /// the checkpoint's `image_processing_siglip2_fast.py`.
    ///
    /// Each edge is rounded UP to a multiple of `patch_size *
    /// spatial_merge_size` (at least one block), starting at `scale = 1.0` and
    /// lowering `scale` by 0.02 until the patch grid fits under
    /// `max_patches_per_image`. There is no lower pixel bound beyond one block
    /// per edge and no upscaling. The reference hardcodes the block as
    /// `patch_size * 2`; the config-driven form is equal for this checkpoint.
    ///
    /// `scale` is decremented in place exactly as Python does; `1.0 - 0.02 *
    /// k` rounds differently and can pick a different grid. For an extreme
    /// aspect ratio `scale` may go to or below zero, where every edge clamps to
    /// one block, so the loop ends once a grid that small fits the cap. The
    /// iteration bound only guards a cap below one block's patch count, which
    /// `try_preprocess_with_spatial` then rejects as `TooManyPatches`.
    fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let block = self.resize_factor() as f64;
        let patch = self.patch_size.max(1) as u64;
        let max_patches = self.max_patches_per_image as u64;
        // Compute in f64 and clamp before casting so a negative scaled edge
        // never reaches an unsigned type.
        let scaled = |edge: u32, scale: f64| -> u32 {
            let size = ((edge as f64 * scale) / block).ceil() * block;
            size.max(block).min(u32::MAX as f64) as u32
        };

        let mut scale = 1.0f64;
        // One block per edge is reached by scale <= 0, i.e. within
        // 1 / SCALE_STEP + 1 steps; a few more cover f64 accumulation.
        let max_steps = (1.0 / SCALE_STEP).ceil() as usize + 2;
        let mut h = scaled(orig_h, scale);
        let mut w = scaled(orig_w, scale);
        for _ in 0..max_steps {
            let patches = (h as u64 / patch) * (w as u64 / patch);
            if patches <= max_patches {
                break;
            }
            scale -= SCALE_STEP;
            h = scaled(orig_h, scale);
            w = scaled(orig_w, scale);
        }
        (h, w)
    }

    /// Compute `(h_patches, w_patches)` per image after resizing.
    pub fn compute_spatial_shapes(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                let h_patches = (h / self.patch_size as u32) as i32;
                let w_patches = (w / self.patch_size as u32) as i32;
                (h_patches, w_patches)
            })
            .collect()
    }

    pub fn try_preprocess_with_spatial(
        &self,
        images: &[DynamicImage],
    ) -> Result<(UniquePtr<MlxArray>, Vec<(i32, i32)>), YoutuVLPreprocessError> {
        self.validate_config()?;
        let spatial_shapes = self.compute_spatial_shapes(images);

        let in_channels = 3usize;
        let patch_area = self
            .patch_size
            .checked_mul(self.patch_size)
            .ok_or(YoutuVLPreprocessError::AllocationOverflow)?;
        let features_per_patch = in_channels
            .checked_mul(patch_area)
            .ok_or(YoutuVLPreprocessError::AllocationOverflow)?;

        let mut total_patches: usize = 0;
        for (image_index, &(h, w)) in spatial_shapes.iter().enumerate() {
            let patches = (h as usize)
                .checked_mul(w as usize)
                .ok_or(YoutuVLPreprocessError::AllocationOverflow)?;
            if patches > self.max_patches_per_image {
                return Err(YoutuVLPreprocessError::TooManyPatches {
                    image_index,
                    patches,
                    max_patches: self.max_patches_per_image,
                });
            }
            total_patches = total_patches
                .checked_add(patches)
                .ok_or(YoutuVLPreprocessError::AllocationOverflow)?;
        }

        let total_patch_values = total_patches
            .checked_mul(features_per_patch)
            .ok_or(YoutuVLPreprocessError::AllocationOverflow)?;

        let mut all_patches = vec![0f32; total_patch_values];
        let mut write_offset: usize = 0;

        for (img_idx, img) in images.iter().enumerate() {
            let (h_patches, w_patches) = spatial_shapes[img_idx];
            let target_h = (h_patches as u32) * (self.patch_size as u32);
            let target_w = (w_patches as u32) * (self.patch_size as u32);

            let resized = img.resize_exact(target_w, target_h, FilterType::Lanczos3);
            let rgb = resized.to_rgb8();

            let h = target_h as usize;
            let w = target_w as usize;
            let mut normalized = vec![0f32; in_channels * h * w];
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    for c in 0..in_channels {
                        let val = pixel[c] as f32 / 255.0;
                        let normed = (val - self.mean[c]) / self.std[c];
                        normalized[c * h * w + y * w + x] = normed;
                    }
                }
            }

            // Emit one row per spatial patch in the layout
            // `[h_patches * w_patches, patch_size * patch_size * channels]`.
            //
            // Rows are merge-block-major, not raster: they run
            // `(block_y, block_x, inner_y, inner_x)`, so each consecutive run
            // of `spatial_merge_size ** 2` rows is exactly one spatial-merge
            // block. Inner features are channel-last, `(dy, dx, c)`.
            //
            // Both orders come from this checkpoint's own processor,
            // `convert_image_to_patches` in
            // https://huggingface.co/tencent/Youtu-VL-4B-Instruct/blob/main/image_processing_siglip2_fast.py
            // which reshapes the image to `(C, nh/m, m, ps, nw/m, m, ps)` and
            // permutes `(1, 4, 2, 5, 3, 6, 0)`.
            // `Siglip2VisionEmbeddings.patch_embedding` is an `nn.Linear` over
            // those rows and `remap_youtu_vl_weights` only renames that weight,
            // so any other order feeds the trained projection a permuted
            // vector. `YoutuVLVisionEncoder::rot_pos_emb` and the merge-unit
            // gather in `forward_with_spatial` independently assume the same
            // block-major grouping, so raster rows also mislabel every token's
            // rotary position. This is not Qwen2-VL's `(c, dy, dx)` layout:
            // that family unfolds and this one does not.
            let merge = self.spatial_merge_size;
            let hp = h_patches as usize;
            let wp = w_patches as usize;
            // `smart_resize` rounds both edges up to a multiple of
            // `patch_size * spatial_merge_size`, so the grid is always
            // divisible and the block loops never need padding. Check it
            // rather than relying on that invariant silently: a future change
            // to the resize policy would otherwise drop patches here.
            if !hp.is_multiple_of(merge) || !wp.is_multiple_of(merge) {
                return Err(YoutuVLPreprocessError::UnalignedPatchGrid {
                    image_index: img_idx,
                    h_patches: hp,
                    w_patches: wp,
                    spatial_merge_size: merge,
                });
            }

            let total_patches_img = hp * wp;
            let mut row = 0usize;
            for block_y in 0..hp / merge {
                for block_x in 0..wp / merge {
                    for inner_y in 0..merge {
                        for inner_x in 0..merge {
                            let py = block_y * merge + inner_y;
                            let px = block_x * merge + inner_x;
                            let y_start = py * self.patch_size;
                            let x_start = px * self.patch_size;

                            let row_start = (write_offset + row) * features_per_patch;
                            let mut k = 0usize;
                            for dy in 0..self.patch_size {
                                for dx in 0..self.patch_size {
                                    let y = y_start + dy;
                                    let x = x_start + dx;
                                    for c in 0..in_channels {
                                        all_patches[row_start + k] =
                                            normalized[c * h * w + y * w + x];
                                        k += 1;
                                    }
                                }
                            }
                            row += 1;
                        }
                    }
                }
            }
            debug_assert_eq!(row, total_patches_img);

            write_offset += total_patches_img;
        }

        let total_patches_i32 = i32::try_from(total_patches).map_err(|_| {
            YoutuVLPreprocessError::DimensionTooLarge {
                dimension: total_patches,
            }
        })?;
        let features_per_patch_i32 = i32::try_from(features_per_patch).map_err(|_| {
            YoutuVLPreprocessError::DimensionTooLarge {
                dimension: features_per_patch,
            }
        })?;
        let pixel_values =
            mlxcel_core::from_slice_f32(&all_patches, &[total_patches_i32, features_per_patch_i32]);

        Ok((pixel_values, spatial_shapes))
    }

    /// Preprocess the input images and return `(pixel_values, spatial_shapes)`
    /// in the layout expected by `YoutuVLVisionEncoder::forward_with_spatial`.
    ///
    /// Runtime call sites should prefer [`Self::try_preprocess_with_spatial`]
    /// so invalid configs and oversized images surface as request errors.
    pub fn preprocess_with_spatial(
        &self,
        images: &[DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32)>) {
        match self.try_preprocess_with_spatial(images) {
            Ok(result) => result,
            Err(err) => {
                tracing::warn!("Youtu-VL preprocessing failed: {err}");
                let features_per_patch = self
                    .patch_size
                    .checked_mul(self.patch_size)
                    .and_then(|area| area.checked_mul(3))
                    .and_then(|n| i32::try_from(n).ok())
                    .unwrap_or(0);
                (
                    mlxcel_core::zeros(&[0, features_per_patch], mlxcel_core::dtype::FLOAT32),
                    Vec::new(),
                )
            }
        }
    }
}

impl ImageProcessor for YoutuVLProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_spatial(images);
        pixel_values
    }
}

#[cfg(test)]
#[path = "youtu_vl_tests.rs"]
mod tests;
