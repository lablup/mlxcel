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

//! Step-3.7 image processor (base pass + tiled-patch pass).
//!
//! Per image:
//! 1. Convert to RGB and square-pad to a black canvas whose side is
//!    `min(max(w, h), 3024)` (content pasted top-left), so all later math sees
//!    a square image.
//! 2. When `max(w, h)` exceeds `3024` the content is scaled down to fit the
//!    clamped square *before* the canvas is allocated, so the canvas is never
//!    larger than `3024 x 3024` regardless of the input aspect ratio.
//! 3. Window decision: side `<= 728` gives no patches; side `> 728` gives a
//!    `504` px window. (After square padding `long == short`, so the general
//!    upstream rule collapses to exactly these two outcomes.)
//! 4. Base pass: bilinear resize the padded square to `728x728`.
//! 5. Patch pass (only when windowed): crop target per axis
//!    `d' = 504 * (floor(r) + 1 if frac(r) > 0.2 else floor(r))` where
//!    `r = side / 504`; resize the square to `d' x d'` and tile `504x504`
//!    windows row-major. A newline marker follows every row-end patch except
//!    the final one.
//! 6. Normalize `x/255`, then `(x - mean)/std` per channel with the CLIP RGB
//!    constants. Output layout is channels-first `(num_images, 3, H, W)`; the
//!    base and patch tensors are separate batches.
//!
//! EXIF orientation transpose is a best-effort no-op here (the `image`
//! `DynamicImage` decode path does not carry orientation); documented as a
//! real-checkpoint follow-up.
//!
//! Used by: Step-3.7 (step3p7).

use image::imageops::FilterType;
use image::{DynamicImage, Rgb, RgbImage};
use mlxcel_core::{MlxArray, UniquePtr};

/// CLIP RGB normalization mean (per channel).
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
/// CLIP RGB normalization std (per channel).
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Base-image feature tokens after the two stride-2 downsamplers:
/// `728/14 = 52` grid, halved twice to `13`, `13^2 = 169`.
pub const BASE_FEATURE_TOKENS: usize = 169;
/// Patch feature tokens: `504/14 = 36` grid, halved twice to `9`, `9^2 = 81`.
pub const PATCH_FEATURE_TOKENS: usize = 81;

/// Per-image patch layout, shared between the processor (pixel tiling) and the
/// prompt expansion (placeholder blocks) so the two cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step3p7ImageLayout {
    /// Total `504x504` patches for this image (`0` when unwindowed).
    pub num_patches: usize,
    /// Patches per grid row (`sqrt(num_patches)`); newline markers land at row
    /// ends. `0` when unwindowed.
    pub patches_per_row: usize,
}

impl Step3p7ImageLayout {
    /// Total projected feature rows this image contributes:
    /// `169` base + `81 * num_patches`.
    pub fn feature_tokens(&self) -> usize {
        BASE_FEATURE_TOKENS + PATCH_FEATURE_TOKENS * self.num_patches
    }
}

/// Preprocessed Step-3.7 image tensors and per-image layout.
pub struct Step3p7PreprocessOutput {
    /// Base pass, channels-first `(num_images, 3, 728, 728)`.
    pub base_pixel_values: UniquePtr<MlxArray>,
    /// Patch pass, channels-first `(total_patches, 3, 504, 504)`; `None` when no
    /// image is windowed.
    pub patch_pixel_values: Option<UniquePtr<MlxArray>>,
    /// Per-image layouts in input order.
    pub layouts: Vec<Step3p7ImageLayout>,
}

pub struct Step3p7Processor {
    pub base_size: usize,
    pub patch_window: usize,
    pub max_side: usize,
    pub patch_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for Step3p7Processor {
    fn default() -> Self {
        Self::new()
    }
}

impl Step3p7Processor {
    pub fn new() -> Self {
        Self {
            base_size: 728,
            patch_window: 504,
            max_side: 3024,
            patch_size: 14,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// Clamp the square side to `max_side`.
    fn clamped_side(&self, raw_side: u32) -> u32 {
        (raw_side as usize).min(self.max_side) as u32
    }

    /// Compute the patch layout for an image of the given raw dimensions,
    /// independent of pixel data (used by the prompt expansion and tests).
    pub fn compute_layout(&self, width: u32, height: u32) -> Step3p7ImageLayout {
        let side = self.clamped_side(width.max(height));
        if (side as usize) <= self.base_size {
            return Step3p7ImageLayout {
                num_patches: 0,
                patches_per_row: 0,
            };
        }
        let r = side as f64 / self.patch_window as f64;
        let floor = r.floor();
        let k = floor as usize + usize::from((r - floor) > 0.2);
        Step3p7ImageLayout {
            num_patches: k * k,
            patches_per_row: k,
        }
    }

    /// Convert to RGB, then square-pad to a black canvas whose side is
    /// `min(max(w, h), max_side)` (content pasted top-left, black fill).
    ///
    /// When `max(w, h)` exceeds `max_side` the source is scaled down by
    /// `max_side / max(w, h)` before the square canvas is allocated, so the
    /// canvas never exceeds `max_side x max_side`. Allocating the full
    /// `max(w, h)` square first (then resizing down) would let an adversarial
    /// or merely extreme aspect ratio commit a quadratic amount of memory: a
    /// `16000 x 100` image is a few MB decoded but a `16000^2 * 3` (~768 MB)
    /// canvas, defeating the caller's decode budget. Scaling first bounds the
    /// canvas to `max_side^2` (~27 MB at the default `3024`).
    fn square_padded(&self, image: &DynamicImage) -> RgbImage {
        let rgb = image.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let raw_side = w.max(h);
        let side = self.clamped_side(raw_side);

        if side == raw_side {
            // Within `max_side`: pad at full resolution, no scaling.
            let mut canvas = RgbImage::from_pixel(raw_side, raw_side, Rgb([0, 0, 0]));
            image::imageops::overlay(&mut canvas, &rgb, 0, 0);
            return canvas;
        }

        // Oversized: scale the content down to fit the clamped square first,
        // then pad. `raw_side > side >= 1` here, so the ratio is well defined;
        // each scaled edge is clamped to `[1, side]`.
        let scale = side as f64 / raw_side as f64;
        let scaled_w = ((w as f64 * scale).round() as u32).clamp(1, side);
        let scaled_h = ((h as f64 * scale).round() as u32).clamp(1, side);
        let scaled = DynamicImage::ImageRgb8(rgb)
            .resize_exact(scaled_w, scaled_h, FilterType::Triangle)
            .to_rgb8();
        let mut canvas = RgbImage::from_pixel(side, side, Rgb([0, 0, 0]));
        image::imageops::overlay(&mut canvas, &scaled, 0, 0);
        canvas
    }

    /// Append `rgb` as channels-first normalized floats `(3, H, W)`.
    fn push_normalized(&self, rgb: &RgbImage, out: &mut Vec<f32>) {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    let v = pixel[c] as f32 / 255.0;
                    out.push((v - self.mean[c]) / self.std[c]);
                }
            }
        }
    }

    /// Preprocess a batch of images into the base and patch tensors plus the
    /// per-image patch layout.
    pub fn preprocess(&self, images: &[DynamicImage]) -> Step3p7PreprocessOutput {
        let base = self.base_size as u32;
        let window = self.patch_window as u32;

        let mut base_data: Vec<f32> = Vec::new();
        let mut patch_data: Vec<f32> = Vec::new();
        let mut layouts: Vec<Step3p7ImageLayout> = Vec::with_capacity(images.len());
        let mut total_patches = 0usize;

        for image in images {
            let padded = self.square_padded(image);
            let layout = self.compute_layout(image.width(), image.height());

            // Base pass: bilinear resize the padded square to 728x728.
            let base_img = DynamicImage::ImageRgb8(padded.clone())
                .resize_exact(base, base, FilterType::Triangle)
                .to_rgb8();
            self.push_normalized(&base_img, &mut base_data);

            // Patch pass: resize the square to d' x d' and tile 504 windows.
            if layout.num_patches > 0 {
                let k = layout.patches_per_row as u32;
                let crop_side = k * window;
                let resized = DynamicImage::ImageRgb8(padded)
                    .resize_exact(crop_side, crop_side, FilterType::Triangle)
                    .to_rgb8();
                for row in 0..k {
                    for col in 0..k {
                        let window_img = image::imageops::crop_imm(
                            &resized,
                            col * window,
                            row * window,
                            window,
                            window,
                        )
                        .to_image();
                        self.push_normalized(&window_img, &mut patch_data);
                    }
                }
                total_patches += layout.num_patches;
            }

            layouts.push(layout);
        }

        let base_pixel_values = mlxcel_core::from_slice_f32(
            &base_data,
            &[images.len() as i32, 3, base as i32, base as i32],
        );

        let patch_pixel_values = if total_patches > 0 {
            Some(mlxcel_core::from_slice_f32(
                &patch_data,
                &[total_patches as i32, 3, window as i32, window as i32],
            ))
        } else {
            None
        };

        Step3p7PreprocessOutput {
            base_pixel_values,
            patch_pixel_values,
            layouts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> Step3p7Processor {
        Step3p7Processor::new()
    }

    #[test]
    fn window_decision_no_patches_when_side_within_base() {
        let p = processor();
        // 728 and below -> no patches.
        assert_eq!(p.compute_layout(728, 728).num_patches, 0);
        assert_eq!(p.compute_layout(500, 728).num_patches, 0);
        assert_eq!(p.compute_layout(300, 200).num_patches, 0);
    }

    #[test]
    fn crop_and_tile_counts_follow_the_0_2_rounding_rule() {
        let p = processor();

        // side 1000 -> r = 1.984, frac 0.984 > 0.2 -> k = 2 -> 4 patches, 2/row.
        let l = p.compute_layout(1000, 800);
        assert_eq!((l.num_patches, l.patches_per_row), (4, 2));

        // side 1050 -> r = 2.083, frac 0.083 <= 0.2 -> k = 2 -> 4 patches.
        let l = p.compute_layout(1050, 700);
        assert_eq!((l.num_patches, l.patches_per_row), (4, 2));

        // side 1210 -> r = 2.40, frac 0.40 > 0.2 -> k = 3 -> 9 patches, 3/row.
        let l = p.compute_layout(1210, 900);
        assert_eq!((l.num_patches, l.patches_per_row), (9, 3));
    }

    #[test]
    fn max_side_clamp_caps_the_crop_grid() {
        let p = processor();
        // Raw side 6000 clamps to 3024 -> r = 6.0 exactly -> k = 6 -> 36 patches.
        let l = p.compute_layout(6000, 4000);
        assert_eq!((l.num_patches, l.patches_per_row), (36, 6));
    }

    #[test]
    fn square_pad_of_oversized_extreme_aspect_ratio_stays_bounded_by_max_side() {
        let p = processor();
        // Extreme aspect ratio (6000x100) whose raw max side (6000) exceeds
        // max_side (3024). Squaring the raw side before clamping would have
        // allocated a 6000x6000 canvas; the fixed path scales the content down
        // first, so the canvas must never exceed max_side x max_side.
        let side = p.max_side as u32;
        let wide = DynamicImage::ImageRgb8(RgbImage::from_pixel(6000, 100, Rgb([200, 10, 10])));
        let padded = p.square_padded(&wide);
        assert_eq!(padded.width(), side);
        assert_eq!(padded.height(), side);

        // Content scales to `round(100 * 3024/6000) = 50` rows, pasted at the
        // top; the solid source color survives the resize unchanged.
        let scaled_h = ((100.0_f64 * side as f64 / 6000.0).round() as u32).clamp(1, side);
        assert!(scaled_h < side, "expected padding below the scaled content");
        assert_eq!(*padded.get_pixel(0, scaled_h - 1), Rgb([200, 10, 10]));

        // The row immediately below the scaled content must be pure black:
        // scaling the content before pasting hard-pastes it onto the black
        // canvas with no blending. The pre-fix algorithm padded to the raw
        // (unclamped) square first and resized the *whole* canvas down
        // afterward, which blends the content/background boundary across
        // several destination rows (e.g. `Rgb([80, 4, 4])` at this row for a
        // 6000x100 source), so this assertion would fail against that order.
        assert_eq!(*padded.get_pixel(0, scaled_h), Rgb([0, 0, 0]));
        assert_eq!(*padded.get_pixel(0, side - 1), Rgb([0, 0, 0]));
    }

    #[test]
    fn feature_token_identity_169_plus_81_per_patch() {
        // 169 base tokens, 81 per patch.
        assert_eq!(BASE_FEATURE_TOKENS, 169);
        assert_eq!(PATCH_FEATURE_TOKENS, 81);
        let unwindowed = Step3p7ImageLayout {
            num_patches: 0,
            patches_per_row: 0,
        };
        assert_eq!(unwindowed.feature_tokens(), 169);
        let windowed = Step3p7ImageLayout {
            num_patches: 4,
            patches_per_row: 2,
        };
        assert_eq!(windowed.feature_tokens(), 169 + 81 * 4);
    }

    #[test]
    fn normalization_constants_are_the_clip_rgb_values() {
        let p = processor();
        assert_eq!(p.mean, CLIP_MEAN);
        assert_eq!(p.std, CLIP_STD);
        // Sanity-check the leading channels against the known CLIP values.
        assert!((p.mean[0] - 0.481).abs() < 1e-3);
        assert!((p.std[0] - 0.268).abs() < 1e-3);
    }

    #[test]
    fn preprocess_shapes_base_and_patch_batches() {
        let p = processor();
        // One small image (no patches) + one large image (4 patches).
        let small = DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 400, Rgb([120, 60, 30])));
        let large = DynamicImage::ImageRgb8(RgbImage::from_pixel(1000, 1000, Rgb([10, 200, 90])));
        let out = p.preprocess(&[small, large]);

        assert_eq!(
            mlxcel_core::array_shape(out.base_pixel_values.as_ref().unwrap()),
            vec![2, 3, 728, 728]
        );
        let patch = out
            .patch_pixel_values
            .as_ref()
            .expect("windowed image yields patches");
        assert_eq!(
            mlxcel_core::array_shape(patch.as_ref().unwrap()),
            vec![4, 3, 504, 504]
        );
        assert_eq!(out.layouts[0].num_patches, 0);
        assert_eq!(out.layouts[1].num_patches, 4);
    }
}
