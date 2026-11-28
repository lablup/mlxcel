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

//! Qwen2-VL Image Processor
//!
//! Handles dynamic resolution image preprocessing:
//! 1. Resize preserving aspect ratio within constraints
//! 2. Pad to multiples of patch_size * spatial_merge_size (= 28)
//! 3. Duplicate frame for temporal_patch_size (single image -> 2 frames)
//! 4. Flatten to patch format for vision encoder
//!
//! Used by: Qwen2-VL

use super::ImageProcessor;
use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Qwen2VLProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub video_default_fps: f64,
    pub video_min_frames: usize,
    pub video_max_frames: usize,
    resize_filter: FilterType,
}

/// Default lower pixel bound the Qwen2-VL processors are constructed with.
pub const DEFAULT_MIN_PIXELS: usize = 4 * 28 * 28;

/// Default upper pixel bound the Qwen2-VL processors are constructed with.
///
/// `smart_resize` caps the resized area at this value, so it is also what
/// bounds the visual token count for any input image. Shared with
/// [`max_image_tokens`] so a capacity floor derived from it cannot drift away
/// from what the processor actually admits.
pub const DEFAULT_MAX_PIXELS: usize = 16384 * 28 * 28;

/// Qwen-VL video sampling defaults from the published processor sidecars.
pub const DEFAULT_VIDEO_FPS: f64 = crate::multimodal::video::DEFAULT_FPS;
pub const DEFAULT_VIDEO_MIN_FRAMES: usize = crate::multimodal::video::FPS_MIN_FRAMES;
pub const DEFAULT_VIDEO_MAX_FRAMES: usize = crate::multimodal::video::FPS_MAX_FRAMES;

/// Qwen video sampling policy parsed from `processor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenVideoProcessorConfig {
    pub fps: f64,
    pub min_frames: usize,
    pub max_frames: usize,
}

impl Default for QwenVideoProcessorConfig {
    fn default() -> Self {
        Self {
            fps: DEFAULT_VIDEO_FPS,
            min_frames: DEFAULT_VIDEO_MIN_FRAMES,
            max_frames: DEFAULT_VIDEO_MAX_FRAMES,
        }
    }
}

impl QwenVideoProcessorConfig {
    /// Resolve Qwen's video sampling sidecar. Missing keys keep the upstream
    /// defaults; present malformed values are rejected by the loader before a
    /// model can serve video with a silently widened or inverted bound.
    pub fn from_processor_config(config: Option<&serde_json::Value>) -> Result<Self, String> {
        let mut resolved = Self::default();
        if let Some(config) = config {
            if let Some(fps) = config.get("fps") {
                resolved.fps = fps
                    .as_f64()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| {
                        "Qwen-VL processor_config.json `fps` must be a finite positive number"
                            .to_string()
                    })?;
            }
            if let Some(min_frames) = config.get("min_frames") {
                resolved.min_frames =
                    usize_from_json(min_frames, "Qwen-VL processor_config.json `min_frames`")?;
            }
            if let Some(max_frames) = config.get("max_frames") {
                resolved.max_frames =
                    usize_from_json(max_frames, "Qwen-VL processor_config.json `max_frames`")?;
            }
        }

        if resolved.min_frames == 0 {
            return Err("Qwen-VL processor_config.json `min_frames` must be positive".to_string());
        }
        if resolved.max_frames < resolved.min_frames {
            return Err(format!(
                "Qwen-VL processor_config.json requires min_frames <= max_frames, got {}..={}",
                resolved.min_frames, resolved.max_frames
            ));
        }

        Ok(resolved)
    }

    #[must_use]
    pub fn frame_sampling(
        self,
        temporal_patch_size: usize,
    ) -> crate::multimodal::video::FrameSamplingPolicy {
        crate::multimodal::video::FrameSamplingPolicy {
            min_frames: self.min_frames,
            max_frames: self.max_frames,
            frame_factor: temporal_patch_size.max(1),
            reject_over_max: true,
            keep_all_without_target_fps: false,
        }
    }
}

fn usize_from_json(value: &serde_json::Value, field: &str) -> Result<usize, String> {
    let raw = value
        .as_u64()
        .ok_or_else(|| format!("{field} must be a non-negative integer"))?;
    usize::try_from(raw).map_err(|_| format!("{field} is too large for this platform"))
}

#[derive(Debug, Clone, Copy)]
pub enum Qwen2VLMediaInput<'a> {
    Image(&'a DynamicImage),
    Video(&'a [DynamicImage]),
}

/// Largest visual token count one image can expand into under these bounds.
///
/// `smart_resize` rounds both edges to `patch_size * spatial_merge_size` and
/// caps the area at `max_pixels`; `insert_qwen_vl_image_tokens` then emits
/// `(h / merge) * (w / merge)` tokens for a single-frame image. Both reduce to
/// `pixels / (patch_size * spatial_merge_size)^2`, so the maximum is that
/// quotient evaluated at the pixel cap.
///
/// Returns `None` for a degenerate geometry rather than dividing by zero.
#[must_use]
pub fn max_image_tokens(
    patch_size: usize,
    spatial_merge_size: usize,
    max_pixels: usize,
) -> Option<usize> {
    let factor = patch_size.checked_mul(spatial_merge_size)?;
    let divisor = factor.checked_mul(factor)?;
    if divisor == 0 {
        return None;
    }
    Some(max_pixels / divisor)
}

impl Qwen2VLProcessor {
    /// Used by: Qwen2-VL, Qwen2.5-VL (CLIP normalization)
    pub fn new(patch_size: usize, temporal_patch_size: usize, spatial_merge_size: usize) -> Self {
        crate::vision::image_token_overrides::note_dynamic_resolution_processor();
        Self {
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
            video_default_fps: DEFAULT_VIDEO_FPS,
            video_min_frames: DEFAULT_VIDEO_MIN_FRAMES,
            video_max_frames: DEFAULT_VIDEO_MAX_FRAMES,
            resize_filter: FilterType::Lanczos3,
        }
    }

    /// Used by: Qwen3-VL, Qwen3-VL MoE, Qwen3.5 / 3.6 VLM, Qwen3-Omni MoE, Cohere Compass,
    /// Qwen3.8-Flash-Next, Mage VL (simple 0.5/0.5 normalization)
    pub fn new_with_norm(
        patch_size: usize,
        temporal_patch_size: usize,
        spatial_merge_size: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Self {
        crate::vision::image_token_overrides::note_dynamic_resolution_processor();
        Self {
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            mean,
            std,
            video_default_fps: DEFAULT_VIDEO_FPS,
            video_min_frames: DEFAULT_VIDEO_MIN_FRAMES,
            video_max_frames: DEFAULT_VIDEO_MAX_FRAMES,
            resize_filter: FilterType::Lanczos3,
        }
    }

    /// Override the checkpoint's resize bounds in pixels.
    ///
    /// Qwen2-VL / Qwen2.5-VL checkpoints ship the family defaults, so their
    /// loaders take [`DEFAULT_MIN_PIXELS`] / [`DEFAULT_MAX_PIXELS`]. The
    /// Qwen3-VL generation does not: its sidecars set `size.shortest_edge` to
    /// 65536 and `size.longest_edge` to 16777216, and reading them is what makes
    /// `smart_resize` pick the same grid as the HF processor.
    ///
    /// Used by: the Qwen3-VL-generation loaders through the checkpoint resize
    /// policy, Mage VL, CLEF.
    #[must_use]
    pub fn with_pixel_bounds(mut self, min_pixels: usize, max_pixels: usize) -> Self {
        self.min_pixels = min_pixels;
        self.max_pixels = max_pixels;
        self
    }

    #[must_use]
    pub fn with_video_config(mut self, config: QwenVideoProcessorConfig) -> Self {
        self.video_default_fps = config.fps;
        self.video_min_frames = config.min_frames;
        self.video_max_frames = config.max_frames;
        self
    }

    /// Override the image resampler while preserving the historical Qwen2
    /// default for callers that do not opt in.
    #[must_use]
    pub fn with_resize_filter(mut self, resize_filter: FilterType) -> Self {
        self.resize_filter = resize_filter;
        self
    }

    /// The image resampler this processor resizes with.
    #[must_use]
    pub fn resize_filter(&self) -> FilterType {
        self.resize_filter
    }

    #[must_use]
    pub fn video_sampling_policy(&self) -> crate::multimodal::video::FrameSamplingPolicy {
        QwenVideoProcessorConfig {
            fps: self.video_default_fps,
            min_frames: self.video_min_frames,
            max_frames: self.video_max_frames,
        }
        .frame_sampling(self.temporal_patch_size)
    }

    /// The pixel bounds this processor resizes against.
    ///
    /// Normally the checkpoint's own `min_pixels` / `max_pixels`. When an
    /// operator passed b10621's `--image-min-tokens` / `--image-max-tokens`,
    /// [`crate::vision::image_token_overrides`] converts the token budget into
    /// the same two bounds with upstream's own `tokens * patch_size^2 *
    /// merge^2` arithmetic and returns those instead, counting the application
    /// so startup can tell an ignored budget from an applied one.
    ///
    /// Used by: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Qwen3-VL-MoE,
    /// Qwen3-Omni, GLM-4V, GLM-4V-MoE, GLM-OCR, ColQwen2.5, Qwen3.8-Flash-Next.
    #[must_use]
    pub(crate) fn effective_pixel_bounds(&self) -> (usize, usize) {
        crate::vision::image_token_overrides::resolve_pixel_bounds(
            self.patch_size,
            self.spatial_merge_size,
            self.min_pixels,
            self.max_pixels,
        )
    }
    /// Compute target size that satisfies constraints
    /// Returns (height, width) padded to multiples of factor
    pub(crate) fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u64; // 28
        let (min_pixels, max_pixels) = self.effective_pixel_bounds();
        let (min_pixels, max_pixels) = (min_pixels as u64, max_pixels as u64);
        let area = |h: u64, w: u64| h.saturating_mul(w);
        // The reference's `round(height / factor) * factor` has no floor: a
        // thin edge that rounds to 0 drops the area under `min_pixels`, and
        // that branch rescales the original size (10x1000 at factor 28 gives
        // 28x560, not the 28x1008 a one-factor floor here would keep).
        let round_to =
            |edge: f64| -> u64 { (edge / factor as f64).round_ties_even() as u64 * factor };

        // Start with original size, round to factor. Arithmetic is u64 and
        // saturating so a huge image or a huge checkpoint bound cannot wrap.
        // The reference's `round(height / factor)` is Python's `round`, which
        // rounds half to even: 720 / 32 = 22.5 goes to 22, not 23.
        let mut h = round_to(f64::from(orig_h));
        let mut w = round_to(f64::from(orig_w));

        // Out of bounds, scale the original size once as the reference does:
        // `beta = sqrt(h * w / max_pixels)`, then `floor(h / beta / factor)`
        // (`ceil(h * beta / factor)` under `min_pixels`). Rescaling the
        // rounded size instead lands on another multiple (1372x772 under
        // 1003520 pixels gives 756 rows, not 728). Operation order as upstream.
        // A zero edge (the reference raises on it) is rescaled as one pixel,
        // so `beta` stays finite and no edge saturates to `u32::MAX`.
        let (orig_h, orig_w) = (orig_h.max(1), orig_w.max(1));
        let orig_area = area(u64::from(orig_h), u64::from(orig_w)) as f64;
        let (orig_h, orig_w) = (f64::from(orig_h), f64::from(orig_w));
        let factor_f = factor as f64;
        let snap = |steps: f64| (steps as u64).max(1).saturating_mul(factor);
        let pixels = area(h, w);
        if pixels > max_pixels {
            let beta = (orig_area / max_pixels as f64).sqrt();
            h = snap((orig_h / beta / factor_f).floor());
            w = snap((orig_w / beta / factor_f).floor());
        } else if pixels < min_pixels {
            let beta = (min_pixels as f64 / orig_area).sqrt();
            h = snap((orig_h * beta / factor_f).ceil());
            w = snap((orig_w * beta / factor_f).ceil());
        }
        // Past 200:1 the reference raises instead of resizing. There the
        // short side can clamp to one `factor` with the area still over the
        // cap (after either branch); trim the long side so
        // `max_image_tokens` still holds.
        if area(h, w) > max_pixels && orig_h.max(orig_w) / orig_h.min(orig_w) > 200.0 {
            if h <= w {
                w = (max_pixels / h / factor).max(1) * factor;
            } else {
                h = (max_pixels / w / factor).max(1) * factor;
            }
        }
        // Deviation: with `min_pixels == 0` an edge that rounds to 0 passes
        // both bounds checks and the reference returns a zero edge, which no
        // resize can produce. Floor it to one `factor`.
        h = h.max(factor);
        w = w.max(factor);

        (
            u32::try_from(h).unwrap_or(u32::MAX),
            u32::try_from(w).unwrap_or(u32::MAX),
        )
    }

    /// Compute grid_thw for a set of images
    /// Returns Vec of (temporal, h_patches, w_patches)
    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                let h_patches = h as i32 / self.patch_size as i32;
                let w_patches = w as i32 / self.patch_size as i32;
                (1i32, h_patches, w_patches) // temporal=1 for single images
            })
            .collect()
    }

    /// Preprocess images and return (pixel_values, grid_thw)
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let (all_patches, grid_thw) = self.preprocess_values_with_grid(images);
        let in_channels = 3usize;
        let patch_area = self.patch_size * self.patch_size;
        let features_per_pixel = in_channels * patch_area;
        let total_rows: usize = grid_thw
            .iter()
            .map(|&(t, h, w)| (t as usize) * (h as usize) * (w as usize) * self.temporal_patch_size)
            .sum();
        let pixel_values = mlxcel_core::from_slice_f32(
            &all_patches,
            &[total_rows as i32, features_per_pixel as i32],
        );
        (pixel_values, grid_thw)
    }

    /// Pure host form of [`Self::preprocess_with_grid`].
    ///
    /// The OpenXLA path consumes these owned F32 values directly, while the MLX
    /// path wraps the exact same values in an `MlxArray`. Keeping one processor
    /// implementation prevents resize/normalize/patch-order drift.
    pub fn preprocess_values_with_grid(
        &self,
        images: &[DynamicImage],
    ) -> (Vec<f32>, Vec<(i32, i32, i32)>) {
        self.preprocess_media_values_with_grid(
            &images
                .iter()
                .map(Qwen2VLMediaInput::Image)
                .collect::<Vec<_>>(),
        )
        .expect("image-only Qwen-VL preprocessing cannot fail")
    }

    /// Preprocess a mixed image/video sequence in media order and return the
    /// flattened Conv3d patch rows plus one `(t, h, w)` grid per media item.
    ///
    /// For videos, `t` is the number of temporal-patch groups, not raw frames:
    /// a 4-frame clip with `temporal_patch_size=2` has `t=2`. The spatial
    /// axes are shared across all frames in one clip, matching upstream's
    /// `_smart_resize_video` contract.
    pub fn preprocess_media_values_with_grid(
        &self,
        media: &[Qwen2VLMediaInput<'_>],
    ) -> Result<(Vec<f32>, Vec<(i32, i32, i32)>), String> {
        let mut all_patches: Vec<f32> = Vec::new();
        let mut grid_thw: Vec<(i32, i32, i32)> = Vec::with_capacity(media.len());
        let in_channels = 3usize;
        let patch_area = self.patch_size * self.patch_size;
        let features_per_pixel = in_channels * patch_area;

        for item in media {
            let (frames, target_h, target_w, grid_t) = match *item {
                Qwen2VLMediaInput::Image(image) => {
                    let (h, w) = self.smart_resize(image.height(), image.width());
                    let frames =
                        std::iter::repeat_n(image, self.temporal_patch_size).collect::<Vec<_>>();
                    (frames, h, w, 1usize)
                }
                Qwen2VLMediaInput::Video(frames) => {
                    let frames = self.pad_video_frames(frames)?;
                    let Some(first) = frames.first() else {
                        return Err("Qwen-VL video contains no decoded frames".to_string());
                    };
                    let (h, w) = self.smart_resize(first.height(), first.width());
                    let grid_t = frames.len() / self.temporal_patch_size;
                    (frames, h, w, grid_t)
                }
            };

            let h_patches = target_h as usize / self.patch_size;
            let w_patches = target_w as usize / self.patch_size;
            grid_thw.push((grid_t as i32, h_patches as i32, w_patches as i32));

            let normalized_frames = frames
                .iter()
                .map(|frame| self.normalize_resized_frame(frame, target_h, target_w))
                .collect::<Vec<_>>();

            for temporal_group in 0..grid_t {
                for block_y in 0..h_patches / self.spatial_merge_size {
                    for block_x in 0..w_patches / self.spatial_merge_size {
                        for inner_y in 0..self.spatial_merge_size {
                            for inner_x in 0..self.spatial_merge_size {
                                let py = block_y * self.spatial_merge_size + inner_y;
                                let px = block_x * self.spatial_merge_size + inner_x;
                                let y_start = py * self.patch_size;
                                let x_start = px * self.patch_size;
                                for tp in 0..self.temporal_patch_size {
                                    let frame_idx = temporal_group * self.temporal_patch_size + tp;
                                    let normalized = &normalized_frames[frame_idx];
                                    for c in 0..in_channels {
                                        for dy in 0..self.patch_size {
                                            for dx in 0..self.patch_size {
                                                let y = y_start + dy;
                                                let x = x_start + dx;
                                                all_patches.push(
                                                    normalized[c
                                                        * target_h as usize
                                                        * target_w as usize
                                                        + y * target_w as usize
                                                        + x],
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        debug_assert_eq!(
            all_patches.len(),
            grid_thw
                .iter()
                .map(|&(t, h, w)| {
                    t as usize
                        * h as usize
                        * w as usize
                        * self.temporal_patch_size
                        * features_per_pixel
                })
                .sum::<usize>()
        );
        Ok((all_patches, grid_thw))
    }

    pub fn preprocess_media_with_grid(
        &self,
        media: &[Qwen2VLMediaInput<'_>],
    ) -> Result<(UniquePtr<MlxArray>, Vec<(i32, i32, i32)>), String> {
        let (all_patches, grid_thw) = self.preprocess_media_values_with_grid(media)?;
        let in_channels = 3usize;
        let patch_area = self.patch_size * self.patch_size;
        let features_per_pixel = in_channels * patch_area;
        let total_rows: usize = grid_thw
            .iter()
            .map(|&(t, h, w)| (t as usize) * (h as usize) * (w as usize) * self.temporal_patch_size)
            .sum();
        let pixel_values = mlxcel_core::from_slice_f32(
            &all_patches,
            &[total_rows as i32, features_per_pixel as i32],
        );
        Ok((pixel_values, grid_thw))
    }

    fn normalize_resized_frame(
        &self,
        image: &DynamicImage,
        target_h: u32,
        target_w: u32,
    ) -> Vec<f32> {
        let resized = image.resize_exact(target_w, target_h, self.resize_filter);
        let rgb = resized.to_rgb8();
        let h = target_h as usize;
        let w = target_w as usize;
        let mut normalized = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    normalized[c * h * w + y * w + x] = (val - self.mean[c]) / self.std[c];
                }
            }
        }
        normalized
    }

    fn pad_video_frames<'a>(
        &self,
        frames: &'a [DynamicImage],
    ) -> Result<Vec<&'a DynamicImage>, String> {
        if frames.len() < self.video_min_frames {
            return Err(format!(
                "Qwen-VL video has {} decoded frame(s), below min_frames={}",
                frames.len(),
                self.video_min_frames
            ));
        }
        if frames.len() > self.video_max_frames {
            return Err(format!(
                "Qwen-VL video has {} decoded frame(s), exceeding max_frames={}",
                frames.len(),
                self.video_max_frames
            ));
        }
        if self.temporal_patch_size == 0 {
            return Err("Qwen-VL temporal_patch_size must be positive".to_string());
        }
        let padded_len = frames.len().div_ceil(self.temporal_patch_size) * self.temporal_patch_size;
        if padded_len > self.video_max_frames {
            return Err(format!(
                "Qwen-VL video needs {} frame slot(s) after temporal padding, exceeding \
                 max_frames={}",
                padded_len, self.video_max_frames
            ));
        }
        let mut padded = frames.iter().collect::<Vec<_>>();
        if let Some(last) = frames.last() {
            while padded.len() < padded_len {
                padded.push(last);
            }
        }
        Ok(padded)
    }
}

impl ImageProcessor for Qwen2VLProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::*;

    /// The Qwen3-VL-generation geometry: 16-pixel patches, 2x2 merge, and
    /// the 65536 / 16777216 bounds those sidecars declare.
    fn factor_32_processor() -> Qwen2VLProcessor {
        Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3])
            .with_pixel_bounds(65_536, 16_777_216)
    }

    /// transformers' `smart_resize` rounds `edge / factor` with Python's
    /// `round`, which breaks a tie towards the even neighbour. Expected
    /// values are transformers 5.24.1 `image_processing_pil_qwen2_vl.
    /// smart_resize` (and mlx-vlm's Qwen3-VL `_smart_resize_image`).
    #[test]
    fn smart_resize_breaks_ties_to_even_like_python_round() {
        // Factor 28 (Qwen2-VL, Qwen2.5-VL, GLM-4V): 854 / 28 = 30.5 -> 30,
        // 630 / 28 = 22.5 -> 22, 350 / 28 = 12.5 -> 12.
        let p28 = Qwen2VLProcessor::new(14, 2, 2);
        assert_eq!(p28.smart_resize(480, 854), (476, 840));
        assert_eq!(p28.smart_resize(630, 1200), (616, 1204));
        assert_eq!(p28.smart_resize(350, 350), (336, 336));
        // Factor 32 (Qwen3-VL generation, Mage-VL): 720 / 32 = 22.5 -> 22,
        // 3024 / 32 = 94.5 -> 94, 1680 / 32 = 52.5 -> 52, 336 / 32 = 10.5 -> 10.
        let p32 = factor_32_processor();
        assert_eq!(p32.smart_resize(720, 1280), (704, 1280));
        assert_eq!(p32.smart_resize(3024, 4032), (3008, 4032));
        assert_eq!(p32.smart_resize(1050, 1680), (1056, 1664));
        assert_eq!(p32.smart_resize(336, 336), (320, 320));
        // A 1280x720 image is 44x80 patches, 22 * 40 = 880 merged tokens.
        let grid = p32.compute_grid_thw(&[DynamicImage::new_rgb8(1280, 720)]);
        assert_eq!(grid, vec![(1, 44, 80)]);
    }

    /// Non-ties and ties whose lower neighbour is odd round the same either
    /// way; these must not move.
    #[test]
    fn smart_resize_non_ties_and_odd_ties_are_unchanged() {
        let p28 = Qwen2VLProcessor::new(14, 2, 2);
        assert_eq!(p28.smart_resize(1080, 1920), (1092, 1932));
        assert_eq!(p28.smart_resize(480, 640), (476, 644));
        // 1050 / 28 = 37.5 -> 38 under both rules.
        assert_eq!(p28.smart_resize(1050, 1680), (1064, 1680));
        assert_eq!(p28.smart_resize(700, 1050), (700, 1064));
        let p32 = factor_32_processor();
        assert_eq!(p32.smart_resize(1080, 1920), (1088, 1920));
        // 1200 / 32 = 37.5 -> 38 under both rules.
        assert_eq!(p32.smart_resize(1200, 1920), (1216, 1920));
    }

    #[test]
    fn owned_and_mlx_paths_share_spatial_merge_grouped_patch_order() {
        let processor = Qwen2VLProcessor::new(14, 2, 2);
        let mut image = RgbImage::new(56, 56);
        for patch_y in 0..4u32 {
            for patch_x in 0..4u32 {
                let value = (patch_y * 4 + patch_x) as u8 * 10;
                for y in patch_y * 14..(patch_y + 1) * 14 {
                    for x in patch_x * 14..(patch_x + 1) * 14 {
                        image.put_pixel(x, y, Rgb([value, 0, 0]));
                    }
                }
            }
        }
        let image = DynamicImage::ImageRgb8(image);
        let (values, grids) = processor.preprocess_values_with_grid(&[image]);
        assert_eq!(grids, vec![(1, 4, 4)]);
        let row_width = 3 * 14 * 14;
        let observed = (0..16)
            .map(|patch| values[patch * 2 * row_width])
            .collect::<Vec<_>>();
        let grouped_patch_ids = [0u8, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
        for (actual, patch_id) in observed.into_iter().zip(grouped_patch_ids) {
            let expected = (patch_id as f32 * 10.0 / 255.0 - processor.mean[0]) / processor.std[0];
            assert!((actual - expected).abs() < 1e-6);
        }
        let (mlx, mlx_grids) = processor.preprocess_with_grid(&[DynamicImage::new_rgb8(56, 56)]);
        assert_eq!(mlx_grids, grids);
        assert_eq!(mlxcel_core::array_shape(&mlx), vec![32, row_width as i32]);
    }

    /// Mage-VL is the first caller with `temporal_patch_size = 1`: each image
    /// must yield `t*h*w` rows of `3*P*P`, not the duplicated `2*t*h*w` the
    /// Qwen2-VL checkpoints get.
    #[test]
    fn qwen2_vl_processor_temporal_patch_1_row_count() {
        let processor = Qwen2VLProcessor::new_with_norm(16, 1, 2, [0.5; 3], [0.5; 3])
            .with_pixel_bounds(3136, 4_000_000);
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(224, 224, Rgb([255, 0, 0])));
        let (pixel_values, grids) = processor.preprocess_with_grid(&[image]);
        assert_eq!(grids, vec![(1, 14, 14)]);
        assert_eq!(
            mlxcel_core::array_shape(&pixel_values),
            vec![14 * 14, 3 * 16 * 16]
        );
        let (values, _) = processor.preprocess_values_with_grid(&[DynamicImage::ImageRgb8(
            RgbImage::from_pixel(224, 224, Rgb([255, 0, 0])),
        )]);
        // Row 0 starts with channel 0 (red, (1 - 0.5) / 0.5 = 1) and its
        // second 256-value block is channel 1 (green, -1).
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[256] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn video_processor_config_reads_qwen_sidecar_bounds() {
        let config = serde_json::json!({
            "fps": 1.5,
            "min_frames": 6,
            "max_frames": 128
        });
        let parsed = QwenVideoProcessorConfig::from_processor_config(Some(&config)).unwrap();
        assert_eq!(parsed.fps, 1.5);
        assert_eq!(parsed.min_frames, 6);
        assert_eq!(parsed.max_frames, 128);
    }

    #[test]
    fn video_processor_config_rejects_inverted_frame_bounds() {
        let config = serde_json::json!({
            "min_frames": 8,
            "max_frames": 4
        });
        let err = QwenVideoProcessorConfig::from_processor_config(Some(&config)).unwrap_err();
        assert!(err.contains("min_frames <= max_frames"), "{err}");
    }

    #[test]
    fn preprocess_video_pads_frames_to_temporal_patch_grid() {
        let processor =
            Qwen2VLProcessor::new(14, 2, 2).with_video_config(QwenVideoProcessorConfig {
                fps: 2.0,
                min_frames: 1,
                max_frames: 4,
            });
        let frames = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([10, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([20, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([30, 0, 0]))),
        ];
        let (pixel_values, grids) = processor
            .preprocess_media_with_grid(&[Qwen2VLMediaInput::Video(&frames)])
            .unwrap();

        assert_eq!(grids, vec![(2, 4, 4)]);
        assert_eq!(
            mlxcel_core::array_shape(&pixel_values),
            vec![64, 3 * 14 * 14]
        );
    }

    #[test]
    fn qwen3_checkpoint_policy_lifts_224_square_to_sixteen_by_sixteen_grid() {
        let processor = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3])
            .with_pixel_bounds(65_536, 16_777_216)
            .with_resize_filter(FilterType::CatmullRom);
        assert_eq!(
            processor.compute_grid_thw(&[DynamicImage::new_rgb8(224, 224)]),
            vec![(1, 16, 16)]
        );
    }

    /// Out of bounds, transformers' `smart_resize` (and mlx-vlm's Qwen3-VL
    /// `_smart_resize_image`) scales the original size by
    /// `beta = sqrt(h * w / bound)` and floors (or ceils) once. Expected
    /// values are transformers 5.24.1 `image_processing_pil_qwen2_vl.
    /// smart_resize`. No edge is an `x.5` tie with an even lower neighbour
    /// (2160 / 32 = 67.5 rounds to 68 either way), so tie rounding (#3463)
    /// does not enter.
    #[test]
    fn smart_resize_rescales_out_of_bounds_images_from_the_original_size() {
        // Qwen2.5-VL checkpoint bounds (3136..12845056, factor 28): a 5K
        // screenshot and an 18 MP photo.
        let p28 = Qwen2VLProcessor::new(14, 2, 2);
        assert_eq!(p28.smart_resize(2880, 5120), (2688, 4760));
        assert_eq!(p28.smart_resize(2690, 4782), (2688, 4760));
        assert_eq!(p28.smart_resize(3456, 5184), (2912, 4368));
        // The 1280-token cap (`max_pixels = 1003520`).
        let capped = Qwen2VLProcessor::new(14, 2, 2).with_pixel_bounds(3136, 1_003_520);
        assert_eq!(capped.smart_resize(772, 1372), (728, 1316));
        assert_eq!(capped.smart_resize(3024, 4032), (840, 1148));
        // Qwen3-VL bounds (65536..16777216, factor 32): small images grow
        // from the original size, a 24 MP photo shrinks.
        let p32 = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3])
            .with_pixel_bounds(65_536, 16_777_216);
        assert_eq!(p32.smart_resize(100, 177), (224, 352));
        assert_eq!(p32.smart_resize(120, 160), (224, 320));
        assert_eq!(p32.smart_resize(180, 320), (192, 352));
        assert_eq!(p32.smart_resize(4000, 6000), (3328, 4992));
        // Mage-VL bounds (3136..4000000, factor 32): a 4K screenshot is
        // 46 x 83 = 3818 merged tokens, as mlx-vlm counts it.
        let mage = Qwen2VLProcessor::new_with_norm(16, 1, 2, [0.5; 3], [0.5; 3])
            .with_pixel_bounds(3136, 4_000_000);
        assert_eq!(mage.smart_resize(2160, 3840), (1472, 2656));
        assert_eq!(mage.smart_resize(4000, 6000), (1632, 2432));
        assert_eq!(
            mage.compute_grid_thw(&[DynamicImage::new_rgb8(3840, 2160)]),
            vec![(1, 92, 166)]
        );
    }

    /// In-bounds images and out-of-bounds ones where both rules land on the
    /// same multiple must not move.
    #[test]
    fn smart_resize_in_bounds_and_agreeing_clamps_are_unchanged() {
        let p28 = Qwen2VLProcessor::new(14, 2, 2);
        assert_eq!(p28.smart_resize(1080, 1920), (1092, 1932));
        assert_eq!(p28.smart_resize(480, 640), (476, 644));
        assert_eq!(p28.smart_resize(4000, 6000), (2912, 4368));
        assert_eq!(p28.smart_resize(30, 50), (56, 84));
        let capped = Qwen2VLProcessor::new(14, 2, 2).with_pixel_bounds(3136, 1_003_520);
        assert_eq!(capped.smart_resize(1080, 1920), (728, 1316));
        let p32 = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3])
            .with_pixel_bounds(65_536, 16_777_216);
        assert_eq!(p32.smart_resize(1080, 1920), (1088, 1920));
        assert_eq!(p32.smart_resize(100, 100), (256, 256));
        assert_eq!(p32.smart_resize(200, 300), (224, 320));
        let mage = Qwen2VLProcessor::new_with_norm(16, 1, 2, [0.5; 3], [0.5; 3])
            .with_pixel_bounds(3136, 4_000_000);
        assert_eq!(mage.smart_resize(1800, 2880), (1568, 2528));
        assert_eq!(mage.smart_resize(1440, 2560), (1440, 2560));
    }

    /// Under 200:1 the reference result is kept even where the short side
    /// clamps to one factor and the area overshoots a small cap (here 64
    /// tokens). Past 200:1, where the reference raises, the long side is
    /// trimmed so the area stays under the cap.
    #[test]
    fn smart_resize_extreme_aspect_follows_the_reference_then_keeps_the_cap() {
        let tiny_cap = Qwen2VLProcessor::new(14, 2, 2).with_pixel_bounds(3136, 50_176);
        assert_eq!(tiny_cap.smart_resize(30, 3000), (28, 2240));
        let cap = 200_704; // 256 tokens
        let p = Qwen2VLProcessor::new(14, 2, 2).with_pixel_bounds(3136, cap);
        for (h, w) in [(1, 16_384), (16_384, 1), (40, 16_000)] {
            let (rh, rw) = p.smart_resize(h, w);
            assert_eq!((rh % 28, rw % 28), (0, 0), "{h}x{w}");
            assert!(
                (rh as usize) * (rw as usize) <= cap,
                "{h}x{w} -> {rh}x{rw} exceeds {cap}"
            );
        }
    }

    /// A zero edge must not turn `beta` infinite and saturate the other edge
    /// to `u32::MAX` (a multi-gigabyte resize); it is rescaled as one pixel.
    #[test]
    fn smart_resize_zero_edge_stays_bounded() {
        let p = Qwen2VLProcessor::new(14, 2, 2);
        assert_eq!(p.smart_resize(0, 10), (28, 196));
        assert_eq!(p.smart_resize(10, 0), (196, 28));
        assert_eq!(p.smart_resize(0, 0), (56, 56));
    }

    /// A thin edge rounds to 0 in the reference's first step, so the area
    /// falls under `min_pixels` and the original size is rescaled. Expected
    /// values are transformers 5.16.1 `image_processing_qwen2_vl.
    /// smart_resize`: `(10, 1000, 28, 3136, 12845056)` gives
    /// `beta = sqrt(3136 / 10000) = 0.56`, `ceil(10 * 0.56 / 28) = 1`,
    /// `ceil(1000 * 0.56 / 28) = 20`, so 28x560 (a one-factor floor on the
    /// first step kept 28x1008); `(10, 1000, 32, 65536, 16777216)` gives
    /// 32x2560.
    #[test]
    fn smart_resize_thin_edge_rescales_like_the_reference() {
        let p28 = Qwen2VLProcessor::new(14, 2, 2);
        assert_eq!(p28.smart_resize(10, 1000), (28, 560));
        assert_eq!(p28.smart_resize(1000, 10), (560, 28));
        let p32 = factor_32_processor();
        assert_eq!(p32.smart_resize(10, 1000), (32, 2560));
        assert_eq!(p32.smart_resize(1000, 10), (2560, 32));
        // `min_pixels == 0`: the reference returns a zero edge; one factor here.
        let p0 = Qwen2VLProcessor::new(14, 2, 2).with_pixel_bounds(0, 12_845_056);
        assert_eq!(p0.smart_resize(10, 1000), (28, 1008));
    }

    /// Past 200:1 a thin edge now takes the `min_pixels` branch, whose long
    /// side can still exceed a small cap; the trim must apply there too.
    #[test]
    fn smart_resize_extreme_aspect_under_min_pixels_keeps_the_cap() {
        let cap = 200_704;
        let p = Qwen2VLProcessor::new(14, 2, 2).with_pixel_bounds(3136, cap);
        for (h, w) in [(1, 100_000), (100_000, 1)] {
            let (rh, rw) = p.smart_resize(h, w);
            assert_eq!((rh % 28, rw % 28), (0, 0), "{h}x{w}");
            assert!((rh as usize) * (rw as usize) <= cap, "{h}x{w} -> {rh}x{rw}");
        }
    }
}
