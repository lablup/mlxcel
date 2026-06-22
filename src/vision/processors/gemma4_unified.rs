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

//! Gemma 4 Unified (`gemma4_unified`) image and audio front-end processor.
//!
//! Encoder-free: there is no ViT and no Conformer. Images are turned into flat
//! `model_patch_size² · 3` patch vectors (one per soft token) plus a 2-D
//! `(x, y)` grid position, padded to `num_soft_tokens` with position `-1`.
//! Audio is the raw waveform chunked into `audio_samples_per_token`-sample
//! frames plus a validity mask. Both are projected into the language-model
//! hidden space downstream by the shared multimodal embedder.

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// Default soft-token budget per video frame. Mirrors upstream
/// `Gemma4UnifiedConfig.vision_soft_tokens_per_video_frame` default of 70
/// (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4_unified/config.py#L101).
/// Video frames carry a much smaller per-frame budget than still images
/// (`num_soft_tokens`, 280) because a clip supplies many frames.
pub const DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME: usize = 70;

/// One preprocessed image for the Gemma 4 Unified vision embedder.
pub struct Gemma4UnifiedImageInput {
    /// Flat patch matrix: `[num_soft_tokens, model_patch_size² · 3]` float32.
    /// Padding rows (beyond the real patch count) are zeros.
    pub patches: UniquePtr<MlxArray>,
    /// `(x, y)` grid position per patch: `[num_soft_tokens, 2]` int32. Padding
    /// patches carry `-1` on both axes.
    pub positions: UniquePtr<MlxArray>,
    /// Number of real (non-padding) patches == soft tokens for this image.
    pub num_soft_tokens: usize,
}

/// Audio features chunked from a raw waveform.
pub struct Gemma4UnifiedAudioInput {
    /// `[num_frames, audio_samples_per_token]` float32 waveform frames.
    pub features: UniquePtr<MlxArray>,
    /// `[num_frames]` bool validity mask (`true` = real frame, `false` = pad).
    pub mask: UniquePtr<MlxArray>,
    /// Number of frames == audio soft tokens.
    pub num_frames: usize,
}

/// Gemma 4 Unified image + audio processor.
pub struct Gemma4UnifiedProcessor {
    /// Side length (pixels) of each square patch fed to the projector.
    pub model_patch_size: usize,
    /// Maximum soft tokens (patches) per image.
    pub num_soft_tokens: usize,
    /// Maximum soft tokens (patches) per video frame. Smaller than
    /// `num_soft_tokens` because a clip supplies many frames. Defaults to
    /// [`DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME`]; the loader overrides it from
    /// `vision_soft_tokens_per_video_frame` in the checkpoint config.
    pub video_soft_tokens_per_frame: usize,
    /// Raw samples per audio frame/token.
    pub audio_samples_per_token: usize,
    /// Pixel rescale factor (1/255).
    pub rescale_factor: f32,
}

impl Gemma4UnifiedProcessor {
    pub fn new(
        model_patch_size: usize,
        num_soft_tokens: usize,
        audio_samples_per_token: usize,
    ) -> Self {
        Self {
            model_patch_size: model_patch_size.max(1),
            num_soft_tokens: num_soft_tokens.max(1),
            video_soft_tokens_per_frame: DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME,
            audio_samples_per_token: audio_samples_per_token.max(1),
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// Override the per-video-frame soft-token budget (config-driven). Clamped
    /// to at least 1 so the patch loop always emits a row.
    pub fn set_video_soft_tokens_per_frame(&mut self, value: usize) {
        self.video_soft_tokens_per_frame = value.max(1);
    }

    /// Flat patch dimension (`model_patch_size² · 3`).
    #[inline]
    pub fn patch_dim(&self) -> usize {
        self.model_patch_size * self.model_patch_size * 3
    }

    /// Preprocess a batch of images.
    pub fn preprocess(&self, images: &[DynamicImage]) -> Vec<Gemma4UnifiedImageInput> {
        images
            .iter()
            .map(|image| self.preprocess_single(image))
            .collect()
    }

    /// Preprocess decoded video frames into per-frame patch inputs.
    ///
    /// Video is "images per frame": each decoded frame runs through the same
    /// encoder-free patchify as a still image, but the per-frame soft-token
    /// budget is capped at [`Self::video_soft_tokens_per_frame`] (70 by
    /// default, from `vision_soft_tokens_per_video_frame`) rather than the
    /// larger image budget [`Self::num_soft_tokens`]. Each returned
    /// [`Gemma4UnifiedImageInput`] therefore has its `patches`/`positions`
    /// padded to `video_soft_tokens_per_frame` rows with `num_soft_tokens`
    /// equal to the real patch count for that frame.
    pub fn preprocess_video_frames(&self, frames: &[DynamicImage]) -> Vec<Gemma4UnifiedImageInput> {
        frames
            .iter()
            .map(|frame| self.patchify(frame, self.video_soft_tokens_per_frame))
            .collect()
    }

    fn preprocess_single(&self, image: &DynamicImage) -> Gemma4UnifiedImageInput {
        self.patchify(image, self.num_soft_tokens)
    }

    /// Patchify one image (or video frame) into flat patch vectors plus grid
    /// positions, capped and padded to `soft_token_cap` rows. Shared by the
    /// image ([`Self::preprocess_single`]) and video
    /// ([`Self::preprocess_video_frames`]) paths so the patch loop lives in one
    /// place; only the soft-token budget differs.
    fn patchify(&self, image: &DynamicImage, soft_token_cap: usize) -> Gemma4UnifiedImageInput {
        let soft_token_cap = soft_token_cap.max(1);
        let rgb = image.to_rgb8();
        let (target_h, target_w) =
            self.resize_dims(rgb.height() as usize, rgb.width() as usize, soft_token_cap);

        let resized = if rgb.height() as usize == target_h && rgb.width() as usize == target_w {
            rgb
        } else {
            DynamicImage::ImageRgb8(rgb)
                .resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom)
                .to_rgb8()
        };

        let ps = self.model_patch_size;
        let grid_h = target_h / ps;
        let grid_w = target_w / ps;
        let real_patches = (grid_h * grid_w).min(soft_token_cap);
        let patch_dim = self.patch_dim();

        // Pad row count up to soft_token_cap; pad rows stay zero and get
        // position -1 so the embedder contributes zero for them.
        let mut patch_data = vec![0.0f32; soft_token_cap * patch_dim];
        let mut positions = vec![-1i32; soft_token_cap * 2];

        let mut patch_idx = 0usize;
        'outer: for py in 0..grid_h {
            for px in 0..grid_w {
                if patch_idx >= soft_token_cap {
                    break 'outer;
                }
                // Fill this patch's flat vector in (y, x, c) row-major order:
                // [model_patch_size, model_patch_size, 3].
                let dst_base = patch_idx * patch_dim;
                for ly in 0..ps {
                    let src_y = py * ps + ly;
                    for lx in 0..ps {
                        let src_x = px * ps + lx;
                        let pixel = resized.get_pixel(src_x as u32, src_y as u32);
                        let cell = dst_base + (ly * ps + lx) * 3;
                        patch_data[cell] = pixel[0] as f32 * self.rescale_factor;
                        patch_data[cell + 1] = pixel[1] as f32 * self.rescale_factor;
                        patch_data[cell + 2] = pixel[2] as f32 * self.rescale_factor;
                    }
                }
                positions[patch_idx * 2] = px as i32;
                positions[patch_idx * 2 + 1] = py as i32;
                patch_idx += 1;
            }
        }

        Gemma4UnifiedImageInput {
            patches: mlxcel_core::from_slice_f32(
                &patch_data,
                &[soft_token_cap as i32, patch_dim as i32],
            ),
            positions: mlxcel_core::from_slice_i32(&positions, &[soft_token_cap as i32, 2]),
            num_soft_tokens: real_patches,
        }
    }

    /// Aspect-ratio-preserving resize so the number of `model_patch_size`
    /// patches stays at or below `max_patches` and each side is a multiple
    /// of `model_patch_size`.
    fn resize_dims(
        &self,
        image_height: usize,
        image_width: usize,
        max_patches: usize,
    ) -> (usize, usize) {
        let ps = self.model_patch_size;
        let target_px = max_patches as f64 * (ps * ps) as f64;
        let factor = (target_px / ((image_height * image_width).max(1) as f64)).sqrt();

        let mut target_h = ((factor * image_height as f64 / ps as f64).floor() as usize) * ps;
        let mut target_w = ((factor * image_width as f64 / ps as f64).floor() as usize) * ps;

        let max_side = max_patches.max(1) * ps;
        if target_h == 0 && target_w == 0 {
            target_h = ps;
            target_w = ps;
        } else if target_h == 0 {
            target_h = ps;
            target_w = ((image_width / image_height.max(1)).max(1) * ps).min(max_side);
        } else if target_w == 0 {
            target_w = ps;
            target_h = ((image_height / image_width.max(1)).max(1) * ps).min(max_side);
        }

        // Final guard: shrink the larger axis until grid_h * grid_w <=
        // num_soft_tokens (the area bound can be exceeded by rounding).
        while (target_h / ps) * (target_w / ps) > max_patches {
            if target_h >= target_w && target_h > ps {
                target_h -= ps;
            } else if target_w > ps {
                target_w -= ps;
            } else {
                break;
            }
        }

        (target_h, target_w)
    }

    /// Chunk a raw waveform into `audio_samples_per_token`-sample frames.
    ///
    /// The trailing partial frame is zero-padded to a full `audio_samples_per_token`
    /// vector and is treated as a real audio soft token — the checkpoint is built
    /// expecting exactly `num_frames = ceil(samples / audio_samples_per_token)` soft
    /// tokens, including the zero-padded tail. All frames in the returned mask are
    /// therefore marked valid (`true`). The placeholder count, the mask length, and
    /// the projected-feature count all equal `num_frames`.
    pub fn process_audio(&self, samples: &[f32]) -> Gemma4UnifiedAudioInput {
        let frame = self.audio_samples_per_token;
        let num_frames = samples.len().div_ceil(frame).max(1);
        let mut data = vec![0.0f32; num_frames * frame];

        for f in 0..num_frames {
            let start = f * frame;
            if start >= samples.len() {
                break;
            }
            let end = (start + frame).min(samples.len());
            data[start..start + (end - start)].copy_from_slice(&samples[start..end]);
            // Any remaining bytes in this frame slot stay at 0.0 (zero-pad).
        }

        // Every frame slot — including the zero-padded trailing partial — is a
        // real audio soft token and is marked valid.
        let mask_i32 = vec![1i32; num_frames];

        Gemma4UnifiedAudioInput {
            features: mlxcel_core::from_slice_f32(&data, &[num_frames as i32, frame as i32]),
            mask: {
                let arr = mlxcel_core::from_slice_i32(&mask_i32, &[num_frames as i32]);
                mlxcel_core::astype(&arr, mlxcel_core::dtype::BOOL)
            },
            num_frames,
        }
    }

    /// Number of audio frames (== soft tokens) a waveform of `len` samples
    /// produces. Mirrors [`Self::process_audio`]'s frame count.
    pub fn audio_num_frames(&self, len: usize) -> usize {
        len.div_ceil(self.audio_samples_per_token).max(1)
    }
}

#[cfg(test)]
#[path = "gemma4_unified_tests.rs"]
mod tests;
