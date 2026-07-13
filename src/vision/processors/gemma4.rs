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

//! Gemma4 image and video preprocessor.
//!
//! Used by: Gemma4 VLM
//!
//! The image side mirrors https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/processing_gemma4.py
//! :Gemma4ImageProcessor`. The video side ([`process_videos`] +
//! [`Gemma4VideoFeatures`]) mirrors `Gemma4VideoProcessor` from the
//! same file: per-frame uniform sampling, aspect-ratio-preserving resize
//! to a smaller per-frame token budget (default 70 soft tokens), rescale
//! to `[0, 1]`, and concatenate channel-first frames into one
//! `(N_total_frames, C, H, W)` tensor.

use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

/// Default soft-token budget per video frame. Matches upstream
/// `Gemma4VideoProcessor.max_soft_tokens` default of 70 (one of
/// `_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)`).
pub const DEFAULT_VIDEO_MAX_SOFT_TOKENS: usize = 70;

/// Default frame ceiling per video. Mirrors upstream
/// `Gemma4VideoProcessor.num_frames = 32`.
pub const DEFAULT_VIDEO_NUM_FRAMES: usize = 32;

/// Default sampling fps used to compute per-frame timestamps when the
/// caller does not supply per-video FPS values. Matches upstream
/// `default_fps = 2.0`.
pub const DEFAULT_VIDEO_FPS: f64 = 2.0;

/// Allowed soft-token budgets accepted by Gemma 4 preprocessing. Mirrors
/// upstream `_SUPPORTED_SOFT_TOKENS`.
///
/// Images and video frames intentionally share one ladder: the resize math is
/// the same function of the budget in both paths
/// ([`Gemma4Processor::aspect_ratio_preserving_resize_dims`] and
/// [`Gemma4Processor::video_resize_dims`] differ only in which budget they
/// read), and the vision tower is fully resolution-driven, so any budget that
/// is valid for a frame is valid for a still image. Keeping one ladder means a
/// per-request image override cannot land on a value the video path would
/// reject.
pub const SUPPORTED_SOFT_TOKENS: &[usize] = &[70, 140, 280, 560, 1120];

/// Allowed per-frame soft-token budgets accepted by Gemma 4 video
/// processing. Alias of [`SUPPORTED_SOFT_TOKENS`], kept under its original
/// name because `set_video_config` and its callers already reference it.
pub const SUPPORTED_VIDEO_SOFT_TOKENS: &[usize] = SUPPORTED_SOFT_TOKENS;

/// Allowed per-image soft-token budgets accepted by a per-request override.
/// Alias of [`SUPPORTED_SOFT_TOKENS`]; see that constant for why images and
/// video share the ladder.
pub const SUPPORTED_IMAGE_SOFT_TOKENS: &[usize] = SUPPORTED_SOFT_TOKENS;

/// Validate an untrusted per-request image soft-token budget against
/// [`SUPPORTED_IMAGE_SOFT_TOKENS`].
///
/// The budget arrives from the HTTP request body (or a CLI flag) and drives
/// the resize target: the preprocessed image, its patch grid, and the number
/// of soft tokens injected into the prompt all scale with it. An unbounded
/// value is therefore a memory-amplification vector, so anything outside the
/// ladder is rejected rather than clamped: a silent clamp would leave the
/// caller believing they got a budget they did not get.
///
/// # Errors
/// Returns `Err` with a message naming the supported values when
/// `max_soft_tokens` is not on the ladder.
pub fn validate_image_soft_tokens(max_soft_tokens: usize) -> Result<usize, String> {
    if SUPPORTED_IMAGE_SOFT_TOKENS.contains(&max_soft_tokens) {
        Ok(max_soft_tokens)
    } else {
        Err(format!(
            "image max_soft_tokens must be one of {SUPPORTED_IMAGE_SOFT_TOKENS:?}, got {max_soft_tokens}"
        ))
    }
}

/// Preprocessed output for a single image (or a single video frame) ready
/// for the Gemma 4 vision tower.
///
/// Mirrors the per-sample output of upstream `Gemma4ImageProcessor.__call__`:
/// a channel-first `[1, 3, H, W]` float32 tensor rescaled to `[0, 1]`,
/// the patch grid dimensions used to compute soft-token counts, and the
/// resulting soft-token count fed to the multimodal projector.
pub struct Gemma4ImageInput {
    /// Channel-first pixel tensor: shape `[1, 3, H, W]`, values in `[0, 1]`.
    pub pixel_values: UniquePtr<MlxArray>,
    /// `(patch_h, patch_w)` — number of patches along each spatial axis.
    pub patch_grid: (usize, usize),
    /// Number of soft tokens this image contributes to the token sequence.
    pub num_soft_tokens: usize,
}

/// Per-frame metadata for a single video processed by [`Gemma4Processor`].
///
/// One [`Gemma4VideoFeatures`] is emitted per input video. The Gemma 4
/// vision tower expects each frame as its own `[1, 3, H, W]` tensor (the
/// existing image path), so we reuse [`Gemma4ImageInput`] for the
/// per-frame output and stash video-level metadata (timestamps,
/// soft-token count, total frame count) on the wrapper.
pub struct Gemma4VideoFeatures {
    /// One [`Gemma4ImageInput`] per sampled frame. The vision tower runs
    /// on each of these independently, exactly as it does for static
    /// images.
    pub frames: Vec<Gemma4ImageInput>,
    /// Soft tokens emitted per frame (same value for every frame within
    /// a single video — frames share the resized H/W).
    pub num_soft_tokens_per_frame: usize,
    /// Per-frame timestamp (seconds since video start) used when the
    /// processor expands `<|video|>` placeholders into
    /// `MM:SS <boi>...<eoi>` runs.
    pub frame_timestamps: Vec<f64>,
}

impl Gemma4VideoFeatures {
    /// Number of frames sampled from this video.
    #[must_use]
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
}

/// Gemma 4 image and video preprocessor.
///
/// Encapsulates the parameters that govern both the static-image pipeline
/// (patch size, pooling kernel, soft-token budget) and the video pipeline
/// (per-frame token budget, frame ceiling). Construct via [`Self::new`] and
/// optionally tune video behaviour with [`Self::set_video_config`].
pub struct Gemma4Processor {
    /// Pixel size of each vision-tower patch (e.g. 14).
    pub patch_size: usize,
    /// Maximum soft tokens per image input. Drives aspect-ratio-preserving
    /// resize so the resulting patch grid never exceeds this budget.
    pub max_soft_tokens: usize,
    /// Spatial pooling kernel size applied after patch embedding (e.g. 2).
    pub pooling_kernel_size: usize,
    /// Pixel rescale factor (1/255). Applied element-wise after RGB conversion.
    pub rescale_factor: f32,
    /// Per-frame soft-token budget for video inputs. Smaller than
    /// `max_soft_tokens` because each video supplies many frames.
    pub video_max_soft_tokens: usize,
    /// Maximum frames sampled per video. The processor uniformly
    /// downsamples longer clips to this count.
    pub video_num_frames: usize,
}

impl Gemma4Processor {
    /// Create a new processor with image-pipeline defaults. Video defaults
    /// ([`DEFAULT_VIDEO_MAX_SOFT_TOKENS`], [`DEFAULT_VIDEO_NUM_FRAMES`]) are
    /// used until overridden by [`Self::set_video_config`].
    pub fn new(patch_size: usize, max_soft_tokens: usize, pooling_kernel_size: usize) -> Self {
        Self {
            patch_size,
            max_soft_tokens,
            pooling_kernel_size,
            rescale_factor: 1.0 / 255.0,
            video_max_soft_tokens: DEFAULT_VIDEO_MAX_SOFT_TOKENS,
            video_num_frames: DEFAULT_VIDEO_NUM_FRAMES,
        }
    }

    /// Override video processing defaults. Mirrors the constructor
    /// kwargs of upstream `Gemma4VideoProcessor`.
    ///
    /// # Errors
    /// Returns `Err` when `max_soft_tokens` is not in
    /// [`SUPPORTED_VIDEO_SOFT_TOKENS`].
    pub fn set_video_config(
        &mut self,
        max_soft_tokens: usize,
        num_frames: usize,
    ) -> Result<(), String> {
        if !SUPPORTED_VIDEO_SOFT_TOKENS.contains(&max_soft_tokens) {
            return Err(format!(
                "Gemma4 video max_soft_tokens must be one of {:?}, got {}",
                SUPPORTED_VIDEO_SOFT_TOKENS, max_soft_tokens
            ));
        }
        self.video_max_soft_tokens = max_soft_tokens;
        self.video_num_frames = num_frames.max(1);
        Ok(())
    }

    /// Preprocess a batch of static images at the checkpoint's configured
    /// soft-token budget.
    ///
    /// Each image is resized to the aspect-ratio-preserving target dimensions
    /// derived from [`Self::max_soft_tokens`], rescaled to `[0, 1]`, and
    /// packed into a `[1, 3, H, W]` channel-first tensor. Returns one
    /// [`Gemma4ImageInput`] per input image, in the same order.
    ///
    /// Equivalent to [`Self::preprocess_with_budget`] with `None`.
    pub fn preprocess(&self, images: &[DynamicImage]) -> Vec<Gemma4ImageInput> {
        self.preprocess_with_budget(images, None)
    }

    /// Preprocess a batch of static images under an optional per-call
    /// soft-token budget.
    ///
    /// `max_soft_tokens` shadows [`Self::max_soft_tokens`] for this call only;
    /// `None` uses the checkpoint's configured budget and is byte-identical to
    /// [`Self::preprocess`]. A larger budget resizes the image to a larger
    /// target, which yields a denser patch grid and therefore more soft tokens.
    ///
    /// Callers **must** derive the prompt's placeholder expansion from the
    /// `num_soft_tokens` on the returned [`Gemma4ImageInput`]s rather than from
    /// any independently computed budget. The vision tower emits exactly
    /// `num_soft_tokens` rows per image, so a placeholder count derived from a
    /// different budget would desync the prompt from the features and the model
    /// would attend to garbage.
    ///
    /// The budget is expected to already have passed
    /// [`validate_image_soft_tokens`] at the request boundary. A zero budget is
    /// floored to 1 so the resize math cannot divide the target area to nothing.
    pub fn preprocess_with_budget(
        &self,
        images: &[DynamicImage],
        max_soft_tokens: Option<usize>,
    ) -> Vec<Gemma4ImageInput> {
        images
            .iter()
            .map(|image| self.preprocess_single(image, max_soft_tokens))
            .collect()
    }

    /// Resolve the effective image soft-token budget for one call: the
    /// per-call override when present, otherwise the checkpoint's configured
    /// [`Self::max_soft_tokens`].
    fn effective_image_soft_tokens(&self, max_soft_tokens: Option<usize>) -> usize {
        max_soft_tokens.unwrap_or(self.max_soft_tokens).max(1)
    }

    /// Process one or more videos into Gemma 4 video features.
    ///
    /// Each input video is a list of decoded frames (`Vec<DynamicImage>`)
    /// — typically the output of [`crate::multimodal::video::load_video`].
    /// Frames are uniformly downsampled to at most [`Self::video_num_frames`],
    /// resized to fit the per-frame patch budget
    /// ([`Self::video_max_soft_tokens`]), rescaled to `[0, 1]`, and
    /// returned as a [`Gemma4VideoFeatures`] per video.
    ///
    /// `fps` is an optional per-video sampling rate used to compute
    /// per-frame timestamps. When `None`, [`DEFAULT_VIDEO_FPS`] is used
    /// for every video. When the slice is shorter than `videos.len()`,
    /// the last fps value is reused for the remaining videos. When
    /// `fps[i]` is non-positive, [`DEFAULT_VIDEO_FPS`] is substituted
    /// for that single video.
    pub fn process_videos(
        &self,
        videos: &[Vec<DynamicImage>],
        fps: Option<&[f64]>,
    ) -> Vec<Gemma4VideoFeatures> {
        videos
            .iter()
            .enumerate()
            .map(|(idx, frames)| {
                let sampling_fps = fps
                    .and_then(|s| {
                        if s.is_empty() {
                            None
                        } else {
                            s.get(idx).copied().or_else(|| s.last().copied())
                        }
                    })
                    .filter(|f| f.is_finite() && *f > 0.0)
                    .unwrap_or(DEFAULT_VIDEO_FPS);
                self.process_single_video(frames, sampling_fps)
            })
            .collect()
    }

    fn process_single_video(
        &self,
        frames: &[DynamicImage],
        sampling_fps: f64,
    ) -> Gemma4VideoFeatures {
        let sampled = uniform_sample_frames(frames, self.video_num_frames);
        let (target_h, target_w) = if let Some(first) = sampled.first() {
            let rgb = first.to_rgb8();
            self.video_resize_dims(rgb.height() as usize, rgb.width() as usize)
        } else {
            // Empty input — return a zero-frame feature bag. Downstream
            // call sites should already short-circuit on empty videos.
            (
                self.pooling_kernel_size * self.patch_size,
                self.pooling_kernel_size * self.patch_size,
            )
        };

        let mut frames_out = Vec::with_capacity(sampled.len());
        for frame in &sampled {
            frames_out.push(self.preprocess_video_frame(frame, target_h, target_w));
        }

        let num_soft_tokens_per_frame = if frames_out.is_empty() {
            0
        } else {
            frames_out[0].num_soft_tokens
        };
        let frame_timestamps: Vec<f64> = (0..frames_out.len())
            .map(|i| (i as f64) / sampling_fps.max(f64::EPSILON))
            .collect();

        Gemma4VideoFeatures {
            frames: frames_out,
            num_soft_tokens_per_frame,
            frame_timestamps,
        }
    }

    /// Resize and pack a single video frame to the shared target size
    /// chosen for the entire video. Cuts down on PIL round-trips by
    /// reusing the [`Self::preprocess_single`] machinery for the
    /// rescale + channel-first packing once we already know the target
    /// dimensions.
    fn preprocess_video_frame(
        &self,
        frame: &DynamicImage,
        target_h: usize,
        target_w: usize,
    ) -> Gemma4ImageInput {
        let rgb = frame.to_rgb8();
        let resized = if rgb.height() as usize == target_h && rgb.width() as usize == target_w {
            rgb
        } else {
            DynamicImage::ImageRgb8(rgb)
                .resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom)
                .to_rgb8()
        };

        let channels = 3usize;
        let mut data = vec![0.0f32; channels * target_h * target_w];
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..channels {
                    let dst = c * target_h * target_w + y * target_w + x;
                    data[dst] = pixel[c] as f32 * self.rescale_factor;
                }
            }
        }

        let patch_h = target_h / self.patch_size;
        let patch_w = target_w / self.patch_size;
        let num_soft_tokens = (patch_h * patch_w) / self.pooling_kernel_size.pow(2);

        Gemma4ImageInput {
            pixel_values: mlxcel_core::from_slice_f32(
                &data,
                &[1, channels as i32, target_h as i32, target_w as i32],
            ),
            patch_grid: (patch_h, patch_w),
            num_soft_tokens,
        }
    }

    /// Compute the target `(height, width)` for a single video given a
    /// reference frame's input dimensions. The math mirrors the image
    /// path's [`Self::aspect_ratio_preserving_resize_dims`] but uses
    /// [`Self::video_max_soft_tokens`] (smaller per-frame budget).
    fn video_resize_dims(&self, image_height: usize, image_width: usize) -> (usize, usize) {
        let max_patches = self.video_max_soft_tokens * self.pooling_kernel_size.pow(2);
        let target_px = max_patches as f64 * (self.patch_size * self.patch_size) as f64;
        let factor = (target_px / ((image_height * image_width).max(1) as f64)).sqrt();
        let side_mult = self.pooling_kernel_size * self.patch_size;
        let mut target_h =
            ((factor * image_height as f64 / side_mult as f64).floor() as usize) * side_mult;
        let mut target_w =
            ((factor * image_width as f64 / side_mult as f64).floor() as usize) * side_mult;

        let max_side_length = (max_patches / self.pooling_kernel_size.pow(2)).max(1) * side_mult;
        if target_h == 0 && target_w == 0 {
            target_h = side_mult;
            target_w = side_mult;
        } else if target_h == 0 {
            target_h = side_mult;
            target_w =
                ((image_width / image_height.max(1)).max(1) * side_mult).min(max_side_length);
        } else if target_w == 0 {
            target_w = side_mult;
            target_h =
                ((image_height / image_width.max(1)).max(1) * side_mult).min(max_side_length);
        }
        (target_h, target_w)
    }

    fn preprocess_single(
        &self,
        image: &DynamicImage,
        max_soft_tokens: Option<usize>,
    ) -> Gemma4ImageInput {
        let rgb = image.to_rgb8();
        let (target_h, target_w) = self.aspect_ratio_preserving_resize_dims(
            rgb.height() as usize,
            rgb.width() as usize,
            max_soft_tokens,
        );

        let resized = if rgb.height() as usize == target_h && rgb.width() as usize == target_w {
            rgb
        } else {
            DynamicImage::ImageRgb8(rgb)
                .resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom)
                .to_rgb8()
        };

        let channels = 3usize;
        let mut data = vec![0.0f32; channels * target_h * target_w];
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for c in 0..channels {
                    let dst = c * target_h * target_w + y * target_w + x;
                    data[dst] = pixel[c] as f32 * self.rescale_factor;
                }
            }
        }

        let patch_h = target_h / self.patch_size;
        let patch_w = target_w / self.patch_size;
        let num_soft_tokens = (patch_h * patch_w) / self.pooling_kernel_size.pow(2);

        Gemma4ImageInput {
            pixel_values: mlxcel_core::from_slice_f32(
                &data,
                &[1, channels as i32, target_h as i32, target_w as i32],
            ),
            patch_grid: (patch_h, patch_w),
            num_soft_tokens,
        }
    }

    /// Target `(height, width)` for one image under an optional per-call
    /// soft-token budget. `max_soft_tokens = None` uses the checkpoint's
    /// configured [`Self::max_soft_tokens`], which reproduces the pre-override
    /// behavior exactly.
    fn aspect_ratio_preserving_resize_dims(
        &self,
        image_height: usize,
        image_width: usize,
        max_soft_tokens: Option<usize>,
    ) -> (usize, usize) {
        let budget = self.effective_image_soft_tokens(max_soft_tokens);
        let max_patches = budget * self.pooling_kernel_size.pow(2);
        let target_px = max_patches as f64 * (self.patch_size * self.patch_size) as f64;
        let factor = (target_px / ((image_height * image_width).max(1) as f64)).sqrt();
        let side_mult = self.pooling_kernel_size * self.patch_size;

        let mut target_h =
            ((factor * image_height as f64 / side_mult as f64).floor() as usize) * side_mult;
        let mut target_w =
            ((factor * image_width as f64 / side_mult as f64).floor() as usize) * side_mult;

        let max_side_length = (max_patches / self.pooling_kernel_size.pow(2)).max(1) * side_mult;

        if target_h == 0 && target_w == 0 {
            target_h = side_mult;
            target_w = side_mult;
        } else if target_h == 0 {
            target_h = side_mult;
            target_w =
                ((image_width / image_height.max(1)).max(1) * side_mult).min(max_side_length);
        } else if target_w == 0 {
            target_w = side_mult;
            target_h =
                ((image_height / image_width.max(1)).max(1) * side_mult).min(max_side_length);
        }

        (target_h, target_w)
    }
}

/// Uniformly subsample `frames` down to at most `target` items.
///
/// Mirrors the upstream `Gemma4VideoProcessor._sample_frames`
/// implementation: returns `frames` as-is when its length is at most
/// `target`; otherwise picks `target` linearly-spaced indices using
/// `linspace(0, T-1, target).round()`.
fn uniform_sample_frames(frames: &[DynamicImage], target: usize) -> Vec<DynamicImage> {
    let total = frames.len();
    if total == 0 {
        return Vec::new();
    }
    if total <= target {
        return frames.to_vec();
    }
    if target <= 1 {
        return vec![frames[0].clone()];
    }
    let last = (total - 1) as f64;
    let step = last / (target as f64 - 1.0);
    (0..target)
        .map(|i| {
            let idx = (i as f64 * step).round() as usize;
            frames[idx.min(total - 1)].clone()
        })
        .collect()
}

#[cfg(test)]
#[path = "gemma4_tests.rs"]
mod tests;
