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

//! Clef video sampling and frame geometry.
//!
//! Upstream's `Clef._sequence` samples a clip from every source frame with
//! `count = int(total / 24 * fps)` clamped to the video processor's
//! `[min_frames, max_frames]` and to `total`, takes the frames at
//! `linspace(0, total - 1, count).round()`, and stamps them as a 24 fps
//! clip, so the group timestamps are `index / 24` whatever the source rate.
//! The Qwen3-VL video processor then resizes every frame to one size chosen
//! from the clip's total pixel budget (`smart_resize` of
//! `video_processing_qwen3_vl.py`), which differs from the per-image rule.

use image::DynamicImage;
use image::imageops::FilterType;

use super::DecisionError;

/// The frame rate upstream stamps every sampled clip with.
pub(crate) const CLIP_FPS: f64 = 24.0;

/// Most decoded bytes (four per source pixel) the videos of one request may
/// take. 768 frames, the processor's frame cap, fit at 1280 x 1080; a
/// 4096 x 4096 source (the decoder's resolution cap) fits 64 frames.
pub(crate) const MAX_DECODED_VIDEO_BYTES: usize = 4 << 30;

/// The video processor's sampling and pixel bounds from
/// `processor_config.json` `video_processor`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VideoSettings {
    /// Target sampling rate (`fps`).
    pub(crate) fps: f64,
    /// `min_frames`.
    pub(crate) min_frames: usize,
    /// `max_frames`.
    pub(crate) max_frames: usize,
    /// Clip pixel floor (`min_pixels`, the `shortest_edge` of `size`).
    pub(crate) min_pixels: usize,
    /// Clip pixel budget (`max_pixels`, the `longest_edge` of `size`).
    pub(crate) max_pixels: usize,
}

impl VideoSettings {
    /// Transformers' Qwen3-VL video processor defaults.
    pub(crate) const DEFAULT: Self = Self {
        fps: 2.0,
        min_frames: 4,
        max_frames: 768,
        min_pixels: 128 * 32 * 32,
        max_pixels: 32 * 32 * 768,
    };

    /// Read `video_processor` from a parsed `processor_config.json`; absent
    /// keys keep [`Self::DEFAULT`].
    pub(crate) fn from_processor_config(config: Option<&serde_json::Value>) -> Self {
        let section = config.and_then(|c| c.get("video_processor"));
        let count = |key: &str, default: usize| {
            section
                .and_then(|s| s.get(key))
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .unwrap_or(default)
        };
        let size = section.and_then(|s| s.get("size"));
        let edge = |key: &str| {
            size.and_then(|s| s.get(key))
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
        };
        let fps = section
            .and_then(|s| s.get("fps"))
            .and_then(serde_json::Value::as_f64)
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .unwrap_or(Self::DEFAULT.fps);
        let min_frames = count("min_frames", Self::DEFAULT.min_frames);
        Self {
            fps,
            min_frames,
            max_frames: count("max_frames", Self::DEFAULT.max_frames).max(min_frames),
            min_pixels: edge("shortest_edge")
                .unwrap_or_else(|| count("min_pixels", Self::DEFAULT.min_pixels)),
            max_pixels: edge("longest_edge")
                .unwrap_or_else(|| count("max_pixels", Self::DEFAULT.max_pixels)),
        }
    }
}

/// Upstream's frame count for a clip of `total` source frames.
pub(crate) fn frame_count(total: usize, settings: &VideoSettings) -> usize {
    // Python's `int(total / 24 * fps)` truncates toward zero.
    let count = (total as f64 / CLIP_FPS * settings.fps) as usize;
    count
        .max(settings.min_frames)
        .min(settings.max_frames)
        .min(total)
}

/// `np.linspace(0, total - 1, count).round()` (half to even, as NumPy).
pub(crate) fn linspace_indices(total: usize, count: usize) -> Vec<usize> {
    if count == 0 || total == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0];
    }
    let stop = (total - 1) as f64;
    let step = stop / (count - 1) as f64;
    (0..count)
        .map(|index| {
            let value = if index == count - 1 {
                stop
            } else {
                index as f64 * step
            };
            value.round_ties_even() as usize
        })
        .collect()
}

/// `smart_resize` of transformers' Qwen3-VL video processor: one
/// `(height, width)` for every frame of a `num_frames` clip.
pub(crate) fn video_frame_size(
    num_frames: usize,
    height: u32,
    width: u32,
    temporal_factor: usize,
    factor: usize,
    settings: &VideoSettings,
) -> Result<(u32, u32), DecisionError> {
    let invalid = |message: String| DecisionError::InvalidInput(message);
    if num_frames < temporal_factor {
        return Err(invalid(format!(
            "a video needs at least {temporal_factor} frames, got {num_frames}"
        )));
    }
    let factor_f = factor as f64;
    let (mut height, mut width) = (f64::from(height), f64::from(width));
    if height < factor_f || width < factor_f {
        let scale = (factor_f / height).max(factor_f / width);
        height = (height * scale).trunc();
        width = (width * scale).trunc();
    }
    if height.max(width) / height.min(width) > 200.0 {
        return Err(invalid(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            height.max(width) / height.min(width)
        )));
    }
    let frames = num_frames as f64;
    let mut h_bar = (height / factor_f).round_ties_even() * factor_f;
    let mut w_bar = (width / factor_f).round_ties_even() * factor_f;
    let t_bar = (frames / temporal_factor as f64).round_ties_even() * temporal_factor as f64;
    if t_bar * h_bar * w_bar > settings.max_pixels as f64 {
        let beta = ((frames * height * width) / settings.max_pixels as f64).sqrt();
        h_bar = factor_f.max((height / beta / factor_f).floor() * factor_f);
        w_bar = factor_f.max((width / beta / factor_f).floor() * factor_f);
    } else if t_bar * h_bar * w_bar < settings.min_pixels as f64 {
        let beta = (settings.min_pixels as f64 / (frames * height * width)).sqrt();
        h_bar = (height * beta / factor_f).ceil() * factor_f;
        w_bar = (width * beta / factor_f).ceil() * factor_f;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// Resize every frame to `(height, width)` (bicubic, as the processor).
pub(crate) fn resize_frames(frames: &[DynamicImage], height: u32, width: u32) -> Vec<DynamicImage> {
    frames
        .iter()
        .map(|frame| {
            if frame.height() == height && frame.width() == width {
                frame.clone()
            } else {
                frame.resize_exact(width, height, FilterType::CatmullRom)
            }
        })
        .collect()
}
