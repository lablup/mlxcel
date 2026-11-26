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

//! Video-prompted arm of `logit_trace` (#3234).
//!
//! A change to how video tokens are positioned moves only text conditioned on
//! a video. With `MLXCEL_TRACE_VIDEO=PATH` the harness samples the clip the
//! way the Qwen-VL video path of `mlxcel generate --video` does (the
//! processor's sampling policy, at `MLXCEL_TRACE_VIDEO_FPS` or the processor's
//! default rate), renders the chat template around one video and
//! `MLXCEL_TRACE_VIDEO_PROMPT` (default `Describe the video.`), prepares the
//! embeddings through `vlm_runtime::compute_qwen_vl_media_embeddings` with the
//! clip's `second_per_grid`, and teacher-forces the first `CHUNK_TOKENS`
//! tokens of `TEXT_FILE` as the reply, exactly as the image arm does.
//!
//! Two switches make the comparison against a reference possible:
//!
//! - `MLXCEL_TRACE_VIDEO_FRAMES_DIR=DIR` writes the sampled frames there as
//!   `frame_0000.png`, ... so the reference arm
//!   (`scripts/mlx_vlm_image_logit_trace.py --video`) encodes the same frames
//!   instead of sampling the clip a second time. The trace's `# video_timing`
//!   line carries the sampled fps to hand it as `--video-fps`.
//! - `MLXCEL_TRACE_VIDEO_UNTIMED=1` withholds the timing, which places the
//!   video's temporal indices one position apart: the positions every
//!   Qwen-VL family other than Qwen2.5-VL uses, and the ones Qwen2.5-VL used
//!   before #3234. It is the reverted arm of that change.
//!
//! Only the flat video layout is covered (one run per video: Qwen2-VL,
//! Qwen2.5-VL, Qwen4-exp-VL, MiMo-V2-VL). The Qwen3-VL family renders one
//! timestamped run per temporal group and is refused.

use anyhow::{Context, Result};
use mlxcel::tokenizer::MlxcelTokenizer;
use mlxcel::video::{self, VideoLimits, VideoSource};

use super::image_arm::{self, ImageTrace};

/// Default user prompt when `MLXCEL_TRACE_VIDEO_PROMPT` is unset.
const DEFAULT_PROMPT: &str = "Describe the video.";

/// The video arm's settings, read from the environment.
pub struct VideoArm {
    pub video_path: String,
    pub prompt: String,
    /// `MLXCEL_TRACE_VIDEO_FPS`; `None` uses the processor's default rate.
    pub fps: Option<f64>,
    /// `MLXCEL_TRACE_VIDEO_FRAMES_DIR`: where to write the sampled frames.
    pub frames_dir: Option<String>,
    /// `MLXCEL_TRACE_VIDEO_UNTIMED=1`: withhold `second_per_grid`.
    pub untimed: bool,
}

impl VideoArm {
    /// `Some` when `MLXCEL_TRACE_VIDEO` names a clip.
    pub fn from_env() -> Result<Option<Self>> {
        let Ok(video_path) = std::env::var("MLXCEL_TRACE_VIDEO") else {
            return Ok(None);
        };
        let prompt = std::env::var("MLXCEL_TRACE_VIDEO_PROMPT")
            .unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
        let fps = parse_fps(std::env::var("MLXCEL_TRACE_VIDEO_FPS").ok().as_deref())?;
        let frames_dir = std::env::var("MLXCEL_TRACE_VIDEO_FRAMES_DIR").ok();
        let untimed = std::env::var("MLXCEL_TRACE_VIDEO_UNTIMED").as_deref() == Ok("1");
        Ok(Some(Self {
            video_path,
            prompt,
            fps,
            frames_dir,
            untimed,
        }))
    }
}

/// A positive finite rate; unset (or empty) keeps the processor default.
fn parse_fps(value: Option<&str>) -> Result<Option<f64>> {
    match value.map(str::trim) {
        None | Some("") => Ok(None),
        Some(raw) => {
            let fps: f64 = raw
                .parse()
                .with_context(|| format!("MLXCEL_TRACE_VIDEO_FPS must be a number, got {raw:?}"))?;
            anyhow::ensure!(
                fps.is_finite() && fps > 0.0,
                "MLXCEL_TRACE_VIDEO_FPS must be positive and finite, got {raw:?}"
            );
            Ok(Some(fps))
        }
    }
}

/// One video-prompted continuation and the timing of the clip behind it.
pub struct VideoTrace {
    pub trace: ImageTrace,
    pub sampled_frames: usize,
    pub total_frames: usize,
    pub source_fps: f64,
    pub sampled_fps: f64,
    pub second_per_grid: f32,
}

/// Run the video-prompted arm over the first `max_tokens` tokens of `text`.
pub fn trace(
    arm: &VideoArm,
    loaded: &mlxcel::LoadedModel,
    tokenizer: &MlxcelTokenizer,
    model_dir: &str,
    text: &str,
    max_tokens: usize,
) -> Result<VideoTrace> {
    let qwen = mlxcel::vlm_runtime::qwen_video_runtime(loaded)
        .context("MLXCEL_TRACE_VIDEO needs a Qwen-VL family with video input")?;
    let info = qwen.prompt_info();
    anyhow::ensure!(
        !info.video_timestamps,
        "MLXCEL_TRACE_VIDEO covers the flat video layout only; this family renders per-group \
         timestamps"
    );

    // Sample as `server::model_worker::prepare_qwen_vl_video_embeddings` does.
    let canonical = std::fs::canonicalize(&arm.video_path)
        .with_context(|| format!("resolving video {}", arm.video_path))?;
    let source = VideoSource::from_path(canonical);
    let fps = arm.fps.unwrap_or(info.processor.video_default_fps);
    let sampled = video::load_video_source_sampled_with_policy(
        &source,
        Some(fps),
        None,
        info.processor.video_sampling_policy(),
        &VideoLimits::from_env(),
    )
    .with_context(|| format!("loading video {}", arm.video_path))?;
    let sampled_fps = sampled.sampled_fps()?;
    let second_per_grid = mlxcel::qwen_vl::qwen_video_second_per_grid(
        sampled_fps,
        info.processor.temporal_patch_size,
    )?;
    if let Some(dir) = &arm.frames_dir {
        write_frames(dir, &sampled.frames)?;
    }

    let prompt = image_arm::render_prompt(model_dir, tokenizer, &arm.prompt, None, "video");
    let add_special = !tokenizer.prompt_carries_bos(&prompt);
    let mut prompt_ids = image_arm::encode(tokenizer, &prompt, add_special);
    let timing = [second_per_grid];
    let sampled_frames = sampled.frames.len();
    let (embeddings, _stats) = mlxcel::vlm_runtime::compute_qwen_vl_media_embeddings(
        qwen,
        &mut prompt_ids,
        &[],
        std::slice::from_ref(&sampled.frames),
        None,
        (!arm.untimed).then_some(&timing[..]),
        None,
        None,
    )?;

    // By-id forwards of a scheduler-owned family go through a pool (#3254).
    let model = &mlxcel::models::pooled_model::Pooled::new(loaded);
    let trace = image_arm::trace_prepared(
        model,
        tokenizer,
        prompt_ids,
        &embeddings,
        text,
        max_tokens,
        None,
    )?;
    Ok(VideoTrace {
        trace,
        sampled_frames,
        total_frames: sampled.total_frames,
        source_fps: sampled.source_fps,
        sampled_fps,
        second_per_grid,
    })
}

/// Write the sampled frames as `frame_NNNN.png`, in order, into `dir`.
fn write_frames(dir: &str, frames: &[image::DynamicImage]) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {dir}"))?;
    for (index, frame) in frames.iter().enumerate() {
        let path = std::path::Path::new(dir).join(format!("frame_{index:04}.png"));
        frame
            .to_rgb8()
            .save(&path)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_fps;

    #[test]
    fn fps_accepts_a_positive_rate_and_unset() {
        assert_eq!(parse_fps(None).unwrap(), None);
        assert_eq!(parse_fps(Some("")).unwrap(), None);
        assert_eq!(parse_fps(Some(" 2 ")).unwrap(), Some(2.0));
        assert_eq!(parse_fps(Some("0.5")).unwrap(), Some(0.5));
    }

    #[test]
    fn fps_rejects_non_positive_and_non_numeric_values() {
        for raw in ["0", "-1", "nan", "inf", "fast"] {
            assert!(parse_fps(Some(raw)).is_err(), "{raw:?} must be rejected");
        }
    }
}
