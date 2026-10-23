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

//! CLI handler for `mlxcel predict` map, keypoint, and instance-mask models.

use std::io::Read;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

use mlxcel::initialize_runtime_checked;
use mlxcel::multimodal::video::{
    DEFAULT_FPS, DEFAULT_MAX_DECODED_VIDEO_BYTES, FrameSamplingPolicy, VideoLimits, VideoSource,
    load_video_source_with_policy_and_budget,
};
use mlxcel::vision::dense::{
    DenseFamily, DenseInput, DenseOutputInfo, FAMILIES, resolve_family, write_dense_output,
};

use crate::PredictArgs;

const DEFAULT_DENSE_MAX_FRAMES: usize = 32;

fn parse_opts(raw: Option<&str>) -> Result<Map<String, Value>> {
    let Some(raw) = raw else {
        return Ok(Map::new());
    };
    match serde_json::from_str::<Value>(raw).context("parse --opts JSON")? {
        Value::Object(map) => Ok(map),
        _ => bail!("--opts must be a JSON object"),
    }
}

fn parse_boxes(raw: &[String]) -> Result<Vec<[f32; 4]>> {
    let mut boxes = Vec::new();
    for group in raw {
        for value in group.split(';') {
            ensure!(!value.trim().is_empty(), "parse --boxes: empty box");
            let coords = value
                .split(',')
                .map(str::trim)
                .map(|coord| {
                    coord
                        .parse::<f32>()
                        .with_context(|| format!("parse --boxes coordinate `{coord}`"))
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                coords.len() == 4,
                "parse --boxes: expected four comma-separated coordinates in `{value}`"
            );
            ensure!(
                coords.iter().all(|coord| coord.is_finite()),
                "parse --boxes: coordinates must be finite in `{value}`"
            );
            ensure!(
                coords[2] > coords[0] && coords[3] > coords[1],
                "parse --boxes: right and bottom must exceed left and top in `{value}`"
            );
            boxes.push([coords[0], coords[1], coords[2], coords[3]]);
        }
    }
    Ok(boxes)
}

fn select_task<'a>(
    requested: Option<&'a str>,
    predictor: &'a dyn mlxcel::vision::dense::DensePredictor,
    input: &DenseInput,
) -> Result<&'a str> {
    let tasks = predictor.tasks();
    ensure!(!tasks.is_empty(), "dense predictor registered no tasks");
    if let Some(task) = requested {
        ensure!(
            tasks.contains(&task),
            "unsupported task `{task}`; choose one of: {}",
            tasks.join(", ")
        );
        return Ok(task);
    }
    let task = predictor.default_task(input).ok_or_else(|| {
        anyhow::anyhow!(
            "--task is required; this family supports: {}",
            tasks.join(", ")
        )
    })?;
    ensure!(
        tasks.contains(&task),
        "dense predictor default task `{task}` is not registered"
    );
    Ok(task)
}

fn load_predict_video(
    path: &std::path::Path,
    target_fps: Option<f64>,
    max_frames: Option<usize>,
    keep_all_without_target_fps: bool,
) -> Result<Vec<image::DynamicImage>> {
    let max_frames = max_frames.unwrap_or(DEFAULT_DENSE_MAX_FRAMES);
    ensure!(max_frames > 0, "load video: --max-frames must be positive");
    let source = VideoSource::from_path(path.to_path_buf());
    load_video_source_with_policy_and_budget(
        &source,
        target_fps,
        None,
        FrameSamplingPolicy {
            min_frames: 1,
            max_frames,
            frame_factor: 1,
            reject_over_max: false,
            keep_all_without_target_fps,
        },
        &VideoLimits::from_env(),
        Some(DEFAULT_MAX_DECODED_VIDEO_BYTES),
    )
    .with_context(|| format!("load video: {}", path.display()))
}

fn keep_all_video_frames(model_type: &str, task: Option<&str>) -> bool {
    model_type == "video_depth_anything"
        || (task == Some("track")
            && matches!(
                model_type,
                "sam3" | "sam3_video" | "sam3.1" | "sam3.1_video"
            ))
}

fn load_predict_image(path: &std::path::Path) -> Result<image::DynamicImage> {
    let limits = mlxcel::current_image_input_limits();
    let read_limit = limits
        .max_payload_bytes
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("load image: configured payload limit overflowed"))?;
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("load image: open {}", path.display()))?
        .take(read_limit as u64)
        .read_to_end(&mut bytes)
        .with_context(|| format!("load image: read {}", path.display()))?;
    ensure!(
        bytes.len() <= limits.max_payload_bytes,
        "load image: payload {} is {} bytes, exceeding the configured maximum of {}",
        path.display(),
        bytes.len(),
        limits.max_payload_bytes
    );
    let mut images = mlxcel::decode_image_payloads_with_limits(&[bytes], limits)
        .with_context(|| format!("load image: decode {}", path.display()))?;
    let image = images.pop().ok_or_else(|| {
        anyhow::anyhow!(
            "load image: decoder returned no image for {}",
            path.display()
        )
    })?;
    validate_dense_image_allocation(image.width(), image.height(), limits.max_decode_alloc_bytes)?;
    Ok(image)
}

fn validate_dense_image_allocation(width: u32, height: u32, max_bytes: u64) -> Result<()> {
    let expanded_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|channels| channels.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| anyhow::anyhow!("load image: dense RGB f32 allocation size overflowed"))?;
    ensure!(
        expanded_bytes <= max_bytes,
        "load image: dense RGB f32 expansion for {width}x{height} requires {expanded_bytes} bytes, exceeding the configured maximum of {max_bytes}"
    );
    Ok(())
}

pub(crate) fn run_predict(args: PredictArgs) -> Result<()> {
    run_predict_with(args, FAMILIES)
}

pub(crate) fn run_predict_with(args: PredictArgs, families: &[DenseFamily]) -> Result<()> {
    ensure!(
        args.model.is_dir(),
        "load model: directory does not exist: {}",
        args.model.display()
    );
    let source_count = usize::from(args.image.is_some()) + usize::from(args.video.is_some());
    ensure!(
        source_count == 1,
        "load input: exactly one of --image or --video is required"
    );
    if let Some(fps) = args.target_fps {
        ensure!(
            fps.is_finite() && fps > 0.0,
            "load video: --target-fps must be finite and positive"
        );
    }
    let mut opts = parse_opts(args.opts.as_deref())?;
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("write output: create {}", args.output.display()))?;

    let family =
        resolve_family(&args.model, families).context("load model: classify checkpoint")?;
    let has_prompts = !args.prompt.is_empty()
        || !args.boxes.is_empty()
        || opts.contains_key("prompts")
        || opts.contains_key("boxes");
    ensure!(
        !has_prompts || family.supports_prompts,
        "load model: dense-prediction family {} does not support --prompt or --boxes",
        family.model_type
    );
    ensure!(
        args.prompt.iter().all(|prompt| !prompt.trim().is_empty()),
        "parse --prompt: prompt text must not be empty"
    );
    let boxes = parse_boxes(&args.boxes)?;
    if !args.prompt.is_empty() {
        opts.insert(
            "prompts".into(),
            Value::Array(args.prompt.iter().cloned().map(Value::String).collect()),
        );
    }
    if !boxes.is_empty() {
        opts.insert("boxes".into(), serde_json::to_value(boxes)?);
    }
    let input = if let Some(path) = &args.image {
        ensure!(
            path.is_file(),
            "load image: file does not exist: {}",
            path.display()
        );
        DenseInput::Image(load_predict_image(path)?)
    } else {
        let path = args
            .video
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("load video: --video is required without --image"))?;
        ensure!(
            path.is_file(),
            "load video: file does not exist: {}",
            path.display()
        );
        let keep_all_without_target_fps =
            keep_all_video_frames(family.model_type, args.task.as_deref());
        let frames = load_predict_video(
            path,
            args.target_fps,
            args.max_frames,
            keep_all_without_target_fps,
        )?;
        DenseInput::Video {
            frames,
            fps: args.target_fps.unwrap_or(DEFAULT_FPS),
        }
    };
    let input_size = input
        .dimensions()
        .ok_or_else(|| anyhow::anyhow!("load input: video decoded no frames"))?;
    let is_video = input.is_video();

    let runtime = initialize_runtime_checked().context("load model: initialize runtime")?;
    super::generate::print_runtime_setup(&runtime);
    let predictor = (family.load)(&args.model).context("load model: construct predictor")?;
    let task = select_task(args.task.as_deref(), predictor.as_ref(), &input)?;
    opts.insert("task".into(), Value::String(task.into()));

    let started = Instant::now();
    let output = predictor
        .predict(&input, &opts)
        .with_context(|| format!("predict {task}"))?;
    for tensor in output.tensors.values() {
        mlxcel_core::eval(tensor);
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let info = DenseOutputInfo {
        family: predictor.family(),
        task,
        input_size,
        elapsed_ms,
        is_video,
    };
    let paths = write_dense_output(&args.output, &output, &info).context("write output")?;
    for path in paths {
        println!("saved {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
#[path = "predict_tests.rs"]
mod tests;
