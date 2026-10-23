// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::Parser;
use mlxcel::vision::dense::{DenseOutput, DensePredictor, PersonKeypoints};

use super::*;
use crate::{Cli, Commands};

struct Stub;

impl DensePredictor for Stub {
    fn family(&self) -> &'static str {
        "sapiens2"
    }

    fn tasks(&self) -> &[&'static str] {
        &["depth", "pose"]
    }

    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput> {
        let task = opts.get("task").and_then(Value::as_str).unwrap();
        let (tensors, keypoints) = match input {
            DenseInput::Image(image) => {
                use image::GenericImageView;
                let (width, height) = image.dimensions();
                let n = width as usize * height as usize;
                let depth = mlxcel_core::from_slice_f32(
                    &(0..n).map(|value| value as f32).collect::<Vec<_>>(),
                    &[height as i32, width as i32],
                );
                let mut tensors = BTreeMap::new();
                tensors.insert("depth".into(), depth);
                let keypoints = if task == "pose" {
                    vec![PersonKeypoints {
                        bbox: [0.0, 0.0, width as f32, height as f32],
                        points: vec![[1.0, 1.0, 1.0]],
                        names: None,
                    }]
                } else {
                    Vec::new()
                };
                (tensors, keypoints)
            }
            DenseInput::Video { frames, .. } => {
                let width = frames[0].width() as usize;
                let height = frames[0].height() as usize;
                let n = frames.len() * width * height;
                let depth = mlxcel_core::from_slice_f32(
                    &(0..n).map(|value| value as f32).collect::<Vec<_>>(),
                    &[frames.len() as i32, height as i32, width as i32],
                );
                let mut tensors = BTreeMap::new();
                tensors.insert("depth".into(), depth);
                (tensors, Vec::new())
            }
        };
        Ok(DenseOutput {
            tensors,
            keypoints,
            instances: Vec::new(),
            meshes: Vec::new(),
            gaussians: Vec::new(),
            meta: Map::new(),
        })
    }
}

fn load_stub(_: &Path) -> Result<Box<dyn DensePredictor>> {
    Ok(Box::new(Stub))
}

const STUB_FAMILIES: &[DenseFamily] = &[DenseFamily {
    model_type: "sapiens2",
    supports_prompts: false,
    supports_sidecars: false,
    load: load_stub,
}];

struct PromptStub;

impl DensePredictor for PromptStub {
    fn family(&self) -> &'static str {
        "sam3"
    }

    fn tasks(&self) -> &[&'static str] {
        &["segment"]
    }

    fn predict(&self, input: &DenseInput, opts: &Map<String, Value>) -> Result<DenseOutput> {
        assert_eq!(opts["prompts"], serde_json::json!(["a cat", "a dog"]));
        assert_eq!(
            opts["boxes"],
            serde_json::json!([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        );
        assert_eq!(opts["task"], "segment");
        let (width, height) = input.dimensions().unwrap();
        let mut output = DenseOutput::default();
        output.tensors.insert(
            "depth".into(),
            mlxcel_core::zeros(&[height as i32, width as i32], mlxcel_core::dtype::FLOAT32),
        );
        Ok(output)
    }
}

fn load_prompt_stub(_: &Path) -> Result<Box<dyn DensePredictor>> {
    Ok(Box::new(PromptStub))
}

const PROMPT_FAMILIES: &[DenseFamily] = &[DenseFamily {
    model_type: "sam3",
    supports_prompts: true,
    supports_sidecars: false,
    load: load_prompt_stub,
}];

fn png_chunk(name: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(name, data).to_be_bytes());
    out
}

fn crc32(name: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in name.iter().chain(data) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn synthetic_png_header(width: u32, height: u32) -> Vec<u8> {
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
    png.extend_from_slice(&png_chunk(b"IEND", &[]));
    png
}

fn model_dir() -> tempfile::TempDir {
    model_dir_with_type("sapiens2")
}

fn model_dir_with_type(model_type: &str) -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        format!(r#"{{"model_type":"{model_type}"}}"#),
    )
    .unwrap();
    dir
}

#[test]
fn vda_keeps_consecutive_frames_without_target_fps() {
    assert!(keep_all_video_frames("video_depth_anything", None));
    assert!(keep_all_video_frames("video_depth_anything", Some("depth")));
    assert!(!keep_all_video_frames("sapiens2", None));
}

fn args(model: &Path, output: PathBuf) -> PredictArgs {
    PredictArgs {
        mask: None,
        pointmap: None,
        model: model.to_path_buf(),
        image: None,
        video: None,
        output,
        task: Some("depth".into()),
        prompt: Vec::new(),
        boxes: Vec::new(),
        max_frames: None,
        target_fps: None,
        opts: None,
    }
}

#[test]
fn tracking_frame_sampling_keeps_all_sam_generations_only() {
    for model_type in ["sam3", "sam3_video", "sam3.1", "sam3.1_video"] {
        assert!(keep_all_video_frames(model_type, Some("track")));
        assert!(!keep_all_video_frames(model_type, Some("segment")));
    }
    assert!(!keep_all_video_frames("sapiens2", Some("track")));
    assert!(!keep_all_video_frames("sam3.1", None));
}

fn parse_predict(argv: Vec<OsString>) -> PredictArgs {
    let command = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || Cli::try_parse_from(argv).map(|cli| cli.command))
        .unwrap()
        .join()
        .unwrap()
        .unwrap();
    match command {
        Commands::Predict(args) => args,
        other => panic!("expected predict command, got {other:?}"),
    }
}

#[test]
fn image_runs_through_stub_family_and_writes_contract() {
    let model = model_dir();
    let scratch = tempfile::TempDir::new().unwrap();
    let image_path = scratch.path().join("input.png");
    image::DynamicImage::new_rgb8(3, 2)
        .save(&image_path)
        .unwrap();
    let output = scratch.path().join("nested/output");
    let args = parse_predict(vec![
        "mlxcel".into(),
        "predict".into(),
        "--model".into(),
        model.path().as_os_str().to_owned(),
        "--image".into(),
        image_path.as_os_str().to_owned(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--task".into(),
        "pose".into(),
        "--opts".into(),
        r#"{"threshold":0.5}"#.into(),
    ]);

    run_predict_with(args, STUB_FAMILIES).unwrap();
    for name in [
        "result.safetensors",
        "meta.json",
        "keypoints.json",
        "depth.png",
    ] {
        assert!(output.join(name).is_file(), "missing {name}");
    }
}

#[test]
fn repeated_prompts_and_boxes_are_parsed_and_passed_to_predictor() {
    let model = model_dir_with_type("sam3");
    let scratch = tempfile::TempDir::new().unwrap();
    let image_path = scratch.path().join("input.png");
    image::DynamicImage::new_rgb8(2, 3)
        .save(&image_path)
        .unwrap();
    let output = scratch.path().join("output");
    let args = parse_predict(vec![
        "mlxcel".into(),
        "predict".into(),
        "--model".into(),
        model.path().as_os_str().to_owned(),
        "--image".into(),
        image_path.as_os_str().to_owned(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--prompt".into(),
        "a cat".into(),
        "--prompt".into(),
        "a dog".into(),
        "--boxes".into(),
        "0,1,2,3".into(),
        "--boxes".into(),
        "4,5,6,7".into(),
    ]);
    assert_eq!(args.prompt, ["a cat", "a dog"]);
    assert_eq!(args.boxes, ["0,1,2,3", "4,5,6,7"]);

    run_predict_with(args, PROMPT_FAMILIES).unwrap();
    assert!(output.join("result.safetensors").is_file());
}

#[test]
fn boxes_parser_accepts_semicolon_groups_and_rejects_malformed_values() {
    assert_eq!(
        parse_boxes(&["0,1,2,3; 4,5,6,7".into()]).unwrap(),
        [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    );
    for raw in ["0,1,2", "0,1,nope,3", "0,1,2,3;", "0,1,inf,3", "2,1,0,3"] {
        let error = parse_boxes(&[raw.into()]).unwrap_err().to_string();
        assert!(error.contains("--boxes"), "{raw}: {error}");
    }
}

#[test]
fn prompts_are_rejected_for_families_without_prompt_capability() {
    let model = model_dir();
    let scratch = tempfile::TempDir::new().unwrap();
    let mut args = args(model.path(), scratch.path().join("output"));
    args.image = Some(scratch.path().join("not-loaded.png"));
    args.prompt.push("a person".into());
    let error = run_predict_with(args, STUB_FAMILIES)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not support --prompt or --boxes"),
        "{error}"
    );
}

#[test]
fn prompt_options_are_rejected_for_families_without_prompt_capability() {
    let model = model_dir();
    let scratch = tempfile::TempDir::new().unwrap();
    let mut args = args(model.path(), scratch.path().join("output"));
    args.image = Some(scratch.path().join("not-loaded.png"));
    args.opts = Some(r#"{"prompts":["a person"]}"#.into());
    let error = run_predict_with(args, STUB_FAMILIES)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not support --prompt or --boxes"),
        "{error}"
    );
}

#[test]
fn exactly_one_input_and_task_are_validated() {
    let model = model_dir();
    let scratch = tempfile::TempDir::new().unwrap();
    let no_input = args(model.path(), scratch.path().join("none"));
    assert!(
        run_predict_with(no_input, STUB_FAMILIES)
            .unwrap_err()
            .to_string()
            .contains("exactly one")
    );

    let image_path = scratch.path().join("input.png");
    image::DynamicImage::new_rgb8(2, 2)
        .save(&image_path)
        .unwrap();
    let mut missing_task = args(model.path(), scratch.path().join("missing-task"));
    missing_task.image = Some(image_path);
    missing_task.task = None;
    assert!(
        run_predict_with(missing_task, STUB_FAMILIES)
            .unwrap_err()
            .to_string()
            .contains("--task is required")
    );
}

#[test]
fn image_loader_rejects_an_oversized_header_before_allocation() {
    let scratch = tempfile::TempDir::new().unwrap();
    let path = scratch.path().join("oversized.png");
    std::fs::write(&path, synthetic_png_header(100_000, 100_000)).unwrap();
    let error = load_predict_image(&path).unwrap_err();
    assert!(
        format!("{error:#}").contains("Image decode rejected by configured limits"),
        "{error:#}"
    );
}

#[test]
fn image_loader_accounts_for_dense_rgb_f32_expansion() {
    let limit = 512 * 1024 * 1024;
    validate_dense_image_allocation(4096, 4096, limit).unwrap();
    let error = validate_dense_image_allocation(8192, 8192, limit)
        .unwrap_err()
        .to_string();
    assert!(error.contains("dense RGB f32 expansion"), "{error}");
    assert!(error.contains("805306368"), "{error}");
}

#[test]
fn video_cli_honors_fps_and_true_max_frame_cap() {
    if !mlxcel::multimodal::video::ffmpeg_available() {
        eprintln!("skipping: ffmpeg and ffprobe are required");
        return;
    }
    let model = model_dir();
    let scratch = tempfile::TempDir::new().unwrap();
    let video = scratch.path().join("input.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=4x4:r=2:d=2",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&video)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());

    let output = scratch.path().join("video-output");
    let args = parse_predict(vec![
        "mlxcel".into(),
        "predict".into(),
        "--model".into(),
        model.path().as_os_str().to_owned(),
        "--video".into(),
        video.as_os_str().to_owned(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--task".into(),
        "depth".into(),
        "--max-frames".into(),
        "3".into(),
        "--target-fps".into(),
        "2".into(),
    ]);
    run_predict_with(args, STUB_FAMILIES).unwrap();
    assert!(output.join("result.safetensors").is_file());
    assert!(output.join("depth/000000.png").is_file());
    assert!(output.join("depth/000002.png").is_file());
    assert!(!output.join("depth/000003.png").exists());

    let short_cap = load_predict_video(&video, Some(2.0), Some(10), false).unwrap();
    assert_eq!(short_cap.len(), 4, "a maximum above the clip must not fail");

    let one = load_predict_video(&video, Some(2.0), Some(1), false).unwrap();
    assert_eq!(one.len(), 1, "a one-frame cap must remain one frame");

    let tracking_video = scratch.path().join("tracking-input.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=4x4:r=6:d=0.5",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&tracking_video)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());

    let default_sampling = load_predict_video(&tracking_video, None, Some(3), false).unwrap();
    assert_eq!(
        default_sampling.len(),
        1,
        "other dense tasks preserve the generic 2 FPS default"
    );

    let tracking_sampling = load_predict_video(&tracking_video, None, Some(3), true).unwrap();
    assert_eq!(
        tracking_sampling.len(),
        3,
        "SAM tracking keeps source frames up to its cap"
    );
}

#[test]
fn object_sidecars_are_rejected_before_other_families_load_inputs() {
    let model = model_dir();
    let scratch = tempfile::TempDir::new().unwrap();
    for pointmap in [false, true] {
        let mut args = args(model.path(), scratch.path().join("out"));
        args.image = Some(scratch.path().join("not-loaded.png"));
        if pointmap {
            args.pointmap = Some(scratch.path().join("points.safetensors"));
        } else {
            args.mask = Some(scratch.path().join("mask.png"));
        }
        let error = run_predict_with(args, STUB_FAMILIES)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("does not support --mask or --pointmap"),
            "{error}"
        );
    }
}
