// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use image::{DynamicImage, RgbImage};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::Map;

use super::*;
use crate::vision::dense::DenseInput;

const FIXTURE_DIR: &str = "tests/fixtures/video_depth_anything";

fn fixture(path: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(FIXTURE_DIR)
            .join(path),
    )
    .unwrap()
}

fn reference(path: &str) -> Vec<u8> {
    match std::env::var_os("MLXCEL_TEST_VDA_REFERENCE_DIR") {
        Some(directory) => std::fs::read(PathBuf::from(directory).join(path)).unwrap(),
        None => fixture(path),
    }
}

fn f32_values(tensor: TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::F32);
    tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn frames() -> Vec<DynamicImage> {
    let bytes = fixture("davis_40_frames.safetensors");
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor("frames_nhwc_u8").unwrap();
    assert_eq!(tensor.dtype(), Dtype::U8);
    let shape = tensor.shape();
    assert_eq!(shape, [40, 96, 128, 3]);
    let frame_size = shape[1] * shape[2] * 3;
    tensor
        .data()
        .chunks_exact(frame_size)
        .map(|bytes| {
            DynamicImage::ImageRgb8(
                RgbImage::from_raw(shape[2] as u32, shape[1] as u32, bytes.to_vec()).unwrap(),
            )
        })
        .collect()
}

fn assert_relative_rms(actual: &[f32], expected: &[f32], tolerance: f64, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name} element count");
    let (error, reference) = actual.iter().zip(expected).fold(
        (0.0f64, 0.0f64),
        |(error, reference), (&actual, &expected)| {
            let delta = f64::from(actual) - f64::from(expected);
            (
                error + delta * delta,
                reference + f64::from(expected) * f64::from(expected),
            )
        },
    );
    let rms = (error / reference.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("{name} relative RMS: {rms:.9e}");
    assert!(
        rms <= tolerance,
        "{name} relative RMS {rms} exceeds {tolerance}"
    );
}

fn run_checkpoint(
    checkpoint: PathBuf,
    reference_file: &str,
    tolerance: f64,
    expected_metric: bool,
) {
    let predictor = VideoDepthPredictor {
        model: VideoDepthModel::load(&checkpoint).unwrap(),
        processor: VideoDepthProcessor::from_path(&checkpoint).unwrap(),
    };
    assert_eq!(predictor.model.config.metric, expected_metric);
    assert_eq!(
        predictor.model.representative_weight_dtype(),
        mlxcel_core::dtype::FLOAT32
    );
    assert_eq!(
        predictor.model.representative_head_dtype(),
        mlxcel_core::dtype::FLOAT32
    );
    assert_eq!(
        predictor.model.representative_norm_dtype(),
        mlxcel_core::dtype::FLOAT32
    );
    let frames = frames();
    let prepared = predictor.processor.preprocess(&frames[0]).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(&prepared),
        mlxcel_core::dtype::FLOAT32
    );
    assert_eq!(mlxcel_core::array_shape(&prepared), [518, 686, 3]);
    let output = predictor
        .predict(&DenseInput::Video { frames, fps: 24.0 }, &Map::new())
        .unwrap();
    let depth = output.tensors.get("depth").unwrap();
    assert_eq!(
        output
            .tensors
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["depth"]
    );
    assert_eq!(
        output.meta.get("metric"),
        Some(&serde_json::Value::Bool(expected_metric))
    );
    assert_eq!(mlxcel_core::array_dtype(depth), mlxcel_core::dtype::FLOAT32);
    let reference_bytes = reference(reference_file);
    let reference = SafeTensors::deserialize(&reference_bytes).unwrap();
    let expected = reference.tensor("depth").unwrap();
    assert_eq!(
        expected.shape(),
        mlxcel_core::array_shape(depth)
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>()
    );
    let actual = mlxcel_core::utils::array_to_vec_f32(depth);
    assert!(actual.iter().all(|value| value.is_finite()));
    let expected = f32_values(expected);
    assert!(expected.iter().all(|value| value.is_finite()));
    assert_relative_rms(&actual, &expected, tolerance, reference_file);
}

fn run_both(tolerance: f64) {
    let relative = PathBuf::from(
        std::env::var("MLXCEL_TEST_VDA_SMALL_DIR").expect("set MLXCEL_TEST_VDA_SMALL_DIR"),
    );
    let metric = PathBuf::from(
        std::env::var("MLXCEL_TEST_VDA_METRIC_SMALL_DIR")
            .expect("set MLXCEL_TEST_VDA_METRIC_SMALL_DIR"),
    );
    run_checkpoint(relative, "small_relative_40.safetensors", tolerance, false);
    run_checkpoint(metric, "small_metric_40.safetensors", tolerance, true);
}

#[test]
#[ignore = "requires all six pinned Small, Base, and Large relative/metric checkpoints"]
fn authentic_all_published_checkpoints_load_with_exact_layouts() {
    let root = PathBuf::from(
        std::env::var("MLXCEL_TEST_VDA_CHECKPOINT_ROOT")
            .expect("set MLXCEL_TEST_VDA_CHECKPOINT_ROOT"),
    );
    let _device_lock = mlxcel_core::streams::lock_default_device();
    let _cpu_device = mlxcel_core::streams::DefaultDeviceGuard::cpu();
    for (directory, hidden_size, depth, metric) in [
        ("relative", 384, 12, false),
        ("metric", 384, 12, true),
        ("base-relative", 768, 12, false),
        ("base-metric", 768, 12, true),
        ("large-relative", 1024, 24, false),
        ("large-metric", 1024, 24, true),
    ] {
        let model = VideoDepthModel::load(&root.join(directory))
            .unwrap_or_else(|error| panic!("load {directory}: {error:#}"));
        assert_eq!(model.config.metric, metric, "{directory} metric mode");
        assert_eq!(
            model.config.dinov2.hidden_size, hidden_size,
            "{directory} hidden size"
        );
        assert_eq!(model.config.dinov2.num_hidden_layers, depth);
        assert_eq!(
            model.representative_weight_dtype(),
            mlxcel_core::dtype::FLOAT32,
            "{directory} backbone dtype"
        );
        assert_eq!(
            model.representative_head_dtype(),
            mlxcel_core::dtype::FLOAT32,
            "{directory} temporal-head dtype"
        );
        assert_eq!(
            model.representative_norm_dtype(),
            mlxcel_core::dtype::FLOAT32,
            "{directory} norm dtype"
        );
        drop(model);
        mlxcel_core::memory::clear_cache();
    }
}

#[test]
#[ignore = "requires both real Small checkpoints and full committed 40-frame CPU references"]
fn authentic_small_cpu_full_40_frame_parity() {
    let _device_lock = mlxcel_core::streams::lock_default_device();
    let _cpu_device = mlxcel_core::streams::DefaultDeviceGuard::cpu();
    assert!(!mlxcel_core::default_device_is_gpu());
    run_both(1e-4);
}

#[test]
#[ignore = "requires both real Small checkpoints, ROCm, and full committed 40-frame CPU references"]
fn authentic_small_rocm_full_40_frame_parity() {
    let _device_lock = mlxcel_core::streams::lock_default_device();
    assert!(
        mlxcel_core::gpu_backend_available(),
        "ROCm parity requires a GPU backend"
    );
    let _gpu_device = mlxcel_core::streams::DefaultDeviceGuard::gpu();
    assert!(mlxcel_core::default_device_is_gpu());
    run_both(1e-2);
}
