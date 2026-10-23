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

//! Gated parity tests against pinned PP-DocLayoutV3 checkpoints and raw outputs.

use std::path::{Path, PathBuf};

use image::{DynamicImage, RgbImage};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;

use super::model::PpDocLayoutV3Model;
use super::postprocess::decode_detections;
use super::predictor::read_output_f32;
use super::processor::PpDocLayoutV3Processor;
use super::sanitize::{needs_sanitize, sanitize};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};

const FIXTURE_DIR: &str = "tests/fixtures/pp_doclayout_v3";

fn reference_directory() -> PathBuf {
    std::env::var_os("MLXCEL_TEST_PP_DOCLAYOUT_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR))
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(reference_directory().join(name)).unwrap()
}

fn checkpoint(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {variable} to a pinned PP-DocLayoutV3 checkpoint"))
}

fn weight_signature(weights: &WeightMap) -> Vec<(String, Vec<i32>)> {
    let mut signature: Vec<_> = weights
        .iter()
        .map(|(name, value)| (name.clone(), mlxcel_core::array_shape(value)))
        .collect();
    signature.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    signature
}

fn f32_tensor<'a>(tensors: &'a SafeTensors<'a>, name: &str, shape: &[usize]) -> TensorView<'a> {
    let tensor = tensors.tensor(name).unwrap();
    assert_eq!(tensor.dtype(), Dtype::F32, "{name} dtype");
    assert_eq!(tensor.shape(), shape, "{name} shape");
    tensor
}

fn f32_values(tensor: TensorView<'_>) -> Vec<f32> {
    tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn input() -> (DynamicImage, Vec<f32>) {
    let bytes = fixture("input.safetensors");
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let decoded = tensors.tensor("decoded_rgb_u8").unwrap();
    assert_eq!(decoded.dtype(), Dtype::U8);
    let shape = decoded.shape();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[2], 3);
    let image = DynamicImage::ImageRgb8(
        RgbImage::from_raw(shape[1] as u32, shape[0] as u32, decoded.data().to_vec()).unwrap(),
    );
    let pixels = f32_values(f32_tensor(&tensors, "pixel_values_nhwc", &[1, 800, 800, 3]));
    (image, pixels)
}

fn assert_max_abs(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name} element count");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{name} actual finite"
    );
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "{name} reference finite"
    );
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("PP-DocLayoutV3 {name} max abs: {maximum:.9e}");
    assert!(
        maximum <= tolerance,
        "{name} max abs {maximum} exceeds {tolerance}"
    );
}

struct DifferenceMetrics {
    maximum: f32,
    rms: f64,
    maximum_index: usize,
    actual_at_maximum: f32,
    expected_at_maximum: f32,
}

fn difference_metrics(name: &str, actual: &[f32], expected: &[f32]) -> DifferenceMetrics {
    assert_eq!(actual.len(), expected.len(), "{name} element count");
    assert!(!actual.is_empty(), "{name} cannot be empty");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{name} actual finite"
    );
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "{name} reference finite"
    );
    let mut maximum = -1.0_f32;
    let mut maximum_index = 0;
    let mut squared = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).abs();
        squared += f64::from(difference) * f64::from(difference);
        if difference > maximum {
            maximum = difference;
            maximum_index = index;
        }
    }
    DifferenceMetrics {
        maximum,
        rms: (squared / actual.len() as f64).sqrt(),
        maximum_index,
        actual_at_maximum: actual[maximum_index],
        expected_at_maximum: expected[maximum_index],
    }
}

fn run_raw_parity(checkpoint: PathBuf, layout: &str, reference_name: &str) -> Vec<String> {
    let _capture_scope = super::diagnostics::CaptureScope::begin(layout);
    let (_, pixels) = input();
    let input = mlxcel_core::from_slice_f32(&pixels, &[1, 800, 800, 3]);
    let model = PpDocLayoutV3Model::load(checkpoint).unwrap();
    assert_eq!(model.weight_dtype(), mlxcel_core::dtype::FLOAT32);
    let output = model.forward(&input).unwrap();
    assert_eq!(
        mlxcel_core::hardware::gpu_backend_kind(),
        mlxcel_core::hardware::GpuBackendKind::Rocm
    );
    eprintln!(
        "PP-DocLayoutV3 actual_backend=rocm actual_weight_dtype=F32 reference={reference_name}"
    );
    let bytes = fixture(reference_name);
    let reference = SafeTensors::deserialize(&bytes).unwrap();
    let mut names = reference.names();
    names.sort_unstable();
    assert_eq!(names, ["logits", "order_logits", "pred_boxes"]);
    let mut failures = Vec::new();
    for (name, actual, shape) in [
        ("logits", &output.logits, [1, 300, 25].as_slice()),
        ("pred_boxes", &output.pred_boxes, [1, 300, 4].as_slice()),
        (
            "order_logits",
            &output.order_logits,
            [1, 300, 300].as_slice(),
        ),
    ] {
        assert_eq!(
            mlxcel_core::array_shape(actual),
            shape.iter().map(|&v| v as i32).collect::<Vec<_>>()
        );
        let expected = f32_values(f32_tensor(&reference, name, shape));
        let actual = read_output_f32(actual);
        let metrics = difference_metrics(name, &actual, &expected);
        eprintln!(
            "PP-DocLayoutV3 layout={layout} tensor={name} max_abs={:.9e} rms={:.9e} max_index={} actual={:.9e} reference={:.9e}",
            metrics.maximum,
            metrics.rms,
            metrics.maximum_index,
            metrics.actual_at_maximum,
            metrics.expected_at_maximum,
        );
        if metrics.maximum > 1.0e-3 {
            failures.push(format!(
                "{layout} {name} max abs {} exceeds 0.001 (rms {}, index {}, actual {}, reference {})",
                metrics.maximum,
                metrics.rms,
                metrics.maximum_index,
                metrics.actual_at_maximum,
                metrics.expected_at_maximum,
            ));
        }
    }
    failures
}

#[derive(Deserialize)]
struct ExpectedRoot {
    threshold: f32,
    size: [u32; 2],
    results: std::collections::BTreeMap<String, ExpectedLayout>,
}

#[derive(Deserialize)]
struct ExpectedLayout {
    rust_expected_detections: Vec<ExpectedDetection>,
    order_differs: bool,
}

#[derive(Deserialize)]
struct ExpectedDetection {
    bbox: [f32; 4],
    score: f32,
    label: usize,
    class_name: String,
    reading_order: u32,
}

const LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "header",
    "image",
    "formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "text",
    "vision_footnote",
];

fn assert_postprocess(layout: &str, raw_name: &str, expected: &ExpectedRoot) {
    let raw_bytes = fixture(raw_name);
    let raw = SafeTensors::deserialize(&raw_bytes).unwrap();
    let logits = f32_values(f32_tensor(&raw, "logits", &[1, 300, 25]));
    let boxes = f32_values(f32_tensor(&raw, "pred_boxes", &[1, 300, 4]));
    let order = f32_values(f32_tensor(&raw, "order_logits", &[1, 300, 300]));
    let labels = LABELS.map(str::to_string);
    let actual = decode_detections(
        &logits,
        &boxes,
        &order,
        300,
        25,
        expected.size[0] as f32,
        expected.size[1] as f32,
        expected.threshold,
        Some(&labels),
    )
    .unwrap();
    let layout = &expected.results[layout];
    assert!(layout.order_differs, "fixture must exercise model order");
    assert_eq!(actual.len(), layout.rust_expected_detections.len());
    for (index, (actual, expected)) in actual
        .iter()
        .zip(&layout.rust_expected_detections)
        .enumerate()
    {
        assert_eq!(actual.label, expected.label, "detection {index} label");
        assert_eq!(
            actual.class_name, expected.class_name,
            "detection {index} class"
        );
        assert_eq!(
            actual.reading_order,
            Some(expected.reading_order),
            "detection {index} order"
        );
        assert!(
            (actual.score - expected.score).abs() <= 1.0e-6,
            "detection {index} score: {} != {}",
            actual.score,
            expected.score
        );
        for axis in 0..4 {
            assert!(
                (actual.bbox[axis] - expected.bbox[axis]).abs() <= 1.0e-3,
                "detection {index} bbox[{axis}]: {} != {}",
                actual.bbox[axis],
                expected.bbox[axis]
            );
        }
    }
}

#[test]
fn authentic_raw_outputs_preserve_global_two_column_reading_order() {
    let expected: ExpectedRoot =
        serde_json::from_slice(&fixture("expected_detections.json")).unwrap();
    assert_eq!(expected.size, [1191, 1684]);
    assert_eq!(expected.threshold, 0.3);
    assert_postprocess("hashnuke", "hashnuke_raw.safetensors", &expected);
    assert_postprocess("paddle", "paddle_raw.safetensors", &expected);
}

#[test]
#[ignore = "requires pinned Paddle and HashNuke checkpoints"]
fn real_raw_parity_both_released_layouts() {
    let _device_lock = mlxcel_core::streams::lock_default_device();
    assert!(
        mlxcel_core::gpu_backend_available(),
        "PP-DocLayoutV3 raw parity requires a GPU backend"
    );
    let _gpu_device = mlxcel_core::streams::DefaultDeviceGuard::gpu();
    assert!(mlxcel_core::default_device_is_gpu());
    let mut failures = run_raw_parity(
        checkpoint("MLXCEL_TEST_PP_DOCLAYOUT_PADDLE_DIR"),
        "paddle",
        "paddle_raw.safetensors",
    );
    mlxcel_core::memory::clear_cache();
    failures.extend(run_raw_parity(
        checkpoint("MLXCEL_TEST_PP_DOCLAYOUT_HASHNUKE_DIR"),
        "hashnuke",
        "hashnuke_raw.safetensors",
    ));
    assert!(
        failures.is_empty(),
        "PP-DocLayoutV3 raw parity failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "requires pinned processor fixture and both released checkpoints"]
fn real_processor_matches_huggingface_uint8_bicubic_exactly() {
    let (image, expected) = input();
    for variable in [
        "MLXCEL_TEST_PP_DOCLAYOUT_PADDLE_DIR",
        "MLXCEL_TEST_PP_DOCLAYOUT_HASHNUKE_DIR",
    ] {
        let processor = PpDocLayoutV3Processor::from_pretrained(checkpoint(variable)).unwrap();
        let actual = processor.process_image(&image).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&actual.pixel_values),
            [1, 800, 800, 3]
        );
        assert_max_abs(
            variable,
            &read_output_f32(&actual.pixel_values),
            &expected,
            0.0,
        );
    }
}

#[test]
#[ignore = "requires both released Paddle and HashNuke checkpoints"]
fn real_sanitizer_is_idempotent_and_both_layouts_have_identical_key_shapes() {
    let paddle = load_weights_from_dir(checkpoint("MLXCEL_TEST_PP_DOCLAYOUT_PADDLE_DIR")).unwrap();
    assert!(needs_sanitize(&paddle).unwrap());
    let paddle = sanitize(paddle).unwrap();
    let paddle_once = weight_signature(&paddle);
    let paddle = sanitize(paddle).unwrap();
    assert!(!needs_sanitize(&paddle).unwrap());
    assert_eq!(weight_signature(&paddle), paddle_once);
    drop(paddle);
    mlxcel_core::memory::clear_cache();

    let hashnuke =
        load_weights_from_dir(checkpoint("MLXCEL_TEST_PP_DOCLAYOUT_HASHNUKE_DIR")).unwrap();
    assert!(!needs_sanitize(&hashnuke).unwrap());
    let hashnuke = sanitize(hashnuke).unwrap();
    assert_eq!(weight_signature(&hashnuke), paddle_once);
}
