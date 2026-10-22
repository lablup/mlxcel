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

//! Gated real-checkpoint parity against the pinned mlx-vlm reference.
use safetensors::SafeTensors;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

use super::super::{DenseInput, DensePredictor};
use super::predictor::Sam3Predictor;
use crate::vision::dense::preprocess::{PillowResizeFilter, resize_pillow_rgb8};

#[path = "real_test_utils.rs"]
mod real_test_utils;
use real_test_utils::{
    assert_bf16_stage, check_bf16_values_exact, report_bf16_rows, report_bf16_stage,
    tensor_bf16_as_f32, tensor_f32, tensor_i32, tensor_i64_as_f32,
};

const TEXT_REFERENCE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/text-reference.safetensors"
));
const PIXEL_REFERENCE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/pixel-reference.safetensors"
));
const CAT_IMAGE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/cats.png"
));
const CAT_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/cats-reference.json"
));
const CAT_MASKS: &[&[u8]] = &[
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/cats-mask-0.png"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/cats-mask-1.png"
    )),
];
const BOX_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/box-reference.json"
));
const BOX_MASKS: &[&[u8]] = &[
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/box-mask-0.png"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/box-mask-1.png"
    )),
];
const CAR_VIDEO: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/car-clip.mp4"
));
const VIDEO_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/video-reference.json"
));
const VIDEO_NATIVE_MASKS: &[&[u8]] = &[
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/video-native-mask-0.png"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/video-native-mask-1.png"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/video-native-mask-2.png"
    )),
];
const VIDEO_SOURCE_MASKS: &[&[u8]] = &[
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/video-source-mask-0.png"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/video-source-mask-1.png"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3/video-source-mask-2.png"
    )),
];

#[derive(Deserialize)]
struct ImageReference {
    boxes: Vec<[f32; 4]>,
    scores: Vec<f32>,
}

#[derive(Deserialize)]
struct VideoReference {
    frames: Vec<VideoFrameReference>,
}

#[derive(Deserialize)]
struct VideoFrameReference {
    frame: u32,
    object_id: u32,
    score: f32,
    native_foreground: usize,
    source_foreground: usize,
}

#[test]
#[ignore = "loads the real SAM 3 checkpoint and runs CLIP on the accelerator"]
fn pinned_checkpoint_matches_mlx_vlm_text() {
    let Ok(model_dir) = std::env::var("MLXCEL_SAM3_MODEL") else {
        eprintln!("skip: MLXCEL_SAM3_MODEL is not set");
        return;
    };
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor = Sam3Predictor::load(Path::new(&model_dir)).expect("load SAM 3 checkpoint");
    let actual_dtype = predictor
        .model
        .prompt_projection_dtype()
        .expect("bf16 checkpoint must load a dense text projection");
    eprintln!("SAM 3 text projection dtype: {actual_dtype}");
    if std::env::var_os("MLXCEL_KEEP_BF16").is_some() {
        assert_eq!(actual_dtype, mlxcel_core::dtype::BFLOAT16);
    } else {
        assert!(matches!(
            actual_dtype,
            mlxcel_core::dtype::BFLOAT16 | mlxcel_core::dtype::FLOAT16
        ));
    }
    let tokens = predictor
        .processor
        .tokenize("a cat")
        .expect("tokenize prompt");
    let reference = SafeTensors::deserialize(TEXT_REFERENCE).expect("parse text reference");
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&tokens.input_ids),
        tensor_i64_as_f32(reference.tensor("input_ids").unwrap())
    );
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&tokens.attention_mask),
        tensor_i64_as_f32(reference.tensor("attention_mask").unwrap())
    );

    let detailed_path = Path::new(&model_dir)
        .parent()
        .map(|parent| parent.join("sam3-exact-text-stages.safetensors"));
    if let Some(Ok(bytes)) = detailed_path.as_ref().map(std::fs::read) {
        let detailed = SafeTensors::deserialize(&bytes).expect("parse detailed text stages");
        let mut mask_check = Ok(());
        for (name, actual) in predictor
            .model
            .prompt_diagnostic_stages(&tokens.input_ids, &tokens.attention_mask)
        {
            let expected = detailed
                .tensor(&name)
                .unwrap_or_else(|_| panic!("detailed reference missing {name}"));
            if name == "layer_0_mask" {
                mask_check = check_bf16_values_exact(
                    &name,
                    &actual,
                    detailed.tensor(&name).expect("detailed mask reference"),
                );
            }
            report_bf16_stage(&name, &actual, expected);
        }
        assert!(mask_check.is_ok(), "{}", mask_check.unwrap_err());
    }

    let hidden = predictor
        .model
        .prompt_hidden(&tokens.input_ids, &tokens.attention_mask);
    report_bf16_rows(
        "text hidden",
        &hidden,
        reference.tensor("text_hidden").unwrap(),
        &mlxcel_core::utils::array_to_vec_f32(&tokens.attention_mask),
    );
    assert_bf16_stage(
        "text hidden",
        &hidden,
        reference.tensor("text_hidden").unwrap(),
        0.02,
        0.002,
    );

    let projected = predictor
        .model
        .prompt_features(&tokens.input_ids, &tokens.attention_mask);
    let projected = mlxcel_core::astype(&projected, mlxcel_core::dtype::FLOAT32);
    let actual = mlxcel_core::utils::array_to_vec_f32(&projected);
    let expected = tensor_bf16_as_f32(reference.tensor("text_projected").unwrap());
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    let mut total = 0.0_f32;
    for (actual, expected) in actual.iter().zip(&expected) {
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        total += error;
    }
    let mean = total / actual.len() as f32;
    assert!(maximum <= 0.02, "projected token max error {maximum}");
    assert!(mean <= 0.002, "projected token mean error {mean}");
}

#[test]
#[ignore = "loads the real SAM 3 checkpoint and runs cat segmentation"]
fn pinned_checkpoint_matches_mlx_vlm_cat() {
    let Ok(model_dir) = std::env::var("MLXCEL_SAM3_MODEL") else {
        eprintln!("skip: MLXCEL_SAM3_MODEL is not set");
        return;
    };
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor = Sam3Predictor::load(Path::new(&model_dir)).expect("load SAM 3 checkpoint");
    assert_image_parity(&predictor, None, CAT_REFERENCE, CAT_MASKS, "no-box cat");
}

#[test]
#[ignore = "loads the real SAM 3 checkpoint and runs box-prompt segmentation"]
fn pinned_checkpoint_matches_mlx_vlm_box_prompt() {
    let Ok(model_dir) = std::env::var("MLXCEL_SAM3_MODEL") else {
        eprintln!("skip: MLXCEL_SAM3_MODEL is not set");
        return;
    };
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor = Sam3Predictor::load(Path::new(&model_dir)).expect("load SAM 3 checkpoint");
    assert_image_parity(
        &predictor,
        Some(json!([[0.0, 50.0, 350.0, 480.0]])),
        BOX_REFERENCE,
        BOX_MASKS,
        "box-prompt cat",
    );
}

fn assert_image_parity(
    predictor: &Sam3Predictor,
    boxes: Option<Value>,
    reference: &str,
    masks: &[&[u8]],
    case: &str,
) {
    let image = image::load_from_memory(CAT_IMAGE).expect("decode cat fixture");
    assert_processor_pixels(predictor, &image);
    let input = DenseInput::Image(image);
    let mut opts = Map::from_iter([
        ("task".into(), Value::String("segment".into())),
        (
            "prompts".into(),
            Value::Array(vec![Value::String("a cat".into())]),
        ),
        ("threshold".into(), Value::from(0.3)),
        ("nms".into(), Value::from(0.5)),
    ]);
    if let Some(boxes) = boxes {
        opts.insert("boxes".into(), boxes);
    }
    let output = predictor
        .predict(&input, &opts)
        .unwrap_or_else(|error| panic!("run {case} segmentation: {error:#}"));
    let reference: ImageReference = serde_json::from_str(reference)
        .unwrap_or_else(|error| panic!("parse {case} reference: {error}"));
    assert_cat_parity(case, &output, &reference, masks);
}

#[test]
#[ignore = "loads the real SAM 3 checkpoint and propagates three video frames"]
fn pinned_checkpoint_matches_mlx_vlm_video_tracking() {
    let Ok(model_dir) = std::env::var("MLXCEL_SAM3_MODEL") else {
        eprintln!("skip: MLXCEL_SAM3_MODEL is not set");
        return;
    };
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor = Sam3Predictor::load(Path::new(&model_dir)).expect("load SAM 3 checkpoint");
    let scratch = tempfile::TempDir::new().expect("video scratch directory");
    let path = scratch.path().join("car-clip.mp4");
    std::fs::write(&path, CAR_VIDEO).expect("write video fixture");
    let frames = crate::multimodal::video::load_video_source_with_policy_and_budget(
        &crate::multimodal::video::VideoSource::from_path(path),
        None,
        None,
        crate::multimodal::video::FrameSamplingPolicy {
            min_frames: 1,
            max_frames: 3,
            frame_factor: 1,
            reject_over_max: false,
            keep_all_without_target_fps: true,
        },
        &crate::multimodal::video::VideoLimits::default(),
        Some(crate::multimodal::video::DEFAULT_MAX_DECODED_VIDEO_BYTES),
    )
    .expect("decode exact video fixture");
    assert_eq!(frames.len(), 3);
    let output = predictor
        .predict(
            &DenseInput::Video { frames, fps: 6.0 },
            &Map::from_iter([
                ("task".into(), Value::String("track".into())),
                (
                    "prompts".into(),
                    Value::Array(vec![Value::String("a car".into())]),
                ),
                ("threshold".into(), Value::from(0.15)),
            ]),
        )
        .expect("run SAM 3 video tracking");
    let reference: VideoReference =
        serde_json::from_str(VIDEO_REFERENCE).expect("parse video reference");
    assert_eq!(output.instances.len(), reference.frames.len());
    let masks = output.tensors.get("masks").expect("tracking masks");
    assert_eq!(mlxcel_core::array_shape(masks), vec![3, 480, 854]);
    let actual_masks = mlxcel_core::array_to_raw_bytes(masks);
    let mask_pixels = 480 * 854;
    for (index, (instance, expected)) in output.instances.iter().zip(&reference.frames).enumerate()
    {
        assert_eq!(instance.frame, Some(expected.frame));
        assert_eq!(instance.object_id, Some(expected.object_id));
        assert!((instance.score - expected.score).abs() <= 0.02);
        let native = image::load_from_memory(VIDEO_NATIVE_MASKS[index])
            .expect("decode native tracking mask")
            .to_luma8();
        assert_eq!(
            native.pixels().filter(|pixel| pixel[0] != 0).count(),
            expected.native_foreground
        );
        let source = image::load_from_memory(VIDEO_SOURCE_MASKS[index])
            .expect("decode source tracking mask")
            .to_luma8();
        assert_eq!(
            source.pixels().filter(|pixel| pixel[0] != 0).count(),
            expected.source_foreground
        );
        let actual = &actual_masks[index * mask_pixels..(index + 1) * mask_pixels];
        if expected.source_foreground == 0 {
            assert!(actual.iter().all(|value| *value == 0));
        } else {
            assert!(binary_u8_mask_iou(actual, source.as_raw()) >= 0.95);
        }
    }
}

fn assert_processor_pixels(predictor: &Sam3Predictor, image: &image::DynamicImage) {
    let resized = resize_pillow_rgb8(&image.to_rgb8(), 1008, 1008, PillowResizeFilter::Bilinear)
        .expect("Pillow-compatible cat resize");
    assert_eq!(
        format!("{:x}", Sha256::digest(resized.as_raw())),
        "a05b0197d0c56d9761ff0dd7d0370c00b7f9275d464e15c2bccedb1860051583",
        "full resized RGB8 output differs from Pillow"
    );
    let pixels = predictor
        .processor
        .preprocess_image(image)
        .expect("preprocess cat fixture");
    let actual = mlxcel_core::utils::array_to_vec_f32(&pixels);
    let reference = SafeTensors::deserialize(PIXEL_REFERENCE).expect("parse pixel reference");
    let positions = tensor_i32(reference.tensor("positions").unwrap());
    let expected = tensor_f32(reference.tensor("pixels").unwrap());
    assert_eq!(positions.len() * 3, expected.len());
    let mut maximum = 0.0_f32;
    let mut total = 0.0_f32;
    for (sample, position) in positions.iter().enumerate() {
        for channel in 0..3 {
            let index = (position[0] * 1008 + position[1]) * 3 + channel;
            let error = (actual[index] - expected[sample * 3 + channel]).abs();
            maximum = maximum.max(error);
            total += error;
        }
    }
    let mean = total / expected.len() as f32;
    assert!(maximum <= 1e-6, "processor pixel max error {maximum}");
    assert!(mean <= 1e-7, "processor pixel mean error {mean}");
}

fn assert_cat_parity(
    case: &str,
    output: &super::super::DenseOutput,
    reference: &ImageReference,
    expected_masks: &[&[u8]],
) {
    assert_eq!(reference.boxes.len(), 2);
    assert_eq!(reference.scores.len(), 2);
    assert_eq!(expected_masks.len(), 2);
    assert_eq!(output.instances.len(), 2, "expected both cats");
    let masks = output.tensors.get("masks").expect("segmentation masks");
    assert_eq!(mlxcel_core::array_shape(masks), vec![2, 480, 640]);
    let actual_masks = mlxcel_core::utils::array_to_vec_f32(masks);

    let mut unmatched = vec![0_usize, 1];
    for instance in &output.instances {
        assert_eq!(instance.label, "a cat");
        let (slot, &reference_index) = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                box_error(instance.bbox, &reference.boxes, **left).total_cmp(&box_error(
                    instance.bbox,
                    &reference.boxes,
                    **right,
                ))
            })
            .expect("unmatched reference cat");
        let max_box_error = (0..4)
            .map(|coordinate| {
                (instance.bbox[coordinate] - reference.boxes[reference_index][coordinate]).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(
            max_box_error <= 2.0,
            "cat box {reference_index} max error {max_box_error}"
        );
        let score_error = (instance.score - reference.scores[reference_index]).abs();
        assert!(
            score_error <= 0.01,
            "cat score {reference_index} error {score_error}"
        );
        let actual_index = instance.mask_index.expect("cat mask index");
        let expected_mask = image::load_from_memory(expected_masks[reference_index])
            .expect("decode reference mask")
            .to_luma8();
        assert_eq!(expected_mask.dimensions(), (640, 480));
        let iou = binary_mask_iou(
            &actual_masks[actual_index * 480 * 640..(actual_index + 1) * 480 * 640],
            expected_mask.as_raw(),
        );
        let actual_foreground = actual_masks
            [actual_index * 480 * 640..(actual_index + 1) * 480 * 640]
            .iter()
            .filter(|value| **value > 0.5)
            .count();
        let expected_foreground = expected_mask.pixels().filter(|pixel| pixel[0] != 0).count();
        eprintln!(
            "{case} instance {reference_index}: score={}, box={:?}, mask_foreground={actual_foreground}, reference_foreground={expected_foreground}, IoU={iou}",
            instance.score, instance.bbox
        );
        assert!(iou >= 0.98, "{case} mask {reference_index} IoU {iou}");
        unmatched.swap_remove(slot);
    }
}

fn box_error(actual: [f32; 4], expected: &[[f32; 4]], index: usize) -> f32 {
    (0..4)
        .map(|coordinate| (actual[coordinate] - expected[index][coordinate]).abs())
        .sum()
}

fn binary_mask_iou(actual: &[f32], expected: &[u8]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut intersection = 0_u32;
    let mut union = 0_u32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let actual = actual > 0.5;
        let expected = expected != 0;
        intersection += u32::from(actual && expected);
        union += u32::from(actual || expected);
    }
    intersection as f32 / union as f32
}

fn binary_u8_mask_iou(actual: &[u8], expected: &[u8]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut intersection = 0_u32;
    let mut union = 0_u32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        intersection += u32::from(actual != 0 && expected != 0);
        union += u32::from(actual != 0 || expected != 0);
    }
    intersection as f32 / union as f32
}
