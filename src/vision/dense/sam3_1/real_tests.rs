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

//! Gated parity tests against pinned upstream SAM 3.1 outputs.

use std::path::Path;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::super::{DenseInput, DensePredictor};
use super::predictor::Sam31Predictor;

const CAT_IMAGE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/cats.png"
));
const CAT_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3_1/cats-reference.json"
));
const CAT_MASKS: &[&[u8]] = &[
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3_1/cats-mask-0.png"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sam3_1/cats-mask-1.png"
    )),
];
const CAR_VIDEO: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3_1/three-cars.mp4"
));
const VIDEO_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3_1/video-reference.json"
));
const VIDEO_MASKS: &[&[&[u8]]] = &[
    &[
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-0.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-1.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-2.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-3.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-4.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-5.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-6.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-7.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-0-mask-8.png"
        )),
    ],
    &[
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-0.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-1.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-2.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-3.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-4.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-5.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-6.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-7.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-1-mask-8.png"
        )),
    ],
    &[
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-0.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-1.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-2.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-3.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-4.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-5.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-6.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-7.png"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sam3_1/video-frame-2-mask-8.png"
        )),
    ],
];

#[derive(Deserialize)]
struct ImageReference {
    boxes: Vec<[f32; 4]>,
    scores: Vec<f32>,
}

#[derive(Deserialize)]
struct VideoReference {
    track_step_calls: usize,
    frames: Vec<VideoFrameReference>,
}

#[derive(Deserialize)]
struct VideoFrameReference {
    frame: u32,
    objects: Vec<VideoObjectReference>,
}

#[derive(Deserialize)]
struct VideoObjectReference {
    object_id: u32,
    #[serde(rename = "box")]
    bbox: [f32; 4],
    score: f32,
    foreground: usize,
}

#[test]
#[ignore = "loads the real SAM 3.1 checkpoint and runs cat segmentation"]
fn pinned_checkpoint_matches_upstream_cat_and_f32_policy() {
    let model_dir = model_dir();
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor = Sam31Predictor::load(Path::new(&model_dir)).expect("load SAM 3.1 checkpoint");
    for (name, dtype) in &predictor.parameter_dtypes {
        assert_eq!(
            *dtype,
            mlxcel_core::dtype::FLOAT32,
            "{name} must preserve the checkpoint's actual f32 dtype"
        );
    }
    let image = image::load_from_memory(CAT_IMAGE).expect("decode cat fixture");
    let pixels = predictor
        .processor
        .preprocess_image(&image)
        .expect("preprocess cat image");
    assert_eq!(
        mlxcel_core::array_dtype(&pixels),
        mlxcel_core::dtype::FLOAT32
    );
    let output = predictor
        .predict(
            &DenseInput::Image(image),
            &Map::from_iter([
                ("task".into(), Value::String("segment".into())),
                (
                    "prompts".into(),
                    Value::Array(vec![Value::String("a cat".into())]),
                ),
                ("threshold".into(), Value::from(0.3)),
                ("nms".into(), Value::from(0.5)),
            ]),
        )
        .expect("run SAM 3.1 cat segmentation");
    let reference: ImageReference =
        serde_json::from_str(CAT_REFERENCE).expect("parse cat reference");
    assert_eq!(output.instances.len(), reference.scores.len());
    assert_eq!(
        mlxcel_core::array_dtype(output.tensors.get("scores").unwrap()),
        mlxcel_core::dtype::FLOAT32
    );
    assert_eq!(
        mlxcel_core::array_dtype(output.tensors.get("masks").unwrap()),
        mlxcel_core::dtype::UINT8
    );
    let actual_masks = mlxcel_core::array_to_raw_bytes(output.tensors.get("masks").unwrap());
    let pixels_per_mask = 640 * 480;
    for index in 0..reference.scores.len() {
        let instance = &output.instances[index];
        assert!(
            (instance.score - reference.scores[index]).abs() <= 0.01,
            "cat {index} score {} vs {}",
            instance.score,
            reference.scores[index]
        );
        for coordinate in 0..4 {
            assert!(
                (instance.bbox[coordinate] - reference.boxes[index][coordinate]).abs() <= 2.0,
                "cat {index} box coordinate {coordinate}: {} vs {}",
                instance.bbox[coordinate],
                reference.boxes[index][coordinate]
            );
        }
        let expected = image::load_from_memory(CAT_MASKS[index])
            .expect("decode cat mask")
            .to_luma8();
        let actual = &actual_masks[index * pixels_per_mask..(index + 1) * pixels_per_mask];
        assert!(mask_iou(actual, expected.as_raw()) >= 0.98);
    }
}

#[test]
#[ignore = "loads the real SAM 3.1 checkpoint and propagates nine cars"]
fn pinned_checkpoint_matches_upstream_three_car_multiplex_tracking() {
    let model_dir = model_dir();
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor = Sam31Predictor::load(Path::new(&model_dir)).expect("load SAM 3.1 checkpoint");
    let scratch = tempfile::TempDir::new().expect("video scratch directory");
    let path = scratch.path().join("three-cars.mp4");
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
    let first_frame = frames[0].clone();
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
                ("nms".into(), Value::from(0.5)),
            ]),
        )
        .expect("run SAM 3.1 video tracking");
    let reference: VideoReference =
        serde_json::from_str(VIDEO_REFERENCE).expect("parse video reference");
    assert_eq!(
        output.meta["track_step_calls"],
        Value::from(reference.track_step_calls)
    );
    assert_eq!(reference.track_step_calls, reference.frames.len() - 1);
    assert_eq!(output.instances.len(), 27);
    let actual_masks = mlxcel_core::array_to_raw_bytes(output.tensors.get("masks").unwrap());
    let pixels_per_mask = 640 * 360;
    for (frame_index, frame) in reference.frames.iter().enumerate() {
        assert_eq!(frame.frame as usize, frame_index);
        assert_eq!(frame.objects.len(), 9);
        for (object_index, expected) in frame.objects.iter().enumerate() {
            let row = frame_index * 9 + object_index;
            let instance = &output.instances[row];
            assert_eq!(instance.frame, Some(frame.frame));
            assert_eq!(instance.object_id, Some(expected.object_id));
            assert!(
                (instance.score - expected.score).abs() <= 0.01,
                "frame {frame_index} object {object_index} score {} vs {}",
                instance.score,
                expected.score
            );
            for coordinate in 0..4 {
                assert!(
                    (instance.bbox[coordinate] - expected.bbox[coordinate]).abs() <= 2.0,
                    "frame {frame_index} object {object_index} box {coordinate}: {} vs {}",
                    instance.bbox[coordinate],
                    expected.bbox[coordinate]
                );
            }
            let expected_mask = image::load_from_memory(VIDEO_MASKS[frame_index][object_index])
                .expect("decode upstream video mask")
                .to_luma8();
            assert_eq!(
                expected_mask.pixels().filter(|pixel| pixel[0] != 0).count(),
                expected.foreground
            );
            assert!(expected.foreground > 0);
            let actual = &actual_masks[row * pixels_per_mask..(row + 1) * pixels_per_mask];
            assert!(
                mask_iou(actual, expected_mask.as_raw()) >= 0.95,
                "frame {frame_index} object {object_index} mask IoU below 0.95"
            );
        }
    }

    let single = predictor
        .predict(
            &DenseInput::Video {
                frames: vec![first_frame.clone()],
                fps: 6.0,
            },
            &Map::from_iter([
                ("task".into(), Value::String("track".into())),
                (
                    "prompts".into(),
                    Value::Array(vec![Value::String("a car".into())]),
                ),
                ("threshold".into(), Value::from(0.15)),
                ("nms".into(), Value::from(0.5)),
            ]),
        )
        .expect("track a single-frame video");
    assert_eq!(single.instances.len(), 9);
    assert!(single.instances.iter().all(|row| row.frame == Some(0)));
    assert_eq!(single.meta["frames"], Value::from(1));
    assert_eq!(single.meta["track_step_calls"], Value::from(0));

    let empty = predictor
        .predict(
            &DenseInput::Video {
                frames: vec![first_frame],
                fps: 6.0,
            },
            &Map::from_iter([
                ("task".into(), Value::String("track".into())),
                (
                    "prompts".into(),
                    Value::Array(vec![Value::String("a car".into())]),
                ),
                ("threshold".into(), Value::from(1.0)),
                ("nms".into(), Value::from(0.5)),
            ]),
        )
        .expect("write an empty one-frame result when no detection passes");
    assert!(empty.instances.is_empty());
    assert_eq!(empty.meta["frames"], Value::from(1));
    assert_eq!(empty.meta["track_step_calls"], Value::from(0));
    assert_eq!(
        mlxcel_core::array_shape(empty.tensors.get("masks").unwrap()),
        [0, 360, 640]
    );
}

fn model_dir() -> String {
    std::env::var("MLXCEL_SAM3_MODEL")
        .expect("MLXCEL_SAM3_MODEL must select the pinned SAM 3.1 checkpoint")
}

fn mask_iou(left: &[u8], right: &[u8]) -> f32 {
    let intersection = left
        .iter()
        .zip(right)
        .filter(|(left, right)| **left != 0 && **right != 0)
        .count();
    let union = left
        .iter()
        .zip(right)
        .filter(|(left, right)| **left != 0 || **right != 0)
        .count();
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}
