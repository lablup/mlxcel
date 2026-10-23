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

use std::fs;

use super::deformable::grid_sample_f32;
use super::mask_box::mask_to_box_coordinate;
use super::processor::{ProcessorConfig, checked_values};

fn values(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32))
}

#[test]
fn mask_boxes_cover_empty_single_pixel_and_bottom_right_extent() {
    let mask = mlxcel_core::from_slice_f32(
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // empty
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // top-left
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // bottom-right
        ],
        &[1, 3, 2, 3],
    );
    let actual = mask_to_box_coordinate(&mask).unwrap();
    assert_eq!(mlxcel_core::array_shape(&actual), [1, 3, 4]);
    let actual = values(&actual);
    let expected = [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / 6.0,
        1.0 / 4.0,
        1.0 / 3.0,
        1.0 / 2.0,
        5.0 / 6.0,
        3.0 / 4.0,
        1.0 / 3.0,
        1.0 / 2.0,
    ];
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "index {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn deformable_sampling_computes_f32_and_restores_half_dtype() {
    let input = mlxcel_core::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[1, 2, 2, 1]);
    let grid = mlxcel_core::from_slice_f32(&[0.0, 0.0], &[1, 1, 1, 2]);
    for dtype in [mlxcel_core::dtype::FLOAT16, mlxcel_core::dtype::BFLOAT16] {
        let input = mlxcel_core::astype(&input, dtype);
        let grid = mlxcel_core::astype(&grid, dtype);
        let output = grid_sample_f32(&input, &grid);
        assert_eq!(mlxcel_core::array_dtype(&output), dtype);
        assert!((values(&output)[0] - 1.5).abs() < 1e-3);
    }
}

#[test]
fn processor_config_reads_exact_huggingface_semantics() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("preprocessor_config.json"),
        r#"{
            "do_resize":true,"size":{"height":800,"width":800},"resample":3,
            "do_rescale":true,"rescale_factor":0.00392156862745098,
            "do_normalize":true,"image_mean":[0,0,0],"image_std":[1,1,1]
        }"#,
    )
    .unwrap();
    let config = ProcessorConfig::from_pretrained(directory.path()).unwrap();
    assert_eq!((config.height, config.width), (800, 800));
    assert_eq!(config.image_mean, [0.0; 3]);
    assert_eq!(config.image_std, [1.0; 3]);
}

#[test]
fn processor_rejects_unsupported_resize_and_unbounded_size() {
    for body in [
        r#"{"resample":2,"size":{"height":800,"width":800}}"#,
        r#"{"size":{"height":800,"width":1025}}"#,
    ] {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("preprocessor_config.json"), body).unwrap();
        assert!(ProcessorConfig::from_pretrained(directory.path()).is_err());
    }
}

#[test]
fn processor_rejects_large_resize_intermediate_before_allocation() {
    let error = checked_values(100_000, 800, "test intermediate").unwrap_err();
    assert!(error.contains("limit"));
}

#[test]
fn public_processor_constructor_rejects_invalid_normalization() {
    let mut config = ProcessorConfig::default();
    config.image_std[1] = 0.0;
    assert!(super::processor::PpDocLayoutV3Processor::new(config).is_err());
}

#[test]
fn detector_dispatch_rejects_unknown_model_type() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("config.json"),
        r#"{"model_type":"not_a_detector"}"#,
    )
    .unwrap();
    let error = crate::vision::detection::load_detector(directory.path(), 0.3)
        .err()
        .expect("unknown detector must fail");
    assert_eq!(error, "unsupported detector model_type 'not_a_detector'");
}
