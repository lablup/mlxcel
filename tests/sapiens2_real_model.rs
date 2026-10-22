// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Opt-in parity against a published Sapiens2 checkpoint.

use mlxcel::vision::dense::{DenseInput, DenseOutput};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
struct PoseReference {
    keypoints: Vec<Vec<Vec<f32>>>,
    scores: Vec<Vec<f32>>,
    boxes: Vec<Vec<f32>>,
}

fn predict(env: &str, opts: Map<String, Value>) -> DenseOutput {
    let path = std::env::var(env).unwrap_or_else(|_| panic!("set {env}"));
    let predictor = mlxcel::vision::dense::sapiens2::load(path.as_ref()).unwrap();
    let image = image::open("tests/fixtures/sapiens2/person.png").unwrap();
    predictor.predict(&DenseInput::Image(image), &opts).unwrap()
}

fn assert_finite(array: &mlxcel_core::MlxArray) {
    let values = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        array,
        mlxcel_core::dtype::FLOAT32,
    ));
    assert!(values.iter().all(|value| value.is_finite()));
}

fn load_reference(env: &str) -> mlxcel_core::weights::WeightMap {
    let path = std::env::var(env).unwrap_or_else(|_| panic!("set {env}"));
    mlxcel_core::weights::load_safetensors(path).expect("load upstream reference")
}

fn assert_relative_rms(
    actual: &mlxcel_core::MlxArray,
    expected: &mlxcel_core::MlxArray,
    threshold: f64,
) -> f64 {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected)
    );
    assert_finite(actual);
    assert_finite(expected);
    let actual = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        actual,
        mlxcel_core::dtype::FLOAT32,
    ));
    let expected = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        expected,
        mlxcel_core::dtype::FLOAT32,
    ));
    let (squared_error, squared_scale) = actual.iter().zip(&expected).fold(
        (0.0f64, 0.0f64),
        |(error, scale), (actual, expected)| {
            let actual = f64::from(*actual);
            let expected = f64::from(*expected);
            (
                error + (actual - expected).powi(2),
                scale + expected.powi(2),
            )
        },
    );
    let rms = (squared_error / actual.len() as f64).sqrt();
    let scale = (squared_scale / actual.len() as f64).sqrt();
    let relative_rms = rms / scale.max(f64::MIN_POSITIVE);
    assert!(
        relative_rms < threshold,
        "relative RMS {relative_rms:.8} is not below {threshold}"
    );
    relative_rms
}

#[test]
#[ignore = "requires SAPIENS2_POSE_MODEL pointing to sapiens2-pose-0.4b-bf16"]
fn sapiens2_pose_real_checkpoint_flip_off_parity() {
    let mut opts = Map::new();
    opts.insert("flip_test".into(), Value::Bool(false));
    let output = predict("SAPIENS2_POSE_MODEL", opts);
    assert_pose_parity(
        &output,
        include_str!("fixtures/sapiens2/pose-rocm-bf16-flip0.json"),
    );
}

#[test]
#[ignore = "requires SAPIENS2_POSE_MODEL pointing to sapiens2-pose-0.4b-bf16"]
fn sapiens2_pose_real_checkpoint_flip_on_parity() {
    let mut opts = Map::new();
    opts.insert("flip_test".into(), Value::Bool(true));
    let output = predict("SAPIENS2_POSE_MODEL", opts);
    assert_pose_parity(
        &output,
        include_str!("fixtures/sapiens2/pose-rocm-bf16-flip1.json"),
    );
}

fn assert_pose_parity(output: &DenseOutput, reference_json: &str) {
    let reference: PoseReference = serde_json::from_str(reference_json).unwrap();
    assert_finite(&output.tensors["keypoints"]);
    assert_finite(&output.tensors["scores"]);
    assert_finite(&output.tensors["boxes"]);

    let keypoints = mlxcel_core::utils::array_to_vec_f32(&output.tensors["keypoints"]);
    let scores = mlxcel_core::utils::array_to_vec_f32(&output.tensors["scores"]);
    let boxes = mlxcel_core::utils::array_to_vec_f32(&output.tensors["boxes"]);
    let expected_keypoints: Vec<f32> = reference
        .keypoints
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let expected_scores: Vec<f32> = reference.scores.into_iter().flatten().collect();
    let expected_boxes: Vec<f32> = reference.boxes.into_iter().flatten().collect();
    assert!(expected_keypoints.iter().all(|value| value.is_finite()));
    assert!(expected_scores.iter().all(|value| value.is_finite()));
    assert!(expected_boxes.iter().all(|value| value.is_finite()));
    assert_eq!(keypoints.len(), 308 * 2);
    assert_eq!(scores.len(), 308);
    assert_eq!(boxes, expected_boxes);
    let max_error = keypoints
        .chunks_exact(2)
        .zip(expected_keypoints.chunks_exact(2))
        .map(|(actual, expected)| {
            ((actual[0] - expected[0]).powi(2) + (actual[1] - expected[1]).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(max_error <= 1.5, "maximum keypoint error {max_error} px");
    let max_score_error = scores
        .iter()
        .zip(expected_scores)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "Sapiens2 pose parity: points=308, max_keypoint_error={max_error:.6}px, max_score_error={max_score_error:.9}"
    );
    assert!(
        max_score_error <= 0.05,
        "maximum keypoint score error {max_score_error}"
    );
}

#[test]
#[ignore = "requires SAPIENS2_SEG_MODEL pointing to sapiens2-seg-0.4b-bf16"]
fn sapiens2_segmentation_real_checkpoint_parity() {
    let output = predict("SAPIENS2_SEG_MODEL", Map::new());
    assert_finite(&output.tensors["segmentation"]);
    let labels = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        &output.tensors["segmentation"],
        mlxcel_core::dtype::FLOAT32,
    ));
    let reference = image::open("tests/fixtures/sapiens2/seg-rocm-bf16.png")
        .unwrap()
        .into_luma8();
    assert_eq!(labels.len(), reference.as_raw().len());
    let matching = labels
        .iter()
        .zip(reference.as_raw())
        .filter(|(actual, expected)| **actual as u8 == **expected)
        .count();
    let agreement = matching as f64 / labels.len() as f64;
    eprintln!("Sapiens2 segmentation parity: label_agreement={agreement:.9}");
    assert!(
        agreement >= 0.995,
        "segmentation label agreement {agreement:.6}"
    );
}

#[test]
#[ignore = "requires SAPIENS2_NORMAL_MODEL and SAPIENS2_NORMAL_REFERENCE"]
fn sapiens2_normal_real_checkpoint_parity() {
    let output = predict("SAPIENS2_NORMAL_MODEL", Map::new());
    assert_eq!(
        mlxcel_core::array_shape(&output.tensors["normals"]),
        [720, 1280, 3]
    );
    let reference = load_reference("SAPIENS2_NORMAL_REFERENCE");
    let relative_rms = assert_relative_rms(&output.tensors["normals"], &reference["normals"], 1e-2);
    eprintln!("Sapiens2 normal parity: relative_rms={relative_rms:.9}");
}

#[test]
#[ignore = "requires SAPIENS2_POINTMAP_MODEL and SAPIENS2_POINTMAP_REFERENCE"]
fn sapiens2_pointmap_real_checkpoint_parity() {
    let output = predict("SAPIENS2_POINTMAP_MODEL", Map::new());
    assert_eq!(
        mlxcel_core::array_shape(&output.tensors["pointmaps"]),
        [720, 1280, 3]
    );
    assert_eq!(mlxcel_core::array_shape(&output.tensors["scales"]), [1]);
    let reference = load_reference("SAPIENS2_POINTMAP_REFERENCE");
    let relative_rms =
        assert_relative_rms(&output.tensors["pointmaps"], &reference["pointmaps"], 1e-2);
    eprintln!("Sapiens2 point-map parity: relative_rms={relative_rms:.9}");
    assert_finite(&output.tensors["scales"]);
}

#[test]
#[ignore = "requires SAPIENS2_MATTING_MODEL pointing to sapiens2-matting-1b-bf16"]
fn sapiens2_matting_real_checkpoint_smoke() {
    let output = predict("SAPIENS2_MATTING_MODEL", Map::new());
    assert_eq!(
        mlxcel_core::array_shape(&output.tensors["alphas"]),
        [720, 1280]
    );
    assert_eq!(
        mlxcel_core::array_shape(&output.tensors["foregrounds"]),
        [720, 1280, 3]
    );
    assert_finite(&output.tensors["alphas"]);
    assert_finite(&output.tensors["foregrounds"]);
    eprintln!("Sapiens2 matting smoke: finite alphas and foregrounds");
}
