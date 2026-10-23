// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Opt-in parity tests against authenticated RF-DETR references.

use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;

use super::postprocess::postprocess;
use super::predictor::{RfDetrPredictor, decode_masks};

const FIXTURE_DIR: &str = "tests/fixtures/rfdetr";

fn reference_directory() -> PathBuf {
    std::env::var_os("MLXCEL_TEST_RFDETR_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR))
}

fn checkpoint(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {variable} to an authenticated RF-DETR checkpoint"))
}

fn bytes(variant: &str, name: &str) -> Vec<u8> {
    std::fs::read(reference_directory().join(variant).join(name)).unwrap()
}

fn tensor<'a>(tensors: &'a SafeTensors<'a>, name: &str, shape: &[usize]) -> TensorView<'a> {
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

fn read(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32))
}

fn max_abs(name: &str, actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{name} finite"
    );
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "{name} reference finite"
    );
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

#[derive(Deserialize)]
struct ExpectedRoot {
    threshold: f32,
    nms_iou: f32,
    num_select: usize,
    source_size: [u32; 2],
    detections: Vec<ExpectedDetection>,
}

#[derive(Deserialize)]
struct ExpectedDetection {
    query_index: usize,
    label: usize,
    class_name: String,
    score: f32,
    #[serde(rename = "box")]
    bbox: [f32; 4],
    mask: Option<String>,
}

fn run_variant(variant: &str, checkpoint_variable: &str, queries: usize, masks: bool) {
    let input_bytes = bytes(variant, "input.safetensors");
    let input_tensors = SafeTensors::deserialize(&input_bytes).unwrap();
    let resolution = if masks { 384 } else { 560 };
    let input = f32_values(tensor(
        &input_tensors,
        "pixel_values",
        &[1, resolution, resolution, 3],
    ));
    let predictor = RfDetrPredictor::from_pretrained(checkpoint(checkpoint_variable), 0.5).unwrap();
    assert_eq!(predictor.processor().num_select(), 300);
    let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sam3/cats.png");
    let processed = predictor.processor().preprocess_path(&image).unwrap();
    assert_eq!(processed.original_size, (480, 640));
    let processed_values = read(&processed.pixel_values);
    let processor_max = max_abs("pixel_values", &processed_values, &input);
    eprintln!("RF-DETR variant={variant} processor_max_abs={processor_max:.9e}");
    assert_eq!(processor_max, 0.0, "processor output must be byte exact");
    let model = predictor.model();
    assert_eq!(model.weight_dtype(), mlxcel_core::dtype::FLOAT32);
    let output = model.forward(&processed.pixel_values).unwrap();
    assert_eq!(
        mlxcel_core::hardware::gpu_backend_kind(),
        mlxcel_core::hardware::GpuBackendKind::Rocm
    );
    let reference_bytes = bytes(variant, "raw.safetensors");
    let reference = SafeTensors::deserialize(&reference_bytes).unwrap();
    let mut names = reference.names();
    names.sort_unstable();
    let expected_names = if masks {
        vec!["pred_boxes", "pred_logits", "pred_masks"]
    } else {
        vec!["pred_boxes", "pred_logits"]
    };
    assert_eq!(names, expected_names);
    let actual_logits = read(&output.pred_logits);
    let actual_boxes = read(&output.pred_boxes);
    let expected_logits = f32_values(tensor(&reference, "pred_logits", &[1, queries, 91]));
    let expected_boxes = f32_values(tensor(&reference, "pred_boxes", &[1, queries, 4]));
    let logits_max = max_abs("pred_logits", &actual_logits, &expected_logits);
    let boxes_max = max_abs("pred_boxes", &actual_boxes, &expected_boxes);
    eprintln!(
        "RF-DETR variant={variant} actual_backend=rocm actual_weight_dtype=F32 logits_max_abs={logits_max:.9e} boxes_max_abs={boxes_max:.9e}"
    );
    assert!(logits_max <= 1.0e-3, "logits max abs {logits_max}");
    assert!(boxes_max <= 1.0e-3, "boxes max abs {boxes_max}");
    if masks {
        let pred_masks = output.pred_masks.as_deref().expect("seg masks");
        let actual_masks = read(pred_masks);
        let expected_masks = f32_values(tensor(
            &reference,
            "pred_masks",
            &[1, queries, resolution / 4, resolution / 4],
        ));
        let masks_max = max_abs("pred_masks", &actual_masks, &expected_masks);
        eprintln!("RF-DETR variant=seg pred_masks_max_abs={masks_max:.9e}");
        assert!(masks_max <= 1.0e-3, "masks max abs {masks_max}");
    }

    let expected: ExpectedRoot =
        serde_json::from_slice(&bytes(variant, "detections.json")).unwrap();
    let decoded = postprocess(
        &actual_logits,
        &actual_boxes,
        queries,
        91,
        expected.source_size[1],
        expected.source_size[0],
        expected.threshold,
        expected.num_select,
        expected.nms_iou,
        None,
    )
    .unwrap();
    assert_eq!(decoded.len(), expected.detections.len());
    for (actual, expected) in decoded.iter().zip(&expected.detections) {
        assert_eq!(actual.query_index, expected.query_index);
        assert_eq!(actual.label, expected.label);
        assert_eq!(actual.class_name, expected.class_name);
        assert!((actual.score - expected.score).abs() <= 1.0e-3);
        for (actual, expected) in actual.bbox.iter().zip(expected.bbox) {
            assert!(
                (actual - expected).abs() <= 1.0,
                "box {actual} vs {expected}"
            );
        }
    }
    if masks {
        let actual_masks = decode_masks(
            output.pred_masks.as_deref().unwrap(),
            &decoded,
            queries,
            expected.source_size[1],
            expected.source_size[0],
        )
        .unwrap();
        for (index, (actual, expected)) in actual_masks
            .into_iter()
            .zip(&expected.detections)
            .enumerate()
        {
            let actual = actual.expect("segmentation detection mask");
            let expected_name = expected.mask.as_deref().expect("reference mask filename");
            let expected = image::open(reference_directory().join(variant).join(expected_name))
                .unwrap()
                .to_luma8();
            assert_eq!(actual.dimensions(), expected.dimensions());
            let mut intersection = 0usize;
            let mut union = 0usize;
            for (&actual, &expected) in actual.as_raw().iter().zip(expected.as_raw()) {
                let actual = actual > 0;
                let expected = expected > 0;
                intersection += usize::from(actual && expected);
                union += usize::from(actual || expected);
            }
            let iou = intersection as f64 / union.max(1) as f64;
            eprintln!("RF-DETR mask={index} IoU={iou:.9}");
            assert!(iou >= 0.98, "mask {index} IoU {iou}");
        }
    }
}

#[test]
#[ignore = "requires authenticated Base checkpoint and ROCm GPU"]
fn base_raw_and_postprocess_parity() {
    run_variant("base", "MLXCEL_TEST_RFDETR_BASE_DIR", 300, false);
}

#[test]
#[ignore = "requires authenticated Seg Small checkpoint and ROCm GPU"]
fn seg_small_raw_and_postprocess_parity() {
    run_variant("seg", "MLXCEL_TEST_RFDETR_SEG_DIR", 100, true);
}
