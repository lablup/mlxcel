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

use std::path::Path;

use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::{Map, Value, json};

use super::super::{DenseInput, DensePredictor};
use super::predictor::Sam3Predictor;

const TEXT_REFERENCE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/text-reference.safetensors"
));
const CAT_IMAGE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/cats.png"
));
const CAT_REFERENCE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/cats-reference.safetensors"
));
const BOX_REFERENCE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/box-reference.safetensors"
));

#[test]
#[ignore = "loads the real SAM 3 checkpoint and runs CLIP on the accelerator"]
fn pinned_checkpoint_matches_mlx_vlm_text_and_cat() {
    let Ok(model_dir) = std::env::var("MLXCEL_SAM3_MODEL") else {
        eprintln!("skip: MLXCEL_SAM3_MODEL is not set");
        return;
    };
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor = Sam3Predictor::load(Path::new(&model_dir)).expect("load SAM 3 checkpoint");
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

    let image = image::load_from_memory(CAT_IMAGE).expect("decode cat fixture");
    let input = DenseInput::Image(image);
    let opts = Map::from_iter([
        ("task".into(), Value::String("segment".into())),
        (
            "prompts".into(),
            Value::Array(vec![Value::String("a cat".into())]),
        ),
        ("threshold".into(), Value::from(0.3)),
        ("nms".into(), Value::from(0.5)),
    ]);
    let output = predictor
        .predict(&input, &opts)
        .expect("run SAM 3 cat segmentation");
    let reference = SafeTensors::deserialize(CAT_REFERENCE).expect("parse cat reference");
    assert_cat_parity(&output, &reference);

    let box_opts = Map::from_iter([
        ("task".into(), Value::String("segment".into())),
        (
            "prompts".into(),
            Value::Array(vec![Value::String("a cat".into())]),
        ),
        ("boxes".into(), json!([[0.0, 50.0, 350.0, 480.0]])),
        ("threshold".into(), Value::from(0.3)),
        ("nms".into(), Value::from(0.5)),
    ]);
    let box_output = predictor
        .predict(&input, &box_opts)
        .expect("run SAM 3 box-prompt segmentation");
    let box_reference =
        SafeTensors::deserialize(BOX_REFERENCE).expect("parse box-prompt reference");
    assert_cat_parity(&box_output, &box_reference);
}

fn assert_cat_parity(output: &super::super::DenseOutput, reference: &SafeTensors<'_>) {
    let expected_boxes = tensor_f32(reference.tensor("boxes").unwrap());
    let expected_scores = tensor_f32(reference.tensor("scores").unwrap());
    let expected_masks = reference.tensor("masks").unwrap();
    assert_eq!(expected_masks.dtype(), Dtype::U8);
    assert_eq!(expected_masks.shape(), &[2, 480, 640]);
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
                box_error(instance.bbox, &expected_boxes, **left).total_cmp(&box_error(
                    instance.bbox,
                    &expected_boxes,
                    **right,
                ))
            })
            .expect("unmatched reference cat");
        let max_box_error = (0..4)
            .map(|coordinate| {
                (instance.bbox[coordinate] - expected_boxes[reference_index * 4 + coordinate]).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(
            max_box_error <= 2.0,
            "cat box {reference_index} max error {max_box_error}"
        );
        let score_error = (instance.score - expected_scores[reference_index]).abs();
        assert!(
            score_error <= 0.01,
            "cat score {reference_index} error {score_error}"
        );
        let actual_index = instance.mask_index.expect("cat mask index");
        let iou = binary_mask_iou(
            &actual_masks[actual_index * 480 * 640..(actual_index + 1) * 480 * 640],
            &expected_masks.data()[reference_index * 480 * 640..(reference_index + 1) * 480 * 640],
        );
        assert!(iou >= 0.98, "cat mask {reference_index} IoU {iou}");
        unmatched.swap_remove(slot);
    }
}

fn box_error(actual: [f32; 4], expected: &[f32], index: usize) -> f32 {
    (0..4)
        .map(|coordinate| (actual[coordinate] - expected[index * 4 + coordinate]).abs())
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

fn tensor_i64_as_f32(tensor: TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::I64);
    tensor
        .data()
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()) as f32)
        .collect()
}

fn tensor_bf16_as_f32(tensor: TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::BF16);
    tensor
        .data()
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_le_bytes(bytes.try_into().unwrap());
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect()
}

fn tensor_f32(tensor: TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), Dtype::F32);
    tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}
