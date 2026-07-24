// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

use super::*;
use crate::prepared::{
    DecodePositionState, PreparedInputError, PreparedIreePrefill, PreparedPositionMode,
};
use mlxcel_core::session::{
    PreparedAttentionBias, PreparedModality, PreparedPositions, PreparedTensorDType,
};

fn tensor_i32(shape: &[usize], values: impl IntoIterator<Item = i32>) -> OwnedTensor {
    let bytes = values
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    OwnedTensor::new(bytes, PreparedTensorDType::Int32, shape.to_vec()).unwrap()
}

fn tensor_f32(shape: &[usize], values: impl IntoIterator<Item = f32>) -> OwnedTensor {
    let bytes = values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    OwnedTensor::new(bytes, PreparedTensorDType::Float32, shape.to_vec()).unwrap()
}

fn prepared(modality_tokens: usize) -> PreparedPrefill {
    PreparedPrefill::new(
        vec![10, 11, 12, 13],
        tensor_f32(&[1, 4, 2], [0.0; 8]),
        PreparedPositions::Sequential {
            start: 0,
            length: 4,
        },
        PreparedAttentionBias {
            tensor: tensor_f32(&[1, 1, 1, 4], [0.0; 4]),
            causal: true,
        },
        vec![PreparedModality {
            family: "qwen3_vl".into(),
            item_count: usize::from(modality_tokens != 0),
            token_count: modality_tokens,
        }],
    )
    .unwrap()
}

fn features(positions: &[i32], layers: &[i32], hidden: usize) -> DeepStackFeatures {
    let value_count = layers.len() * positions.len() * hidden;
    DeepStackFeatures::new(
        tensor_i32(&[positions.len()], positions.iter().copied()),
        tensor_f32(
            &[layers.len(), positions.len(), hidden],
            (0..value_count).map(|index| index as f32 + 1.0),
        ),
        tensor_i32(&[layers.len()], layers.iter().copied()),
    )
    .unwrap()
}

fn schema() -> DeepStackConfig {
    DeepStackConfig {
        target_layer_indices: vec![0, 2, 3],
        max_visual_positions: 4,
    }
}

fn public_mrope_request(rope_delta: i32) -> DeepStackPreparedPrefill {
    let mut value = prepared(2);
    value.positions = PreparedPositions::Mrope3D {
        tensor: tensor_i32(&[3, 4], [0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 2, 3]),
        rope_delta,
    };
    DeepStackPreparedPrefill::new(value, features(&[1, 3], &[0, 2, 3], 2)).unwrap()
}

#[test]
fn public_deepstack_request_preserves_mrope_delta_and_rejects_wrong_mode() {
    for rope_delta in [-2, 5] {
        let request = public_mrope_request(rope_delta);
        let prepared = PreparedIreePrefill::prepare_for_mode(
            request.prepared(),
            2,
            8,
            PreparedPositionMode::Mrope3D,
        )
        .unwrap();
        assert_eq!(prepared.positions.mode(), PreparedPositionMode::Mrope3D);
        assert_eq!(prepared.positions.rope_delta(), rope_delta);
        assert_eq!(prepared.positions.values().len(), 3 * 8);

        let error = PreparedIreePrefill::prepare_for_mode(
            request.prepared(),
            2,
            8,
            PreparedPositionMode::OneD,
        )
        .unwrap_err();
        assert_eq!(
            error,
            PreparedInputError::PositionModeMismatch {
                expected: PreparedPositionMode::OneD,
                actual: PreparedPositionMode::Mrope3D,
            }
        );
    }
}

#[test]
fn public_deepstack_delta_commits_single_decode_coordinate_only_on_success() {
    let successful = public_mrope_request(-2);
    let successful = PreparedIreePrefill::prepare_for_mode(
        successful.prepared(),
        2,
        8,
        PreparedPositionMode::Mrope3D,
    )
    .unwrap();
    let mut state = DecodePositionState::default();
    assert_eq!(
        state
            .complete_prefill(successful.positions.rope_delta(), Ok(17))
            .unwrap(),
        17
    );
    assert_eq!(
        state.mrope_coordinate(successful.effective_len as i32),
        Ok(2),
        "the first decode coordinate is sequence_len + signed delta"
    );

    let failed = public_mrope_request(5);
    let failed = PreparedIreePrefill::prepare_for_mode(
        failed.prepared(),
        2,
        8,
        PreparedPositionMode::Mrope3D,
    )
    .unwrap();
    assert_eq!(
        state.complete_prefill::<i32>(
            failed.positions.rope_delta(),
            Err("native prefill failed".to_string()),
        ),
        Err("native prefill failed".to_string())
    );
    assert_eq!(
        state.mrope_coordinate(successful.effective_len as i32),
        Ok(2),
        "a failed prefill must retain the previously committed coordinate"
    );
}

#[test]
fn owns_compact_payload_and_materializes_only_static_runtime_padding() {
    let compact = features(&[1, 3], &[0, 2, 3], 2);
    let request = DeepStackPreparedPrefill::new(prepared(2), compact.clone()).unwrap();
    assert_eq!(request.deepstack().visual_count(), 2);
    assert_eq!(request.deepstack().layer_count(), 3);
    assert_eq!(request.deepstack().hidden_size(), 2);

    let runtime = PreparedDeepStack::prepare(&compact, &schema(), 2).unwrap();
    assert_eq!(runtime.visual_positions, [1, 3, -1, -1]);
    assert_eq!(runtime.layer_indices, [0, 2, 3]);
    assert_eq!(runtime.actual_layer_count, 3);
    assert_eq!(runtime.actual_visual_count, 2);
    assert_eq!(runtime.max_layer_count, 3);
    assert_eq!(runtime.max_visual_count, 4);
    assert_eq!(runtime.hidden_size, 2);
    assert_eq!(runtime.layer_features.len(), 3 * 4 * 2);
    assert_eq!(&runtime.layer_features[0..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&runtime.layer_features[4..8], &[0.0; 4]);
    assert_eq!(&runtime.layer_features[8..12], &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn accepts_zero_visual_positions_without_special_dense_buffers() {
    let compact = features(&[], &[0, 2, 3], 2);
    DeepStackPreparedPrefill::new(prepared(0), compact.clone()).unwrap();
    let runtime = PreparedDeepStack::prepare(&compact, &schema(), 2).unwrap();
    assert_eq!(runtime.actual_visual_count, 0);
    assert_eq!(runtime.visual_positions, [-1; 4]);
    assert!(runtime.layer_features.iter().all(|&value| value == 0.0));
}

#[test]
fn fixed_qwen3_vl_fixture_matches_every_post_hook_residual() {
    // Port of Qwen3VLModel::deepstack_process's issue #650 fixture: hidden
    // states start at one, image positions are 1..=3, and every visual feature
    // component is ten. Repeating the same MLX add-after-layer operation at the
    // first, middle, and last target gives the fixed 11/21/31 residual table.
    const SEQUENCE: usize = 5;
    const HIDDEN: usize = 4;
    const POSITIONS: [usize; 3] = [1, 2, 3];
    const EXPECTED_VISUAL_ROWS: [f32; 3] = [11.0, 21.0, 31.0];

    let mut hidden = vec![1.0f32; SEQUENCE * HIDDEN];
    let mut snapshots = Vec::new();
    for _target_layer in 0..3 {
        for &position in &POSITIONS {
            for column in 0..HIDDEN {
                hidden[position * HIDDEN + column] += 10.0;
            }
        }
        snapshots.push(hidden.clone());
    }

    for (layer, snapshot) in snapshots.iter().enumerate() {
        for position in 0..SEQUENCE {
            for column in 0..HIDDEN {
                let expected = if POSITIONS.contains(&position) {
                    EXPECTED_VISUAL_ROWS[layer]
                } else {
                    1.0
                };
                assert_eq!(
                    snapshot[position * HIDDEN + column],
                    expected,
                    "post-hook layer={layer} position={position} hidden={column}"
                );
            }
        }
    }
}

#[test]
fn request_validation_rejects_position_metadata_and_non_finite_features() {
    for (positions, expected) in [
        (vec![1, 1], DeepStackInputError::PositionsNotSortedUnique),
        (vec![3, 1], DeepStackInputError::PositionsNotSortedUnique),
    ] {
        let error = DeepStackPreparedPrefill::new(prepared(2), features(&positions, &[0, 2, 3], 2))
            .unwrap_err();
        assert_eq!(error, expected);
    }

    let error =
        DeepStackPreparedPrefill::new(prepared(2), features(&[1, 4], &[0, 2, 3], 2)).unwrap_err();
    assert!(matches!(
        error,
        DeepStackInputError::InvalidPosition {
            index: 1,
            position: 4,
            sequence_len: 4
        }
    ));

    let error =
        DeepStackPreparedPrefill::new(prepared(1), features(&[1, 3], &[0, 2, 3], 2)).unwrap_err();
    assert!(matches!(
        error,
        DeepStackInputError::ModalityPositionCount {
            positions: 2,
            modality_tokens: 1
        }
    ));

    let error =
        DeepStackPreparedPrefill::new(prepared(2), features(&[1, 3], &[-1, 2, 3], 2)).unwrap_err();
    assert_eq!(error, DeepStackInputError::InvalidLayerIndex(-1));

    let mut invalid = features(&[1, 3], &[0, 2, 3], 2);
    invalid.layer_features.bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(
        DeepStackPreparedPrefill::new(prepared(2), invalid).unwrap_err(),
        DeepStackInputError::NonFiniteFeature
    );
}

#[test]
fn runtime_validation_rejects_schema_hidden_and_static_bound_mismatches() {
    let compact = features(&[1, 3], &[0, 2, 3], 2);
    let wrong_layers = DeepStackConfig {
        target_layer_indices: vec![0, 1, 3],
        max_visual_positions: 4,
    };
    assert!(matches!(
        PreparedDeepStack::prepare(&compact, &wrong_layers, 2),
        Err(DeepStackInputError::LayerContract { .. })
    ));
    assert!(matches!(
        PreparedDeepStack::prepare(&compact, &schema(), 3),
        Err(DeepStackInputError::HiddenSize {
            actual: 2,
            expected: 3
        })
    ));
    let too_small = DeepStackConfig {
        target_layer_indices: vec![0, 2, 3],
        max_visual_positions: 1,
    };
    assert!(matches!(
        PreparedDeepStack::prepare(&compact, &too_small, 2),
        Err(DeepStackInputError::StaticMaxOverflow {
            actual: 2,
            maximum: 1
        })
    ));
}

#[test]
fn compact_tensor_constructor_rejects_dtype_shape_and_storage_drift() {
    let error = DeepStackFeatures::new(
        tensor_f32(&[2], [1.0, 3.0]),
        tensor_f32(&[1, 2, 2], [0.0; 4]),
        tensor_i32(&[1], [0]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DeepStackInputError::DType {
            tensor: "visual_positions",
            ..
        }
    ));

    let error = DeepStackFeatures::new(
        tensor_i32(&[2], [1, 3]),
        tensor_f32(&[1, 3, 2], [0.0; 6]),
        tensor_i32(&[1], [0]),
    )
    .unwrap_err();
    assert!(matches!(error, DeepStackInputError::Shape { .. }));

    let mut positions = tensor_i32(&[2], [1, 3]);
    positions.bytes.pop();
    let error = DeepStackFeatures::new(
        positions,
        tensor_f32(&[1, 2, 2], [0.0; 4]),
        tensor_i32(&[1], [0]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DeepStackInputError::ByteCount {
            tensor: "visual_positions",
            expected: 8,
            actual: 7
        }
    ));
}
