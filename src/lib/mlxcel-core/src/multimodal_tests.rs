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

use super::{
    OwnedTensor, PreparedAttentionBias, PreparedModality, PreparedPositions, PreparedPrefill,
    PreparedPrefillError, PreparedTensorDType,
};

fn zeros(dtype: PreparedTensorDType, shape: &[usize]) -> OwnedTensor {
    let count = shape.iter().product::<usize>() * dtype.size_bytes();
    OwnedTensor::new(vec![0; count], dtype, shape.to_vec()).unwrap()
}

#[test]
fn owned_tensor_checks_each_supported_model_dtype() {
    for dtype in [
        PreparedTensorDType::Float16,
        PreparedTensorDType::BFloat16,
        PreparedTensorDType::Float32,
    ] {
        let tensor = zeros(dtype, &[1, 3, 5]);
        assert_eq!(tensor.bytes.len(), 15 * dtype.size_bytes());
        assert_eq!(tensor.element_count(), 15);
    }
}

#[test]
fn owned_tensor_rejects_wrong_byte_count() {
    let error = OwnedTensor::new(vec![0; 7], PreparedTensorDType::Float32, vec![1, 2]).unwrap_err();
    assert_eq!(
        error,
        PreparedPrefillError::ByteCountMismatch {
            expected: 8,
            actual: 7,
        }
    );
}

#[test]
fn prepared_prefill_is_owned_send_data() {
    fn assert_send<T: Send>() {}
    assert_send::<PreparedPrefill>();

    let prepared = PreparedPrefill::new(
        vec![1, 2, 3],
        zeros(PreparedTensorDType::Float16, &[1, 3, 4]),
        PreparedPositions::Sequential {
            start: 0,
            length: 3,
        },
        PreparedAttentionBias {
            tensor: zeros(PreparedTensorDType::Float32, &[1, 1, 1, 3]),
            causal: true,
        },
        vec![PreparedModality {
            family: "llava".to_string(),
            item_count: 1,
            token_count: 2,
        }],
    )
    .unwrap();

    let moved = std::thread::spawn(move || prepared.sequence_len)
        .join()
        .unwrap();
    assert_eq!(moved, 3);
}

#[test]
fn prepared_prefill_rejects_shape_drift() {
    let error = PreparedPrefill::new(
        vec![1, 2, 3],
        zeros(PreparedTensorDType::Float16, &[1, 2, 4]),
        PreparedPositions::Sequential {
            start: 0,
            length: 3,
        },
        PreparedAttentionBias {
            tensor: zeros(PreparedTensorDType::Float32, &[1, 1, 1, 3]),
            causal: true,
        },
        Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(error, PreparedPrefillError::EmbeddingShape { .. }));
}

#[test]
fn prepared_prefill_rejects_floating_point_explicit_positions() {
    let error = PreparedPrefill::new(
        vec![1, 2, 3],
        zeros(PreparedTensorDType::Float16, &[1, 3, 4]),
        PreparedPositions::Explicit(zeros(PreparedTensorDType::Float32, &[1, 3])),
        PreparedAttentionBias {
            tensor: zeros(PreparedTensorDType::Float32, &[1, 1, 1, 3]),
            causal: true,
        },
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        PreparedPrefillError::PositionDType(PreparedTensorDType::Float32)
    );
}
