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

//! Validation and static-bucket materialization for owned multimodal prefill.

use std::fmt;

use mlxcel_core::session::{OwnedTensor, PreparedPositions, PreparedPrefill, PreparedTensorDType};

const MASKED_VALUE: f32 = -1.0e30;

/// A validated prepared prefill in the exact static shapes consumed by IREE.
#[derive(Debug)]
pub(crate) struct PreparedIreePrefill {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) embeddings: Vec<f32>,
    pub(crate) positions: Vec<i32>,
    pub(crate) attention_bias: Vec<f32>,
    pub(crate) effective_len: usize,
    pub(crate) hidden_size: usize,
    pub(crate) context_capacity: usize,
}

/// Why an owned prepared payload cannot enter the IREE runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedInputError {
    Empty,
    SequenceLength {
        declared: usize,
        token_ids: usize,
    },
    Capacity {
        effective_len: usize,
        context_capacity: usize,
    },
    Slot {
        slot: usize,
        slot_count: usize,
    },
    EmbeddingDType(PreparedTensorDType),
    EmbeddingShape {
        shape: Vec<usize>,
        expected: Vec<usize>,
    },
    HiddenSize {
        actual: usize,
        expected: usize,
    },
    PositionDType(PreparedTensorDType),
    PositionShape(Vec<usize>),
    UnsupportedPositions,
    AttentionBiasDType(PreparedTensorDType),
    AttentionBiasShape(Vec<usize>),
    ByteCount {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidFloat {
        tensor: &'static str,
    },
    ShapeOverflow,
}

impl fmt::Display for PreparedInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("prepared IREE prefill requires a non-empty sequence"),
            Self::SequenceLength {
                declared,
                token_ids,
            } => write!(
                f,
                "prepared sequence_len={declared} disagrees with {} logical token ids",
                token_ids
            ),
            Self::Capacity {
                effective_len,
                context_capacity,
            } => write!(
                f,
                "prepared effective length {effective_len} exceeds context_capacity={context_capacity}"
            ),
            Self::Slot { slot, slot_count } => {
                write!(f, "prepared slot {slot} is outside [0,{slot_count})")
            }
            Self::EmbeddingDType(dtype) => {
                write!(f, "IREE prepared embeddings must be Float32, got {dtype:?}")
            }
            Self::EmbeddingShape { shape, expected } => {
                write!(f, "prepared embedding shape {shape:?} must be {expected:?}")
            }
            Self::HiddenSize { actual, expected } => {
                write!(f, "prepared hidden size {actual} must match model hidden size {expected}")
            }
            Self::PositionDType(dtype) => {
                write!(f, "IREE prepared positions must be Int32, got {dtype:?}")
            }
            Self::PositionShape(shape) => write!(
                f,
                "prepared explicit positions shape {shape:?} must be [sequence] or [1, sequence]"
            ),
            Self::UnsupportedPositions => f.write_str(
                "IREE prepared positions must be the dense zero-based sequence; M-RoPE/offset positions are not supported",
            ),
            Self::AttentionBiasDType(dtype) => {
                write!(f, "IREE prepared attention bias must be Float32, got {dtype:?}")
            }
            Self::AttentionBiasShape(shape) => write!(
                f,
                "prepared attention bias shape {shape:?} must be [sequence], [1,1,1,sequence], [sequence,sequence], or [1,1,sequence,sequence]"
            ),
            Self::ByteCount {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "prepared {tensor} byte count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidFloat { tensor } => {
                write!(f, "prepared {tensor} contains NaN or an invalid positive mask value")
            }
            Self::ShapeOverflow => f.write_str("prepared IREE static shape overflowed"),
        }
    }
}

impl std::error::Error for PreparedInputError {}

pub(crate) fn validate_slot(slot: usize, slot_count: usize) -> Result<(), PreparedInputError> {
    if slot >= slot_count {
        return Err(PreparedInputError::Slot { slot, slot_count });
    }
    Ok(())
}

fn checked_count(shape: &[usize], element_size: usize) -> Result<usize, PreparedInputError> {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .and_then(|count| count.checked_mul(element_size))
        .ok_or(PreparedInputError::ShapeOverflow)
}

fn validate_bytes(tensor: &'static str, value: &OwnedTensor) -> Result<(), PreparedInputError> {
    let expected = checked_count(&value.shape, value.dtype.size_bytes())?;
    if value.bytes.len() != expected {
        return Err(PreparedInputError::ByteCount {
            tensor,
            expected,
            actual: value.bytes.len(),
        });
    }
    Ok(())
}

fn read_f32(tensor: &'static str, value: &OwnedTensor) -> Result<Vec<f32>, PreparedInputError> {
    validate_bytes(tensor, value)?;
    Ok(value
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn read_i32(tensor: &'static str, value: &OwnedTensor) -> Result<Vec<i32>, PreparedInputError> {
    validate_bytes(tensor, value)?;
    Ok(value
        .bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

impl PreparedIreePrefill {
    pub(crate) fn prepare(
        value: &PreparedPrefill,
        hidden_size: usize,
        context_capacity: usize,
    ) -> Result<Self, PreparedInputError> {
        let effective_len = value.sequence_len;
        if effective_len == 0 {
            return Err(PreparedInputError::Empty);
        }
        if effective_len != value.token_ids.len() {
            return Err(PreparedInputError::SequenceLength {
                declared: effective_len,
                token_ids: value.token_ids.len(),
            });
        }
        if effective_len > context_capacity {
            return Err(PreparedInputError::Capacity {
                effective_len,
                context_capacity,
            });
        }
        if value.embeddings.dtype != PreparedTensorDType::Float32 {
            return Err(PreparedInputError::EmbeddingDType(value.embeddings.dtype));
        }
        let expected_embedding_shape = vec![1, effective_len, hidden_size];
        if value.embeddings.shape != expected_embedding_shape {
            let actual_hidden = value.embeddings.shape.get(2).copied();
            if value.embeddings.shape.len() == 3
                && value.embeddings.shape[0] == 1
                && value.embeddings.shape[1] == effective_len
                && actual_hidden != Some(hidden_size)
            {
                return Err(PreparedInputError::HiddenSize {
                    actual: actual_hidden.unwrap_or(0),
                    expected: hidden_size,
                });
            }
            return Err(PreparedInputError::EmbeddingShape {
                shape: value.embeddings.shape.clone(),
                expected: expected_embedding_shape,
            });
        }
        let input_embeddings = read_f32("embeddings", &value.embeddings)?;
        if input_embeddings.iter().any(|v| !v.is_finite()) {
            return Err(PreparedInputError::InvalidFloat {
                tensor: "embeddings",
            });
        }
        let embedding_count = context_capacity
            .checked_mul(hidden_size)
            .ok_or(PreparedInputError::ShapeOverflow)?;
        let mut embeddings = vec![0.0; embedding_count];
        embeddings[..input_embeddings.len()].copy_from_slice(&input_embeddings);

        let mut positions: Vec<i32> = (0..context_capacity)
            .map(|position| i32::try_from(position).map_err(|_| PreparedInputError::ShapeOverflow))
            .collect::<Result<_, _>>()?;
        match &value.positions {
            PreparedPositions::Sequential { start, length } => {
                if *start != 0 || *length != effective_len {
                    return Err(PreparedInputError::UnsupportedPositions);
                }
            }
            PreparedPositions::Explicit(tensor) => {
                if tensor.dtype != PreparedTensorDType::Int32 {
                    return Err(PreparedInputError::PositionDType(tensor.dtype));
                }
                if tensor.shape != [effective_len] && tensor.shape != [1, effective_len] {
                    return Err(PreparedInputError::PositionShape(tensor.shape.clone()));
                }
                let explicit = read_i32("positions", tensor)?;
                if explicit
                    .iter()
                    .enumerate()
                    .any(|(index, &position)| position != index as i32)
                {
                    return Err(PreparedInputError::UnsupportedPositions);
                }
                positions[..effective_len].copy_from_slice(&explicit);
            }
            _ => return Err(PreparedInputError::UnsupportedPositions),
        }

        let bias_tensor = &value.attention_bias.tensor;
        if bias_tensor.dtype != PreparedTensorDType::Float32 {
            return Err(PreparedInputError::AttentionBiasDType(bias_tensor.dtype));
        }
        let compact_bias = read_f32("attention bias", bias_tensor)?;
        if compact_bias
            .iter()
            .any(|value| value.is_nan() || *value > 0.0)
        {
            return Err(PreparedInputError::InvalidFloat {
                tensor: "attention bias",
            });
        }
        let key_bias =
            bias_tensor.shape == [effective_len] || bias_tensor.shape == [1, 1, 1, effective_len];
        let matrix_bias = bias_tensor.shape == [effective_len, effective_len]
            || bias_tensor.shape == [1, 1, effective_len, effective_len];
        if !key_bias && !matrix_bias {
            return Err(PreparedInputError::AttentionBiasShape(
                bias_tensor.shape.clone(),
            ));
        }
        let bias_count = context_capacity
            .checked_mul(context_capacity)
            .ok_or(PreparedInputError::ShapeOverflow)?;
        let mut attention_bias = vec![MASKED_VALUE; bias_count];
        for query in 0..effective_len {
            for key in 0..effective_len {
                let base = if key_bias {
                    compact_bias[key]
                } else {
                    compact_bias[query * effective_len + key]
                };
                attention_bias[query * context_capacity + key] =
                    if value.attention_bias.causal && key > query {
                        MASKED_VALUE
                    } else {
                        base
                    };
            }
        }

        Ok(Self {
            token_ids: value.token_ids.clone(),
            embeddings,
            positions,
            attention_bias,
            effective_len,
            hidden_size,
            context_capacity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::session::{PreparedAttentionBias, PreparedModality};

    fn tensor_f32(shape: &[usize], values: impl IntoIterator<Item = f32>) -> OwnedTensor {
        let bytes = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        OwnedTensor::new(bytes, PreparedTensorDType::Float32, shape.to_vec()).unwrap()
    }

    fn fixture() -> PreparedPrefill {
        PreparedPrefill::new(
            vec![11, 22, 33],
            tensor_f32(&[1, 3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            PreparedPositions::Sequential {
                start: 0,
                length: 3,
            },
            PreparedAttentionBias {
                tensor: tensor_f32(&[1, 1, 1, 3], [0.0, 0.0, 0.0]),
                causal: true,
            },
            vec![PreparedModality {
                family: "test".into(),
                item_count: 1,
                token_count: 1,
            }],
        )
        .unwrap()
    }

    #[test]
    fn materializes_static_shapes_and_causal_bias() {
        let prepared = PreparedIreePrefill::prepare(&fixture(), 2, 5).unwrap();
        assert_eq!(prepared.effective_len, 3);
        assert_eq!(prepared.token_ids, [11, 22, 33]);
        assert_eq!(prepared.embeddings.len(), 10);
        assert_eq!(&prepared.embeddings[..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(&prepared.positions, &[0, 1, 2, 3, 4]);
        assert_eq!(prepared.attention_bias[0], 0.0);
        assert_eq!(prepared.attention_bias[1], MASKED_VALUE);
        assert_eq!(prepared.attention_bias[2 * 5 + 2], 0.0);
        assert_eq!(prepared.attention_bias[3 * 5], MASKED_VALUE);
    }

    #[test]
    fn rejects_wrong_dtype_rank_hidden_bytes_mask_and_capacity() {
        let mut value = fixture();
        value.embeddings.dtype = PreparedTensorDType::Float16;
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 5),
            Err(PreparedInputError::EmbeddingDType(_))
        ));

        let mut value = fixture();
        value.embeddings.shape = vec![3, 2];
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 5),
            Err(PreparedInputError::EmbeddingShape { .. })
        ));

        let value = fixture();
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 4, 5),
            Err(PreparedInputError::HiddenSize { .. })
        ));

        let mut value = fixture();
        value.embeddings.bytes.pop();
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 5),
            Err(PreparedInputError::ByteCount { .. })
        ));

        let mut value = fixture();
        value.attention_bias.tensor.shape = vec![1, 3, 1];
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 5),
            Err(PreparedInputError::AttentionBiasShape(_))
        ));

        let value = fixture();
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 2),
            Err(PreparedInputError::Capacity { .. })
        ));
    }

    #[test]
    fn rejects_overlong_host_buffer_without_touching_canary() {
        let mut value = fixture();
        let canary = [0xA5, 0x5A, 0xC3, 0x3C];
        value.embeddings.bytes.extend_from_slice(&canary);

        let error = PreparedIreePrefill::prepare(&value, 2, 5).unwrap_err();
        assert_eq!(
            error,
            PreparedInputError::ByteCount {
                tensor: "embeddings",
                expected: 24,
                actual: 28,
            }
        );
        assert_eq!(&value.embeddings.bytes[24..], &canary);
    }

    #[test]
    fn rejects_invalid_real_len_slot_inputs_and_positions() {
        assert_eq!(
            validate_slot(4, 4).unwrap_err(),
            PreparedInputError::Slot {
                slot: 4,
                slot_count: 4
            }
        );

        let mut value = fixture();
        value.sequence_len = 2;
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 5),
            Err(PreparedInputError::SequenceLength { .. })
        ));

        let mut value = fixture();
        value.positions = PreparedPositions::Sequential {
            start: 1,
            length: 3,
        };
        assert_eq!(
            PreparedIreePrefill::prepare(&value, 2, 5).unwrap_err(),
            PreparedInputError::UnsupportedPositions
        );

        let mut value = fixture();
        value.attention_bias.tensor = tensor_f32(&[1, 1, 1, 3], [0.0, f32::NAN, 0.0]);
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 5),
            Err(PreparedInputError::InvalidFloat { .. })
        ));
    }
}
