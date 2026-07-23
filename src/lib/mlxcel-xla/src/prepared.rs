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

/// Position schema compiled into an IREE language-model bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedPositionMode {
    OneD,
    Mrope3D,
}

impl PreparedPositionMode {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::OneD => "1D",
            Self::Mrope3D => "M-RoPE 3D",
        }
    }

    pub(crate) const fn ffi_code(self) -> i32 {
        match self {
            Self::OneD => 0,
            Self::Mrope3D => 1,
        }
    }
}

/// Static-bucket positions retained with their explicit semantic mode.
#[derive(Debug)]
pub(crate) enum PreparedIreePositions {
    OneD(Vec<i32>),
    Mrope3D {
        /// Row-major `[3, context_capacity]`, ordered temporal/height/width.
        values: Vec<i32>,
        rope_delta: i32,
    },
}

impl PreparedIreePositions {
    pub(crate) const fn mode(&self) -> PreparedPositionMode {
        match self {
            Self::OneD(_) => PreparedPositionMode::OneD,
            Self::Mrope3D { .. } => PreparedPositionMode::Mrope3D,
        }
    }

    pub(crate) fn values(&self) -> &[i32] {
        match self {
            Self::OneD(values) | Self::Mrope3D { values, .. } => values,
        }
    }

    pub(crate) const fn rope_delta(&self) -> i32 {
        match self {
            Self::OneD(_) => 0,
            Self::Mrope3D { rope_delta, .. } => *rope_delta,
        }
    }
}

/// A validated prepared prefill in the exact static shapes consumed by IREE.
#[derive(Debug)]
pub(crate) struct PreparedIreePrefill {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) embeddings: Vec<f32>,
    pub(crate) positions: PreparedIreePositions,
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
    PositionShape {
        shape: Vec<usize>,
        expected: Vec<usize>,
    },
    PositionModeMismatch {
        expected: PreparedPositionMode,
        actual: PreparedPositionMode,
    },
    NegativePosition {
        axis: usize,
        index: usize,
        value: i32,
    },
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
            Self::PositionShape { shape, expected } => write!(
                f,
                "prepared explicit positions shape {shape:?} must be {expected:?}"
            ),
            Self::PositionModeMismatch { expected, actual } => write!(
                f,
                "prepared position mode {} disagrees with loaded model mode {}",
                actual.name(),
                expected.name()
            ),
            Self::NegativePosition {
                axis,
                index,
                value,
            } => write!(
                f,
                "prepared M-RoPE position axis {axis} index {index} is negative ({value})"
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MropeCoordinateError {
    Overflow { cache_len: i32, rope_delta: i32 },
    Negative { cache_len: i32, rope_delta: i32 },
}

impl fmt::Display for MropeCoordinateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow {
                cache_len,
                rope_delta,
            } => write!(
                f,
                "M-RoPE decode coordinate overflow: cache_len={cache_len}, rope_delta={rope_delta}"
            ),
            Self::Negative {
                cache_len,
                rope_delta,
            } => write!(
                f,
                "M-RoPE decode coordinate is negative: cache_len={cache_len}, rope_delta={rope_delta}"
            ),
        }
    }
}

impl std::error::Error for MropeCoordinateError {}

/// Build the static position buffer for text-only input.
///
/// Accepting backend-neutral sequential or explicit 1D positions in an
/// M-RoPE bundle is intentional: text tokens use the same logical coordinate
/// on the temporal, height, and width axes. The resulting row-major buffer is
/// therefore `[axis0, axis1, axis2]`, with each axis containing the same
/// zero-based sequence. Multimodal callers must instead provide
/// [`PreparedPositions::Mrope3D`] so their axes and signed decode delta remain
/// explicit.
pub(crate) fn canonical_text_positions(
    mode: PreparedPositionMode,
    context_capacity: usize,
) -> Result<Vec<i32>, PreparedInputError> {
    let one_axis: Vec<i32> = (0..context_capacity)
        .map(|position| i32::try_from(position).map_err(|_| PreparedInputError::ShapeOverflow))
        .collect::<Result<_, _>>()?;
    match mode {
        PreparedPositionMode::OneD => Ok(one_axis),
        PreparedPositionMode::Mrope3D => {
            let capacity = context_capacity
                .checked_mul(3)
                .ok_or(PreparedInputError::ShapeOverflow)?;
            let mut values = Vec::with_capacity(capacity);
            for _ in 0..3 {
                values.extend_from_slice(&one_axis);
            }
            Ok(values)
        }
    }
}

pub(crate) fn mrope_decode_coordinate(
    cache_len: i32,
    rope_delta: i32,
) -> Result<i32, MropeCoordinateError> {
    let coordinate = cache_len
        .checked_add(rope_delta)
        .ok_or(MropeCoordinateError::Overflow {
            cache_len,
            rope_delta,
        })?;
    if coordinate < 0 {
        return Err(MropeCoordinateError::Negative {
            cache_len,
            rope_delta,
        });
    }
    Ok(coordinate)
}

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
        Self::prepare_for_mode(
            value,
            hidden_size,
            context_capacity,
            PreparedPositionMode::OneD,
        )
    }

    pub(crate) fn prepare_for_mode(
        value: &PreparedPrefill,
        hidden_size: usize,
        context_capacity: usize,
        expected_mode: PreparedPositionMode,
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

        let sequential = canonical_text_positions(PreparedPositionMode::OneD, context_capacity)?;
        let validate_1d = |tensor: &OwnedTensor| -> Result<Vec<i32>, PreparedInputError> {
            if tensor.dtype != PreparedTensorDType::Int32 {
                return Err(PreparedInputError::PositionDType(tensor.dtype));
            }
            if tensor.shape != [effective_len] && tensor.shape != [1, effective_len] {
                return Err(PreparedInputError::PositionShape {
                    shape: tensor.shape.clone(),
                    expected: vec![effective_len],
                });
            }
            let explicit = read_i32("positions", tensor)?;
            if explicit
                .iter()
                .enumerate()
                .any(|(index, &position)| position != index as i32)
            {
                return Err(PreparedInputError::UnsupportedPositions);
            }
            Ok(explicit)
        };
        let positions = match (expected_mode, &value.positions) {
            (PreparedPositionMode::OneD, PreparedPositions::Sequential { start, length }) => {
                if *start != 0 || *length != effective_len {
                    return Err(PreparedInputError::UnsupportedPositions);
                }
                PreparedIreePositions::OneD(sequential)
            }
            (PreparedPositionMode::OneD, PreparedPositions::Explicit(tensor)) => {
                let explicit = validate_1d(tensor)?;
                let mut positions = sequential;
                positions[..effective_len].copy_from_slice(&explicit);
                PreparedIreePositions::OneD(positions)
            }
            (PreparedPositionMode::OneD, PreparedPositions::Mrope3D { .. }) => {
                return Err(PreparedInputError::PositionModeMismatch {
                    expected: PreparedPositionMode::OneD,
                    actual: PreparedPositionMode::Mrope3D,
                });
            }
            (PreparedPositionMode::Mrope3D, PreparedPositions::Sequential { start, length }) => {
                if *start != 0 || *length != effective_len {
                    return Err(PreparedInputError::UnsupportedPositions);
                }
                PreparedIreePositions::Mrope3D {
                    values: canonical_text_positions(
                        PreparedPositionMode::Mrope3D,
                        context_capacity,
                    )?,
                    rope_delta: 0,
                }
            }
            (PreparedPositionMode::Mrope3D, PreparedPositions::Explicit(tensor)) => {
                let explicit = validate_1d(tensor)?;
                let mut values =
                    canonical_text_positions(PreparedPositionMode::Mrope3D, context_capacity)?;
                for axis in values.chunks_exact_mut(context_capacity) {
                    axis[..effective_len].copy_from_slice(&explicit);
                }
                PreparedIreePositions::Mrope3D {
                    values,
                    rope_delta: 0,
                }
            }
            (PreparedPositionMode::Mrope3D, PreparedPositions::Mrope3D { tensor, rope_delta }) => {
                if tensor.dtype != PreparedTensorDType::Int32 {
                    return Err(PreparedInputError::PositionDType(tensor.dtype));
                }
                if tensor.shape != [3, effective_len] {
                    return Err(PreparedInputError::PositionShape {
                        shape: tensor.shape.clone(),
                        expected: vec![3, effective_len],
                    });
                }
                let explicit = read_i32("positions", tensor)?;
                if let Some((flat_index, &position)) = explicit
                    .iter()
                    .enumerate()
                    .find(|(_, position)| **position < 0)
                {
                    return Err(PreparedInputError::NegativePosition {
                        axis: flat_index / effective_len,
                        index: flat_index % effective_len,
                        value: position,
                    });
                }
                let mut values = Vec::with_capacity(3 * context_capacity);
                for axis in 0..3 {
                    let mut padded = sequential.clone();
                    let start = axis * effective_len;
                    padded[..effective_len]
                        .copy_from_slice(&explicit[start..start + effective_len]);
                    values.extend(padded);
                }
                PreparedIreePositions::Mrope3D {
                    values,
                    rope_delta: *rope_delta,
                }
            }
            _ => return Err(PreparedInputError::UnsupportedPositions),
        };

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

    fn mrope_fixture(axes: [&[i32]; 3], rope_delta: i32) -> PreparedPrefill {
        let sequence_len = axes[0].len();
        assert!(axes.iter().all(|axis| axis.len() == sequence_len));
        let coordinates = axes
            .into_iter()
            .flat_map(|axis| axis.iter().copied())
            .flat_map(i32::to_le_bytes)
            .collect();
        PreparedPrefill::new(
            vec![1; sequence_len],
            tensor_f32(&[1, sequence_len, 2], vec![0.25; sequence_len * 2]),
            PreparedPositions::Mrope3D {
                tensor: OwnedTensor::new(
                    coordinates,
                    PreparedTensorDType::Int32,
                    vec![3, sequence_len],
                )
                .unwrap(),
                rope_delta,
            },
            PreparedAttentionBias {
                tensor: tensor_f32(&[1, 1, 1, sequence_len], vec![0.0; sequence_len]),
                causal: true,
            },
            Vec::new(),
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
        assert!(matches!(
            prepared.positions,
            PreparedIreePositions::OneD(ref values) if values == &[0, 1, 2, 3, 4]
        ));
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

    #[test]
    fn canonicalizes_mrope_axes_and_preserves_signed_delta() {
        let mut value = fixture();
        value.positions = PreparedPositions::Mrope3D {
            tensor: OwnedTensor::new(
                [0i32, 1, 2, 0, 4, 5, 0, 6, 7]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
                PreparedTensorDType::Int32,
                vec![3, 3],
            )
            .unwrap(),
            rope_delta: -4,
        };

        let prepared =
            PreparedIreePrefill::prepare_for_mode(&value, 2, 5, PreparedPositionMode::Mrope3D)
                .unwrap();
        assert_eq!(prepared.positions.rope_delta(), -4);
        assert_eq!(
            prepared.positions.values(),
            &[0, 1, 2, 3, 4, 0, 4, 5, 3, 4, 0, 6, 7, 3, 4]
        );
    }

    #[test]
    fn intentionally_canonicalizes_text_only_positions_for_mrope_bundles() {
        let expected = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
        assert_eq!(
            canonical_text_positions(PreparedPositionMode::Mrope3D, 5).unwrap(),
            expected
        );

        let sequential =
            PreparedIreePrefill::prepare_for_mode(&fixture(), 2, 5, PreparedPositionMode::Mrope3D)
                .expect("text-only sequential positions in an M-RoPE bundle");
        assert_eq!(sequential.positions.mode(), PreparedPositionMode::Mrope3D);
        assert_eq!(sequential.positions.values(), expected);
        assert_eq!(sequential.positions.rope_delta(), 0);

        let mut explicit_value = fixture();
        explicit_value.positions = PreparedPositions::Explicit(
            OwnedTensor::new(
                [0i32, 1, 2]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
                PreparedTensorDType::Int32,
                vec![1, 3],
            )
            .unwrap(),
        );
        let explicit = PreparedIreePrefill::prepare_for_mode(
            &explicit_value,
            2,
            5,
            PreparedPositionMode::Mrope3D,
        )
        .expect("text-only explicit positions in an M-RoPE bundle");
        assert_eq!(explicit.positions.values(), expected);
        assert_eq!(explicit.positions.rope_delta(), 0);
    }

    #[test]
    fn rejects_mrope_payload_for_one_dimensional_bundle() {
        let mut value = fixture();
        value.positions = PreparedPositions::Mrope3D {
            tensor: OwnedTensor::new(vec![0; 3 * 3 * 4], PreparedTensorDType::Int32, vec![3, 3])
                .unwrap(),
            rope_delta: 0,
        };
        assert!(matches!(
            PreparedIreePrefill::prepare(&value, 2, 5),
            Err(PreparedInputError::PositionModeMismatch { .. })
        ));
    }

    #[test]
    fn ports_qwen_text_image_multi_image_video_and_padding_coordinates() {
        // These row-major fixtures are the direct outputs of the existing MLX
        // Qwen `compute_rope_index` walk for merge=1. Padding fixtures pin the
        // explicit prepared-input convention supported by this backend seam.
        let fixtures: Vec<(&str, [&[i32]; 3], i32)> = vec![
            (
                "text-only",
                [&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]],
                0,
            ),
            (
                "one-image-1x2x2",
                [
                    &[0, 1, 2, 2, 2, 2, 4],
                    &[0, 1, 2, 2, 3, 3, 4],
                    &[0, 1, 2, 3, 2, 3, 4],
                ],
                -2,
            ),
            (
                "multiple-images",
                [
                    &[0, 1, 1, 3, 4, 4, 6],
                    &[0, 1, 1, 3, 4, 5, 6],
                    &[0, 1, 2, 3, 4, 4, 6],
                ],
                0,
            ),
            (
                "video-2x1x2",
                [
                    &[0, 1, 1, 2, 2, 3],
                    &[0, 1, 1, 1, 1, 3],
                    &[0, 1, 2, 1, 2, 3],
                ],
                -2,
            ),
            (
                "left-padding",
                [&[0, 0, 0, 1, 2], &[0, 0, 0, 1, 2], &[0, 0, 0, 1, 2]],
                -2,
            ),
            (
                "right-padding",
                [&[0, 1, 2, 0, 0], &[0, 1, 2, 0, 0], &[0, 1, 2, 0, 0]],
                -2,
            ),
        ];

        for (name, axes, delta) in fixtures {
            let sequence_len = axes[0].len();
            let prepared = PreparedIreePrefill::prepare_for_mode(
                &mrope_fixture(axes, delta),
                2,
                sequence_len + 2,
                PreparedPositionMode::Mrope3D,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(prepared.positions.rope_delta(), delta, "{name}");
            for (axis, expected) in axes.iter().enumerate() {
                let start = axis * (sequence_len + 2);
                assert_eq!(
                    &prepared.positions.values()[start..start + sequence_len],
                    *expected,
                    "{name} axis {axis}"
                );
            }
        }

        let long = (0..2048).collect::<Vec<i32>>();
        let prepared = PreparedIreePrefill::prepare_for_mode(
            &mrope_fixture([&long, &long, &long], 17),
            2,
            2050,
            PreparedPositionMode::Mrope3D,
        )
        .expect("long M-RoPE fixture");
        assert_eq!(&prepared.positions.values()[..2048], long);
        assert_eq!(prepared.positions.rope_delta(), 17);
    }

    #[test]
    fn decode_coordinate_supports_signed_deltas_and_typed_failures() {
        assert_eq!(mrope_decode_coordinate(12, -5), Ok(7));
        assert_eq!(mrope_decode_coordinate(12, 5), Ok(17));
        assert_eq!(
            mrope_decode_coordinate(2, -3),
            Err(MropeCoordinateError::Negative {
                cache_len: 2,
                rope_delta: -3,
            })
        );
        assert_eq!(
            mrope_decode_coordinate(i32::MAX, 1),
            Err(MropeCoordinateError::Overflow {
                cache_len: i32::MAX,
                rope_delta: 1,
            })
        );
    }
}
