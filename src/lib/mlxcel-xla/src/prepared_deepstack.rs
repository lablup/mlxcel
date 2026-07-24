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

//! Compact owned DeepStack side inputs for embeddings prefill.

use std::fmt;

use mlxcel_core::session::{OwnedTensor, PreparedPrefill, PreparedTensorDType};

#[cfg(any(feature = "iree", test))]
use crate::emitter::DeepStackConfig;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeepStackInputError {
    DType {
        tensor: &'static str,
        actual: PreparedTensorDType,
        expected: PreparedTensorDType,
    },
    Shape {
        tensor: &'static str,
        actual: Vec<usize>,
        expected: String,
    },
    ByteCount {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    ShapeOverflow,
    EmptyLayerSet,
    InvalidPosition {
        index: usize,
        position: i32,
        sequence_len: usize,
    },
    PositionsNotSortedUnique,
    ModalityPositionCount {
        positions: usize,
        modality_tokens: usize,
    },
    InvalidLayerIndex(i32),
    LayerContract {
        actual: Vec<i32>,
        expected: Vec<usize>,
    },
    StaticMaxOverflow {
        actual: usize,
        maximum: usize,
    },
    HiddenSize {
        actual: usize,
        expected: usize,
    },
    NonFiniteFeature,
}

impl fmt::Display for DeepStackInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DType {
                tensor,
                actual,
                expected,
            } => write!(
                f,
                "DeepStack {tensor} dtype {actual:?} must be {expected:?}"
            ),
            Self::Shape {
                tensor,
                actual,
                expected,
            } => write!(f, "DeepStack {tensor} shape {actual:?} must be {expected}"),
            Self::ByteCount {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "DeepStack {tensor} byte count mismatch: expected {expected}, got {actual}"
            ),
            Self::ShapeOverflow => f.write_str("DeepStack tensor shape arithmetic overflowed"),
            Self::EmptyLayerSet => f.write_str("DeepStack requires at least one feature layer"),
            Self::InvalidPosition {
                index,
                position,
                sequence_len,
            } => write!(
                f,
                "DeepStack visual position {index}={position} is outside [0,{sequence_len})"
            ),
            Self::PositionsNotSortedUnique => {
                f.write_str("DeepStack visual positions must be strictly sorted and unique")
            }
            Self::ModalityPositionCount {
                positions,
                modality_tokens,
            } => write!(
                f,
                "DeepStack has {positions} visual positions but placeholder expansion metadata accounts for {modality_tokens} modality tokens"
            ),
            Self::InvalidLayerIndex(layer) => {
                write!(f, "DeepStack layer index {layer} must be non-negative")
            }
            Self::LayerContract { actual, expected } => write!(
                f,
                "DeepStack layer indices {actual:?} do not match compiled target layers {expected:?}"
            ),
            Self::StaticMaxOverflow { actual, maximum } => write!(
                f,
                "DeepStack visual position count {actual} exceeds static maximum {maximum}"
            ),
            Self::HiddenSize { actual, expected } => write!(
                f,
                "DeepStack feature hidden size {actual} must match model hidden size {expected}"
            ),
            Self::NonFiniteFeature => {
                f.write_str("DeepStack layer features contain a non-finite value")
            }
        }
    }
}

impl std::error::Error for DeepStackInputError {}

fn checked_byte_count(
    tensor: &'static str,
    value: &OwnedTensor,
) -> Result<(), DeepStackInputError> {
    let expected = value
        .shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .and_then(|count| count.checked_mul(value.dtype.size_bytes()))
        .ok_or(DeepStackInputError::ShapeOverflow)?;
    if value.bytes.len() != expected {
        return Err(DeepStackInputError::ByteCount {
            tensor,
            expected,
            actual: value.bytes.len(),
        });
    }
    Ok(())
}

fn read_i32(tensor: &'static str, value: &OwnedTensor) -> Result<Vec<i32>, DeepStackInputError> {
    checked_byte_count(tensor, value)?;
    Ok(value
        .bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn read_f32(tensor: &'static str, value: &OwnedTensor) -> Result<Vec<f32>, DeepStackInputError> {
    checked_byte_count(tensor, value)?;
    Ok(value
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

/// Compact side tensors with shapes `[Nv]`, `[K, Nv, hidden]`, and `[K]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepStackFeatures {
    visual_positions: OwnedTensor,
    layer_features: OwnedTensor,
    layer_indices: OwnedTensor,
}

impl DeepStackFeatures {
    pub fn new(
        visual_positions: OwnedTensor,
        layer_features: OwnedTensor,
        layer_indices: OwnedTensor,
    ) -> Result<Self, DeepStackInputError> {
        if visual_positions.dtype != PreparedTensorDType::Int32 {
            return Err(DeepStackInputError::DType {
                tensor: "visual_positions",
                actual: visual_positions.dtype,
                expected: PreparedTensorDType::Int32,
            });
        }
        if layer_indices.dtype != PreparedTensorDType::Int32 {
            return Err(DeepStackInputError::DType {
                tensor: "layer_indices",
                actual: layer_indices.dtype,
                expected: PreparedTensorDType::Int32,
            });
        }
        if layer_features.dtype != PreparedTensorDType::Float32 {
            return Err(DeepStackInputError::DType {
                tensor: "layer_features",
                actual: layer_features.dtype,
                expected: PreparedTensorDType::Float32,
            });
        }
        if visual_positions.shape.len() != 1 {
            return Err(DeepStackInputError::Shape {
                tensor: "visual_positions",
                actual: visual_positions.shape.clone(),
                expected: "[Nv]".to_string(),
            });
        }
        if layer_indices.shape.len() != 1 {
            return Err(DeepStackInputError::Shape {
                tensor: "layer_indices",
                actual: layer_indices.shape.clone(),
                expected: "[K]".to_string(),
            });
        }
        if layer_indices.shape[0] == 0 {
            return Err(DeepStackInputError::EmptyLayerSet);
        }
        let layer_count = layer_indices.shape[0];
        let visual_count = visual_positions.shape[0];
        if layer_features.shape.len() != 3
            || layer_features.shape[0] != layer_count
            || layer_features.shape[1] != visual_count
            || layer_features.shape[2] == 0
        {
            return Err(DeepStackInputError::Shape {
                tensor: "layer_features",
                actual: layer_features.shape.clone(),
                expected: format!("[{layer_count}, {visual_count}, hidden>0]"),
            });
        }
        checked_byte_count("visual_positions", &visual_positions)?;
        checked_byte_count("layer_features", &layer_features)?;
        checked_byte_count("layer_indices", &layer_indices)?;
        Ok(Self {
            visual_positions,
            layer_features,
            layer_indices,
        })
    }

    #[must_use]
    pub fn visual_count(&self) -> usize {
        self.visual_positions.shape[0]
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layer_indices.shape[0]
    }

    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.layer_features.shape[2]
    }
}

/// Prepared embeddings coupled to compact, prefill-only DeepStack additions.
#[derive(Clone, Debug)]
pub struct DeepStackPreparedPrefill {
    prepared: PreparedPrefill,
    deepstack: DeepStackFeatures,
}

impl DeepStackPreparedPrefill {
    pub fn new(
        prepared: PreparedPrefill,
        deepstack: DeepStackFeatures,
    ) -> Result<Self, DeepStackInputError> {
        let positions = read_i32("visual_positions", &deepstack.visual_positions)?;
        if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(DeepStackInputError::PositionsNotSortedUnique);
        }
        if let Some((index, &position)) = positions
            .iter()
            .enumerate()
            .find(|(_, position)| **position < 0 || **position as usize >= prepared.sequence_len)
        {
            return Err(DeepStackInputError::InvalidPosition {
                index,
                position,
                sequence_len: prepared.sequence_len,
            });
        }
        let modality_tokens = prepared
            .modalities
            .iter()
            .try_fold(0usize, |count, modality| {
                count.checked_add(modality.token_count)
            })
            .ok_or(DeepStackInputError::ShapeOverflow)?;
        if modality_tokens != positions.len() {
            return Err(DeepStackInputError::ModalityPositionCount {
                positions: positions.len(),
                modality_tokens,
            });
        }
        let layers = read_i32("layer_indices", &deepstack.layer_indices)?;
        if let Some(&layer) = layers.iter().find(|&&layer| layer < 0) {
            return Err(DeepStackInputError::InvalidLayerIndex(layer));
        }
        if read_f32("layer_features", &deepstack.layer_features)?
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(DeepStackInputError::NonFiniteFeature);
        }
        Ok(Self {
            prepared,
            deepstack,
        })
    }

    #[must_use]
    pub fn prepared(&self) -> &PreparedPrefill {
        &self.prepared
    }

    #[must_use]
    pub fn deepstack(&self) -> &DeepStackFeatures {
        &self.deepstack
    }

    #[cfg(any(feature = "iree", test))]
    pub(crate) fn into_parts(self) -> (PreparedPrefill, DeepStackFeatures) {
        (self.prepared, self.deepstack)
    }
}

#[cfg(any(feature = "iree", test))]
#[derive(Debug)]
pub(crate) struct PreparedDeepStack {
    pub(crate) visual_positions: Vec<i32>,
    pub(crate) layer_features: Vec<f32>,
    pub(crate) layer_indices: Vec<i32>,
    pub(crate) actual_layer_count: usize,
    pub(crate) actual_visual_count: usize,
    pub(crate) max_layer_count: usize,
    pub(crate) max_visual_count: usize,
    pub(crate) hidden_size: usize,
}

#[cfg(any(feature = "iree", test))]
impl PreparedDeepStack {
    pub(crate) fn prepare(
        value: &DeepStackFeatures,
        schema: &DeepStackConfig,
        hidden_size: usize,
    ) -> Result<Self, DeepStackInputError> {
        let actual_visual_count = value.visual_count();
        if actual_visual_count > schema.max_visual_positions {
            return Err(DeepStackInputError::StaticMaxOverflow {
                actual: actual_visual_count,
                maximum: schema.max_visual_positions,
            });
        }
        if value.hidden_size() != hidden_size {
            return Err(DeepStackInputError::HiddenSize {
                actual: value.hidden_size(),
                expected: hidden_size,
            });
        }
        let actual_layers = read_i32("layer_indices", &value.layer_indices)?;
        if actual_layers
            != schema
                .target_layer_indices
                .iter()
                .map(|&layer| layer as i32)
                .collect::<Vec<_>>()
        {
            return Err(DeepStackInputError::LayerContract {
                actual: actual_layers,
                expected: schema.target_layer_indices.clone(),
            });
        }
        let compact_positions = read_i32("visual_positions", &value.visual_positions)?;
        let compact_features = read_f32("layer_features", &value.layer_features)?;
        let max_layer_count = schema.target_layer_indices.len();
        let max_visual_count = schema.max_visual_positions;
        let static_count = max_layer_count
            .checked_mul(max_visual_count)
            .and_then(|count| count.checked_mul(hidden_size))
            .ok_or(DeepStackInputError::ShapeOverflow)?;
        let mut layer_features = vec![0.0; static_count];
        for layer in 0..max_layer_count {
            for visual in 0..actual_visual_count {
                let source = (layer * actual_visual_count + visual) * hidden_size;
                let destination = (layer * max_visual_count + visual) * hidden_size;
                layer_features[destination..destination + hidden_size]
                    .copy_from_slice(&compact_features[source..source + hidden_size]);
            }
        }
        let mut visual_positions = vec![-1; max_visual_count];
        visual_positions[..actual_visual_count].copy_from_slice(&compact_positions);
        let layer_indices = schema
            .target_layer_indices
            .iter()
            .map(|&layer| layer as i32)
            .collect();
        Ok(Self {
            visual_positions,
            layer_features,
            layer_indices,
            actual_layer_count: max_layer_count,
            actual_visual_count,
            max_layer_count,
            max_visual_count,
            hidden_size,
        })
    }
}

#[cfg(test)]
#[path = "prepared_deepstack_tests.rs"]
mod tests;
