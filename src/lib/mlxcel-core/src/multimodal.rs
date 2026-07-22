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

//! Backend-neutral multimodal prefill values.
//!
//! These types form the ownership boundary between a host-side media
//! preprocessor and an inference engine. They deliberately contain only owned
//! Rust values: no MLX arrays, safetensor views, model handles, or server
//! request state. A prepared value can therefore be moved to a compiler-backend
//! worker without tying that worker to the producer's tensor runtime.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Element types accepted at the prepared-prefill boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PreparedTensorDType {
    Float16,
    BFloat16,
    Float32,
    Int32,
}

impl PreparedTensorDType {
    /// Number of bytes occupied by one element.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Float16 | Self::BFloat16 => 2,
            Self::Float32 | Self::Int32 => 4,
        }
    }

    /// Whether this dtype is valid for embeddings and additive attention bias.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float16 | Self::BFloat16 | Self::Float32)
    }
}

/// A contiguous, row-major tensor whose storage is owned by this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnedTensor {
    pub bytes: Vec<u8>,
    pub dtype: PreparedTensorDType,
    pub shape: Vec<usize>,
}

impl OwnedTensor {
    /// Construct an owned tensor after checking shape arithmetic and byte size.
    ///
    /// # Errors
    ///
    /// Returns [`PreparedPrefillError::ShapeOverflow`] when the shape cannot be
    /// represented and [`PreparedPrefillError::ByteCountMismatch`] when the
    /// storage length does not match `shape * dtype.size_bytes()`.
    pub fn new(
        bytes: Vec<u8>,
        dtype: PreparedTensorDType,
        shape: Vec<usize>,
    ) -> Result<Self, PreparedPrefillError> {
        let expected = checked_byte_count(&shape, dtype)?;
        if bytes.len() != expected {
            return Err(PreparedPrefillError::ByteCountMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            bytes,
            dtype,
            shape,
        })
    }

    /// Number of logical elements in the tensor.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

/// Position encoding associated with a prepared prefill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PreparedPositions {
    /// Dense positions `start..start + length`, the LLaVA reference layout.
    Sequential { start: i32, length: usize },
    /// Explicit position rows for model families that do not use a scalar
    /// sequence position (for example multimodal RoPE families).
    Explicit(OwnedTensor),
}

impl PreparedPositions {
    #[must_use]
    fn sequence_len(&self) -> Option<usize> {
        match self {
            Self::Sequential { length, .. } => Some(*length),
            Self::Explicit(tensor) => tensor.shape.last().copied(),
        }
    }
}

/// Additive attention bias and its causal interpretation.
///
/// The tensor may use broadcast dimensions. The LLaVA producer emits a compact
/// `[1, 1, 1, sequence]` all-zero padding bias and sets `causal = true`, which
/// is equivalent to the existing MLX path's implicit causal mask without an
/// unnecessary quadratic host allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedAttentionBias {
    pub tensor: OwnedTensor,
    pub causal: bool,
}

/// Metrics and validation facts for one prepared modality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedModality {
    /// Stable model-family identifier, such as `"llava"`.
    pub family: String,
    /// Number of decoded media items consumed.
    pub item_count: usize,
    /// Number of logical sequence positions occupied by the modality.
    pub token_count: usize,
}

/// Owned prefill payload crossing from host preprocessing to an engine worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedPrefill {
    /// Logical ids after family-specific placeholder expansion.
    pub token_ids: Vec<i32>,
    /// `[batch, sequence, hidden]` merged embeddings.
    pub embeddings: OwnedTensor,
    pub positions: PreparedPositions,
    pub attention_bias: PreparedAttentionBias,
    pub sequence_len: usize,
    pub modalities: Vec<PreparedModality>,
}

impl PreparedPrefill {
    /// Construct and validate a backend-neutral prepared prefill.
    ///
    /// # Errors
    ///
    /// Returns a typed error when token, embedding, position, or attention
    /// shapes disagree. Producers should call this only after their
    /// family-specific hidden-size and media-cardinality checks.
    pub fn new(
        token_ids: Vec<i32>,
        embeddings: OwnedTensor,
        positions: PreparedPositions,
        attention_bias: PreparedAttentionBias,
        modalities: Vec<PreparedModality>,
    ) -> Result<Self, PreparedPrefillError> {
        let sequence_len = token_ids.len();
        if embeddings.shape.len() != 3
            || embeddings.shape[0] != 1
            || embeddings.shape[1] != sequence_len
        {
            return Err(PreparedPrefillError::EmbeddingShape {
                shape: embeddings.shape.clone(),
                sequence_len,
            });
        }
        if !embeddings.dtype.is_float() {
            return Err(PreparedPrefillError::EmbeddingDType(embeddings.dtype));
        }
        let position_len = positions.sequence_len();
        if position_len != Some(sequence_len) {
            return Err(PreparedPrefillError::PositionLength {
                expected: sequence_len,
                actual: position_len,
            });
        }
        if let PreparedPositions::Explicit(tensor) = &positions
            && tensor.dtype != PreparedTensorDType::Int32
        {
            return Err(PreparedPrefillError::PositionDType(tensor.dtype));
        }
        if !attention_bias.tensor.dtype.is_float() {
            return Err(PreparedPrefillError::AttentionBiasDType(
                attention_bias.tensor.dtype,
            ));
        }
        let bias_sequence = attention_bias.tensor.shape.last().copied();
        if bias_sequence != Some(sequence_len) {
            return Err(PreparedPrefillError::AttentionBiasShape {
                shape: attention_bias.tensor.shape.clone(),
                sequence_len,
            });
        }
        let modality_tokens = modalities.iter().try_fold(0usize, |total, modality| {
            total
                .checked_add(modality.token_count)
                .ok_or(PreparedPrefillError::ShapeOverflow)
        })?;
        if modality_tokens > sequence_len {
            return Err(PreparedPrefillError::ModalityTokenCount {
                token_count: modality_tokens,
                sequence_len,
            });
        }

        Ok(Self {
            token_ids,
            embeddings,
            positions,
            attention_bias,
            sequence_len,
            modalities,
        })
    }
}

/// Structural errors at the owned prefill boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PreparedPrefillError {
    #[error("prepared tensor shape or byte-size calculation overflowed")]
    ShapeOverflow,
    #[error("prepared tensor byte count mismatch: expected {expected}, got {actual}")]
    ByteCountMismatch { expected: usize, actual: usize },
    #[error("prepared embedding shape {shape:?} must be [1, {sequence_len}, hidden]")]
    EmbeddingShape {
        shape: Vec<usize>,
        sequence_len: usize,
    },
    #[error("prepared embedding dtype {0:?} is not floating point")]
    EmbeddingDType(PreparedTensorDType),
    #[error("prepared position length mismatch: expected {expected}, got {actual:?}")]
    PositionLength {
        expected: usize,
        actual: Option<usize>,
    },
    #[error("prepared explicit-position dtype {0:?} must be Int32")]
    PositionDType(PreparedTensorDType),
    #[error("prepared attention-bias dtype {0:?} is not floating point")]
    AttentionBiasDType(PreparedTensorDType),
    #[error(
        "prepared attention-bias shape {shape:?} does not end in sequence length {sequence_len}"
    )]
    AttentionBiasShape {
        shape: Vec<usize>,
        sequence_len: usize,
    },
    #[error(
        "prepared modality metadata accounts for {token_count} tokens, exceeding sequence length {sequence_len}"
    )]
    ModalityTokenCount {
        token_count: usize,
        sequence_len: usize,
    },
}

fn checked_byte_count(
    shape: &[usize],
    dtype: PreparedTensorDType,
) -> Result<usize, PreparedPrefillError> {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .and_then(|count| count.checked_mul(dtype.size_bytes()))
        .ok_or(PreparedPrefillError::ShapeOverflow)
}

#[cfg(test)]
#[path = "multimodal_tests.rs"]
mod tests;
