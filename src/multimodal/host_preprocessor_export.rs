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

//! Validation and owned-export helpers for the host multimodal producer.

use mlxcel_core::MlxArray;
use mlxcel_core::session::{
    OwnedTensor, PreparedAttentionBias, PreparedModality, PreparedPositions, PreparedPrefill,
    PreparedTensorDType,
};

use crate::vision::merge::InputEmbeddings;

use super::HostPreprocessorError;

pub(super) fn validate_sequence_capacity(
    sequence_len: usize,
    maximum: usize,
) -> Result<(), HostPreprocessorError> {
    if sequence_len == 0 {
        return Err(HostPreprocessorError::InvalidConfig(
            "tokenized prompt must not be empty".to_string(),
        ));
    }
    if sequence_len > maximum {
        return Err(HostPreprocessorError::SequenceCapacity {
            actual: sequence_len,
            maximum,
        });
    }
    Ok(())
}

pub(super) fn validate_processor_shape(
    shape: &[i32],
    image_count: usize,
    image_size: usize,
) -> Result<(), HostPreprocessorError> {
    let expected = [
        usize_to_i32(image_count, "image count")?,
        3,
        usize_to_i32(image_size, "image size")?,
        usize_to_i32(image_size, "image size")?,
    ];
    if shape != expected {
        return Err(HostPreprocessorError::ProcessorShape {
            actual: shape.to_vec(),
            image_count,
            image_size,
        });
    }
    Ok(())
}

pub(super) fn validate_embedding_shape(
    shape: &[i32],
    sequence_len: usize,
    hidden_size: usize,
    source_name: &'static str,
) -> Result<(), HostPreprocessorError> {
    let expected = [
        1,
        usize_to_i32(sequence_len, "sequence length")?,
        usize_to_i32(hidden_size, "hidden size")?,
    ];
    if shape != expected {
        return Err(HostPreprocessorError::EmbeddingShape {
            source_name,
            actual: shape.to_vec(),
            sequence_len,
            hidden_size,
        });
    }
    Ok(())
}

pub(super) fn validate_projected_shape(
    shape: &[i32],
    image_count: usize,
    tokens_per_image: usize,
    hidden_size: usize,
) -> Result<(), HostPreprocessorError> {
    let expected = [
        usize_to_i32(image_count, "image count")?,
        usize_to_i32(tokens_per_image, "tokens per image")?,
        usize_to_i32(hidden_size, "hidden size")?,
    ];
    if shape != expected {
        return Err(HostPreprocessorError::ProjectedShape {
            actual: shape.to_vec(),
            image_count,
            tokens_per_image,
            hidden_size,
        });
    }
    Ok(())
}

pub(super) fn export_llava_prefill(
    logical_tokens: Vec<i32>,
    merged: InputEmbeddings,
    image_token_id: i32,
    image_count: usize,
    tokens_per_image: usize,
    hidden_size: usize,
) -> Result<PreparedPrefill, HostPreprocessorError> {
    let actual_image_tokens = logical_tokens
        .iter()
        .filter(|&&token| token == image_token_id)
        .count();
    let expected_image_tokens = image_count
        .checked_mul(tokens_per_image)
        .ok_or(HostPreprocessorError::ShapeOverflow)?;
    if actual_image_tokens != expected_image_tokens {
        return Err(HostPreprocessorError::ExpandedLength {
            actual: actual_image_tokens,
            expected: expected_image_tokens,
        });
    }
    validate_embedding_shape(
        &mlxcel_core::array_shape(&merged.inputs_embeds),
        logical_tokens.len(),
        hidden_size,
        "merged embedding",
    )?;
    if merged.attention_mask_4d.is_some() {
        return Err(HostPreprocessorError::InvalidConfig(
            "LLaVA host preprocessing requires standard causal masking, not a family-specific 4D mask"
                .to_string(),
        ));
    }

    let embeddings = export_mlx_tensor(&merged.inputs_embeds, "merged embedding")?;
    build_prepared_prefill(
        logical_tokens,
        embeddings,
        image_count,
        expected_image_tokens,
        "llava",
    )
}

pub(super) fn export_mlx_tensor(
    array: &MlxArray,
    label: &'static str,
) -> Result<OwnedTensor, HostPreprocessorError> {
    let shape = mlxcel_core::array_shape(array)
        .into_iter()
        .map(|dim| {
            usize::try_from(dim).map_err(|_| HostPreprocessorError::TensorExport {
                tensor: label,
                message: format!("negative dimension {dim}"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dtype = prepared_dtype(mlxcel_core::array_dtype(array))?;

    // This one fallible FFI operation makes the lazy result contiguous,
    // evaluates it, and copies its bytes. No earlier raw view is retained.
    let bytes = mlxcel_core::try_array_to_raw_bytes(array).map_err(|error| {
        HostPreprocessorError::TensorExport {
            tensor: label,
            message: error.to_string(),
        }
    })?;
    OwnedTensor::new(bytes, dtype, shape).map_err(HostPreprocessorError::from)
}

fn prepared_dtype(dtype: i32) -> Result<PreparedTensorDType, HostPreprocessorError> {
    match dtype {
        mlxcel_core::dtype::FLOAT16 => Ok(PreparedTensorDType::Float16),
        mlxcel_core::dtype::BFLOAT16 => Ok(PreparedTensorDType::BFloat16),
        mlxcel_core::dtype::FLOAT32 => Ok(PreparedTensorDType::Float32),
        other => Err(HostPreprocessorError::UnsupportedDType(other)),
    }
}

pub(super) fn build_prepared_prefill(
    logical_tokens: Vec<i32>,
    embeddings: OwnedTensor,
    image_count: usize,
    image_token_count: usize,
    family: &str,
) -> Result<PreparedPrefill, HostPreprocessorError> {
    let sequence_len = logical_tokens.len();
    let bias_bytes = vec![
        0u8;
        sequence_len
            .checked_mul(PreparedTensorDType::Float32.size_bytes())
            .ok_or(HostPreprocessorError::ShapeOverflow)?
    ];
    let attention_bias = PreparedAttentionBias {
        tensor: OwnedTensor::new(
            bias_bytes,
            PreparedTensorDType::Float32,
            vec![1, 1, 1, sequence_len],
        )?,
        causal: true,
    };
    let modalities = if image_count == 0 {
        Vec::new()
    } else {
        vec![PreparedModality {
            family: family.to_string(),
            item_count: image_count,
            token_count: image_token_count,
        }]
    };
    PreparedPrefill::new(
        logical_tokens,
        embeddings,
        PreparedPositions::Sequential {
            start: 0,
            length: sequence_len,
        },
        attention_bias,
        modalities,
    )
    .map_err(HostPreprocessorError::from)
}

pub(super) fn usize_to_i32(
    value: usize,
    label: &'static str,
) -> Result<i32, HostPreprocessorError> {
    i32::try_from(value).map_err(|_| HostPreprocessorError::DimensionOverflow { label, value })
}
