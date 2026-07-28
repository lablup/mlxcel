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

// Reachable in production only from the `xla-iree` Qwen2-VL prefill path, and
// exercised by the module tests regardless, so the gate carries `test` too.
// Without it a default-feature `cargo clippy --lib -- -D warnings` fails the
// build on `function is never used`.
#[cfg(any(feature = "xla-iree", test))]
pub(super) fn export_qwen2_vl_prefill(
    logical_tokens: Vec<i32>,
    merged: InputEmbeddings,
    grids: &[(i32, i32, i32)],
    image_token_id: i32,
    video_token_id: i32,
    spatial_merge_size: usize,
    hidden_size: usize,
) -> Result<PreparedPrefill, HostPreprocessorError> {
    if spatial_merge_size == 0 {
        return Err(HostPreprocessorError::InvalidConfig(
            "Qwen2-VL spatial_merge_size must be positive".to_string(),
        ));
    }
    if logical_tokens.contains(&video_token_id) {
        return Err(HostPreprocessorError::InvalidConfig(
            "Qwen2-VL XLA image preprocessing does not support video placeholders".to_string(),
        ));
    }
    validate_embedding_shape(
        &mlxcel_core::array_shape(&merged.inputs_embeds),
        logical_tokens.len(),
        hidden_size,
        "Qwen2-VL merged embedding",
    )?;
    if merged.attention_mask_4d.is_some() {
        return Err(HostPreprocessorError::InvalidConfig(
            "Qwen2-VL prepared prefill requires the standard causal mask".to_string(),
        ));
    }
    let merge = i32::try_from(spatial_merge_size).map_err(|_| {
        HostPreprocessorError::InvalidConfig(
            "Qwen2-VL spatial_merge_size does not fit i32".to_string(),
        )
    })?;
    let expected_per_image = grids
        .iter()
        .enumerate()
        .map(|(index, &(temporal, height, width))| {
            if temporal != 1
                || height <= 0
                || width <= 0
                || height % merge != 0
                || width % merge != 0
            {
                return Err(HostPreprocessorError::InvalidConfig(format!(
                    "invalid Qwen2-VL image grid {index}: ({temporal},{height},{width})"
                )));
            }
            let count = temporal
                .checked_mul(height / merge)
                .and_then(|value| value.checked_mul(width / merge))
                .ok_or(HostPreprocessorError::ShapeOverflow)?;
            usize::try_from(count).map_err(|_| HostPreprocessorError::ShapeOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_image_tokens = expected_per_image.iter().sum::<usize>();
    let actual_image_tokens = logical_tokens
        .iter()
        .filter(|&&token| token == image_token_id)
        .count();
    if actual_image_tokens != expected_image_tokens {
        return Err(HostPreprocessorError::ExpandedLength {
            actual: actual_image_tokens,
            expected: expected_image_tokens,
        });
    }

    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    let mut current_position = 0i32;
    let mut segment_start = 0usize;
    let mut image_index = 0usize;
    let mut cursor = 0usize;
    while cursor < logical_tokens.len() {
        if logical_tokens[cursor] != image_token_id {
            cursor += 1;
            continue;
        }
        let vision_start = cursor;
        while cursor < logical_tokens.len() && logical_tokens[cursor] == image_token_id {
            cursor += 1;
        }
        if image_index >= grids.len() {
            return Err(HostPreprocessorError::InvalidConfig(
                "Qwen2-VL prompt has more image-token runs than decoded images".to_string(),
            ));
        }
        let text_length = i32::try_from(vision_start - segment_start)
            .map_err(|_| HostPreprocessorError::ShapeOverflow)?;
        for position in current_position
            ..current_position
                .checked_add(text_length)
                .ok_or(HostPreprocessorError::ShapeOverflow)?
        {
            for axis in &mut axes {
                axis.push(position);
            }
        }
        current_position = current_position
            .checked_add(text_length)
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        let run_length = cursor - vision_start;
        if run_length != expected_per_image[image_index] {
            return Err(HostPreprocessorError::InvalidConfig(format!(
                "Qwen2-VL image-token run {image_index} has {run_length} tokens, expected {}",
                expected_per_image[image_index]
            )));
        }
        let (temporal, height, width) = grids[image_index];
        let llm_height = height / merge;
        let llm_width = width / merge;
        for t in 0..temporal {
            for h in 0..llm_height {
                for w in 0..llm_width {
                    axes[0].push(
                        current_position
                            .checked_add(t)
                            .ok_or(HostPreprocessorError::ShapeOverflow)?,
                    );
                    axes[1].push(
                        current_position
                            .checked_add(h)
                            .ok_or(HostPreprocessorError::ShapeOverflow)?,
                    );
                    axes[2].push(
                        current_position
                            .checked_add(w)
                            .ok_or(HostPreprocessorError::ShapeOverflow)?,
                    );
                }
            }
        }
        current_position = current_position
            .checked_add(temporal.max(llm_height).max(llm_width))
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        image_index += 1;
        segment_start = cursor;
    }
    if image_index != grids.len() {
        return Err(HostPreprocessorError::InvalidConfig(format!(
            "Qwen2-VL prompt has {image_index} image-token runs, expected {}",
            grids.len()
        )));
    }
    let trailing = logical_tokens.len() - segment_start;
    for offset in 0..trailing {
        let position = current_position
            .checked_add(i32::try_from(offset).map_err(|_| HostPreprocessorError::ShapeOverflow)?)
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        for axis in &mut axes {
            axis.push(position);
        }
    }
    if axes.iter().any(|axis| axis.len() != logical_tokens.len()) {
        return Err(HostPreprocessorError::InvalidConfig(
            "Qwen2-VL M-RoPE position construction did not cover the logical prompt".to_string(),
        ));
    }
    let maximum = axes
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .ok_or_else(|| {
            HostPreprocessorError::InvalidConfig(
                "Qwen2-VL logical prompt must not be empty".to_string(),
            )
        })?;
    let sequence_len_i32 =
        i32::try_from(logical_tokens.len()).map_err(|_| HostPreprocessorError::ShapeOverflow)?;
    let rope_delta = maximum
        .checked_add(1)
        .and_then(|value| value.checked_sub(sequence_len_i32))
        .ok_or(HostPreprocessorError::ShapeOverflow)?;
    let positions = axes.into_iter().flatten().collect::<Vec<_>>();
    let position_bytes = positions
        .into_iter()
        .flat_map(i32::to_ne_bytes)
        .collect::<Vec<_>>();
    let sequence_len = logical_tokens.len();
    let embeddings = export_mlx_tensor(&merged.inputs_embeds, "Qwen2-VL merged embedding")?;
    let attention_bytes = vec![
        0u8;
        sequence_len
            .checked_mul(PreparedTensorDType::Float32.size_bytes())
            .ok_or(HostPreprocessorError::ShapeOverflow)?
    ];
    PreparedPrefill::new(
        logical_tokens,
        embeddings,
        PreparedPositions::Mrope3D {
            tensor: OwnedTensor::new(
                position_bytes,
                PreparedTensorDType::Int32,
                vec![3, sequence_len],
            )?,
            rope_delta,
        },
        PreparedAttentionBias {
            tensor: OwnedTensor::new(
                attention_bytes,
                PreparedTensorDType::Float32,
                vec![1, 1, 1, sequence_len],
            )?,
            causal: true,
        },
        vec![PreparedModality {
            family: "qwen2_vl".to_string(),
            item_count: grids.len(),
            token_count: expected_image_tokens,
        }],
    )
    .map_err(HostPreprocessorError::from)
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
