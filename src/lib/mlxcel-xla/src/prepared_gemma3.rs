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

//! Canonical Gemma3 VLM merge at the XLA embeddings-prefill boundary.
//!
//! The ordinary Gemma3 token graph gathers embeddings and multiplies them by
//! `sqrt(hidden_size)`. The embeddings entry deliberately does neither, so this
//! producer exports text rows after that scale while keeping projected image
//! rows at their original magnitude. It also owns the reference's bidirectional
//! padding mask instead of substituting a causal or sliding-window mask.

use std::fmt;

use mlxcel_core::session::{
    OwnedTensor, PreparedAttentionBias, PreparedModality, PreparedPositions, PreparedPrefill,
    PreparedPrefillError, PreparedTensorDType,
};

/// Exact masked value emitted by `vision::merge::prepare_inputs_for_multimodal`.
pub const GEMMA3_VLM_MASKED_VALUE: f32 = f32::MIN;

/// Stable identity component for the qualified external-mask behavior.
pub const GEMMA3_VLM_MASK_MODE: &str = "gemma3-vlm-bidirectional-padding-f32-min-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma3VlmPreparedError {
    Empty,
    ZeroHiddenSize,
    Capacity {
        sequence_len: usize,
        context_capacity: usize,
    },
    ShapeOverflow,
    EmbeddingCount {
        expected: usize,
        actual: usize,
    },
    ProjectedShape {
        values: usize,
        hidden_size: usize,
    },
    AttentionMaskCount {
        expected: usize,
        actual: usize,
    },
    InvalidAttentionMask {
        index: usize,
        value: i32,
    },
    PaddingMaskMismatch {
        index: usize,
        token_id: i32,
        mask: i32,
    },
    PlaceholderCount {
        positions: usize,
        projected_tokens: usize,
    },
    ImageCount {
        images: usize,
        projected_tokens: usize,
    },
    NonFinite {
        tensor: &'static str,
        index: usize,
    },
    InvalidTokenConfiguration,
    Prepared(PreparedPrefillError),
}

impl fmt::Display for Gemma3VlmPreparedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("Gemma3 VLM prefill requires at least one token"),
            Self::ZeroHiddenSize => {
                formatter.write_str("Gemma3 VLM hidden size must be greater than zero")
            }
            Self::Capacity {
                sequence_len,
                context_capacity,
            } => write!(
                formatter,
                "Gemma3 VLM sequence length {sequence_len} exceeds context capacity {context_capacity}"
            ),
            Self::ShapeOverflow => formatter.write_str("Gemma3 VLM tensor shape overflowed"),
            Self::EmbeddingCount { expected, actual } => write!(
                formatter,
                "Gemma3 VLM raw text embeddings have {actual} values, expected {expected}"
            ),
            Self::ProjectedShape {
                values,
                hidden_size,
            } => write!(
                formatter,
                "Gemma3 VLM projected image features have {values} values, which is not divisible by hidden size {hidden_size}"
            ),
            Self::AttentionMaskCount { expected, actual } => write!(
                formatter,
                "Gemma3 VLM attention mask has {actual} values, expected {expected}"
            ),
            Self::InvalidAttentionMask { index, value } => write!(
                formatter,
                "Gemma3 VLM attention mask[{index}] is {value}; expected 0 or 1"
            ),
            Self::PaddingMaskMismatch {
                index,
                token_id,
                mask,
            } => write!(
                formatter,
                "Gemma3 VLM token {index} id {token_id} disagrees with padding mask {mask}"
            ),
            Self::PlaceholderCount {
                positions,
                projected_tokens,
            } => write!(
                formatter,
                "Gemma3 VLM prompt has {positions} image positions but projector returned {projected_tokens} tokens"
            ),
            Self::ImageCount {
                images,
                projected_tokens,
            } => write!(
                formatter,
                "Gemma3 VLM declares {images} image(s) but projector returned {projected_tokens} token(s)"
            ),
            Self::NonFinite { tensor, index } => write!(
                formatter,
                "Gemma3 VLM {tensor} contains a non-finite value at flat index {index}"
            ),
            Self::InvalidTokenConfiguration => {
                formatter.write_str("Gemma3 VLM pad_token_id and image_token_id must be distinct")
            }
            Self::Prepared(error) => write!(formatter, "Gemma3 VLM prepared prefill: {error}"),
        }
    }
}

impl std::error::Error for Gemma3VlmPreparedError {}

impl From<PreparedPrefillError> for Gemma3VlmPreparedError {
    fn from(value: PreparedPrefillError) -> Self {
        Self::Prepared(value)
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn first_non_finite(values: &[f32]) -> Option<usize> {
    values.iter().position(|value| !value.is_finite())
}

/// Build the exact post-scale embedding and additive-mask payload for one
/// Gemma3 VLM request.
#[allow(clippy::too_many_arguments)]
pub fn prepare_gemma3_vlm_prefill(
    token_ids: Vec<i32>,
    raw_text_embeddings: &[f32],
    projected_image_features: &[f32],
    attention_mask: &[i32],
    hidden_size: usize,
    context_capacity: usize,
    pad_token_id: i32,
    image_token_id: i32,
    image_count: usize,
) -> Result<PreparedPrefill, Gemma3VlmPreparedError> {
    let sequence_len = token_ids.len();
    if sequence_len == 0 {
        return Err(Gemma3VlmPreparedError::Empty);
    }
    if hidden_size == 0 {
        return Err(Gemma3VlmPreparedError::ZeroHiddenSize);
    }
    if sequence_len > context_capacity {
        return Err(Gemma3VlmPreparedError::Capacity {
            sequence_len,
            context_capacity,
        });
    }
    if pad_token_id == image_token_id {
        return Err(Gemma3VlmPreparedError::InvalidTokenConfiguration);
    }

    let expected_embeddings = sequence_len
        .checked_mul(hidden_size)
        .ok_or(Gemma3VlmPreparedError::ShapeOverflow)?;
    if raw_text_embeddings.len() != expected_embeddings {
        return Err(Gemma3VlmPreparedError::EmbeddingCount {
            expected: expected_embeddings,
            actual: raw_text_embeddings.len(),
        });
    }
    if attention_mask.len() != sequence_len {
        return Err(Gemma3VlmPreparedError::AttentionMaskCount {
            expected: sequence_len,
            actual: attention_mask.len(),
        });
    }
    if let Some(index) = first_non_finite(raw_text_embeddings) {
        return Err(Gemma3VlmPreparedError::NonFinite {
            tensor: "raw text embeddings",
            index,
        });
    }
    if let Some(index) = first_non_finite(projected_image_features) {
        return Err(Gemma3VlmPreparedError::NonFinite {
            tensor: "projected image features",
            index,
        });
    }
    if projected_image_features.len() % hidden_size != 0 {
        return Err(Gemma3VlmPreparedError::ProjectedShape {
            values: projected_image_features.len(),
            hidden_size,
        });
    }

    let mut image_positions = Vec::new();
    for (index, (&token_id, &mask)) in token_ids.iter().zip(attention_mask).enumerate() {
        if mask != 0 && mask != 1 {
            return Err(Gemma3VlmPreparedError::InvalidAttentionMask { index, value: mask });
        }
        let is_padding = token_id == pad_token_id;
        if is_padding != (mask == 0) {
            return Err(Gemma3VlmPreparedError::PaddingMaskMismatch {
                index,
                token_id,
                mask,
            });
        }
        if token_id == image_token_id {
            image_positions.push(index);
        }
    }
    let projected_tokens = projected_image_features.len() / hidden_size;
    if image_positions.len() != projected_tokens {
        return Err(Gemma3VlmPreparedError::PlaceholderCount {
            positions: image_positions.len(),
            projected_tokens,
        });
    }
    if (image_count == 0) != (projected_tokens == 0)
        || (image_count > 0 && !projected_tokens.is_multiple_of(image_count))
    {
        return Err(Gemma3VlmPreparedError::ImageCount {
            images: image_count,
            projected_tokens,
        });
    }

    let normalizer = (hidden_size as f64).sqrt() as f32;
    let mut merged = vec![0.0f32; expected_embeddings];
    let mut image_row = 0usize;
    for (position, &token_id) in token_ids.iter().enumerate() {
        let destination = position * hidden_size;
        if token_id == pad_token_id {
            continue;
        }
        if token_id == image_token_id {
            let source = image_row * hidden_size;
            merged[destination..destination + hidden_size]
                .copy_from_slice(&projected_image_features[source..source + hidden_size]);
            image_row += 1;
            continue;
        }
        for offset in 0..hidden_size {
            let value = raw_text_embeddings[destination + offset] * normalizer;
            if !value.is_finite() {
                return Err(Gemma3VlmPreparedError::NonFinite {
                    tensor: "post-scale text embeddings",
                    index: destination + offset,
                });
            }
            merged[destination + offset] = value;
        }
    }

    let bias_count = sequence_len
        .checked_mul(sequence_len)
        .ok_or(Gemma3VlmPreparedError::ShapeOverflow)?;
    let mut attention_bias = vec![GEMMA3_VLM_MASKED_VALUE; bias_count];
    for query in 0..sequence_len {
        for key in 0..sequence_len {
            if attention_mask[query] == 1 && attention_mask[key] == 1 {
                attention_bias[query * sequence_len + key] = 0.0;
            }
        }
    }

    let embeddings = OwnedTensor::new(
        f32_bytes(&merged),
        PreparedTensorDType::Float32,
        vec![1, sequence_len, hidden_size],
    )?;
    let attention_bias = PreparedAttentionBias {
        tensor: OwnedTensor::new(
            f32_bytes(&attention_bias),
            PreparedTensorDType::Float32,
            vec![1, 1, sequence_len, sequence_len],
        )?,
        causal: false,
    };
    let modalities = (image_count > 0).then(|| PreparedModality {
        family: "gemma3".to_string(),
        item_count: image_count,
        token_count: image_positions.len(),
    });
    PreparedPrefill::new(
        token_ids,
        embeddings,
        PreparedPositions::Sequential {
            start: 0,
            length: sequence_len,
        },
        attention_bias,
        modalities.into_iter().collect(),
    )
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(tensor: &OwnedTensor) -> Vec<f32> {
        tensor
            .bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn exports_post_scale_text_unscaled_images_and_exact_bidirectional_mask() {
        let prepared = prepare_gemma3_vlm_prefill(
            vec![0, 9, 7, 0],
            &[1.0, 2.0, 100.0, 200.0, 3.0, 4.0, 5.0, 6.0],
            &[11.0, 13.0],
            &[0, 1, 1, 0],
            2,
            8,
            0,
            9,
            1,
        )
        .unwrap();

        let sqrt_two = 2.0f32.sqrt();
        assert_eq!(
            read_f32(&prepared.embeddings),
            [
                0.0,
                0.0,
                11.0,
                13.0,
                3.0 * sqrt_two,
                4.0 * sqrt_two,
                0.0,
                0.0
            ]
        );
        let bias = read_f32(&prepared.attention_bias.tensor);
        assert_eq!(bias.len(), 16);
        assert_eq!(bias[1 * 4 + 1], 0.0);
        assert_eq!(bias[1 * 4 + 2], 0.0, "valid future keys stay bidirectional");
        assert_eq!(bias[2 * 4 + 1], 0.0);
        assert_eq!(bias[0], f32::MIN);
        assert_eq!(bias[3 * 4 + 2], f32::MIN);
        assert!(!prepared.attention_bias.causal);
        assert_eq!(prepared.modalities[0].token_count, 1);

        let static_payload = crate::prepared::PreparedIreePrefill::prepare(&prepared, 2, 8)
            .expect("Gemma3 mask materializes into the static XLA bucket");
        assert_eq!(static_payload.attention_bias.len(), 64);
        assert_eq!(static_payload.attention_bias[1 * 8 + 2], 0.0);
        assert_eq!(static_payload.attention_bias[1 * 8 + 7], f32::MIN);
        assert_eq!(static_payload.attention_bias[7 * 8 + 1], f32::MIN);
    }

    #[test]
    fn rejects_padding_disagreement_placeholder_mismatch_and_double_scale_overflow() {
        assert!(matches!(
            prepare_gemma3_vlm_prefill(vec![0], &[1.0], &[], &[1], 1, 1, 0, 9, 0),
            Err(Gemma3VlmPreparedError::PaddingMaskMismatch { .. })
        ));
        assert!(matches!(
            prepare_gemma3_vlm_prefill(vec![9], &[1.0], &[], &[1], 1, 1, 0, 9, 1),
            Err(Gemma3VlmPreparedError::PlaceholderCount { .. })
        ));
        assert!(matches!(
            prepare_gemma3_vlm_prefill(
                vec![7],
                &[f32::MAX, 0.0, 0.0, 0.0],
                &[],
                &[1],
                4,
                1,
                0,
                9,
                0,
            ),
            Err(Gemma3VlmPreparedError::NonFinite {
                tensor: "post-scale text embeddings",
                ..
            })
        ));
        assert!(matches!(
            prepare_gemma3_vlm_prefill(vec![9], &[1.0, 2.0], &[3.0], &[1], 2, 1, 0, 9, 1,),
            Err(Gemma3VlmPreparedError::ProjectedShape { .. })
        ));
        assert!(matches!(
            prepare_gemma3_vlm_prefill(vec![9], &[1.0], &[3.0], &[1], 1, 1, 0, 9, 0,),
            Err(Gemma3VlmPreparedError::ImageCount { .. })
        ));
    }
}
