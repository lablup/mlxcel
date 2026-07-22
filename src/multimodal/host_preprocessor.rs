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

//! Host-side multimodal preprocessing for compiler backends.
//!
//! The producer in this module may use MLX to run the existing vision tower,
//! projector, and embedding lookup. Its output is the fully-owned
//! [`PreparedPrefill`] contract from `mlxcel-core`, so no MLX value crosses the
//! session/worker boundary.

use std::path::Path;

use image::DynamicImage;
use mlxcel_core::layers::UnifiedEmbedding;
use mlxcel_core::session::{
    OwnedTensor, PreparedPrefill, PreparedPrefillError, PreparedTensorDType,
};
use thiserror::Error;

use crate::vision::connectors::MultiModalConnector;
use crate::vision::encoders::VisionEncoder;
use crate::vision::merge::{InputEmbeddings, merge_llava};
use crate::vision::processors::ImageProcessor;

use super::vlm_prompt::{ImageTokenBlockError, ImageTokenBlockInfo, apply_image_token_blocks};

#[path = "host_preprocessor_export.rs"]
mod export;
#[cfg(test)]
use export::export_mlx_tensor;
use export::{
    build_prepared_prefill, export_llava_prefill, usize_to_i32, validate_embedding_shape,
    validate_processor_shape, validate_projected_shape, validate_sequence_capacity,
};

/// A host preprocessor consumes tokenized text plus already-decoded images.
///
/// URL fetching, data-URI limits, image decoding, and other request security
/// policy remain at the server boundary. Audio and video are intentionally not
/// admitted by this first image-only contract.
pub trait HostMultimodalPreprocessor {
    /// Prepare an owned prefill payload for an engine worker.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, tensor-export, or family error before any
    /// compiler runtime is invoked.
    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError>;
}

/// LLaVA host preprocessor retaining only the components needed before XLA.
pub struct LlavaHostPreprocessor {
    processor: Box<dyn ImageProcessor>,
    encoder: Box<dyn VisionEncoder>,
    connector: Box<dyn MultiModalConnector>,
    /// Canonical checkpoint embedding lookup. This owns only
    /// `model.embed_tokens.{weight,scales,biases}` handles loaded through the
    /// filtered loader; no decoder layer or LM head is constructed or retained.
    text_embeddings: UnifiedEmbedding,
    image_token_id: i32,
    tokens_per_image: usize,
    hidden_size: usize,
    image_size: usize,
    max_sequence_len: usize,
}

impl LlavaHostPreprocessor {
    /// Load the LLaVA processor, vision tower, projector, and embedding lookup
    /// from a checkpoint without constructing a text decoder.
    ///
    /// The canonical filtered host loader and IREE both read the same immutable
    /// model directory. It explicitly bypasses process-global weight surgery,
    /// which IREE does not apply, so the host representation cannot select a
    /// different embedding revision or transformed value from the uploaded
    /// buffers. Changing either requires changing the shared model path.
    ///
    /// # Errors
    ///
    /// Returns an actionable typed error for incompatible families, malformed
    /// config, missing required weights, or an invalid embedding layout.
    pub fn load(model_path: &Path) -> Result<Self, HostPreprocessorError> {
        crate::loading::load_llava_host_preprocessor(model_path)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        processor: Box<dyn ImageProcessor>,
        encoder: Box<dyn VisionEncoder>,
        connector: Box<dyn MultiModalConnector>,
        text_embeddings: UnifiedEmbedding,
        image_token_id: i32,
        tokens_per_image: usize,
        hidden_size: usize,
        image_size: usize,
        max_sequence_len: usize,
    ) -> Result<Self, HostPreprocessorError> {
        if tokens_per_image == 0 {
            return Err(HostPreprocessorError::InvalidConfig(
                "LLaVA mm_tokens_per_image must be greater than zero".to_string(),
            ));
        }
        if hidden_size == 0 {
            return Err(HostPreprocessorError::InvalidConfig(
                "LLaVA text hidden_size must be greater than zero".to_string(),
            ));
        }
        if image_size == 0 {
            return Err(HostPreprocessorError::InvalidConfig(
                "LLaVA processor image_size must be greater than zero".to_string(),
            ));
        }
        if max_sequence_len == 0 {
            return Err(HostPreprocessorError::InvalidConfig(
                "LLaVA max sequence length must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            processor,
            encoder,
            connector,
            text_embeddings,
            image_token_id,
            tokens_per_image,
            hidden_size,
            image_size,
            max_sequence_len,
        })
    }

    fn token_block_info(&self) -> ImageTokenBlockInfo {
        ImageTokenBlockInfo {
            use_boi_eoi: false,
            image_token_id: self.image_token_id,
            mm_tokens_per_image: self.tokens_per_image,
            boi_token_id: 0,
            eoi_token_id: 0,
            has_bos: true,
            separator_token_id: None,
            suffix_tokens: Vec::new(),
            block_prefix_tokens: Vec::new(),
            block_suffix_tokens: Vec::new(),
        }
    }
}

impl HostMultimodalPreprocessor for LlavaHostPreprocessor {
    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        let mut logical_tokens = token_ids.to_vec();
        apply_image_token_blocks(&mut logical_tokens, self.token_block_info(), images.len())?;
        validate_sequence_capacity(logical_tokens.len(), self.max_sequence_len)?;

        let input_ids = mlxcel_core::from_slice_i32(
            &logical_tokens,
            &[1, usize_to_i32(logical_tokens.len(), "sequence length")?],
        );
        let text_embeddings = self.text_embeddings.forward(&input_ids);
        validate_embedding_shape(
            &mlxcel_core::array_shape(&text_embeddings),
            logical_tokens.len(),
            self.hidden_size,
            "text embedding table",
        )?;

        let merged = if images.is_empty() {
            InputEmbeddings {
                inputs_embeds: text_embeddings,
                attention_mask_4d: None,
            }
        } else {
            let pixels = self.processor.preprocess(images);
            validate_processor_shape(
                &mlxcel_core::array_shape(&pixels),
                images.len(),
                self.image_size,
            )?;

            // The standard processor emits channels-first f32. Normalize layout
            // for the shared CLIP/SigLIP encoder and normalize dtype once to the
            // text embedding dtype before vision execution.
            let pixels = mlxcel_core::transpose_axes(&pixels, &[0, 2, 3, 1]);
            let embed_dtype = mlxcel_core::array_dtype(&text_embeddings);
            let pixels = mlxcel_core::astype(&pixels, embed_dtype);
            let encoded = self.encoder.forward(&pixels);
            let projected = self.connector.forward(&encoded.hidden_states);
            validate_projected_shape(
                &mlxcel_core::array_shape(&projected),
                images.len(),
                self.tokens_per_image,
                self.hidden_size,
            )?;

            merge_llava(
                self.image_token_id,
                &projected,
                &text_embeddings,
                &input_ids,
            )
        };

        export_llava_prefill(
            logical_tokens,
            merged,
            self.image_token_id,
            images.len(),
            self.tokens_per_image,
            self.hidden_size,
        )
    }
}

/// Deterministic checkpoint-free producer for engine and server contract tests.
#[derive(Debug, Clone)]
pub struct FakeHostMultimodalPreprocessor {
    pub image_token_id: i32,
    pub tokens_per_image: usize,
    pub hidden_size: usize,
    pub max_sequence_len: usize,
}

impl Default for FakeHostMultimodalPreprocessor {
    fn default() -> Self {
        Self {
            image_token_id: -200,
            tokens_per_image: 4,
            hidden_size: 8,
            max_sequence_len: 4096,
        }
    }
}

impl HostMultimodalPreprocessor for FakeHostMultimodalPreprocessor {
    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        let mut logical_tokens = token_ids.to_vec();
        let info = ImageTokenBlockInfo {
            use_boi_eoi: false,
            image_token_id: self.image_token_id,
            mm_tokens_per_image: self.tokens_per_image,
            boi_token_id: 0,
            eoi_token_id: 0,
            has_bos: true,
            separator_token_id: None,
            suffix_tokens: Vec::new(),
            block_prefix_tokens: Vec::new(),
            block_suffix_tokens: Vec::new(),
        };
        apply_image_token_blocks(&mut logical_tokens, info, images.len())?;
        validate_sequence_capacity(logical_tokens.len(), self.max_sequence_len)?;

        let element_count = logical_tokens
            .len()
            .checked_mul(self.hidden_size)
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        let byte_count = element_count
            .checked_mul(PreparedTensorDType::Float32.size_bytes())
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        let mut bytes = Vec::with_capacity(byte_count);
        for (position, &token) in logical_tokens.iter().enumerate() {
            for lane in 0..self.hidden_size {
                // Stable across platforms and independent of image contents.
                let value = token as f32 * 0.001 + position as f32 * 0.01 + lane as f32;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let embeddings = OwnedTensor::new(
            bytes,
            PreparedTensorDType::Float32,
            vec![1, logical_tokens.len(), self.hidden_size],
        )?;
        let image_token_count = images
            .len()
            .checked_mul(self.tokens_per_image)
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        build_prepared_prefill(
            logical_tokens,
            embeddings,
            images.len(),
            image_token_count,
            "fake-llava",
        )
    }
}

/// Typed failures surfaced before an XLA runtime invocation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HostPreprocessorError {
    #[error(transparent)]
    Placeholder(#[from] ImageTokenBlockError),
    #[error("incompatible multimodal family: expected LLaVA, got {actual}")]
    FamilyMismatch { actual: String },
    #[error("invalid LLaVA host-preprocessor config: {0}")]
    InvalidConfig(String),
    #[error("failed to load LLaVA host-preprocessor weights: {0}")]
    WeightLoad(String),
    #[error(
        "processor output shape {actual:?} does not match decoded RGB batch [{image_count}, 3, {image_size}, {image_size}]"
    )]
    ProcessorShape {
        actual: Vec<i32>,
        image_count: usize,
        image_size: usize,
    },
    #[error("{source_name} shape {actual:?} does not match [1, {sequence_len}, {hidden_size}]")]
    EmbeddingShape {
        source_name: &'static str,
        actual: Vec<i32>,
        sequence_len: usize,
        hidden_size: usize,
    },
    #[error(
        "projected image shape {actual:?} does not match [{image_count}, {tokens_per_image}, {hidden_size}]"
    )]
    ProjectedShape {
        actual: Vec<i32>,
        image_count: usize,
        tokens_per_image: usize,
        hidden_size: usize,
    },
    #[error("expanded sequence length {actual} exceeds model capacity {maximum}")]
    SequenceCapacity { actual: usize, maximum: usize },
    #[error(
        "expanded prompt contains {actual} image token(s), expected {expected} from decoded media"
    )]
    ExpandedLength { actual: usize, expected: usize },
    #[error("unsupported prepared tensor dtype code {0}")]
    UnsupportedDType(i32),
    #[error("failed to export evaluated contiguous {tensor} tensor: {message}")]
    TensorExport {
        tensor: &'static str,
        message: String,
    },
    #[error("{label} cannot be represented by the MLX i32 shape ABI: {value}")]
    DimensionOverflow { label: &'static str, value: usize },
    #[error("prepared tensor shape calculation overflowed")]
    ShapeOverflow,
    #[error(transparent)]
    Prepared(#[from] PreparedPrefillError),
}

#[cfg(test)]
#[path = "host_preprocessor_tests.rs"]
mod tests;
