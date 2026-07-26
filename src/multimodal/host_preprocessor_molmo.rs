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

//! Molmo v1 prepared-prefill producer.
//!
//! The producer owns only the tokenizer, dual embedding table, processor, and
//! vision tower. The OLMo decoder and LM head remain exclusively in the XLA
//! session. Processor-provided `image_input_idx` values are the authoritative
//! sparse-add map; negative entries never consume a target position.

#[cfg(feature = "xla-iree")]
use std::cell::RefCell;
use std::path::Path;

use image::DynamicImage;
use mlxcel_core::session::{OwnedTensor, PreparedPrefill, PreparedTensorDType};

use crate::models::molmo::Molmo2Embedding;
use crate::tokenizer::MlxcelTokenizer;
use crate::vision::encoders::molmo::MolmoVisionModel;
use crate::vision::processors::molmo::MolmoProcessor;

use super::export::{
    build_prepared_prefill, export_mlx_tensor, usize_to_i32, validate_embedding_shape,
    validate_sequence_capacity,
};
use super::{HostMultimodalPreprocessor, HostPreprocessorError};

const MOLMO_V1_BOS_TOKEN_ID: i32 = 151_643;

/// Host components that build Molmo v1's owned XLA prepared-prefill payload.
pub struct MolmoHostPreprocessor {
    processor: MolmoProcessor,
    vision: MolmoVision,
    text_embeddings: Molmo2Embedding,
    tokenizer: MlxcelTokenizer,
    hidden_size: usize,
    max_sequence_len: usize,
    max_crops: usize,
    patches_per_crop: usize,
    projected_rows_per_crop: usize,
}

enum MolmoVision {
    Host(MolmoVisionModel),
    #[cfg(feature = "xla-iree")]
    Iree(RefCell<mlxcel_xla::IreeMolmoVisionProjector>),
}

impl MolmoHostPreprocessor {
    /// Load only Molmo v1's prefill-side components; no decoder is retained.
    pub fn load(model_path: &Path) -> Result<Self, HostPreprocessorError> {
        crate::loading::load_molmo_host_preprocessor(model_path)
    }

    #[cfg(feature = "xla-iree")]
    pub fn load_iree(model_path: &Path, device: &str) -> Result<Self, HostPreprocessorError> {
        crate::loading::load_molmo_iree_host_preprocessor(model_path, device)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        processor: MolmoProcessor,
        vision_tower: MolmoVisionModel,
        text_embeddings: Molmo2Embedding,
        tokenizer: MlxcelTokenizer,
        hidden_size: usize,
        max_sequence_len: usize,
        max_crops: usize,
        patches_per_crop: usize,
        projected_rows_per_crop: usize,
    ) -> Result<Self, HostPreprocessorError> {
        for (name, value) in [
            ("hidden size", hidden_size),
            ("maximum sequence length", max_sequence_len),
            ("maximum crop count", max_crops),
            ("patches per crop", patches_per_crop),
            ("projected rows per crop", projected_rows_per_crop),
        ] {
            if value == 0 {
                return Err(HostPreprocessorError::InvalidConfig(format!(
                    "Molmo v1 {name} must be non-zero"
                )));
            }
        }
        Ok(Self {
            processor,
            vision: MolmoVision::Host(vision_tower),
            text_embeddings,
            tokenizer,
            hidden_size,
            max_sequence_len,
            max_crops,
            patches_per_crop,
            projected_rows_per_crop,
        })
    }

    #[cfg(feature = "xla-iree")]
    pub(crate) fn from_iree_parts(
        processor: MolmoProcessor,
        text_embeddings: Molmo2Embedding,
        tokenizer: MlxcelTokenizer,
        max_sequence_len: usize,
        projector: mlxcel_xla::IreeMolmoVisionProjector,
    ) -> Result<Self, HostPreprocessorError> {
        let hidden_size = projector.text_hidden();
        let max_crops = projector.max_crops();
        let patches_per_crop = projector.patches_per_crop();
        let projected_rows_per_crop = projector.projected_rows_per_crop();
        if projector.patch_width() == 0 || max_sequence_len == 0 {
            return Err(HostPreprocessorError::InvalidConfig(
                "Molmo IREE static dimensions must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            processor,
            vision: MolmoVision::Iree(RefCell::new(projector)),
            text_embeddings,
            tokenizer,
            hidden_size,
            max_sequence_len,
            max_crops,
            patches_per_crop,
            projected_rows_per_crop,
        })
    }

    fn input_prompt(&self, token_ids: &[i32]) -> Result<String, HostPreprocessorError> {
        let ids = token_ids
            .iter()
            .map(|&token| {
                u32::try_from(token).map_err(|_| {
                    HostPreprocessorError::InvalidConfig(format!(
                        "Molmo v1 prompt token id {token} is negative"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.tokenizer
            .decode(&ids, true)
            .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))
    }

    fn logical_tokens(
        &self,
        token_ids: &[i32],
        image_tokens: &[i32],
    ) -> Result<Vec<i32>, HostPreprocessorError> {
        let prompt = format_prompt(&self.input_prompt(token_ids)?);
        let prompt_ids = self
            .tokenizer
            .encode(&prompt, false)
            .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
        let logical_len = 1usize
            .checked_add(image_tokens.len())
            .and_then(|length| length.checked_add(prompt_ids.len()))
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        validate_sequence_capacity(logical_len, self.max_sequence_len)?;
        let prompt_ids = prompt_ids
            .into_iter()
            .map(|token| {
                i32::try_from(token).map_err(|_| {
                    HostPreprocessorError::InvalidConfig(format!(
                        "Molmo v1 tokenizer id {token} does not fit i32"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut logical_tokens = Vec::with_capacity(logical_len);
        logical_tokens.push(MOLMO_V1_BOS_TOKEN_ID);
        logical_tokens.extend_from_slice(image_tokens);
        logical_tokens.extend(prompt_ids);
        Ok(logical_tokens)
    }

    fn prepare_image(
        &self,
        token_ids: &[i32],
        image: &DynamicImage,
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        let output = self.processor.preprocess_image(image);
        let logical_tokens = self.logical_tokens(token_ids, &output.image_token_ids)?;
        let crop_count = usize::try_from(output.pixel_values_shape[0])
            .map_err(|_| HostPreprocessorError::ShapeOverflow)?;
        let patches_per_crop = usize::try_from(output.pixel_values_shape[1])
            .map_err(|_| HostPreprocessorError::ShapeOverflow)?;
        let patch_width = usize::try_from(output.pixel_values_shape[2])
            .map_err(|_| HostPreprocessorError::ShapeOverflow)?;
        let shifted_indices = output
            .image_input_idx
            .iter()
            .map(|&index| {
                if index < 0 {
                    Ok(index)
                } else {
                    index
                        .checked_add(1)
                        .ok_or(HostPreprocessorError::ShapeOverflow)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prepared_image = mlxcel_xla::MolmoPreparedImage {
            pixel_values: output.pixel_values,
            crop_count,
            patches_per_crop,
            patch_width,
            image_masks: output.image_masks,
            projected_rows_per_crop: self.projected_rows_per_crop,
            image_input_idx: shifted_indices,
        };
        let max_feature_rows = self
            .max_crops
            .checked_mul(self.projected_rows_per_crop)
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        let plan = prepared_image
            .sparse_add_plan(
                logical_tokens.len(),
                self.hidden_size,
                self.max_crops,
                self.patches_per_crop,
                max_feature_rows,
                self.max_sequence_len,
            )
            .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;

        let input_ids = mlxcel_core::from_slice_i32(
            &logical_tokens,
            &[1, usize_to_i32(logical_tokens.len(), "sequence length")?],
        );
        let text = self.text_embeddings.forward(&input_ids);
        let embedding_dtype = match mlxcel_core::array_dtype(&text) {
            mlxcel_core::dtype::FLOAT16 => mlxcel_xla::MolmoEmbeddingDType::Float16,
            mlxcel_core::dtype::BFLOAT16 => mlxcel_xla::MolmoEmbeddingDType::BFloat16,
            mlxcel_core::dtype::FLOAT32 => mlxcel_xla::MolmoEmbeddingDType::Float32,
            other => return Err(HostPreprocessorError::UnsupportedDType(other)),
        };
        validate_embedding_shape(
            &mlxcel_core::array_shape(&text),
            logical_tokens.len(),
            self.hidden_size,
            "Molmo text embedding table",
        )?;

        let text = mlxcel_core::astype(&text, mlxcel_core::dtype::FLOAT32);
        let mut text_values = tensor_f32(export_mlx_tensor(&text, "Molmo text embeddings")?)?;
        let projected_values = match &self.vision {
            MolmoVision::Host(vision_tower) => {
                let pixels = mlxcel_core::from_slice_f32(
                    &prepared_image.pixel_values,
                    &[
                        1,
                        usize_to_i32(crop_count, "crop count")?,
                        usize_to_i32(patches_per_crop, "patches per crop")?,
                        usize_to_i32(patch_width, "patch width")?,
                    ],
                );
                let masks = mlxcel_core::from_slice_f32(
                    &prepared_image.image_masks,
                    &[
                        1,
                        usize_to_i32(crop_count, "crop count")?,
                        usize_to_i32(patches_per_crop, "patches per crop")?,
                    ],
                );
                let projected = mlxcel_core::astype(
                    &vision_tower.forward(&pixels, &masks),
                    mlxcel_core::dtype::FLOAT32,
                );
                let expected = [
                    1,
                    usize_to_i32(crop_count, "crop count")?,
                    usize_to_i32(self.projected_rows_per_crop, "projected rows per crop")?,
                    usize_to_i32(self.hidden_size, "hidden size")?,
                ];
                if mlxcel_core::array_shape(&projected) != expected {
                    return Err(HostPreprocessorError::InvalidConfig(format!(
                        "Molmo projected image shape {:?} does not match {expected:?}",
                        mlxcel_core::array_shape(&projected)
                    )));
                }
                tensor_f32(export_mlx_tensor(&projected, "Molmo projected features")?)?
            }
            #[cfg(feature = "xla-iree")]
            MolmoVision::Iree(projector) => {
                let projection = projector
                    .try_borrow_mut()
                    .map_err(|_| {
                        HostPreprocessorError::Iree(
                            "concurrent/re-entrant Molmo IREE vision invocation is unsupported"
                                .to_string(),
                        )
                    })?
                    .project(
                        &prepared_image.pixel_values,
                        &prepared_image.image_masks,
                        crop_count,
                    )
                    .map_err(HostPreprocessorError::Iree)?;
                tracing::info!(
                    vision_backend = "iree",
                    crop_count,
                    upload_bytes = projection.upload_bytes,
                    transfer_bytes = projection.transfer_bytes,
                    iree_vision_seconds = projection.elapsed_seconds,
                    "Molmo v1 native vision projection completed"
                );
                projection.values
            }
        };
        plan.apply_in_dtype(&mut text_values, &projected_values, embedding_dtype)
            .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
        let mut bytes = Vec::with_capacity(
            text_values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(HostPreprocessorError::ShapeOverflow)?,
        );
        for value in text_values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        let embeddings = OwnedTensor::new(
            bytes,
            PreparedTensorDType::Float32,
            vec![1, logical_tokens.len(), self.hidden_size],
        )?;
        build_prepared_prefill(
            logical_tokens,
            embeddings,
            1,
            output.image_token_ids.len(),
            "molmo-v1",
        )
    }
}

impl HostMultimodalPreprocessor for MolmoHostPreprocessor {
    fn backend(&self) -> super::XlaVisionBackend {
        match &self.vision {
            MolmoVision::Host(_) => super::XlaVisionBackend::Host,
            #[cfg(feature = "xla-iree")]
            MolmoVision::Iree(_) => super::XlaVisionBackend::Iree,
        }
    }

    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        match images {
            [image] => self.prepare_image(token_ids, image),
            [] => Err(HostPreprocessorError::InvalidConfig(
                "Molmo v1 image preprocessor requires exactly one image".to_string(),
            )),
            _ => Err(HostPreprocessorError::InvalidConfig(
                "Molmo v1 currently supports exactly one image per request".to_string(),
            )),
        }
    }
}

fn tensor_f32(tensor: OwnedTensor) -> Result<Vec<f32>, HostPreprocessorError> {
    if tensor.dtype != PreparedTensorDType::Float32 {
        return Err(HostPreprocessorError::InvalidConfig(format!(
            "Molmo sparse add requires float32 tensors, got {:?}",
            tensor.dtype
        )));
    }
    Ok(tensor
        .bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| {
            f32::from_ne_bytes(
                bytes
                    .try_into()
                    .expect("chunks_exact yields one native f32"),
            )
        })
        .collect())
}

fn format_prompt(prompt: &str) -> String {
    let without_image_placeholder = prompt.replace("<|image|>", "");
    let prompt = without_image_placeholder.trim();
    if prompt.starts_with("User:") && prompt.ends_with("Assistant:") {
        format!(" {prompt}")
    } else {
        format!(" User: {prompt} Assistant:")
    }
}

#[cfg(any(
    all(test, feature = "xla-diagnostics"),
    feature = "xla-reference-diagnostics"
))]
#[path = "host_preprocessor_molmo_diagnostics_tests.rs"]
mod diagnostics_tests;
#[cfg(feature = "xla-reference-diagnostics")]
pub use diagnostics_tests::run_pinned_molmo_eager_mlx_iree_boundaries;
#[cfg(test)]
#[path = "host_preprocessor_molmo_tests.rs"]
mod tests;
