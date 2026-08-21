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

use std::cell::RefCell;
use std::path::Path;

use image::DynamicImage;
use mlxcel_core::session::{OwnedTensor, PreparedPrefill, PreparedTensorDType};

use super::export::{
    build_prepared_prefill, export_mlx_tensor, usize_to_i32, validate_embedding_shape,
    validate_sequence_capacity,
};
use super::{HostMultimodalPreprocessor, HostPreprocessorError, XlaVisionBackend};
use crate::models::molmo2::Molmo2Embedding;
use crate::vision::processors::molmo2::Molmo2Processor;

const IMAGE_START_ID: i32 = 151936;
const IMAGE_END_ID: i32 = 151937;
const IMAGE_PATCH_ID: i32 = 151938;
const IMAGE_COL_ID: i32 = 151939;
const LOW_RES_IMAGE_START_ID: i32 = 151940;
const IMAGE_PLACEHOLDER_ID: i32 = 151941;

pub struct Molmo2IreeHostPreprocessor {
    processor: Molmo2Processor,
    text_embeddings: Molmo2Embedding,
    projector: RefCell<mlxcel_xla::IreeMolmo2VisionProjector>,
    hidden_size: usize,
    max_sequence_len: usize,
    device: String,
}

impl Molmo2IreeHostPreprocessor {
    pub fn load(model_path: &Path, device: &str) -> Result<Self, HostPreprocessorError> {
        let (text_embeddings, hidden_size, max_sequence_len) =
            crate::loading::load_molmo2_xla_text_embeddings(model_path)
                .map_err(|error| HostPreprocessorError::WeightLoad(error.to_string()))?;
        let preprocessor_path = model_path.join("preprocessor_config.json");
        let preprocessor_text = std::fs::read_to_string(&preprocessor_path).map_err(|error| {
            HostPreprocessorError::InvalidConfig(format!(
                "read {}: {error}",
                preprocessor_path.display()
            ))
        })?;
        let config: serde_json::Value =
            serde_json::from_str(&preprocessor_text).map_err(|error| {
                HostPreprocessorError::InvalidConfig(format!(
                    "parse {}: {error}",
                    preprocessor_path.display()
                ))
            })?;
        let pair = |key: &str| -> Result<(usize, usize), HostPreprocessorError> {
            let values = config
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    HostPreprocessorError::InvalidConfig(format!(
                        "Molmo2 preprocessor {key} must contain two integers"
                    ))
                })?;
            let first = values.first().and_then(serde_json::Value::as_u64);
            let second = values.get(1).and_then(serde_json::Value::as_u64);
            match (first, second) {
                (Some(first), Some(second)) => Ok((first as usize, second as usize)),
                _ => Err(HostPreprocessorError::InvalidConfig(format!(
                    "Molmo2 preprocessor {key} must contain two nonnegative integers"
                ))),
            }
        };
        let size = config
            .get("size")
            .and_then(serde_json::Value::as_object)
            .and_then(|size| {
                Some((
                    size.get("height")?.as_u64()? as usize,
                    size.get("width")?.as_u64()? as usize,
                ))
            })
            .ok_or_else(|| {
                HostPreprocessorError::InvalidConfig(
                    "Molmo2 preprocessor size.height/width are required".to_string(),
                )
            })?;
        let max_crops = config
            .get("max_crops")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                HostPreprocessorError::InvalidConfig(
                    "Molmo2 preprocessor max_crops is required".to_string(),
                )
            })? as usize;
        let patch_size = config
            .get("patch_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                HostPreprocessorError::InvalidConfig(
                    "Molmo2 preprocessor patch_size is required".to_string(),
                )
            })? as usize;
        let processor = Molmo2Processor::new(
            max_crops,
            Some(pair("overlap_margins")?),
            Some(patch_size),
            Some(pair("pooling_size")?),
            Some(size),
        );
        let projector = mlxcel_xla::IreeMolmo2VisionProjector::load(model_path, device)
            .map_err(HostPreprocessorError::Iree)?;
        if projector.image_patch_id() != IMAGE_PATCH_ID
            || projector.text_hidden_size() != hidden_size
        {
            return Err(HostPreprocessorError::InvalidConfig(
                "Molmo2 vision/text token or hidden-size contract mismatch".to_string(),
            ));
        }
        Ok(Self {
            processor,
            text_embeddings,
            projector: RefCell::new(projector),
            hidden_size,
            max_sequence_len,
            device: device.to_string(),
        })
    }

    fn image_tokens(grid: [i32; 4]) -> Result<Vec<i32>, HostPreprocessorError> {
        let [lo_h, lo_w, hi_h, hi_w] = grid;
        let dimensions = [lo_h, lo_w, hi_h, hi_w]
            .map(|value| usize::try_from(value).map_err(|_| HostPreprocessorError::ShapeOverflow))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let mut tokens = Vec::new();
        tokens.push(LOW_RES_IMAGE_START_ID);
        for _ in 0..dimensions[0] {
            tokens.extend(std::iter::repeat_n(IMAGE_PATCH_ID, dimensions[1]));
            tokens.push(IMAGE_COL_ID);
        }
        tokens.push(IMAGE_END_ID);
        tokens.push(IMAGE_START_ID);
        for _ in 0..dimensions[2] {
            tokens.extend(std::iter::repeat_n(IMAGE_PATCH_ID, dimensions[3]));
            tokens.push(IMAGE_COL_ID);
        }
        tokens.push(IMAGE_END_ID);
        Ok(tokens)
    }

    fn expand_prompt(
        token_ids: &[i32],
        image_tokens: &[i32],
    ) -> Result<Vec<i32>, HostPreprocessorError> {
        let positions = token_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &token)| (token == IMAGE_PLACEHOLDER_ID).then_some(index))
            .collect::<Vec<_>>();
        if positions.len() > 1 {
            return Err(HostPreprocessorError::InvalidConfig(format!(
                "Molmo2 prompt contains {} image placeholders but one image is supported",
                positions.len()
            )));
        }
        let Some(position) = positions.first().copied() else {
            let mut expanded = image_tokens.to_vec();
            expanded.extend_from_slice(token_ids);
            return Ok(expanded);
        };
        let mut expanded = Vec::with_capacity(token_ids.len() - 1 + image_tokens.len());
        expanded.extend_from_slice(&token_ids[..position]);
        expanded.extend_from_slice(image_tokens);
        expanded.extend_from_slice(&token_ids[position + 1..]);
        Ok(expanded)
    }

    fn prepare_one(
        &self,
        token_ids: &[i32],
        image: &DynamicImage,
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        let processed = self.processor.preprocess_image(image);
        let image_tokens = Self::image_tokens(processed.image_grid)?;
        let logical_tokens = Self::expand_prompt(token_ids, &image_tokens)?;
        let prompt_image_patch_count = logical_tokens
            .iter()
            .filter(|&&token| token == IMAGE_PATCH_ID)
            .count();
        validate_sequence_capacity(logical_tokens.len(), self.max_sequence_len)?;
        let input_ids = mlxcel_core::from_slice_i32(
            &logical_tokens,
            &[1, usize_to_i32(logical_tokens.len(), "sequence length")?],
        );
        let text = mlxcel_core::astype(
            &self.text_embeddings.forward(&input_ids),
            mlxcel_core::dtype::FLOAT32,
        );
        validate_embedding_shape(
            &mlxcel_core::array_shape(&text),
            logical_tokens.len(),
            self.hidden_size,
            "Molmo2 dual embedding table",
        )?;
        let text = export_mlx_tensor(&text, "Molmo2 text embeddings")?;
        if text.dtype != PreparedTensorDType::Float32 {
            return Err(HostPreprocessorError::InvalidConfig(
                "Molmo2 prepared embeddings must be Float32".to_string(),
            ));
        }
        let mut text_values = text
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        let patches_shape = processed
            .pixel_values_shape
            .map(|value| usize::try_from(value).map_err(|_| HostPreprocessorError::ShapeOverflow))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let pooling_shape = processed
            .image_token_pooling_shape
            .map(|value| usize::try_from(value).map_err(|_| HostPreprocessorError::ShapeOverflow))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let mut projector = self.projector.try_borrow_mut().map_err(|_| {
            HostPreprocessorError::Iree(
                "concurrent/re-entrant Molmo2 vision invocation is unsupported".to_string(),
            )
        })?;
        let projection = projector
            .project(mlxcel_xla::Molmo2VisionInput {
                patches: &processed.pixel_values,
                patches_shape: [patches_shape[0], patches_shape[1], patches_shape[2]],
                image_token_pooling: &processed.image_token_pooling,
                pooling_shape: [pooling_shape[0], pooling_shape[1]],
                image_grid: processed.image_grid,
                image_num_crops: usize::try_from(processed.image_num_crops)
                    .map_err(|_| HostPreprocessorError::ShapeOverflow)?,
                prompt_image_patch_count,
            })
            .map_err(HostPreprocessorError::Iree)?;
        let positions = mlxcel_xla::add_molmo2_projected_features(
            &logical_tokens,
            IMAGE_PATCH_ID,
            &mut text_values,
            self.hidden_size,
            &projection.values,
        )
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
        let bytes = text_values
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let embeddings = OwnedTensor::new(
            bytes,
            PreparedTensorDType::Float32,
            vec![1, logical_tokens.len(), self.hidden_size],
        )?;
        tracing::info!(
            vision_backend = "iree",
            vision_device = %self.device,
            image_crops = processed.image_num_crops,
            image_tokens = positions.len(),
            iree_vision_seconds = projection.elapsed_seconds,
            "OpenXLA Molmo2 projection completed"
        );
        build_prepared_prefill(logical_tokens, embeddings, 1, positions.len(), "molmo2")
    }
}

impl HostMultimodalPreprocessor for Molmo2IreeHostPreprocessor {
    fn backend(&self) -> XlaVisionBackend {
        XlaVisionBackend::Iree
    }

    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        match images {
            [image] => self.prepare_one(token_ids, image),
            _ => Err(HostPreprocessorError::InvalidConfig(format!(
                "Molmo2 XLA requires exactly one image, got {}",
                images.len()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_expansion_preserves_framing_and_patch_order() {
        let image = Molmo2IreeHostPreprocessor::image_tokens([1, 2, 2, 1]).unwrap();
        assert_eq!(
            image,
            vec![
                LOW_RES_IMAGE_START_ID,
                IMAGE_PATCH_ID,
                IMAGE_PATCH_ID,
                IMAGE_COL_ID,
                IMAGE_END_ID,
                IMAGE_START_ID,
                IMAGE_PATCH_ID,
                IMAGE_COL_ID,
                IMAGE_PATCH_ID,
                IMAGE_COL_ID,
                IMAGE_END_ID,
            ]
        );
        let expanded =
            Molmo2IreeHostPreprocessor::expand_prompt(&[7, IMAGE_PLACEHOLDER_ID, 8], &image)
                .unwrap();
        assert_eq!(expanded.first(), Some(&7));
        assert_eq!(expanded.last(), Some(&8));
        assert_eq!(
            expanded
                .iter()
                .filter(|&&token| token == IMAGE_PATCH_ID)
                .count(),
            4
        );
    }

    #[test]
    fn prompt_without_placeholder_prefixes_image_and_rejects_duplicates() {
        assert_eq!(
            Molmo2IreeHostPreprocessor::expand_prompt(&[7, 8], &[1, 2]).unwrap(),
            vec![1, 2, 7, 8]
        );
        assert!(
            Molmo2IreeHostPreprocessor::expand_prompt(
                &[IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID],
                &[1]
            )
            .is_err()
        );
    }
}
