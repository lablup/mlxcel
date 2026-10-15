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

//! Filtered Molmo v1 host producer loader for XLA prepared prefills.

use std::path::Path;

use mlxcel_core::weights::WeightMap;
use serde_json::Value;

use crate::models;
use crate::multimodal::host_preprocessor::{HostPreprocessorError, MolmoHostPreprocessor};
use crate::vision::encoders::molmo::{MolmoVisionConfig, MolmoVisionModel};
use crate::vision::processors::molmo::{MolmoImageTokens, MolmoProcessor};

use super::special::{
    inherit_quantization_if_missing, molmo_vision_i32, read_clip_triple, rewrite_molmo_weight_key,
};
use super::{
    load_vlm_weights_common_filtered_canonical, read_optional_model_json, read_sanitized_vlm_config,
};

/// Load only Molmo's dual WTE, vision stack, tokenizer, and processor.
pub(crate) fn load_molmo_host_preprocessor(
    model_path: &Path,
) -> Result<MolmoHostPreprocessor, HostPreprocessorError> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
    if full_config.get("model_type").and_then(Value::as_str) != Some("molmo") {
        return Err(HostPreprocessorError::FamilyMismatch {
            actual: full_config
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
        });
    }

    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    inherit_quantization_if_missing(&mut text_config_value, &full_config)
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
    let text_config: models::molmo::MolmoTextConfig = serde_json::from_value(text_config_value)
        .map_err(|error| {
            HostPreprocessorError::InvalidConfig(format!(
                "failed to parse Molmo v1 text config: {error}"
            ))
        })?;

    let empty = Value::Object(Default::default());
    let vision_config = full_config.get("vision_config").unwrap_or(&empty);
    let mut vit_layers = full_config
        .get("vit_layers")
        .or_else(|| vision_config.get("vit_layers"))
        .and_then(Value::as_array)
        .map(|layers| {
            layers
                .iter()
                .filter_map(|layer| layer.as_i64().map(|layer| layer as i32))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![-2, -9]);
    if vit_layers.is_empty() {
        vit_layers = vec![-2, -9];
    }
    let image_patch_size = molmo_vision_i32(vision_config, "image_patch_size", 14) as usize;
    let (input_h, input_w) = vision_config
        .get("image_default_input_size")
        .and_then(Value::as_array)
        .and_then(|dimensions| {
            Some((
                dimensions.first()?.as_i64()? as i32,
                dimensions.get(1)?.as_i64()? as i32,
            ))
        })
        .unwrap_or((336, 336));
    let patch_h = input_h / image_patch_size as i32;
    let patch_w = input_w / image_patch_size as i32;
    let pool_h = molmo_vision_i32(vision_config, "image_pooling_h", 2);
    let pool_w = molmo_vision_i32(vision_config, "image_pooling_w", 2);
    if patch_h <= 0 || patch_w <= 0 || pool_h <= 0 || pool_w <= 0 {
        return Err(HostPreprocessorError::InvalidConfig(
            "Molmo v1 patch and pooling dimensions must be positive".to_string(),
        ));
    }
    let vision_cfg = MolmoVisionConfig {
        image_num_layers: molmo_vision_i32(vision_config, "image_num_layers", 23) as usize,
        image_emb_dim: molmo_vision_i32(vision_config, "image_emb_dim", 1024),
        image_num_heads: molmo_vision_i32(vision_config, "image_num_heads", 16),
        image_num_kv_heads: molmo_vision_i32(vision_config, "image_num_key_value_heads", 16),
        image_head_dim: molmo_vision_i32(vision_config, "image_head_dim", 64),
        image_num_pos: molmo_vision_i32(vision_config, "image_num_pos", 577) as usize,
        image_norm_eps: vision_config
            .get("image_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-5) as f32,
        image_num_patch: (patch_h, patch_w),
        image_pooling_h: pool_h,
        image_pooling_w: pool_w,
        vit_layers,
        group_size: text_config.group_size(),
        bits: text_config.bits(),
    };

    let raw_weights = load_vlm_weights_common_filtered_canonical(model_path, |name| {
        name.starts_with("vision_tower.")
            || name.starts_with("model.vision_backbone.")
            || name.starts_with("language_model.model.wte.")
            || name.starts_with("model.transformer.wte.")
    })
    .map_err(|error| HostPreprocessorError::WeightLoad(error.to_string()))?;
    let mut weights = WeightMap::new();
    for (name, value) in raw_weights {
        weights.insert(rewrite_molmo_weight_key(&name), value);
    }
    let text_embeddings =
        models::molmo::Molmo2Embedding::from_weights(&weights, "language_model.model.wte")
            .map_err(|error| HostPreprocessorError::WeightLoad(error.to_string()))?;
    let vision_tower = MolmoVisionModel::from_weights(&weights, "vision_tower", vision_cfg)
        .map_err(HostPreprocessorError::WeightLoad)?;
    let tokenizer = crate::tokenizer::load_tokenizer(model_path)
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;

    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let max_high_res_crops = preprocessor_config
        .as_ref()
        .and_then(|config| config.get("max_crops"))
        .and_then(Value::as_u64)
        .unwrap_or(12) as usize;
    let overlap = preprocessor_config
        .as_ref()
        .and_then(|config| config.get("overlap_margins"))
        .and_then(Value::as_array)
        .and_then(|values| {
            Some((
                values.first()?.as_u64()? as usize,
                values.get(1)?.as_u64()? as usize,
            ))
        });
    let base_size = preprocessor_config
        .as_ref()
        .and_then(|config| config.get("base_image_input_size"))
        .and_then(Value::as_array)
        .and_then(|values| {
            Some((
                values.first()?.as_u64()? as usize,
                values.get(1)?.as_u64()? as usize,
            ))
        });
    let token_len = preprocessor_config.as_ref().and_then(|config| {
        Some((
            config.get("image_token_length_h")?.as_u64()? as usize,
            config.get("image_token_length_w")?.as_u64()? as usize,
        ))
    });
    let processor = MolmoProcessor::new(
        max_high_res_crops,
        overlap,
        Some(image_patch_size),
        base_size,
        token_len,
        read_clip_triple(preprocessor_config.as_ref(), "image_mean"),
        read_clip_triple(preprocessor_config.as_ref(), "image_std"),
        MolmoImageTokens::default(),
    );
    let max_crops = max_high_res_crops
        .checked_add(1)
        .ok_or(HostPreprocessorError::ShapeOverflow)?;
    let patches_per_crop = usize::try_from(patch_h)
        .ok()
        .and_then(|height| usize::try_from(patch_w).ok().map(|width| height * width))
        .ok_or(HostPreprocessorError::ShapeOverflow)?;
    let projected_rows_per_crop = usize::try_from(patch_h / pool_h)
        .ok()
        .and_then(|height| {
            usize::try_from(patch_w / pool_w)
                .ok()
                .and_then(|width| height.checked_mul(width))
        })
        .ok_or(HostPreprocessorError::ShapeOverflow)?;
    let max_sequence_len = full_config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;

    MolmoHostPreprocessor::from_parts(
        processor,
        vision_tower,
        text_embeddings,
        tokenizer,
        text_config.hidden_size,
        max_sequence_len,
        max_crops,
        patches_per_crop,
        projected_rows_per_crop,
    )
}

#[cfg(feature = "xla-iree")]
pub(crate) fn load_molmo_iree_host_preprocessor(
    model_path: &Path,
    device: &str,
) -> Result<MolmoHostPreprocessor, HostPreprocessorError> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
    if full_config.get("model_type").and_then(Value::as_str) != Some("molmo") {
        return Err(HostPreprocessorError::FamilyMismatch {
            actual: full_config
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
        });
    }
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    inherit_quantization_if_missing(&mut text_config_value, &full_config)
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
    let text_config: models::molmo::MolmoTextConfig = serde_json::from_value(text_config_value)
        .map_err(|error| {
            HostPreprocessorError::InvalidConfig(format!(
                "failed to parse Molmo v1 text config: {error}"
            ))
        })?;
    let raw_weights = load_vlm_weights_common_filtered_canonical(model_path, |name| {
        name.starts_with("language_model.model.wte.") || name.starts_with("model.transformer.wte.")
    })
    .map_err(|error| HostPreprocessorError::WeightLoad(error.to_string()))?;
    let mut weights = WeightMap::new();
    for (name, value) in raw_weights {
        weights.insert(rewrite_molmo_weight_key(&name), value);
    }
    let text_embeddings =
        models::molmo::Molmo2Embedding::from_weights(&weights, "language_model.model.wte")
            .map_err(HostPreprocessorError::WeightLoad)?;
    let tokenizer = crate::tokenizer::load_tokenizer(model_path)
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let max_crops = preprocessor_config
        .as_ref()
        .and_then(|config| config.get("max_crops"))
        .and_then(Value::as_u64)
        .unwrap_or(12) as usize;
    let overlap = preprocessor_config
        .as_ref()
        .and_then(|config| config.get("overlap_margins"))
        .and_then(Value::as_array)
        .and_then(|values| {
            Some((
                values.first()?.as_u64()? as usize,
                values.get(1)?.as_u64()? as usize,
            ))
        });
    let base_size = preprocessor_config
        .as_ref()
        .and_then(|config| config.get("base_image_input_size"))
        .and_then(Value::as_array)
        .and_then(|values| {
            Some((
                values.first()?.as_u64()? as usize,
                values.get(1)?.as_u64()? as usize,
            ))
        });
    let token_len = preprocessor_config.as_ref().and_then(|config| {
        Some((
            config.get("image_token_length_h")?.as_u64()? as usize,
            config.get("image_token_length_w")?.as_u64()? as usize,
        ))
    });
    let vision_config = full_config.get("vision_config").unwrap_or(&full_config);
    let image_patch_size = molmo_vision_i32(vision_config, "image_patch_size", 14) as usize;
    let processor = MolmoProcessor::new(
        max_crops,
        overlap,
        Some(image_patch_size),
        base_size,
        token_len,
        read_clip_triple(preprocessor_config.as_ref(), "image_mean"),
        read_clip_triple(preprocessor_config.as_ref(), "image_std"),
        MolmoImageTokens::default(),
    );
    let max_sequence_len = full_config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    let projector = mlxcel_xla::IreeMolmoVisionProjector::load(model_path, device)
        .map_err(HostPreprocessorError::Iree)?;
    if projector.text_hidden() != text_config.hidden_size {
        return Err(HostPreprocessorError::InvalidConfig(format!(
            "Molmo IREE text hidden {} disagrees with text config {}",
            projector.text_hidden(),
            text_config.hidden_size
        )));
    }
    MolmoHostPreprocessor::from_iree_parts(
        processor,
        text_embeddings,
        tokenizer,
        max_sequence_len,
        projector,
    )
}
