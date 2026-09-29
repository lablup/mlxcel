// Copyright 2025-2026 Lablup Inc.
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

//! Mage-VL (`mage_vl`) loader.
//!
//! Accepts both weight layouts:
//!
//! - hub (`microsoft/Mage-VL`): `model.language_model.*`, `lm_head.*`,
//!   `model.visual.*` with a torch `[out, C, P, P]` patch kernel;
//! - MLX conversion (`mlx-community/Mage-VL-8bit`): `language_model.model.*`,
//!   `language_model.lm_head.*`, `vision_tower.*` with a channels-last patch
//!   kernel.
//!
//! Both are normalized to `model.*` / `lm_head.*` for the Qwen3 decoder and
//! `vision_tower.*` for Mage-ViT. The decoder RoPE base is lifted out of
//! `text_config.rope_parameters`, and the top-level `quantization` block is
//! inherited by `text_config`.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::mage_vl_config::{
    MageVlTokenIds, MageVlVisionConfig, lift_rope_parameters, resolve_eos_token_ids,
    resolve_normalization, resolve_pixel_bounds,
};

use super::{load_vlm_weights_common, read_optional_model_json, read_sanitized_vlm_config};

/// Prefix the tower is loaded from after [`remap_mage_vl_weight_key`].
pub(crate) const MAGE_VL_VISION_PREFIX: &str = "vision_tower";

/// Normalize one checkpoint key to the layout the loaders read.
pub(crate) fn remap_mage_vl_weight_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.visual.") {
        format!("vision_tower.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        rest.to_string()
    } else {
        key.to_string()
    }
}

/// Take `text_config`, lift its RoPE base and inherit the top-level
/// quantization block.
pub(crate) fn mage_vl_text_config(full_config: &Value) -> Result<Value> {
    let mut text_config = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in Mage-VL config.json"))?;
    super::require_object_mut(&mut text_config, "Mage-VL text_config")?;
    lift_rope_parameters(&mut text_config);
    if text_config.get("quantization").is_none_or(Value::is_null)
        && let Some(q) = full_config.get("quantization").filter(|q| !q.is_null())
    {
        super::require_object_mut(&mut text_config, "Mage-VL text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    match text_config.get("model_type").and_then(Value::as_str) {
        None | Some("qwen3") => Ok(text_config),
        Some(other) => Err(anyhow::anyhow!(
            "Unsupported Mage-VL text_config.model_type: '{other}' (expected 'qwen3')"
        )),
    }
}

/// Load a Mage-VL VLM.
pub(crate) fn load_mage_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::mage_vl::MageVlVisionEncoder;
    use vision::mage_vl::MageVlModel;
    use vision::processors::qwen2_vl::Qwen2VLProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let has_vision = models::vlm_has_vision(&full_config, model_path);

    let text_config = mage_vl_text_config(&full_config)?;
    let mut text_args: models::qwen3::ModelArgs = serde_json::from_value(text_config)
        .map_err(|e| anyhow::anyhow!("Failed to parse Mage-VL text_config: {e}"))?;
    text_args.set_checkpoint_label(model_path);

    // Validate the vision config before the multi-GB weight load.
    let vision_config: MageVlVisionConfig = if has_vision {
        let raw = full_config
            .get("vision_config")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let config: MageVlVisionConfig = serde_json::from_value(raw)
            .map_err(|e| anyhow::anyhow!("Failed to parse Mage-VL vision_config: {e}"))?;
        config.validate().map_err(anyhow::Error::msg)?;
        config
    } else {
        MageVlVisionConfig::default()
    };
    let token_ids = MageVlTokenIds::from_config(&full_config).map_err(anyhow::Error::msg)?;

    let raw = load_vlm_weights_common(model_path, None)?;
    let mut weights = WeightMap::new();
    for (key, value) in raw {
        let key = remap_mage_vl_weight_key(&key);
        if !has_vision && key.starts_with("vision_tower.") {
            continue;
        }
        weights.insert(key, value);
    }
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Qwen3Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Mage-VL Qwen3 decoder: {e}"))?;
    weights.retain(|key, _| key.starts_with("vision_tower."));

    let vision = if has_vision {
        Some(
            MageVlVisionEncoder::from_weights(
                &weights,
                &vision_config,
                MAGE_VL_VISION_PREFIX,
                text_args.group_size(),
                text_args.bits(),
            )
            .map_err(|e| anyhow::anyhow!("Failed to load Mage-ViT vision tower: {e}"))?,
        )
    } else {
        None
    };
    drop(weights);

    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let (min_pixels, max_pixels) = resolve_pixel_bounds(preprocessor_config.as_ref());
    let (mean, std) = resolve_normalization(preprocessor_config.as_ref());
    let processor = Qwen2VLProcessor::new_with_norm(
        vision_config.patch_size,
        vision_config.temporal_patch_size,
        vision_config.spatial_merge_size,
        mean,
        std,
    )
    .with_pixel_bounds(min_pixels, max_pixels);

    let generation_config = read_optional_model_json(model_path, "generation_config.json");
    let eos_token_ids = resolve_eos_token_ids(generation_config.as_ref(), &full_config);

    Ok(LoadedModel::MageVLM(MageVlModel {
        text_model,
        vision,
        processor,
        token_ids,
        spatial_merge_size: vision_config.spatial_merge_size,
        eos_token_ids,
        model_path: model_path.display().to_string(),
    }))
}

#[cfg(test)]
#[path = "vlm_mage_vl_tests.rs"]
mod tests;
