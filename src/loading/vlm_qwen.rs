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

//! Qwen-VL family loaders.
//!
//! Families:
//! - Qwen2-VL
//! - Qwen2.5-VL
//! - Qwen3-VL / Qwen3-VL-MoE
//! - Qwen3.5-VL / Qwen3.5-VL-MoE
//!
//! This file centralizes Qwen-specific token defaults, quantization inheritance,
//! and weight-key remapping so the generic VLM router does not need to know
//! about Qwen family details.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{
    Qwen35VlmVariant, QwenVisionTokenIds, inherit_qwen_text_quantization,
    inherit_qwen_vision_quantization, load_vlm_weights_common, parse_required_vlm_subconfig,
    parse_vlm_config, qwen_vl_processor, qwen_vl_processor_with_norm, qwen_vl_token_ids,
    qwen35_vlm_token_defaults, read_sanitized_vlm_config, remap_qwen3_vl_weights,
    strip_language_model_prefix, wrap_qwen35_vlm,
};

/// Load a Qwen2-VL model (custom ViT + Qwen2 language model with MRoPE)
pub(crate) fn load_qwen2_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::qwen2_vl::{Qwen2VLVisionConfig, Qwen2VLVisionEncoder};

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text config is at root level for Qwen2-VL (not inside text_config sub-object)
    let text_config: models::qwen2_vl::Qwen2VLConfig =
        parse_vlm_config(&config_str, "Qwen2VL text config")?;

    // Vision config is in vision_config sub-object
    let mut vision_config: Qwen2VLVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Qwen2VL vision config")?;

    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);

    // Sanitize tied embeddings
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // Build text model
    let text_model = models::Qwen2VLModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen2VL text model: {}", e))?;

    // Build vision encoder
    let vision_encoder =
        Qwen2VLVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Qwen2VL vision encoder: {}", e))?;

    // Build image processor
    let processor = qwen_vl_processor(&vision_config);

    // Get token IDs from config
    let token_ids = qwen_vl_token_ids(
        &full_config,
        QwenVisionTokenIds {
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
        },
    );

    let vlm = vision::Qwen2VLModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::Qwen2VL(vlm))
}

/// Load a Qwen2.5-VL model (windowed ViT + Qwen2 language model with MRoPE)
pub(crate) fn load_qwen2_5_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::qwen2_5_vl::{Qwen25VLVisionConfig, Qwen25VLVisionEncoder};

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let text_config: models::qwen2_vl::Qwen2VLConfig =
        parse_vlm_config(&config_str, "Qwen2.5VL text config")?;

    let mut vision_config: Qwen25VLVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Qwen2.5VL vision config")?;

    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Qwen2VLModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen2.5VL text model: {}", e))?;

    let vision_encoder =
        Qwen25VLVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Qwen2.5VL vision encoder: {}", e))?;

    let processor = qwen_vl_processor(&vision_config);
    let token_ids = qwen_vl_token_ids(
        &full_config,
        QwenVisionTokenIds {
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
        },
    );

    let vlm = vision::Qwen25VLModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::Qwen25VL(vlm))
}

/// Load a Qwen3-VL model (ViT + interleaved MRoPE + DeepStack)
pub(crate) fn load_qwen3_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config: models::qwen3_vl::Qwen3VLConfig =
        parse_required_vlm_subconfig(&full_config, "text_config", "Qwen3VL text config")?;
    inherit_qwen_text_quantization(&mut text_config, &full_config);

    let mut vision_config: Qwen3VLVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Qwen3VL vision config")?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let mut weights = remap_qwen3_vl_weights(load_vlm_weights_common(model_path, None)?, false);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Qwen3VLModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen3VL text model: {}", e))?;

    let vision_encoder =
        Qwen3VLVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Qwen3VL vision encoder: {}", e))?;

    let processor = qwen_vl_processor_with_norm(&vision_config, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    let token_ids = qwen_vl_token_ids(
        &full_config,
        QwenVisionTokenIds {
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
        },
    );

    let vlm = vision::Qwen3VLModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::Qwen3VL(vlm))
}

pub(crate) fn load_qwen3_vl_moe(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config: models::qwen3_vl_moe::Qwen3VLMoeConfig =
        parse_required_vlm_subconfig(&full_config, "text_config", "Qwen3VLMoe text config")?;
    inherit_qwen_text_quantization(&mut text_config, &full_config);

    let mut vision_config: Qwen3VLVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Qwen3VLMoe vision config")?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let mut weights = remap_qwen3_vl_weights(load_vlm_weights_common(model_path, None)?, true);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Qwen3VLMoeModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen3VLMoe text model: {}", e))?;

    let vision_encoder =
        Qwen3VLVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Qwen3VLMoe vision encoder: {}", e))?;

    let processor = qwen_vl_processor_with_norm(&vision_config, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    let token_ids = qwen_vl_token_ids(
        &full_config,
        QwenVisionTokenIds {
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
        },
    );

    let vlm = vision::Qwen3VLMoeModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::Qwen3VLMoe(vlm))
}

/// Load a Qwen3.5 VLM (Qwen3-VL vision encoder + Qwen3.5 hybrid text backbone)
pub(crate) fn load_qwen3_5_vlm(model_path: &Path) -> Result<LoadedModel> {
    load_qwen3_5_vlm_with_variant(model_path, Qwen35VlmVariant::Dense)
}

/// Load a Qwen3.5 MoE VLM (Qwen3-VL vision encoder + Qwen3.5 MoE hybrid text backbone)
pub(crate) fn load_qwen3_5_moe_vlm(model_path: &Path) -> Result<LoadedModel> {
    load_qwen3_5_vlm_with_variant(model_path, Qwen35VlmVariant::Moe)
}

fn load_qwen3_5_vlm_with_variant(
    model_path: &Path,
    variant: Qwen35VlmVariant,
) -> Result<LoadedModel> {
    use vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut vision_config: Qwen3VLVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Qwen3.5 vision config")?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let mut text_weights = mlxcel_core::weights::WeightMap::new();
    let mut vision_weights = mlxcel_core::weights::WeightMap::new();

    for (key, value) in raw_weights {
        if key.starts_with("model.language_model.") || key.starts_with("language_model.model.") {
            let new_key = key
                .replace("model.language_model.", "model.")
                .replace("language_model.model.", "model.");
            text_weights.insert(new_key, value);
        } else if key.starts_with("model.visual.") {
            let new_key = key.replacen("model.visual.", "vision_tower.", 1);
            vision_weights.insert(new_key, value);
        } else if key.starts_with("vision_tower.") {
            vision_weights.insert(key, value);
        } else if key.starts_with("language_model.lm_head.") {
            let new_key = key.replacen("language_model.", "", 1);
            text_weights.insert(new_key, value);
        } else if key.starts_with("lm_head.") || key.starts_with("model.") {
            text_weights.insert(key, value);
        }
    }

    let mut text_config_val = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in config.json"))?;

    if text_config_val.get("quantization").is_none() && full_config.get("quantization").is_some() {
        super::require_object_mut(&mut text_config_val, "Qwen3.5 text_config")?.insert(
            "quantization".to_string(),
            full_config["quantization"].clone(),
        );
    }

    let text_config: models::qwen3_5::Qwen35Config = serde_json::from_value(text_config_val)
        .map_err(|e| anyhow::anyhow!("Failed to parse Qwen3.5 text config: {}", e))?;

    let mut text_weights = models::qwen3_5::sanitize_weights(text_weights, &text_config);
    models::sanitize_tied_embeddings(&mut text_weights, &full_config);

    let mut text_model = models::Qwen35Model::from_weights(&text_weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen3.5 text model: {}", e))?;

    let mrope_section = text_config
        .rope_parameters
        .as_ref()
        .and_then(|rp| rp.get("mrope_section"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![11, 11, 10]);
    let rope_theta = text_config
        .rope_parameters
        .as_ref()
        .and_then(|rp| rp.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(10000000.0);
    let rope_dims = text_config.rope_dims() as usize;
    text_model.set_mrope(mrope_section, rope_theta, rope_dims);

    let vision_encoder =
        Qwen3VLVisionEncoder::from_weights(&vision_weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Qwen3.5 vision encoder: {}", e))?;

    let processor = qwen_vl_processor_with_norm(&vision_config, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    let token_ids = qwen_vl_token_ids(&full_config, qwen35_vlm_token_defaults());

    let vlm = vision::Qwen35VLModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(wrap_qwen35_vlm(vlm, variant))
}

/// Remap raw GLM-4V (`Glm4vForConditionalGeneration`) weight keys to the
/// mlxcel layout: `model.visual.*` -> `vision_tower.*`,
/// `model.language_model.*` -> `model.*`, `lm_head.*` unchanged.
fn remap_glm4v_weights(
    raw_weights: mlxcel_core::weights::WeightMap,
) -> mlxcel_core::weights::WeightMap {
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(rest) = key.strip_prefix("model.visual.") {
            format!("vision_tower.{}", rest)
        } else if let Some(rest) = key.strip_prefix("visual.") {
            format!("vision_tower.{}", rest)
        } else if let Some(rest) = key.strip_prefix("model.language_model.") {
            format!("model.{}", rest)
        } else if let Some(rest) = key.strip_prefix("language_model.model.") {
            format!("model.{}", rest)
        } else if let Some(rest) = key.strip_prefix("language_model.lm_head.") {
            format!("lm_head.{}", rest)
        } else {
            key
        };
        weights.insert(new_key, value);
    }
    weights
}

/// Load a GLM-4V model (GLM-4V ViT + GLM-4 text backbone with sectioned MRoPE).
pub(crate) fn load_glm4v(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::glm4v::{Glm4vVisionConfig, Glm4vVisionEncoder};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config: models::glm4v::Glm4vTextConfig =
        parse_required_vlm_subconfig(&full_config, "text_config", "GLM-4V text config")?;
    inherit_qwen_text_quantization(&mut text_config, &full_config);

    let mut vision_config: Glm4vVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "GLM-4V vision config")?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let mut weights = remap_glm4v_weights(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Glm4vTextModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load GLM-4V text model: {}", e))?;

    let vision_encoder = Glm4vVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load GLM-4V vision encoder: {}", e))?;

    let processor = qwen_vl_processor(&vision_config);
    let token_ids = qwen_vl_token_ids(
        &full_config,
        QwenVisionTokenIds {
            image_token_id: 151363,
            video_token_id: 151364,
            vision_start_token_id: 151339,
        },
    );

    let vlm = vision::Glm4vModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::Glm4v(vlm))
}

/// Drop text decoder layers whose index is `>= num_hidden_layers` (the
/// multi-token-prediction / next-n block). GLM-OCR ships a
/// `num_nextn_predict_layers: 1` MTP block at `model.layers.{num_hidden_layers}.*`
/// that inference never runs; dropping it keeps any layer-count logic honest and
/// avoids leaking unused tensors. Published mlx conversions may have dropped it
/// already, in which case this is a no-op.
fn drop_extra_text_layers(weights: &mut mlxcel_core::weights::WeightMap, num_hidden_layers: usize) {
    let to_remove: Vec<String> = weights
        .keys()
        .filter(|k| {
            k.strip_prefix("model.layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|idx| idx.parse::<usize>().ok())
                .is_some_and(|idx| idx >= num_hidden_layers)
        })
        .cloned()
        .collect();
    for k in to_remove {
        weights.remove(&k);
    }
}

/// Lift GLM-OCR's `rope_parameters` block into the fields `Glm4vTextConfig`
/// deserializes: `mrope_section` under `rope_scaling`, plus top-level
/// `partial_rotary_factor` and `rope_theta`. GLM-OCR nests all three under
/// `rope_parameters` (not `rope_scaling`) and omits the top-level scalars, so
/// feeding the config through unchanged silently yields half-width rotary with
/// the wrong sections.
fn normalize_glm_ocr_rope(text_config: &mut serde_json::Value) -> Result<()> {
    let obj = super::require_object_mut(text_config, "GLM-OCR text_config")?;
    let Some(rp) = obj.get("rope_parameters").cloned() else {
        return Ok(());
    };
    if let Some(section) = rp.get("mrope_section").cloned() {
        let rs = obj
            .entry("rope_scaling".to_string())
            .or_insert_with(|| serde_json::json!({}));
        if let Some(rs_obj) = rs.as_object_mut() {
            rs_obj.insert("mrope_section".to_string(), section);
        }
    }
    if obj
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .is_none()
        && let Some(prf) = rp.get("partial_rotary_factor").cloned()
    {
        obj.insert("partial_rotary_factor".to_string(), prf);
    }
    if obj.get("rope_theta").and_then(|v| v.as_f64()).is_none() {
        let theta = rp
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        obj.insert("rope_theta".to_string(), serde_json::json!(theta));
    }
    Ok(())
}

/// Load a GLM-OCR model (GLM-OCR ViT + GLM-4 text backbone, full-width MRoPE).
pub(crate) fn load_glm_ocr(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::glm_ocr::GlmOcrVisionEncoder;
    use vision::encoders::glm4v::Glm4vVisionConfig;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text config: lift `rope_parameters`, then reuse the GLM-4V text backbone.
    let mut text_config_val = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in config.json"))?;
    normalize_glm_ocr_rope(&mut text_config_val)?;

    let mut text_config: models::glm4v::Glm4vTextConfig =
        serde_json::from_value(text_config_val)
            .map_err(|e| anyhow::anyhow!("Failed to parse GLM-OCR text config: {}", e))?;
    inherit_qwen_text_quantization(&mut text_config, &full_config);

    let mut vision_config: Glm4vVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "GLM-OCR vision config")?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    // Remap raw keys (`model.visual.*` -> `vision_tower.*`,
    // `model.language_model.*` -> `model.*`), then drop the MTP layer.
    let mut weights = remap_glm4v_weights(load_vlm_weights_common(model_path, None)?);
    drop_extra_text_layers(&mut weights, text_config.num_hidden_layers);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Glm4vTextModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load GLM-OCR text model: {}", e))?;

    let vision_encoder =
        GlmOcrVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load GLM-OCR vision encoder: {}", e))?;

    // OCR pixel bounds (preprocessor_config `size` = shortest/longest edge).
    let mut processor = qwen_vl_processor(&vision_config);
    processor.min_pixels = 12544;
    processor.max_pixels = 9633792;

    // GLM-OCR names its start/end tokens `image_start_token_id` /
    // `image_end_token_id` (defaults 59256 / 59257), which are consecutive so the
    // `vision_end = vision_start + 1` insertion assumption still holds.
    let image_token_id = full_config
        .get("image_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(59280) as i32;
    let video_token_id = full_config
        .get("video_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(59281) as i32;
    let vision_start_token_id = full_config
        .get("image_start_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(59256) as i32;

    let vlm = vision::GlmOcrModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::GlmOcr(vlm))
}

/// Extract the `mrope_section` from a GLM-4V MoE text config, checking both
/// `rope_scaling` and `rope_parameters`; defaults to `[64, 32, 32]`.
fn extract_glm4v_mrope_section(text_config: &serde_json::Value) -> Vec<i32> {
    for key in ["rope_scaling", "rope_parameters"] {
        if let Some(section) = text_config
            .get(key)
            .and_then(|rs| rs.get("mrope_section"))
            .and_then(|v| v.as_array())
        {
            let vals: Vec<i32> = section
                .iter()
                .filter_map(|v| v.as_i64().map(|i| i as i32))
                .collect();
            if !vals.is_empty() {
                return vals;
            }
        }
    }
    vec![64, 32, 32]
}

/// Extract EOS token ids from the top-level or text config; defaults to the
/// GLM-4V MoE end tokens.
fn extract_glm4v_eos_token_ids(
    full_config: &serde_json::Value,
    text_config: &serde_json::Value,
) -> Vec<i32> {
    for cfg in [full_config, text_config] {
        if let Some(eos) = cfg.get("eos_token_id") {
            if let Some(arr) = eos.as_array() {
                let vals: Vec<i32> = arr
                    .iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect();
                if !vals.is_empty() {
                    return vals;
                }
            } else if let Some(single) = eos.as_i64() {
                return vec![single as i32];
            }
        }
    }
    vec![151329, 151336, 151338]
}

/// Load a GLM-4V MoE model (GLM-4V ViT + GLM-4 MoE text backbone with MRoPE).
pub(crate) fn load_glm4v_moe(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::glm4v::{Glm4vVisionConfig, Glm4vVisionEncoder};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text config: normalize `rope_theta` (it may live under `rope_parameters`)
    // and inherit top-level quantization, then reuse the shared GLM-4 MoE args.
    let mut text_config_val = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in config.json"))?;
    {
        let obj = super::require_object_mut(&mut text_config_val, "GLM-4V MoE text_config")?;
        if obj.get("rope_theta").and_then(|v| v.as_f64()).is_none() {
            let theta = obj
                .get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0);
            obj.insert("rope_theta".to_string(), serde_json::json!(theta));
        }
        if obj.get("quantization_config").is_none()
            && obj.get("group_size").is_none()
            && let Some(q) = full_config.get("quantization")
        {
            obj.insert("quantization_config".to_string(), q.clone());
        }
    }

    let args: models::glm4_moe::ModelArgs = serde_json::from_value(text_config_val.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse GLM-4V MoE text config: {}", e))?;
    let mrope_section = extract_glm4v_mrope_section(&text_config_val);
    let eos_token_ids = extract_glm4v_eos_token_ids(&full_config, &text_config_val);

    let mut vision_config: Glm4vVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "GLM-4V MoE vision config")?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let mut weights = remap_glm4v_weights(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model =
        models::Glm4vMoeTextModel::from_weights(&weights, &args, &mrope_section, eos_token_ids)
            .map_err(|e| anyhow::anyhow!("Failed to load GLM-4V MoE text model: {}", e))?;

    let vision_encoder = Glm4vVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load GLM-4V MoE vision encoder: {}", e))?;

    let processor = qwen_vl_processor(&vision_config);
    let token_ids = qwen_vl_token_ids(
        &full_config,
        QwenVisionTokenIds {
            image_token_id: 151363,
            video_token_id: 151364,
            vision_start_token_id: 151339,
        },
    );

    let vlm = vision::Glm4vMoeModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::Glm4vMoe(vlm))
}

/// Load a Qwen3-Omni MoE thinker (`qwen3_omni_moe`, stage 1: text output from
/// text + image + audio inputs).
///
/// Verified against `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`: text
/// quantized 4-bit under `thinker.language_model.model.*` with pre-stacked
/// `switch_mlp` experts and an untied quantized `lm_head`; vision and audio
/// towers plain bf16 under `thinker.vision_tower.*` / `thinker.audio_tower.*`
/// with conv weights already channels-last. Raw exports instead ship
/// `thinker.model.*` / `thinker.visual.*`; both spellings are remapped. The
/// talker / code2wav speech stack (`talker.*`, `code2wav.*`) is dropped before
/// remap so its arrays are freed; its absence (thinker-only exports) is fine.
/// Sub-configs live under `thinker_config`, as do the multimodal token ids.
pub(crate) fn load_qwen3_omni_moe(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let thinker = full_config
        .get("thinker_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing thinker_config in Qwen3-Omni config.json"))?;

    // Sub-configs carry explicit nulls (e.g. `num_position_embeddings: null`);
    // strip them so serde defaults apply.
    fn strip_nulls(value: &mut serde_json::Value) {
        if let Some(map) = value.as_object_mut() {
            map.retain(|_, v| !v.is_null());
            for v in map.values_mut() {
                strip_nulls(v);
            }
        }
    }
    let subconfig = |key: &str| -> Result<serde_json::Value> {
        let mut v = thinker
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing thinker_config.{key} in config.json"))?;
        strip_nulls(&mut v);
        // HF exports carry both `in_channels` and its alias `in_chans`;
        // serde rejects the pair as a duplicate field, so drop the alias.
        if let Some(map) = v.as_object_mut()
            && map.contains_key("in_channels")
        {
            map.remove("in_chans");
        }
        Ok(v)
    };

    let mut text_config: models::qwen3_vl_moe::Qwen3VLMoeConfig =
        serde_json::from_value(subconfig("text_config")?)
            .map_err(|e| anyhow::anyhow!("Failed to parse Qwen3-Omni text config: {}", e))?;
    inherit_qwen_text_quantization(&mut text_config, &full_config);

    let mut vision_config: Qwen3VLVisionConfig =
        serde_json::from_value(subconfig("vision_config")?)
            .map_err(|e| anyhow::anyhow!("Failed to parse Qwen3-Omni vision config: {}", e))?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let audio_config: crate::audio::qwen3_omni_moe::Qwen3OmniAudioConfig =
        serde_json::from_value(subconfig("audio_config")?)
            .map_err(|e| anyhow::anyhow!("Failed to parse Qwen3-Omni audio config: {}", e))?;

    // Drop the speech-output stack first so its arrays are freed, then remap
    // the thinker prefixes (converted and raw spellings).
    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let mut weights = mlxcel_core::weights::WeightMap::new();
    for (key, value) in raw_weights {
        if key.starts_with("talker.") || key.starts_with("code2wav.") {
            continue;
        }
        // The sinusoidal audio position table is computed at construction.
        if key.ends_with("audio_tower.positional_embedding") {
            continue;
        }
        let new_key = if let Some(rest) = key.strip_prefix("thinker.language_model.") {
            rest.to_string()
        } else if let Some(rest) = key.strip_prefix("thinker.vision_tower.") {
            format!("vision_tower.{rest}")
        } else if let Some(rest) = key.strip_prefix("thinker.visual.") {
            format!("vision_tower.{rest}")
        } else if let Some(rest) = key.strip_prefix("thinker.audio_tower.") {
            format!("audio_tower.{rest}")
        } else if let Some(rest) = key.strip_prefix("thinker.") {
            rest.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Qwen3VLMoeModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen3-Omni text model: {}", e))?;

    let vision_encoder =
        Qwen3VLVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Qwen3-Omni vision encoder: {}", e))?;

    let audio_encoder = crate::audio::qwen3_omni_moe::Qwen3OmniAudioEncoder::from_weights(
        &weights,
        &audio_config,
        "audio_tower",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Qwen3-Omni audio tower: {}", e))?;

    let processor = qwen_vl_processor_with_norm(&vision_config, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);

    let id = |key: &str, default: i64| -> i32 {
        thinker.get(key).and_then(|v| v.as_i64()).unwrap_or(default) as i32
    };

    let vlm = vision::qwen3_omni_moe::Qwen3OmniMoeModel {
        text_model,
        vision_encoder,
        audio_encoder,
        processor,
        image_token_id: id("image_token_id", 151_655),
        video_token_id: id("video_token_id", 151_656),
        vision_start_token_id: id("vision_start_token_id", 151_652),
        audio_token_id: id("audio_token_id", 151_675),
        audio_start_token_id: id("audio_start_token_id", 151_669),
        audio_end_token_id: id("audio_end_token_id", 151_670),
        im_end_token_id: full_config
            .get("im_end_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(151_645) as i32,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::Qwen3OmniMoe(vlm))
}

#[cfg(test)]
mod glm_ocr_loader_tests {
    use super::{drop_extra_text_layers, normalize_glm_ocr_rope};
    use crate::models::glm4v::Glm4vTextConfig;

    #[test]
    fn normalize_lifts_rope_parameters_into_config_fields() {
        // GLM-OCR nests everything under `rope_parameters` with no top-level
        // `rope_scaling` / `partial_rotary_factor` / `rope_theta`.
        let mut cfg = serde_json::json!({
            "model_type": "glm_ocr_text",
            "hidden_size": 1536,
            "num_hidden_layers": 16,
            "intermediate_size": 4608,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 59392,
            "head_dim": 128,
            "rms_norm_eps": 1e-05,
            "eos_token_id": [59246, 59253],
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000
            }
        });
        normalize_glm_ocr_rope(&mut cfg).unwrap();
        let parsed: Glm4vTextConfig = serde_json::from_value(cfg).unwrap();
        assert_eq!(parsed.partial_rotary_factor, 1.0);
        assert_eq!(parsed.rope_theta, 10000.0);
        assert_eq!(
            parsed.rope_scaling.as_ref().unwrap().mrope_section,
            vec![16, 24, 24]
        );
        assert_eq!(parsed.eos_token_id, Some(vec![59246, 59253]));
    }

    #[test]
    fn normalize_leaves_glm4v_style_config_untouched() {
        // A glm4v config carries `rope_scaling` and top-level scalars; with no
        // `rope_parameters` present, normalization is a no-op.
        let mut cfg = serde_json::json!({
            "hidden_size": 4096,
            "num_hidden_layers": 40,
            "intermediate_size": 13696,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "vocab_size": 151552,
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
            "rope_scaling": {"mrope_section": [8, 12, 12]}
        });
        normalize_glm_ocr_rope(&mut cfg).unwrap();
        let parsed: Glm4vTextConfig = serde_json::from_value(cfg).unwrap();
        assert_eq!(parsed.partial_rotary_factor, 0.5);
        assert_eq!(
            parsed.rope_scaling.as_ref().unwrap().mrope_section,
            vec![8, 12, 12]
        );
    }

    #[test]
    fn drop_extra_text_layers_removes_mtp_block() {
        let dummy = || mlxcel_core::from_slice_f32(&[0.0], &[1]);
        let mut weights = mlxcel_core::weights::WeightMap::new();
        for i in 0..16 {
            weights.insert(format!("model.layers.{i}.input_layernorm.weight"), dummy());
        }
        // Next-n prediction block at layer index 16 (num_hidden_layers = 16).
        weights.insert("model.layers.16.eh_proj.weight".to_string(), dummy());
        weights.insert(
            "model.layers.16.shared_head.head.weight".to_string(),
            dummy(),
        );
        weights.insert(
            "model.layers.16.input_layernorm.weight".to_string(),
            dummy(),
        );
        weights.insert("model.embed_tokens.weight".to_string(), dummy());

        drop_extra_text_layers(&mut weights, 16);

        assert!(!weights.keys().any(|k| k.starts_with("model.layers.16.")));
        assert!(weights.contains_key("model.layers.15.input_layernorm.weight"));
        assert!(weights.contains_key("model.embed_tokens.weight"));
        assert_eq!(
            weights
                .keys()
                .filter(|k| k.starts_with("model.layers."))
                .count(),
            16
        );
    }
}
