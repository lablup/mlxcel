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
