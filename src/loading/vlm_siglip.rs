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

//! SigLIP-backed VLM loaders.
//!
//! Families:
//! - Aya Vision
//! - PaliGemma
//!
//! These models share a SigLIP-family vision tower but diverge in text-backend
//! defaults and projector wiring, so their config normalization is grouped here.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

fn quantization_params(full_config: &Value) -> (i32, i32) {
    let group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;
    (group_size, bits)
}

fn resolve_image_token_id(full_config: &Value, default: i32) -> i32 {
    full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(default as i64) as i32
}

fn inherit_quantization_if_missing(text_config: &mut Value, full_config: &Value) -> Result<()> {
    if text_config.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(text_config, "SigLIP text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    Ok(())
}

fn rewrite_aya_weight_key(key: &str) -> Option<String> {
    if let Some(rest) = key.strip_prefix("model.vision_tower.") {
        Some(format!("vision_tower.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.multi_modal_projector.") {
        Some(format!("multi_modal_projector.{}", rest))
    } else {
        key.strip_prefix("model.language_model.")
            .map(|rest| format!("model.{}", rest))
    }
}

fn sanitize_aya_weights(weights: &mut WeightMap) {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        if let Some(new_key) = rewrite_aya_weight_key(&key)
            && let Some(value) = weights.remove(&key)
        {
            weights.insert(new_key, value);
        }
    }
}

fn inject_aya_text_defaults(text_config: &mut Value, weights: &WeightMap) -> Result<()> {
    let obj = super::require_object_mut(text_config, "Aya Vision text_config")?;
    if !obj.contains_key("vocab_size") {
        obj.insert("vocab_size".to_string(), serde_json::json!(256000));
    }
    if !obj.contains_key("layer_norm_eps") {
        obj.insert("layer_norm_eps".to_string(), serde_json::json!(1e-5));
    }
    if !obj.contains_key("head_dim") {
        let hidden = obj
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096);
        let heads = obj
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        obj.insert("head_dim".to_string(), serde_json::json!(hidden / heads));
    }
    if !obj.contains_key("sliding_window") {
        obj.insert("sliding_window".to_string(), serde_json::json!(4096));
    }
    if !obj.contains_key("tie_word_embeddings") && !weights.contains_key("lm_head.weight") {
        obj.insert("tie_word_embeddings".to_string(), serde_json::json!(true));
    }
    Ok(())
}

fn rewrite_paligemma_weight_key(key: &str) -> Option<String> {
    key.strip_prefix("multi_modal_projector.linear.")
        .map(|rest| format!("multi_modal_projector.linear_1.{}", rest))
}

fn sanitize_paligemma_weights(weights: &mut WeightMap) {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        if let Some(new_key) = rewrite_paligemma_weight_key(&key)
            && let Some(value) = weights.remove(&key)
        {
            weights.insert(new_key, value);
        }
    }
}

fn inject_paligemma_text_defaults(text_config: &mut Value) -> Result<()> {
    let obj = super::require_object_mut(text_config, "PaliGemma text_config")?;
    if !obj.contains_key("rms_norm_eps") {
        obj.insert("rms_norm_eps".to_string(), serde_json::json!(1e-6));
    }
    if !obj.contains_key("head_dim") {
        let default_head_dim = obj
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_u64())
            .unwrap_or(256);
        obj.insert("head_dim".to_string(), serde_json::json!(default_head_dim));
    }
    Ok(())
}

fn build_paligemma_text_model(weights: &WeightMap, text_config: &Value) -> Result<LoadedModel> {
    match text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("gemma")
    {
        "gemma" => {
            let text_args: models::gemma::ModelArgs =
                serde_json::from_value(text_config.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Gemma: {}", e))?;
            let model = models::GemmaModel::from_weights(weights, &text_args)
                .map_err(|e| anyhow::anyhow!("Failed to load Gemma text model: {}", e))?;
            Ok(LoadedModel::Gemma(model))
        }
        "gemma2" => {
            let text_args: models::gemma2::ModelArgs = serde_json::from_value(text_config.clone())
                .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Gemma2: {}", e))?;
            let model = models::Gemma2Model::from_weights(weights, &text_args)
                .map_err(|e| anyhow::anyhow!("Failed to load Gemma2 text model: {}", e))?;
            Ok(LoadedModel::Gemma2(model))
        }
        other => Err(anyhow::anyhow!(
            "Unsupported PaliGemma text backend: {}",
            other
        )),
    }
}

/// Load an Aya Vision VLM model (SigLIP + SwiGLU projector + Cohere2 text).
pub(crate) fn load_aya_vision_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::aya_vision::AyaVisionProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    sanitize_aya_weights(&mut weights);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let vision_config: vision::config::VisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config"))?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse vision_config: {}", e))?;

    let mut text_config = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config"))?;
    inherit_quantization_if_missing(&mut text_config, &full_config)?;
    inject_aya_text_defaults(&mut text_config, &weights)?;

    let text_args: models::cohere2::Cohere2Config = serde_json::from_value(text_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Cohere2: {}", e))?;
    let text_model = models::Cohere2Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Cohere2 text model: {}", e))?;

    let (quant_group_size, quant_bits) = quantization_params(&full_config);
    let vision_feature_layer = full_config
        .get("vision_feature_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let vision_feature_select_strategy = full_config
        .get("vision_feature_select_strategy")
        .and_then(|v| v.as_str())
        .unwrap_or("full")
        .to_string();

    let vision_encoder =
        SigLipVisionModel::from_weights(&weights, &vision_config, "vision_tower.vision_model")
            .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?
            .with_feature_selection(vision_feature_layer, vision_feature_select_strategy);

    let downsample_factor = full_config
        .get("downsample_factor")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    let adapter_layer_norm_eps = full_config
        .get("adapter_layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;

    let connector = AyaVisionProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
        downsample_factor,
        adapter_layer_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Aya Vision projector: {}", e))?;

    let processor = SigLipProcessor::new(vision_config.image_size);
    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
    let mm_tokens_per_image = num_patches / downsample_factor.pow(2);

    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: resolve_image_token_id(&full_config, 255036),
        pad_token_id: 0,
        hidden_size: text_args.hidden_size,
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
        pixtral_layout: None,
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Cohere2(text_model)),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

/// Load a PaliGemma VLM model (SigLIP + Linear projector + Gemma text).
pub(crate) fn load_paligemma_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::linear::LinearProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    sanitize_paligemma_weights(&mut weights);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let vision_config: vision::config::VisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config"))?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse vision_config: {}", e))?;

    let text_config_val = full_config
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("Missing text_config"))?;
    let mut text_config = text_config_val.clone();
    inherit_quantization_if_missing(&mut text_config, &full_config)?;
    inject_paligemma_text_defaults(&mut text_config)?;
    let text_model = build_paligemma_text_model(&weights, &text_config)?;

    let (quant_group_size, quant_bits) = quantization_params(&full_config);
    let vision_encoder = SigLipVisionModel::from_weights_with_quant(
        &weights,
        &vision_config,
        "vision_tower.vision_model",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?;

    let connector = LinearProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load linear projector: {}", e))?;

    let processor = SigLipProcessor::new(vision_config.image_size);
    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
    let text_hidden_size = full_config
        .get("hidden_size")
        .or_else(|| text_config_val.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;

    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: resolve_image_token_id(&full_config, 257152),
        pad_token_id: full_config
            .get("pad_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        hidden_size: text_hidden_size,
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image: num_patches,
        merge_strategy: vision::MergeStrategy::Gemma3,
        has_bos: false,              // PaliGemma: tokenizer has add_bos_token=false
        separator_token_id: Some(2), // BOS(2) between image and text tokens
        suffix_tokens: vec![108],    // newline after text prompt
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
        pixtral_layout: None,
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::Gemma3VLM(vlm))
}

#[cfg(test)]
#[path = "vlm_siglip_tests.rs"]
mod tests;
