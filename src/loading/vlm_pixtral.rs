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

//! Pixtral-family VLM loaders.
//!
//! Families:
//! - Pixtral
//! - Mistral3 VLM
//!
//! These families share Pixtral vision-tower assembly and Mistral-compatible
//! text-config shaping, so the family-specific normalization stays here.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::llava::infer_llama_config_from_weights;
use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};
use crate::model_metadata::is_mistral4_config;

struct PixtralFamilyContext {
    full_config: Value,
    weights: WeightMap,
    text_model: LoadedModel,
    text_hidden_size: usize,
    hidden_size: usize,
    image_token_id: i32,
    pad_token_id: i32,
    quant_group_size: i32,
    quant_bits: i32,
    pixtral_config: vision::encoders::pixtral::PixtralVisionConfig,
    vision_feature_layer: i32,
    rms_norm_eps: f32,
}

fn inherit_quantization_if_missing(
    text_config_value: &mut Value,
    full_config: &Value,
) -> Result<()> {
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(text_config_value, "Pixtral/Mistral3 text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    Ok(())
}

pub(super) fn apply_mistral_attention_head_override(
    text_config_value: &mut Value,
    weights: &WeightMap,
) {
    if let Some(obj) = text_config_value.as_object_mut() {
        let head_dim = obj.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
        if let Some(w) = weights
            .get("model.layers.0.self_attn.q_proj.scales")
            .or_else(|| weights.get("model.layers.0.self_attn.q_proj.weight"))
        {
            let shape = mlxcel_core::array_shape(w);
            if !shape.is_empty() {
                let q_out = shape[0] as usize;
                obj.insert(
                    "num_attention_heads".to_string(),
                    Value::Number(serde_json::Number::from(q_out / head_dim)),
                );
            }
        }
    }
}

pub(super) fn build_mistral_text_config(full_config: &Value, weights: &WeightMap) -> Result<Value> {
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    infer_llama_config_from_weights(&mut text_config_value, weights);
    apply_mistral_attention_head_override(&mut text_config_value, weights);
    inherit_quantization_if_missing(&mut text_config_value, full_config)?;
    Ok(text_config_value)
}

fn build_pixtral_family_context(model_path: &Path) -> Result<PixtralFamilyContext> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    vision::encoders::pixtral::sanitize_pixtral_weights(&mut weights);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // `text_config_value` is the Llama-shaped view of the backbone config. It
    // is only used below for scalar context fields (hidden_size, rms_norm_eps);
    // those keys come straight from text_config and survive the Llama massaging
    // (infer_llama_config_from_weights only fills missing fields and never
    // touches rms_norm_eps; apply_mistral_attention_head_override is a no-op
    // when q_proj weights are absent), so it stays correct on both branches.
    let text_config_value = build_mistral_text_config(&full_config, &weights)?;

    // The Mistral3 VLM container wraps either a standard Llama/Mistral text
    // backbone (q_proj) or a `mistral4` MLA + MoE backbone (q_a_proj /
    // kv_a_proj_with_mqa / kv_b_proj, no q_proj). Route the MLA case to the
    // Mistral4 loader, mirroring the text-only `load_llama_family_from_weights`
    // path in `src/loading/mod.rs`; the Llama loader cannot find `q_proj` for an
    // MLA backbone and fails with a weight-not-found error. The
    // `language_model.` prefix is already stripped above, so the weights sit at
    // the bare `model.layers...` namespace both loaders expect. See
    // lablup/mlxcel#423.
    let text_model = if is_mistral4_config(&full_config) {
        // Build the Mistral4 args from the RAW text_config, not the
        // Llama-massaged `text_config_value`: infer_llama_config_from_weights and
        // apply_mistral_attention_head_override key off `q_proj` and can corrupt
        // the MLA head layout. Inherit the top-level quantization block when
        // text_config omits it so group_size/bits are present.
        let mut mistral4_text_config = full_config
            .get("text_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing text_config for Mistral4 VLM backbone"))?;
        inherit_quantization_if_missing(&mut mistral4_text_config, &full_config)?;
        let text_args: models::mistral4::Mistral4Config =
            serde_json::from_value(mistral4_text_config)
                .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Mistral4: {}", e))?;
        models::mistral4::sanitize_weights(&mut weights, &text_args, "");
        models::Mistral4Model::from_weights(&weights, &text_args)
            .map(LoadedModel::Mistral4)
            .map_err(|e| anyhow::anyhow!("Failed to load Mistral4 text model: {}", e))?
    } else {
        let text_args: models::llama3::ModelArgs =
            serde_json::from_value(text_config_value.clone())
                .map_err(|e| anyhow::anyhow!("Failed to parse text_config as Mistral: {}", e))?;
        models::Llama3Model::from_weights(&weights, &text_args)
            .map(LoadedModel::Llama)
            .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?
    };

    let vision_config_value = full_config
        .get("vision_config")
        .ok_or_else(|| anyhow::anyhow!("vision_config not found in config.json"))?;
    let pixtral_config =
        vision::encoders::pixtral::PixtralVisionConfig::from_json(vision_config_value);

    Ok(PixtralFamilyContext {
        text_hidden_size: text_config_value
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize,
        hidden_size: full_config
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        image_token_id: full_config
            .get("image_token_index")
            .or_else(|| full_config.get("image_token_id"))
            .and_then(|v| v.as_i64())
            .unwrap_or(10) as i32,
        pad_token_id: full_config
            .get("pad_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        quant_group_size: full_config
            .get("quantization")
            .and_then(|q| q.get("group_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(64) as i32,
        quant_bits: full_config
            .get("quantization")
            .and_then(|q| q.get("bits"))
            .and_then(|v| v.as_i64())
            .unwrap_or(4) as i32,
        vision_feature_layer: full_config
            .get("vision_feature_layer")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1) as i32,
        rms_norm_eps: text_config_value
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32,
        full_config,
        weights,
        text_model,
        pixtral_config,
    })
}

/// Load a Pixtral VLM model (Pixtral ViT with 2D RoPE + Mistral text + MLP projector).
pub(crate) fn load_pixtral_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::mlp::MLPProjector;
    use vision::encoders::pixtral::PixtralVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    let context = build_pixtral_family_context(model_path)?;

    let vision_encoder = PixtralVisionModel::from_weights(
        &context.weights,
        &context.pixtral_config,
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Pixtral vision encoder: {}", e))?
    .with_feature_layer(context.vision_feature_layer);

    let connector = MLPProjector::from_weights(
        &context.weights,
        "multi_modal_projector",
        context.quant_group_size,
        context.quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MLP projector: {}", e))?;

    let processor = SigLipProcessor::new_rescale_only(context.pixtral_config.image_size);
    let num_patches =
        (context.pixtral_config.image_size / context.pixtral_config.patch_size).pow(2);
    let mm_tokens_per_image = context
        .full_config
        .get("mm_tokens_per_image")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_patches as u64) as usize;

    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: context.image_token_id,
        pad_token_id: context.pad_token_id,
        hidden_size: if context.hidden_size > 0 {
            context.hidden_size
        } else {
            context.text_hidden_size
        },
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(context.text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

/// Load a Mistral 3 VLM model (Pixtral ViT + PatchMerger projector + Mistral text model).
pub(crate) fn load_mistral3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::mistral3::Mistral3Projector;
    use vision::encoders::pixtral::PixtralVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    let context = build_pixtral_family_context(model_path)?;

    let vision_encoder = PixtralVisionModel::from_weights(
        &context.weights,
        &context.pixtral_config,
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Pixtral vision encoder: {}", e))?
    .with_feature_layer(context.vision_feature_layer);

    let patch_h = (context.pixtral_config.image_size / context.pixtral_config.patch_size) as i32;
    let spatial_merge_size = context
        .full_config
        .get("spatial_merge_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(2) as i32;

    let connector = Mistral3Projector::from_weights(
        &context.weights,
        "multi_modal_projector",
        context.quant_group_size,
        context.quant_bits,
        patch_h,
        spatial_merge_size,
        context.rms_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Mistral3 projector: {}", e))?;

    let processor = SigLipProcessor::new_rescale_only(context.pixtral_config.image_size);
    let num_patches_per_side =
        context.pixtral_config.image_size / context.pixtral_config.patch_size;
    let merged_per_side = num_patches_per_side / spatial_merge_size as usize;
    let mm_tokens_per_image = context
        .full_config
        .get("mm_tokens_per_image")
        .and_then(|v| v.as_u64())
        .unwrap_or((merged_per_side * merged_per_side) as u64)
        as usize;

    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: context.image_token_id,
        pad_token_id: context.pad_token_id,
        hidden_size: if context.hidden_size > 0 {
            context.hidden_size
        } else {
            context.text_hidden_size
        },
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(context.text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

#[cfg(test)]
#[path = "vlm_pixtral_tests.rs"]
mod tests;
