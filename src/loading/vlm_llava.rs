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

//! LLaVA-style VLM loaders.
//!
//! Families:
//! - LLaVA
//! - LLaVA-Bunny
//!
//! These families share SigLIP/CLIP-style vision towers plus text-backend
//! selection and projector wiring, so their config normalization lives here.

use anyhow::Result;
use mlxcel_core::layers::UnifiedEmbedding;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::multimodal::host_preprocessor::{HostPreprocessorError, LlavaHostPreprocessor};
use crate::vision;
use crate::vision::config::VLMConfig;

use super::{
    load_vlm_weights_common, load_vlm_weights_common_filtered_canonical, parse_vlm_config,
    read_sanitized_vlm_config, strip_language_model_prefix,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlavaTextBackend {
    Llama,
    Qwen2,
}

/// Infer missing Llama config fields from weight shapes.
///
/// LLaVA-derived text configs may be minimal. We can infer:
/// - hidden_size from model.norm.weight shape
/// - num_hidden_layers from counting model.layers.N keys
/// - intermediate_size from model.layers.0.mlp.gate_proj shape
/// - num_attention_heads from hidden_size / head_dim (head_dim=128 typical)
pub(super) fn infer_llama_config_from_weights(config: &mut Value, weights: &WeightMap) {
    let obj = match config.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    if obj.get("hidden_size").and_then(|v| v.as_u64()).is_none()
        && let Some(w) = weights.get("model.norm.weight")
    {
        let shape = mlxcel_core::array_shape(w);
        if !shape.is_empty() {
            obj.insert(
                "hidden_size".to_string(),
                Value::Number(serde_json::Number::from(shape[0] as u64)),
            );
        }
    }

    if obj
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .is_none()
    {
        let max_layer = weights
            .keys()
            .filter_map(|k| {
                k.strip_prefix("model.layers.")
                    .and_then(|rest| rest.split('.').next())
                    .and_then(|n| n.parse::<usize>().ok())
            })
            .max()
            .unwrap_or(0);
        obj.insert(
            "num_hidden_layers".to_string(),
            Value::Number(serde_json::Number::from(max_layer + 1)),
        );
    }

    if obj
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .is_none()
        && let Some(w) = weights
            .get("model.layers.0.mlp.gate_proj.scales")
            .or_else(|| weights.get("model.layers.0.mlp.gate_proj.weight"))
    {
        let shape = mlxcel_core::array_shape(w);
        if !shape.is_empty() {
            obj.insert(
                "intermediate_size".to_string(),
                Value::Number(serde_json::Number::from(shape[0] as u64)),
            );
        }
    }

    if obj
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .is_none()
        && let Some(hidden) = obj.get("hidden_size").and_then(|v| v.as_u64())
    {
        obj.insert(
            "num_attention_heads".to_string(),
            Value::Number(serde_json::Number::from(hidden / 128)),
        );
    }
}

fn llava_text_backend(model_type: &str, family_label: &str) -> Result<LlavaTextBackend> {
    match model_type {
        "llama" | "mistral" => Ok(LlavaTextBackend::Llama),
        "qwen2" => Ok(LlavaTextBackend::Qwen2),
        _ => Err(anyhow::anyhow!(
            "Unsupported {} text backend: {}",
            family_label,
            model_type
        )),
    }
}

fn detect_bunny_text_backend(full_config: &Value) -> LlavaTextBackend {
    let text_config = full_config.get("text_config").cloned().unwrap_or_default();
    text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| full_config.get("model_type").and_then(|v| v.as_str()))
        .and_then(|model_type| {
            if model_type.contains("llama") || model_type.contains("mistral") {
                Some(LlavaTextBackend::Llama)
            } else if model_type.contains("qwen") {
                Some(LlavaTextBackend::Qwen2)
            } else {
                None
            }
        })
        .unwrap_or(LlavaTextBackend::Llama)
}

fn inherit_text_quantization_if_missing(
    text_config_value: &mut Value,
    full_config: &Value,
) -> Result<()> {
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(text_config_value, "LLaVA text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    Ok(())
}

fn build_llava_text_model(
    weights: &WeightMap,
    text_config_value: &Value,
    backend: LlavaTextBackend,
    family_label: &str,
) -> Result<LoadedModel> {
    let text_args: models::llama3::ModelArgs = serde_json::from_value(text_config_value.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse {} text_config: {}", family_label, e))?;

    match backend {
        LlavaTextBackend::Llama => {
            let model = models::Llama3Model::from_weights(weights, &text_args)
                .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;
            Ok(LoadedModel::Llama(model))
        }
        LlavaTextBackend::Qwen2 => {
            let model = models::Qwen2Model::from_weights(weights, &text_args)
                .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;
            Ok(LoadedModel::Qwen2(model))
        }
    }
}

fn clip_processor(image_size: usize) -> vision::processors::siglip::SigLipProcessor {
    vision::processors::siglip::SigLipProcessor {
        image_size,
        mean: [0.48145466, 0.4578275, 0.40821073],
        std: [0.26862954, 0.261_302_6, 0.275_777_1],
        do_normalize: true,
    }
}

fn llava_processor(
    vision_model_type: &str,
    image_size: usize,
) -> vision::processors::siglip::SigLipProcessor {
    if vision_model_type.contains("clip") {
        clip_processor(image_size)
    } else {
        vision::processors::siglip::SigLipProcessor::new(image_size)
    }
}

/// Whether a raw checkpoint tensor belongs to the host-first LLaVA prefix.
///
/// The optional `language_model.` wrapper is normalized by
/// [`strip_language_model_prefix`] after loading. Decoder layers, final norm,
/// and LM head deliberately fail this predicate, so they are never retained by
/// the host preprocessor.
pub(super) fn is_llava_host_preprocessor_weight(name: &str) -> bool {
    let canonical = name.strip_prefix("language_model.").unwrap_or(name);
    canonical.starts_with("vision_tower.")
        || canonical.starts_with("multi_modal_projector.")
        || canonical.starts_with("model.embed_tokens.")
}

/// Widen only the host vision tower and projector weights used by the
/// qualified XLA LLaVA path.
///
/// The checkpoint remains immutable BF16 on disk and the text embedding table
/// stays at its checkpoint dtype. MLX otherwise selects BF16 matmul output when
/// either a vision weight or activation is BF16, so widening only the input
/// pixels is insufficient to prevent layer-by-layer reduction drift.
fn widen_llava_host_vision_weights(weights: &mut WeightMap) {
    for (name, tensor) in weights.iter_mut() {
        if !(name.starts_with("vision_tower.") || name.starts_with("multi_modal_projector.")) {
            continue;
        }
        let dtype = mlxcel_core::array_dtype(tensor);
        if dtype == mlxcel_core::dtype::BFLOAT16 || dtype == mlxcel_core::dtype::FLOAT16 {
            *tensor = mlxcel_core::astype(tensor, mlxcel_core::dtype::FLOAT32);
        }
    }
}

fn require_llava_host_family(config: &VLMConfig) -> Result<(), HostPreprocessorError> {
    if !matches!(config.model_type.as_str(), "llava" | "llava_next") {
        return Err(HostPreprocessorError::FamilyMismatch {
            actual: config.model_type.clone(),
        });
    }
    if !(config.vision_config.model_type.contains("clip")
        || config.vision_config.model_type.contains("siglip"))
    {
        return Err(HostPreprocessorError::FamilyMismatch {
            actual: format!(
                "{} with vision tower {}",
                config.model_type, config.vision_config.model_type
            ),
        });
    }
    if config.vision_config.num_channels != 3 {
        return Err(HostPreprocessorError::InvalidConfig(format!(
            "LLaVA host preprocessing supports RGB vision towers only, got {} channels",
            config.vision_config.num_channels
        )));
    }
    if config.vision_config.patch_size == 0 {
        return Err(HostPreprocessorError::InvalidConfig(format!(
            "LLaVA patch_size must be non-zero for image_size {}",
            config.vision_config.image_size
        )));
    }
    Ok(())
}

/// Load only the host-side components needed to prepare LLaVA embeddings for
/// XLA. The full text decoder is neither constructed nor retained.
pub(crate) fn load_llava_host_preprocessor(
    model_path: &Path,
) -> Result<LlavaHostPreprocessor, HostPreprocessorError> {
    use vision::connectors::mlp::MLPProjector;
    use vision::encoders::siglip::SigLipVisionModel;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
    let vlm_config: VLMConfig = parse_vlm_config(&config_str, "VLM config")
        .map_err(|error| HostPreprocessorError::InvalidConfig(error.to_string()))?;
    require_llava_host_family(&vlm_config)?;

    let text_model_type = vlm_config
        .text_config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("llama");
    llava_text_backend(text_model_type, "LLaVA host preprocessor").map_err(|error| {
        HostPreprocessorError::FamilyMismatch {
            actual: error.to_string(),
        }
    })?;

    let mut weights = load_vlm_weights_common_filtered_canonical(model_path, |name| {
        is_llava_host_preprocessor_weight(name)
    })
    .map(strip_language_model_prefix)
    .map_err(|error| HostPreprocessorError::WeightLoad(error.to_string()))?;
    widen_llava_host_vision_weights(&mut weights);

    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(Value::as_i64)
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(Value::as_i64)
        .unwrap_or(4) as i32;

    let text_embeddings = UnifiedEmbedding::from_weights(
        &weights,
        "model.embed_tokens",
        quant_group_size,
        quant_bits,
    )
    .map_err(|error| {
        HostPreprocessorError::WeightLoad(format!(
            "missing or invalid LLaVA text embedding table: {error}"
        ))
    })?;
    let connector = MLPProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|error| {
        HostPreprocessorError::WeightLoad(format!(
            "missing or invalid LLaVA multi_modal_projector weights: {error}"
        ))
    })?;
    let encoder = SigLipVisionModel::from_weights(
        &weights,
        &vlm_config.vision_config,
        "vision_tower.vision_model",
    )
    .map_err(|error| {
        HostPreprocessorError::WeightLoad(format!(
            "missing or invalid LLaVA vision_tower weights: {error}"
        ))
    })?
    .with_feature_selection(
        vlm_config.vision_feature_layer,
        vlm_config.vision_feature_select_strategy.clone(),
    );

    let text_hidden_size = vlm_config
        .text_config
        .get("hidden_size")
        .and_then(Value::as_u64)
        .or_else(|| full_config.get("hidden_size").and_then(Value::as_u64))
        .ok_or_else(|| {
            HostPreprocessorError::InvalidConfig(
                "LLaVA text_config.hidden_size is required for host preprocessing".to_string(),
            )
        })? as usize;
    let max_sequence_len = vlm_config
        .text_config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .or_else(|| {
            full_config
                .get("max_position_embeddings")
                .and_then(Value::as_u64)
        })
        .unwrap_or(4096) as usize;
    let num_patches =
        (vlm_config.vision_config.image_size / vlm_config.vision_config.patch_size).pow(2);
    let tokens_per_image = vlm_config.mm_tokens_per_image.unwrap_or(num_patches);
    let processor = llava_processor(
        &vlm_config.vision_config.model_type,
        vlm_config.vision_config.image_size,
    );

    LlavaHostPreprocessor::from_parts(
        Box::new(processor),
        Box::new(encoder),
        Box::new(connector),
        text_embeddings,
        vlm_config.image_token_index,
        tokens_per_image,
        text_hidden_size,
        vlm_config.vision_config.image_size,
        max_sequence_len,
    )
}

fn bunny_processor(
    mm_vision_tower: &str,
    image_size: usize,
) -> vision::processors::siglip::SigLipProcessor {
    if mm_vision_tower.contains("clip") && !mm_vision_tower.contains("siglip") {
        clip_processor(image_size)
    } else {
        vision::processors::siglip::SigLipProcessor::new(image_size)
    }
}

fn rewrite_bunny_weight_key(key: &str) -> Option<String> {
    if let Some(rest) = key.strip_prefix("vision_tower.vision_tower.") {
        Some(format!("vision_tower.{}", rest))
    } else if let Some(rest) = key.strip_prefix("mm_projector.") {
        Some(format!("multi_modal_projector.{}", rest))
    } else {
        key.strip_prefix("model.lm_head.")
            .map(|rest| format!("lm_head.{}", rest))
    }
}

fn sanitize_bunny_weight_keys(weights: &mut WeightMap) {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        if let Some(new_key) = rewrite_bunny_weight_key(&key)
            && let Some(value) = weights.remove(&key)
        {
            weights.insert(new_key, value);
        }
    }
}

fn parse_bunny_vision_config(
    full_config: &Value,
    vision_config_val: &Value,
) -> vision::config::VisionConfig {
    serde_json::from_value(vision_config_val.clone()).unwrap_or_else(|_| {
        let mm_hidden = full_config
            .get("mm_hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1152) as usize;
        let intermediate = vision_config_val
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4304) as usize;
        vision::config::VisionConfig {
            model_type: "siglip_vision_model".to_string(),
            num_hidden_layers: 27,
            hidden_size: mm_hidden,
            intermediate_size: intermediate,
            num_attention_heads: 16,
            patch_size: 14,
            image_size: 384,
            num_channels: 3,
            layer_norm_eps: 1e-6,
            hidden_act: vision::config::VisionHiddenActivation::ExactGelu,
        }
    })
}

/// Load a LLaVA VLM model (CLIP/SigLIP vision + Llama/Qwen2 text + MLP projector).
pub(crate) fn load_llava_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::config::VLMConfig;
    use vision::connectors::mlp::MLPProjector;
    use vision::encoders::siglip::SigLipVisionModel;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let vlm_config: VLMConfig = parse_vlm_config(&config_str, "VLM config")?;

    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);

    let text_model_type = vlm_config
        .text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");
    let backend = llava_text_backend(text_model_type, "LLaVA")?;

    let mut text_config_value = vlm_config.text_config.clone();
    infer_llama_config_from_weights(&mut text_config_value, &weights);
    inherit_text_quantization_if_missing(&mut text_config_value, &full_config)?;

    models::sanitize_tied_embeddings(&mut weights, &full_config);
    let text_model = build_llava_text_model(&weights, &text_config_value, backend, "LLaVA")?;

    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;

    let vision_feature_layer = vlm_config.vision_feature_layer;
    let vision_feature_select_strategy = vlm_config.vision_feature_select_strategy.clone();

    let vision_encoder = SigLipVisionModel::from_weights(
        &weights,
        &vlm_config.vision_config,
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?
    .with_feature_selection(vision_feature_layer, vision_feature_select_strategy);

    let connector = MLPProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MLP projector: {}", e))?;

    let processor = llava_processor(
        vlm_config.vision_config.model_type.as_str(),
        vlm_config.vision_config.image_size,
    );

    let num_patches =
        (vlm_config.vision_config.image_size / vlm_config.vision_config.patch_size).pow(2);
    let mm_tokens_per_image = vlm_config.mm_tokens_per_image.unwrap_or(num_patches);
    let text_hidden_size = text_config_value
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;

    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: vlm_config.image_token_index,
        pad_token_id: vlm_config.pad_token_id,
        hidden_size: if vlm_config.hidden_size > 0 {
            vlm_config.hidden_size
        } else {
            text_hidden_size
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
        pixtral_layout: None,
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

/// Load a LLaVA-Bunny VLM model (SigLIP + MLP projector + Qwen2/Llama text).
pub(crate) fn load_llava_bunny_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::mlp::MLPProjector;
    use vision::encoders::siglip::SigLipVisionModel;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    sanitize_bunny_weight_keys(&mut weights);

    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_config_val = full_config.get("text_config").cloned().unwrap_or_default();
    let mut text_config_value = if text_config_val.is_object()
        && text_config_val.as_object().map(|o| o.len()).unwrap_or(0) > 0
    {
        text_config_val.clone()
    } else {
        full_config.clone()
    };
    infer_llama_config_from_weights(&mut text_config_value, &weights);
    inherit_text_quantization_if_missing(&mut text_config_value, &full_config)?;

    let text_model = build_llava_text_model(
        &weights,
        &text_config_value,
        detect_bunny_text_backend(&full_config),
        "LLaVA-Bunny",
    )?;

    let quant_group_size = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let quant_bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;

    let vision_config_val = full_config
        .get("vision_config")
        .cloned()
        .unwrap_or_default();
    let vision_config = parse_bunny_vision_config(&full_config, &vision_config_val);

    let vision_feature_layer = full_config
        .get("vision_feature_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-2) as i32;
    let vision_feature_select_strategy = full_config
        .get("vision_feature_select_strategy")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    let vision_encoder = SigLipVisionModel::from_weights_with_quant(
        &weights,
        &vision_config,
        "vision_tower.vision_model",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?
    .with_feature_selection(vision_feature_layer, vision_feature_select_strategy);

    let connector = MLPProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MLP projector: {}", e))?;

    let mm_vision_tower = full_config
        .get("mm_vision_tower")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let processor = bunny_processor(mm_vision_tower, vision_config.image_size);

    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
    let mm_tokens_per_image = full_config
        .get("mm_tokens_per_image")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_patches as u64) as usize;
    let text_hidden_size = text_config_value
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;
    let image_token_index = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(-200) as i32;
    let pad_token_id = full_config
        .get("pad_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: image_token_index,
        pad_token_id,
        hidden_size: text_hidden_size,
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
        text_model: Box::new(text_model),
        vision: vision_module,
    };

    Ok(LoadedModel::LlavaVLM(vlm))
}

#[cfg(test)]
#[path = "vlm_llava_tests.rs"]
mod tests;
