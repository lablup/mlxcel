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

//! Special-case VLM loaders that do not fit another family bucket.
//!
//! Families:
//! - Llama4 VLM
//! - MiniCPM-o
//! - Moondream3
//! - Phi4MM
//! - Phi4-SigLIP
//! - Phi3V
//! - Molmo2
//!
//! These architectures need custom config shaping or weight remapping that is
//! distinct from the LLaVA/Qwen/Gemma/SigLIP families, so they are grouped here.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{
    load_vlm_weights_common, parse_required_vlm_subconfig, parse_vlm_config,
    read_optional_model_json, read_sanitized_vlm_config,
};

fn phi3_vision_config() -> vision::config::VisionConfig {
    vision::config::VisionConfig {
        model_type: "phi3_v".to_string(),
        hidden_size: 1024,
        image_size: 336,
        intermediate_size: 4096,
        num_attention_heads: 16,
        num_hidden_layers: 24,
        num_channels: 3,
        patch_size: 14,
        layer_norm_eps: 1e-5,
    }
}

const PHI4_FUSED_TEXT_FIELDS: &[&str] = &[
    "vocab_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "rms_norm_eps",
    "hidden_size",
    "num_key_value_heads",
    "rope_theta",
    "rope_traditional",
    "partial_rotary_factor",
    "rope_scaling",
    "model_type",
    "quantization",
    "tie_word_embeddings",
    "max_position_embeddings",
];

fn parse_quantization_params(full_config: &Value) -> (i32, i32) {
    let group_size = full_config
        .get("quantization")
        .and_then(|value| value.get("group_size"))
        .and_then(|value| value.as_i64())
        .unwrap_or(0) as i32;
    let bits = full_config
        .get("quantization")
        .and_then(|value| value.get("bits"))
        .and_then(|value| value.as_i64())
        .unwrap_or(0) as i32;
    (group_size, bits)
}

fn moondream3_bits(full_config: &Value) -> i32 {
    full_config
        .get("quantization_config")
        .and_then(|value| value.get("bits"))
        .and_then(|value| value.as_i64())
        .map(|value| value as i32)
        .or_else(|| {
            full_config
                .get("quantization_config")
                .and_then(|value| value.get("quant_method"))
                .and_then(|value| value.as_str())
                .and_then(|method| method.strip_prefix("int"))
                .and_then(|bits| bits.parse::<i32>().ok())
        })
        .unwrap_or(4)
}

fn moondream3_group_size(full_config: &Value, field: &str) -> i32 {
    full_config
        .get(field)
        .and_then(|value| value.as_i64())
        .unwrap_or(128) as i32
}

pub(super) fn moondream3_text_config_value(full_config: &Value) -> Value {
    let bits = moondream3_bits(full_config);
    serde_json::json!({
        "model_type": "moondream3",
        "dim": 2048,
        "ff_dim": 8192,
        "n_layers": 24,
        "vocab_size": 51200,
        "max_context": 4096,
        "n_heads": 32,
        "n_kv_heads": 32,
        "prefix_attn": 730,
        "group_size": moondream3_group_size(full_config, "text_group_size"),
        "bits": bits,
        "eos_token_id": 0,
        "moe": {
            "num_experts": 64,
            "start_layer": 4,
            "experts_per_token": 8,
            "expert_inner_dim": 1024,
            "expert_group_size": moondream3_group_size(full_config, "expert_group_size")
        }
    })
}

pub(super) fn moondream3_vision_config_value(_full_config: &Value) -> Value {
    serde_json::json!({
        "enc_dim": 1152,
        "enc_patch_size": 14,
        "enc_n_layers": 27,
        "enc_ff_dim": 4304,
        "enc_n_heads": 16,
        "proj_out_dim": 2048,
        "crop_size": 378,
        "in_channels": 3,
        "max_crops": 12,
        "overlap_margin": 4,
        "proj_inner_dim": 8192
    })
}

pub(super) fn rewrite_moondream3_weight_key(key: &str) -> Option<String> {
    if key.starts_with("model.region.") {
        return None;
    }

    let key = key.strip_prefix("model.").unwrap_or(key);
    if key == "text.wte" {
        Some("text.wte.weight".to_string())
    } else if let Some(prefix) = key.strip_suffix(".weight.packed") {
        Some(format!("{}.weight.packed", prefix))
    } else if let Some(prefix) = key.strip_suffix(".weight.scale") {
        Some(format!("{}.weight.scale", prefix))
    } else if let Some(prefix) = key.strip_suffix(".weight.zero_point") {
        Some(format!("{}.weight.zero_point", prefix))
    } else {
        Some(key.to_string())
    }
}

fn moondream3_qkv_dim(config: &models::moondream3::ModelArgs) -> i32 {
    let head_dim = (config.dim / config.n_heads) as i32;
    (config.n_heads as i32 + 2 * config.n_kv_heads as i32) * head_dim
}

fn moondream3_dense_weight_shape(
    prefix: &str,
    config: &models::moondream3::ModelArgs,
) -> Result<Vec<i32>> {
    if prefix.ends_with(".attn.qkv") {
        Ok(vec![moondream3_qkv_dim(config), config.dim as i32])
    } else if prefix.ends_with(".attn.proj") {
        Ok(vec![config.dim as i32, config.dim as i32])
    } else if prefix.ends_with(".mlp.fc1") {
        Ok(vec![config.ff_dim as i32, config.dim as i32])
    } else if prefix.ends_with(".mlp.fc2") {
        Ok(vec![config.dim as i32, config.ff_dim as i32])
    } else {
        Err(anyhow::anyhow!(
            "Unsupported Moondream3 dense packed weight prefix: {}",
            prefix
        ))
    }
}

fn moondream3_moe_weight_shape(
    prefix: &str,
    config: &models::moondream3::ModelArgs,
) -> Result<Vec<i32>> {
    let moe = config
        .moe
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Moondream3 MoE weight requires moe config"))?;
    let expert_count = moe.num_experts as i32;
    let dim = config.dim as i32;
    let expert_inner_dim = moe.expert_inner_dim as i32;

    if prefix.ends_with(".mlp.fc1") {
        Ok(vec![expert_count, expert_inner_dim * 2, dim])
    } else if prefix.ends_with(".mlp.fc2") {
        Ok(vec![expert_count, dim, expert_inner_dim])
    } else {
        Err(anyhow::anyhow!(
            "Unsupported Moondream3 MoE packed weight prefix: {}",
            prefix
        ))
    }
}

pub(super) fn dequantize_moondream3_weight(
    packed: &mlxcel_core::MlxArray,
    scale: &mlxcel_core::MlxArray,
    zero_point: &mlxcel_core::MlxArray,
    original_shape: &[i32],
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let packed_shape = mlxcel_core::array_shape(packed);
    let packed_i32 = mlxcel_core::astype(packed, mlxcel_core::dtype::INT32);
    let high_shift = mlxcel_core::from_slice_i32(&[4], &[1]);
    let nibble_mask = mlxcel_core::from_slice_i32(&[0xF], &[1]);

    let high = mlxcel_core::right_shift(&packed_i32, &high_shift);
    let high = mlxcel_core::bitwise_and(&high, &nibble_mask);
    let low = mlxcel_core::bitwise_and(&packed_i32, &nibble_mask);

    let unpacked = if original_shape.len() == 2 {
        let combined = mlxcel_core::concatenate(&high, &low, 0);
        let reorder: Vec<i32> = (0..packed_shape[0])
            .flat_map(|idx| [idx, idx + packed_shape[0]])
            .collect();
        let reorder = mlxcel_core::from_slice_i32(&reorder, &[packed_shape[0] * 2]);
        mlxcel_core::take(&combined, &reorder, 0)
    } else {
        let combined = mlxcel_core::concatenate(&high, &low, 1);
        let reorder: Vec<i32> = (0..packed_shape[1])
            .flat_map(|idx| [idx, idx + packed_shape[1]])
            .collect();
        let reorder = mlxcel_core::from_slice_i32(&reorder, &[packed_shape[1] * 2]);
        mlxcel_core::take(&combined, &reorder, 1)
    };

    let unpacked = mlxcel_core::astype(&unpacked, mlxcel_core::dtype::FLOAT32);
    let centered = mlxcel_core::subtract(&unpacked, zero_point);
    let dequantized = mlxcel_core::multiply(&centered, scale);
    let dequantized = mlxcel_core::astype(&dequantized, mlxcel_core::dtype::BFLOAT16);
    mlxcel_core::reshape(&dequantized, original_shape)
}

fn remap_moondream3_weights(
    raw_weights: &WeightMap,
    text_config: &models::moondream3::ModelArgs,
) -> Result<WeightMap> {
    let mut weights = WeightMap::new();

    for (key, value) in raw_weights {
        let Some(new_key) = rewrite_moondream3_weight_key(key) else {
            continue;
        };

        if new_key.ends_with(".weight.packed")
            || new_key.ends_with(".weight.scale")
            || new_key.ends_with(".weight.zero_point")
        {
            continue;
        }

        weights.insert(new_key, mlxcel_core::copy(value));
    }

    for key in raw_weights.keys() {
        let Some(prefix) = key.strip_suffix(".weight.packed") else {
            continue;
        };
        let Some(new_prefix) = rewrite_moondream3_weight_key(prefix) else {
            continue;
        };

        let packed = raw_weights
            .get(&format!("{}.weight.packed", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing Moondream3 packed weight for {}", prefix))?;
        let scale = raw_weights
            .get(&format!("{}.weight.scale", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing Moondream3 weight scale for {}", prefix))?;
        let zero_point = raw_weights
            .get(&format!("{}.weight.zero_point", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing Moondream3 zero_point for {}", prefix))?;

        let packed_shape = mlxcel_core::array_shape(packed);
        let original_shape = match packed_shape.len() {
            2 => moondream3_dense_weight_shape(&new_prefix, text_config)?,
            3 => moondream3_moe_weight_shape(&new_prefix, text_config)?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported Moondream3 packed rank {} for {}",
                    packed_shape.len(),
                    new_prefix
                ));
            }
        };

        let weight = dequantize_moondream3_weight(packed, scale, zero_point, &original_shape);
        weights.insert(format!("{}.weight", new_prefix), weight);
    }

    let ptrs: Vec<*const mlxcel_core::MlxArray> = weights
        .values()
        .filter_map(|value| value.as_ref().map(|array| array as *const _))
        .collect();
    if !ptrs.is_empty() {
        unsafe { mlxcel_core::eval_all(&ptrs) };
    }

    Ok(weights)
}

pub(super) fn remap_minicpmo_text_weights(raw_weights: &WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key.clone()
        };
        weights.insert(new_key, mlxcel_core::copy(value));
    }
    weights
}

fn minicpmo_processor_config_value(model_path: &Path, full_config: &Value) -> Option<Value> {
    read_optional_model_json(model_path, "preprocessor_config.json")
        .or_else(|| {
            read_optional_model_json(model_path, "processor_config.json")
                .and_then(|value| value.get("image_processor").cloned().or(Some(value)))
        })
        .or_else(|| Some(full_config.clone()))
}

pub(crate) fn load_minicpmo_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::minicpmo::{
        MiniCPMOResampler, MiniCPMOVisionConfig, MiniCPMOVisionModel,
    };
    use vision::processors::minicpmo::MiniCPMOProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    // MiniCPM-o uses standard Qwen3 (NOT Qwen3-VL) as text backbone
    let text_config: models::qwen3::ModelArgs = serde_json::from_value(full_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse MiniCPM-o text config: {}", e))?;
    let vision_config: MiniCPMOVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "MiniCPM-o vision config")?;
    let processor_config = minicpmo_processor_config_value(model_path, &full_config)
        .unwrap_or_else(|| full_config.clone());

    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let text_weights = remap_minicpmo_text_weights(&raw_weights);
    let (group_size, bits) = parse_quantization_params(&full_config);

    let text_model = models::Qwen3Model::from_weights(&text_weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-o text model: {}", e))?;
    let vision_tower = MiniCPMOVisionModel::from_weights(
        &raw_weights,
        &vision_config,
        "vision_tower",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-o vision tower: {}", e))?;
    let resampler = MiniCPMOResampler::from_weights(
        &raw_weights,
        "resampler",
        text_config.hidden_size,
        (text_config.hidden_size / 128).max(1),
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-o resampler: {}", e))?;

    let processor = MiniCPMOProcessor::new(
        processor_config
            .get("patch_size")
            .and_then(|value| value.as_u64())
            .unwrap_or(14) as usize,
        processor_config
            .get("scale_resolution")
            .and_then(|value| value.as_u64())
            .unwrap_or(448) as usize,
        processor_config
            .get("image_feature_size")
            .and_then(|value| value.as_u64())
            .unwrap_or(
                full_config
                    .get("query_num")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(64),
            ) as usize,
    );

    let eos_token_ids = match full_config.get("eos_token_id") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_i64().map(|id| id as i32))
            .collect(),
        Some(Value::Number(value)) => value.as_i64().map(|id| vec![id as i32]).unwrap_or_default(),
        _ => Vec::new(),
    };

    Ok(LoadedModel::MiniCPMOVLM(vision::MiniCPMOVLModel {
        text_model,
        vision_tower,
        resampler,
        processor,
        eos_token_ids,
    }))
}

/// Remap MiniCPM-V 4.6 checkpoint weight keys to flat namespaces.
///
/// Checkpoint layout → mlxcel layout:
///   `model.language_model.*`               → `language_model.*`
///   `model.lm_head.*`                      → `lm_head.*`
///   `model.vision_tower.vit_merger.*`      → `vit_merger.*`
///   `model.vision_tower.*`                 → `vision_tower.*`
///   `model.vpm.*`                          → `vision_tower.*`
///   `model.vit_merger.*`                   → `vit_merger.*`
///   `model.merger.*`                       → `merger.*`
///   `model.llm.*`                          → `language_model.*`  (compat)
///   `model.visual.*`                       → `vision_tower.*`  (compat)
fn remap_minicpmv4_6_weights(raw_weights: &WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();

    for (key, value) in raw_weights {
        if key.contains("position_ids") {
            continue;
        }

        let stripped = key.strip_prefix("model.").unwrap_or(key.as_str());

        let new_key = if let Some(rest) = stripped.strip_prefix("language_model.") {
            format!("language_model.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("lm_head.") {
            format!("lm_head.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("vision_tower.vit_merger.") {
            format!("vit_merger.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("vision_tower.") {
            format!("vision_tower.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("vpm.") {
            format!("vision_tower.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("vit_merger.") {
            format!("vit_merger.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("merger.") {
            format!("merger.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("llm.") {
            // Backward-compat: older naming
            format!("language_model.{}", rest)
        } else if let Some(rest) = stripped.strip_prefix("visual.") {
            format!("vision_tower.{}", rest)
        } else {
            continue;
        };

        weights.insert(new_key, mlxcel_core::copy(value));
    }

    weights
}

/// Extract text-model weights from the full remapped map.
/// Qwen35Model::from_weights expects keys WITHOUT a "language_model." prefix.
fn minicpmv4_6_text_weights(weights: &WeightMap) -> WeightMap {
    let mut text_weights = WeightMap::new();
    for (key, value) in weights {
        if let Some(rest) = key.strip_prefix("language_model.") {
            text_weights.insert(rest.to_string(), mlxcel_core::copy(value));
        } else if key.starts_with("lm_head.") {
            text_weights.insert(key.clone(), mlxcel_core::copy(value));
        }
    }
    text_weights
}

pub(crate) fn load_minicpmv4_6_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::minicpmv4_6::{
        MiniCPMV46Config, MiniCPMV46VisionConfig, MiniCPMV46VisionModel,
    };
    use vision::minicpmv4_6_vl::MiniCPMV46VLModel;
    use vision::processors::minicpmo::MiniCPMOProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // MiniCPM-V 4.6 uses Qwen3.5-Text (hybrid linear+full-attention) as the
    // language backbone. The LLM fields are nested under "text_config"; fall
    // back to root if absent (future-proofing and test configs).
    let text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    let text_config: models::qwen3_5::Qwen35Config = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse MiniCPM-V 4.6 text config: {}", e))?;

    let vision_config: MiniCPMV46VisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "MiniCPM-V 4.6 vision config")?;

    let model_cfg: MiniCPMV46Config = serde_json::from_value(full_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse MiniCPM-V 4.6 model config: {}", e))?;

    let processor_config = minicpmo_processor_config_value(model_path, &full_config)
        .unwrap_or_else(|| full_config.clone());

    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let (group_size, bits) = parse_quantization_params(&full_config);

    // Remap all keys to flat namespaces.
    let weights = remap_minicpmv4_6_weights(&raw_weights);

    // Build text-only weight map (de-prefixed for Qwen35Model).
    let mut text_weights = minicpmv4_6_text_weights(&weights);

    // Apply bf16→f16 for non-quantized text model weights on Apple Silicon.
    if bits == 0 {
        let hw = mlxcel_core::hardware::get_hardware();
        if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
            let had_bf16 = models::convert_bf16_weights_with_keep(&mut text_weights, |key| {
                key.ends_with(".scales") || key.ends_with(".biases")
            });
            if had_bf16 {
                models::warn_bf16_precision();
            }
        }
    }

    // Qwen3.5 hybrid-backbone sanitization. Applies the GatedDelta
    // `conv1d.weight` transpose and the RMSNorm `+1.0` shift, and (for MoE
    // checkpoints) stacks `switch_mlp` experts / splits `gate_up_proj`.
    // Without this, raw-layout / MTP / MoE Qwen3.5 checkpoints load with the
    // WRONG weights. Mirrors the Qwen3.5 VLM loader (`vlm_qwen.rs`) and the
    // special Qwen3.5 text loader (`special.rs`). `sanitize_weights` covers
    // both dense and MoE layouts (steps 6-8), and the text weights are
    // already de-prefixed to the `model.*` / `lm_head.*` namespace it expects.
    let mut text_weights = models::qwen3_5::sanitize_weights(text_weights, &text_config);

    models::sanitize_tied_embeddings(&mut text_weights, &full_config);

    let text_model = models::Qwen35Model::from_weights(&text_weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-V 4.6 text model: {}", e))?;

    // MiniCPMV46VisionModel::from_weights uses the remapped weight map with:
    //   vision tower   prefix = "vision_tower"
    //   vit_merger     prefix = "vit_merger"
    //   merger         prefix = "merger"
    let vision_model = MiniCPMV46VisionModel::from_weights(
        &weights,
        &vision_config,
        &model_cfg,
        "vision_tower",
        "vit_merger",
        "merger",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load MiniCPM-V 4.6 vision model: {}", e))?;

    // "query_num" is the mlx-vlm upstream field; "image_feature_size" appears in
    // the processor_config.json image_processor dict of real checkpoints.
    let image_feature_size = processor_config
        .get("query_num")
        .or_else(|| processor_config.get("image_feature_size"))
        .or_else(|| full_config.get("query_num"))
        .and_then(|v| v.as_u64())
        .unwrap_or(64) as usize;

    let patch_size = processor_config
        .get("patch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(14) as usize;

    let scale_resolution = processor_config
        .get("scale_resolution")
        .and_then(|v| v.as_u64())
        .unwrap_or(448) as usize;

    let resize_multiple_for_axis = |axis: usize| -> Result<usize> {
        let vit_factor = vision_config.window_kernel_size[axis].max(1);
        let merge_factor = model_cfg.merge_kernel_size[axis].max(1);
        let mut downsample = vit_factor;
        for _ in 0..model_cfg.merger_times {
            downsample = downsample.checked_mul(merge_factor).ok_or_else(|| {
                anyhow::anyhow!(
                    "MiniCPM-V 4.6 image downsample factor overflowed while building processor"
                )
            })?;
        }
        patch_size.checked_mul(downsample).ok_or_else(|| {
            anyhow::anyhow!("MiniCPM-V 4.6 resize alignment overflowed while building processor")
        })
    };

    // Upstream MiniCPM-V 4.6 `_find_best_resize` aligns dimensions to
    // `patch_size * 4` for the default 2x2 VitMerger followed by a 2x2
    // Merger.  Compute that alignment from config instead of hardcoding 4 so
    // rectangular images still reserve the same number of text placeholders as
    // vision tokens emitted by the dynamic grid.
    let processor = MiniCPMOProcessor::new_with_resize_multiples(
        patch_size,
        scale_resolution,
        image_feature_size,
        resize_multiple_for_axis(0)?,
        resize_multiple_for_axis(1)?,
    );

    let eos_token_ids = match full_config.get("eos_token_id") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|v| v.as_i64().map(|id| id as i32))
            .collect(),
        Some(Value::Number(value)) => value.as_i64().map(|id| vec![id as i32]).unwrap_or_default(),
        _ => Vec::new(),
    };

    Ok(LoadedModel::MiniCPMV46VLM(MiniCPMV46VLModel {
        text_model,
        vision_model,
        processor,
        eos_token_ids,
    }))
}

pub(crate) fn load_moondream3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::moondream3::{Moondream3VisionConfig, Moondream3VisionModel};
    use vision::processors::moondream3::Moondream3Processor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let text_config: models::moondream3::ModelArgs =
        serde_json::from_value(moondream3_text_config_value(&full_config))
            .map_err(|err| anyhow::anyhow!("Failed to parse Moondream3 text config: {}", err))?;
    let vision_config: Moondream3VisionConfig =
        serde_json::from_value(moondream3_vision_config_value(&full_config))
            .map_err(|err| anyhow::anyhow!("Failed to parse Moondream3 vision config: {}", err))?;

    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let weights = remap_moondream3_weights(&raw_weights, &text_config)?;

    let text_model = models::Moondream3Model::from_weights(&weights, &text_config)
        .map_err(|err| anyhow::anyhow!("Failed to load Moondream3 text model: {}", err))?;
    let vision_tower = Moondream3VisionModel::from_weights(&weights, &vision_config)
        .map_err(|err| anyhow::anyhow!("Failed to load Moondream3 vision tower: {}", err))?;
    let processor = Moondream3Processor::new(
        vision_config.crop_size,
        vision_config.enc_patch_size,
        vision_config.max_crops,
        vision_config.overlap_margin,
    );

    Ok(LoadedModel::Moondream3VLM(vision::Moondream3VLModel {
        text_model,
        vision_tower,
        processor,
        eos_token_ids: vec![text_config.eos_token_id],
    }))
}

pub(super) fn rewrite_phi4_siglip_weight_key(key: &str) -> Option<String> {
    if key.contains("position_ids") || key.contains("vision_model.head.") {
        None
    } else if let Some(rest) = key.strip_prefix("model.vision_tower.") {
        Some(format!("vision_tower.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.mm_projector.0.") {
        Some(format!("mm_projector_linear1.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.mm_projector.2.") {
        Some(format!("mm_projector_linear2.{}", rest))
    } else {
        Some(key.to_string())
    }
}

fn remap_phi4_siglip_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let Some(new_key) = rewrite_phi4_siglip_weight_key(&key) else {
            continue;
        };
        weights.insert(new_key, value);
    }
    weights
}

pub(super) fn phi4_siglip_text_config_value(full_config: &Value) -> Result<Value> {
    let mut text_config = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let text_obj = super::require_object_mut(&mut text_config, "Phi4-SigLIP text_config")?;
    for &field in PHI4_FUSED_TEXT_FIELDS {
        if let Some(value) = full_config.get(field) {
            text_obj
                .entry(field.to_string())
                .or_insert_with(|| value.clone());
        }
    }

    Ok(text_config)
}

fn phi4mm_vision_lora_scale(full_config: &Value) -> f32 {
    let vision_lora = full_config.get("vision_lora").and_then(Value::as_object);
    let rank = vision_lora
        .and_then(|cfg| cfg.get("r"))
        .and_then(Value::as_f64)
        .unwrap_or(256.0);
    let alpha = vision_lora
        .and_then(|cfg| cfg.get("lora_alpha"))
        .and_then(Value::as_f64)
        .unwrap_or(512.0);
    (alpha / rank.max(1.0)) as f32
}

fn flatten_phi4mm_patch_embedding(
    weight: &mlxcel_core::MlxArray,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 4 {
        return mlxcel_core::copy(weight);
    }

    let transposed = mlxcel_core::transpose_axes(weight, &[0, 2, 3, 1]);
    mlxcel_core::reshape(&transposed, &[shape[0], shape[1] * shape[2] * shape[3]])
}

fn rewrite_phi4mm_vision_key(key: &str) -> Option<String> {
    if key.contains("position_ids")
        || key.contains("img_processor.head.")
        || key.starts_with("model.embed_tokens_extend.audio_embed.")
    {
        None
    } else if key == "model.embed_tokens_extend.image_embed.glb_GN" {
        Some("glb_GN".to_string())
    } else if key == "model.embed_tokens_extend.image_embed.sub_GN" {
        Some("sub_GN".to_string())
    } else if let Some(rest) =
        key.strip_prefix("model.embed_tokens_extend.image_embed.img_processor.")
    {
        Some(format!("vision_tower.vision_tower.vision_model.{}", rest))
    } else if let Some(rest) =
        key.strip_prefix("model.embed_tokens_extend.image_embed.img_projection.0.")
    {
        Some(format!("mm_projector_linear1.{}", rest))
    } else if let Some(rest) =
        key.strip_prefix("model.embed_tokens_extend.image_embed.img_projection.2.")
    {
        Some(format!("mm_projector_linear2.{}", rest))
    } else {
        Some(key.to_string())
    }
}

pub(super) fn phi4mm_text_config_value(full_config: &Value) -> Result<Value> {
    let mut text_config = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let text_obj = super::require_object_mut(&mut text_config, "Phi4MM text_config")?;
    for &field in PHI4_FUSED_TEXT_FIELDS {
        if let Some(value) = full_config.get(field) {
            text_obj
                .entry(field.to_string())
                .or_insert_with(|| value.clone());
        }
    }

    Ok(text_config)
}

pub(super) fn phi4mm_vision_config_value(full_config: &Value) -> Value {
    let image_size = full_config
        .get("embd_layer")
        .and_then(|layer| layer.get("image_embd_layer"))
        .and_then(|layer| layer.get("crop_size"))
        .and_then(Value::as_u64)
        .unwrap_or(448);
    let patch_size = full_config
        .get("vision_config")
        .and_then(|cfg| cfg.get("patch_size"))
        .and_then(Value::as_u64)
        .unwrap_or(14);

    serde_json::json!({
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_channels": 3,
        "image_size": image_size,
        "patch_size": patch_size,
        "num_patches": (image_size / patch_size).pow(2),
        "layer_norm_eps": 1e-6
    })
}

/// LoRA pair: (lora_A weight, lora_B weight) keyed by the base weight name.
type LoRAPairs = Vec<(
    String,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
)>;

/// Returns (remapped_weights, lora_pairs) where lora_pairs contains
/// the raw LoRA A/B weights for runtime on-the-fly application.
fn remap_phi4mm_weights(
    raw_weights: WeightMap,
    vision_lora_scale: f32,
) -> Result<(WeightMap, LoRAPairs, f32)> {
    let mut phi4mm_vision_lora: HashMap<
        String,
        (
            Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
            Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
        ),
    > = HashMap::new();
    let mut base_weights = WeightMap::new();

    for (key, value) in raw_weights {
        if let Some(base_key) = key.strip_suffix(".lora_A.vision.weight") {
            let final_key = rewrite_phi4mm_vision_key(&format!("{}.weight", base_key))
                .map(|key| key.replace(".base_layer.", "."));
            if let Some(final_key) = final_key {
                phi4mm_vision_lora
                    .entry(final_key)
                    .or_insert((None, None))
                    .0 = Some(value);
            }
            continue;
        }
        if let Some(base_key) = key.strip_suffix(".lora_B.vision.weight") {
            let final_key = rewrite_phi4mm_vision_key(&format!("{}.weight", base_key))
                .map(|key| key.replace(".base_layer.", "."));
            if let Some(final_key) = final_key {
                phi4mm_vision_lora
                    .entry(final_key)
                    .or_insert((None, None))
                    .1 = Some(value);
            }
            continue;
        }
        if key.contains(".lora_A.speech.") || key.contains(".lora_B.speech.") {
            continue;
        }

        let Some(mut new_key) = rewrite_phi4mm_vision_key(&key) else {
            continue;
        };
        if new_key.contains(".base_layer.") {
            new_key = new_key.replace(".base_layer.", ".");
        }

        let value = if new_key.ends_with("patch_embedding.weight") {
            flatten_phi4mm_patch_embedding(&value)
        } else {
            value
        };
        base_weights.insert(new_key, value);
    }

    let mut lora_pairs: LoRAPairs = Vec::new();
    let effective_scale = vision_lora_scale;
    for (base_key, (lora_a, lora_b)) in phi4mm_vision_lora {
        let (Some(lora_a), Some(lora_b)) = (lora_a, lora_b) else {
            tracing::warn!(
                "Incomplete Phi4MM vision LoRA pair for {}: missing lora_A or lora_B",
                base_key
            );
            continue;
        };

        if !base_weights.contains_key(&base_key) {
            tracing::warn!(
                "Skipping Phi4MM vision LoRA pair for {} because the base weight is missing",
                base_key
            );
            continue;
        }

        // Store raw LoRA A/B for runtime on-the-fly application (not fused)
        lora_pairs.push((base_key, lora_a, lora_b));
    }

    Ok((base_weights, lora_pairs, effective_scale))
}

pub(crate) fn load_phi4mm_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::phi4_siglip::{Phi4SigLipVisionConfig, Phi4SigLipVisionEncoder};
    use vision::processors::phi4mm::Phi4MMProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config_value = phi4mm_text_config_value(&full_config)?;
    inherit_quantization_if_missing(&mut text_config_value, &full_config)?;
    let text_config: models::phi4mm::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Phi4MM text config: {}", e))?;

    let vision_config: Phi4SigLipVisionConfig =
        serde_json::from_value(phi4mm_vision_config_value(&full_config))
            .map_err(|e| anyhow::anyhow!("Failed to parse Phi4MM vision config: {}", e))?;

    let select_layer = -2isize;
    let vision_lora_scale = phi4mm_vision_lora_scale(&full_config);

    // Read HD transform config
    let image_embd_layer = full_config
        .get("embd_layer")
        .and_then(|layer| layer.get("image_embd_layer"));
    let crop_size = image_embd_layer
        .and_then(|cfg| cfg.get("crop_size"))
        .and_then(Value::as_u64)
        .unwrap_or(448) as usize;
    let hd_transform_order = image_embd_layer
        .and_then(|cfg| cfg.get("hd_transform_order"))
        .and_then(Value::as_str)
        .unwrap_or("glb_sub")
        .to_string();

    // Read dynamic_hd from preprocessor_config.json
    let dynamic_hd = {
        let preproc_path = model_path.join("preprocessor_config.json");
        if preproc_path.exists() {
            let preproc_str = std::fs::read_to_string(&preproc_path)
                .map_err(|e| anyhow::anyhow!("Failed to read preprocessor_config.json: {}", e))?;
            let preproc: serde_json::Value = serde_json::from_str(&preproc_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse preprocessor_config.json: {}", e))?;
            preproc
                .get("dynamic_hd")
                .and_then(Value::as_u64)
                .unwrap_or(36) as usize
        } else {
            36
        }
    };

    let (mut weights, lora_pairs, effective_scale) = remap_phi4mm_weights(
        load_vlm_weights_common(model_path, None)?,
        vision_lora_scale,
    )?;
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // Load glb_GN and sub_GN learnable separator weights
    let glb_gn = weights
        .get("glb_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Phi4MM glb_GN weight not found"))?;
    let sub_gn = weights
        .get("sub_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Phi4MM sub_GN weight not found"))?;

    let mut text_model = models::Phi4MMModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Phi4MM text model: {}", e))?;

    // Set runtime LoRA weights on the text model's linear layers.
    // LoRA starts active (applied during prefill) and is deactivated
    // by after_prefill() so decode uses base weights only — matching
    // Python PEFT behavior.
    let has_runtime_lora = !lora_pairs.is_empty();
    let mut lora_set_count = 0usize;
    for (key, lora_a, lora_b) in lora_pairs {
        // key = "model.layers.N.{section}.{proj}.weight"
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() < 5 || parts[0] != "model" || parts[1] != "layers" {
            continue;
        }
        let Some(layer_idx) = parts[2].parse::<usize>().ok() else {
            continue;
        };
        if layer_idx >= text_model.layers.len() {
            continue;
        }
        let layer = &mut text_model.layers[layer_idx];
        match (parts[3], parts[4]) {
            ("self_attn", "qkv_proj") => {
                layer
                    .self_attn
                    .qkv_proj
                    .set_lora(lora_a, lora_b, effective_scale);
                lora_set_count += 1;
            }
            ("self_attn", "o_proj") => {
                layer
                    .self_attn
                    .o_proj
                    .set_lora(lora_a, lora_b, effective_scale);
                lora_set_count += 1;
            }
            ("mlp", "gate_up_proj") => {
                layer
                    .mlp
                    .gate_up_proj
                    .set_lora(lora_a, lora_b, effective_scale);
                lora_set_count += 1;
            }
            ("mlp", "down_proj") => {
                layer
                    .mlp
                    .down_proj
                    .set_lora(lora_a, lora_b, effective_scale);
                lora_set_count += 1;
            }
            _ => {}
        }
    }
    tracing::debug!("Phi4MM: set LoRA on {} linear layers", lora_set_count);

    let vision_tower = Phi4SigLipVisionEncoder::from_weights(
        &weights,
        &vision_config,
        "vision_tower.vision_tower.vision_model",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4MM vision tower: {}", e))?;

    let mm_projector_linear1 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "mm_projector_linear1",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4MM mm_projector_linear1: {}", e))?;
    let mm_projector_linear2 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "mm_projector_linear2",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4MM mm_projector_linear2: {}", e))?;

    let processor = Phi4MMProcessor::new(crop_size, vision_config.patch_size, dynamic_hd);
    let eos_token_ids = match full_config.get("eos_token_id") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_i64().map(|id| id as i32))
            .collect(),
        Some(Value::Number(value)) => value.as_i64().map(|id| vec![id as i32]).unwrap_or_default(),
        _ => Vec::new(),
    };

    Ok(LoadedModel::Phi4MMVLM(vision::Phi4MMVLModel {
        text_model,
        vision_tower,
        mm_projector_linear1,
        mm_projector_linear2,
        processor,
        select_layer,
        eos_token_ids,
        glb_gn,
        sub_gn,
        hd_transform_order,
        has_runtime_lora,
    }))
}

pub(crate) fn load_phi4_siglip_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::phi4_siglip::{Phi4SigLipVisionConfig, Phi4SigLipVisionEncoder};
    use vision::processors::phi4_siglip::Phi4SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config_value = phi4_siglip_text_config_value(&full_config)?;
    inherit_quantization_if_missing(&mut text_config_value, &full_config)?;
    let text_config: models::phi3::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Phi4-SigLIP text config: {}", e))?;

    let vision_config: Phi4SigLipVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Phi4-SigLIP vision config")?;

    let mm_hidden_size = full_config
        .get("mm_hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(vision_config.hidden_size as i64) as usize;
    let select_layer = full_config
        .get("mm_vision_select_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-2) as isize;
    let min_num_patches = full_config
        .get("min_num_patches")
        .and_then(|v| v.as_u64())
        .unwrap_or(vision_config.num_patches as u64) as usize;
    let max_num_patches = full_config
        .get("max_num_patches")
        .and_then(|v| v.as_u64())
        .unwrap_or((vision_config.image_size / vision_config.patch_size).pow(2) as u64)
        as usize;

    let mut weights = remap_phi4_siglip_weights(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Phi3Model::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP text model: {}", e))?;
    let vision_tower = Phi4SigLipVisionEncoder::from_weights(
        &weights,
        &vision_config,
        "vision_tower.vision_tower.vision_model",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP vision tower: {}", e))?;

    let mm_projector_linear1 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "mm_projector_linear1",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP mm_projector_linear1: {}", e))?;
    let mm_projector_linear2 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "mm_projector_linear2",
        text_config.group_size(),
        text_config.bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Phi4-SigLIP mm_projector_linear2: {}", e))?;

    let processor =
        Phi4SigLipProcessor::new(vision_config.patch_size, min_num_patches, max_num_patches);

    let eos_token_ids = match full_config.get("eos_token_id") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_i64().map(|id| id as i32))
            .collect(),
        Some(Value::Number(value)) => value
            .as_i64()
            .map(|id| vec![id as i32])
            .unwrap_or_else(Vec::new),
        _ => Vec::new(),
    };
    let _ = mm_hidden_size;

    let vlm = vision::Phi4SigLipVLModel {
        text_model,
        vision_tower,
        mm_projector_linear1,
        mm_projector_linear2,
        processor,
        select_layer,
        eos_token_ids,
    };

    Ok(LoadedModel::Phi4SigLipVLM(vlm))
}

pub(super) fn rewrite_phi3_weight_key(key: &str) -> Option<String> {
    if key.contains("position_ids") {
        None
    } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_processor.") {
        Some(format!("vision_tower.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_projection.0.") {
        Some(format!("img_projection.0.{}", rest))
    } else if let Some(rest) = key.strip_prefix("model.vision_embed_tokens.img_projection.2.") {
        Some(format!("img_projection.2.{}", rest))
    } else if key == "model.vision_embed_tokens.glb_GN" {
        Some("glb_GN".to_string())
    } else if key == "model.vision_embed_tokens.sub_GN" {
        Some("sub_GN".to_string())
    } else {
        Some(key.to_string())
    }
}

pub(super) fn should_transpose_phi3_patch_embedding(shape: &[i32]) -> bool {
    if shape.len() != 4 {
        return false;
    }
    let out_ch = shape[0];
    let dim1 = shape[1];
    let dim2 = shape[2];
    !(out_ch >= dim1 && out_ch >= dim2 && dim1 == dim2)
}

fn remap_phi3_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let Some(new_key) = rewrite_phi3_weight_key(&key) else {
            continue;
        };

        let mapped_value = if new_key.contains("patch_embedding.weight")
            && should_transpose_phi3_patch_embedding(&mlxcel_core::array_shape(&value))
        {
            mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
        } else {
            value
        };

        weights.insert(new_key, mapped_value);
    }
    weights
}

pub(super) fn phi3_num_crops(full_config: &Value, preprocessor_config: Option<&Value>) -> usize {
    if let Some(config) = preprocessor_config {
        return config
            .get("num_crops")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
    }

    full_config
        .get("vision_config")
        .and_then(|vc| vc.get("num_crops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as usize
}

pub(crate) fn load_phi3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::phi3_v::Phi3VProcessor;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let text_args: models::phi3::ModelArgs = parse_vlm_config(&config_str, "text config")?;

    let image_dim_out = phi3_vision_config().hidden_size;
    let mut weights = remap_phi3_weights(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Phi3Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    let vision_encoder = SigLipVisionModel::from_weights(
        &weights,
        &phi3_vision_config(),
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?
    .with_feature_selection(-2, "default".to_string());

    let glb_gn = weights
        .get("glb_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("glb_GN weight not found"))?;
    let sub_gn = weights
        .get("sub_GN")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("sub_GN weight not found"))?;

    let group_size = text_args.group_size();
    let bits = text_args.bits();
    let img_proj_linear1 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "img_projection.0",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load img_projection.0: {}", e))?;
    let img_proj_linear2 = mlxcel_core::layers::UnifiedLinear::from_weights(
        &weights,
        "img_projection.2",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load img_projection.2: {}", e))?;

    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let processor = Phi3VProcessor::new(phi3_num_crops(&full_config, preprocessor_config.as_ref()));

    let vlm = vision::Phi3VLModel {
        text_model,
        vision_encoder,
        glb_gn,
        sub_gn,
        img_proj_linear1,
        img_proj_linear2,
        processor,
        image_dim_out,
    };

    Ok(LoadedModel::Phi3VLM(vlm))
}

pub(super) fn cap_molmo2_vit_num_layers(num_layers: usize) -> usize {
    num_layers.min(25)
}

pub(super) fn parse_molmo2_vit_layers(adapter_config: &Value) -> Vec<i32> {
    adapter_config
        .get("vit_layers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect()
        })
        .unwrap_or_else(|| vec![-3, -9])
}

pub(super) fn rewrite_molmo2_weight_key(key: &str) -> String {
    let mut new_key = key.to_string();
    if new_key.starts_with("model.transformer.") {
        new_key = new_key.replacen("model.transformer.", "language_model.model.", 1);
    }
    if new_key.starts_with("model.vision_backbone.") {
        new_key = new_key.replacen("model.vision_backbone.", "vision_tower.", 1);
    }
    if new_key.starts_with("lm_head.") {
        new_key = new_key.replacen("lm_head.", "language_model.lm_head.", 1);
    }
    new_key.replace(".transformer.resblocks.", ".transformer.")
}

fn remap_molmo2_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        weights.insert(rewrite_molmo2_weight_key(&key), value);
    }
    weights
}

pub(super) fn molmo2_max_crops(preprocessor_config: Option<&Value>) -> usize {
    preprocessor_config
        .and_then(|config| config.get("max_crops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as usize
}

/// Rewrite original-HF Molmo v1 weight keys to the standard mlx-vlm layout.
///
/// Unlike Molmo2, v1 keeps the `.transformer.resblocks.N` vision path intact
/// (the encoder loads from that exact prefix). mlx-vlm checkpoints already use
/// the target naming, in which case this is a no-op per key.
pub(super) fn rewrite_molmo_weight_key(key: &str) -> String {
    let mut new_key = key.to_string();
    if new_key.starts_with("model.transformer.") {
        new_key = new_key.replacen("model.transformer.", "language_model.model.", 1);
    }
    if new_key.starts_with("model.vision_backbone.") {
        new_key = new_key.replacen("model.vision_backbone.", "vision_tower.", 1);
    }
    new_key
}

fn remap_molmo_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        weights.insert(rewrite_molmo_weight_key(&key), value);
    }
    weights
}

/// Read a vision config field with a reference default. Molmo-7B ships an almost
/// empty `vision_config`, so most fields fall back to the dataclass defaults.
fn molmo_vision_i32(vision_config: &Value, key: &str, default: i32) -> i32 {
    vision_config
        .get(key)
        .and_then(|v| v.as_i64())
        .unwrap_or(default as i64) as i32
}

pub(crate) fn load_molmo_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::molmo::{MolmoVisionConfig, MolmoVisionModel};
    use vision::processors::molmo::{MolmoImageTokens, MolmoProcessor};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text config: Molmo-7B is flat (text fields at the top level). The
    // MolmoTextConfig serde defaults + aliases tolerate both flat and nested
    // `text_config` schemas.
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    inherit_quantization_if_missing(&mut text_config_value, &full_config)?;
    let text_config: models::molmo::MolmoTextConfig = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Molmo text config: {}", e))?;

    // Vision config: apply reference `VisionConfig` defaults for the many fields
    // Molmo-7B omits from its minimal `vision_config`.
    let empty = Value::Object(Default::default());
    let vision_config = full_config.get("vision_config").unwrap_or(&empty);
    let mut vit_layers = full_config
        .get("vit_layers")
        .or_else(|| vision_config.get("vit_layers"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![-2, -9]);
    if vit_layers.is_empty() {
        vit_layers = vec![-2, -9];
    }

    let image_patch_size = molmo_vision_i32(vision_config, "image_patch_size", 14) as usize;
    let image_default_input = vision_config
        .get("image_default_input_size")
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let h = a.first()?.as_i64()? as i32;
            let w = a.get(1)?.as_i64()? as i32;
            Some((h, w))
        })
        .unwrap_or((336, 336));
    let patch_h = image_default_input.0 / image_patch_size as i32;
    let patch_w = image_default_input.1 / image_patch_size as i32;

    let vision_cfg = MolmoVisionConfig {
        image_num_layers: molmo_vision_i32(vision_config, "image_num_layers", 23) as usize,
        image_emb_dim: molmo_vision_i32(vision_config, "image_emb_dim", 1024),
        image_num_heads: molmo_vision_i32(vision_config, "image_num_heads", 16),
        image_num_kv_heads: molmo_vision_i32(vision_config, "image_num_key_value_heads", 16),
        image_head_dim: molmo_vision_i32(vision_config, "image_head_dim", 64),
        image_num_pos: molmo_vision_i32(vision_config, "image_num_pos", 577) as usize,
        image_norm_eps: vision_config
            .get("image_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32,
        image_num_patch: (patch_h, patch_w),
        image_pooling_h: molmo_vision_i32(vision_config, "image_pooling_h", 2),
        image_pooling_w: molmo_vision_i32(vision_config, "image_pooling_w", 2),
        vit_layers,
        group_size: text_config.group_size(),
        bits: text_config.bits(),
    };

    // Load + remap weights, then convert plain bf16 -> f16 on Apple Silicon
    // (keeping quantization scales/biases as bf16). `load_vlm_weights_common`
    // skips conversion because the model is quantized.
    let mut weights = remap_molmo_weights(load_vlm_weights_common(model_path, None)?);
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model =
        models::MolmoModel::from_weights(&weights, &text_config, "language_model.model")
            .map_err(|e| anyhow::anyhow!("Failed to load Molmo text model: {}", e))?;

    let vision_tower = MolmoVisionModel::from_weights(&weights, "vision_tower", vision_cfg)
        .map_err(|e| anyhow::anyhow!("Failed to load Molmo vision model: {}", e))?;

    // Processor configuration (preprocessor_config.json drives crop/margin/mean).
    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let max_crops = preprocessor_config
        .as_ref()
        .and_then(|c| c.get("max_crops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let overlap = preprocessor_config
        .as_ref()
        .and_then(|c| c.get("overlap_margins"))
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let l = a.first()?.as_u64()? as usize;
            let r = a.get(1)?.as_u64()? as usize;
            Some((l, r))
        });
    let base_size = preprocessor_config
        .as_ref()
        .and_then(|c| c.get("base_image_input_size"))
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let h = a.first()?.as_u64()? as usize;
            let w = a.get(1)?.as_u64()? as usize;
            Some((h, w))
        });
    let token_len = preprocessor_config.as_ref().and_then(|c| {
        let h = c.get("image_token_length_h")?.as_u64()? as usize;
        let w = c.get("image_token_length_w")?.as_u64()? as usize;
        Some((h, w))
    });
    let mean = read_clip_triple(preprocessor_config.as_ref(), "image_mean");
    let std = read_clip_triple(preprocessor_config.as_ref(), "image_std");

    let tokens = MolmoImageTokens::default();
    let processor = MolmoProcessor::new(
        max_crops,
        overlap,
        Some(image_patch_size),
        base_size,
        token_len,
        mean,
        std,
        tokens,
    );

    let vlm = vision::MolmoVLModel {
        text_model,
        vision_tower,
        processor,
    };

    Ok(LoadedModel::MolmoVLM(vlm))
}

/// Read a 3-element CLIP normalization triple from the preprocessor config.
fn read_clip_triple(config: Option<&Value>, key: &str) -> Option<[f32; 3]> {
    config
        .and_then(|c| c.get(key))
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let x = a.first()?.as_f64()? as f32;
            let y = a.get(1)?.as_f64()? as f32;
            let z = a.get(2)?.as_f64()? as f32;
            Some([x, y, z])
        })
}

pub(crate) fn load_molmo2_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::molmo2::Molmo2VisionModel;
    use vision::processors::molmo2::Molmo2Processor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    inherit_quantization_if_missing(&mut text_config_value, &full_config)?;
    let text_config: models::molmo2::Molmo2TextConfig =
        serde_json::from_value(text_config_value)
            .map_err(|e| anyhow::anyhow!("Failed to parse text config: {}", e))?;

    let vision_config = full_config.get("vision_config").unwrap_or(&full_config);
    let vit_config = vision_config.get("vit_config").unwrap_or(vision_config);
    let adapter_config = vision_config.get("adapter_config").unwrap_or(vision_config);

    let vit_num_layers = cap_molmo2_vit_num_layers(
        vit_config
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(25) as usize,
    );
    let vit_hidden_size = vit_config
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(1152) as i32;
    let vit_intermediate_size = vit_config
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(4304) as i32;
    let vit_num_heads = vit_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_num_kv_heads = vit_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_head_dim = vit_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let vit_image_num_pos = vit_config
        .get("image_num_pos")
        .and_then(|v| v.as_u64())
        .unwrap_or(729) as usize;
    let vit_layer_norm_eps = vit_config
        .get("layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let vit_float32_attention = vit_config
        .get("float32_attention")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let adapter_hidden_size = adapter_config
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(1152) as i32;
    let adapter_intermediate_size = adapter_config
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(9728) as i32;
    let adapter_text_hidden_size = adapter_config
        .get("text_hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(text_config.hidden_size as i64) as i32;
    let adapter_num_heads = adapter_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_num_kv_heads = adapter_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_head_dim = adapter_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let adapter_float32_attention = adapter_config
        .get("float32_attention")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let pooling_attention_mask = adapter_config
        .get("pooling_attention_mask")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let image_patch_id = full_config
        .get("image_patch_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151938) as i32;
    let image_end_token_id = full_config
        .get("image_end_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151937) as i32;

    let mut weights = remap_molmo2_weights(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model =
        models::Molmo2Model::from_weights(&weights, &text_config, "language_model.model")
            .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    let vision_tower = Molmo2VisionModel::from_weights(
        &weights,
        "vision_tower",
        vit_num_layers,
        vit_hidden_size,
        vit_intermediate_size,
        vit_num_heads,
        vit_num_kv_heads,
        vit_head_dim,
        vit_image_num_pos,
        vit_layer_norm_eps,
        vit_float32_attention,
        adapter_hidden_size,
        adapter_intermediate_size,
        adapter_text_hidden_size,
        adapter_num_heads,
        adapter_num_kv_heads,
        adapter_head_dim,
        adapter_float32_attention,
        &parse_molmo2_vit_layers(adapter_config),
        pooling_attention_mask,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision model: {}", e))?;

    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let processor = Molmo2Processor::new(
        molmo2_max_crops(preprocessor_config.as_ref()),
        None,
        None,
        None,
        None,
    );

    let vlm = vision::Molmo2VLModel {
        text_model,
        vision_tower,
        processor,
        image_patch_id,
        image_end_token_id,
    };

    Ok(LoadedModel::Molmo2VLM(vlm))
}

pub(crate) fn load_molmo_point_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::molmo_point::{MolmoPointConnector, PointPredictor};
    use vision::encoders::molmo2::Molmo2VisionTransformer;
    use vision::molmo_point_vl::{MolmoPointConfig, MolmoPointVLModel};
    use vision::processors::molmo2::Molmo2Processor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Parse text config
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    inherit_quantization_if_missing(&mut text_config_value, &full_config)?;
    let text_config: models::molmo2::Molmo2TextConfig =
        serde_json::from_value(text_config_value)
            .map_err(|e| anyhow::anyhow!("Failed to parse text config: {e}"))?;

    // Parse vision config (may be in vit_config or vision_config)
    let vit_config = full_config
        .get("vit_config")
        .or_else(|| full_config.get("vision_config"))
        .unwrap_or(&full_config);

    // Parse adapter config
    let adapter_config = full_config.get("adapter_config").unwrap_or(&full_config);

    // ViT parameters
    let vit_num_layers = cap_molmo2_vit_num_layers(
        vit_config
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(27) as usize,
    );
    let vit_hidden_size = vit_config
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(1152) as i32;
    let vit_intermediate_size = vit_config
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(4304) as i32;
    let vit_num_heads = vit_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_num_kv_heads = vit_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let vit_head_dim = vit_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let vit_image_num_pos = vit_config
        .get("image_num_pos")
        .and_then(|v| v.as_u64())
        .unwrap_or(729) as usize;
    let vit_layer_norm_eps = vit_config
        .get("layer_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;

    // Adapter parameters
    let adapter_num_heads = adapter_config
        .get("num_attention_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_num_kv_heads = adapter_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;
    let adapter_head_dim = adapter_config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(72) as i32;
    let pooling_attention_mask = adapter_config
        .get("pooling_attention_mask")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let positional_embeddings_size = adapter_config
        .get("positional_embeddings")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    // Resolve vit_layers (may be in adapter_config)
    let vit_layers_raw = parse_molmo2_vit_layers(adapter_config);
    let vit_layers_resolved: Vec<usize> = vit_layers_raw
        .iter()
        .map(|&layer| {
            if layer < 0 {
                (layer + vit_num_layers as i32) as usize
            } else {
                layer as usize
            }
        })
        .collect();

    // Truncate ViT to only the layers we need
    let last_layer_needed = *vit_layers_resolved.iter().max().unwrap_or(&0) + 1;
    let actual_vit_layers = last_layer_needed.min(vit_num_layers);

    // Model-level config
    let image_patch_id = full_config
        .get("image_patch_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151938) as i32;
    let image_end_token_id = full_config
        .get("image_end_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151937) as i32;
    let image_non_indexable_patch_id = full_config
        .get("image_non_indexable_patch_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151942) as i32;
    let patch_token_id = full_config
        .get("patch_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151947) as i32;
    let subpatch_token_id = full_config
        .get("subpatch_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151948) as i32;
    let location_token_id = full_config
        .get("location_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(151949) as i32;
    let no_more_points_class = full_config
        .get("no_more_points_class")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let norm_logits = full_config
        .get("norm_logits")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let patch_location = full_config
        .get("patch_location")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let patch_embed_dim = full_config
        .get("patch_embed_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(512) as i32;
    let layer_norm_x = full_config
        .get("layer_norm_x")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let token_prediction_rotary = full_config
        .get("token_prediction_rotary")
        .and_then(|v| v.as_str())
        .unwrap_or("one_d");
    let token_prediction_rotary_theta = full_config
        .get("token_prediction_rotary_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(50000.0) as f32;

    // Load and remap weights
    let weights = remap_molmo_point_weights(load_vlm_weights_common(model_path, None)?);

    // Load language model (uses Molmo2 text config)
    let language_model =
        models::molmo_point::MolmoPointLanguageModel::from_weights(&weights, &text_config, "lm")
            .map_err(|e| anyhow::anyhow!("Failed to load language model: {e}"))?;

    // Load vision model (ViT, possibly truncated)
    let vision_model = Molmo2VisionTransformer::from_weights(
        &weights,
        "vision_model",
        actual_vit_layers,
        vit_hidden_size,
        vit_intermediate_size,
        vit_num_heads,
        vit_num_kv_heads,
        vit_head_dim,
        vit_image_num_pos,
        vit_layer_norm_eps,
        true, // float32_attention
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision model: {e}"))?;

    // Load connector
    let connector = MolmoPointConnector::from_weights(
        &weights,
        "connector",
        adapter_num_heads,
        adapter_num_kv_heads,
        adapter_head_dim,
        positional_embeddings_size,
        pooling_attention_mask,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load connector: {e}"))?;

    // Load point predictor
    let point_predictor = PointPredictor::from_weights(
        &weights,
        "point_predictor",
        text_config.layer_norm_eps,
        layer_norm_x,
        token_prediction_rotary,
        token_prediction_rotary_theta,
        patch_embed_dim,
        patch_location.is_some(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load point predictor: {e}"))?;

    // Load ViT feature embedding (vit_dim -> llm_dim)
    let build_vit_embedding =
        mlxcel_core::layers::Linear::from_weights(&weights, "build_vit_embedding")
            .map_err(|e| anyhow::anyhow!("Failed to load build_vit_embedding: {e}"))?;

    // Processor
    let preprocessor_config = read_optional_model_json(model_path, "preprocessor_config.json");
    let max_crops = preprocessor_config
        .as_ref()
        .and_then(|c| c.get("max_crops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(24) as usize;
    let processor = Molmo2Processor::new(max_crops, None, None, None, None);

    let config = MolmoPointConfig {
        image_patch_id,
        image_end_token_id,
        image_non_indexable_patch_id,
        patch_token_id,
        subpatch_token_id,
        location_token_id,
        no_more_points_class,
        norm_logits,
        patch_location,
        vit_layers: vit_layers_raw,
        hidden_size: text_config.hidden_size as i32,
    };

    let vlm = MolmoPointVLModel {
        language_model,
        vision_model,
        connector,
        point_predictor,
        build_vit_embedding,
        processor,
        config,
        vit_layers: vit_layers_resolved,
    };

    Ok(LoadedModel::MolmoPointVLM(vlm))
}

/// Remap Molmo-Point weight keys from HuggingFace format.
///
/// HF keys:
///   model.transformer.* -> lm.model.*
///   model.lm_head.* -> lm.lm_head.*
///   model.vit.transformer.resblocks.* -> vision_model.resblocks.*
///   model.vit.* -> vision_model.*
fn remap_molmo_point_weights(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let mut new_key = key.clone();

        // Strip "model." prefix
        if new_key.starts_with("model.") {
            new_key = new_key[6..].to_string();
        }

        // lm_head -> lm.lm_head
        if new_key.starts_with("lm_head.") {
            new_key = format!("lm.{new_key}");
        }

        // transformer.* -> lm.model.*
        if new_key.starts_with("transformer.") {
            new_key = format!("lm.model.{}", &new_key[12..]);
        }

        // vit.transformer.resblocks -> vision_model.transformer
        new_key = new_key.replace("vit.transformer.resblocks", "vision_model.transformer");

        // vit.* -> vision_model.* (remaining keys)
        if new_key.starts_with("vit.") {
            new_key = format!("vision_model.{}", &new_key[4..]);
        }

        // Cast float32 weights to float16
        let dtype = mlxcel_core::array_dtype(&value);
        let value = if dtype == mlxcel_core::dtype::FLOAT32 {
            mlxcel_core::astype(&value, mlxcel_core::dtype::FLOAT16)
        } else {
            mlxcel_core::copy(&value)
        };

        weights.insert(new_key, value);
    }
    weights
}

pub(super) fn inherit_quantization_if_missing(
    text_config: &mut Value,
    full_config: &Value,
) -> Result<()> {
    if text_config.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(text_config, "special VLM text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    Ok(())
}

fn language_model_only_weights(weights: &WeightMap) -> WeightMap {
    let mut text_weights = WeightMap::new();
    for (key, value) in weights {
        if key.starts_with("language_model.") {
            text_weights.insert(key.clone(), mlxcel_core::copy(value));
        }
    }
    text_weights
}

pub(super) fn llama4_vision_prefix(weights: &WeightMap) -> &'static str {
    if weights.contains_key("vision_model.patch_embedding.linear.weight") {
        "vision_model"
    } else {
        "vision_tower"
    }
}

pub(super) fn llama4_quantization_params(full_config: &Value) -> (i32, i32) {
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

pub(super) fn llama4_token_ids(full_config: &Value) -> (i32, i32) {
    let image_token_id = full_config
        .get("image_token_index")
        .or_else(|| full_config.get("image_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(200092) as i32;
    let pad_token_id = full_config
        .get("text_config")
        .and_then(|tc| tc.get("pad_token_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(200018) as i32;
    (image_token_id, pad_token_id)
}

pub(super) fn llama4_mm_tokens_per_image(
    vision_config: &vision::encoders::llama4::Llama4VisionConfig,
) -> usize {
    let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
    (num_patches as f32 * vision_config.pixel_shuffle_ratio.powi(2)) as usize
}

pub(crate) fn load_llama4_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::linear::LinearProjector;
    use vision::encoders::llama4::{Llama4VisionConfig, Llama4VisionModel};
    use vision::processors::siglip::SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let text_config_value = if let Some(tc) = full_config.get("text_config") {
        let mut tc = tc.clone();
        inherit_quantization_if_missing(&mut tc, &full_config)?;
        tc
    } else {
        full_config.clone()
    };

    let text_args: models::llama4::TextArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Llama4 text config: {}", e))?;
    let vision_config: Llama4VisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config in config.json"))?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse vision_config: {}", e))?;

    let weights = load_vlm_weights_common(model_path, None)?;
    let mut text_weights = language_model_only_weights(&weights);
    models::sanitize_tied_embeddings(&mut text_weights, &full_config);

    let text_model = models::Llama4CxxModel::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Llama4 text model: {}", e))?;
    let text_wrapper = models::Llama4Wrapper::new(text_model);

    let (quant_group_size, quant_bits) = llama4_quantization_params(&full_config);
    let vision_encoder = Llama4VisionModel::from_weights(
        &weights,
        &vision_config,
        llama4_vision_prefix(&weights),
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Llama4 vision encoder: {}", e))?;

    let connector = LinearProjector::from_weights(
        &weights,
        "multi_modal_projector",
        quant_group_size,
        quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Llama4 projector: {}", e))?;

    let (image_token_id, pad_token_id) = llama4_token_ids(&full_config);
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(SigLipProcessor::new(vision_config.image_size)),
        image_token_id,
        pad_token_id,
        hidden_size: text_args.hidden_size,
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image: llama4_mm_tokens_per_image(&vision_config),
        merge_strategy: vision::MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Llama4(text_wrapper)),
        vision: vision_module,
    };

    Ok(LoadedModel::Llama4VLM(vlm))
}

#[cfg(test)]
#[path = "vlm_special_tests.rs"]
mod tests;
