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

//! Kimi-VL / Kimi-VL 2.5 loader.
//!
//! Mirrors upstream `Model.__init__` / `Model.sanitize`
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/kimi_vl.py)
//! to translate a HuggingFace `model_type = "kimi_vl"` (or `"kimi_k25"`)
//! safetensors checkpoint into a fully-wired [`crate::vision::KimiVLModel`].
//!
//! Weight-name mapping performed here:
//! - `vision_tower.encoder.<rest>` -> `vision_tower.<rest>` (upstream strips the
//!   `encoder.` segment), plus the MoonViT-specific fixups: transpose the
//!   `patch_embed.proj.weight` conv kernel to MLX channel-last layout and rename
//!   `blocks.{i}.{wqkv,wo}` -> `blocks.{i}.attn.{wqkv,wo}` when not already so.
//! - `language_model.model.<rest>` -> `model.<rest>` and
//!   `language_model.lm_head.<rest>` -> `lm_head.<rest>` (the DeepSeek-V3 backbone
//!   consumes keys directly under `model.*` / `lm_head.*`).
//! - `multi_modal_projector.*` (kimi_vl) / `mm_projector.*` (kimi_k25) kept.
//! - `position_ids` / `rotary_emb` keys dropped (RoPE replaces them).
//!
//! The MoE expert stacking and the MLA `kv_b_proj` -> `embed_q`/`unembed_out`
//! decomposition are delegated to [`DeepSeekV3Model::sanitize_weights_with_args`].

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use std::path::Path;

use crate::LoadedModel;
use crate::models::deepseek_v3::{DeepSeekV3Config, DeepSeekV3Model, QuantizationConfig};
use crate::vision::KimiVLModel;
use crate::vision::encoders::kimi_vl::{KimiVLVisionConfig, KimiVLVisionModel};
use crate::vision::kimi_vl::KimiVLMultiModalProjector;
use crate::vision::processors::kimi_vl::KimiVLProcessor;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};
use crate::loading::conv2d_weight_is_channel_last;

/// Upstream default `media_placeholder_token_id` (kimi_vl/config.py).
const DEFAULT_MEDIA_PLACEHOLDER_TOKEN_ID: i32 = 163_606;
/// Upstream default MoonViT patch-token budget (KimiVLImageProcessor).
const DEFAULT_IN_TOKEN_LIMIT: usize = 4096;

pub(crate) fn load_kimi_vl_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config: DeepSeekV3Config =
        parse_required_vlm_subconfig(&full_config, "text_config", "Kimi-VL text config")?;
    let mut vision_config: KimiVLVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Kimi-VL vision config")?;

    // Inherit a top-level `quantization` block into both sub-configs when the
    // checkpoint stores it once above the sub-configs (matches upstream loaders).
    if let Some(q) = full_config.get("quantization") {
        if text_config.quantization.is_none()
            && let Ok(parsed) = serde_json::from_value::<QuantizationConfig>(q.clone())
        {
            text_config.quantization = Some(parsed);
        }
        if vision_config.quant_bits == 0 {
            if let Some(group_size) = q.get("group_size").and_then(|v| v.as_i64()) {
                vision_config.quant_group_size = group_size as i32;
            }
            if let Some(bits) = q.get("bits").and_then(|v| v.as_i64()) {
                vision_config.quant_bits = bits as i32;
            }
        }
    }

    let media_placeholder_token_id = full_config
        .get("media_placeholder_token_id")
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or(DEFAULT_MEDIA_PLACEHOLDER_TOKEN_ID);
    let eos_token_ids = parse_eos_token_ids(&full_config);

    let spatial_merge_size = vision_config.spatial_merge_size.max(1) as i32;
    let vision_hidden = vision_config.hidden_size as i32;
    let merged_hidden = vision_hidden * spatial_merge_size * spatial_merge_size;
    let group_size = text_config.group_size();
    let bits = text_config.bits();

    // Load raw weights, run the Kimi-VL key remapping, then the shared
    // DeepSeek-V3 sanitize (expert stacking + MLA decomposition).
    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let weights = remap_kimi_vl_weights(raw_weights);
    let weights = DeepSeekV3Model::sanitize_weights_with_args(weights, &text_config);

    let text_model = DeepSeekV3Model::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Kimi-VL text model: {e}"))?;

    let vision_model = KimiVLVisionModel::from_weights(&weights, &vision_config, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load Kimi-VL MoonViT tower: {e}"))?;

    // Connector: kimi_vl uses `multi_modal_projector.{pre_norm,linear_1,linear_2}`;
    // kimi_k25 uses `mm_projector.{pre_norm,proj.0,proj.2}`.
    let projector = build_projector(&weights, merged_hidden, group_size, bits)?;

    let processor = KimiVLProcessor::new(
        vision_config.patch_size,
        [
            vision_config.spatial_merge_size.max(1),
            vision_config.spatial_merge_size.max(1),
        ],
        DEFAULT_IN_TOKEN_LIMIT,
    );

    let vlm = KimiVLModel {
        text_model,
        vision_model,
        projector,
        processor,
        media_placeholder_token_id,
        spatial_merge_size,
        eos_token_ids,
    };

    Ok(LoadedModel::KimiVL(vlm))
}

fn build_projector(
    weights: &WeightMap,
    merged_hidden: i32,
    group_size: i32,
    bits: i32,
) -> Result<KimiVLMultiModalProjector> {
    let (pre_norm, l1, l2) = if weights.contains_key("mm_projector.pre_norm.weight") {
        (
            "mm_projector.pre_norm",
            "mm_projector.proj.0",
            "mm_projector.proj.2",
        )
    } else {
        (
            "multi_modal_projector.pre_norm",
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2",
        )
    };
    KimiVLMultiModalProjector::from_weights(
        weights,
        pre_norm,
        l1,
        l2,
        merged_hidden,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Kimi-VL connector: {e}"))
}

fn parse_eos_token_ids(config: &serde_json::Value) -> Vec<i32> {
    match config.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            n.as_i64().map(|id| vec![id as i32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect(),
        _ => Vec::new(),
    }
}

/// Apply the Kimi-VL weight-key remapping (upstream `Model.sanitize` plus the
/// MoonViT `VisionModel.sanitize` and the language-prefix rewrite).
pub fn remap_kimi_vl_weights(raw: WeightMap) -> WeightMap {
    let mut out = WeightMap::with_capacity(raw.len());

    for (key, value) in raw.into_iter() {
        if key.contains("position_ids") || key.contains("rotary_emb") {
            continue;
        }

        if key.starts_with("vision_tower.") {
            let (new_key, value) = transform_vision_key(&key, value);
            out.insert(new_key, value);
            continue;
        }

        if let Some(rest) = key.strip_prefix("language_model.model.") {
            out.insert(format!("model.{rest}"), value);
        } else if let Some(rest) = key.strip_prefix("language_model.lm_head.") {
            out.insert(format!("lm_head.{rest}"), value);
        } else if let Some(rest) = key.strip_prefix("language_model.") {
            out.insert(rest.to_string(), value);
        } else {
            // Connector (`multi_modal_projector.*` / `mm_projector.*`) and any
            // other top-level keys pass through unchanged.
            out.insert(key, value);
        }
    }

    out
}

fn transform_vision_key(
    key: &str,
    value: mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
) -> (String, mlxcel_core::UniquePtr<mlxcel_core::MlxArray>) {
    // Upstream strips the `encoder.` module segment from vision-tower keys.
    let mut new_key = key.replace("encoder.", "");

    // Rename fused-qkv / output projections onto the `attn.` submodule when the
    // checkpoint stores them flat under the block (`blocks.{i}.wqkv` etc.).
    if new_key.contains("blocks") && !new_key.contains("attn") {
        if new_key.contains("wqkv") {
            new_key = new_key.replace("wqkv", "attn.wqkv");
        } else if new_key.contains("wo.") {
            new_key = new_key.replace("wo.", "attn.wo.");
        }
    }

    // Transpose the conv patch-embed kernel to MLX channel-last layout.
    if new_key.ends_with("patch_embed.proj.weight") {
        let shape = mlxcel_core::array_shape(&value);
        if shape.len() == 4 && !conv2d_weight_is_channel_last(&shape) {
            let transposed = mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1]);
            return (new_key, mlxcel_core::copy(&transposed));
        }
    }

    (new_key, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one() -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
        mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32)
    }

    /// The real kimi-vl-a3b-thinking checkpoint stores `text_config.q_lora_rank`
    /// as JSON `null` (the Moonlight-16B backbone projects the query directly).
    /// This is the exact shape the loader hands to `parse_required_vlm_subconfig`,
    /// which previously failed with "invalid type: null, expected usize".
    #[test]
    fn parses_real_kimi_vl_text_config_with_null_q_lora_rank() {
        let full_config = serde_json::json!({
            "model_type": "kimi_vl",
            "text_config": {
                "model_type": "deepseek_v3",
                "vocab_size": 163840,
                "hidden_size": 2048,
                "intermediate_size": 11264,
                "moe_intermediate_size": 1408,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "kv_lora_rank": 512,
                "q_lora_rank": null,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "qk_nope_head_dim": 128,
                "first_k_dense_replace": 1,
                "rope_theta": 800000.0
            }
        });

        let text_config: DeepSeekV3Config =
            parse_required_vlm_subconfig(&full_config, "text_config", "Kimi-VL text config")
                .expect("real Kimi-VL text config must parse");
        assert_eq!(text_config.q_lora_rank, None);
        assert_eq!(text_config.num_hidden_layers, 27);
    }

    #[test]
    fn remaps_language_and_vision_prefixes() {
        let mut raw = WeightMap::new();
        raw.insert(
            "language_model.model.layers.0.self_attn.q_a_proj.weight".to_string(),
            one(),
        );
        raw.insert("language_model.lm_head.weight".to_string(), one());
        raw.insert(
            "vision_tower.encoder.blocks.0.wqkv.weight".to_string(),
            one(),
        );
        raw.insert("vision_tower.encoder.blocks.0.wo.weight".to_string(), one());
        raw.insert("vision_tower.blocks.0.norm0.weight".to_string(), one());
        raw.insert("multi_modal_projector.linear_1.weight".to_string(), one());
        raw.insert(
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
            one(),
        );

        let out = remap_kimi_vl_weights(raw);

        assert!(out.contains_key("model.layers.0.self_attn.q_a_proj.weight"));
        assert!(out.contains_key("lm_head.weight"));
        assert!(out.contains_key("vision_tower.blocks.0.attn.wqkv.weight"));
        assert!(out.contains_key("vision_tower.blocks.0.attn.wo.weight"));
        assert!(out.contains_key("vision_tower.blocks.0.norm0.weight"));
        assert!(out.contains_key("multi_modal_projector.linear_1.weight"));
        // rotary_emb dropped.
        assert!(!out.keys().any(|k| k.contains("rotary_emb")));
    }

    #[test]
    fn transposes_pytorch_conv_patch_embed_weight() {
        let mut raw = WeightMap::new();
        // PyTorch layout [out, in, kH, kW] = [8, 3, 2, 2] -> should transpose.
        raw.insert(
            "vision_tower.patch_embed.proj.weight".to_string(),
            mlxcel_core::from_slice_f32(&vec![0.0; 8 * 3 * 2 * 2], &[8, 3, 2, 2]),
        );
        let out = remap_kimi_vl_weights(raw);
        let w = out.get("vision_tower.patch_embed.proj.weight").unwrap();
        // MLX channel-last [out, kH, kW, in] = [8, 2, 2, 3].
        assert_eq!(mlxcel_core::array_shape(w), vec![8, 2, 2, 3]);
    }
}
