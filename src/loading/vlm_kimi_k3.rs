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

//! Kimi K3 VLM loader (`model_type: "kimi_k3"` with `vision_config` and
//! `vision_tower.*` tensors, issue #1342).
//!
//! Where this diverges from the text-only route (`loading::special`,
//! `SpecialWeightLoaderKind::KimiK3`):
//!
//! - The `vision_tower.*` and `mm_projector.*` tensors are split off **before**
//!   `KimiK3Model::sanitize_weights` runs, since that sanitizer drops every
//!   key outside `model.` / `lm_head.` (the text-only route keeps dropping
//!   them). The text sanitize itself is reused unchanged.
//! - Weights are read through the index-filtered loader so a layer-truncated
//!   local copy whose `config.json` lowers `num_hidden_layers` never mmap-opens
//!   the shards of the layers it drops.
//! - The Apple Silicon dtype policy uses `config_has_quantization_metadata`,
//!   the text path's predicate: the published checkpoint declares its mxfp4
//!   experts as a compressed-tensors `quantization_config` under
//!   `text_config` and no MLX `quantization` block, so it counts as quantized
//!   and nothing is promoted from bf16 to f16. The tower and projector
//!   therefore run in bf16 next to the bf16 attention and dense planes of the
//!   text side, exactly as the text-only route leaves them.
//!
//! Surgery pipelines are not applied on this route.

use anyhow::{Result, anyhow};
use mlxcel_core::weights::WeightMap;
use std::path::Path;

use crate::LoadedModel;
use crate::models::kimi_k3::KimiK3Config;
use crate::models::{KimiK3Model, bf16_to_f16_at_load, config_has_quantization_metadata};
use crate::multimodal::kimi_k3_prompt::read_media_token_ids;
use crate::vision::encoders::moonvit3d::{MoonViT3DConfig, MoonViT3DVisionModel};
use crate::vision::kimi_k3_vl::{KimiK3VLModel, sanitize_kimi_k3_vision_weights};
use crate::vision::processors::kimi_k3::KimiK3ImageProcessor;

use super::{parse_required_vlm_subconfig, read_sanitized_vlm_config};

const VISION_PREFIXES: [&str; 2] = ["vision_tower.", "mm_projector."];

/// `true` for a raw checkpoint key the VLM route reads: everything except the
/// MTP heads and the text layers at or past `num_hidden_layers`.
pub(crate) fn keep_kimi_k3_vlm_weight(key: &str, num_hidden_layers: usize) -> bool {
    let key = key.strip_prefix("language_model.").unwrap_or(key);
    if key.starts_with("model.mtp") {
        return false;
    }
    if let Some(rest) = key.strip_prefix("model.layers.") {
        let layer: Option<usize> = rest.split('.').next().and_then(|n| n.parse().ok());
        if let Some(n) = layer
            && n >= num_hidden_layers
        {
            return false;
        }
    }
    true
}

pub(crate) fn is_kimi_k3_vision_key(key: &str) -> bool {
    VISION_PREFIXES.iter().any(|p| key.starts_with(p))
}

/// Split the raw map into `(vision, text)`.
pub(crate) fn split_kimi_k3_vision_weights(weights: WeightMap) -> (WeightMap, WeightMap) {
    let mut vision = WeightMap::new();
    let mut text = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        if is_kimi_k3_vision_key(&key) {
            vision.insert(key, value);
        } else {
            text.insert(key, value);
        }
    }
    (vision, text)
}

pub(crate) fn load_kimi_k3_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let config = KimiK3Config::from_json_str(&config_str).map_err(|e| anyhow!("{e}"))?;
    let vision_config: MoonViT3DConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Kimi K3 vision config")?;
    vision_config.validate().map_err(|e| anyhow!("{e}"))?;
    if vision_config.text_hidden_size != config.text_config.hidden_size {
        return Err(anyhow!(
            "Kimi K3: vision_config.text_hidden_size ({}) does not match text_config.hidden_size ({})",
            vision_config.text_hidden_size,
            config.text_config.hidden_size
        ));
    }
    let media_placeholder_token_id = config
        .media_placeholder_token_id
        .ok_or_else(|| anyhow!("Kimi K3: config.json has no media_placeholder_token_id"))?
        as i32;
    let media_token_ids = read_media_token_ids(model_path)?;
    if media_token_ids.pad != media_placeholder_token_id {
        return Err(anyhow!(
            "Kimi K3: media_placeholder_token_id {media_placeholder_token_id} does not match the \
             tokenizer's <|media_pad|> id {}",
            media_token_ids.pad
        ));
    }
    let processor = KimiK3ImageProcessor::from_model_dir(model_path).map_err(|e| anyhow!("{e}"))?;

    let num_layers = config.text_config.num_hidden_layers;
    println!("[KimiK3-VLM] Loading weights...");
    let mut weights = mlxcel_core::weights::load_weights_from_dir_index_filtered(model_path, |k| {
        keep_kimi_k3_vlm_weight(k, num_layers)
    })
    .map_err(|e| anyhow!("{e}"))?;

    if bf16_to_f16_at_load(
        config_has_quantization_metadata(&full_config),
        Some(&full_config),
    ) && crate::models::convert_bf16_weights(&mut weights)
    {
        crate::models::warn_bf16_precision();
    }

    let (vision_weights, text_weights) = split_kimi_k3_vision_weights(weights);
    if vision_weights.is_empty() {
        return Err(anyhow!(
            "Kimi K3: no vision_tower.* / mm_projector.* tensors in {}; load it as the text \
             backbone instead",
            model_path.display()
        ));
    }

    println!("[KimiK3-VLM] Building text backbone ({num_layers} layers)...");
    let text_weights = KimiK3Model::sanitize_weights(text_weights, &config.text_config)
        .map_err(|e| anyhow!("{e}"))?;
    let mut text = KimiK3Model::from_weights(&text_weights, &config.text_config)
        .map_err(|e| anyhow!("Failed to load Kimi K3 text backbone: {e}"))?;
    drop(text_weights);
    text.set_eos_token_ids(config.eos_token_ids());
    text.set_eos_token_ids(crate::loading::read_eos_token_ids(model_path));

    println!(
        "[KimiK3-VLM] Building MoonViT3D tower ({} blocks) and patchmergerv2 projector...",
        vision_config.vt_num_hidden_layers
    );
    let vision_weights = sanitize_kimi_k3_vision_weights(vision_weights);
    let vision =
        MoonViT3DVisionModel::from_weights(&vision_weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow!("Failed to load Kimi K3 MoonViT3D tower: {e}"))?;
    let projector = KimiK3VLModel::projector_from_weights(&vision_weights, &vision_config)
        .map_err(|e| anyhow!("Failed to load Kimi K3 projector: {e}"))?;

    println!("[KimiK3-VLM] Model loaded successfully");
    Ok(LoadedModel::KimiK3VLM(KimiK3VLModel {
        text,
        vision,
        projector,
        processor,
        media_placeholder_token_id,
        media_token_ids,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_vision_and_in_range_text_layers_only() {
        assert!(keep_kimi_k3_vlm_weight(
            "vision_tower.patch_embed.proj.weight",
            4
        ));
        assert!(keep_kimi_k3_vlm_weight("mm_projector.proj.0.weight", 4));
        assert!(keep_kimi_k3_vlm_weight(
            "language_model.model.embed_tokens.weight",
            4
        ));
        assert!(keep_kimi_k3_vlm_weight(
            "language_model.model.layers.3.mlp.gate.weight",
            4
        ));
        assert!(!keep_kimi_k3_vlm_weight(
            "language_model.model.layers.4.mlp.gate.weight",
            4
        ));
        assert!(!keep_kimi_k3_vlm_weight(
            "model.layers.92.self_attn.o_proj.weight",
            4
        ));
        assert!(!keep_kimi_k3_vlm_weight(
            "language_model.model.mtp.layers.0.x",
            4
        ));
    }

    #[test]
    fn splits_vision_keys_from_text_keys() {
        let one = || mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32);
        let mut raw = WeightMap::new();
        raw.insert("vision_tower.encoder.blocks.0.wqkv.weight".into(), one());
        raw.insert("mm_projector.post_norm.weight".into(), one());
        raw.insert("language_model.model.norm.weight".into(), one());
        let (vision, text) = split_kimi_k3_vision_weights(raw);
        assert_eq!(vision.len(), 2);
        assert_eq!(text.len(), 1);
        assert!(text.contains_key("language_model.model.norm.weight"));
    }
}
