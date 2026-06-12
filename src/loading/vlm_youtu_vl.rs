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

//! Youtu-VL loader.
//!
//! Mirrors https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/youtu_vl.py (Model.sanitize)
//! and `Model.__init__` to translate a HuggingFace `model_type = "youtu_vl"`
//! safetensors checkpoint into a fully-wired [`crate::vision::YoutuVLModel`].
//!
//! Weight-name mapping performed here:
//! - `siglip2.vision_model.<rest>` → `vision_tower.<rest>`
//! - `siglip2.<rest>`              → `vision_tower.<rest>`
//! - `merger.<rest>`               → `vision_tower.merger.<rest>`
//! - `model.<rest>`                → `<rest>` (we drop the `language_model.`
//!   prefix that `Model.sanitize` otherwise injects, since the language
//!   model exposed by this crate consumes keys directly under `model.*`).
//! - `lm_head.<rest>`              → `lm_head.<rest>` (kept; dropped later
//!   when `tie_word_embeddings = true`).
//! - `position_ids` / `position_embedding` keys are dropped (RoPE replaces
//!   them).
//!
//! The MLA `kv_b_proj` decomposition is delegated to
//! [`crate::models::youtu_vl_lm::sanitize_text_weights`].

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use std::path::Path;

use crate::LoadedModel;
use crate::models::youtu_vl_lm::{YoutuLanguageModel, YoutuTextConfig, sanitize_text_weights};
use crate::vision::YoutuVLModel;
use crate::vision::encoders::youtu_vl::{YoutuVLVisionEncoder, YoutuVisionConfig};
use crate::vision::processors::youtu_vl::YoutuVLProcessor;

use super::{
    load_vlm_weights_common, parse_required_vlm_subconfig, parse_vlm_config,
    read_sanitized_vlm_config,
};

/// Default token IDs from upstream `youtu_vl/config.py::ModelConfig`.
const DEFAULT_IMAGE_TOKEN_ID: i32 = 128_264;
const DEFAULT_VIDEO_TOKEN_ID: i32 = 128_265;
const DEFAULT_VISION_START_TOKEN_ID: i32 = 128_262;
const DEFAULT_VISION_END_TOKEN_ID: i32 = 128_263;

pub(crate) fn load_youtu_vl_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Youtu-VL flattens its text fields at the root of `config.json` (mirroring
    // the `from_dict` classmethod that copies non-vision keys into `text_config`).
    // Parsing the root struct directly therefore yields the language config.
    let mut text_config: YoutuTextConfig = parse_vlm_config(&config_str, "Youtu-VL text config")?;

    // Carry root-level quantization into the text config when the safetensors
    // were quantized but the inline `quantization` key sat one level above
    // `text_config` (matches how upstream Python loaders treat quantization).
    if text_config.quantization.is_none()
        && let Some(qjson) = full_config.get("quantization").cloned()
        && let Ok(parsed) =
            serde_json::from_value::<crate::models::youtu_vl_lm::QuantizationConfig>(qjson)
    {
        text_config.quantization = Some(parsed);
    }

    let mut vision_config: YoutuVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Youtu-VL vision config")?;

    // Inherit quantization params for the vision tower from the top-level
    // `quantization` block (the vision sub-config typically does not repeat
    // them).
    if vision_config.quant_bits == 0
        && let Some(q) = full_config.get("quantization")
    {
        if let Some(group_size) = q.get("group_size").and_then(|v| v.as_i64()) {
            vision_config.quant_group_size = group_size as i32;
        }
        if let Some(bits) = q.get("bits").and_then(|v| v.as_i64()) {
            vision_config.quant_bits = bits as i32;
        }
    }

    // Top-level token ids — fall back to the defaults defined by upstream when
    // the checkpoint config omits them.
    let image_token_id = full_config
        .get("image_token_id")
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID);
    let video_token_id = full_config
        .get("video_token_id")
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or(DEFAULT_VIDEO_TOKEN_ID);
    let vision_start_token_id = full_config
        .get("vision_start_token_id")
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or(DEFAULT_VISION_START_TOKEN_ID);
    let vision_end_token_id = full_config
        .get("vision_end_token_id")
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or(DEFAULT_VISION_END_TOKEN_ID);

    let eos_token_ids = parse_eos_token_ids(&full_config);

    // Load raw weights and run the Youtu-VL key remapping.
    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let weights = remap_youtu_vl_weights(raw_weights);
    let weights = sanitize_text_weights(weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to sanitize Youtu-VL text weights: {}", e))?;

    // Build language model.
    let language_model = YoutuLanguageModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Youtu-VL text model: {}", e))?
        .with_eos_token_ids(eos_token_ids);

    // Build vision tower (rooted at `vision_tower.*` after remapping).
    let vision_encoder =
        YoutuVLVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Youtu-VL vision tower: {}", e))?;

    // Build processor — read normalization from `preprocessor_config.json` if
    // present, otherwise fall back to SigLIP2 defaults.
    let processor = build_processor(model_path, &vision_config);

    let vlm = YoutuVLModel {
        text_model: language_model,
        vision_encoder,
        processor,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        vision_end_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::YoutuVL(vlm))
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

/// Apply Youtu-VL's vision-side weight-key remapping. This mirrors the first
/// half of `Model.sanitize` in upstream `youtu_vl.py`. The MLA decomposition
/// step is split out into [`sanitize_text_weights`] because it depends on the
/// text config dimensions.
///
/// Used by: `load_youtu_vl_vlm` (this module), Youtu-VL test harness.
pub fn remap_youtu_vl_weights(raw: WeightMap) -> WeightMap {
    let mut out = WeightMap::with_capacity(raw.len());

    for (key, value) in raw.into_iter() {
        // Skip RoPE-replaced position tables.
        if key.contains("position_ids") || key.contains("position_embedding") {
            continue;
        }

        let new_key = transform_youtu_vl_key(&key);
        out.insert(new_key, value);
    }

    out
}

fn transform_youtu_vl_key(key: &str) -> String {
    // siglip2.vision_model.* → vision_tower.*
    if let Some(rest) = key.strip_prefix("siglip2.vision_model.") {
        return format!("vision_tower.{rest}");
    }
    if let Some(rest) = key.strip_prefix("siglip2.") {
        return format!("vision_tower.{rest}");
    }

    // merger.* → vision_tower.merger.*
    if let Some(rest) = key.strip_prefix("merger.") {
        return format!("vision_tower.merger.{rest}");
    }

    // model.* — keep the prefix because our YoutuLanguageModel consumes
    // weights at `model.<...>` directly. Upstream rewrites this to
    // `language_model.model.<...>` to mirror Python's nested module layout,
    // but the Rust implementation expects the original path.
    if key.starts_with("model.") || key == "model" {
        return key.to_string();
    }
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        return format!("model.{rest}");
    }
    if let Some(rest) = key.strip_prefix("language_model.lm_head.") {
        return format!("lm_head.{rest}");
    }

    // lm_head.* — kept; sanitize_text_weights drops it later when ties.
    if key.starts_with("lm_head.") {
        return key.to_string();
    }

    // Unmodified.
    key.to_string()
}

fn build_processor(model_path: &Path, vision_config: &YoutuVisionConfig) -> YoutuVLProcessor {
    let mut processor =
        YoutuVLProcessor::new(vision_config.patch_size, vision_config.spatial_merge_size)
            .with_max_patches_per_image(vision_config.num_patches);

    // Try to read `preprocessor_config.json` for `image_mean`, `image_std`,
    // and any min/max pixel hints. Fail silently (use defaults) if anything
    // is missing or malformed — the SigLIP2 defaults still produce valid
    // numerical output.
    let preproc_path = model_path.join("preprocessor_config.json");
    let Ok(text) = std::fs::read_to_string(&preproc_path) else {
        return processor;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return processor;
    };

    if let Some(mean_arr) = json.get("image_mean").and_then(|v| v.as_array())
        && mean_arr.len() == 3
    {
        let mean = [
            mean_arr[0].as_f64().unwrap_or(0.5) as f32,
            mean_arr[1].as_f64().unwrap_or(0.5) as f32,
            mean_arr[2].as_f64().unwrap_or(0.5) as f32,
        ];
        let std = json
            .get("image_std")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 3 {
                    Some([
                        arr[0].as_f64().unwrap_or(0.5) as f32,
                        arr[1].as_f64().unwrap_or(0.5) as f32,
                        arr[2].as_f64().unwrap_or(0.5) as f32,
                    ])
                } else {
                    None
                }
            })
            .unwrap_or([0.5, 0.5, 0.5]);
        processor = processor.with_norm(mean, std);
    }

    let min_pixels = json.get("min_pixels").and_then(|v| v.as_u64());
    let max_pixels = json.get("max_pixels").and_then(|v| v.as_u64());
    if let (Some(min_p), Some(max_p)) = (min_pixels, max_pixels) {
        processor = processor.with_pixel_bounds(min_p as usize, max_p as usize);
    }
    if let Some(num_patches) = json.get("num_patches").and_then(|v| v.as_u64()) {
        processor = processor.with_max_patches_per_image(num_patches as usize);
    }

    processor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_remap_handles_youtu_vl_prefixes() {
        let mut raw = WeightMap::new();
        let one = mlxcel_core::ones(&[1], mlxcel_core::dtype::FLOAT32);

        raw.insert(
            "siglip2.vision_model.embeddings.patch_embedding.weight".to_string(),
            mlxcel_core::copy(&one),
        );
        raw.insert(
            "siglip2.vision_model.encoder.layers.0.self_attn.q_proj.weight".to_string(),
            mlxcel_core::copy(&one),
        );
        raw.insert("merger.ln_q.weight".to_string(), mlxcel_core::copy(&one));
        raw.insert("merger.mlp.0.weight".to_string(), mlxcel_core::copy(&one));
        raw.insert(
            "model.layers.0.self_attn.q_a_proj.weight".to_string(),
            mlxcel_core::copy(&one),
        );
        raw.insert("lm_head.weight".to_string(), mlxcel_core::copy(&one));
        raw.insert(
            "model.embed_positions.position_ids".to_string(),
            mlxcel_core::copy(&one),
        );

        let out = remap_youtu_vl_weights(raw);

        assert!(out.contains_key("vision_tower.embeddings.patch_embedding.weight"));
        assert!(out.contains_key("vision_tower.encoder.layers.0.self_attn.q_proj.weight"));
        assert!(out.contains_key("vision_tower.merger.ln_q.weight"));
        assert!(out.contains_key("vision_tower.merger.mlp.0.weight"));
        // Language model keys keep the `model.` prefix as expected by
        // YoutuLanguageModel::from_weights.
        assert!(out.contains_key("model.layers.0.self_attn.q_a_proj.weight"));
        assert!(out.contains_key("lm_head.weight"));
        // Position ids should be stripped.
        assert!(!out.keys().any(|k| k.contains("position_ids")));
    }
}
