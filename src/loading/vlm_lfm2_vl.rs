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

//! LFM2-VL (`lfm2_vl` / `lfm2-vl`) VLM loader.
//!
//! Layout of `mlx-community/LFM2-VL-*` checkpoints: the text backbone lives under
//! `language_model.model.*` (stripped to the plain `model.*` layout
//! [`Lfm2Model::from_weights`] expects), the vision tower under `vision_tower.*`
//! (all weights plain BF16, not quantized), and the projector under
//! `multi_modal_projector.*` (quantized linears + a plain LayerNorm). The text
//! embedding and projector linears are 4-bit; the vision tower is not, and
//! `UnifiedLinear` loads plain tensors as a regular `Linear` when no `.scales`
//! companion is present.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::encoders::lfm2_vl::Lfm2VlVisionConfig;

use super::{load_vlm_weights_common, read_sanitized_vlm_config};

const DEFAULT_IMAGE_TOKEN_ID: i32 = 396;
const DEFAULT_IMAGE_START_ID: i32 = 498;
const DEFAULT_IMAGE_END_ID: i32 = 499;

fn get_usize(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(default)
}

fn parse_vision_config(full_config: &Value) -> Lfm2VlVisionConfig {
    let vc = full_config
        .get("vision_config")
        .cloned()
        .unwrap_or(Value::Null);
    let d = Lfm2VlVisionConfig::default();
    Lfm2VlVisionConfig {
        hidden_size: get_usize(&vc, "hidden_size", d.hidden_size),
        intermediate_size: get_usize(&vc, "intermediate_size", d.intermediate_size),
        num_hidden_layers: get_usize(&vc, "num_hidden_layers", d.num_hidden_layers),
        num_attention_heads: get_usize(&vc, "num_attention_heads", d.num_attention_heads),
        patch_size: get_usize(&vc, "patch_size", d.patch_size),
        num_patches: get_usize(&vc, "num_patches", d.num_patches),
        layer_norm_eps: vc
            .get("layer_norm_eps")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(d.layer_norm_eps),
        // vision_feature_layer is a top-level key, not in vision_config.
        vision_feature_layer: full_config
            .get("vision_feature_layer")
            .and_then(|x| x.as_i64())
            .map(|x| x as i32)
            .unwrap_or(d.vision_feature_layer),
    }
}

/// Strip the `language_model.` prefix so the LFM2 backbone sees its canonical
/// `model.*` layout (it applies its own sanitize internally).
fn lfm2_text_weights(weights: &mlxcel_core::weights::WeightMap) -> mlxcel_core::weights::WeightMap {
    let mut out = mlxcel_core::weights::WeightMap::new();
    for (key, value) in weights.iter() {
        if let Some(rest) = key.strip_prefix("language_model.") {
            out.insert(rest.to_string(), mlxcel_core::copy(value));
        }
    }
    out
}

/// Load an LFM2-VL VLM (packed-patch ViT + pixel-unshuffle projector + LFM2
/// hybrid text backbone).
pub(crate) fn load_lfm2_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::lfm2_vl::Lfm2VlConnector;
    use vision::encoders::lfm2_vl::Lfm2VlVisionTower;
    use vision::lfm2_vl::Lfm2VlModel;
    use vision::processors::lfm2_vl::Lfm2VlProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text backbone args from `text_config`, inheriting the top-level quantization.
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in LFM2-VL config.json"))?;
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(&mut text_config_value, "LFM2-VL text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    let text_args: models::lfm2::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse LFM2-VL text_config: {}", e))?;

    let vision_config = parse_vision_config(&full_config);
    let gs = text_args.group_size();
    let bits = text_args.bits();

    // Load weights; convert plain bf16 tensors (vision tower + projector norm +
    // linear biases) to f16 on Apple Silicon, keeping quant scales/biases.
    let mut weights = load_vlm_weights_common(model_path, None)?;
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    // Text backbone from the stripped `model.*` subset.
    let text_weights = lfm2_text_weights(&weights);
    let text_model = models::lfm2::Lfm2Model::from_weights(text_args.clone(), text_weights)
        .map_err(|e| anyhow::anyhow!("Failed to load LFM2-VL text backbone: {}", e))?;

    // Vision tower + connector.
    let vision_tower =
        Lfm2VlVisionTower::from_weights(&weights, "vision_tower", &vision_config, gs, bits)
            .map_err(|e| anyhow::anyhow!("Failed to load LFM2-VL vision tower: {}", e))?;

    let downsample_factor = full_config
        .get("downsample_factor")
        .and_then(|v| v.as_i64())
        .unwrap_or(2) as i32;
    let use_layernorm = full_config
        .get("projector_use_layernorm")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let connector = Lfm2VlConnector::from_weights(
        &weights,
        "multi_modal_projector",
        downsample_factor,
        use_layernorm,
        gs,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load LFM2-VL projector: {}", e))?;

    // Processor.
    let patch_size = full_config
        .get("encoder_patch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(vision_config.patch_size as u64) as usize;
    let min_tokens = get_usize(&full_config, "min_image_tokens", 64);
    let max_tokens = get_usize(&full_config, "max_image_tokens", 256);
    let processor = Lfm2VlProcessor::new(
        patch_size,
        downsample_factor as usize,
        min_tokens,
        max_tokens,
    );

    let image_token_id = full_config
        .get("image_token_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID as i64) as i32;
    let use_image_special_tokens = full_config
        .get("use_image_special_tokens")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let eos_token_ids = text_args.eos_token_ids();

    let vlm = Lfm2VlModel {
        text_model,
        vision_tower,
        connector,
        processor,
        image_token_id,
        image_start_id: DEFAULT_IMAGE_START_ID,
        image_end_id: DEFAULT_IMAGE_END_ID,
        use_image_special_tokens,
        downsample_factor,
        patch_dim: (patch_size * patch_size * 3) as i32,
        eos_token_ids,
    };

    Ok(LoadedModel::Lfm2VL(vlm))
}
