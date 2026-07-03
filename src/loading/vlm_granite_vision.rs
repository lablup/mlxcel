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

//! Granite Vision loader (SigLIP multi-tap tower + Granite text backbone).
//!
//! Routes both checkpoint spellings: `model_type: "granite_vision"` and the
//! original `model_type: "llava_next"` with a `granite` text config.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

/// Default LLaVA-Next `image_grid_pinpoints` for Granite Vision (27 `(h, w)`
/// pairs), used when the config omits them.
fn default_pinpoints() -> Vec<(i32, i32)> {
    let mut p: Vec<(i32, i32)> = Vec::new();
    for w in (384..=3840).step_by(384) {
        p.push((384, w));
    }
    for w in (384..=1920).step_by(384) {
        p.push((768, w));
    }
    for w in (384..=1152).step_by(384) {
        p.push((1152, w));
    }
    for h in [1536, 1920] {
        for w in [384, 768] {
            p.push((h, w));
        }
    }
    for h in [2304, 2688, 3072, 3456, 3840] {
        p.push((h, 384));
    }
    p
}

fn parse_pinpoints(full_config: &serde_json::Value) -> Vec<(i32, i32)> {
    full_config
        .get("image_grid_pinpoints")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|pair| {
                    let p = pair.as_array()?;
                    Some((p.first()?.as_i64()? as i32, p.get(1)?.as_i64()? as i32))
                })
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(default_pinpoints)
}

/// `vision_feature_layer` may be a list or a single int; normalize to a list.
fn parse_feature_layers(full_config: &serde_json::Value) -> Vec<i32> {
    match full_config.get("vision_feature_layer") {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|i| i as i32))
            .collect(),
        Some(serde_json::Value::Number(n)) => vec![n.as_i64().unwrap_or(-1) as i32],
        _ => vec![-24, -20, -12, -1],
    }
}

/// Load a Granite Vision VLM.
pub(crate) fn load_granite_vision_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text backbone config: inherit top-level quantization, then parse as Granite.
    let mut text_config_val = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in config.json"))?;
    if let Some(obj) = text_config_val.as_object_mut()
        && obj.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        obj.insert("quantization".to_string(), q.clone());
    }
    let text_args: models::granite::ModelArgs = serde_json::from_value(text_config_val)
        .map_err(|e| anyhow::anyhow!("Failed to parse Granite text config: {}", e))?;

    let vision_config: vision::config::VisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config in config.json"))?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse Granite vision config: {}", e))?;

    let group_size = text_args.group_size();
    let bits = text_args.bits();

    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::GraniteModel::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Granite text model: {}", e))?;

    // Vision tower (plain bf16 in the released checkpoint; UnifiedLinear loads it
    // as a regular Linear when no `.scales` companion is present).
    let vision_tower = vision::encoders::siglip::SigLipVisionModel::from_weights_with_quant(
        &weights,
        &vision_config,
        "vision_tower.vision_model",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Granite vision tower: {}", e))?;

    let projector = vision::connectors::mlp::MLPProjector::from_weights(
        &weights,
        "multi_modal_projector",
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Granite projector: {}", e))?;

    let image_newline = weights
        .get("image_newline")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing image_newline weight"))?;

    let image_token_index = full_config
        .get("image_token_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(49155) as i32;
    let pinpoints = parse_pinpoints(&full_config);
    let feature_layers = parse_feature_layers(&full_config);

    let image_size = vision_config.image_size as u32;
    let feature_side = (vision_config.image_size / vision_config.patch_size) as i32;
    let base_tokens = feature_side * feature_side;
    let processor = vision::processors::anyres::AnyResProcessor::new(pinpoints, image_size);

    let vlm = vision::GraniteVisionVLModel {
        text_model,
        vision_tower,
        projector,
        image_newline,
        processor,
        image_token_index,
        vision_feature_layers: feature_layers,
        feature_side,
        base_tokens,
        eos_token_id: 0,
    };

    Ok(LoadedModel::GraniteVisionVLM(vlm))
}
