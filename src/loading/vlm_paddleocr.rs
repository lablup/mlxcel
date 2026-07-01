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

//! PaddleOCR-VL loader (NaViT vision encoder + ERNIE-4.5 MRoPE text decoder).

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use mlxcel_core::weights::WeightMap;

use super::{
    load_vlm_weights_common, parse_required_vlm_subconfig, parse_vlm_config,
    read_sanitized_vlm_config,
};

/// Load a PaddleOCR-VL model.
pub(crate) fn load_paddleocr_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::paddleocr_vl::PaddleOcrProjector;
    use vision::encoders::paddleocr_vl::{PaddleOcrVisionConfig, PaddleOcrVisionEncoder};
    use vision::processors::paddleocr_vl::PaddleOcrVlProcessor;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text params live at the config root (mirrors the reference `from_dict`,
    // which copies every non-`vision_config` key into `text_config`); some
    // exports also nest them under `text_config`.
    let text_config: models::paddleocr_vl::PaddleOcrTextConfig = if full_config
        .get("text_config")
        .and_then(|v| v.get("hidden_size"))
        .is_some()
    {
        parse_required_vlm_subconfig(&full_config, "text_config", "PaddleOCR-VL text config")?
    } else {
        parse_vlm_config(&config_str, "PaddleOCR-VL text config")?
    };

    let mut vision_config: PaddleOcrVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "PaddleOCR-VL vision config")?;

    if let Some(q) = full_config.get("quantization") {
        if let Some(gs) = q.get("group_size").and_then(|v| v.as_i64()) {
            vision_config.quant_group_size = gs as i32;
        }
        if let Some(b) = q.get("bits").and_then(|v| v.as_i64()) {
            vision_config.quant_bits = b as i32;
        }
    }

    let weights = remap_paddleocr_weights(load_vlm_weights_common(model_path, None)?)?;

    let text_model = models::paddleocr_vl::PaddleOcrTextModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load PaddleOCR-VL text model: {}", e))?;

    let vision_encoder =
        PaddleOcrVisionEncoder::from_weights(&weights, &vision_config, "visual")
            .map_err(|e| anyhow::anyhow!("Failed to load PaddleOCR-VL vision encoder: {}", e))?;

    let connector = PaddleOcrProjector::from_weights(
        &weights,
        "visual.projector",
        vision_config.spatial_merge_size,
        vision_config.quant_group_size,
        vision_config.quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load PaddleOCR-VL projector: {}", e))?;

    let processor =
        PaddleOcrVlProcessor::new(vision_config.patch_size, vision_config.spatial_merge_size);

    let read_token = |key: &str, default: i64| -> i32 {
        full_config
            .get(key)
            .and_then(|v| v.as_i64())
            .unwrap_or(default) as i32
    };

    let vlm = vision::PaddleOcrVlModel {
        text_model,
        vision_encoder,
        connector,
        processor,
        image_token_id: read_token("image_token_id", 100295),
        video_token_id: read_token("video_token_id", 100296),
        vision_start_token_id: read_token("vision_start_token_id", 101305),
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::PaddleOcrVL(vlm))
}

/// Remap raw checkpoint keys into mlxcel's PaddleOCR-VL namespace.
///
/// Mirrors the reference `Model.sanitize`:
/// - `visual.vision_model.{embeddings,post_layernorm}` -> `visual.{...}`
/// - `visual.vision_model.encoder.layers` -> `visual.layers`
/// - vision `q_proj`/`k_proj`/`v_proj` -> fused `qkv` (concatenated on axis 0)
/// - `mlp_AR` -> `visual.projector`
/// - drop `packing_position_embedding`, `vision_model.head`, `position_ids`
/// - text (`model.*`) and `lm_head.*` pass through unchanged
fn remap_paddleocr_weights(mut raw: WeightMap) -> Result<WeightMap> {
    let mut out = WeightMap::new();
    let keys: Vec<String> = raw.keys().cloned().collect();

    for key in keys {
        if key.contains("packing_position_embedding")
            || key.contains("vision_model.head")
            || key.contains("position_ids")
        {
            continue;
        }

        let is_visual = key.contains("visual");
        if is_visual && (key.contains("k_proj") || key.contains("v_proj")) {
            // Consumed by the fused qkv path below.
            continue;
        }

        if is_visual && key.contains("q_proj") {
            let k_key = key.replace("q_proj", "k_proj");
            let v_key = key.replace("q_proj", "v_proj");
            let q = raw
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("missing {}", key))?;
            let k = raw
                .get(&k_key)
                .ok_or_else(|| anyhow::anyhow!("missing {}", k_key))?;
            let v = raw
                .get(&v_key)
                .ok_or_else(|| anyhow::anyhow!("missing {}", v_key))?;
            let qk = mlxcel_core::concatenate(q.as_ref().unwrap(), k.as_ref().unwrap(), 0);
            let qkv = mlxcel_core::concatenate(qk.as_ref().unwrap(), v.as_ref().unwrap(), 0);
            let new_key = remap_key(&key).replace("q_proj", "qkv");
            out.insert(new_key, qkv);
            continue;
        }

        let new_key = remap_key(&key);
        if let Some(value) = raw.remove(&key) {
            out.insert(new_key, value);
        }
    }

    Ok(out)
}

fn remap_key(key: &str) -> String {
    if key.contains("visual.vision_model.encoder") {
        key.replace("visual.vision_model.encoder", "visual")
    } else if key.contains("visual.vision_model") {
        key.replace("visual.vision_model", "visual")
    } else if let Some(rest) = key.strip_prefix("mlp_AR") {
        format!("visual.projector{rest}")
    } else {
        key.to_string()
    }
}
