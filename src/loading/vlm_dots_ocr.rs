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

//! dots.ocr loader (`dots_vit` vision tower + Qwen2 text decoder).

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::encoders::dots_ocr::{DotsVisionConfig, DotsVisionEncoder};
use crate::vision::processors::dots_ocr::DotsOcrProcessor;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};

/// Remap raw checkpoint keys into the layout the text and vision loaders expect.
///
/// - strip the `language_model.` wrapper so the Qwen2 backbone sees `model.*`
/// - fold the nested `model.lm_head.*` up to the top-level `lm_head.*` the
///   untied-head loader looks for
/// - strip a leading `model.` from `model.vision_tower.*` (alternative nesting)
/// - drop `position_ids` / rotary `inv_freq` buffers
fn remap_dots_ocr_weights(weights: WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (k, v) in weights {
        if k.contains("position_ids") || k.contains("inv_freq") {
            continue;
        }
        let k = if let Some(rest) = k.strip_prefix("language_model.") {
            rest.to_string()
        } else if let Some(rest) = k.strip_prefix("model.vision_tower.") {
            format!("vision_tower.{rest}")
        } else {
            k
        };
        let k = if let Some(rest) = k.strip_prefix("model.lm_head.") {
            format!("lm_head.{rest}")
        } else {
            k
        };
        out.insert(k, v);
    }
    out
}

pub(crate) fn load_dots_ocr_vl(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text params sit flat at the config root (no `text_config` block).
    let text_args: models::qwen2::ModelArgs =
        super::parse_vlm_config(&config_str, "dots.ocr text")?;

    let mut vision_config: DotsVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "dots.ocr vision config")?;
    if let Some(q) = full_config.get("quantization") {
        if let Some(gs) = q.get("group_size").and_then(|v| v.as_i64()) {
            vision_config.quant_group_size = gs as i32;
        }
        if let Some(b) = q.get("bits").and_then(|v| v.as_i64()) {
            vision_config.quant_bits = b as i32;
        }
    }

    let weights = remap_dots_ocr_weights(load_vlm_weights_common(model_path, None)?);

    let text_model = models::qwen2::Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load dots.ocr text model: {}", e))?;

    let vision_encoder = DotsVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load dots.ocr vision encoder: {}", e))?;

    let read_token = |key: &str, default: i64| -> i32 {
        full_config
            .get(key)
            .and_then(|v| v.as_i64())
            .unwrap_or(default) as i32
    };

    let mut eos_token_ids = crate::loading::read_eos_token_ids(model_path);
    if eos_token_ids.is_empty() {
        eos_token_ids = vec![151643, 151673];
    }

    let vlm = vision::dots_ocr::DotsOcrVlModel {
        text_model,
        vision_encoder,
        processor: DotsOcrProcessor::default(),
        image_token_id: read_token("image_token_id", 151665),
        video_token_id: read_token("video_token_id", 151656),
        vision_start_token_id: 151666,
        spatial_merge_size: vision_config.spatial_merge_size,
        eos_token_ids,
    };
    Ok(LoadedModel::DotsOcrVL(vlm))
}
