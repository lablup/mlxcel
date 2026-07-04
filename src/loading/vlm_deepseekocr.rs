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

//! DeepSeek-OCR loader: SAM + CLIP towers, linear projector, `image_newline` /
//! `view_separator`, and the shared DeepSeek MoE text decoder.

use anyhow::Result;
use serde_json::json;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::encoders::deepseekocr_clip::{ClipConfig, ClipEncoder};
use crate::vision::encoders::deepseekocr_sam::{SamConfig, SamEncoder};
use crate::vision::processors::deepseekocr::DeepSeekOcrProcessor;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

const IMAGE_TOKEN_ID: i32 = 128815;
const N_EMBED: i32 = 1280;

/// Rename the misspelled `view_seperator` and strip a leading `model.` from the
/// vision / projector keys (layout B). Layout A keys are already canonical, so
/// this is a no-op there.
fn remap_deepseekocr_weights(weights: WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (k, v) in weights {
        let key = if k == "model.view_seperator" || k == "view_seperator" {
            "view_separator".to_string()
        } else if let Some(rest) = k.strip_prefix("model.") {
            if rest.starts_with("sam_model.")
                || rest.starts_with("vision_model.")
                || rest.starts_with("projector.")
                || rest == "image_newline"
                || rest == "view_separator"
            {
                rest.to_string()
            } else {
                k
            }
        } else {
            k
        };
        out.insert(key, v);
    }
    out
}

pub(crate) fn load_deepseekocr_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text decoder config from `language_config`, inheriting the root
    // quantization block and a `model_type` (the sub-config omits both).
    let mut lc = full_config
        .get("language_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing language_config in DeepSeek-OCR config"))?;
    if let Some(obj) = lc.as_object_mut() {
        obj.entry("model_type".to_string())
            .or_insert_with(|| json!("deepseek"));
        if !obj.contains_key("quantization")
            && let Some(q) = full_config.get("quantization")
        {
            obj.insert("quantization".to_string(), q.clone());
        }
    }
    let args: models::deepseek::ModelArgs = serde_json::from_value(lc)
        .map_err(|e| anyhow::anyhow!("Failed to parse DeepSeek-OCR language_config: {}", e))?;
    let (gs, bits) = (args.group_size(), args.bits());

    let weights = remap_deepseekocr_weights(load_vlm_weights_common(model_path, None)?);

    let sam = SamEncoder::from_weights(&weights, "sam_model", SamConfig::default())
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR SAM encoder: {}", e))?;
    let clip = ClipEncoder::from_weights(&weights, "vision_model", ClipConfig::default())
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR CLIP encoder: {}", e))?;
    let projector = UnifiedLinear::from_weights(&weights, "projector.layers", gs, bits)
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR projector: {}", e))?;
    let image_newline = weights
        .get("image_newline")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing image_newline"))?;
    let view_separator = weights
        .get("view_separator")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing view_separator"))?;

    // The text backbone consumes the `model.*` / `lm_head` keys (layout A ships
    // them under `language_model.`); the vision keys are already loaded by ref.
    let text_weights = strip_language_model_prefix(weights);
    let text_model = models::deepseek::DeepSeekModel::from_weights(&text_weights, &args)
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR text model: {}", e))?;

    let vlm = vision::deepseekocr::DeepSeekOcrVlModel {
        text_model,
        sam,
        clip,
        projector,
        image_newline,
        view_separator,
        processor: DeepSeekOcrProcessor::default(),
        image_token_id: IMAGE_TOKEN_ID,
        eos_token_id: 1,
        n_embed: N_EMBED,
    };
    Ok(LoadedModel::DeepSeekOcrVLM(vlm))
}
