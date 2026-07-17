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

//! DeepSeek-OCR loaders. V1: SAM + CLIP towers, linear projector,
//! `image_newline` / `view_separator`. V2: SAM (896-channel compressor) + Qwen2
//! query resampler, linear projector, `view_separator`. Both feed the shared
//! DeepSeek MoE text decoder.

use anyhow::Result;
use serde_json::json;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::encoders::deepseekocr_clip::{ClipConfig, ClipEncoder};
use crate::vision::encoders::deepseekocr_qwen2::{Qwen2Resampler, Qwen2ResamplerConfig};
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

const DEFAULT_SLIDING_WINDOW: i32 = 128;

/// Load `baidu/Unlimited-OCR`. The vision + text stack is the DeepSeek-OCR V1
/// layout (SAM + CLIP + linear projector + DeepSeek MoE decoder) shipped under
/// the same layout-B `model.*` prefixes, so it loads exactly like
/// [`load_deepseekocr_vlm`]; the raw checkpoint stores MoE experts per-expert
/// (`experts.{idx}`) rather than pre-stacked, which the DeepSeek `SwitchLinear`
/// loader now folds in transparently. The only Unlimited-OCR-specific step is
/// wrapping the runtime with a ring sliding decode cache whose window comes from
/// `language_config.sliding_window_size`.
pub(crate) fn load_unlimited_ocr_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut lc = full_config
        .get("language_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing language_config in Unlimited-OCR config"))?;
    let window = lc
        .get("sliding_window_size")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(DEFAULT_SLIDING_WINDOW)
        .max(1);
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
        .map_err(|e| anyhow::anyhow!("Failed to parse Unlimited-OCR language_config: {}", e))?;
    let (gs, bits) = (args.group_size(), args.bits());

    let weights = remap_deepseekocr_weights(load_vlm_weights_common(model_path, None)?);

    let sam = SamEncoder::from_weights(&weights, "sam_model", SamConfig::default())
        .map_err(|e| anyhow::anyhow!("Failed to load Unlimited-OCR SAM encoder: {}", e))?;
    let clip = ClipEncoder::from_weights(&weights, "vision_model", ClipConfig::default())
        .map_err(|e| anyhow::anyhow!("Failed to load Unlimited-OCR CLIP encoder: {}", e))?;
    let projector = UnifiedLinear::from_weights(&weights, "projector.layers", gs, bits)
        .map_err(|e| anyhow::anyhow!("Failed to load Unlimited-OCR projector: {}", e))?;
    let image_newline = weights
        .get("image_newline")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing image_newline"))?;
    let view_separator = weights
        .get("view_separator")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing view_separator"))?;

    let text_weights = strip_language_model_prefix(weights);
    let text_model = models::deepseek::DeepSeekModel::from_weights(&text_weights, &args)
        .map_err(|e| anyhow::anyhow!("Failed to load Unlimited-OCR text model: {}", e))?;

    let inner = vision::deepseekocr::DeepSeekOcrVlModel {
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
    let vlm = vision::unlimited_ocr::UnlimitedOcrVlModel::new(inner, window);
    Ok(LoadedModel::UnlimitedOcrVLM(vlm))
}

/// V2 key remap on top of V1's rules: fold the layout-B `qwen2_model` nesting
/// (`model.qwen2_model.model.model.*` and the `query_*` banks) into the
/// canonical `vision_model.qwen2_encoder.*`, and normalize the query banks to
/// their bare (no `.weight`) names. A no-op on the already-canonical layout A.
fn remap_deepseekocr_2_weights(weights: WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (k, v) in weights {
        let mut key = if k == "model.view_seperator" || k == "view_seperator" {
            "view_separator".to_string()
        } else if let Some(rest) = k.strip_prefix("model.qwen2_model.model.model.") {
            format!("vision_model.qwen2_encoder.{rest}")
        } else if let Some(rest) = k.strip_prefix("model.qwen2_model.") {
            format!("vision_model.qwen2_encoder.{rest}")
        } else if let Some(rest) = k.strip_prefix("model.") {
            if rest.starts_with("sam_model.")
                || rest.starts_with("vision_model.")
                || rest.starts_with("projector.")
                || rest == "view_separator"
            {
                rest.to_string()
            } else {
                k
            }
        } else {
            k
        };
        if key == "vision_model.qwen2_encoder.query_1024.weight" {
            key = "vision_model.qwen2_encoder.query_1024".to_string();
        } else if key == "vision_model.qwen2_encoder.query_768.weight" {
            key = "vision_model.qwen2_encoder.query_768".to_string();
        }
        out.insert(key, v);
    }
    out
}

pub(crate) fn load_deepseekocr_2_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut lc = full_config
        .get("language_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing language_config in DeepSeek-OCR 2 config"))?;
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
        .map_err(|e| anyhow::anyhow!("Failed to parse DeepSeek-OCR 2 language_config: {}", e))?;
    let (gs, bits) = (args.group_size(), args.bits());

    let weights = remap_deepseekocr_2_weights(load_vlm_weights_common(model_path, None)?);

    let sam = SamEncoder::from_weights(
        &weights,
        "sam_model",
        SamConfig {
            final_out_chans: 896,
            ..SamConfig::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR 2 SAM encoder: {}", e))?;
    let resampler = Qwen2Resampler::from_weights(
        &weights,
        "vision_model.qwen2_encoder",
        Qwen2ResamplerConfig::default(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR 2 query resampler: {}", e))?;
    let projector = UnifiedLinear::from_weights(&weights, "projector.layers", gs, bits)
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR 2 projector: {}", e))?;
    let view_separator = weights
        .get("view_separator")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing view_separator"))?;

    let text_weights = strip_language_model_prefix(weights);
    let text_model = models::deepseek::DeepSeekModel::from_weights(&text_weights, &args)
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-OCR 2 text model: {}", e))?;

    let vlm = vision::deepseekocr_2::DeepSeekOcr2VlModel {
        text_model,
        sam,
        resampler,
        projector,
        view_separator,
        processor: DeepSeekOcrProcessor::v2(),
        image_token_id: IMAGE_TOKEN_ID,
        eos_token_id: 1,
        n_embed: N_EMBED,
    };
    Ok(LoadedModel::DeepSeekOcr2VLM(vlm))
}
