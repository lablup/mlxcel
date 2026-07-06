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

//! Hunyuan-VL loader (`hunyuan_vl`, e.g. HunyuanOCR).
//!
//! Weight layout (verified against `hadeseus/HunyuanOCR-mlx-4bit`): text under
//! `language_model.model.*` (stripped to `model.*`; tied embeddings, no
//! `lm_head`), vision under `vision_tower.*` (plain bf16 while the text is
//! 4-bit; conv weights already channels-last). Text config fields live at the
//! config root next to a nested `vision_config`; token ids
//! (`image_token_id` 120120 etc.) at the root.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models::hunyuan_vl::{HunyuanVlTextConfig, HunyuanVlTextModel};
use crate::vision;
use crate::vision::encoders::hunyuan_vl::{HunyuanVlVisionConfig, HunyuanVlVisionEncoder};
use crate::vision::processors::hunyuan_vl::HunyuanVlProcessor;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

fn read_i32(config: &Value, key: &str, default: i32) -> i32 {
    config
        .get(key)
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(default)
}

fn read_usize(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn read_eos_ids(config: &Value) -> Vec<i32> {
    match config.get("eos_token_id") {
        Some(Value::Number(n)) => n.as_i64().map(|v| vec![v as i32]).unwrap_or_default(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn load_hunyuan_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text config fields sit at the config root (including quantization and
    // rope_scaling with the xdrope section).
    let text_config: HunyuanVlTextConfig = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse Hunyuan-VL text config: {}", e))?;
    let gs = full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let bits = full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let vision_value = full_config
        .get("vision_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing vision_config in Hunyuan-VL config"))?;
    let mut vcfg: HunyuanVlVisionConfig = serde_json::from_value(vision_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Hunyuan-VL vision_config: {}", e))?;
    vcfg.quant_group_size = gs;
    vcfg.quant_bits = bits;

    let weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);

    let mut text_model = HunyuanVlTextModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Hunyuan-VL text model: {}", e))?;
    let eos = read_eos_ids(&full_config);
    if !eos.is_empty() {
        text_model.eos_token_ids = eos;
    }

    let vision_encoder = HunyuanVlVisionEncoder::from_weights(&weights, &vcfg, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load Hunyuan-VL vision tower: {}", e))?;

    // Processor: honor preprocessor_config.json min/max pixel overrides.
    let mut processor = HunyuanVlProcessor {
        patch_size: vcfg.patch_size,
        spatial_merge_size: vcfg.spatial_merge_size,
        ..HunyuanVlProcessor::default()
    };
    let pre_path = model_path.join("preprocessor_config.json");
    if let Ok(text) = std::fs::read_to_string(&pre_path)
        && let Ok(pre) = serde_json::from_str::<Value>(&text)
    {
        processor.min_pixels = read_usize(&pre, "min_pixels", processor.min_pixels);
        processor.max_pixels = read_usize(&pre, "max_pixels", processor.max_pixels);
    }

    let vlm = vision::hunyuan_vl::HunyuanVlModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: read_i32(&full_config, "image_token_id", 120_120),
        image_start_token_id: read_i32(&full_config, "image_start_token_id", 120_118),
        image_end_token_id: read_i32(&full_config, "image_end_token_id", 120_119),
        image_newline_token_id: read_i32(&full_config, "image_newline_token_id", 120_121),
        spatial_merge_size: vcfg.spatial_merge_size,
    };
    Ok(LoadedModel::HunyuanVLM(vlm))
}
