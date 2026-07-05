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

//! ERNIE-4.5 MoE VL loader: DFNRope tower + variable-resolution resampler +
//! modality-split MoE MRoPE decoder (`ernie4_5_moe_vl`).
//!
//! Weight layout (verified against
//! `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`): text under
//! `language_model.model.*` (stripped to `model.*`; no `lm_head`, embeddings are
//! tied), vision under `vision_tower.*` (plain bf16 while the text and
//! resampler are 4-bit quantized; the unified loaders pick the mode per key),
//! resampler under `resampler_model.*` with `spatial_linear.layers.{0,2,3}` /
//! `temporal_linear.layers.{0,2,3}` / `mlp` / `after_norm`. Text config fields
//! live at the config root (no nested `text_config`), with the MoE fields in
//! int-or-2-list form and `mrope_section` under `rope_scaling`.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models::ernie4_5_moe_vl::{Ernie45MoeVlTextConfig, Ernie45MoeVlTextModel};
use crate::vision;
use crate::vision::connectors::ernie4_5_vl::Ernie45VlResampler;
use crate::vision::encoders::ernie4_5_vl::{Ernie45VlVisionConfig, Ernie45VlVisionEncoder};
use crate::vision::processors::ernie4_5_vl::Ernie45VlProcessor;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

fn read_i32(config: &Value, key: &str, default: i32) -> i32 {
    config
        .get(key)
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
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

pub(crate) fn load_ernie4_5_moe_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text config: fields at the config root, including the root quantization
    // block and rope_scaling.mrope_section.
    let text_config: Ernie45MoeVlTextConfig = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse ERNIE-4.5-VL text config: {}", e))?;
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
        .ok_or_else(|| anyhow::anyhow!("Missing vision_config in ERNIE-4.5-VL config"))?;
    let mut vcfg: Ernie45VlVisionConfig = serde_json::from_value(vision_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse ERNIE-4.5-VL vision_config: {}", e))?;
    vcfg.quant_group_size = gs;
    vcfg.quant_bits = bits;

    let weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);

    let mut text_model = Ernie45MoeVlTextModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load ERNIE-4.5-VL text model: {}", e))?;
    let eos = read_eos_ids(&full_config);
    if !eos.is_empty() {
        text_model.eos_token_ids = eos;
    }

    let vision_encoder = Ernie45VlVisionEncoder::from_weights(&weights, &vcfg, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load ERNIE-4.5-VL vision tower: {}", e))?;

    let spatial_conv_size = read_i32(&full_config, "spatial_conv_size", 2);
    let use_temporal_conv = full_config
        .get("use_temporal_conv")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let resampler = Ernie45VlResampler::from_weights(
        &weights,
        "resampler_model",
        spatial_conv_size,
        use_temporal_conv,
        gs,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load ERNIE-4.5-VL resampler: {}", e))?;

    let processor = Ernie45VlProcessor {
        patch_size: vcfg.patch_size,
        spatial_merge_size: vcfg.spatial_merge_size,
        ..Ernie45VlProcessor::default()
    };

    let im_patch_id = read_i32(&full_config, "im_patch_id", 100_295);
    let vlm = vision::ernie4_5_moe_vl::Ernie45MoeVlModel {
        text_model,
        vision_encoder,
        resampler,
        processor,
        image_token_id: im_patch_id,
        video_token_id: read_i32(&full_config, "video_token_id", im_patch_id),
        vision_start_token_id: read_i32(&full_config, "image_start_token_id", 101_304),
        vision_end_token_id: read_i32(&full_config, "image_end_token_id", 101_305),
        video_start_token_id: read_i32(&full_config, "video_start_token_id", 101_306),
        video_end_token_id: read_i32(&full_config, "video_end_token_id", 101_307),
        spatial_merge_size: vcfg.spatial_merge_size,
    };
    Ok(LoadedModel::Ernie45MoeVLM(vlm))
}
