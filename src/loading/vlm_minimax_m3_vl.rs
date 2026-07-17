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

//! MiniMax-M3-VL loader (`model_type: "minimax_m3_vl"`).
//!
//! The only public checkpoint (`MiniMaxAI/MiniMax-M3`, 427B) is itself the VL
//! model: a top-level `minimax_m3_vl` config with nested `text_config` and
//! `vision_config`. Text weights live under `language_model.model.*` and the
//! CLIP-style vision tower under `vision_tower.vision_model.*`, with the
//! two-stage projector at `multi_modal_projector.*` / `patch_merge_mlp.*`.
//!
//! The tower is built from the raw weight map first (its verbatim keys are
//! intact there), after which the map is moved into the MiniMax-M3 text
//! sanitizer, which drops the vision/projector tensors and rewrites
//! `language_model.model.*` -> `model.*` for the text decoder. Vision weights
//! are f32/bf16 non-quantized while the text tower may be quantized in
//! community exports; the text `ModelArgs` inherits the top-level quantization
//! block when its own `text_config` omits it.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models::MiniMaxM3Model;
use crate::models::minimax_m3::{ModelArgs, sanitize_weights};
use crate::vision;
use crate::vision::encoders::minimax_m3_vl::{MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder};
use crate::vision::processors::minimax_m3::MiniMaxM3Processor;

use super::{load_vlm_weights_common, read_sanitized_vlm_config};

pub(crate) fn load_minimax_m3_vl(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text config lives under `text_config`. Community 4-bit exports commonly
    // keep the quantization block at the config root, so inject it when the
    // nested block omits it.
    let mut text_value = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in MiniMax-M3-VL config.json"))?;
    if let (Some(obj), Some(quant)) = (text_value.as_object_mut(), full_config.get("quantization"))
    {
        obj.entry("quantization").or_insert_with(|| quant.clone());
    }
    let text_args: ModelArgs = serde_json::from_value(text_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse MiniMax-M3-VL text_config: {}", e))?;
    if let Some(sparse) = text_args.sparse_attention_config.as_ref() {
        sparse.validate().map_err(|e| anyhow::anyhow!(e))?;
    }

    let vision_value = full_config
        .get("vision_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing vision_config in MiniMax-M3-VL config.json"))?;
    let vision_config: MiniMaxM3VisionConfig = serde_json::from_value(vision_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse MiniMax-M3-VL vision_config: {}", e))?;

    let weights = load_vlm_weights_common(model_path, None)?;

    // Build the vision tower from the raw keys first (borrowing), then move the
    // map into the text sanitizer.
    let vision_encoder = MiniMaxM3VisionEncoder::from_weights(&weights, &vision_config)
        .map_err(|e| anyhow::anyhow!("Failed to load MiniMax-M3-VL vision tower: {}", e))?;

    let text_weights = sanitize_weights(weights, &text_args);
    let text_model = MiniMaxM3Model::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load MiniMax-M3-VL text model: {}", e))?;

    let read_token = |key: &str, default: i64| -> i32 {
        full_config
            .get(key)
            .and_then(|v| v.as_i64())
            .unwrap_or(default) as i32
    };

    let mut eos_token_ids = crate::loading::read_eos_token_ids(model_path);
    if eos_token_ids.is_empty() {
        // MiniMax `[e~[` sentinel at 200020 in its 200k vocab.
        eos_token_ids = vec![200020];
    }

    let vlm = vision::minimax_m3_vl::MiniMaxM3VlModel {
        text_model,
        vision_encoder,
        processor: MiniMaxM3Processor::default(),
        image_token_id: read_token("image_token_id", 200025),
        video_token_id: read_token("video_token_id", 200026),
        vision_start_token_id: read_token("vision_start_token_id", 200029),
        vision_end_token_id: read_token("vision_end_token_id", 200030),
        spatial_merge_size: vision_config.spatial_merge_size() as i32,
        eos_token_ids,
    };
    Ok(LoadedModel::MiniMaxM3VL(vlm))
}
