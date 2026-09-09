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

//! Cohere Compass / North-Micro-Vision (`cohere_compass`) loader.
//!
//! The vision half is a Qwen3-VL tower down to the config keys, so the loader
//! reuses the Qwen3-VL vision config, encoder, processor and weight-prefix
//! remap; only the text decoder is this family's own.
//!
//! Two published weight layouts land on one key space through
//! `remap_qwen3_vl_weights`:
//!
//! ```text
//! CohereLabs original:  model.language_model.*   model.visual.*
//! mlx conversion:       language_model.model.*   vision_tower.*
//! after remap:          model.*                  vision_tower.*
//! ```

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{
    QwenVisionTokenIds, inherit_qwen_vision_quantization, load_vlm_weights_common,
    parse_required_vlm_subconfig, qwen_vl_processor_with_norm, qwen_vl_token_ids,
    read_optional_model_json, read_sanitized_vlm_config, remap_qwen3_vl_weights,
};

/// Vision token ids of the published Compass checkpoints. Every one supplies
/// them at the top level of `config.json`; these are the fallback.
const COMPASS_IMAGE_TOKEN_ID: i32 = 255031;
const COMPASS_VIDEO_TOKEN_ID: i32 = 255032;
const COMPASS_VISION_START_TOKEN_ID: i32 = 255028;

pub(crate) fn load_cohere_compass_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut text_config: models::CompassTextConfig =
        parse_required_vlm_subconfig(&full_config, "text_config", "Cohere Compass text config")?;
    // `text_config` carries no `quantization` block of its own in the mlx
    // conversion; the weights are quantized per the top-level block.
    if text_config.quantization.is_none()
        && let Some(q) = full_config.get("quantization")
    {
        text_config.quantization = serde_json::from_value(q.clone()).ok();
    }

    let mut vision_config: Qwen3VLVisionConfig = parse_required_vlm_subconfig(
        &full_config,
        "vision_config",
        "Cohere Compass vision config",
    )?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let mut weights = remap_qwen3_vl_weights(load_vlm_weights_common(model_path, None)?, false);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::CohereCompassTextModel::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Cohere Compass text model: {}", e))?;

    let vision_encoder =
        Qwen3VLVisionEncoder::from_weights(&weights, &vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load Cohere Compass vision encoder: {}", e))?;

    // `preprocessor_config.json` normalizes to mean/std 0.5, the same as the
    // Qwen3-VL processor default this helper takes. Its resize bounds are not
    // the family defaults, though, so they are read explicitly; see
    // `compass_pixel_bounds`.
    let processor =
        qwen_vl_processor_with_norm(model_path, &vision_config, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])?;
    let (min_pixels, max_pixels) = compass_pixel_bounds(model_path);
    let processor = processor.with_pixel_bounds(min_pixels, max_pixels);
    let token_ids = qwen_vl_token_ids(
        &full_config,
        QwenVisionTokenIds {
            image_token_id: COMPASS_IMAGE_TOKEN_ID,
            video_token_id: COMPASS_VIDEO_TOKEN_ID,
            vision_start_token_id: COMPASS_VISION_START_TOKEN_ID,
        },
    )?;

    let vlm = vision::CohereCompassModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    Ok(LoadedModel::CohereCompassVLM(vlm))
}

/// Resize bounds from `preprocessor_config.json`.
///
/// Compass stores them as `size.shortest_edge` / `size.longest_edge`, which in
/// this processor family are pixel *areas*, not edge lengths (65536 = 256x256,
/// 16777216 = 4096x4096). `min_pixels` / `max_pixels` at the top level win when
/// present, matching what the HF image processor reads. Anything missing or
/// malformed falls back to the Qwen-VL family default so a partial sidecar
/// cannot silently produce a degenerate grid.
fn compass_pixel_bounds(model_path: &Path) -> (usize, usize) {
    use vision::processors::qwen2_vl::{DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS};

    let config = read_optional_model_json(model_path, "preprocessor_config.json");
    let read = |primary: &str, nested: &str| -> Option<usize> {
        let config = config.as_ref()?;
        let value = config
            .get(primary)
            .filter(|v| !v.is_null())
            .or_else(|| config.get("size").and_then(|s| s.get(nested)))?;
        usize::try_from(value.as_u64()?).ok().filter(|v| *v > 0)
    };

    let min_pixels = read("min_pixels", "shortest_edge").unwrap_or(DEFAULT_MIN_PIXELS);
    let max_pixels = read("max_pixels", "longest_edge").unwrap_or(DEFAULT_MAX_PIXELS);
    if min_pixels > max_pixels {
        return (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);
    }
    (min_pixels, max_pixels)
}

#[cfg(test)]
#[path = "vlm_cohere_compass_tests.rs"]
mod tests;
