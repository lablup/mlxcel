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

//! Granite 4 Vision loader (SigLIP + 8 window-QFormer projectors + Granite 4
//! hybrid text backbone with multi-depth injection).

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::connectors::granite4_vision::{Downsampler, WindowQFormerProjector};

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

/// Default LLaVA-Next `image_grid_pinpoints` for Granite 4 Vision.
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

/// `downsample_rate` `"q/w"` -> `(q, w)`; defaults to `(4, 8)`.
fn parse_downsample_rate(full_config: &serde_json::Value) -> (i32, i32) {
    full_config
        .get("downsample_rate")
        .and_then(|v| v.as_str())
        .and_then(|s| {
            let mut it = s.split('/');
            let q = it.next()?.trim().parse::<i32>().ok()?;
            let w = it.next()?.trim().parse::<i32>().ok()?;
            Some((q, w))
        })
        .unwrap_or((4, 8))
}

fn parse_i32_pairs(v: Option<&serde_json::Value>) -> Vec<[i32; 2]> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|pair| {
                    let p = pair.as_array()?;
                    Some([p.first()?.as_i64()? as i32, p.get(1)?.as_i64()? as i32])
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_i32_list(v: Option<&serde_json::Value>) -> Vec<i32> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_i64().map(|i| i as i32))
                .collect()
        })
        .unwrap_or_default()
}

/// Load a Granite 4 Vision VLM.
pub(crate) fn load_granite4_vision_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text backbone config: inherit top-level quantization, parse as granitemoehybrid.
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
    let text_args: models::granitemoehybrid::ModelArgs = serde_json::from_value(text_config_val)
        .map_err(|e| anyhow::anyhow!("Failed to parse Granite 4 text config: {}", e))?;

    let vision_config: vision::config::VisionConfig = serde_json::from_value(
        full_config
            .get("vision_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing vision_config in config.json"))?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse Granite 4 vision config: {}", e))?;

    let group_size = text_args.group_size();
    let bits = text_args.bits();
    let (q, w) = parse_downsample_rate(&full_config);

    let weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);

    // Vision tower and 8 window-QFormer projectors read by ref before the text
    // backbone consumes the weight map. The granite4 SigLIP tower uses the
    // sigmoid `GELU(approx="fast")` in its MLP (unlike Granite Vision #539, which
    // uses the tanh variant), matching the reference vision encoder.
    let vision_tower =
        vision::encoders::siglip::SigLipVisionModel::from_weights_with_quant_and_gelu(
            &weights,
            &vision_config,
            "vision_tower.vision_model",
            group_size,
            bits,
            true, // use_fast_gelu (sigmoid GELU)
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Granite 4 vision tower: {}", e))?;

    let spatial_offsets = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let mut projectors = Vec::with_capacity(8);
    for i in 0..4 {
        projectors.push(
            WindowQFormerProjector::from_weights(
                &weights,
                &format!("layerwise_projectors.{i}"),
                Downsampler::MeanPool,
                q,
                w,
                group_size,
                bits,
            )
            .map_err(|e| anyhow::anyhow!("Failed to load layerwise projector {i}: {}", e))?,
        );
    }
    for (i, (row_off, col_off)) in spatial_offsets.iter().enumerate() {
        projectors.push(
            WindowQFormerProjector::from_weights(
                &weights,
                &format!("spatial_projectors.{i}"),
                Downsampler::Strided {
                    row_off: *row_off,
                    col_off: *col_off,
                },
                q,
                w,
                group_size,
                bits,
            )
            .map_err(|e| anyhow::anyhow!("Failed to load spatial projector {i}: {}", e))?,
        );
    }

    let image_newline = weights
        .get("image_newline")
        .map(|x| mlxcel_core::copy(x))
        .ok_or_else(|| anyhow::anyhow!("Missing image_newline weight"))?;

    // Stream routing from config.
    let deepstack_map = parse_i32_pairs(full_config.get("deepstack_layer_map"));
    let spatial_targets = parse_i32_list(full_config.get("spatial_target_layers"));
    let spatial_vision_layer = full_config
        .get("spatial_vision_layer")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let use_spatial = full_config
        .get("use_spatial_sampling")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // Collect the deepstack vision taps; the spatial tap reuses one of them when
    // present, otherwise it is appended.
    let mut taps: Vec<i32> = deepstack_map.iter().map(|p| p[0]).collect();
    let spatial_tap_idx = match taps.iter().position(|&t| t == spatial_vision_layer) {
        Some(idx) => idx,
        None => {
            taps.push(spatial_vision_layer);
            taps.len() - 1
        }
    };

    let mut stream_specs: Vec<(usize, usize, usize)> = Vec::new();
    for (i, pair) in deepstack_map.iter().enumerate() {
        stream_specs.push((i, i, pair[1] as usize));
    }
    if use_spatial {
        for (i, &layer) in spatial_targets.iter().enumerate() {
            stream_specs.push((4 + i, spatial_tap_idx, layer as usize));
        }
    }

    let image_token_index = full_config
        .get("image_token_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(100352) as i32;
    let pinpoints = parse_pinpoints(&full_config);
    let image_size = vision_config.image_size as u32;
    let patch_grid = (vision_config.image_size / vision_config.patch_size) as i32; // 24
    let feature_side = patch_grid / (w / q); // 24 / 2 = 12
    let base_tokens = feature_side * feature_side; // 144
    let processor = vision::processors::anyres::AnyResProcessor::new(pinpoints, image_size);

    // Text backbone consumes the weight map last.
    let eos_token_id = text_args.eos_token_ids().first().copied().unwrap_or(100257);
    let text_model = models::GraniteMoeHybridModel::from_weights(text_args, weights)
        .map_err(|e| anyhow::anyhow!("Failed to load Granite 4 text model: {}", e))?;

    let vlm = vision::Granite4VisionVLModel::new(
        text_model,
        vision_tower,
        projectors,
        image_newline,
        processor,
        image_token_index,
        taps,
        stream_specs,
        feature_side,
        base_tokens,
        eos_token_id,
    );

    Ok(LoadedModel::Granite4VisionVLM(vlm))
}
