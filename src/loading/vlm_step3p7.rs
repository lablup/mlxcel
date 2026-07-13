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

//! Step-3.7 (`step3p7`) loader: `perception_encoder` ViT + Step-3.5 MoE text
//! backbone.
//!
//! Parses the nested config (`text_config` + `vision_config`), resolves the
//! `<im_patch>` scatter target and the image-framing special-token ids,
//! sanitizes weights (prefix normalization + conv permutation, then the shared
//! Step-3.5 text sanitize), and builds the [`vision::Step3p7VlModel`] wrapper.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::models::step3p5::Step3p5Config;
use crate::vision;
use crate::vision::encoders::step3p7::permute_conv_weight_to_channels_last;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};

/// Default `<im_patch>` id when neither `image_token_index` nor `image_token_id`
/// is present (the value shipped by known step3p7 configs).
const DEFAULT_IMAGE_TOKEN_ID: i64 = 128001;

/// Load a Step-3.7 (`step3p7`) VLM.
pub(crate) fn load_step3p7_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::step3p7::Step3p7Connector;
    use vision::encoders::step3p7::{Step3p7VisionConfig, Step3p7VisionEncoder};
    use vision::processors::step3p7::Step3p7Processor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text backbone config from the nested `text_config`, with the step3p7
    // default flips applied.
    let text_config_value = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in step3p7 config.json"))?;
    let mut text_config = Step3p5Config::from_nested_text_config(&text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to build step3p7 text config: {}", e))?;
    if text_config.quantization.is_none()
        && let Some(q) = full_config.get("quantization")
    {
        text_config.quantization = serde_json::from_value(q.clone()).ok();
    }

    // Vision (`perception_encoder`) config, inheriting the global quantization.
    let mut vision_config: Step3p7VisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "step3p7 vision config")?;
    if let Some(q) = full_config.get("quantization") {
        if let Some(gs) = q.get("group_size").and_then(|v| v.as_i64()) {
            vision_config.quant_group_size = gs as i32;
        }
        if let Some(b) = q.get("bits").and_then(|v| v.as_i64()) {
            vision_config.quant_bits = b as i32;
        }
    }

    let projector_stride = full_config
        .get("understand_projector_stride")
        .and_then(|v| v.as_i64())
        .unwrap_or(2) as usize;

    let tokens = resolve_token_ids(model_path, &full_config);
    let eos_token_ids = Step3p5Config::resolve_step3p7_eos_token_ids(&full_config);

    // Weights.
    let raw = load_vlm_weights_common(model_path, None)?;
    let weights = sanitize_step3p7_weights(raw, &text_config, &vision_config);

    let backbone = models::Step3p5Model::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load step3p7 text backbone: {}", e))?;

    let encoder = Step3p7VisionEncoder::from_weights(&weights, &vision_config, "vision_model")
        .map_err(|e| anyhow::anyhow!("Failed to load step3p7 vision encoder: {}", e))?;

    let connector = Step3p7Connector::from_weights(
        &weights,
        "vision_model",
        "vit_large_projector",
        vision_config.width,
        projector_stride,
        vision_config.quant_group_size,
        vision_config.quant_bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load step3p7 connector: {}", e))?;

    let processor = Step3p7Processor::new();
    let base_grid = (processor.base_size / vision_config.patch_size) as i32;
    let patch_grid = (processor.patch_window / vision_config.patch_size) as i32;

    let vlm = vision::Step3p7VlModel {
        backbone,
        encoder,
        connector,
        processor,
        tokens,
        base_grid,
        patch_grid,
        eos_token_ids,
    };

    Ok(LoadedModel::Step3p7VL(vlm))
}

/// Resolve `<im_patch>` and the image-framing special-token ids.
///
/// `image_token_index` (falling back to `image_token_id`, then
/// [`DEFAULT_IMAGE_TOKEN_ID`]) is the scatter target. The framing tokens
/// (`<im_start>`/`<im_end>`/`<patch_start>`/`<patch_end>`/`<patch_newline>`)
/// are resolved from `added_tokens.json`; the fallbacks are placeholders that
/// the real-checkpoint smoke test must confirm.
fn resolve_token_ids(model_path: &Path, full_config: &Value) -> vision::step3p7::Step3p7TokenIds {
    let image_token_id = full_config
        .get("image_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID);
    let image_token_index = full_config
        .get("image_token_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(image_token_id) as i32;

    let added_tokens: Option<Value> = std::fs::read_to_string(model_path.join("added_tokens.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let resolve = |name: &str, default: i32| -> i32 {
        added_tokens
            .as_ref()
            .and_then(|v| v.get(name))
            .and_then(|id| id.as_i64())
            .map(|id| id as i32)
            .unwrap_or(default)
    };

    vision::step3p7::Step3p7TokenIds {
        image_token_index,
        im_start: resolve("<im_start>", image_token_index + 1),
        im_end: resolve("<im_end>", image_token_index + 2),
        patch_start: resolve("<patch_start>", image_token_index + 3),
        patch_end: resolve("<patch_end>", image_token_index + 4),
        patch_newline: resolve("<patch_newline>", image_token_index + 5),
    }
}

/// Sanitize step3p7 weights.
///
/// 1. Normalize every prefix variant to the shape-table target names.
/// 2. Permute the three conv kernels to MLX channels-last (idempotent guard).
/// 3. Split text vs vision keys and reuse `Step3p5Model::sanitize_weights` so
///    the MoE fused-name remap and zero-centered-norm `+1` offset live in one
///    place (and never touch the vision LayerNorms).
fn sanitize_step3p7_weights(
    raw: WeightMap,
    text_config: &Step3p5Config,
    vision_config: &vision::encoders::step3p7::Step3p7VisionConfig,
) -> WeightMap {
    let width = vision_config.width as i32;
    let num_channels = vision_config.num_channels as i32;

    let mut normalized = WeightMap::new();
    for (key, value) in raw {
        let new_key = normalize_key(&key);
        let value = match new_key.as_str() {
            "vision_model.conv1.weight" => {
                permute_conv_weight_to_channels_last(value.as_ref().unwrap(), num_channels)
            }
            "vision_model.vit_downsampler1.weight" => {
                permute_conv_weight_to_channels_last(value.as_ref().unwrap(), width)
            }
            "vision_model.vit_downsampler2.weight" => {
                permute_conv_weight_to_channels_last(value.as_ref().unwrap(), 2 * width)
            }
            _ => value,
        };
        normalized.insert(new_key, value);
    }

    let mut text = WeightMap::new();
    let mut vision = WeightMap::new();
    for (key, value) in normalized {
        if key.starts_with("vision_model.") || key.starts_with("vit_large_projector.") {
            vision.insert(key, value);
        } else {
            text.insert(key, value);
        }
    }

    let mut sanitized = models::Step3p5Model::sanitize_weights(text, text_config);
    for (key, value) in vision {
        sanitized.insert(key, value);
    }
    sanitized
}

/// Normalize one checkpoint key to the shape-table target name.
fn normalize_key(key: &str) -> String {
    let mut k = key.to_string();

    // Text backbone wrappers.
    if let Some(rest) = k.strip_prefix("model.language_model.") {
        k = format!("model.{rest}");
    } else if let Some(rest) = k.strip_prefix("language_model.") {
        // `language_model.model.*` -> `model.*`; `language_model.lm_head.*` -> top-level.
        k = rest.to_string();
    }

    // Vision tower + projector prefixes.
    if let Some(rest) = k.strip_prefix("model.vision_model.") {
        k = format!("vision_model.{rest}");
    } else if let Some(rest) = k.strip_prefix("model.vit_large_projector.") {
        k = format!("vit_large_projector.{rest}");
    }

    // Bare vision keys -> vision_model.*
    if is_bare_vision_key(&k) {
        k = format!("vision_model.{k}");
    }

    // Both `transformer.resblocks.{i}` and `transformer.{i}` spellings occur.
    k = k.replace("transformer.resblocks.", "transformer.");

    // Fused qkv tensor-attribute naming -> submodule naming.
    k = k.replace("attn.in_proj_weight", "attn.in_proj.weight");
    k = k.replace("attn.in_proj_bias", "attn.in_proj.bias");

    k
}

fn is_bare_vision_key(k: &str) -> bool {
    if k.starts_with("vision_model.")
        || k.starts_with("vit_large_projector.")
        || k.starts_with("model.")
        || k.starts_with("lm_head.")
    {
        return false;
    }
    k.starts_with("conv1.")
        || k.starts_with("ln_pre.")
        || k.starts_with("ln_post.")
        || k.starts_with("transformer.")
        || k.starts_with("vit_downsampler")
        || k.starts_with("positional_embedding")
        || k.starts_with("class_embedding")
}

#[cfg(test)]
#[path = "vlm_step3p7_tests.rs"]
mod tests;
