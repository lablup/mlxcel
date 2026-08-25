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

//! LFM2-VL loader metadata helpers for processor policy and tokenizer markers.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::path::Path;

use crate::vision::encoders::lfm2_vl::Lfm2VlVisionConfig;
use crate::vision::processors::lfm2_vl::Lfm2VlTilingPolicy;

const DEFAULT_IMG_ROW_COL_BASE_ID: i32 = 397;
const DEFAULT_IMG_THUMBNAIL_ID: i32 = 497;
pub(super) const MAX_LFM2_VL_VIEW_PATCHES: usize = 16 * 1024;
const MAX_LFM2_VL_CANVAS_PIXELS: usize = 32 * 1024 * 1024;

fn get_usize(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(default)
}

fn get_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
}

fn get_f32(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(default)
}

pub(super) fn read_optional_json_file(path: &Path, label: &str) -> Result<Option<Value>> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str::<Value>(&content)
            .with_context(|| format!("Failed to parse LFM2-VL {label} at {}", path.display()))
            .map(Some),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err)
            .with_context(|| format!("Failed to read LFM2-VL {label} at {}", path.display())),
    }
}

fn token_i64_to_i32(id: i64) -> Option<i32> {
    i32::try_from(id).ok().filter(|id| *id >= 0)
}

pub(super) fn token_field(v: &Value, key: &str) -> Result<Option<i32>> {
    v.get(key)
        .and_then(|value| value.as_i64())
        .map(|id| {
            token_i64_to_i32(id)
                .ok_or_else(|| anyhow::anyhow!("LFM2-VL {key} token id {id} is outside i32 range"))
        })
        .transpose()
}

pub(super) fn positive_i32_field(v: &Value, key: &str) -> Result<Option<i32>> {
    v.get(key)
        .and_then(|value| value.as_i64())
        .map(|value| {
            i32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    anyhow::anyhow!("LFM2-VL {key} value {value} must be a positive i32")
                })
        })
        .transpose()
}

pub(super) fn sidecar_or_config_usize(
    sidecar: Option<&Value>,
    full_config: &Value,
    key: &str,
    default: usize,
) -> usize {
    sidecar
        .and_then(|v| v.get(key))
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| get_usize(full_config, key, default))
}

pub(super) fn sidecar_or_config_bool(
    sidecar: Option<&Value>,
    full_config: &Value,
    key: &str,
    default: bool,
) -> bool {
    sidecar
        .and_then(|v| v.get(key))
        .and_then(|x| x.as_bool())
        .unwrap_or_else(|| get_bool(full_config, key, default))
}

pub(super) fn lfm2_processor_sidecar(config: &Value) -> &Value {
    config.get("image_processor").unwrap_or(config)
}

pub(super) fn parse_tiling_policy(processor_config: Option<&Value>) -> Lfm2VlTilingPolicy {
    let defaults = Lfm2VlTilingPolicy::default();
    let Some(config) = processor_config else {
        return defaults;
    };
    Lfm2VlTilingPolicy {
        do_image_splitting: get_bool(config, "do_image_splitting", defaults.do_image_splitting),
        tile_size: get_usize(config, "tile_size", defaults.tile_size),
        min_tiles: get_usize(config, "min_tiles", defaults.min_tiles),
        max_tiles: get_usize(config, "max_tiles", defaults.max_tiles),
        max_pixels_tolerance: get_f32(
            config,
            "max_pixels_tolerance",
            defaults.max_pixels_tolerance,
        ),
        use_thumbnail: get_bool(config, "use_thumbnail", defaults.use_thumbnail),
    }
}

pub(super) fn validate_tiling_policy(
    policy: Lfm2VlTilingPolicy,
    patch_size: usize,
    downsample_factor: usize,
    min_image_tokens: usize,
    max_image_tokens: usize,
    max_num_patches: usize,
) -> Result<()> {
    if patch_size == 0 {
        bail!("LFM2-VL processor encoder_patch_size must be positive");
    }
    if downsample_factor == 0 {
        bail!("LFM2-VL processor downsample_factor must be positive");
    }
    if min_image_tokens == 0 || min_image_tokens > max_image_tokens {
        bail!(
            "LFM2-VL processor requires 1 <= min_image_tokens <= max_image_tokens, got {}..={}",
            min_image_tokens,
            max_image_tokens
        );
    }
    let max_single_view_patches = max_image_tokens
        .checked_mul(downsample_factor)
        .and_then(|tokens| tokens.checked_mul(downsample_factor))
        .ok_or_else(|| anyhow::anyhow!("LFM2-VL max image patch budget overflows"))?;
    if max_single_view_patches > MAX_LFM2_VL_VIEW_PATCHES {
        bail!(
            "LFM2-VL max_image_tokens={} with downsample_factor={} can allocate {} patches, exceeding the safety limit {}",
            max_image_tokens,
            downsample_factor,
            max_single_view_patches,
            MAX_LFM2_VL_VIEW_PATCHES
        );
    }
    if max_num_patches > MAX_LFM2_VL_VIEW_PATCHES {
        bail!(
            "LFM2-VL max_num_patches={} exceeds the safety limit {}",
            max_num_patches,
            MAX_LFM2_VL_VIEW_PATCHES
        );
    }
    if policy.tile_size == 0 {
        bail!("LFM2-VL processor_config.json tile_size must be positive");
    }
    if policy.min_tiles == 0 || policy.min_tiles > policy.max_tiles {
        bail!(
            "LFM2-VL processor_config.json requires 1 <= min_tiles <= max_tiles, got {}..={}",
            policy.min_tiles,
            policy.max_tiles
        );
    }
    if !policy.max_pixels_tolerance.is_finite() || policy.max_pixels_tolerance <= 0.0 {
        bail!(
            "LFM2-VL processor_config.json max_pixels_tolerance must be finite and positive, got {}",
            policy.max_pixels_tolerance
        );
    }
    if policy.max_tiles > 10 {
        bail!(
            "LFM2-VL processor_config.json max_tiles={} exceeds the shipped <|img_row_r_col_c|> marker table size 10",
            policy.max_tiles
        );
    }
    let total = patch_size
        .checked_mul(downsample_factor)
        .ok_or_else(|| anyhow::anyhow!("LFM2-VL patch_size * downsample_factor overflows"))?;
    if !policy.tile_size.is_multiple_of(total) {
        bail!(
            "LFM2-VL processor_config.json tile_size={} must be divisible by encoder_patch_size * downsample_factor ({})",
            policy.tile_size,
            total
        );
    }
    let tile_patch_side = policy.tile_size / patch_size.max(1);
    if tile_patch_side.saturating_mul(tile_patch_side) > max_num_patches.max(1) {
        bail!(
            "LFM2-VL processor_config.json tile_size={} creates {}x{} patches, exceeding max_num_patches={}",
            policy.tile_size,
            tile_patch_side,
            tile_patch_side,
            max_num_patches
        );
    }
    let max_canvas_pixels = policy
        .tile_size
        .checked_mul(policy.tile_size)
        .and_then(|pixels_per_tile| pixels_per_tile.checked_mul(policy.max_tiles))
        .ok_or_else(|| anyhow::anyhow!("LFM2-VL tiling canvas pixel count overflows"))?;
    if max_canvas_pixels > MAX_LFM2_VL_CANVAS_PIXELS {
        bail!(
            "LFM2-VL processor_config.json tile_size={} and max_tiles={} can allocate {} canvas pixels, exceeding the safety limit {}",
            policy.tile_size,
            policy.max_tiles,
            max_canvas_pixels,
            MAX_LFM2_VL_CANVAS_PIXELS
        );
    }
    Ok(())
}

fn resolve_token_id_from_json(value: &Value, name: &str) -> Option<i32> {
    value
        .get(name)
        .and_then(|id| id.as_i64())
        .and_then(token_i64_to_i32)
        .or_else(|| {
            value
                .get("added_tokens")
                .and_then(|tokens| tokens.as_array())
                .and_then(|tokens| {
                    tokens.iter().find_map(|token| {
                        (token.get("content").and_then(|content| content.as_str()) == Some(name))
                            .then(|| {
                                token
                                    .get("id")
                                    .and_then(|id| id.as_i64())
                                    .and_then(token_i64_to_i32)
                            })
                            .flatten()
                    })
                })
        })
        .or_else(|| {
            value
                .get("model")
                .and_then(|model| model.get("vocab"))
                .and_then(|vocab| vocab.get(name))
                .and_then(|id| id.as_i64())
                .and_then(token_i64_to_i32)
        })
        .or_else(|| {
            value
                .get("added_tokens_decoder")
                .and_then(|decoder| decoder.as_object())
                .and_then(|decoder| {
                    decoder.iter().find_map(|(id, token)| {
                        (token.get("content").and_then(|content| content.as_str()) == Some(name))
                            .then(|| id.parse::<i32>().ok().filter(|id| *id >= 0))
                            .flatten()
                    })
                })
        })
}

pub(super) fn resolve_added_token_id(
    added_tokens: Option<&Value>,
    tokenizer_json: Option<&Value>,
    name: &str,
) -> Option<i32> {
    added_tokens
        .and_then(|value| resolve_token_id_from_json(value, name))
        .or_else(|| tokenizer_json.and_then(|value| resolve_token_id_from_json(value, name)))
}

fn lfm2_row_col_default(row: usize, col: usize) -> i32 {
    DEFAULT_IMG_ROW_COL_BASE_ID + ((row - 1) * 10 + (col - 1)) as i32
}

pub(super) fn resolve_lfm2_vl_marker_ids(
    added_tokens: Option<&Value>,
    tokenizer_json: Option<&Value>,
    policy: Lfm2VlTilingPolicy,
) -> Result<([[i32; 10]; 10], i32)> {
    let mut row_col_ids = [[0i32; 10]; 10];
    for row in 1..=10 {
        for col in 1..=10 {
            row_col_ids[row - 1][col - 1] = resolve_added_token_id(
                added_tokens,
                tokenizer_json,
                &format!("<|img_row_{row}_col_{col}|>"),
            )
            .unwrap_or_else(|| lfm2_row_col_default(row, col));
        }
    }
    let thumbnail_id = resolve_added_token_id(added_tokens, tokenizer_json, "<|img_thumbnail|>")
        .unwrap_or(DEFAULT_IMG_THUMBNAIL_ID);

    if policy.do_image_splitting {
        for row in 1..=policy.max_tiles {
            for col in 1..=policy.max_tiles {
                let name = format!("<|img_row_{row}_col_{col}|>");
                if resolve_added_token_id(added_tokens, tokenizer_json, &name).is_none() {
                    bail!("LFM2-VL tokenizer is missing required added token {name}");
                }
            }
        }
        if policy.use_thumbnail
            && resolve_added_token_id(added_tokens, tokenizer_json, "<|img_thumbnail|>").is_none()
        {
            bail!("LFM2-VL tokenizer is missing required added token <|img_thumbnail|>");
        }
    }

    Ok((row_col_ids, thumbnail_id))
}

pub(super) fn parse_vision_config(full_config: &Value) -> Lfm2VlVisionConfig {
    let vc = full_config
        .get("vision_config")
        .cloned()
        .unwrap_or(Value::Null);
    let d = Lfm2VlVisionConfig::default();
    Lfm2VlVisionConfig {
        hidden_size: get_usize(&vc, "hidden_size", d.hidden_size),
        intermediate_size: get_usize(&vc, "intermediate_size", d.intermediate_size),
        num_hidden_layers: get_usize(&vc, "num_hidden_layers", d.num_hidden_layers),
        num_attention_heads: get_usize(&vc, "num_attention_heads", d.num_attention_heads),
        patch_size: get_usize(&vc, "patch_size", d.patch_size),
        num_patches: get_usize(&vc, "num_patches", d.num_patches),
        layer_norm_eps: vc
            .get("layer_norm_eps")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(d.layer_norm_eps),
        // vision_feature_layer is a top-level key, not in vision_config.
        vision_feature_layer: full_config
            .get("vision_feature_layer")
            .and_then(|x| x.as_i64())
            .map(|x| x as i32)
            .unwrap_or(d.vision_feature_layer),
    }
}
