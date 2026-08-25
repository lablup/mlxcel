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

//! LFM2-VL (`lfm2_vl` / `lfm2-vl`) VLM loader.
//!
//! Layout of `mlx-community/LFM2-VL-*` checkpoints: the text backbone lives under
//! `language_model.model.*` (stripped to the plain `model.*` layout
//! [`Lfm2Model::from_weights`] expects), the vision tower under `vision_tower.*`
//! (all weights plain BF16, not quantized), and the projector under
//! `multi_modal_projector.*` (quantized linears + a plain LayerNorm). The text
//! embedding and projector linears are 4-bit; the vision tower is not, and
//! `UnifiedLinear` loads plain tensors as a regular `Linear` when no `.scales`
//! companion is present.

use anyhow::{Result, bail};
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::encoders::lfm2_vl::Lfm2VlVisionConfig;
use crate::vision::processors::lfm2_vl::Lfm2VlTilingPolicy;

use super::{load_vlm_weights_common, read_sanitized_vlm_config};

const DEFAULT_IMAGE_TOKEN_ID: i32 = 396;
const DEFAULT_IMAGE_START_ID: i32 = 498;
const DEFAULT_IMAGE_END_ID: i32 = 499;
const DEFAULT_IMG_ROW_COL_BASE_ID: i32 = 397;
const DEFAULT_IMG_THUMBNAIL_ID: i32 = 497;

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

fn read_json_file(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
}

fn sidecar_or_config_usize(
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

fn parse_tiling_policy(processor_config: Option<&Value>) -> Lfm2VlTilingPolicy {
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

fn validate_tiling_policy(
    policy: Lfm2VlTilingPolicy,
    patch_size: usize,
    downsample_factor: usize,
    max_num_patches: usize,
) -> Result<()> {
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
    let total = patch_size.max(1) * downsample_factor.max(1);
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
    Ok(())
}

fn resolve_token_id_from_json(value: &Value, name: &str) -> Option<i32> {
    value
        .get(name)
        .and_then(|id| id.as_i64())
        .map(|id| id as i32)
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
                                    .map(|id| id as i32)
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
                .map(|id| id as i32)
        })
        .or_else(|| {
            value
                .get("added_tokens_decoder")
                .and_then(|decoder| decoder.as_object())
                .and_then(|decoder| {
                    decoder.iter().find_map(|(id, token)| {
                        (token.get("content").and_then(|content| content.as_str()) == Some(name))
                            .then(|| id.parse::<i32>().ok())
                            .flatten()
                    })
                })
        })
}

fn resolve_added_token_id(
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

fn resolve_lfm2_vl_marker_ids(
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

fn parse_vision_config(full_config: &Value) -> Lfm2VlVisionConfig {
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

/// Strip the `language_model.` prefix so the LFM2 backbone sees its canonical
/// `model.*` layout (it applies its own sanitize internally).
fn lfm2_text_weights(weights: &mlxcel_core::weights::WeightMap) -> mlxcel_core::weights::WeightMap {
    let mut out = mlxcel_core::weights::WeightMap::new();
    for (key, value) in weights.iter() {
        if let Some(rest) = key.strip_prefix("language_model.") {
            out.insert(rest.to_string(), mlxcel_core::copy(value));
        }
    }
    out
}

/// Load an LFM2-VL VLM (packed-patch ViT + pixel-unshuffle projector + LFM2
/// hybrid text backbone).
pub(crate) fn load_lfm2_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::lfm2_vl::Lfm2VlConnector;
    use vision::encoders::lfm2_vl::Lfm2VlVisionTower;
    use vision::lfm2_vl::Lfm2VlModel;
    use vision::processors::lfm2_vl::Lfm2VlProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text backbone args from `text_config`, inheriting the top-level quantization.
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in LFM2-VL config.json"))?;
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(&mut text_config_value, "LFM2-VL text_config")?
            .insert("quantization".to_string(), q.clone());
    }
    let text_args: models::lfm2::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse LFM2-VL text_config: {}", e))?;

    let vision_config = parse_vision_config(&full_config);
    let gs = text_args.group_size();
    let bits = text_args.bits();

    // Load weights; convert plain bf16 tensors (vision tower + projector norm +
    // linear biases) to f16 on Apple Silicon, keeping quant scales/biases.
    let mut weights = load_vlm_weights_common(model_path, None)?;
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    // Text backbone from the stripped `model.*` subset.
    let text_weights = lfm2_text_weights(&weights);
    let text_model = models::lfm2::Lfm2Model::from_weights(text_args.clone(), text_weights)
        .map_err(|e| anyhow::anyhow!("Failed to load LFM2-VL text backbone: {}", e))?;

    // Vision tower + connector.
    let vision_tower =
        Lfm2VlVisionTower::from_weights(&weights, "vision_tower", &vision_config, gs, bits)
            .map_err(|e| anyhow::anyhow!("Failed to load LFM2-VL vision tower: {}", e))?;

    let downsample_factor = full_config
        .get("downsample_factor")
        .and_then(|v| v.as_i64())
        .unwrap_or(2) as i32;
    let use_layernorm = full_config
        .get("projector_use_layernorm")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let connector = Lfm2VlConnector::from_weights(
        &weights,
        "multi_modal_projector",
        downsample_factor,
        use_layernorm,
        gs,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load LFM2-VL projector: {}", e))?;

    // Processor. LFM2-VL stores image-splitting policy in processor_config.json;
    // keep config.json as the compatibility fallback for the keys mlxcel already
    // read before splitting support.
    let processor_config = read_json_file(&model_path.join("processor_config.json"));
    let patch_size = sidecar_or_config_usize(
        processor_config.as_ref(),
        &full_config,
        "encoder_patch_size",
        vision_config.patch_size,
    );
    let min_tokens = sidecar_or_config_usize(
        processor_config.as_ref(),
        &full_config,
        "min_image_tokens",
        64,
    );
    let max_tokens = sidecar_or_config_usize(
        processor_config.as_ref(),
        &full_config,
        "max_image_tokens",
        256,
    );
    let tiling = parse_tiling_policy(processor_config.as_ref());
    let downsample_factor_usize = downsample_factor.max(1) as usize;
    validate_tiling_policy(
        tiling,
        patch_size,
        downsample_factor_usize,
        vision_config.num_patches,
    )?;
    let processor = Lfm2VlProcessor::new(
        patch_size,
        downsample_factor_usize,
        min_tokens,
        max_tokens,
        tiling,
    );

    let added_tokens = read_json_file(&model_path.join("added_tokens.json"));
    let tokenizer_json = read_json_file(&model_path.join("tokenizer.json"));
    let (img_row_col_ids, img_thumbnail_id) =
        resolve_lfm2_vl_marker_ids(added_tokens.as_ref(), tokenizer_json.as_ref(), tiling)?;

    let image_token_id = full_config
        .get("image_token_index")
        .and_then(|v| v.as_i64())
        .or_else(|| {
            resolve_added_token_id(added_tokens.as_ref(), tokenizer_json.as_ref(), "<image>")
                .map(i64::from)
        })
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID as i64) as i32;
    let image_start_id = resolve_added_token_id(
        added_tokens.as_ref(),
        tokenizer_json.as_ref(),
        "<|image_start|>",
    )
    .unwrap_or(DEFAULT_IMAGE_START_ID);
    let image_end_id = resolve_added_token_id(
        added_tokens.as_ref(),
        tokenizer_json.as_ref(),
        "<|image_end|>",
    )
    .unwrap_or(DEFAULT_IMAGE_END_ID);
    let use_image_special_tokens = full_config
        .get("use_image_special_tokens")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let eos_token_ids = text_args.eos_token_ids();

    let vlm = Lfm2VlModel {
        text_model,
        vision_tower,
        connector,
        processor,
        image_token_id,
        image_start_id,
        image_end_id,
        img_row_col_ids,
        img_thumbnail_id,
        use_image_special_tokens,
        downsample_factor,
        patch_dim: (patch_size * patch_size * 3) as i32,
        eos_token_ids,
    };

    Ok(LoadedModel::Lfm2VL(vlm))
}
