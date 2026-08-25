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

use anyhow::Result;
use std::path::Path;

use super::{load_vlm_weights_common, read_sanitized_vlm_config};
use crate::LoadedModel;
use crate::models;
use crate::vision;

#[path = "vlm_lfm2_vl_metadata.rs"]
mod metadata;
#[cfg(test)]
#[path = "vlm_lfm2_vl_tests.rs"]
mod tests;

use self::metadata::{
    lfm2_processor_sidecar, parse_tiling_policy, parse_vision_config, positive_i32_field,
    read_optional_json_file, resolve_added_token_id, resolve_lfm2_vl_marker_ids,
    sidecar_or_config_bool, sidecar_or_config_usize, token_field, validate_tiling_policy,
};

const DEFAULT_IMAGE_TOKEN_ID: i32 = 396;
const DEFAULT_IMAGE_START_ID: i32 = 498;
const DEFAULT_IMAGE_END_ID: i32 = 499;

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

    let downsample_factor = positive_i32_field(&full_config, "downsample_factor")?.unwrap_or(2);
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
    let processor_config_json = read_optional_json_file(
        &model_path.join("processor_config.json"),
        "processor_config.json",
    )?;
    let processor_config = processor_config_json.as_ref().map(lfm2_processor_sidecar);
    let patch_size = sidecar_or_config_usize(
        processor_config,
        &full_config,
        "encoder_patch_size",
        vision_config.patch_size,
    );
    let min_tokens =
        sidecar_or_config_usize(processor_config, &full_config, "min_image_tokens", 64);
    let max_tokens =
        sidecar_or_config_usize(processor_config, &full_config, "max_image_tokens", 256);
    let tiling = parse_tiling_policy(processor_config);
    let downsample_factor_usize = downsample_factor.max(1) as usize;
    let tile_patch_side = tiling.tile_size / patch_size.max(1);
    let default_max_num_patches = (max_tokens
        .saturating_mul(downsample_factor_usize)
        .saturating_mul(downsample_factor_usize))
    .max(tile_patch_side.saturating_mul(tile_patch_side));
    let max_num_patches = sidecar_or_config_usize(
        processor_config,
        &full_config,
        "max_num_patches",
        default_max_num_patches,
    );
    validate_tiling_policy(
        tiling,
        patch_size,
        downsample_factor_usize,
        min_tokens,
        max_tokens,
        max_num_patches,
    )?;
    let processor = Lfm2VlProcessor::new(
        patch_size,
        downsample_factor_usize,
        min_tokens,
        max_tokens,
        tiling,
    );

    let added_tokens =
        read_optional_json_file(&model_path.join("added_tokens.json"), "added_tokens.json")?;
    let tokenizer_json =
        read_optional_json_file(&model_path.join("tokenizer.json"), "tokenizer.json")?;
    let (img_row_col_ids, img_thumbnail_id) =
        resolve_lfm2_vl_marker_ids(added_tokens.as_ref(), tokenizer_json.as_ref(), tiling)?;

    let image_token_id = token_field(&full_config, "image_token_index")?
        .or_else(|| {
            resolve_added_token_id(added_tokens.as_ref(), tokenizer_json.as_ref(), "<image>")
        })
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID);
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
    let use_image_special_tokens = sidecar_or_config_bool(
        processor_config,
        &full_config,
        "use_image_special_tokens",
        true,
    );
    let eos_token_ids = text_args.eos_token_ids();

    let patch_dim = patch_size
        .checked_mul(patch_size)
        .and_then(|dim| dim.checked_mul(3))
        .and_then(|dim| i32::try_from(dim).ok())
        .ok_or_else(|| {
            anyhow::anyhow!("LFM2-VL patch_dim overflows for patch_size={patch_size}")
        })?;

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
        patch_dim,
        eos_token_ids,
    };

    Ok(LoadedModel::Lfm2VL(vlm))
}
