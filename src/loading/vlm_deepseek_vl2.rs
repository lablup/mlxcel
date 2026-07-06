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

//! DeepSeek-VL2 loader: SigLIP tower + `downsample_mlp_gelu` projector +
//! `image_newline` / `view_separator` feeding the shared DeepSeek-V2 MoE decoder.
//!
//! Weight layout (verified against `mlx-community/deepseek-vl2-small-4bit`):
//! text under `language_model.model.*` / `language_model.lm_head` (MLA, pre-fused
//! `switch_mlp`), vision under `vision.vision_tower.*`, projector under
//! `projector.*`, and the `image_newline` / `view_separator` vectors at the top
//! level. The unused `vision.vision_tower.attn_pool.*` head is dropped at load.

use anyhow::Result;
use serde_json::json;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::connectors::deepseek_vl2::DownsampleMlpGelu;
use crate::vision::encoders::deepseek_vl2::{DeepSeekVl2VisionConfig, DeepSeekVl2VisionEncoder};
use crate::vision::processors::deepseek_vl2::DeepSeekVl2Processor;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

/// `<image>` placeholder id and EOS from the checkpoint tokenizer (config.json
/// carries no `image_token_index` for this family).
const IMAGE_TOKEN_ID: i32 = 100003;
const EOS_TOKEN_ID: i32 = 100001;
const DEFAULT_DOWNSAMPLE_RATIO: i32 = 2;

/// Drop the unused attention-pool head and normalize the `view_seperator`
/// misspelling to the canonical name (this checkpoint already ships the correct
/// spelling, so the rename is a no-op there but guards other exports).
fn remap_deepseek_vl2_weights(weights: WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (k, v) in weights {
        if k.contains(".attn_pool.") {
            continue;
        }
        let key = if k == "view_seperator" {
            "view_separator".to_string()
        } else {
            k
        };
        out.insert(key, v);
    }
    out
}

fn parse_candidate_resolutions(full_config: &serde_json::Value) -> Vec<(i32, i32)> {
    full_config
        .get("candidate_resolutions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    let p = p.as_array()?;
                    Some((p.first()?.as_i64()? as i32, p.get(1)?.as_i64()? as i32))
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn load_deepseek_vl2_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Text decoder config from `language_config`, inheriting the root
    // quantization block and the `deepseek_v2` model_type (the sub-config omits
    // both). MLA head dims are absent and fall back to DeepSeek-V2 defaults.
    let mut lc = full_config
        .get("language_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing language_config in DeepSeek-VL2 config"))?;
    if let Some(obj) = lc.as_object_mut() {
        obj.entry("model_type".to_string())
            .or_insert_with(|| json!("deepseek_v2"));
        if !obj.contains_key("quantization")
            && let Some(q) = full_config.get("quantization")
        {
            obj.insert("quantization".to_string(), q.clone());
        }
    }
    let args: models::deepseek_v2::ModelArgs = serde_json::from_value(lc)
        .map_err(|e| anyhow::anyhow!("Failed to parse DeepSeek-VL2 language_config: {}", e))?;
    let (gs, bits) = (args.group_size(), args.bits());

    // Vision tower config; quantization is inherited from the root block.
    let vision_config = full_config
        .get("vision_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing vision_config in DeepSeek-VL2 config"))?;
    let mut vcfg: DeepSeekVl2VisionConfig = serde_json::from_value(vision_config)
        .map_err(|e| anyhow::anyhow!("Failed to parse DeepSeek-VL2 vision_config: {}", e))?;
    vcfg.quant_group_size = gs;
    vcfg.quant_bits = bits;

    let candidate_resolutions = parse_candidate_resolutions(&full_config);
    if candidate_resolutions.is_empty() {
        anyhow::bail!("DeepSeek-VL2 config is missing candidate_resolutions");
    }
    let downsample_ratio = full_config
        .get("projector_config")
        .and_then(|p| p.get("downsample_ratio"))
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(DEFAULT_DOWNSAMPLE_RATIO);
    let processor =
        DeepSeekVl2Processor::new(candidate_resolutions, vcfg.patch_size, downsample_ratio);

    let weights = remap_deepseek_vl2_weights(load_vlm_weights_common(model_path, None)?);

    let encoder = DeepSeekVl2VisionEncoder::from_weights(&weights, &vcfg, "vision.vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-VL2 vision tower: {}", e))?;
    let projector =
        DownsampleMlpGelu::from_weights(&weights, "projector", downsample_ratio, gs, bits)
            .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-VL2 projector: {}", e))?;
    let image_newline = weights
        .get("image_newline")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing image_newline"))?;
    let view_separator = weights
        .get("view_separator")
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| anyhow::anyhow!("Missing view_separator"))?;
    let n_embed = mlxcel_core::array_shape(&image_newline)[0];

    // The text backbone consumes the `model.*` / `lm_head` keys; the vision keys
    // are already loaded by reference above.
    let text_weights = strip_language_model_prefix(weights);
    let text_model = models::deepseek_v2::DeepSeekV2Model::from_weights(&text_weights, &args)
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-VL2 text model: {}", e))?;

    let vlm = vision::deepseek_vl2::DeepSeekVl2VlModel {
        text_model,
        encoder,
        projector,
        image_newline,
        view_separator,
        processor,
        image_token_id: IMAGE_TOKEN_ID,
        eos_token_id: EOS_TOKEN_ID,
        n_embed,
    };
    Ok(LoadedModel::DeepSeekVL2(vlm))
}
