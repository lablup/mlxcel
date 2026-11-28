// Copyright 2025-2026 Lablup Inc.
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
//! `image_newline` / `view_separator` feeding a plain-MHA (tiny) or MLA MoE decoder.
//!
//! Weight layout (verified against `mlx-community/deepseek-vl2-small-4bit`):
//! text under `language_model.model.*` / `language_model.lm_head` (MLA, pre-fused
//! `switch_mlp`), vision under `vision.vision_tower.*`, projector under
//! `projector.*`, and the `image_newline` / `view_separator` vectors at the top
//! level. The unused `vision.vision_tower.attn_pool.*` head is dropped at load.

use anyhow::{Context, Result};
use serde_json::json;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::connectors::deepseek_vl2::DownsampleMlpGelu;
use crate::vision::deepseek_vl2::TextDecoder;
use crate::vision::encoders::deepseek_vl2::{DeepSeekVl2VisionConfig, DeepSeekVl2VisionEncoder};
use crate::vision::processors::deepseek_vl2::DeepSeekVl2Processor;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

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
        } else if k == "view_seperator.scales" {
            "view_separator.scales".to_string()
        } else if k == "view_seperator.biases" {
            "view_separator.biases".to_string()
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

/// Canonical decoder config for loading and KV sizing. Tiny explicitly turns
/// MLA off (zero MLA head dimension) but still labels itself deepseek_v2.
/// The other releases omit the head dimensions and use the MLA defaults.
/// All inherit the outer quantization block unless the decoder supplies one.
///
/// Used by: [`load_deepseek_vl2_vlm`] and
/// `execution::config_fields::kv_decoder_config`, so the KV cache mode rules
/// and the sizing see the decoder the runtime builds (issue #3155).
pub(crate) fn deepseek_vl2_language_config(
    full_config: &serde_json::Value,
) -> Option<serde_json::Value> {
    let mut lc = full_config.get("language_config")?.clone();
    if let Some(obj) = lc.as_object_mut() {
        let use_mla = obj
            .get("use_mla")
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| obj.get("qk_nope_head_dim").and_then(|v| v.as_u64()) != Some(0));
        if use_mla {
            obj.entry("model_type")
                .or_insert_with(|| json!("deepseek_v2"));
        } else {
            obj.insert("model_type".into(), json!("deepseek"));
            // KV sizing recognizes MLA by a numeric rank; the MHA decoder ignores it.
            obj.remove("kv_lora_rank");
        }
        if !obj.contains_key("quantization")
            && let Some(q) = full_config.get("quantization")
        {
            obj.insert("quantization".to_string(), q.clone());
        }
    }
    Some(lc)
}

pub(crate) fn load_deepseek_vl2_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let lc = deepseek_vl2_language_config(&full_config)
        .ok_or_else(|| anyhow::anyhow!("Missing language_config in DeepSeek-VL2 config"))?;
    let tokenizer = crate::tokenizer::load_tokenizer(model_path)?;
    let special_id = |token: &str| -> Result<i32> {
        let id = tokenizer
            .token_to_id(token)
            .with_context(|| format!("Missing DeepSeek-VL2 tokenizer entry {token}"))?;
        Ok(i32::try_from(id)?)
    };
    let image_token_id = special_id("<image>")?;
    let eos_token_id = special_id("<｜end▁of▁sentence｜>")?;

    let weights = strip_language_model_prefix(remap_deepseek_vl2_weights(load_vlm_weights_common(
        model_path, None,
    )?));
    let (text_model, hidden_size, gs, bits) = if lc["model_type"] == "deepseek" {
        let args: models::deepseek::ModelArgs = serde_json::from_value(lc)
            .context("Failed to parse DeepSeek-VL2 plain-MHA language_config")?;
        let model = models::deepseek::DeepSeekModel::from_weights(&weights, &args)
            .map_err(anyhow::Error::msg)?;
        (
            TextDecoder::Mha(model),
            args.hidden_size,
            args.group_size(),
            args.bits(),
        )
    } else {
        let args: models::deepseek_v2::ModelArgs = serde_json::from_value(lc)
            .context("Failed to parse DeepSeek-VL2 MLA language_config")?;
        let model = models::deepseek_v2::DeepSeekV2Model::from_weights(&weights, &args)
            .map_err(anyhow::Error::msg)?;
        (
            TextDecoder::Mla(model),
            args.hidden_size,
            args.group_size(),
            args.bits(),
        )
    };

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

    let encoder = DeepSeekVl2VisionEncoder::from_weights(&weights, &vcfg, "vision.vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-VL2 vision tower: {}", e))?;
    let projector =
        DownsampleMlpGelu::from_weights(&weights, "projector", downsample_ratio, gs, bits)
            .map_err(|e| anyhow::anyhow!("Failed to load DeepSeek-VL2 projector: {}", e))?;
    let n_embed = i32::try_from(hidden_size)
        .map_err(|_| anyhow::anyhow!("DeepSeek-VL2 hidden_size does not fit in i32"))?;
    if !weights.contains_key("image_newline") && !weights.contains_key("image_newline.scales") {
        anyhow::bail!("Missing image_newline");
    }
    let image_newline = crate::vision::weights::load_dense_param(
        &weights,
        "image_newline",
        &[n_embed],
        Some((gs, bits)),
    )
    .map_err(anyhow::Error::msg)?;
    if !weights.contains_key("view_separator") && !weights.contains_key("view_separator.scales") {
        anyhow::bail!("Missing view_separator");
    }
    let view_separator = crate::vision::weights::load_dense_param(
        &weights,
        "view_separator",
        &[n_embed],
        Some((gs, bits)),
    )
    .map_err(anyhow::Error::msg)?;

    let vlm = vision::deepseek_vl2::DeepSeekVl2VlModel {
        text_model,
        encoder,
        projector,
        image_newline,
        view_separator,
        processor,
        image_token_id,
        eos_token_id,
        n_embed,
    };
    Ok(LoadedModel::DeepSeekVL2(vlm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_vl2_tiny_selects_a_parseable_plain_decoder_for_loading_and_kv_sizing() {
        let config = json!({
            "model_type":"deepseek_vl_v2",
            "quantization":{"group_size":32,"bits":4},
            "language_config":{
                "model_type":"deepseek_v2", "use_mla":false,
                "kv_lora_rank":null,"qk_nope_head_dim":0,
                "hidden_size":1280,"intermediate_size":6848,"num_hidden_layers":12,
                "num_attention_heads":10,"num_key_value_heads":10,
                "vocab_size":129280,"max_position_embeddings":4096
            }
        });
        let lc = deepseek_vl2_language_config(&config).unwrap();
        assert_eq!(lc["model_type"], "deepseek");
        let args: models::deepseek::ModelArgs = serde_json::from_value(lc).unwrap();
        assert_eq!(
            (args.head_dim(), args.group_size(), args.bits()),
            (128, 32, 4)
        );
        assert_eq!(
            crate::execution::config_fields::kv_decoder_model_type(&config).as_deref(),
            Some("deepseek")
        );
    }

    #[test]
    fn deepseek_vl2_missing_head_dimensions_keep_mla_defaults() {
        let cfg = json!({"language_config":{"hidden_size":2048}});
        assert_eq!(
            deepseek_vl2_language_config(&cfg).unwrap()["model_type"],
            "deepseek_v2"
        );
    }
}
