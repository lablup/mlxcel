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

//! Jina VLM loader (`model_type: jvlm`).
//!
//! The checkpoint declares `jvlm` at the top level and in both sub-configs, and
//! ships an OLMo-flavoured nested schema (`text_config.block_config.attn_config
//! .n_heads`, `vision_config.vl_connector_config.attn_pooling_config`, ...)
//! rather than flat HF keys, so config normalization is all done here.
//!
//! The MLX conversion is mixed precision: `language_model.layers.0`, every norm,
//! the embedding tables, the ViT patch embedding, the positional table and
//! `vision_model.layers.N.ffn.down` stay bf16 while everything else is 4-bit
//! affine. Nothing special is needed for that because `UnifiedLinear::
//! from_weights` decides per prefix by probing for a `.scales` tensor, but it
//! does mean the top-level `quantization` block must be threaded into the
//! vision tower and connector too, not just the text stack.
//!
//! Weight keys are already in their final form (`language_model.*`,
//! `vision_model.*`, `vl_connector.*`), so there is no key remapping step.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::models::jina_vlm::{
    JINA_VLM_DEFAULT_EOS_IDS, JinaVlmTextConfig, JinaVlmTextModel, Quantization,
};
use crate::vision::encoders::jina_vlm::{JinaVlmVisionConfig, JinaVlmVisionModel};
use crate::vision::processors::jina_vlm::{JinaVlmImageTokens, JinaVlmProcessor};
use crate::{LoadedModel, models, vision};

use super::{load_vlm_weights_common, read_optional_model_json, read_sanitized_vlm_config};

/// `<|image|>`: the prompt-side placeholder the image-token block replaces.
const DEFAULT_IMAGE_PROMPT_TOKEN_ID: i32 = 151940;

pub(crate) fn load_jina_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let empty = Value::Object(Default::default());
    let text_value = full_config.get("text_config").unwrap_or(&empty);
    let vision_value = full_config.get("vision_config").unwrap_or(&empty);

    let (group_size, bits) = read_quantization(&full_config);
    let mut text_config = JinaVlmTextConfig::from_json(text_value);
    text_config.quantization = Some(Quantization { group_size, bits });

    let vision_config = parse_vision_config(vision_value, group_size, bits);
    let processor = build_processor(model_path);

    let mut weights = load_vlm_weights_common(model_path, None)?;
    // Apple Silicon runs f16 faster than bf16; quantization planes must stay
    // bf16 or the dequantize kernels reject them.
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    let eos_token_ids = resolve_eos_token_ids(model_path, &full_config);

    let text_model =
        JinaVlmTextModel::from_weights(&weights, &text_config, "language_model", eos_token_ids)
            .map_err(|e| anyhow::anyhow!("Failed to load Jina VLM text model: {}", e))?;

    let vision_model =
        JinaVlmVisionModel::from_weights(&weights, "vision_model", "vl_connector", vision_config)
            .map_err(|e| anyhow::anyhow!("Failed to load Jina VLM vision model: {}", e))?;

    let image_prompt_token_id = lookup_added_token_id(model_path, "<|image|>")
        .or_else(|| {
            full_config
                .get("image_token_index")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
        })
        .unwrap_or(DEFAULT_IMAGE_PROMPT_TOKEN_ID);

    let always_start_with_space = read_optional_model_json(model_path, "processor_config.json")
        .and_then(|c| c.get("always_start_with_space").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    Ok(LoadedModel::JinaVLM(vision::JinaVlmModel {
        text_model,
        vision_model,
        processor,
        image_prompt_token_id,
        always_start_with_space,
    }))
}

/// Top-level `quantization` block. Both the text stack and the vision side
/// inherit it; the checkpoint declares it only once.
fn read_quantization(full_config: &Value) -> (i32, i32) {
    let quant = full_config.get("quantization");
    let group_size = quant
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let bits = quant
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;
    (group_size, bits)
}

fn parse_vision_config(vision: &Value, group_size: i32, bits: i32) -> JinaVlmVisionConfig {
    let d = JinaVlmVisionConfig::default();
    let block = vision.get("block_config");
    let attn = block.and_then(|b| b.get("attn_config"));
    let ffn = block.and_then(|b| b.get("ffn_config"));
    let connector = vision.get("vl_connector_config");
    let pooling_attn = connector.and_then(|c| c.get("attn_pooling_config"));
    let projector = connector.and_then(|c| c.get("mlp_projector_config"));
    let lnorm = attn
        .and_then(|a| a.get("lnorm_config"))
        .or_else(|| block.and_then(|b| b.get("lnorm_config")));

    let i32_at = |v: Option<&Value>, key: &str, default: i32| -> i32 {
        v.and_then(|o| o.get(key))
            .and_then(|x| x.as_i64())
            .map(|x| x as i32)
            .unwrap_or(default)
    };

    // `input_size` is `[height, width]`; the tower is square in every released
    // config, so the height drives the crop side.
    let image_size = vision
        .get("input_size")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(d.image_size);

    let vit_layers = vision
        .get("vit_layers")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| d.vit_layers.clone());

    let pooling_num_heads = i32_at(pooling_attn, "n_heads", d.pooling_num_heads);
    let pooling_head_dim = i32_at(pooling_attn, "head_dim", d.pooling_head_dim);

    JinaVlmVisionConfig {
        hidden_size: i32_at(Some(vision), "hidden_size", d.hidden_size),
        num_hidden_layers: i32_at(Some(vision), "n_layers", d.num_hidden_layers as i32) as usize,
        num_attention_heads: i32_at(attn, "n_heads", d.num_attention_heads),
        head_dim: i32_at(attn, "head_dim", d.head_dim),
        patch_size: i32_at(Some(vision), "patch_size", d.patch_size),
        image_size,
        num_channels: i32_at(Some(vision), "n_channels", d.num_channels),
        intermediate_size: i32_at(ffn, "size", d.intermediate_size),
        layer_norm_eps: lnorm
            .and_then(|l| l.get("eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(d.layer_norm_eps as f64) as f32,
        use_cls_token: vision
            .get("use_cls_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.use_cls_token),
        post_layer_norm: vision
            .get("post_lnorm")
            .and_then(|v| v.as_bool())
            .unwrap_or(d.post_layer_norm),
        vit_layers,
        output_size: i32_at(Some(vision), "output_size", d.output_size),
        pooling_num_heads,
        pooling_head_dim,
        connector_hidden_size: i32_at(projector, "size", d.connector_hidden_size),
        pooling_h: i32_at(connector, "pooling_h", d.pooling_h),
        pooling_w: i32_at(connector, "pooling_w", d.pooling_w),
        group_size,
        bits,
    }
}

/// Build the image processor from `preprocessor_config.json`.
///
/// Special token ids come from the preprocessor config first (it names all four
/// explicitly), then from `tokenizer.json`, then from the released defaults.
fn build_processor(model_path: &Path) -> JinaVlmProcessor {
    let d = JinaVlmProcessor::default();
    let config = read_optional_model_json(model_path, "preprocessor_config.json");
    let config = config.as_ref();

    let usize_at = |key: &str, default: usize| -> usize {
        config
            .and_then(|c| c.get(key))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };
    let token_at = |key: &str, name: &str, default: i32| -> i32 {
        config
            .and_then(|c| c.get(key))
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .or_else(|| lookup_added_token_id(model_path, name))
            .unwrap_or(default)
    };

    let base_input_size = config
        .and_then(|c| c.get("base_input_size"))
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let h = a.first()?.as_u64()? as usize;
            let w = a.get(1)?.as_u64()? as usize;
            Some((h, w))
        })
        .unwrap_or(d.base_input_size);

    let overlap_margins = config
        .and_then(|c| c.get("overlap_margins"))
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let l = a.first()?.as_u64()? as usize;
            let r = a.get(1)?.as_u64()? as usize;
            Some((l, r))
        })
        .unwrap_or(d.overlap_margins);

    // `min_pixels` / `max_pixels` are also mirrored into `size` as
    // `shortest_edge` / `longest_edge`; the explicit keys win when present.
    let size = config.and_then(|c| c.get("size"));
    let min_pixels = config
        .and_then(|c| c.get("min_pixels"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            size.and_then(|s| s.get("shortest_edge"))
                .and_then(|v| v.as_u64())
        })
        .map(|v| v as usize)
        .unwrap_or(d.min_pixels);
    let max_pixels = config
        .and_then(|c| c.get("max_pixels"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            size.and_then(|s| s.get("longest_edge"))
                .and_then(|v| v.as_u64())
        })
        .map(|v| v as usize)
        .unwrap_or(d.max_pixels);

    // These three are divisors below; a malformed config must not panic the
    // whole load.
    let patch_size = usize_at("patch_size", d.patch_size).max(1);
    let pooling_h = usize_at("pooling_h", d.pooling_h).max(1);
    let pooling_w = usize_at("pooling_w", d.pooling_w).max(1);

    // `token_length_*` must equal ceil(crop_patches / pooling) or the pooled
    // feature count and the `<im_patch>` count disagree; derive rather than
    // trust when the config omits them.
    let crop_patches_h = base_input_size.0 / patch_size;
    let crop_patches_w = base_input_size.1 / patch_size;
    let token_length_h = usize_at("token_length_h", crop_patches_h.div_ceil(pooling_h));
    let token_length_w = usize_at("token_length_w", crop_patches_w.div_ceil(pooling_w));

    JinaVlmProcessor {
        base_input_size,
        patch_size,
        max_crops: usize_at("max_crops", d.max_crops),
        min_pixels,
        max_pixels,
        overlap_margins,
        pooling_h,
        pooling_w,
        token_length_h,
        token_length_w,
        use_column_tokens: config
            .and_then(|c| c.get("use_column_tokens"))
            .and_then(|v| v.as_bool())
            .unwrap_or(d.use_column_tokens),
        image_min: config
            .and_then(|c| c.get("image_min"))
            .and_then(|v| v.as_f64())
            .unwrap_or(d.image_min as f64) as f32,
        image_max: config
            .and_then(|c| c.get("image_max"))
            .and_then(|v| v.as_f64())
            .unwrap_or(d.image_max as f64) as f32,
        tokens: JinaVlmImageTokens {
            image_start_id: token_at("start_token_id", "<im_start>", d.tokens.image_start_id),
            image_end_id: token_at("end_token_id", "<im_end>", d.tokens.image_end_id),
            image_patch_id: token_at("patch_token_id", "<im_patch>", d.tokens.image_patch_id),
            image_col_id: token_at("column_token_id", "<im_col>", d.tokens.image_col_id),
        },
    }
}

/// `generation_config.json` first (absent in the released MLX conversion), then
/// the `eos_token_id` in `config.json`, then the tokenizer defaults.
fn resolve_eos_token_ids(model_path: &Path, full_config: &Value) -> Vec<i32> {
    let mut ids = crate::loading::read_eos_token_ids(model_path);
    if ids.is_empty() {
        ids = match full_config.get("eos_token_id") {
            Some(Value::Number(n)) => n.as_i64().map(|v| vec![v as i32]).unwrap_or_default(),
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect(),
            _ => Vec::new(),
        };
    }
    if ids.is_empty() {
        ids = JINA_VLM_DEFAULT_EOS_IDS.to_vec();
    }
    ids
}

/// Resolve an added token's id from `tokenizer.json`.
fn lookup_added_token_id(model_path: &Path, content: &str) -> Option<i32> {
    let tokenizer = std::fs::read_to_string(model_path.join("tokenizer.json")).ok()?;
    let tokenizer: Value = serde_json::from_str(&tokenizer).ok()?;
    tokenizer
        .get("added_tokens")?
        .as_array()?
        .iter()
        .find(|t| t.get("content").and_then(|c| c.as_str()) == Some(content))
        .and_then(|t| t.get("id"))
        .and_then(|id| id.as_i64())
        .map(|id| id as i32)
}

#[cfg(test)]
#[path = "vlm_jina_vlm_tests.rs"]
mod tests;
