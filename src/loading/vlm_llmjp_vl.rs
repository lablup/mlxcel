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

//! LLM-jp-VL (`llmjpvl`) VLM loader.
//!
//! Two released checkpoints share this architecture and differ only in the text
//! backbone named by `llm_config.model_type`:
//!
//! - `llm-jp/llm-jp-4-vl-9B-beta`: `llama`, hidden 4096, 32 layers, vocab
//!   196608, untied LM head, image ids 14 / 15 / 16.
//! - `llm-jp/Jagle-VL-2.2B-Jagle-FineVision`: `qwen3`, hidden 2048, 28 layers,
//!   vocab 151676, tied embeddings, image ids 151655 / 151669 / 151670.
//!
//! Three details separate this loader from the InternVL one it is shaped after:
//! the decoder config lives under `llm_config` (there is no `text_config`), the
//! tower is SigLIP2 under `vision_backbone.vision_model.*` rather than
//! InternViT under `vision_model.*`, and the `mlp1` LayerNorm takes torch's
//! default epsilon instead of `vision_config.layer_norm_eps`.

use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};

/// Weight prefix of the SigLIP2 tower.
const VISION_PREFIX: &str = "vision_backbone.vision_model";

/// Attention-pooling head. `last_hidden_state` is the feature source, so these
/// tensors are never read; dropping them keeps roughly 20 MB of bf16 out of the
/// resident weight map.
const VISION_HEAD_PREFIX: &str = "vision_backbone.vision_model.head.";

/// `llm-jp-4-vl-9B-beta` image-framing ids, the fallback when a checkpoint
/// ships no `added_tokens_decoder`.
const DEFAULT_IMAGE_PAD_TOKEN_ID: i32 = 14;
const DEFAULT_IMAGE_START_TOKEN_ID: i32 = 15;
const DEFAULT_IMAGE_END_TOKEN_ID: i32 = 16;

/// `llm-jp-4-vl-9B-beta` stop ids (`<|return|>`, `<|end|>`).
const DEFAULT_EOS_TOKEN_IDS: [i32; 2] = [2, 11];

/// Map `content -> id` from a `tokenizer_config.json` `added_tokens_decoder`
/// block, whose keys are stringified ids and whose values carry the literal
/// token text.
///
/// The two checkpoints disagree on every image id (14 / 15 / 16 versus
/// 151655 / 151669 / 151670), so resolving by content rather than trusting a
/// constant is what makes one loader serve both.
pub(crate) fn added_token_ids(tokenizer_config: Option<&Value>) -> HashMap<String, i32> {
    let mut map = HashMap::new();
    let Some(decoder) = tokenizer_config
        .and_then(|c| c.get("added_tokens_decoder"))
        .and_then(Value::as_object)
    else {
        return map;
    };
    for (id, entry) in decoder {
        let Ok(id) = id.parse::<i32>() else {
            continue;
        };
        if let Some(content) = entry.get("content").and_then(Value::as_str) {
            map.insert(content.to_string(), id);
        }
    }
    map
}

/// Resolve `(<|image_pad|>, <|image_start|>, <|image_end|>)`.
///
/// `img_context_token_id` in `config.json` names the pad id directly and wins
/// when present; the other two only exist in the tokenizer.
pub(crate) fn resolve_image_token_ids(
    full_config: &Value,
    added: &HashMap<String, i32>,
) -> (i32, i32, i32) {
    let pad = full_config
        .get("img_context_token_id")
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .or_else(|| added.get("<|image_pad|>").copied())
        .unwrap_or(DEFAULT_IMAGE_PAD_TOKEN_ID);
    let start = added
        .get("<|image_start|>")
        .copied()
        .unwrap_or(DEFAULT_IMAGE_START_TOKEN_ID);
    let end = added
        .get("<|image_end|>")
        .copied()
        .unwrap_or(DEFAULT_IMAGE_END_TOKEN_ID);
    (pad, start, end)
}

/// Resolve the stop ids.
///
/// `generation_config.json` names them for both checkpoints (`[2, 2]` for the
/// 9B, `[151675, 151645]` for Jagle-VL), and the Harmony turn terminators
/// `<|return|>` / `<|end|>` / `<|im_end|>` are added from the tokenizer so a
/// turn that ends on the non-final terminator still stops.
pub(crate) fn resolve_eos_token_ids(
    generation_config: Option<&Value>,
    llm_config: Option<&Value>,
    added: &HashMap<String, i32>,
) -> Vec<i32> {
    fn push(ids: &mut Vec<i32>, id: i32) {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }

    fn push_eos_field(ids: &mut Vec<i32>, source: Option<&Value>) {
        let Some(value) = source.and_then(|c| c.get("eos_token_id")) else {
            return;
        };
        match value {
            Value::Number(_) => {
                if let Some(id) = value.as_i64() {
                    push(ids, id as i32);
                }
            }
            Value::Array(items) => {
                for id in items.iter().filter_map(Value::as_i64) {
                    push(ids, id as i32);
                }
            }
            _ => {}
        }
    }

    let mut ids: Vec<i32> = Vec::new();
    push_eos_field(&mut ids, generation_config);
    push_eos_field(&mut ids, llm_config);
    for name in ["<|return|>", "<|end|>", "<|im_end|>"] {
        if let Some(&id) = added.get(name) {
            push(&mut ids, id);
        }
    }

    if ids.is_empty() {
        ids = DEFAULT_EOS_TOKEN_IDS.to_vec();
    }
    ids
}

/// Take the decoder config out of `config.json`.
///
/// Released checkpoints put it under `llm_config`; `text_config` is accepted
/// as a fallback because that is what a converted checkpoint following the
/// InternVL naming would carry. The top-level `quantization` block is inherited
/// when the sub-config has none, so a quantized conversion reaches the backbone
/// loader with its group size and bit width.
pub(crate) fn select_text_config(full_config: &Value) -> Result<Value> {
    let mut text_config = full_config
        .get("llm_config")
        .or_else(|| full_config.get("text_config"))
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!("Missing llm_config (and text_config fallback) in config.json")
        })?;

    if text_config.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(&mut text_config, "LLM-jp-VL llm_config")?
            .insert("quantization".to_string(), q.clone());
    }
    Ok(text_config)
}

/// Reject an `llm_config.model_type` this runtime has no backbone for.
///
/// `qwen2` is accepted alongside `llama` because mlxcel already serves that
/// graph through the Llama backbone, so a converted checkpoint relabelling the
/// decoder still loads.
pub(crate) fn validate_text_model_type(model_type: &str) -> Result<()> {
    match model_type {
        "llama" | "qwen2" | "qwen3" => Ok(()),
        other => Err(anyhow::anyhow!(
            "Unsupported LLM-jp-VL llm_config.model_type: '{other}' (expected 'llama' or 'qwen3')"
        )),
    }
}

/// Reject `select_layer != -1` at load with the value named.
///
/// Upstream reads `hidden_states[select_layer]` for any other value, which this
/// runtime does not implement; failing here is better than silently serving
/// `last_hidden_state` features the checkpoint was not trained against.
pub(crate) fn validate_select_layer(full_config: &Value) -> Result<()> {
    match full_config.get("select_layer").and_then(Value::as_i64) {
        None | Some(-1) => Ok(()),
        Some(other) => Err(anyhow::anyhow!(
            "LLM-jp-VL select_layer = {other} is not supported (only -1, the post-layernorm last hidden state, is implemented)"
        )),
    }
}

/// Load an LLM-jp-VL (`llmjpvl`) VLM.
pub(crate) fn load_llmjp_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::config::VisionConfig;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::internvl::InternVLConnector;
    use vision::llmjp_vl::{
        LLMJP_DEFAULT_IMAGE_SEQ_LENGTH, LLMJP_DEFAULT_MODEL_MAX_LENGTH, LLMJP_MLP1_LAYER_NORM_EPS,
        LlmJpVlModel,
    };
    use vision::llmjp_vl_text::LlmJpTextModel;
    use vision::processors::internvl::InternVLProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    validate_select_layer(&full_config)?;

    let vision_config: VisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "LLM-jp-VL vision config")?;
    let text_config = select_text_config(&full_config)?;
    let text_model_type = text_config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    // Fail before the weight load, not after: the 9B checkpoint is 18 GB and
    // an operator pointing at an unsupported backbone should learn that in a
    // second rather than a minute.
    validate_text_model_type(&text_model_type)?;

    // Weights. `load_vlm_weights_common` applies the shared load-time dtype
    // policy; a quantized conversion opts out of it, so convert the remaining
    // plain bf16 tensors on Apple Silicon here while keeping quantization
    // scales/biases in bf16 for `quantized_matmul` (the InternVL rule).
    let mut weights = load_vlm_weights_common(model_path, None)?;
    weights.retain(|key, _| !key.starts_with(VISION_HEAD_PREFIX));
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    // The backbone loaders expect `model.*` / `lm_head.*`. The decoder tensors
    // are *moved* out of the map rather than copied: the 9B checkpoint's
    // decoder is 18 GB and neither the tower nor the connector reads it.
    let text_weights =
        super::strip_language_model_prefix(take_language_model_weights(&mut weights));

    let (text_model, group_size, bits) = match text_model_type.as_str() {
        // `qwen2` names the same graph mlxcel already serves with the Llama
        // backbone, so a conversion relabelling the decoder still loads.
        "llama" | "qwen2" => {
            let mut args: models::llama3::ModelArgs =
                serde_json::from_value(text_config).map_err(|e| {
                    anyhow::anyhow!("Failed to parse LLM-jp-VL llm_config as llama: {}", e)
                })?;
            args.set_checkpoint_label(model_path);
            let (group_size, bits) = (args.group_size(), args.bits());
            let model = models::Llama3Model::from_weights(&text_weights, &args)
                .map_err(|e| anyhow::anyhow!("Failed to load LLM-jp-VL llama text model: {}", e))?;
            (LlmJpTextModel::Llama(model), group_size, bits)
        }
        "qwen3" => {
            let mut args: models::qwen3::ModelArgs =
                serde_json::from_value(text_config).map_err(|e| {
                    anyhow::anyhow!("Failed to parse LLM-jp-VL llm_config as qwen3: {}", e)
                })?;
            args.set_checkpoint_label(model_path);
            let (group_size, bits) = (args.group_size(), args.bits());
            let model = models::Qwen3Model::from_weights(&text_weights, &args)
                .map_err(|e| anyhow::anyhow!("Failed to load LLM-jp-VL qwen3 text model: {}", e))?;
            (LlmJpTextModel::Qwen3(model), group_size, bits)
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported LLM-jp-VL llm_config.model_type: '{other}' (expected 'llama' or 'qwen3')"
            ));
        }
    };
    drop(text_weights);

    // SigLIP2 tower. `from_weights_with_quant_and_gelu` sanitizes the conv
    // layout itself, accepting both the HF `[O, I, kH, kW]` shipped here and an
    // already-converted `[O, kH, kW, I]`.
    let vision_model = SigLipVisionModel::from_weights_with_quant_and_gelu(
        &weights,
        &vision_config,
        VISION_PREFIX,
        group_size,
        bits,
        false,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load LLM-jp-VL SigLIP2 vision tower: {}", e))?;

    let downsample_ratio = full_config
        .get("downsample_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.5) as f32;
    let connector = InternVLConnector::from_weights(
        &weights,
        "mlp1",
        LLMJP_MLP1_LAYER_NORM_EPS,
        downsample_ratio,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load LLM-jp-VL mlp1 connector: {}", e))?;
    drop(weights);

    let image_size = full_config
        .get("force_image_size")
        .and_then(Value::as_u64)
        .unwrap_or(vision_config.image_size as u64) as usize;
    let min_dynamic_patch = full_config
        .get("min_dynamic_patch")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    let max_dynamic_patch = full_config
        .get("max_dynamic_patch")
        .and_then(Value::as_u64)
        .unwrap_or(12) as usize;
    let use_thumbnail = full_config
        .get("use_thumbnail")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let mut processor = InternVLProcessor::new(
        image_size,
        min_dynamic_patch,
        max_dynamic_patch,
        use_thumbnail,
    );
    // SigLIP normalization: (pixel/255 - 0.5) / 0.5, not the ImageNet
    // statistics the InternVL processor defaults to.
    let (mean, std) = siglip_normalization(model_path);
    processor.mean = mean;
    processor.std = std;

    let patches_per_side = image_size / vision_config.patch_size.max(1);
    let num_image_token = ((patches_per_side * patches_per_side) as f32
        * downsample_ratio
        * downsample_ratio)
        .round() as usize;

    let tokenizer_config = super::read_optional_model_json(model_path, "tokenizer_config.json");
    let generation_config = super::read_optional_model_json(model_path, "generation_config.json");
    let processor_config = super::read_optional_model_json(model_path, "processor_config.json");
    let added = added_token_ids(tokenizer_config.as_ref());
    let (image_context_token_id, img_start_token_id, img_end_token_id) =
        resolve_image_token_ids(&full_config, &added);
    let eos_token_ids = resolve_eos_token_ids(
        generation_config.as_ref(),
        full_config.get("llm_config"),
        &added,
    );

    let image_seq_length = processor_config
        .as_ref()
        .and_then(|c| c.get("image_seq_length"))
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(LLMJP_DEFAULT_IMAGE_SEQ_LENGTH);
    let model_max_length = tokenizer_config
        .as_ref()
        .and_then(|c| c.get("model_max_length"))
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(LLMJP_DEFAULT_MODEL_MAX_LENGTH);

    Ok(LoadedModel::LlmJpVL(LlmJpVlModel {
        text_model,
        vision_model,
        connector,
        processor,
        image_context_token_id,
        img_start_token_id,
        img_end_token_id,
        num_image_token,
        eos_token_ids,
        image_seq_length,
        model_max_length,
        max_dynamic_patch,
    }))
}

/// Read `preprocessor_config.json` normalization, defaulting to SigLIP's
/// `[0.5; 3]` mean and std.
fn siglip_normalization(model_path: &Path) -> ([f32; 3], [f32; 3]) {
    let default = [0.5f32; 3];
    let Some(config) = super::read_optional_model_json(model_path, "preprocessor_config.json")
    else {
        return (default, default);
    };
    let read = |key: &str| -> [f32; 3] {
        config
            .get(key)
            .and_then(Value::as_array)
            .and_then(|values| {
                let parsed: Vec<f32> = values
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|v| v as f32)
                    .collect();
                <[f32; 3]>::try_from(parsed.as_slice()).ok()
            })
            .unwrap_or(default)
    };
    (read("image_mean"), read("image_std"))
}

/// Move every `language_model.*` tensor out of `weights`.
///
/// Moving rather than copying matters at this scale: `llm-jp-4-vl-9B-beta`
/// carries an 18 GB decoder, and the vision tower and connector read none of
/// it. What is left behind is exactly `vision_backbone.*` plus `mlp1.*`.
fn take_language_model_weights(
    weights: &mut mlxcel_core::weights::WeightMap,
) -> mlxcel_core::weights::WeightMap {
    let keys: Vec<String> = weights
        .keys()
        .filter(|key| key.starts_with("language_model."))
        .cloned()
        .collect();
    let mut out = mlxcel_core::weights::WeightMap::new();
    for key in keys {
        if let Some(value) = weights.remove(&key) {
            out.insert(key, value);
        }
    }
    out
}

#[cfg(test)]
#[path = "vlm_llmjp_vl_tests.rs"]
mod tests;
