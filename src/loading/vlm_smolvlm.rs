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

//! SmolVLM / SmolVLM2 (`smolvlm`) and SmolVLM-Instruct (`idefics3`) VLM loader.
//!
//! SmolVLM was built on Idefics3, so both checkpoints share one runtime: a
//! SigLIP vision tower, a `pixel_shuffle(scale_factor)` + bias-free Linear
//! connector, and a plain Llama text backbone. Two on-disk layouts are handled:
//!
//! - `SmolVLMForConditionalGeneration` (SmolVLM / SmolVLM2, mlx-community):
//!   vision `model.vision_model.*`, connector
//!   `model.connector.modality_projection.proj.*`, SmolLM2 text
//!   `model.text_model.*` with a separate top-level untied head `lm_head.*`.
//! - `Idefics3ForConditionalGeneration` (SmolVLM-Instruct): vision
//!   `vision_model.*`, connector `connector.modality_projection.proj.*`, and the
//!   whole Llama-with-head nested under `language_model.*` (including
//!   `language_model.lm_head.*`).
//!
//! Both layouts are remapped into the plain Llama key layout mlxcel's
//! [`crate::models::Llama3Model`] expects (see [`remap_smolvlm_text_key`]), and
//! the vision/connector prefixes are resolved from whichever form is on disk.
//!
//! On Apple Silicon, non-quantized bf16 weights are converted to f16 by
//! [`load_vlm_weights_common`]. For quantized checkpoints that conversion is
//! skipped (to preserve quantization scales/biases), so this loader converts
//! the remaining plain bf16 tensors (vision tower, connector, norms) to f16
//! while keeping `.scales` / `.biases` as bf16, mirroring the InternVL loader.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};

/// Released SmolVLM `<image>` placeholder id (`image_token_id` in config.json).
const DEFAULT_IMAGE_TOKEN_ID: i32 = 49153;
/// Default pixel-shuffle compression factor.
const DEFAULT_SCALE_FACTOR: i32 = 2;

/// Resolve a special token id from the tokenizer's `added_tokens.json`,
/// returning `0` (unknown) when the file or key is absent.
fn resolve_added_token_id(added_tokens: Option<&Value>, name: &str) -> i32 {
    added_tokens
        .and_then(|v| v.get(name))
        .and_then(|id| id.as_i64())
        .map(|id| id as i32)
        .unwrap_or(0)
}

/// Resolve the EOS/stop token ids. Prefer the text config's `eos_token_id`
/// (single or array); fall back to the SmolLM2 default (`<|im_end|>` = 2 and
/// `<end_of_utterance>` when present).
fn resolve_eos_token_ids(full_config: &Value, added_tokens: Option<&Value>) -> Vec<i32> {
    let mut ids = Vec::new();

    let text_eos = full_config
        .get("text_config")
        .and_then(|tc| tc.get("eos_token_id"))
        .or_else(|| full_config.get("eos_token_id"));
    match text_eos {
        Some(Value::Number(n)) => {
            if let Some(v) = n.as_i64() {
                ids.push(v as i32);
            }
        }
        Some(Value::Array(arr)) => {
            for v in arr {
                if let Some(v) = v.as_i64() {
                    ids.push(v as i32);
                }
            }
        }
        _ => {}
    }

    // `<end_of_utterance>` is the SmolVLM chat-template turn terminator.
    let eou = resolve_added_token_id(added_tokens, "<end_of_utterance>");
    if eou != 0 && !ids.contains(&eou) {
        ids.push(eou);
    }

    if ids.is_empty() {
        ids.push(2); // SmolLM2 default EOS.
    }
    ids
}

/// Map one checkpoint text-weight key into the plain Llama key layout the
/// [`models::Llama3Model`] loader expects, returning `None` for keys that are
/// not text weights (vision tower / connector keys are handled separately).
///
/// Two on-disk layouts are supported:
/// - SmolVLM / SmolVLM2 (`SmolVLMForConditionalGeneration`, mlx-community): the
///   SmolLM2 backbone lives under `model.text_model.*` with a separate
///   top-level untied head at `lm_head.*`.
/// - Idefics3 / SmolVLM-Instruct (`Idefics3ForConditionalGeneration`): the whole
///   Llama-with-head is nested under `language_model.*`, so the untied head is
///   `language_model.lm_head.*`. It must be promoted to the top-level `lm_head.*`
///   key. [`models::Llama3Model::from_weights`] reads the untied head from
///   `lm_head` (not `model.lm_head`), so mapping it to `model.lm_head.*` would
///   leave the head unfound and fail the load.
fn remap_smolvlm_text_key(key: &str) -> Option<String> {
    if let Some(rest) = key.strip_prefix("model.text_model.") {
        Some(format!("model.{rest}"))
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        if let Some(head) = rest.strip_prefix("lm_head.") {
            // Promote the idefics3 nested untied head to the top-level key.
            Some(format!("lm_head.{head}"))
        } else {
            Some(format!("model.{rest}"))
        }
    } else if key.starts_with("lm_head.") {
        Some(key.to_string())
    } else {
        None
    }
}

/// Collect the SmolLM2 text weights and rewrite them into the plain Llama key
/// layout the [`models::Llama3Model`] loader expects. See
/// [`remap_smolvlm_text_key`] for the per-key mapping rules.
fn smolvlm_text_weights(
    weights: &mlxcel_core::weights::WeightMap,
) -> mlxcel_core::weights::WeightMap {
    let mut out = mlxcel_core::weights::WeightMap::new();
    for (key, value) in weights.iter() {
        if let Some(dest) = remap_smolvlm_text_key(key) {
            out.insert(dest, mlxcel_core::copy(value));
        }
    }
    out
}

/// Pick the weight prefix that actually exists in the checkpoint for a
/// sub-module, so both the HuggingFace (`model.vision_model.*`) and an
/// already-sanitized mlx-vlm (`vision_model.*`) layout load.
fn resolve_prefix<'a>(
    weights: &mlxcel_core::weights::WeightMap,
    hf_prefix: &'a str,
    bare_prefix: &'a str,
) -> &'a str {
    let has_hf = weights.iter().any(|(k, _)| k.starts_with(hf_prefix));
    if has_hf { hf_prefix } else { bare_prefix }
}

/// Load a SmolVLM / SmolVLM2 (`smolvlm`) VLM (SigLIP + pixel-shuffle connector
/// + SmolLM2 language model).
pub(crate) fn load_smolvlm_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::smolvlm::SmolVLMProcessor;
    use vision::smolvlm::{SmolVLMConnector, SmolVLMModel};

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Vision config lives in the `vision_config` sub-object (SigLIP).
    let vision_config: vision::config::VisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "SmolVLM vision config")?;

    // Text config is the `text_config` sub-object (SmolLM2 = Llama).
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in config.json"))?;

    // Inherit the top-level quantization into the text config when the text
    // config does not carry its own.
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(&mut text_config_value, "SmolVLM text_config")?
            .insert("quantization".to_string(), q.clone());
    }

    let text_args: models::llama3::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse SmolVLM text_config: {}", e))?;

    let group_size = text_args.group_size();
    let bits = text_args.bits();

    // Load weights. `load_vlm_weights_common` converts non-quantized bf16 -> f16
    // on Apple Silicon; for quantized checkpoints it keeps bf16, so we convert
    // the remaining plain bf16 tensors ourselves (keeping quant scales/biases).
    let mut weights = load_vlm_weights_common(model_path, None)?;
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown
        && text_args.quantization.is_some()
    {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    // Build the SmolLM2 (Llama) text backbone from the remapped text subset.
    let text_weights = smolvlm_text_weights(&weights);
    let text_model = models::Llama3Model::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load SmolVLM text model: {}", e))?;

    // Build the SigLIP vision tower.
    let vision_prefix = resolve_prefix(&weights, "model.vision_model", "vision_model");
    let vision_model = SigLipVisionModel::from_weights_with_quant(
        &weights,
        &vision_config,
        vision_prefix,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load SmolVLM SigLIP vision tower: {}", e))?;

    // Build the pixel-shuffle connector.
    let scale_factor = full_config
        .get("scale_factor")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_SCALE_FACTOR as i64) as i32;
    let connector_prefix = resolve_prefix(&weights, "model.connector", "connector");
    let connector =
        SmolVLMConnector::from_weights(&weights, connector_prefix, scale_factor, group_size, bits)
            .map_err(|e| anyhow::anyhow!("Failed to load SmolVLM connector: {}", e))?;

    // Image processor. The splitting policy comes from processor_config.json
    // when present.
    let processor_config = std::fs::read_to_string(model_path.join("processor_config.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok());

    // Derive num_image_token from the vision config so it always equals the
    // runtime pixel_shuffle output per tile, `(side / scale_factor)^2`, where
    // `side = image_size / patch_size` is the SigLIP patch grid side. This keeps
    // the merge invariant exact (image-feature rows == `<image>` placeholders);
    // for a valid checkpoint it equals the processor's `image_seq_len`.
    let image_size = vision_config.image_size;
    let side = (image_size / vision_config.patch_size.max(1)) / (scale_factor.max(1) as usize);
    let num_image_token = (side * side).max(1);

    let do_image_splitting = processor_config
        .as_ref()
        .and_then(|c| c.get("do_image_splitting"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Cap the tile grid using the outer longest-edge budget when available.
    let max_splits_per_side = processor_config
        .as_ref()
        .and_then(|c| c.get("size"))
        .and_then(|s| s.get("longest_edge"))
        .and_then(|v| v.as_u64())
        .map(|edge| (edge as usize).div_ceil(image_size.max(1)).max(1))
        .unwrap_or(4);

    let processor = SmolVLMProcessor::new(image_size, do_image_splitting, max_splits_per_side);

    // Resolve token ids from config + the tokenizer's added tokens.
    let image_token_id = full_config
        .get("image_token_id")
        .and_then(|v| v.as_i64())
        .or_else(|| {
            full_config
                .get("image_token_index")
                .and_then(|v| v.as_i64())
        })
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID as i64) as i32;

    let added_tokens = std::fs::read_to_string(model_path.join("added_tokens.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok());
    let fake_image_token_id =
        resolve_added_token_id(added_tokens.as_ref(), "<fake_token_around_image>");
    let global_image_token_id = resolve_added_token_id(added_tokens.as_ref(), "<global-img>");
    let eos_token_ids = resolve_eos_token_ids(&full_config, added_tokens.as_ref());

    let vlm = SmolVLMModel {
        text_model,
        vision_model,
        connector,
        processor,
        image_token_id,
        fake_image_token_id,
        global_image_token_id,
        num_image_token,
        eos_token_ids,
    };

    Ok(LoadedModel::SmolVLM(vlm))
}

#[cfg(test)]
mod tests {
    use super::remap_smolvlm_text_key;
    use serde_json::json;

    #[test]
    fn idefics3_text_config_parses_into_llama_args() {
        // The `text_config` sub-object of a real SmolVLM-Instruct (idefics3)
        // config.json must deserialize into the Llama backbone args, including
        // the untied-head flag the loader relies on to locate `lm_head`.
        let text_config = json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 64,
            "rms_norm_eps": 1e-05,
            "rope_theta": 273768.0,
            "vocab_size": 49155,
            "tie_word_embeddings": false
        });

        let args: crate::models::llama3::ModelArgs =
            serde_json::from_value(text_config).expect("idefics3 text_config parses as Llama args");

        assert_eq!(args.hidden_size, 2048);
        assert_eq!(args.num_hidden_layers, 24);
        assert_eq!(args.num_attention_heads, 32);
        assert_eq!(args.num_key_value_heads, Some(32));
        assert_eq!(args.head_dim, Some(64));
        assert_eq!(args.vocab_size, 49155);
        // Untied head: the loader must fetch a real top-level `lm_head`.
        assert!(!args.tie_word_embeddings);
    }

    #[test]
    fn idefics3_vision_config_parses_into_siglip_vision_config() {
        // The `vision_config` sub-object (model_type "idefics3") is a SigLIP
        // tower and must deserialize into mlxcel's VisionConfig.
        let vision_config = json!({
            "model_type": "idefics3",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "patch_size": 14,
            "image_size": 384
        });

        let cfg: crate::vision::config::VisionConfig =
            serde_json::from_value(vision_config).expect("idefics3 vision_config parses");

        assert_eq!(cfg.model_type, "idefics3");
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.image_size, 384);
    }

    #[test]
    fn remap_promotes_idefics3_nested_lm_head_to_top_level() {
        // idefics3 nests the whole Llama-with-head under `language_model.*`.
        // The untied head must land at the top-level `lm_head.*` key that
        // Llama3Model::from_weights reads; the backbone maps to `model.*`.
        assert_eq!(
            remap_smolvlm_text_key("language_model.embed_tokens.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            remap_smolvlm_text_key("language_model.layers.0.self_attn.q_proj.weight").as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            remap_smolvlm_text_key("language_model.norm.weight").as_deref(),
            Some("model.norm.weight")
        );
        // The critical regression: the nested head is promoted to top-level,
        // not left under `model.lm_head.*` where the loader never reads it.
        assert_eq!(
            remap_smolvlm_text_key("language_model.lm_head.weight").as_deref(),
            Some("lm_head.weight")
        );
        // Vision tower and connector keys are not text weights.
        assert!(remap_smolvlm_text_key("vision_model.post_layernorm.weight").is_none());
        assert!(remap_smolvlm_text_key("connector.modality_projection.proj.weight").is_none());
    }

    #[test]
    fn remap_preserves_smolvlm2_layout() {
        // SmolVLM/SmolVLM2 keep the backbone under `model.text_model.*` with a
        // separate top-level untied head; that mapping stays unchanged.
        assert_eq!(
            remap_smolvlm_text_key("model.text_model.embed_tokens.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            remap_smolvlm_text_key("model.text_model.layers.3.mlp.gate_proj.weight").as_deref(),
            Some("model.layers.3.mlp.gate_proj.weight")
        );
        assert_eq!(
            remap_smolvlm_text_key("lm_head.weight").as_deref(),
            Some("lm_head.weight")
        );
    }
}
