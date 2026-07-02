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

//! Idefics2 (`idefics2`) VLM loader.
//!
//! Idefics2 pairs a SigLIP vision tower with a perceiver-resampler connector and
//! a Mistral text backbone. The on-disk layout
//! (`Idefics2ForConditionalGeneration`) nests the whole Mistral-with-head under
//! `language_model.*`, exactly like the Idefics3 layout the SmolVLM loader
//! handles, so the text weights are remapped into the plain Llama key layout
//! [`crate::models::Llama3Model`] expects (see [`remap_idefics2_text_key`]).
//!
//! The mlx-community 4-bit checkpoint strips the Mistral dimensions from
//! `text_config` (only `vocab_size`, `rms_norm_eps`, `max_position_embeddings`,
//! `pad_token_id` survive). [`normalize_text_config`] fills the released
//! `HuggingFaceM4/idefics2-8b` text config (Mistral-7B-v0.1) as explicit
//! defaults, all confirmed against the checkpoint's quantized weight shapes
//! (`q_proj` 4096, `k_proj` 1024 -> 8 kv heads, 32 layers, intermediate 14336).
//! Any field the checkpoint does carry wins over the default.
//!
//! On Apple Silicon, non-quantized bf16 weights are converted to f16 by
//! [`load_vlm_weights_common`]. For quantized checkpoints that conversion is
//! skipped (to preserve quantization scales/biases), so this loader converts the
//! remaining plain bf16 tensors (vision tower, connector, norms) to f16 while
//! keeping `.scales` / `.biases` as bf16, mirroring the SmolVLM/InternVL loader.

use anyhow::Result;
use serde_json::{Map, Value};
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::idefics2::Idefics2PerceiverHeads;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};

/// Released Idefics2 `<image>` placeholder id (`image_token_id` in config.json).
const DEFAULT_IMAGE_TOKEN_ID: i32 = 32001;
/// Mistral default EOS (`</s>`).
const MISTRAL_EOS_ID: i32 = 2;

/// Resolve a special token id from the tokenizer's `added_tokens.json`,
/// returning `0` (unknown) when the file or key is absent.
fn resolve_added_token_id(added_tokens: Option<&Value>, name: &str) -> i32 {
    added_tokens
        .and_then(|v| v.get(name))
        .and_then(|id| id.as_i64())
        .map(|id| id as i32)
        .unwrap_or(0)
}

/// Fill missing Mistral-7B-v0.1 fields into the Idefics2 `text_config` so it
/// deserializes into [`models::llama3::ModelArgs`]. Values already present in the
/// checkpoint win over these defaults.
fn normalize_text_config(text_config: &mut Value, full_config: &Value) -> Result<()> {
    let obj = super::require_object_mut(text_config, "Idefics2 text_config")?;

    // HuggingFaceM4/idefics2-8b text_config == Mistral-7B-v0.1. Any field the
    // checkpoint already carries wins over these defaults.
    let defaults: [(&str, Value); 10] = [
        ("hidden_size", Value::from(4096)),
        ("num_hidden_layers", Value::from(32)),
        ("intermediate_size", Value::from(14336)),
        ("num_attention_heads", Value::from(32)),
        ("num_key_value_heads", Value::from(8)),
        ("head_dim", Value::from(128)),
        ("rms_norm_eps", Value::from(1e-5)),
        ("vocab_size", Value::from(32003)),
        ("rope_theta", Value::from(10000.0)),
        ("tie_word_embeddings", Value::from(false)),
    ];
    for (key, value) in defaults {
        obj.entry(key.to_string()).or_insert(value);
    }

    // Inherit the top-level quantization when the text config lacks its own.
    if !obj.contains_key("quantization")
        && let Some(q) = full_config.get("quantization")
    {
        obj.insert("quantization".to_string(), q.clone());
    }
    Ok(())
}

/// Map one checkpoint text-weight key into the plain Llama key layout the
/// [`models::Llama3Model`] loader expects, returning `None` for non-text keys
/// (vision tower / connector keys are handled separately).
///
/// Idefics2 nests the whole Mistral-with-head under `language_model.*`, so the
/// untied head `language_model.lm_head.*` is promoted to the top-level `lm_head.*`
/// key that [`models::Llama3Model::from_weights`] reads.
fn remap_idefics2_text_key(key: &str) -> Option<String> {
    if let Some(rest) = key.strip_prefix("language_model.") {
        if let Some(head) = rest.strip_prefix("lm_head.") {
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

/// Collect the Mistral text weights and rewrite them into the plain Llama key
/// layout. See [`remap_idefics2_text_key`].
fn idefics2_text_weights(
    weights: &mlxcel_core::weights::WeightMap,
) -> mlxcel_core::weights::WeightMap {
    let mut out = mlxcel_core::weights::WeightMap::new();
    for (key, value) in weights.iter() {
        if let Some(dest) = remap_idefics2_text_key(key) {
            out.insert(dest, mlxcel_core::copy(value));
        }
    }
    out
}

/// Parse the perceiver head split from `perceiver_config`, falling back to the
/// released Idefics2 defaults when the 4-bit config strips the fields.
fn resolve_perceiver_heads(full_config: &Value) -> Idefics2PerceiverHeads {
    let pc = full_config.get("perceiver_config");
    let get = |key: &str| -> Option<usize> {
        pc.and_then(|c| c.get(key))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    };
    let d = Idefics2PerceiverHeads::default();
    Idefics2PerceiverHeads {
        n_heads: get("resampler_n_heads").unwrap_or(d.n_heads),
        head_dim: get("resampler_head_dim").unwrap_or(d.head_dim),
        n_kv_heads: get("num_key_value_heads").unwrap_or(d.n_kv_heads),
    }
}

/// Load an Idefics2 (`idefics2`) VLM (SigLIP + perceiver-resampler connector +
/// Mistral language model).
pub(crate) fn load_idefics2_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::idefics2::{Idefics2Connector, Idefics2Model};
    use vision::processors::idefics2::Idefics2Processor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Vision config (`vision_config`) is a SigLIP tower.
    let vision_config: vision::config::VisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Idefics2 vision config")?;

    // Text config (`text_config`) is Mistral; normalize the stripped 4-bit
    // config into full Mistral-7B args.
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    normalize_text_config(&mut text_config_value, &full_config)?;

    let text_args: models::llama3::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Idefics2 text_config: {}", e))?;

    let group_size = text_args.group_size();
    let bits = text_args.bits();

    // Load weights; convert leftover plain bf16 tensors to f16 on Apple Silicon
    // for quantized checkpoints (keeping quant scales/biases as bf16).
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

    // Mistral text backbone from the remapped `language_model.*` subset.
    let text_weights = idefics2_text_weights(&weights);
    let text_model = models::Llama3Model::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Idefics2 text model: {}", e))?;

    // SigLIP vision tower (`vision_model.*`). Idefics2's vision MLP uses the
    // sigmoid `GELU(approx="fast")` (not the tanh GELU the other SigLIP consumers
    // use), so request the fast-GELU encoder variant.
    let vision_model = SigLipVisionModel::from_weights_with_quant_and_gelu(
        &weights,
        &vision_config,
        "vision_model",
        group_size,
        bits,
        true,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Idefics2 SigLIP vision tower: {}", e))?;

    // Promote the connector's f16 params (quant scales/biases, latents, norm
    // weights) to f32. The modality-projection SwiGLU (14336-wide) and the
    // perceiver MLP (16384-wide) overflow f16 to inf on some images; the
    // reference avoids this by running the connector at the f32 pixel dtype.
    // The packed int4 `.weight` tensors stay as-is. `get_input_embeddings` feeds
    // the vision output as f32 so the whole connector computes in f32.
    let connector_f16_keys: Vec<String> = weights
        .iter()
        .filter(|(k, v)| {
            k.starts_with("connector.")
                && mlxcel_core::array_dtype(v) == mlxcel_core::dtype::FLOAT16
        })
        .map(|(k, _)| k.clone())
        .collect();
    for key in connector_f16_keys {
        if let Some(v) = weights.get(&key) {
            let promoted = mlxcel_core::astype(v, mlxcel_core::dtype::FLOAT32);
            weights.insert(key, promoted);
        }
    }

    // Perceiver-resampler connector (`connector.*`).
    let heads = resolve_perceiver_heads(&full_config);
    let connector = Idefics2Connector::from_weights(
        &weights,
        "connector",
        &text_args,
        heads,
        text_args.rms_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Idefics2 connector: {}", e))?;
    let num_image_token = connector.num_image_token();

    let patch_size = vision_config.patch_size.max(1);
    let num_patches_per_side = (vision_config.image_size / patch_size).max(1);
    let processor = Idefics2Processor::new(vision_config.image_size, patch_size);

    // Token ids.
    let image_token_id = full_config
        .get("image_token_id")
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_IMAGE_TOKEN_ID as i64) as i32;

    let added_tokens = std::fs::read_to_string(model_path.join("added_tokens.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok());
    let fake_image_token_id =
        resolve_added_token_id(added_tokens.as_ref(), "<fake_token_around_image>");

    // EOS: Mistral `</s>` (2) plus `<end_of_utterance>` (the chat-template turn
    // terminator) when the tokenizer exposes it.
    let mut eos_token_ids = vec![MISTRAL_EOS_ID];
    let eou = resolve_added_token_id(added_tokens.as_ref(), "<end_of_utterance>");
    if eou != 0 && !eos_token_ids.contains(&eou) {
        eos_token_ids.push(eou);
    }

    let vlm = Idefics2Model {
        text_model,
        vision_model,
        connector,
        processor,
        image_token_id,
        fake_image_token_id,
        num_image_token,
        patch_size,
        num_patches_per_side,
        eos_token_ids,
    };

    Ok(LoadedModel::Idefics2(vlm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn remap_promotes_language_model_head_to_top_level() {
        assert_eq!(
            remap_idefics2_text_key("language_model.embed_tokens.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            remap_idefics2_text_key("language_model.layers.0.self_attn.q_proj.weight").as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            remap_idefics2_text_key("language_model.norm.weight").as_deref(),
            Some("model.norm.weight")
        );
        assert_eq!(
            remap_idefics2_text_key("language_model.lm_head.weight").as_deref(),
            Some("lm_head.weight")
        );
        // Vision tower / connector keys are not text weights.
        assert!(remap_idefics2_text_key("vision_model.post_layernorm.weight").is_none());
        assert!(remap_idefics2_text_key("connector.perceiver_resampler.latents").is_none());
    }

    #[test]
    fn normalize_fills_mistral_defaults_and_respects_overrides() {
        // Stripped 4-bit text_config.
        let mut tc = json!({
            "model_type": "mistral",
            "vocab_size": 32003,
            "rms_norm_eps": 1e-5,
            "num_hidden_layers": 40
        });
        let full = json!({ "quantization": { "group_size": 64, "bits": 4 } });
        normalize_text_config(&mut tc, &full).unwrap();

        let args: crate::models::llama3::ModelArgs =
            serde_json::from_value(tc).expect("normalized text_config parses as Llama args");
        // Default filled.
        assert_eq!(args.hidden_size, 4096);
        assert_eq!(args.num_attention_heads, 32);
        assert_eq!(args.num_key_value_heads, Some(8));
        assert_eq!(args.intermediate_size, 14336);
        assert_eq!(args.vocab_size, 32003);
        assert!(!args.tie_word_embeddings);
        // Checkpoint value wins over the default (40, not 32).
        assert_eq!(args.num_hidden_layers, 40);
        // Top-level quantization inherited.
        assert!(args.quantization.is_some());
    }

    #[test]
    fn perceiver_heads_default_when_config_stripped() {
        let full = json!({ "perceiver_config": { "model_type": "idefics2" } });
        let heads = resolve_perceiver_heads(&full);
        assert_eq!(heads.n_heads, 16);
        assert_eq!(heads.head_dim, 96);
        assert_eq!(heads.n_kv_heads, 4);
    }
}
