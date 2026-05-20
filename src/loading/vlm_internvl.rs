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

//! InternVL (`internvl_chat`) VLM loader.
//!
//! `internvl3-1b` composition (verified from `config.json` + tensor names):
//! - Language model: **Qwen2** (`text_config.model_type == "qwen2"`,
//!   `Qwen2ForCausalLM`, hidden=896, 24 layers, 14 heads, 2 kv-heads, QKV
//!   bias, `tie_word_embeddings=false`). Reuses mlxcel's existing Qwen2 /
//!   Llama backbone — NOT InternLM.
//! - Vision tower: InternViT (`vision_model.*`), non-quantized bf16.
//! - Connector `mlp1.*`: LayerNorm (bf16) + two 4-bit Linears, with a
//!   `pixel_shuffle(0.5)` in front.
//!
//! Mixed precision: the top-level `quantization` block makes
//! [`load_vlm_weights_common`] treat the checkpoint as quantized and skip its
//! bf16 -> f16 conversion (so the quantized LLM keeps bf16 scales/biases for
//! `quantized_matmul`). The InternViT tower and the connector LayerNorm are
//! plain bf16, which would JIT-crash on M5, so this loader explicitly converts
//! every remaining bf16 tensor to f16 **except** quantization `.scales` /
//! `.biases` (which must stay bf16). This mirrors the standard Apple Silicon
//! bf16 -> f16 policy for non-quantized weights documented in
//! `docs/apple-silicon-precision.md`.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};

/// Default InternVL3 image-framing token ids (resolved from the tokenizer
/// when present; these are the values in `internvl3-1b`).
const DEFAULT_IMG_CONTEXT_TOKEN_ID: i32 = 151667; // <IMG_CONTEXT>
const DEFAULT_IMG_START_TOKEN_ID: i32 = 151665; // <img>
const DEFAULT_IMG_END_TOKEN_ID: i32 = 151666; // </img>

/// Resolve a special token id from the tokenizer's `added_tokens.json`,
/// falling back to `default` when the file or key is absent.
fn resolve_added_token_id(added_tokens: Option<&Value>, name: &str, default: i32) -> i32 {
    added_tokens
        .and_then(|v| v.get(name))
        .and_then(|id| id.as_i64())
        .map(|id| id as i32)
        .unwrap_or(default)
}

/// Resolve the EOS/stop token ids for the Qwen2 backbone. The `Llama3Model`
/// trait default returns Llama-3 ids, which are wrong for Qwen2, so the
/// server stop path needs the correct ids supplied here. We include both the
/// configured `eos_token_id` and `<|im_end|>` (the chat-template turn
/// terminator) when available.
fn resolve_eos_token_ids(full_config: &Value, added_tokens: Option<&Value>) -> Vec<i32> {
    let mut ids = Vec::new();

    if let Some(eos) = full_config
        .get("text_config")
        .and_then(|tc| tc.get("eos_token_id"))
        .and_then(|v| v.as_i64())
    {
        ids.push(eos as i32);
    }

    // `<|im_end|>` is the Qwen2 chat-template turn terminator.
    if let Some(im_end) = added_tokens
        .and_then(|v| v.get("<|im_end|>"))
        .and_then(|id| id.as_i64())
    {
        let im_end = im_end as i32;
        if !ids.contains(&im_end) {
            ids.push(im_end);
        }
    }

    if ids.is_empty() {
        // Qwen2 defaults: <|endoftext|> and <|im_end|>.
        ids = vec![151643, 151645];
    }
    ids
}

/// Load an InternVL (`internvl_chat`) VLM (InternViT + pixel-shuffle `mlp1`
/// connector + Qwen2 language model).
pub(crate) fn load_internvl_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::internvl::{InternVitConfig, InternVitVisionModel};
    use vision::internvl::{InternVLChatVLM, InternVLConnector};
    use vision::processors::internvl::InternVLProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    // Vision config lives in the `vision_config` sub-object.
    let vision_config: InternVitConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "InternVL vision config")?;

    // Text config is the real `text_config` sub-object (NOT the bogus
    // `llm_config`, which carries the upstream 32B template dimensions).
    let mut text_config_value = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in config.json"))?;

    // Inherit the top-level quantization into the text config when the text
    // config does not carry its own (the LLM weights are 4-bit).
    if text_config_value.get("quantization").is_none()
        && let Some(q) = full_config.get("quantization")
    {
        super::require_object_mut(&mut text_config_value, "InternVL text_config")?
            .insert("quantization".to_string(), q.clone());
    }

    let text_args: models::llama3::ModelArgs = serde_json::from_value(text_config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse InternVL text_config: {}", e))?;

    // Load weights. `load_vlm_weights_common` skips bf16 -> f16 because the
    // model is quantized, so we convert the plain bf16 tensors ourselves
    // while keeping quantization scales/biases as bf16.
    let mut weights = load_vlm_weights_common(model_path, None)?;
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            // Keep quantization parameters as bf16 (the `quantized_matmul`
            // path consumes them directly); convert everything else (the
            // InternViT tower, the connector LayerNorm, and the q/k/v linear
            // `.bias` tensors) to f16.
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    // Quantization parameters for the 4-bit connector Linears.
    let group_size = text_args.group_size();
    let bits = text_args.bits();

    // Build the Qwen2 text backbone. The weight keys are
    // `language_model.model.*` / `language_model.lm_head.*`; the Qwen2 loader
    // expects `model.*` / `lm_head.*`, so strip the `language_model.` prefix.
    let text_weights = super::strip_language_model_prefix(weights_subset_language(&weights));
    let text_model = models::Qwen2Model::from_weights(&text_weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load InternVL Qwen2 text model: {}", e))?;

    // Build the InternViT vision tower (`vision_model.*`).
    let vision_model = InternVitVisionModel::from_weights(&weights, &vision_config, "vision_model")
        .map_err(|e| anyhow::anyhow!("Failed to load InternViT vision tower: {}", e))?;

    // Build the `mlp1` connector. `downsample_ratio` lives at the top level.
    let downsample_ratio = full_config
        .get("downsample_ratio")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let connector = InternVLConnector::from_weights(
        &weights,
        "mlp1",
        vision_config.layer_norm_eps,
        downsample_ratio,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load InternVL mlp1 connector: {}", e))?;

    // Image processor (dynamic tiling).
    let image_size = full_config
        .get("force_image_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(vision_config.image_size as u64) as usize;
    let min_dynamic_patch = full_config
        .get("min_dynamic_patch")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let max_dynamic_patch = full_config
        .get("max_dynamic_patch")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let use_thumbnail = full_config
        .get("use_thumbnail")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let processor = InternVLProcessor::new(
        image_size,
        min_dynamic_patch,
        max_dynamic_patch,
        use_thumbnail,
    );

    // num_image_token = (image_size / patch_size)^2 * downsample_ratio^2.
    let patches_per_side = image_size / vision_config.patch_size;
    let num_image_token = ((patches_per_side * patches_per_side) as f32
        * downsample_ratio
        * downsample_ratio)
        .round() as usize;

    // Resolve image-framing + EOS token ids from the tokenizer's added tokens
    // (parsed once), falling back to the InternVL3 defaults.
    let added_tokens = std::fs::read_to_string(model_path.join("added_tokens.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok());
    let image_context_token_id = resolve_added_token_id(
        added_tokens.as_ref(),
        "<IMG_CONTEXT>",
        DEFAULT_IMG_CONTEXT_TOKEN_ID,
    );
    let img_start_token_id =
        resolve_added_token_id(added_tokens.as_ref(), "<img>", DEFAULT_IMG_START_TOKEN_ID);
    let img_end_token_id =
        resolve_added_token_id(added_tokens.as_ref(), "</img>", DEFAULT_IMG_END_TOKEN_ID);
    let eos_token_ids = resolve_eos_token_ids(&full_config, added_tokens.as_ref());

    let vlm = InternVLChatVLM {
        text_model,
        vision_model,
        connector,
        processor,
        image_context_token_id,
        img_start_token_id,
        img_end_token_id,
        num_image_token,
        eos_token_ids,
    };

    Ok(LoadedModel::InternVLChatVLM(vlm))
}

/// Collect only the `language_model.*` keys so the Qwen2 backbone loader does
/// not see vision / connector tensors. (`strip_language_model_prefix` then
/// rewrites `language_model.model.*` -> `model.*` etc.)
fn weights_subset_language(
    weights: &mlxcel_core::weights::WeightMap,
) -> mlxcel_core::weights::WeightMap {
    let mut out = mlxcel_core::weights::WeightMap::new();
    for (key, value) in weights.iter() {
        if key.starts_with("language_model.") {
            out.insert(key.clone(), mlxcel_core::copy(value));
        }
    }
    out
}
