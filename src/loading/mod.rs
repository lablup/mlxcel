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

//! Shared model-loading entry points and routing helpers.
//!
//! Family-specific registries live in sibling modules while this file keeps the
//! public `load_model*` APIs thin and focused on route selection. Model kind
//! and route policy live in `src/model_metadata.rs`.
//!
//! Rationale:
//! - keep directory and weight loading policy in one place
//! - keep family-specific construction out of the public router
//! - make new model additions update policy tables before implementation code

use anyhow::Result;
use serde::de::DeserializeOwned;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use crate::LoadedModel;
use crate::distributed::{
    ShardConfig, TensorParallelErnie45Model, TensorParallelGemma3Model, TensorParallelGemma4Model,
    TensorParallelHunyuanV1DenseModel, TensorParallelLlamaModel, TensorParallelQwen3Model,
    TensorParallelQwen35Model, validate_supported_runtime,
};
use crate::lora;
use crate::model_metadata::{
    DirectoryLoadRoute, ModelLoadPolicy, WeightLoadRoute, is_ministral3_config, is_mistral4_config,
    model_load_policy,
};
use crate::models::{self, ModelType, get_model_type, sanitize_config_json};
use crate::tokenizer::{self, MlxcelTokenizer};
use mlxcel_core::weights::WeightMap;

pub(crate) mod config_backed;
pub(crate) mod nonstandard;
pub(crate) mod special;
mod vlm;

use self::config_backed::{
    try_load_config_backed_model_from_dir, try_load_config_backed_model_from_weights,
};
use self::nonstandard::try_load_nonstandard_model_from_dir;
use self::special::try_load_special_model_from_weights;
use self::vlm::*;

/// Resolve model path: if a file is given, use its parent directory.
///
/// This provides compatibility with callers that pass a specific model file
/// (e.g. `/path/to/model.safetensors`) instead of the model directory.
fn resolve_model_dir(model_path: &Path) -> PathBuf {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    } else {
        model_path.to_path_buf()
    }
}

fn parse_eos_token_ids(config: &serde_json::Value) -> Vec<i32> {
    match config.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            if let Some(id) = n.as_i64() {
                vec![id as i32]
            } else {
                Vec::new()
            }
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect(),
        _ => Vec::new(),
    }
}

pub(super) fn parse_model_config<T: DeserializeOwned>(config_str: &str) -> Result<T> {
    serde_json::from_str(config_str).map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))
}

/// Layout-detection guard for conv weights loaded from a checkpoint.
///
/// Several loaders transpose conv weights from PyTorch to MLX channel-last
/// layout while sanitizing a checkpoint. Pre-converted `mlx-community`
/// checkpoints already store these weights channel-last, so an unconditional
/// transpose double-converts and corrupts the shape (see issue #428). These
/// helpers detect the layout from the shape so the transpose fires only for
/// genuine PyTorch-layout weights, which is safe in both directions: a true
/// PyTorch weight is still transposed and an already-MLX weight is left as-is.
///
/// Returns `true` when a 4D conv2d weight is already in MLX channel-last
/// `[out, kH, kW, in]` layout (skip the transpose). PyTorch layout is
/// `[out, in, kH, kW]`.
///
/// Heuristic: in channel-last layout the output-channel count is the leading
/// dim and dominates both spatial dims, and the two kernel dims are equal
/// (square kernels are the norm for these patch/subsample/backbone convs).
/// PyTorch layout puts `in` in dim 1, which generally breaks `dim1 == dim2`
/// (e.g. `[128, 1, 3, 3]` has `dim1=1 != dim2=3`) or `out >= dim1` once the
/// in-channel count exceeds the output count. This mirrors the validated
/// `should_transpose_phi3_patch_embedding` predicate.
#[must_use]
pub(crate) fn conv2d_weight_is_channel_last(shape: &[i32]) -> bool {
    shape.len() == 4 && shape[0] >= shape[1] && shape[0] >= shape[2] && shape[1] == shape[2]
}

/// Returns `true` when a 3D depthwise conv1d weight is already in MLX
/// channel-last `[out, kW, in]` layout (skip the transpose). PyTorch layout is
/// `[out, in, kW]`.
///
/// This predicate is valid only for depthwise kernels, where the in-channel
/// count per group is `1`. In that case channel-last `[out, kW, 1]` carries the
/// `1` in the trailing dim, while PyTorch `[out, 1, kW]` carries it in the
/// middle dim. Detecting `shape[2] == 1 && shape[1] > 1` therefore distinguishes
/// the two depthwise layouts unambiguously: the confirmed Gemma 4 audio kernel
/// `[1024, 5, 1]` is recognized as already-MLX (skip), and its PyTorch form
/// `[1024, 1, 5]` as needing the transpose.
///
/// It does NOT generalize to pointwise conv1d (`kernel == 1`): PyTorch pointwise
/// `[out, in, 1]` and MLX depthwise `[out, kW, 1]` share the same shape
/// signature and cannot be told apart from shape alone. Callers that may receive
/// pointwise weights must scope this guard to depthwise keys.
#[must_use]
pub(crate) fn conv1d_weight_is_channel_last(shape: &[i32]) -> bool {
    shape.len() == 3 && shape[2] == 1 && shape[1] > 1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35VlmKind {
    Dense,
    Moe,
}

fn qwen35_vlm_kind(model_type: ModelType) -> Option<Qwen35VlmKind> {
    match model_type {
        ModelType::Qwen35VLM => Some(Qwen35VlmKind::Dense),
        ModelType::Qwen35MoeVLM => Some(Qwen35VlmKind::Moe),
        _ => None,
    }
}

fn require_qwen35_vlm_kind(model_type: ModelType) -> Result<Qwen35VlmKind> {
    qwen35_vlm_kind(model_type).ok_or_else(|| {
        anyhow::anyhow!(
            "Expected a Qwen3.5 VLM variant but got model type: {:?}",
            model_type
        )
    })
}

fn try_load_vlm_model_from_dir(
    model_type: ModelType,
    model_path: &Path,
) -> Result<Option<LoadedModel>> {
    Ok(match model_type {
        ModelType::Llama4VLM => Some(load_llama4_vlm(model_path)?),
        ModelType::Qwen35VLM | ModelType::Qwen35MoeVLM => {
            Some(match require_qwen35_vlm_kind(model_type)? {
                Qwen35VlmKind::Dense => load_qwen3_5_vlm(model_path)?,
                Qwen35VlmKind::Moe => load_qwen3_5_moe_vlm(model_path)?,
            })
        }
        ModelType::Gemma3VLM => Some(load_gemma3_vlm(model_path)?),
        ModelType::Gemma4VLM => Some(load_gemma4_vlm(model_path)?),
        ModelType::Gemma4Unified => Some(load_gemma4_unified(model_path)?),
        ModelType::LlavaVLM => Some(load_llava_vlm(model_path)?),
        ModelType::LlavaBunnyVLM => Some(load_llava_bunny_vlm(model_path)?),
        ModelType::AyaVisionVLM => Some(load_aya_vision_vlm(model_path)?),
        ModelType::PaliGemmaVLM => Some(load_paligemma_vlm(model_path)?),
        ModelType::PixtralVLM => Some(load_pixtral_vlm(model_path)?),
        ModelType::Mistral3VLM => Some(load_mistral3_vlm(model_path)?),
        ModelType::Qwen2VL => Some(load_qwen2_vl(model_path)?),
        ModelType::Qwen25VL => Some(load_qwen2_5_vl(model_path)?),
        ModelType::Qwen3VL => Some(load_qwen3_vl(model_path)?),
        ModelType::Qwen3VLMoe => Some(load_qwen3_vl_moe(model_path)?),
        ModelType::MiniCPMOVLM => Some(load_minicpmo_vlm(model_path)?),
        ModelType::MiniCPMV46VLM => Some(load_minicpmv4_6_vlm(model_path)?),
        ModelType::Moondream3VLM => Some(load_moondream3_vlm(model_path)?),
        ModelType::Gemma3nVLM => Some(load_gemma3n_vlm(model_path)?),
        ModelType::Phi4MMVLM => Some(load_phi4mm_vlm(model_path)?),
        ModelType::Phi4SigLipVLM => Some(load_phi4_siglip_vlm(model_path)?),
        ModelType::Phi3VLM => Some(load_phi3_vlm(model_path)?),
        ModelType::MolmoVLM => Some(load_molmo_vlm(model_path)?),
        ModelType::Molmo2VLM => Some(load_molmo2_vlm(model_path)?),
        ModelType::MolmoPointVLM => Some(load_molmo_point_vlm(model_path)?),
        ModelType::NemotronHNanoOmniVLM => Some(load_nemotron_h_nano_omni_vlm(model_path)?),
        ModelType::YoutuVLM => Some(load_youtu_vl_vlm(model_path)?),
        ModelType::InternVLChatVLM => Some(load_internvl_vlm(model_path)?),
        _ => None,
    })
}

fn load_pair_from_dir<T, U, E, F>(path_str: &str, load: F) -> Result<T>
where
    F: FnOnce(String) -> std::result::Result<(T, U), E>,
    E: Display,
{
    let (model, _) = load(path_str.to_owned()).map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(model)
}

fn model_path_str(model_path: &Path) -> Result<&str> {
    model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Model path contains invalid UTF-8: {:?}", model_path))
}

fn load_mistral3_text_directory_variant(path_str: &str) -> Result<LoadedModel> {
    Ok(LoadedModel::Ministral3(models::Ministral3Wrapper::new(
        load_pair_from_dir(path_str, models::Ministral3Model::load_from_text_config)?,
    )))
}

fn load_mistral3_mistral4_directory_variant(path_str: &str) -> Result<LoadedModel> {
    Ok(LoadedModel::Mistral4(load_pair_from_dir(
        path_str,
        models::Mistral4Model::load_from_text_config,
    )?))
}

fn load_mistral3_llama_directory_variant(path_str: &str) -> Result<LoadedModel> {
    Ok(LoadedModel::Llama(load_pair_from_dir(
        path_str,
        models::Llama3Model::load,
    )?))
}

fn load_llama_family_from_weights(
    model_type: ModelType,
    config_str: &str,
    config: &serde_json::Value,
    weights: &mut WeightMap,
) -> Result<LoadedModel> {
    if model_type == ModelType::Mistral3 && is_ministral3_config(config) {
        let text_config = config
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("Missing text_config for Ministral3"))?;
        let args: models::ministral3::ModelArgs = serde_json::from_value(text_config.clone())
            .map_err(|err| anyhow::anyhow!("Failed to parse text_config: {}", err))?;
        let model = models::Ministral3Model::from_weights(weights, &args)
            .map_err(|err| anyhow::anyhow!("{}", err))?;
        return Ok(LoadedModel::Ministral3(models::Ministral3Wrapper::new(
            model,
        )));
    }

    if model_type == ModelType::Mistral3 && is_mistral4_config(config) {
        let text_config = config
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("Missing text_config for Mistral4"))?;
        let args: models::mistral4::Mistral4Config = serde_json::from_value(text_config.clone())
            .map_err(|err| anyhow::anyhow!("Failed to parse text_config: {}", err))?;
        models::mistral4::sanitize_weights(weights, &args, "");
        let model = models::Mistral4Model::from_weights(weights, &args)
            .map_err(|err| anyhow::anyhow!("{}", err))?;
        return Ok(LoadedModel::Mistral4(model));
    }

    let args: models::llama3::ModelArgs = parse_model_config(config_str)?;
    let model = models::Llama3Model::from_weights(weights, &args)
        .map_err(|err| anyhow::anyhow!("{}", err))?;
    Ok(LoadedModel::Llama(model))
}

/// Read EOS token IDs from generation_config.json
///
/// Returns the token IDs from the `eos_token_id` field, which can be either
/// a single integer or an array of integers. Returns empty vec if not found.
pub fn read_eos_token_ids(model_dir: &Path) -> Vec<i32> {
    let config_path = model_dir.join("generation_config.json");
    let Ok(content) = std::fs::read_to_string(&config_path) else {
        return Vec::new();
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) else {
        return Vec::new();
    };
    parse_eos_token_ids(&config)
}

/// Load a model from a directory (or file — parent directory will be used)
pub fn load_model(model_path: &Path) -> Result<(LoadedModel, MlxcelTokenizer)> {
    let model_path = resolve_model_dir(model_path);
    let model_path = model_path.as_path();
    let model_type = get_model_type(model_path)?;

    // Whisper is an encoder-decoder ASR model, not a text generator. It is
    // wired into the speech-to-text audio endpoints at server startup rather
    // than the LanguageModel path, so fail clearly here instead of attempting
    // a text-model load.
    if model_type == ModelType::Whisper {
        anyhow::bail!(
            "Whisper is a speech-to-text model served through the /v1/audio/* \
             endpoints, not a text-generation model. The mlxcel-server populates \
             its speech-to-text slot from this checkpoint automatically."
        );
    }

    let path_str = model_path_str(model_path)?;

    let policy = if model_type == ModelType::Mistral3 {
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = parse_model_config(&config_str)?;
        (model_load_policy(model_type, Some(&config))?, Some(config))
    } else {
        (model_load_policy(model_type, None)?, None)
    };

    let model = match policy {
        (
            ModelLoadPolicy {
                directory_route: DirectoryLoadRoute::Mistral3TextWrapper,
                ..
            },
            Some(_),
        ) => load_mistral3_text_directory_variant(path_str)?,
        (
            ModelLoadPolicy {
                directory_route: DirectoryLoadRoute::Mistral3Mistral4Wrapper,
                ..
            },
            Some(_),
        ) => load_mistral3_mistral4_directory_variant(path_str)?,
        (
            ModelLoadPolicy {
                directory_route: DirectoryLoadRoute::Mistral3LlamaFallback,
                ..
            },
            Some(_),
        ) => load_mistral3_llama_directory_variant(path_str)?,
        (
            ModelLoadPolicy {
                directory_route: DirectoryLoadRoute::Vlm,
                ..
            },
            _,
        ) => try_load_vlm_model_from_dir(model_type, model_path)?.ok_or_else(|| {
            anyhow::anyhow!("Missing directory loader for model type: {:?}", model_type)
        })?,
        (
            ModelLoadPolicy {
                directory_route: DirectoryLoadRoute::Nonstandard,
                ..
            },
            _,
        ) => try_load_nonstandard_model_from_dir(model_type, model_path, path_str)?.ok_or_else(
            || anyhow::anyhow!("Missing directory loader for model type: {:?}", model_type),
        )?,
        (
            ModelLoadPolicy {
                directory_route: DirectoryLoadRoute::ConfigBacked,
                ..
            },
            _,
        ) => try_load_config_backed_model_from_dir(model_type, path_str)?.ok_or_else(|| {
            anyhow::anyhow!("Missing directory loader for model type: {:?}", model_type)
        })?,
        (
            ModelLoadPolicy {
                directory_route:
                    DirectoryLoadRoute::Mistral3TextWrapper
                    | DirectoryLoadRoute::Mistral3Mistral4Wrapper
                    | DirectoryLoadRoute::Mistral3LlamaFallback,
                ..
            },
            None,
        ) => {
            unreachable!("Mistral3 routes require config context")
        }
    };

    let tokenizer = tokenizer::load_tokenizer(model_path)?;
    Ok((model, tokenizer))
}

pub fn load_model_with_tensor_parallel(
    model_path: &Path,
    adapter_path: Option<&Path>,
    shard_config: &ShardConfig,
) -> Result<(LoadedModel, MlxcelTokenizer)> {
    if shard_config.tp_size == 1 {
        return match adapter_path {
            Some(adapter) => load_model_with_adapter(model_path, adapter),
            None => load_model(model_path),
        };
    }

    let model_path = resolve_model_dir(model_path);
    let model_path = model_path.as_path();
    let support = validate_supported_runtime(model_path, shard_config.clone(), adapter_path)?;
    let model = match support.summary.model_type {
        ModelType::Llama | ModelType::Qwen2 => LoadedModel::TensorParallelLlama(
            TensorParallelLlamaModel::from_model_dir(model_path, shard_config.clone())?,
        ),
        ModelType::Qwen3 => LoadedModel::TensorParallelQwen3(
            TensorParallelQwen3Model::from_model_dir(model_path, shard_config.clone())?,
        ),
        ModelType::Qwen35 | ModelType::Qwen35VLM => LoadedModel::TensorParallelQwen35(
            TensorParallelQwen35Model::from_model_dir(model_path, shard_config.clone())?,
        ),
        ModelType::Gemma3 => LoadedModel::TensorParallelGemma3(
            TensorParallelGemma3Model::from_model_dir(model_path, shard_config.clone())?,
        ),
        ModelType::Gemma4 | ModelType::Gemma4VLM => LoadedModel::TensorParallelGemma4(
            TensorParallelGemma4Model::from_model_dir(model_path, shard_config.clone())?,
        ),
        ModelType::Ernie45 => LoadedModel::TensorParallelErnie45(
            TensorParallelErnie45Model::from_model_dir(model_path, shard_config.clone())?,
        ),
        ModelType::HunyuanV1Dense => LoadedModel::TensorParallelHunyuanV1Dense(
            TensorParallelHunyuanV1DenseModel::from_model_dir(model_path, shard_config.clone())?,
        ),
        other => anyhow::bail!(
            "tensor-parallel runtime does not support model type: {:?}",
            other
        ),
    };

    let tokenizer = tokenizer::load_tokenizer(model_path)?;
    Ok((model, tokenizer))
}

/// Load a model with LoRA adapter weights fused in
///
/// This loads the base model weights, fuses them with LoRA adapter weights,
/// then constructs the model from the fused weights.
pub fn load_model_with_adapter(
    model_path: &Path,
    adapter_path: &Path,
) -> Result<(LoadedModel, MlxcelTokenizer)> {
    let model_path = resolve_model_dir(model_path);
    let model_path = model_path.as_path();
    // Load base weights
    let base_weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Fuse with adapter weights
    let mut fused_weights = lora::apply_lora_adapters(&base_weights, adapter_path)?;

    // Build model from fused weights
    let model = load_model_from_weights(model_path, &mut fused_weights)?;

    let tokenizer = tokenizer::load_tokenizer(model_path)?;
    Ok((model, tokenizer))
}

/// Build a model from pre-loaded weights (used by adapter loading)
fn load_model_from_weights(model_path: &Path, weights: &mut WeightMap) -> Result<LoadedModel> {
    let model_type = get_model_type(model_path)?;
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config_str = sanitize_config_json(&config_str);

    // Sanitize tied embeddings: copy embed_tokens → lm_head if needed
    let config_value: serde_json::Value = parse_model_config(&config_str)?;
    models::sanitize_tied_embeddings(weights, &config_value);

    let policy = model_load_policy(model_type, Some(&config_value))?;

    if let Some(message) = policy.capabilities.adapter_unsupported_message {
        return Err(anyhow::anyhow!(message));
    }

    let weight_route = policy
        .weight_route
        .ok_or_else(|| anyhow::anyhow!("Missing adapter weight route for {:?}", model_type))?;

    let model = match weight_route {
        WeightLoadRoute::LlamaFamily => {
            load_llama_family_from_weights(model_type, &config_str, &config_value, weights)?
        }
        WeightLoadRoute::Special => {
            try_load_special_model_from_weights(model_type, &config_str, weights)?.ok_or_else(
                || anyhow::anyhow!("Missing weight loader for model type: {:?}", model_type),
            )?
        }
        WeightLoadRoute::ConfigBacked => {
            try_load_config_backed_model_from_weights(model_type, &config_str, weights)?
                .ok_or_else(|| {
                    anyhow::anyhow!("Missing weight loader for model type: {:?}", model_type)
                })?
        }
    };

    Ok(model)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
