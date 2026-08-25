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

//! `load_embedding_model`: the dispatcher every embedding family registers
//! with.
//!
//! Adding a family means adding one arm to [`build_family_model`] that
//! returns its `Box<dyn EmbeddingModel>`; tokenizer loading, pad-token
//! resolution, vocabulary size and the length limits are shared and happen
//! in [`finish_loaded_model`]. Weights come through
//! [`load_embedding_weights`], which walks module subfolders
//! (`2_Dense/...`) and applies the text-model bf16 rule.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir_with_subfolders};
use serde_json::Value;

use crate::model_metadata::is_embedding_model_type;
use crate::models::bert::BertVariant;
use crate::models::bert_heads::BertEmbeddingModel;
use crate::models::siglip_text::load_siglip_text_model;
use crate::models::{
    ModelType, config_has_quantization_metadata, convert_bf16_weights, get_model_type,
    sanitize_config_json, should_convert_bf16_to_f16, warn_bf16_precision,
};
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::limits::{
    EmbeddingLimits, is_absolute_position_family, resolve_pad_token_id, resolve_vocab_size,
};
use super::model::EmbeddingModel;
use super::tokenize::strip_padding_and_truncation;

/// Everything the engine needs to serve one embedding checkpoint.
pub struct LoadedEmbeddingModel {
    /// The family forward pass.
    pub model: Box<dyn EmbeddingModel>,
    /// Tokenizer with any built-in padding / truncation removed.
    pub tokenizer: MlxcelTokenizer,
    /// Sequence, width and multi-vector limits.
    pub limits: EmbeddingLimits,
    /// Id used to right-pad micro-batches.
    pub pad_token_id: u32,
    /// Vocabulary size token-id inputs are validated against.
    pub vocab_size: usize,
    /// Detected family.
    pub model_type: ModelType,
}

/// Operator overrides applied while loading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbeddingLoadOptions {
    /// `--embedding-max-length`: lowers the derived `max_length`.
    pub max_length: Option<usize>,
}

/// `config.json` `quantization` block (`{group_size, bits}`) a family passes
/// to `UnifiedLinear::from_weights` / `UnifiedEmbedding::from_weights`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationParams {
    pub group_size: i32,
    pub bits: i32,
}

/// Read the checkpoint's `quantization` block when present.
pub fn quantization_params(config: &Value) -> Option<QuantizationParams> {
    let block = config.get("quantization")?;
    Some(QuantizationParams {
        group_size: block.get("group_size")?.as_i64()? as i32,
        bits: block.get("bits")?.as_i64()? as i32,
    })
}

/// Read and sanitize `<model_dir>/config.json`.
pub fn read_embedding_config(model_dir: &Path) -> Result<Value> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&sanitize_config_json(&raw))
        .with_context(|| format!("failed to parse {}", path.display()))
}

/// Load every safetensors tensor of an embedding checkpoint, including
/// module subfolders, and apply the text-model precision rule: on Apple
/// Silicon non-quantized bf16 tensors convert to f16, while quantized
/// checkpoints (scales and biases included) stay bf16.
///
/// Used by: every embedding family loader.
pub fn load_embedding_weights(model_dir: &Path, config: &Value) -> Result<WeightMap> {
    let mut weights = load_weights_from_dir_with_subfolders(model_dir)
        .map_err(|e| anyhow::anyhow!("failed to load embedding weights: {e}"))?;
    if should_convert_bf16_to_f16()
        && !config_has_quantization_metadata(config)
        && convert_bf16_weights(&mut weights)
    {
        warn_bf16_precision();
    }
    Ok(weights)
}

/// Error for a detected embedding family whose port has not landed yet.
pub fn embedding_family_not_yet_supported(model_type: ModelType) -> anyhow::Error {
    anyhow::anyhow!(
        "{} ({model_type:?}) is detected as an embedding checkpoint, but this embedding \
         family is not yet supported by /v1/embeddings (epic #1348 tracks the port)",
        model_type.display_name()
    )
}

/// Construct the family model for a detected embedding variant.
///
/// Each family sub-issue replaces its `not yet supported` arm with the real
/// constructor. Families resolve their pooling mode through
/// [`super::pooling::resolve_pooling_mode`] and read weights through
/// [`load_embedding_weights`].
fn build_family_model(
    model_type: ModelType,
    model_dir: &Path,
    config: &Value,
) -> Result<Box<dyn EmbeddingModel>> {
    match model_type {
        ModelType::Bert | ModelType::XlmRoberta => {
            let variant = BertVariant::from_model_type(model_type)
                .expect("Bert and XlmRoberta both map to a BertVariant");
            Ok(Box::new(BertEmbeddingModel::load(
                model_dir, config, variant,
            )?))
        }
        ModelType::SiglipText => load_siglip_text_model(model_dir, config),
        ModelType::ModernBert => Ok(Box::new(
            crate::models::modernbert_heads::ModernBertEmbeddingModel::load(model_dir, config)?,
        )),
        ModelType::Gemma3Embedding => Ok(Box::new(
            crate::models::gemma3_embedding::Gemma3EmbeddingModel::load(model_dir, config)?,
        )),
        ModelType::Qwen3Embedding => Ok(Box::new(
            crate::models::qwen3_embedding::Qwen3EmbeddingModel::load(model_dir, config)?,
        )),
        ModelType::ColIdefics3 => Ok(Box::new(
            crate::models::colidefics3::ColIdefics3Model::load(model_dir, config)?,
        )),
        ModelType::ColQwen25 => Ok(Box::new(crate::models::colqwen2_5::ColQwen25Model::load(
            model_dir, config,
        )?)),
        ModelType::LlamaBidirec => Ok(Box::new(
            crate::models::llama_bidirec::LlamaBidirecModel::load(model_dir, config)?,
        )),
        ModelType::Ministral3Embedding => Ok(Box::new(
            crate::models::ministral3_embedding::Ministral3EmbeddingModel::load(model_dir, config)?,
        )),
        ModelType::Lfm2Embedding => Ok(Box::new(
            crate::models::lfm2_embedding::Lfm2EmbeddingModel::load(model_dir, config)?,
        )),
        ModelType::Qwen3VLEmbedding | ModelType::LlamaNemotronVLEmbedding => {
            Err(embedding_family_not_yet_supported(model_type))
        }
        other => bail!(
            "{} ({other:?}) is not an embedding checkpoint; load it with the generation \
             loader instead",
            other.display_name()
        ),
    }
}

/// Resolve a path that may point at a file inside the checkpoint directory.
fn resolve_model_dir(model_path: &Path) -> PathBuf {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    } else {
        model_path.to_path_buf()
    }
}

/// Load an embedding checkpoint with default options.
pub fn load_embedding_model(model_path: &Path) -> Result<LoadedEmbeddingModel> {
    load_embedding_model_with_options(model_path, EmbeddingLoadOptions::default())
}

/// Load an embedding checkpoint: detect the family, build its model, then
/// attach the shared tokenizer, limits, pad id and vocabulary size.
pub fn load_embedding_model_with_options(
    model_path: &Path,
    options: EmbeddingLoadOptions,
) -> Result<LoadedEmbeddingModel> {
    let model_dir = resolve_model_dir(model_path);
    let model_type = get_model_type(&model_dir)?;
    if !is_embedding_model_type(model_type) {
        bail!(
            "{} at {} is not an embedding checkpoint; /v1/embeddings needs an encoder or a \
             bidirectional / last-token embedding export (a `1_Pooling/config.json`, a \
             `modules.json` Pooling entry, or an embedding architecture in config.json)",
            model_type.display_name(),
            model_dir.display()
        );
    }
    let config = read_embedding_config(&model_dir)?;
    let model = build_family_model(model_type, &model_dir, &config)?;
    finish_loaded_model(model, &model_dir, &config, model_type, options)
}

/// Attach the shared tokenizer, limits, pad id and vocabulary size to a
/// constructed family model.
///
/// Used by: [`load_embedding_model_with_options`] and the test-only stub
/// loader, so the stub exercises exactly the production tail.
pub(crate) fn finish_loaded_model(
    model: Box<dyn EmbeddingModel>,
    model_dir: &Path,
    config: &Value,
    model_type: ModelType,
    options: EmbeddingLoadOptions,
) -> Result<LoadedEmbeddingModel> {
    let tokenizer = strip_padding_and_truncation(load_tokenizer(model_dir)?);
    let mut limits = EmbeddingLimits::derive(
        model_dir,
        model.as_ref(),
        is_absolute_position_family(model_type),
        options.max_length,
    );
    if let Some(fixed) = model.pad_to_max_length() {
        limits.max_length = limits.max_length.min(fixed);
    }
    if let Some(cap) = model.max_sequence_length() {
        limits.max_length = limits.max_length.min(cap);
    }
    let pad_token_id = resolve_pad_token_id(model_dir, &tokenizer);
    let vocab_size = resolve_vocab_size(config, &tokenizer);
    tracing::info!(
        target: "mlxcel::embeddings",
        model_type = ?model_type,
        dim = limits.dim,
        max_length = limits.max_length,
        multi_vector = limits.multi_vector,
        pad_token_id,
        vocab_size,
        "embedding model loaded"
    );
    Ok(LoadedEmbeddingModel {
        model,
        tokenizer,
        limits,
        pad_token_id,
        vocab_size,
        model_type,
    })
}
