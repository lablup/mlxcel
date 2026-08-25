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

//! Per-checkpoint limits: the sequence length cap, the vector width, the
//! pad token, and the vocabulary size the request validator checks against.

use std::path::Path;

use serde_json::Value;

use crate::models::ModelType;
use crate::tokenizer::MlxcelTokenizer;

use super::model::EmbeddingModel;

/// Hard upper bound on the tokens one embedding input may carry, whatever
/// the checkpoint declares.
pub const EMBEDDING_MAX_LENGTH_CAP: usize = 8192;

/// `tokenizer_config.json` `model_max_length` values at or above this are
/// the HuggingFace "unset" sentinel (1e30 style) and are ignored.
const MODEL_MAX_LENGTH_UNSET: u64 = 1_000_000;

/// Static limits of one loaded embedding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingLimits {
    /// Inputs longer than this are truncated from the right (keeping a
    /// trailing special token the tokenizer appends).
    pub max_length: usize,
    /// Width `D` of one vector.
    pub dim: usize,
    /// Whether the model emits `[num_real_tokens, D]` per input.
    pub multi_vector: bool,
}

impl EmbeddingLimits {
    /// Derive the limits for a loaded model. `is_absolute_position` marks
    /// families whose `max_position_embeddings` is a hard positional table
    /// (BERT, XLM-RoBERTa, SigLIP); `max_length_override` is
    /// `--embedding-max-length`.
    pub fn derive(
        model_dir: &Path,
        model: &dyn EmbeddingModel,
        is_absolute_position: bool,
        max_length_override: Option<usize>,
    ) -> Self {
        Self {
            max_length: derive_max_length(model_dir, is_absolute_position, max_length_override),
            dim: model.embedding_dim(),
            multi_vector: model.multi_vector(),
        }
    }
}

/// Read and parse a JSON file, returning `None` when it is absent or
/// malformed (a limit derivation never fails the load over a side file).
pub fn read_json(path: &Path) -> Option<Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Families whose position table is absolute and therefore caps the input
/// length through `max_position_embeddings`.
pub fn is_absolute_position_family(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::Bert | ModelType::XlmRoberta | ModelType::SiglipText
    )
}

/// `max_length` derivation: the smallest of `sentence_bert_config.json`
/// `max_seq_length`, `tokenizer_config.json` `model_max_length` (when
/// `0 < value < 1_000_000`), `config.json` `max_position_embeddings` for
/// absolute-position encoders, the hard cap, and the operator override.
pub fn derive_max_length(
    model_dir: &Path,
    is_absolute_position: bool,
    max_length_override: Option<usize>,
) -> usize {
    let mut candidates: Vec<usize> = vec![EMBEDDING_MAX_LENGTH_CAP];

    if let Some(v) = read_json(&model_dir.join("sentence_bert_config.json"))
        .and_then(|cfg| cfg.get("max_seq_length")?.as_u64())
        .filter(|&v| v > 0)
    {
        candidates.push(v as usize);
    }

    if let Some(v) = read_json(&model_dir.join("tokenizer_config.json"))
        .and_then(|cfg| cfg.get("model_max_length")?.as_u64())
        .filter(|&v| v > 0 && v < MODEL_MAX_LENGTH_UNSET)
    {
        candidates.push(v as usize);
    }

    if is_absolute_position
        && let Some(v) = read_json(&model_dir.join("config.json"))
            .and_then(|cfg| cfg.get("max_position_embeddings")?.as_u64())
            .filter(|&v| v > 0)
    {
        candidates.push(v as usize);
    }

    if let Some(v) = max_length_override.filter(|&v| v > 0) {
        candidates.push(v);
    }

    candidates
        .into_iter()
        .min()
        .unwrap_or(EMBEDDING_MAX_LENGTH_CAP)
}

/// The token string a `tokenizer_config.json` entry names; accepts both the
/// plain string form and the `{"content": ...}` added-token object form.
fn token_string(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}

/// Resolve the pad token id: `tokenizer_config.json` `pad_token`, falling
/// back to `eos_token`, then `0`.
pub fn resolve_pad_token_id(model_dir: &Path, tokenizer: &MlxcelTokenizer) -> u32 {
    let config = read_json(&model_dir.join("tokenizer_config.json"));
    let lookup = |key: &str| -> Option<u32> {
        let token = token_string(config.as_ref()?.get(key)?)?;
        tokenizer.hf_tokenizer()?.token_to_id(token)
    };
    lookup("pad_token")
        .or_else(|| lookup("eos_token"))
        .unwrap_or(0)
}

/// Vocabulary size the request validator checks token-id inputs against:
/// `config.json` `vocab_size` (top level, then `text_config`), falling back
/// to the tokenizer's own vocabulary.
pub fn resolve_vocab_size(config: &Value, tokenizer: &MlxcelTokenizer) -> usize {
    config
        .get("vocab_size")
        .or_else(|| config.get("text_config")?.get("vocab_size"))
        .and_then(Value::as_u64)
        .filter(|&v| v > 0)
        .map(|v| v as usize)
        .or_else(|| tokenizer.hf_tokenizer().map(|t| t.get_vocab_size(true)))
        .unwrap_or(0)
}

/// `config.json` `normalize` flag (default `true`), for families that let
/// the checkpoint turn L2 normalization off.
pub fn config_normalize_flag(config: &Value) -> bool {
    config
        .get("normalize")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}
