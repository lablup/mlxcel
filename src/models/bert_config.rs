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

//! Config resolution for the BERT / XLM-RoBERTa encoder.
//!
//! Split out of [`super::bert`] so the encoder module stays inside the
//! 500-line file guideline. Both types are re-exported there, so callers keep
//! using `crate::models::bert::{BertArgs, BertVariant}`.

use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::Value;

use super::ModelType;

/// Which of the two encoder dialects a checkpoint speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BertVariant {
    /// `model_type: bert`. Position ids are `0..L`, segment ids are real.
    Bert,
    /// `model_type: xlm-roberta`. Position ids skip the padding index, and
    /// the single-row token-type table is always indexed at 0.
    XlmRoberta,
}

impl BertVariant {
    /// Map the detected family to its dialect.
    pub fn from_model_type(model_type: ModelType) -> Option<Self> {
        match model_type {
            ModelType::Bert => Some(Self::Bert),
            ModelType::XlmRoberta => Some(Self::XlmRoberta),
            _ => None,
        }
    }

    /// Dialect implied by `config.json` `model_type`, for the classification
    /// head, which is reached directly rather than through
    /// [`super::get_model_type`] (a `ForSequenceClassification` checkpoint is
    /// a reranker and never detects as an embedding variant).
    pub fn from_config(config: &Value) -> Option<Self> {
        match config
            .get("model_type")
            .and_then(Value::as_str)?
            .to_ascii_lowercase()
            .as_str()
        {
            "bert" => Some(Self::Bert),
            "xlm-roberta" | "xlm_roberta" => Some(Self::XlmRoberta),
            _ => None,
        }
    }

    /// Weight-key prefix a task-head checkpoint puts the encoder under
    /// (`bert.encoder.layer.0...` / `roberta.encoder.layer.0...`).
    pub fn weight_prefix(&self) -> &'static str {
        match self {
            Self::Bert => "bert.",
            Self::XlmRoberta => "roberta.",
        }
    }

    fn default_type_vocab_size(&self) -> usize {
        match self {
            Self::Bert => 2,
            Self::XlmRoberta => 1,
        }
    }

    fn default_layer_norm_eps(&self) -> f32 {
        match self {
            Self::Bert => 1e-12,
            Self::XlmRoberta => 1e-5,
        }
    }

    fn default_pad_token_id(&self) -> i32 {
        match self {
            Self::Bert => 0,
            Self::XlmRoberta => 1,
        }
    }
}

/// Fields read straight off `config.json`. Everything whose default depends
/// on the variant stays optional here and is resolved in
/// [`BertArgs::from_config`].
#[derive(Debug, Clone, Deserialize)]
struct RawBertArgs {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    type_vocab_size: Option<usize>,
    #[serde(default)]
    layer_norm_eps: Option<f32>,
    #[serde(default)]
    pad_token_id: Option<i32>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    num_labels: Option<usize>,
    #[serde(default)]
    id2label: Option<serde_json::Map<String, Value>>,
}

/// Resolved encoder geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct BertArgs {
    pub variant: BertVariant,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    /// Rows of the position table. XLM-RoBERTa stores `max_len + pad + 1`.
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    pub pad_token_id: i32,
    pub hidden_act: String,
    /// Width of the classification head; `1` when the config declares
    /// neither `num_labels` nor `id2label`.
    pub num_labels: usize,
}

impl BertArgs {
    /// Read the encoder geometry, filling the variant-dependent defaults.
    pub fn from_config(config: &Value, variant: BertVariant) -> Result<Self> {
        let raw: RawBertArgs = serde_json::from_value(config.clone())?;
        if raw.hidden_size == 0 || raw.num_attention_heads == 0 {
            bail!("bert config: hidden_size and num_attention_heads must be non-zero");
        }
        if !raw.hidden_size.is_multiple_of(raw.num_attention_heads) {
            bail!(
                "bert config: hidden_size {} is not divisible by num_attention_heads {}",
                raw.hidden_size,
                raw.num_attention_heads
            );
        }
        let num_labels = raw
            .num_labels
            .or_else(|| raw.id2label.as_ref().map(serde_json::Map::len))
            .filter(|&n| n > 0)
            .unwrap_or(1);
        Ok(Self {
            variant,
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            intermediate_size: raw.intermediate_size,
            max_position_embeddings: raw.max_position_embeddings,
            type_vocab_size: raw
                .type_vocab_size
                .unwrap_or_else(|| variant.default_type_vocab_size()),
            layer_norm_eps: raw
                .layer_norm_eps
                .unwrap_or_else(|| variant.default_layer_norm_eps()),
            pad_token_id: raw
                .pad_token_id
                .unwrap_or_else(|| variant.default_pad_token_id()),
            hidden_act: raw.hidden_act.unwrap_or_else(|| "gelu".to_string()),
            num_labels,
        })
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Real tokens the absolute position table can address.
    ///
    /// BERT indexes the table at `0..L`. XLM-RoBERTa starts at
    /// `pad_token_id + 1`, so the same table holds `pad_token_id + 1` fewer
    /// real tokens: `bge-m3`'s 8194 rows cap at 8192 tokens.
    pub fn max_sequence_length(&self) -> usize {
        match self.variant {
            BertVariant::Bert => self.max_position_embeddings,
            BertVariant::XlmRoberta => self
                .max_position_embeddings
                .saturating_sub(self.pad_token_id.max(0) as usize + 1),
        }
    }
}
