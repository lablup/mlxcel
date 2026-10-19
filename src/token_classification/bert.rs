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

//! [`TokenClassifierModel`] over the BERT / XLM-RoBERTa token head
//! ([`BertTokenClassifier`]), matching mlx-vlm's `load_token_classifier` +
//! `TokenClassifier` defaults: per-token argmax, word grouping on, the
//! tokenizer's CLS / SEP around every window, and `max_position_embeddings`
//! as the context.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

use crate::models::bert::BertVariant;
use crate::models::bert_heads::BertTokenClassifier;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::loader::id2label_labels;
use super::{DecodeMode, TokenClassifierModel};

/// Rampart and every other BIO checkpoint read with per-token argmax only.
const BERT_DECODE_MODES: &[DecodeMode] = &[DecodeMode::Argmax];

/// The BERT / XLM-RoBERTa token head as a pipeline model.
pub struct BertTokenTagger {
    model: BertTokenClassifier,
    labels: Vec<String>,
    special_tokens: (Vec<u32>, Vec<u32>),
}

/// A `cls_token` / `sep_token` entry of `tokenizer_config.json`: a string or
/// an `AddedToken` object with `content`.
fn special_token_text(config: &Value, key: &str) -> Option<String> {
    match config.get(key)? {
        Value::String(text) => Some(text.clone()),
        Value::Object(token) => token.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

/// `([cls], [sep])` when the tokenizer defines both, else `([], [])`, as
/// upstream reads `tokenizer.cls_token_id` / `sep_token_id`. The token text
/// comes from `tokenizer_config.json`, falling back to the family's default
/// (`[CLS]` / `[SEP]` for BERT, `<s>` / `</s>` for XLM-RoBERTa) exactly where
/// a `transformers` fast tokenizer would.
pub fn resolve_special_tokens(
    model_dir: &Path,
    tokenizer: &MlxcelTokenizer,
    variant: BertVariant,
) -> (Vec<u32>, Vec<u32>) {
    let MlxcelTokenizer::HuggingFace(hf) = tokenizer else {
        return (Vec::new(), Vec::new());
    };
    let config: Value = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or(Value::Null);
    let (default_cls, default_sep) = match variant {
        BertVariant::Bert => ("[CLS]", "[SEP]"),
        BertVariant::XlmRoberta => ("<s>", "</s>"),
    };
    let id = |key: &str, default: &str| {
        let text = special_token_text(&config, key).unwrap_or_else(|| default.to_string());
        hf.token_to_id(&text)
    };
    match (id("cls_token", default_cls), id("sep_token", default_sep)) {
        (Some(cls), Some(sep)) => (vec![cls], vec![sep]),
        _ => (Vec::new(), Vec::new()),
    }
}

impl BertTokenTagger {
    /// Load the head, its tag set and its tokenizer.
    ///
    /// Fails when `id2label` does not cover the classifier's output width.
    pub fn load(model_dir: &Path, config: &Value) -> Result<(Self, MlxcelTokenizer)> {
        let labels = id2label_labels(config)?;
        let model = BertTokenClassifier::load(model_dir)?;
        if labels.len() != model.num_labels() {
            bail!(
                "id2label has {} entries but the classifier has {} outputs",
                labels.len(),
                model.num_labels()
            );
        }
        let tokenizer = load_tokenizer(model_dir).with_context(|| {
            format!(
                "failed to load the tokenizer of {}",
                model_dir.display()
            )
        })?;
        let special_tokens = resolve_special_tokens(model_dir, &tokenizer, model.args().variant);
        Ok((
            Self {
                model,
                labels,
                special_tokens,
            },
            tokenizer,
        ))
    }

    /// Build from an already loaded head (tests and callers with their own
    /// weights).
    pub fn from_parts(
        model: BertTokenClassifier,
        labels: Vec<String>,
        special_tokens: (Vec<u32>, Vec<u32>),
    ) -> Result<Self> {
        if labels.len() != model.num_labels() {
            bail!(
                "id2label has {} entries but the classifier has {} outputs",
                labels.len(),
                model.num_labels()
            );
        }
        Ok(Self {
            model,
            labels,
            special_tokens,
        })
    }

    fn window_logits(&self, window_ids: &[u32]) -> Result<UniquePtr<MlxArray>> {
        let length = i32::try_from(window_ids.len()).context("token window is too long")?;
        let ids: Vec<i32> = window_ids
            .iter()
            .map(|&id| i32::try_from(id).context("token id does not fit in int32"))
            .collect::<Result<_>>()?;
        let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, length]);
        let attention_mask = mlxcel_core::ones(&[1, length], dtype::INT32);
        let logits = self.model.logits(&input_ids, &attention_mask, None)?;
        let logits = mlxcel_core::astype(&logits, dtype::FLOAT32);
        Ok(mlxcel_core::log_softmax(&logits, -1))
    }
}

impl TokenClassifierModel for BertTokenTagger {
    fn labels(&self) -> &[String] {
        &self.labels
    }

    fn context_size(&self) -> usize {
        // `max_position_embeddings` for BERT. XLM-RoBERTa's table starts at
        // `pad_token_id + 1`, so it addresses that many fewer positions; using
        // the raw config value there would index past the table.
        self.model.max_sequence_length()
    }

    fn special_tokens(&self) -> (Vec<u32>, Vec<u32>) {
        self.special_tokens.clone()
    }

    fn group_words(&self) -> bool {
        true
    }

    fn decode_modes(&self) -> &[DecodeMode] {
        BERT_DECODE_MODES
    }

    fn log_probs(&mut self, window_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if window_ids.is_empty() {
            return Ok(Vec::new());
        }
        let log_probs = self.window_logits(window_ids)?;
        mlxcel_core::try_eval(&log_probs)
            .map_err(|e| anyhow::anyhow!("token classifier evaluation failed: {e}"))?;
        let flat = array_to_vec_f32(&log_probs);
        let width = self.labels.len();
        if flat.len() != window_ids.len() * width {
            bail!(
                "token classifier produced {} scores for {} tokens x {width} labels",
                flat.len(),
                window_ids.len()
            );
        }
        Ok(flat.chunks(width).map(<[f32]>::to_vec).collect())
    }
}
