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

//! Query/document relevance scoring behind `POST /v1/rerank` and
//! `mlxcel rerank`.
//!
//! Three checkpoint shapes score a pair, and all three end at a probability in
//! `[0, 1]`:
//!
//! - [`RerankerKind::SequenceClassifier`]: a one-label cross-encoder on the
//!   BERT, XLM-RoBERTa or ModernBERT trunk. The pair is tokenized as a single
//!   sequence and the head's logit becomes `sigmoid(logit)`.
//! - [`RerankerKind::GenerativeText`]: the Qwen3 reranker, which is asked to
//!   answer `yes` or `no` and is read as
//!   `sigmoid(logit("yes") - logit("no"))` at the last prompt position.
//! - [`RerankerKind::GenerativeVl`]: the Qwen3-VL reranker, the same yes/no
//!   read on a multimodal prompt whose query and documents may carry images.
//!
//! Layout:
//!
//! - [`loader`]: `load_reranker`, the kind dispatcher.
//! - [`sequence_classifier`]: the cross-encoder path over the merged
//!   `BertSequenceClassifier` / `ModernBertSequenceClassifier` heads.
//! - [`qwen3_generative`]: the Qwen3 yes/no path.
//! - [`qwen3_vl_generative`]: the Qwen3-VL yes/no path.
//!
//! Everything here holds MLX handles and runs on one thread: the server's
//! rerank worker thread or the `mlxcel rerank` main thread.

pub mod loader;
pub mod qwen3_generative;
pub mod qwen3_vl_generative;
pub mod sequence_classifier;

#[cfg(test)]
pub(crate) mod stub;

#[cfg(test)]
#[path = "real_checkpoint_tests.rs"]
mod real_checkpoint_tests;

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, dtype, utils::array_to_vec_f32};
use serde_json::Value;

pub use crate::embeddings::ImageInput;
pub use loader::{LoadedReranker, RerankLoadOptions, load_reranker, load_reranker_with_options};

/// Hard upper bound on the tokens one query/document pair may carry, whatever
/// the checkpoint declares.
pub const RERANK_MAX_LENGTH_CAP: usize = 8192;

/// Default `--rerank-batch-size`: pairs per forward pass for a text reranker.
pub const DEFAULT_RERANK_BATCH_SIZE: usize = 8;

/// Default pairs per forward pass for the multimodal reranker, whose rows each
/// carry a full image's worth of visual tokens.
pub const DEFAULT_RERANK_VL_BATCH_SIZE: usize = 2;

/// Which scoring recipe a loaded checkpoint uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RerankerKind {
    /// One-label cross-encoder; `score = sigmoid(logit)`.
    SequenceClassifier,
    /// Qwen3 yes/no reranker; `score = sigmoid(logit(yes) - logit(no))`.
    GenerativeText,
    /// Qwen3-VL yes/no reranker over text and image documents.
    GenerativeVl,
}

impl RerankerKind {
    /// Stable identifier reported by `/v1/rerank` diagnostics and the CLI.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SequenceClassifier => "sequence_classifier",
            Self::GenerativeText => "generative_text",
            Self::GenerativeVl => "generative_vl",
        }
    }

    /// Whether an `instruction` means anything for this kind.
    ///
    /// A cross-encoder has no place to put one: the pair is the whole input,
    /// so an instruction would silently be ignored and the route rejects it
    /// instead.
    #[must_use]
    pub fn accepts_instruction(self) -> bool {
        !matches!(self, Self::SequenceClassifier)
    }

    /// Default micro-batch width before `--rerank-batch-size` overrides it.
    #[must_use]
    pub fn default_batch_size(self) -> usize {
        match self {
            Self::GenerativeVl => DEFAULT_RERANK_VL_BATCH_SIZE,
            _ => DEFAULT_RERANK_BATCH_SIZE,
        }
    }
}

/// One side of a scored pair: the query, or one document.
///
/// Both fields are optional so an image-only document is expressible; the
/// caller is responsible for rejecting an item that carries neither.
#[derive(Debug, Clone, Default)]
pub struct RerankItem {
    /// Text content, absent for an image-only item.
    pub text: Option<String>,
    /// Decoded image, accepted only by a reranker whose
    /// [`Reranker::supports_images`] is `true`.
    pub image: Option<ImageInput>,
}

impl RerankItem {
    /// A text-only item.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            image: None,
        }
    }

    /// An image-only item.
    #[must_use]
    pub fn image(image: ImageInput) -> Self {
        Self {
            text: None,
            image: Some(image),
        }
    }

    /// Text content, or `""` for an image-only item.
    #[must_use]
    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }

    /// Whether this item carries an image.
    #[must_use]
    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    /// Whether this item carries neither usable text nor an image.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.image.is_none() && self.text.as_deref().unwrap_or("").trim().is_empty()
    }
}

/// Result of one scoring call, in document order.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankScores {
    /// Relevance probability per document, same length and order as the input.
    pub scores: Vec<f32>,
    /// Real (non-padding) tokens across every scored pair.
    pub prompt_tokens: usize,
}

/// A loaded reranker.
///
/// Implementors hold MLX array handles directly and are used from exactly one
/// thread, so the trait requires neither `Send` nor `Sync`.
pub trait Reranker {
    /// Which scoring recipe this checkpoint uses.
    fn kind(&self) -> RerankerKind;

    /// Score every document against `query`, in document order.
    ///
    /// `instruction` is ignored by the cross-encoder path and rendered into
    /// the prompt by the generative ones.
    fn score(
        &self,
        query: &RerankItem,
        documents: &[RerankItem],
        instruction: Option<&str>,
    ) -> Result<RerankScores>;

    /// Whether image items are accepted.
    fn supports_images(&self) -> bool {
        false
    }

    /// Token cap on one scored pair.
    fn max_length(&self) -> usize;

    /// Pairs per forward pass.
    fn batch_size(&self) -> usize;
}

/// Label count a classifier config declares: `num_labels`, else
/// `len(id2label)`, else `1`.
///
/// This is the config's claim. Each classifier head overrules it with the row
/// count of its own projection tensor, which is what actually decides the
/// logit width; the two are compared at load so a mismatch is visible.
#[must_use]
pub fn num_labels(config: &Value) -> usize {
    config
        .get("num_labels")
        .and_then(Value::as_u64)
        .or_else(|| {
            config
                .get("id2label")
                .and_then(Value::as_object)
                .map(|labels| labels.len() as u64)
        })
        .filter(|&n| n > 0)
        .map_or(1, |n| n as usize)
}

/// Message for a checkpoint whose family has no reranker port.
fn unsupported_reranker(model_type: &str, architecture: Option<&str>) -> anyhow::Error {
    let arch = architecture.unwrap_or("<no architectures entry>");
    anyhow::anyhow!(
        "Unsupported reranker model type `{model_type}` (architectures[0] = `{arch}`). \
         /v1/rerank serves one-label BertForSequenceClassification, \
         XLMRobertaForSequenceClassification and ModernBertForSequenceClassification \
         cross-encoders, the Qwen3 generative reranker (model_type `qwen3`) and the \
         Qwen3-VL multimodal reranker (model_type `qwen3_vl`); see docs/embeddings.md"
    )
}

/// Pick the scoring recipe for a reranker checkpoint's `config.json`.
///
/// A `ForSequenceClassification` export is a cross-encoder and is accepted only
/// on the three encoder families with a head port. Everything else is keyed on
/// `model_type`, because a generative reranker's config is indistinguishable
/// from a chat model's: `mlx-community/Qwen3-Reranker-0.6B-4bit` declares
/// `Qwen3ForCausalLM` and `Qwen/Qwen3-VL-Reranker-2B` declares
/// `Qwen3VLForConditionalGeneration`. That is why the generative kinds are only
/// reachable through `--reranker-model`, never through `-m` auto-detection.
pub fn detect_reranker_kind(config: &Value) -> Result<RerankerKind> {
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_str);
    let model_type_raw = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    let model_type = model_type_raw.to_ascii_lowercase();

    if crate::models::is_sequence_classification_architecture(config) {
        return match model_type.as_str() {
            "bert" | "xlm-roberta" | "xlm_roberta" | "modernbert" => {
                Ok(RerankerKind::SequenceClassifier)
            }
            _ => Err(unsupported_reranker(model_type_raw, architecture)),
        };
    }

    match model_type.as_str() {
        "qwen3" => Ok(RerankerKind::GenerativeText),
        "qwen3_vl" => Ok(RerankerKind::GenerativeVl),
        _ => Err(unsupported_reranker(model_type_raw, architecture)),
    }
}

/// Reject a classifier that does not expose exactly one output label.
///
/// A multi-label head produces a distribution over classes rather than one
/// relevance score, so `sigmoid(logit)` would be meaningless. The check runs on
/// the tensor-derived count, not on `config.json`.
pub(crate) fn require_single_label(num_labels: usize) -> Result<()> {
    if num_labels != 1 {
        bail!(
            "reranker sequence classifiers must expose exactly one output label, this \
             checkpoint has {num_labels}"
        );
    }
    Ok(())
}

/// Read a `[B, ...]` logit array back as sigmoid probabilities.
///
/// The array is cast to f32 first so a bf16 or f16 forward pass reads back at
/// full precision, then evaluated on the calling thread.
pub(crate) fn sigmoid_to_vec(logits: &MlxArray) -> Result<Vec<f32>> {
    let scores = mlxcel_core::sigmoid(&mlxcel_core::astype(logits, dtype::FLOAT32));
    mlxcel_core::try_eval(&scores)
        .map_err(|e| anyhow::anyhow!("reranker score evaluation failed: {e}"))?;
    Ok(array_to_vec_f32(&scores))
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;
