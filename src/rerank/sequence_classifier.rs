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

//! The one-label cross-encoder path: BERT, XLM-RoBERTa and ModernBERT
//! `ForSequenceClassification` checkpoints.
//!
//! The heads themselves already exist
//! ([`BertSequenceClassifier`] from #1321, [`ModernBertSequenceClassifier`]
//! from #1332); this module only supplies what `/v1/rerank` adds on top: pair
//! tokenization, the length limit, micro-batching, and
//! `score = sigmoid(logit)`.
//!
//! Pairs are encoded through the checkpoint's own pair template
//! (`[CLS] query [SEP] document [SEP]` for BERT, `<s> query </s></s> document
//! </s>` for XLM-RoBERTa), with the tokenizer's longest-first truncation turned
//! on at `max_length`. Longest-first is what the reference
//! `tokenizer(query, document, truncation=True)` call does: it drops tokens
//! from whichever side is currently longer, and the post-processor's special
//! tokens are reserved before any dropping happens, so a long document never
//! costs the query its `[CLS]`.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;
use tokenizers::{TruncationParams, TruncationStrategy};

use crate::embeddings::limits::{derive_max_length, resolve_pad_token_id};
use crate::embeddings::loader::read_embedding_config;
use crate::embeddings::tokenize::{EncodeOptions, EncodedBatch, encode_pair_row};
use crate::models::bert::BertVariant;
use crate::models::bert_heads::BertSequenceClassifier;
use crate::models::modernbert_heads::ModernBertSequenceClassifier;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::{
    RerankItem, RerankScores, Reranker, RerankerKind, require_single_label, sigmoid_to_vec,
};

/// The two classifier trunks behind one scoring path.
enum ClassifierBackbone {
    /// `BertForSequenceClassification` / `XLMRobertaForSequenceClassification`.
    Bert(Box<BertSequenceClassifier>),
    /// `ModernBertForSequenceClassification`.
    ModernBert(Box<ModernBertSequenceClassifier>),
}

impl ClassifierBackbone {
    fn num_labels(&self) -> usize {
        match self {
            Self::Bert(model) => model.num_labels(),
            Self::ModernBert(model) => model.num_labels(),
        }
    }

    /// BERT cross-encoders put segment `1` on the document half;
    /// XLM-RoBERTa and ModernBERT have no segment table to index.
    fn needs_token_type_ids(&self) -> bool {
        match self {
            Self::Bert(model) => model.needs_token_type_ids(),
            Self::ModernBert(_) => false,
        }
    }

    /// Hard token cap the loaded weights impose, if any. Only the absolute
    /// position table of the BERT trunk has one; ModernBERT is RoPE.
    fn weight_max_length(&self) -> Option<usize> {
        match self {
            Self::Bert(model) => Some(model.max_sequence_length()),
            Self::ModernBert(_) => None,
        }
    }

    fn logits(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
        token_type_ids: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        match self {
            Self::Bert(model) => model.logits(input_ids, attention_mask, token_type_ids),
            Self::ModernBert(model) => model.logits(input_ids, attention_mask),
        }
    }
}

/// A one-label cross-encoder scored with `sigmoid(logit)`.
pub struct SequenceClassifierReranker {
    backbone: ClassifierBackbone,
    tokenizer: MlxcelTokenizer,
    pad_token_id: u32,
    max_length: usize,
    batch_size: usize,
}

/// Turn on the tokenizer's own longest-first truncation at `max_length`.
///
/// The embedding loader strips whatever `tokenizer.json` baked in, so this is
/// the only truncation configured, and putting it on the tokenizer (rather than
/// trimming ids afterwards) is what makes the special-token accounting and the
/// per-side balance match the reference implementation exactly.
fn with_longest_first_truncation(
    tokenizer: MlxcelTokenizer,
    max_length: usize,
) -> Result<MlxcelTokenizer> {
    let MlxcelTokenizer::HuggingFace(mut hf) = tokenizer else {
        bail!(
            "a cross-encoder reranker needs a tokenizer.json (HuggingFace) tokenizer to encode \
             query/document pairs"
        );
    };
    hf.with_truncation(Some(TruncationParams {
        max_length,
        strategy: TruncationStrategy::LongestFirst,
        ..Default::default()
    }))
    .map_err(|e| anyhow!("failed to configure reranker truncation: {e}"))?;
    Ok(MlxcelTokenizer::HuggingFace(hf))
}

impl SequenceClassifierReranker {
    /// Load a `ForSequenceClassification` checkpoint from its directory.
    ///
    /// `max_length_override` is `--rerank-max-length`; without it the cap comes
    /// from `sentence_bert_config.json`, `tokenizer_config.json`,
    /// `config.json` `max_position_embeddings` for the absolute-position BERT
    /// trunk, and the shared 8192 ceiling.
    pub fn load(
        model_dir: &Path,
        batch_size: usize,
        max_length_override: Option<usize>,
    ) -> Result<Self> {
        let config = read_embedding_config(model_dir)?;
        let backbone = build_backbone(model_dir, &config)?;
        require_single_label(backbone.num_labels())?;

        let tokenizer = crate::embeddings::tokenize::strip_padding_and_truncation(
            load_tokenizer(model_dir).with_context(|| {
                format!("failed to load the tokenizer in {}", model_dir.display())
            })?,
        );
        let pad_token_id = resolve_pad_token_id(model_dir, &tokenizer);
        let mut max_length = derive_max_length(
            model_dir,
            matches!(backbone, ClassifierBackbone::Bert(_)),
            max_length_override,
        );
        if let Some(cap) = backbone.weight_max_length() {
            max_length = max_length.min(cap);
        }
        let tokenizer = with_longest_first_truncation(tokenizer, max_length)?;

        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        tracing::info!(
            target: "mlxcel::rerank",
            model_type,
            max_length,
            pad_token_id,
            batch_size,
            "sequence-classifier reranker loaded"
        );
        Ok(Self {
            backbone,
            tokenizer,
            pad_token_id,
            max_length,
            batch_size: batch_size.max(1),
        })
    }

    /// Score one micro-batch of `(query, document)` text pairs.
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<(Vec<f32>, usize)> {
        let opts = EncodeOptions {
            add_special_tokens: true,
            max_length: self.max_length,
            with_token_type_ids: self.backbone.needs_token_type_ids(),
        };
        let rows = pairs
            .iter()
            .map(|(query, document)| encode_pair_row(&self.tokenizer, query, document, opts))
            .collect::<Result<Vec<_>>>()?;
        let batch = EncodedBatch::from_rows(&rows, self.pad_token_id, None);

        let input_ids = batch.input_ids_array();
        let attention_mask = batch.attention_mask_array();
        let token_type_ids = self
            .backbone
            .needs_token_type_ids()
            .then(|| batch.token_type_ids_array())
            .flatten();
        let logits =
            self.backbone
                .logits(&input_ids, &attention_mask, token_type_ids.as_deref())?;
        let shape = mlxcel_core::array_shape(&logits);
        if shape != vec![batch.batch as i32, 1] {
            bail!(
                "cross-encoder produced logits of shape {shape:?}, expected [{}, 1]",
                batch.batch
            );
        }
        Ok((sigmoid_to_vec(&logits)?, batch.total_tokens()))
    }
}

/// Construct the classifier trunk named by `config.json`.
fn build_backbone(model_dir: &Path, config: &Value) -> Result<ClassifierBackbone> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("<missing>")
        .to_ascii_lowercase();
    match model_type.as_str() {
        "modernbert" => Ok(ClassifierBackbone::ModernBert(Box::new(
            ModernBertSequenceClassifier::load(model_dir)?,
        ))),
        _ if BertVariant::from_config(config).is_some() => Ok(ClassifierBackbone::Bert(Box::new(
            BertSequenceClassifier::load(model_dir)?,
        ))),
        other => bail!(
            "{} declares model_type `{other}`, which has no cross-encoder port",
            model_dir.display()
        ),
    }
}

impl Reranker for SequenceClassifierReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::SequenceClassifier
    }

    fn score(
        &self,
        query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
    ) -> Result<RerankScores> {
        if query.has_image() || documents.iter().any(RerankItem::has_image) {
            bail!("a cross-encoder reranker is text-only and does not accept image items");
        }
        let query_text = query.text_or_empty();
        let mut scores = Vec::with_capacity(documents.len());
        let mut prompt_tokens = 0usize;
        for chunk in documents.chunks(self.batch_size) {
            let pairs: Vec<(&str, &str)> = chunk
                .iter()
                .map(|document| (query_text, document.text_or_empty()))
                .collect();
            let (chunk_scores, tokens) = self.score_pairs(&pairs)?;
            prompt_tokens += tokens;
            scores.extend(chunk_scores);
        }
        Ok(RerankScores {
            scores,
            prompt_tokens,
        })
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
#[path = "sequence_classifier_tests.rs"]
mod sequence_classifier_tests;
