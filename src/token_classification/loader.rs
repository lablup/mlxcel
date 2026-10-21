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

//! Detection and loading for `mlxcel tag` / `mlxcel extract`.
//!
//! Detection reads `config.json` alone. Adding a family is one
//! [`TokenModelKind`] variant, one arm in each of its `match`es, one detector
//! function in [`DETECTORS`], and one arm in [`load_token_classifier_with`]
//! or [`load_span_extractor`].
//!
//! This detection is independent of [`crate::models::get_model_type`]: the
//! generation, `/v1/embeddings` and `/v1/rerank` routing is unchanged.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::embeddings::loader::read_embedding_config;

use super::bert::BertTokenTagger;
use super::gliner2_5::{self, Gliner2_5Model};
use super::gliner2_5_prepare::WordSplitter;
use super::privacy_filter::{DEFAULT_OPERATING_POINT, PrivacyFilterTagger};
use super::{SpanExtractorModel, TokenClassifier, TokenClassifierModel};

/// The two per-token output shapes, one CLI subcommand each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenOutputShape {
    /// Fixed tag set, tagged spans plus redaction: `mlxcel tag`.
    TokenClassification,
    /// Labels per request: `mlxcel extract`.
    SpanExtraction,
}

impl TokenOutputShape {
    /// The subcommand that serves this shape.
    pub fn subcommand(self) -> &'static str {
        match self {
            Self::TokenClassification => "mlxcel tag",
            Self::SpanExtraction => "mlxcel extract",
        }
    }

    fn noun(self) -> &'static str {
        match self {
            Self::TokenClassification => "token-classification",
            Self::SpanExtraction => "span-extraction",
        }
    }
}

/// A per-token family mlxcel can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TokenModelKind {
    /// `BertForTokenClassification` / `XLMRobertaForTokenClassification`.
    BertTokenClassifier,
    /// OpenAI Privacy Filter (`model_type` `openai_privacy_filter`), a BIOES
    /// PII tagger decoded with a calibrated Viterbi pass.
    PrivacyFilter,
    /// GLiNER2.5 (`model_type` `gliner2` / `gliner2_5`), a DeBERTa-v3 span
    /// extractor with request-time labels.
    Gliner2_5,
}

impl TokenModelKind {
    /// Every kind, for listing what a subcommand supports.
    pub const ALL: &'static [Self] = &[
        Self::BertTokenClassifier,
        Self::PrivacyFilter,
        Self::Gliner2_5,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::BertTokenClassifier => "bert_token_classifier",
            Self::PrivacyFilter => "openai_privacy_filter",
            Self::Gliner2_5 => "gliner2_5",
        }
    }

    /// Human-readable family, for error messages and docs.
    pub fn description(self) -> &'static str {
        match self {
            Self::BertTokenClassifier => {
                "BERT / XLM-RoBERTa ForTokenClassification (architectures[0] ending in \
                 ForTokenClassification, model_type bert / xlm-roberta)"
            }
            Self::PrivacyFilter => {
                "OpenAI Privacy Filter PII tagger (model_type openai_privacy_filter, BIOES labels \
                 decoded with a calibrated Viterbi pass)"
            }
            Self::Gliner2_5 => {
                "GLiNER2.5 boundary span extractor (model_type gliner2 / gliner2_5 or architectures \
                 Gliner2ForSchemaExtraction / BoundaryExtractor, architecture boundary)"
            }
        }
    }

    pub fn output_shape(self) -> TokenOutputShape {
        match self {
            Self::BertTokenClassifier | Self::PrivacyFilter => {
                TokenOutputShape::TokenClassification
            }
            Self::Gliner2_5 => TokenOutputShape::SpanExtraction,
        }
    }
}

/// `model_type` values of the BERT trunk in `src/models/bert.rs`.
const BERT_TOKEN_MODEL_TYPES: &[&str] = &["bert", "xlm-roberta", "xlm_roberta"];

fn model_type(config: &Value) -> Option<String> {
    config
        .get("model_type")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase)
}

fn first_architecture(config: &Value) -> Option<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|archs| archs.first())
        .and_then(Value::as_str)
}

fn detect_bert_token_classifier(config: &Value) -> Result<Option<TokenModelKind>> {
    let is_bert = model_type(config).is_some_and(|t| BERT_TOKEN_MODEL_TYPES.contains(&t.as_str()));
    let is_token_head =
        first_architecture(config).is_some_and(|a| a.ends_with("ForTokenClassification"));
    if !(is_bert && is_token_head) {
        return Ok(None);
    }
    id2label_labels(config)?;
    Ok(Some(TokenModelKind::BertTokenClassifier))
}

fn detect_privacy_filter(config: &Value) -> Result<Option<TokenModelKind>> {
    if model_type(config).as_deref() != Some("openai_privacy_filter") {
        return Ok(None);
    }
    id2label_labels(config)?;
    Ok(Some(TokenModelKind::PrivacyFilter))
}

fn detect_gliner2_5(config: &Value) -> Result<Option<TokenModelKind>> {
    if !gliner2_5::is_gliner2_config(config) {
        return Ok(None);
    }
    gliner2_5::check_boundary_architecture(config)?;
    Ok(Some(TokenModelKind::Gliner2_5))
}

/// One detector per family; the first that claims the config wins.
type Detector = fn(&Value) -> Result<Option<TokenModelKind>>;
const DETECTORS: &[Detector] = &[
    detect_bert_token_classifier,
    detect_privacy_filter,
    detect_gliner2_5,
];

/// Recognize a token-classification or span-extraction checkpoint from its
/// `config.json`.
pub fn detect_token_model_kind(config: &Value) -> Result<TokenModelKind> {
    for detector in DETECTORS {
        if let Some(kind) = detector(config)? {
            return Ok(kind);
        }
    }
    bail!(
        "not a token-classification or span-extraction checkpoint (model_type `{}`, \
         architectures[0] `{}`); supported families: {}",
        model_type(config).unwrap_or_else(|| "<missing>".to_string()),
        first_architecture(config).unwrap_or("<missing>"),
        TokenModelKind::ALL
            .iter()
            .map(|kind| kind.description())
            .collect::<Vec<_>>()
            .join("; ")
    )
}

/// The ordered tag set from `config.json` `id2label`.
///
/// Keys must be the indices `0..n` exactly once each and every value a
/// string; the result is in output order.
pub fn id2label_labels(config: &Value) -> Result<Vec<String>> {
    let Some(map) = config
        .get("id2label")
        .and_then(Value::as_object)
        .filter(|map| !map.is_empty())
    else {
        bail!("Token-classification checkpoints must define id2label");
    };
    let mut labels: Vec<Option<String>> = vec![None; map.len()];
    for (key, value) in map {
        let index: usize = key
            .trim()
            .parse()
            .with_context(|| format!("id2label key '{key}' is not a label index"))?;
        let Some(slot) = labels.get_mut(index) else {
            bail!(
                "id2label key {index} is out of range for {} labels",
                map.len()
            );
        };
        let Some(label) = value.as_str() else {
            bail!("id2label[{index}] must be a string, got {value}");
        };
        if slot.replace(label.to_string()).is_some() {
            bail!("id2label maps index {index} twice");
        }
    }
    labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| label.with_context(|| format!("id2label has no entry {index}")))
        .collect()
}

/// The error for a checkpoint handed to the other shape's subcommand.
pub fn wrong_subcommand_error(
    model_dir: &Path,
    kind: TokenModelKind,
    requested: TokenOutputShape,
) -> anyhow::Error {
    let actual = kind.output_shape();
    anyhow::anyhow!(
        "{} is a {} checkpoint ({}), not a {} one; use `{}` instead of `{}`",
        model_dir.display(),
        actual.noun(),
        kind.as_str(),
        requested.noun(),
        actual.subcommand(),
        requested.subcommand()
    )
}

fn resolve_model_dir(model_path: &Path) -> PathBuf {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    } else {
        model_path.to_path_buf()
    }
}

/// Read the config and detect the kind, insisting on `shape`.
fn detect_for(
    model_path: &Path,
    shape: TokenOutputShape,
) -> Result<(PathBuf, Value, TokenModelKind)> {
    let model_dir = resolve_model_dir(model_path);
    let config = read_embedding_config(&model_dir)?;
    let kind = detect_token_model_kind(&config).with_context(|| {
        format!(
            "{} cannot be loaded by `{}`",
            model_dir.display(),
            shape.subcommand()
        )
    })?;
    if kind.output_shape() != shape {
        return Err(wrong_subcommand_error(&model_dir, kind, shape));
    }
    Ok((model_dir, config, kind))
}

/// A loaded shape-1 model with its pipeline.
pub struct LoadedTokenClassifier {
    pub classifier: TokenClassifier<Box<dyn TokenClassifierModel>>,
    pub kind: TokenModelKind,
    pub model_type: String,
}

/// A loaded shape-2 model.
pub struct LoadedSpanExtractor {
    pub extractor: Box<dyn SpanExtractorModel>,
    pub kind: TokenModelKind,
    pub model_type: String,
}

/// Load-time choices of `mlxcel tag` that are not per-request options.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenClassifierLoadOptions {
    /// Calibration operating point of a Viterbi-decoded checkpoint
    /// (`--operating-point`); `None` is the checkpoint's `default`. Refused
    /// before any weight loads for a family without a Viterbi decoder.
    pub operating_point: Option<String>,
}

/// Load a token-classification checkpoint for `mlxcel tag` with the default
/// load options.
pub fn load_token_classifier(model_path: &Path) -> Result<LoadedTokenClassifier> {
    load_token_classifier_with(model_path, &TokenClassifierLoadOptions::default())
}

/// Load a token-classification checkpoint for `mlxcel tag`.
pub fn load_token_classifier_with(
    model_path: &Path,
    options: &TokenClassifierLoadOptions,
) -> Result<LoadedTokenClassifier> {
    let (model_dir, config, kind) = detect_for(model_path, TokenOutputShape::TokenClassification)?;
    if let Some(point) = options.operating_point.as_deref()
        && kind != TokenModelKind::PrivacyFilter
    {
        bail!(
            "--operating-point '{point}' applies only to Viterbi-decoded checkpoints \
             (openai_privacy_filter); {} is {} and decodes with argmax",
            model_dir.display(),
            kind.as_str()
        );
    }
    let (model, tokenizer): (Box<dyn TokenClassifierModel>, _) = match kind {
        TokenModelKind::BertTokenClassifier => {
            let (model, tokenizer) = BertTokenTagger::load(&model_dir, &config)?;
            (Box::new(model), tokenizer)
        }
        TokenModelKind::PrivacyFilter => {
            let point = options
                .operating_point
                .as_deref()
                .unwrap_or(DEFAULT_OPERATING_POINT);
            let (model, tokenizer) = PrivacyFilterTagger::load(&model_dir, &config, point)?;
            (Box::new(model), tokenizer)
        }
        TokenModelKind::Gliner2_5 => {
            return Err(wrong_subcommand_error(
                &model_dir,
                kind,
                TokenOutputShape::TokenClassification,
            ));
        }
    };
    let classifier = TokenClassifier::new(model, tokenizer)
        .with_context(|| format!("failed to build the tagger for {}", model_dir.display()))?;
    Ok(LoadedTokenClassifier {
        classifier,
        kind,
        model_type: model_type(&config).unwrap_or_else(|| "unknown".to_string()),
    })
}

/// Load a span-extraction checkpoint for `mlxcel extract`.
pub fn load_span_extractor(model_path: &Path) -> Result<LoadedSpanExtractor> {
    load_span_extractor_with(model_path, &SpanExtractorLoadOptions::default())
}

/// Load-time choices of `mlxcel extract` that are not per-request options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpanExtractorLoadOptions {
    /// Override of the GLiNER2.5 `max_len` (`--max-len`); `None` keeps the
    /// checkpoint's value. Outside `1..=MAX_SUPPORTED_MAX_LEN` the load fails
    /// with the config error text.
    pub max_len: Option<usize>,
    /// How GLiNER2.5 cuts the text into words (`--word-splitter`); `None`
    /// keeps the checkpoint's default (`whitespace`).
    pub word_splitter: Option<WordSplitter>,
}

/// Load a span-extraction checkpoint for `mlxcel extract` with load-time
/// GLiNER2.5 choices from the CLI flags.
pub fn load_span_extractor_with(
    model_path: &Path,
    options: &SpanExtractorLoadOptions,
) -> Result<LoadedSpanExtractor> {
    let (model_dir, config, kind) = detect_for(model_path, TokenOutputShape::SpanExtraction)?;
    let model_type = model_type(&config).unwrap_or_else(|| "unknown".to_string());
    // `detect_for` already refused every token-classification kind, so their
    // arms only restate that refusal.
    match kind {
        TokenModelKind::Gliner2_5 => {
            let mut model = Gliner2_5Model::load(&model_dir, &config)?;
            if let Some(max_len) = options.max_len {
                model.set_max_len(max_len)?;
            }
            if let Some(splitter) = options.word_splitter {
                model.set_word_splitter(splitter);
            }
            Ok(LoadedSpanExtractor {
                extractor: Box::new(model),
                kind,
                model_type,
            })
        }
        TokenModelKind::BertTokenClassifier | TokenModelKind::PrivacyFilter => Err(
            wrong_subcommand_error(&model_dir, kind, TokenOutputShape::SpanExtraction),
        ),
    }
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod loader_tests;
