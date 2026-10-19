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

//! Per-token output models: token classification and span extraction.
//!
//! Two output shapes, two traits, because they differ in what a request
//! carries and in what comes back:
//!
//! - **Token classification** ([`TokenClassifierModel`], run by
//!   [`TokenClassifier`]): a fixed BIO / BIOES tag set, per-token scores,
//!   tagged character spans and a redacted copy of the text. `mlxcel tag`.
//!   BERT / XLM-RoBERTa `*ForTokenClassification` checkpoints (Rampart)
//!   implement it here; OpenAI Privacy Filter adds a Viterbi decoder.
//! - **Span extraction** ([`SpanExtractorModel`]): labels chosen per request,
//!   label-keyed span lists, plus zero-shot text classification.
//!   `mlxcel extract`. GLiNER2.5 implements it.
//!
//! One trait with optional arguments was rejected: it would let a caller pass
//! labels to a fixed-label model and silently ignore them.
//!
//! The pipeline mirrors `mlx_vlm/token_classification.py`: tokenize without
//! special tokens (char offsets required), split the ids into windows of
//! `context_size - prefix - suffix`, wrap each window in the special tokens,
//! score it, log-softmax in f32, strip the special positions, concatenate,
//! decode a label path, then build spans and the redaction.

pub mod bert;
pub mod extract;
pub mod loader;
pub mod spans;

use anyhow::{Result, bail};
use tokenizers::Tokenizer;

use crate::tokenizer::MlxcelTokenizer;

pub use extract::{ClassifyTask, EntityItem, EntityResult, ExtractOptions, SpanExtractorModel};
pub use loader::{
    LoadedSpanExtractor, LoadedTokenClassifier, TokenModelKind, TokenOutputShape,
    detect_token_model_kind, load_span_extractor, load_token_classifier,
};
pub use spans::{LabelInfo, TokenClassificationResult, TokenSpan, build_label_info};

/// How a label path is read off the per-token scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeMode {
    /// Independent per-token argmax.
    Argmax,
    /// A constrained Viterbi pass over the tag transitions.
    Viterbi,
}

impl DecodeMode {
    /// The spelling upstream uses (`"argmax"` / `"viterbi"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Argmax => "argmax",
            Self::Viterbi => "viterbi",
        }
    }
}

/// A fixed-label token classifier the [`TokenClassifier`] pipeline drives.
///
/// Contract, which every implementation relies on:
///
/// - [`Self::log_probs`] receives one already wrapped window
///   (`prefix + tokens + suffix`, the ids from [`Self::special_tokens`]) of at
///   most [`Self::context_size`] ids and returns exactly one row per id, each
///   with `labels().len()` scores. The pipeline strips the prefix and suffix
///   rows and renormalizes every row with a log-softmax in f32, so a model
///   may return log-probabilities or raw logits.
/// - `decode_modes()[0]` is the default decode mode; a request for a mode not
///   in the list is refused before scoring.
/// - [`Self::decode_path`] returns one output index per token of the
///   concatenated, stripped score matrix.
pub trait TokenClassifierModel {
    /// The tag of every output, in output order (`O`, `B-x`, `I-x`, ...).
    fn labels(&self) -> &[String];
    /// Ids per scored window, special tokens included.
    fn context_size(&self) -> usize;
    /// `(prefix, suffix)` ids wrapped around every window.
    fn special_tokens(&self) -> (Vec<u32>, Vec<u32>);
    /// Whether sub-tokens are grouped into words (by the tokenizer's word
    /// ids) before spans are built.
    fn group_words(&self) -> bool;
    /// Supported decode modes; the first is the default.
    fn decode_modes(&self) -> &[DecodeMode];
    /// `[window_ids.len()][labels]` scores for one wrapped window.
    fn log_probs(&mut self, window_ids: &[u32]) -> Result<Vec<Vec<f32>>>;
    /// Read a label path off `[tokens][labels]` log-probabilities. The
    /// default is per-token argmax whatever the mode.
    fn decode_path(&self, scores: &[Vec<f32>], mode: DecodeMode) -> Vec<usize> {
        let _ = mode;
        argmax_path(scores)
    }
}

impl<T: TokenClassifierModel + ?Sized> TokenClassifierModel for Box<T> {
    fn labels(&self) -> &[String] {
        (**self).labels()
    }
    fn context_size(&self) -> usize {
        (**self).context_size()
    }
    fn special_tokens(&self) -> (Vec<u32>, Vec<u32>) {
        (**self).special_tokens()
    }
    fn group_words(&self) -> bool {
        (**self).group_words()
    }
    fn decode_modes(&self) -> &[DecodeMode] {
        (**self).decode_modes()
    }
    fn log_probs(&mut self, window_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        (**self).log_probs(window_ids)
    }
    fn decode_path(&self, scores: &[Vec<f32>], mode: DecodeMode) -> Vec<usize> {
        (**self).decode_path(scores, mode)
    }
}

/// Per-token argmax; the first maximum wins, as with `numpy.argmax`.
pub fn argmax_path(scores: &[Vec<f32>]) -> Vec<usize> {
    scores
        .iter()
        .map(|row| {
            let mut best: Option<(usize, f32)> = None;
            for (index, &value) in row.iter().enumerate() {
                if best.is_none_or(|(_, top)| value > top) {
                    best = Some((index, value));
                }
            }
            best.map_or(0, |(index, _)| index)
        })
        .collect()
}

/// `row - logsumexp(row)` in f32.
pub fn log_softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    let log_sum = max + sum.ln();
    for value in row.iter_mut() {
        *value -= log_sum;
    }
}

/// Per-request options of [`TokenClassifier::classify`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagOptions {
    /// Decode mode; `None` takes the model's default (`decode_modes()[0]`).
    pub decode: Option<DecodeMode>,
    /// Text every redacted span becomes; `None` uses the span placeholder.
    pub replacement: Option<String>,
    /// Entity labels reported but left unredacted.
    pub keep_labels: Vec<String>,
    /// Trim whitespace off both ends of every span (upstream default `true`).
    pub trim_whitespace: bool,
}

impl Default for TagOptions {
    fn default() -> Self {
        Self {
            decode: None,
            replacement: None,
            keep_labels: Vec::new(),
            trim_whitespace: true,
        }
    }
}

/// Token ids of one text with their char offsets and (optionally) word ids.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenizedText {
    pub ids: Vec<u32>,
    /// Unicode scalar `(start, end)` of every token.
    pub offsets: Vec<(usize, usize)>,
    /// Word index of every token; `None` unless the model groups words.
    pub word_ids: Option<Vec<Option<u32>>>,
}

/// Tokenize, window, score, decode, and redact with a [`TokenClassifierModel`].
pub struct TokenClassifier<M> {
    model: M,
    tokenizer: Tokenizer,
    label_info: LabelInfo,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    window: usize,
}

impl<M: TokenClassifierModel> TokenClassifier<M> {
    /// Wrap a model and its tokenizer.
    ///
    /// Fails on an invalid tag set, on a context that leaves no room for a
    /// token, and on a tokenizer that cannot report character offsets
    /// (anything but a `tokenizer.json` tokenizer), all before any inference.
    pub fn new(model: M, tokenizer: MlxcelTokenizer) -> Result<Self> {
        let MlxcelTokenizer::HuggingFace(mut tokenizer) = tokenizer else {
            bail!("Token classification requires a tokenizer with offsets");
        };
        // Upstream tokenizes with `truncation=False` and no padding; whatever
        // `tokenizer.json` baked in would otherwise cut long text silently.
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);

        let label_info = build_label_info(model.labels())?;
        if model.decode_modes().is_empty() {
            bail!("a token classifier must support at least one decode mode");
        }
        let (prefix, suffix) = model.special_tokens();
        let window = model
            .context_size()
            .checked_sub(prefix.len() + suffix.len())
            .filter(|&w| w > 0);
        let Some(window) = window else {
            bail!("context_size must leave room for at least one token");
        };
        Ok(Self {
            model,
            tokenizer,
            label_info,
            prefix,
            suffix,
            window,
        })
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn label_info(&self) -> &LabelInfo {
        &self.label_info
    }

    /// Real tokens per window (`context_size - prefix - suffix`).
    pub fn window(&self) -> usize {
        self.window
    }

    /// Tokenize without special tokens, with char offsets and, when the model
    /// groups words, word ids.
    pub fn tokenize(&self, text: &str) -> Result<TokenizedText> {
        let encoding = self
            .tokenizer
            .encode_char_offsets(text, false)
            .map_err(|e| anyhow::anyhow!("token classification tokenization failed: {e}"))?;
        Ok(TokenizedText {
            ids: encoding.get_ids().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
            word_ids: self
                .model
                .group_words()
                .then(|| encoding.get_word_ids().to_vec()),
        })
    }

    /// Per-token log-probabilities `[tokens][labels]`, scored window by window.
    pub fn scores(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let num_labels = self.label_info.labels.len();
        let (head, tail) = (self.prefix.len(), self.suffix.len());
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(self.window) {
            let mut window = Vec::with_capacity(head + chunk.len() + tail);
            window.extend_from_slice(&self.prefix);
            window.extend_from_slice(chunk);
            window.extend_from_slice(&self.suffix);
            let rows = self.model.log_probs(&window)?;
            if rows.len() != window.len() {
                bail!(
                    "token classifier returned {} score rows for a window of {} ids",
                    rows.len(),
                    window.len()
                );
            }
            for mut row in rows.into_iter().skip(head).take(chunk.len()) {
                if row.len() != num_labels {
                    bail!(
                        "token classifier returned {} scores per token for {num_labels} labels",
                        row.len()
                    );
                }
                log_softmax_row(&mut row);
                out.push(row);
            }
        }
        Ok(out)
    }

    /// Resolve the requested decode mode against the model's list.
    pub fn resolve_decode(&self, requested: Option<DecodeMode>) -> Result<DecodeMode> {
        let modes = self.model.decode_modes();
        let Some(&default) = modes.first() else {
            bail!("a token classifier must support at least one decode mode");
        };
        let mode = requested.unwrap_or(default);
        if !modes.contains(&mode) {
            let names: Vec<String> = modes.iter().map(|m| format!("'{}'", m.as_str())).collect();
            bail!("decode must be {}", names.join(" or "));
        }
        Ok(mode)
    }

    /// Tag `text`, returning its spans and redacted copy.
    pub fn classify(&mut self, text: &str, options: &TagOptions) -> Result<TokenClassificationResult> {
        let mode = self.resolve_decode(options.decode)?;
        let tokens = self.tokenize(text)?;
        if tokens.ids.is_empty() {
            return Ok(TokenClassificationResult {
                text: text.to_string(),
                spans: Vec::new(),
                redacted_text: text.to_string(),
            });
        }
        let scores = self.scores(&tokens.ids)?;
        let path = self.model.decode_path(&scores, mode);
        if path.len() != tokens.ids.len() {
            bail!(
                "token classifier decoded {} labels for {} tokens",
                path.len(),
                tokens.ids.len()
            );
        }
        let spans = spans::decode_spans(
            text,
            &path,
            &tokens.offsets,
            &self.label_info,
            tokens.word_ids.as_deref(),
            options.trim_whitespace,
        )?;
        let redacted_text = spans::redact(
            text,
            &spans,
            options.replacement.as_deref(),
            &options.keep_labels,
        );
        Ok(TokenClassificationResult {
            text: text.to_string(),
            spans,
            redacted_text,
        })
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;
