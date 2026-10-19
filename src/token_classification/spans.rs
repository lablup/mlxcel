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

//! BIO / BIOES span decoding and redaction.
//!
//! A one-to-one port of the label, span and redaction helpers in
//! `mlx_vlm/token_classification.py` (`build_label_info`, `token_spans`,
//! `group_words`, `placeholder`, `decode_spans`, `redact`). Every character
//! offset here is a Unicode scalar index into the input text, which is what
//! Python string indexing (and the fast tokenizer's `offset_mapping`) uses, so
//! a span decoded here and one decoded upstream compare equal field by field.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, bail};
use serde::Serialize;

/// The position of a token inside its entity in a BIO / BIOES tag set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Boundary {
    /// `B-`: the first token of a multi-token entity.
    Begin,
    /// `I-`: a token inside an entity.
    Inside,
    /// `E-`: the last token of a multi-token entity (BIOES).
    End,
    /// `S-`: a one-token entity (BIOES).
    Single,
}

impl Boundary {
    fn parse(prefix: &str) -> Option<Self> {
        Some(match prefix {
            "B" => Self::Begin,
            "I" => Self::Inside,
            "E" => Self::End,
            "S" => Self::Single,
            _ => return None,
        })
    }
}

/// An indexed tag set: the result of [`build_label_info`].
///
/// Field names follow upstream's `LabelInfo` so a decoder ported from Python
/// (the Privacy Filter's constrained Viterbi, for instance) reads the same
/// tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelInfo {
    /// The tag of every classifier output, in output order.
    pub labels: Vec<String>,
    /// Output index of the background tag `O` (the last one, if repeated).
    pub background: usize,
    /// Entity names in first-seen order; index `0` is always `O`.
    pub span_names: Vec<String>,
    /// Entity name to its index in [`Self::span_names`].
    pub span_index: HashMap<String, usize>,
    /// Output index to the index of its entity in [`Self::span_names`].
    pub token_to_span: Vec<usize>,
    /// Output index to its boundary; `None` for `O`.
    pub boundaries: Vec<Option<Boundary>>,
    /// Entity name to the output index of each of its boundary tags.
    pub states_by_span: HashMap<String, BTreeMap<Boundary, usize>>,
}

/// Index a BIO or BIOES tag set (`O`, `B-x`, `I-x`, `E-x`, `S-x`).
///
/// A label is split at its first `-` (as Python's `str.partition` does), so
/// `B-STREET-NAME` names the entity `STREET-NAME`.
pub fn build_label_info(labels: &[String]) -> Result<LabelInfo> {
    let mut background = None;
    let mut span_names = vec!["O".to_string()];
    let mut span_index = HashMap::from([("O".to_string(), 0usize)]);
    let mut token_to_span = Vec::with_capacity(labels.len());
    let mut boundaries = Vec::with_capacity(labels.len());
    let mut states_by_span: HashMap<String, BTreeMap<Boundary, usize>> = HashMap::new();

    for (index, label) in labels.iter().enumerate() {
        if label == "O" {
            background = Some(index);
            token_to_span.push(0);
            boundaries.push(None);
            continue;
        }
        let parsed = label
            .split_once('-')
            .and_then(|(prefix, name)| Some((Boundary::parse(prefix)?, name)))
            .filter(|(_, name)| !name.is_empty());
        let Some((boundary, name)) = parsed else {
            bail!("Invalid token label '{label}'; expected BIO or BIOES labels");
        };
        let span = match span_index.get(name) {
            Some(&span) => span,
            None => {
                span_names.push(name.to_string());
                span_index.insert(name.to_string(), span_names.len() - 1);
                span_names.len() - 1
            }
        };
        token_to_span.push(span);
        boundaries.push(Some(boundary));
        states_by_span
            .entry(name.to_string())
            .or_default()
            .insert(boundary, index);
    }

    let Some(background) = background else {
        bail!("Token labels must include the background label 'O'");
    };
    Ok(LabelInfo {
        labels: labels.to_vec(),
        background,
        span_names,
        span_index,
        token_to_span,
        boundaries,
        states_by_span,
    })
}

/// One `(span_index, token_start, token_end)` run, `token_end` exclusive.
pub type TokenRun = (usize, usize, usize);

/// Return the entity runs of a label path.
///
/// `I` after a different entity starts a new span, and `E` without a matching
/// open span becomes a one-token span, exactly as upstream builds them. Every
/// entry of `path` must be an output index of `labels`; [`decode_spans`]
/// checks that before calling this.
pub fn token_spans(path: &[usize], labels: &LabelInfo) -> Vec<TokenRun> {
    let mut spans = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    for (token, &tag) in path.iter().enumerate() {
        let span = labels.token_to_span.get(tag).copied().unwrap_or(0);
        let boundary = labels.boundaries.get(tag).copied().flatten();
        if tag == labels.background {
            if let Some((open, start)) = current.take() {
                spans.push((open, start, token));
            }
            continue;
        }
        match boundary {
            Some(Boundary::Single) => {
                if let Some((open, start)) = current.take() {
                    spans.push((open, start, token));
                }
                spans.push((span, token, token + 1));
            }
            Some(Boundary::Begin) => {
                if let Some((open, start)) = current.take() {
                    spans.push((open, start, token));
                }
                current = Some((span, token));
            }
            Some(Boundary::Inside) => {
                if current.map(|(open, _)| open) != Some(span) {
                    if let Some((open, start)) = current.take() {
                        spans.push((open, start, token));
                    }
                    current = Some((span, token));
                }
            }
            Some(Boundary::End) => {
                match current.take() {
                    Some((open, start)) if open == span => spans.push((span, start, token + 1)),
                    Some((open, start)) => {
                        spans.push((open, start, token));
                        spans.push((span, token, token + 1));
                    }
                    None => spans.push((span, token, token + 1)),
                }
            }
            // A repeated `O` that is not the background index: upstream's
            // branch chain has no arm for it, so the open span continues.
            None => {}
        }
    }
    if let Some((open, start)) = current {
        spans.push((open, start, path.len()));
    }
    spans
}

/// Give every sub-token of a word one tag so the word forms a single span.
///
/// The word takes the tag of its first non-`O` sub-token, so a word is never
/// dropped because only a later piece was tagged; the remaining sub-tokens
/// become that entity's `I` state. Tokens with no word id stay on their own.
pub fn group_words(path: &[usize], word_ids: &[Option<u32>], labels: &LabelInfo) -> Vec<usize> {
    let mut grouped = path.to_vec();
    let mut start = 0;
    while start < path.len() {
        let mut end = start + 1;
        if let Some(word) = word_ids.get(start).copied().flatten() {
            while end < path.len() && word_ids.get(end).copied().flatten() == Some(word) {
                end += 1;
            }
        }
        if let Some(&first) = path[start..end]
            .iter()
            .find(|&&tag| tag != labels.background)
        {
            let span = labels.token_to_span.get(first).copied().unwrap_or(0);
            let inside = labels
                .span_names
                .get(span)
                .and_then(|name| labels.states_by_span.get(name))
                .and_then(|states| states.get(&Boundary::Inside))
                .copied()
                .unwrap_or(first);
            grouped[start] = first;
            for tag in &mut grouped[start + 1..end] {
                *tag = inside;
            }
        }
        start = end;
    }
    grouped
}

/// The redaction placeholder for an entity label: `<` + the label upper-cased
/// with every run of non-alphanumerics replaced by `_` and trimmed of `_`
/// (`REDACTED` when nothing is left) + `>`.
pub fn placeholder(label: &str) -> String {
    let mut normalized = String::with_capacity(label.len());
    let mut in_run = false;
    for ch in label.to_uppercase().chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch);
            in_run = false;
        } else if !in_run {
            normalized.push('_');
            in_run = true;
        }
    }
    let trimmed = normalized.trim_matches('_');
    format!("<{}>", if trimmed.is_empty() { "REDACTED" } else { trimmed })
}

/// One detected entity. `start` / `end` are Unicode scalar offsets into the
/// input text, `end` exclusive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenSpan {
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub placeholder: String,
}

/// The shape-1 output: `{"text", "spans": [...], "redacted_text"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenClassificationResult {
    pub text: String,
    pub spans: Vec<TokenSpan>,
    pub redacted_text: String,
}

/// Python's `str.isspace`, which also counts the ASCII information separators
/// (`U+001C`..`U+001F`) that Rust's `char::is_whitespace` does not.
fn is_python_space(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

/// Turn a per-token label path into non-overlapping character spans.
///
/// With `word_ids` (WordPiece-style tokenizers) sub-tokens are grouped into
/// words first, and same-entity spans that touch with no gap (`617` `-` `555`)
/// are merged, because the pre-tokenizer splits punctuation into separate
/// words that a checkpoint may tag `B-` individually. Spans are sorted by
/// `(start, -length, label)` and a span starting before the end of the
/// previous kept one is dropped.
pub fn decode_spans(
    text: &str,
    path: &[usize],
    offsets: &[(usize, usize)],
    labels: &LabelInfo,
    word_ids: Option<&[Option<u32>]>,
    trim_whitespace: bool,
) -> Result<Vec<TokenSpan>> {
    if offsets.len() != path.len() {
        bail!(
            "token classification: {} offsets for a path of {} tokens",
            offsets.len(),
            path.len()
        );
    }
    if let Some(&tag) = path.iter().find(|&&tag| tag >= labels.labels.len()) {
        bail!(
            "token classification: label index {tag} is out of range for {} labels",
            labels.labels.len()
        );
    }
    let grouped;
    let path: &[usize] = match word_ids {
        Some(ids) => {
            if ids.len() != path.len() {
                bail!(
                    "token classification: {} word ids for a path of {} tokens",
                    ids.len(),
                    path.len()
                );
            }
            grouped = group_words(path, ids, labels);
            &grouped
        }
        None => path,
    };

    let chars: Vec<char> = text.chars().collect();
    let mut detected: Vec<(&str, usize, usize)> = Vec::new();
    for (span, token_start, token_end) in token_spans(path, labels) {
        let mut start = offsets[token_start].0.min(chars.len());
        let mut end = offsets[token_end - 1].1.min(chars.len());
        if trim_whitespace {
            while start < end && is_python_space(chars[start]) {
                start += 1;
            }
            while end > start && is_python_space(chars[end - 1]) {
                end -= 1;
            }
        }
        if end <= start {
            continue;
        }
        detected.push((labels.span_names[span].as_str(), start, end));
    }

    // Byte-level tokenizers can give several tokens the same character range.
    // Keep one deterministic, globally non-overlapping span set.
    detected.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then((b.2 - b.1).cmp(&(a.2 - a.1)))
            .then(a.0.cmp(b.0))
    });
    let mut kept: Vec<(&str, usize, usize)> = Vec::new();
    let mut cursor = 0;
    for (label, start, end) in detected {
        if start < cursor {
            continue;
        }
        if word_ids.is_some()
            && let Some(last) = kept.last_mut()
            && last.0 == label
            && last.2 == start
        {
            last.2 = end;
            cursor = end;
            continue;
        }
        kept.push((label, start, end));
        cursor = end;
    }
    Ok(kept
        .into_iter()
        .map(|(label, start, end)| TokenSpan {
            label: label.to_string(),
            start,
            end,
            text: chars[start..end].iter().collect(),
            placeholder: placeholder(label),
        })
        .collect())
}

/// Replace every span not in `keep_labels`, right to left, with `replacement`
/// or the span's placeholder. Offsets are Unicode scalar indices and are
/// clamped to the text the way Python slicing clamps them.
pub fn redact(
    text: &str,
    spans: &[TokenSpan],
    replacement: Option<&str>,
    keep_labels: &[String],
) -> String {
    let keep: HashSet<&str> = keep_labels.iter().map(String::as_str).collect();
    let mut output: Vec<char> = text.chars().collect();
    for span in spans.iter().rev() {
        if keep.contains(span.label.as_str()) {
            continue;
        }
        let value = replacement.unwrap_or(&span.placeholder);
        let start = span.start.min(output.len());
        let end = span.end.min(output.len());
        let mut next = Vec::with_capacity(output.len() + value.len());
        next.extend_from_slice(&output[..start]);
        next.extend(value.chars());
        next.extend_from_slice(&output[end..]);
        output = next;
    }
    output.into_iter().collect()
}

#[cfg(test)]
#[path = "spans_tests.rs"]
mod spans_tests;
