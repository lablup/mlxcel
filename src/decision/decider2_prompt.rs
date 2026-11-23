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

//! Decider-2b question rendering and prompt token ids.
//!
//! A port of upstream's `_text`, `_annotate`, `_render_question` and
//! `Decider2._prompt` (`mlx_vlm/models/decider2/decider2.py`). Everything here
//! runs on the tokenizer alone, so the prompt tests need no weights.

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use super::{DecisionError, Question, QuestionKind};

/// Label tokens the checkpoint answers with.
pub(crate) const LABEL_COUNT: usize = 255;
/// Largest option count that is rendered as one string.
const INLINE_OPTIONS: usize = 10;
/// Upstream's `_annotate` threshold: lists this long get an `_index` per item.
const ANNOTATE_MIN_ITEMS: usize = 8;

fn invalid(message: &str) -> DecisionError {
    DecisionError::InvalidInput(message.to_string())
}

/// What a question renders to: the kind the temperature is looked up by,
/// the instruction, and one option line per answer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RenderedQuestion {
    /// `choice`, `score` or `noul` (upstream's rendered type).
    pub(crate) kind: &'static str,
    /// The instruction text.
    pub(crate) instruction: String,
    /// The option lines, in answer order.
    pub(crate) options: Vec<String>,
    /// `score` level texts; empty otherwise.
    pub(crate) legend: Vec<String>,
}

/// Upstream's `_text`: a string as is, anything else as Python JSON.
pub(crate) fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_json(other),
    }
}

/// Python's `json.dumps(value, ensure_ascii=False)` with default separators.
///
/// Object keys keep their request order (`serde_json` is built with
/// `preserve_order`). String escaping matches Python's for every character a
/// Rust string can hold, so `serde_json` serializes them.
pub(crate) fn python_json(value: &Value) -> String {
    let mut out = String::new();
    write_python_json(value, &mut out);
    out
}

fn write_python_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                out.push_str(&number.to_string());
            } else {
                out.push_str(&python_float(number.as_f64().unwrap_or(0.0)));
            }
        }
        Value::String(_) => out.push_str(&value.to_string()),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push_str(": ");
                write_python_json(item, out);
            }
            out.push('}');
        }
    }
}

/// Python's `repr(float)`: the shortest round-trip digits, positional for
/// decimal exponents in `-4..16` and `d.ddde+XX` otherwise.
pub(crate) fn python_float(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "NaN".to_string()
        } else if value > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    // `{:e}` prints the shortest round-trip mantissa, e.g. `1.25e-5`.
    let formatted = format!("{:e}", value.abs());
    let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            let (whole, fraction) = digits.split_at(point as usize);
            format!("{whole}.{fraction}")
        };
        format!("{sign}{body}")
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{first}{rest}e{exp_sign}{:02}", exponent.abs())
    }
}

/// Upstream's `_annotate`: a list of 8 or more items gets an `_index` per
/// item, recursively. `_index` goes first; an item's own `_index` replaces
/// the value but keeps that position, as a Python dict merge does.
pub(crate) fn annotate(value: &Value) -> Value {
    match value {
        Value::Array(items) if items.len() >= ANNOTATE_MIN_ITEMS => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let mut entry = Map::new();
                    entry.insert("_index".to_string(), Value::from(index));
                    match annotate(item) {
                        Value::Object(fields) => entry.extend(fields),
                        other => {
                            entry.insert("value".to_string(), other);
                        }
                    }
                    Value::Object(entry)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(annotate).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), annotate(item)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The prompt context: a string state as is, anything else annotated and
/// serialized.
pub(crate) fn render_state(state: Option<&Value>) -> String {
    match state {
        Some(Value::String(text)) => text.clone(),
        Some(other) => python_json(&annotate(other)),
        None => "null".to_string(),
    }
}

/// A description that renders as a bare label (`None` or `""` upstream).
fn is_blank(value: Option<&Value>) -> bool {
    matches!(value, None | Some(Value::Null)) || value.and_then(Value::as_str) == Some("")
}

/// Upstream's `_render_question` for a question the shared layer validated.
///
/// Descriptions and levels are read from the raw request so a non-string
/// value is JSON-encoded the way Python encodes it.
pub(crate) fn render_question(question: &Question) -> Result<RenderedQuestion, DecisionError> {
    let instruction = match question.raw.get("instructions") {
        Some(value) => text(value),
        None => question.raw.get("question").map(text).unwrap_or_default(),
    };
    if instruction.is_empty() {
        return Err(invalid("Question needs instructions"));
    }
    let criteria = question.raw.get("criteria");
    let (kind, options, legend) = match question.kind {
        QuestionKind::Choice => {
            let labels = question.labels();
            if !(2..=LABEL_COUNT).contains(&labels.len()) {
                return Err(invalid("choice needs 2 to 255 options"));
            }
            let descriptions = criteria.and_then(Value::as_object);
            let options = labels
                .iter()
                .map(|label| {
                    let description = descriptions.and_then(|map| map.get(&label.name));
                    if is_blank(description) {
                        label.name.clone()
                    } else {
                        format!(
                            "{}: {}",
                            label.name,
                            text(description.unwrap_or(&Value::Null))
                        )
                    }
                })
                .collect();
            ("choice", options, Vec::new())
        }
        QuestionKind::Score => {
            let levels: Vec<String> = match criteria {
                Some(Value::Array(items)) => items.iter().map(text).collect(),
                _ => question.levels().to_vec(),
            };
            if !(2..=10).contains(&levels.len()) {
                return Err(invalid("score needs 2 to 10 levels"));
            }
            let options = levels
                .iter()
                .enumerate()
                .map(|(index, level)| format!("{index}: {level}"))
                .collect();
            ("score", options, levels)
        }
        QuestionKind::Bool => {
            let map = criteria.and_then(Value::as_object);
            let line = |name: &str, key: &str| {
                let description = map.and_then(|map| map.get(key));
                if is_blank(description) {
                    name.to_string()
                } else {
                    format!("{name}: {}", text(description.unwrap_or(&Value::Null)))
                }
            };
            (
                "noul",
                vec![line("no", "false"), line("yes", "true")],
                Vec::new(),
            )
        }
        other => {
            return Err(invalid(&format!(
                "Unsupported question type: '{}'",
                other.as_str()
            )));
        }
    };
    Ok(RenderedQuestion {
        kind,
        instruction,
        options,
        legend,
    })
}

/// The isolated-row instruction for one score level, with a leading
/// `"<int>:"` stripped from the level text.
pub(crate) fn isolated_level_instruction(instruction: &str, level: &str) -> String {
    static LEADING_INDEX: OnceLock<Option<fancy_regex::Regex>> = OnceLock::new();
    let pattern = LEADING_INDEX.get_or_init(|| fancy_regex::Regex::new(r"^\s*-?\d+\s*:\s*").ok());
    let level = match pattern {
        Some(pattern) => pattern.replace(level, ""),
        None => std::borrow::Cow::Borrowed(level),
    };
    format!("{instruction}\nProposed answer: {level}\nDoes the proposed answer fit?")
}

/// The tokenizer and label tokens: everything the prompt needs.
pub(crate) struct Decider2Prompter {
    tokenizer: tokenizers::Tokenizer,
    /// `A`, `B`, ... in label order.
    pub(crate) labels: Vec<String>,
    /// The single token id of each label.
    pub(crate) label_ids: Vec<u32>,
    /// State tokens kept from the start of `"Context:\n" + state`.
    pub(crate) max_state_tokens: usize,
}

impl Decider2Prompter {
    /// Load `tokenizer.json` and pick the label tokens.
    pub(crate) fn load(model_dir: &Path, max_state_tokens: usize) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!("{error}"))
            .with_context(|| format!("loading {}", path.display()))?;
        let (labels, label_ids) = select_labels(&tokenizer)?;
        Ok(Self {
            tokenizer,
            labels,
            label_ids,
            max_state_tokens,
        })
    }

    /// The id of a token string, if the vocabulary has it.
    pub(crate) fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Encode without special tokens.
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| DecisionError::Internal(format!("tokenizer: {error}")))
    }

    /// The ids of `"Context:\n" + context`, truncated to `max_state_tokens`.
    ///
    /// Every row of a request shares them, so a request encodes its state
    /// once rather than once per question.
    pub(crate) fn context_ids(&self, context: &str) -> Result<Vec<u32>, DecisionError> {
        let mut ids = self.encode(&format!("Context:\n{context}"))?;
        ids.truncate(self.max_state_tokens);
        Ok(ids)
    }

    /// Upstream's `_prompt`: the truncated context (from
    /// [`Self::context_ids`]), the question, one line per option and the open
    /// answer slot. Past 10 options each label id is spliced in directly so
    /// it cannot merge with its neighbours.
    pub(crate) fn prompt(
        &self,
        context_ids: &[u32],
        instruction: &str,
        options: &[String],
    ) -> Result<Vec<u32>, DecisionError> {
        let mut ids = context_ids.to_vec();
        let head = format!("\n\nQuestion: {instruction}\nOptions:");
        let tail = "\nAnswer: (";
        if options.len() <= INLINE_OPTIONS {
            let mut rendered = head;
            for (label, option) in self.labels.iter().zip(options) {
                rendered.push_str(&format!("\n({label}) {option}"));
            }
            rendered.push_str(tail);
            ids.extend(self.encode(&rendered)?);
        } else {
            ids.extend(self.encode(&head)?);
            let open = self.encode("\n(")?;
            for (label_id, option) in self.label_ids.iter().zip(options) {
                ids.extend_from_slice(&open);
                ids.push(*label_id);
                ids.extend(self.encode(&format!(") {option}"))?);
            }
            ids.extend(self.encode(tail)?);
        }
        Ok(ids)
    }
}

/// `A`..`Z` then `AA`..`ZZ`, kept when they encode to one token, first 255.
fn select_labels(tokenizer: &tokenizers::Tokenizer) -> Result<(Vec<String>, Vec<u32>)> {
    let letters = || (b'A'..=b'Z').map(char::from);
    let names = letters()
        .map(String::from)
        .chain(letters().flat_map(|a| letters().map(move |b| format!("{a}{b}"))));
    let mut labels = Vec::with_capacity(LABEL_COUNT);
    let mut ids = Vec::with_capacity(LABEL_COUNT);
    for name in names {
        if labels.len() == LABEL_COUNT {
            break;
        }
        let encoding = tokenizer
            .encode(name.as_str(), false)
            .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
        if let [id] = encoding.get_ids() {
            labels.push(name);
            ids.push(*id);
        }
    }
    let distinct: HashSet<u32> = ids.iter().copied().collect();
    if labels.len() != LABEL_COUNT || distinct.len() != LABEL_COUNT {
        bail!("The decider tokenizer does not provide 255 distinct label tokens");
    }
    Ok((labels, ids))
}
