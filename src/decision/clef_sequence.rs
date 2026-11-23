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

//! Clef sequence building: a port of upstream's `_render_criterion`,
//! `_options` and `Clef._sequence` (`mlx_vlm/models/clef/clef.py`).
//!
//! Every piece is encoded on its own without special tokens, exactly as
//! upstream concatenates separate `tokenizer(text)` calls, so the spans of
//! each question's instruction and each option's description are token
//! positions in the final sequence. Everything here runs on the tokenizer
//! alone, so the sequence tests need no weights.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use super::decider2_prompt::python_float;
use super::{DecisionError, Question, QuestionKind};

/// Most tokens one sequence may hold (upstream's `max_length`).
pub(crate) const MAX_LENGTH: usize = 16_384;

const SYSTEM: &str = "<|im_start|>system\nRead the complete state and schema. Decide every \
                      field jointly. Each answer must be exactly one of that field's allowed \
                      options.<|im_end|>\n<|im_start|>user\nSTATE:\n";
const SUFFIX: &str =
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";
const BOOL_TRUE: &str = "The proposition is true or the answer is yes.";
const BOOL_FALSE: &str = "The proposition is false or the answer is no.";

/// Upstream's rendered type name (`bool` is sent to the head as `noul`).
pub(crate) fn type_name(kind: QuestionKind) -> &'static str {
    match kind {
        QuestionKind::Score => "score",
        QuestionKind::Bool => "noul",
        _ => "choice",
    }
}

/// `QUESTION_TYPES`: the head's type embedding row.
pub(crate) fn type_id(kind: QuestionKind) -> i32 {
    match kind {
        QuestionKind::Bool => 0,
        QuestionKind::Score => 2,
        _ => 1,
    }
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"),
/// sort_keys=True)`.
pub(crate) fn compact_json(value: &Value) -> String {
    let mut out = String::new();
    write_compact(value, &mut out);
    out
}

fn write_compact(value: &Value, out: &mut String) {
    match value {
        Value::Number(number) if !(number.is_i64() || number.is_u64()) => {
            out.push_str(&python_float(number.as_f64().unwrap_or(0.0)));
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_compact(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_compact(&map[key], out);
            }
            out.push('}');
        }
        // Null, booleans, integers and strings print as serde_json prints
        // them, which is what Python prints for every value a Rust string
        // can hold.
        other => out.push_str(&other.to_string()),
    }
}

/// `_render_criterion`: a string as is, anything else as compact JSON.
pub(crate) fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => compact_json(other),
    }
}

/// Python truthiness of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// `_options`: `(option id, description)` in the order the schema lists
/// them. Descriptions are the raw request values (`Null` when absent).
pub(crate) fn options(question: &Question) -> Vec<(String, Value)> {
    let criteria = question
        .raw
        .get("criteria")
        .filter(|value| !value.is_null());
    match question.kind {
        QuestionKind::Bool => {
            let given = criteria.and_then(Value::as_object);
            let pick = |key: &str, default: &str| {
                given
                    .and_then(|map| map.get(key))
                    .cloned()
                    .unwrap_or_else(|| Value::String(default.to_string()))
            };
            vec![
                ("true".to_string(), pick("true", BOOL_TRUE)),
                ("false".to_string(), pick("false", BOOL_FALSE)),
            ]
        }
        QuestionKind::Score => match criteria {
            Some(Value::Array(levels)) => levels
                .iter()
                .enumerate()
                .map(|(index, level)| (index.to_string(), level.clone()))
                .collect(),
            _ => Vec::new(),
        },
        _ => {
            let mut labels: Vec<(String, Value)> = match criteria {
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|item| (render_label(item), Value::Null))
                    .collect(),
                Some(Value::Object(map)) => map
                    .iter()
                    .map(|(name, description)| (name.clone(), description.clone()))
                    .collect(),
                _ => Vec::new(),
            };
            labels.sort_by(|a, b| a.0.cmp(&b.0));
            labels
        }
    }
}

fn render_label(item: &Value) -> String {
    item.as_str()
        .map_or_else(|| item.to_string(), str::to_string)
}

/// One question's place in the sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClefRow {
    /// `[start, end)` of the rendered instruction.
    pub(crate) question_span: (usize, usize),
    /// `[start, end)` of each option's rendered `{option_id, description}`.
    pub(crate) option_spans: Vec<(usize, usize)>,
    /// Option ids in schema order (choice labels sorted).
    pub(crate) labels: Vec<String>,
}

/// A whole sequence and the rows the head reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClefSequence {
    /// Token ids, prefix (media included), state, schema, suffix.
    pub(crate) ids: Vec<u32>,
    /// One row per question, in request order.
    pub(crate) rows: Vec<ClefRow>,
}

/// The tokenizer plus the special ids media blocks are written with.
pub(crate) struct ClefTokenizer {
    tokenizer: tokenizers::Tokenizer,
    pub(crate) image_pad: u32,
    pub(crate) video_pad: u32,
    pub(crate) vision_start: u32,
    pub(crate) vision_end: u32,
}

impl ClefTokenizer {
    /// Load `tokenizer.json` from the checkpoint.
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!("{error}"))
            .with_context(|| format!("loading {}", path.display()))?;
        let id = |token: &str| {
            tokenizer
                .token_to_id(token)
                .with_context(|| format!("the clef tokenizer has no `{token}` token"))
        };
        Ok(Self {
            image_pad: id("<|image_pad|>")?,
            video_pad: id("<|video_pad|>")?,
            vision_start: id("<|vision_start|>")?,
            vision_end: id("<|vision_end|>")?,
            tokenizer,
        })
    }

    /// Encode without special tokens added (special tokens in the text are
    /// still parsed as themselves).
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| DecisionError::Internal(format!("clef tokenizer: {error}")))
    }
}

/// The media part of the prefix: what upstream's processor expands
/// `"<|vision_start|><|image_pad|><|vision_end|>"` per image and
/// `"<|vision_start|><|vision_start|><|video_pad|><|vision_end|><|vision_end|>"`
/// per video into, followed by `"\n"`.
///
/// `images` holds each image's token count; `videos` each video's per-group
/// token count and its group timestamps in seconds.
pub(crate) fn media_block(
    tokenizer: &ClefTokenizer,
    images: &[usize],
    videos: &[(usize, Vec<f64>)],
) -> Result<Vec<u32>, DecisionError> {
    let mut ids = Vec::new();
    for &count in images {
        ids.push(tokenizer.vision_start);
        ids.extend(std::iter::repeat_n(tokenizer.image_pad, count));
        ids.push(tokenizer.vision_end);
    }
    for (group_tokens, seconds) in videos {
        ids.push(tokenizer.vision_start);
        for &second in seconds {
            ids.extend(tokenizer.encode(&crate::qwen_vl::qwen_video_timestamp_text(second))?);
            ids.push(tokenizer.vision_start);
            ids.extend(std::iter::repeat_n(tokenizer.video_pad, *group_tokens));
            ids.push(tokenizer.vision_end);
        }
        ids.push(tokenizer.vision_end);
    }
    ids.extend(tokenizer.encode("\n")?);
    Ok(ids)
}

/// The schema tokens and each row's spans relative to the schema start.
pub(crate) fn schema(
    tokenizer: &ClefTokenizer,
    questions: &[(String, Question)],
) -> Result<(Vec<u32>, Vec<ClefRow>), DecisionError> {
    let mut ids = tokenizer.encode("\n\nSCHEMA FIELDS:\n")?;
    let mut rows = Vec::with_capacity(questions.len());
    for (index, (name, question)) in questions.iter().enumerate() {
        ids.extend(tokenizer.encode(&format!(
            "\nFIELD {}\nID: {name}\nTYPE: {}\nINSTRUCTION: ",
            index + 1,
            type_name(question.kind)
        ))?);
        let instruction = question
            .raw
            .get("instructions")
            .filter(|value| truthy(value))
            .map_or_else(|| name.clone(), render_criterion);
        let start = ids.len();
        ids.extend(tokenizer.encode(&instruction)?);
        let question_span = (start, ids.len());
        ids.extend(tokenizer.encode("\nALLOWED OPTIONS:\n")?);
        let mut option_spans = Vec::new();
        let mut labels = Vec::new();
        for (number, (option, description)) in options(question).into_iter().enumerate() {
            ids.extend(tokenizer.encode(&format!("OPTION {}: ", number + 1))?);
            let mut semantics = Map::new();
            semantics.insert("option_id".to_string(), Value::String(option.clone()));
            if !description.is_null() {
                semantics.insert("description".to_string(), description);
            }
            let start = ids.len();
            ids.extend(tokenizer.encode(&compact_json(&Value::Object(semantics)))?);
            option_spans.push((start, ids.len()));
            labels.push(option);
            ids.extend(tokenizer.encode("\n")?);
        }
        ids.extend(tokenizer.encode("END FIELD\n")?);
        rows.push(ClefRow {
            question_span,
            option_spans,
            labels,
        });
    }
    Ok((ids, rows))
}

/// The fixed part of a sequence (prefix with media, schema and suffix) and
/// the rows, before the state is placed.
pub(crate) struct Frame {
    prefix: Vec<u32>,
    schema: Vec<u32>,
    suffix: Vec<u32>,
    rows: Vec<ClefRow>,
}

impl Frame {
    /// Tokens the sequence holds without its state.
    pub(crate) fn fixed_len(&self) -> usize {
        self.prefix.len() + self.schema.len() + self.suffix.len()
    }
}

/// Everything but the state. Refuses a schema (with its media) that alone
/// exceeds [`MAX_LENGTH`], with upstream's message.
pub(crate) fn frame(
    tokenizer: &ClefTokenizer,
    questions: &[(String, Question)],
    media: &[u32],
) -> Result<Frame, DecisionError> {
    let mut prefix = tokenizer.encode(SYSTEM)?;
    if !media.is_empty() {
        prefix.extend_from_slice(media);
    }
    let (schema, rows) = schema(tokenizer, questions)?;
    let suffix = tokenizer.encode(SUFFIX)?;
    let frame = Frame {
        prefix,
        schema,
        suffix,
        rows,
    };
    let fixed = frame.fixed_len();
    if fixed > MAX_LENGTH {
        return Err(DecisionError::InvalidInput(format!(
            "schema requires {fixed} tokens before state; maximum is {MAX_LENGTH}"
        )));
    }
    Ok(frame)
}

/// Place the state (truncated to what fits) and shift the rows onto the
/// final positions. `state` of `None` renders as upstream renders `None`.
pub(crate) fn place_state(
    tokenizer: &ClefTokenizer,
    frame: Frame,
    state: Option<&Value>,
) -> Result<ClefSequence, DecisionError> {
    let rendered = state.map_or_else(|| "null".to_string(), render_criterion);
    let mut state_ids = tokenizer.encode(&rendered)?;
    state_ids.truncate(MAX_LENGTH - frame.fixed_len());
    let offset = frame.prefix.len() + state_ids.len();
    let shift = |(start, end): (usize, usize)| (start + offset, end + offset);
    let rows = frame
        .rows
        .into_iter()
        .map(|row| ClefRow {
            question_span: shift(row.question_span),
            option_spans: row.option_spans.into_iter().map(shift).collect(),
            labels: row.labels,
        })
        .collect();
    let mut ids = frame.prefix;
    ids.extend(state_ids);
    ids.extend(frame.schema);
    ids.extend(frame.suffix);
    Ok(ClefSequence { ids, rows })
}

/// [`frame`] then [`place_state`].
#[cfg(test)]
pub(crate) fn sequence(
    tokenizer: &ClefTokenizer,
    state: Option<&Value>,
    questions: &[(String, Question)],
    media: &[u32],
) -> Result<ClefSequence, DecisionError> {
    let frame = frame(tokenizer, questions, media)?;
    place_state(tokenizer, frame, state)
}
