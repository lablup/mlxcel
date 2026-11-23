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

/// One image's placeholder run: `<|vision_start|>`, `count` image pads and
/// `<|vision_end|>`.
pub(crate) fn image_block(tokenizer: &ClefTokenizer, count: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(count + 2);
    ids.push(tokenizer.vision_start);
    ids.extend(std::iter::repeat_n(tokenizer.image_pad, count));
    ids.push(tokenizer.vision_end);
    ids
}

/// One video's placeholder run: every temporal group framed and preceded by
/// its timestamp text, the whole video framed again.
pub(crate) fn video_block(
    tokenizer: &ClefTokenizer,
    group_tokens: usize,
    seconds: &[f64],
) -> Result<Vec<u32>, DecisionError> {
    let mut ids = vec![tokenizer.vision_start];
    for &second in seconds {
        ids.extend(tokenizer.encode(&crate::qwen_vl::qwen_video_timestamp_text(second))?);
        ids.push(tokenizer.vision_start);
        ids.extend(std::iter::repeat_n(tokenizer.video_pad, group_tokens));
        ids.push(tokenizer.vision_end);
    }
    ids.push(tokenizer.vision_end);
    Ok(ids)
}

/// The media part of the prefix: what upstream's processor expands
/// `"<|vision_start|><|image_pad|><|vision_end|>"` per image and
/// `"<|vision_start|><|vision_start|><|video_pad|><|vision_end|><|vision_end|>"`
/// per video into, followed by `"\n"`.
///
/// `images` holds each image's token count; `videos` each video's per-group
/// token count and its group timestamps in seconds.
#[cfg(test)]
pub(crate) fn media_block(
    tokenizer: &ClefTokenizer,
    images: &[usize],
    videos: &[(usize, Vec<f64>)],
) -> Result<Vec<u32>, DecisionError> {
    let mut next = videos.iter();
    let (ids, _) = gated_media_block(tokenizer, images, videos.len(), 0, |_| {
        let (tokens, seconds) = next.next().cloned().unwrap_or_default();
        Ok(((), tokens, seconds))
    })?;
    Ok(ids)
}

/// [`media_block`] for media that is still to be decoded, refusing a request
/// as soon as it cannot fit.
///
/// `text_tokens` is the frame's length without media. Before each video is
/// prepared (`prepare(index)` decodes it and returns it with its per-group
/// token count and group timestamps), the tokens so far are checked against
/// [`MAX_LENGTH`]: every video only adds tokens, so a request already over the
/// cap is refused without decoding the rest. The final, exact check is
/// [`Frame::with_media`].
pub(crate) fn gated_media_block<V>(
    tokenizer: &ClefTokenizer,
    images: &[usize],
    videos: usize,
    text_tokens: usize,
    mut prepare: impl FnMut(usize) -> Result<(V, usize, Vec<f64>), DecisionError>,
) -> Result<(Vec<u32>, Vec<V>), DecisionError> {
    let newline = tokenizer.encode("\n")?;
    let mut ids = Vec::new();
    for &count in images {
        ids.extend(image_block(tokenizer, count));
    }
    let mut prepared = Vec::with_capacity(videos);
    for index in 0..videos {
        let fixed = text_tokens + ids.len() + newline.len();
        if fixed > MAX_LENGTH {
            return Err(schema_too_long(fixed, false));
        }
        let (video, group_tokens, seconds) = prepare(index)?;
        ids.extend(video_block(tokenizer, group_tokens, &seconds)?);
        prepared.push(video);
    }
    ids.extend(newline);
    Ok((ids, prepared))
}

/// Upstream's refusal of a sequence whose fixed part exceeds [`MAX_LENGTH`].
/// `complete` is false when encoding stopped early, so `tokens` is a lower
/// bound.
pub(crate) fn schema_too_long(tokens: usize, complete: bool) -> DecisionError {
    let at_least = if complete { "" } else { "at least " };
    DecisionError::InvalidInput(format!(
        "schema requires {at_least}{tokens} tokens before state; maximum is {MAX_LENGTH}"
    ))
}

/// Refuse a request whose own text (state, schema) carries image or video
/// placeholder tokens when media is present.
///
/// Every piece is encoded with special tokens parsed, as upstream does, so a
/// literal `<|image_pad|>` in the text becomes a placeholder the vision
/// features and MRoPE positions are matched against; with media present the
/// counts no longer agree. `expected` is the pad count of the media block.
pub(crate) fn check_placeholders(
    tokenizer: &ClefTokenizer,
    ids: &[u32],
    expected: usize,
) -> Result<(), DecisionError> {
    if placeholder_count(tokenizer, ids) != expected {
        return Err(DecisionError::InvalidInput(
            "state and questions may not contain the <|image_pad|> or <|video_pad|> tokens \
             when images or videos are given"
                .to_string(),
        ));
    }
    Ok(())
}

/// Image and video pad tokens in `ids`.
pub(crate) fn placeholder_count(tokenizer: &ClefTokenizer, ids: &[u32]) -> usize {
    ids.iter()
        .filter(|&&id| id == tokenizer.image_pad || id == tokenizer.video_pad)
        .count()
}

/// The schema tokens and each row's spans relative to the schema start.
///
/// `base` is the length of everything else in the frame. Encoding stops
/// with [`schema_too_long`] as soon as `base` plus the schema so far exceeds
/// [`MAX_LENGTH`] while input remains, so an oversized schema is refused
/// after at most one piece past the cap rather than encoded in full.
pub(crate) fn schema(
    tokenizer: &ClefTokenizer,
    questions: &[(String, Question)],
    base: usize,
) -> Result<(Vec<u32>, Vec<ClefRow>), DecisionError> {
    let mut ids = tokenizer.encode("\n\nSCHEMA FIELDS:\n")?;
    let mut rows = Vec::with_capacity(questions.len());
    let over = |ids: &[u32]| base + ids.len() > MAX_LENGTH;
    for (index, (name, question)) in questions.iter().enumerate() {
        let last_question = index + 1 == questions.len();
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
        if over(&ids) {
            return Err(schema_too_long(base + ids.len(), false));
        }
        let options = options(question);
        let option_count = options.len();
        for (number, (option, description)) in options.into_iter().enumerate() {
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
            if !(last_question && number + 1 == option_count) && over(&ids) {
                return Err(schema_too_long(base + ids.len(), false));
            }
        }
        ids.extend(tokenizer.encode("END FIELD\n")?);
        rows.push(ClefRow {
            question_span,
            option_spans,
            labels,
        });
        if !last_question && over(&ids) {
            return Err(schema_too_long(base + ids.len(), false));
        }
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

    /// Insert the media block after the system prompt (where [`frame`] puts
    /// it) and check the cap again. Row spans are schema-relative until
    /// [`place_state`], so they need no shift.
    pub(crate) fn with_media(mut self, media: &[u32]) -> Result<Self, DecisionError> {
        self.prefix.extend_from_slice(media);
        self.checked()
    }

    fn checked(self) -> Result<Self, DecisionError> {
        let fixed = self.fixed_len();
        if fixed > MAX_LENGTH {
            return Err(schema_too_long(fixed, true));
        }
        Ok(self)
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
    let suffix = tokenizer.encode(SUFFIX)?;
    let (schema, rows) = schema(tokenizer, questions, prefix.len() + suffix.len())?;
    Frame {
        prefix,
        schema,
        suffix,
        rows,
    }
    .checked()
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
