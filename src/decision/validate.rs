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

//! The shared request validation, rules 1 to 8 of the decision contract.
//!
//! The order and every message follow upstream (`mlx_vlm/decision.py`
//! `predict` and `_validate_question_structure`, and the state check of
//! `mlx_vlm/server/decisions.py`), so a client written against mlx-vlm sees
//! the same 400 text here. `videos` counts as media for the state rule, which
//! is the one place this module goes beyond upstream's wording.

use serde_json::{Map, Value};

use super::{Criteria, DecisionError, Label, MediaKind, Question, QuestionKind};

/// Default `multi_label` threshold.
pub const DEFAULT_THRESHOLD: f64 = 0.5;

/// Which media fields a request carries (present, non-null, non-empty).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaPresence {
    /// `images` carries at least one entry.
    pub images: bool,
    /// `videos` carries at least one entry.
    pub videos: bool,
    /// `audio` carries a clip.
    pub audio: bool,
}

impl MediaPresence {
    /// Whether any media is present.
    #[must_use]
    pub fn any(self) -> bool {
        self.images || self.videos || self.audio
    }

    fn has(self, kind: MediaKind) -> bool {
        match kind {
            MediaKind::Image => self.images,
            MediaKind::Video => self.videos,
            MediaKind::Audio => self.audio,
        }
    }
}

fn invalid(message: impl Into<String>) -> DecisionError {
    DecisionError::InvalidInput(message.into())
}

/// Rule 4: `questions` must be a non-empty map of name to object.
///
/// Also run on its own by `mlxcel decide` before the model loads.
pub fn validate_question_structure(
    questions: &Value,
) -> Result<&Map<String, Value>, DecisionError> {
    let map = match questions {
        Value::Object(map) if !map.is_empty() => map,
        _ => return Err(invalid("At least one named question is required")),
    };
    if map.values().any(|question| !question.is_object()) {
        return Err(invalid("Questions must map names to question dictionaries"));
    }
    Ok(map)
}

/// Validate a whole request against a model's declarations (rules 1 to 8).
///
/// `state` of `None` means absent or `null`. On success every question is
/// returned typed, in request order.
pub fn validate_questions(
    decision_types: &[QuestionKind],
    decision_media: &[MediaKind],
    state: Option<&Value>,
    questions: &Value,
    media: MediaPresence,
) -> Result<Vec<(String, Question)>, DecisionError> {
    // Rule 1.
    if state.is_none_or(Value::is_null) && !media.any() {
        return Err(invalid(
            "state is required unless images or audio are given",
        ));
    }
    // Rule 2.
    if decision_types.is_empty() {
        return Err(invalid("This model does not support decision prediction"));
    }
    // Rule 3, in upstream's `_MEDIA` order.
    for kind in [MediaKind::Image, MediaKind::Video, MediaKind::Audio] {
        if media.has(kind) && !decision_media.contains(&kind) {
            return Err(invalid(format!(
                "This model does not support {} input",
                kind.noun()
            )));
        }
    }
    // Rule 4.
    let map = validate_question_structure(questions)?;
    // Rules 5 to 8, question by question.
    map.iter()
        .map(|(name, question)| {
            let raw = question.as_object().cloned().unwrap_or_default();
            validate_one(decision_types, raw).map(|question| (name.clone(), question))
        })
        .collect()
}

fn validate_one(
    decision_types: &[QuestionKind],
    raw: Map<String, Value>,
) -> Result<Question, DecisionError> {
    // Rule 5: `type` defaults to `choice`.
    let default_type = Value::String(QuestionKind::Choice.as_str().to_string());
    let requested = raw.get("type").unwrap_or(&default_type);
    let kind = requested
        .as_str()
        .and_then(QuestionKind::parse)
        .filter(|kind| decision_types.contains(kind))
        .ok_or_else(|| {
            invalid(format!(
                "This model does not support {} decisions",
                python_repr(requested)
            ))
        })?;
    let criteria_value = raw.get("criteria").filter(|value| !value.is_null());

    let criteria = match kind {
        QuestionKind::Choice | QuestionKind::MultiLabel | QuestionKind::Score => {
            // Rule 6.
            let minimum = if kind == QuestionKind::MultiLabel {
                1
            } else {
                2
            };
            let count = match criteria_value {
                Some(Value::Array(items)) => items.len(),
                Some(Value::Object(map)) => map.len(),
                _ => 0,
            };
            if count < minimum {
                return Err(invalid(format!(
                    "{} needs at least {minimum} criteria",
                    kind.as_str()
                )));
            }
            if kind == QuestionKind::Score {
                match criteria_value {
                    Some(Value::Array(items)) => {
                        Criteria::Levels(items.iter().map(describe).collect())
                    }
                    _ => return Err(invalid("score criteria must be an ordered list")),
                }
            } else {
                Criteria::Labels(labels(criteria_value)?)
            }
        }
        QuestionKind::Bool => {
            // Rule 8.
            match criteria_value {
                None => Criteria::Bool {
                    when_false: None,
                    when_true: None,
                },
                Some(Value::Object(map)) => Criteria::Bool {
                    when_false: map.get("false").and_then(optional_description),
                    when_true: map.get("true").and_then(optional_description),
                },
                Some(_) => {
                    return Err(invalid(
                        "bool criteria must map false and true to descriptions",
                    ));
                }
            }
        }
    };

    // Rule 7.
    let threshold = if kind == QuestionKind::MultiLabel {
        match raw.get("threshold") {
            None => DEFAULT_THRESHOLD,
            Some(Value::Number(number))
                if number
                    .as_f64()
                    .is_some_and(|value| (0.0..=1.0).contains(&value)) =>
            {
                number.as_f64().unwrap_or(DEFAULT_THRESHOLD)
            }
            Some(_) => {
                return Err(invalid("threshold must be a number between zero and one"));
            }
        }
    } else {
        DEFAULT_THRESHOLD
    };

    Ok(Question {
        kind,
        instructions: raw
            .get("instructions")
            .and_then(Value::as_str)
            .map(str::to_string),
        criteria,
        threshold,
        raw,
    })
}

/// `choice` / `multi_label` labels: non-empty strings, unique.
fn labels(criteria: Option<&Value>) -> Result<Vec<Label>, DecisionError> {
    let labels: Vec<Label> = match criteria {
        Some(Value::Array(items)) => {
            let mut labels = Vec::with_capacity(items.len());
            for item in items {
                match item.as_str() {
                    Some(name) if !name.is_empty() => labels.push(Label {
                        name: name.to_string(),
                        description: None,
                    }),
                    _ => return Err(invalid("Decision labels must be nonempty strings")),
                }
            }
            labels
        }
        Some(Value::Object(map)) => {
            if map.keys().any(String::is_empty) {
                return Err(invalid("Decision labels must be nonempty strings"));
            }
            map.iter()
                .map(|(name, description)| Label {
                    name: name.clone(),
                    description: optional_description(description),
                })
                .collect()
        }
        _ => Vec::new(),
    };
    let mut seen = std::collections::HashSet::with_capacity(labels.len());
    if !labels.iter().all(|label| seen.insert(label.name.as_str())) {
        return Err(invalid("Decision labels must be unique"));
    }
    Ok(labels)
}

/// A description value: `null` is none, a string is itself, anything else
/// is its JSON text.
fn optional_description(value: &Value) -> Option<String> {
    (!value.is_null()).then(|| describe(value))
}

fn describe(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Python's `repr` of a JSON value, for the "does not support X decisions"
/// message (upstream formats the type with `{kind!r}`).
pub(crate) fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => {
            let quote = if text.contains('\'') && !text.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut out = String::with_capacity(text.len() + 2);
            out.push(quote);
            for ch in text.chars() {
                match ch {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
        other => other.to_string(),
    }
}
