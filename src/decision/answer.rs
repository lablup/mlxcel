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

//! Typed answers and their wire shape.
//!
//! The constructors fill the fields every model reports the same way, so a
//! scorer passes only its probabilities and its own metrics:
//!
//! - `choice`: `value` is the argmax label, `probabilities` are keyed by label
//!   in request order, `metadata.confidence` is the argmax probability.
//! - `score`: `value` is the expected level index, `probabilities` are keyed
//!   `"0"` to `"n-1"`, `metadata.legend` maps each level to its description.
//! - `bool`: `value` is `probability >= 0.5` (also the answer to `noul`).
//! - `multi_label`: `value` lists the labels at or above the threshold and
//!   `scores` are independent, so they need not sum to one.
//!
//! Model-specific metrics go in `metadata` and are not comparable across
//! models.

use serde_json::{Map, Value, json};

use super::{DecisionError, Question};

/// One answer.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Answer {
    /// A `choice` answer.
    Choice {
        /// The argmax label.
        value: String,
        /// `(label, probability)` in request order.
        probabilities: Vec<(String, f64)>,
        /// `confidence` plus model-specific metrics.
        metadata: Map<String, Value>,
    },
    /// A `score` answer.
    Score {
        /// Expected level index.
        value: f64,
        /// Probability of level `i` at index `i`.
        probabilities: Vec<f64>,
        /// `legend` plus model-specific metrics.
        metadata: Map<String, Value>,
    },
    /// A `bool` (or `noul`) answer.
    Bool {
        /// Probability of `true`.
        probability: f64,
        /// Model-specific metrics; omitted from the JSON when empty.
        metadata: Map<String, Value>,
    },
    /// A `multi_label` answer.
    MultiLabel {
        /// Labels at or above the threshold, in request order.
        value: Vec<String>,
        /// `(label, independent score)` in request order.
        scores: Vec<(String, f64)>,
        /// Model-specific metrics; omitted from the JSON when empty.
        metadata: Map<String, Value>,
    },
}

fn length_mismatch(what: &str, got: usize, want: usize) -> DecisionError {
    DecisionError::Internal(format!("{what}: {got} values for {want} criteria"))
}

impl Answer {
    /// A `choice` answer from one probability per label of `question`.
    pub fn choice(
        question: &Question,
        probabilities: Vec<f64>,
        mut metadata: Map<String, Value>,
    ) -> Result<Self, DecisionError> {
        let labels = question.labels();
        if probabilities.len() != labels.len() || labels.is_empty() {
            return Err(length_mismatch("choice", probabilities.len(), labels.len()));
        }
        let best = argmax(&probabilities);
        metadata
            .entry("confidence")
            .or_insert_with(|| json!(probabilities[best]));
        Ok(Answer::Choice {
            value: labels[best].name.clone(),
            probabilities: labels
                .iter()
                .map(|label| label.name.clone())
                .zip(probabilities)
                .collect(),
            metadata,
        })
    }

    /// A `score` answer from one probability per level of `question`.
    pub fn score(
        question: &Question,
        probabilities: Vec<f64>,
        mut metadata: Map<String, Value>,
    ) -> Result<Self, DecisionError> {
        let levels = question.levels();
        if probabilities.len() != levels.len() || levels.is_empty() {
            return Err(length_mismatch("score", probabilities.len(), levels.len()));
        }
        let value = probabilities
            .iter()
            .enumerate()
            .map(|(index, p)| index as f64 * p)
            .sum();
        let legend: Map<String, Value> = levels
            .iter()
            .enumerate()
            .map(|(index, level)| (index.to_string(), Value::String(level.clone())))
            .collect();
        metadata
            .entry("legend")
            .or_insert_with(|| Value::Object(legend));
        Ok(Answer::Score {
            value,
            probabilities,
            metadata,
        })
    }

    /// A `bool` answer from the probability of `true`.
    #[must_use]
    pub fn bool(probability: f64, metadata: Map<String, Value>) -> Self {
        Answer::Bool {
            probability,
            metadata,
        }
    }

    /// A `multi_label` answer from one independent score per label.
    pub fn multi_label(
        question: &Question,
        scores: Vec<f64>,
        metadata: Map<String, Value>,
    ) -> Result<Self, DecisionError> {
        let labels = question.labels();
        if scores.len() != labels.len() || labels.is_empty() {
            return Err(length_mismatch("multi_label", scores.len(), labels.len()));
        }
        let value = labels
            .iter()
            .zip(&scores)
            .filter(|(_, score)| **score >= question.threshold)
            .map(|(label, _)| label.name.clone())
            .collect();
        Ok(Answer::MultiLabel {
            value,
            scores: labels
                .iter()
                .map(|label| label.name.clone())
                .zip(scores)
                .collect(),
            metadata,
        })
    }

    /// Wire type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Answer::Choice { .. } => "choice",
            Answer::Score { .. } => "score",
            Answer::Bool { .. } => "bool",
            Answer::MultiLabel { .. } => "multi_label",
        }
    }

    /// Whether every number in the answer is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        match self {
            Answer::Choice { probabilities, .. } => {
                probabilities.iter().all(|(_, p)| p.is_finite())
            }
            Answer::Score {
                value,
                probabilities,
                ..
            } => value.is_finite() && probabilities.iter().all(|p| p.is_finite()),
            Answer::Bool { probability, .. } => probability.is_finite(),
            Answer::MultiLabel { scores, .. } => scores.iter().all(|(_, s)| s.is_finite()),
        }
    }

    /// The answer object as sent on the wire.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("type".to_string(), json!(self.type_name()));
        match self {
            Answer::Choice {
                value,
                probabilities,
                metadata,
            } => {
                out.insert("value".to_string(), json!(value));
                out.insert("probabilities".to_string(), keyed(probabilities));
                out.insert("metadata".to_string(), Value::Object(metadata.clone()));
            }
            Answer::Score {
                value,
                probabilities,
                metadata,
            } => {
                out.insert("value".to_string(), json!(value));
                let indexed: Map<String, Value> = probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, p)| (index.to_string(), json!(p)))
                    .collect();
                out.insert("probabilities".to_string(), Value::Object(indexed));
                out.insert("metadata".to_string(), Value::Object(metadata.clone()));
            }
            Answer::Bool {
                probability,
                metadata,
            } => {
                out.insert("value".to_string(), json!(*probability >= 0.5));
                out.insert("probability".to_string(), json!(probability));
                if !metadata.is_empty() {
                    out.insert("metadata".to_string(), Value::Object(metadata.clone()));
                }
            }
            Answer::MultiLabel {
                value,
                scores,
                metadata,
            } => {
                out.insert("value".to_string(), json!(value));
                out.insert("scores".to_string(), keyed(scores));
                if !metadata.is_empty() {
                    out.insert("metadata".to_string(), Value::Object(metadata.clone()));
                }
            }
        }
        Value::Object(out)
    }
}

/// `(label, number)` pairs as a JSON object in pair order.
fn keyed(pairs: &[(String, f64)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(name, value)| (name.clone(), json!(value)))
            .collect(),
    )
}

/// Index of the first maximum.
fn argmax(values: &[f64]) -> usize {
    values.iter().enumerate().fold(
        0,
        |best, (index, value)| {
            if *value > values[best] { index } else { best }
        },
    )
}
