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

//! Test-only decision model: deterministic answers, no MLX and no checkpoint.
//!
//! It declares `choice`, `score` and `bool` and reads `images` only, which is
//! what the media-gate and type-gate tests need. Answers depend on the state
//! text so the assertions mean something:
//!
//! - `choice`: a label that appears in the state gets weight 3, every other
//!   label weight 1, normalized.
//! - `score`: uniform over the levels.
//! - `bool`: 0.9 when the state contains "yes" or images were sent, else 0.2.
//!
//! Every answer's `metadata.images` reports how many images reached
//! `predict`, and `input_tokens` is the state's word count.

use std::path::Path;

use serde_json::{Map, Value, json};

use super::loader::{DecisionKind, LoadedDecisionModel};
use super::{
    Answer, DecisionError, DecisionMedia, DecisionModel, DecisionResult, DecisionUsage, MediaKind,
    Question, QuestionKind,
};

/// `config.json` `model_type` the loader detects as the stub.
pub(crate) const STUB_MODEL_TYPE: &str = "mlxcel_decision_stub";

const STUB_TYPES: &[QuestionKind] = &[
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::Bool,
];
const STUB_MEDIA: &[MediaKind] = &[MediaKind::Image];

pub(crate) struct StubDecisionModel;

impl StubDecisionModel {
    pub(crate) fn load(_model_dir: &Path) -> Self {
        Self
    }
}

fn state_text(state: Option<&Value>) -> String {
    match state {
        Some(Value::String(text)) => text.to_lowercase(),
        Some(other) => other.to_string().to_lowercase(),
        None => String::new(),
    }
}

impl DecisionModel for StubDecisionModel {
    fn decision_types(&self) -> &[QuestionKind] {
        STUB_TYPES
    }

    fn decision_media(&self) -> &[MediaKind] {
        STUB_MEDIA
    }

    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError> {
        let text = state_text(state);
        let images = media.images.len();
        let mut answers = Vec::with_capacity(questions.len());
        for (name, question) in questions {
            let mut metadata = Map::new();
            metadata.insert("images".to_string(), json!(images));
            let answer = match question.kind {
                QuestionKind::Choice => {
                    let weights: Vec<f64> = question
                        .labels()
                        .iter()
                        .map(|label| {
                            if text.contains(&label.name.to_lowercase()) {
                                3.0
                            } else {
                                1.0
                            }
                        })
                        .collect();
                    let total: f64 = weights.iter().sum();
                    Answer::choice(
                        question,
                        weights.iter().map(|w| w / total).collect(),
                        metadata,
                    )?
                }
                QuestionKind::Score => {
                    let n = question.levels().len();
                    Answer::score(question, vec![1.0 / n as f64; n], metadata)?
                }
                QuestionKind::Bool => {
                    let p = if text.contains("yes") || images > 0 {
                        0.9
                    } else {
                        0.2
                    };
                    Answer::bool(p, metadata)
                }
                other => {
                    return Err(DecisionError::Internal(format!(
                        "stub does not answer {}",
                        other.as_str()
                    )));
                }
            };
            answers.push((name.clone(), answer));
        }
        Ok(DecisionResult {
            answers,
            usage: DecisionUsage {
                input_tokens: text.split_whitespace().count(),
            },
        })
    }
}

/// A fully assembled stub for the worker and route tests.
pub(crate) fn stub_loaded_decision_model() -> LoadedDecisionModel {
    LoadedDecisionModel {
        model: Box::new(StubDecisionModel),
        kind: DecisionKind::Stub,
        model_type: STUB_MODEL_TYPE.to_string(),
    }
}
