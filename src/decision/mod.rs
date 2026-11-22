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

//! Typed decision models behind `POST /v1/decisions` and `mlxcel decide`.
//!
//! A decision model answers named questions about a state (text, any JSON
//! value, or media) with calibrated probabilities instead of generated text.
//! The contract is mlx-vlm's shared decision API (`mlx_vlm/decision.py`,
//! `mlx_vlm/decide.py`, `mlx_vlm/server/decisions.py`): every question has a
//! type (`choice`, `score`, `bool`, `multi_label`, with `noul` an alias of
//! `bool`), and each model declares the types and media it reads.
//!
//! Layout:
//!
//! - this module: the question and answer types, the [`DecisionModel`] trait
//!   and [`predict`], the one entry point that validates before scoring.
//! - [`validate`]: rules 1 to 8 of the contract, with upstream's messages.
//! - [`answer`]: answer constructors and the response JSON.
//! - [`loader`]: [`load_decision_model`], the kind dispatcher.
//! - [`cli`]: `mlxcel decide` input preparation (pure, testable before load).
//! - [`media`]: audio decoding to 16 kHz mono and CLI media loading.
//!
//! # Adding a decision model
//!
//! A model port touches three places and nothing else:
//!
//! 1. A [`loader::DecisionKind`] variant for the family.
//! 2. A detection arm in [`loader::detect_decision_kind`] that recognizes the
//!    checkpoint from `config.json`, plus a constructor arm in
//!    [`loader::load_decision_model`].
//! 3. A scorer: a type implementing [`DecisionModel`]. It declares
//!    [`DecisionModel::decision_types`] and [`DecisionModel::decision_media`],
//!    and its [`DecisionModel::predict`] receives questions that already
//!    passed [`validate::validate_questions`]. Model-specific limits (option
//!    counts, level counts, media exclusivity) are the scorer's to check and
//!    are reported as [`DecisionError::InvalidInput`], which surfaces as a 400.
//!    Build answers with the [`Answer`] constructors so the shared fields
//!    (`confidence`, `legend`, the bool `value`) are filled the same way for
//!    every model.
//!
//! The server worker, route, CLI and `/v1/models` listing need no change.
//!
//! Everything a scorer holds runs on one thread: the server's decision worker
//! thread or the `mlxcel decide` main thread.

pub mod answer;
pub mod cli;
pub mod loader;
pub mod media;
pub mod validate;

#[cfg(test)]
pub(crate) mod stub;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

use serde_json::{Map, Value};
use thiserror::Error;

pub use answer::Answer;
pub use loader::{DecisionKind, LoadedDecisionModel, detect_decision_kind, load_decision_model};
pub use validate::{MediaPresence, validate_question_structure, validate_questions};

/// A question type, as declared by a model and requested by a question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuestionKind {
    /// One label out of several; probabilities sum to one.
    Choice,
    /// An ordered level; the value is the expected level index.
    Score,
    /// A yes/no question; `noul` requests parse to this kind too.
    Bool,
    /// Any subset of labels, scored independently against a threshold.
    MultiLabel,
}

impl QuestionKind {
    /// Every kind, in the order upstream lists them.
    pub const ALL: [QuestionKind; 4] = [
        QuestionKind::Choice,
        QuestionKind::Score,
        QuestionKind::Bool,
        QuestionKind::MultiLabel,
    ];

    /// Wire name of the kind; `noul` requests answer as `bool`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionKind::Choice => "choice",
            QuestionKind::Score => "score",
            QuestionKind::Bool => "bool",
            QuestionKind::MultiLabel => "multi_label",
        }
    }

    /// Parse a requested `type`; `noul` is an alias of `bool`.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "choice" => Some(QuestionKind::Choice),
            "score" => Some(QuestionKind::Score),
            "bool" | "noul" => Some(QuestionKind::Bool),
            "multi_label" => Some(QuestionKind::MultiLabel),
            _ => None,
        }
    }
}

/// A media input a model may read, as declared in `decision_media`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MediaKind {
    /// `images`: decoded still images.
    Image,
    /// `videos`: clips handed to the model as sources it decodes itself.
    Video,
    /// `audio`: one clip, 16 kHz mono samples.
    Audio,
}

impl MediaKind {
    /// Request field that carries this media (`images`, `videos`, `audio`).
    #[must_use]
    pub fn field(self) -> &'static str {
        match self {
            MediaKind::Image => "images",
            MediaKind::Video => "videos",
            MediaKind::Audio => "audio",
        }
    }

    /// Noun used in "This model does not support {noun} input".
    #[must_use]
    pub fn noun(self) -> &'static str {
        match self {
            MediaKind::Image => "image",
            MediaKind::Video => "video",
            MediaKind::Audio => "audio",
        }
    }
}

/// One `choice` or `multi_label` label, with its optional description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// The label returned in `value` and used as the probability key.
    pub name: String,
    /// The description from a `label -> description` map, if any.
    pub description: Option<String>,
}

/// A question's validated `criteria`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Criteria {
    /// `choice` and `multi_label`: labels in request order.
    Labels(Vec<Label>),
    /// `score`: level descriptions, level `i` at index `i`.
    Levels(Vec<String>),
    /// `bool`: optional descriptions of the false and true answers.
    Bool {
        /// Description of the `false` answer.
        when_false: Option<String>,
        /// Description of the `true` answer.
        when_true: Option<String>,
    },
}

/// One validated question.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    /// The question type (`noul` already folded into [`QuestionKind::Bool`]).
    pub kind: QuestionKind,
    /// `instructions`, when given as a string.
    pub instructions: Option<String>,
    /// The validated criteria for `kind`.
    pub criteria: Criteria,
    /// `multi_label` threshold in `[0, 1]` (0.5 when absent); 0.5 otherwise.
    pub threshold: f64,
    /// The question object as sent, for model-specific keys.
    pub raw: Map<String, Value>,
}

impl Question {
    /// Labels of a `choice` or `multi_label` question; empty otherwise.
    #[must_use]
    pub fn labels(&self) -> &[Label] {
        match &self.criteria {
            Criteria::Labels(labels) => labels,
            _ => &[],
        }
    }

    /// Levels of a `score` question; empty otherwise.
    #[must_use]
    pub fn levels(&self) -> &[String] {
        match &self.criteria {
            Criteria::Levels(levels) => levels,
            _ => &[],
        }
    }
}

/// Media handed to [`DecisionModel::predict`], already acquired and decoded.
#[derive(Debug, Default)]
pub struct DecisionMedia {
    /// Decoded images, in request order.
    pub images: Vec<image::DynamicImage>,
    /// Video sources, in request order. On the server these are fd-backed
    /// handles from the allowlist-guarded resolver, so a model decodes the
    /// file that was checked rather than reopening a path.
    pub videos: Vec<crate::multimodal::video::VideoSource>,
    /// One audio clip as 16 kHz mono samples.
    pub audio: Option<Vec<f32>>,
}

impl DecisionMedia {
    /// Which media fields carry something.
    #[must_use]
    pub fn presence(&self) -> MediaPresence {
        MediaPresence {
            images: !self.images.is_empty(),
            videos: !self.videos.is_empty(),
            audio: self.audio.is_some(),
        }
    }
}

/// Token accounting for one prediction. No tokens are generated, so the
/// response's `output_tokens` is always 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecisionUsage {
    /// Prompt tokens the model read.
    pub input_tokens: usize,
}

/// The answers to one request, in question order.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionResult {
    /// `(question name, answer)` pairs in request order.
    pub answers: Vec<(String, Answer)>,
    /// Token accounting.
    pub usage: DecisionUsage,
}

impl DecisionResult {
    /// The wire response: `{"model", "answers", "usage"}`.
    #[must_use]
    pub fn to_response_json(&self, model_id: &str) -> Value {
        let mut answers = Map::new();
        for (name, answer) in &self.answers {
            answers.insert(name.clone(), answer.to_json());
        }
        serde_json::json!({
            "model": model_id,
            "answers": answers,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": 0,
            },
        })
    }
}

/// Failure of one prediction.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecisionError {
    /// The request is invalid for this model: a contract rule, or a
    /// model-specific limit. Surfaces as a 400 with this message.
    #[error("{0}")]
    InvalidInput(String),
    /// The model failed while scoring. Logged; clients see a generic 500.
    #[error("decision prediction failed: {0}")]
    Internal(String),
}

/// A model that answers typed questions.
///
/// See the module docs for how a port registers one. `predict` never sees a
/// question that failed [`validate_questions`]: [`predict`] and the server
/// route both validate first.
pub trait DecisionModel: Send {
    /// Question types this model answers. Empty means the model does not
    /// support decision prediction at all.
    fn decision_types(&self) -> &[QuestionKind];

    /// Media this model reads. A request carrying any other media is refused
    /// before `predict` runs.
    fn decision_media(&self) -> &[MediaKind];

    /// Answer every question, in order.
    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError>;
}

/// Validate a request against `model`'s declarations, then score it.
///
/// `state` of `None` or JSON `null` means no state. Used by `mlxcel decide`;
/// the server route validates with the worker's declared types first (so it
/// can refuse media before fetching it) and then calls the model directly.
pub fn predict(
    model: &mut dyn DecisionModel,
    state: Option<&Value>,
    questions: &Value,
    media: DecisionMedia,
) -> Result<DecisionResult, DecisionError> {
    let state = state.filter(|value| !value.is_null());
    let validated = validate_questions(
        model.decision_types(),
        model.decision_media(),
        state,
        questions,
        media.presence(),
    )?;
    let result = model.predict(state, &validated, media)?;
    check_result(&validated, &result)?;
    Ok(result)
}

/// Reject a model result that does not answer exactly the asked questions or
/// carries a non-finite number. Both are model bugs, so they are internal.
pub fn check_result(
    questions: &[(String, Question)],
    result: &DecisionResult,
) -> Result<(), DecisionError> {
    let same_names = questions.len() == result.answers.len()
        && questions
            .iter()
            .zip(&result.answers)
            .all(|((asked, _), (answered, _))| asked == answered);
    if !same_names {
        return Err(DecisionError::Internal(
            "the model did not answer the questions it was asked, in order".to_string(),
        ));
    }
    if let Some((name, _)) = result
        .answers
        .iter()
        .find(|(_, answer)| !answer.is_finite())
    {
        return Err(DecisionError::Internal(format!(
            "the model returned a non-finite number for `{name}`"
        )));
    }
    Ok(())
}
