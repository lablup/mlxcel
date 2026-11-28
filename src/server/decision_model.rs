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

//! Decision-model provider interface.
//!
//! The seam between `POST /v1/decisions` and the decision worker, mirroring
//! [`super::rerank_model`]. The route validates the request against the
//! declarations exposed here, acquires the media, and hands typed questions to
//! the provider; scoring runs on the worker thread. Until `--decision-model`
//! loads a checkpoint the [`AppState`](crate::server::AppState) slot stays
//! `None` and the route returns a structured `501 Not Implemented`.

use serde_json::Value;
use thiserror::Error;

use crate::multimodal::video::PrivateTempDir;

pub use crate::decision::{DecisionMedia, DecisionResult, MediaKind, Question, QuestionKind};

/// Failure modes of one decision request.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecisionProviderError {
    /// The bounded worker queue was full at admission time (`503`).
    #[error("decision worker queue is full")]
    QueueFull,
    /// The worker did not reply within the per-request timeout (`504`). A
    /// request still queued is skipped; one already running finishes and its
    /// answer is discarded.
    #[error("decision request timed out")]
    Timeout,
    /// The request is invalid for the loaded model (`400`).
    #[error("{0}")]
    InvalidInput(String),
    /// The model failed (`500`, with a generic detail).
    #[error("decision prediction failed: {0}")]
    Internal(String),
}

impl From<crate::decision::DecisionError> for DecisionProviderError {
    fn from(err: crate::decision::DecisionError) -> Self {
        match err {
            crate::decision::DecisionError::InvalidInput(message) => Self::InvalidInput(message),
            crate::decision::DecisionError::Internal(message) => Self::Internal(message),
        }
    }
}

/// Provider-facing interface implemented by the decision worker.
///
/// The call is synchronous and blocking, so routes invoke it inside
/// `spawn_blocking`.
pub trait DecisionModelProvider: Send + Sync {
    /// Answer validated questions about `state` and `media`.
    ///
    /// `temp_files` hold the server-owned files behind `media` (downloaded
    /// videos). The provider keeps them until the model has stopped reading
    /// them, which can be after this call returned `Timeout`.
    fn predict(
        &self,
        state: Option<Value>,
        questions: Vec<(String, Question)>,
        media: DecisionMedia,
        temp_files: Vec<PrivateTempDir>,
    ) -> Result<DecisionResult, DecisionProviderError>;

    /// Served model id, reported in responses and `/v1/models`.
    fn model_id(&self) -> &str;

    /// Unix timestamp the model was loaded at, for `/v1/models`.
    fn created_at(&self) -> i64;

    /// Question types the loaded model answers.
    fn decision_types(&self) -> &[QuestionKind];

    /// Media the loaded model reads.
    fn decision_media(&self) -> &[MediaKind];

    /// Ask the worker to exit and wait until it has released the model, up to
    /// `timeout`. Called by the process exit path (#2738, #2745). Idempotent.
    fn shutdown_and_wait(&self, timeout: std::time::Duration) -> anyhow::Result<()>;
}
