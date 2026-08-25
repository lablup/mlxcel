// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Reranker-model provider interface.
//!
//! The transport-agnostic seam between `POST /v1/rerank` and the rerank
//! worker, mirroring [`super::embedding_model`]. The route only sees plain
//! `f32` scores and a token count; prompt assembly, tokenization, batching and
//! the forward pass happen behind this trait on the worker thread. Until a
//! provider is registered the [`AppState`](crate::server::AppState) slot stays
//! `None` and the route returns a structured `501 Not Implemented`.

use thiserror::Error;

pub use crate::rerank::{ImageInput, RerankItem, RerankScores, RerankerKind};

/// Failure modes of one rerank request.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RerankError {
    /// The bounded worker queue was full at admission time. Routes map this to
    /// the shared `503` "all slots are busy" envelope.
    #[error("rerank worker queue is full")]
    QueueFull,
    /// The worker did not reply within the per-request timeout. Routes map this
    /// to `504`; the in-flight MLX work is not cancelled.
    #[error("rerank request timed out")]
    Timeout,
    /// The request content is invalid (empty query or document, an image for a
    /// text-only reranker). Routes map this to `400 invalid_request_error`.
    #[error("{0}")]
    InvalidInput(String),
    /// The model failed while scoring the request.
    #[error("rerank inference failed: {0}")]
    Internal(String),
}

/// Provider-facing interface implemented by the rerank worker.
///
/// The call is synchronous and blocking (it waits for the worker's reply), so
/// routes invoke it inside `spawn_blocking`.
pub trait RerankModelProvider: Send + Sync {
    /// Score every document against `query`, in document order.
    fn rerank(
        &self,
        query: RerankItem,
        documents: Vec<RerankItem>,
        instruction: Option<String>,
    ) -> Result<RerankScores, RerankError>;

    /// Served model id, reported in responses and `/v1/models`.
    fn model_id(&self) -> &str;

    /// Unix timestamp the model was loaded at, for `/v1/models`.
    fn created_at(&self) -> i64;

    /// Which scoring recipe the loaded checkpoint uses.
    fn kind(&self) -> RerankerKind;

    /// Whether image items are accepted.
    fn supports_images(&self) -> bool {
        false
    }

    /// Token cap on one scored pair.
    fn max_length(&self) -> usize;
}
