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

//! Embedding-model provider interface.
//!
//! The transport-agnostic seam between `POST /v1/embeddings` and the
//! embedding worker, mirroring [`super::audio_model`]. The route only sees
//! plain `Vec<f32>` vectors and token counts; tokenization, batching and
//! pooling happen behind this trait on the worker thread. Until a provider
//! is registered the [`AppState`](crate::server::AppState) slot stays `None`
//! and the route returns a structured `501 Not Implemented`.

use thiserror::Error;

use crate::embeddings::EmbeddingEngineError;
pub use crate::embeddings::{EmbedOptions, EmbedReply, EmbeddingVector, ImageInput};

/// Failure modes of one embedding request.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EmbeddingError {
    /// The bounded worker queue was full at admission time. Routes map this
    /// to the shared `503` "all slots are busy" envelope.
    #[error("embedding worker queue is full")]
    QueueFull,
    /// The worker did not reply within the per-request timeout. Routes map
    /// this to `504`; the in-flight MLX work is not cancelled.
    #[error("embedding request timed out")]
    Timeout,
    /// The request content is invalid (empty item, token id out of range,
    /// `dimensions` out of range, image for a text-only model). Routes map
    /// this to `400 invalid_request_error`.
    #[error("{0}")]
    InvalidInput(String),
    /// The model failed while processing the request.
    #[error("embedding inference failed: {0}")]
    Internal(String),
}

impl From<EmbeddingEngineError> for EmbeddingError {
    fn from(err: EmbeddingEngineError) -> Self {
        match err {
            EmbeddingEngineError::InvalidInput(message) => EmbeddingError::InvalidInput(message),
            EmbeddingEngineError::Internal(message) => EmbeddingError::Internal(message),
        }
    }
}

/// Provider-facing interface implemented by the embedding worker.
///
/// Every method is synchronous and blocking (it waits for the worker's
/// reply), so routes call it inside `spawn_blocking`.
pub trait EmbeddingModelProvider: Send + Sync {
    /// Embed texts (special tokens added, per-family formatting applied).
    fn embed_texts(
        &self,
        texts: Vec<String>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError>;

    /// Embed verbatim token-id rows (no special tokens added).
    fn embed_tokens(
        &self,
        token_rows: Vec<Vec<u32>>,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError>;

    /// Embed one image (VLM embedders only).
    fn embed_image(
        &self,
        image: ImageInput,
        opts: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError>;

    /// Served model id, reported in responses and `/v1/models`.
    fn model_id(&self) -> &str;

    /// Unix timestamp the model was loaded at, for `/v1/models`.
    fn created_at(&self) -> i64;

    /// Width `D` of one vector.
    fn dim(&self) -> usize;

    /// Whether outputs are `[num_real_tokens, D]` matrices.
    fn multi_vector(&self) -> bool;

    /// Whether `image_url` items are accepted.
    fn supports_images(&self) -> bool {
        false
    }

    /// Vocabulary size token-id inputs must stay below.
    fn vocab_size(&self) -> usize;

    /// Token cap per input.
    fn max_length(&self) -> usize;
}
