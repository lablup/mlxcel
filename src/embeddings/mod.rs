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

//! Shared foundation for embedding models served through `/v1/embeddings`
//! and `mlxcel embed`.
//!
//! - [`model`]: the [`EmbeddingModel`] trait and batch / output types.
//! - [`pooling`]: `cls` / `mean` / `max` / `lasttoken` pooling, the
//!   `1_Pooling/config.json` reader, L2 normalization and `dimensions`
//!   truncation.
//! - [`limits`]: `max_length` derivation, pad token and vocabulary size.
//! - [`tokenize`]: right-padded batch tokenization with trailing-special
//!   truncation and pair encoding.
//! - [`loader`]: `load_embedding_model`, the family dispatcher.
//! - [`engine`]: length-sorted micro-batching, normalization and readback.
//!
//! No family lives here; each family sub-issue adds its constructor to
//! [`loader`] and its module under `src/models/`.

pub mod engine;
pub mod limits;
pub mod loader;
pub mod model;
pub mod pooling;
pub mod tokenize;

#[cfg(test)]
pub(crate) mod stub;

#[cfg(test)]
#[path = "pooling_tests.rs"]
mod pooling_tests;

#[cfg(test)]
#[path = "tokenize_tests.rs"]
mod tokenize_tests;

#[cfg(test)]
#[path = "engine_tests.rs"]
mod engine_tests;

#[cfg(test)]
#[path = "loader_tests.rs"]
mod loader_tests;

#[cfg(test)]
#[path = "real_checkpoint_tests.rs"]
mod real_checkpoint_tests;

pub use engine::{
    DEFAULT_EMBEDDING_BATCH_SIZE, EmbedOptions, EmbedReply, EmbeddingEngine, EmbeddingEngineError,
    EmbeddingVector,
};
pub use limits::{EMBEDDING_MAX_LENGTH_CAP, EmbeddingLimits};
pub use loader::{
    EmbeddingLoadOptions, LoadedEmbeddingModel, load_embedding_model,
    load_embedding_model_with_options,
};
pub use model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput, ImageInput};
pub use pooling::{
    POOLING_ENV, PoolingConfig, PoolingMode, normalize_l2, pool, resolve_pooling_mode,
    truncate_dimensions,
};
