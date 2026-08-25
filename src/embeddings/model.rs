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

//! The `EmbeddingModel` trait and the batch / output types every embedding
//! family implements against.
//!
//! A family owns its forward pass and its pooling; the engine in
//! [`super::engine`] owns tokenization, micro-batching, normalization and
//! `dimensions` truncation. The trait therefore returns already-pooled
//! vectors (`[B, D]`), or the padded token matrix (`[B, L, D]`) for
//! multi-vector (late-interaction) families.

use anyhow::Result;
use mlxcel_core::{MlxArray, UniquePtr};

use super::pooling::PoolingMode;

/// One decoded image handed to a vision-language embedder.
///
/// The HTTP layer decodes and bounds the payload with the shared image
/// limits before the batch reaches the worker, so a family only sees a
/// ready-to-preprocess image.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Decoded pixels.
    pub image: image::DynamicImage,
}

/// One right-padded micro-batch presented to [`EmbeddingModel::embed`].
///
/// All arrays are `[B, L]` int32. `attention_mask` is `1` for a real token
/// and `0` for padding; the family builds its attention mask from it with
/// `mlxcel_core::utils::create_bidirectional_padding_mask` or
/// `create_causal_padding_mask`.
pub struct EmbeddingBatch<'a> {
    /// `[B, L]` int32, right-padded with the checkpoint's pad token id.
    pub input_ids: &'a MlxArray,
    /// `[B, L]` int32, `1` = real token, `0` = padding.
    pub attention_mask: &'a MlxArray,
    /// `[B, L]` int32 segment ids; `Some` only when the family asks for them
    /// through [`EmbeddingModel::needs_token_type_ids`] (BERT pairs).
    pub token_type_ids: Option<&'a MlxArray>,
    /// Images for VLM embedders; `None` for text-only batches.
    pub images: Option<&'a [ImageInput]>,
}

/// Result of one forward pass.
pub struct EmbeddingOutput {
    /// `[B, D]` for single-vector models; `[B, L, D]` with padding rows
    /// zeroed for multi-vector models.
    pub embeddings: UniquePtr<MlxArray>,
    /// The `[B, L, D]` hidden states before pooling, when the family keeps
    /// them (debugging and the reranker path).
    pub last_hidden_state: Option<UniquePtr<MlxArray>>,
}

/// A loaded embedding family.
///
/// Implementors hold MLX array handles directly and are used from exactly
/// one thread (the embedding worker thread or the `mlxcel embed` main
/// thread), so the trait requires neither `Send` nor `Sync`.
pub trait EmbeddingModel {
    /// Run the forward pass on one padded micro-batch.
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput>;

    /// Pooling the family applies when the checkpoint ships no
    /// `1_Pooling/config.json`. Families resolve the effective mode at load
    /// time through [`super::pooling::resolve_pooling_mode`].
    fn default_pooling(&self) -> PoolingMode;

    /// Whether the engine L2-normalizes the pooled vectors (default `true`;
    /// `config.json` `normalize: false` turns it off for families that
    /// declare it).
    fn normalize(&self) -> bool {
        true
    }

    /// `true` for late-interaction (ColBERT-style) families whose
    /// [`EmbeddingOutput::embeddings`] is `[B, L, D]`.
    fn multi_vector(&self) -> bool {
        false
    }

    /// Width `D` of one embedding vector.
    fn embedding_dim(&self) -> usize;

    /// Whether `image_url` items are accepted for this family.
    fn supports_images(&self) -> bool {
        false
    }

    /// Whether the batch must carry `token_type_ids` (BERT pairs).
    fn needs_token_type_ids(&self) -> bool {
        false
    }

    /// Per-family text formatting (chat-template wrapping for
    /// Qwen3-VL-Embedding, an instruction prefix for Qwen3-Embedding,
    /// identity otherwise).
    fn format_text(&self, text: &str, _instruction: Option<&str>) -> String {
        text.to_string()
    }

    /// Fixed sequence width the family requires (SigLIP text pads to exactly
    /// 64); `None` pads each micro-batch to its longest member.
    fn pad_to_max_length(&self) -> Option<usize> {
        None
    }

    /// Hard token cap the loaded weights impose, lowering the `max_length`
    /// derived from the checkpoint's side files.
    ///
    /// Absolute position tables are the case that needs it: XLM-RoBERTa
    /// indexes its table from `pad_token_id + 1`, so `bge-m3`'s 8194 rows
    /// address only 8192 real tokens and a config-derived 8194 would gather
    /// out of bounds. `None` means the derived limit already holds.
    fn max_sequence_length(&self) -> Option<usize> {
        None
    }
}
