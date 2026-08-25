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

//! Test-only embedding model: mean-pooled one-hot token embeddings.
//!
//! The stub lets the engine, the worker and the `/v1/embeddings` route run
//! end to end without a checkpoint. Each token id `t < dim` maps to the unit
//! vector `e_t`; a text's embedding is the mask-weighted mean of its
//! one-hot rows, L2-normalized by the engine. Two texts that share tokens
//! therefore have a positive cosine similarity and disjoint texts have zero,
//! which the tests assert on.

use anyhow::{Result, bail};
use mlxcel_core::{UniquePtr, dtype};

use crate::models::ModelType;

use super::limits::EmbeddingLimits;
use super::loader::LoadedEmbeddingModel;
use super::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use super::pooling::{PoolingMode, pool};
use super::tokenize_tests::bert_like_tokenizer;

/// Width of the stub vectors. Ids at or above it (the `[CLS]` / `[SEP]`
/// ids of the test tokenizer) map to zero rows.
pub(crate) const STUB_DIM: usize = 16;

/// Vocabulary size the stub validates token-id inputs against.
pub(crate) const STUB_VOCAB_SIZE: usize = 128;

/// Token cap of the stub model.
pub(crate) const STUB_MAX_LENGTH: usize = 32;

pub(crate) struct StubEmbeddingModel {
    dim: usize,
    pooling: PoolingMode,
    multi_vector: bool,
}

impl StubEmbeddingModel {
    pub(crate) fn new() -> Self {
        Self {
            dim: STUB_DIM,
            pooling: PoolingMode::Mean,
            multi_vector: false,
        }
    }

    /// Variant that returns the `[B, L, D]` one-hot matrix itself (padding
    /// rows zeroed), for the multi-vector route tests.
    pub(crate) fn multi_vector() -> Self {
        Self {
            dim: STUB_DIM,
            pooling: PoolingMode::Mean,
            multi_vector: true,
        }
    }

    /// `[B, L, D]` one-hot rows for ids below `dim`; other ids give zeros.
    fn one_hot(&self, input_ids: &mlxcel_core::MlxArray) -> UniquePtr<mlxcel_core::MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let (b, l) = (shape[0], shape[1]);
        let ids = mlxcel_core::reshape(input_ids, &[b, l, 1]);
        let classes = mlxcel_core::reshape(
            &mlxcel_core::arange_i32(0, self.dim as i32, 1),
            &[1, 1, self.dim as i32],
        );
        mlxcel_core::astype(&mlxcel_core::equal(&ids, &classes), dtype::FLOAT32)
    }
}

impl EmbeddingModel for StubEmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("stub embedding model does not accept images");
        }
        let hidden = self.one_hot(batch.input_ids);
        let embeddings = if self.multi_vector {
            let shape = mlxcel_core::array_shape(&hidden);
            let mask = mlxcel_core::astype(
                &mlxcel_core::reshape(batch.attention_mask, &[shape[0], shape[1], 1]),
                dtype::FLOAT32,
            );
            mlxcel_core::multiply(&hidden, &mask)
        } else {
            pool(&hidden, batch.attention_mask, self.pooling)
        };
        Ok(EmbeddingOutput {
            embeddings,
            last_hidden_state: Some(hidden),
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        self.pooling
    }

    fn multi_vector(&self) -> bool {
        self.multi_vector
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// A fully assembled stub model over the BERT-shaped word-level test
/// tokenizer (`hello` = 3, `world` = 4, `[CLS]` = 101, `[SEP]` = 102).
pub(crate) fn stub_loaded_model(multi_vector: bool) -> LoadedEmbeddingModel {
    let model: Box<dyn EmbeddingModel> = if multi_vector {
        Box::new(StubEmbeddingModel::multi_vector())
    } else {
        Box::new(StubEmbeddingModel::new())
    };
    LoadedEmbeddingModel {
        limits: EmbeddingLimits {
            max_length: STUB_MAX_LENGTH,
            dim: STUB_DIM,
            multi_vector,
        },
        model,
        tokenizer: bert_like_tokenizer(false),
        pad_token_id: 0,
        vocab_size: STUB_VOCAB_SIZE,
        model_type: ModelType::Bert,
    }
}
