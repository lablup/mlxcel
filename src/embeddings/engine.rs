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

//! The embedding engine: tokenization, length-sorted micro-batching, the
//! family forward pass, normalization and `dimensions` truncation.
//!
//! One engine wraps one [`LoadedEmbeddingModel`] and is driven from a single
//! thread: the server's embedding worker thread or the `mlxcel embed` main
//! thread. Everything MLX-related stays inside; callers only see plain
//! `Vec<f32>` vectors and token counts.

use mlxcel_core::{MlxArray, UniquePtr, dtype, utils::array_to_vec_f32};
use thiserror::Error;

use crate::models::ModelType;

use super::loader::LoadedEmbeddingModel;
use super::model::{EmbeddingBatch, EmbeddingOutput, ImageInput};
use super::pooling::{normalize_l2, truncate_dimensions};
use super::tokenize::{EncodeOptions, EncodedBatch, EncodedRow, encode_row, truncate_token_ids};

/// Default `--embedding-batch-size`: texts per forward pass.
pub const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 16;

/// Failure modes of one engine call.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EmbeddingEngineError {
    /// The caller's input is invalid (empty text, token id out of range,
    /// `dimensions` out of range, image for a text-only family). Routes map
    /// this to `400`.
    #[error("{0}")]
    InvalidInput(String),
    /// The forward pass or a post-processing step failed.
    #[error("{0}")]
    Internal(String),
}

/// Per-request options forwarded to the engine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbedOptions {
    /// Forwarded to `EmbeddingModel::format_text`.
    pub instruction: Option<String>,
    /// Keep only the first `dimensions` components (re-normalized when the
    /// family normalizes). Validated against the model width.
    pub dimensions: Option<usize>,
}

/// One embedding result: `shape == [D]` for single-vector models,
/// `[num_real_tokens, D]` for multi-vector models. `values` is row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVector {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

impl EmbeddingVector {
    /// `true` when this is a `[n, D]` token matrix rather than one vector.
    #[must_use]
    pub fn is_multi_vector(&self) -> bool {
        self.shape.len() == 2
    }

    /// Row-major rows of width `D` (a single row for single-vector output).
    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        let width = self
            .shape
            .last()
            .copied()
            .unwrap_or(self.values.len())
            .max(1);
        self.values.chunks(width)
    }
}

/// Result of one engine call, in the caller's input order.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedReply {
    pub vectors: Vec<EmbeddingVector>,
    /// Real (non-padding) tokens across all inputs, special tokens included.
    pub prompt_tokens: usize,
}

/// Tokenize, batch, embed and post-process on the calling thread.
pub struct EmbeddingEngine {
    loaded: LoadedEmbeddingModel,
    batch_size: usize,
}

impl EmbeddingEngine {
    /// Wrap a loaded model. `batch_size` is clamped to at least one.
    pub fn new(loaded: LoadedEmbeddingModel, batch_size: usize) -> Self {
        Self {
            loaded,
            batch_size: batch_size.max(1),
        }
    }

    /// Width `D` of one vector.
    pub fn dim(&self) -> usize {
        self.loaded.limits.dim
    }

    /// Token cap per input.
    pub fn max_length(&self) -> usize {
        self.loaded.limits.max_length
    }

    /// Vocabulary size token-id inputs are validated against.
    pub fn vocab_size(&self) -> usize {
        self.loaded.vocab_size
    }

    /// Whether outputs are `[num_real_tokens, D]` matrices.
    pub fn multi_vector(&self) -> bool {
        self.loaded.limits.multi_vector
    }

    /// Whether `image_url` items are accepted.
    pub fn supports_images(&self) -> bool {
        self.loaded.model.supports_images()
    }

    /// Detected family.
    pub fn model_type(&self) -> ModelType {
        self.loaded.model_type
    }

    /// Texts per forward pass.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Reject a `dimensions` value outside `1..=D`.
    pub fn validate_dimensions(
        &self,
        dimensions: Option<usize>,
    ) -> Result<(), EmbeddingEngineError> {
        match dimensions {
            Some(n) if n == 0 || n > self.dim() => {
                Err(EmbeddingEngineError::InvalidInput(format!(
                    "`dimensions` must be between 1 and {} for this model, got {n}",
                    self.dim()
                )))
            }
            _ => Ok(()),
        }
    }

    /// Embed texts (special tokens added, per-family formatting applied).
    pub fn embed_texts(
        &self,
        texts: &[String],
        options: &EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingEngineError> {
        self.validate_dimensions(options.dimensions)?;
        let encode = EncodeOptions {
            add_special_tokens: true,
            max_length: self.max_length(),
            with_token_type_ids: self.loaded.model.needs_token_type_ids(),
        };
        let mut rows = Vec::with_capacity(texts.len());
        for (index, text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(EmbeddingEngineError::InvalidInput(format!(
                    "input[{index}] is an empty string"
                )));
            }
            let formatted = self
                .loaded
                .model
                .format_text(text, options.instruction.as_deref());
            let row = encode_row(&self.loaded.tokenizer, &formatted, encode)
                .map_err(|e| EmbeddingEngineError::Internal(e.to_string()))?;
            rows.push(row);
        }
        self.run_rows(rows, None, options.dimensions)
    }

    /// Embed verbatim token-id rows (no special tokens added).
    pub fn embed_tokens(
        &self,
        token_rows: &[Vec<u32>],
        options: &EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingEngineError> {
        self.validate_dimensions(options.dimensions)?;
        let vocab_size = self.vocab_size();
        let needs_type_ids = self.loaded.model.needs_token_type_ids();
        let mut rows = Vec::with_capacity(token_rows.len());
        for (index, ids) in token_rows.iter().enumerate() {
            if ids.is_empty() {
                return Err(EmbeddingEngineError::InvalidInput(format!(
                    "input[{index}] is an empty token list"
                )));
            }
            if vocab_size > 0
                && let Some(bad) = ids.iter().find(|&&id| id as usize >= vocab_size)
            {
                return Err(EmbeddingEngineError::InvalidInput(format!(
                    "input[{index}] contains token id {bad}, which is >= vocab_size {vocab_size}"
                )));
            }
            let ids = truncate_token_ids(ids, self.max_length());
            let type_ids = needs_type_ids.then(|| vec![0; ids.len()]);
            rows.push(EncodedRow { ids, type_ids });
        }
        self.run_rows(rows, None, options.dimensions)
    }

    /// Embed one image (VLM embedders only). The text side of the batch is
    /// the family's formatting of an empty text plus the instruction.
    pub fn embed_image(
        &self,
        image: ImageInput,
        options: &EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingEngineError> {
        self.validate_dimensions(options.dimensions)?;
        if !self.supports_images() {
            return Err(EmbeddingEngineError::InvalidInput(
                "this embedding model does not accept image inputs".to_string(),
            ));
        }
        let encode = EncodeOptions {
            add_special_tokens: true,
            max_length: self.max_length(),
            with_token_type_ids: self.loaded.model.needs_token_type_ids(),
        };
        let formatted = self
            .loaded
            .model
            .format_text("", options.instruction.as_deref());
        let row = encode_row(&self.loaded.tokenizer, &formatted, encode)
            .map_err(|e| EmbeddingEngineError::Internal(e.to_string()))?;
        let images = [image];
        // Expand the prompt's image placeholder into the run of image tokens
        // the forward pass consumes, so `usage.prompt_tokens` and the row
        // count of a multi-vector output describe the same sequence.
        let ids = self
            .loaded
            .model
            .expand_image_tokens(&row.ids, &images)
            .map_err(|e| EmbeddingEngineError::Internal(e.to_string()))?;
        if ids.is_empty() {
            return Err(EmbeddingEngineError::Internal(
                "image prompt expansion produced no tokens".to_string(),
            ));
        }
        let row = EncodedRow {
            type_ids: row.type_ids.map(|_| vec![0; ids.len()]),
            ids,
        };
        self.run_rows(vec![row], Some(&images), options.dimensions)
    }

    /// Sort rows by length, cut them into micro-batches, run each, and write
    /// the results back in the caller's order.
    fn run_rows(
        &self,
        rows: Vec<EncodedRow>,
        images: Option<&[ImageInput]>,
        dimensions: Option<usize>,
    ) -> Result<EmbedReply, EmbeddingEngineError> {
        let prompt_tokens: usize = rows.iter().map(|r| r.ids.len()).sum();
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());

        let mut vectors: Vec<Option<EmbeddingVector>> = (0..rows.len()).map(|_| None).collect();
        for chunk in order.chunks(self.batch_size) {
            let chunk_rows: Vec<EncodedRow> = chunk.iter().map(|&i| rows[i].clone()).collect();
            let batch = EncodedBatch::from_rows(
                &chunk_rows,
                self.loaded.pad_token_id,
                self.loaded.model.pad_to_max_length(),
            );
            let produced = self.forward_batch(&batch, images, dimensions)?;
            for (&index, vector) in chunk.iter().zip(produced) {
                vectors[index] = Some(vector);
            }
        }

        let vectors = vectors
            .into_iter()
            .map(|v| {
                v.ok_or_else(|| {
                    EmbeddingEngineError::Internal("micro-batch produced no vector".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(EmbedReply {
            vectors,
            prompt_tokens,
        })
    }

    /// Run one padded micro-batch through the family and post-process.
    fn forward_batch(
        &self,
        batch: &EncodedBatch,
        images: Option<&[ImageInput]>,
        dimensions: Option<usize>,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingEngineError> {
        let input_ids = batch.input_ids_array();
        let attention_mask = batch.attention_mask_array();
        let token_type_ids = if self.loaded.model.needs_token_type_ids() {
            batch.token_type_ids_array()
        } else {
            None
        };
        let embed_batch = EmbeddingBatch {
            input_ids: &input_ids,
            attention_mask: &attention_mask,
            token_type_ids: token_type_ids.as_deref(),
            images,
        };
        let output = self.loaded.model.embed(&embed_batch).map_err(|e| {
            EmbeddingEngineError::Internal(format!("embedding forward failed: {e}"))
        })?;
        self.postprocess(output, &batch.token_counts, dimensions)
    }

    /// Normalize, truncate and read back one forward pass.
    fn postprocess(
        &self,
        output: EmbeddingOutput,
        token_counts: &[usize],
        dimensions: Option<usize>,
    ) -> Result<Vec<EmbeddingVector>, EmbeddingEngineError> {
        let normalize = self.loaded.model.normalize();
        let shape = mlxcel_core::array_shape(&output.embeddings);
        let batch = token_counts.len();
        let expected_rank = if self.multi_vector() { 3 } else { 2 };
        if shape.len() != expected_rank || shape[0] as usize != batch {
            return Err(EmbeddingEngineError::Internal(format!(
                "embedding output has shape {shape:?}, expected rank {expected_rank} with batch {batch}"
            )));
        }
        let width = *shape.last().unwrap_or(&0) as usize;
        if width != self.dim() {
            return Err(EmbeddingEngineError::Internal(format!(
                "embedding output width {width} does not match the model width {}",
                self.dim()
            )));
        }

        let mut processed: UniquePtr<MlxArray> =
            mlxcel_core::astype(&output.embeddings, dtype::FLOAT32);
        if normalize {
            processed = normalize_l2(&processed);
        }
        let out_width = match dimensions {
            Some(n) if n < width => {
                processed = truncate_dimensions(&processed, n);
                if normalize {
                    processed = normalize_l2(&processed);
                }
                n
            }
            _ => width,
        };
        mlxcel_core::try_eval(&processed)
            .map_err(|e| EmbeddingEngineError::Internal(format!("embedding eval failed: {e}")))?;
        let values = array_to_vec_f32(&processed);

        if self.multi_vector() {
            let padded_len = shape[1] as usize;
            let stride = padded_len * out_width;
            Ok(token_counts
                .iter()
                .enumerate()
                .map(|(b, &count)| {
                    let rows = count.min(padded_len);
                    let start = b * stride;
                    EmbeddingVector {
                        values: values[start..start + rows * out_width].to_vec(),
                        shape: vec![rows, out_width],
                    }
                })
                .collect())
        } else {
            Ok(values
                .chunks(out_width)
                .take(batch)
                .map(|chunk| EmbeddingVector {
                    values: chunk.to_vec(),
                    shape: vec![out_width],
                })
                .collect())
        }
    }
}
