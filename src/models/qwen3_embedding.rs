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

//! Qwen3-Embedding: the causal Qwen3 backbone with last-token pooling.
//!
//! Unlike EmbeddingGemma this family keeps the decoder causal. The export is
//! an ordinary `Qwen3ForCausalLM` (`model_type: qwen3`) with a
//! sentence-transformers `1_Pooling` module declaring
//! `pooling_mode_lasttoken: true`, and the sentence embedding is the hidden
//! state at the `<|endoftext|>` token the tokenizer appends. Because the last
//! real token attends every earlier token, a causal mask over a right-padded
//! batch produces exactly the single-row result: padding sits after the
//! pooled position and is blocked as a key everywhere.
//!
//! The only backbone work is reaching the hidden states without the head:
//! [`Qwen3Model::forward_hidden`] stops after the final norm, so no
//! `[B, L, vocab_size]` logit tensor is ever materialized. The tied `lm_head`
//! is dropped at load for the same reason.
//!
//! The query format (`Instruct: {task}\nQuery: {query}`) is caller-side;
//! passing `instruction` on the request or `--instruction` on the CLI applies
//! it here, and documents embed as raw text.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::create_causal_padding_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::limits::config_normalize_flag;
use crate::embeddings::loader::load_embedding_weights;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::models::embedding_sanitize::sanitize_decoder_embedding_weights;
use crate::models::qwen3::{ModelArgs, Qwen3Model};

/// Qwen3-Embedding: causal Qwen3 layers, last-token pooling.
pub struct Qwen3EmbeddingModel {
    model: Qwen3Model,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

impl Qwen3EmbeddingModel {
    /// Load a Qwen3-Embedding checkpoint from `model_dir`.
    ///
    /// The published checkpoints save the inner `Qwen3Model`, so the backbone
    /// roots arrive bare (`embed_tokens.weight`, `layers.0.…`, `norm.weight`)
    /// and are prefixed with `model.` before the Qwen3 constructor runs.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let mut args: ModelArgs = serde_json::from_value(config.clone())
            .context("failed to parse the Qwen3-Embedding config.json as a Qwen3 config")?;
        args.set_checkpoint_label(model_dir);
        // The embedder stops at the final norm, so no head is ever applied.
        // Forcing the tied flag both drops an untied `lm_head` from memory (on
        // the 0.6B checkpoint that is a 151669 x 1024 tensor) and keeps the
        // constructor from failing on a head this path would never read.
        args.tie_word_embeddings = true;

        let mut weights = load_embedding_weights(model_dir, config)?;
        sanitize_decoder_embedding_weights(&mut weights);

        let pooling = resolve_pooling_mode(model_dir, PoolingMode::LastToken)?;
        Self::from_weights(&weights, &args, pooling, config_normalize_flag(config))
    }

    /// Build the model from an already sanitized weight map.
    ///
    /// Split from [`Self::load`] so the mask and pooling behaviour can be
    /// exercised on a synthetic checkpoint-free model.
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        pooling: PoolingMode,
        normalize: bool,
    ) -> Result<Self> {
        let model = Qwen3Model::from_weights(weights, args)
            .map_err(|e| anyhow::anyhow!("Qwen3-Embedding: {e}"))?;
        Ok(Self {
            model,
            pooling,
            normalize,
            embedding_dim: args.hidden_size,
        })
    }

    /// Run the causal backbone and the final norm, returning the
    /// `[B, L, hidden_size]` hidden states before pooling.
    pub(crate) fn forward_hidden(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mask = create_causal_padding_mask(attention_mask, 0);
        let mut caches = self.model.make_caches();
        self.model
            .forward_hidden(input_ids, None, &mut caches, Some(&mask))
    }
}

impl EmbeddingModel for Qwen3EmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("Qwen3-Embedding is a text-only embedder and does not accept images");
        }

        let hidden = self.forward_hidden(batch.input_ids, batch.attention_mask);
        let pooled = pool(&hidden, batch.attention_mask, self.pooling);

        Ok(EmbeddingOutput {
            embeddings: pooled,
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::LastToken
    }

    fn pooling(&self) -> PoolingMode {
        self.pooling
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Wrap a query in the checkpoint's instruction format when the caller
    /// supplies one. Documents are embedded raw, which is what the model card
    /// prescribes, so an absent or blank instruction is the identity.
    fn format_text(&self, text: &str, instruction: Option<&str>) -> String {
        match instruction.map(str::trim).filter(|task| !task.is_empty()) {
            Some(task) => format!("Instruct: {task}\nQuery: {text}"),
            None => text.to_string(),
        }
    }
}

#[cfg(test)]
#[path = "qwen3_embedding_tests.rs"]
mod qwen3_embedding_tests;
