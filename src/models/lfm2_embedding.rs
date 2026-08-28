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

//! LFM2.5-Embedding: the hybrid LFM2 backbone run bidirectionally, CLS pooled.
//!
//! The checkpoint declares `model_type: lfm2` with `architectures:
//! ["Lfm2BidirectionalModel"]`: 16 layers alternating 10 gated short
//! convolutions with 6 full-attention layers, `conv_L_cache` 3, GQA 16/8 with
//! per-head Q/K RMSNorm, `embedding_norm` as the final norm. The layers are
//! exactly [`crate::models::lfm2`]'s, with two directional changes:
//!
//! - Attention layers see `create_bidirectional_padding_mask` rather than a
//!   causal mask.
//! - The short-convolution mixer becomes non-causal. This is the one change
//!   that reaches into the backbone: [`crate::models::lfm2::ModelArgs`] grows a
//!   `conv_causal` flag (default `true`, so generation is byte-identical) and
//!   this loader sets it to `false` before construction. The mixer then splits
//!   its `L_cache - 1` zero padding across both sides instead of prepending all
//!   of it, so position `t` mixes `t - 1`, `t` and `t + 1` at `L_cache = 3` and
//!   the output length stays `L`.
//! - The conv input is zeroed at padding positions. A convolution has no key
//!   axis, so the attention mask does not reach it, and without the zeroing the
//!   pad-token embeddings mix into the real positions next to the boundary and
//!   then spread across the whole row through the attention layers above.
//!   Measured on the published checkpoint before the fix: changing only the
//!   masked tail of a padded row moved the pooled vector by cosine 0.94.
//!
//! Pooling is CLS: the tokenizer's post-processor prepends `<|startoftext|>`
//! and the checkpoint's `1_Pooling/config.json` sets
//! `pooling_mode_cls_token: true`, so the sentence vector is the hidden state
//! at that position. Batches are right-padded, which puts it at index 0 in
//! every row; `pool` finds it by first-real-token argmax rather than assuming
//! that, so a left-padded batch would pool correctly too.
//!
//! The late-interaction (ColBERT) LFM2 checkpoints share this `model_type` and
//! architecture but ship a `1_Dense` projection and emit one vector per token.
//! They are out of scope here and are rejected at load with a message saying
//! so, rather than loading and silently returning a single pooled vector.
//!
//! Prompt prefixes (`query: ` / `document: `) are caller-side and documented in
//! `docs/embeddings.md`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::limits::config_normalize_flag;
use crate::embeddings::loader::load_embedding_weights;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::models::embedding_sanitize::sanitize_decoder_embedding_weights;
use crate::models::lfm2::{Lfm2Model, ModelArgs};

/// LFM2's final norm root. The shared decoder-embedding sanitize prefixes
/// `embed_tokens.`, `layers.` and `norm.`; LFM2 spells its final norm
/// `embedding_norm.weight`, which matches none of those, so this family adds
/// the one root the shared list does not cover.
const LFM2_EXTRA_BACKBONE_ROOT: &str = "embedding_norm.";

/// LFM2.5-Embedding: bidirectional LFM2 layers, CLS pooling.
pub struct Lfm2EmbeddingModel {
    model: Lfm2Model,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

/// Prefix a bare `embedding_norm.` root with `model.`.
///
/// A key that already carries the prefix is left alone, so this is a no-op on
/// an mlx conversion and idempotent when applied twice, matching the shared
/// helper's contract.
fn prefix_embedding_norm(weights: &mut WeightMap) {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter(|key| key.starts_with(LFM2_EXTRA_BACKBONE_ROOT))
        .map(|key| (key.clone(), format!("model.{key}")))
        .collect();
    for (from, to) in renames {
        if let Some(tensor) = weights.remove(&from) {
            weights.insert(to, tensor);
        }
    }
}

/// Reject a late-interaction export.
///
/// The multi-vector LFM2 checkpoints ship a `1_Dense` module folder in place of
/// (or alongside) the pooling module. `dense_folders` is what
/// [`sanitize_decoder_embedding_weights`] folded, which catches the same layout
/// through its tensors even when the folder is named differently.
fn reject_late_interaction(model_dir: &Path, dense_folders: usize) -> Result<()> {
    let folder_on_disk = std::fs::read_dir(model_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry.file_name().to_str().is_some_and(|name| {
                name.strip_suffix("_Dense")
                    .is_some_and(|index| index.parse::<u32>().is_ok())
            })
        });
    if dense_folders == 0 && !folder_on_disk {
        return Ok(());
    }
    bail!(
        "{} carries a sentence-transformers Dense module. That is the LFM2 / LFM2.5 ColBERT \
         late-interaction layout, which emits one vector per token through a projection this \
         single-vector embedder does not apply; multi-vector LFM2 checkpoints are not supported \
         yet",
        model_dir.display()
    )
}

impl Lfm2EmbeddingModel {
    /// Load an LFM2.5-Embedding checkpoint from `model_dir`.
    ///
    /// The published export saves the inner `Lfm2BidirectionalModel`, so the
    /// backbone roots arrive bare (`embed_tokens.weight`, `layers.0.…`,
    /// `embedding_norm.weight`) and are prefixed with `model.` before
    /// `Lfm2Model::from_weights` runs its own sanitize (the `w1`/`w2`/`w3`
    /// feed-forward rename and the depthwise conv transpose this checkpoint
    /// both need).
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let mut args: ModelArgs = serde_json::from_value(config.clone())
            .context("failed to parse the LFM2.5-Embedding config.json as an LFM2 config")?;
        // The one backbone behaviour change: the short-conv mixer looks both
        // ways for this family. Generation keeps the default `true`.
        args.conv_causal = false;

        let mut weights = load_embedding_weights(model_dir, config)?;
        let dense_folders = sanitize_decoder_embedding_weights(&mut weights);
        reject_late_interaction(model_dir, dense_folders)?;
        prefix_embedding_norm(&mut weights);

        let pooling = resolve_pooling_mode(model_dir, PoolingMode::Cls)?;
        Self::from_weights(weights, args, pooling, config_normalize_flag(config))
    }

    /// Build the model from an already sanitized weight map.
    ///
    /// Split from [`Self::load`] so the mask, the non-causal conv and the
    /// pooling can be exercised on a synthetic checkpoint-free model.
    /// `Lfm2Model::from_weights` takes the weights by value because it applies
    /// the LFM2 rename pass in place.
    pub(crate) fn from_weights(
        weights: WeightMap,
        args: ModelArgs,
        pooling: PoolingMode,
        normalize: bool,
    ) -> Result<Self> {
        let embedding_dim = args.hidden_size;
        let model = Lfm2Model::from_weights(args, weights)
            .map_err(|e| anyhow::anyhow!("LFM2.5-Embedding: {e}"))?;
        Ok(Self {
            model,
            pooling,
            normalize,
            embedding_dim,
        })
    }

    /// Run the bidirectional backbone and `embedding_norm`, returning the
    /// `[B, L, hidden_size]` hidden states before pooling.
    ///
    /// `attention_mask` is the `[B, L]` int32 padding mask (`1` = real token).
    pub(crate) fn forward_hidden(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        self.model
            .forward_hidden_bidirectional(input_ids, attention_mask)
    }
}

impl EmbeddingModel for Lfm2EmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("LFM2.5-Embedding is a text-only embedder and does not accept images");
        }

        let hidden = self.forward_hidden(batch.input_ids, batch.attention_mask);
        let pooled = pool(&hidden, batch.attention_mask, self.pooling);

        Ok(EmbeddingOutput {
            embeddings: pooled,
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::Cls
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
}

#[cfg(test)]
#[path = "lfm2_embedding_tests.rs"]
mod lfm2_embedding_tests;
