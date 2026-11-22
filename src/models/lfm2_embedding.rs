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
//! Two more checkpoints share this backbone and `model_type`, and the loader
//! tells them apart by their weights rather than by a separate `ModelType`:
//!
//! - LFM2.5-Encoder (`Lfm2BidirectionalForMaskedLM`) stores the backbone under
//!   `lfm2.`, which is stripped before the shared sanitize. Its masked-LM head
//!   is tied to `embed_tokens`, so nothing else is dropped, and it pools like
//!   LFM2.5-Embedding (CLS unless `1_Pooling` or `--pooling` says otherwise).
//!   It is a fine-tuning backbone: the pooled vector is not a trained sentence
//!   embedding.
//! - LFM2.5-ColBERT ships one token projection (`1_Dense/`, `dense.*` in the
//!   mlx-community conversions, or `projection.*` in an mlx-vlm re-export;
//!   see [`head`]). With it the model is
//!   multi-vector: every token's hidden state goes through the projection and
//!   is L2-normalized by [`project_and_normalize`], and pooling is skipped. The
//!   upstream reference runs this head with `mask_convolutions=False`; this
//!   port keeps the padding-zeroed conv input for it too, which agrees with the
//!   reference on unpadded rows and keeps a padded row equal to the same row
//!   run alone.
//!
//! Prompt prefixes (`query: ` / `document: `) are caller-side and documented in
//! `docs/embeddings.md`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::limits::config_normalize_flag;
use crate::embeddings::loader::load_embedding_weights;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::models::col_late_interaction::project_and_normalize;
use crate::models::embedding_sanitize::sanitize_decoder_embedding_weights;
use crate::models::lfm2::{Lfm2Model, ModelArgs};

#[path = "lfm2_embedding_head.rs"]
mod head;

/// Wrapper prefix the `Lfm2BidirectionalForMaskedLM` export stores the
/// backbone under.
const MASKED_LM_PREFIX: &str = "lfm2.";

/// LFM2.5-Embedding, -Encoder and -ColBERT: bidirectional LFM2 layers, then
/// either pooling (single-vector) or a token projection (multi-vector).
pub struct Lfm2EmbeddingModel {
    model: Lfm2Model,
    /// The ColBERT token projection; `None` for the pooled embedders.
    projection: Option<UnifiedLinear>,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

/// The key passes [`Lfm2EmbeddingModel::load`] applies before
/// `Lfm2Model::from_weights` runs its own sanitize: strip the masked-LM
/// export's single `lfm2.` wrapper, then the shared decoder-embedding
/// normalization (Dense fold, head drop, `model.` prefix).
///
/// Returns the number of `Dense` folders folded.
pub(crate) fn sanitize_family_weights(weights: &mut WeightMap) -> usize {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter_map(|key| {
            key.strip_prefix(MASKED_LM_PREFIX)
                .map(|rest| (key.clone(), rest.to_string()))
        })
        .collect();
    for (from, to) in renames {
        if let Some(tensor) = weights.remove(&from) {
            weights.insert(to, tensor);
        }
    }
    sanitize_decoder_embedding_weights(weights)
}

impl Lfm2EmbeddingModel {
    /// Load an LFM2.5-Embedding, -Encoder or -ColBERT checkpoint from
    /// `model_dir`.
    ///
    /// The published export saves the inner `Lfm2BidirectionalModel`, so the
    /// backbone roots arrive bare (`embed_tokens.weight`, `layers.0.…`,
    /// `embedding_norm.weight`) and are prefixed with `model.` by the shared
    /// [`sanitize_decoder_embedding_weights`] before `Lfm2Model::from_weights`
    /// runs its own sanitize (the `w1`/`w2`/`w3` feed-forward rename and the
    /// depthwise conv transpose this checkpoint both need).
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let mut args: ModelArgs = serde_json::from_value(config.clone())
            .context("failed to parse the LFM2.5-Embedding config.json as an LFM2 config")?;
        // The one backbone behaviour change: the short-conv mixer looks both
        // ways for this family. Generation keeps the default `true`.
        args.conv_causal = false;

        let mut weights = load_embedding_weights(model_dir, config)?;
        let dense_folders = sanitize_family_weights(&mut weights);
        let prefix = head::find_projection(&weights, dense_folders)
            .with_context(|| format!("loading {}", model_dir.display()))?;
        head::validate_projection(model_dir, config, &weights, prefix, args.group_size())?;

        // A multi-vector head keeps every token and never pools, so the
        // pooling resolver (and its `--pooling` override) only applies to the
        // single-vector exports.
        let pooling = if prefix.is_some() {
            PoolingMode::Cls
        } else {
            resolve_pooling_mode(model_dir, PoolingMode::Cls)?
        };
        Self::from_weights(weights, args, pooling, config_normalize_flag(config))
    }

    /// Build the model from an already sanitized weight map.
    ///
    /// Split from [`Self::load`] so the mask, the non-causal conv and the
    /// pooling can be exercised on a synthetic checkpoint-free model.
    /// `Lfm2Model::from_weights` takes the weights by value because it applies
    /// the LFM2 rename pass in place.
    ///
    /// A linear under one of `head::PROJECTION_PREFIXES` (`dense.0`, `dense`,
    /// `projection`, `model.projection`) in `weights` becomes the ColBERT
    /// token projection and makes the model multi-vector; its tensors are
    /// taken out before the backbone is built.
    pub(crate) fn from_weights(
        mut weights: WeightMap,
        args: ModelArgs,
        pooling: PoolingMode,
        normalize: bool,
    ) -> Result<Self> {
        let (group_size, bits) = (args.group_size(), args.bits());
        let mut embedding_dim = args.hidden_size;
        let projection = match head::find_projection(&weights, 0)? {
            Some(prefix) => {
                let linear = UnifiedLinear::from_weights(&weights, prefix, group_size, bits)
                    .map_err(|e| anyhow::anyhow!("LFM2 ColBERT projection `{prefix}`: {e}"))?;
                let (out_features, in_features) = head::projection_widths(&linear, prefix)?;
                if in_features != args.hidden_size {
                    bail!(
                        "LFM2 ColBERT projection `{prefix}` takes {in_features} input features \
                         but the backbone hidden size is {}",
                        args.hidden_size
                    );
                }
                head::take_projection_keys(&mut weights, prefix);
                embedding_dim = out_features;
                Some(linear)
            }
            None => None,
        };
        let model = Lfm2Model::from_weights(args, weights)
            .map_err(|e| anyhow::anyhow!("LFM2.5-Embedding: {e}"))?;
        Ok(Self {
            model,
            projection,
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
        let embeddings = match &self.projection {
            // `[B, L, D]`, unit rows, padding rows zeroed.
            Some(projection) => project_and_normalize(&hidden, projection, batch.attention_mask),
            None => pool(&hidden, batch.attention_mask, self.pooling),
        };

        Ok(EmbeddingOutput {
            embeddings,
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

    fn multi_vector(&self) -> bool {
        self.projection.is_some()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
#[path = "lfm2_embedding_tests.rs"]
mod lfm2_embedding_tests;

#[cfg(test)]
#[path = "lfm2_embedding_head_tests.rs"]
mod lfm2_embedding_head_tests;
