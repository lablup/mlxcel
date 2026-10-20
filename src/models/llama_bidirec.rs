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

//! Bidirectional Llama (the LLM2Vec recipe): the Llama 3 decoder run with a
//! padding-only mask and mean pooled.
//!
//! The checkpoint is an ordinary Llama 3.2 1B shape (`hidden_size` 2048, 16
//! layers, 32/8 heads, `head_dim` 64, `rope_scaling` `llama3` at factor 32)
//! exported as `LlamaBidirectionalModel` with `use_bidirectional_attention:
//! true` and a sentence-transformers `1_Pooling` module declaring mean
//! pooling. Nothing about the layers changes: the only differences from the
//! generator are the mask and the missing head.
//!
//! - Every layer sees `create_bidirectional_padding_mask`, a `[B, 1, 1, L]`
//!   additive f32 mask that blocks only padding keys. A real query row
//!   therefore attends every real token in both directions, which is what the
//!   LLM2Vec conversion trains for.
//! - There is no `lm_head`. The backbone is a
//!   [`crate::models::headless_llama::HeadlessLlama`] rather than a
//!   [`crate::models::Llama3Model`], whose `lm_head` field is non-optional and
//!   would otherwise materialize a tied `128256 x 2048` projection this path
//!   never applies.
//! - Each call runs on fresh per-layer `KVCache`s at offset 0, so RoPE
//!   positions and the mask key axis both start at the first token of the
//!   input and a padded batch reproduces the unpadded single row exactly.
//!
//! Prompt prefixes (`query: ` / `passage: `) are caller-side and documented in
//! `docs/embeddings.md`.
//!
//! LLM2Vec checkpoints published as PEFT adapters are not loadable here: a
//! directory that carries only `adapter_model.safetensors` is rejected at load
//! with a message saying the adapter has to be merged into a full
//! `LlamaBidirectionalModel` export first.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::create_bidirectional_padding_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::limits::config_normalize_flag;
use crate::embeddings::loader::load_embedding_weights;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::models::embedding_sanitize::{
    drop_derived_buffers, sanitize_decoder_embedding_weights, strip_language_model_wrapper,
};
use crate::models::headless_llama::HeadlessLlama;
use crate::models::llama3::ModelArgs;

/// Bidirectional Llama: Llama 3 layers under a padding-only mask, mean pooled.
pub struct LlamaBidirecModel {
    backbone: HeadlessLlama,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

/// Strip a VLM wrapper prefix and drop the derived buffers, then apply the
/// shared decoder-embedding normalization.
///
/// Ordering matters: the wrapper prefix has to come off before
/// [`sanitize_decoder_embedding_weights`] decides whether a key is a bare
/// backbone root, otherwise a wrapped `layers.0.…` key would keep the wrapper
/// and never be prefixed with `model.`.
///
/// Returns the number of sentence-transformers `Dense` folders the shared pass
/// folded, which this family rejects (it applies none).
fn sanitize_llama_bidirec_weights(weights: &mut WeightMap) -> usize {
    strip_language_model_wrapper(weights);
    drop_derived_buffers(weights);
    sanitize_decoder_embedding_weights(weights)
}

/// Reject a directory that ships a PEFT adapter instead of a merged model.
///
/// An adapter-only export carries `adapter_config.json` and
/// `adapter_model.safetensors` and no full shard, so every backbone tensor
/// lookup would fail one at a time with a "weight not found" message that says
/// nothing about the real problem.
fn reject_adapter_only_directory(model_dir: &Path) -> Result<()> {
    let has_adapter = model_dir.join("adapter_model.safetensors").is_file()
        || model_dir.join("adapter_model.bin").is_file();
    if !has_adapter {
        return Ok(());
    }
    let has_full_weights = model_dir.join("model.safetensors").is_file()
        || model_dir.join("model.safetensors.index.json").is_file();
    if has_full_weights {
        return Ok(());
    }
    bail!(
        "{} ships a PEFT adapter (adapter_model.safetensors) and no full model shard. \
         mlxcel does not merge LLM2Vec adapters: merge the adapter into its base model and \
         export a complete LlamaBidirectionalModel checkpoint, then point -m at that directory",
        model_dir.display()
    )
}

impl LlamaBidirecModel {
    /// Load a bidirectional Llama checkpoint from `model_dir`.
    ///
    /// The published export saves the inner `LlamaBidirectionalModel`, so the
    /// backbone roots arrive bare (`embed_tokens.weight`, `layers.0.…`,
    /// `norm.weight`) and are prefixed with `model.` before the headless Llama
    /// backbone is assembled.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        reject_adapter_only_directory(model_dir)?;

        let mut args: ModelArgs = serde_json::from_value(config.clone())
            .context("failed to parse the bidirectional Llama config.json as a Llama 3 config")?;
        args.set_checkpoint_label(model_dir);
        // The embedder stops at the final norm. Forcing the tied flag keeps a
        // hand-built checkpoint that still carries an untied `lm_head` from
        // being read for a head this path never applies; the head tensors are
        // dropped by the sanitize pass either way.
        args.tie_word_embeddings = true;

        let mut weights = load_embedding_weights(model_dir, config)?;
        let dense_folders = sanitize_llama_bidirec_weights(&mut weights);
        if dense_folders > 0 {
            bail!(
                "{} carries {dense_folders} sentence-transformers Dense module(s). Bidirectional \
                 Llama applies no post-pooling projection, so loading it would silently drop a \
                 trained layer; this checkpoint variant is not supported",
                model_dir.display()
            );
        }

        let pooling = resolve_pooling_mode(model_dir, PoolingMode::Mean)?;
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
        let backbone = HeadlessLlama::from_weights(weights, args)
            .map_err(|e| anyhow::anyhow!("bidirectional Llama: {e}"))?;
        Ok(Self {
            backbone,
            pooling,
            normalize,
            embedding_dim: args.hidden_size,
        })
    }

    /// Run the bidirectional backbone and the final norm, returning the
    /// `[B, L, hidden_size]` hidden states before pooling.
    ///
    /// `attention_mask` is the `[B, L]` int32 padding mask (`1` = real token).
    pub(crate) fn forward_hidden(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mask = create_bidirectional_padding_mask(attention_mask);

        // Fresh per-call caches at offset 0: the embedder never reuses a KV
        // cache, and offset 0 keeps RoPE and the mask key axis aligned with the
        // input length.
        let mut caches = self.backbone.make_caches();
        self.backbone
            .forward_hidden(input_ids, None, &mut caches, Some(&mask))
    }
}

impl EmbeddingModel for LlamaBidirecModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("bidirectional Llama is a text-only embedder and does not accept images");
        }

        let hidden = self.forward_hidden(batch.input_ids, batch.attention_mask);
        let pooled = pool(&hidden, batch.attention_mask, self.pooling);

        Ok(EmbeddingOutput {
            embeddings: pooled,
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::Mean
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
#[path = "llama_bidirec_tests.rs"]
mod llama_bidirec_tests;
