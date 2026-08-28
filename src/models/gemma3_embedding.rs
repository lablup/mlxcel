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

//! EmbeddingGemma: the Gemma 3 text backbone run bidirectionally, mean
//! pooled, then projected through two bias-free `Dense` modules.
//!
//! The checkpoint is a Gemma 3 decoder (`model_type: gemma3_text`,
//! `architectures: ["Gemma3TextModel"]`) exported with
//! `use_bidirectional_attention: true`, so the layers, the norms and the RoPE
//! schedule are exactly [`crate::models::gemma3`]'s. Only three things differ
//! from the generator:
//!
//! - Every layer sees a bidirectional mask. The full-attention layers get a
//!   padding-only mask; the sliding layers get the same padding mask
//!   intersected with a symmetric `|q - k| < sliding_window` band, which is
//!   how `transformers` builds the bidirectional sliding overlay. For inputs
//!   up to `sliding_window` tokens the two masks are identical, so the window
//!   only starts to matter past 512 tokens.
//! - There is no `lm_head`. The final norm output is mask-weighted mean
//!   pooled into one vector per row.
//! - Two `Dense` projections (`768 -> 3072 -> 768`, no bias, no activation)
//!   run after pooling, before the L2 normalization the engine applies.
//!
//! Because the masks are owned here, each layer's built-in causal window
//! parameter is cleared after construction: an explicit mask plus a causal
//! window would re-impose causality on the Metal 4 attention path.
//!
//! Prompt prefixes (`task: search result | query: `, `title: none | text: `)
//! are caller-side and documented in `docs/embeddings.md`; the trained
//! Matryoshka widths are 768, 512, 256 and 128.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::{GemmaRMSNorm, KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_bidirectional_padding_mask, create_bidirectional_window_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::limits::config_normalize_flag;
use crate::embeddings::loader::load_embedding_weights;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::models::embedding_sanitize::{linear_features, sanitize_decoder_embedding_weights};
use crate::models::gemma3::{Cache, ModelArgs, TransformerBlock};

/// `layer_types` entry marking a full-attention (global) layer.
const FULL_ATTENTION: &str = "full_attention";

/// EmbeddingGemma: bidirectional Gemma 3 layers, mean pooling, two `Dense`
/// projections.
pub struct Gemma3EmbeddingModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<TransformerBlock>,
    norm: GemmaRMSNorm,
    /// Post-pooling projections in application order (`dense.0`, `dense.1`).
    /// Empty for a bidirectional Gemma 3 export that ships no `Dense` module.
    dense: Vec<UnifiedLinear>,
    hidden_size: usize,
    /// Symmetric half-width of the sliding-layer band, in tokens.
    sliding_window: i32,
    /// Layer `i` is a full-attention layer iff `(i + 1) % pattern == 0`.
    sliding_window_pattern: usize,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

/// Resolve the sliding / full alternation period.
///
/// `layer_types` is authoritative when present: transformers 4.57 renamed the
/// scalar to `_sliding_window_pattern` (which [`ModelArgs`] does not parse,
/// so it would silently fall back to the family default of 6) and lists the
/// per-layer kinds instead. The list is validated against the derived period
/// rather than only sampled, because a period that is off by one still loads,
/// still runs, and only shows up as a quietly wrong embedding.
fn resolve_sliding_window_pattern(config: &Value, fallback: usize) -> Result<usize> {
    let Some(layer_types) = config.get("layer_types").and_then(Value::as_array) else {
        let scalar = config
            .get("sliding_window_pattern")
            .or_else(|| config.get("_sliding_window_pattern"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .filter(|&v| v > 0);
        return Ok(scalar.unwrap_or(fallback));
    };
    if layer_types.is_empty() {
        return Ok(fallback);
    }

    let is_full: Vec<bool> = layer_types
        .iter()
        .map(|entry| entry.as_str() == Some(FULL_ATTENTION))
        .collect();
    // No full-attention layer at all: every layer is sliding, which
    // `(i + 1) % pattern == 0` expresses with a period past the last index.
    let pattern = match is_full.iter().position(|&full| full) {
        Some(first) => first + 1,
        None => return Ok(layer_types.len() + 1),
    };
    for (i, &full) in is_full.iter().enumerate() {
        if full != (i + 1).is_multiple_of(pattern) {
            bail!(
                "config.json `layer_types` is not a repeating period-{pattern} pattern: layer {i} \
                 is `{}` but the period derived from the first `{FULL_ATTENTION}` entry says \
                 otherwise",
                layer_types[i].as_str().unwrap_or("?")
            );
        }
    }
    Ok(pattern)
}

impl Gemma3EmbeddingModel {
    /// Load an EmbeddingGemma checkpoint from `model_dir`.
    ///
    /// Accepts both published layouts: the mlx conversion, which folds the
    /// projections into the main shards as `dense.{k}.*` and keeps the
    /// `model.` prefix, and the sentence-transformers original, whose
    /// `2_Dense/` and `3_Dense/` module folders reach us as
    /// `{N}_Dense.linear.*` and whose backbone roots are bare.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let mut args: ModelArgs = serde_json::from_value(config.clone())
            .context("failed to parse the EmbeddingGemma config.json as a Gemma 3 config")?;
        args.sliding_window_pattern =
            resolve_sliding_window_pattern(config, args.sliding_window_pattern)?;

        let mut weights = load_embedding_weights(model_dir, config)?;
        sanitize_decoder_embedding_weights(&mut weights);

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
        let group_size = args.group_size();
        let bits = args.bits();
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)
                .map_err(|e| anyhow::anyhow!("EmbeddingGemma: {e}"))?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let mut layer = TransformerBlock::from_weights(weights, args, i)
                .map_err(|e| anyhow::anyhow!("EmbeddingGemma layer {i}: {e}"))?;
            // The embedding path builds and owns every mask, so the layer must
            // not also apply its causal sliding window.
            layer.self_attn.window_size = 0;
            layers.push(layer);
        }

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| {
                anyhow::anyhow!("EmbeddingGemma: weight not found: model.norm.weight")
            })?;
        let norm = GemmaRMSNorm::new(norm_weight, args.rms_norm_eps);

        let dense = load_dense_stack(weights, group_size, bits, args.hidden_size)?;
        let embedding_dim = match dense.len().checked_sub(1) {
            Some(last) => linear_features(weights, &format!("dense.{last}"), group_size)
                .map(|(out, _)| out as usize)
                .unwrap_or(args.hidden_size),
            None => args.hidden_size,
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            dense,
            hidden_size: args.hidden_size,
            sliding_window: args.sliding_window as i32,
            sliding_window_pattern: args.sliding_window_pattern,
            pooling,
            normalize,
            embedding_dim,
        })
    }

    /// `true` when layer `i` is a full-attention layer.
    fn is_full_attention(&self, i: usize) -> bool {
        (i + 1).is_multiple_of(self.sliding_window_pattern)
    }

    /// `true` when at least one layer needs the windowed mask.
    fn has_sliding_layers(&self) -> bool {
        (0..self.layers.len()).any(|i| !self.is_full_attention(i))
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
        // Gemma scales the token embeddings by sqrt(hidden_size), exactly as
        // `Gemma3Model::get_embed_tokens` does for the generator.
        let mut h = self.embed_tokens.forward(input_ids);
        h = mlxcel_core::multiply_scalar(&h, (self.hidden_size as f32).sqrt());

        let full_mask = create_bidirectional_padding_mask(attention_mask);
        let sliding_mask = (self.has_sliding_layers() && self.sliding_window >= 1)
            .then(|| create_bidirectional_window_mask(attention_mask, self.sliding_window));

        // Fresh per-call caches at offset 0: the embedder never reuses a KV
        // cache, and offset 0 keeps RoPE and the mask key axis aligned with
        // the input length.
        let mut caches: Vec<Cache> = (0..self.layers.len())
            .map(|_| Cache::Standard(KVCache::new()))
            .collect();

        for (i, layer) in self.layers.iter().enumerate() {
            let mask: &MlxArray = if self.is_full_attention(i) {
                &full_mask
            } else {
                sliding_mask.as_deref().unwrap_or(&full_mask)
            };
            h = layer.forward(&h, caches[i].as_interface(), Some(mask));
        }
        self.norm.forward(&h)
    }
}

/// Load the post-pooling `Dense` stack (`dense.0`, `dense.1`, ...), verifying
/// that the widths chain from the backbone hidden size.
///
/// A shape mismatch here is the failure mode a silent `dense.0` / `dense.1`
/// swap produces: both projections load, the forward pass runs, and only the
/// vectors are wrong. Checking the chain turns that into a load error.
fn load_dense_stack(
    weights: &WeightMap,
    group_size: i32,
    bits: i32,
    hidden_size: usize,
) -> Result<Vec<UnifiedLinear>> {
    let mut stack = Vec::new();
    let mut expected_in = hidden_size as i32;
    for index in 0.. {
        let prefix = format!("dense.{index}");
        let Some((out_features, in_features)) = linear_features(weights, &prefix, group_size)
        else {
            break;
        };
        if in_features != expected_in {
            bail!(
                "EmbeddingGemma: {prefix} expects {in_features} input features but the previous \
                 stage produces {expected_in}; the Dense modules are out of order or belong to a \
                 different checkpoint"
            );
        }
        stack.push(
            UnifiedLinear::from_weights(weights, &prefix, group_size, bits)
                .map_err(|e| anyhow::anyhow!("EmbeddingGemma {prefix}: {e}"))?,
        );
        expected_in = out_features;
    }
    Ok(stack)
}

impl EmbeddingModel for Gemma3EmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        if batch.images.is_some_and(|images| !images.is_empty()) {
            bail!("EmbeddingGemma is a text-only embedder and does not accept images");
        }

        let hidden = self.forward_hidden(batch.input_ids, batch.attention_mask);
        let hidden_dtype = mlxcel_core::array_dtype(&hidden);
        let mut pooled: UniquePtr<MlxArray> = pool(&hidden, batch.attention_mask, self.pooling);
        if !self.dense.is_empty() {
            // Pooling returns f32; the projections carry the checkpoint's own
            // dtype, so cast back before the (possibly quantized) matmuls. With
            // no projections the f32 pooled vector goes straight to the engine
            // instead of taking a needless round trip through f16.
            pooled = mlxcel_core::astype(&pooled, hidden_dtype);
            for dense in &self.dense {
                pooled = dense.forward(&pooled);
            }
        }

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
#[path = "gemma3_embedding_tests.rs"]
mod gemma3_embedding_tests;
