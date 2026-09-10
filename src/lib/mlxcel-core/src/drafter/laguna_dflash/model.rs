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

//! [`LagunaDFlashDraftModel`]: the assembled Laguna DFlash drafter.
//!
//! Per draft round the model consumes the proposal block `[bonus, mask, ...,
//! mask]` (`[B, L]`) and the target's captured residual streams for the new
//! context positions (`[B, T, num_layers * target_hidden]`), and produces
//! `[B, L, vocab]` logits. The context path is
//! `hidden_norm(fc(concat_i aux_hidden_norms[i](h_i)))`, which every layer
//! then passes through its own `input_layernorm` before the K/V projection
//! (vLLM `DFlashLagunaForCausalLM::combine_hidden_states` plus
//! `DFlashLagunaModel::_project_context_kv`).
//!
//! The drafter ships neither `embed_tokens` nor `lm_head`; both are bound
//! from the target at [`crate::drafter::Drafter::bind`] time.

use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use crate::ops::concatenate;
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::cache::LagunaDFlashContextCache;
use super::config::LagunaDFlashConfig;
use super::layer::LagunaDFlashDecoderLayer;

/// Laguna DFlash drafter.
pub struct LagunaDFlashDraftModel {
    pub config: LagunaDFlashConfig,
    /// Target embedding table, installed by `bind`.
    pub embed_tokens: Option<UnifiedEmbedding>,
    /// Target output head, installed by `bind`; `None` falls back to the
    /// tied `embed_tokens.as_linear` projection.
    pub lm_head: Option<UnifiedLinear>,
    /// One RMSNorm per captured target layer, applied to that layer's slice
    /// of the concatenated hidden input.
    pub aux_hidden_norms: Vec<RMSNorm>,
    /// `[hidden_size, num_layers * target_hidden]` projection.
    pub fc: UnifiedLinear,
    pub hidden_norm: RMSNorm,
    pub layers: Vec<LagunaDFlashDecoderLayer>,
    pub norm: RMSNorm,
    /// Width of each captured target slice (`aux_hidden_norms[i].weight` len).
    pub target_hidden_size: i32,
    /// Activation dtype the drafter weights were loaded in. Target-provided
    /// tensors (embeddings, captured hidden states) are cast to it so the
    /// drafter never runs a mixed-dtype matmul.
    pub compute_dtype: i32,
}

impl LagunaDFlashDraftModel {
    /// Build from a sanitized weight map (see [`super::sanitize`]).
    pub fn from_weights(weights: &WeightMap, config: LagunaDFlashConfig) -> Result<Self, String> {
        // Only consulted for quantized variants; the published bf16 drafters
        // carry no `.scales`.
        let group_size = 64;
        let bits = 4;
        let norm = |key: &str| -> Result<RMSNorm, String> {
            let w = weights
                .get(key)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            Ok(RMSNorm::new(w, config.rms_norm_eps))
        };
        let mut aux_hidden_norms = Vec::with_capacity(config.target_layer_ids.len());
        let mut target_hidden_size = 0;
        for i in 0..config.target_layer_ids.len() {
            let key = format!("aux_hidden_norms.{i}.weight");
            let width = weights
                .get(&key)
                .map(|w| ffi::array_shape(w)[0])
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            if i == 0 {
                target_hidden_size = width;
            } else if width != target_hidden_size {
                return Err(format!(
                    "{key} has width {width} but aux_hidden_norms.0 has {target_hidden_size}"
                ));
            }
            aux_hidden_norms.push(norm(&key)?);
        }
        let fc_rows = weights
            .get("fc.weight")
            .map(|w| ffi::array_shape(w)[0])
            .ok_or_else(|| "Weight not found: fc.weight".to_string())?;
        if fc_rows != config.hidden_size as i32 {
            return Err(format!(
                "fc.weight has {fc_rows} rows but hidden_size is {}",
                config.hidden_size
            ));
        }
        let fc = UnifiedLinear::from_weights(weights, "fc", group_size, bits)?;
        let hidden_norm = norm("hidden_norm.weight")?;
        let final_norm = norm("norm.weight")?;
        let compute_dtype = weights
            .get("norm.weight")
            .map(|w| ffi::array_dtype(w))
            .ok_or_else(|| "Weight not found: norm.weight".to_string())?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(LagunaDFlashDecoderLayer::from_weights(
                weights,
                &format!("layers.{i}"),
                &config,
                config.sliding_windows[i],
                group_size,
                bits,
            )?);
        }
        Ok(Self {
            config,
            embed_tokens: None,
            lm_head: None,
            aux_hidden_norms,
            fc,
            hidden_norm,
            layers,
            norm: final_norm,
            target_hidden_size,
            compute_dtype,
        })
    }

    /// Install the target's embedding table (shared buffer, no copy).
    pub fn bind_target_embedding(&mut self, embed: UnifiedEmbedding) {
        self.embed_tokens = Some(embed);
    }

    /// Install the target's output head; `None` keeps the tied fallback.
    pub fn bind_target_lm_head(&mut self, lm_head: Option<UnifiedLinear>) {
        self.lm_head = lm_head;
    }

    /// Whether `bind` still has to supply the embedding table.
    pub fn needs_embed_binding(&self) -> bool {
        self.embed_tokens.is_none()
    }

    fn embed(&self) -> &UnifiedEmbedding {
        self.embed_tokens.as_ref().expect(
            "Laguna DFlash drafter embed_tokens is not bound; the drafter borrows the \
             target's embedding table via Drafter::bind() before the first forward()",
        )
    }

    /// Fresh per-layer context caches.
    pub fn make_cache(&self) -> Vec<LagunaDFlashContextCache> {
        self.config
            .sliding_windows
            .iter()
            .map(|w| LagunaDFlashContextCache::new(*w as i32))
            .collect()
    }

    fn to_compute_dtype(&self, x: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        if ffi::array_dtype(&x) == self.compute_dtype {
            x
        } else {
            ffi::astype(&x, self.compute_dtype)
        }
    }

    /// Borrow `x` as the compute dtype: `None` when it already is, so the
    /// caller slices the original without a copy.
    fn cast_if_needed(&self, x: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        (ffi::array_dtype(x) != self.compute_dtype).then(|| ffi::astype(x, self.compute_dtype))
    }

    /// `hidden_norm(fc(concat_i aux_hidden_norms[i](slice_i)))` over the
    /// `[B, T, num_layers * target_hidden]` concatenation.
    pub fn combine_hidden(&self, target_hidden: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(target_hidden);
        debug_assert_eq!(shape.len(), 3, "target hidden must be [B, T, D]");
        let n = self.aux_hidden_norms.len() as i32;
        let width = self.target_hidden_size;
        assert_eq!(
            shape[2],
            n * width,
            "Laguna DFlash drafter expected {n} captured slices of {width} features, got a \
             {}-wide hidden input",
            shape[2]
        );
        let cast = self.cast_if_needed(target_hidden);
        let hidden: &MlxArray = match &cast {
            Some(c) => c,
            None => target_hidden,
        };
        let mut normed: Option<UniquePtr<MlxArray>> = None;
        for (i, norm) in self.aux_hidden_norms.iter().enumerate() {
            let start = i as i32 * width;
            let slab = ffi::slice(hidden, &[0, 0, start], &[shape[0], shape[1], start + width]);
            let slab = norm.forward(&slab);
            normed = Some(match normed {
                Some(acc) => concatenate(&acc, &slab, -1),
                None => slab,
            });
        }
        let normed = normed.expect("at least one captured target layer");
        let projected = self.fc.forward(&normed);
        self.hidden_norm.forward(&projected)
    }

    /// Forward over the proposal block.
    ///
    /// - `inputs`: `[B, L]` int32 token ids.
    /// - `target_hidden`: `[B, T, num_layers * target_hidden]` captured
    ///   residual streams for the `T` new context positions.
    /// - `caches`: one context cache per drafter layer.
    ///
    /// Returns `[B, L, vocab]` logits.
    pub fn forward(
        &self,
        inputs: &MlxArray,
        target_hidden: &MlxArray,
        caches: &mut [LagunaDFlashContextCache],
    ) -> UniquePtr<MlxArray> {
        debug_assert_eq!(
            caches.len(),
            self.layers.len(),
            "one context cache per layer"
        );
        let embed = self.embed();
        let mut x = self.to_compute_dtype(embed.forward(inputs));
        // Rows older than the widest window can never be attended: drop them
        // before `aux_hidden_norms` and `fc` run over the whole prompt, and
        // advance every layer's absolute offset past them. Each layer still
        // trims to its own window afterwards.
        let shape = ffi::array_shape(target_hidden);
        let keep = self
            .config
            .sliding_windows
            .iter()
            .max()
            .map(|w| *w as i32 - 1)
            .unwrap_or(shape[1]);
        let sliced;
        let target_hidden = if shape[1] > keep {
            let dropped = shape[1] - keep;
            for cache in caches.iter_mut() {
                cache.advance(dropped);
            }
            sliced = ffi::slice(
                target_hidden,
                &[0, dropped, 0],
                &[shape[0], shape[1], shape[2]],
            );
            &sliced
        } else {
            target_hidden
        };
        let ctx = self.combine_hidden(target_hidden);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            x = layer.forward(&x, &ctx, cache);
        }
        let x = self.norm.forward(&x);
        match &self.lm_head {
            Some(head) => head.forward(&x),
            None => embed.as_linear(&x),
        }
    }
}
