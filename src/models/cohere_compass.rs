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

//! Cohere Compass text decoder (`cohere_compass_text`), the language half of
//! CohereLabs North-Micro-Vision.
//!
//! It is a Command-style decoder (parallel attention + MLP over one LayerNorm,
//! tied embeddings, `logit_scale`) crossed with the Qwen3-VL positional stack:
//!
//! - `sliding_attention` layers rotate with the interleaved 3-axis MRoPE of
//!   [`crate::models::qwen3_vl::InterleavedMRoPE`] at base 50000 and attend at
//!   most `sliding_window` keys;
//! - `full_attention` layers get **no** positional encoding at all
//!   (`rope_parameters.full_attention` is JSON `null`) and attend the whole
//!   prefix;
//! - the first three layers take a DeepStack injection from the vision tower.
//!
//! Reference: `CohereCompassForConditionalGeneration` in transformers 5.15.

use crate::models::cohere_compass_config::{CompassTextConfig, SLIDING_ATTENTION};
use crate::models::cohere_compass_layers::{
    CompassAttention, CompassBlock, CompassMLP, CompassNorm,
};
use crate::models::qwen_mrope_state::MRopeState;
use crate::models::qwen3_vl::Qwen3VLModel;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{KVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask_dense};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::cell::RefCell;

/// Cohere Compass language model.
pub struct CohereCompassTextModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<CompassBlock>,
    norm: CompassNorm,
    lm_head: UnifiedLinear,
    logit_scale: f32,
    /// `true` where `layer_types[i] == "sliding_attention"`.
    layer_is_sliding: Vec<bool>,
    sliding_window: i32,
    /// First sliding / first full layer, used to size the two prefill masks.
    swa_idx: usize,
    ga_idx: usize,
    eos_token_ids: Vec<i32>,
    mrope_state: MRopeState,
    visual_pos_masks: RefCell<Option<UniquePtr<MlxArray>>>,
    deepstack_visual_embeds: RefCell<Option<Vec<UniquePtr<MlxArray>>>>,
}

impl CohereCompassTextModel {
    pub fn from_weights(weights: &WeightMap, config: &CompassTextConfig) -> Result<Self, String> {
        if config.transformer_block_type != "parallel" {
            return Err(format!(
                "cohere_compass: transformer_block_type {:?} is not implemented; only the \
                 parallel block (attention and MLP over one input LayerNorm) ships in this family",
                config.transformer_block_type
            ));
        }
        config.validate_rope_style()?;
        let layer_types = config.resolved_layer_types()?;
        let gs = config.group_size();
        let bits = config.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_is_sliding = Vec::with_capacity(config.num_hidden_layers);
        for (i, layer_type) in layer_types.iter().enumerate() {
            let is_sliding = layer_type == SLIDING_ATTENTION;
            let window = if is_sliding {
                config.sliding_window as i32
            } else {
                0
            };
            let rope_spec = config.rope_for_layer_type(layer_type)?;
            let prefix = format!("model.layers.{i}");
            layers.push(CompassBlock {
                self_attn: CompassAttention::from_weights(
                    weights, config, &prefix, rope_spec, window,
                )?,
                mlp: CompassMLP::from_weights(weights, config, &prefix)?,
                input_layernorm: CompassNorm::from_weights(
                    weights,
                    &format!("{prefix}.input_layernorm"),
                    config,
                )?,
            });
            layer_is_sliding.push(is_sliding);
        }

        let norm = CompassNorm::from_weights(weights, "model.norm", config)?;
        let lm_head = if config.tie_word_embeddings {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", gs, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?
        };

        let swa_idx = layer_is_sliding.iter().position(|&s| s).unwrap_or(0);
        let ga_idx = layer_is_sliding.iter().position(|&s| !s).unwrap_or(0);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            logit_scale: config.logit_scale,
            layer_is_sliding,
            sliding_window: config.sliding_window as i32,
            swa_idx,
            ga_idx,
            eos_token_ids: config.eos_token_ids(),
            mrope_state: MRopeState::new(),
            visual_pos_masks: RefCell::new(None),
            deepstack_visual_embeds: RefCell::new(None),
        })
    }

    /// `true` where `layer_types[i] == "sliding_attention"`. Test surface for
    /// the per-layer cache/window split.
    #[must_use]
    pub fn layer_is_sliding(&self) -> &[bool] {
        &self.layer_is_sliding
    }

    #[must_use]
    pub fn sliding_window(&self) -> i32 {
        self.sliding_window
    }

    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        self.mrope_state.set_fallback(position_ids, rope_deltas);
    }

    pub fn clear_mrope_state(&self) {
        self.mrope_state.clear_fallback();
    }

    pub fn release_mrope_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.release_sequence(seq_id);
    }

    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.bind_fallback_to_sequence(seq_id);
    }

    pub(crate) fn take_mrope_entry(
        &self,
        seq_id: SequenceId,
    ) -> Option<crate::models::qwen_mrope_state::MRopeEntry> {
        self.mrope_state.take_for_sequence(seq_id)
    }

    pub(crate) fn install_mrope_entry(
        &self,
        seq_id: SequenceId,
        entry: crate::models::qwen_mrope_state::MRopeEntry,
    ) {
        self.mrope_state.bind_for_sequence(seq_id, entry);
    }

    pub fn set_deepstack_state(
        &self,
        visual_pos_masks: UniquePtr<MlxArray>,
        deepstack_visual_embeds: Vec<UniquePtr<MlxArray>>,
    ) {
        *self.visual_pos_masks.borrow_mut() = Some(visual_pos_masks);
        *self.deepstack_visual_embeds.borrow_mut() = Some(deepstack_visual_embeds);
    }

    pub fn clear_deepstack_state(&self) {
        *self.visual_pos_masks.borrow_mut() = None;
        *self.deepstack_visual_embeds.borrow_mut() = None;
    }

    /// Build the two prefill masks. `None` at decode width, where every live
    /// key is permitted for the full layers and the sliding layers get their
    /// window from `causal_attention`'s `window_size` argument instead.
    ///
    /// The caller-supplied `mask` is deliberately not threaded through: this
    /// model needs two differently shaped masks per step, so it always builds
    /// them from its own caches. Same contract as `cohere2.rs` / `olmo3.rs`.
    fn prefill_masks(
        &self,
        l: i32,
        caches: &[KVCache],
    ) -> (Option<UniquePtr<MlxArray>>, Option<UniquePtr<MlxArray>>) {
        if l <= 1 {
            return (None, None);
        }
        // Size from the cache's live window, not the monotonic `offset`: under
        // `--max-kv-size` a trimmed cache returns only `live_len` keys and an
        // `offset`-sized mask would be wider than the K/V (issue #419).
        let full = create_causal_mask(l, caches[self.ga_idx].live_len());
        let sliding = create_sliding_window_prefill_mask_dense(
            l,
            caches[self.swa_idx].live_len(),
            self.sliding_window,
        );
        (Some(full), Some(sliding))
    }

    fn mask_for_layer<'m>(
        &self,
        layer_idx: usize,
        full: &'m Option<UniquePtr<MlxArray>>,
        sliding: &'m Option<UniquePtr<MlxArray>>,
    ) -> Option<&'m MlxArray> {
        let chosen = if self.layer_is_sliding[layer_idx] {
            sliding
        } else {
            full
        };
        chosen.as_ref().map(|m| m.as_ref().unwrap())
    }

    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, input_embeddings, caches, None)
    }

    pub(crate) fn forward_for_sequence(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let h = self.forward_hidden_for_sequence(input_ids, input_embeddings, caches, seq_id);
        let logits = self.lm_head.forward(&h);
        // Command-family logit scaling, applied after the (tied) head.
        let scale =
            mlxcel_core::full_f32(&[1], self.logit_scale, mlxcel_core::array_dtype(&logits));
        mlxcel_core::multiply(&logits, &scale)
    }

    fn forward_hidden_for_sequence(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let cache_offset = caches[0].offset;
        if input_embeddings.is_none() && cache_offset == 0 {
            self.clear_mrope_state();
            self.clear_deepstack_state();
        }

        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        let ids_shape = mlxcel_core::array_shape(input_ids);
        let batch = ids_shape[0];
        let seq_len = ids_shape[1];
        let position_ids = self.mrope_state.with_entry(seq_id, |entry| {
            if let Some(ref stored) = entry.position_ids {
                let pos_shape = mlxcel_core::array_shape(stored);
                if pos_shape.len() == 3
                    && pos_shape[1] == batch
                    && pos_shape[2] >= cache_offset + seq_len
                {
                    return mlxcel_core::slice(
                        stored,
                        &[0, 0, cache_offset],
                        &[pos_shape[0], pos_shape[1], cache_offset + seq_len],
                    );
                }
            }
            Qwen3VLModel::compute_position_ids_with_delta(
                entry.rope_deltas.unwrap_or(0),
                batch,
                seq_len,
                cache_offset,
            )
        });

        let (full_mask, sliding_mask) = self.prefill_masks(seq_len, caches);

        let ds_masks = self.visual_pos_masks.borrow();
        let ds_embeds = self.deepstack_visual_embeds.borrow();

        for layer_idx in 0..self.layers.len() {
            let mask = self.mask_for_layer(layer_idx, &full_mask, &sliding_mask);
            h = self.layers[layer_idx].forward(&h, &mut caches[layer_idx], mask, &position_ids);

            if let (Some(masks), Some(embeds)) = (&*ds_masks, &*ds_embeds)
                && layer_idx < embeds.len()
                && cache_offset == 0
            {
                h = Qwen3VLModel::deepstack_process(&h, masks, &embeds[layer_idx]);
            }
        }

        self.norm.forward(&h)
    }

    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl mlxcel_core::generate::LanguageModel for CohereCompassTextModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, caches)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, input_embeddings, caches)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, None, caches, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, input_embeddings, caches, seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.release_mrope_sequence(seq_id);
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        CohereCompassTextModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "cohere_compass_tests.rs"]
mod cohere_compass_tests;
