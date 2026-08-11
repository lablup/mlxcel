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

//! Muse Glimmer text decoder (`text_config.model_type: "muse_glimmer_text"`).
//!
//! The decoder owns a mixed cache layout: sliding layers use rotating caches
//! and full-attention layers use growing KV caches. Scheduler calls must carry
//! `SequenceId`s so each row resolves to an isolated model-owned cache vector.

use crate::models::model_owned::ModelOwnedSequenceState;
use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::{KVCache, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

pub(crate) use super::muse_glimmer_cache::MuseCache;
pub use super::muse_glimmer_config::{
    DEFAULT_IMAGE_END_TOKEN_ID, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, DEFAULT_IMAGE_START_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID, DEFAULT_PAD_TOKEN_ID, DEFAULT_VIDEO_TOKEN_ID, MuseGlimmerConfig,
    MuseGlimmerTextConfig, MuseGlimmerVisionConfig,
};
pub(crate) use super::muse_glimmer_layers::{MuseGlimmerDecoderLayer, MuseRmsNorm};

pub struct MuseGlimmerTextModel {
    embed_tokens: UnifiedEmbedding,
    embed_norm: MuseRmsNorm,
    layers: Vec<MuseGlimmerDecoderLayer>,
    norm: MuseRmsNorm,
    lm_head: UnifiedLinear,
    sliding_window: usize,
    eos_token_ids: Vec<i32>,
    suppressed_token_ids: Vec<i32>,
    output_multiplier: f32,
    final_logit_softcapping: Option<f32>,
}

impl MuseGlimmerTextModel {
    pub fn forward_with_muse_caches(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [MuseCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed_input_ids(input_ids),
        };
        for (idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[idx], mask);
        }
        h = self.norm.forward(&h);
        let logits = self.lm_head.forward(&h);
        Self::softcap_logits(
            &logits,
            self.output_multiplier,
            self.final_logit_softcapping,
        )
    }

    pub fn softcap_logits(
        logits: &MlxArray,
        output_multiplier: f32,
        final_logit_softcapping: Option<f32>,
    ) -> UniquePtr<MlxArray> {
        let scaled = mlxcel_core::multiply_scalar(logits, output_multiplier);
        match final_logit_softcapping {
            Some(cap) if cap > 0.0 => {
                let divided = mlxcel_core::multiply_scalar(&scaled, 1.0 / cap);
                let capped = mlxcel_core::tanh(&divided);
                mlxcel_core::multiply_scalar(&capped, cap)
            }
            _ => scaled,
        }
    }

    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_input_ids(input_ids)
    }

    fn embed_input_ids(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embeddings = self.embed_tokens.forward(input_ids);
        self.embed_norm.forward(&embeddings)
    }

    pub fn make_muse_caches(&self) -> Vec<MuseCache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.use_sliding {
                    MuseCache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                } else {
                    MuseCache::Standard(KVCache::new())
                }
            })
            .collect()
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerTextConfig,
        model_prefix: &str,
        lm_head_prefix: &str,
        eos_token_ids: Vec<i32>,
        suppressed_token_ids: Vec<i32>,
    ) -> Result<Self, String> {
        config.validate()?;
        if config.tie_word_embeddings {
            return Err(
                "Muse Glimmer requires an untied lm_head in the published checkpoint".to_string(),
            );
        }

        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{model_prefix}.embed_tokens"),
            config.group_size(),
            config.bits(),
        )?;
        let embed_norm = MuseRmsNorm::no_weight(config.rms_norm_eps);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MuseGlimmerDecoderLayer::from_weights(
                weights,
                config,
                i,
                model_prefix,
                get_weight_copy,
            )?);
        }
        let norm = MuseRmsNorm::standard(
            get_weight_copy(weights, &format!("{model_prefix}.norm.weight"))?,
            config.rms_norm_eps,
        );
        let lm_head = UnifiedLinear::from_weights(
            weights,
            lm_head_prefix,
            config.group_size(),
            config.bits(),
        )?;

        Ok(Self {
            embed_tokens,
            embed_norm,
            layers,
            norm,
            lm_head,
            sliding_window: config.sliding_window,
            eos_token_ids,
            suppressed_token_ids,
            output_multiplier: config.output_multiplier,
            final_logit_softcapping: config.final_logit_softcapping,
        })
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, MuseGlimmerTextConfig), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config: MuseGlimmerConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let eos_token_ids = crate::loading::read_eos_token_ids(model_dir);
        let model = Self::from_weights(
            &weights,
            &config.text_config,
            "model.language_model",
            "lm_head",
            if eos_token_ids.is_empty() {
                vec![200_001, 200_008]
            } else {
                eos_token_ids
            },
            vec![
                config.image_token_id.unwrap_or(DEFAULT_IMAGE_TOKEN_ID),
                config.video_token_id.unwrap_or(DEFAULT_VIDEO_TOKEN_ID),
                DEFAULT_PAD_TOKEN_ID,
            ],
        )?;
        Ok((model, config.text_config))
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

pub struct MuseGlimmerTextWrapper {
    model: MuseGlimmerTextModel,
    sequence_state: ModelOwnedSequenceState<MuseCache>,
}

impl MuseGlimmerTextWrapper {
    pub fn new(model: MuseGlimmerTextModel) -> Self {
        let caches = model.make_muse_caches();
        Self {
            model,
            sequence_state: ModelOwnedSequenceState::new(caches),
        }
    }

    fn with_state(
        &self,
        seq_id: Option<SequenceId>,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_muse_caches(),
            |state| {
                self.model
                    .forward_with_muse_caches(input_ids, input_embeddings, state, mask)
            },
        )
    }

    fn forward_batched_without_sequence_ids(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let b = batch_caches.len();
        if b == 0 {
            return mlxcel_core::zeros(&[0, 1, 1], mlxcel_core::dtype::FLOAT32);
        }
        let shape = mlxcel_core::array_shape(input_ids);
        let row_len = if shape.len() >= 2 { shape[1] } else { 1 };
        if b == 1 {
            return self.with_state(None, input_ids, None, mask);
        }

        let token_0 = mlxcel_core::slice(input_ids, &[0, 0], &[1, row_len]);
        let mut row_caches = self.model.make_muse_caches();
        let mut result = self
            .model
            .forward_with_muse_caches(&token_0, None, &mut row_caches, None);
        for i in 1..b {
            let token_i = mlxcel_core::slice(input_ids, &[i as i32, 0], &[i as i32 + 1, row_len]);
            let mut row_caches = self.model.make_muse_caches();
            let logits_i =
                self.model
                    .forward_with_muse_caches(&token_i, None, &mut row_caches, None);
            result = mlxcel_core::concatenate(&result, &logits_i, 0);
        }
        result
    }

    fn forward_batched_with_sequence_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: &[SequenceId],
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let b = batch_caches.len();
        if b == 0 {
            return mlxcel_core::zeros(&[0, 1, 1], mlxcel_core::dtype::FLOAT32);
        }
        if seq_ids.len() != b {
            tracing::warn!(
                seq_ids = seq_ids.len(),
                batch = b,
                "Muse Glimmer batched decode called with mismatched sequence ids"
            );
            return self.forward_batched_without_sequence_ids(input_ids, batch_caches, mask);
        }

        let shape = mlxcel_core::array_shape(input_ids);
        let row_len = if shape.len() >= 2 { shape[1] } else { 1 };
        if b == 1 {
            return self.with_state(seq_ids.first().copied(), input_ids, None, mask);
        }

        let token_0 = mlxcel_core::slice(input_ids, &[0, 0], &[1, row_len]);
        let mut result = self.with_state(seq_ids.first().copied(), &token_0, None, None);
        for (i, seq_id) in seq_ids.iter().copied().enumerate().skip(1) {
            let token_i = mlxcel_core::slice(input_ids, &[i as i32, 0], &[i as i32 + 1, row_len]);
            let logits_i = self.with_state(Some(seq_id), &token_i, None, None);
            result = mlxcel_core::concatenate(&result, &logits_i, 0);
        }
        result
    }

    #[cfg(test)]
    pub(crate) fn sequence_cache_summaries(
        &self,
        seq_id: SequenceId,
    ) -> Option<Vec<(bool, i32, i32)>> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                state
                    .iter()
                    .map(|cache| (cache.is_sliding(), cache.offset(), cache.live_len()))
                    .collect()
            })
    }
}

impl LanguageModel for MuseGlimmerTextWrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.with_state(None, input_ids, None, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.with_state(None, input_ids, input_embeddings, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.with_state(seq_id, input_ids, None, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.with_state(seq_id, input_ids, input_embeddings, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(self.model.make_muse_caches());
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        self.model.suppressed_token_ids.clone()
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.model.get_embed_tokens(input_ids))
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.model.layers.len())
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.model.make_muse_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn reset_runtime_state(&self) {
        self.sequence_state
            .replace_internal(self.model.make_muse_caches());
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot = ModelStateSnapshot::new("muse_glimmer", token_len);
                for (idx, cache) in state.iter().enumerate() {
                    if let Err(error) = cache.snapshot_into(&mut snapshot, &format!("layer{idx}")) {
                        tracing::warn!(
                            error,
                            layer_idx = idx,
                            "Muse Glimmer snapshot prompt-cache donation skipped"
                        );
                        return None;
                    }
                }
                if snapshot.is_empty() {
                    None
                } else {
                    Some(snapshot)
                }
            })
            .flatten()
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != "muse_glimmer" {
            return Err(format!(
                "cannot restore {} snapshot into Muse Glimmer",
                snapshot.family()
            ));
        }
        let mut state = self.model.make_muse_caches();
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"))?;
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_without_sequence_ids(input_ids, batch_caches, mask)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        _context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_without_sequence_ids(input_ids, batch_caches, mask)
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        _context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        if let Some(seq_ids) = seq_ids {
            return self.forward_batched_with_sequence_ids(input_ids, seq_ids, batch_caches, mask);
        }
        self.forward_batched_without_sequence_ids(input_ids, batch_caches, mask)
    }
}

#[cfg(test)]
#[path = "muse_glimmer_tests.rs"]
mod tests;
