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

//! `LanguageModel` wrapper for Limite: the model owns its mixed sliding and
//! global caches per sequence, honors the injected KV cache modes, reports
//! the per-layer storage modes, and batches width-one decode natively.

use super::LimiteModel;
use crate::models::limite_layers::Cache;
use crate::models::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
use mlxcel_core::cache::LayerStateKind;
use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, route_batched_decode_with_ids};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct LimiteWrapper {
    model: LimiteModel,
    sequence_state: ModelOwnedSequenceState<Cache>,
    kv_cache_layer_modes: KvCacheLayerModes,
}

impl LimiteWrapper {
    pub fn new(model: LimiteModel) -> Self {
        let caches = model.make_caches_with_modes(None);
        Self {
            model,
            sequence_state: ModelOwnedSequenceState::new(caches),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn sequence_offsets(&self, id: SequenceId) -> Vec<i32> {
        self.sequence_state
            .with_sequence_state_ref(id, |caches| caches.iter().map(Cache::offset).collect())
            .expect("test sequence exists")
    }

    #[cfg(test)]
    pub(crate) fn sequence_kv_modes(&self, id: SequenceId) -> Vec<KVCacheMode> {
        self.sequence_state
            .with_sequence_state_ref(id, |caches| caches.iter().map(Cache::mode).collect())
            .expect("test sequence exists")
    }

    #[cfg(test)]
    pub(crate) fn sequence_visible_lens(&self, id: SequenceId) -> Vec<usize> {
        self.sequence_state
            .with_sequence_state_ref(id, |caches| caches.iter().map(Cache::visible_len).collect())
            .expect("test sequence exists")
    }

    /// Fresh caches in the KV cache modes the server or CLI injected.
    fn fresh_caches(&self) -> Vec<Cache> {
        self.model
            .make_caches_with_modes(Some(&self.kv_cache_layer_modes))
    }

    pub fn reset_caches(&self) {
        self.sequence_state.replace_internal(self.fresh_caches());
    }
}

impl mlxcel_core::generate::LanguageModel for LimiteWrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_sequence_state(None, |caches| {
            self.model.forward_rows(input_ids, caches, None)
        })
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // The model owns its sliding/global state; the returned slice is a
        // placeholder the engine threads through unused.
        self.reset_caches();
        (0..self.model.num_layers())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.reset_caches();
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    /// Sliding layers hold a `RotatingKVCache` of `sliding_window + 1` keys,
    /// which stores FP16 for every mode but an aligned `Turbo4Asym` (#2783).
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        let window = self.model.sliding_keys;
        self.kv_cache_layer_modes.storage_modes(|layer_idx| {
            self.model
                .layers
                .get(layer_idx)
                .is_some_and(|layer| !layer.attn.is_global)
                .then_some(window)
                .map_or(LayerStateKind::Full, LayerStateKind::sliding)
        })
    }

    fn supports_batching(&self) -> bool {
        true
    }
    fn supports_padded_prefill(&self) -> bool {
        false
    }
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }
    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.model.num_layers())
    }
    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.fresh_caches());
    }
    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }
    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.fresh_caches(),
            |caches| self.model.forward_rows(input_ids, caches, None),
        )
    }
    /// The sigmoid softcap runs on the sampled row only (#3070).
    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // No embeddings path: decline so the default keeps its behavior.
        if input_embeddings.is_some() {
            return None;
        }
        Some(self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.fresh_caches(),
            |caches| self.model.forward_rows(input_ids, caches, Some(last_pos)),
        ))
    }
    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut mlxcel_core::cache::CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_existing_sequence_state(seq_id, |caches| {
                let visible_lens: Vec<_> = caches.iter().map(Cache::visible_len).collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
            })?
    }
    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_ids(input_ids, None, caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(
            shape.len(),
            2,
            "limite batched decode expects rank-2 input ids"
        );
        assert_eq!(
            shape[0] as usize,
            batch_caches.len(),
            "limite input batch must match cache rows"
        );
        if let Some(ids) = seq_ids {
            assert_eq!(
                ids.len(),
                batch_caches.len(),
                "limite requires one sequence id per row"
            );
        }
        route_batched_decode_with_ids(self, input_ids, seq_ids, batch_caches, |batch_caches| {
            let ids = seq_ids.expect("limite model-owned batched decode requires sequence ids");
            if shape[1] != 1 || mask.is_some() {
                self.sequence_state
                    .validate_sequence_states(ids)
                    .unwrap_or_else(|err| panic!("limite batched decode: {err}"));
                let mut outputs = batch_caches.iter_mut().enumerate().map(|(row, caches)| {
                    let input = mlxcel_core::slice(
                        input_ids,
                        &[row as i32, 0],
                        &[row as i32 + 1, shape[1]],
                    );
                    self.forward_with_sequence_id(&input, Some(ids[row]), caches, mask)
                });
                let first = outputs.next().expect("nonempty batch");
                return outputs.fold(first, |out, next| mlxcel_core::concatenate(&out, &next, 0));
            }
            self.sequence_state
                .with_batched_sequence_states(ids, |caches| {
                    self.model.forward_batched_decode(input_ids, caches)
                })
                .unwrap_or_else(|err| panic!("limite batched decode: {err}"))
        })
    }

    fn reset_runtime_state(&self) {
        self.reset_caches();
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }
}
