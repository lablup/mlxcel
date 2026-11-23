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

//! Test driver for a scheduler-owned model without a scheduler (issue #3254).
//!
//! A scheduler-owned family keeps no per-sequence state in its wrapper, so a
//! model-level test can no longer prepare a sequence on the model and forward
//! it by id. [`Pooled`] owns the model and a real [`CachePool`], and exposes
//! the by-id calls those tests were written against: each one resolves the
//! test's sequence id to a pool sequence and goes through the same entry
//! points the engine and the prompt cache use (`forward_layer_state`, the
//! per-layer rewinds, `server::prompt_cache::state_snapshot`). It implements
//! `LanguageModel` itself, so the generic test helpers written over the trait
//! run on it, and derefs to the model for the wrapper's own methods.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;

use mlxcel_core::cache::{
    CachePool, KVCacheMode, LayerState, LayerStateForward, LayerStateKind, LayerStateSnapshotSpec,
    SequenceId, SequenceStateLayout,
};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::server::prompt_cache::state_snapshot;

pub(crate) struct Pooled<M: LanguageModel> {
    model: M,
    pool: RefCell<CachePool>,
    /// The test's sequence ids, mapped to the ids the pool assigned.
    ids: RefCell<HashMap<SequenceId, SequenceId>>,
}

impl<M: LanguageModel> Deref for Pooled<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.model
    }
}

impl<M: LanguageModel> std::ops::DerefMut for Pooled<M> {
    fn deref_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<M: LanguageModel> Pooled<M> {
    pub(crate) fn new(model: M) -> Self {
        assert!(
            model.layer_state_kinds().is_some(),
            "Pooled drives scheduler-owned models"
        );
        Self {
            model,
            pool: RefCell::new(CachePool::new(64)),
            ids: RefCell::new(HashMap::new()),
        }
    }

    /// The pool sequence behind `seq`, opened on first use.
    fn pool_id(&self, seq: SequenceId) -> SequenceId {
        if let Some(id) = self.ids.borrow().get(&seq) {
            return *id;
        }
        let id = self
            .pool
            .borrow_mut()
            .allocate(&self.model)
            .expect("pool sequence");
        self.ids.borrow_mut().insert(seq, id);
        id
    }

    /// Whether `seq` has a pool sequence.
    pub(crate) fn has_sequence_state(&self, seq: SequenceId) -> bool {
        self.ids.borrow().contains_key(&seq)
    }

    /// Read `seq`'s layer state.
    pub(crate) fn with_layer_state<R>(
        &self,
        seq: SequenceId,
        f: impl FnOnce(&[LayerState]) -> R,
    ) -> R {
        let id = self.pool_id(seq);
        let pool = self.pool.borrow();
        f(pool.layer_state(id).expect("scheduler-owned sequence"))
    }

    /// Mutate `seq`'s layer state.
    pub(crate) fn with_layer_state_mut<R>(
        &self,
        seq: SequenceId,
        f: impl FnOnce(&mut [LayerState]) -> R,
    ) -> R {
        let id = self.pool_id(seq);
        let mut pool = self.pool.borrow_mut();
        f(pool.layer_state_mut(id).expect("scheduler-owned sequence"))
    }

    /// The KV cache mode each of `seq`'s attention layers was built with.
    pub(crate) fn sequence_kv_modes(&self, seq: SequenceId) -> Vec<KVCacheMode> {
        self.with_layer_state(seq, |state| {
            state
                .iter()
                .filter_map(|layer| match layer {
                    LayerState::Kv(cache) => Some(cache.mode),
                    LayerState::Rotating(cache) => Some(cache.mode),
                    _ => None,
                })
                .collect()
        })
    }

    fn forward_row(&self, seq: SequenceId, request: LayerStateForward<'_>) -> UniquePtr<MlxArray> {
        let ids = [seq];
        let request = request.with_seq_ids(Some(&ids));
        self.with_layer_state_mut(seq, |state| {
            self.model
                .forward_layer_state(&request, &mut [state])
                .expect("scheduler-owned forward")
        })
    }

    /// The batched forward over the rows' pool state. `Err` is the model's
    /// refusal or the pool's (a duplicate or unknown row).
    pub(crate) fn try_forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        _batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let seqs = seq_ids.ok_or("a scheduler-owned batch needs sequence ids")?;
        if let [seq] = seqs {
            // One row keeps the single-sequence contract, lazy open included.
            return Ok(self.forward_row(*seq, LayerStateForward::tokens(input_ids).with_mask(mask)));
        }
        let ids = self.ids.borrow();
        let pool_ids: Vec<SequenceId> = seqs
            .iter()
            .map(|seq| {
                ids.get(seq).copied().ok_or_else(|| {
                    format!("missing scheduler-owned sequence state for sequence {seq}")
                })
            })
            .collect::<Result<_, _>>()?;
        let mut pool = self.pool.borrow_mut();
        let mut rows = pool.batch_layer_state_mut(&pool_ids)?;
        let request = LayerStateForward::tokens(input_ids)
            .with_mask(mask)
            .with_seq_ids(Some(seqs));
        self.model.forward_layer_state(&request, &mut rows)
    }
}

impl<M: LanguageModel> LanguageModel for Pooled<M> {
    /// Open a fresh pool sequence for `seq`, dropping any earlier one.
    fn prepare_sequence_state(&self, seq: SequenceId) {
        self.release_sequence_state_by_id(seq);
        self.model.prepare_sequence_state(seq);
        let _ = self.pool_id(seq);
    }

    fn release_sequence_state_by_id(&self, seq: SequenceId) {
        if let Some(id) = self.ids.borrow_mut().remove(&seq) {
            self.pool.borrow_mut().release(id);
        }
        self.model.release_sequence_state_by_id(seq);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match seq_id {
            Some(seq) => {
                self.forward_row(seq, LayerStateForward::tokens(input_ids).with_mask(mask))
            }
            None => self
                .model
                .forward_with_sequence_id(input_ids, None, caches, mask),
        }
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        match seq_id {
            Some(seq) => self.forward_row(
                seq,
                LayerStateForward::tokens(input_ids)
                    .with_mask(mask)
                    .with_last_pos(Some(last_pos)),
            ),
            None => self
                .model
                .forward_last_logits_with_sequence_id(input_ids, None, caches, mask, last_pos),
        }
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match seq_id {
            Some(seq) => self.forward_row(
                seq,
                LayerStateForward::tokens(input_ids)
                    .with_embeddings(input_embeddings)
                    .with_mask(mask),
            ),
            None => self.model.forward_with_embeddings_and_sequence_id(
                input_ids,
                input_embeddings,
                None,
                caches,
                mask,
            ),
        }
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        match seq_id {
            Some(seq) => self.forward_row(
                seq,
                LayerStateForward::tokens(input_ids)
                    .with_embeddings(input_embeddings)
                    .with_mask(mask)
                    .with_last_pos(Some(last_pos)),
            ),
            None => self
                .model
                .forward_last_logits_with_embeddings_and_sequence_id(
                    input_ids,
                    input_embeddings,
                    None,
                    caches,
                    mask,
                    last_pos,
                ),
        }
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if seq_ids.is_none() {
            // The id-less entry runs on the model's fallback slot.
            return self
                .model
                .forward_batched_with_ids(input_ids, None, batch_caches, mask);
        }
        self.try_forward_batched_with_ids(input_ids, seq_ids, batch_caches, mask)
            .unwrap_or_else(|err| panic!("scheduler-owned batched forward: {err}"))
    }

    fn snapshot_sequence_state(
        &self,
        seq: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        let id = *self.ids.borrow().get(&seq)?;
        state_snapshot::snapshot_sequence_state(&self.model, &self.pool.borrow(), id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        let id = self.pool_id(seq);
        state_snapshot::restore_sequence_state(
            &self.model,
            &mut self.pool.borrow_mut(),
            id,
            snapshot,
        )
    }

    fn restore_sequence_state_truncated(
        &self,
        seq: SequenceId,
        snapshot: &ModelStateSnapshot,
        target_len: usize,
    ) -> Result<(), String> {
        let id = self.pool_id(seq);
        state_snapshot::restore_sequence_state_truncated(
            &self.model,
            &mut self.pool.borrow_mut(),
            id,
            snapshot,
            target_len,
        )
    }

    fn snapshot_truncatable_to(&self, snapshot: &ModelStateSnapshot, target_len: usize) -> bool {
        state_snapshot::snapshot_truncatable_to(&self.model, snapshot, target_len)
    }

    fn state_checkpoint_from_snapshot(
        &self,
        snapshot: &ModelStateSnapshot,
    ) -> Option<ModelStateSnapshot> {
        state_snapshot::state_checkpoint_from_snapshot(&self.model, snapshot)
    }

    fn restore_sequence_state_from_checkpoint(
        &self,
        seq: SequenceId,
        checkpoint: &ModelStateSnapshot,
        donor: &ModelStateSnapshot,
    ) -> Result<(), String> {
        let id = self.pool_id(seq);
        state_snapshot::restore_sequence_state_from_checkpoint(
            &self.model,
            &mut self.pool.borrow_mut(),
            id,
            checkpoint,
            donor,
        )
    }

    /// The engine's decode rewind: every layer, in order.
    fn rewind_decode_appends(&self, seq: SequenceId, n: i32) -> Result<(), String> {
        if !self.has_sequence_state(seq) {
            return Err(format!(
                "missing scheduler-owned sequence state for sequence {seq}"
            ));
        }
        self.with_layer_state_mut(seq, |state| {
            state
                .iter_mut()
                .enumerate()
                .try_for_each(|(layer_idx, layer)| {
                    layer.rewind_decode_appends(n).map_err(|err| {
                        format!("decode rewind, sequence {seq}, layer {layer_idx}: {err}")
                    })
                })
        })
    }

    /// The engine's pad trim: every layer for a sequence, the model's
    /// fallback slot for `None`.
    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        let Some(seq) = seq else {
            return self.model.trim_state(None, excess);
        };
        if !self.has_sequence_state(seq) {
            return Err(format!(
                "missing scheduler-owned sequence state for sequence {seq}"
            ));
        }
        if excess <= 0 {
            return Ok(());
        }
        self.with_layer_state_mut(seq, |state| {
            state
                .iter_mut()
                .enumerate()
                .try_for_each(|(layer_idx, layer)| {
                    layer
                        .rewind_padded_prefill(excess)
                        .map_err(|err| format!("pad trim, layer {layer_idx}: {err}"))
                })
        })
    }

    /// Mirror `seq`'s layer state lengths into `cache_pool`'s paged shadow,
    /// as the engine does for a scheduler-owned sequence.
    fn sync_sequence_storage(
        &self,
        seq: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        if !self.has_sequence_state(seq) {
            return Ok(());
        }
        let visible: Vec<usize> = self.with_layer_state(seq, |state| {
            state.iter().map(LayerState::visible_len).collect()
        });
        cache_pool.sync_paged_state_with_lengths(seq, &visible)
    }

    // Everything that is not per-sequence state goes to the model. Each
    // method the migrated families override is listed, because a method
    // left out would answer with the trait default instead of the model's.
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.model
            .forward_last_logits(input_ids, caches, mask, last_pos)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.model.forward_batched(input_ids, batch_caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        self.model.embed_tokens(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids()
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.model.set_kv_cache_layer_modes(modes);
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.model.kv_cache_layer_modes()
    }

    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        self.model.kv_cache_storage_modes()
    }

    fn layer_state_kinds(&self) -> Option<Vec<LayerStateKind>> {
        self.model.layer_state_kinds()
    }

    fn make_layer_state(&self) -> Option<Vec<LayerState>> {
        self.model.make_layer_state()
    }

    fn layer_state_snapshot_spec(&self) -> Option<LayerStateSnapshotSpec> {
        self.model.layer_state_snapshot_spec()
    }

    fn forward_layer_state(
        &self,
        request: &LayerStateForward<'_>,
        rows: &mut [&mut [LayerState]],
    ) -> Result<UniquePtr<MlxArray>, String> {
        self.model.forward_layer_state(request, rows)
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        self.model.sequence_state_layout()
    }

    fn supports_batching(&self) -> bool {
        self.model.supports_batching()
    }

    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        self.model.batched_decode_kind()
    }

    fn supports_batched_prefill(&self) -> bool {
        self.model.supports_batched_prefill()
    }

    fn supports_padded_prefill(&self) -> bool {
        self.model.supports_padded_prefill()
    }

    fn supports_paged_decode_backend(&self) -> bool {
        self.model.supports_paged_decode_backend()
    }

    fn supports_decode_lookahead_rewind(&self) -> bool {
        self.model.supports_decode_lookahead_rewind()
    }

    fn supports_snapshot_reuse(&self) -> bool {
        self.model.supports_snapshot_reuse()
    }

    fn reset_runtime_state(&self) {
        self.model.reset_runtime_state();
    }
}
