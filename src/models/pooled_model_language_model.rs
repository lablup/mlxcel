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

//! `LanguageModel` for [`Pooled`]: the by-id entries over the pool, and
//! everything else delegated to the model.

use mlxcel_core::cache::{
    CachePool, KVCacheMode, LayerState, LayerStateForward, LayerStateKind, LayerStateSnapshotSpec,
    SequenceId, SequenceStateLayout,
};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Pooled;
use crate::server::prompt_cache::state_snapshot;

impl<M: LanguageModel> LanguageModel for Pooled<M> {
    /// Open a fresh pool sequence for `seq`, dropping any earlier one.
    ///
    /// Only the pool entry is dropped. The model's own per-sequence side
    /// state (Qwen 3.5's MRoPE positions, installed before this call) stays,
    /// as the model-owned prepare left it.
    fn prepare_sequence_state(&self, seq: SequenceId) {
        if !self.scheduler_owned {
            return self.model.prepare_sequence_state(seq);
        }
        if let Some(id) = self.ids.borrow_mut().remove(&seq) {
            self.pool.borrow_mut().release(id);
        }
        self.model.prepare_sequence_state(seq);
        let _ = self.pool_id(seq);
    }

    fn release_sequence_state_by_id(&self, seq: SequenceId) {
        if !self.scheduler_owned {
            return self.model.release_sequence_state_by_id(seq);
        }
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
        if !self.scheduler_owned {
            return self
                .model
                .forward_with_sequence_id(input_ids, seq_id, caches, mask);
        }
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
        if !self.scheduler_owned {
            return self
                .model
                .forward_last_logits_with_sequence_id(input_ids, seq_id, caches, mask, last_pos);
        }
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
        if !self.scheduler_owned {
            return self.model.forward_with_embeddings_and_sequence_id(
                input_ids,
                input_embeddings,
                seq_id,
                caches,
                mask,
            );
        }
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
        if !self.scheduler_owned {
            return self
                .model
                .forward_last_logits_with_embeddings_and_sequence_id(
                    input_ids,
                    input_embeddings,
                    seq_id,
                    caches,
                    mask,
                    last_pos,
                );
        }
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
        if !self.scheduler_owned {
            return self
                .model
                .forward_batched_with_ids(input_ids, seq_ids, batch_caches, mask);
        }
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
        if !self.scheduler_owned {
            return self.model.snapshot_sequence_state(seq, token_len);
        }
        let id = *self.ids.borrow().get(&seq)?;
        state_snapshot::snapshot_sequence_state(&self.model, &self.pool.borrow(), id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        if !self.scheduler_owned {
            return self.model.restore_sequence_state(seq, snapshot);
        }
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
        if !self.scheduler_owned {
            return self
                .model
                .restore_sequence_state_truncated(seq, snapshot, target_len);
        }
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
        if !self.scheduler_owned {
            return self
                .model
                .restore_sequence_state_from_checkpoint(seq, checkpoint, donor);
        }
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
        if !self.scheduler_owned {
            return self.model.rewind_decode_appends(seq, n);
        }
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
        if !self.scheduler_owned {
            return self.model.trim_state(seq, excess);
        }
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
        if !self.scheduler_owned {
            return self.model.sync_sequence_storage(seq, cache_pool);
        }
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

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        self.model.output_suppressed_token_ids()
    }

    fn supports_chunked_prefill(&self) -> bool {
        self.model.supports_chunked_prefill()
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        self.model.supports_maskless_padded_prefill()
    }

    fn supports_speculative_verify_rewind(&self) -> bool {
        self.model.supports_speculative_verify_rewind()
    }

    fn kv_cache_captures_sequence_state(&self) -> bool {
        self.model.kv_cache_captures_sequence_state()
    }

    fn rope_table_regime(&self, total_prompt_len: usize) -> Option<u8> {
        self.model.rope_table_regime(total_prompt_len)
    }

    fn after_prefill(&self) {
        self.model.after_prefill();
    }

    fn release_sequence_state(&self, caches: &mut [KVCache]) {
        self.model.release_sequence_state(caches);
    }
}
