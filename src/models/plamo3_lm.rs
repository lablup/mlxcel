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

//! [`LanguageModel`] for PLaMo 3: model-owned hybrid sliding/full state,
//! native batched decode, last-row prefill logits and model-state snapshots.
//! The state bodies are AFMoE's, shared through
//! [`crate::models::sliding_hybrid_lm`].

use mlxcel_core::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma3::Cache;
use crate::models::kv_snapshot::KvSnapshotNames;
use crate::models::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
use crate::models::plamo3::Plamo3Model;
use crate::models::sliding_hybrid_lm::{self as hybrid, SlidingHybridModel};

impl SlidingHybridModel for Plamo3Model {
    /// A PLaMo 3 snapshot never restores into another family that shares
    /// the `Cache` enum.
    const FAMILY: &'static str = "plamo3";
    const KV_NAMES: KvSnapshotNames = KvSnapshotNames::new("PLaMo 3", "standard", "rotating");

    fn hybrid_sequence_state(&self) -> &ModelOwnedSequenceState<Cache> {
        &self.sequence_state
    }

    fn hybrid_kv_cache_layer_modes(&self) -> &KvCacheLayerModes {
        &self.kv_cache_layer_modes
    }

    fn hybrid_make_internal_caches(&self) -> Vec<Cache> {
        self.make_internal_caches()
    }

    fn hybrid_num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hybrid_layer_uses_sliding(&self, layer_idx: usize) -> bool {
        !self.layers[layer_idx].is_full
    }

    fn hybrid_sliding_window(&self) -> i32 {
        self.sliding_window
    }

    fn hybrid_forward_batched_decode(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_decode_with_caches(input_ids, caches)
    }
}

impl LanguageModel for Plamo3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, None, None)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, seq_id, None)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        _input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Text-only family: there is no image-embedding prefill path.
        self.forward_for_sequence(input_ids, seq_id, None)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        _input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        Some(self.forward_for_sequence(input_ids, seq_id, Some(last_pos)))
    }

    fn reset_runtime_state(&self) {
        hybrid::reset_runtime_state(self)
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        hybrid::sequence_state_layout(self)
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        hybrid::prepare_sequence_state(self, seq_id)
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        hybrid::sync_sequence_storage(self, seq_id, cache_pool)
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        hybrid::set_kv_cache_layer_modes(self, modes)
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    /// Sliding layers hold a `RotatingKVCache`, which stores FP16 for every
    /// mode but an aligned `Turbo4Asym` (issue #2783).
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        hybrid::kv_cache_storage_modes(self)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        hybrid::snapshot_sequence_state(self, seq_id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        hybrid::restore_sequence_state(self, seq_id, snapshot)
    }

    fn snapshot_truncatable_to(&self, snapshot: &ModelStateSnapshot, target_len: usize) -> bool {
        hybrid::snapshot_truncatable_to(self, snapshot, target_len)
    }

    fn restore_sequence_state_truncated(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
        target_len: usize,
    ) -> Result<(), String> {
        hybrid::restore_sequence_state_truncated(self, seq_id, snapshot, target_len)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        hybrid::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn batched_decode_kind(&self) -> mlxcel_core::generate::BatchedDecodeKind {
        mlxcel_core::generate::BatchedDecodeKind::Native
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
        hybrid::forward_batched_with_ids(self, input_ids, seq_ids, batch_caches, mask)
    }

    /// Opt out of tile-aligned padded prefill for the same reason as AFMoE
    /// (issue #1335): the scheduler's post-pad trim reaches only `CachePool`
    /// caches, and this family's state is model-owned.
    fn supports_padded_prefill(&self) -> bool {
        false
    }

    /// An id-less multi-token forward starts an independent prompt by
    /// resetting the model-owned caches, so token-ID chunking would lose state
    /// for callers that chunk without ids (the same contract as AFMoE).
    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
