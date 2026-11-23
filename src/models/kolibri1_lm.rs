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

//! [`LanguageModel`] for Kolibri-1: model-owned hybrid sliding/full state,
//! native batched decode, last-row prefill logits and model-state snapshots,
//! on the same contracts as [`crate::models::afmoe`].

use mlxcel_core::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma3::Cache;
use crate::models::kolibri1::Kolibri1Model;
use crate::models::kv_snapshot::KvSnapshotNames;
use mlxcel_core::cache::LayerStateKind;

/// The snapshot family tag; a Kolibri-1 snapshot never restores into another
/// family that shares the `Cache` enum.
const FAMILY: &str = "kolibri1";

const KOLIBRI1_KV_SNAPSHOT_NAMES: KvSnapshotNames =
    KvSnapshotNames::new("Kolibri-1", "standard", "rotating");

impl LanguageModel for Kolibri1Model {
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
        self.sequence_state
            .replace_internal(self.make_internal_caches());
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.layers.len())
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_internal_caches());
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |caches| {
                let visible_lens: Vec<_> = caches.iter().map(Cache::visible_len).collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
            })
            .unwrap_or_else(|| {
                Err(format!(
                    "missing Kolibri-1 sequence state while synchronizing sequence {seq_id}"
                ))
            })
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.sequence_state
            .replace_internal(self.make_internal_caches());
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    /// Sliding layers hold a `RotatingKVCache`, which stores FP16 for every
    /// mode but an aligned `Turbo4Asym`; the 513-token window is not aligned,
    /// so on this family they always store FP16 (issue #2783).
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        self.kv_cache_layer_modes.storage_modes(|layer_idx| {
            self.layers
                .get(layer_idx)
                .is_some_and(|layer| layer.uses_sliding)
                .then_some(self.sliding_window)
                .map_or(LayerStateKind::Full, LayerStateKind::sliding)
        })
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
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot = ModelStateSnapshot::new(FAMILY, token_len);
                for (idx, cache) in state.iter().enumerate() {
                    if let Err(error) = cache.snapshot_into(
                        &mut snapshot,
                        &format!("layer{idx}"),
                        KOLIBRI1_KV_SNAPSHOT_NAMES,
                    ) {
                        tracing::warn!(
                            error,
                            layer_idx = idx,
                            "Kolibri-1 snapshot prompt-cache donation skipped"
                        );
                        return None;
                    }
                }
                (!snapshot.is_empty()).then_some(snapshot)
            })
            .flatten()
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != FAMILY {
            return Err(format!(
                "cannot restore {} snapshot into Kolibri-1",
                snapshot.family()
            ));
        }
        let mut state = self.make_internal_caches();
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"), KOLIBRI1_KV_SNAPSHOT_NAMES)?;
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    /// A truncating restore is sound only while every sliding layer is still
    /// unwrapped at `target_len`; checked per layer, since one wrapped ring is
    /// enough to make it unsound.
    fn snapshot_truncatable_to(&self, snapshot: &ModelStateSnapshot, target_len: usize) -> bool {
        if snapshot.family() != FAMILY || target_len > snapshot.token_len() {
            return false;
        }
        let Ok(target) = i32::try_from(target_len) else {
            return false;
        };
        (0..self.layers.len()).all(|idx| {
            Cache::snapshot_truncatable_to(
                snapshot,
                &format!("layer{idx}"),
                target,
                KOLIBRI1_KV_SNAPSHOT_NAMES,
            )
        })
    }

    fn restore_sequence_state_truncated(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
        target_len: usize,
    ) -> Result<(), String> {
        if snapshot.family() != FAMILY {
            return Err(format!(
                "cannot restore {} snapshot into Kolibri-1",
                snapshot.family()
            ));
        }
        let target = i32::try_from(target_len).map_err(|_| {
            format!("Kolibri-1 truncated restore: target {target_len} out of range")
        })?;
        // Re-check rather than trust the caller: an error here falls back to a
        // cold prefill, a partially truncated state would corrupt generation.
        if !self.snapshot_truncatable_to(snapshot, target_len) {
            return Err(format!(
                "Kolibri-1 truncated restore: snapshot cannot be truncated to {target_len} tokens"
            ));
        }
        let mut state = self.make_internal_caches();
        for (idx, cache) in state.iter_mut().enumerate() {
            let prefix = format!("layer{idx}");
            cache.restore_from(snapshot, &prefix, KOLIBRI1_KV_SNAPSHOT_NAMES)?;
            cache
                .truncate_to(target, KOLIBRI1_KV_SNAPSHOT_NAMES)
                .map_err(|err| format!("{prefix}: {err}"))?;
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Compatibility only: the real state lives in `sequence_state`.
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
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
        let shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(shape.len(), 2, "kolibri1 batched input must be rank 2");
        assert_eq!(
            shape[0] as usize,
            batch_caches.len(),
            "kolibri1 input batch must match cache rows"
        );
        if let Some(ids) = seq_ids {
            assert_eq!(
                ids.len(),
                batch_caches.len(),
                "kolibri1 requires one sequence id per row"
            );
        }
        mlxcel_core::generate::route_batched_decode_with_ids(
            self,
            input_ids,
            seq_ids,
            batch_caches,
            |batch_caches| {
                let ids =
                    seq_ids.expect("kolibri1 model-owned batched decode requires sequence ids");
                if shape[1] != 1 || mask.is_some() {
                    self.sequence_state
                        .validate_sequence_states(ids)
                        .unwrap_or_else(|error| panic!("kolibri1 batched decode: {error}"));
                    let outputs: Vec<_> = batch_caches
                        .iter_mut()
                        .enumerate()
                        .map(|(row, caches)| {
                            let input = mlxcel_core::utils::slice_axis(
                                input_ids,
                                0,
                                row as i32,
                                row as i32 + 1,
                            );
                            self.forward_with_sequence_id(&input, Some(ids[row]), caches, mask)
                        })
                        .collect();
                    let output_refs: Vec<_> = outputs.iter().map(|output| &**output).collect();
                    return mlxcel_core::concatenate_many(&output_refs, 0);
                }
                self.sequence_state
                    .with_batched_sequence_states(ids, |states| {
                        self.forward_batched_decode_with_caches(input_ids, states)
                    })
                    .unwrap_or_else(|error| panic!("kolibri1 batched decode: {error}"))
            },
        )
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
