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

//! Shared [`LanguageModel`] bodies for the model-owned sliding/full hybrids.
//!
//! AFMoE and Kolibri-1 keep their heterogeneous per-layer caches (a
//! `RotatingKVCache` on sliding layers, a `KVCache` on full layers, both held
//! in the `gemma3::Cache` enum) in a [`ModelOwnedSequenceState`], and their
//! `LanguageModel` impls differed only in the snapshot family tag and the
//! label in messages. The bodies live here, parameterized by
//! [`SlidingHybridModel::FAMILY`] and [`SlidingHybridModel::KV_NAMES`], and
//! each impl delegates to them (issue #3297).
//!
//! Used by: AFMoE (`src/models/afmoe.rs`), Kolibri-1
//! (`src/models/kolibri1_lm.rs`) and PLaMo 3 (`src/models/plamo3_lm.rs`).

use mlxcel_core::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma3::Cache;
use crate::models::kv_ring_checkpoint;
use crate::models::kv_snapshot::KvSnapshotNames;
use crate::models::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
use mlxcel_core::cache::LayerStateKind;

/// What a sliding/full hybrid exposes so the shared bodies can drive it.
///
/// The accessor names carry a `hybrid_` prefix so they never collide with a
/// `LanguageModel` method or the model's own inherent methods.
pub(crate) trait SlidingHybridModel: LanguageModel {
    /// Snapshot family tag (`"afmoe"`, `"kolibri1"`), also the prefix of the
    /// batched-decode panic messages. A snapshot never restores into a family
    /// with another tag, even one that shares the `Cache` enum.
    const FAMILY: &'static str;

    /// Snapshot tensor names; `KV_NAMES.family` is the label used in errors
    /// and log lines (`"AFMoE"`, `"Kolibri-1"`).
    const KV_NAMES: KvSnapshotNames;

    /// The model-owned per-sequence caches.
    fn hybrid_sequence_state(&self) -> &ModelOwnedSequenceState<Cache>;

    /// The injected per-layer KV cache modes.
    fn hybrid_kv_cache_layer_modes(&self) -> &KvCacheLayerModes;

    /// A fresh set of per-layer caches, built with the injected modes.
    fn hybrid_make_internal_caches(&self) -> Vec<Cache>;

    /// Number of decoder layers.
    fn hybrid_num_layers(&self) -> usize;

    /// Whether layer `layer_idx` is a sliding-window layer.
    fn hybrid_layer_uses_sliding(&self, layer_idx: usize) -> bool;

    /// The sliding window, in tokens.
    fn hybrid_sliding_window(&self) -> i32;

    /// Width-one batched decode over the rows' own caches.
    fn hybrid_forward_batched_decode(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray>;
}

/// [`LanguageModel::reset_runtime_state`].
pub(crate) fn reset_runtime_state<M: SlidingHybridModel>(model: &M) {
    model
        .hybrid_sequence_state()
        .replace_internal(model.hybrid_make_internal_caches());
}

/// [`LanguageModel::sequence_state_layout`].
pub(crate) fn sequence_state_layout<M: SlidingHybridModel>(model: &M) -> SequenceStateLayout {
    SequenceStateLayout::model_owned(model.hybrid_num_layers())
}

/// [`LanguageModel::prepare_sequence_state`].
pub(crate) fn prepare_sequence_state<M: SlidingHybridModel>(model: &M, seq_id: SequenceId) {
    model
        .hybrid_sequence_state()
        .prepare_sequence_state(seq_id, model.hybrid_make_internal_caches());
}

/// [`LanguageModel::sync_sequence_storage`].
pub(crate) fn sync_sequence_storage<M: SlidingHybridModel>(
    model: &M,
    seq_id: SequenceId,
    cache_pool: &mut CachePool,
) -> Result<(), String> {
    model
        .hybrid_sequence_state()
        .with_sequence_state_ref(seq_id, |caches| {
            let visible_lens: Vec<_> = caches.iter().map(Cache::visible_len).collect();
            cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
        })
        .unwrap_or_else(|| {
            Err(format!(
                "missing {} sequence state while synchronizing sequence {seq_id}",
                M::KV_NAMES.family
            ))
        })
}

/// [`LanguageModel::set_kv_cache_layer_modes`]: store the table and rebuild
/// the internal caches with it.
pub(crate) fn set_kv_cache_layer_modes<M: SlidingHybridModel>(model: &M, modes: Vec<KVCacheMode>) {
    model.hybrid_kv_cache_layer_modes().set(modes);
    reset_runtime_state(model);
}

/// [`LanguageModel::kv_cache_storage_modes`]. Sliding layers hold a
/// `RotatingKVCache`, which stores FP16 for every mode but an aligned
/// `Turbo4Asym` (issue #2783).
pub(crate) fn kv_cache_storage_modes<M: SlidingHybridModel>(
    model: &M,
) -> Option<Vec<Option<KVCacheMode>>> {
    model
        .hybrid_kv_cache_layer_modes()
        .storage_modes(|layer_idx| {
            (layer_idx < model.hybrid_num_layers() && model.hybrid_layer_uses_sliding(layer_idx))
                .then_some(model.hybrid_sliding_window())
                .map_or(LayerStateKind::Full, LayerStateKind::sliding)
        })
}

/// [`LanguageModel::snapshot_sequence_state`]. Returns `None` (no donation)
/// when any layer cannot be snapshotted or the state is empty.
pub(crate) fn snapshot_sequence_state<M: SlidingHybridModel>(
    model: &M,
    seq_id: SequenceId,
    token_len: usize,
) -> Option<ModelStateSnapshot> {
    model
        .hybrid_sequence_state()
        .with_sequence_state_ref(seq_id, |state| {
            let mut snapshot = ModelStateSnapshot::new(M::FAMILY, token_len);
            for (idx, cache) in state.iter().enumerate() {
                if let Err(error) =
                    cache.snapshot_into(&mut snapshot, &format!("layer{idx}"), M::KV_NAMES)
                {
                    tracing::warn!(
                        error,
                        layer_idx = idx,
                        "{} snapshot prompt-cache donation skipped",
                        M::KV_NAMES.family
                    );
                    return None;
                }
            }
            (!snapshot.is_empty()).then_some(snapshot)
        })
        .flatten()
}

fn check_family<M: SlidingHybridModel>(snapshot: &ModelStateSnapshot) -> Result<(), String> {
    if snapshot.family() == M::FAMILY {
        Ok(())
    } else {
        Err(format!(
            "cannot restore {} snapshot into {}",
            snapshot.family(),
            M::KV_NAMES.family
        ))
    }
}

/// [`LanguageModel::restore_sequence_state`].
pub(crate) fn restore_sequence_state<M: SlidingHybridModel>(
    model: &M,
    seq_id: SequenceId,
    snapshot: &ModelStateSnapshot,
) -> Result<(), String> {
    check_family::<M>(snapshot)?;
    let mut state = model.hybrid_make_internal_caches();
    for (idx, cache) in state.iter_mut().enumerate() {
        cache.restore_from(snapshot, &format!("layer{idx}"), M::KV_NAMES)?;
    }
    model
        .hybrid_sequence_state()
        .replace_sequence_state(seq_id, state);
    Ok(())
}

/// [`LanguageModel::snapshot_truncatable_to`]. Both layer kinds are ordinary
/// KV state, so a truncating restore is sound whenever every sliding layer is
/// still unwrapped at `target_len`. The check is per layer at the requested
/// target rather than a conversation-length heuristic: one wrapped sliding
/// layer is enough to make the restore unsound.
pub(crate) fn snapshot_truncatable_to<M: SlidingHybridModel>(
    model: &M,
    snapshot: &ModelStateSnapshot,
    target_len: usize,
) -> bool {
    if snapshot.family() != M::FAMILY || target_len > snapshot.token_len() {
        return false;
    }
    let Ok(target) = i32::try_from(target_len) else {
        return false;
    };
    (0..model.hybrid_num_layers()).all(|idx| {
        Cache::snapshot_truncatable_to(snapshot, &format!("layer{idx}"), target, M::KV_NAMES)
    })
}

/// [`LanguageModel::restore_sequence_state_truncated`].
pub(crate) fn restore_sequence_state_truncated<M: SlidingHybridModel>(
    model: &M,
    seq_id: SequenceId,
    snapshot: &ModelStateSnapshot,
    target_len: usize,
) -> Result<(), String> {
    check_family::<M>(snapshot)?;
    let label = M::KV_NAMES.family;
    let target = i32::try_from(target_len)
        .map_err(|_| format!("{label} truncated restore: target {target_len} out of range"))?;
    // Re-check rather than trust the caller: installing a partially truncated
    // state would corrupt generation silently, while an error here just falls
    // back to a cold prefill.
    if !snapshot_truncatable_to(model, snapshot, target_len) {
        return Err(format!(
            "{label} truncated restore: snapshot cannot be truncated to {target_len} tokens"
        ));
    }
    let mut state = model.hybrid_make_internal_caches();
    for (idx, cache) in state.iter_mut().enumerate() {
        let prefix = format!("layer{idx}");
        cache.restore_from(snapshot, &prefix, M::KV_NAMES)?;
        cache
            .truncate_to(target, M::KV_NAMES)
            .map_err(|err| format!("{prefix}: {err}"))?;
    }
    model
        .hybrid_sequence_state()
        .replace_sequence_state(seq_id, state);
    Ok(())
}

/// [`LanguageModel::state_checkpoint_from_snapshot`]: a ring-only checkpoint
/// (issue #3255) holding the sliding layers' rings, which a longer snapshot
/// of the same conversation cannot give back once they have wrapped.
pub(crate) fn state_checkpoint_from_snapshot<M: SlidingHybridModel>(
    model: &M,
    snapshot: &ModelStateSnapshot,
) -> Option<ModelStateSnapshot> {
    kv_ring_checkpoint::ring_checkpoint(snapshot, M::FAMILY, model.hybrid_num_layers(), M::KV_NAMES)
}

/// [`LanguageModel::restore_sequence_state_from_checkpoint`]: the sliding
/// layers from `checkpoint`, the global layers from `donor`.
pub(crate) fn restore_sequence_state_from_checkpoint<M: SlidingHybridModel>(
    model: &M,
    seq_id: SequenceId,
    checkpoint: &ModelStateSnapshot,
    donor: &ModelStateSnapshot,
) -> Result<(), String> {
    let target = kv_ring_checkpoint::checkpoint_target(checkpoint, donor, M::FAMILY, M::KV_NAMES)?;
    let mut state = model.hybrid_make_internal_caches();
    for (idx, cache) in state.iter_mut().enumerate() {
        cache.restore_from_ring_checkpoint(
            checkpoint,
            donor,
            &format!("layer{idx}"),
            target,
            M::KV_NAMES,
        )?;
    }
    model
        .hybrid_sequence_state()
        .replace_sequence_state(seq_id, state);
    Ok(())
}

/// [`LanguageModel::make_caches`]: compatibility only, the real state lives in
/// the model-owned sequence state.
pub(crate) fn make_caches<M: SlidingHybridModel>(model: &M) -> Vec<KVCache> {
    (0..model.hybrid_num_layers())
        .map(|_| KVCache::new())
        .collect()
}

/// [`LanguageModel::forward_batched_with_ids`].
///
/// A multi-token or masked batch runs each row through
/// `forward_with_sequence_id` and concatenates the rows; a width-one unmasked
/// batch runs the native batched decode over the rows' model-owned caches.
pub(crate) fn forward_batched_with_ids<M: SlidingHybridModel>(
    model: &M,
    input_ids: &MlxArray,
    seq_ids: Option<&[SequenceId]>,
    batch_caches: &mut [&mut [KVCache]],
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let family = M::FAMILY;
    let shape = mlxcel_core::array_shape(input_ids);
    assert_eq!(shape.len(), 2, "{family} batched input must be rank 2");
    assert_eq!(
        shape[0] as usize,
        batch_caches.len(),
        "{family} input batch must match cache rows"
    );
    if let Some(ids) = seq_ids {
        assert_eq!(
            ids.len(),
            batch_caches.len(),
            "{family} requires one sequence id per row"
        );
    }
    mlxcel_core::generate::route_batched_decode_with_ids(
        model,
        input_ids,
        seq_ids,
        batch_caches,
        |batch_caches| {
            let ids = seq_ids.unwrap_or_else(|| {
                panic!("{family} model-owned batched decode requires sequence ids")
            });
            let state = model.hybrid_sequence_state();
            if shape[1] != 1 || mask.is_some() {
                state
                    .validate_sequence_states(ids)
                    .unwrap_or_else(|error| panic!("{family} batched decode: {error}"));
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
                        model.forward_with_sequence_id(&input, Some(ids[row]), caches, mask)
                    })
                    .collect();
                let output_refs: Vec<_> = outputs.iter().map(|output| &**output).collect();
                return mlxcel_core::concatenate_many(&output_refs, 0);
            }
            state
                .with_batched_sequence_states(ids, |states| {
                    model.hybrid_forward_batched_decode(input_ids, states)
                })
                .unwrap_or_else(|error| panic!("{family} batched decode: {error}"))
        },
    )
}
