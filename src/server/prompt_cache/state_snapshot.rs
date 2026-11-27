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

//! Snapshot, restore, truncation and turn-boundary checkpoints of
//! scheduler-owned sequence state (issue #3254, ADR 0010).
//!
//! A scheduler-owned family reports one [`LayerStateKind`] per layer and the
//! pool holds one [`LayerState`] per layer. Everything the prompt cache does
//! with a sequence's state is written here once, over that per-layer storage,
//! instead of once per family inside the model wrapper:
//!
//! | Layer kind | Whole snapshot | Truncation | Checkpoint |
//! |---|---|---|---|
//! | `Full` | every row | always | rebuilt from the donor, cut back |
//! | `Sliding` | the ring and its scalars | while the ring is unwrapped | the ring itself (#3255) |
//! | `Recurrent` | every slot | never | the slots themselves (#3047) |
//! | `None` | nothing | always | nothing, restores fresh |
//!
//! No tensor is encoded here. The encoders are the ones the model-owned
//! families use (`models::kv_snapshot`, `models::kv_ring_checkpoint`,
//! `models::hybrid_snapshot`), selected by the family's
//! [`LayerStateSnapshotSpec`], so a snapshot written by this path has the
//! tensor names, shapes and scalars the family stored before it was migrated.
//!
//! A restore fills a fresh state vector and returns it. The caller installs
//! it only on `Ok`, so a mismatch leaves the sequence untouched and the
//! scheduler falls back to a cold prefill.

use crate::models::kv_snapshot::{self, KvSnapshotNames};
use crate::models::{hybrid_snapshot, kv_ring_checkpoint};
use mlxcel_core::cache::{
    KVCacheMode, LayerState, LayerStateEncoding, LayerStateKind, LayerStateSnapshotSpec,
};
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::RotatingKVCache;

/// What the generic snapshot code needs to know about a family: its snapshot
/// vocabulary and its per-layer kinds.
#[derive(Debug, Clone)]
pub(crate) struct StatePolicy {
    spec: LayerStateSnapshotSpec,
    kinds: Vec<LayerStateKind>,
}

impl StatePolicy {
    pub(crate) fn new(spec: LayerStateSnapshotSpec, kinds: Vec<LayerStateKind>) -> Self {
        Self { spec, kinds }
    }

    fn label(&self) -> &'static str {
        self.spec.label
    }

    /// The `kv_snapshot` name table of a windowed family.
    fn windowed_names(&self) -> Option<KvSnapshotNames> {
        match self.spec.encoding {
            LayerStateEncoding::Windowed { standard, rotating } => {
                Some(KvSnapshotNames::new(self.spec.label, standard, rotating))
            }
            LayerStateEncoding::Hybrid => None,
        }
    }

    fn check_family(&self, snapshot: &ModelStateSnapshot) -> Result<(), String> {
        if snapshot.family() == self.spec.family {
            Ok(())
        } else {
            Err(format!(
                "cannot restore {} snapshot into {}",
                snapshot.family(),
                self.label()
            ))
        }
    }

    /// The layer count and every layer's storage must be what the family's
    /// table says, before anything is read from or written to it.
    fn check_layers(&self, layers: &[LayerState]) -> Result<(), String> {
        if layers.len() != self.kinds.len() {
            return Err(format!(
                "{} state: {} layers of state, the model has {}",
                self.label(),
                layers.len(),
                self.kinds.len()
            ));
        }
        for (layer_idx, (layer, kind)) in layers.iter().zip(&self.kinds).enumerate() {
            if !layer.matches_kind(*kind) {
                return Err(format!(
                    "{} state layer{layer_idx}: holds {} storage, the model declares {kind:?}",
                    self.label(),
                    layer.storage_name()
                ));
            }
        }
        Ok(())
    }

    fn unsupported(&self, layer_idx: usize, layer: &LayerState) -> String {
        format!(
            "{} state layer{layer_idx}: {} storage has no {:?} snapshot encoding",
            self.label(),
            layer.storage_name(),
            self.spec.encoding
        )
    }

    /// Capture `layers` at `token_len` tokens.
    ///
    /// `Ok(None)` means there is nothing to store (a windowed family whose
    /// layers are all empty).
    ///
    /// # Errors
    ///
    /// A layer whose storage or KV mode the snapshot container cannot carry.
    /// The caller logs it and skips the donation.
    pub(crate) fn snapshot(
        &self,
        layers: &[LayerState],
        token_len: usize,
    ) -> Result<Option<ModelStateSnapshot>, String> {
        self.check_layers(layers)?;
        let mut snapshot = ModelStateSnapshot::new(self.spec.family, token_len);
        let names = self.windowed_names();
        for (layer_idx, layer) in layers.iter().enumerate() {
            let prefix = format!("layer{layer_idx}");
            match (layer, names) {
                (LayerState::Empty, _) => {}
                (LayerState::Kv(cache), Some(names)) => {
                    kv_snapshot::snapshot_standard(cache, &mut snapshot, &prefix, names)?;
                }
                (LayerState::Rotating(cache), Some(names)) => {
                    kv_snapshot::snapshot_rotating(cache, &mut snapshot, &prefix, names)?;
                }
                (LayerState::Kv(cache), None) => {
                    hybrid_snapshot::snapshot_attention(cache, &mut snapshot, &prefix)?;
                }
                (LayerState::Recurrent(state), None) => {
                    hybrid_snapshot::snapshot_recurrent(state, &mut snapshot, &prefix);
                }
                _ => return Err(self.unsupported(layer_idx, layer)),
            }
        }
        // A windowed family with nothing stored donates nothing; a hybrid
        // always stored its snapshot, whose token length is its state.
        if names.is_some() && snapshot.is_empty() {
            return Ok(None);
        }
        Ok(Some(snapshot))
    }

    /// Fill `fresh` (a state vector from the model's `make_layer_state`) from
    /// the whole of `snapshot` and return it.
    ///
    /// # Errors
    ///
    /// Another family's snapshot, a layer layout that differs from the
    /// model's table, or a tensor the per-layer decoder refuses.
    pub(crate) fn restore(
        &self,
        mut fresh: Vec<LayerState>,
        snapshot: &ModelStateSnapshot,
    ) -> Result<Vec<LayerState>, String> {
        self.check_family(snapshot)?;
        self.check_layers(&fresh)?;
        let names = self.windowed_names();
        for (layer_idx, layer) in fresh.iter_mut().enumerate() {
            let prefix = format!("layer{layer_idx}");
            match (&mut *layer, names) {
                (LayerState::Empty, _) => {}
                (LayerState::Kv(cache), Some(names)) => {
                    kv_snapshot::restore_standard(cache, snapshot, &prefix, names)?;
                }
                (LayerState::Rotating(cache), Some(names)) => {
                    kv_snapshot::restore_rotating(cache, snapshot, &prefix, names)?;
                }
                (LayerState::Kv(cache), None) => {
                    hybrid_snapshot::restore_attention(cache, snapshot, &prefix)?;
                }
                (LayerState::Recurrent(state), None) => {
                    hybrid_snapshot::restore_recurrent(state, snapshot, &prefix);
                }
                _ => return Err(self.unsupported(layer_idx, layer)),
            }
            self.check_restored_layer(layer, &prefix)?;
        }
        Ok(fresh)
    }

    /// [`check_kv_tensors`] with the layer named in the error. Every restore
    /// (whole, truncated and from a checkpoint) runs it on every layer before
    /// the state is installed.
    fn check_restored_layer(&self, layer: &LayerState, prefix: &str) -> Result<(), String> {
        check_kv_tensors(layer).map_err(|err| format!("{} {prefix}: {err}", self.label()))
    }

    /// Whether `snapshot` can be restored covering only its first
    /// `target_len` tokens (issue #1145).
    ///
    /// Answered per layer from the snapshot's own scalars, because the store
    /// decides before a sequence exists: a full layer truncates always, a
    /// sliding layer while its ring is unwrapped at `target_len`, and a
    /// recurrent layer never, so a family that holds one cannot truncate.
    pub(crate) fn truncatable_to(&self, snapshot: &ModelStateSnapshot, target_len: usize) -> bool {
        if snapshot.family() != self.spec.family || target_len > snapshot.token_len() {
            return false;
        }
        let Ok(target) = i32::try_from(target_len) else {
            return false;
        };
        let Some(names) = self.windowed_names() else {
            return false;
        };
        self.kinds.iter().enumerate().all(|(layer_idx, kind)| {
            let prefix = format!("layer{layer_idx}");
            match kind {
                // Each helper is vacuously true when its own tensors are
                // absent, so both are asked whatever the kind.
                LayerStateKind::Full | LayerStateKind::Sliding { .. } | LayerStateKind::None => {
                    kv_snapshot::rotating_truncatable_to(snapshot, &prefix, target, names)
                        && kv_snapshot::standard_truncatable_to(snapshot, &prefix, target, names)
                }
                LayerStateKind::Chunked { .. } | LayerStateKind::Recurrent => false,
            }
        })
    }

    /// [`Self::restore`], then cut every layer back to `target_len` tokens.
    ///
    /// # Errors
    ///
    /// As [`Self::restore`], and a snapshot that [`Self::truncatable_to`]
    /// refuses: the check is repeated here rather than trusted, because a
    /// partially truncated state would corrupt generation silently.
    pub(crate) fn restore_truncated(
        &self,
        fresh: Vec<LayerState>,
        snapshot: &ModelStateSnapshot,
        target_len: usize,
    ) -> Result<Vec<LayerState>, String> {
        self.check_family(snapshot)?;
        let label = self.label();
        let target = i32::try_from(target_len)
            .map_err(|_| format!("{label} truncated restore: target {target_len} out of range"))?;
        let names = match self.windowed_names() {
            Some(names) if self.truncatable_to(snapshot, target_len) => names,
            _ => {
                return Err(format!(
                    "{label} truncated restore: snapshot cannot be truncated to {target_len} tokens"
                ));
            }
        };
        let mut state = self.restore(fresh, snapshot)?;
        for (layer_idx, layer) in state.iter_mut().enumerate() {
            let cut = match layer {
                LayerState::Kv(cache) => kv_snapshot::truncate_standard(cache, target, names),
                LayerState::Rotating(cache) => kv_snapshot::truncate_rotating(cache, target, names),
                LayerState::Empty => Ok(()),
                LayerState::Chunked(_) | LayerState::Recurrent(_) => {
                    Err("this storage cannot be truncated".to_string())
                }
            };
            cut.map_err(|err| format!("layer{layer_idx}: {err}"))?;
        }
        Ok(state)
    }

    /// The turn-boundary state checkpoint of `snapshot`: the part a longer
    /// snapshot of the same conversation cannot give back.
    ///
    /// For a windowed family that is the sliding rings, once one has wrapped
    /// (#3255); before that the whole snapshot truncates on its own and there
    /// is no checkpoint. For a hybrid it is the recurrent slots (#3047).
    pub(crate) fn state_checkpoint(
        &self,
        snapshot: &ModelStateSnapshot,
    ) -> Option<ModelStateSnapshot> {
        match self.windowed_names() {
            Some(names) => kv_ring_checkpoint::ring_checkpoint(
                snapshot,
                self.spec.family,
                self.kinds.len(),
                names,
            ),
            None => (snapshot.family() == self.spec.family)
                .then(|| snapshot.subset(hybrid_snapshot::is_checkpoint_tensor)),
        }
    }

    /// Fill `fresh` with the state for `checkpoint.token_len()` tokens: the
    /// checkpoint's rings or recurrent slots, and the full layers of `donor`
    /// cut back to that length.
    ///
    /// # Errors
    ///
    /// A family mismatch, a donor shorter than the checkpoint, a layer the
    /// checkpoint should hold and does not, or a donor layer that cannot be
    /// cut back.
    pub(crate) fn restore_from_checkpoint(
        &self,
        mut fresh: Vec<LayerState>,
        checkpoint: &ModelStateSnapshot,
        donor: &ModelStateSnapshot,
    ) -> Result<Vec<LayerState>, String> {
        self.check_layers(&fresh)?;
        match self.windowed_names() {
            Some(names) => {
                let target = kv_ring_checkpoint::checkpoint_target(
                    checkpoint,
                    donor,
                    self.spec.family,
                    names,
                )?;
                for (layer_idx, layer) in fresh.iter_mut().enumerate() {
                    let prefix = format!("layer{layer_idx}");
                    match &mut *layer {
                        LayerState::Kv(cache) => kv_ring_checkpoint::restore_standard_from_donor(
                            cache, donor, &prefix, target, names,
                        )?,
                        LayerState::Rotating(cache) => {
                            kv_ring_checkpoint::restore_rotating_from_checkpoint(
                                cache, checkpoint, donor, &prefix, target, names,
                            )?;
                        }
                        LayerState::Empty => {}
                        other => return Err(self.unsupported(layer_idx, other)),
                    }
                    self.check_restored_layer(layer, &prefix)?;
                }
            }
            None => {
                let family = self.spec.family;
                if checkpoint.family() != family || donor.family() != family {
                    return Err(format!(
                        "cannot restore a {} checkpoint over a {} snapshot into {}",
                        checkpoint.family(),
                        donor.family(),
                        self.label()
                    ));
                }
                for (layer_idx, layer) in fresh.iter_mut().enumerate() {
                    let prefix = format!("layer{layer_idx}");
                    let (keep, have) =
                        hybrid_snapshot::checkpoint_lengths(checkpoint, donor, &prefix)?;
                    match &mut *layer {
                        LayerState::Recurrent(state) => {
                            hybrid_snapshot::require_checkpoint_fields(
                                checkpoint,
                                &prefix,
                                state.names(),
                            )?;
                            hybrid_snapshot::restore_recurrent(state, checkpoint, &prefix);
                        }
                        LayerState::Kv(cache) => hybrid_snapshot::restore_attention_from_donor(
                            cache, donor, &prefix, keep, have,
                        )?,
                        LayerState::Empty => {}
                        other => return Err(self.unsupported(layer_idx, other)),
                    }
                    self.check_restored_layer(layer, &prefix)?;
                }
            }
        }
        Ok(fresh)
    }
}

/// The restored keys and values of an attention layer must be one pair: both
/// present or both absent, rank 4, the same dtype and the same batch, head
/// and row counts. A full layer's buffer must also hold at least `offset`
/// rows, since attention reads its first `offset` rows, so a full layer with
/// no buffer must be at offset 0 (a ring's offset counts every token it has
/// seen, so it is not bounded by its rows). A snapshot that fails this was
/// not written by this layer's encoder.
///
/// That is the layout of a layer that stores plain rows. A quantized layer
/// (issue #3250) holds a different set of buffers, of different dtypes: an
/// `Int8` full layer keeps scales beside int8 rows, `Turbo4Asym` has packed V
/// and no values, symmetric `Turbo4` has neither keys nor values. Those are
/// checked against their own mode's layout instead, by the same check the
/// encoder ran. A sliding ring is judged by the mode it stores, since every
/// label but an aligned `Turbo4Asym` leaves it holding plain rows.
fn check_kv_tensors(layer: &LayerState) -> Result<(), String> {
    match layer {
        LayerState::Kv(cache) if cache.mode != KVCacheMode::Fp16 => {
            let parts = cache.snapshot_parts();
            if !parts.holds_no_rows() {
                return parts.validate();
            }
            return if parts.offset > 0 {
                Err(format!("no rows behind an offset of {}", parts.offset))
            } else {
                Ok(())
            };
        }
        LayerState::Rotating(cache)
            if RotatingKVCache::stored_mode(cache.max_size, cache.mode) != KVCacheMode::Fp16 =>
        {
            return cache.snapshot_parts().check_layout();
        }
        _ => {}
    }
    let (keys, values, full_offset) = match layer {
        LayerState::Kv(cache) => (
            cache.keys.as_deref(),
            cache.values.as_deref(),
            Some(cache.offset),
        ),
        LayerState::Rotating(cache) => (cache.keys.as_deref(), cache.values.as_deref(), None),
        _ => return Ok(()),
    };
    let (keys, values) = match (keys, values) {
        // A full layer with no rows cannot claim tokens: attention would read
        // rows that were never restored.
        (None, None) => {
            return match full_offset {
                Some(offset) if offset > 0 => {
                    Err(format!("no keys or values behind an offset of {offset}"))
                }
                _ => Ok(()),
            };
        }
        (Some(keys), Some(values)) => (keys, values),
        _ => return Err("the snapshot holds only one of keys and values".to_string()),
    };
    let (k_shape, v_shape) = (
        mlxcel_core::array_shape(keys),
        mlxcel_core::array_shape(values),
    );
    if k_shape.len() != 4 || v_shape.len() != 4 || k_shape[..3] != v_shape[..3] {
        return Err(format!(
            "keys {k_shape:?} and values {v_shape:?} are not one [B, H, T, D] pair"
        ));
    }
    let (k_dtype, v_dtype) = (
        mlxcel_core::array_dtype(keys),
        mlxcel_core::array_dtype(values),
    );
    if k_dtype != v_dtype {
        return Err(format!(
            "keys dtype {k_dtype} and values dtype {v_dtype} differ"
        ));
    }
    if let Some(offset) = full_offset
        && k_shape[2] < offset
    {
        return Err(format!(
            "{} rows of keys cannot cover an offset of {offset}",
            k_shape[2]
        ));
    }
    Ok(())
}

#[path = "state_snapshot_dispatch.rs"]
mod dispatch;
pub(crate) use dispatch::{
    restore_sequence_state, restore_sequence_state_from_checkpoint,
    restore_sequence_state_truncated, snapshot_sequence_state, snapshot_truncatable_to,
    state_checkpoint_from_snapshot,
};

#[cfg(test)]
#[path = "state_snapshot_tests.rs"]
mod tests;
