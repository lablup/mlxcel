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

//! Snapshot encoding of the attention + GatedDeltaNet hybrids.
//!
//! An attention layer is stored as `{prefix}.attention.{mode,keys,values}` and
//! a recurrent layer as `{prefix}.linear.{slot}`; no offset is stored, because
//! every layer of a hybrid has consumed exactly the snapshot's tokens. The
//! encoding was written inside `Qwen3NextCache` first. It lives here so the
//! model-owned cache enum and the scheduler's generic snapshot code
//! (`server::prompt_cache::state_snapshot`, issue #3254) write and read the
//! same tensors through the same functions.
//!
//! Used by: Qwen3-Next and Qwen4-Exp through `Qwen3NextCache`, and Qwen 3.5
//! through the scheduler-owned layer state.

use crate::models::recurrent_snapshot::{
    push_kv_cache_mode, push_optional, restore_kv_cache_mode, restore_optional,
};
use mlxcel_core::cache::{KVCacheMode, RecurrentState};
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::KVCache;

/// Tensor-name segment of a recurrent layer's slots.
pub(crate) const LINEAR: &str = "linear";

/// Whether a snapshot tensor belongs in a turn-boundary state checkpoint
/// (issue #3047): the recurrent layers' state and convolution window. The
/// attention layers' rows are taken from a longer snapshot instead.
pub(crate) fn is_checkpoint_tensor(name: &str) -> bool {
    name.contains(".linear.")
}

/// Copy an attention layer into `snapshot` under `{prefix}.attention`.
///
/// # Errors
///
/// A quantized mode, whose sidecars the container does not model.
pub(crate) fn snapshot_attention(
    kv: &KVCache,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    if kv.mode != KVCacheMode::Fp16 {
        return Err(format!(
            "{prefix}.attention: model-state snapshots do not yet serialize {:?} KV sidecars",
            kv.mode
        ));
    }
    push_kv_cache_mode(snapshot, format!("{prefix}.attention.mode"), kv.mode);
    push_optional(snapshot, format!("{prefix}.attention.keys"), &kv.keys);
    push_optional(snapshot, format!("{prefix}.attention.values"), &kv.values);
    Ok(())
}

/// Restore an attention layer from `{prefix}.attention`, at the snapshot's
/// token length.
///
/// # Errors
///
/// The stored mode differs from the cache's, or the mode is quantized.
pub(crate) fn restore_attention(
    kv: &mut KVCache,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    let snapshot_mode = restore_kv_cache_mode(snapshot, format!("{prefix}.attention.mode"))?
        .unwrap_or(KVCacheMode::Fp16);
    if snapshot_mode != kv.mode {
        return Err(format!(
            "{prefix}.attention: snapshot KV mode {:?} does not match configured mode {:?}",
            snapshot_mode, kv.mode
        ));
    }
    if kv.mode != KVCacheMode::Fp16 {
        return Err(format!(
            "{prefix}.attention: model-state restores do not yet deserialize {:?} KV sidecars",
            kv.mode
        ));
    }
    kv.keys = restore_optional(snapshot, format!("{prefix}.attention.keys"));
    kv.values = restore_optional(snapshot, format!("{prefix}.attention.values"));
    kv.offset = snapshot.token_len() as i32;
    Ok(())
}

/// Validate that `checkpoint` can be rebuilt over `donor` and return
/// `(keep, have)`: the checkpoint's and the donor's token lengths.
pub(crate) fn checkpoint_lengths(
    checkpoint: &ModelStateSnapshot,
    donor: &ModelStateSnapshot,
    prefix: &str,
) -> Result<(usize, usize), String> {
    let keep = checkpoint.token_len();
    let have = donor.token_len();
    if keep == 0 || keep > have {
        return Err(format!(
            "{prefix}: a {keep}-token checkpoint cannot be rebuilt from a {have}-token snapshot"
        ));
    }
    Ok((keep, have))
}

/// Rebuild an attention layer for `keep` tokens from the `have`-token donor:
/// the donor's rows, cut back with [`KVCache::trim`], the same trim a
/// speculative rollback uses (issue #3047).
pub(crate) fn restore_attention_from_donor(
    kv: &mut KVCache,
    donor: &ModelStateSnapshot,
    prefix: &str,
    keep: usize,
    have: usize,
) -> Result<(), String> {
    restore_attention(kv, donor, prefix)?;
    if kv.keys.is_none() || kv.values.is_none() {
        return Err(format!(
            "{prefix}.attention: the {have}-token snapshot holds no attention rows"
        ));
    }
    let trim = (have - keep) as i32;
    if trim > 0 && kv.trim(trim) != trim {
        return Err(format!(
            "{prefix}.attention: could not cut {have} rows back to {keep}"
        ));
    }
    if kv.offset != keep as i32 {
        return Err(format!(
            "{prefix}.attention: {} rows after the cut, expected {keep}",
            kv.offset
        ));
    }
    Ok(())
}

/// The checkpoint must hold every one of `fields` under `{prefix}.linear`.
pub(crate) fn require_checkpoint_fields<'a>(
    checkpoint: &ModelStateSnapshot,
    prefix: &str,
    fields: impl IntoIterator<Item = &'a str>,
) -> Result<(), String> {
    let keep = checkpoint.token_len();
    for field in fields {
        if checkpoint
            .tensor(&format!("{prefix}.{LINEAR}.{field}"))
            .is_none()
        {
            return Err(format!(
                "{prefix}.{LINEAR}: checkpoint has no {field} for {keep} tokens"
            ));
        }
    }
    Ok(())
}

/// Copy a recurrent layer's slots into `snapshot` as
/// `{prefix}.linear.{slot}`, in slot order. An empty slot stores nothing.
pub(crate) fn snapshot_recurrent(
    state: &RecurrentState,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
) {
    for (name, tensor) in state.slots() {
        if let Some(tensor) = tensor {
            snapshot.push_tensor(format!("{prefix}.{LINEAR}.{name}"), tensor);
        }
    }
}

/// Restore every declared slot of a recurrent layer from
/// `{prefix}.linear.{slot}`, at the snapshot's token length. A slot the
/// snapshot does not hold is left empty.
pub(crate) fn restore_recurrent(
    state: &mut RecurrentState,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
) {
    let names: Vec<&'static str> = state.names().collect();
    for name in names {
        state.put(
            name,
            restore_optional(snapshot, format!("{prefix}.{LINEAR}.{name}")),
        );
    }
    state.offset = snapshot.token_len() as i32;
}
