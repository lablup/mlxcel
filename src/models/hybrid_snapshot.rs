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
//! An attention layer in `Int8`, `Turbo4Asym` or `Turbo4` also stores the
//! sidecars of its mode and its `turbo_seed` and `live_start` under the same
//! `{prefix}.attention.` prefix (issue #3250, `kv_snapshot::push_kv_sidecars`).
//! `Turbo4Delegated` and `Turbo3Asym` are still refused.
//!
//! Used by: Qwen3-Next and Qwen4-Exp through `Qwen3NextCache`, Qwen 3.5
//! through the scheduler-owned layer state, and the attention arms of LFM2
//! and Bailing MoE Linear.

use crate::models::kv_snapshot::{push_kv_sidecars, restore_kv_sidecars};
use crate::models::recurrent_snapshot::{
    push_kv_cache_mode, push_optional, restore_kv_cache_mode, restore_optional,
};
use mlxcel_core::cache::{KVCacheMode, RecurrentState, kv_mode_is_snapshotable};
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

/// Whether a quantized attention cache holds anything a snapshot has to carry
/// beside the mode tag. A cache that has seen no token holds no buffer, in
/// any mode.
fn holds_quantized_rows(kv: &KVCache) -> bool {
    kv.mode != KVCacheMode::Fp16 && !(kv.offset == 0 && kv.snapshot_parts().holds_no_rows())
}

/// Copy an attention layer into `snapshot` under `{prefix}.attention`.
///
/// An FP16 layer stores `.mode`, `.keys` and `.values`. A quantized layer
/// stores the same three names (those its mode holds) and its sidecars,
/// `.turbo_seed` and `.live_start`.
///
/// # Errors
///
/// `Turbo4Delegated` or `Turbo3Asym`, whose layouts the container does not
/// model, or a quantized cache whose buffers are not its mode's layout.
/// Nothing is stored for the layer in either case.
///
/// Used by: Qwen3-Next and Qwen4-Exp (`Qwen3NextCache`), Qwen 3.5 (the
/// scheduler-owned layer state), LFM2 and Bailing MoE Linear.
pub(crate) fn snapshot_attention(
    kv: &KVCache,
    snapshot: &mut ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    if !kv_mode_is_snapshotable(kv.mode) {
        return Err(format!(
            "{prefix}.attention: model-state snapshots do not yet serialize {:?} KV sidecars",
            kv.mode
        ));
    }
    // The sidecars go first: `push_kv_sidecars` validates the layout before
    // it stores anything, so a refused layer leaves no tensor behind.
    if holds_quantized_rows(kv) {
        push_kv_sidecars(kv, snapshot, &format!("{prefix}.attention"))
            .map_err(|err| format!("{prefix}.attention: {err}"))?;
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
/// The stored mode differs from the cache's, the mode is `Turbo4Delegated`
/// or `Turbo3Asym`, or a quantized snapshot lacks a buffer or scalar its mode
/// needs. A quantized restore assigns nothing unless everything validated.
///
/// Used by: Qwen3-Next and Qwen4-Exp (`Qwen3NextCache`), Qwen 3.5 (the
/// scheduler-owned layer state), LFM2 and Bailing MoE Linear.
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
    if !kv_mode_is_snapshotable(kv.mode) {
        return Err(format!(
            "{prefix}.attention: model-state restores do not yet deserialize {:?} KV sidecars",
            kv.mode
        ));
    }
    if kv.mode != KVCacheMode::Fp16 {
        let base = format!("{prefix}.attention");
        let stored_rows = ["keys", "k_packed"]
            .iter()
            .any(|name| snapshot.tensor(&format!("{base}.{name}")).is_some());
        if !stored_rows && snapshot.token_len() == 0 {
            // A layer that had seen no token stored its mode tag only.
            return Ok(());
        }
        return restore_kv_sidecars(kv, snapshot, &base, kv.mode, snapshot.token_len() as i32)
            .map_err(|err| format!("{base}: {err}"));
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
    // An FP16 layer holds its rows in `keys` and `values`. A quantized one
    // does not have both: `Turbo4Asym` has no `values` and symmetric `Turbo4`
    // has neither, so it is asked whether it holds any buffer at all.
    let no_rows = if kv.mode == KVCacheMode::Fp16 {
        kv.keys.is_none() || kv.values.is_none()
    } else {
        kv.snapshot_parts().holds_no_rows()
    };
    if no_rows {
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

#[cfg(test)]
#[path = "hybrid_snapshot_tests.rs"]
mod tests;
