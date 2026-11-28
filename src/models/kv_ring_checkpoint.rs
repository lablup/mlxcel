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

//! Ring-only turn-boundary state checkpoints for sliding-window families
//! (issue #3255).
//!
//! A whole snapshot of a sliding-window model is cut back to a shorter prefix
//! only while every sliding ring is unwrapped (#1145). Once a ring has
//! wrapped, logical token `t` no longer sits at slot `t`, so a request that
//! edits an earlier turn cannot reuse a later, longer snapshot of the same
//! conversation and falls back to a much shorter prefix or a cold prefill.
//!
//! The full-attention layers of that longer snapshot still hold every token at
//! its own slot, so they truncate. Only the rings cannot be rebuilt, which is
//! the same split the hybrid families make between attention KV and recurrent
//! state (#3047). A ring checkpoint therefore keeps only the `rotating` tensors
//! and scalars of each sliding layer, and a restore takes the sliding layers
//! from the checkpoint and the full-attention layers from the donor, cut back
//! to the checkpoint's length. The store's donor rule (same session bucket,
//! donor tokens extend the checkpoint, lineage agrees) guarantees those rows
//! were computed by the same prefill.
//!
//! Used by: Gemma 3, Gemma 4, AFMoE.

use crate::models::kv_snapshot::{
    KvSnapshotNames, restore_rotating, restore_standard, rotating_truncatable_to,
    standard_truncatable_to, truncate_standard,
};
use crate::models::recurrent_snapshot::restore_i32;
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{KVCache, RotatingKVCache};

/// Whether `name` is one of the sliding-ring tensors or scalars
/// [`crate::models::kv_snapshot::snapshot_rotating`] writes:
/// `layer{idx}.{rotating}.{field}`.
fn is_ring_tensor(name: &str, names: KvSnapshotNames) -> bool {
    name.split('.').nth(1) == Some(names.rotating)
}

/// The ring checkpoint of a whole snapshot of `family` with `num_layers`
/// layers: the snapshot's sliding-ring tensors, sharing their buffers.
///
/// Returns `None` for another family's snapshot, for one with no sliding
/// layer, and when no ring has wrapped yet: the whole snapshot then truncates
/// on its own (#1145), so a checkpoint would only duplicate it.
pub(crate) fn ring_checkpoint(
    snapshot: &ModelStateSnapshot,
    family: &str,
    num_layers: usize,
    names: KvSnapshotNames,
) -> Option<ModelStateSnapshot> {
    if snapshot.family() != family {
        return None;
    }
    let wrapped = (0..num_layers).any(|idx| {
        let prefix = format!("layer{idx}");
        snapshot
            .tensor(&format!("{prefix}.{}.keys", names.rotating))
            .is_some()
            && !rotating_truncatable_to(snapshot, &prefix, 0, names)
    });
    if !wrapped {
        return None;
    }
    let checkpoint = snapshot.subset(|name| is_ring_tensor(name, names));
    (!checkpoint.is_empty()).then_some(checkpoint)
}

/// Validate a checkpoint and donor pair of `family` and return the token
/// length to restore: the checkpoint's.
///
/// # Errors
///
/// A family mismatch, or a donor shorter than the checkpoint.
pub(crate) fn checkpoint_target(
    checkpoint: &ModelStateSnapshot,
    donor: &ModelStateSnapshot,
    family: &str,
    names: KvSnapshotNames,
) -> Result<i32, String> {
    let label = names.family;
    if checkpoint.family() != family || donor.family() != family {
        return Err(format!(
            "cannot restore a {} checkpoint over a {} snapshot into {label}",
            checkpoint.family(),
            donor.family()
        ));
    }
    if checkpoint.token_len() > donor.token_len() {
        return Err(format!(
            "{label} checkpoint restore: the {}-token checkpoint is longer than its {}-token donor",
            checkpoint.token_len(),
            donor.token_len()
        ));
    }
    i32::try_from(checkpoint.token_len()).map_err(|_| {
        format!(
            "{label} checkpoint restore: length {} out of range",
            checkpoint.token_len()
        )
    })
}

/// Restore a full-attention layer from the donor, cut back to `target`
/// tokens.
///
/// A layer the donor stored nothing for (a KV-shared layer) stays fresh.
pub(crate) fn restore_standard_from_donor(
    cache: &mut KVCache,
    donor: &ModelStateSnapshot,
    prefix: &str,
    target: i32,
    names: KvSnapshotNames,
) -> Result<(), String> {
    if !standard_truncatable_to(donor, prefix, target, names) {
        return Err(format!(
            "{} checkpoint restore {prefix}: donor cannot be cut back to {target} tokens",
            names.family
        ));
    }
    restore_standard(cache, donor, prefix, names)?;
    truncate_standard(cache, target, names).map_err(|err| format!("{prefix}: {err}"))
}

/// Restore a sliding layer from the checkpoint's ring.
///
/// The ring must cover exactly `target` tokens. A layer for which neither the
/// checkpoint nor the donor stored a ring (a KV-shared layer) stays fresh; a
/// donor ring with no checkpoint ring is refused, since the donor's ring holds
/// other tokens.
pub(crate) fn restore_rotating_from_checkpoint(
    cache: &mut RotatingKVCache,
    checkpoint: &ModelStateSnapshot,
    donor: &ModelStateSnapshot,
    prefix: &str,
    target: i32,
    names: KvSnapshotNames,
) -> Result<(), String> {
    let (label, kind) = (names.family, names.rotating);
    if checkpoint
        .tensor(&format!("{prefix}.{kind}.keys"))
        .is_none()
    {
        if donor.tensor(&format!("{prefix}.{kind}.keys")).is_some() {
            return Err(format!(
                "{label} checkpoint restore {prefix}: the checkpoint has no ring for a sliding layer the donor holds"
            ));
        }
        return Ok(());
    }
    let offset = restore_i32(checkpoint, format!("{prefix}.{kind}.offset"));
    if offset != Some(target) {
        return Err(format!(
            "{label} checkpoint restore {prefix}: ring covers {offset:?} tokens, not {target}"
        ));
    }
    restore_rotating(cache, checkpoint, prefix, names)
}

#[cfg(test)]
#[path = "kv_ring_checkpoint_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kv_ring_checkpoint_quantized_tests.rs"]
mod quantized_tests;
