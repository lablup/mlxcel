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

//! Exact-prefix state snapshots of a Raon-1 mini sequence for the server
//! prompt cache (issue #3458).
//!
//! Per layer, an RDA layer stores its raw conv window, its float32 delta-rule
//! state (kept float32, no cast) and its offset; an MLA layer stores its
//! latent pair through the shared full-attention serializer
//! ([`kv_snapshot`]), which carries fp16 caches only and refuses a quantized
//! latent, so such a sequence donates nothing and prefills cold.
//!
//! The recurrent state summarizes every token consumed and cannot be rewound,
//! so `snapshot_truncatable_to` keeps its `false` default and only a hit on
//! the whole stored prefix restores. A state that does not hold exactly the
//! snapshot's tokens is not donated: a DFlash2 burst keeps its caches outside
//! the sequence (and its drafter's feature history is not snapshotted), so a
//! drafted request stores nothing. The snapshot holds no reasoning effort:
//! the prompt-cache key folds the request's resolved level into the template
//! signature, so a request at another effort looks up another bucket.

use super::Raon1MiniLayerCache;
use crate::models::kv_snapshot::{self, KvSnapshotNames};
use crate::models::recurrent_snapshot::{push_i32, push_optional, restore_i32, restore_optional};
use mlxcel_core::generate::ModelStateSnapshot;

/// Family tag of a Raon-1 mini snapshot; a restore refuses any other.
pub(super) const FAMILY: &str = "raon1_mini";

const NAMES: KvSnapshotNames = KvSnapshotNames::new("Raon-1 mini", "mla", "rotating");

impl Raon1MiniLayerCache {
    /// Copy this layer's state into `snapshot` under `prefix`.
    pub(super) fn snapshot_into(
        &self,
        snapshot: &mut ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            Raon1MiniLayerCache::Delta(state) => {
                push_optional(snapshot, format!("{prefix}.rda.conv"), &state.conv);
                push_optional(snapshot, format!("{prefix}.rda.ssm"), &state.ssm);
                push_i32(snapshot, format!("{prefix}.rda.offset"), state.offset);
                Ok(())
            }
            Raon1MiniLayerCache::Attn(cache) => {
                kv_snapshot::snapshot_standard(cache, snapshot, prefix, NAMES)
            }
        }
    }

    /// Fill this fresh layer state from `snapshot` under `prefix`.
    pub(super) fn restore_from(
        &mut self,
        snapshot: &ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            Raon1MiniLayerCache::Delta(state) => {
                let conv = restore_optional(snapshot, format!("{prefix}.rda.conv"));
                let ssm = restore_optional(snapshot, format!("{prefix}.rda.ssm"));
                let offset = restore_i32(snapshot, format!("{prefix}.rda.offset"))
                    .unwrap_or(snapshot.token_len() as i32);
                check_rda_state(conv.as_deref(), ssm.as_deref(), offset, snapshot, prefix)?;
                // Validated before assigning, so a failure leaves the fresh
                // state untouched and the scheduler prefills cold.
                state.conv = conv;
                state.ssm = ssm;
                state.offset = offset;
                Ok(())
            }
            Raon1MiniLayerCache::Attn(cache) => {
                kv_snapshot::restore_standard(cache, snapshot, prefix, NAMES)
            }
        }
    }
}

/// Refuse an RDA snapshot that is missing its state or has a state of the
/// wrong shape: a layer a snapshot of a started sequence does not carry, a
/// conv window that is not `[B, K-1, 3P]`, a recurrent state that is not
/// `[B, H, D, D]` with the conv window's batch, or an offset other than the
/// tokens the snapshot is keyed by.
fn check_rda_state(
    conv: Option<&mlxcel_core::MlxArray>,
    ssm: Option<&mlxcel_core::MlxArray>,
    offset: i32,
    snapshot: &ModelStateSnapshot,
    prefix: &str,
) -> Result<(), String> {
    let fail = |what: String| Err(format!("Raon-1 mini restore {prefix}: {what}"));
    let (conv, ssm) = match (conv, ssm) {
        (Some(conv), Some(ssm)) => (conv, ssm),
        (None, None) if snapshot.token_len() == 0 => return Ok(()),
        (None, None) => {
            return fail(format!(
                "RDA snapshot of {} tokens has neither a conv window nor a recurrent state",
                snapshot.token_len()
            ));
        }
        _ => {
            return fail(
                "RDA snapshot has only one of its conv window and recurrent state".to_string(),
            );
        }
    };
    let (conv_shape, ssm_shape) = (
        mlxcel_core::array_shape(conv),
        mlxcel_core::array_shape(ssm),
    );
    if conv_shape.len() != 3 || ssm_shape.len() != 4 || conv_shape.contains(&0) {
        return fail(format!(
            "RDA conv window {conv_shape:?} or recurrent state {ssm_shape:?} has the wrong rank"
        ));
    }
    if conv_shape[0] != ssm_shape[0] || ssm_shape[2] != ssm_shape[3] {
        return fail(format!(
            "RDA conv window {conv_shape:?} and recurrent state {ssm_shape:?} disagree"
        ));
    }
    if usize::try_from(offset).ok() != Some(snapshot.token_len()) {
        return fail(format!(
            "RDA snapshot offset {offset} is not its {} tokens",
            snapshot.token_len()
        ));
    }
    Ok(())
}

/// The snapshot of one sequence's layer states at `token_len` tokens, or the
/// reason a layer cannot be carried.
pub(super) fn snapshot_layers(
    layers: &[Raon1MiniLayerCache],
    token_len: usize,
) -> Result<ModelStateSnapshot, String> {
    let mut snapshot = ModelStateSnapshot::new(FAMILY, token_len);
    for (idx, layer) in layers.iter().enumerate() {
        // The state must hold exactly the tokens the snapshot is keyed by. A
        // DFlash2 burst runs on its own caches and leaves the sequence's
        // state untouched, so its finish has nothing to donate here.
        if usize::try_from(layer.offset()).ok() != Some(token_len) {
            return Err(format!(
                "layer {idx} holds {} tokens, not the {token_len} the snapshot is keyed by",
                layer.offset()
            ));
        }
        layer.snapshot_into(&mut snapshot, &format!("layer{idx}"))?;
    }
    Ok(snapshot)
}

/// Fill the fresh `layers` from `snapshot`; on an error the caller drops them.
pub(super) fn restore_layers(
    layers: &mut [Raon1MiniLayerCache],
    snapshot: &ModelStateSnapshot,
) -> Result<(), String> {
    if snapshot.family() != FAMILY {
        return Err(format!(
            "cannot restore {} snapshot into Raon-1 mini",
            snapshot.family()
        ));
    }
    for (idx, layer) in layers.iter_mut().enumerate() {
        // `restore_standard` accepts a layer with no stored tensors as a
        // fresh one, so a snapshot that lost an MLA layer is caught here.
        if snapshot.token_len() > 0
            && matches!(layer, Raon1MiniLayerCache::Attn(_))
            && snapshot.tensor(&format!("layer{idx}.mla.keys")).is_none()
        {
            return Err(format!(
                "Raon-1 mini restore layer{idx}: snapshot of {} tokens has no latent cache",
                snapshot.token_len()
            ));
        }
        layer.restore_from(snapshot, &format!("layer{idx}"))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "raon1_mini_snapshot_tests.rs"]
mod tests;
