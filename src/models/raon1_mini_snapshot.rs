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
                state.conv = restore_optional(snapshot, format!("{prefix}.rda.conv"));
                state.ssm = restore_optional(snapshot, format!("{prefix}.rda.ssm"));
                if state.conv.is_some() != state.ssm.is_some() {
                    return Err(format!(
                        "Raon-1 mini restore {prefix}: RDA snapshot has only one of its conv \
                         window and recurrent state"
                    ));
                }
                state.offset = restore_i32(snapshot, format!("{prefix}.rda.offset"))
                    .unwrap_or(snapshot.token_len() as i32);
                Ok(())
            }
            Raon1MiniLayerCache::Attn(cache) => {
                kv_snapshot::restore_standard(cache, snapshot, prefix, NAMES)
            }
        }
    }
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
        layer.restore_from(snapshot, &format!("layer{idx}"))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "raon1_mini_snapshot_tests.rs"]
mod tests;
