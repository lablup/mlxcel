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

//! Ring checkpoints under a quantized KV mode table (issues #3255, #3250).
//!
//! A ring checkpoint restore takes each sliding layer from the checkpoint and
//! each full-attention layer from a longer donor snapshot, cut back to the
//! checkpoint's length. With a quantized table the donor's full layer holds
//! real quantized state, so the cut runs through the quantized trim, and the
//! ring is either plain rows behind a label or, for `Turbo4Asym`, packed V.
//! The restored pair must be the pair a cold run to the checkpoint's length
//! builds.

use super::*;
use crate::models::kv_snapshot::test_support::{assert_same_bytes, keys, values};
use crate::models::kv_snapshot::{snapshot_rotating, snapshot_standard};
use mlxcel_core::cache::KVCacheMode;

const NAMES: KvSnapshotNames = KvSnapshotNames::new("Test", "standard", "rotating");

/// A multiple of the turbo block, so a `Turbo4Asym` ring really quantizes.
const WINDOW: i32 = 64;

/// The table modes under test and one ring sidecar a `Turbo4Asym` ring adds.
const TABLES: [KVCacheMode; 3] = [
    KVCacheMode::Int8,
    KVCacheMode::Turbo4,
    KVCacheMode::Turbo4Asym,
];

/// A sliding layer and a full layer of `mode` after `tokens` decode steps.
fn layers_after(mode: KVCacheMode, tokens: i32) -> (RotatingKVCache, KVCache) {
    let mut ring = RotatingKVCache::new_with_mode(WINDOW, mode);
    let mut full = KVCache::new_with_mode(mode);
    for t in 0..tokens {
        let _ = ring.update_and_fetch(keys(t, 1), values(t, 1));
        let _ = full.update_and_fetch(keys(t, 1), values(t, 1));
    }
    (ring, full)
}

/// The two-layer snapshot (layer0 sliding, layer1 full) of those layers.
fn snapshot_after(mode: KVCacheMode, tokens: i32) -> ModelStateSnapshot {
    let (ring, full) = layers_after(mode, tokens);
    let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
    snapshot_rotating(&ring, &mut snapshot, "layer0", NAMES)
        .unwrap_or_else(|err| panic!("{mode:?} ring snapshot: {err}"));
    snapshot_standard(&full, &mut snapshot, "layer1", NAMES)
        .unwrap_or_else(|err| panic!("{mode:?} full snapshot: {err}"));
    snapshot
}

/// A label does not make an unwrapped ring look wrapped: the whole snapshot
/// still truncates on its own, so there is nothing to checkpoint.
#[test]
fn an_unwrapped_ring_needs_no_checkpoint_under_any_table() {
    for mode in TABLES {
        for tokens in [1, WINDOW - 1, WINDOW] {
            let snapshot = snapshot_after(mode, tokens);
            assert!(
                ring_checkpoint(&snapshot, "test", 2, NAMES).is_none(),
                "{mode:?} at {tokens}: an unwrapped ring"
            );
        }
    }
}

/// The checkpoint of a wrapped ring is the ring alone, packed V included when
/// the ring holds it, and none of the full layer's buffers.
#[test]
fn a_checkpoint_keeps_the_whole_ring_and_only_the_ring() {
    for mode in TABLES {
        let snapshot = snapshot_after(mode, WINDOW + 20);
        let checkpoint = ring_checkpoint(&snapshot, "test", 2, NAMES)
            .unwrap_or_else(|| panic!("{mode:?}: a wrapped ring is checkpointed"));
        let names: Vec<&str> = checkpoint.tensor_names().collect();
        assert!(
            names
                .iter()
                .all(|name| name.starts_with("layer0.rotating.")),
            "{mode:?}: only ring tensors, got {names:?}"
        );
        let ring_tensors = snapshot
            .tensor_names()
            .filter(|name| name.starts_with("layer0.rotating."))
            .count();
        assert_eq!(names.len(), ring_tensors, "{mode:?}: every ring tensor");
        assert_eq!(
            checkpoint.tensor("layer0.rotating.v_packed").is_some(),
            mode == KVCacheMode::Turbo4Asym,
            "{mode:?}: packed V travels with a quantized ring only"
        );
        assert!(checkpoint.nbytes() < snapshot.nbytes(), "{mode:?}");
    }
}

/// The restored layers are the layers a cold run to the checkpoint's length
/// holds: they return the same K/V, byte for byte, step after step and across
/// the ring's next wrap.
#[test]
fn a_checkpoint_restore_matches_a_cold_run_to_the_same_length() {
    for mode in TABLES {
        let target = WINDOW + 20;
        let checkpoint = ring_checkpoint(&snapshot_after(mode, target), "test", 2, NAMES)
            .unwrap_or_else(|| panic!("{mode:?}: a wrapped ring is checkpointed"));
        let donor = snapshot_after(mode, WINDOW + 45);
        assert_eq!(
            checkpoint_target(&checkpoint, &donor, "test", NAMES),
            Ok(target),
            "{mode:?}"
        );

        let mut ring = RotatingKVCache::new_with_mode(WINDOW, mode);
        restore_rotating_from_checkpoint(&mut ring, &checkpoint, &donor, "layer0", target, NAMES)
            .unwrap_or_else(|err| panic!("{mode:?} ring: {err}"));
        let mut full = KVCache::new_with_mode(mode);
        restore_standard_from_donor(&mut full, &donor, "layer1", target, NAMES)
            .unwrap_or_else(|err| panic!("{mode:?} full: {err}"));
        assert_eq!((ring.offset, full.offset), (target, target), "{mode:?}");

        let (mut cold_ring, mut cold_full) = layers_after(mode, target);
        for t in target..target + WINDOW + 3 {
            let (ck, cv) = cold_ring.update_and_fetch(keys(t, 1), values(t, 1));
            let (rk, rv) = ring.update_and_fetch(keys(t, 1), values(t, 1));
            assert_same_bytes(&rk, &ck, &format!("{mode:?} ring: keys after token {t}"));
            assert_same_bytes(&rv, &cv, &format!("{mode:?} ring: values after token {t}"));
            let (ck, cv) = cold_full.update_and_fetch(keys(t, 1), values(t, 1));
            let (fk, fv) = full.update_and_fetch(keys(t, 1), values(t, 1));
            assert_same_bytes(&fk, &ck, &format!("{mode:?} full: keys after token {t}"));
            assert_same_bytes(&fv, &cv, &format!("{mode:?} full: values after token {t}"));
        }
    }
}

/// A donor that cannot be cut back to the checkpoint's length is refused, and
/// so is a ring that covers other tokens.
#[test]
fn a_mismatched_pair_is_refused_under_a_quantized_table() {
    for mode in TABLES {
        let target = WINDOW + 20;
        let donor = snapshot_after(mode, WINDOW + 45);
        let mut full = KVCache::new_with_mode(mode);
        assert!(
            restore_standard_from_donor(&mut full, &donor, "layer1", WINDOW + 46, NAMES).is_err(),
            "{mode:?}: the donor does not hold that many tokens"
        );
        assert!(full.is_empty(), "{mode:?}: nothing was installed");

        let wrong = ring_checkpoint(&donor, "test", 2, NAMES).expect("wrapped");
        let mut ring = RotatingKVCache::new_with_mode(WINDOW, mode);
        let err =
            restore_rotating_from_checkpoint(&mut ring, &wrong, &donor, "layer0", target, NAMES)
                .expect_err("a ring covering other tokens must be refused");
        assert!(err.contains("ring covers"), "{mode:?}: {err}");
        assert!(ring.is_empty(), "{mode:?}: nothing was installed");
    }
}
