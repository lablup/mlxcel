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

//! Cache-level tests for ring-only state checkpoints (issue #3255). The
//! model-level parity test is in `gemma4_ring_checkpoint_tests.rs`.

use super::*;
use crate::models::kv_snapshot::{snapshot_rotating, snapshot_standard};
use mlxcel_core::{MlxArray, UniquePtr};

const NAMES: KvSnapshotNames = KvSnapshotNames::new("Test", "standard", "rotating");
const WINDOW: i32 = 4;

fn kv(start: i32, n: i32, base: f32) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..n * 2)
        .map(|i| base + ((start * 2 + i) as f32) * 0.5)
        .collect();
    mlxcel_core::from_slice_f32(&data, &[1, 1, n, 2])
}

/// A two-layer snapshot (layer0 sliding, layer1 full) after `tokens` tokens.
fn snapshot_after(tokens: i32) -> ModelStateSnapshot {
    let mut ring = RotatingKVCache::new(WINDOW);
    let mut full = KVCache::new();
    ring.update_and_fetch(kv(0, tokens, 0.0), kv(0, tokens, 100.0));
    full.update_and_fetch(kv(0, tokens, 0.0), kv(0, tokens, 100.0));
    let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
    snapshot_rotating(&ring, &mut snapshot, "layer0", NAMES).expect("ring");
    snapshot_standard(&full, &mut snapshot, "layer1", NAMES).expect("full");
    snapshot
}

#[test]
fn an_unwrapped_ring_needs_no_checkpoint() {
    let snapshot = snapshot_after(WINDOW - 1);
    assert!(ring_checkpoint(&snapshot, "test", 2, NAMES).is_none());
}

#[test]
fn a_checkpoint_keeps_only_the_ring() {
    let snapshot = snapshot_after(WINDOW + 3);
    let checkpoint = ring_checkpoint(&snapshot, "test", 2, NAMES).expect("wrapped ring");
    assert_eq!(checkpoint.token_len(), snapshot.token_len());
    assert!(checkpoint.tensor("layer0.rotating.keys").is_some());
    assert!(checkpoint.tensor("layer0.rotating.offset").is_some());
    assert!(checkpoint.tensor("layer1.standard.keys").is_none());
    assert!(checkpoint.nbytes() < snapshot.nbytes());
}

#[test]
fn another_familys_snapshot_yields_no_checkpoint() {
    let snapshot = snapshot_after(WINDOW + 3);
    assert!(ring_checkpoint(&snapshot, "other", 2, NAMES).is_none());
}

#[test]
fn a_pair_is_validated_before_anything_is_restored() {
    let checkpoint = ring_checkpoint(&snapshot_after(WINDOW + 3), "test", 2, NAMES).unwrap();
    let donor = snapshot_after(WINDOW + 6);
    assert_eq!(
        checkpoint_target(&checkpoint, &donor, "test", NAMES),
        Ok(WINDOW + 3)
    );
    assert!(checkpoint_target(&checkpoint, &donor, "other", NAMES).is_err());
    let short = snapshot_after(WINDOW + 1);
    assert!(
        checkpoint_target(&checkpoint, &short, "test", NAMES).is_err(),
        "a donor shorter than the checkpoint cannot supply its rows"
    );
}

#[test]
fn the_restore_takes_the_ring_from_the_checkpoint_and_the_rest_from_the_donor() {
    let target = WINDOW + 3;
    let checkpoint = ring_checkpoint(&snapshot_after(target), "test", 2, NAMES).unwrap();
    let donor = snapshot_after(WINDOW + 6);

    let mut ring = RotatingKVCache::new(WINDOW);
    restore_rotating_from_checkpoint(&mut ring, &checkpoint, &donor, "layer0", target, NAMES)
        .expect("ring");
    assert_eq!(ring.offset, target);

    let mut full = KVCache::new();
    restore_standard_from_donor(&mut full, &donor, "layer1", target, NAMES).expect("full");
    assert_eq!(full.offset, target);
}

#[test]
fn a_ring_of_another_length_is_refused() {
    let donor = snapshot_after(WINDOW + 6);
    let wrong = ring_checkpoint(&donor, "test", 2, NAMES).unwrap();
    let mut ring = RotatingKVCache::new(WINDOW);
    let err =
        restore_rotating_from_checkpoint(&mut ring, &wrong, &donor, "layer0", WINDOW + 3, NAMES)
            .expect_err("a ring covering other tokens must be refused");
    assert!(err.contains("ring covers"), "{err}");
}

#[test]
fn a_donor_ring_without_a_checkpoint_ring_is_refused_and_an_empty_layer_stays_fresh() {
    let donor = snapshot_after(WINDOW + 6);
    let empty = ModelStateSnapshot::new("test", (WINDOW + 3) as usize);
    let mut ring = RotatingKVCache::new(WINDOW);
    assert!(
        restore_rotating_from_checkpoint(&mut ring, &empty, &donor, "layer0", WINDOW + 3, NAMES)
            .is_err()
    );
    // A KV-shared layer: neither side stored anything for it.
    let mut shared = RotatingKVCache::new(WINDOW);
    restore_rotating_from_checkpoint(&mut shared, &empty, &donor, "layer7", WINDOW + 3, NAMES)
        .expect("a layer nobody stored stays fresh");
    assert!(shared.keys.is_none());
    let mut shared_full = KVCache::new();
    restore_standard_from_donor(&mut shared_full, &donor, "layer7", WINDOW + 3, NAMES)
        .expect("a full layer the donor did not store stays fresh");
    assert!(shared_full.keys.is_none());
}
