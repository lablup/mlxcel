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

//! What a quantized snapshot stores, buffer by buffer (issue #3250).
//!
//! `kv_snapshot_quantized_tests.rs` compares what a restored cache returns.
//! These tests look inside: every buffer a restored cache holds (keys,
//! values, scales, packed K and V, norms, rescale) and every scalar is
//! compared with the source cache byte for byte, and the tensor names each
//! mode writes are pinned, since they are what a later restore looks up.

use super::test_support::{assert_same_bytes, keys, values};
use super::*;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{KVCache, RotatingKVCache};
use mlxcel_core::{MlxArray, UniquePtr};

const WINDOW: i32 = 64;
const NAMES: KvSnapshotNames = KvSnapshotNames::new("Test", "standard", "rotating");

const ALL_STANDARD: [KVCacheMode; 4] = [
    KVCacheMode::Fp16,
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
];

fn tensor_names(snapshot: &ModelStateSnapshot) -> Vec<String> {
    snapshot.tensor_names().map(str::to_string).collect()
}

/// The first `rows` rows of a `[B, H, T, D]` buffer.
fn leading_rows(array: &MlxArray, rows: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(array);
    assert!(
        shape[2] >= rows,
        "buffer holds {} rows, need {rows}",
        shape[2]
    );
    mlxcel_core::slice(array, &[0, 0, 0, 0], &[shape[0], shape[1], rows, shape[3]])
}

fn assert_same_optional(
    actual: Option<&MlxArray>,
    expected: Option<&MlxArray>,
    rows: Option<i32>,
    what: &str,
) {
    match (actual, expected) {
        (None, None) => {}
        (Some(actual), Some(expected)) => match rows {
            Some(rows) => assert_same_bytes(
                &leading_rows(actual, rows),
                &leading_rows(expected, rows),
                what,
            ),
            None => assert_same_bytes(actual, expected, what),
        },
        (actual, expected) => panic!(
            "{what}: present in one cache only (restored {}, reference {})",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

/// Every buffer and scalar of the two caches agrees. With `rows`, the buffers
/// are compared over their first `rows` rows only: a logical rewind keeps the
/// physical capacity, and what lies past the live rows is not state.
fn assert_same_cache(restored: &KVCache, reference: &KVCache, rows: Option<i32>, what: &str) {
    let (r, s) = (restored.snapshot_parts(), reference.snapshot_parts());
    assert_eq!(
        (r.mode, r.offset, r.live_start, r.turbo_seed),
        (s.mode, s.offset, s.live_start, s.turbo_seed),
        "{what}: scalars"
    );
    assert_same_optional(r.keys, s.keys, rows, &format!("{what}: keys"));
    assert_same_optional(r.values, s.values, rows, &format!("{what}: values"));
    for name in KV_SIDECAR_FIELDS {
        assert_same_optional(
            r.sidecar(name),
            s.sidecar(name),
            rows,
            &format!("{what}: {name}"),
        );
    }
}

fn filled(mode: KVCacheMode, tokens: i32) -> KVCache {
    let mut cache = KVCache::new_with_mode(mode);
    let _ = cache.update_and_fetch(keys(0, tokens), values(0, tokens));
    cache
}

fn snapshot_of(cache: &KVCache, tokens: i32) -> ModelStateSnapshot {
    let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
    snapshot_standard(cache, &mut snapshot, "layer0", NAMES)
        .unwrap_or_else(|err| panic!("{:?} snapshot: {err}", cache.mode));
    snapshot
}

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

#[test]
fn a_restored_cache_holds_every_buffer_of_the_source_byte_for_byte() {
    for mode in ALL_STANDARD {
        // One token, a turbo block edge, an ordinary length, and both sides
        // of a buffer growth step.
        for tokens in [1, 32, 37, 256, 257] {
            let source = filled(mode, tokens);
            let snapshot = snapshot_of(&source, tokens);
            let mut restored = KVCache::new_with_mode(mode);
            restore_standard(&mut restored, &snapshot, "layer0", NAMES)
                .unwrap_or_else(|err| panic!("{mode:?} restore at {tokens}: {err}"));
            assert_same_cache(&restored, &source, None, &format!("{mode:?} at {tokens}"));
            assert_eq!(restored.nbytes(), source.nbytes(), "{mode:?} at {tokens}");
        }
    }
}

#[test]
fn a_truncated_restore_holds_the_cold_prefix_byte_for_byte() {
    for mode in ALL_STANDARD {
        for (stored, target) in [
            (37, 36),
            (37, 32),
            (37, 1),
            (300, 256),
            (300, 257),
            (300, 255),
        ] {
            let source = filled(mode, stored);
            let snapshot = snapshot_of(&source, stored);
            let mut restored = KVCache::new_with_mode(mode);
            restore_standard(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
            truncate_standard(&mut restored, target, NAMES).expect("truncate");
            let cold = filled(mode, target);
            assert_same_cache(
                &restored,
                &cold,
                Some(target),
                &format!("{mode:?} {stored} tokens cut to {target}"),
            );
        }
    }
}

/// The names are what a restore looks up, so they are pinned per mode: the
/// rows the mode holds, its sidecars, then the scalars. `Turbo4Asym` stores no
/// `values` and symmetric `Turbo4` no `keys`, because the cache holds none.
#[test]
fn each_quantized_mode_stores_exactly_these_tensors() {
    let names = |mode| tensor_names(&snapshot_of(&filled(mode, 8), 8));
    assert_eq!(
        names(KVCacheMode::Int8),
        [
            "layer0.standard.keys",
            "layer0.standard.values",
            "layer0.standard.key_scales",
            "layer0.standard.val_scales",
            "layer0.standard.turbo_seed",
            "layer0.standard.live_start",
            "layer0.standard.offset",
            "layer0.standard.mode",
        ]
    );
    assert_eq!(
        names(KVCacheMode::Turbo4Asym),
        [
            "layer0.standard.keys",
            "layer0.standard.v_packed",
            "layer0.standard.v_norms",
            "layer0.standard.v_rescale",
            "layer0.standard.turbo_seed",
            "layer0.standard.live_start",
            "layer0.standard.offset",
            "layer0.standard.mode",
        ]
    );
    assert_eq!(
        names(KVCacheMode::Turbo4),
        [
            "layer0.standard.v_packed",
            "layer0.standard.v_norms",
            "layer0.standard.v_rescale",
            "layer0.standard.k_packed",
            "layer0.standard.k_norms",
            "layer0.standard.turbo_seed",
            "layer0.standard.live_start",
            "layer0.standard.offset",
            "layer0.standard.mode",
        ]
    );
    // And FP16 writes neither a sidecar nor the two new scalars.
    assert_eq!(
        names(KVCacheMode::Fp16),
        [
            "layer0.standard.keys",
            "layer0.standard.values",
            "layer0.standard.offset",
            "layer0.standard.mode",
        ]
    );
}

/// A quantized snapshot that lost one of its scalars is refused whole. The
/// offset, the seed and the live-window start have no safe default for a
/// quantized cache: packed rows decode under one seed only.
#[test]
fn a_quantized_snapshot_missing_a_scalar_is_refused_whole() {
    for mode in [
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ] {
        let snapshot = snapshot_of(&filled(mode, 8), 8);
        for scalar in ["offset", "turbo_seed", "live_start"] {
            let missing = format!("layer0.standard.{scalar}");
            let partial = snapshot.subset(|name| name != missing);
            let mut target = KVCache::new_with_mode(mode);
            let err = restore_standard(&mut target, &partial, "layer0", NAMES)
                .expect_err("a quantized snapshot without one of its scalars");
            assert!(err.contains(scalar), "{mode:?} without {scalar}: {err}");
            assert!(
                target.is_empty() && target.offset == 0 && target.nbytes() == 0,
                "{mode:?} without {scalar}: the cache must be left untouched"
            );
        }
    }
}

/// A cache built with one seed restores from a snapshot taken under another:
/// the restored cache takes the snapshot's seed, the only one its packed rows
/// decode under.
#[test]
fn a_restored_cache_carries_the_seed_its_rows_were_packed_with() {
    for mode in [KVCacheMode::Turbo4Asym, KVCacheMode::Turbo4] {
        let mut source = KVCache::new_with_mode_and_seed(mode, 0x0BAD_5EED);
        let _ = source.update_and_fetch(keys(0, 37), values(0, 37));
        let snapshot = snapshot_of(&source, 37);
        let mut restored = KVCache::new_with_mode(mode);
        restore_standard(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
        assert_eq!(restored.snapshot_parts().turbo_seed, 0x0BAD_5EED);
        assert_same_cache(
            &restored,
            &source,
            None,
            &format!("{mode:?} under its own seed"),
        );
        for t in 37..41 {
            let (sk, sv) = source.update_and_fetch(keys(t, 1), values(t, 1));
            let (rk, rv) = restored.update_and_fetch(keys(t, 1), values(t, 1));
            assert_same_bytes(&rk, &sk, &format!("{mode:?}: keys after {t}"));
            assert_same_bytes(&rv, &sv, &format!("{mode:?}: values after {t}"));
        }
    }
}

/// Taking and restoring a snapshot copies handles, not rows, so neither
/// depends on how many tokens the cache holds. The bound is far above what
/// either takes and far below what reading 32k rows back to the host would.
#[test]
fn snapshot_and_restore_do_not_scale_with_the_cache_length() {
    for mode in [
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ] {
        for tokens in [4096, 32768] {
            let source = filled(mode, tokens);
            // Evaluate the source first, so its prefill is not what is timed.
            let parts = source.snapshot_parts();
            for array in [parts.keys, parts.values]
                .into_iter()
                .chain(
                    KV_SIDECAR_FIELDS
                        .into_iter()
                        .map(|name| parts.sidecar(name)),
                )
                .flatten()
            {
                mlxcel_core::eval(array);
            }
            let started = std::time::Instant::now();
            let snapshot = snapshot_of(&source, tokens);
            let mut restored = KVCache::new_with_mode(mode);
            restore_standard(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
            assert!(standard_truncatable_to(
                &snapshot,
                "layer0",
                tokens - 1,
                NAMES
            ));
            let elapsed = started.elapsed();
            println!("{mode:?} at {tokens} tokens: snapshot + restore in {elapsed:?}");
            assert!(
                elapsed < std::time::Duration::from_secs(2),
                "{mode:?} at {tokens}: snapshot + restore took {elapsed:?}"
            );
            assert_eq!(restored.offset, tokens);
            assert_eq!(restored.nbytes(), source.nbytes());
        }
    }
}

// ---------------------------------------------------------------------------
// RotatingKVCache
// ---------------------------------------------------------------------------

fn stepped_ring(mode: KVCacheMode, tokens: i32) -> RotatingKVCache {
    let mut ring = RotatingKVCache::new_with_mode(WINDOW, mode);
    for t in 0..tokens {
        let _ = ring.update_and_fetch(keys(t, 1), values(t, 1));
    }
    ring
}

fn assert_same_ring(restored: &RotatingKVCache, source: &RotatingKVCache, what: &str) {
    let (r, s) = (restored.snapshot_parts(), source.snapshot_parts());
    assert_eq!(r.state, s.state, "{what}: ring scalars");
    assert_same_optional(r.keys, s.keys, None, &format!("{what}: keys"));
    assert_same_optional(r.values, s.values, None, &format!("{what}: values"));
    for name in ROTATING_SIDECAR_FIELDS {
        assert_same_optional(
            r.sidecar(name),
            s.sidecar(name),
            None,
            &format!("{what}: {name}"),
        );
    }
}

#[test]
fn a_restored_ring_holds_every_buffer_of_the_source_byte_for_byte() {
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
        KVCacheMode::Turbo3Asym,
        KVCacheMode::Turbo4Asym,
    ] {
        // One token, before the window, exactly full, one past it, wrapped.
        for tokens in [1, 40, WINDOW, WINDOW + 1, 3 * WINDOW + 5] {
            let source = stepped_ring(mode, tokens);
            let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
            snapshot_rotating(&source, &mut snapshot, "layer0", NAMES)
                .unwrap_or_else(|err| panic!("{mode:?} ring snapshot at {tokens}: {err}"));
            let mut restored = RotatingKVCache::new_with_mode(WINDOW, mode);
            restore_rotating(&mut restored, &snapshot, "layer0", NAMES)
                .unwrap_or_else(|err| panic!("{mode:?} ring restore at {tokens}: {err}"));
            assert_same_ring(&restored, &source, &format!("{mode:?} ring at {tokens}"));
        }
    }
}

#[test]
fn a_turbo4_asym_ring_stores_exactly_these_tensors() {
    let ring = stepped_ring(KVCacheMode::Turbo4Asym, 8);
    let mut snapshot = ModelStateSnapshot::new("test", 8);
    snapshot_rotating(&ring, &mut snapshot, "layer0", NAMES).expect("snapshot");
    assert_eq!(
        tensor_names(&snapshot),
        [
            "layer0.rotating.keys",
            "layer0.rotating.v_packed",
            "layer0.rotating.v_norms",
            "layer0.rotating.v_rescale",
            "layer0.rotating.max_size",
            "layer0.rotating.buffer_size",
            "layer0.rotating.offset",
            "layer0.rotating.start_position",
            "layer0.rotating.idx",
            "layer0.rotating.step",
            "layer0.rotating.mode",
            "layer0.rotating.turbo_seed",
        ]
    );
}

/// A truncated restore of an unwrapped ring holds the cold prefix. The ring's
/// rewind moves cursors only, so the rows past the cut are still in the
/// buffer; the comparison is over the live rows.
#[test]
fn a_truncated_ring_holds_the_cold_prefix_byte_for_byte() {
    for mode in [KVCacheMode::Int8, KVCacheMode::Turbo4Asym] {
        for target in [33, 32, 25, 1] {
            let source = stepped_ring(mode, 40);
            let mut snapshot = ModelStateSnapshot::new("test", 40);
            snapshot_rotating(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");
            let mut restored = RotatingKVCache::new_with_mode(WINDOW, mode);
            restore_rotating(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
            truncate_rotating(&mut restored, target, NAMES).expect("truncate");
            let cold = stepped_ring(mode, target);

            let (r, c) = (restored.snapshot_parts(), cold.snapshot_parts());
            let what = format!("{mode:?} ring cut to {target}");
            assert_eq!(
                (r.state.offset, r.state.idx, r.state.mode),
                (c.state.offset, c.state.idx, c.state.mode),
                "{what}: cursors"
            );
            assert_same_optional(r.keys, c.keys, Some(target), &format!("{what}: keys"));
            assert_same_optional(r.values, c.values, Some(target), &format!("{what}: values"));
            for name in ROTATING_SIDECAR_FIELDS {
                assert_same_optional(
                    r.sidecar(name),
                    c.sidecar(name),
                    Some(target),
                    &format!("{what}: {name}"),
                );
            }

            // A multi-token append after the cut, which is what the scheduler
            // runs for the rest of the prompt.
            let mut restored = restored;
            let mut cold = cold;
            let (ck, cv) = cold.update_and_fetch(keys(target, 9), values(target, 9));
            let (rk, rv) = restored.update_and_fetch(keys(target, 9), values(target, 9));
            assert_same_bytes(&rk, &ck, &format!("{what}: keys after a chunk"));
            assert_same_bytes(&rv, &cv, &format!("{what}: values after a chunk"));
        }
    }
}

// ---------------------------------------------------------------------------
// The mode table check
// ---------------------------------------------------------------------------

#[test]
fn the_first_mode_a_snapshot_cannot_carry_is_reported() {
    use KVCacheMode::{Fp16, Int8, Turbo3Asym, Turbo4, Turbo4Asym, Turbo4Delegated};
    assert_eq!(first_snapshot_unsupported_mode(&[]), None);
    assert_eq!(first_snapshot_unsupported_mode(&[None, None]), None);
    assert_eq!(
        first_snapshot_unsupported_mode(&[
            Some(Fp16),
            None,
            Some(Int8),
            Some(Turbo4Asym),
            Some(Turbo4)
        ]),
        None,
        "every mode a snapshot carries"
    );
    assert_eq!(
        first_snapshot_unsupported_mode(&[Some(Turbo4Delegated)]),
        Some(Turbo4Delegated)
    );
    assert_eq!(
        first_snapshot_unsupported_mode(&[Some(Turbo3Asym)]),
        Some(Turbo3Asym)
    );
    assert_eq!(
        first_snapshot_unsupported_mode(&[
            Some(Fp16),
            None,
            Some(Turbo3Asym),
            Some(Turbo4Delegated)
        ]),
        Some(Turbo3Asym),
        "the first one in layer order"
    );
    // A sliding layer under either label reports the FP16 its ring stores, so
    // a table like Gemma 4's is fine as long as its full layers are.
    assert_eq!(
        first_snapshot_unsupported_mode(&[Some(Fp16), Some(Fp16), None, Some(Int8)]),
        None
    );
}
