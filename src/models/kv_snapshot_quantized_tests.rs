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

//! Snapshot and restore of the non-FP16 KV cache modes (issue #3250).
//!
//! Every comparison here is between a restored cache and an uninterrupted
//! cache of the same mode fed the same tokens, never against FP16: a quantized
//! cache is not expected to equal an FP16 one, it is expected to equal itself.
//! Returned K/V are compared byte for byte, dtype included.
//!
//! The tests in this file use only the snapshot functions and the caches'
//! public surface, so they also compile against a tree without the change,
//! where they fail. The stored-buffer comparisons, which need the new
//! accessors, live in `kv_snapshot_parts_tests.rs`.

use super::test_support::{assert_same_bytes, keys, kv, values};
use super::*;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{KVCache, RotatingKVCache};

/// Head dim of the shared fixture, for the byte counts below.
const QH: i32 = super::test_support::QH;
const QD: i32 = super::test_support::QD;

/// Sliding window of the rotating fixtures: a multiple of the turbo block
/// size (32), so `Turbo4Asym` really quantizes there.
const WINDOW: i32 = 64;

const NAMES: KvSnapshotNames = KvSnapshotNames::new("Test", "standard", "rotating");

/// The modes a full-attention [`KVCache`] snapshot carries.
const STANDARD_MODES: [KVCacheMode; 3] = [
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
];

/// The labels a [`RotatingKVCache`] accepts while storing plain FP16 rows.
const FP16_RING_LABELS: [KVCacheMode; 4] = [
    KVCacheMode::Int8,
    KVCacheMode::Turbo4,
    KVCacheMode::Turbo4Delegated,
    KVCacheMode::Turbo3Asym,
];

/// Append tokens `[start, start + n)` one at a time to both caches and require
/// the returned K/V to agree after every single append.
fn assert_standard_steps_agree(a: &mut KVCache, b: &mut KVCache, start: i32, n: i32, what: &str) {
    for t in start..start + n {
        let (ak, av) = a.update_and_fetch(keys(t, 1), values(t, 1));
        let (bk, bv) = b.update_and_fetch(keys(t, 1), values(t, 1));
        assert_same_bytes(&ak, &bk, &format!("{what}: keys after token {t}"));
        assert_same_bytes(&av, &bv, &format!("{what}: values after token {t}"));
    }
    assert_eq!(a.offset, b.offset, "{what}: offset");
    assert_eq!(a.seq_len(), b.seq_len(), "{what}: live length");
}

fn assert_rotating_steps_agree(
    a: &mut RotatingKVCache,
    b: &mut RotatingKVCache,
    start: i32,
    n: i32,
    what: &str,
) {
    for t in start..start + n {
        let (ak, av) = a.update_and_fetch(keys(t, 1), values(t, 1));
        let (bk, bv) = b.update_and_fetch(keys(t, 1), values(t, 1));
        assert_same_bytes(&ak, &bk, &format!("{what}: keys after token {t}"));
        assert_same_bytes(&av, &bv, &format!("{what}: values after token {t}"));
    }
    assert_eq!(
        a.snapshot_state(),
        b.snapshot_state(),
        "{what}: ring scalars"
    );
}

/// Snapshot `source` and restore it into a fresh cache of the same mode.
fn standard_round_trip(source: &KVCache, tokens: i32) -> KVCache {
    let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
    snapshot_standard(source, &mut snapshot, "layer0", NAMES)
        .unwrap_or_else(|err| panic!("{:?} snapshot at {tokens} tokens: {err}", source.mode));
    let mut restored = KVCache::new_with_mode(source.mode);
    restore_standard(&mut restored, &snapshot, "layer0", NAMES)
        .unwrap_or_else(|err| panic!("{:?} restore at {tokens} tokens: {err}", source.mode));
    restored
}

fn rotating_round_trip(source: &RotatingKVCache, tokens: i32) -> RotatingKVCache {
    let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
    snapshot_rotating(source, &mut snapshot, "layer0", NAMES)
        .unwrap_or_else(|err| panic!("{:?} ring snapshot at {tokens} tokens: {err}", source.mode));
    let mut restored = RotatingKVCache::new_with_mode(source.max_size, source.mode);
    restore_rotating(&mut restored, &snapshot, "layer0", NAMES)
        .unwrap_or_else(|err| panic!("{:?} ring restore at {tokens} tokens: {err}", source.mode));
    restored
}

fn tensor_names(snapshot: &ModelStateSnapshot) -> Vec<String> {
    snapshot.tensor_names().map(str::to_string).collect()
}

/// Every cold-versus-restored comparison below feeds a prefix once as part of
/// a longer push and once on its own, so the fixture must not depend on how
/// the tokens were chunked.
#[test]
fn the_fixture_depends_on_the_position_and_not_on_the_chunking() {
    let long = keys(0, 9);
    assert_same_bytes(
        &mlxcel_core::slice(&long, &[0, 0, 4, 0], &[1, QH, 9, QD]),
        &keys(4, 5),
        "a slice of a long push against a short push of the same positions",
    );
    assert_same_bytes(
        &mlxcel_core::slice(&values(0, 9), &[0, 0, 0, 0], &[1, QH, 3, QD]),
        &values(0, 3),
        "a prefix of a long push against a short push",
    );
}

// ---------------------------------------------------------------------------
// KVCache: Int8, Turbo4Asym, Turbo4
// ---------------------------------------------------------------------------

/// A prefill, a snapshot, a restore, and then decode steps: the restored cache
/// returns the K/V the uninterrupted cache returns, at every step.
///
/// The prefill lengths are the boundary cases: one token, one short of and
/// exactly a turbo block (32), an ordinary length, and one short of, exactly
/// and one past a buffer growth step (256). The decode run after each restore
/// is long enough to cross the next turbo block.
#[test]
fn a_restored_quantized_cache_decodes_like_the_uninterrupted_one() {
    for mode in STANDARD_MODES {
        for prefill in [1, 31, 32, 37, 255, 256, 257] {
            let mut source = KVCache::new_with_mode(mode);
            let _ = source.update_and_fetch(keys(0, prefill), values(0, prefill));
            assert_eq!(source.offset, prefill);

            let mut restored = standard_round_trip(&source, prefill);
            assert_eq!(restored.offset, prefill, "{mode:?}: restored offset");
            assert_eq!(restored.mode, mode, "{mode:?}: restored mode");
            assert_eq!(
                restored.nbytes(),
                source.nbytes(),
                "{mode:?} at {prefill}: the restored cache holds the same buffers"
            );
            assert_standard_steps_agree(
                &mut source,
                &mut restored,
                prefill,
                34,
                &format!("{mode:?} restored at {prefill}"),
            );
        }
    }
}

/// The same after a cache that was built by decode steps only, and after one
/// built by two prefill chunks, since each takes a different growth path.
#[test]
fn a_restored_quantized_cache_matches_whatever_built_the_source() {
    for mode in STANDARD_MODES {
        let mut stepped = KVCache::new_with_mode(mode);
        for t in 0..40 {
            let _ = stepped.update_and_fetch(keys(t, 1), values(t, 1));
        }
        let mut restored = standard_round_trip(&stepped, 40);
        assert_standard_steps_agree(
            &mut stepped,
            &mut restored,
            40,
            8,
            &format!("{mode:?} built by decode steps"),
        );

        let mut chunked = KVCache::new_with_mode(mode);
        let _ = chunked.update_and_fetch(keys(0, 200), values(0, 200));
        let _ = chunked.update_and_fetch(keys(200, 100), values(200, 100));
        let mut restored = standard_round_trip(&chunked, 300);
        assert_standard_steps_agree(
            &mut chunked,
            &mut restored,
            300,
            8,
            &format!("{mode:?} built by two prefill chunks"),
        );
    }
}

/// A multi-token append after the restore (a second prefill chunk, which is
/// what the scheduler runs for the new suffix of a prompt) agrees too.
#[test]
fn a_restored_quantized_cache_takes_a_prefill_suffix() {
    for mode in STANDARD_MODES {
        for (prefill, suffix) in [(37, 19), (256, 1), (256, 300), (1, 255)] {
            let mut source = KVCache::new_with_mode(mode);
            let _ = source.update_and_fetch(keys(0, prefill), values(0, prefill));
            let mut restored = standard_round_trip(&source, prefill);

            let (sk, sv) = source.update_and_fetch(keys(prefill, suffix), values(prefill, suffix));
            let (rk, rv) =
                restored.update_and_fetch(keys(prefill, suffix), values(prefill, suffix));
            let what = format!("{mode:?} restored at {prefill}, suffix of {suffix}");
            assert_same_bytes(&rk, &sk, &format!("{what}: keys"));
            assert_same_bytes(&rv, &sv, &format!("{what}: values"));
            assert_standard_steps_agree(&mut source, &mut restored, prefill + suffix, 3, &what);
        }
    }
}

/// Restoring a snapshot and then appending to the restored cache must leave
/// the snapshot as it was: a second restore of the same snapshot still
/// reproduces the source. The snapshot shares its buffers with the cache it
/// was taken from and with every cache restored from it.
#[test]
fn a_quantized_snapshot_survives_appends_to_the_caches_that_share_it() {
    for mode in STANDARD_MODES {
        let mut source = KVCache::new_with_mode(mode);
        let _ = source.update_and_fetch(keys(0, 37), values(0, 37));
        let mut snapshot = ModelStateSnapshot::new("test", 37);
        snapshot_standard(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");

        // The source moves on, and so does a first restored cache.
        let mut first = KVCache::new_with_mode(mode);
        restore_standard(&mut first, &snapshot, "layer0", NAMES).expect("first restore");
        for t in 37..60 {
            let _ = source.update_and_fetch(keys(t, 1), kv(t, 1, 5.0));
            let _ = first.update_and_fetch(kv(t, 1, 9.0), values(t, 1));
        }

        // A cold cache of the first 37 tokens is the reference for the second
        // restore, which must not have seen either tail.
        let mut cold = KVCache::new_with_mode(mode);
        let _ = cold.update_and_fetch(keys(0, 37), values(0, 37));
        let mut second = KVCache::new_with_mode(mode);
        restore_standard(&mut second, &snapshot, "layer0", NAMES).expect("second restore");
        assert_standard_steps_agree(
            &mut cold,
            &mut second,
            37,
            4,
            &format!("{mode:?} second restore"),
        );
    }
}

/// A truncated restore is the cache a cold fill of the same prefix builds.
/// The targets cover the whole snapshot, one token, a turbo block edge and an
/// ordinary cut, from snapshots on both sides of a buffer growth step.
#[test]
fn a_truncated_quantized_restore_matches_a_cold_fill_of_the_prefix() {
    for mode in STANDARD_MODES {
        for (stored, target) in [
            (37, 37),
            (37, 36),
            (37, 32),
            (37, 20),
            (37, 1),
            (300, 256),
            (300, 255),
            (300, 257),
            (256, 32),
        ] {
            let mut source = KVCache::new_with_mode(mode);
            let _ = source.update_and_fetch(keys(0, stored), values(0, stored));
            let mut snapshot = ModelStateSnapshot::new("test", stored as usize);
            snapshot_standard(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");

            let what = format!("{mode:?} {stored} tokens cut to {target}");
            assert!(
                standard_truncatable_to(&snapshot, "layer0", target, NAMES),
                "{what}: a full-attention snapshot truncates to any earlier length"
            );
            let mut restored = KVCache::new_with_mode(mode);
            restore_standard(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
            truncate_standard(&mut restored, target, NAMES).expect("truncate");
            assert_eq!(restored.offset, target, "{what}: offset after the cut");

            let mut cold = KVCache::new_with_mode(mode);
            let _ = cold.update_and_fetch(keys(0, target), values(0, target));
            assert_standard_steps_agree(&mut cold, &mut restored, target, 34, &what);
        }
    }
}

/// The truncation check answers from the snapshot's own length in every mode.
/// Symmetric `Turbo4` stores no `keys` tensor at all, so a check that reads
/// "no keys" as "nothing stored" would call any target truncatable.
#[test]
fn quantized_truncation_is_refused_past_the_stored_length() {
    for mode in STANDARD_MODES {
        let mut source = KVCache::new_with_mode(mode);
        let _ = source.update_and_fetch(keys(0, 37), values(0, 37));
        let mut snapshot = ModelStateSnapshot::new("test", 37);
        snapshot_standard(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");

        assert!(standard_truncatable_to(&snapshot, "layer0", 37, NAMES));
        assert!(standard_truncatable_to(&snapshot, "layer0", 0, NAMES));
        assert!(
            !standard_truncatable_to(&snapshot, "layer0", 38, NAMES),
            "{mode:?}: a target past the stored length is refused"
        );
        assert!(!standard_truncatable_to(&snapshot, "layer0", -1, NAMES));

        let mut restored = KVCache::new_with_mode(mode);
        restore_standard(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
        let err = truncate_standard(&mut restored, 38, NAMES)
            .expect_err("a target past the cached offset is refused");
        assert!(err.contains("exceeds cached offset"), "{mode:?}: {err}");
        assert_eq!(
            restored.offset, 37,
            "{mode:?}: a refused cut changes nothing"
        );
    }
}

/// Cutting a restored cache back to nothing leaves a cache that fills like a
/// fresh one.
#[test]
fn a_quantized_restore_truncated_to_zero_fills_like_a_fresh_cache() {
    for mode in STANDARD_MODES {
        let mut source = KVCache::new_with_mode(mode);
        let _ = source.update_and_fetch(keys(0, 37), values(0, 37));
        let mut restored = standard_round_trip(&source, 37);
        truncate_standard(&mut restored, 0, NAMES).expect("truncate to zero");
        assert_eq!(restored.offset, 0);
        assert!(restored.is_empty(), "{mode:?}: nothing is left");

        let mut fresh = KVCache::new_with_mode(mode);
        let (fk, fv) = fresh.update_and_fetch(keys(0, 5), values(0, 5));
        let (rk, rv) = restored.update_and_fetch(keys(0, 5), values(0, 5));
        assert_same_bytes(&rk, &fk, &format!("{mode:?}: keys refilled from zero"));
        assert_same_bytes(&rv, &fv, &format!("{mode:?}: values refilled from zero"));
    }
}

/// An Int8 cache whose front was trimmed (`--max-kv-size`) keeps its live
/// window start across the round trip.
#[test]
fn an_int8_cache_with_a_trimmed_front_restores_its_live_window() {
    let mut source = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = source.update_and_fetch(keys(0, 40), values(0, 40));
    assert_eq!(source.trim_front(10), 10);
    assert_eq!((source.offset, source.seq_len()), (40, 30));

    let mut restored = standard_round_trip(&source, 40);
    assert_eq!(
        (restored.offset, restored.seq_len()),
        (40, 30),
        "the monotonic offset and the live length both survive"
    );
    assert_standard_steps_agree(
        &mut source,
        &mut restored,
        40,
        5,
        "Int8 with a trimmed front",
    );
}

/// A cache whose front was trimmed no longer holds the first rows, so its
/// snapshot cannot stand in for any shorter prefix: a cold prefill of fewer
/// tokens would start at row 0. Only the whole length is "truncatable",
/// which is no cut at all.
#[test]
fn a_front_trimmed_int8_snapshot_truncates_only_to_its_own_length() {
    let mut source = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = source.update_and_fetch(keys(0, 40), values(0, 40));
    assert_eq!(source.trim_front(10), 10);
    let mut snapshot = ModelStateSnapshot::new("test", 40);
    snapshot_standard(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");

    assert!(
        standard_truncatable_to(&snapshot, "layer0", 40, NAMES),
        "the whole snapshot is still restorable"
    );
    for target in [39, 30, 11, 10, 1, 0] {
        assert!(
            !standard_truncatable_to(&snapshot, "layer0", target, NAMES),
            "a front-trimmed snapshot must not be cut back to {target}"
        );
    }
    assert!(!standard_truncatable_to(&snapshot, "layer0", 41, NAMES));

    // The same cache without the front trim truncates to all of them.
    let mut untrimmed = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = untrimmed.update_and_fetch(keys(0, 40), values(0, 40));
    let mut snapshot = ModelStateSnapshot::new("test", 40);
    snapshot_standard(&untrimmed, &mut snapshot, "layer0", NAMES).expect("snapshot");
    for target in [40, 39, 30, 11, 10, 1, 0] {
        assert!(standard_truncatable_to(&snapshot, "layer0", target, NAMES));
    }
}

/// The store sizes an entry from the snapshot's tensors, so a quantized
/// snapshot has to weigh what the quantized cache weighs, at any length.
#[test]
fn a_quantized_snapshot_weighs_what_the_cache_holds() {
    for tokens in [4096, 32768] {
        let mut fp16 = KVCache::new();
        let _ = fp16.update_and_fetch(keys(0, tokens), values(0, tokens));
        let mut fp16_snapshot = ModelStateSnapshot::new("test", tokens as usize);
        snapshot_standard(&fp16, &mut fp16_snapshot, "layer0", NAMES).expect("fp16 snapshot");
        // Two i32 scalars (offset, mode) ride with the two FP16 buffers.
        let rows = (QH * tokens) as usize;
        let fp16_bytes = 2 * rows * QD as usize * 2 + 2 * 4;
        assert_eq!(fp16_snapshot.nbytes(), fp16_bytes, "fp16 at {tokens}");

        for mode in STANDARD_MODES {
            let mut cache = KVCache::new_with_mode(mode);
            let _ = cache.update_and_fetch(keys(0, tokens), values(0, tokens));
            let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
            snapshot_standard(&cache, &mut snapshot, "layer0", NAMES)
                .unwrap_or_else(|err| panic!("{mode:?} snapshot at {tokens}: {err}"));
            let scalars: usize = snapshot
                .tensor_names()
                .filter(|name| mlxcel_core::array_shape(snapshot.tensor(name).unwrap()) == vec![1])
                .count()
                * 4;
            assert_eq!(
                snapshot.nbytes(),
                cache.nbytes() + scalars,
                "{mode:?} at {tokens}: the snapshot is the cache's buffers plus its scalars"
            );
            let per_row = match mode {
                // int8 K and V, one FP16 scale each.
                KVCacheMode::Int8 => 2 * QD as usize + 2 * 2,
                // FP16 K, nibble-packed V, FP16 norm and rescale.
                KVCacheMode::Turbo4Asym => QD as usize * 2 + QD as usize / 2 + 2 * 2,
                // Nibble-packed K and V, FP16 K norm, V norm and V rescale.
                KVCacheMode::Turbo4 => 2 * (QD as usize / 2) + 3 * 2,
                other => unreachable!("{other:?} is not in STANDARD_MODES"),
            };
            assert_eq!(
                snapshot.nbytes() - scalars,
                rows * per_row,
                "{mode:?} at {tokens}: bytes per stored row"
            );
            assert!(
                snapshot.nbytes() < fp16_snapshot.nbytes(),
                "{mode:?} at {tokens}: smaller than the fp16 snapshot"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// KVCache: what stays refused, and what must not change
// ---------------------------------------------------------------------------

/// `Turbo4Delegated` (hot and cold V) and `Turbo3Asym` (3-bit packing) are
/// still not carried, and the refusal names the mode.
#[test]
fn delegated_and_three_bit_caches_are_still_refused_by_name() {
    for (mode, name) in [
        (KVCacheMode::Turbo4Delegated, "Turbo4Delegated"),
        (KVCacheMode::Turbo3Asym, "Turbo3Asym"),
    ] {
        let mut cache = KVCache::new_with_mode(mode);
        let _ = cache.update_and_fetch(keys(0, 8), values(0, 8));
        let mut snapshot = ModelStateSnapshot::new("test", 8);
        let err = snapshot_standard(&cache, &mut snapshot, "layer0", NAMES)
            .expect_err("this mode is not carried");
        assert!(
            err.contains(&format!(
                "cache mode {name} is not supported by model-state snapshots"
            )),
            "{err}"
        );
        assert!(snapshot.is_empty(), "a refused snapshot stores nothing");
    }
}

/// An empty quantized cache (a KV-shared layer) stores nothing and restores
/// to a fresh cache, in every mode.
#[test]
fn an_empty_quantized_cache_stores_nothing_and_restores_fresh() {
    for mode in [
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
        KVCacheMode::Turbo3Asym,
    ] {
        let mut snapshot = ModelStateSnapshot::new("test", 0);
        snapshot_standard(
            &KVCache::new_with_mode(mode),
            &mut snapshot,
            "layer0",
            NAMES,
        )
        .expect("an empty cache is not an error");
        assert!(snapshot.is_empty(), "{mode:?}: nothing stored");
        assert!(standard_truncatable_to(&snapshot, "layer0", 0, NAMES));

        let mut restored = KVCache::new_with_mode(mode);
        restore_standard(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
        assert!(restored.is_empty(), "{mode:?}: the cache stays fresh");
        truncate_standard(&mut restored, 0, NAMES).expect("a fresh cache truncates to zero");
    }
}

/// A snapshot lands only in a cache configured for the mode it was taken in.
#[test]
fn a_quantized_snapshot_is_refused_by_a_cache_of_another_mode() {
    let all = [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ];
    for stored in all {
        let mut source = KVCache::new_with_mode(stored);
        let _ = source.update_and_fetch(keys(0, 8), values(0, 8));
        let mut snapshot = ModelStateSnapshot::new("test", 8);
        snapshot_standard(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");
        for live in all.into_iter().filter(|live| *live != stored) {
            let mut target = KVCache::new_with_mode(live);
            let err = restore_standard(&mut target, &snapshot, "layer0", NAMES)
                .expect_err("a snapshot of another mode is refused");
            assert!(
                err.contains("does not match configured cache mode"),
                "{stored:?} into {live:?}: {err}"
            );
            assert!(
                target.is_empty() && target.offset == 0,
                "{stored:?} into {live:?}: a refused restore leaves the cache untouched"
            );
        }
    }
}

/// A snapshot that lost one of the buffers its mode needs is refused before
/// anything is assigned, so the caller falls back to a cold prefill.
#[test]
fn a_quantized_snapshot_missing_a_buffer_is_refused_whole() {
    for mode in STANDARD_MODES {
        let mut source = KVCache::new_with_mode(mode);
        let _ = source.update_and_fetch(keys(0, 8), values(0, 8));
        let mut snapshot = ModelStateSnapshot::new("test", 8);
        snapshot_standard(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");

        // Every stored buffer in turn (the rank-4 tensors; the scalars have
        // documented fallbacks and are covered by the parts tests).
        let buffers: Vec<String> = tensor_names(&snapshot)
            .into_iter()
            .filter(|name| mlxcel_core::array_shape(snapshot.tensor(name).unwrap()).len() == 4)
            .collect();
        assert!(buffers.len() >= 4, "{mode:?} stores {buffers:?}");
        for missing in &buffers {
            let partial = snapshot.subset(|name| name != missing);
            let mut target = KVCache::new_with_mode(mode);
            let outcome = restore_standard(&mut target, &partial, "layer0", NAMES);
            assert!(
                outcome.is_err(),
                "{mode:?}: a snapshot without {missing} must be refused"
            );
            assert!(
                target.is_empty() && target.offset == 0 && target.nbytes() == 0,
                "{mode:?} without {missing}: the cache must be left untouched"
            );
        }
    }
}

/// What an FP16 cache stores is exactly what it stored before quantized
/// snapshots existed: these names, in this order, and nothing else.
#[test]
fn an_fp16_snapshot_stores_the_same_names_as_before() {
    let mut standard = KVCache::new();
    let _ = standard.update_and_fetch(keys(0, 8), values(0, 8));
    let mut snapshot = ModelStateSnapshot::new("test", 8);
    snapshot_standard(&standard, &mut snapshot, "layer0", NAMES).expect("snapshot");
    assert_eq!(
        tensor_names(&snapshot),
        [
            "layer0.standard.keys",
            "layer0.standard.values",
            "layer0.standard.offset",
            "layer0.standard.mode",
        ]
    );

    let mut rotating = RotatingKVCache::new(WINDOW);
    let _ = rotating.update_and_fetch(keys(0, 8), values(0, 8));
    let mut snapshot = ModelStateSnapshot::new("test", 8);
    snapshot_rotating(&rotating, &mut snapshot, "layer1", NAMES).expect("snapshot");
    assert_eq!(
        tensor_names(&snapshot),
        [
            "layer1.rotating.keys",
            "layer1.rotating.values",
            "layer1.rotating.max_size",
            "layer1.rotating.buffer_size",
            "layer1.rotating.offset",
            "layer1.rotating.start_position",
            "layer1.rotating.idx",
            "layer1.rotating.step",
            "layer1.rotating.mode",
            "layer1.rotating.turbo_seed",
        ]
    );
}

// ---------------------------------------------------------------------------
// RotatingKVCache: an FP16 ring under a quantized label
// ---------------------------------------------------------------------------

/// A sliding ring built with `Int8`, `Turbo4`, `Turbo4Delegated` or
/// `Turbo3Asym` stores plain FP16 rows, so its snapshot is an FP16 ring
/// snapshot that keeps the label. Filled to before the window, exactly the
/// window, and past it (wrapped), by decode steps.
#[test]
fn a_labelled_fp16_ring_round_trips_before_at_and_after_the_wrap() {
    for label in FP16_RING_LABELS {
        for filled in [1, 40, WINDOW - 1, WINDOW, WINDOW + 1, 100, 3 * WINDOW] {
            let mut source = RotatingKVCache::new_with_mode(WINDOW, label);
            for t in 0..filled {
                let _ = source.update_and_fetch(keys(t, 1), values(t, 1));
            }
            let mut restored = rotating_round_trip(&source, filled);
            assert_eq!(
                restored.snapshot_state(),
                source.snapshot_state(),
                "{label:?} at {filled}: ring scalars, the label included"
            );
            assert_eq!(restored.mode, label, "{label:?}: the label is kept");
            // Far enough to cross the next wrap from any of the fill levels.
            assert_rotating_steps_agree(
                &mut source,
                &mut restored,
                filled,
                WINDOW + 3,
                &format!("{label:?} ring restored at {filled}"),
            );
        }
    }
}

/// The rows such a ring stores are the rows an FP16 ring stores: the two
/// snapshots differ in the `mode` scalar and in nothing else.
#[test]
fn a_labelled_fp16_ring_stores_what_an_fp16_ring_stores() {
    for label in FP16_RING_LABELS {
        for filled in [40, 100] {
            let mut plain = RotatingKVCache::new(WINDOW);
            let mut labelled = RotatingKVCache::new_with_mode(WINDOW, label);
            for t in 0..filled {
                let _ = plain.update_and_fetch(keys(t, 1), values(t, 1));
                let _ = labelled.update_and_fetch(keys(t, 1), values(t, 1));
            }
            let mut plain_snapshot = ModelStateSnapshot::new("test", filled as usize);
            snapshot_rotating(&plain, &mut plain_snapshot, "layer0", NAMES).expect("fp16");
            let mut labelled_snapshot = ModelStateSnapshot::new("test", filled as usize);
            snapshot_rotating(&labelled, &mut labelled_snapshot, "layer0", NAMES)
                .unwrap_or_else(|err| panic!("{label:?} ring snapshot: {err}"));

            assert_eq!(
                tensor_names(&labelled_snapshot),
                tensor_names(&plain_snapshot),
                "{label:?}: the same tensors and scalars"
            );
            for name in tensor_names(&plain_snapshot) {
                let plain_tensor = plain_snapshot.tensor(&name).unwrap();
                let labelled_tensor = labelled_snapshot.tensor(&name).unwrap();
                if name.ends_with(".mode") {
                    assert_ne!(
                        mlxcel_core::item_i32(labelled_tensor),
                        mlxcel_core::item_i32(plain_tensor),
                        "{label:?}: the label is what is stored"
                    );
                } else {
                    assert_same_bytes(labelled_tensor, plain_tensor, &format!("{label:?} {name}"));
                }
            }
        }
    }
}

/// A multi-token prefill longer than the window leaves an oversized buffer
/// for one step; the snapshot taken there restores under a label too.
#[test]
fn a_labelled_fp16_ring_round_trips_after_an_over_window_prefill() {
    for label in FP16_RING_LABELS {
        let prefill = WINDOW + 20;
        let mut source = RotatingKVCache::new_with_mode(WINDOW, label);
        let _ = source.update_and_fetch(keys(0, prefill), values(0, prefill));
        let mut restored = rotating_round_trip(&source, prefill);
        assert_rotating_steps_agree(
            &mut source,
            &mut restored,
            prefill,
            WINDOW + 3,
            &format!("{label:?} ring after an over-window prefill"),
        );
    }
}

/// Truncation follows the ring, not the label: allowed while the ring is
/// unwrapped, refused once it has wrapped.
#[test]
fn a_labelled_fp16_ring_truncates_only_while_unwrapped() {
    for label in FP16_RING_LABELS {
        let mut unwrapped = RotatingKVCache::new_with_mode(WINDOW, label);
        for t in 0..40 {
            let _ = unwrapped.update_and_fetch(keys(t, 1), values(t, 1));
        }
        let mut snapshot = ModelStateSnapshot::new("test", 40);
        snapshot_rotating(&unwrapped, &mut snapshot, "layer0", NAMES)
            .unwrap_or_else(|err| panic!("{label:?} ring snapshot: {err}"));
        assert!(
            rotating_truncatable_to(&snapshot, "layer0", 25, NAMES),
            "{label:?}: an unwrapped ring truncates"
        );
        assert!(!rotating_truncatable_to(&snapshot, "layer0", 41, NAMES));

        let mut restored = RotatingKVCache::new_with_mode(WINDOW, label);
        restore_rotating(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
        truncate_rotating(&mut restored, 25, NAMES).expect("truncate");
        let mut cold = RotatingKVCache::new_with_mode(WINDOW, label);
        for t in 0..25 {
            let _ = cold.update_and_fetch(keys(t, 1), values(t, 1));
        }
        // Across the wrap, so a wrong cursor would show.
        for t in 25..25 + WINDOW + 3 {
            let (ck, cv) = cold.update_and_fetch(keys(t, 1), values(t, 1));
            let (rk, rv) = restored.update_and_fetch(keys(t, 1), values(t, 1));
            assert_same_bytes(
                &rk,
                &ck,
                &format!("{label:?} cut ring: keys after token {t}"),
            );
            assert_same_bytes(
                &rv,
                &cv,
                &format!("{label:?} cut ring: values after token {t}"),
            );
        }

        let mut wrapped = RotatingKVCache::new_with_mode(WINDOW, label);
        for t in 0..100 {
            let _ = wrapped.update_and_fetch(keys(t, 1), values(t, 1));
        }
        let mut snapshot = ModelStateSnapshot::new("test", 100);
        snapshot_rotating(&wrapped, &mut snapshot, "layer0", NAMES)
            .unwrap_or_else(|err| panic!("{label:?} wrapped ring snapshot: {err}"));
        assert!(
            !rotating_truncatable_to(&snapshot, "layer0", 90, NAMES),
            "{label:?}: a wrapped ring does not truncate"
        );
        let mut restored = RotatingKVCache::new_with_mode(WINDOW, label);
        restore_rotating(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
        let err = truncate_rotating(&mut restored, 90, NAMES)
            .expect_err("a wrapped ring cannot be trimmed");
        assert!(err.contains("has wrapped"), "{label:?}: {err}");
    }
}

/// A ring snapshot lands only in a ring built with the same label, even
/// though every one of these labels stores the same FP16 rows.
#[test]
fn a_ring_snapshot_is_refused_by_a_ring_of_another_label() {
    let mut source = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Int8);
    for t in 0..40 {
        let _ = source.update_and_fetch(keys(t, 1), values(t, 1));
    }
    let mut snapshot = ModelStateSnapshot::new("test", 40);
    snapshot_rotating(&source, &mut snapshot, "layer0", NAMES).expect("Int8 ring snapshot");
    for live in [
        KVCacheMode::Fp16,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo3Asym,
    ] {
        let mut target = RotatingKVCache::new_with_mode(WINDOW, live);
        let err = restore_rotating(&mut target, &snapshot, "layer0", NAMES)
            .expect_err("a ring of another label refuses the snapshot");
        assert!(
            err.contains("does not match configured cache mode"),
            "Int8 into {live:?}: {err}"
        );
        assert!(
            target.is_empty(),
            "Int8 into {live:?}: the ring stays fresh"
        );
    }
}

// ---------------------------------------------------------------------------
// RotatingKVCache: Turbo4Asym, which really quantizes V
// ---------------------------------------------------------------------------

/// The quantized ring round-trips before the window is full, exactly at it,
/// one past it and well after it has wrapped, and the restored ring then
/// agrees with the source step by step across the next wrap.
#[test]
fn a_turbo4_asym_ring_round_trips_before_at_and_after_the_wrap() {
    for filled in [
        1,
        31,
        32,
        40,
        WINDOW - 1,
        WINDOW,
        WINDOW + 1,
        100,
        3 * WINDOW,
    ] {
        let mut source = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
        for t in 0..filled {
            let _ = source.update_and_fetch(keys(t, 1), values(t, 1));
        }
        let mut restored = rotating_round_trip(&source, filled);
        assert_eq!(
            restored.snapshot_state(),
            source.snapshot_state(),
            "Turbo4Asym ring at {filled}: ring scalars"
        );
        assert_rotating_steps_agree(
            &mut source,
            &mut restored,
            filled,
            WINDOW + 3,
            &format!("Turbo4Asym ring restored at {filled}"),
        );
    }
}

/// The same from prefilled rings: a short prefill, one that fills the window
/// exactly, and one longer than the window.
#[test]
fn a_turbo4_asym_ring_round_trips_after_a_prefill() {
    for prefill in [20, WINDOW, WINDOW + 20] {
        let mut source = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
        let _ = source.update_and_fetch(keys(0, prefill), values(0, prefill));
        let mut restored = rotating_round_trip(&source, prefill);
        assert_rotating_steps_agree(
            &mut source,
            &mut restored,
            prefill,
            WINDOW + 3,
            &format!("Turbo4Asym ring after a {prefill}-token prefill"),
        );

        // And a second prefill chunk after the restore.
        let mut source = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
        let _ = source.update_and_fetch(keys(0, prefill), values(0, prefill));
        let mut restored = rotating_round_trip(&source, prefill);
        let (sk, sv) = source.update_and_fetch(keys(prefill, 9), values(prefill, 9));
        let (rk, rv) = restored.update_and_fetch(keys(prefill, 9), values(prefill, 9));
        assert_same_bytes(
            &rk,
            &sk,
            &format!("chunk after a {prefill}-token prefill: keys"),
        );
        assert_same_bytes(
            &rv,
            &sv,
            &format!("chunk after a {prefill}-token prefill: values"),
        );
        assert_rotating_steps_agree(
            &mut source,
            &mut restored,
            prefill + 9,
            WINDOW + 3,
            &format!("Turbo4Asym ring after a {prefill}-token prefill and a chunk"),
        );
    }
}

/// An unwrapped quantized ring truncates to a cold fill of the prefix, and a
/// wrapped one is refused, as for FP16.
#[test]
fn a_turbo4_asym_ring_truncates_only_while_unwrapped() {
    let mut unwrapped = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
    for t in 0..40 {
        let _ = unwrapped.update_and_fetch(keys(t, 1), values(t, 1));
    }
    let mut snapshot = ModelStateSnapshot::new("test", 40);
    snapshot_rotating(&unwrapped, &mut snapshot, "layer0", NAMES).expect("snapshot");
    for target in [40, 33, 32, 25, 1] {
        assert!(
            rotating_truncatable_to(&snapshot, "layer0", target, NAMES),
            "an unwrapped ring truncates to {target}"
        );
        let mut restored = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
        restore_rotating(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
        truncate_rotating(&mut restored, target, NAMES).expect("truncate");
        let mut cold = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
        for t in 0..target {
            let _ = cold.update_and_fetch(keys(t, 1), values(t, 1));
        }
        for t in target..target + WINDOW + 3 {
            let (ck, cv) = cold.update_and_fetch(keys(t, 1), values(t, 1));
            let (rk, rv) = restored.update_and_fetch(keys(t, 1), values(t, 1));
            assert_same_bytes(&rk, &ck, &format!("cut to {target}: keys after token {t}"));
            assert_same_bytes(
                &rv,
                &cv,
                &format!("cut to {target}: values after token {t}"),
            );
        }
    }
    assert!(!rotating_truncatable_to(&snapshot, "layer0", 41, NAMES));

    let mut wrapped = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
    for t in 0..100 {
        let _ = wrapped.update_and_fetch(keys(t, 1), values(t, 1));
    }
    let mut snapshot = ModelStateSnapshot::new("test", 100);
    snapshot_rotating(&wrapped, &mut snapshot, "layer0", NAMES).expect("snapshot");
    assert!(
        !rotating_truncatable_to(&snapshot, "layer0", 90, NAMES),
        "a wrapped ring does not truncate"
    );
    let mut restored = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
    restore_rotating(&mut restored, &snapshot, "layer0", NAMES).expect("restore");
    let err =
        truncate_rotating(&mut restored, 90, NAMES).expect_err("a wrapped ring cannot be trimmed");
    assert!(err.contains("has wrapped"), "{err}");
}

/// A quantized ring snapshot that lost one of its buffers is refused whole.
#[test]
fn a_turbo4_asym_ring_snapshot_missing_a_buffer_is_refused_whole() {
    let mut source = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
    for t in 0..40 {
        let _ = source.update_and_fetch(keys(t, 1), values(t, 1));
    }
    let mut snapshot = ModelStateSnapshot::new("test", 40);
    snapshot_rotating(&source, &mut snapshot, "layer0", NAMES).expect("snapshot");
    let buffers: Vec<String> = tensor_names(&snapshot)
        .into_iter()
        .filter(|name| mlxcel_core::array_shape(snapshot.tensor(name).unwrap()).len() == 4)
        .collect();
    assert_eq!(buffers.len(), 4, "keys and three V sidecars: {buffers:?}");
    for missing in &buffers {
        let partial = snapshot.subset(|name| name != missing);
        let mut target = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
        // Missing keys are refused as well: packed values with no keys are a
        // quantized ring that lost a buffer, not a layer that stored nothing.
        let outcome = restore_rotating(&mut target, &partial, "layer0", NAMES);
        assert!(
            outcome.is_err(),
            "a snapshot without {missing} must be refused"
        );
        assert!(
            target.is_empty() && target.get_offset() == 0,
            "without {missing}: the ring must be left untouched"
        );
    }
}
