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

//! The snapshot views of [`KVCache`] and [`RotatingKVCache`] (issue #3250).
//!
//! A restored cache has to be the cache it was copied from: every stored
//! buffer is compared byte for byte, dtype and shape included, and so are the
//! K/V the two caches return afterwards. The other half of the file is the
//! layout check, which has to refuse a malformed set without assigning and
//! without reaching a panic in the Turbo4 parameter builder.

use super::*;
use crate::dtype;

const HEADS: i32 = 2;
const HEAD_DIM: i32 = 64;
const WINDOW: i32 = 64;

const QUANTIZED: [KVCacheMode; 3] = [
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
];

/// `n` FP16 tokens whose values depend on the absolute position only, with a
/// per-token amplitude so that scales and norms differ from row to row.
fn rows(start: i32, n: i32, phase: f32) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity((HEADS * n * HEAD_DIM) as usize);
    for h in 0..HEADS {
        for t in 0..n {
            let pos = start + t;
            let amplitude = 0.25 + (pos % 7) as f32 * 0.125;
            for d in 0..HEAD_DIM {
                let angle = h as f32 * 1.7 + pos as f32 * 0.37 + d as f32 * 0.11 + phase;
                data.push(angle.sin() * amplitude);
            }
        }
    }
    ffi::astype(
        &ffi::from_slice_f32(&data, &[1, HEADS, n, HEAD_DIM]),
        dtype::FLOAT16,
    )
}

fn bytes(array: &MlxArray) -> Vec<u8> {
    ffi::eval(array);
    ffi::array_to_raw_bytes(array)
}

fn assert_same(actual: Option<&MlxArray>, expected: Option<&MlxArray>, what: &str) {
    match (actual, expected) {
        (None, None) => {}
        (Some(actual), Some(expected)) => {
            assert_eq!(
                ffi::array_shape(actual),
                ffi::array_shape(expected),
                "{what}: shape"
            );
            assert_eq!(
                ffi::array_dtype(actual),
                ffi::array_dtype(expected),
                "{what}: dtype"
            );
            assert!(bytes(actual) == bytes(expected), "{what}: bytes differ");
        }
        (actual, expected) => panic!(
            "{what}: present in one cache only (restored {}, source {})",
            actual.is_some(),
            expected.is_some()
        ),
    }
}

/// Every buffer and scalar of the two caches agrees.
fn assert_same_cache(restored: &KVCache, source: &KVCache, what: &str) {
    let (r, s) = (restored.snapshot_parts(), source.snapshot_parts());
    assert_eq!(
        (r.mode, r.offset, r.live_start, r.turbo_seed),
        (s.mode, s.offset, s.live_start, s.turbo_seed),
        "{what}: scalars"
    );
    assert_same(r.keys, s.keys, &format!("{what}: keys"));
    assert_same(r.values, s.values, &format!("{what}: values"));
    for name in KV_SIDECAR_FIELDS {
        assert_same(r.sidecar(name), s.sidecar(name), &format!("{what}: {name}"));
    }
    assert_eq!(restored.nbytes(), source.nbytes(), "{what}: nbytes");
}

/// A copy of the cache's buffers, the way a snapshot hands them to a restore.
fn owned(parts: KVCacheSnapshotParts<'_>) -> KVCacheSnapshotOwned {
    let copy = |array: Option<&MlxArray>| array.map(ffi::copy);
    KVCacheSnapshotOwned {
        mode: parts.mode,
        offset: parts.offset,
        live_start: parts.live_start,
        turbo_seed: parts.turbo_seed,
        keys: copy(parts.keys),
        values: copy(parts.values),
        key_scales: copy(parts.key_scales),
        val_scales: copy(parts.val_scales),
        v_packed: copy(parts.v_packed),
        v_norms: copy(parts.v_norms),
        v_rescale: copy(parts.v_rescale),
        k_packed: copy(parts.k_packed),
        k_norms: copy(parts.k_norms),
    }
}

fn filled(mode: KVCacheMode, tokens: i32) -> KVCache {
    let mut cache = KVCache::new_with_mode(mode);
    let _ = cache.update_and_fetch(rows(0, tokens, 0.0), rows(0, tokens, 1.0));
    cache
}

fn restored_from(source: &KVCache) -> KVCache {
    let mut restored = KVCache::new_with_mode(source.mode);
    restored
        .restore_snapshot_parts(owned(source.snapshot_parts()))
        .unwrap_or_else(|err| panic!("{:?} restore: {err}", source.mode));
    restored
}

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

#[test]
fn a_restored_cache_holds_the_source_buffers_byte_for_byte() {
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ] {
        // One token, a turbo block edge, an ordinary length, and both sides
        // of a buffer growth step.
        for tokens in [1, 32, 37, 256, 257] {
            let mut source = filled(mode, tokens);
            let mut restored = restored_from(&source);
            let what = format!("{mode:?} at {tokens}");
            assert_same_cache(&restored, &source, &what);

            // And they stay the same cache through decode steps and a chunk.
            for t in tokens..tokens + 3 {
                let (sk, sv) = source.update_and_fetch(rows(t, 1, 0.0), rows(t, 1, 1.0));
                let (rk, rv) = restored.update_and_fetch(rows(t, 1, 0.0), rows(t, 1, 1.0));
                assert_same(Some(&rk), Some(&sk), &format!("{what}: keys after {t}"));
                assert_same(Some(&rv), Some(&sv), &format!("{what}: values after {t}"));
            }
            let next = tokens + 3;
            let _ = source.update_and_fetch(rows(next, 40, 0.0), rows(next, 40, 1.0));
            let _ = restored.update_and_fetch(rows(next, 40, 0.0), rows(next, 40, 1.0));
            assert_eq!(restored.offset, source.offset);
            // The physical capacity may differ after a growth, the live rows
            // may not: compare what the next fetch returns.
            let (sk, sv) =
                source.update_and_fetch(rows(next + 40, 1, 0.0), rows(next + 40, 1, 1.0));
            let (rk, rv) =
                restored.update_and_fetch(rows(next + 40, 1, 0.0), rows(next + 40, 1, 1.0));
            assert_same(Some(&rk), Some(&sk), &format!("{what}: keys after a chunk"));
            assert_same(
                Some(&rv),
                Some(&sv),
                &format!("{what}: values after a chunk"),
            );
        }
    }
}

#[test]
fn each_mode_stores_exactly_its_own_buffers() {
    let present = |cache: &KVCache| -> Vec<&'static str> {
        let parts = cache.snapshot_parts();
        let mut names = Vec::new();
        if parts.keys.is_some() {
            names.push("keys");
        }
        if parts.values.is_some() {
            names.push("values");
        }
        names.extend(
            KV_SIDECAR_FIELDS
                .into_iter()
                .filter(|name| parts.sidecar(name).is_some()),
        );
        names
    };
    assert_eq!(present(&filled(KVCacheMode::Fp16, 8)), ["keys", "values"]);
    assert_eq!(
        present(&filled(KVCacheMode::Int8, 8)),
        ["keys", "values", "key_scales", "val_scales"]
    );
    assert_eq!(
        present(&filled(KVCacheMode::Turbo4Asym, 8)),
        ["keys", "v_packed", "v_norms", "v_rescale"]
    );
    assert_eq!(
        present(&filled(KVCacheMode::Turbo4, 8)),
        ["v_packed", "v_norms", "v_rescale", "k_packed", "k_norms"]
    );
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ] {
        filled(mode, 8)
            .snapshot_parts()
            .validate()
            .unwrap_or_else(|err| panic!("{mode:?}: a cache the mode filled validates: {err}"));
        assert!(
            KVCache::new_with_mode(mode)
                .snapshot_parts()
                .holds_no_rows(),
            "{mode:?}: a fresh cache holds nothing"
        );
    }
}

#[test]
fn a_fresh_cache_and_an_empty_set_restore_to_a_fresh_cache() {
    for mode in QUANTIZED {
        let mut cache = KVCache::new_with_mode(mode);
        cache
            .restore_snapshot_parts(KVCacheSnapshotOwned {
                mode,
                ..KVCacheSnapshotOwned::default()
            })
            .expect("a set with no rows and no tokens is a layer that stored nothing");
        assert!(cache.is_empty() && cache.nbytes() == 0);

        let err = cache
            .restore_snapshot_parts(KVCacheSnapshotOwned {
                mode,
                offset: 5,
                ..KVCacheSnapshotOwned::default()
            })
            .expect_err("tokens with no rows behind them");
        assert!(err.contains("holds no rows"), "{mode:?}: {err}");
        assert_eq!(cache.offset, 0);
    }
}

#[test]
fn the_restored_cache_takes_the_snapshot_seed_and_its_parameters() {
    for mode in [KVCacheMode::Turbo4Asym, KVCacheMode::Turbo4] {
        let mut source = KVCache::new_with_mode_and_seed(mode, 0x5EED_1234);
        let _ = source.update_and_fetch(rows(0, 37, 0.0), rows(0, 37, 1.0));

        // The receiving cache was built with another seed.
        let mut restored = KVCache::new_with_mode(mode);
        restored
            .restore_snapshot_parts(owned(source.snapshot_parts()))
            .expect("restore");
        assert_eq!(restored.turbo_seed, 0x5EED_1234);

        // Built at restore time, for a reader that dequantizes before it
        // appends, and equal to what the source's first update built.
        let (built, lazy) = (
            restored.turbo_params().expect("parameters are ready"),
            source.turbo_params().expect("the source has updated"),
        );
        assert_eq!(built.head_dim, lazy.head_dim);
        assert_eq!(built.signs1, lazy.signs1);
        assert_eq!(built.signs2, lazy.signs2);
        assert_eq!(built.k_signs1, lazy.k_signs1);
        assert_eq!(built.k_signs2, lazy.k_signs2);

        let (sk, sv) = source.update_and_fetch(rows(37, 1, 0.0), rows(37, 1, 1.0));
        let (rk, rv) = restored.update_and_fetch(rows(37, 1, 0.0), rows(37, 1, 1.0));
        assert_same(
            Some(&rk),
            Some(&sk),
            &format!("{mode:?}: keys under the carried seed"),
        );
        assert_same(
            Some(&rv),
            Some(&sv),
            &format!("{mode:?}: values under the carried seed"),
        );
    }
}

#[test]
fn an_int8_front_trim_survives_the_round_trip() {
    let mut source = filled(KVCacheMode::Int8, 40);
    assert_eq!(source.trim_front(10), 10);
    let parts = source.snapshot_parts();
    assert_eq!((parts.offset, parts.live_start), (40, 10));
    parts
        .validate()
        .expect("a front-trimmed Int8 cache validates");

    let mut restored = restored_from(&source);
    assert_same_cache(&restored, &source, "Int8 with a trimmed front");
    let (sk, sv) = source.update_and_fetch(rows(40, 1, 0.0), rows(40, 1, 1.0));
    let (rk, rv) = restored.update_and_fetch(rows(40, 1, 0.0), rows(40, 1, 1.0));
    assert_eq!(
        ffi::array_shape(&rk)[2],
        31,
        "the live window, not the offset"
    );
    assert_same(Some(&rk), Some(&sk), "keys after the front trim");
    assert_same(Some(&rv), Some(&sv), "values after the front trim");
}

#[test]
fn a_restore_shares_the_snapshot_buffers_without_changing_them() {
    for mode in QUANTIZED {
        let source = filled(mode, 37);
        let snapshot = owned(source.snapshot_parts());
        let reference = owned(snapshot.as_parts());

        let mut restored = KVCache::new_with_mode(mode);
        restored
            .restore_snapshot_parts(owned(snapshot.as_parts()))
            .expect("restore");
        for t in 37..70 {
            let _ = restored.update_and_fetch(rows(t, 1, 3.0), rows(t, 1, 4.0));
        }
        let (kept, was) = (snapshot.as_parts(), reference.as_parts());
        assert_same(kept.keys, was.keys, &format!("{mode:?}: snapshot keys"));
        assert_same(
            kept.values,
            was.values,
            &format!("{mode:?}: snapshot values"),
        );
        for name in KV_SIDECAR_FIELDS {
            assert_same(
                kept.sidecar(name),
                was.sidecar(name),
                &format!("{mode:?}: snapshot {name}"),
            );
        }
    }
}

/// A zero buffer with the shape and dtype of `source`, after `edit` changed
/// its shape, or in `dtype` when one is given. `None` when `source` is absent, so a
/// case only replaces a buffer the mode really holds.
///
/// Every malformed case below is built this way, from the source's own
/// buffer with exactly one property changed. A replacement made up from
/// scratch would differ in row count too (the buffers are allocated a growth
/// step at a time), and the case would then pass on the length check
/// whatever happened to the check it is named after.
fn like(
    source: &Option<UniquePtr<MlxArray>>,
    dtype: Option<i32>,
    edit: impl FnOnce(&mut Vec<i32>),
) -> Option<UniquePtr<MlxArray>> {
    source.as_deref().map(|array| {
        let mut shape = ffi::array_shape(array);
        edit(&mut shape);
        ffi::zeros(&shape, dtype.unwrap_or_else(|| ffi::array_dtype(array)))
    })
}

/// Replace the per-token FP16 sidecar the mode holds (`val_scales` for Int8,
/// `v_norms` for the Turbo4 modes) with an edited copy.
fn edit_per_token_sidecar(
    parts: &mut KVCacheSnapshotOwned,
    dtype: Option<i32>,
    edit: impl Fn(&mut Vec<i32>),
) {
    parts.val_scales = like(&parts.val_scales, dtype, &edit);
    parts.v_norms = like(&parts.v_norms, dtype, &edit);
}

/// Apply `corrupt` to a good set of `mode` and require the restore to refuse
/// it and leave the target untouched.
fn assert_refused(mode: KVCacheMode, what: &str, corrupt: impl FnOnce(&mut KVCacheSnapshotOwned)) {
    let source = filled(mode, 37);
    let mut parts = owned(source.snapshot_parts());
    corrupt(&mut parts);
    let mut target = KVCache::new_with_mode(mode);
    let outcome = target.restore_snapshot_parts(parts);
    assert!(outcome.is_err(), "{mode:?}: {what} must be refused");
    assert!(
        target.is_empty() && target.offset == 0 && target.nbytes() == 0,
        "{mode:?}: {what} must leave the cache untouched"
    );
    assert!(target.turbo_params().is_none(), "{mode:?}: {what}");
}

#[test]
fn a_malformed_set_is_refused_before_anything_is_assigned() {
    for mode in QUANTIZED {
        // Each buffer the mode holds, missing in turn.
        let held: Vec<&str> = {
            let source = filled(mode, 8);
            let parts = source.snapshot_parts();
            ["keys", "values"]
                .into_iter()
                .filter(|name| match *name {
                    "keys" => parts.keys.is_some(),
                    _ => parts.values.is_some(),
                })
                .chain(
                    KV_SIDECAR_FIELDS
                        .into_iter()
                        .filter(|name| parts.sidecar(name).is_some()),
                )
                .collect()
        };
        for name in held {
            assert_refused(mode, &format!("a set without {name}"), |parts| match name {
                "keys" => parts.keys = None,
                "values" => parts.values = None,
                other => parts.set_sidecar(other, None),
            });
        }
        // The buffers are allocated a growth step at a time, so they hold
        // more rows than the 37 tokens written. What the check can see is an
        // offset the buffers cannot physically cover.
        assert_refused(mode, "an offset past the buffers", |parts| {
            let rows = parts.v_packed.as_deref().or(parts.keys.as_deref());
            parts.offset = ffi::array_shape(rows.expect("a row buffer"))[2] + 1;
        });
        assert_refused(mode, "a negative offset", |parts| parts.offset = -1);
        assert_refused(mode, "a live start past the offset", |parts| {
            parts.live_start = 38;
        });
        assert_refused(mode, "a negative live start", |parts| parts.live_start = -1);
        assert_refused(mode, "another mode's set", |parts| {
            parts.mode = KVCacheMode::Fp16;
        });
        assert_refused(mode, "a sidecar of another length", |parts| {
            edit_per_token_sidecar(parts, None, |shape| shape[2] += 1);
        });
        assert_refused(mode, "a sidecar of another head count", |parts| {
            edit_per_token_sidecar(parts, None, |shape| shape[1] += 1);
        });
        assert_refused(mode, "a sidecar of another batch", |parts| {
            edit_per_token_sidecar(parts, None, |shape| shape[0] += 1);
        });
        assert_refused(mode, "a rank-3 sidecar", |parts| {
            edit_per_token_sidecar(parts, None, |shape| {
                shape.remove(0);
            });
        });
        assert_refused(mode, "a sidecar of another dtype", |parts| {
            edit_per_token_sidecar(parts, Some(dtype::FLOAT32), |_| {});
        });
        assert_refused(mode, "a per-token sidecar two wide", |parts| {
            edit_per_token_sidecar(parts, None, |shape| shape[3] = 2);
        });
    }
    // A front trim only exists on Fp16 and Int8 caches.
    for mode in [KVCacheMode::Turbo4Asym, KVCacheMode::Turbo4] {
        assert_refused(mode, "a trimmed front", |parts| parts.live_start = 1);
        // A packed width that is not half a power of two would panic in the
        // Turbo4 parameter builder; the restore has to refuse it first. K and
        // V keep one width here, so the width is the only thing wrong.
        for width in [0, 3, 24, 48] {
            assert_refused(mode, &format!("a packed width of {width}"), |parts| {
                parts.v_packed = like(&parts.v_packed, None, |shape| shape[3] = width);
                parts.k_packed = like(&parts.k_packed, None, |shape| shape[3] = width);
            });
        }
        assert_refused(mode, "packed values that are not uint8", |parts| {
            parts.v_packed = like(&parts.v_packed, Some(dtype::INT8), |_| {});
        });
    }
    // Symmetric Turbo4 has one head dim for K and V. 16 bytes is a valid
    // width on its own (a head dim of 32), so only the disagreement is wrong.
    assert_refused(
        KVCacheMode::Turbo4,
        "K and V of different head dims",
        |parts| {
            parts.k_packed = like(&parts.k_packed, None, |shape| shape[3] = 16);
        },
    );
    // Turbo4Asym keeps K as FP16 rows.
    assert_refused(KVCacheMode::Turbo4Asym, "keys that are not FP16", |parts| {
        parts.keys = like(&parts.keys, Some(dtype::FLOAT32), |_| {});
    });
    // An Int8 cache stores int8 rows. Rows of another dtype are not this
    // layout.
    assert_refused(KVCacheMode::Int8, "rows that are not int8", |parts| {
        parts.keys = like(&parts.keys, Some(dtype::FLOAT16), |_| {});
    });
}

/// An absorbed MLA latent under `Int8` keeps its affine rows in the slots a
/// per-head Int8 cache uses: uint32 packed rows in `keys`, FP16 `k_pe` in
/// `values`, group scales and biases in the scale slots. A snapshot does not
/// carry that layout, and says so by name rather than by its first
/// mismatching buffer.
#[test]
fn an_int8_cache_holding_a_quantized_mla_latent_is_refused_by_name() {
    let mut latent = KVCache::new_with_mode(KVCacheMode::Int8);
    latent.keys = Some(ffi::zeros(&[1, 1, 37, 128], dtype::UINT32));
    latent.values = Some(ffi::zeros(&[1, 1, 37, 64], dtype::FLOAT16));
    latent.key_scales = Some(ffi::zeros(&[1, 1, 37, 8], dtype::FLOAT16));
    latent.val_scales = Some(ffi::zeros(&[1, 1, 37, 8], dtype::FLOAT16));
    latent.offset = 37;
    assert!(latent.holds_quantized_latent());

    let err = latent
        .snapshot_parts()
        .validate()
        .expect_err("a quantized latent is not carried");
    assert!(
        err.contains("quantized MLA latent is not supported by model-state snapshots"),
        "{err}"
    );
    let mut target = KVCache::new_with_mode(KVCacheMode::Int8);
    let err = target
        .restore_snapshot_parts(owned(latent.snapshot_parts()))
        .expect_err("and not restored");
    assert!(err.contains("quantized MLA latent"), "{err}");
    assert!(target.is_empty() && target.nbytes() == 0);
}

#[test]
fn delegated_and_three_bit_caches_are_not_snapshotable() {
    for mode in [KVCacheMode::Turbo4Delegated, KVCacheMode::Turbo3Asym] {
        assert!(!kv_mode_is_snapshotable(mode));
        let source = filled(mode, 8);
        let err = source
            .snapshot_parts()
            .validate()
            .expect_err("this mode is not carried");
        assert!(
            err.contains("is not supported by model-state snapshots"),
            "{mode:?}: {err}"
        );
        let mut target = KVCache::new_with_mode(mode);
        assert!(
            target
                .restore_snapshot_parts(owned(source.snapshot_parts()))
                .is_err()
        );
        assert!(target.is_empty());
    }
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ] {
        assert!(kv_mode_is_snapshotable(mode));
    }
}

#[test]
fn a_populated_target_is_refused() {
    for mode in QUANTIZED {
        let source = filled(mode, 8);
        let mut target = filled(mode, 3);
        let err = target
            .restore_snapshot_parts(owned(source.snapshot_parts()))
            .expect_err("the target already holds rows");
        assert!(err.contains("not empty"), "{mode:?}: {err}");
        assert_eq!(target.offset, 3, "{mode:?}: the target keeps its own rows");
    }
}

// ---------------------------------------------------------------------------
// RotatingKVCache
// ---------------------------------------------------------------------------

fn ring_buffers(parts: RotatingKVCacheSnapshotParts<'_>) -> RotatingKVCacheSnapshotBuffers {
    let copy = |array: Option<&MlxArray>| array.map(ffi::copy);
    RotatingKVCacheSnapshotBuffers {
        keys: copy(parts.keys),
        values: copy(parts.values),
        v_packed: copy(parts.v_packed),
        v_norms: copy(parts.v_norms),
        v_rescale: copy(parts.v_rescale),
    }
}

fn stepped_ring(mode: KVCacheMode, tokens: i32) -> RotatingKVCache {
    let mut ring = RotatingKVCache::new_with_mode(WINDOW, mode);
    for t in 0..tokens {
        let _ = ring.update_and_fetch(rows(t, 1, 0.0), rows(t, 1, 1.0));
    }
    ring
}

fn assert_same_ring(restored: &RotatingKVCache, source: &RotatingKVCache, what: &str) {
    let (r, s) = (restored.snapshot_parts(), source.snapshot_parts());
    assert_eq!(r.state, s.state, "{what}: ring scalars");
    assert_same(r.keys, s.keys, &format!("{what}: keys"));
    assert_same(r.values, s.values, &format!("{what}: values"));
    for name in ROTATING_SIDECAR_FIELDS {
        assert_same(r.sidecar(name), s.sidecar(name), &format!("{what}: {name}"));
    }
}

#[test]
fn a_restored_ring_holds_the_source_buffers_byte_for_byte() {
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
        KVCacheMode::Turbo3Asym,
        KVCacheMode::Turbo4Asym,
    ] {
        // Before the window, exactly full, one past it, and well wrapped.
        for tokens in [1, 40, WINDOW, WINDOW + 1, 3 * WINDOW + 5] {
            let mut source = stepped_ring(mode, tokens);
            let parts = source.snapshot_parts();
            parts.check_layout().expect("a ring the mode filled");
            let mut restored = RotatingKVCache::new_with_mode(WINDOW, mode);
            restored
                .restore_snapshot_state(parts.state, ring_buffers(parts))
                .unwrap_or_else(|err| panic!("{mode:?} ring at {tokens}: {err}"));
            let what = format!("{mode:?} ring at {tokens}");
            assert_same_ring(&restored, &source, &what);
            assert_eq!(restored.mode, mode, "{what}: the label is kept");

            for t in tokens..tokens + WINDOW + 3 {
                let (sk, sv) = source.update_and_fetch(rows(t, 1, 0.0), rows(t, 1, 1.0));
                let (rk, rv) = restored.update_and_fetch(rows(t, 1, 0.0), rows(t, 1, 1.0));
                assert_same(Some(&rk), Some(&sk), &format!("{what}: keys after {t}"));
                assert_same(Some(&rv), Some(&sv), &format!("{what}: values after {t}"));
            }
            assert_same_ring(&restored, &source, &format!("{what}, after the next wrap"));
        }
    }
}

#[test]
fn only_an_aligned_turbo4_asym_ring_holds_packed_values() {
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
        KVCacheMode::Turbo3Asym,
    ] {
        let ring = stepped_ring(mode, 8);
        let parts = ring.snapshot_parts();
        assert!(parts.keys.is_some() && parts.values.is_some(), "{mode:?}");
        assert!(
            ROTATING_SIDECAR_FIELDS
                .iter()
                .all(|name| parts.sidecar(name).is_none()),
            "{mode:?}: plain rows only"
        );
        assert_eq!(
            RotatingKVCache::stored_mode(WINDOW, mode),
            KVCacheMode::Fp16
        );
    }
    let ring = stepped_ring(KVCacheMode::Turbo4Asym, 8);
    let parts = ring.snapshot_parts();
    assert!(parts.keys.is_some() && parts.values.is_none());
    assert!(
        ROTATING_SIDECAR_FIELDS
            .iter()
            .all(|name| parts.sidecar(name).is_some())
    );
}

#[test]
fn a_malformed_ring_set_is_refused_before_anything_is_assigned() {
    let source = stepped_ring(KVCacheMode::Turbo4Asym, 40);
    let good = source.snapshot_parts();
    let refused = |what: &str,
                   state: RotatingKVCacheSnapshotState,
                   corrupt: &dyn Fn(&mut RotatingKVCacheSnapshotBuffers)| {
        let mut buffers = ring_buffers(good);
        corrupt(&mut buffers);
        let mut target = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Turbo4Asym);
        assert!(
            target.restore_snapshot_state(state, buffers).is_err(),
            "{what} must be refused"
        );
        assert!(
            target.is_empty() && target.offset == 0 && target.v_packed.is_none(),
            "{what} must leave the ring untouched"
        );
    };
    refused("a ring without keys", good.state, &|b| b.keys = None);
    refused("a ring without packed V", good.state, &|b| {
        b.v_packed = None
    });
    refused("a ring without V norms", good.state, &|b| b.v_norms = None);
    refused("a ring without V rescale", good.state, &|b| {
        b.v_rescale = None
    });
    // Each replacement keeps the ring's own shape and dtype but for the one
    // property the case is about (see `like`).
    refused("a packed ring with FP16 values", good.state, &|b| {
        b.values = like(&b.keys, None, |_| {});
    });
    refused("a sidecar of another length", good.state, &|b| {
        b.v_norms = like(&b.v_norms, None, |shape| shape[2] += 1);
    });
    refused("a sidecar of another head count", good.state, &|b| {
        b.v_rescale = like(&b.v_rescale, None, |shape| shape[1] += 1);
    });
    refused("ring keys that are not FP16", good.state, &|b| {
        b.keys = like(&b.keys, Some(dtype::FLOAT32), |_| {});
    });
    for width in [0, 3, 24] {
        refused(&format!("a packed width of {width}"), good.state, &|b| {
            b.v_packed = like(&b.v_packed, None, |shape| shape[3] = width);
        });
    }
    let mut past = good.state;
    past.idx = 4096;
    refused("a write position past the buffer", past, &|_| {});
    // `enable_speculative_buffer` is FP16-only, so no other label has this.
    let mut buffered = good.state;
    buffered.buffer_size = 8;
    buffered.speculative_ring_origin = Some((40, 40 % WINDOW));
    buffered.start_position = 0;
    refused("speculative buffer state under a label", buffered, &|_| {});
    // A Turbo4Asym label on a window that is not a multiple of the block
    // cannot be built, so a snapshot that claims it is not a real one.
    let mut misaligned = good.state;
    misaligned.max_size = WINDOW + 1;
    refused("a misaligned Turbo4Asym window", misaligned, &|_| {});

    // A ring of plain rows must not carry packed V.
    let plain = stepped_ring(KVCacheMode::Int8, 40);
    let parts = plain.snapshot_parts();
    let mut buffers = ring_buffers(parts);
    buffers.v_packed = like(&buffers.keys, Some(dtype::UINT8), |shape| shape[3] = 32);
    let mut target = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Int8);
    assert!(target.restore_snapshot_state(parts.state, buffers).is_err());
    assert!(target.is_empty());
}
