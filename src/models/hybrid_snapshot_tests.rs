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

//! The attention arm of the attention + recurrent hybrids in the quantized KV
//! modes (issue #3250).
//!
//! Qwen3-Next, Qwen4-Exp, Qwen 3.5, LFM2 and Bailing MoE Linear all store an
//! attention layer through `snapshot_attention` and `restore_attention`. They
//! used to refuse every non-FP16 mode ("do not yet serialize"). Each test
//! compares a restored cache with an uninterrupted cache of the same mode,
//! byte for byte, and uses only the cache's public surface.

use super::*;
use crate::models::kv_snapshot::test_support::{assert_same_bytes, keys, values};
use crate::models::qwen3_next::Qwen3NextCache;

const CARRIED: [KVCacheMode; 3] = [
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
];

fn filled(mode: KVCacheMode, tokens: i32) -> KVCache {
    let mut kv = KVCache::new_with_mode(mode);
    let _ = kv.update_and_fetch(keys(0, tokens), values(0, tokens));
    kv
}

fn assert_steps_agree(a: &mut KVCache, b: &mut KVCache, start: i32, n: i32, what: &str) {
    for t in start..start + n {
        let (ak, av) = a.update_and_fetch(keys(t, 1), values(t, 1));
        let (bk, bv) = b.update_and_fetch(keys(t, 1), values(t, 1));
        assert_same_bytes(&ak, &bk, &format!("{what}: keys after token {t}"));
        assert_same_bytes(&av, &bv, &format!("{what}: values after token {t}"));
    }
    assert_eq!(a.offset, b.offset, "{what}: offset");
}

fn tensor_names(snapshot: &ModelStateSnapshot) -> Vec<String> {
    snapshot.tensor_names().map(str::to_string).collect()
}

#[test]
fn a_quantized_attention_layer_round_trips() {
    for mode in CARRIED {
        // One token, a turbo block edge, an ordinary length, a growth step.
        for tokens in [1, 32, 37, 256, 257] {
            let mut source = filled(mode, tokens);
            let mut snapshot = ModelStateSnapshot::new("test", tokens as usize);
            snapshot_attention(&source, &mut snapshot, "layer0")
                .unwrap_or_else(|err| panic!("{mode:?} snapshot at {tokens}: {err}"));

            let mut restored = KVCache::new_with_mode(mode);
            restore_attention(&mut restored, &snapshot, "layer0")
                .unwrap_or_else(|err| panic!("{mode:?} restore at {tokens}: {err}"));
            assert_eq!(restored.offset, tokens);
            assert_eq!(restored.nbytes(), source.nbytes(), "{mode:?} at {tokens}");
            assert_steps_agree(
                &mut source,
                &mut restored,
                tokens,
                34,
                &format!("{mode:?} attention restored at {tokens}"),
            );
        }
    }
}

/// An FP16 attention layer stores the three names it always has, in the same
/// order, and a quantized one adds its sidecars and two scalars under the
/// same `{prefix}.attention.` prefix.
#[test]
fn an_attention_layer_stores_these_tensors() {
    let names = |mode| {
        let mut snapshot = ModelStateSnapshot::new("test", 8);
        snapshot_attention(&filled(mode, 8), &mut snapshot, "layer0").expect("snapshot");
        tensor_names(&snapshot)
    };
    assert_eq!(
        names(KVCacheMode::Fp16),
        [
            "layer0.attention.mode",
            "layer0.attention.keys",
            "layer0.attention.values",
        ]
    );
    assert_eq!(
        names(KVCacheMode::Int8),
        [
            "layer0.attention.key_scales",
            "layer0.attention.val_scales",
            "layer0.attention.turbo_seed",
            "layer0.attention.live_start",
            "layer0.attention.mode",
            "layer0.attention.keys",
            "layer0.attention.values",
        ]
    );
    assert_eq!(
        names(KVCacheMode::Turbo4Asym),
        [
            "layer0.attention.v_packed",
            "layer0.attention.v_norms",
            "layer0.attention.v_rescale",
            "layer0.attention.turbo_seed",
            "layer0.attention.live_start",
            "layer0.attention.mode",
            "layer0.attention.keys",
        ]
    );
    assert_eq!(
        names(KVCacheMode::Turbo4),
        [
            "layer0.attention.v_packed",
            "layer0.attention.v_norms",
            "layer0.attention.v_rescale",
            "layer0.attention.k_packed",
            "layer0.attention.k_norms",
            "layer0.attention.turbo_seed",
            "layer0.attention.live_start",
            "layer0.attention.mode",
        ]
    );
}

/// A turn-boundary checkpoint restore takes the attention rows from a longer
/// snapshot and cuts them back (issue #3047). In a quantized mode the result
/// is the cache a cold prefill of the shorter prefix builds.
#[test]
fn a_quantized_attention_layer_is_cut_back_from_a_longer_donor() {
    for mode in CARRIED {
        for (have, keep) in [(37, 36), (37, 32), (37, 1), (300, 256), (300, 257)] {
            let donor_cache = filled(mode, have);
            let mut donor = ModelStateSnapshot::new("test", have as usize);
            snapshot_attention(&donor_cache, &mut donor, "layer0").expect("donor snapshot");

            let mut restored = KVCache::new_with_mode(mode);
            restore_attention_from_donor(
                &mut restored,
                &donor,
                "layer0",
                keep as usize,
                have as usize,
            )
            .unwrap_or_else(|err| panic!("{mode:?} {have} cut to {keep}: {err}"));
            assert_eq!(restored.offset, keep);

            let mut cold = filled(mode, keep);
            assert_steps_agree(
                &mut cold,
                &mut restored,
                keep,
                34,
                &format!("{mode:?} attention, {have} rows cut to {keep}"),
            );
        }
    }
}

#[test]
fn delegated_and_three_bit_attention_layers_are_still_refused() {
    for mode in [KVCacheMode::Turbo4Delegated, KVCacheMode::Turbo3Asym] {
        let source = filled(mode, 8);
        let mut snapshot = ModelStateSnapshot::new("test", 8);
        let err = snapshot_attention(&source, &mut snapshot, "layer0")
            .expect_err("this mode is not carried");
        assert!(err.contains("do not yet serialize"), "{mode:?}: {err}");
        assert!(
            snapshot.is_empty(),
            "{mode:?}: a refused layer stores nothing"
        );
    }
}

#[test]
fn an_attention_snapshot_lands_only_in_a_cache_of_its_mode() {
    let all = [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ];
    for stored in all {
        let mut snapshot = ModelStateSnapshot::new("test", 8);
        snapshot_attention(&filled(stored, 8), &mut snapshot, "layer0").expect("snapshot");
        for live in all.into_iter().filter(|live| *live != stored) {
            let mut target = KVCache::new_with_mode(live);
            let err = restore_attention(&mut target, &snapshot, "layer0")
                .expect_err("a snapshot of another mode is refused");
            assert!(
                err.contains("does not match configured mode"),
                "{stored:?} into {live:?}: {err}"
            );
            assert!(target.is_empty() && target.offset == 0);
        }
    }
}

/// A quantized attention snapshot that lost a buffer or a scalar is refused
/// before anything is assigned.
#[test]
fn a_quantized_attention_snapshot_missing_a_tensor_is_refused_whole() {
    for mode in CARRIED {
        let mut snapshot = ModelStateSnapshot::new("test", 8);
        snapshot_attention(&filled(mode, 8), &mut snapshot, "layer0").expect("snapshot");
        for missing in tensor_names(&snapshot) {
            if missing.ends_with(".mode") {
                // Without its tag the layer reads as FP16, which the cache's
                // own mode then refuses.
                continue;
            }
            let partial = snapshot.subset(|name| name != missing);
            let mut target = KVCache::new_with_mode(mode);
            assert!(
                restore_attention(&mut target, &partial, "layer0").is_err(),
                "{mode:?}: a snapshot without {missing} must be refused"
            );
            assert!(
                target.is_empty() && target.offset == 0 && target.nbytes() == 0,
                "{mode:?} without {missing}: the cache must be left untouched"
            );
        }
        let untagged = snapshot.subset(|name| !name.ends_with(".mode"));
        let mut target = KVCache::new_with_mode(mode);
        let err = restore_attention(&mut target, &untagged, "layer0")
            .expect_err("an untagged snapshot is an FP16 one");
        assert!(
            err.contains("does not match configured mode"),
            "{mode:?}: {err}"
        );
        assert!(target.is_empty());
    }
}

/// A layer that has seen no token stores its mode tag only and restores to a
/// fresh cache, quantized or not.
#[test]
fn an_attention_layer_with_no_tokens_stores_its_tag_only() {
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ] {
        let mut snapshot = ModelStateSnapshot::new("test", 0);
        snapshot_attention(&KVCache::new_with_mode(mode), &mut snapshot, "layer0")
            .expect("an empty layer is not an error");
        assert_eq!(
            tensor_names(&snapshot),
            ["layer0.attention.mode"],
            "{mode:?}"
        );
        let mut restored = KVCache::new_with_mode(mode);
        restore_attention(&mut restored, &snapshot, "layer0").expect("restore");
        assert!(restored.is_empty() && restored.offset == 0, "{mode:?}");
    }
}

/// Qwen3-Next reaches the shared arm through its cache enum.
#[test]
fn a_qwen3_next_attention_cache_round_trips_in_int8() {
    let mut source = Qwen3NextCache::Attention(filled(KVCacheMode::Int8, 37));
    let mut snapshot = ModelStateSnapshot::new("qwen3_next", 37);
    source
        .snapshot_into(&mut snapshot, "layer3")
        .expect("an Int8 attention layer is stored");
    assert!(snapshot.tensor("layer3.attention.key_scales").is_some());

    let mut restored = Qwen3NextCache::Attention(KVCache::new_with_mode(KVCacheMode::Int8));
    restored
        .restore_from(&snapshot, "layer3")
        .expect("and restored");
    let (Qwen3NextCache::Attention(source), Qwen3NextCache::Attention(restored)) =
        (&mut source, &mut restored)
    else {
        unreachable!("both are attention caches");
    };
    assert_steps_agree(source, restored, 37, 4, "Qwen3-Next Int8 attention");
}
