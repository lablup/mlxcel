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

//! The generic snapshot path over synthetic layer state: byte compatibility
//! with the model-owned encoders, round trips, truncation and checkpoints.
//! The real-model round trips live in `models::gemma3_tests` and
//! `models::qwen3_5_checkpoint_tests`.

use super::*;
use crate::models::gated_delta::GatedDeltaCache;
use crate::models::gemma3::Cache;
use crate::models::qwen3_next::Qwen3NextCache;
use mlxcel_core::cache::{KVCacheMode, RecurrentState};
use mlxcel_core::layers::{KVCache, RotatingKVCache};
use mlxcel_core::{MlxArray, UniquePtr};

const WINDOW: i32 = 4;
const GEMMA3: LayerStateSnapshotSpec = LayerStateSnapshotSpec {
    family: "gemma3",
    label: "Gemma 3",
    encoding: LayerStateEncoding::Windowed {
        standard: "standard",
        rotating: "rotating",
    },
};
const QWEN35: LayerStateSnapshotSpec = LayerStateSnapshotSpec {
    family: "qwen3_5",
    label: "Qwen 3.5",
    encoding: LayerStateEncoding::Hybrid,
};

/// `[1, 2, n, 3]` rows whose values depend on the head, the row and the
/// column but not on `n`, so a prefix of a longer fill equals a shorter fill.
fn rows(n: i32, base: f32) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity((n * 2 * 3) as usize);
    for head in 0..2 {
        for row in 0..n {
            for col in 0..3 {
                data.push(base + (head * 100 + row * 3 + col) as f32);
            }
        }
    }
    mlxcel_core::from_slice_f32(&data, &[1, 2, n, 3])
}

fn windowed_kinds() -> Vec<LayerStateKind> {
    vec![
        LayerStateKind::sliding(WINDOW),
        LayerStateKind::Full,
        LayerStateKind::None,
    ]
}

fn windowed_policy() -> StatePolicy {
    StatePolicy::new(GEMMA3, windowed_kinds())
}

fn fresh_windowed() -> Vec<LayerState> {
    LayerState::for_kinds(&windowed_kinds(), None, 0)
}

/// A `prefill`-token prompt as one chunk, then `decode` single-token steps,
/// written to a sliding and a full cache. Returns the token count.
fn fill(sliding: &mut RotatingKVCache, full: &mut KVCache, prefill: i32, decode: i32) -> usize {
    let mut base = 0.0;
    let mut step = |n: i32, sliding: &mut RotatingKVCache, full: &mut KVCache| {
        let _ = sliding.update_and_fetch(rows(n, base), rows(n, base + 0.5));
        let _ = full.update_and_fetch(rows(n, base + 100.0), rows(n, base + 100.5));
        base += 1000.0;
    };
    step(prefill, sliding, full);
    for _ in 0..decode {
        step(1, sliding, full);
    }
    (prefill + decode) as usize
}

fn filled_windowed(prefill: i32, decode: i32) -> (Vec<LayerState>, usize) {
    let mut state = fresh_windowed();
    let [
        LayerState::Rotating(sliding),
        LayerState::Kv(full),
        LayerState::Empty,
    ] = state.as_mut_slice()
    else {
        panic!("windowed fixture layout");
    };
    let len = fill(sliding, full, prefill, decode);
    (state, len)
}

/// The same content written through the model-owned `gemma3::Cache` enum,
/// which AFMoE and Kolibri-1 still snapshot through.
fn model_owned_snapshot(prefill: i32, decode: i32) -> ModelStateSnapshot {
    let mut sliding = RotatingKVCache::new_with_mode(WINDOW, KVCacheMode::Fp16);
    let mut full = KVCache::new_with_mode(KVCacheMode::Fp16);
    let len = fill(&mut sliding, &mut full, prefill, decode);
    let caches = [
        Cache::Rotating(sliding),
        Cache::Standard(full),
        Cache::Standard(KVCache::new()),
    ];
    let names = KvSnapshotNames::new("Gemma 3", "standard", "rotating");
    let mut snapshot = ModelStateSnapshot::new("gemma3", len);
    for (idx, cache) in caches.iter().enumerate() {
        cache
            .snapshot_into(&mut snapshot, &format!("layer{idx}"), names)
            .expect("model-owned snapshot");
    }
    snapshot
}

/// Every tensor of `snapshot` as `(name, shape, dtype, bytes)`.
fn contents(snapshot: &ModelStateSnapshot) -> Vec<(String, Vec<i32>, i32, Vec<u8>)> {
    snapshot
        .tensor_names()
        .map(|name| {
            let tensor = snapshot.tensor(name).expect("named tensor");
            mlxcel_core::eval(tensor);
            (
                name.to_string(),
                mlxcel_core::array_shape(tensor),
                mlxcel_core::array_dtype(tensor),
                mlxcel_core::array_to_raw_bytes(tensor),
            )
        })
        .collect()
}

fn assert_same_snapshot(got: &ModelStateSnapshot, want: &ModelStateSnapshot, what: &str) {
    assert_eq!(got.family(), want.family(), "{what}: family");
    assert_eq!(got.token_len(), want.token_len(), "{what}: token length");
    let (got, want) = (contents(got), contents(want));
    let names = |c: &[(String, Vec<i32>, i32, Vec<u8>)]| -> Vec<String> {
        c.iter().map(|t| t.0.clone()).collect()
    };
    assert_eq!(names(&got), names(&want), "{what}: tensor names");
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.1, w.1, "{what}: shape of {}", g.0);
        assert_eq!(g.2, w.2, "{what}: dtype of {}", g.0);
        assert!(g.3 == w.3, "{what}: bytes of {}", g.0);
    }
}

#[test]
fn a_windowed_snapshot_is_what_the_model_owned_encoder_writes() {
    // Unwrapped (3 of 4 window rows), exactly full, and wrapped by decode
    // writes: the generic path must produce the tensors the per-family code
    // on main produced, name for name, shape for shape and byte for byte.
    for (prefill, decode) in [(3, 0), (4, 0), (3, 6), (9, 2)] {
        let (state, len) = filled_windowed(prefill, decode);
        let got = windowed_policy()
            .snapshot(&state, len)
            .expect("snapshot")
            .expect("non-empty");
        let want = model_owned_snapshot(prefill, decode);
        assert_same_snapshot(&got, &want, &format!("prefill {prefill}, decode {decode}"));
    }
}

#[test]
fn the_windowed_tensor_names_are_pinned() {
    // A stored snapshot is read back by name. This list is the contract: a
    // renamed or dropped tensor fails here before it orphans stored entries.
    let (state, len) = filled_windowed(3, 6);
    let snapshot = windowed_policy()
        .snapshot(&state, len)
        .expect("snapshot")
        .expect("non-empty");
    let names: Vec<&str> = snapshot.tensor_names().collect();
    assert_eq!(names, PINNED_WINDOWED_NAMES);
}

const PINNED_WINDOWED_NAMES: &[&str] = &[
    "layer0.rotating.keys",
    "layer0.rotating.values",
    "layer0.rotating.max_size",
    "layer0.rotating.buffer_size",
    "layer0.rotating.offset",
    "layer0.rotating.start_position",
    "layer0.rotating.idx",
    "layer0.rotating.step",
    "layer0.rotating.mode",
    "layer0.rotating.turbo_seed",
    "layer1.standard.keys",
    "layer1.standard.values",
    "layer1.standard.offset",
    "layer1.standard.mode",
];

/// The first `rows` live rows of a layer's keys and values. A cut cache keeps
/// its physical buffer, so only the live rows are comparable.
fn live_rows(layer: &LayerState, rows: i32) -> (Vec<u8>, Vec<u8>) {
    let (keys, values) = match layer {
        LayerState::Kv(cache) => (cache.keys.as_deref(), cache.values.as_deref()),
        LayerState::Rotating(cache) => (cache.keys.as_deref(), cache.values.as_deref()),
        _ => panic!("attention layer expected"),
    };
    let cut = |tensor: Option<&MlxArray>| {
        let tensor = tensor.expect("stored tensor");
        let shape = mlxcel_core::array_shape(tensor);
        let head = mlxcel_core::slice(tensor, &[0, 0, 0, 0], &[shape[0], shape[1], rows, shape[3]]);
        let head = mlxcel_core::contiguous(&head, false);
        mlxcel_core::eval(&head);
        mlxcel_core::array_to_raw_bytes(&head)
    };
    (cut(keys), cut(values))
}

#[test]
fn a_windowed_round_trip_restores_every_layer() {
    for (prefill, decode) in [(3, 0), (3, 6), (9, 2)] {
        let policy = windowed_policy();
        let (state, len) = filled_windowed(prefill, decode);
        let snapshot = policy.snapshot(&state, len).unwrap().unwrap();
        let restored = policy
            .restore(fresh_windowed(), &snapshot)
            .expect("restore");
        for (layer, (got, want)) in restored.iter().zip(&state).enumerate() {
            assert_eq!(got.offset(), want.offset(), "layer {layer} offset");
            assert_eq!(
                got.visible_len(),
                want.visible_len(),
                "layer {layer} length"
            );
        }
        let again = policy.snapshot(&restored, len).unwrap().unwrap();
        assert_same_snapshot(&again, &snapshot, "snapshot of the restored state");
    }
}

#[test]
fn truncation_follows_the_sliding_ring() {
    let policy = windowed_policy();
    // Unwrapped: any shorter prefix is reachable, and the cut state is the
    // state of a prefill that stopped there.
    let (state, len) = filled_windowed(3, 0);
    let snapshot = policy.snapshot(&state, len).unwrap().unwrap();
    assert!(policy.truncatable_to(&snapshot, 2));
    assert!(policy.truncatable_to(&snapshot, 3));
    assert!(!policy.truncatable_to(&snapshot, 4), "past the snapshot");
    let cut = policy
        .restore_truncated(fresh_windowed(), &snapshot, 2)
        .expect("truncated restore");
    assert_eq!(cut[0].offset(), 2);
    assert_eq!(cut[1].offset(), 2);
    let (short, _) = filled_windowed(2, 0);
    for layer in 0..2 {
        assert_eq!(cut[layer].visible_len(), 2, "layer {layer} live rows");
        assert!(
            live_rows(&cut[layer], 2) == live_rows(&short[layer], 2),
            "layer {layer}: a cut state holds the rows of a shorter prefill"
        );
    }

    // Wrapped: logical token t no longer sits at slot t.
    let (state, len) = filled_windowed(3, 6);
    let snapshot = policy.snapshot(&state, len).unwrap().unwrap();
    assert!(!policy.truncatable_to(&snapshot, 5));
    let err = policy
        .restore_truncated(fresh_windowed(), &snapshot, 5)
        .err()
        .expect("a wrapped ring declines truncation");
    assert!(err.contains("cannot be truncated"), "{err}");
}

#[test]
fn a_mismatched_restore_is_a_named_error() {
    let policy = windowed_policy();
    let (state, len) = filled_windowed(3, 0);
    let snapshot = policy.snapshot(&state, len).unwrap().unwrap();

    let foreign = ModelStateSnapshot::new("gemma4", 4);
    let err = policy
        .restore(fresh_windowed(), &foreign)
        .err()
        .expect("family");
    assert_eq!(err, "cannot restore gemma4 snapshot into Gemma 3");
    assert!(!policy.truncatable_to(&foreign, 2));
    assert!(policy.state_checkpoint(&foreign).is_none());

    let mut short = fresh_windowed();
    short.pop();
    let err = policy.restore(short, &snapshot).err().expect("layer count");
    assert!(err.contains("2 layers of state, the model has 3"), "{err}");

    let mut swapped = fresh_windowed();
    swapped.swap(0, 1);
    let err = policy
        .restore(swapped, &snapshot)
        .err()
        .expect("layer kinds");
    assert!(
        err.contains("layer0: holds full storage, the model declares Sliding"),
        "{err}"
    );

    // An Fp16 snapshot does not land in a quantized cache.
    let quantized = LayerState::for_kinds(&windowed_kinds(), Some(&[KVCacheMode::Int8; 3]), 0);
    let err = policy
        .restore(quantized, &snapshot)
        .err()
        .expect("cache mode");
    assert!(
        err.contains("does not match configured cache mode"),
        "{err}"
    );

    // A windowed family has no encoding for a recurrent layer.
    let policy = StatePolicy::new(GEMMA3, vec![LayerStateKind::Recurrent]);
    let state = LayerState::for_kinds(&[LayerStateKind::Recurrent], None, 0);
    let err = policy.snapshot(&state, 1).err().expect("no encoding");
    assert!(err.contains("recurrent storage has no"), "{err}");
}

#[test]
fn a_ring_checkpoint_exists_once_a_ring_has_wrapped() {
    let policy = windowed_policy();
    let (unwrapped, len) = filled_windowed(3, 0);
    let snapshot = policy.snapshot(&unwrapped, len).unwrap().unwrap();
    assert!(
        policy.state_checkpoint(&snapshot).is_none(),
        "an unwrapped snapshot truncates on its own"
    );

    // The checkpoint at 9 tokens and a donor at 11 tokens of the same run.
    let (at_boundary, boundary_len) = filled_windowed(3, 6);
    let boundary = policy
        .snapshot(&at_boundary, boundary_len)
        .unwrap()
        .unwrap();
    let checkpoint = policy.state_checkpoint(&boundary).expect("ring checkpoint");
    assert!(
        checkpoint
            .tensor_names()
            .all(|name| name.contains(".rotating.")),
        "a ring checkpoint holds only ring tensors"
    );
    let (longer, longer_len) = filled_windowed(3, 8);
    let donor = policy.snapshot(&longer, longer_len).unwrap().unwrap();

    let rebuilt = policy
        .restore_from_checkpoint(fresh_windowed(), &checkpoint, &donor)
        .expect("checkpoint restore");
    // The ring comes from the checkpoint, tensor for tensor. The full layer
    // is the donor's, cut back: the boundary's live rows.
    let rebuilt_snapshot = policy.snapshot(&rebuilt, boundary_len).unwrap().unwrap();
    assert_same_snapshot(
        &rebuilt_snapshot.subset(|name| name.starts_with("layer0.")),
        &boundary.subset(|name| name.starts_with("layer0.")),
        "the ring of checkpoint + donor versus the boundary state",
    );
    assert_eq!(rebuilt[1].offset(), boundary_len as i32);
    assert!(
        live_rows(&rebuilt[1], boundary_len as i32)
            == live_rows(&at_boundary[1], boundary_len as i32),
        "the full layer holds the boundary's rows"
    );

    let err = policy
        .restore_from_checkpoint(fresh_windowed(), &donor, &boundary)
        .err()
        .expect("a donor shorter than the checkpoint");
    assert!(err.contains("longer than its"), "{err}");
}

fn hybrid_kinds() -> Vec<LayerStateKind> {
    vec![LayerStateKind::Recurrent, LayerStateKind::Full]
}

fn fresh_hybrid() -> Vec<LayerState> {
    vec![
        LayerState::Recurrent(RecurrentState::with_slots(&["conv_state", "state_cache"])),
        LayerState::Kv(KVCache::new()),
    ]
}

fn filled_hybrid(tokens: i32) -> Vec<LayerState> {
    let mut state = fresh_hybrid();
    let [LayerState::Recurrent(recurrent), LayerState::Kv(attention)] = state.as_mut_slice() else {
        panic!("hybrid fixture layout");
    };
    recurrent.put("conv_state", Some(rows(1, tokens as f32)));
    recurrent.put("state_cache", Some(rows(2, 7.0 + tokens as f32)));
    recurrent.offset = tokens;
    let _ = attention.update_and_fetch(rows(tokens, 1.0), rows(tokens, 2.0));
    state
}

#[test]
fn a_hybrid_snapshot_is_what_the_model_owned_encoder_writes() {
    let tokens = 5;
    let state = filled_hybrid(tokens);
    let policy = StatePolicy::new(QWEN35, hybrid_kinds());
    let got = policy
        .snapshot(&state, tokens as usize)
        .expect("snapshot")
        .expect("a hybrid always stores its snapshot");

    let mut linear = GatedDeltaCache::new();
    linear.conv_state = Some(rows(1, tokens as f32));
    linear.state_cache = Some(rows(2, 7.0 + tokens as f32));
    linear.offset = tokens;
    let mut attention = KVCache::new();
    let _ = attention.update_and_fetch(rows(tokens, 1.0), rows(tokens, 2.0));
    let caches = [
        Qwen3NextCache::Linear(linear),
        Qwen3NextCache::Attention(attention),
    ];
    let mut want = ModelStateSnapshot::new("qwen3_5", tokens as usize);
    for (idx, cache) in caches.iter().enumerate() {
        cache
            .snapshot_into(&mut want, &format!("layer{idx}"))
            .expect("model-owned snapshot");
    }
    assert_same_snapshot(&got, &want, "hybrid snapshot");
    assert_eq!(
        got.tensor_names().collect::<Vec<_>>(),
        vec![
            "layer0.linear.conv_state",
            "layer0.linear.state_cache",
            "layer1.attention.mode",
            "layer1.attention.keys",
            "layer1.attention.values",
        ]
    );
}

#[test]
fn a_hybrid_restores_whole_and_from_a_checkpoint_but_never_truncates() {
    let policy = StatePolicy::new(QWEN35, hybrid_kinds());
    let boundary_state = filled_hybrid(3);
    let boundary = policy.snapshot(&boundary_state, 3).unwrap().unwrap();
    let donor_state = filled_hybrid(5);
    let donor = policy.snapshot(&donor_state, 5).unwrap().unwrap();

    let restored = policy.restore(fresh_hybrid(), &donor).expect("restore");
    assert_eq!(restored[0].offset(), 5);
    assert_eq!(restored[1].offset(), 5);
    assert_same_snapshot(
        &policy.snapshot(&restored, 5).unwrap().unwrap(),
        &donor,
        "hybrid round trip",
    );

    assert!(
        !policy.truncatable_to(&donor, 3),
        "recurrent state cannot be rewound"
    );
    assert!(policy.restore_truncated(fresh_hybrid(), &donor, 3).is_err());

    let checkpoint = policy.state_checkpoint(&boundary).expect("checkpoint");
    assert_eq!(
        checkpoint.tensor_names().collect::<Vec<_>>(),
        vec!["layer0.linear.conv_state", "layer0.linear.state_cache"]
    );
    let rebuilt = policy
        .restore_from_checkpoint(fresh_hybrid(), &checkpoint, &donor)
        .expect("checkpoint restore");
    assert_eq!(rebuilt[0].offset(), 3);
    assert_eq!(
        rebuilt[1].offset(),
        3,
        "the donor's rows cut back to the checkpoint"
    );
    let recurrent = rebuilt[0].as_recurrent().expect("recurrent layer");
    let conv = recurrent.get("conv_state").expect("conv state");
    let want = boundary.tensor("layer0.linear.conv_state").expect("stored");
    mlxcel_core::eval(conv);
    mlxcel_core::eval(want);
    assert!(mlxcel_core::array_to_raw_bytes(conv) == mlxcel_core::array_to_raw_bytes(want));

    // A checkpoint that lost a recurrent slot is refused, as is a foreign pair.
    let partial = checkpoint.subset(|name| name.ends_with("conv_state"));
    let err = policy
        .restore_from_checkpoint(fresh_hybrid(), &partial, &donor)
        .err()
        .expect("missing slot");
    assert!(err.contains("checkpoint has no state_cache"), "{err}");
    let foreign = ModelStateSnapshot::new("qwen3_next", 3);
    let err = policy
        .restore_from_checkpoint(fresh_hybrid(), &foreign, &donor)
        .err()
        .expect("foreign checkpoint");
    assert!(err.contains("into Qwen 3.5"), "{err}");
}

/// `donor` with its attention keys and values replaced by `keys` and `values`.
fn with_attention(
    donor: &ModelStateSnapshot,
    keys: &MlxArray,
    values: &MlxArray,
) -> ModelStateSnapshot {
    let mut forged = donor
        .subset(|name| !name.ends_with(".attention.keys") && !name.ends_with(".attention.values"));
    forged.push_tensor("layer1.attention.keys", keys);
    forged.push_tensor("layer1.attention.values", values);
    forged
}

/// Every restore validates the attention tensors it installs, the checkpoint
/// restore included: a donor whose keys and values are not one pair, or hold
/// fewer rows than the restored offset, is a named error, never a state.
#[test]
fn every_restore_refuses_attention_tensors_that_do_not_fit() {
    let policy = StatePolicy::new(QWEN35, hybrid_kinds());
    let boundary = policy.snapshot(&filled_hybrid(3), 3).unwrap().unwrap();
    let donor = policy.snapshot(&filled_hybrid(5), 5).unwrap().unwrap();
    let checkpoint = policy.state_checkpoint(&boundary).expect("checkpoint");

    // Values of another head count than the keys.
    let values = mlxcel_core::from_slice_f32(&[0.0; 5 * 3], &[1, 1, 5, 3]);
    let mismatched = with_attention(&donor, &rows(5, 1.0), &values);
    let err = policy
        .restore(fresh_hybrid(), &mismatched)
        .err()
        .expect("whole restore of a mismatched pair");
    assert!(err.contains("not one [B, H, T, D] pair"), "{err}");
    let err = policy
        .restore_from_checkpoint(fresh_hybrid(), &checkpoint, &mismatched)
        .err()
        .expect("checkpoint restore over a mismatched pair");
    assert!(
        err.contains("Qwen 3.5 layer1") && err.contains("not one [B, H, T, D] pair"),
        "{err}"
    );

    // Two rows behind a five-token donor and a three-token checkpoint.
    let short = with_attention(&donor, &rows(2, 1.0), &rows(2, 2.0));
    let err = policy
        .restore(fresh_hybrid(), &short)
        .err()
        .expect("whole restore of too few rows");
    assert!(
        err.contains("2 rows of keys cannot cover an offset of 5"),
        "{err}"
    );
    let err = policy
        .restore_from_checkpoint(fresh_hybrid(), &checkpoint, &short)
        .err()
        .expect("checkpoint restore over too few rows");
    assert!(
        err.contains("2 rows of keys cannot cover an offset of 3"),
        "{err}"
    );

    // The well-formed pair still restores.
    assert!(
        policy
            .restore_from_checkpoint(fresh_hybrid(), &checkpoint, &donor)
            .is_ok()
    );
}
