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

//! Snapshot and restore of `qwen4_exp` per-sequence state against a cold
//! prefill of the same tokens, compared bit for bit (issue #2885).

use mlxcel_core::cache::{KVCacheMode, SequenceId};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::parity_tests::{fixture, tiny_model};
use super::snapshot::{SNAPSHOT_FAMILY, SnapshotParts};
use super::{Qwen4ExpCache, Qwen4ExpModel};

/// The fixture's `eos_token_id`, where the PLE n-gram context resets.
const EOS: i32 = 1;
/// `indexer_compress_ratio` of the fixture.
const RATIO: usize = 4;
/// Tokens through which the fixture's QSA stays dense: its 16-key budget is
/// 4 blocks, so the 20th token is the first with 5 complete blocks.
const DENSE_THROUGH: usize = 19;
/// Decode steps after the restore, as in the acceptance criterion.
const DECODE_STEPS: usize = 16;

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn tokens(len: usize) -> Vec<i32> {
    let prompt = fixture("model")["prompt"]
        .as_array()
        .expect("prompt")
        .iter()
        .map(|t| t.as_i64().expect("token") as i32)
        .collect::<Vec<_>>();
    assert!(
        len <= prompt.len() + 64,
        "fixture prompt is too short for {len}"
    );
    (0..len)
        .map(|i| {
            if i < prompt.len() {
                prompt[i]
            } else {
                (i as i32 % 31) + 2
            }
        })
        .collect()
}

fn raw_bytes(array: &MlxArray) -> Vec<u8> {
    let evaluated = mlxcel_core::astype(array, mlxcel_core::array_dtype(array));
    mlxcel_core::eval(&evaluated);
    mlxcel_core::array_to_raw_bytes(&evaluated)
}

/// Whether two logit tensors match bit for bit, including their dtype.
fn identical(a: &MlxArray, b: &MlxArray) -> bool {
    mlxcel_core::array_dtype(a) == mlxcel_core::array_dtype(b)
        && mlxcel_core::array_shape(a) == mlxcel_core::array_shape(b)
        && raw_bytes(a) == raw_bytes(b)
}

fn float_values(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}

fn decode_tokens() -> Vec<i32> {
    (0..DECODE_STEPS as i32).map(|i| (i * 7) % 29 + 2).collect()
}

fn decode_all(
    model: &Qwen4ExpModel,
    caches: &mut [Qwen4ExpCache],
    probes: &[i32],
) -> Vec<UniquePtr<MlxArray>> {
    probes
        .iter()
        .map(|&token| model.logits(&model.forward_hidden(&ids(&[token]), None, caches, None)))
        .collect()
}

fn cold_caches(model: &Qwen4ExpModel, prompt: &[i32]) -> Vec<Qwen4ExpCache> {
    let mut caches = model.make_internal_caches();
    let _ = model.forward_hidden(&ids(prompt), None, &mut caches, None);
    caches
}

fn assert_restored_matches_cold(model: &Qwen4ExpModel, prompt: &[i32]) {
    let probes = decode_tokens();
    let mut reference = cold_caches(model, prompt);
    let reference_logits = decode_all(model, &mut reference, &probes);

    let source = cold_caches(model, prompt);
    let snapshot = model
        .capture_snapshot(&source, prompt.len(), SnapshotParts::ALL)
        .expect("capture");
    assert_eq!(snapshot.family(), SNAPSHOT_FAMILY);
    assert_eq!(snapshot.token_len(), prompt.len());
    drop(source);
    let mut restored = model.restore_snapshot_caches(&snapshot).expect("restore");
    let restored_logits = decode_all(model, &mut restored, &probes);

    for (step, (cold, warm)) in reference_logits.iter().zip(&restored_logits).enumerate() {
        assert!(
            identical(cold, warm),
            "decode step {step} after a {}-token prompt differs from cold prefill",
            prompt.len()
        );
    }
}

/// The acceptance cases, on the tiny fixture's scale: dense QSA, selected
/// blocks, a prompt ending mid-block, and a prompt containing EOS.
#[test]
fn restored_sequences_match_cold_prefill_bit_for_bit() {
    let model = tiny_model(&fixture("model"));
    let dense = DENSE_THROUGH - 1;
    assert!(
        !dense.is_multiple_of(RATIO),
        "the dense case sits mid-block as well"
    );
    assert_restored_matches_cold(&model, &tokens(dense));
    assert_restored_matches_cold(&model, &tokens(36)); // whole blocks, selected
    assert_restored_matches_cold(&model, &tokens(38)); // ends mid-block, selected

    let mut with_eos = tokens(30);
    with_eos[28] = EOS; // context resets two positions before the end
    assert_restored_matches_cold(&model, &with_eos);
}

fn snapshot_of(model: &Qwen4ExpModel, prompt: &[i32], parts: SnapshotParts) -> ModelStateSnapshot {
    model
        .capture_snapshot(&cold_caches(model, prompt), prompt.len(), parts)
        .expect("capture")
}

fn differs_from_cold(model: &Qwen4ExpModel, prompt: &[i32], parts: SnapshotParts) -> bool {
    let probes = decode_tokens();
    let mut reference = cold_caches(model, prompt);
    let reference_logits = decode_all(model, &mut reference, &probes);
    let snapshot = snapshot_of(model, prompt, parts);
    let mut restored = model.restore_snapshot_caches(&snapshot).expect("restore");
    let restored_logits = decode_all(model, &mut restored, &probes);
    reference_logits
        .iter()
        .zip(&restored_logits)
        .any(|(cold, warm)| !identical(cold, warm))
}

/// Omitting the n-gram contexts changes the n-gram ids of the first token
/// after the restore, which the cold-prefill comparison catches.
#[test]
fn omitting_the_ngram_contexts_makes_the_comparison_fail() {
    let model = tiny_model(&fixture("model"));
    let parts = SnapshotParts {
        ple_contexts: false,
        ..SnapshotParts::ALL
    };
    assert!(
        differs_from_cold(&model, &tokens(30), parts),
        "the comparison must fail without the n-gram contexts"
    );
}

/// The block keys are derived state. A snapshot without them restores a
/// cache holding no summaries, which the state comparison sees, and the next
/// selection recomputes every summary from the stored raw keys and positions
/// to the same bits, so the logits do not change. Wrong summaries are consumed
/// as they stand, which is what shows the logit comparison covers this state.
#[test]
fn block_keys_are_compared_and_omitted_ones_are_recomputed_to_the_same_bits() {
    let model = tiny_model(&fixture("model"));
    let prompt = tokens(36);
    let snapshot = snapshot_of(&model, &prompt, SnapshotParts::ALL);
    let name = "layer3.qsa.block_keys";
    let stored = snapshot
        .tensor(name)
        .expect("the fixture must have selected blocks at 36 tokens");
    let parts = SnapshotParts {
        qsa_block_keys: false,
        ..SnapshotParts::ALL
    };

    let without = model
        .restore_snapshot_caches(&snapshot_of(&model, &prompt, parts))
        .expect("restore");
    let blocks_of = |caches: &[Qwen4ExpCache]| {
        caches.iter().find_map(|cache| match cache {
            Qwen4ExpCache::Attention { index, .. } => Some(index.cached_blocks()),
            Qwen4ExpCache::Linear { .. } => None,
        })
    };
    assert_eq!(blocks_of(&without), Some(0));
    assert_eq!(
        blocks_of(&model.restore_snapshot_caches(&snapshot).expect("restore")),
        blocks_of(&cold_caches(&model, &prompt)),
        "a full snapshot restores the summaries a cold run holds"
    );
    assert!(
        !differs_from_cold(&model, &prompt, parts),
        "recomputing the summaries from the raw keys reproduces them exactly"
    );

    let zeros = mlxcel_core::zeros(
        &mlxcel_core::array_shape(stored),
        mlxcel_core::array_dtype(stored),
    );
    let names = snapshot_names(&model, &snapshot);
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let corrupted = with_tensor(&snapshot, &names, name, &zeros);
    let probes = decode_tokens();
    let mut reference = cold_caches(&model, &prompt);
    let reference_logits = decode_all(&model, &mut reference, &probes);
    let mut restored = model.restore_snapshot_caches(&corrupted).expect("restore");
    let restored_logits = decode_all(&model, &mut restored, &probes);
    assert!(
        reference_logits
            .iter()
            .zip(&restored_logits)
            .any(|(cold, warm)| !identical(cold, warm)),
        "the comparison must fail on wrong block keys"
    );
}

fn expect_restore_error(model: &Qwen4ExpModel, snapshot: &ModelStateSnapshot, wanted: &str) {
    let error = match model.restore_snapshot_caches(snapshot) {
        Ok(_) => panic!("restore must fail mentioning {wanted:?}"),
        Err(error) => error,
    };
    assert!(
        error.contains(wanted),
        "error {error:?} does not name {wanted:?}"
    );
}

/// A copy of `snapshot` with `tensor` replaced (or added) under `name`.
fn with_tensor(
    snapshot: &ModelStateSnapshot,
    names: &[&str],
    name: &str,
    tensor: &MlxArray,
) -> ModelStateSnapshot {
    let mut copy = ModelStateSnapshot::new(snapshot.family(), snapshot.token_len());
    for &kept in names {
        if kept != name {
            copy.push_tensor(kept, snapshot.tensor(kept).expect("listed tensor"));
        }
    }
    copy.push_tensor(name, tensor);
    copy
}

/// Every tensor name a production snapshot of the fixture writes.
fn snapshot_names(model: &Qwen4ExpModel, snapshot: &ModelStateSnapshot) -> Vec<String> {
    let mut names = vec!["layers".to_string()];
    for (idx, layer) in model.layers.iter().enumerate() {
        let prefix = format!("layer{idx}");
        names.push(format!("{prefix}.kind"));
        if layer.is_linear() {
            names.push(format!("{prefix}.conv_state"));
            names.push(format!("{prefix}.state_cache"));
            names.push(format!("{prefix}.ple.conv_state"));
            names.push(format!("{prefix}.ple.contexts"));
        } else {
            for field in ["kv.keys", "kv.values", "kv.offset", "kv.mode"] {
                names.push(format!("{prefix}.{field}"));
            }
            for field in ["qsa.raw_keys", "qsa.positions", "qsa.block_keys"] {
                names.push(format!("{prefix}.{field}"));
            }
        }
    }
    names.retain(|name| snapshot.tensor(name).is_some());
    names
}

#[test]
fn a_foreign_family_or_mismatched_shapes_fail_with_named_errors() {
    let model = tiny_model(&fixture("model"));
    let prompt = tokens(30);
    let snapshot = snapshot_of(&model, &prompt, SnapshotParts::ALL);
    let names = snapshot_names(&model, &snapshot);
    let names: Vec<&str> = names.iter().map(String::as_str).collect();

    let foreign = ModelStateSnapshot::new("qwen3_5", prompt.len());
    expect_restore_error(&model, &foreign, "qwen3_5");

    let bad_layers = mlxcel_core::from_slice_i32(&[3], &[1]);
    let snapshot_with = with_tensor(&snapshot, &names, "layers", &bad_layers);
    expect_restore_error(&model, &snapshot_with, "layer count");

    let attention = (0..model.layers.len())
        .find(|idx| !model.layers[*idx].is_linear())
        .expect("fixture has an attention layer");
    let linear = (0..model.layers.len())
        .find(|idx| model.layers[*idx].is_linear())
        .expect("fixture has a linear layer");

    let wrong = mlxcel_core::zeros(&[1, 2, 2, 2], dtype::FLOAT32);
    let name = format!("layer{linear}.state_cache");
    expect_restore_error(
        &model,
        &with_tensor(&snapshot, &names, &name, &wrong),
        "state_cache",
    );
    let name = format!("layer{attention}.qsa.raw_keys");
    let wrong_keys = mlxcel_core::zeros(&[1, 7, 16], dtype::FLOAT32);
    expect_restore_error(
        &model,
        &with_tensor(&snapshot, &names, &name, &wrong_keys),
        "qsa.raw_keys",
    );
}

#[test]
fn a_snapshot_under_another_kv_mode_is_refused_by_name() {
    let model = tiny_model(&fixture("model"));
    let snapshot = snapshot_of(&model, &tokens(30), SnapshotParts::ALL);
    let other = tiny_model(&fixture("model"));
    other.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; other.layers.len()]);
    expect_restore_error(&other, &snapshot, "mode");
}

/// A speculative block that has not been rewound leaves positions past the
/// claimed prefix, so the capture is refused; once the appended positions are
/// the claimed length the state is restorable and equal to a cold run.
#[test]
fn a_speculative_tail_is_refused_and_a_committed_one_snapshots_like_cold() {
    let model = tiny_model(&fixture("model"));
    let seq = SequenceId::from_raw(11);
    model.prepare_sequence_state(seq);
    let prompt = tokens(30);
    let block = [5, 9, 21];
    let _ = model.forward_prefill_with_streams_for_sequence(&ids(&prompt), Some(seq), prompt.len());
    let _ = model.forward_speculative_for_sequence(&ids(&block), Some(seq));

    assert!(
        model.sequence_snapshot(seq, prompt.len()).is_none(),
        "the unrewound tail must not be stored under the shorter prefix"
    );
    let committed = model
        .sequence_snapshot(seq, prompt.len() + block.len())
        .expect("every appended position is claimed");

    let mut whole = prompt.clone();
    whole.extend_from_slice(&block);
    let probes = decode_tokens();
    let cold_logits = decode_all(&model, &mut cold_caches(&model, &whole), &probes);
    let mut restored = model.restore_snapshot_caches(&committed).expect("restore");
    let restored_logits = decode_all(&model, &mut restored, &probes);
    // Same bits up to the kernel a block forward uses for its tail: the
    // assertion that matters is agreement with the cold run on every token.
    for (step, (cold, warm)) in cold_logits.iter().zip(&restored_logits).enumerate() {
        let (cold, warm) = (float_values(cold), float_values(warm));
        assert_eq!(cold.len(), warm.len());
        for (a, b) in cold.iter().zip(&warm) {
            assert!(
                (a - b).abs() <= 2e-5 * a.abs().max(1.0),
                "step {step}: {a} against {b}"
            );
        }
    }

    model.release_sequence_state_by_id(seq);
}

/// A restore starts the sequence with an empty rollback log, so a rewind
/// past the restore point is refused instead of replaying an unrelated
/// capture.
#[test]
fn a_restored_sequence_refuses_a_rewind_past_the_restore_point() {
    let model = tiny_model(&fixture("model"));
    let seq = SequenceId::from_raw(21);
    model.prepare_sequence_state(seq);
    let prompt = tokens(24);
    let snapshot = snapshot_of(&model, &prompt, SnapshotParts::ALL);
    model
        .restore_sequence_state(seq, &snapshot)
        .expect("restore");
    let error = model
        .rewind_decode_appends(seq, 1)
        .expect_err("nothing to rewind");
    assert!(error.contains("rewind"), "unexpected rewind error: {error}");
    model.release_sequence_state_by_id(seq);
}

/// Decision for image and video requests: reuse is declined. A sequence
/// holding image or video MRoPE state does not snapshot.
#[test]
fn a_sequence_with_media_positions_declines_the_snapshot() {
    let model = tiny_model(&fixture("model"));
    let seq = SequenceId::from_raw(31);
    model.prepare_sequence_state(seq);
    let prompt = tokens(24);
    let _ = model.forward_prefill_with_streams_for_sequence(&ids(&prompt), Some(seq), prompt.len());
    assert!(
        model.sequence_snapshot(seq, prompt.len()).is_some(),
        "the text-only sequence snapshots"
    );
    let positions = mlxcel_core::zeros(&[3, 1, 24], dtype::INT32);
    model.set_mrope_state(positions, 0);
    model.bind_mrope_state_to_sequence(seq);
    assert!(
        model.sequence_snapshot(seq, prompt.len()).is_none(),
        "the media sequence must not snapshot"
    );
    model.release_sequence_state_by_id(seq);
}

/// A row store that delegates to another and asks for gather-ahead, standing
/// in for the external (`ple_storage`) table.
struct CountingStore {
    inner: Box<dyn crate::models::qwen4_exp::ple_storage::PleRowStore>,
    lookups: std::rc::Rc<std::cell::Cell<usize>>,
}

impl crate::models::qwen4_exp::ple_storage::PleRowStore for CountingStore {
    fn lookup(
        &self,
        ids: &[i64],
    ) -> Result<UniquePtr<MlxArray>, crate::models::qwen4_exp::ple_storage::PleStoreError> {
        self.lookups.set(self.lookups.get() + 1);
        self.inner.lookup(ids)
    }

    fn wants_prefetch(&self) -> bool {
        true
    }
}

/// The snapshot holds per-sequence state only: captured under one row store,
/// it restores to the same decode under another, and it stores no row data.
#[test]
fn a_snapshot_restores_identically_over_another_row_store() {
    let mut model = tiny_model(&fixture("model"));
    let lookups = std::rc::Rc::new(std::cell::Cell::new(0));
    let prompt = tokens(36);
    let snapshot = snapshot_of(&model, &prompt, SnapshotParts::ALL);
    let probes = decode_tokens();
    let mut resident = cold_caches(&model, &prompt);
    let resident_logits = decode_all(&model, &mut resident, &probes);

    let counter = lookups.clone();
    for layer in &mut model.layers {
        if let Some(ple) = layer.ple.as_mut() {
            let counter = counter.clone();
            ple.wrap_table_for_test(|inner| {
                Box::new(CountingStore {
                    inner,
                    lookups: counter,
                })
            });
        }
    }
    let mut restored = model.restore_snapshot_caches(&snapshot).expect("restore");
    let restored_logits = decode_all(&model, &mut restored, &probes);
    assert!(lookups.get() > 0, "the decode must read the other store");
    for (step, (cold, warm)) in resident_logits.iter().zip(&restored_logits).enumerate() {
        assert!(identical(cold, warm), "step {step} differs across stores");
    }
}

#[test]
fn snapshots_keep_every_tensor_at_its_stored_dtype() {
    let model = tiny_model(&fixture("model"));
    let caches = cold_caches(&model, &tokens(30));
    let snapshot = model
        .capture_snapshot(&caches, 30, SnapshotParts::ALL)
        .expect("capture");
    let restored = model.restore_snapshot_caches(&snapshot).expect("restore");
    for (source, copy) in caches.iter().zip(&restored) {
        match (source, copy) {
            (Qwen4ExpCache::Linear { delta: a, .. }, Qwen4ExpCache::Linear { delta: b, .. }) => {
                let dtypes = |c: &crate::models::gated_delta::GatedDeltaCache| {
                    (
                        c.state_cache.as_ref().map(|t| mlxcel_core::array_dtype(t)),
                        c.conv_state.as_ref().map(|t| mlxcel_core::array_dtype(t)),
                    )
                };
                assert_eq!(dtypes(a), dtypes(b));
                assert!(a.state_cache.is_some(), "linear state must be stored");
            }
            (
                Qwen4ExpCache::Attention { kv: a, index: ia },
                Qwen4ExpCache::Attention { kv: b, index: ib },
            ) => {
                assert_eq!(
                    a.keys.as_ref().map(|t| mlxcel_core::array_dtype(t)),
                    b.keys.as_ref().map(|t| mlxcel_core::array_dtype(t))
                );
                assert_eq!(
                    ia.raw_keys.as_ref().map(|t| mlxcel_core::array_dtype(t)),
                    ib.raw_keys.as_ref().map(|t| mlxcel_core::array_dtype(t))
                );
            }
            _ => panic!("layer order differs"),
        }
    }
}

/// A multi-token suffix after a restore that lands mid-block, the shape a
/// second turn takes, is bit-identical to a cold prefill split at the same
/// boundary. Against a cold prefill that forwards the whole prompt at once it
/// agrees to rounding, as any two chunkings of this model do.
#[test]
fn a_multi_token_suffix_after_a_mid_block_restore_matches_a_cold_split() {
    let model = tiny_model(&fixture("model"));
    for boundary in [37_usize, 38, 39, 41] {
        let whole = tokens(boundary + 9);
        let probes = decode_tokens();

        let mut split = cold_caches(&model, &whole[..boundary]);
        let _ = model.forward_hidden(&ids(&whole[boundary..]), None, &mut split, None);
        let split_logits = decode_all(&model, &mut split, &probes);

        let snapshot = snapshot_of(&model, &whole[..boundary], SnapshotParts::ALL);
        let mut warm = model.restore_snapshot_caches(&snapshot).expect("restore");
        let _ = model.forward_hidden(&ids(&whole[boundary..]), None, &mut warm, None);
        let warm_logits = decode_all(&model, &mut warm, &probes);

        let mut whole_cold = cold_caches(&model, &whole);
        let whole_logits = decode_all(&model, &mut whole_cold, &probes);

        for step in 0..probes.len() {
            assert!(
                identical(&split_logits[step], &warm_logits[step]),
                "boundary {boundary}: step {step} differs from the same split run cold"
            );
            let (a, b) = (
                float_values(&whole_logits[step]),
                float_values(&warm_logits[step]),
            );
            let worst = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
                .fold(0.0_f32, f32::max);
            assert!(
                worst < 1e-3,
                "boundary {boundary}: step {step} is {worst} from a one-shot cold prefill"
            );
        }
    }
}

/// The donor sequence keeps running after the capture, as a history-boundary
/// snapshot's sequence does; its later forwards must not reach the snapshot
/// through a shared buffer.
#[test]
fn a_snapshot_is_unaffected_by_the_donor_continuing() {
    let model = tiny_model(&fixture("model"));
    let prompt = tokens(30);
    let probes = decode_tokens();
    let mut cold = cold_caches(&model, &prompt);
    let cold_logits = decode_all(&model, &mut cold, &probes);

    let mut donor = cold_caches(&model, &prompt);
    let snapshot = model
        .capture_snapshot(&donor, prompt.len(), SnapshotParts::ALL)
        .expect("capture");
    let continuation: Vec<i32> = (0..64).map(|i| i % 31 + 2).collect();
    let _ = decode_all(&model, &mut donor, &continuation);
    let _ = model.forward_hidden(&ids(&tokens(24)), None, &mut donor, None);

    let mut restored = model.restore_snapshot_caches(&snapshot).expect("restore");
    let restored_logits = decode_all(&model, &mut restored, &probes);
    for (step, (a, b)) in cold_logits.iter().zip(&restored_logits).enumerate() {
        assert!(
            identical(a, b),
            "step {step} differs after the donor ran on"
        );
    }
}

/// Issue #2974: a chained prompt-cache hit, as a multi-turn chat on the
/// server produces it. Turn 1 prefills `[0..a)` and snapshots; turn 2
/// restores that, prefills `[a..b)`, snapshots again and runs on (its
/// generation-prompt suffix and decode); turn 3 restores the second snapshot
/// and prefills `[b..n)`. Its logits must equal a cold run that forwards the
/// same three pieces bit for bit, so the restore chain itself loses no state.
/// Against a cold run with another partition the chain agrees only to
/// rounding, which is the partition effect #2974 measured on the real model.
#[test]
fn a_chained_restore_matches_a_cold_run_of_the_same_partition() {
    let model = tiny_model(&fixture("model"));
    let probes = decode_tokens();
    for (a, b, n) in [(22_usize, 38_usize, 55_usize), (24, 40, 56), (10, 18, 47)] {
        let prompt = tokens(n);

        let first = snapshot_of(&model, &prompt[..a], SnapshotParts::ALL);
        let mut turn2 = model
            .restore_snapshot_caches(&first)
            .expect("restore turn 2");
        let _ = model.forward_hidden(&ids(&prompt[a..b]), None, &mut turn2, None);
        let second = model
            .capture_snapshot(&turn2, b, SnapshotParts::ALL)
            .expect("capture turn 2");
        // Turn 2 goes on with its generation-prompt suffix and its reply.
        let _ = model.forward_hidden(&ids(&[5, 6, 7]), None, &mut turn2, None);
        let _ = decode_all(&model, &mut turn2, &probes[..4]);
        drop(turn2);

        let mut turn3 = model
            .restore_snapshot_caches(&second)
            .expect("restore turn 3");
        let _ = model.forward_hidden(&ids(&prompt[b..]), None, &mut turn3, None);
        let chained = decode_all(&model, &mut turn3, &probes);

        let mut cold = model.make_internal_caches();
        for range in [0..a, a..b, b..n] {
            let _ = model.forward_hidden(&ids(&prompt[range]), None, &mut cold, None);
        }
        let same_partition = decode_all(&model, &mut cold, &probes);

        let mut whole = cold_caches(&model, &prompt);
        let one_piece = decode_all(&model, &mut whole, &probes);

        for step in 0..probes.len() {
            assert!(
                identical(&chained[step], &same_partition[step]),
                "({a}, {b}, {n}): step {step} differs from the same partition run cold"
            );
            let (x, y) = (float_values(&one_piece[step]), float_values(&chained[step]));
            let worst = x
                .iter()
                .zip(&y)
                .map(|(p, q)| (p - q).abs() / p.abs().max(1.0))
                .fold(0.0_f32, f32::max);
            assert!(
                worst < 1e-3,
                "({a}, {b}, {n}): step {step} is {worst} from a one-piece cold prefill"
            );
        }
    }
}
