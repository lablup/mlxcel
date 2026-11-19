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

//! Turn-boundary state checkpoints for `qwen4_exp` (issue #3047).
//!
//! A checkpoint keeps the recurrent state at a turn boundary and takes the
//! attention rows from a longer snapshot of the same conversation, cut back
//! with the speculative-rollback trim. These tests compare the rebuilt state
//! with a direct prefill of the shorter prefix, bit for bit, before the
//! server is allowed to serve it.

use mlxcel_core::{MlxArray, UniquePtr};

use super::parity_tests::{fixture, tiny_model};
use super::snapshot::SnapshotParts;
use super::snapshot_tests::{cold_caches, decode_all, decode_tokens, identical, ids, tokens};
use super::{Qwen4ExpCache, Qwen4ExpModel};

/// `(checkpoint, donor)` lengths across the fixture's QSA regimes: a dense
/// prefix under a selecting donor, a block-aligned prefix, a prefix that ends
/// mid-block, and a one-token cut.
const CASES: &[(usize, usize)] = &[(18, 30), (24, 38), (30, 38), (36, 37)];

/// Bytes of `array` at its own dtype.
fn bytes(array: &MlxArray) -> Vec<u8> {
    let evaluated = mlxcel_core::contiguous(array, false);
    mlxcel_core::eval(&evaluated);
    mlxcel_core::array_to_raw_bytes(&evaluated)
}

/// The first `rows` positions of a `[B, H, T, D]` KV buffer.
fn kv_rows(array: &MlxArray, rows: i32) -> Vec<u8> {
    let shape = mlxcel_core::array_shape(array);
    bytes(&mlxcel_core::slice(
        array,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], rows, shape[3]],
    ))
}

/// Every attention layer's visible KV rows and QSA index, and every linear
/// layer's recurrent state, as bytes.
fn state_bytes(caches: &[Qwen4ExpCache]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for cache in caches {
        match cache {
            Qwen4ExpCache::Attention { kv, index } => {
                let rows = kv.offset;
                out.push(kv_rows(kv.keys.as_ref().expect("keys"), rows));
                out.push(kv_rows(kv.values.as_ref().expect("values"), rows));
                for part in [&index.raw_keys, &index.positions, &index.block_keys] {
                    out.push(part.as_ref().map(|a| bytes(a)).unwrap_or_default());
                }
            }
            Qwen4ExpCache::Linear { delta, ple } => {
                for part in [&delta.conv_state, &delta.state_cache] {
                    out.push(part.as_ref().map(|a| bytes(a)).unwrap_or_default());
                }
                if let Some(ple) = ple {
                    out.push(
                        ple.conv_state
                            .as_ref()
                            .map(|a| bytes(a))
                            .unwrap_or_default(),
                    );
                    out.push(
                        ple.contexts
                            .iter()
                            .flatten()
                            .flat_map(|t| t.to_le_bytes())
                            .collect(),
                    );
                }
            }
        }
    }
    out
}

/// The checkpoint at `keep` tokens and the donor at `have`, from one chain:
/// prefill `[0, keep)`, capture, prefill `[keep, have)`, capture. This is how
/// a conversation's turns build them in the server.
fn chain(
    model: &Qwen4ExpModel,
    prompt: &[i32],
    keep: usize,
) -> (
    mlxcel_core::generate::ModelStateSnapshot,
    mlxcel_core::generate::ModelStateSnapshot,
) {
    let mut caches = cold_caches(model, &prompt[..keep]);
    let at_keep = model
        .capture_snapshot(&caches, keep, SnapshotParts::ALL)
        .expect("capture the turn boundary");
    let checkpoint = model.state_checkpoint(&at_keep).expect("checkpoint");
    drop(at_keep);
    let _ = model.forward_hidden(&ids(&prompt[keep..]), None, &mut caches, None);
    let donor = model
        .capture_snapshot(&caches, prompt.len(), SnapshotParts::ALL)
        .expect("capture the newest turn");
    (checkpoint, donor)
}

fn logits_differ(a: &[UniquePtr<MlxArray>], b: &[UniquePtr<MlxArray>]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| !identical(x, y))
}

/// The acceptance check of #3047: the state rebuilt from a checkpoint and a
/// longer snapshot is the state a direct prefill of the shorter prefix
/// writes, bit for bit, and so is every decode step after it.
#[test]
fn a_checkpoint_over_a_longer_snapshot_matches_a_direct_prefill_bit_for_bit() {
    let model = tiny_model(&fixture("model"));
    let probes = decode_tokens();
    for &(keep, have) in CASES {
        let prompt = tokens(have);
        let (checkpoint, donor) = chain(&model, &prompt, keep);
        assert_eq!(checkpoint.token_len(), keep);
        for name in ["layer3.kv.keys", "layer3.qsa.raw_keys"] {
            assert!(
                checkpoint.tensor(name).is_none(),
                "({keep}, {have}): the checkpoint must not carry {name}"
            );
        }
        assert!(
            checkpoint.nbytes() < donor.nbytes(),
            "({keep}, {have}): the checkpoint is not smaller than a whole snapshot"
        );

        let direct = cold_caches(&model, &prompt[..keep]);
        let mut rebuilt = model
            .checkpoint_caches(&checkpoint, &donor)
            .expect("rebuild from the checkpoint");
        assert_eq!(Qwen4ExpCache::offset(&rebuilt), keep as i32);
        let (want, got) = (state_bytes(&direct), state_bytes(&rebuilt));
        assert_eq!(want.len(), got.len());
        for (part, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(
                w == g,
                "({keep}, {have}): state part {part} differs from a direct prefill"
            );
        }

        let mut direct = direct;
        let direct_logits = decode_all(&model, &mut direct, &probes);
        let rebuilt_logits = decode_all(&model, &mut rebuilt, &probes);
        assert_eq!(
            logits_differ(&direct_logits, &rebuilt_logits),
            None,
            "({keep}, {have}): a decode step after the rebuilt state differs"
        );
    }
}

/// The next turn's suffix prefilled over the rebuilt state reproduces the
/// same suffix prefilled over a direct prefill, which is what a request that
/// edits history after the checkpoint runs.
#[test]
fn a_suffix_prefilled_over_a_rebuilt_state_matches_the_direct_chain() {
    let model = tiny_model(&fixture("model"));
    let keep = 24;
    let prompt = tokens(38);
    let (checkpoint, donor) = chain(&model, &prompt, keep);
    let edited: Vec<i32> = (0..10).map(|i| (i * 5) % 23 + 2).collect();

    let mut direct = cold_caches(&model, &prompt[..keep]);
    let direct_suffix = model.logits(&model.forward_hidden(&ids(&edited), None, &mut direct, None));
    let mut rebuilt = model
        .checkpoint_caches(&checkpoint, &donor)
        .expect("rebuild");
    let rebuilt_suffix =
        model.logits(&model.forward_hidden(&ids(&edited), None, &mut rebuilt, None));
    assert!(identical(&direct_suffix, &rebuilt_suffix));
    let probes = decode_tokens();
    assert_eq!(
        logits_differ(
            &decode_all(&model, &mut direct, &probes),
            &decode_all(&model, &mut rebuilt, &probes)
        ),
        None
    );
}

/// A donor shorter than the checkpoint, a foreign family, and a checkpoint
/// missing a linear layer's state are refused with a named error.
#[test]
fn mismatched_checkpoints_are_refused() {
    let model = tiny_model(&fixture("model"));
    let prompt = tokens(30);
    let (checkpoint, donor) = chain(&model, &prompt, 18);
    let (_, short_donor) = chain(&model, &prompt[..20], 12);
    let (long_checkpoint, _) = chain(&model, &prompt, 24);
    assert!(
        model
            .checkpoint_caches(&long_checkpoint, &short_donor)
            .is_err()
    );

    let foreign = mlxcel_core::generate::ModelStateSnapshot::new("qwen3_5", 18);
    assert!(model.checkpoint_caches(&foreign, &donor).is_err());

    let gutted = checkpoint.subset(|name| !name.ends_with("state_cache"));
    let err = model
        .checkpoint_caches(&gutted, &donor)
        .err()
        .expect("a checkpoint without GatedDeltaNet state must be refused");
    assert!(err.contains("missing GatedDeltaNet state"), "{err}");
}
