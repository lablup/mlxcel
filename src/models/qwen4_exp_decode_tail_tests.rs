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

//! State written by decode steps against state written by a prefill of the
//! same tokens (issue #3040).
//!
//! A completion snapshot holds `prompt + reply`, and the reply part of its
//! state was written one token at a time by decode. A cold run of the next
//! turn prefills the same reply tokens as part of a longer piece. These tests
//! compare the two on the tiny fixture, so the cause of the live-replies
//! `--verify` mismatches is pinned at the model level.

use mlxcel_core::MlxArray;

use super::Qwen4ExpModel;
use super::parity_tests::{fixture, tiny_model};
use super::snapshot::SnapshotParts;
use super::snapshot_tests::{
    cold_caches, decode_all, decode_tokens, float_values, identical, ids, tokens,
};

/// Worst relative difference between two logit tensors.
fn worst_relative(a: &MlxArray, b: &MlxArray) -> f32 {
    let (x, y) = (float_values(a), float_values(b));
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(&y)
        .map(|(p, q)| (p - q).abs() / p.abs().max(1.0))
        .fold(0.0_f32, f32::max)
}

/// Probe logits after restoring a snapshot of `caches` at `len` tokens, the
/// way the next turn consumes a stored entry.
fn probe_after_restore(
    model: &Qwen4ExpModel,
    caches: &[super::Qwen4ExpCache],
    len: usize,
) -> Vec<mlxcel_core::UniquePtr<MlxArray>> {
    let snapshot = model
        .capture_snapshot(caches, len, SnapshotParts::ALL)
        .expect("capture");
    let mut restored = model.restore_snapshot_caches(&snapshot).expect("restore");
    decode_all(model, &mut restored, &decode_tokens())
}

/// The completion snapshot's tail is decode-written: prefill the prompt, then
/// forward the reply one token at a time. Its state is not the state a
/// prefill of the same reply tokens writes, neither split at the prompt
/// boundary nor in one piece with the prompt, so restoring it reproduces no
/// cold run of the next turn. On the fixture the gap is a rounding-sized
/// 1.5e-7 to 2.1e-7 relative, the same order as the gap between the two
/// cold partitions; on the bfloat16 real model that is the 1-3 ULP class
/// that flips greedy near-ties (#2974).
///
/// What #3040 stores instead, the prompt-boundary state with the reply
/// prefilled after the restore, is bit-identical to the cold run split at
/// the prompt boundary.
#[test]
fn a_decode_written_tail_differs_from_a_prefill_of_the_same_tokens() {
    let model = tiny_model(&fixture("model"));
    for (prompt_len, reply_len) in [(22_usize, 9_usize), (30, 8), (38, 12)] {
        let whole = tokens(prompt_len + reply_len);
        let (prompt, reply) = whole.split_at(prompt_len);
        let len = whole.len();
        let case = format!("({prompt_len}, {reply_len})");

        let mut decoded = cold_caches(&model, prompt);
        let _ = decode_all(&model, &mut decoded, reply);
        let decoded = probe_after_restore(&model, &decoded, len);

        let mut split = cold_caches(&model, prompt);
        let _ = model.forward_hidden(&ids(reply), None, &mut split, None);
        let split = probe_after_restore(&model, &split, len);

        let one_piece = probe_after_restore(&model, &cold_caches(&model, &whole), len);

        let boundary = model
            .capture_snapshot(&cold_caches(&model, prompt), prompt_len, SnapshotParts::ALL)
            .expect("capture the prompt boundary");
        let mut reprefilled = model.restore_snapshot_caches(&boundary).expect("restore");
        let _ = model.forward_hidden(&ids(reply), None, &mut reprefilled, None);
        let reprefilled = decode_all(&model, &mut reprefilled, &decode_tokens());

        let mut decode_bits_differ = false;
        for step in 0..decoded.len() {
            decode_bits_differ |= !identical(&decoded[step], &split[step])
                && !identical(&decoded[step], &one_piece[step]);
            assert!(
                worst_relative(&one_piece[step], &decoded[step]) < 1e-3,
                "{case}: step {step} is beyond rounding from the cold run"
            );
            assert!(
                identical(&reprefilled[step], &split[step]),
                "{case}: step {step} of the re-prefilled reply differs from the cold split"
            );
        }
        assert!(
            decode_bits_differ,
            "{case}: decode-written state matched a prefill of the same tokens"
        );
    }
}
