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

//! Speculative rollback on the tiny `qwen4_exp` fixture (issue #2880).
//!
//! Every test verifies a block of `K` tokens, rolls `R` of them back, decodes
//! a few probe tokens and compares the probe logits with a run that never saw
//! the rejected tokens. The fixture (`tests/fixtures/qwen4_exp/model.json`)
//! has one PLE layer, one QSA layer and MoE on four layers, with
//! `indexer_budget 16` and `compress_ratio 4`: attention is dense while a
//! query sees at most four complete blocks (through its 19th token) and sparse
//! from the 20th, the analogue of the released config's 2051/2052 switch.
//!
//! A rollback that forgets a component produces a plausible wrong state
//! rather than a crash, so each component also has a check that fails when
//! that component's rollback is switched off ([`RollbackParts`]).

use mlxcel_core::cache::{DecodeLookaheadAppendScope, SequenceId};
use mlxcel_core::engine::{Engine, SequenceSpec};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::speculative::mtp::target::MtpTarget;
use mlxcel_core::speculative::prompt_lookup::{
    PromptLookupConfig, prompt_lookup_unsupported_reason,
};
use mlxcel_core::speculative::prompt_lookup_drafter::PromptLookupDrafter;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::parity_tests::{fixture, quantized_tiny_model, tiny_model};
use super::rollback::RollbackParts;
use super::{Qwen4ExpCache, Qwen4ExpModel};
use crate::models::qwen4_exp_mtp_target::Qwen4ExpMtpTargetAdapter;

/// The fixture's `eos_token_id`, where the PLE n-gram context resets.
const EOS: i32 = 1;
/// `indexer_compress_ratio` of the fixture.
const RATIO: i32 = 4;
/// Both arms are float32 and differ only in kernel shape: the reference
/// forwards the kept tokens as a chunk of their own, the candidate as the
/// head of a wider block, so the roundings differ by ulps at most.
const RTOL: f32 = 2e-5;

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn values(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}

/// `max |a - b| / max |b|`.
fn rel_diff(actual: &MlxArray, expected: &MlxArray) -> f32 {
    let (actual, expected) = (values(actual), values(expected));
    assert_eq!(actual.len(), expected.len(), "shape mismatch");
    let scale = expected
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
        / scale
}

#[track_caller]
fn assert_close(label: &str, actual: &MlxArray, expected: &MlxArray) {
    let diff = rel_diff(actual, expected);
    assert!(
        diff <= RTOL,
        "{label}: relative difference {diff:e} exceeds {RTOL:e}"
    );
}

#[track_caller]
fn assert_differs(label: &str, actual: &MlxArray, expected: &MlxArray) {
    let diff = rel_diff(actual, expected);
    assert!(
        diff > RTOL,
        "{label}: relative difference {diff:e} is within {RTOL:e}, so this check would not \
         catch a rollback that skipped the component"
    );
}

/// The MTP exactness gate retries a failing probe with `qmv_wide` switched
/// off: on GPU generation 15 and later MLX sends `M >= 2` affine quantized
/// matmuls down `qmv_wide`, which reduces in another order than the `qmv` a
/// single row takes, so no multi-row block can match the chain under it.
/// The block-versus-chain tests run under the same switch and restore it.
struct QmvWideOff {
    previous: bool,
}

impl QmvWideOff {
    fn enter() -> Self {
        let previous = mlxcel_core::qmv_wide_enabled();
        mlxcel_core::set_qmv_wide(false);
        Self { previous }
    }
}

impl Drop for QmvWideOff {
    fn drop(&mut self) {
        mlxcel_core::set_qmv_wide(self.previous);
    }
}

fn prompt_tokens(len: usize) -> Vec<i32> {
    let fixture = fixture("model");
    fixture["prompt"]
        .as_array()
        .expect("prompt")
        .iter()
        .take(len)
        .map(|t| t.as_i64().expect("token") as i32)
        .collect()
}

fn prefill(model: &Qwen4ExpModel, prompt: &[i32]) -> Vec<Qwen4ExpCache> {
    let mut caches = model.make_internal_caches();
    let _ = model.forward_hidden(&ids(prompt), None, &mut caches, None);
    caches
}

fn step(
    model: &Qwen4ExpModel,
    caches: &mut [Qwen4ExpCache],
    tokens: &[i32],
) -> UniquePtr<MlxArray> {
    model.logits(&model.forward_hidden(&ids(tokens), None, caches, None))
}

/// Probe logits after `prompt` and `kept`, decoded one token at a time, with
/// the caches they leave behind.
fn reference(
    model: &Qwen4ExpModel,
    prompt: &[i32],
    kept: &[i32],
    probes: &[i32],
) -> (Vec<UniquePtr<MlxArray>>, Vec<Qwen4ExpCache>) {
    let mut caches = prefill(model, prompt);
    if !kept.is_empty() {
        let _ = step(model, &mut caches, kept);
    }
    let logits = probes
        .iter()
        .map(|&token| step(model, &mut caches, &[token]))
        .collect();
    (logits, caches)
}

/// Verify `block` after `prompt`, keep its first `keep` positions through a
/// rollback restoring `parts`, and return the caches.
fn rolled_back(
    model: &Qwen4ExpModel,
    prompt: &[i32],
    block: &[i32],
    keep: i32,
    parts: RollbackParts,
) -> Vec<Qwen4ExpCache> {
    let mut caches = prefill(model, prompt);
    let out = model.forward_speculative(&ids(block), &mut caches, None);
    model
        .rollback_speculative_parts(&mut caches, &out.capture, keep, parts)
        .expect("rollback");
    caches
}

/// [`rolled_back`] followed by the probe decodes.
fn candidate(
    model: &Qwen4ExpModel,
    prompt: &[i32],
    block: &[i32],
    keep: i32,
    probes: &[i32],
    parts: RollbackParts,
) -> (Vec<UniquePtr<MlxArray>>, Vec<Qwen4ExpCache>) {
    let mut caches = rolled_back(model, prompt, block, keep, parts);
    let logits = probes
        .iter()
        .map(|&token| step(model, &mut caches, &[token]))
        .collect();
    (logits, caches)
}

fn attention(
    caches: &[Qwen4ExpCache],
) -> (&mlxcel_core::layers::KVCache, &super::qsa::QsaIndexCache) {
    caches
        .iter()
        .find_map(|cache| match cache {
            Qwen4ExpCache::Attention { kv, index } => Some((kv, index)),
            Qwen4ExpCache::Linear { .. } => None,
        })
        .expect("the fixture has an attention layer")
}

fn ple_contexts(caches: &[Qwen4ExpCache]) -> &Vec<Vec<i64>> {
    caches
        .iter()
        .find_map(|cache| match cache {
            Qwen4ExpCache::Linear { ple: Some(ple), .. } => Some(&ple.contexts),
            _ => None,
        })
        .expect("the fixture has a PLE layer")
}

fn assert_optional_close(
    label: &str,
    actual: &Option<UniquePtr<MlxArray>>,
    expected: &Option<UniquePtr<MlxArray>>,
) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_close(label, actual, expected),
        (None, None) => {}
        _ => panic!("{label}: one side holds a tensor and the other does not"),
    }
}

/// Every piece of per-sequence state of `candidate` agrees with `reference`
/// (the block keys are a lazy cache, so they are compared only when both
/// sides materialized them, and the candidate's never exceed the complete
/// blocks of its length).
fn assert_states_match(reference: &[Qwen4ExpCache], candidate: &[Qwen4ExpCache]) {
    assert_eq!(reference.len(), candidate.len());
    for (layer, (r, c)) in reference.iter().zip(candidate).enumerate() {
        match (r, c) {
            (
                Qwen4ExpCache::Attention { kv: rk, index: ri },
                Qwen4ExpCache::Attention { kv: ck, index: ci },
            ) => {
                assert_eq!(rk.offset, ck.offset, "layer {layer}: kv offset");
                assert_eq!(ri.len(), ci.len(), "layer {layer}: index length");
                assert_optional_close(
                    &format!("layer {layer}: raw keys"),
                    &ci.raw_keys,
                    &ri.raw_keys,
                );
                assert_optional_close(
                    &format!("layer {layer}: positions"),
                    &ci.positions,
                    &ri.positions,
                );
                assert!(
                    ci.cached_blocks() <= ci.len() / RATIO,
                    "layer {layer}: {} block keys for {} positions",
                    ci.cached_blocks(),
                    ci.len()
                );
                if let (Some(rb), Some(cb)) = (&ri.block_keys, &ci.block_keys) {
                    assert_eq!(
                        ri.cached_blocks(),
                        ci.cached_blocks(),
                        "layer {layer}: block count"
                    );
                    assert_close(&format!("layer {layer}: block keys"), cb, rb);
                }
            }
            (
                Qwen4ExpCache::Linear { delta: rd, ple: rp },
                Qwen4ExpCache::Linear { delta: cd, ple: cp },
            ) => {
                assert_eq!(rd.offset, cd.offset, "layer {layer}: delta offset");
                assert_optional_close(
                    &format!("layer {layer}: conv state"),
                    &cd.conv_state,
                    &rd.conv_state,
                );
                assert_optional_close(
                    &format!("layer {layer}: recurrent state"),
                    &cd.state_cache,
                    &rd.state_cache,
                );
                match (rp, cp) {
                    (Some(rp), Some(cp)) => {
                        assert_eq!(rp.contexts, cp.contexts, "layer {layer}: n-gram contexts");
                        assert_optional_close(
                            &format!("layer {layer}: PLE conv state"),
                            &cp.conv_state,
                            &rp.conv_state,
                        );
                    }
                    (None, None) => {}
                    _ => panic!("layer {layer}: PLE cache present on one side only"),
                }
            }
            _ => panic!("layer {layer}: cache kinds differ"),
        }
    }
}

#[track_caller]
fn assert_probes_close(reference: &[UniquePtr<MlxArray>], candidate: &[UniquePtr<MlxArray>]) {
    assert_eq!(reference.len(), candidate.len());
    for (i, (r, c)) in reference.iter().zip(candidate).enumerate() {
        assert_close(&format!("probe {i} logits"), c, r);
    }
}

/// Reject part of a block that completed a QSA block in the sparse regime:
/// 39 prompt tokens, a 5-token block (positions 39..43 complete the 11th
/// block), keep 2 (41 positions, 10 complete blocks). The 11th block summary
/// has to go, and everything else has to agree with a run of the two kept
/// tokens alone.
#[test]
fn rollback_across_a_qsa_block_boundary_matches_a_run_without_the_rejected_tokens() {
    let model = tiny_model(&fixture("model"));
    let prompt = prompt_tokens(39);
    let block = [5, 9, 21, 33, 40];
    let probes = [12, 7, 3];
    let (ref_logits, ref_caches) = reference(&model, &prompt, &block[..2], &probes);
    let (cand_logits, cand_caches) =
        candidate(&model, &prompt, &block, 2, &probes, RollbackParts::ALL);
    assert_states_match(&ref_caches, &cand_caches);
    assert_probes_close(&ref_logits, &cand_logits);

    // Without trimming the block keys the 11th summary (over positions 40..43,
    // three of them rejected) survives the rollback.
    let trimmed = rolled_back(&model, &prompt, &block, 2, RollbackParts::ALL);
    assert_eq!(attention(&trimmed).1.cached_blocks(), 10);
    assert_eq!(attention(&trimmed).1.len(), 41);
    let stale = rolled_back(
        &model,
        &prompt,
        &block,
        2,
        RollbackParts {
            qsa_blocks: false,
            ..RollbackParts::ALL
        },
    );
    assert_eq!(attention(&stale).1.cached_blocks(), 11);
}

/// Reject across the dense/sparse switch: 17 prompt tokens (dense), a
/// 5-token block to 22 positions (the indexer selects for the first time and
/// caches five block summaries), keep 2 back to 19 (dense again), then probe
/// into the sparse regime, where the retained summaries must equal the ones a
/// fresh run computes.
#[test]
fn rollback_across_the_dense_sparse_switch_matches_a_run_without_the_rejected_tokens() {
    let model = tiny_model(&fixture("model"));
    let prompt = prompt_tokens(17);
    let block = [5, 9, 21, 33, 40];
    let probes = [12, 7, 3];
    let (ref_logits, ref_caches) = reference(&model, &prompt, &block[..2], &probes);
    let rolled = rolled_back(&model, &prompt, &block, 2, RollbackParts::ALL);
    let (kv, index) = attention(&rolled);
    assert_eq!(kv.offset, 19);
    assert_eq!(index.len(), 19);
    assert_eq!(
        index.cached_blocks(),
        4,
        "the fifth block is no longer complete"
    );
    let (cand_logits, cand_caches) =
        candidate(&model, &prompt, &block, 2, &probes, RollbackParts::ALL);
    assert_eq!(attention(&cand_caches).1.cached_blocks(), 5);
    assert_states_match(&ref_caches, &cand_caches);
    assert_probes_close(&ref_logits, &cand_logits);
}

/// A drafted EOS that the target rejects must not reset the n-gram context:
/// after keeping one of `[5, EOS, 9, 21]` the context is the kept token and
/// its predecessor, and the probe logits match a run that never saw the EOS.
#[test]
fn rejected_eos_does_not_reset_the_ngram_context() {
    let model = tiny_model(&fixture("model"));
    let prompt = prompt_tokens(12);
    let block = [5, EOS, 9, 21];
    let probes = [12, 7, 3];
    let (ref_logits, ref_caches) = reference(&model, &prompt, &block[..1], &probes);
    let (cand_logits, cand_caches) =
        candidate(&model, &prompt, &block, 1, &probes, RollbackParts::ALL);
    assert_eq!(
        ple_contexts(&rolled_back(&model, &prompt, &block, 1, RollbackParts::ALL)),
        &vec![vec![i64::from(prompt[11]), 5]]
    );
    assert_states_match(&ref_caches, &cand_caches);
    assert_probes_close(&ref_logits, &cand_logits);

    // Without the context rollback the block's last two tokens, which came
    // after the rejected EOS, stay as the context the next probe hashes
    // against.
    let parts = RollbackParts {
        ple_context: false,
        ..RollbackParts::ALL
    };
    let stale = rolled_back(&model, &prompt, &block, 1, parts);
    assert_eq!(ple_contexts(&stale)[0], vec![9, 21]);
    let (stale_logits, _) = candidate(&model, &prompt, &block, 1, &probes, parts);
    assert_differs(
        "probe 0 logits with a stale n-gram context",
        &stale_logits[0],
        &ref_logits[0],
    );

    // Keeping the EOS is the other half of the contract: the context then
    // holds it and the n-gram hash restarts its segment there, exactly as
    // the reference run that decoded the EOS does.
    let (ref_logits, ref_caches) = reference(&model, &prompt, &block[..2], &probes);
    let (cand_logits, cand_caches) =
        candidate(&model, &prompt, &block, 2, &probes, RollbackParts::ALL);
    assert_eq!(ple_contexts(&cand_caches), ple_contexts(&ref_caches));
    assert_states_match(&ref_caches, &cand_caches);
    assert_probes_close(&ref_logits, &cand_logits);
}

/// The GatedDeltaNet replay and the PLE convolution window each change the
/// probe logits when skipped, and the all-rejected and one-rejected cases
/// match the reference like the mixed one.
#[test]
fn gated_delta_replay_and_ple_window_each_matter_and_every_keep_count_matches() {
    let model = tiny_model(&fixture("model"));
    let prompt = prompt_tokens(12);
    let block = [5, 9, 21, 33];
    let probes = [12, 7];
    for keep in [0, 1, 3] {
        let (ref_logits, ref_caches) = reference(&model, &prompt, &block[..keep as usize], &probes);
        let (cand_logits, cand_caches) =
            candidate(&model, &prompt, &block, keep, &probes, RollbackParts::ALL);
        assert_states_match(&ref_caches, &cand_caches);
        assert_probes_close(&ref_logits, &cand_logits);
    }
    let (ref_logits, _) = reference(&model, &prompt, &block[..1], &probes);
    for (label, parts) in [
        (
            "GatedDeltaNet replay",
            RollbackParts {
                gdn: false,
                ..RollbackParts::ALL
            },
        ),
        (
            "PLE convolution window",
            RollbackParts {
                ple_conv: false,
                ..RollbackParts::ALL
            },
        ),
    ] {
        let (stale_logits, _) = candidate(&model, &prompt, &block, 1, &probes, parts);
        assert_differs(
            &format!("probe 0 logits without the {label}"),
            &stale_logits[0],
            &ref_logits[0],
        );
    }
}

/// The verify block is byte-identical to the single-token chain on the
/// quantized fixture: every component of the captured forward runs in its
/// decode-equivalent form, so the exactness probe the MTP gate runs on the
/// real checkpoint passes here at every width a drafter uses. Quantized
/// rather than float32 weights because a dense float32 matmul takes a
/// different kernel for one row than for several, while the quantized
/// matmul takes `qmv` per row below the batch limit, as the real checkpoint
/// does; the probe's prompt also crosses into the sparse QSA regime.
#[test]
fn block_chain_probe_passes_on_the_quantized_fixture() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let _qmv_wide_off = QmvWideOff::enter();
    let model = quantized_tiny_model(&fixture("model"));
    for block_size in [2, 4, 8] {
        let verdict = model.probe_block_chain_exactness(block_size);
        assert!(
            verdict.is_equal(),
            "block {block_size}: {}",
            verdict.reason()
        );
    }
}

/// Bisect a block-versus-chain difference: every traced stage of a 2-row
/// verify block, row by row, against the two single-row steps. Prints the
/// stages that differ; passes only when none does.
#[test]
fn block_vs_chain_stages_agree_byte_for_byte_on_the_quantized_fixture() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let _qmv_wide_off = QmvWideOff::enter();
    let model = quantized_tiny_model(&fixture("model"));
    let prompt: Vec<i32> = (0..48).map(|i| (i * 7 + 2) % 64).collect();
    let block = [5, 9];
    let bytes = |array: &MlxArray| -> Vec<u8> {
        mlxcel_core::eval(array);
        mlxcel_core::array_to_raw_bytes(array)
    };
    let row = |array: &MlxArray, pos: i32| -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(array);
        let mut start = vec![0; shape.len()];
        let mut stop = shape.clone();
        start[1] = pos;
        stop[1] = pos + 1;
        mlxcel_core::contiguous(&mlxcel_core::slice(array, &start, &stop), false)
    };

    let mut chain: Vec<Vec<(String, UniquePtr<MlxArray>)>> = Vec::new();
    let mut caches = prefill(&model, &prompt);
    for token in block {
        super::trace::start();
        let _ = model.forward_speculative(&ids(&[token]), &mut caches, None);
        chain.push(super::trace::take());
    }
    let mut caches = prefill(&model, &prompt);
    super::trace::start();
    let _ = model.forward_speculative(&ids(&block), &mut caches, None);
    let traced = super::trace::take();
    assert_eq!(traced.len(), chain[0].len(), "stage counts");

    let mut differing = Vec::new();
    for (index, (label, tensor)) in traced.iter().enumerate() {
        let shape = mlxcel_core::array_shape(tensor);
        if shape.len() < 2 || shape[1] != block.len() as i32 {
            continue;
        }
        for (pos, step) in chain.iter().enumerate() {
            let (step_label, step_tensor) = &step[index];
            assert_eq!(step_label, label);
            if bytes(&row(tensor, pos as i32)) != bytes(step_tensor) {
                differing.push(format!("stage {index} {label} at position {pos}"));
            }
        }
    }
    assert!(
        differing.is_empty(),
        "stages that differ from the single-token chain:\n  {}",
        differing.join("\n  ")
    );
}

/// The engine path: a speculative forward inside the lookahead scope is
/// captured on the sequence's log and `rewind_decode_appends` unwinds into it,
/// repeatedly; only the newest forward is kept, a rewind past it is refused
/// and leaves the state alone, and a plain forward commits and clears the log.
#[test]
fn rewind_decode_appends_unwinds_the_newest_forward_and_refuses_beyond_it() {
    let model = tiny_model(&fixture("model"));
    // Nothing unwinds across forwards as long as lookahead stays off; turning
    // it on needs `ROLLBACK_LOG_DEPTH` to cover its speculative appends.
    assert!(!model.supports_decode_lookahead_rewind());
    let prompt = prompt_tokens(12);
    let (block_a, block_b) = ([5, 9, 21], [33, 40, 50]);
    let probes = [12, 7];
    let kept = [5, 9, 21, 33];
    let (ref_logits, _) = reference(&model, &prompt, &kept, &probes);

    let seq = SequenceId::from_raw(7);
    model.prepare_sequence_state(seq);
    let mut none: [mlxcel_core::layers::KVCache; 0] = [];
    let _ = model.forward_with_sequence_id(&ids(&prompt), Some(seq), &mut none, None);
    {
        let _scope = DecodeLookaheadAppendScope::enter();
        let _ = model.forward_with_sequence_id(&ids(&block_a), Some(seq), &mut none, None);
        let _ = model.forward_with_sequence_id(&ids(&block_b), Some(seq), &mut none, None);
    }
    // Block A was evicted: only block B's three positions can be unwound.
    let err = model
        .rewind_decode_appends(seq, 4)
        .expect_err("beyond the newest forward");
    assert!(err.contains("appended 3"), "{err}");
    // Two separate rewinds: the second reaches into the truncated capture.
    model.rewind_decode_appends(seq, 1).expect("first rewind");
    model.rewind_decode_appends(seq, 1).expect("second rewind");
    for (i, probe) in probes.iter().enumerate() {
        let logits = model.forward_with_sequence_id(&ids(&[*probe]), Some(seq), &mut none, None);
        assert_close(&format!("probe {i} logits"), &logits, &ref_logits[i]);
    }
    // The plain probe forwards committed everything before them.
    let err = model
        .rewind_decode_appends(seq, 1)
        .expect_err("nothing speculative left");
    assert!(err.contains("appended 0"), "{err}");
    model.release_sequence_state_by_id(seq);
}

/// The whole prompt-lookup eligibility and verify-round path on the engine:
/// the family is admitted, `Engine::verify` captures, `unwind_appends`
/// rewinds through the model, and the next verify matches the reference.
#[test]
fn engine_verify_rounds_admit_the_family_and_rewind_through_the_model() {
    let model = tiny_model(&fixture("model"));
    let prompt = prompt_tokens(12);
    let block = [5, 9, 21, 33];
    let probes = [12, 7];
    let (ref_logits, _) = reference(&model, &prompt, &block[..1], &probes);
    assert_eq!(prompt_lookup_unsupported_reason(&model), None);

    let mut engine = Engine::with_capacity(model, 1);
    let id = engine.open(SequenceSpec::default()).expect("open");
    let drafter = PromptLookupDrafter::new(PromptLookupConfig::default());
    assert_eq!(
        engine.verify_rounds_unsupported(id, &SamplingConfig::greedy(), &drafter),
        None
    );
    let logits = engine.verify(id, &ids(&prompt)).expect("prefill");
    mlxcel_core::eval(&logits);
    engine.commit_appends(id, prompt.len() as i32);

    let logits = engine.verify(id, &ids(&block)).expect("verify block");
    mlxcel_core::eval(&logits);
    engine
        .unwind_appends(id, 3)
        .expect("unwind the rejected tail");
    engine.commit_appends(id, 1);
    for (i, probe) in probes.iter().enumerate() {
        let logits = engine.verify(id, &ids(&[*probe])).expect("probe");
        engine.commit_appends(id, 1);
        assert_close(&format!("probe {i} logits"), &logits, &ref_logits[i]);
    }
}

/// An MTP prefill longer than the prefill chunk runs the same pieces classic
/// prefill runs: the last logits match a chunked forward byte for byte and
/// the streams still cover every prompt position for the drafter.
#[test]
fn mtp_prefill_follows_the_classic_chunk_plan() {
    let model = tiny_model(&fixture("model"));
    let prompt = prompt_tokens(39);
    let seq = SequenceId::from_raw(11);
    model.prepare_sequence_state(seq);
    let (logits, streams) =
        model.forward_prefill_with_streams_for_sequence(&ids(&prompt), Some(seq), 16);
    assert_eq!(mlxcel_core::array_shape(&streams)[1], prompt.len() as i32);

    let mut caches = model.make_internal_caches();
    let mut hidden = None;
    for piece in prompt.chunks(16) {
        hidden = Some(model.forward_hidden(&ids(piece), None, &mut caches, None));
    }
    let hidden = hidden.expect("a prompt");
    let shape = mlxcel_core::array_shape(&hidden);
    let last = shape[1] - 1;
    let chunked_last = model.logits(&mlxcel_core::slice(
        &hidden,
        &[0, last, 0],
        &[1, last + 1, shape[2]],
    ));
    let bytes = |array: &MlxArray| {
        mlxcel_core::eval(array);
        mlxcel_core::array_to_raw_bytes(array)
    };
    assert_eq!(bytes(&logits), bytes(&chunked_last));
    model.release_sequence_state_by_id(seq);
}

/// The adapter prefills in the chunk the server configures, not the process
/// default: the classic decode it must match byte for byte follows the
/// server's `prefill_chunk_size`, and `0` is a single call.
#[test]
fn mtp_adapter_prefill_runs_in_the_configured_chunk_size() {
    let model = tiny_model(&fixture("model"));
    let prompt = prompt_tokens(39);
    for (seq, chunk, expected_pieces) in [(21, 16, 3), (22, 39, 1), (23, 0, 1)] {
        let seq = SequenceId::from_raw(seq);
        model.prepare_sequence_state(seq);
        let adapter =
            Qwen4ExpMtpTargetAdapter::new(&model, Some(seq)).with_prefill_chunk_size(chunk);
        super::trace::start();
        let (_, output, _) = adapter.prefill_and_seed(
            &prompt,
            &SamplingConfig::greedy(),
            &[],
            &LogprobsConfig::default(),
        );
        let pieces = super::trace::take()
            .iter()
            .filter(|(label, _)| label == "prefill piece")
            .count();
        assert_eq!(pieces, expected_pieces, "chunk {chunk}");
        assert_eq!(output.kv_offset, prompt.len(), "chunk {chunk}");
        model.release_sequence_state_by_id(seq);
    }
}
