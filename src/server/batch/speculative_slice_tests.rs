// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Unit tests for the tick-cooperative B=1 MTP slice driver
//! ([`super::speculative_slice`], issue #734).
//!
//! ## What these tests pin
//!
//! - **Stream identity (acceptance)**: for the same target/drafter script,
//!   the slice-driven path (one round per `step_slice_session` call, the
//!   generator reconstructed for every call, exactly the per-tick
//!   lifecycle the scheduler imposes) commits the identical token stream,
//!   emits the identical client events, and produces the identical
//!   acceptance-summary counts as the run-to-completion burst composition
//!   (`MtpGenerator::generate` + lump streaming).
//! - **Per-round streaming**: each slice commits its round's accepted
//!   tokens immediately: the committed stream grows round by round, not
//!   in one lump at finalize (the documented lump behaviour of the legacy
//!   burst).
//! - **Cross-slice cancellation**: a client disconnect between slices
//!   finishes the session at the next slice without another draft/verify.
//! - **Probe rounds (#736)**: probe slices emit one greedy token each and
//!   stay out of the acceptance aggregates, exactly as in the first rounds
//!   of a legacy burst.
//!
//! The tick-arbitration policy (`slice_takes_tick`, strict round/classic
//! alternation with bounded gaps) is pinned in
//! `speculative_slice.rs`'s own test module; the mixed
//! speculative+classic interleaving at the real scheduler loop needs a
//! real model and belongs to the on-hardware E2E lane
//! (`scripts/bench_serving_concurrency.py --concurrency 2`).

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::speculative::mtp::MtpGenerator;
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, from_slice_f32};

use crate::server::batch::sequence::{RequestPriority, SequenceInfo, SequenceState};
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::thinking_budget::ThinkingState;

use super::speculative_burst::{begin_burst_stream, finalize_burst_stream, stream_burst_tokens};
use super::speculative_slice::{MtpSliceJob, begin_slice_session, step_slice_session};

// =============================================================================
// Mocks: mirror `speculative_burst_tests.rs` / the mlxcel-core mtp tests,
// inlined here because test modules cannot share private items.
// =============================================================================

fn dummy_tensor() -> UniquePtr<MlxArray> {
    from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 4])
}

struct MockMtpTarget {
    first_bonus: i32,
    scripted_target_tokens: RefCell<Vec<Vec<i32>>>,
    eos: Vec<i32>,
    cumulative_offset: RefCell<usize>,
}

impl MockMtpTarget {
    fn new(scripted: Vec<Vec<i32>>, eos: Vec<i32>) -> Self {
        Self {
            first_bonus: 100,
            scripted_target_tokens: RefCell::new(scripted),
            eos,
            cumulative_offset: RefCell::new(0),
        }
    }

    fn build_verify_output(&self, advance: usize) -> MtpVerifyOutput {
        *self.cumulative_offset.borrow_mut() += advance;
        let kv_offset = *self.cumulative_offset.borrow();
        MtpVerifyOutput {
            next_hidden: dummy_tensor(),
            next_shared_kv: vec![
                dummy_tensor(),
                dummy_tensor(),
                dummy_tensor(),
                dummy_tensor(),
            ],
            kv_offset,
            bonus_position: kv_offset,
        }
    }
}

impl MtpTarget for MockMtpTarget {
    fn prefill_and_seed(
        &self,
        _prompt_tokens: &[i32],
        _sampler: &SamplingConfig,
        _token_history: &[i32],
        _logprobs_config: &LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        (self.first_bonus, self.build_verify_output(1), None)
    }

    fn embed_token(&self, _token_id: i32) -> UniquePtr<MlxArray> {
        dummy_tensor()
    }

    fn verify_forward(
        &self,
        _verify_input: &[i32],
        _sampler: &SamplingConfig,
        _logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput {
        let target_tokens = {
            let mut q = self.scripted_target_tokens.borrow_mut();
            if q.len() > 1 {
                q.remove(0)
            } else if !q.is_empty() {
                q[0].clone()
            } else {
                vec![0]
            }
        };
        VerifyForwardOutput {
            target_tokens,
            target_logprobs: None,
            captured: VerifyCaptured {
                tensors: Vec::new(),
                scalars: Vec::new(),
            },
        }
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        _block_size: usize,
        _captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.build_verify_output(accepted + 1)
    }

    fn num_layers(&self) -> usize {
        4
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos.clone()
    }
}

/// Scripted drafter. Records `set_shared_kv` calls so the slice tests can
/// pin the resume re-arm (one arm at the start of every non-probe round,
/// which is what makes an un-reset drafter resumable across ticks).
struct ScriptedDrafter {
    scripted_draft_tokens: RefCell<Vec<Vec<i32>>>,
    set_shared_kv_calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl ScriptedDrafter {
    fn new(scripted: Vec<Vec<i32>>, calls: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self {
            scripted_draft_tokens: RefCell::new(scripted),
            set_shared_kv_calls: calls,
        }
    }
}

impl Drafter for ScriptedDrafter {
    fn bind(
        &mut self,
        _target: &dyn mlxcel_core::generate::LanguageModel,
    ) -> Result<(), DrafterError> {
        Ok(())
    }

    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        _kv_offset: usize,
        _position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        self.set_shared_kv_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn draft_block(
        &mut self,
        _last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        let mut q = self.scripted_draft_tokens.borrow_mut();
        if q.len() > 1 {
            Ok(q.remove(0))
        } else if !q.is_empty() {
            Ok(q[0].clone())
        } else {
            Ok(vec![0; block_size.saturating_sub(1)])
        }
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

/// Build a minimal in-flight `SequenceInfo` (Prefilling: the slice path
/// transitions Queued -> Prefilling before slice 0, and the stream
/// finalize transitions Prefilling -> Finished).
fn make_slice_sequence(max_tokens: usize) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);
    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(77),
        state: SequenceState::Prefilling,
        prompt_tokens,
        sampling: SamplingConfig::greedy(),
        max_tokens,
        eos_token_ids: Vec::new(),
        priority: RequestPriority::Normal,
        logprobs_config: LogprobsConfig::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset: 0,
        already_cached_tokens: 0,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: Some(Instant::now()),
        first_token_time: None,
        token_history: Vec::new(),
        sampler_state: None,
        merged_eos: Vec::new(),
        thinking: ThinkingState::disabled(),
        structured: None,
    };
    (seq, rx)
}

/// Two full-accept rounds terminated by an EOS (9999) in round 2.
/// Committed stream: [100, 10, 11, 12, 13, 20, 21, 22] (the EOS itself is
/// classified, not committed, same as the classic decode path).
fn two_round_script() -> (Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<i32>) {
    (
        vec![vec![10, 11, 12, 13], vec![20, 21, 22, 9999]],
        vec![vec![10, 11, 12], vec![20, 21, 22]],
        vec![9999],
    )
}

/// Render the channel's events into a comparable form. The stub tokenizer
/// decodes to empty text, so most runs see only the `Done` event; the
/// comparison still pins that the two paths emit the SAME events in the
/// SAME order.
fn drain_events(rx: &mpsc::Receiver<GenerateEvent>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        out.push(match ev {
            GenerateEvent::Token(t) => ("token".to_string(), t),
            GenerateEvent::TokenWithLogprobs(t, _) => ("token_lp".to_string(), t),
            GenerateEvent::Done(_) => ("done".to_string(), String::new()),
            GenerateEvent::Error(e) => panic!("unexpected error event: {e}"),
        });
    }
    out
}

/// Drive the run-to-completion reference: `MtpGenerator::generate` over the
/// script, then the lump-streaming composition the legacy
/// `finalize_burst_success` performs (begin + one stream call + finalize).
fn run_to_completion_reference(
    max_tokens: usize,
) -> (
    Vec<i32>,
    mlxcel_core::speculative::mtp::MtpAcceptanceSummary,
    Vec<(String, String)>,
    super::speculative_burst::FinalizeOutcome,
) {
    let (target_script, draft_script, eos) = two_round_script();
    let target = MockMtpTarget::new(target_script, eos.clone());
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let drafter: Box<dyn Drafter> = Box::new(ScriptedDrafter::new(draft_script, calls));
    let mut generator = MtpGenerator::new(target, drafter, 4);
    let (mut seq, rx) = make_slice_sequence(max_tokens);
    let token_history: Vec<i32> = Vec::new();
    let (tokens, logprobs, _stats) = generator.generate(
        &seq.prompt_tokens.clone(),
        max_tokens,
        &seq.sampling.clone(),
        &token_history,
        &AtomicBool::new(false),
        &seq.logprobs_config.clone(),
    );
    let summary = generator.last_acceptance().expect("summary");
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let mut stream = begin_burst_stream(eos, &seq);
    stream_burst_tokens(&tokenizer, &mut seq, &mut stream, &tokens, &logprobs);
    let outcome = finalize_burst_stream(&tokenizer, seq, &stream);
    let events = drain_events(&rx);
    (tokens, summary, events, outcome)
}

/// Drive the same script through the tick-cooperative slice driver,
/// returning the job's finalize outcome plus the committed-token snapshot
/// AFTER EVERY SLICE (the per-round streaming evidence).
fn run_slices(
    max_tokens: usize,
    cancel_after_slice: Option<usize>,
) -> (
    MtpSliceJob,
    Vec<Vec<i32>>,
    mpsc::Receiver<GenerateEvent>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    let (target_script, draft_script, eos) = two_round_script();
    let target = MockMtpTarget::new(target_script.clone(), eos.clone());
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let drafter: Box<dyn Drafter> = Box::new(ScriptedDrafter::new(draft_script, calls.clone()));
    let (seq, rx) = make_slice_sequence(max_tokens);
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let token_history: Vec<i32> = Vec::new();

    // Slice 0: prefill + seed + first bonus. The target is consumed per
    // slice, exactly what the scheduler does with the per-tick adapter.
    let mut job = begin_slice_session(
        target,
        drafter,
        seq,
        &tokenizer,
        eos.clone(),
        /* block_size */ 4,
        /* profile_probe_rounds */ 0,
        /* prefill_start_offset */ 0,
        &token_history,
    );
    let mut committed_after_each_slice = vec![job.seq.generated_tokens.clone()];

    // The mock target's script position must survive across slices the
    // way the real model's per-sequence KV slot does. `step_slice_session`
    // takes the target by value (the scheduler reconstructs the borrowing
    // adapter per tick), so pass a `&MockMtpTarget` view over one
    // RefCell-backed script; the blanket `impl MtpTarget for &T` mirrors
    // exactly the "stateless view over persistent model state" shape of
    // the real per-tick adapters. Slice 0 runs no verify rounds, so the
    // stepping script starts at round 1.
    let stepping_target = MockMtpTarget::new(target_script, eos);
    let mut slice_index = 0usize;
    while !job.finished() {
        slice_index += 1;
        if cancel_after_slice == Some(slice_index - 1) {
            job.seq.cancelled.store(true, Ordering::Relaxed);
        }
        step_slice_session(&stepping_target, &mut job, &tokenizer);
        committed_after_each_slice.push(job.seq.generated_tokens.clone());
        assert!(slice_index < 64, "runaway slice loop");
    }
    (job, committed_after_each_slice, rx, calls)
}

#[test]
fn slice_driver_commits_identical_stream_to_run_to_completion() {
    let max_tokens = 64;
    let (ref_tokens, ref_summary, ref_events, ref_outcome) =
        run_to_completion_reference(max_tokens);
    let (mut job, _committed, rx, _calls) = run_slices(max_tokens, None);

    // The generator-level emitted stream is [100, 10..13, 20..22, 9999];
    // the committed stream excludes the EOS. Compare via the finalize
    // outcome (committed tokens + finish classification).
    let finish = job.finish.take().expect("job finished");
    let summary = finish.summary.expect("summary present");
    assert_eq!(summary.rounds, ref_summary.rounds);
    assert_eq!(summary.proposed_tokens, ref_summary.proposed_tokens);
    assert_eq!(
        summary.accepted_draft_tokens,
        ref_summary.accepted_draft_tokens
    );
    assert_eq!(summary.probe_rounds, ref_summary.probe_rounds);

    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let outcome = finalize_burst_stream(&tokenizer, job.seq, &job.stream);
    assert_eq!(
        outcome.generated_tokens, ref_outcome.generated_tokens,
        "slice-driven committed stream must equal the run-to-completion stream token-for-token"
    );
    assert_eq!(outcome.tokens_generated, ref_outcome.tokens_generated);
    assert_eq!(outcome.healthy_finish, ref_outcome.healthy_finish);
    assert_eq!(
        outcome.generated_tokens,
        vec![100, 10, 11, 12, 13, 20, 21, 22],
        "hand-computed committed stream"
    );
    // Sanity on the reference emitted stream (incl. the EOS the stream
    // layer classifies away).
    assert_eq!(ref_tokens, vec![100, 10, 11, 12, 13, 20, 21, 22, 9999]);

    // Client-visible events must be pairwise identical.
    let events = drain_events(&rx);
    assert_eq!(
        events, ref_events,
        "slice-driven event stream must equal the run-to-completion event stream"
    );
}

#[test]
fn slice_driver_streams_tokens_per_round_not_in_a_lump() {
    let (job, committed, _rx, _calls) = run_slices(64, None);
    assert!(job.finished());
    // Slice 0 commits the first bonus immediately.
    assert_eq!(committed[0], vec![100]);
    // Round 1 commits its 4 walk tokens in the SAME tick it ran.
    assert_eq!(committed[1], vec![100, 10, 11, 12, 13]);
    // Round 2 commits up to (and excluding) the EOS.
    assert_eq!(committed[2], vec![100, 10, 11, 12, 13, 20, 21, 22]);
    assert_eq!(job.slices, 3, "slice 0 + two rounds");
    assert!(job.max_slice_wall_ms <= job.total_slice_wall_ms);
}

#[test]
fn slice_driver_cancel_between_slices_finishes_without_another_round() {
    // Cancel after slice 1 (round 1). The next slice must finish without
    // drafting/verifying again: the committed stream stays at round 1's.
    let (mut job, committed, _rx, _calls) = run_slices(64, Some(1));
    assert!(job.finished());
    assert_eq!(
        committed.last().expect("slices ran"),
        &vec![100, 10, 11, 12, 13],
        "no tokens after the cancellation slice"
    );
    let finish = job.finish.take().expect("finished");
    assert_eq!(
        finish.summary.expect("summary").rounds,
        1,
        "only round 1 ran before the cancel"
    );
    // Early bail (not EOS, not budget) classifies as a clean Stop, same
    // as the legacy burst's early-bail classification.
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let outcome = finalize_burst_stream(&tokenizer, job.seq, &job.stream);
    assert!(outcome.healthy_finish);
}

#[test]
fn slice_driver_rearms_drafter_once_per_round() {
    // Two rounds ran; the resume re-arm at the top of each round is what
    // lets the un-reset drafter continue across ticks. Exactly one
    // `set_shared_kv` per (non-probe) round, the same count the legacy
    // run-to-completion loop performs for an EOS-terminated run (seed arm
    // + one arm after each non-terminal round).
    let (_job, _committed, _rx, calls) = run_slices(64, None);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        2,
        "one shared-KV arm per executed round"
    );
}

#[test]
fn slice_driver_probe_rounds_run_in_first_slices() {
    // [#736] Probe rounds run in the first slices exactly as they run in
    // the first rounds of a legacy burst: one greedy token each, kept out
    // of the acceptance aggregates.
    let target_script = vec![vec![10], vec![11], vec![20, 21, 22, 23]];
    let draft_script = vec![vec![20, 21, 22]];
    let target = MockMtpTarget::new(target_script.clone(), vec![]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let drafter: Box<dyn Drafter> = Box::new(ScriptedDrafter::new(draft_script, calls.clone()));
    let (seq, _rx) = make_slice_sequence(7);
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();

    let mut job = begin_slice_session(
        target,
        drafter,
        seq,
        &tokenizer,
        Vec::new(),
        4,
        /* profile_probe_rounds */ 2,
        0,
        &[],
    );
    let stepping_target = MockMtpTarget::new(target_script, vec![]);
    let mut committed = vec![job.seq.generated_tokens.clone()];
    while !job.finished() {
        step_slice_session(&stepping_target, &mut job, &tokenizer);
        committed.push(job.seq.generated_tokens.clone());
        assert!(committed.len() < 64, "runaway");
    }
    assert_eq!(committed[0], vec![100]);
    assert_eq!(committed[1], vec![100, 10], "probe slice 1: one token");
    assert_eq!(committed[2], vec![100, 10, 11], "probe slice 2: one token");
    assert_eq!(
        committed[3],
        vec![100, 10, 11, 20, 21, 22, 23],
        "real round after the probes"
    );
    let summary = job
        .finish
        .take()
        .expect("finished")
        .summary
        .expect("summary");
    assert_eq!(summary.probe_rounds, 2);
    assert_eq!(summary.rounds, 1);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "probe rounds never arm the drafter; only the real round does"
    );
}
