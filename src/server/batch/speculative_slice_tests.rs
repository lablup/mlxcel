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
use super::speculative_slice::{
    MTP_SLICE_GRANT_SKIP_CAP, MtpSliceJob, begin_slice_session, next_grant_index,
    slice_grant_expired, step_slice_session,
};

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

    /// Override the seed bonus so two concurrent mock requests emit
    /// disjoint token ranges (rotation tests, issue #746).
    fn with_first_bonus(mut self, first_bonus: i32) -> Self {
        self.first_bonus = first_bonus;
        self
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
    make_slice_sequence_with_id(max_tokens, 77)
}

fn make_slice_sequence_with_id(
    max_tokens: usize,
    raw_id: u64,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);
    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(raw_id),
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

// =============================================================================
// Slot-grant rotation (issue #746): the scheduler's park / promote protocol
// driven over the REAL policy functions (`slice_grant_expired`,
// `next_grant_index`) and the real slice driver, with ONE shared drafter
// handle rotating across jobs exactly as the worker's single drafter does.
// The mixed speculative+classic tick arbitration is pinned separately in
// `speculative_slice.rs`'s own test module (`slice_takes_tick` is common to
// both phases), and the real-model interleaving stays in the on-hardware
// E2E lane, as for #734.
// =============================================================================

/// Deterministic drafter: proposals are a pure function of the bonus token
/// (`bonus+1, bonus+2, ...`), the shape of the real MTP assistant whose
/// draft depends only on the per-round `set_shared_kv` arm plus the bonus
/// and hidden inputs. One instance is SHARED across all jobs of a rotation
/// run, so any cross-session state leak the rotation failed to overwrite
/// would corrupt the streams and break parity with the isolated runs. Like
/// the real Gemma 4 assistant drafter, it does not override
/// `Drafter::reset` (the park-boundary `return_drafter` reset is the trait
/// default no-op).
struct DeterministicDrafter;

impl Drafter for DeterministicDrafter {
    fn bind(
        &mut self,
        _target: &dyn mlxcel_core::generate::LanguageModel,
    ) -> Result<(), DrafterError> {
        Ok(())
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        Ok((1..block_size as i32).map(|i| last_bonus + i).collect())
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

/// One request spec for the rotation harness. The target script is written
/// against the deterministic drafter: a fully accepted round's row is
/// `[bonus+1, bonus+2, bonus+3, next_bonus]`, and a terminal row leads
/// with the EOS id (zero drafts accepted, the correction token is the
/// EOS).
#[derive(Clone)]
struct RotSpec {
    tag: char,
    first_bonus: i32,
    target_script: Vec<Vec<i32>>,
    eos: Vec<i32>,
    lane: RequestPriority,
}

fn spec_a() -> RotSpec {
    RotSpec {
        tag: 'A',
        first_bonus: 100,
        target_script: vec![
            vec![101, 102, 103, 104],
            vec![105, 106, 107, 108],
            vec![9999, 0, 0, 0],
        ],
        eos: vec![9999],
        lane: RequestPriority::Normal,
    }
}

fn spec_b() -> RotSpec {
    RotSpec {
        tag: 'B',
        first_bonus: 500,
        target_script: vec![vec![501, 502, 503, 504], vec![8888, 0, 0, 0]],
        eos: vec![8888],
        lane: RequestPriority::Normal,
    }
}

/// Build a spec of `full_rounds` fully accepted rounds (each committing
/// 4 tokens against the deterministic drafter's `bonus+1..bonus+3`
/// proposals) followed by an EOS round, in the given priority lane.
fn chain_spec(
    tag: char,
    first_bonus: i32,
    lane: RequestPriority,
    full_rounds: usize,
    eos: i32,
) -> RotSpec {
    let mut script = Vec::with_capacity(full_rounds + 1);
    let mut bonus = first_bonus;
    for _ in 0..full_rounds {
        script.push(vec![bonus + 1, bonus + 2, bonus + 3, bonus + 4]);
        bonus += 4;
    }
    script.push(vec![eos, 0, 0, 0]);
    RotSpec {
        tag,
        first_bonus,
        target_script: script,
        eos: vec![eos],
        lane,
    }
}

/// Per-request terminal accounting produced by the harness.
struct RotOutcome {
    committed: Vec<i32>,
    healthy_finish: bool,
    rounds: usize,
    slices: usize,
}

struct RotJobState {
    spec: RotSpec,
    /// `Some` until slice 0 runs (the waiter phase).
    seq: Option<SequenceInfo>,
    _rx: mpsc::Receiver<GenerateEvent>,
    /// `Some` after slice 0 until finalize.
    job: Option<MtpSliceJob>,
    /// Persistent per-request stepping target: emulates the model's
    /// per-sequence KV slot surviving parks (same pattern as
    /// `run_slices`).
    stepping_target: Option<MockMtpTarget>,
    outcome: Option<RotOutcome>,
    aborted: bool,
    /// The waiter declined at promotion time (the B=1 verdict re-check)
    /// and was routed to the classic prefill path instead of starting
    /// slice 0.
    routed_to_classic: bool,
}

fn finalize_rot_job(
    mut job: MtpSliceJob,
    tokenizer: &crate::tokenizer::MlxcelTokenizer,
) -> RotOutcome {
    let slices = job.slices;
    let finish = job.finish.take().expect("job finished");
    let rounds = finish.summary.map(|s| s.rounds).unwrap_or(0);
    let outcome = finalize_burst_stream(tokenizer, job.seq, &job.stream);
    RotOutcome {
        committed: outcome.generated_tokens,
        healthy_finish: outcome.healthy_finish,
        rounds,
        slices,
    }
}

/// Emulate the scheduler's slot-grant protocol (issue #746) over the real
/// policy functions and the real slice driver:
///
/// - spec 0 is admitted directly (the slot was free); every later spec is
///   a WAITER in the backlog ring.
/// - one speculative action per tick: an active job's round, a promoted
///   parked job's round (promotion is bookkeeping), or a promoted
///   waiter's slice 0.
/// - at a round boundary with the grant spent and the backlog non-empty
///   (`slice_grant_expired`), the active job parks: its drafter is
///   released (the trait-default no-op reset, as through
///   `WorkerDrafterSlot::return_drafter`) and handed to the next grantee
///   chosen by `next_grant_index`.
/// - a cancelled waiter aborts at promotion with bookkeeping only; a
///   cancelled parked job is promoted normally and finishes at its next
///   step.
/// - a waiter's promotion re-checks the B=1 verdict (mirroring the
///   scheduler's `mtp_b1_should_run` re-check for an adaptive policy that
///   settled to Decline while the request waited); a declined waiter is
///   routed to classic prefill with bookkeeping only and the loop tries
///   the next grantee in the same tick.
/// - a waiter promotion whose slice 0 already spends the grant budget
///   (`MLXCEL_MTP_SLICE_GRANT_ROUNDS=1`) parks the job immediately, so
///   an admission grant cannot hold a free extra round beyond the budget.
///
/// `cancel_waiter` / `cancel_on_first_park` cancel the given spec index
/// up front / when it first parks; `decline_waiter_at_promotion` fails
/// the given spec index's promotion-time verdict re-check.
fn run_rotation_harness(
    budget: usize,
    specs: Vec<RotSpec>,
    cancel_waiter: Option<usize>,
    cancel_on_first_park: Option<usize>,
    decline_waiter_at_promotion: Option<usize>,
) -> (Vec<RotJobState>, Vec<(char, usize)>) {
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let mut free_drafter: Option<Box<dyn Drafter>> = Some(Box::new(DeterministicDrafter));
    let mut states: Vec<RotJobState> = specs
        .into_iter()
        .enumerate()
        .map(|(i, spec)| {
            let (mut seq, rx) = make_slice_sequence_with_id(64, 200 + i as u64);
            seq.priority = spec.lane;
            if cancel_waiter == Some(i) {
                seq.cancelled.store(true, Ordering::Relaxed);
            }
            RotJobState {
                spec,
                seq: Some(seq),
                _rx: rx,
                job: None,
                stepping_target: None,
                outcome: None,
                aborted: false,
                routed_to_classic: false,
            }
        })
        .collect();

    fn start_slice0(
        state: &mut RotJobState,
        drafter: Box<dyn Drafter>,
        tokenizer: &crate::tokenizer::MlxcelTokenizer,
    ) {
        let seq = state.seq.take().expect("slice 0 needs the parked sequence");
        let target = MockMtpTarget::new(state.spec.target_script.clone(), state.spec.eos.clone())
            .with_first_bonus(state.spec.first_bonus);
        let job = begin_slice_session(
            target,
            drafter,
            seq,
            tokenizer,
            state.spec.eos.clone(),
            /* block_size */ 4,
            /* profile_probe_rounds */ 0,
            /* prefill_start_offset */ 0,
            &[],
        );
        state.stepping_target = Some(
            MockMtpTarget::new(state.spec.target_script.clone(), state.spec.eos.clone())
                .with_first_bonus(state.spec.first_bonus),
        );
        state.job = Some(job);
    }

    let mut backlog: std::collections::VecDeque<usize> = (1..states.len()).collect();
    // Per-spec skip counters mirroring `SliceBacklogEntry::skipped_grants`
    // (a fresh ring entry starts at 0; every grant decision increments
    // the non-selected ring members; reset when granted or re-parked).
    let mut skip_counts: Vec<usize> = vec![0; states.len()];
    let mut active: Option<usize> = None;
    let mut grant_log: Vec<(char, usize)> = Vec::new();

    // Direct admission of spec 0 (mirrors `start_mtp_slice_b1` on a free
    // slot): slice 0 runs and opens the first grant. No immediate-expiry
    // check here: in the real scheduler a direct admission always sees an
    // EMPTY backlog (`try_speculative_burst` parks or declines arrivals
    // while it is non-empty), so `slice_grant_expired(1, budget, false)`
    // is always false; the harness's upfront backlog models requests that
    // arrive DURING this first grant.
    start_slice0(
        &mut states[0],
        free_drafter.take().expect("drafter free at start"),
        &tokenizer,
    );
    let mut grant_slices = 1usize;
    if states[0].job.as_ref().expect("job started").finished() {
        grant_log.push((states[0].spec.tag, grant_slices));
        let mut job = states[0].job.take().expect("job");
        free_drafter = job.take_drafter();
        states[0].outcome = Some(finalize_rot_job(job, &tokenizer));
    } else {
        active = Some(0);
    }

    for _tick in 0..200 {
        if active.is_none() {
            // Promotion, mirroring `promote_next_speculative_grantee`.
            let mut tick_consumed = false;
            loop {
                let entries: Vec<(RequestPriority, usize)> = backlog
                    .iter()
                    .map(|&i| {
                        let lane = match &states[i].job {
                            Some(job) => job.seq.priority,
                            None => {
                                states[i]
                                    .seq
                                    .as_ref()
                                    .expect("waiter holds a sequence")
                                    .priority
                            }
                        };
                        (lane, skip_counts[i])
                    })
                    .collect();
                let Some(pos) = next_grant_index(&entries) else {
                    break;
                };
                // Mirror `pop_next_speculative_grantee`: every grant
                // decision increments the skip counter of every
                // non-selected ring entry; the selected entry leaves
                // the ring with its counter reset.
                for (ring_pos, &j) in backlog.iter().enumerate() {
                    if ring_pos != pos {
                        skip_counts[j] += 1;
                    }
                }
                let i = backlog.remove(pos).expect("index from next_grant_index");
                skip_counts[i] = 0;
                if states[i].job.is_none() {
                    // Waiter.
                    let cancelled = states[i]
                        .seq
                        .as_ref()
                        .expect("waiter holds a sequence")
                        .cancelled
                        .load(Ordering::Relaxed);
                    if cancelled {
                        // Bookkeeping-only abort; try the next grantee in
                        // the same tick, as the scheduler does.
                        states[i].seq = None;
                        states[i].aborted = true;
                        continue;
                    }
                    if decline_waiter_at_promotion == Some(i) {
                        // Mirror of the scheduler's promotion-time
                        // `mtp_b1_should_run` re-check: an adaptive
                        // policy that settled to Decline while the
                        // request waited routes the waiter to classic
                        // prefill with bookkeeping only, and the loop
                        // tries the next grantee in the same tick.
                        states[i].seq = None;
                        states[i].routed_to_classic = true;
                        continue;
                    }
                    let drafter = free_drafter.take().expect("drafter free at promotion");
                    start_slice0(&mut states[i], drafter, &tokenizer);
                    grant_slices = 1;
                    if states[i].job.as_ref().expect("job started").finished() {
                        grant_log.push((states[i].spec.tag, grant_slices));
                        let mut job = states[i].job.take().expect("job");
                        free_drafter = job.take_drafter();
                        states[i].outcome = Some(finalize_rot_job(job, &tokenizer));
                    } else if slice_grant_expired(grant_slices, budget, !backlog.is_empty()) {
                        // Budget of 1 with other grantees waiting: slice 0
                        // already spent the whole grant, so the job parks
                        // immediately instead of holding a free extra
                        // round (mirrors `start_mtp_slice_b1`'s install
                        // site).
                        grant_log.push((states[i].spec.tag, grant_slices));
                        let job = states[i].job.as_mut().expect("job");
                        free_drafter =
                            Some(job.take_drafter().expect("drafter held after slice 0"));
                        // Re-parking enters the ring as a fresh entry.
                        skip_counts[i] = 0;
                        backlog.push_back(i);
                    } else {
                        active = Some(i);
                    }
                    // A waiter's slice 0 is the tick's one action.
                    tick_consumed = true;
                    break;
                }
                // Parked job: re-attach the drafter; its round runs this
                // tick below.
                let drafter = free_drafter.take().expect("drafter free at promotion");
                states[i]
                    .job
                    .as_mut()
                    .expect("parked job present")
                    .attach_drafter(drafter);
                grant_slices = 0;
                active = Some(i);
                break;
            }
            if tick_consumed {
                continue;
            }
            if active.is_none() {
                // Backlog drained with no active job: the run is over.
                break;
            }
        }

        // Run exactly one round of the active job.
        let i = active.expect("active job");
        {
            let state = &mut states[i];
            let stepping = state
                .stepping_target
                .as_ref()
                .expect("stepping target present");
            let job = state.job.as_mut().expect("active job present");
            step_slice_session(stepping, job, &tokenizer);
        }
        grant_slices += 1;
        let finished = states[i].job.as_ref().expect("job").finished();
        if finished {
            grant_log.push((states[i].spec.tag, grant_slices));
            let mut job = states[i].job.take().expect("job");
            free_drafter = job.take_drafter();
            states[i].outcome = Some(finalize_rot_job(job, &tokenizer));
            active = None;
        } else if slice_grant_expired(grant_slices, budget, !backlog.is_empty()) {
            // Park at the round boundary: release the drafter for the
            // next grantee (no-op reset, as through the worker slot) and
            // rejoin the ring.
            grant_log.push((states[i].spec.tag, grant_slices));
            let job = states[i].job.as_mut().expect("job");
            free_drafter = Some(job.take_drafter().expect("drafter held while active"));
            if cancel_on_first_park == Some(i) {
                job.seq.cancelled.store(true, Ordering::Relaxed);
            }
            // Re-parking enters the ring as a fresh entry.
            skip_counts[i] = 0;
            backlog.push_back(i);
            active = None;
        }
    }

    (states, grant_log)
}

/// Reference: each spec driven alone through the same harness (a
/// single-job run never rotates, so this is exactly the #734 behavior).
fn isolated_outcome(spec: RotSpec) -> RotOutcome {
    let (mut states, _log) = run_rotation_harness(usize::MAX, vec![spec], None, None, None);
    states.remove(0).outcome.expect("isolated run finishes")
}

/// AC1 + AC2 + AC5 (issue #746): two long concurrent speculative
/// requests BOTH receive speculative rounds, the slot rotates in grants
/// of at most the budget, and each request's committed stream is
/// byte-identical to its isolated single-request run (greedy parity
/// across rotation).
#[test]
fn rotation_shares_the_slot_and_preserves_stream_parity() {
    let iso_a = isolated_outcome(spec_a());
    let iso_b = isolated_outcome(spec_b());
    assert_eq!(
        iso_a.committed,
        vec![100, 101, 102, 103, 104, 105, 106, 107, 108]
    );
    assert_eq!(iso_b.committed, vec![500, 501, 502, 503, 504]);

    let budget = 2;
    let (states, grant_log) =
        run_rotation_harness(budget, vec![spec_a(), spec_b()], None, None, None);
    let a = states[0].outcome.as_ref().expect("A finished");
    let b = states[1].outcome.as_ref().expect("B finished");

    // Both requests were speculatively accelerated (not one speculative
    // plus one permanently classic).
    assert!(
        a.rounds >= 2 && b.rounds >= 2,
        "both must run speculative rounds"
    );
    assert_eq!(a.rounds, iso_a.rounds);
    assert_eq!(b.rounds, iso_b.rounds);
    assert_eq!(a.slices, iso_a.slices);
    assert_eq!(b.slices, iso_b.slices);

    // Greedy parity: byte-identical committed streams per request.
    assert_eq!(
        a.committed, iso_a.committed,
        "A's stream must survive rotation"
    );
    assert_eq!(
        b.committed, iso_b.committed,
        "B's stream must survive rotation"
    );
    assert!(a.healthy_finish && b.healthy_finish);

    // The grant schedule alternates in bounded grants: A's admission
    // grant (slice 0 + round 1), B's admission grant (slice 0 + round
    // 1), A's resumed grant (rounds 2 and 3, finishing), B's resumed
    // grant (round 2, finishing).
    assert_eq!(grant_log, vec![('A', 2), ('B', 2), ('A', 2), ('B', 1)]);
    for (_, slices) in &grant_log {
        assert!(
            *slices <= budget,
            "no grant may exceed the budget: {grant_log:?}"
        );
    }
}

/// Uncontended no-op (issue #746): a single request never rotates
/// regardless of the budget, and its slice count matches the pre-change
/// behavior (the expiry condition requires a non-empty backlog).
#[test]
fn rotation_uncontended_single_request_never_parks() {
    let budget = 2;
    let (states, grant_log) = run_rotation_harness(budget, vec![spec_a()], None, None, None);
    let a = states[0].outcome.as_ref().expect("A finished");
    // Slice 0 + three rounds, all in ONE grant despite exceeding the
    // budget: uncontended holds never expire.
    assert_eq!(a.slices, 4);
    assert_eq!(grant_log, vec![('A', 4)]);
    assert_eq!(
        a.committed,
        vec![100, 101, 102, 103, 104, 105, 106, 107, 108]
    );
}

/// A parked job whose client disconnected is promoted normally; its next
/// step observes the cancel at the round top and finishes with the
/// tokens emitted so far, and the ring does not wedge (the other request
/// still completes with full parity).
#[test]
fn rotation_cancelled_parked_job_resolves_without_wedging() {
    let iso_b = isolated_outcome(spec_b());
    let (states, _grant_log) = run_rotation_harness(
        2,
        vec![spec_a(), spec_b()],
        None,
        /* cancel A at its first park */ Some(0),
        None,
    );
    let a = states[0].outcome.as_ref().expect("A resolved");
    let b = states[1].outcome.as_ref().expect("B finished");
    // A stopped at its cancellation boundary: slice 0 + round 1 only.
    assert_eq!(a.committed, vec![100, 101, 102, 103, 104]);
    assert_eq!(a.rounds, 1);
    assert!(a.healthy_finish, "cancel classifies as a clean early stop");
    // B is unaffected.
    assert_eq!(b.committed, iso_b.committed);
    assert!(b.healthy_finish);
}

/// A waiter whose client disconnected before its first grant aborts with
/// bookkeeping only, and the ring does not wedge: the parked job behind
/// it is promoted in the same tick and completes with full parity.
#[test]
fn rotation_cancelled_waiter_is_skipped_without_wedging() {
    let iso_a = isolated_outcome(spec_a());
    let (states, grant_log) = run_rotation_harness(
        2,
        vec![spec_a(), spec_b()],
        /* cancel B before any grant */ Some(1),
        None,
        None,
    );
    let a = states[0].outcome.as_ref().expect("A finished");
    assert!(states[1].aborted, "cancelled waiter must abort");
    assert!(states[1].outcome.is_none());
    assert_eq!(a.committed, iso_a.committed);
    // A's schedule: admission grant (2 slices), then, after B's abort,
    // an uncontended resumed grant to completion (rounds 2 and 3).
    assert_eq!(grant_log, vec![('A', 2), ('A', 2)]);
}

/// Budget 1: strict per-slice interleave. A waiter's slice 0 spends its
/// whole grant and parks immediately, so an admission grant cannot hold
/// a free extra round beyond the budget; every later grant is exactly
/// one slice. The opening grant still holds slice 0 plus round 1
/// because a direct admission always starts with an empty backlog (the
/// contention modeled here arrives during that first grant) and the
/// first expiry check is the round boundary after it appears.
#[test]
fn rotation_budget_one_interleaves_strictly_per_slice() {
    let iso_a = isolated_outcome(spec_a());
    let iso_b = isolated_outcome(spec_b());
    let (states, grant_log) = run_rotation_harness(1, vec![spec_a(), spec_b()], None, None, None);
    let a = states[0].outcome.as_ref().expect("A finished");
    let b = states[1].outcome.as_ref().expect("B finished");
    assert_eq!(a.committed, iso_a.committed);
    assert_eq!(b.committed, iso_b.committed);
    assert!(a.healthy_finish && b.healthy_finish);
    assert_eq!(
        grant_log,
        vec![('A', 2), ('B', 1), ('A', 1), ('B', 1), ('A', 1), ('B', 1)],
        "B's admission grant must park right after slice 0 at budget 1"
    );
}

/// A waiter whose promotion-time B=1 verdict re-check declines (the
/// adaptive policy settled to Decline while the request waited) is
/// routed to classic prefill with bookkeeping only: it never runs a
/// speculative slice, the ring does not wedge, and the other request
/// completes with full parity. Mirrors the scheduler's
/// `mtp_b1_should_run` re-check in the Waiter promotion arm.
#[test]
fn rotation_declined_waiter_at_promotion_routes_to_classic() {
    let iso_a = isolated_outcome(spec_a());
    let (states, grant_log) = run_rotation_harness(
        2,
        vec![spec_a(), spec_b()],
        None,
        None,
        /* decline B's promotion verdict */ Some(1),
    );
    let a = states[0].outcome.as_ref().expect("A finished");
    assert!(
        states[1].routed_to_classic,
        "declined waiter must route to classic"
    );
    assert!(
        states[1].outcome.is_none(),
        "no speculative slices for the declined waiter"
    );
    assert!(!states[1].aborted, "routing to classic is not an abort");
    assert_eq!(a.committed, iso_a.committed);
    // Same shape as the cancelled-waiter schedule: B resolves with
    // bookkeeping only and A's next grant runs uncontended to the end.
    assert_eq!(grant_log, vec![('A', 2), ('A', 2)]);
}

/// Anti-starvation floor (issue #746 security hardening): two long
/// High-lane jobs rotating between themselves keep a High entry in the
/// ring at every grant decision, so under strict lane precedence the
/// Normal-lane parked job A would receive NO grant (zero progress, a
/// stalled mid-stream client) until both High jobs completed; a
/// sustained High-lane arrival stream would extend that indefinitely.
/// With the skip-cap escalation, A's skip counter reaches the cap
/// within two decisions and A must be granted next, so between any two
/// consecutive A grants at most `MTP_SLICE_GRANT_SKIP_CAP + 1` other
/// grants run, and every stream still matches its isolated run.
#[test]
fn rotation_skip_cap_prevents_high_lane_starvation() {
    let a = chain_spec('A', 100, RequestPriority::Normal, 5, 9999);
    let b = chain_spec('B', 500, RequestPriority::High, 5, 8888);
    let c = chain_spec('C', 900, RequestPriority::High, 5, 7777);
    let iso = [
        isolated_outcome(a.clone()),
        isolated_outcome(b.clone()),
        isolated_outcome(c.clone()),
    ];
    let (states, grant_log) = run_rotation_harness(2, vec![a, b, c], None, None, None);
    for (idx, iso) in iso.iter().enumerate() {
        let out = states[idx].outcome.as_ref().expect("job finished");
        assert_eq!(
            out.committed, iso.committed,
            "stream parity for spec {idx} under lane contention"
        );
        assert!(out.healthy_finish);
    }
    // Bounded gap for the Normal-lane job: between consecutive A grants
    // at most MTP_SLICE_GRANT_SKIP_CAP + 1 non-A grants run.
    let mut gap = 0usize;
    for &(tag, _) in &grant_log {
        if tag == 'A' {
            gap = 0;
        } else {
            gap += 1;
            assert!(
                gap <= MTP_SLICE_GRANT_SKIP_CAP + 1,
                "Normal-lane job starved by the High lane: {grant_log:?}"
            );
        }
    }
    // A kept receiving grants throughout, not only the opening one.
    let a_grants = grant_log.iter().filter(|&&(tag, _)| tag == 'A').count();
    assert!(
        a_grants >= 3,
        "A must keep progressing under High-lane pressure: {grant_log:?}"
    );
}
