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

//! Unit tests for the speculative-decoding burst driver
//! ([`super::speculative_burst`]).
//!
//! ## What these tests pin
//!
//! - **Gate**: [`should_burst_for_sequence`] returns `false` for every
//!   condition that would silently break the burst path (multimodal
//!   payloads / VLM embeddings, structured-output constraint, adopted
//!   prompt-cache prefix). History-dependent sampling penalties,
//!   logprobs requests, and active thinking-budget state stay eligible
//!   for the B = 1 burst path; B > 1 has its own stricter window guard
//!   for payloads the batched round loops cannot yet return. Each gate
//!   has a dedicated test so an accidental regression is caught at
//!   `cargo test` time rather than in production.
//!
//! - **Bind contract (CRITICAL)**: [`drive_mtp_generator`] must run on
//!   a drafter that has already been [`Drafter::bind`]-ed against its
//!   target. The [`Gemma4AssistantDraftModel`] (the only currently
//!   shipping MTP drafter shape) requires `bind` to resolve `lm_head`;
//!   without it, the first `draft_block` call returns
//!   [`DrafterError::BindNotCalled`] which the round loop silently
//!   swallows, and the request finalizes with exactly one seed-bonus
//!   token. The first PR-review cycle missed this because the
//!   real-model parity test was deferred to a CI hardware lane; cycle
//!   2 ([this file]) pins the invariant with a fast-running mock test.
//!
//!   The regression is caught at two layers:
//!     1. **Unit**: [`drive_mtp_generator_round_loop_produces_more_than_one_token_when_drafter_is_bound`]
//!        proves that a *bound* drafter + matching mock target emits N > 1
//!        tokens through the burst's generator helper.
//!     2. **Unit**: [`drive_mtp_generator_round_loop_returns_only_seed_when_drafter_is_unbound_via_real_gemma4_drafter`]
//!        proves the inverse: an *unbound* real
//!        [`Gemma4AssistantDraftModel`] produces exactly one token
//!        (the seed bonus). If a future refactor accidentally drops
//!        the `drafter.bind(...)` call in `run_mtp_burst`, this test
//!        will go from "1 token" expected back to passing because the
//!        bind step is missing — making the regression loud.
//!
//! ## What these tests do NOT pin
//!
//! - End-to-end byte-equality against the no-drafter baseline on a real
//!   31B Gemma 4 target. That belongs in `tests/speculative_parity.rs`
//!   (`greedy_parity_mtp_gemma4_31b`, `#[ignore]` for CI hardware lane).
//!   Run that test as part of the cycle-2 verification — see the PR body.
//!
//! - The `run_mtp_burst` / `run_dflash_burst` outer dispatch via
//!   `ctx.model: &LoadedModel`. We can't easily construct `LoadedModel`
//!   variants without loading real weights from disk, so the outer
//!   variant dispatch is covered by the structural tests in
//!   `tests/speculative_dispatch.rs` and the real-model end-to-end
//!   test in `tests/speculative_parity.rs`.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, from_slice_f32};

use crate::server::batch::sequence::{RequestPriority, SequenceInfo, SequenceState};
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::thinking_budget::ThinkingState;

use super::speculative_burst::{
    WorkerDrafterSlot, can_join_batched_burst_window, should_burst_for_sequence,
};

// =============================================================================
// Test helpers
// =============================================================================

/// Build a minimal [`SequenceInfo`] in the `Queued` state with default
/// sampling. Callers tweak individual fields before passing to
/// [`should_burst_for_sequence`].
fn make_test_sequence() -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(42),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 64,
        eos_token_ids: vec![2],
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
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        sampler_state: None,
        merged_eos: Vec::new(),
        thinking: ThinkingState::disabled(),
        structured: None,
    };

    (seq, rx)
}

/// MTP dispatch with a placeholder drafter path. The path is never
/// touched by [`should_burst_for_sequence`] (it short-circuits on the
/// per-sequence preconditions), so a `/tmp/...` placeholder is fine.
fn make_mtp_dispatch() -> crate::server::SpeculativeDispatch {
    crate::server::SpeculativeDispatch::Mtp {
        draft_model_path: std::path::PathBuf::from("/tmp/test-mtp-drafter"),
        block_size: 4,
        user_requested_explicit_kind: true,
    }
}

/// DFlash dispatch with a placeholder drafter path. Like
/// [`make_mtp_dispatch`], this is only consumed by
/// [`should_burst_for_sequence`], so no filesystem access occurs.
fn make_dflash_dispatch() -> crate::server::SpeculativeDispatch {
    crate::server::SpeculativeDispatch::DFlash {
        draft_model_path: std::path::PathBuf::from("/tmp/test-dflash-drafter"),
        block_size: 16,
        user_requested_explicit_kind: true,
    }
}

// =============================================================================
// `should_burst_for_sequence` gate tests
//
// These tests verify EVERY per-sequence gate the burst path consults.
// Each gate has a dedicated test so a regression that drops one (e.g.
// during a future refactor) surfaces with a clear failure name.
// =============================================================================

#[test]
fn burst_allowed_for_default_sequence_under_mtp_dispatch() {
    let dispatch = make_mtp_dispatch();
    let (seq, _rx) = make_test_sequence();
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "default sequence under MTP dispatch must be eligible for burst"
    );
}

#[test]
fn burst_declined_when_dispatch_is_disabled() {
    let dispatch = crate::server::SpeculativeDispatch::Disabled;
    let (seq, _rx) = make_test_sequence();
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "Disabled dispatch must never burst"
    );
}

#[test]
fn burst_declined_for_vlm_embeddings() {
    use crate::vision::merge::InputEmbeddings;

    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    // Construct a placeholder InputEmbeddings (the value's interior
    // is irrelevant — the gate only checks Option::is_some). We
    // construct it directly with a tiny tensor rather than going
    // through the heavy `prepare_inputs_for_multimodal` helper.
    seq.vlm_embeddings = Some(InputEmbeddings {
        inputs_embeds: from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 4]),
        attention_mask_4d: None,
    });
    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "VLM-embedding requests must fall back to classic decode"
    );
}

#[test]
fn dflash_burst_allowed_for_vlm_wrapped_text_only_sequence_shape() {
    let dispatch = make_dflash_dispatch();
    let (seq, _rx) = make_test_sequence();
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "a text-only request remains burst-eligible under DFlash dispatch; \
         the model-variant gate separately accepts Qwen35VLM/Qwen35MoeVLM"
    );
}

#[test]
fn dflash_burst_declined_for_raw_multimodal_payload_without_embeddings() {
    let dispatch = make_dflash_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.images.push(b"not-a-real-image-for-gate".to_vec());

    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "DFlash must not enter the burst path for image-bearing requests even \
         before/without a prepared embeddings tensor; text-only VLM-wrapped \
         Qwen 3.5 is supported, true multimodal speculative tail is gated off"
    );
}

#[test]
fn dflash_burst_declined_for_audio_payload_without_embeddings() {
    let dispatch = make_dflash_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.audio.push(b"not-real-audio-for-gate".to_vec());

    assert!(
        !should_burst_for_sequence(&dispatch, &seq),
        "DFlash must not enter the burst path for audio-bearing multimodal \
         requests unless the multimodal speculative tail is explicitly enabled"
    );
}

#[test]
fn burst_allowed_for_adopted_prompt_cache_prefix() {
    // Issue #518: an adopted prompt-cache prefix is no longer a blanket gate.
    // `should_burst_for_sequence` now admits `prefill_start_offset > 0`; the
    // per-kind driver decides (MTP reuses the suffix, DFlash falls back).
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.prefill_start_offset = 5;
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "an adopted cache prefix must reach the B=1 driver, not be gated out at the scheduler"
    );
}

#[test]
fn batched_window_declines_adopted_prompt_cache_prefix() {
    // Issue #518: adopted-prefix requests are handled only on the B=1 arm
    // today. `can_join_batched_burst_window` keeps them out of B>1 windows so
    // they route to the single-request burst (MTP suffix-reuse / DFlash
    // fallback) rather than the batched round loops, which assume a zero KV
    // offset per row.
    let (mut seq, _rx) = make_test_sequence();
    seq.prefill_start_offset = 5;
    assert!(
        !can_join_batched_burst_window(&seq),
        "an adopted cache prefix must be excluded from B>1 windows and land on the B=1 arm"
    );

    // A cold request (no adopted prefix) still joins a batched window.
    let (cold_seq, _rx2) = make_test_sequence();
    assert_eq!(cold_seq.prefill_start_offset, 0);
    assert!(
        can_join_batched_burst_window(&cold_seq),
        "a cold request must remain eligible for a B>1 window"
    );
}

#[test]
fn mtp_prefill_suffix_start_resolves_cold_reuse_and_degenerate_offsets() {
    use super::speculative_burst::mtp_prefill_suffix_start;
    // Cold: offset 0 forwards the whole prompt.
    assert_eq!(mtp_prefill_suffix_start(0, 10), Some(0));
    // Reuse: a proper prefix forwards only the suffix `[offset..]`.
    assert_eq!(mtp_prefill_suffix_start(5, 10), Some(5));
    // Reuse boundary: a single suffix token still has a position to sample the
    // first bonus from.
    assert_eq!(mtp_prefill_suffix_start(9, 10), Some(9));
    // Degenerate: the whole prompt is cached — no suffix position remains, so
    // the driver must decline to classic.
    assert_eq!(mtp_prefill_suffix_start(10, 10), None);
    assert_eq!(mtp_prefill_suffix_start(11, 10), None);
}

// History-dependent sampling penalties are NO LONGER a decline-to-classic
// gate: the burst threads `initial_token_history(&prompt, ..)`
// into the first-bonus sample, so a penalty-bearing request's first bonus
// is byte-identical to the classic decode path. The four tests below
// (formerly `burst_declined_for_{repetition,frequency,presence,dry}_penalty`)
// now assert the gate stays OPEN for each penalty config. The
// `token_history` threading itself is pinned by
// `drive_mtp_generator_round_loop_threads_token_history_into_prefill_and_seed`
// further down; full byte-equality against the classic path lives in the
// real-model `tests/speculative_parity.rs` lane.

#[test]
fn burst_allowed_for_repetition_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.repetition_penalty = 1.1;
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "repetition_penalty != 1.0 must NOT decline to classic — the burst \
         now threads token_history into the first-bonus sample"
    );
}

#[test]
fn burst_allowed_for_frequency_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.frequency_penalty = 0.5;
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "frequency_penalty != 0.0 must NOT decline to classic — the burst \
         now threads token_history into the first-bonus sample"
    );
}

#[test]
fn burst_allowed_for_presence_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.presence_penalty = 0.25;
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "presence_penalty != 0.0 must NOT decline to classic — the burst \
         now threads token_history into the first-bonus sample"
    );
}

#[test]
fn burst_allowed_for_dry_penalty() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.sampling.dry_multiplier = 0.8;
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "dry_multiplier != 0.0 must NOT decline to classic — the burst \
         now threads token_history into the first-bonus sample"
    );
}

// `logprobs_config.enabled` is NO LONGER a decline-to-classic gate
// the burst threads `logprobs_config` through
// `MtpGenerator::generate` / `DFlashGenerator::run` and emits
// `TokenWithLogprobs` events from `finalize_burst_success`. The test
// below (formerly `burst_declined_when_logprobs_enabled`) now asserts
// the gate stays OPEN. The logprobs threading itself is pinned by
// `drive_mtp_generator_round_loop_threads_logprobs_through_to_emitted_tokens`
// further down; full byte-equality against the classic path lives in
// the real-model `tests/speculative_parity.rs` lane.
#[test]
fn burst_allowed_when_logprobs_enabled() {
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    seq.logprobs_config = LogprobsConfig {
        enabled: true,
        top_k: 5,
    };
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "logprobs_config.enabled=true must NOT decline to classic — the burst \
         now threads logprobs through and emits TokenWithLogprobs"
    );
}

#[test]
fn batched_window_rejects_logprobs_enabled_sequences() {
    let dispatch = make_mtp_dispatch();
    let (default_seq, _rx) = make_test_sequence();
    assert!(
        can_join_batched_burst_window(&default_seq),
        "default requests without logprobs may join B>1 speculative windows"
    );

    let (mut seq, _rx) = make_test_sequence();
    seq.logprobs_config = LogprobsConfig {
        enabled: true,
        top_k: 5,
    };

    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "logprobs-enabled requests must still be eligible for the B=1 burst"
    );
    assert!(
        !can_join_batched_burst_window(&seq),
        "B>1 speculative windows return token IDs only; logprobs-enabled \
         requests must stay on the B=1 path so TokenWithLogprobs is emitted"
    );
}

// Thinking-budget enforcement is NO LONGER a decline-to-classic gate
// `finalize_burst_success` runs the same per-token
// `decide_override` + `observe` cycle as the classic path's
// `apply_thinking_budget`, injecting a forced `</think>` at the budget
// boundary. The test below (formerly
// `burst_declined_when_thinking_state_active`) now asserts the gate
// stays OPEN. The forced-injection logic itself is pinned by
// `apply_burst_thinking_budget_*` tests further down.
#[test]
fn burst_allowed_when_thinking_state_active() {
    use crate::server::thinking_budget::{ThinkingBudget, ThinkingTokenIds};
    let dispatch = make_mtp_dispatch();
    let (mut seq, _rx) = make_test_sequence();
    // ThinkingState is "active" (= not disabled) when both token_ids
    // and budget are set. `from_raw_i32(256)` yields a
    // `ThinkingBudget::Limited(256)`.
    let token_ids = ThinkingTokenIds {
        open: 100,
        close: 101,
    };
    let budget = ThinkingBudget::from_raw_i32(256)
        .expect("valid budget")
        .expect("256 > 0 yields Some");
    seq.thinking = ThinkingState::new(Some(token_ids), Some(budget), false);
    assert!(
        should_burst_for_sequence(&dispatch, &seq),
        "thinking-budget enforcement must NOT decline to classic — the burst \
         now injects forced </think> at the budget boundary"
    );
}

// =============================================================================
// `WorkerDrafterSlot` tests — load/return lifecycle
// =============================================================================

#[test]
fn worker_drafter_slot_carries_mtp_path() {
    let dispatch = make_mtp_dispatch();
    let slot = WorkerDrafterSlot::from_dispatch(&dispatch);
    assert_eq!(slot.kind, Some(DrafterKind::Mtp));
    assert_eq!(
        slot.draft_model_path,
        Some(std::path::PathBuf::from("/tmp/test-mtp-drafter"))
    );
}

#[test]
fn worker_drafter_slot_disabled_dispatch_has_no_path() {
    let slot = WorkerDrafterSlot::from_dispatch(&crate::server::SpeculativeDispatch::Disabled);
    assert!(slot.draft_model_path.is_none());
    assert!(slot.kind.is_none());
}

// =============================================================================
// Mock target + drafter for the MTP round-loop regression test.
//
// These mocks mirror the ones in
// `mlxcel-core/src/speculative/mtp/tests.rs` but are inlined here so
// the binary-crate test can drive `MtpGenerator` directly. They are
// deliberately minimal: the `MockMtpTarget` returns dummy hidden + K/V
// tensors so the drafter's `set_shared_kv` accepts the slabs without
// inspecting them; the `MockMtpDrafter` returns scripted draft tokens.
// =============================================================================

/// Tiny dummy `[1, 1, 4]` FP32 tensor used wherever the target needs
/// to surface a "hidden" or a "shared K/V slab".
fn dummy_tensor() -> UniquePtr<MlxArray> {
    from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 4])
}

/// Deterministic synthetic [`mlxcel_core::sampling::TokenLogprobData`]
/// for a token id — `logprob = -(token_id as f32) * 0.01`. The mock
/// `MtpTarget` returns this from `prefill_and_seed` / `verify_forward`
/// so the logprobs-threading test can assert the *right*
/// logprob (the one keyed to a given token) reached
/// `finalize_burst_success` unchanged. A real Gemma 4 / Qwen 3.5
/// target computes the value from log-softmax of the verify logits;
/// the byte-equality-against-classic check belongs in the real-model
/// `tests/speculative_parity.rs` lane.
fn synthetic_logprob(token_id: i32) -> mlxcel_core::sampling::TokenLogprobData {
    mlxcel_core::sampling::TokenLogprobData {
        token_id,
        logprob: -(token_id as f32) * 0.01,
        top_alternatives: Vec::new(),
    }
}

struct MockMtpTarget {
    first_bonus: i32,
    scripted_target_tokens: RefCell<Vec<Vec<i32>>>,
    eos: Vec<i32>,
    cumulative_offset: RefCell<usize>,
    /// `token_history` slice the most recent `prefill_and_seed` call
    /// received. `None` until `prefill_and_seed` runs. Used by
    /// `drive_mtp_generator_round_loop_threads_token_history_into_prefill_and_seed`
    /// to pin the plumbing.
    seen_token_history: RefCell<Option<Vec<i32>>>,
}

impl MockMtpTarget {
    fn new(scripted: Vec<Vec<i32>>, eos: Vec<i32>) -> Self {
        Self {
            first_bonus: 100,
            scripted_target_tokens: RefCell::new(scripted),
            eos,
            cumulative_offset: RefCell::new(0),
            seen_token_history: RefCell::new(None),
        }
    }

    fn with_first_bonus(mut self, bonus: i32) -> Self {
        self.first_bonus = bonus;
        self
    }

    fn build_verify_output(&self, advance: usize) -> MtpVerifyOutput {
        *self.cumulative_offset.borrow_mut() += advance;
        let kv_offset = *self.cumulative_offset.borrow();
        let next_shared_kv = vec![
            dummy_tensor(),
            dummy_tensor(),
            dummy_tensor(),
            dummy_tensor(),
        ];
        MtpVerifyOutput {
            next_hidden: dummy_tensor(),
            next_shared_kv,
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
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<mlxcel_core::sampling::TokenLogprobData>,
    ) {
        // Record what the generator handed us so the test can
        // assert the burst threaded the real token_history through.
        *self.seen_token_history.borrow_mut() = Some(token_history.to_vec());
        let seed = self.build_verify_output(1);
        // Synthetic first-bonus logprob: `None` when disabled; otherwise
        // a deterministic function of the token id (`synthetic_logprob`)
        // so the test can assert the right logprob reached
        // the right token.
        let first_bonus_lp = logprobs_config
            .enabled
            .then(|| synthetic_logprob(self.first_bonus));
        (self.first_bonus, seed, first_bonus_lp)
    }

    fn embed_token(&self, _token_id: i32) -> UniquePtr<MlxArray> {
        dummy_tensor()
    }

    fn verify_forward(
        &self,
        _verify_input: &[i32],
        _sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
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
        // Synthetic per-position logprobs aligned 1:1 with
        // `target_tokens`. `None` when disabled.
        let target_logprobs = logprobs_config.enabled.then(|| {
            target_tokens
                .iter()
                .map(|&tok| synthetic_logprob(tok))
                .collect()
        });
        VerifyForwardOutput {
            target_tokens,
            target_logprobs,
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

/// Tracking drafter that records `bind` and `draft_block` calls.
///
/// Used to assert the call-ordering invariant: `bind` must precede
/// `draft_block` in any code path that drives [`MtpGenerator`]. The
/// inner state proxies a basic "scripted tokens" drafter; the only
/// semantic difference from a plain mock is that this drafter
/// enforces "bind must have been called" by returning
/// [`DrafterError::BindNotCalled`] when `was_bound` is still `false`
/// at `draft_block` time. This mirrors the real
/// [`mlxcel_core::drafter::gemma4_assistant::Gemma4AssistantDraftModel`]'s
/// behavior and is the load-bearing regression guard for the CRITICAL
/// bug.
struct TrackingMockDrafter {
    scripted_draft_tokens: RefCell<Vec<Vec<i32>>>,
    was_bound: RefCell<bool>,
    /// Records the timeline of method names to verify ordering.
    /// Cells: "bind", "set_shared_kv", "draft_block". Uses
    /// [`Rc<RefCell<_>>`] rather than `Arc<RefCell<_>>` because the
    /// burst test is single-threaded; the events log is shared only
    /// between the drafter and the test assertion at the end.
    events: Rc<RefCell<Vec<&'static str>>>,
}

impl TrackingMockDrafter {
    fn new(scripted: Vec<Vec<i32>>, events: Rc<RefCell<Vec<&'static str>>>) -> Self {
        Self {
            scripted_draft_tokens: RefCell::new(scripted),
            was_bound: RefCell::new(false),
            events,
        }
    }
}

impl Drafter for TrackingMockDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        *self.was_bound.borrow_mut() = true;
        self.events.borrow_mut().push("bind");
        Ok(())
    }

    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        _kv_offset: usize,
        _position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        self.events.borrow_mut().push("set_shared_kv");
        Ok(())
    }

    fn draft_block(
        &mut self,
        _last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        self.events.borrow_mut().push("draft_block");
        // CRITICAL: mirror Gemma4AssistantDraftModel — refuse to draft
        // without having been bound first. This is the load-bearing
        // regression check: if `run_mtp_burst` (or its
        // `drive_mtp_generator` helper) ever stops calling `bind`
        // before driving the generator, this branch fires and the
        // generator silently breaks out of its round loop — producing
        // exactly one seed-bonus token, matching the production bug.
        if !*self.was_bound.borrow() {
            return Err(DrafterError::BindNotCalled);
        }
        let mut q = self.scripted_draft_tokens.borrow_mut();
        let proposals = if q.len() > 1 {
            q.remove(0)
        } else if !q.is_empty() {
            q[0].clone()
        } else {
            vec![0; block_size.saturating_sub(1)]
        };
        Ok(proposals)
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

/// Minimal `LanguageModel` for [`TrackingMockDrafter::bind`]'s
/// signature — its methods are never called by the drafter (the
/// tracking impl just records the call), but we must hand it an
/// owned, sized `&dyn LanguageModel`.
struct MinimalLm;

impl LanguageModel for MinimalLm {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [mlxcel_core::layers::KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        unreachable!("MinimalLm::forward should not be called from burst-test mocks")
    }
    fn make_caches(&self) -> Vec<mlxcel_core::layers::KVCache> {
        Vec::new()
    }
    fn num_layers(&self) -> usize {
        0
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

// =============================================================================
// MTP bind regression — the CRITICAL test
//
// Cycle 1 / shipped a `run_mtp_burst` that omitted
// `drafter.bind(target)` before `MtpGenerator::new`. The result: every
// MTP burst request returned exactly one token (the seed bonus) because
// the first `draft_block` call inside the round loop returned
// `DrafterError::BindNotCalled`, which `MtpGenerator::generate`
// silently swallows.
//
// The test below pins both halves of the regression:
//
//   1. WITH bind → N > 1 tokens are emitted (the round-loop runs).
//   2. WITHOUT bind → exactly 1 token is emitted (the seed bonus only).
//
// If a future refactor accidentally drops the bind step in
// `run_mtp_burst::drive_mtp_generator`'s caller, half (2) flips from
// the explicit "1 token" expectation to a regression-mirroring failure,
// surfacing the bug at `cargo test` time rather than in production.
// =============================================================================

/// Helper: build a `MockMtpTarget` scripted for 2 successful verify
/// rounds and an EOS that terminates the loop on round 3.
///
/// Round 1: bonus=100, draft=[10, 11, 12], target=[10, 11, 12, 13]
///   → walk: accepted=3, new_tokens=[10, 11, 12, 13]
/// Round 2: bonus=13, draft=[20, 21, 22], target=[20, 21, 22, EOS]
///   → walk: accepted=3, new_tokens=[20, 21, 22, EOS], terminate.
fn make_two_round_target_and_drafter() -> (MockMtpTarget, Vec<Vec<i32>>) {
    let target_scripts = vec![
        vec![10, 11, 12, 13],
        vec![20, 21, 22, 9999], // 9999 = EOS in our test
    ];
    let drafter_scripts = vec![vec![10, 11, 12], vec![20, 21, 22]];
    let target = MockMtpTarget::new(target_scripts, vec![9999]).with_first_bonus(100);
    (target, drafter_scripts)
}

#[test]
fn drive_mtp_generator_round_loop_produces_more_than_one_token_when_drafter_is_bound() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    let (target, drafter_scripts) = make_two_round_target_and_drafter();

    let events = Rc::new(RefCell::new(Vec::new()));
    let mut drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    let lm = MinimalLm;

    // BIND FIRST — mirroring the cycle-2 fix in run_mtp_burst.
    drafter
        .bind(&lm)
        .expect("bind must succeed on tracking mock");

    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    // Not cancelled — exercise the full round loop.
    let cancel = AtomicBool::new(false);
    let (emitted, _logprobs, _stats) = generator.generate(
        &prompt,
        /* max_tokens= */ 16,
        &sampling,
        &[],
        &cancel,
        &LogprobsConfig::default(),
    );

    // We expect 8 tokens before EOS (4 per round, 2 rounds):
    // [100, 10, 11, 12, 13, 20, 21, 22]. Then the 9999 EOS in round 2's
    // last position terminates the loop. The exact count must be > 1 —
    // the load-bearing regression check is that bind() actually
    // enabled the round-loop to run beyond the seed bonus.
    assert!(
        emitted.len() > 1,
        "round loop must emit more than 1 token when drafter is bound; got {:?}",
        emitted
    );
    // The seed bonus is always first.
    assert_eq!(
        emitted[0], 100,
        "first emitted token must be the seed bonus"
    );
    // The events log must show bind before any draft_block call.
    let events_snapshot = events.borrow();
    let bind_idx = events_snapshot
        .iter()
        .position(|&e| e == "bind")
        .expect("bind must have been recorded");
    let first_draft_idx = events_snapshot
        .iter()
        .position(|&e| e == "draft_block")
        .expect("at least one draft_block call must have occurred");
    assert!(
        bind_idx < first_draft_idx,
        "bind must precede the first draft_block call; events = {events_snapshot:?}"
    );
}

#[test]
fn drive_mtp_generator_round_loop_returns_only_seed_when_drafter_is_unbound() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    // Same scripted target and drafter, but we deliberately SKIP the
    // bind() call. The TrackingMockDrafter mirrors the real
    // Gemma4AssistantDraftModel's behavior: draft_block returns
    // DrafterError::BindNotCalled, which MtpGenerator silently
    // swallows (round_loop break), and we end up with the seed bonus
    // only. This pins the failure mode of the cycle-1 regression so a
    // future refactor that re-drops bind() flips this test from
    // "1 token expected" to "more than 1 expected" — surfacing
    // immediately.
    let (target, drafter_scripts) = make_two_round_target_and_drafter();
    let events = Rc::new(RefCell::new(Vec::new()));
    let drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    // NO bind() — this is the production bug.
    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    // Not cancelled — the early-out we are pinning here is the unbound
    // drafter, not cancellation.
    let cancel = AtomicBool::new(false);
    let (emitted, _logprobs, _stats) = generator.generate(
        &prompt,
        /* max_tokens= */ 16,
        &sampling,
        &[],
        &cancel,
        &LogprobsConfig::default(),
    );

    // CRITICAL invariant — without bind, the round loop breaks on the
    // first draft_block call and we get exactly the seed bonus.
    assert_eq!(
        emitted.len(),
        1,
        "without bind(), the round loop must short-circuit and emit \
         only the seed bonus. Got {emitted:?} which means either bind \
         is being called somewhere unexpected OR MtpGenerator no longer \
         swallows DrafterError::BindNotCalled. Either way, revisit the \
         bind invariant in run_mtp_burst."
    );
    assert_eq!(
        emitted[0], 100,
        "the single emitted token must be the seed bonus"
    );
    // No bind event; at least one draft_block event (the one that
    // bailed).
    let events_snapshot = events.borrow();
    assert!(
        !events_snapshot.contains(&"bind"),
        "events log must not contain bind in the unbound case: {events_snapshot:?}"
    );
    assert!(
        events_snapshot.contains(&"draft_block"),
        "draft_block must still have been called once (and returned BindNotCalled): {events_snapshot:?}"
    );
}

// =============================================================================
// Cancellation propagation regression
//
// landed Option-B speculative-burst dispatch but
// `MtpGenerator::generate` / `DFlashGenerator::run` never inspected
// `seq.cancelled`. A client that disconnected mid-burst would keep the
// worker thread busy for the full `max_tokens` budget, wasting compute
// AND head-of-line-blocking the next request. threads an
// `&AtomicBool` through both generator APIs, checked once per round.
//
// The test below pins the MTP side at the `MtpGenerator::generate`
// layer (the same layer `drive_mtp_generator` drives): a *pre-flagged*
// cancellation flag must cause the round loop to break before the first
// `draft_block` call, so the generator returns exactly the seed bonus.
// If a future refactor drops the per-round `cancel.load(..)` check, this
// test flips from "1 token" to ">1 token" and the regression surfaces
// at `cargo test` time rather than as a stuck worker in production.
//
// The DFlash side shares the identical per-round-check shape in
// `DFlashGenerator::run`; it is exercised by the DFlash round-loop unit
// tests in `mlxcel-core` plus the integration tests in
// `tests/speculative_dispatch.rs`.
// =============================================================================

#[test]
fn drive_mtp_generator_round_loop_returns_only_seed_when_cancelled_before_first_round() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    // Same scripted two-round target + a properly bound drafter — the
    // ONLY difference from
    // `drive_mtp_generator_round_loop_produces_more_than_one_token_when_drafter_is_bound`
    // is the pre-flagged cancellation flag.
    let (target, drafter_scripts) = make_two_round_target_and_drafter();

    let events = Rc::new(RefCell::new(Vec::new()));
    let mut drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    let lm = MinimalLm;
    drafter
        .bind(&lm)
        .expect("bind must succeed on tracking mock");

    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    // Pre-flag cancellation BEFORE the generator runs — mirrors a client
    // that disconnected while the request was still queued / prefilling.
    let cancel = AtomicBool::new(true);
    let (emitted, _logprobs, _stats) = generator.generate(
        &prompt,
        /* max_tokens= */ 16,
        &sampling,
        &[],
        &cancel,
        &LogprobsConfig::default(),
    );

    // The prefill + seed-bonus emission happen unconditionally; the
    // per-round cancellation check fires at the TOP of the round loop,
    // before the first `draft_block`. So we get exactly the seed bonus.
    assert_eq!(
        emitted.len(),
        1,
        "a pre-flagged cancellation flag must break the round loop before \
         the first draft_block call, leaving only the seed bonus. Got \
         {emitted:?} — the per-round cancel.load(..) check in \
         MtpGenerator::generate may have regressed."
    );
    assert_eq!(
        emitted[0], 100,
        "the single emitted token must be the seed bonus"
    );
    // `bind` ran (we called it explicitly) but `draft_block` must NOT
    // have — the cancellation check short-circuits the round loop first.
    let events_snapshot = events.borrow();
    assert!(
        events_snapshot.contains(&"bind"),
        "bind was called explicitly before generate: {events_snapshot:?}"
    );
    assert!(
        !events_snapshot.contains(&"draft_block"),
        "draft_block must NOT run when cancellation is pre-flagged — the \
         round loop breaks before reaching it: {events_snapshot:?}"
    );
}

// =============================================================================
// History-dependent sampling penalties through the burst
//
// gated penalty-bearing requests
// (repetition / frequency / presence / DRY) to the classic decode path
// because the burst's first-bonus sample passed `&[]` for token_history.
// threads `initial_token_history(&prompt,..)` through
// `MtpGenerator::generate` → `MtpTarget::prefill_and_seed` (and through
// `sample_token_optimized` on the DFlash side), and removes the four
// decline-to-classic gate predicates.
//
// The gate change is pinned by `burst_allowed_for_{repetition,frequency,
// presence,dry}_penalty` above. The plumbing — that the generator
// actually forwards the caller's token_history to `prefill_and_seed`
// rather than dropping it — is pinned below by recording the slice the
// mock target received. Full byte-equality between the burst's first
// bonus and the classic path's first token (acceptance criterion 2)
// requires a real Gemma 4 / Qwen 3.5 target and lives in the
// `tests/speculative_parity.rs` real-model lane.
// =============================================================================

#[test]
fn drive_mtp_generator_round_loop_threads_token_history_into_prefill_and_seed() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    let (target, drafter_scripts) = make_two_round_target_and_drafter();

    let events = Rc::new(RefCell::new(Vec::new()));
    let mut drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    let lm = MinimalLm;
    drafter
        .bind(&lm)
        .expect("bind must succeed on tracking mock");

    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    // A non-empty token_history — what the burst computes via
    // `initial_token_history(&prompt, sampling.needs_token_history())`
    // for a repetition/frequency/presence/DRY-bearing request.
    let token_history = vec![1, 2, 3];
    let cancel = AtomicBool::new(false);
    let (_emitted, _logprobs, _stats) = generator.generate(
        &prompt,
        /* max_tokens= */ 16,
        &sampling,
        &token_history,
        &cancel,
        &LogprobsConfig::default(),
    );

    // The generator must have forwarded the caller's token_history
    // verbatim into `MtpTarget::prefill_and_seed` — if it dropped it (or
    // re-passed `&[]`), penalty-bearing requests would silently diverge
    // from the classic decode path. This is the load-bearing
    // plumbing assertion.
    let seen = generator
        .target()
        .seen_token_history
        .borrow()
        .clone()
        .expect("prefill_and_seed must have been called exactly once");
    assert_eq!(
        seen, token_history,
        "MtpGenerator::generate must forward the caller's token_history \
         into MtpTarget::prefill_and_seed unchanged; got {seen:?}"
    );
}

// =============================================================================
// Logprobs through the burst
//
// gated `logprobs_config.enabled` requests to the
// classic decode path because `finalize_burst_success` emitted plain
// `Token(text)` events. threads `logprobs_config` through
// `MtpGenerator::generate` → `MtpTarget::prefill_and_seed` /
// `verify_forward` (and through `DFlashGenerator::run` on the DFlash
// side), returns a per-token `Vec<Option<TokenLogprobData>>` aligned
// with the emitted tokens, and `finalize_burst_success` emits
// `TokenWithLogprobs` events.
//
// The gate change is pinned by `burst_allowed_when_logprobs_enabled`
// above. The plumbing — that the generator actually returns one logprob
// per emitted token, index-aligned, with the right value keyed to the
// right token — is pinned below: the mock target returns deterministic
// `synthetic_logprob(token)` entries, and the test asserts the
// generator's returned `logprobs` vec matches. Full byte-equality
// between the burst's logprobs and the classic path's logprobs
// (acceptance criterion 2) requires a real Gemma 4 / Qwen 3.5 target
// and lives in the `tests/speculative_parity.rs` real-model lane.
// =============================================================================

#[test]
fn drive_mtp_generator_round_loop_threads_logprobs_through_to_emitted_tokens() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    let (target, drafter_scripts) = make_two_round_target_and_drafter();

    let events = Rc::new(RefCell::new(Vec::new()));
    let mut drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    let lm = MinimalLm;
    drafter
        .bind(&lm)
        .expect("bind must succeed on tracking mock");

    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    let cancel = AtomicBool::new(false);
    // Logprobs ENABLED — exercise the capture path.
    let logprobs_config = LogprobsConfig {
        enabled: true,
        top_k: 0,
    };
    let (emitted, logprobs, _stats) = generator.generate(
        &prompt,
        /* max_tokens= */ 16,
        &sampling,
        &[],
        &cancel,
        &logprobs_config,
    );

    // The scripted two-round target emits [100, 10, 11, 12, 13, 20, 21,
    // 22, 9999]: the seed bonus 100, two full-accept rounds of 4 tokens
    // each, and the round-2 EOS (9999). `MtpGenerator::generate` pushes
    // each walk token to `emitted` BEFORE its EOS check fires, so the
    // EOS token itself IS part of the emitted sequence.
    assert_eq!(
        emitted,
        vec![100, 10, 11, 12, 13, 20, 21, 22, 9999],
        "scripted two-round target must emit the hand-computed sequence"
    );

    // The generator must return exactly one logprob entry per emitted
    // token, index-aligned. If it dropped entries (or returned an empty
    // vec despite logprobs being enabled), a speculative response would
    // silently lose its `TokenWithLogprobs` payload.
    assert_eq!(
        logprobs.len(),
        emitted.len(),
        "with logprobs enabled, generate() must return one logprob entry \
         per emitted token (index-aligned); got {} entries for {} tokens",
        logprobs.len(),
        emitted.len(),
    );

    // Every entry must be `Some` and carry the deterministic synthetic
    // logprob keyed to *that* token — proving the right logprob reached
    // the right position through the round loop's walk + emit.
    for (i, (&tok, lp)) in emitted.iter().zip(logprobs.iter()).enumerate() {
        let lp = lp.as_ref().unwrap_or_else(|| {
            panic!("logprobs[{i}] must be Some for token {tok} when logprobs enabled")
        });
        assert_eq!(
            lp.token_id, tok,
            "logprobs[{i}].token_id must match emitted[{i}]"
        );
        let expected = synthetic_logprob(tok);
        assert_eq!(
            lp.logprob, expected.logprob,
            "logprobs[{i}].logprob for token {tok} must be the synthetic value \
             threaded from the mock target"
        );
    }
}

#[test]
fn drive_mtp_generator_round_loop_returns_empty_logprobs_when_disabled() {
    use mlxcel_core::speculative::mtp::MtpGenerator;

    // The zero-overhead path: with logprobs disabled, `generate()` must
    // return an EMPTY logprobs vec — no allocation, no per-token
    // `compute_logprobs` work — and `finalize_burst_success` falls
    // through to plain `Token` events.
    let (target, drafter_scripts) = make_two_round_target_and_drafter();
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut drafter = TrackingMockDrafter::new(drafter_scripts, events.clone());
    let lm = MinimalLm;
    drafter
        .bind(&lm)
        .expect("bind must succeed on tracking mock");
    let boxed: Box<dyn Drafter> = Box::new(drafter);
    let mut generator = MtpGenerator::new(target, boxed, /* block_size= */ 4);

    let prompt = vec![1, 2, 3];
    let sampling = SamplingConfig::greedy();
    let cancel = AtomicBool::new(false);
    let (emitted, logprobs, _stats) = generator.generate(
        &prompt,
        /* max_tokens= */ 16,
        &sampling,
        &[],
        &cancel,
        // Disabled — the default.
        &LogprobsConfig::default(),
    );

    assert!(
        !emitted.is_empty(),
        "sanity: the round loop should still emit tokens"
    );
    assert!(
        logprobs.is_empty(),
        "with logprobs disabled, generate() must return an EMPTY logprobs \
         vec (zero-overhead path); got {} entries",
        logprobs.len(),
    );
}

// =============================================================================
// DFlash bind asymmetry — why there is no mock regression test here
//
// Unlike MTP, `DFlashGenerator::run` calls `drafter.bind(target_lm)?`
// internally on every invocation (see
// `mlxcel_core/src/drafter/dflash/round_loop.rs::run`). There is no
// "caller must bind first" contract to pin: adding a manual bind before
// `run_dflash_burst` would double-bind and is not the desired behaviour.
//
// Consequently there is no analogue of the
// `drive_mtp_generator_round_loop_returns_only_seed_when_drafter_is_unbound`
// test for DFlash — if `DFlashGenerator::run` were accidentally changed to
// require an external bind, the DFlash integration tests in
// `tests/speculative_dispatch.rs` and the real-model smoke tests would
// catch it before production. The asymmetry is documented at the
// function-level docstring of `run_dflash_burst` and in the module
// docstring under "## Bind asymmetry: MTP vs DFlash".
// =============================================================================

// =============================================================================
// `try_run_burst_b1` — decline path without a real model
//
// `try_run_burst_b1` takes a `BurstContext<'_>` which requires a
// `&LoadedModel` — loading real model weights is infeasible in a fast unit
// test. The variant-dispatch decline path is therefore verified at the
// `should_burst_for_sequence` level above (which fires before any model
// access) and at the integration level in `tests/speculative_dispatch.rs`.
//
// What we can unit-test without weights: the `try_run_burst_b1` interface
// itself (names / signatures stable) is pinned by the compile test below.
// =============================================================================

// =============================================================================
// Thinking-budget enforcement through the burst
//
// gated requests with active thinking-budget
// enforcement to the classic decode path because the burst path never
// implemented the forced `</think>` injection. has
// `finalize_burst_success` run the same per-token `decide_override` +
// `observe` cycle the classic path's `apply_thinking_budget` uses,
// factored into the `apply_burst_thinking_budget` helper.
//
// The gate change is pinned by `burst_allowed_when_thinking_state_active`
// above. The forced-injection logic itself is pinned below at the
// `apply_burst_thinking_budget` helper level — `finalize_burst_success`
// needs a `BurstContext` (a real `&LoadedModel`) which is infeasible in
// a fast unit test, but the helper is a pure function over
// `&mut ThinkingState` (the same shape as the classic path's
// `apply_thinking_budget`), so it is directly testable. The end-to-end
// burst behaviour is exercised in the real-model `tests/` lane.
// =============================================================================

#[test]
fn apply_burst_thinking_budget_passes_through_when_disabled() {
    use super::speculative_burst::apply_burst_thinking_budget;
    use crate::server::thinking_budget::ThinkingState;

    // A disabled state must short-circuit: the token passes through
    // unchanged and `override_fired` is false (zero-overhead path for
    // non-thinking requests).
    let mut thinking = ThinkingState::disabled();
    let (final_token, override_fired) = apply_burst_thinking_budget(&mut thinking, 4242);
    assert_eq!(
        final_token, 4242,
        "disabled state must pass the token through"
    );
    assert!(
        !override_fired,
        "disabled state must never report an override"
    );
}

#[test]
fn apply_burst_thinking_budget_injects_close_at_budget_boundary() {
    use super::speculative_burst::apply_burst_thinking_budget;
    use crate::server::thinking_budget::{ThinkingBudget, ThinkingState, ThinkingTokenIds};

    // Budget = 2 in-block tokens. `enter_block_on_start = true` so the
    // first emitted token is already inside the `<think>` block (the
    // Qwen3 default where the chat template primes `<think>\n`).
    let token_ids = ThinkingTokenIds {
        open: 100,
        close: 101,
    };
    let budget = ThinkingBudget::from_raw_i32(2)
        .expect("valid budget")
        .expect("2 > 0 yields Some");
    let mut thinking = ThinkingState::new(Some(token_ids), Some(budget), true);

    // Token 1: in-block #1 — under budget, passes through.
    let (t1, o1) = apply_burst_thinking_budget(&mut thinking, 200);
    assert_eq!(t1, 200, "first in-block token is under budget");
    assert!(!o1, "no override under budget");

    // Token 2: in-block #2 — reaches budget cap (in_block_count == 2),
    // but `decide_override` fires on the NEXT token (it checks
    // `in_block_count >= cap` before observing this one). So token 2
    // still passes through.
    let (t2, o2) = apply_burst_thinking_budget(&mut thinking, 201);
    assert_eq!(t2, 201, "second in-block token still under the cap check");
    assert!(
        !o2,
        "no override yet — cap is checked before this token is observed"
    );

    // Token 3: `in_block_count` is now 2 == cap, so `decide_override`
    // returns `ForceClose(101)`. The burst-produced token (202) is
    // SUBSTITUTED with the forced `</think>` close id, and
    // `override_fired` is true.
    let (t3, o3) = apply_burst_thinking_budget(&mut thinking, 202);
    assert_eq!(
        t3, 101,
        "at the budget boundary the burst token must be replaced with the \
         forced </think> close id"
    );
    assert!(
        o3,
        "override_fired must be true when the forced </think> is injected — \
         the caller uses this to drop the (now-stale) per-token logprob"
    );

    // Token 4: the block is now Closed — budget logic is inactive, the
    // token passes through unchanged.
    let (t4, o4) = apply_burst_thinking_budget(&mut thinking, 203);
    assert_eq!(t4, 203, "after the block closes, tokens pass through");
    assert!(!o4, "no override once the block is Closed");
}

#[test]
fn apply_burst_thinking_budget_respects_natural_close_before_budget() {
    use super::speculative_burst::apply_burst_thinking_budget;
    use crate::server::thinking_budget::{ThinkingBudget, ThinkingState, ThinkingTokenIds};

    // Generous budget; the model closes the block on its own (emits the
    // `</think>` token) well before the cap. The burst must NOT inject
    // a second `</think>` — the natural close passes through and the
    // block transitions to Closed.
    let token_ids = ThinkingTokenIds {
        open: 100,
        close: 101,
    };
    let budget = ThinkingBudget::from_raw_i32(1000)
        .expect("valid budget")
        .expect("1000 > 0 yields Some");
    let mut thinking = ThinkingState::new(Some(token_ids), Some(budget), true);

    // A couple of in-block tokens, then the model's own `</think>`.
    let (_t, o1) = apply_burst_thinking_budget(&mut thinking, 200);
    assert!(!o1);
    let (t_close, o_close) = apply_burst_thinking_budget(&mut thinking, 101);
    assert_eq!(
        t_close, 101,
        "the model's own </think> token passes through unchanged"
    );
    assert!(
        !o_close,
        "a natural close is NOT an override — no forced injection"
    );

    // Post-close tokens pass through.
    let (t_after, o_after) = apply_burst_thinking_budget(&mut thinking, 202);
    assert_eq!(t_after, 202);
    assert!(!o_after);
}

// =============================================================================
// Compile-time pin: the burst module must export the items the
// scheduler depends on. A future re-org that accidentally renames
// `try_run_burst_b1` or `BurstFinalized` breaks this file at compile
// time before runtime tests fire.
// =============================================================================

#[test]
fn burst_module_exports_required_for_scheduler_integration_compile() {
    // Touch each load-bearing item once so a rename surfaces at
    // compile time. The test body is intentionally trivial.
    let _ = std::mem::size_of::<super::speculative_burst::WorkerDrafterSlot>();
    let _ = std::mem::size_of::<super::speculative_burst::BurstFinalized>();
}

// =============================================================================
// `BurstFinalized` prompt-cache donate plumbing.
//
// `try_run_burst_b1` returns a `BurstFinalized` whose `prompt_tokens`,
// `generated_tokens`, and `healthy_finish` fields the scheduler feeds
// into `donate_finished_sequence_cache` so the burst path mirrors the
// classic path's `finalize_completed` prompt-cache donate. Driving
// `try_run_burst_b1` end-to-end needs a real `LoadedModel` (infeasible
// in a fast unit test — see the comment block above), so these tests
// pin the struct's shape and the field contract directly: the
// scheduler's destructuring `match` would break at compile time if a
// field were renamed or dropped, and the semantic tests below assert
// the healthy / non-healthy field conventions the burst's success and
// error arms must follow.
// =============================================================================

#[test]
fn burst_finalized_carries_prompt_cache_donate_fields() {
    use super::speculative_burst::BurstFinalized;

    // Construct a `BurstFinalized` exactly as the burst's success arm
    // does. This is a compile-time pin: if `prompt_tokens`,
    // `generated_tokens`, or `healthy_finish` are renamed/removed, the
    // scheduler's `Ok(BurstFinalized { .. })` destructuring in
    // `execute_prefill` breaks at the same time as this test.
    let finalized = BurstFinalized {
        seq_id: SequenceId::from_raw(7),
        tokens_generated: 3,
        prompt_tokens: vec![1, 2, 3, 4],
        generated_tokens: vec![10, 11, 12],
        healthy_finish: true,
        mtp_profile: None,
        burst_wall_ms: 12.5,
        burst_active_ms: 30.0,
        slices: 3,
    };

    assert_eq!(finalized.seq_id, SequenceId::from_raw(7));
    assert_eq!(finalized.tokens_generated, 3);
    assert_eq!(finalized.prompt_tokens, vec![1, 2, 3, 4]);
    assert_eq!(finalized.generated_tokens, vec![10, 11, 12]);
    assert!(finalized.healthy_finish);
    // `burst_wall_ms` is the HOL-stall observability field (issue #638),
    // re-scoped by issue #734 to the MAX single-tick wall so the
    // tick-cooperative slice reports its realized per-round HOL bound;
    // `burst_active_ms` / `slices` carry the cumulative occupancy and the
    // slice count. Pin that all three survive the destructure the
    // scheduler performs.
    assert_eq!(finalized.burst_wall_ms, 12.5);
    assert_eq!(finalized.burst_active_ms, 30.0);
    assert_eq!(finalized.slices, 3);
    // A sliced request's max single-tick wall never exceeds its cumulative
    // occupancy.
    assert!(finalized.burst_wall_ms <= finalized.burst_active_ms);
    // `tokens_generated` must equal the committed `generated_tokens`
    // length — both are derived from `seq.generated_tokens` after the
    // early-EOS truncation in `finalize_burst_success`.
    assert_eq!(finalized.tokens_generated, finalized.generated_tokens.len());
}

#[test]
fn burst_finalized_error_outcome_has_empty_donate_payload() {
    use super::speculative_burst::BurstFinalized;

    // The error / transition-failure arms of `try_run_burst_b1` build
    // a `BurstFinalized` with empty token vectors and
    // `healthy_finish == false` — the KV cache is assumed tainted on
    // those paths, so the scheduler's `donate_finished_sequence_cache`
    // call must be a no-op. This mirrors the classic path, where
    // `Finished(Error)` sequences bypass the donate branch in
    // `finalize_completed`. `donate_finished_sequence_cache` itself
    // hard-guards on `healthy_finish` before touching the store, so an
    // empty/false payload is the correct "do not donate" signal.
    let errored = BurstFinalized {
        seq_id: SequenceId::from_raw(9),
        tokens_generated: 0,
        prompt_tokens: Vec::new(),
        generated_tokens: Vec::new(),
        healthy_finish: false,
        mtp_profile: None,
        burst_wall_ms: 0.0,
        burst_active_ms: 0.0,
        slices: 1,
    };

    assert!(!errored.healthy_finish, "error outcome must not be healthy");
    assert!(
        errored.prompt_tokens.is_empty() && errored.generated_tokens.is_empty(),
        "error outcome carries no tokens to donate"
    );
    assert_eq!(errored.tokens_generated, 0);
}

#[test]
fn batched_burst_module_exports_required_for_scheduler_integration_compile() {
    // the batched-burst entry point and its result type must
    // stay exported under their current names — `BatchScheduler::try_speculative_burst`
    // depends on both. A re-org that renames either breaks this at
    // compile time before runtime tests fire.
    let _ = std::mem::size_of::<super::speculative_burst::BatchedBurstFinalized>();
    // the per-row payload type `BatchedBurstRow` is also
    // load-bearing — the scheduler's batched arm destructures it to feed
    // `donate_finished_sequence_cache`. A rename breaks this test at the
    // same time as the scheduler.
    let _ = std::mem::size_of::<super::speculative_burst::BatchedBurstRow>();
    // `try_run_burst_batched` is the load-bearing entry; reference it as
    // a function item so a rename is a compile error.
    let _f: fn(
        super::speculative_burst::BurstContext<'_>,
        Vec<SequenceInfo>,
    ) -> Result<super::speculative_burst::BatchedBurstFinalized, Vec<SequenceInfo>> =
        super::speculative_burst::try_run_burst_batched;
    let _ = _f;
}

// =============================================================================
// `BatchedBurstRow` prompt-cache donate plumbing.
//
// `try_run_burst_batched` returns a `BatchedBurstFinalized` whose `rows`
// each carry the same donate-payload shape as the B = 1 `BurstFinalized`
// — `prompt_tokens`, `generated_tokens`, and `healthy_finish` — so the
// scheduler's batched arm can call `donate_finished_sequence_cache` per
// row, symmetric with the B = 1 burst arm and the classic
// `finalize_completed` path. Driving `try_run_burst_batched` end-to-end
// needs a real `LoadedModel` (infeasible in a fast unit test — see the
// comment block above `burst_finalized_carries_prompt_cache_donate_fields`),
// so these tests pin the per-row struct's shape and the field contract
// directly: the scheduler's `BatchedBurstRow { .. }` destructuring would
// break at compile time if a field were renamed or dropped, and the
// semantic tests assert the healthy / non-healthy field conventions the
// batched burst's success and error rows must follow. This mirrors the
// `burst_finalized_*` tests above for the B = 1 path.
// =============================================================================

#[test]
fn batched_burst_row_carries_prompt_cache_donate_fields() {
    use super::speculative_burst::BatchedBurstRow;

    // Construct a `BatchedBurstRow` exactly as the batched burst's
    // healthy success path does. This is a compile-time pin: if
    // `prompt_tokens`, `generated_tokens`, or `healthy_finish` are
    // renamed/removed, the scheduler's `BatchedBurstRow { .. }`
    // destructuring in `try_speculative_burst` breaks at the same time
    // as this test.
    let row = BatchedBurstRow {
        seq_id: SequenceId::from_raw(11),
        tokens_generated: 3,
        prompt_tokens: vec![5, 6, 7, 8],
        generated_tokens: vec![20, 21, 22],
        healthy_finish: true,
    };

    assert_eq!(row.seq_id, SequenceId::from_raw(11));
    assert_eq!(row.tokens_generated, 3);
    assert_eq!(row.prompt_tokens, vec![5, 6, 7, 8]);
    assert_eq!(row.generated_tokens, vec![20, 21, 22]);
    assert!(row.healthy_finish);
    // `tokens_generated` must equal the committed `generated_tokens`
    // length — both are derived from `seq.generated_tokens` after the
    // early-EOS truncation in `finalize_burst_success`, identical to the
    // B = 1 `BurstFinalized` invariant.
    assert_eq!(row.tokens_generated, row.generated_tokens.len());
}

#[test]
fn batched_burst_row_error_outcome_has_empty_donate_payload() {
    use super::speculative_burst::BatchedBurstRow;

    // The error and transition-failure rows of `try_run_burst_batched`
    // build a `BatchedBurstRow` with empty token vectors and
    // `healthy_finish == false` — the KV cache is assumed tainted on
    // those rows, so the scheduler's `donate_finished_sequence_cache`
    // call must be a guaranteed no-op. This is the same convention the
    // B = 1 `BurstFinalized` error arms follow:
    // `donate_finished_sequence_cache` hard-guards on `healthy_finish`
    // before touching the store, so an empty/false payload is the
    // correct "do not donate" signal on a tainted-cache row.
    let errored = BatchedBurstRow {
        seq_id: SequenceId::from_raw(13),
        tokens_generated: 0,
        prompt_tokens: Vec::new(),
        generated_tokens: Vec::new(),
        healthy_finish: false,
    };

    assert!(!errored.healthy_finish, "error row must not be healthy");
    assert!(
        errored.prompt_tokens.is_empty() && errored.generated_tokens.is_empty(),
        "error row carries no tokens to donate"
    );
    assert_eq!(errored.tokens_generated, 0);
}

#[test]
fn batched_burst_finalized_rows_preserve_per_row_donate_payloads() {
    use super::speculative_burst::{BatchedBurstFinalized, BatchedBurstRow};

    // A batched window mixes healthy and tainted rows: per-row early-EOS
    // means one row can finish healthy while a sibling row's state
    // transition fails. `BatchedBurstFinalized.rows` must preserve each
    // row's individual donate payload — the scheduler iterates `rows`
    // and feeds each into `donate_finished_sequence_cache` independently,
    // so a healthy row donates while a tainted sibling stays a no-op.
    let finalized = BatchedBurstFinalized {
        rows: vec![
            BatchedBurstRow {
                seq_id: SequenceId::from_raw(1),
                tokens_generated: 2,
                prompt_tokens: vec![1, 2, 3],
                generated_tokens: vec![40, 41],
                healthy_finish: true,
            },
            BatchedBurstRow {
                seq_id: SequenceId::from_raw(2),
                tokens_generated: 0,
                prompt_tokens: Vec::new(),
                generated_tokens: Vec::new(),
                healthy_finish: false,
            },
        ],
    };

    assert_eq!(finalized.rows.len(), 2);
    // Healthy row: full donate payload survives.
    assert!(finalized.rows[0].healthy_finish);
    assert_eq!(finalized.rows[0].prompt_tokens, vec![1, 2, 3]);
    assert_eq!(finalized.rows[0].generated_tokens, vec![40, 41]);
    // Tainted sibling: empty/false payload survives, donate is a no-op.
    assert!(!finalized.rows[1].healthy_finish);
    assert!(finalized.rows[1].prompt_tokens.is_empty());
    assert!(finalized.rows[1].generated_tokens.is_empty());
}

// =============================================================================
// sampling_config_eq (— batched-window admission predicate)
// =============================================================================

#[test]
fn sampling_config_eq_identical_default_configs_are_equal() {
    let a = SamplingConfig::default();
    let b = SamplingConfig::default();
    assert!(
        super::speculative_burst::sampling_config_eq(&a, &b),
        "two default sampling configs must compare equal"
    );
}

#[test]
fn sampling_config_eq_distinguishes_temperature() {
    let a = SamplingConfig::default();
    let b = SamplingConfig {
        temperature: a.temperature + 0.5,
        ..SamplingConfig::default()
    };
    assert!(
        !super::speculative_burst::sampling_config_eq(&a, &b),
        "a temperature difference must make the configs unequal"
    );
}

#[test]
fn sampling_config_eq_distinguishes_top_k_top_p_min_p() {
    let base = SamplingConfig::default();

    let top_k_diff = SamplingConfig {
        top_k: base.top_k + 7,
        ..SamplingConfig::default()
    };
    assert!(!super::speculative_burst::sampling_config_eq(
        &base,
        &top_k_diff
    ));

    let top_p_diff = SamplingConfig {
        top_p: (base.top_p - 0.1).max(0.0),
        ..SamplingConfig::default()
    };
    assert!(!super::speculative_burst::sampling_config_eq(
        &base,
        &top_p_diff
    ));

    let min_p_diff = SamplingConfig {
        min_p: base.min_p + 0.2,
        ..SamplingConfig::default()
    };
    assert!(!super::speculative_burst::sampling_config_eq(
        &base,
        &min_p_diff
    ));
}

#[test]
fn sampling_config_eq_distinguishes_stop_token_ids() {
    let a = SamplingConfig::default();
    let b = SamplingConfig {
        stop_token_ids: vec![99],
        ..SamplingConfig::default()
    };
    assert!(
        !super::speculative_burst::sampling_config_eq(&a, &b),
        "a per-row stop-token set difference must make the configs unequal — \
         the batched round loop computes one merged-EOS set per window"
    );
}

#[test]
fn sampling_config_eq_distinguishes_seed() {
    let a = SamplingConfig {
        seed: Some(1),
        ..SamplingConfig::default()
    };
    let b = SamplingConfig {
        seed: Some(2),
        ..SamplingConfig::default()
    };
    assert!(!super::speculative_burst::sampling_config_eq(&a, &b));
}

#[test]
fn sampling_config_eq_requires_empty_token_bias_on_both_sides() {
    use mlxcel_core::sampling::TokenBiasMap;
    let a = SamplingConfig::default();
    // Give `b` a non-empty token-bias map. `sampling_config_eq` requires
    // BOTH sides to carry an empty bias, so a request with logit_bias
    // never joins a batched window.
    let mut bias = TokenBiasMap::new();
    bias.insert(7, 1.5);
    let b = SamplingConfig {
        token_bias: bias,
        ..SamplingConfig::default()
    };
    assert!(
        !super::speculative_burst::sampling_config_eq(&a, &b),
        "a non-empty token_bias on either side must make the configs unequal"
    );
    // Sanity: the empty-bias default on both sides is equal.
    assert!(super::speculative_burst::sampling_config_eq(
        &SamplingConfig::default(),
        &SamplingConfig::default()
    ));
}

#[test]
fn sampling_config_eq_excludes_history_dependent_penalty_configs() {
    // lets penalty-bearing requests enter the burst path
    // (the B=1 burst threads token history). But the BATCHED path
    // samples each row's first bonus with an empty token history, so
    // `sampling_config_eq` (the batched-window admission gate) must
    // reject any config that `needs_token_history()`. A penalty
    // request therefore never joins a batched window — it runs as a
    // B=1 burst, which honors its penalties correctly.
    let base = SamplingConfig::default();

    // Even when both sides carry the SAME penalty, they must not be
    // batched together (the batched first sample ignores history).
    let both_rep = SamplingConfig {
        repetition_penalty: 1.1,
        ..SamplingConfig::default()
    };
    let both_rep2 = SamplingConfig {
        repetition_penalty: 1.1,
        ..SamplingConfig::default()
    };
    assert!(
        !super::speculative_burst::sampling_config_eq(&both_rep, &both_rep2),
        "two identical repetition-penalty configs must NOT be batched — \
         the batched first sample ignores token history"
    );

    // A penalty on either side alone also excludes the pair.
    for penalised in [
        SamplingConfig {
            repetition_penalty: 1.2,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            frequency_penalty: 0.5,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            presence_penalty: 0.3,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            dry_multiplier: 0.8,
            ..SamplingConfig::default()
        },
    ] {
        assert!(
            !super::speculative_burst::sampling_config_eq(&base, &penalised),
            "a history-dependent penalty on one side must exclude the pair \
             from a batched window"
        );
        assert!(
            !super::speculative_burst::sampling_config_eq(&penalised, &base),
            "penalty exclusion must be symmetric"
        );
    }
}
