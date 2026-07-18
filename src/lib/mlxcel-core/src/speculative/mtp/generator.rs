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

//! [`MtpGenerator`] — round-loop driver for Gemma 4 MTP speculative decoding
//!
//! Top-level lifecycle (single-batch path):
//!
//! 1. **Prefill.** Call [`MtpTarget::forward_prefill`] over the whole prompt
//!    to populate the target's KV cache. Sample the first bonus token from
//!    the prefill's last-position logits (handled by the caller's
//!    sampler).
//! 2. **Seed.** Call [`MtpTarget::seed_verify`] with the bonus token to
//!    capture the initial hidden + shared K/V slabs. Bind the drafter and
//!    arm its `set_shared_kv` with the seed slabs.
//! 3. **Round loop.** While more tokens remain:
//!    a. Drafter produces `K-1` proposals via
//!    [`crate::drafter::Drafter::draft_block`] (autoregressive, with
//!    RoPE queries frozen at the bonus token's absolute position).
//!    b. Target verify on `[bonus, draft_0, …, draft_{K-2}]` via
//!    [`MtpTarget::verify`] — produces `target_tokens`, the next
//!    hidden, and the re-sliced shared K/V.
//!    c. Compare draft vs target via [`super::speculative_walk`].
//!    d. Emit `new_tokens`. Update `bonus` to the last emitted token.
//!    e. Rebind the drafter against the new shared K/V via
//!    [`crate::drafter::Drafter::set_shared_kv`].
//! 4. **Termination.** Loop exits when an emitted token is in the
//!    target's `eos_token_ids()` or `emitted >= max_tokens`.

use crate::drafter::{Drafter, SharedKv};
use crate::generate::{GenerationStats, SamplingConfig};
use crate::generation_policy::merged_eos_token_ids;
use crate::sampling::{LogprobsConfig, TokenLogprobData};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::adaptive::effective_mtp_block_size;
use super::target::{MtpTarget, MtpVerifyOutput};
use super::walk::speculative_walk;

#[derive(Debug, Default)]
struct MtpRoundDiagnostics {
    rounds: usize,
    proposed_tokens: usize,
    accepted_draft_tokens: usize,
    emitted_from_verify_tokens: usize,
    zero_accept_rounds: usize,
    partial_accept_rounds: usize,
    full_accept_rounds: usize,
    prefill_seed_ms: f64,
    set_shared_kv_ms: f64,
    draft_ms: f64,
    verify_forward_ms: f64,
    speculative_walk_ms: f64,
    verify_finalize_ms: f64,
    /// Classic-step probe rounds executed (issue #736): drafterless rounds
    /// whose `[1, 1]` verify forward stands in for a classic decode step.
    /// Kept out of `rounds`/acceptance so probes never dilute the
    /// speculative signal.
    probe_rounds: usize,
    /// Cumulative probe verify-forward wall-clock (ms). `probe_ms /
    /// probe_rounds` is the measured classic single-token step time the
    /// adaptive policy's measured-cost estimator divides by.
    probe_ms: f64,
}

impl MtpRoundDiagnostics {
    fn new(prefill_seed_time: Duration) -> Self {
        Self {
            prefill_seed_ms: duration_ms(prefill_seed_time),
            ..Self::default()
        }
    }

    fn record_probe(&mut self, verify_ms: f64) {
        self.probe_rounds += 1;
        self.probe_ms += verify_ms;
    }

    fn record_round(&mut self, proposed_tokens: usize, accepted: usize, emitted_tokens: usize) {
        self.rounds += 1;
        self.proposed_tokens += proposed_tokens;
        self.accepted_draft_tokens += accepted.min(proposed_tokens);
        self.emitted_from_verify_tokens += emitted_tokens;
        if accepted == 0 {
            self.zero_accept_rounds += 1;
        } else if accepted >= proposed_tokens {
            self.full_accept_rounds += 1;
        } else {
            self.partial_accept_rounds += 1;
        }
    }

    fn acceptance_rate(&self) -> f64 {
        if self.proposed_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.proposed_tokens as f64
        }
    }

    fn emitted_per_verify(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.emitted_from_verify_tokens as f64 / self.rounds as f64
        }
    }

    /// Project the internal per-round diagnostics into the public
    /// [`MtpAcceptanceSummary`] the server-side adaptive MTP policy consumes.
    /// Carries aggregate round counts and the draft/verify wall-clock split
    /// only; no prompt data and nothing request-identifying.
    fn summary(&self) -> MtpAcceptanceSummary {
        MtpAcceptanceSummary {
            rounds: self.rounds,
            proposed_tokens: self.proposed_tokens,
            accepted_draft_tokens: self.accepted_draft_tokens,
            draft_ms: self.draft_ms,
            verify_forward_ms: self.verify_forward_ms,
            overhead_ms: self.speculative_walk_ms + self.verify_finalize_ms + self.set_shared_kv_ms,
            probe_rounds: self.probe_rounds,
            probe_ms: self.probe_ms,
        }
    }

    fn log(
        &self,
        block_size: usize,
        prompt_tokens: usize,
        generated_tokens: usize,
        decode_time: Duration,
    ) {
        tracing::info!(
            block_size,
            prompt_tokens,
            generated_tokens,
            rounds = self.rounds,
            proposed_tokens = self.proposed_tokens,
            accepted_draft_tokens = self.accepted_draft_tokens,
            acceptance_rate = self.acceptance_rate(),
            emitted_from_verify_tokens = self.emitted_from_verify_tokens,
            emitted_per_verify = self.emitted_per_verify(),
            zero_accept_rounds = self.zero_accept_rounds,
            partial_accept_rounds = self.partial_accept_rounds,
            full_accept_rounds = self.full_accept_rounds,
            prefill_seed_ms = self.prefill_seed_ms,
            set_shared_kv_ms = self.set_shared_kv_ms,
            draft_ms = self.draft_ms,
            verify_forward_ms = self.verify_forward_ms,
            speculative_walk_ms = self.speculative_walk_ms,
            verify_finalize_ms = self.verify_finalize_ms,
            probe_rounds = self.probe_rounds,
            probe_ms = self.probe_ms,
            decode_ms = duration_ms(decode_time),
            "MTP round-loop diagnostics",
        );
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Coarse per-run acceptance + latency summary surfaced from a single
/// [`MtpGenerator::generate`] call.
///
/// This is the public projection of the generator's internal
/// `MtpRoundDiagnostics`: only the fields the server-side adaptive MTP policy
/// (`crate::server::batch::mtp_policy`) needs to decide whether the
/// speculative path pays for itself on a given (target, drafter, hardware)
/// pairing. It carries no prompt data and nothing request-identifying, just
/// aggregate round counts and the draft/verify wall-clock split.
///
/// The time fields are cumulative milliseconds across the run's rounds; a
/// consumer divides by [`Self::rounds`] for a per-round mean.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MtpAcceptanceSummary {
    /// Speculative rounds executed (one drafter forward + one verify forward
    /// each). Zero when the request hit EOS on the seed bonus or
    /// `max_tokens == 1`, in which case the speculative path produced no
    /// timing signal.
    pub rounds: usize,
    /// Total draft tokens proposed across all rounds.
    pub proposed_tokens: usize,
    /// Total draft tokens accepted by the target's argmax walk across all
    /// rounds. The realized acceptance length per round is
    /// `accepted_draft_tokens / rounds`.
    pub accepted_draft_tokens: usize,
    /// Cumulative drafter `draft_block` wall-clock time (ms) across all rounds.
    pub draft_ms: f64,
    /// Cumulative target verify-forward wall-clock time (ms) across all rounds.
    pub verify_forward_ms: f64,
    /// Cumulative non-forward round overhead (ms) across the run: the
    /// speculative walk, the verify finalize (cache rollback + shared-KV
    /// slicing), and the drafter shared-KV re-arm. Part of the real
    /// per-round cost the measured-cost policy estimator charges
    /// (issue #736).
    pub overhead_ms: f64,
    /// Classic-step probe rounds executed while the adaptive policy was
    /// profiling (issue #736). A probe skips the drafter and verifies only
    /// the bonus token, so its `[1, 1]` verify forward is shape-identical to
    /// a classic decode step. Probes emit exactly one greedy token each and
    /// are excluded from `rounds` / acceptance aggregates.
    pub probe_rounds: usize,
    /// Cumulative probe verify-forward wall-clock (ms); `probe_ms /
    /// probe_rounds` is the measured classic single-token step time.
    pub probe_ms: f64,
}

impl MtpAcceptanceSummary {
    /// Fraction of proposed draft tokens the target accepted, in `[0, 1]`.
    /// Returns `0.0` when no tokens were proposed.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.proposed_tokens as f64
        }
    }
}

/// Owned cross-round state of a resumable MTP generation session
/// (issue #734).
///
/// Everything the round loop carries between two rounds lives here, and
/// nothing in here borrows the model: [`MtpVerifyOutput`] owns MLX
/// `UniquePtr` array handles plus plain integers, so the state can be
/// stashed across scheduler ticks while the borrowing target adapter (and
/// the [`MtpGenerator`] wrapping it) is dropped and reconstructed per tick.
/// The per-sequence KV cache itself lives in the model's sequence slot, not
/// here; this state only references it logically via `kv_offset` /
/// `bonus_position`.
///
/// Produced by [`MtpGenerator::begin_session`], advanced by
/// [`MtpGenerator::step_session`], and consumed by
/// [`MtpGenerator::finish_session`].
///
/// Not `Send`: the MLX handles are thread-affine to the worker thread that
/// created them, which is also the only thread that drives the scheduler
/// tick loop.
#[derive(Debug)]
pub struct MtpSessionState {
    /// Latest verify output: seed capture after `begin_session`, then the
    /// post-rollback slabs of the most recent round. The next round re-arms
    /// the drafter from this and reads `next_hidden` for the draft.
    verify_out: MtpVerifyOutput,
    /// Current bonus token (the last emitted token).
    bonus: i32,
    /// Tokens emitted so far, including the first bonus. The session does
    /// not keep the token stream itself: each step hands its new tokens to
    /// the caller, who owns accumulation/streaming.
    emitted_count: usize,
    /// Per-round accepted counts feeding the adaptive block-size controller
    /// ([`effective_mtp_block_size`]). Probe rounds are excluded.
    accept_lens: Vec<f64>,
    /// Classic-step probe rounds still to run (issue #736).
    probes_remaining: usize,
    /// Round diagnostics, accumulated across steps exactly as the
    /// run-to-completion loop accumulates them within one call.
    diagnostics: MtpRoundDiagnostics,
    /// Prefill + seed wall-clock, for the final [`GenerationStats`].
    prefill_time: Duration,
    /// Sum of per-step wall-clocks. Cross-tick gaps between steps are NOT
    /// included, so the reported decode time reflects worker occupancy, not
    /// scheduling delay.
    decode_elapsed: Duration,
    /// Total emission budget, including the first bonus.
    max_tokens: usize,
    /// Prompt length, for stats/diagnostics.
    prompt_len: usize,
    /// Merged EOS set (target EOS + per-request stop tokens).
    eos_tokens: Vec<i32>,
    /// Whether the session reached a terminal condition (EOS, budget,
    /// cancellation, drafter failure, or degenerate block size).
    finished: bool,
}

impl MtpSessionState {
    /// Whether the session reached a terminal condition. Once true,
    /// further [`MtpGenerator::step_session`] calls are no-ops.
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Tokens emitted so far, including the first bonus.
    pub fn emitted_count(&self) -> usize {
        self.emitted_count
    }

    /// Speculative rounds completed so far (probe rounds excluded).
    pub fn rounds(&self) -> usize {
        self.diagnostics.rounds
    }
}

/// Newly emitted tokens of one session slice ([`MtpGenerator::begin_session`]
/// or [`MtpGenerator::step_session`]).
#[derive(Debug)]
pub struct MtpStepOutput {
    /// Tokens emitted by this slice, in order. The first slice carries the
    /// first bonus; a round slice carries the walk's accepted tokens plus
    /// the target bonus. Empty when the slice terminated without emitting
    /// (budget/cancel checks, drafter failure).
    pub new_tokens: Vec<i32>,
    /// Per-token log-probability data, index-aligned 1:1 with
    /// [`Self::new_tokens`]. Empty when logprobs are disabled.
    pub new_logprobs: Vec<Option<TokenLogprobData>>,
    /// Whether the session finished during this slice. Mirrors
    /// [`MtpSessionState::finished`] so callers can stop without touching
    /// the state again.
    pub finished: bool,
}

impl MtpStepOutput {
    fn finished_empty() -> Self {
        Self {
            new_tokens: Vec::new(),
            new_logprobs: Vec::new(),
            finished: true,
        }
    }
}

/// Round-loop driver for Gemma 4 MTP speculative decoding (B=1).
///
/// Generic over `T: MtpTarget` so we get static dispatch into the target's
/// sink-aware forward / rollback hooks (one v-table hop per call would
/// otherwise show up as measurable noise at K=4 and decode-bound rounds).
/// The drafter is held as `Box<dyn Drafter>` so the call site can swap in
/// Gemma 4 assistant / DFlash / future MTP shapes without touching the
/// generator type.
///
/// ## Construction
///
/// Holds the target and drafter by value, plus a sampler and block size.
/// The drafter MUST already have been loaded from disk (e.g. via
/// [`crate::drafter::load_drafter`]) AND bound to the target via
/// [`crate::drafter::Drafter::bind`]. The generator does not own the
/// load or bind path — both happen at the call site before
/// constructing the generator. This avoids cross-trait coupling
/// between [`MtpTarget`] (concrete model wrapper) and the
/// [`crate::generate::LanguageModel`] trait expected by
/// [`crate::drafter::Drafter::bind`].
///
/// ## Lifecycle
///
/// [`MtpGenerator::generate`] is the only public entrypoint. It takes a
/// prompt and `max_tokens`, returns the emitted tokens and stats.
///
/// ## Threading
///
/// Single-threaded by design. The MTP round-loop's drafter ↔ target
/// dependence is too tight for the classic generator's
/// `install_thread_local_default_stream` pattern to help. Future
/// concurrency lands (batched MTP).
pub struct MtpGenerator<T: MtpTarget> {
    target: T,
    drafter: Box<dyn Drafter>,
    /// User-requested ceiling for the verify block length.
    block_size: usize,
    /// Drafter checkpoint's configured block length. User-requested values
    /// above this start here and expand adaptively after high acceptance.
    configured_block_size: usize,
    prefer_requested_block_size: bool,
    /// Coarse acceptance + latency summary of the most recent `generate`
    /// call, surfaced via [`Self::last_acceptance`] for the server's adaptive
    /// MTP policy. `None` until the first run, and reset at the start of every
    /// `generate`. Holds no prompt data.
    last_acceptance: Option<MtpAcceptanceSummary>,
    /// Number of classic-step probe rounds to run at the start of the round
    /// loop (issue #736). Zero (the default) disables probing; the server's
    /// adaptive policy requests a few probes per burst while profiling so the
    /// measured-cost estimator has a classic single-token step time to divide
    /// by. See [`Self::with_profile_probe_rounds`].
    profile_probe_rounds: usize,
}

impl<T: MtpTarget> MtpGenerator<T> {
    /// Construct a new generator.
    ///
    /// `block_size` is `K` — the draft block length. The drafter
    /// produces `K-1` proposals per round; the verify pass takes
    /// `K` tokens (`[bonus, draft_0, …, draft_{K-2}]`). `K=4` is the
    /// upstream Gemma 4 default.
    pub fn new(target: T, drafter: Box<dyn Drafter>, block_size: usize) -> Self {
        assert!(
            block_size >= 2,
            "MtpGenerator: block_size must be >= 2 (block_size=1 produces no draft proposals)",
        );
        let configured_block_size = drafter.configured_block_size().unwrap_or(block_size).max(2);
        let prefer_requested_block_size = drafter.prefer_requested_block_size();
        Self {
            target,
            drafter,
            block_size,
            configured_block_size,
            prefer_requested_block_size,
            last_acceptance: None,
            profile_probe_rounds: 0,
        }
    }

    /// Request `rounds` classic-step probe rounds at the start of the next
    /// [`Self::generate`] call (issue #736). A probe round skips the drafter
    /// and verifies only the bonus token: its `[1, 1]` verify forward is
    /// shape-identical to a classic decode step, so its wall-clock is the
    /// measured classic single-token step time surfaced through
    /// [`MtpAcceptanceSummary::probe_ms`]. Each probe emits exactly one
    /// greedy token (the target argmax), so temperature-0 output stays
    /// byte-identical to classic decode; probes are excluded from the
    /// acceptance and round aggregates. Zero (the default) disables probing.
    #[must_use]
    pub fn with_profile_probe_rounds(mut self, rounds: usize) -> Self {
        self.profile_probe_rounds = rounds;
        self
    }

    /// Block size (K). Test/diagnostic accessor.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Coarse acceptance + latency summary of the most recent [`Self::generate`]
    /// call, or `None` if no speculative round ran (the request hit EOS on the
    /// seed bonus or `max_tokens == 1`).
    ///
    /// Surfaced for the server's adaptive MTP policy, which profiles the first
    /// few requests of a (target, drafter, hardware) pairing to decide whether
    /// the speculative path is worth running. Cleared at the start of every
    /// `generate`, so it always reflects the latest run, and carries no prompt
    /// data.
    pub fn last_acceptance(&self) -> Option<MtpAcceptanceSummary> {
        self.last_acceptance
    }

    /// Target accessor. Test/diagnostic.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Drafter accessor (read-only). Test/diagnostic.
    pub fn drafter(&self) -> &dyn Drafter {
        self.drafter.as_ref()
    }

    /// Consume the generator and return the boxed drafter handle.
    ///
    /// Used by the server-side speculative burst path so a
    /// loaded drafter can be reused across multiple requests on the same
    /// worker thread without re-loading from disk. The caller is
    /// expected to [`Drafter::reset`] the returned handle before the
    /// next burst so per-run drafter state (KV cache, accept counters)
    /// is cleared.
    pub fn into_drafter(self) -> Box<dyn Drafter> {
        self.drafter
    }

    /// Consume the generator and return both the target and the boxed
    /// drafter handle.
    ///
    /// Used by the tick-cooperative slice path (issue #734): the target
    /// adapter borrows the model, so the generator cannot live across
    /// scheduler ticks. The caller reconstructs the adapter + generator
    /// each tick, recovers the drafter here at the end of the tick
    /// WITHOUT resetting it (the [`MtpSessionState`] plus a re-arm at
    /// the next round's start carry the cross-round contract), and only
    /// resets it once the whole session finishes.
    pub fn into_parts(self) -> (T, Box<dyn Drafter>) {
        (self.target, self.drafter)
    }

    /// Run the full MTP generate cycle.
    ///
    /// # Arguments
    ///
    /// - `prompt_tokens`: prompt token ids (non-empty). The generator
    ///   runs prefill through the target's
    ///   [`MtpTarget::prefill_and_seed`] hook, samples the first bonus,
    ///   and immediately enters the round loop.
    /// - `max_tokens`: total cap on emitted tokens, including the first
    ///   bonus.
    /// - `sampling`: sampling config. **Greedy parity requires
    ///   `temperature == 0`** — this is the load-bearing correctness
    ///   gate of the MTP path.
    /// - `token_history`: history-dependent-penalty context forwarded to
    ///   [`MtpTarget::prefill_and_seed`] for the first-bonus sample
    ///   (repetition / frequency / presence / DRY). The server burst
    ///   path passes `initial_token_history(&prompt, ..)` so the first
    ///   bonus is byte-identical to the classic decode path; callers
    ///   with no penalty configured pass `&[]`.
    /// - `cancel`: cooperative-cancellation flag. Checked **once per
    ///   round** (not per token) at the top of the round loop; when set,
    ///   the generator returns early with whatever tokens it has already
    ///   emitted (at minimum the first bonus). The server-side burst
    ///   path passes `&seq.cancelled` so a client disconnect mid-burst
    ///   stops occupying the worker thread. The offline CLI
    ///   path passes `&AtomicBool::new(false)`.
    /// - `logprobs_config`: per-token log-probability capture control.
    ///   When [`LogprobsConfig::enabled`] is false the returned logprobs
    ///   vec is empty (zero-overhead path); when true it carries one
    ///   `Option<TokenLogprobData>` per emitted token, index-aligned
    ///   with the returned `tokens` vec. The server burst path forwards
    ///   this to `finalize_burst_success` so speculative responses carry
    ///   the same `TokenWithLogprobs` payload as the classic decode
    ///   path.
    ///
    /// # Returns
    ///
    /// `(tokens, logprobs, stats)` where `tokens` is the emitted
    /// sequence (including the first bonus at index 0), `logprobs` is
    /// the index-aligned per-token log-probability data (empty when
    /// `logprobs_config.enabled` is false), and `stats` contains the
    /// prefill + decode timing breakdown.
    pub fn generate(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        token_history: &[i32],
        cancel: &AtomicBool,
        logprobs_config: &LogprobsConfig,
    ) -> (Vec<i32>, Vec<Option<TokenLogprobData>>, GenerationStats) {
        assert!(
            !prompt_tokens.is_empty(),
            "MtpGenerator: prompt_tokens must be non-empty",
        );

        // Reset the per-run acceptance summary so `last_acceptance` always
        // reflects this call. The `max_tokens == 0` early return below leaves
        // it `None` (no decode ran); every later exit stamps the round
        // diagnostics' summary via `finish_session`.
        self.last_acceptance = None;

        let prompt_len = prompt_tokens.len();
        if max_tokens == 0 {
            return (
                Vec::new(),
                Vec::new(),
                Self::build_stats(
                    prompt_len,
                    0,
                    std::time::Duration::ZERO,
                    std::time::Duration::ZERO,
                ),
            );
        }

        // Run-to-completion = the resumable session API driven in a tight
        // loop. `begin_session` is the prefill + seed slice; each
        // `step_session` is exactly one speculative round (or one probe
        // round). Because the tick-cooperative server path (issue #734)
        // drives the identical methods, the two paths compute the same
        // rounds, walks, and diagnostics by construction.
        let (mut state, first) = self.begin_session(
            prompt_tokens,
            max_tokens,
            sampling,
            token_history,
            logprobs_config,
        );
        let mut emitted: Vec<i32> = Vec::with_capacity(max_tokens);
        emitted.extend_from_slice(&first.new_tokens);
        // `logprobs` is kept index-aligned with `emitted`: a push to one
        // is always paired with a push to the other. Stays empty (no
        // allocation) when `logprobs_config.enabled` is false.
        let mut logprobs: Vec<Option<TokenLogprobData>> = first.new_logprobs;
        while !state.finished() {
            let out = self.step_session(&mut state, sampling, cancel, logprobs_config);
            emitted.extend_from_slice(&out.new_tokens);
            logprobs.extend(out.new_logprobs);
        }
        let (stats, _summary) = self.finish_session(state);
        (emitted, logprobs, stats)
    }

    /// Start a resumable MTP generation session (issue #734): run the
    /// prefill + seed capture and emit the first bonus token.
    ///
    /// This is "slice 0" of the tick-cooperative server path. The returned
    /// [`MtpSessionState`] owns every cross-round value (the seed
    /// [`MtpVerifyOutput`] holds MLX array handles, not model borrows), so
    /// the caller can drop this generator, stash the state across scheduler
    /// ticks, reconstruct a generator over a fresh target adapter, and
    /// continue with [`Self::step_session`]. The companion [`MtpStepOutput`]
    /// carries the first bonus token (and its logprob when enabled); when
    /// the first bonus is EOS or `max_tokens == 1`, the session is already
    /// finished.
    ///
    /// The drafter is NOT armed here: [`Self::step_session`] re-arms it from
    /// the stored verify output at the start of every round, which is what
    /// makes the session resumable with a drafter that was held (unreset)
    /// between ticks.
    ///
    /// Argument semantics match [`Self::generate`]. `max_tokens` must be
    /// `>= 1` (the `max_tokens == 0` no-op is `generate`'s early return).
    pub fn begin_session(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (MtpSessionState, MtpStepOutput) {
        assert!(
            !prompt_tokens.is_empty(),
            "MtpGenerator: prompt_tokens must be non-empty",
        );
        assert!(
            max_tokens >= 1,
            "MtpGenerator::begin_session: max_tokens must be >= 1",
        );
        self.last_acceptance = None;

        let prompt_len = prompt_tokens.len();
        let eos_tokens =
            merged_eos_token_ids(self.target.eos_token_ids(), &sampling.stop_token_ids);

        // PREFILL + SEED.
        //
        // Combined: prefill the prompt through the target with sinks
        // armed, sample the first bonus from the last-position logits,
        // and capture hidden + shared K/V for the drafter's first
        // round.
        let prefill_start = Instant::now();
        let (first_bonus, verify_out, first_bonus_lp) =
            self.target
                .prefill_and_seed(prompt_tokens, sampling, token_history, logprobs_config);
        let prefill_time = prefill_start.elapsed();
        let diagnostics = MtpRoundDiagnostics::new(prefill_time);

        // Emit the first bonus and short-circuit if it's EOS or
        // max_tokens=1.
        let mut new_logprobs: Vec<Option<TokenLogprobData>> = Vec::new();
        if logprobs_config.enabled {
            new_logprobs.push(first_bonus_lp);
        }
        let finished = eos_tokens.contains(&first_bonus) || max_tokens == 1;
        let state = MtpSessionState {
            verify_out,
            bonus: first_bonus,
            emitted_count: 1,
            accept_lens: Vec::new(),
            probes_remaining: self.profile_probe_rounds,
            diagnostics,
            prefill_time,
            decode_elapsed: Duration::ZERO,
            max_tokens,
            prompt_len,
            eos_tokens,
            finished,
        };
        (
            state,
            MtpStepOutput {
                new_tokens: vec![first_bonus],
                new_logprobs,
                finished,
            },
        )
    }

    /// Run exactly ONE speculative round (or one classic-step probe round,
    /// issue #736) of a session started by [`Self::begin_session`].
    ///
    /// A round is: re-arm the drafter from the stored verify output, draft
    /// `K-1` proposals (skipped for probes), verify-forward
    /// `[bonus, draft_0, …]`, walk, verify-finalize (cache rollback +
    /// shared-KV re-slice), and emit the accepted tokens. The returned
    /// [`MtpStepOutput`] carries this round's newly emitted tokens (with
    /// logprobs when enabled) and whether the session finished (EOS,
    /// budget, cancellation, or drafter failure).
    ///
    /// Calling this after the session finished is a no-op that returns an
    /// empty, finished output.
    ///
    /// ## Resumability contract (issue #734)
    ///
    /// This method touches only `self` (target + drafter + config) and
    /// `state`. The generator may be a FRESH instance over a reconstructed
    /// target adapter, as long as (a) the underlying model's per-sequence KV
    /// slot is untouched since the previous round, (b) the drafter is a
    /// bound handle whose draft-relevant state was not destroyed since the
    /// previous round: held WITHOUT [`Drafter::reset`], or reset by an
    /// implementation whose reset provably preserves it (the MTP assistant
    /// drafter's reset is the trait default no-op, which is what lets the
    /// server rotate one drafter handle across sessions between rounds,
    /// issue #746), and (c) the generator config (`block_size`, probe
    /// count) is the same. The re-arm at the top of the round
    /// re-establishes the drafter's shared-KV binding from `state`'s stored
    /// [`MtpVerifyOutput`], so no drafter-side state needs to survive
    /// between rounds.
    ///
    /// The per-round wall-clock is accumulated into the session's decode
    /// time, so cross-tick gaps between rounds never inflate the reported
    /// decode stats.
    pub fn step_session(
        &mut self,
        state: &mut MtpSessionState,
        sampling: &SamplingConfig,
        cancel: &AtomicBool,
        logprobs_config: &LogprobsConfig,
    ) -> MtpStepOutput {
        if state.finished {
            return MtpStepOutput::finished_empty();
        }
        let step_start = Instant::now();
        let out = self.run_one_round(state, sampling, cancel, logprobs_config);
        state.decode_elapsed += step_start.elapsed();
        out
    }

    /// Finish a session: log the round diagnostics and stamp
    /// [`Self::last_acceptance`], exactly as a completed
    /// [`Self::generate`] call does. Returns the run's
    /// [`GenerationStats`] plus the acceptance summary the server's
    /// adaptive MTP policy consumes.
    ///
    /// Must be called exactly once per session, after the session
    /// reports finished (or when the caller abandons it early, e.g. on
    /// cancellation).
    pub fn finish_session(
        &mut self,
        state: MtpSessionState,
    ) -> (GenerationStats, Option<MtpAcceptanceSummary>) {
        state.diagnostics.log(
            self.block_size,
            state.prompt_len,
            state.emitted_count,
            state.decode_elapsed,
        );
        let summary = state.diagnostics.summary();
        self.last_acceptance = Some(summary);
        (
            Self::build_stats(
                state.prompt_len,
                state.emitted_count,
                state.prefill_time,
                state.decode_elapsed,
            ),
            Some(summary),
        )
    }

    /// One round of the session round-loop. See [`Self::step_session`] for
    /// the contract; this private body is the former `generate` loop body,
    /// restructured so every cross-round value lives in `state`.
    fn run_one_round(
        &mut self,
        state: &mut MtpSessionState,
        sampling: &SamplingConfig,
        cancel: &AtomicBool,
        logprobs_config: &LogprobsConfig,
    ) -> MtpStepOutput {
        if state.emitted_count >= state.max_tokens {
            state.finished = true;
            return MtpStepOutput::finished_empty();
        }
        // Cooperative cancellation: checked once per round (cheap,
        // a single relaxed atomic load), not per token. On a client
        // disconnect the server flips `seq.cancelled` and the session
        // finishes with the tokens emitted so far rather than running
        // the full `max_tokens` budget.
        if cancel.load(Ordering::Relaxed) {
            state.finished = true;
            return MtpStepOutput::finished_empty();
        }
        // [#736] Classic-step probe rounds. While the adaptive policy is
        // profiling, the first `profile_probe_rounds` rounds skip the
        // drafter and verify only the bonus token: the resulting `[1, 1]`
        // verify forward is shape-identical to a classic decode step, so
        // its wall-clock is the measured classic step time the policy's
        // measured-cost estimator divides by. A probe emits exactly one
        // greedy token (the walk over an empty draft returns the target
        // argmax), so temperature-0 output stays byte-identical to
        // classic decode. Probes are recorded separately from the
        // speculative aggregates below.
        let is_probe = state.probes_remaining > 0;
        let draft_tokens: Vec<i32> = if is_probe {
            state.probes_remaining -= 1;
            Vec::new()
        } else {
            // Bound the block size by the remaining budget. When the
            // operator requested a block larger than the drafter's
            // configured depth, mirror upstream's adaptive controller:
            // stay at configured depth until recent acceptance proves the
            // configured prefix is usually fully accepted, then expand to
            // the requested ceiling. The `+1` is because the verify input
            // is `[bonus, draft_0, …, draft_{K-2}]`: one prefix bonus
            // position that the round-loop already counts as emitted.
            let remaining = state.max_tokens - state.emitted_count + 1;
            let bs = if self.prefer_requested_block_size {
                self.block_size.min(remaining)
            } else {
                effective_mtp_block_size(
                    self.block_size,
                    self.configured_block_size,
                    &state.accept_lens,
                    remaining,
                )
            };
            if bs <= 1 {
                state.finished = true;
                return MtpStepOutput::finished_empty();
            }

            // Arm the drafter's shared K/V from the stored verify output
            // (the seed capture on round 1, the previous round's
            // post-rollback slabs afterwards). Running the arm at the
            // START of the round, rather than at the end of the previous
            // one, is what makes the session resumable across scheduler
            // ticks: a drafter held (unreset) between ticks is re-armed
            // here from `state` alone, and the data is identical to the
            // legacy end-of-round arm because `set_shared_kv` is a
            // stateless overwrite consumed only by the `draft_block`
            // below. Probe rounds never consult the drafter, so they
            // skip the arm entirely.
            let set_shared_start = Instant::now();
            self.set_shared_kv_from_verify(&state.verify_out);
            state.diagnostics.set_shared_kv_ms += duration_ms(set_shared_start.elapsed());

            // Drafter produces K-1 proposals. Pass the bonus + last
            // hidden captured from the previous verify pass (or the
            // seed verify on the first round).
            let hidden = state.verify_out.next_hidden.as_ref();
            let draft_start = Instant::now();
            match self.drafter.draft_block(state.bonus, hidden, bs, sampling) {
                Ok(t) => {
                    state.diagnostics.draft_ms += duration_ms(draft_start.elapsed());
                    t
                }
                Err(e) => {
                    state.diagnostics.draft_ms += duration_ms(draft_start.elapsed());
                    // Drafter failed; bail out cleanly. We've already
                    // emitted at least the seed bonus, so finish with what
                    // we have rather than panicking. Future hardening
                    // can surface this through GenerationStats.
                    let _ = e;
                    state.finished = true;
                    return MtpStepOutput::finished_empty();
                }
            }
        };

        // Verify input = [bonus, draft_0, ..., draft_{K-2}].
        // If the drafter returned fewer than bs-1 proposals (e.g.
        // it short-circuited on a non-greedy path), `bs` shrinks
        // accordingly so the verify shape stays consistent.
        let actual_bs = draft_tokens.len() + 1;
        let mut verify_input = Vec::with_capacity(actual_bs);
        verify_input.push(state.bonus);
        verify_input.extend_from_slice(&draft_tokens);

        // Phase 1: sink-aware forward. The target's KV cache is now
        // `bs` longer than before the call. We have target_tokens
        // for the walk; the captured state holds hidden + shared
        // K/V slabs for the finalize step.
        let verify_forward_start = Instant::now();
        let forward_out = self
            .target
            .verify_forward(&verify_input, sampling, logprobs_config);
        let verify_elapsed_ms = duration_ms(verify_forward_start.elapsed());

        // Walk the draft against the target's argmax tokens.
        let budget = state.max_tokens - state.emitted_count;
        let walk_start = Instant::now();
        let walk = speculative_walk(&draft_tokens, &forward_out.target_tokens, budget);
        if !is_probe {
            state.diagnostics.speculative_walk_ms += duration_ms(walk_start.elapsed());
        }
        if is_probe {
            // Probe rounds stand in for classic decode steps: their
            // verify time is the classic-step signal and they stay out
            // of the speculative round/acceptance aggregates (and out of
            // `accept_lens`, so the adaptive block controller's recent
            // window only sees real speculative rounds).
            state.diagnostics.record_probe(verify_elapsed_ms);
        } else {
            state.diagnostics.verify_forward_ms += verify_elapsed_ms;
            state.diagnostics.record_round(
                draft_tokens.len(),
                walk.accepted,
                walk.new_tokens.len(),
            );
            state.accept_lens.push(walk.accepted as f64);
        }

        // Phase 2: rollback the cache + slice shared K/V based on
        // the walk's accepted count. This consumes the captured
        // state from phase 1. `target_logprobs` is pulled out
        // *before* `forward_out` is moved into `verify_finalize`.
        let target_logprobs = forward_out.target_logprobs;
        let verify_finalize_start = Instant::now();
        state.verify_out =
            self.target
                .verify_finalize(walk.accepted, actual_bs, forward_out.captured);
        if !is_probe {
            state.diagnostics.verify_finalize_ms += duration_ms(verify_finalize_start.elapsed());
        }

        // Emit accepted tokens. `walk.new_tokens[i] == target_tokens[i]`
        // for every `i` (accepted draft tokens matched the target by
        // construction; the final entry is the target's bonus), so
        // `target_logprobs[i]` is the correct log-probability for
        // `walk.new_tokens[i]`.
        let mut new_tokens: Vec<i32> = Vec::with_capacity(walk.new_tokens.len());
        let mut new_logprobs: Vec<Option<TokenLogprobData>> = Vec::new();
        for (i, &tok) in walk.new_tokens.iter().enumerate() {
            new_tokens.push(tok);
            state.emitted_count += 1;
            if logprobs_config.enabled {
                // Defensive `.get(i)`: `target_logprobs` is aligned
                // 1:1 with `target_tokens` (length `actual_bs`) and
                // `walk.new_tokens.len() <= actual_bs`, so `i` is
                // always in range, but a missing entry degrades to
                // `None` rather than panicking.
                let lp = target_logprobs.as_ref().and_then(|v| v.get(i).cloned());
                new_logprobs.push(lp);
            }
            if state.eos_tokens.contains(&tok) || state.emitted_count >= state.max_tokens {
                state.finished = true;
                return MtpStepOutput {
                    new_tokens,
                    new_logprobs,
                    finished: true,
                };
            }
        }

        // Next round's bonus is the last emitted token. `walk.new_tokens`
        // always carries at least the target's own token at position 0
        // (the budget is >= 1 at the top of the round), so the fallback
        // to the previous bonus is defensive only.
        state.bonus = *new_tokens.last().unwrap_or(&state.bonus);

        MtpStepOutput {
            new_tokens,
            new_logprobs,
            finished: false,
        }
    }

    /// Wire the verify output's `next_shared_kv` into the drafter's
    /// `set_shared_kv`.
    ///
    /// Best-effort: the round-loop continues even if the drafter
    /// rejects the call (the next `draft_block` will fail closed and
    /// the outer loop bails out). Tests assert the slabs reach the
    /// drafter; real-model integration gates on a stricter
    /// error path via `GenerationStats.errors` or similar.
    fn set_shared_kv_from_verify(&mut self, verify_out: &MtpVerifyOutput) {
        let refs = verify_out.shared_kv_refs();
        let shared_kv = SharedKv::new(&refs);
        let _ = self.drafter.set_shared_kv(
            shared_kv,
            verify_out.kv_offset,
            verify_out.bonus_position,
            /* left_padding */ 0,
        );
    }

    fn build_stats(
        prompt_count: usize,
        gen_count: usize,
        prefill_time: Duration,
        decode_time: Duration,
    ) -> GenerationStats {
        let prefill_ms = duration_ms(prefill_time);
        let decode_ms = duration_ms(decode_time);
        GenerationStats {
            prompt_tokens: prompt_count,
            generated_tokens: gen_count,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            prefill_tok_per_sec: if prefill_ms > 0.0 {
                prompt_count as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            },
            decode_tok_per_sec: if decode_ms > 0.0 {
                gen_count as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}
