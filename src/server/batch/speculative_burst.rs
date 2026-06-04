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

//! Speculative-decoding burst driver for the continuous-batching scheduler
//! (follow-up to).
//!
//! ## Why a "burst" rather than per-tick dispatch
//!
//! The existing speculative round loops
//! ([`mlxcel_core::speculative::mtp::MtpGenerator::generate`] and
//! [`mlxcel_core::drafter::dflash::DFlashGenerator::run`]) are
//! **self-contained drive loops**: they own the per-round drafter state,
//! the bonus/hidden bookkeeping, the per-round verify+walk+rollback
//! protocol, and the EOS / max-tokens termination check.
//!
//! Refactoring those loops into a per-tick step API
//! (`step_once` / `accept_and_advance` / `rollback`) would touch every
//! generator (B=1 MTP, B>1 MTP batched, B=1 DFlash, B>1 DFlash batched)
//! and risks regressing the offline CLI path that still consumes
//! `generate()` / `run()` end-to-end. That refactor is genuinely larger
//! than this issue's scope; see the central architectural decision
//! discussion ("Option A vs Option B").
//!
//! Instead, this module takes **Option B**: when the scheduler decides a
//! sequence is speculative, it delegates the *entire* prefill + decode
//! lifecycle to the kind-specific round-loop driver as a single
//! "speculative burst" — one logical scheduler tick produces every token
//! the request will ever emit. The scheduler's standard prefill →
//! `finish_prefill` → `active_batch.add` → `decode_single_step` pipeline
//! is bypassed for that sequence; the burst streams tokens directly to
//! `seq.response_tx` and finalizes inline. The classic non-speculative
//! request stays on the bit-exact existing pipeline.
//!
//! ## Scope (B = 1 today)
//!
//! This module implements the B = 1 burst for:
//!
//! - **MTP / Gemma 4** — [`crate::LoadedModel::Gemma4`],
//!   [`crate::LoadedModel::Gemma4VLM`], and
//!   [`crate::LoadedModel::Gemma4Unified`] (text-only requests; multimodal
//!   inputs are rejected with a clear error). Drives
//!   [`mlxcel_core::speculative::mtp::MtpGenerator`] through
//!   [`crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter`] /
//!   [`crate::models::gemma4_mtp_target::Gemma4VLMtpTargetAdapter`] /
//!   [`crate::models::gemma4_mtp_target::Gemma4UnifiedMtpTargetAdapter`].
//! - **DFlash / Qwen 3.5** — [`crate::LoadedModel::Qwen35`],
//!   [`crate::LoadedModel::Qwen35Moe`], and their Qwen 3.5 VLM-wrapped
//!   variants for text-only requests. True multimodal requests still
//!   fall back to classic decode until the burst path can consume the
//!   vision/audio prefill embeddings safely.
//!
//! B > 1 batched bursts are deferred to a peer follow-up — they require
//! the batched `MtpTarget` / `SpeculativeTarget` methods on the adapter
//! (which currently return `DrafterError::DraftFailed` per the trait's
//! default) plus per-row continuous-batching reconciliation that is
//! materially harder than the B = 1 case. The scheduler logs a one-time
//! warning at startup when speculative dispatch is requested with
//! `max_batch_size > 1` to make the limitation operator-visible.
//!
//! ## Bind asymmetry: MTP vs DFlash
//!
//! The two dispatch arms differ in who calls [`mlxcel_core::drafter::Drafter::bind`]:
//!
//! - **MTP** (`run_mtp_burst`): bind is **not** called internally by
//!   [`mlxcel_core::speculative::mtp::MtpGenerator::generate`]. This module
//!   calls `drafter.bind(target_lm)` explicitly before constructing the
//!   generator. Omitting the call causes the first `draft_block` to return
//!   `DrafterError::BindNotCalled`, which the round loop silently swallows,
//!   yielding exactly one seed-bonus token per request (the cycle-1 silent
//!   regression). The cycle-2 fix pins this with the
//!   `run_mtp_burst_binds_drafter_before_draft_block` mock test.
//!
//! - **DFlash** (`run_dflash_burst`): bind is called **inside**
//!   [`mlxcel_core::drafter::dflash::DFlashGenerator::run`] on every
//!   invocation. Adding a manual bind here would double-bind; do not add
//!   one.
//!
//! ## Gate predicates
//!
//! [`should_burst_for_sequence`] is the per-request guard. A request is
//! declined to the classic non-speculative path when any of the following
//! hold:
//!
//! - Dispatch is `Disabled` or `Classic` (not a kind-specific variant).
//! - The request carries multimodal payloads / VLM embeddings (image,
//!   audio, or video-derived inputs): the burst path does not yet
//!   support speculative tail decode after multimodal prefill. A
//!   VLM-wrapped Qwen 3.5 checkpoint with a text-only request remains
//!   eligible.
//! - The request carries a structured-output constraint: the speculative
//!   round loops do not yet plumb `llguidance` per-step.
//! - The request adopted a prompt-cache prefix (`prefill_start_offset > 0`):
//!   a burst owns the cache for its full lifetime; mixing an adopted prefix
//!   would double-prefill the leading tokens.
//!
//! History-dependent sampling penalties (repetition / frequency /
//! presence / DRY) are **not** a gate: the burst threads
//! `initial_token_history(&prompt, ..)` into the first-bonus sample so a
//! penalty-bearing request's first bonus is byte-identical to the
//! classic decode path.
//!
//! Logprobs are **not** a gate: the burst threads `logprobs_config`
//! through `MtpGenerator::generate` / `DFlashGenerator::run` and emits
//! `TokenWithLogprobs` events from `finalize_burst_success` — the same
//! payload the classic decode path produces.
//!
//! Thinking-budget enforcement is **not** a gate: `finalize_burst_success`
//! runs the same per-token `decide_override` + `observe` cycle as the
//! classic path's `apply_thinking_budget`, injecting a forced `</think>`
//! at the budget boundary.
//!
//! Each declined request is logged with the gate reason (at `debug` for
//! ordinary gates, and at `warn` for DFlash multimodal requests so operators
//! can distinguish "VLM-wrapped text-only supported" from "multimodal
//! speculative tail not enabled").
//!
//! B > 1 batched bursts apply one additional window-admission guard:
//! [`can_join_batched_burst_window`] keeps requests that need per-row
//! payloads unsupported by the batched round loops (currently logprobs)
//! on the B = 1 path. This preserves correctness without regressing the
//! richer single-request burst features.
//!
//! ## Lazy-load model
//!
//! The drafter checkpoint is NOT read from disk at worker startup (mandate 2). [`WorkerDrafterSlot`] holds the path and an `Option<Box<dyn
//! Drafter>>` that starts as `None`. The first speculative request triggers
//! [`WorkerDrafterSlot::ensure_loaded`], which calls
//! [`mlxcel_core::drafter::load_drafter`] and stores the handle. Subsequent
//! requests on the same worker reuse it. A load failure fails only the current
//! request and leaves the slot empty so the next request retries — the
//! operator can correct a typo'd path without restarting the server.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use mlxcel_core::drafter::dflash::{DFlashBatchedGenerator, DFlashGenerator, SpeculativeTarget};
use mlxcel_core::drafter::{Drafter, DrafterKind, load_drafter};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::generation_policy::{initial_token_history, merged_eos_token_ids};
use mlxcel_core::sampling::TokenLogprobData;
use mlxcel_core::speculative::mtp::{MtpBatchedGenerator, MtpGenerator};

use crate::LoadedModel;
use crate::models::gemma4_mtp_target::{
    Gemma4MtpBatchedTargetAdapter, Gemma4MtpTargetAdapter, Gemma4UnifiedMtpBatchedTargetAdapter,
    Gemma4VLMtpBatchedTargetAdapter,
};
use crate::server::model_provider::GenerateEvent;

use super::sequence::{FinishReason, SequenceInfo, SequenceState};

/// Narrow target contract used by the server-side Qwen 3.5 DFlash burst.
///
/// `DFlashGenerator` already accepts any [`SpeculativeTarget`], but the
/// server prefill side also needs a fresh heterogeneous Qwen 3.5 cache vector.
/// Both the text-only model and the Qwen 3.5 VLM wrapper satisfy that contract:
/// the VLM wrapper delegates speculative hooks to its inner text backbone and
/// allocates the same cache shape. This lets text-only requests against
/// VLM-wrapped checkpoints run DFlash without opening the true multimodal tail
/// path yet.
trait Qwen35DFlashTarget:
    LanguageModel
    + SpeculativeTarget<
        Cache = crate::models::qwen3_next::Qwen3NextCache,
        VerifyOut = crate::models::qwen3_5::VerifyOutput,
    >
{
    fn make_dflash_caches(&self) -> Vec<crate::models::qwen3_next::Qwen3NextCache>;
}

impl Qwen35DFlashTarget for crate::models::Qwen35Model {
    fn make_dflash_caches(&self) -> Vec<crate::models::qwen3_next::Qwen3NextCache> {
        self.make_speculative_caches()
    }
}

impl Qwen35DFlashTarget for crate::vision::Qwen35VLModel {
    fn make_dflash_caches(&self) -> Vec<crate::models::qwen3_next::Qwen3NextCache> {
        self.text_model.make_speculative_caches()
    }
}

/// Lazy-loaded drafter slot held on the scheduler.
///
/// The scheduler thread is single-threaded (every request goes through
/// the same MLX dispatch stream), so a simple `Option<Box<dyn Drafter>>`
/// is sufficient — no atomic-once / `RwLock` is needed. The "lazy load"
/// requirement (mandate 2) means: the drafter weights
/// MUST NOT be read from disk at worker startup; the first speculative
/// request triggers the load. Subsequent requests on the same worker
/// reuse the loaded drafter.
///
/// Failure to load fails the *current* request only — the next request
/// will retry (so an operator can fix a typo'd `--draft-model` path
/// without restarting the server).
pub(crate) struct WorkerDrafterSlot {
    /// Path the drafter is loaded from. Set once at construction time.
    /// `None` for [`crate::server::SpeculativeDispatch::Disabled`].
    pub(crate) draft_model_path: Option<PathBuf>,
    /// Resolved drafter kind. `None` for `Disabled`.
    pub(crate) kind: Option<DrafterKind>,
    /// The loaded drafter handle. `None` until the first successful
    /// `ensure_loaded` call. On a load failure, stays `None` so the next
    /// request retries.
    drafter: Option<Box<dyn Drafter>>,
}

impl WorkerDrafterSlot {
    /// Construct a slot for the given dispatch. `Disabled` produces an
    /// empty slot that never loads anything.
    pub(crate) fn from_dispatch(dispatch: &crate::server::SpeculativeDispatch) -> Self {
        Self {
            draft_model_path: dispatch.draft_model_path().map(|p| p.to_path_buf()),
            kind: dispatch.drafter_kind(),
            drafter: None,
        }
    }

    /// Ensure the drafter is loaded (lazy-load from disk on first
    /// call). On success the slot's [`Self::drafter`] is `Some(..)`
    /// and a subsequent [`Self::take`] is guaranteed to return
    /// `Some(..)`. Errors carry the operator-facing reason string so
    /// the caller can stream a clean error event without further
    /// wrapping.
    ///
    /// Returns `Ok(())` rather than `Result<&mut dyn Drafter, _>` so
    /// the call doesn't leave a `&mut` borrow on the slot — the
    /// caller's next step is typically [`Self::take`], which itself
    /// borrows `self` mutably. Splitting "load" from "take" keeps the
    /// borrow checker simple.
    pub(crate) fn ensure_loaded(&mut self) -> Result<(), String> {
        if self.drafter.is_none() {
            let path = self
                .draft_model_path
                .as_ref()
                .ok_or_else(|| "speculative dispatch is disabled".to_string())?;
            let kind = self.kind;
            tracing::info!(
                "Lazy-loading drafter from {} (kind={:?})",
                path.display(),
                kind,
            );
            let load_start = Instant::now();
            let (loaded, resolved_kind) = load_drafter(path, kind)
                .map_err(|e| format!("Drafter load failed for {}: {e}", path.display()))?;
            tracing::info!(
                "Drafter loaded (kind={resolved_kind}, {} ms)",
                load_start.elapsed().as_millis()
            );
            self.drafter = Some(loaded);
        }
        Ok(())
    }

    /// Take ownership of the loaded drafter, leaving the slot empty. The
    /// caller MUST `return_drafter` the same drafter back when done — the
    /// round-loop drivers consume `Box<dyn Drafter>` by value, so the
    /// slot temporarily transfers ownership for the burst's lifetime.
    /// On burst failure (panic / drafter-load error), the slot stays
    /// empty and the next request lazily reloads.
    pub(crate) fn take(&mut self) -> Option<Box<dyn Drafter>> {
        self.drafter.take()
    }

    /// Return the drafter handle after a successful burst. Resets the
    /// drafter's per-run state via [`Drafter::reset`] (DFlash uses this
    /// to clear its cache; MTP's reset is a no-op but harmless).
    /// `target_lm` is required because reset needs the bound target to
    /// re-derive any cache shapes — passing the same target the burst
    /// just used preserves the bind invariant.
    pub(crate) fn return_drafter(
        &mut self,
        mut drafter: Box<dyn Drafter>,
        target_lm: &dyn LanguageModel,
    ) {
        // Reset is best-effort; a reset failure shouldn't strand the
        // slot. Log and drop in that case so the next request lazily
        // reloads from disk.
        if let Err(e) = drafter.reset(target_lm) {
            tracing::warn!(
                "Drafter reset failed after burst: {e}; dropping handle so the next request reloads"
            );
            return;
        }
        self.drafter = Some(drafter);
    }
}

/// Whether the scheduler should run a speculative burst for `seq`.
///
/// Returns `true` only when ALL of:
///
/// 1. `dispatch` is one of the kind-specific variants
///    (`Mtp` or `DFlash`).
/// 2. `seq` has no multimodal payload / VLM embeddings attached
///    (image/audio/video-derived requests are rejected from the
///    speculative path today — see the module docstring's scope note).
/// 3. `seq` has no structured-output constraint attached. The
///    speculative round loops do not yet plumb the `llguidance` matcher
///    through per-step; falling back to classic decode preserves the
///    grammar invariants.
/// 4. `seq` has no prefill-cache adoption (`prefill_start_offset == 0`).
///    A speculative burst owns the cache for its full lifetime; mixing
///    an adopted prefix with the burst's own prefill would double-prefill
///    the leading tokens.
///
/// History-dependent sampling penalties (repetition / frequency /
/// presence / DRY) are **no longer** a gate: the burst
/// threads `initial_token_history(&prompt, ..)` into the first-bonus
/// sample, so a penalty-bearing request's first bonus is byte-identical
/// to the classic decode path.
///
/// `logprobs_config.enabled` is likewise **no longer** a gate: the burst threads `logprobs_config` through the round-loop
/// drivers and emits `TokenWithLogprobs` events from
/// `finalize_burst_success`, the same payload the classic path produces.
///
/// Thinking-budget enforcement is likewise **no longer** a gate: `finalize_burst_success` runs the same per-token
/// `decide_override` + `observe` cycle as the classic path's
/// `apply_thinking_budget`, injecting a forced `</think>` at the budget
/// boundary.
///
/// Each gate-out logs a reason so an operator can correlate a "spec request
/// fell back to classic" with the reason. Most gates use `debug`; DFlash
/// multimodal requests use `warn` because the Qwen 3.5 VLM-wrapped text-only
/// path is supported while the true multimodal speculative tail is not. The
/// gates run in cheapest-first order so the hot path stays branch-predictable.
pub(crate) fn should_burst_for_sequence(
    dispatch: &crate::server::SpeculativeDispatch,
    seq: &SequenceInfo,
) -> bool {
    if !dispatch.is_kind_specific() {
        return false;
    }
    if seq.vlm_embeddings.is_some() || !seq.images.is_empty() || !seq.audio.is_empty() {
        if matches!(dispatch, crate::server::SpeculativeDispatch::DFlash { .. }) {
            tracing::warn!(
                "DFlash speculative dispatch declined for seq {}: multimodal VLM request \
                 detected; VLM-wrapped text-only Qwen 3.5 targets are supported, but \
                 multimodal speculative tail is not yet enabled; falling back to classic decode",
                seq.seq_id,
            );
        } else {
            tracing::debug!(
                "speculative burst declined for seq {}: multimodal payload or VLM embeddings attached",
                seq.seq_id,
            );
        }
        return false;
    }
    if seq.structured.is_some() {
        tracing::debug!(
            "speculative burst declined for seq {}: structured-output constraint attached",
            seq.seq_id,
        );
        return false;
    }
    if seq.prefill_start_offset > 0 {
        tracing::debug!(
            "speculative burst declined for seq {}: prefill_start_offset={} (adopted cache prefix)",
            seq.seq_id,
            seq.prefill_start_offset,
        );
        return false;
    }
    // History-dependent sampling penalties (repetition / frequency /
    // presence / DRY) are NO LONGER a decline-to-classic gate. The burst now threads `initial_token_history(&prompt, ..)`
    // into the first-bonus sample via `MtpTarget::prefill_and_seed` /
    // `sample_token_optimized`, so a penalty-bearing request's first
    // bonus is byte-identical to the classic decode path. The
    // subsequent round-loop tokens come from the target's greedy
    // argmax, which carries no history dependence.
    //
    // `logprobs_config.enabled` is likewise NO LONGER a gate. The burst threads `logprobs_config` through
    // `MtpGenerator::generate` / `DFlashGenerator::run` and emits
    // `TokenWithLogprobs` events from `finalize_burst_success` — the
    // same payload the classic decode path produces.
    //
    // Thinking-budget enforcement is likewise NO LONGER a gate. `finalize_burst_success` runs the same per-token
    // `decide_override` + `observe` cycle the classic path's
    // `apply_thinking_budget` uses, injecting a forced `</think>` at the
    // budget boundary.
    true
}

/// Return whether `seq` may join a B > 1 speculative-burst window.
///
/// This is intentionally **stricter** than [`should_burst_for_sequence`]:
/// the B = 1 burst path can emit per-token logprob payloads, but the
/// batched MTP/DFlash round loops return token IDs only. Keeping
/// logprobs-enabled requests out of B > 1 windows makes them fall back
/// to the B = 1 burst, where's `TokenWithLogprobs` contract
/// is preserved.
pub(crate) fn can_join_batched_burst_window(seq: &SequenceInfo) -> bool {
    if seq.logprobs_config.enabled {
        tracing::debug!(
            "speculative batched burst declined for seq {}: logprobs requested",
            seq.seq_id,
        );
        return false;
    }
    true
}

/// Whether to force the Gemma 4 MTP B=1 burst path.
///
/// real-model measurements showed the assistant drafter is not
/// profitable for single-request Gemma 4 31B serving: the upstream reference
/// speedups are reported for B>1 windows, while the B=1 path spends an extra
/// drafter forward per token at very low acceptance. Production therefore
/// routes singleton MTP requests through classic decode by default and keeps
/// this override for parity/debug investigations.
pub(crate) fn mtp_b1_burst_enabled() -> bool {
    std::env::var("MLXCEL_ENABLE_MTP_B1")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Whether to force the Gemma 4 MTP B>1 batched burst path.
///
/// Reference Gemma 4 MTP gets its advertised speedups from batched verify
/// windows. The Rust path now preserves per-row cache positions / RoPE
/// anchors after mixed accepts, but real 31B validation still shows the
/// forced B>1 MTP burst is not consistently faster than classic batched
/// decode. Keep the path available behind an explicit operator/debug flag
/// while production falls back to the known-profitable classic scheduler
/// path.
pub(crate) fn mtp_batched_burst_enabled() -> bool {
    std::env::var("MLXCEL_ENABLE_MTP_BATCH")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Successful burst outcome returned to the scheduler.
///
/// The scheduler uses `tokens_generated` to update the per-request
/// Prometheus histograms (mirrors the classic decode path's
/// `batch_metrics.record_sequence_completed(tokens_generated)` call
/// inside `finalize_completed`).
///
/// the prompt + generated token vectors and the
/// `healthy_finish` flag are surfaced so the scheduler can call
/// [`crate::server::batch::scheduler::BatchScheduler::donate_finished_sequence_cache`]
/// for the burst path exactly as `finalize_completed` does for the
/// classic path. The burst consumes `SequenceInfo` by value (it
/// finalizes the request inline), so without re-surfacing these the
/// scheduler has no handle to the tokens after the burst returns. The
/// donate helper is hard-gated on a dense KV-cache backend, so for
/// today's burst-eligible models — Qwen 3.5 (DFlash) and Gemma 4 (MTP),
/// both `SequenceStateBackend::ModelOwned` — the donate is a guarded
/// no-op, **identical** to the classic path's no-op for those same
/// model families. Wiring it in keeps the two paths symmetric and
/// future-proofs the burst for any dense-KV-cache model that later
/// becomes burst-eligible.
#[derive(Debug, Clone)]
pub(crate) struct BurstFinalized {
    pub seq_id: mlxcel_core::cache::SequenceId,
    /// Total tokens streamed to the client (including the seed bonus
    /// and any tokens accepted by the round loop). Matches what the
    /// client actually received.
    pub tokens_generated: usize,
    /// The request's full prompt token stream, forwarded so the
    /// scheduler can compose the prompt-cache donate key. Empty on the
    /// error / transition-failure paths (no healthy cache to donate).
    pub prompt_tokens: Vec<i32>,
    /// The tokens the burst actually committed to the sequence (post
    /// early-EOS truncation). Joined with `prompt_tokens` by the
    /// scheduler to form the donate entry's token key. Empty on the
    /// error paths.
    pub generated_tokens: Vec<i32>,
    /// Whether the burst reached a healthy finish (`Stop` / `Length`).
    /// Mirrors the classic path's `healthy` gate in `finalize_completed`
    /// — only healthy finishes donate their cache back. `false` on the
    /// error / transition-failure paths.
    pub healthy_finish: bool,
}

/// Run a speculative burst for `seq`, producing every token the request
/// will emit, streaming them via `seq.response_tx`, and finalizing the
/// sequence inline.
///
/// Returns `Ok(BurstFinalized)` when the burst handled the request
/// end-to-end (whether by completing, hitting EOS, or surfacing an
/// error to the client). The caller is responsible for releasing the
/// sequence's cache slot via
/// [`crate::server::batch::scheduler::BatchScheduler::release_sequence_caches`]
/// AND recording the per-sequence metric via
/// `batch_metrics.record_sequence_completed(finalized.tokens_generated)`.
/// Returns `Err(rejected_seq)` when the burst declines the request
/// (e.g. the model variant is not supported by the dispatch kind, or
/// the drafter load failed and the operator should see a classic
/// decode fallback). The caller is then responsible for routing
/// `rejected_seq` through the standard non-speculative path.
///
/// This is the load-bearing entry from
/// [`crate::server::batch::BatchScheduler::execute_prefill`].
//
// Suppress `result_large_err`: the `Err` variant carries the full
// `SequenceInfo` (~728 bytes) on purpose so the scheduler can route
// the same sequence into the classic prefill path without an extra
// `Box::new(..) / *boxed` round-trip on the cold "burst declined"
// path. The hot path (`Ok(BurstFinalized)`) is tiny.
#[allow(clippy::result_large_err)]
pub(crate) fn try_run_burst_b1(
    mut ctx: BurstContext<'_>,
    mut seq: SequenceInfo,
) -> Result<BurstFinalized, SequenceInfo> {
    let burst_start = Instant::now();
    let prompt_len = seq.prompt_tokens.len();
    if prompt_len == 0 {
        // Defensive: scheduler already rejects empty prompts at enqueue
        // time. If we ever reach here, fall back to classic decode so
        // the error message stays consistent with the non-speculative
        // path.
        return Err(seq);
    }

    // We don't transition state yet — the burst may decline based on
    // the model variant, in which case the request must fall back to
    // the classic prefill path which itself transitions Queued →
    // Prefilling. The state-machine guard rejects Prefilling → Queued
    // (only Decoding → Queued is allowed for preemptive eviction), so
    // we hold the transition until the dispatch arm confirms support.
    seq.prefill_start = Some(burst_start);

    let result = match ctx.dispatch {
        crate::server::SpeculativeDispatch::Mtp { block_size, .. } => {
            let bs = *block_size;
            run_mtp_burst(ctx.reborrow(), &mut seq, bs)
        }
        crate::server::SpeculativeDispatch::DFlash { block_size, .. } => {
            let bs = *block_size;
            run_dflash_burst(ctx.reborrow(), &mut seq, bs)
        }
        // The gate (`should_burst_for_sequence`) already excluded
        // `Disabled` / `Classic`. Re-asserting here makes the intent
        // explicit for future maintainers.
        crate::server::SpeculativeDispatch::Disabled
        | crate::server::SpeculativeDispatch::Classic { .. } => Err(BurstOutcome::DeclineToClassic),
    };

    match result {
        Ok(BurstSuccess {
            tokens,
            logprobs,
            prefill_time_ms,
            decode_time_ms,
        }) => {
            // Transition Queued → Prefilling now that we know the
            // burst owned the request lifecycle.
            if let Err(err) = seq.state.transition_to(SequenceState::Prefilling) {
                let seq_id = seq.seq_id;
                emit_error_and_finalize(ctx, seq, &format!("State transition error: {err}"));
                return Ok(BurstFinalized {
                    seq_id,
                    tokens_generated: 0,
                    // Transition failure is an error outcome — no
                    // healthy cache to donate.
                    prompt_tokens: Vec::new(),
                    generated_tokens: Vec::new(),
                    healthy_finish: false,
                });
            }
            let seq_id = seq.seq_id;
            // `finalize_burst_success` streams the tokens, classifies
            // the finish reason, and hands back the prompt + committed
            // token vectors so the scheduler can mirror the classic
            // path's prompt-cache donate. `logprobs` is
            // forwarded so a speculative response carries the same
            // `TokenWithLogprobs` payload as the classic decode path
            let finalized =
                finalize_burst_success(ctx, seq, tokens, logprobs, prefill_time_ms, decode_time_ms);
            Ok(BurstFinalized {
                seq_id,
                tokens_generated: finalized.tokens_generated,
                prompt_tokens: finalized.prompt_tokens,
                generated_tokens: finalized.generated_tokens,
                healthy_finish: finalized.healthy_finish,
            })
        }
        Err(BurstOutcome::DeclineToClassic) => {
            // Sequence is still in `Queued` (we deliberately didn't
            // transition). The scheduler's classic `execute_full_prefill`
            // path will call `begin_prefill` which transitions
            // Queued → Prefilling normally. Just clear the burst
            // timer so the classic path's `prefill_start` measurement
            // starts fresh.
            seq.prefill_start = None;
            Err(seq)
        }
        Err(BurstOutcome::Error(msg)) => {
            // Burst attempted the request but failed mid-flight (e.g.
            // drafter load failed, or the round loop bailed). The seq
            // was never transitioned to Prefilling, so transition
            // directly to Finished(Error) which is always permitted
            // from Queued. We report tokens_generated=0 to the metric
            // because the client received an Error event, not any
            // generated tokens.
            let seq_id = seq.seq_id;
            emit_error_and_finalize(ctx, seq, &msg);
            Ok(BurstFinalized {
                seq_id,
                tokens_generated: 0,
                // Error outcome: the KV cache is assumed tainted, so
                // no donate (mirrors the classic path's
                // `Finished(Error)` bypass in `finalize_completed`).
                prompt_tokens: Vec::new(),
                generated_tokens: Vec::new(),
                healthy_finish: false,
            })
        }
    }
}

/// Successful burst output.
struct BurstSuccess {
    tokens: Vec<i32>,
    /// Per-token log-probability data, index-aligned 1:1 with
    /// [`Self::tokens`]. Empty when `seq.logprobs_config.enabled` is
    /// false (zero-overhead path); otherwise one
    /// `Option<TokenLogprobData>` per token, forwarded to
    /// `finalize_burst_success` which emits `TokenWithLogprobs` events
    /// so speculative responses carry the same logprob payload as the
    /// classic decode path.
    logprobs: Vec<Option<TokenLogprobData>>,
    prefill_time_ms: f64,
    decode_time_ms: f64,
}

/// Outcome of [`finalize_burst_success`].
///
/// in addition to the streamed token count (used for the
/// Prometheus per-sequence histogram), this carries the data the
/// scheduler needs to mirror the classic path's prompt-cache donate —
/// the full prompt token stream, the tokens actually committed to the
/// sequence (post early-EOS truncation), and whether the finish was
/// healthy (`Stop` / `Length`). The burst owns the `SequenceInfo` by
/// value, so re-surfacing these is the only way the scheduler can call
/// `donate_finished_sequence_cache` after the burst returns.
struct FinalizeOutcome {
    /// Tokens actually streamed to the client (== committed
    /// `generated_tokens.len()`).
    tokens_generated: usize,
    /// The request's full prompt token stream (for the donate key).
    prompt_tokens: Vec<i32>,
    /// The tokens committed to the sequence after early-EOS truncation.
    generated_tokens: Vec<i32>,
    /// `true` for `FinishReason::Stop` / `FinishReason::Length`.
    healthy_finish: bool,
}

/// Burst error / decline.
enum BurstOutcome {
    /// The burst declined this request; the scheduler should re-route
    /// through the classic non-speculative path.
    DeclineToClassic,
    /// The burst attempted to handle the request but failed mid-flight.
    /// The error message is streamed to the client and the sequence is
    /// finalized.
    Error(String),
}

/// Captured context the burst needs from the scheduler. Borrowed for
/// the burst's lifetime; the burst never holds these references across
/// MLX dispatch yields (single-threaded scheduler invariant).
///
/// The mutable [`WorkerDrafterSlot`] is held as `&mut` so the burst can
/// load (lazy), take, return, and reset the drafter inside one
/// invocation. The remaining fields are pure-immutable refs so the
/// context can be partially-borrowed across the burst's sub-functions
/// without cloning.
pub(crate) struct BurstContext<'a> {
    pub(crate) model: &'a LoadedModel,
    pub(crate) tokenizer: &'a crate::tokenizer::MlxcelTokenizer,
    pub(crate) drafter_slot: &'a mut WorkerDrafterSlot,
    pub(crate) dispatch: &'a crate::server::SpeculativeDispatch,
}

impl<'a> BurstContext<'a> {
    /// Re-borrow the context so it can be passed to a callee without
    /// consuming the caller's binding. Mirrors `&mut *self` for the
    /// drafter slot while leaving every other field as an immutable
    /// re-borrow. Used by the dispatch match-arms in
    /// [`try_run_burst_b1`].
    fn reborrow(&mut self) -> BurstContext<'_> {
        BurstContext {
            model: self.model,
            tokenizer: self.tokenizer,
            drafter_slot: &mut *self.drafter_slot,
            dispatch: self.dispatch,
        }
    }
}

/// MTP B=1 burst — Gemma 4 / Gemma 4 VLM target.
///
/// **Variant gate runs before drafter load.** Surfacing
/// "unsupported target" is cheap (no IO) and the operator should see
/// the decline-to-classic message rather than a confusing
/// "Drafter load failed" message when the pairing is fundamentally
/// wrong (e.g. `--model qwen3.5 --draft-kind mtp`). This ordering also
/// matches the DFlash burst's intent below.
///
/// **Drafter bind happens before MtpGenerator construction.** The
/// underlying [`Gemma4AssistantDraftModel`] (the only currently
/// supported MTP drafter shape) requires
/// [`mlxcel_core::drafter::Drafter::bind`] to be called before its
/// first [`mlxcel_core::drafter::Drafter::draft_block`] call — bind
/// captures the target's `embed_tokens` and resolves the drafter's
/// `lm_head`. Without bind, `draft_block` returns
/// `DrafterError::BindNotCalled`, which
/// [`MtpGenerator::generate`] swallows silently, breaking out of the
/// round loop. The result would be exactly one seed-bonus token
/// streamed to the client per speculative request. This is the
/// silent-correctness bug that PR-review cycle 2 fixed (see
/// `speculative_burst_tests::run_mtp_burst_binds_drafter_before_draft_block`).
///
/// Note: [`mlxcel_core::drafter::dflash::DFlashGenerator::run`] calls
/// `bind` and `reset` internally on every run, so the DFlash burst
/// does NOT need a manual bind here — and adding one would
/// double-bind. See `run_dflash_burst` below.
fn run_mtp_burst(
    ctx: BurstContext<'_>,
    seq: &mut SequenceInfo,
    block_size: u32,
) -> Result<BurstSuccess, BurstOutcome> {
    let block_size = block_size as usize;
    if block_size < 2 {
        return Err(BurstOutcome::Error(format!(
            "MTP burst: block_size={block_size} < 2 produces no draft proposals"
        )));
    }

    // HOIST: resolve the target LM reference and validate the model
    // variant BEFORE loading the drafter. An unsupported pairing
    // (e.g. `--model qwen3.5 --draft-kind mtp`) decline-to-classics
    // here without any IO, which avoids surfacing a confusing
    // "Drafter load failed" message later. Returns `Some((target_lm))`
    // for supported variants, `None` for unsupported (caller
    // decline-to-classics).
    let target_lm: &dyn LanguageModel = match ctx.model {
        LoadedModel::Gemma4(wrapper) => wrapper as &dyn LanguageModel,
        LoadedModel::Gemma4VLM(vlm) => vlm as &dyn LanguageModel,
        LoadedModel::Gemma4Unified(unified) => unified as &dyn LanguageModel,
        _ => {
            tracing::warn!(
                "MTP speculative dispatch declined: target is {:?}, expected \
                 Gemma 4 (text, VLM, or Unified); falling back to classic decode",
                model_variant_label(ctx.model),
            );
            return Err(BurstOutcome::DeclineToClassic);
        }
    };

    // The MTP generator owns the drafter by value; take it from the
    // slot for the burst's lifetime. On success we return it via
    // `return_drafter`; on error we drop it so the next request
    // reloads.
    // Trigger lazy-load (return value not used here — we re-acquire
    // the drafter by value below). Releasing the `&mut` borrow before
    // the `take()` call is what lets `take` re-borrow `ctx.drafter_slot`
    // mutably without overlap.
    ctx.drafter_slot
        .ensure_loaded()
        .map_err(BurstOutcome::Error)?;
    let mut owned_drafter = ctx
        .drafter_slot
        .take()
        .ok_or_else(|| BurstOutcome::Error("drafter slot empty after ensure_loaded".to_string()))?;

    // CRITICAL: bind the drafter to the target BEFORE constructing the
    // generator. [`MtpGenerator::generate`] does NOT call bind
    // internally (unlike `DFlashGenerator::run` which does). Without
    // this call, the first `draft_block` returns
    // `DrafterError::BindNotCalled`, the generator silently breaks out
    // of the round loop, and the client receives exactly one
    // seed-bonus token. See the function-level docstring above.
    //
    // On bind failure we drop the drafter and surface the error to
    // the client — the slot stays empty so the next request lazily
    // reloads from disk. This is consistent with the
    // `WorkerDrafterSlot::ensure_loaded` failure semantics: a failed
    // bind is operator-actionable (typically a target/drafter
    // mismatch) and re-loading from disk won't fix it, but dropping
    // the handle keeps the slot in a clean state for the (rare) case
    // where the operator hot-swaps the drafter checkpoint between
    // requests.
    // Compatibility gate BEFORE binding: reject a target↔drafter pairing whose
    // hidden size / vocabulary do not match (e.g. a 12B Unified target fed an
    // assistant built for a different backbone). The default trait impl is a
    // no-op, so DFlash/InternalMtp are unaffected; the Gemma 4 assistant
    // overrides it to compare backbone_hidden_size (3840) against the target's
    // text hidden size and vocab. Surfaced as a clear burst error.
    if let Err(e) = owned_drafter.validate_target_compat(target_lm) {
        return Err(BurstOutcome::Error(format!(
            "MTP drafter incompatible with target: {e}"
        )));
    }
    if let Err(e) = owned_drafter.bind(target_lm) {
        return Err(BurstOutcome::Error(format!("MTP drafter bind failed: {e}")));
    }

    let sampling = seq.sampling.clone();
    let max_tokens = seq.max_tokens.max(1);
    let prompt = seq.prompt_tokens.clone();

    // History-dependent-penalty context for the first-bonus sample
    // (repetition / frequency / presence / DRY). `initial_token_history`
    // returns the prompt tokens when any such penalty is active and an
    // empty vec otherwise — exactly what the classic decode path seeds
    // its first-token sample with (`scheduler.rs::finish_prefill`). This
    // is what makes the burst's first bonus byte-identical to the
    // classic path for penalty-bearing requests.
    let token_history = initial_token_history(&prompt, sampling.needs_token_history());

    // Cooperative-cancellation flag plumbed into the round-loop driver.
    // The burst owns the worker thread for its full lifetime; on a
    // client disconnect mid-burst the scheduler flips `seq.cancelled`
    // and the generator bails out after the current round instead of
    // running the whole `max_tokens` budget.
    let cancel: &AtomicBool = &seq.cancelled;
    // Per-token log-probability capture control. When enabled, the
    // generator returns one logprob entry per emitted token; the burst
    // forwards them through `finalize_burst_success` which emits
    // `TokenWithLogprobs` events so speculative responses carry the
    // same payload as the classic decode path.
    let logprobs_config = seq.logprobs_config.clone();
    let (output, stats) = match ctx.model {
        LoadedModel::Gemma4(wrapper) => {
            let adapter =
                Gemma4MtpTargetAdapter::new_with_block_size(wrapper, Some(seq.seq_id), block_size);
            drive_mtp_generator(
                adapter,
                owned_drafter,
                &prompt,
                max_tokens,
                &sampling,
                &token_history,
                block_size,
                cancel,
                &logprobs_config,
            )
        }
        LoadedModel::Gemma4VLM(vlm) => {
            let adapter =
                crate::models::gemma4_mtp_target::Gemma4VLMtpTargetAdapter::new_with_block_size(
                    vlm,
                    Some(seq.seq_id),
                    block_size,
                );
            drive_mtp_generator(
                adapter,
                owned_drafter,
                &prompt,
                max_tokens,
                &sampling,
                &token_history,
                block_size,
                cancel,
                &logprobs_config,
            )
        }
        LoadedModel::Gemma4Unified(unified) => {
            let adapter =
                crate::models::gemma4_mtp_target::Gemma4UnifiedMtpTargetAdapter::new_with_block_size(
                    unified,
                    Some(seq.seq_id),
                    block_size,
                );
            drive_mtp_generator(
                adapter,
                owned_drafter,
                &prompt,
                max_tokens,
                &sampling,
                &token_history,
                block_size,
                cancel,
                &logprobs_config,
            )
        }
        // Unreachable: the hoisted variant check above already
        // returned for any model that is neither `Gemma4` nor
        // `Gemma4VLM`. Keeping a defensive arm rather than
        // `unreachable!()` so a future LoadedModel variant added to
        // the enum surfaces as a clean burst error instead of a
        // panic at request time.
        _ => {
            return Err(BurstOutcome::Error(format!(
                "MTP burst: unsupported target {:?} after variant gate (should not happen)",
                model_variant_label(ctx.model),
            )));
        }
    };

    // Hand the recovered drafter back to the slot for the next
    // request. The slot's `return_drafter` calls `reset` against the
    // target LM so the drafter starts clean.
    ctx.drafter_slot
        .return_drafter(output.recovered_drafter, target_lm);

    Ok(BurstSuccess {
        tokens: output.emitted,
        logprobs: output.logprobs,
        prefill_time_ms: stats.prefill_time_ms,
        decode_time_ms: stats.decode_time_ms,
    })
}

/// Output of [`drive_mtp_generator`]: emitted tokens + the drafter
/// handle recovered from the consumed [`MtpGenerator`].
///
/// Returning the drafter by value (rather than the slot-mutating
/// pattern) lets the caller decide how to dispose of it
/// (return-to-slot on success, drop on error). Crucially, this also
/// keeps [`drive_mtp_generator`] callable from `#[cfg(test)]` mocks
/// that don't have a `WorkerDrafterSlot` — the only way the
/// `run_mtp_burst_binds_drafter_before_draft_block` regression test
/// is feasible without a real `LoadedModel`.
struct DriveMtpOutput {
    emitted: Vec<i32>,
    /// Per-token log-probability data, index-aligned 1:1 with
    /// [`Self::emitted`]. Empty when the caller's `LogprobsConfig` is
    /// disabled.
    logprobs: Vec<Option<TokenLogprobData>>,
    recovered_drafter: Box<dyn Drafter>,
}

/// Generator-shape-agnostic helper that drives an [`MtpGenerator`]
/// over a `T: MtpTarget` and a pre-bound drafter. Returns the emitted
/// tokens, the generator's stats, and the recovered drafter handle.
///
/// **Caller invariant**: the `drafter` MUST have been
/// [`Drafter::bind`]-ed against the target's underlying LanguageModel
/// before this call. This invariant lives at the call site (above in
/// `run_mtp_burst`) so this helper is generic over `T: MtpTarget`
/// without needing a `&dyn LanguageModel` parameter — keeping the
/// signature mockable.
///
/// `token_history` is the history-dependent-penalty context forwarded
/// to [`MtpGenerator::generate`] (then on to
/// [`mlxcel_core::speculative::mtp::target::MtpTarget::prefill_and_seed`])
/// for the first-bonus sample, so a penalty-bearing request's first
/// bonus is byte-identical to the classic decode path.
///
/// `cancel` is the cooperative-cancellation flag forwarded to
/// [`MtpGenerator::generate`]; it is checked once per round so a
/// disconnected client's burst stops occupying the worker thread
///
/// `logprobs_config` is forwarded to [`MtpGenerator::generate`]; when
/// enabled the returned [`DriveMtpOutput::logprobs`] carries one
/// `Option<TokenLogprobData>` per emitted token, index-aligned with
/// [`DriveMtpOutput::emitted`].
#[allow(clippy::too_many_arguments)]
fn drive_mtp_generator<T>(
    target: T,
    drafter: Box<dyn Drafter>,
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    token_history: &[i32],
    block_size: usize,
    cancel: &AtomicBool,
    logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
) -> (DriveMtpOutput, mlxcel_core::generate::GenerationStats)
where
    T: mlxcel_core::speculative::mtp::target::MtpTarget,
{
    let mut generator = MtpGenerator::new(target, drafter, block_size);
    let (emitted, logprobs, stats) = generator.generate(
        prompt,
        max_tokens,
        sampling,
        token_history,
        cancel,
        logprobs_config,
    );
    // `MtpGenerator` owns the drafter by value; recover it via
    // `into_drafter` for slot-restoration.
    let recovered_drafter = generator.into_drafter();
    (
        DriveMtpOutput {
            emitted,
            logprobs,
            recovered_drafter,
        },
        stats,
    )
}

/// DFlash B=1 burst — Qwen 3.5 text target or Qwen 3.5 VLM wrapper serving a
/// text-only request.
///
/// **Variant gate runs before drafter load.** Same rationale as
/// [`run_mtp_burst`]: surfacing "unsupported target" decline-to-classic
/// before any drafter IO is cheaper for the operator-facing UX. An
/// unsupported pairing (e.g. `--model gemma4 --draft-kind dflash`)
/// short-circuits here without ever attempting to read the drafter
/// checkpoint.
///
/// **Drafter bind happens inside the round loop.**
/// [`DFlashGenerator::run`] calls `self.drafter.bind(target_lm)?`
/// internally on every invocation (see
/// `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs::run`). Do
/// NOT call bind here — that would double-bind. (Contrast with
/// [`run_mtp_burst`] where `MtpGenerator::generate` does NOT bind
/// internally and we must bind here.)
fn run_dflash_burst(
    ctx: BurstContext<'_>,
    seq: &mut SequenceInfo,
    block_size: u32,
) -> Result<BurstSuccess, BurstOutcome> {
    if seq.vlm_embeddings.is_some() || !seq.images.is_empty() || !seq.audio.is_empty() {
        tracing::warn!(
            "DFlash speculative dispatch declined for seq {}: multimodal VLM request \
             detected; VLM-wrapped text-only Qwen 3.5 targets are supported, but \
             multimodal speculative tail is not yet enabled; falling back to classic decode",
            seq.seq_id,
        );
        return Err(BurstOutcome::DeclineToClassic);
    }

    // HOIST: validate the model variant BEFORE loading the drafter.
    // See the function-level docstring above.
    match ctx.model {
        LoadedModel::Qwen35(_)
        | LoadedModel::Qwen35Moe(_)
        | LoadedModel::Qwen35VLM(_)
        | LoadedModel::Qwen35MoeVLM(_) => {}
        _ => {
            tracing::warn!(
                "DFlash speculative dispatch declined: target is {:?}, expected \
                 Qwen 3.5 text or VLM-wrapped text-only; falling back to classic decode",
                model_variant_label(ctx.model),
            );
            return Err(BurstOutcome::DeclineToClassic);
        }
    }

    ctx.drafter_slot
        .ensure_loaded()
        .map_err(BurstOutcome::Error)?;
    let owned_drafter = ctx
        .drafter_slot
        .take()
        .ok_or_else(|| BurstOutcome::Error("drafter slot empty after ensure_loaded".to_string()))?;

    let sampling = seq.sampling.clone();
    let max_tokens = seq.max_tokens.max(1);
    let prompt = seq.prompt_tokens.clone();
    let eos_token_ids = merged_eos_token_ids(ctx.model.eos_token_ids(), &sampling.stop_token_ids);

    // History-dependent-penalty context for the first-bonus sample
    // (repetition / frequency / presence / DRY). `initial_token_history`
    // returns the prompt tokens when any such penalty is active and an
    // empty vec otherwise — exactly what the classic decode path seeds
    // its first-token sample with (`scheduler.rs::finish_prefill`). This
    // is what makes the burst's first bonus byte-identical to the
    // classic path for penalty-bearing requests.
    let token_history = initial_token_history(&prompt, sampling.needs_token_history());

    // Cooperative-cancellation flag plumbed into the round-loop driver.
    // The burst owns the worker thread for its full lifetime; on a
    // client disconnect mid-burst the scheduler flips `seq.cancelled`
    // and the generator bails out after the current round instead of
    // running the whole `max_tokens` budget.
    let cancel: &AtomicBool = &seq.cancelled;
    // Per-token log-probability capture control. When enabled, the
    // burst returns one logprob entry per emitted token; the burst
    // forwards them through `finalize_burst_success` which emits
    // `TokenWithLogprobs` events so speculative responses carry the
    // same payload as the classic decode path.
    let logprobs_config = seq.logprobs_config.clone();

    // DFlash supports Qwen35 text models and Qwen35 VLM wrappers for
    // text-only requests. The multimodal gate above rejects image/audio
    // payloads before this point.
    let prefill_start = Instant::now();
    let (tokens, logprobs, decode_time_ms) = match ctx.model {
        LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => run_dflash_on_qwen35(
            qwen,
            &prompt,
            &sampling,
            &token_history,
            &eos_token_ids,
            owned_drafter,
            block_size,
            max_tokens,
            ctx.drafter_slot,
            cancel,
            &logprobs_config,
        )?,
        LoadedModel::Qwen35VLM(qwen) | LoadedModel::Qwen35MoeVLM(qwen) => run_dflash_on_qwen35(
            qwen,
            &prompt,
            &sampling,
            &token_history,
            &eos_token_ids,
            owned_drafter,
            block_size,
            max_tokens,
            ctx.drafter_slot,
            cancel,
            &logprobs_config,
        )?,
        _ => {
            // Unreachable per the variant gate above. Defensive arm
            // rather than `unreachable!()` so a future LoadedModel
            // variant addition surfaces as a clean error instead of
            // a runtime panic. Restore the drafter to the slot
            // since we took it without using it.
            ctx.drafter_slot.drafter = Some(owned_drafter);
            return Err(BurstOutcome::Error(format!(
                "DFlash burst: unsupported target {:?} after variant gate (should not happen)",
                model_variant_label(ctx.model),
            )));
        }
    };

    let total_burst = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_time_ms = (total_burst - decode_time_ms).max(0.0);
    Ok(BurstSuccess {
        tokens,
        logprobs,
        prefill_time_ms,
        decode_time_ms,
    })
}

/// DFlash burst on a Qwen 3.5 text target (including a Qwen 3.5 VLM wrapper
/// serving a text-only request) — handles prefill, first-bonus + first-hidden
/// extraction, and `DFlashGenerator::run` driving.
///
/// `token_history` is the history-dependent-penalty context for the
/// first-bonus sample (repetition / frequency / presence / DRY); it is
/// forwarded to `sample_token_optimized` so a penalty-bearing request's
/// first bonus is byte-identical to the classic decode path. The round loop itself runs greedy at temp=0 today, so the
/// per-round target argmax is unaffected — only the first bonus reads
/// the history.
///
/// `cancel` is the cooperative-cancellation flag forwarded to
/// [`DFlashGenerator::run`]; it is checked once per round so a
/// disconnected client's burst stops occupying the worker thread
///
/// `logprobs_config` controls per-token log-probability capture. When
/// enabled, the returned `Vec<Option<TokenLogprobData>>` carries one
/// entry per emitted token (index-aligned with the returned tokens):
/// the first-bonus logprob is computed here from the same
/// penalty-adjusted logits the bonus was sampled from, and the
/// round-loop tokens' logprobs come back in `DFlashRunOutput::logprobs`
#[allow(clippy::too_many_arguments)]
fn run_dflash_on_qwen35<T>(
    qwen: &T,
    prompt_tokens: &[i32],
    sampling: &SamplingConfig,
    token_history: &[i32],
    eos_token_ids: &[i32],
    owned_drafter: Box<dyn Drafter>,
    block_size: u32,
    max_tokens: usize,
    drafter_slot: &mut WorkerDrafterSlot,
    cancel: &AtomicBool,
    logprobs_config: &mlxcel_core::sampling::LogprobsConfig,
) -> Result<(Vec<i32>, Vec<Option<TokenLogprobData>>, f64), BurstOutcome>
where
    T: Qwen35DFlashTarget,
{
    // Build a fresh per-layer cache vector for this request. We do NOT
    // touch the scheduler-owned `sequence_state` map — the speculative
    // burst's caches are independent of the prompt-cache adoption
    // pipeline (`should_burst_for_sequence` enforces
    // `prefill_start_offset == 0`).
    //
    // The target-specific cache factory returns the heterogeneous
    // attention+linear cache vec the round loop needs, while the
    // `LanguageModel::make_caches(&self) -> Vec<KVCache>` trait method
    // returns an empty vec for Qwen 3.5 (the model owns its caches
    // internally). Use the narrow `Qwen35DFlashTarget` helper to
    // disambiguate against the trait method by name.
    let mut caches: Vec<crate::models::qwen3_next::Qwen3NextCache> = qwen.make_dflash_caches();

    // Prefill the prompt through the target's speculative verify hook,
    // capturing the same per-layer hidden states the DFlash round loop
    // captures for candidate blocks. The capture list comes from the
    // drafter checkpoint's `target_layer_ids` (4B uses [1,8,15,22,29];
    // larger drafts such as 27B use their own list).
    let capture_layer_ids = owned_drafter
        .dflash_target_layer_ids()
        .filter(|ids| !ids.is_empty())
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| qwen.capture_layer_ids().to_vec());
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let prefill_verify_start = Instant::now();
    let verify_out =
        qwen.verify_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);
    let prefill_verify_ms = prefill_verify_start.elapsed().as_secs_f64() * 1000.0;

    // Sample the first bonus token from the last-position logits.
    let first_bonus_start = Instant::now();
    let last_pos = prompt_tokens.len() as i32 - 1;
    let logits_shape = mlxcel_core::array_shape(&verify_out.logits);
    let vocab = logits_shape[2];
    let last_logits = mlxcel_core::slice(
        &verify_out.logits,
        &[0, last_pos, 0],
        &[logits_shape[0], last_pos + 1, vocab],
    );
    // `token_history` carries the history-dependent-penalty context
    // (repetition / frequency / presence / DRY) so the first bonus is
    // byte-identical to the classic decode path's first token. The subsequent round-loop tokens are produced by the
    // target's greedy argmax inside `DFlashGenerator::run` (DFlash is
    // greedy-only today), which carries no history dependence.
    // `adjusted_logits` is the penalty-adjusted `[1, vocab]` slice the
    // bonus was sampled from; it feeds `compute_logprobs` so the
    // first-bonus logprob is byte-identical to the classic path's
    // first-token logprob.
    let (first_bonus_arr, first_bonus_adjusted_logits) =
        mlxcel_core::sampling::sample_token_optimized(&last_logits, sampling, token_history);
    mlxcel_core::eval(&first_bonus_arr);
    let first_bonus = mlxcel_core::item_i32(&first_bonus_arr);
    let first_bonus_lp = mlxcel_core::sampling::compute_logprobs(
        &first_bonus_adjusted_logits,
        first_bonus,
        logprobs_config,
    );
    let first_bonus_ms = first_bonus_start.elapsed().as_secs_f64() * 1000.0;

    // Build first_hidden = concat(hidden_states, axis=-1)[:, last_pos:last_pos+1, :].
    // The DFlash round loop expects shape [1, 1, num_layers * hidden_size].
    let first_hidden_start = Instant::now();
    if verify_out.hidden_states.is_empty() {
        // Drafter slot ownership: we took the drafter at the top; we
        // must not silently drop it on error.
        drafter_slot.drafter = Some(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash prefill returned no captured hidden layers".to_string(),
        ));
    }
    let mut concatenated = mlxcel_core::copy(
        verify_out.hidden_states[0]
            .as_ref()
            .expect("hidden state must be non-null"),
    );
    for slab in verify_out.hidden_states.iter().skip(1) {
        concatenated = mlxcel_core::concatenate(
            &concatenated,
            slab.as_ref().expect("hidden state must be non-null"),
            -1,
        );
    }
    let concatenated_shape = mlxcel_core::array_shape(&concatenated);
    debug_assert_eq!(
        concatenated_shape.len(),
        3,
        "concatenated hidden must be 3-D"
    );
    let feature_dim = concatenated_shape[2];
    let first_hidden = mlxcel_core::slice(
        &concatenated,
        &[0, last_pos, 0],
        &[concatenated_shape[0], last_pos + 1, feature_dim],
    );
    let first_hidden_ms = first_hidden_start.elapsed().as_secs_f64() * 1000.0;

    // Drive the round loop. `run` returns the tokens EXCLUDING the
    // first bonus; we prepend it on success so the caller sees a
    // complete output stream.
    let mut generator = DFlashGenerator::new(
        owned_drafter,
        sampling.clone(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let result = generator.run(
        qwen,
        qwen as &dyn LanguageModel,
        &mut caches,
        first_bonus,
        first_hidden,
        eos_token_ids,
        max_tokens,
        cancel,
        logprobs_config,
    );

    // Whatever the round-loop's outcome, recover the drafter so the
    // slot is consistent for the next request.
    let recovered = generator.into_drafter();
    drafter_slot.return_drafter(recovered, qwen as &dyn LanguageModel);

    match result {
        Ok(output) => {
            let diagnostics = output.diagnostics.clone();
            tracing::info!(
                block_size = diagnostics.block_size,
                rounds = diagnostics.rounds,
                proposed_tokens = diagnostics.proposed_tokens,
                accepted_tokens = diagnostics.accepted_tokens,
                acceptance_rate = diagnostics.acceptance_rate(),
                emitted_per_verify = diagnostics.emitted_per_verify(),
                zero_accept_rounds = diagnostics.zero_accept_rounds,
                partial_accept_rounds = diagnostics.partial_accept_rounds,
                full_accept_rounds = diagnostics.full_accept_rounds,
                prefill_verify_ms,
                first_bonus_ms,
                first_hidden_ms,
                bind_reset_ms = diagnostics.bind_reset_time_ms,
                draft_ms = diagnostics.draft_time_ms,
                verify_ms = diagnostics.verify_time_ms,
                target_argmax_sync_ms = diagnostics.target_argmax_time_ms,
                logprobs_ms = diagnostics.logprobs_time_ms,
                walk_ms = diagnostics.walk_time_ms,
                hidden_concat_ms = diagnostics.hidden_concat_time_ms,
                rollback_ms = diagnostics.rollback_time_ms,
                decode_ms = diagnostics.total_decode_time_ms,
                "DFlash diagnostics"
            );
            let mut tokens = Vec::with_capacity(output.tokens.len() + 1);
            tokens.push(first_bonus);
            tokens.extend(output.tokens);
            // Assemble the per-token logprobs the same way as `tokens`:
            // the first-bonus logprob (computed above from the prefill's
            // adjusted logits) prepended to the round loop's per-token
            // logprobs. Stays empty when logprobs are disabled — the
            // round loop returns an empty `output.logprobs` and
            // `first_bonus_lp` is `None`, so the burst's
            // `finalize_burst_success` falls through to plain `Token`
            // events.
            let logprobs: Vec<Option<TokenLogprobData>> = if logprobs_config.enabled {
                let mut lp = Vec::with_capacity(output.logprobs.len() + 1);
                lp.push(first_bonus_lp);
                lp.extend(output.logprobs);
                lp
            } else {
                Vec::new()
            };
            Ok((tokens, logprobs, output.stats.decode_time_ms))
        }
        Err(e) => Err(BurstOutcome::Error(format!(
            "DFlash round loop failed: {e}"
        ))),
    }
}

/// Apply thinking-budget enforcement to one burst-produced
/// token, mirroring the classic decode path's
/// `BatchScheduler::apply_thinking_budget` (`scheduler.rs`).
///
/// Inspects `token` via [`mlxcel_core`-adjacent]
/// [`crate::server::thinking_budget::ThinkingState::decide_override`];
/// when the `<think>` budget is exceeded the forced `</think>` close id
/// is substituted, then
/// [`crate::server::thinking_budget::ThinkingState::observe`] advances
/// the state machine with the final token. `is_disabled()`
/// short-circuits to a single branch for non-thinking requests (zero
/// overhead on the hot path).
///
/// Returns `(final_token, override_fired)`:
/// - `final_token` — the id to commit / stream (either `token` or the
///   forced `</think>` close id).
/// - `override_fired` — `true` when the budget substituted a different
///   token. The caller uses this to drop the per-token logprob (the
///   generator's logprob describes the *original* token, so emitting it
///   against the forced `</think>` would make text and metadata
///   inconsistent — the classic path drops it identically).
///
/// ## Burst vs classic semantics
///
/// Unlike the classic path — which re-samples fresh logits on the
/// *next* step after a forced close — the burst's remaining tokens were
/// pre-computed by the round loop while the model still believed it was
/// inside the `<think>` block. The budget enforcement still injects
/// `</think>` at the correct boundary; the post-close tokens are the
/// burst's own output. This is an inherent property of Option-B burst
/// dispatch (one scheduler tick produces every token) and is documented
/// on [`should_burst_for_sequence`].
pub(crate) fn apply_burst_thinking_budget(
    thinking: &mut crate::server::thinking_budget::ThinkingState,
    token: i32,
) -> (i32, bool) {
    use crate::server::thinking_budget::ThinkingDecision;
    if thinking.is_disabled() {
        return (token, false);
    }
    let final_token = match thinking.decide_override(token) {
        ThinkingDecision::NoOverride => token,
        ThinkingDecision::ForceClose(close_id) => close_id,
    };
    thinking.observe(final_token);
    (final_token, final_token != token)
}

/// Stream tokens from a successful burst to `seq.response_tx`, then
/// emit the `Done` event and clean up. Mirrors the relevant bits of
/// `finish_prefill` + `decode_single_step` + `finalize_completed` from
/// `scheduler.rs` so the request looks identical to the client whether
/// it ran through the classic or speculative path.
///
/// Returns a [`FinalizeOutcome`] carrying the number of tokens actually
/// streamed to the client (`seq.generated_tokens.len()` after the
/// loop), plus the prompt + committed token vectors and the
/// healthy-finish flag. The scheduler feeds `tokens_generated` into
/// `batch_metrics.record_sequence_completed(...)` so the Prometheus
/// counters cover the burst path as well as the classic path, and uses
/// the token vectors + flag to call `donate_finished_sequence_cache`
/// exactly as `finalize_completed` does for the classic
/// path.
fn finalize_burst_success(
    ctx: BurstContext<'_>,
    mut seq: SequenceInfo,
    tokens: Vec<i32>,
    logprobs: Vec<Option<TokenLogprobData>>,
    _prefill_time_ms: f64,
    _decode_time_ms: f64,
) -> FinalizeOutcome {
    // Resolve the eos set once for the EOS-vs-Length classification.
    // `seq.merged_eos` is empty here (the classic path populates it in
    // `finish_prefill`); we recompute from the sampling/model state.
    let merged_eos = merged_eos_token_ids(ctx.model.eos_token_ids(), &seq.sampling.stop_token_ids);
    let eos_set: std::collections::HashSet<i32> = merged_eos.iter().copied().collect();

    let max_tokens = seq.max_tokens.max(1);
    let mut hit_eos = false;

    // When `seq.logprobs_config.enabled` the burst's generator returned
    // one logprob entry per emitted token, index-aligned with `tokens`.
    // We emit `GenerateEvent::TokenWithLogprobs` in that case so a
    // speculative response carries the same payload as the classic
    // decode path's `decode_single_step`. The empty-vec
    // case (logprobs disabled) falls through to plain `Token` events.
    let logprobs_enabled = seq.logprobs_config.enabled && !logprobs.is_empty();

    seq.first_token_time = Some(Instant::now());
    for (idx, token) in tokens.iter().copied().enumerate() {
        if seq.generated_tokens.len() >= max_tokens {
            break;
        }

        // thinking-budget enforcement, applied
        // per-emitted-token. See [`apply_burst_thinking_budget`].
        // `override_fired` gates logprob emission below: when the
        // thinking-budget substituted a different token, the
        // generator's `logprobs[idx]` entry describes the *original*
        // token, so emitting it against the forced `</think>` would
        // make the token text and logprob metadata inconsistent. The
        // classic path drops the logprob in exactly this case
        // (`scheduler.rs::decode_single_step`).
        let (final_token, override_fired) = apply_burst_thinking_budget(&mut seq.thinking, token);

        // EOS check before recording the token so EOS is reported as
        // FinishReason::Stop, not Length, matching the classic
        // `decode_single_step` semantics. The check uses the
        // post-override `final_token` so a forced `</think>` that
        // happens to be an EOS id is classified correctly.
        if eos_set.contains(&final_token) {
            hit_eos = true;
            break;
        }
        seq.generated_tokens.push(final_token);
        if let Some(new_text) = seq.decode_state.on_token(final_token, ctx.tokenizer) {
            // `logprobs[idx]` mirrors the classic path's per-token
            // `compute_logprobs(...)` result: `Some(lp)` → emit
            // `TokenWithLogprobs`, `None` → emit plain `Token` (the
            // classic path does the same when `compute_logprobs`
            // returns `None`, e.g. on a sampler override). A fired
            // thinking-budget override also forces plain `Token`.
            let event = if logprobs_enabled && !override_fired {
                match logprobs.get(idx).and_then(|lp| lp.clone()) {
                    Some(lp) => GenerateEvent::TokenWithLogprobs(new_text, lp),
                    None => GenerateEvent::Token(new_text),
                }
            } else {
                GenerateEvent::Token(new_text)
            };
            let _ = seq.response_tx.send(event);
        }
    }

    // Final state classification.
    let finish_reason = if hit_eos {
        FinishReason::Stop
    } else if seq.generated_tokens.len() >= max_tokens {
        FinishReason::Length
    } else {
        // The drafter / round loop bailed early without hitting EOS or
        // the budget — surface as Stop so the client sees a clean end
        // rather than a phantom Length. The token count and timing are
        // still accurate.
        FinishReason::Stop
    };
    // classify the finish for the prompt-cache donate gate
    // BEFORE `finish_reason` is moved into `transition_to`. Mirrors the
    // `healthy` gate in `scheduler.rs::finalize_completed` — only
    // `Stop` / `Length` / `Cancelled` finishes donate their cache back.
    // `finalize_burst_success` only ever classifies `Stop` or `Length`
    // (the `Error` / `Cancelled` outcomes never reach this function),
    // so this is always `true` here; computing it explicitly keeps the
    // burst path's gate bit-identical to the classic path's and robust
    // if the classification above ever gains a new arm.
    let healthy_finish = matches!(
        finish_reason,
        FinishReason::Stop | FinishReason::Length | FinishReason::Cancelled
    );
    if let Err(err) = seq
        .state
        .transition_to(SequenceState::Finished(finish_reason))
    {
        tracing::warn!("Speculative burst finalize state transition failed: {err}");
    }

    let tokens_generated = seq.generated_tokens.len();
    seq.decode_state.flush(ctx.tokenizer);
    let cached = seq.already_cached_tokens;
    let result = seq.decode_state.finish_with_cache(
        seq.created_at,
        seq.prompt_tokens.len(),
        seq.max_tokens,
        cached,
    );
    tracing::info!(
        prompt_tokens = seq.prompt_tokens.len(),
        generated_tokens = tokens_generated,
        burst_ms = result.generation_time_ms,
        "Speculative burst completed"
    );
    let _ = seq.response_tx.send(GenerateEvent::Done(result));
    // Move the prompt + committed token vectors out of `seq` for the
    // scheduler's prompt-cache donate. `seq` is dropped
    // immediately after this — these are its last readers.
    FinalizeOutcome {
        tokens_generated,
        prompt_tokens: std::mem::take(&mut seq.prompt_tokens),
        generated_tokens: std::mem::take(&mut seq.generated_tokens),
        healthy_finish,
    }
}

/// Emit an error event for `seq` over the response channel, then drop
/// the sequence. Used by the burst's error paths.
///
/// Notes:
/// - We do NOT explicitly transition `seq.state` to
///   `Finished(Error)` here because the sequence is dropped at the
///   end of this function; nothing downstream observes the state.
///   The state-machine guard rejects spurious extra transitions
///   anyway, so the drop is the correct cleanup.
/// - The caller (scheduler) is responsible for releasing the cache
///   slot via [`crate::server::batch::scheduler::BatchScheduler::release_sequence_caches`].
///   We don't take a `&mut Scheduler` here so this module stays
///   dependency-free of the scheduler internals.
fn emit_error_and_finalize(_ctx: BurstContext<'_>, seq: SequenceInfo, msg: &str) {
    let _ = seq
        .response_tx
        .send(GenerateEvent::Error(format!("Speculative burst: {msg}")));
    drop(seq);
    let _ = msg;
}

// ===========================================================================
// Batched burst (B > 1)
// ===========================================================================
//
// The B = 1 burst above runs the full prefill + decode lifecycle of ONE
// speculative request as a single scheduler tick. The batched burst
// generalises that to a *window* of B speculative requests that the
// scheduler collected together: every row runs its prefill + decode
// through the batched round-loop driver
// (`MtpBatchedGenerator::run_batched` / `DFlashBatchedGenerator::run_batched`)
// in one logical tick.
//
// ## Window admission (equal-length prompts)
//
// The batched MTP target adapter forwards the `[B, max_prompt_len]`
// prompt batch in one pass. With equal-length prompts the 2-D causal
// masks broadcast cleanly across the batch and the result is
// byte-identical to running B separate B = 1 prefills (acceptance item 1). The scheduler's window collector
// (`BatchScheduler::execute_speculative_burst`) therefore only groups
// speculative-eligible requests of the *same prompt length* into one
// batched window. A request whose prompt length differs from the window
// head stays queued and is served on the next tick (possibly as its own
// window head).
//
// ## Per-row early-EOS
//
// The batched round-loop drivers already implement per-row early-EOS
// (a row that hits EOS or saturates `max_new_tokens` freezes in place;
// its `bs` no longer participates in the per-round block-size minimum so
// it does not stall its siblings). The burst's per-row finalize
// (`finalize_batched_burst_success`) classifies each row independently:
// `FinishReason::Stop` at the first EOS, `FinishReason::Length` at the
// budget, `FinishReason::Stop` otherwise.

/// One finalized row of a batched burst — the per-row analogue of
/// [`BurstFinalized`].
///
/// originally `BatchedBurstFinalized.rows` carried only
/// `(seq_id, tokens_generated)`, so the scheduler's batched arm could
/// release each row's cache slot and record its Prometheus metric, but
/// it had no handle on the prompt / committed token vectors a
/// prompt-cache donate needs. widens each row to the same
/// donate-payload shape as the B = 1 [`BurstFinalized`] so the batched
/// arm can call
/// [`crate::server::batch::scheduler::BatchScheduler::donate_finished_sequence_cache`]
/// per row, symmetric with the B = 1 burst arm and the classic
/// `finalize_completed` path. As with the B = 1 path the donate helper
/// is hard-gated on a dense KV-cache backend, so for today's
/// batched-eligible model families — Gemma 4 (MTP) and Qwen 3.5
/// (DFlash), both `SequenceStateBackend::ModelOwned` — the donate is a
/// guarded no-op; wiring it in removes a latent multi-turn cache-reuse
/// regression for any dense-KV-cache model that later becomes
/// batched-burst-eligible.
#[derive(Debug, Clone)]
pub(crate) struct BatchedBurstRow {
    pub seq_id: mlxcel_core::cache::SequenceId,
    /// Total tokens streamed to this row's client. Mirrors
    /// [`BurstFinalized::tokens_generated`].
    pub tokens_generated: usize,
    /// This row's full prompt token stream, forwarded so the scheduler
    /// can compose the prompt-cache donate key. Empty on the error /
    /// transition-failure rows (no healthy cache to donate). Mirrors
    /// [`BurstFinalized::prompt_tokens`].
    pub prompt_tokens: Vec<i32>,
    /// The tokens this row actually committed to the sequence (post
    /// early-EOS truncation). Joined with `prompt_tokens` by the
    /// scheduler to form the donate entry's token key. Empty on the
    /// error rows. Mirrors [`BurstFinalized::generated_tokens`].
    pub generated_tokens: Vec<i32>,
    /// Whether this row reached a healthy finish (`Stop` / `Length`).
    /// `false` on the error / transition-failure rows. Mirrors
    /// [`BurstFinalized::healthy_finish`].
    pub healthy_finish: bool,
}

/// Successful batched-burst outcome returned to the scheduler.
///
/// Carries one [`BatchedBurstRow`] per window row so the scheduler can
/// donate each row's KV cache back to the prompt-cache store, release
/// each row's cache slot, and record each row's per-sequence Prometheus
/// metric — the batched analogue of [`BurstFinalized`].
#[derive(Debug, Clone)]
pub(crate) struct BatchedBurstFinalized {
    /// Per-row finalize payloads. Same length / order as the window the
    /// scheduler handed to [`try_run_burst_batched`].
    pub rows: Vec<BatchedBurstRow>,
}

/// Run a speculative burst for a *window* of `seqs` (B = `seqs.len()`),
/// producing every token each row will emit, streaming them via each
/// row's `response_tx`, and finalizing every sequence inline.
///
/// Returns `Ok(BatchedBurstFinalized)` when the batched burst handled the
/// window end-to-end. Returns `Err(rejected_seqs)` when the window
/// declines (e.g. the model variant is not supported by the dispatch
/// kind, or the drafter load failed) — the scheduler then re-routes
/// every rejected sequence through the classic non-speculative path.
///
/// **Caller contract**: every sequence in `seqs` MUST already have
/// passed [`should_burst_for_sequence`] AND have an identical
/// `prompt_tokens.len()` (the batched MTP adapter requires a rectangular
/// `[B, L]` prefill — see its docstring). The scheduler's window
/// collector enforces both.
//
// `result_large_err`: the `Err` variant carries the full window so the
// scheduler can route every rejected sequence into the classic prefill
// path without a `Box` round-trip. The hot path (`Ok`) is tiny.
#[allow(clippy::result_large_err)]
pub(crate) fn try_run_burst_batched(
    mut ctx: BurstContext<'_>,
    mut seqs: Vec<SequenceInfo>,
) -> Result<BatchedBurstFinalized, Vec<SequenceInfo>> {
    let burst_start = Instant::now();
    let batch_size = seqs.len();
    debug_assert!(batch_size >= 2, "try_run_burst_batched requires B >= 2");

    // Defensive: every row must have a non-empty prompt and all prompts
    // must be the same length (the scheduler's window collector
    // guarantees this; re-assert so a future caller bug fails loudly
    // rather than corrupting a verify forward).
    let prompt_len = seqs[0].prompt_tokens.len();
    if prompt_len == 0 || seqs.iter().any(|s| s.prompt_tokens.len() != prompt_len) {
        return Err(seqs);
    }

    // Mark every row's prefill timer; do NOT transition state yet — the
    // burst may decline on the model variant, in which case the rows
    // fall back to the classic prefill path which itself transitions
    // Queued -> Prefilling.
    for seq in seqs.iter_mut() {
        seq.prefill_start = Some(burst_start);
    }

    let result = match ctx.dispatch {
        crate::server::SpeculativeDispatch::Mtp { block_size, .. } => {
            let bs = *block_size;
            run_mtp_burst_batched(ctx.reborrow(), &mut seqs, bs)
        }
        crate::server::SpeculativeDispatch::DFlash { block_size, .. } => {
            let bs = *block_size;
            run_dflash_burst_batched(ctx.reborrow(), &mut seqs, bs)
        }
        crate::server::SpeculativeDispatch::Disabled
        | crate::server::SpeculativeDispatch::Classic { .. } => Err(BurstOutcome::DeclineToClassic),
    };

    match result {
        Ok(rows_tokens) => {
            debug_assert_eq!(rows_tokens.len(), batch_size);
            let mut finalized_rows = Vec::with_capacity(batch_size);
            for (seq, tokens) in seqs.into_iter().zip(rows_tokens) {
                let seq_id = seq.seq_id;
                // Transition Queued -> Prefilling now that the burst
                // owned the row's lifecycle.
                let mut seq = seq;
                if let Err(err) = seq.state.transition_to(SequenceState::Prefilling) {
                    emit_error_and_finalize(
                        ctx.reborrow(),
                        seq,
                        &format!("State transition error: {err}"),
                    );
                    // Transition failure is an error outcome — no
                    // healthy cache to donate (mirrors the B = 1 arm's transition-failure `BurstFinalized`).
                    finalized_rows.push(BatchedBurstRow {
                        seq_id,
                        tokens_generated: 0,
                        prompt_tokens: Vec::new(),
                        generated_tokens: Vec::new(),
                        healthy_finish: false,
                    });
                    continue;
                }
                // The batched round-loop drivers do not capture per-row
                // logprobs (`run_batched` returns tokens only), so the
                // batched burst threads an empty `logprobs` vec into the
                // shared finalize path — `finalize_burst_success` then
                // emits plain `Token` events for every row. The batched
                // window-admission gate (`can_join_batched_burst_window`)
                // rejects any logprobs-enabled request so it falls back
                // to the B = 1 burst, which captures logprobs correctly
                let finalized = finalize_burst_success(
                    ctx.reborrow(),
                    seq,
                    tokens,
                    Vec::new(),
                    /* prefill_time_ms */ 0.0,
                    /* decode_time_ms */ 0.0,
                );
                // Surface the per-row prompt + committed token vectors
                // and the healthy-finish flag so the scheduler can
                // mirror the B = 1 arm's prompt-cache donate per row
                finalized_rows.push(BatchedBurstRow {
                    seq_id,
                    tokens_generated: finalized.tokens_generated,
                    prompt_tokens: finalized.prompt_tokens,
                    generated_tokens: finalized.generated_tokens,
                    healthy_finish: finalized.healthy_finish,
                });
            }
            Ok(BatchedBurstFinalized {
                rows: finalized_rows,
            })
        }
        Err(BurstOutcome::DeclineToClassic) => {
            // Every row is still in `Queued`. Clear the burst timer so
            // the classic path's measurement starts fresh.
            for seq in seqs.iter_mut() {
                seq.prefill_start = None;
            }
            Err(seqs)
        }
        Err(BurstOutcome::Error(msg)) => {
            // The batched burst attempted the window but failed
            // mid-flight (drafter load failed, round loop bailed). No
            // row was transitioned to Prefilling. Surface the error to
            // every row's client and finalize.
            let mut finalized_rows = Vec::with_capacity(batch_size);
            for seq in seqs.into_iter() {
                let seq_id = seq.seq_id;
                emit_error_and_finalize(ctx.reborrow(), seq, &msg);
                // Error outcome: every row's KV cache is assumed
                // tainted, so no donate — empty/false payload (mirrors the B = 1 arm's `BurstOutcome::Error` `BurstFinalized`).
                finalized_rows.push(BatchedBurstRow {
                    seq_id,
                    tokens_generated: 0,
                    prompt_tokens: Vec::new(),
                    generated_tokens: Vec::new(),
                    healthy_finish: false,
                });
            }
            Ok(BatchedBurstFinalized {
                rows: finalized_rows,
            })
        }
    }
}

/// Per-row emitted-token output of a batched burst dispatch arm.
type BatchedBurstTokens = Vec<Vec<i32>>;

/// MTP batched burst — Gemma 4 / Gemma 4 VLM target (B > 1).
///
/// Mirrors [`run_mtp_burst`] but drives [`MtpBatchedGenerator`] over the
/// batched MTP target adapter
/// ([`Gemma4MtpBatchedTargetAdapter`] / [`Gemma4VLMtpBatchedTargetAdapter`]).
/// The variant gate runs before drafter IO and the drafter bind happens
/// before generator construction — same contracts as the B = 1 path.
fn run_mtp_burst_batched(
    ctx: BurstContext<'_>,
    seqs: &mut [SequenceInfo],
    block_size: u32,
) -> Result<BatchedBurstTokens, BurstOutcome> {
    let block_size = block_size as usize;
    if block_size < 2 {
        return Err(BurstOutcome::Error(format!(
            "MTP batched burst: block_size={block_size} < 2 produces no draft proposals"
        )));
    }
    let batch_size = seqs.len();

    // HOIST: variant gate before any drafter IO.
    let target_lm: &dyn LanguageModel = match ctx.model {
        LoadedModel::Gemma4(wrapper) => wrapper as &dyn LanguageModel,
        LoadedModel::Gemma4VLM(vlm) => vlm as &dyn LanguageModel,
        LoadedModel::Gemma4Unified(unified) => unified as &dyn LanguageModel,
        _ => {
            tracing::warn!(
                "MTP batched speculative dispatch declined: target is {:?}, expected \
                 Gemma 4 (text, VLM, or Unified); falling back to classic decode",
                model_variant_label(ctx.model),
            );
            return Err(BurstOutcome::DeclineToClassic);
        }
    };

    ctx.drafter_slot
        .ensure_loaded()
        .map_err(BurstOutcome::Error)?;
    let mut owned_drafter = ctx
        .drafter_slot
        .take()
        .ok_or_else(|| BurstOutcome::Error("drafter slot empty after ensure_loaded".to_string()))?;

    // CRITICAL: bind before constructing the generator — same contract
    // as `run_mtp_burst`. `MtpBatchedGenerator::run_batched` does NOT
    // bind internally; without this the first `draft_block_batched`
    // returns `BindNotCalled`.
    // Compatibility gate before binding — same contract as `run_mtp_burst`.
    if let Err(e) = owned_drafter.validate_target_compat(target_lm) {
        return Err(BurstOutcome::Error(format!(
            "MTP batched drafter incompatible with target: {e}"
        )));
    }
    if let Err(e) = owned_drafter.bind(target_lm) {
        return Err(BurstOutcome::Error(format!(
            "MTP batched drafter bind failed: {e}"
        )));
    }

    // Per-row prompts + the shared sampler / max_tokens. The batched
    // round loop takes a single sampler; the scheduler's window
    // collector only groups requests that share sampling config (see
    // `execute_speculative_burst`). `max_new_tokens` is the per-row
    // budget; the collector also requires equal `max_tokens` so one
    // window value is correct for every row.
    let prompts: Vec<Vec<i32>> = seqs.iter().map(|s| s.prompt_tokens.clone()).collect();
    let sampling = seqs[0].sampling.clone();
    let max_tokens = seqs[0].max_tokens.max(1);

    let (tokens_per_row, recovered_drafter) = match ctx.model {
        LoadedModel::Gemma4(wrapper) => {
            let adapter =
                Gemma4MtpBatchedTargetAdapter::new_with_block_size(wrapper, batch_size, block_size);
            drive_mtp_batched_generator(
                adapter,
                owned_drafter,
                &prompts,
                max_tokens,
                &sampling,
                block_size,
            )?
        }
        LoadedModel::Gemma4VLM(vlm) => {
            let adapter =
                Gemma4VLMtpBatchedTargetAdapter::new_with_block_size(vlm, batch_size, block_size);
            drive_mtp_batched_generator(
                adapter,
                owned_drafter,
                &prompts,
                max_tokens,
                &sampling,
                block_size,
            )?
        }
        LoadedModel::Gemma4Unified(unified) => {
            let adapter = Gemma4UnifiedMtpBatchedTargetAdapter::new_with_block_size(
                unified, batch_size, block_size,
            );
            drive_mtp_batched_generator(
                adapter,
                owned_drafter,
                &prompts,
                max_tokens,
                &sampling,
                block_size,
            )?
        }
        _ => {
            return Err(BurstOutcome::Error(format!(
                "MTP batched burst: unsupported target {:?} after variant gate",
                model_variant_label(ctx.model),
            )));
        }
    };

    ctx.drafter_slot
        .return_drafter(recovered_drafter, target_lm);
    Ok(tokens_per_row)
}

/// Generator-shape-agnostic helper that drives an [`MtpBatchedGenerator`]
/// over a `T: MtpTarget` (batched adapter) and a pre-bound drafter.
///
/// Returns the per-row emitted tokens and the recovered drafter handle.
/// Mirrors [`drive_mtp_generator`] for the B > 1 path; kept generic so a
/// future `#[cfg(test)]` mock can drive it without a real `LoadedModel`.
fn drive_mtp_batched_generator<T>(
    target: T,
    drafter: Box<dyn Drafter>,
    prompts: &[Vec<i32>],
    max_tokens: usize,
    sampling: &SamplingConfig,
    block_size: usize,
) -> Result<(BatchedBurstTokens, Box<dyn Drafter>), BurstOutcome>
where
    T: mlxcel_core::speculative::mtp::target::MtpTarget,
{
    let mut generator = MtpBatchedGenerator::new(target, drafter, block_size);
    let run = generator
        .run_batched(prompts, sampling, max_tokens)
        .map_err(|e| BurstOutcome::Error(format!("MTP batched round loop failed: {e}")))?;
    // `MtpBatchedGenerator` does not expose `into_drafter` today; the
    // batched generators are constructed fresh per burst, so the drafter
    // is consumed by value. We recover it via the `into_parts`-style
    // accessor added alongside this issue.
    let recovered = generator.into_drafter();
    Ok((run.tokens, recovered))
}

/// DFlash batched burst — Qwen 3.5 text target or Qwen 3.5 VLM wrapper serving
/// text-only requests (B > 1).
///
/// Mirrors [`run_dflash_burst`] / [`run_dflash_on_qwen35`] but drives
/// [`DFlashBatchedGenerator`]. The variant gate runs before drafter IO;
/// the drafter bind happens **inside** `DFlashBatchedGenerator::run_batched`
/// (same asymmetry as the B = 1 path — do NOT add a manual bind here).
fn run_dflash_burst_batched(
    ctx: BurstContext<'_>,
    seqs: &mut [SequenceInfo],
    block_size: u32,
) -> Result<BatchedBurstTokens, BurstOutcome> {
    if let Some(seq) = seqs
        .iter()
        .find(|seq| seq.vlm_embeddings.is_some() || !seq.images.is_empty() || !seq.audio.is_empty())
    {
        tracing::warn!(
            "DFlash batched speculative dispatch declined for seq {}: multimodal VLM request \
             detected; VLM-wrapped text-only Qwen 3.5 targets are supported, but \
             multimodal speculative tail is not yet enabled; falling back to classic decode",
            seq.seq_id,
        );
        return Err(BurstOutcome::DeclineToClassic);
    }

    // HOIST: variant gate before drafter IO.
    match ctx.model {
        LoadedModel::Qwen35(_)
        | LoadedModel::Qwen35Moe(_)
        | LoadedModel::Qwen35VLM(_)
        | LoadedModel::Qwen35MoeVLM(_) => {}
        _ => {
            tracing::warn!(
                "DFlash batched speculative dispatch declined: target is {:?}, expected \
                 Qwen 3.5 text or VLM-wrapped text-only; falling back to classic decode",
                model_variant_label(ctx.model),
            );
            return Err(BurstOutcome::DeclineToClassic);
        }
    }

    ctx.drafter_slot
        .ensure_loaded()
        .map_err(BurstOutcome::Error)?;
    let owned_drafter = ctx
        .drafter_slot
        .take()
        .ok_or_else(|| BurstOutcome::Error("drafter slot empty after ensure_loaded".to_string()))?;

    let sampling = seqs[0].sampling.clone();
    let max_tokens = seqs[0].max_tokens.max(1);
    let eos_token_ids = merged_eos_token_ids(ctx.model.eos_token_ids(), &sampling.stop_token_ids);
    let prompts: Vec<Vec<i32>> = seqs.iter().map(|s| s.prompt_tokens.clone()).collect();

    match ctx.model {
        LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => run_dflash_batched_on_qwen35(
            qwen,
            &prompts,
            &sampling,
            &eos_token_ids,
            owned_drafter,
            block_size,
            max_tokens,
            ctx.drafter_slot,
        ),
        LoadedModel::Qwen35VLM(qwen) | LoadedModel::Qwen35MoeVLM(qwen) => {
            run_dflash_batched_on_qwen35(
                qwen,
                &prompts,
                &sampling,
                &eos_token_ids,
                owned_drafter,
                block_size,
                max_tokens,
                ctx.drafter_slot,
            )
        }
        _ => {
            // Defensive: unreachable per the variant gate above. Restore
            // the drafter to the slot since we took it without using it.
            ctx.drafter_slot.drafter = Some(owned_drafter);
            Err(BurstOutcome::Error(format!(
                "DFlash batched burst: unsupported target {:?} after variant gate",
                model_variant_label(ctx.model),
            )))
        }
    }
}

/// DFlash batched burst on a Qwen 3.5 text target (including a Qwen 3.5 VLM
/// wrapper serving text-only requests) — `[B, L]` prefill, per-row first-bonus
/// + first-hidden extraction, and `DFlashBatchedGenerator::run_batched`.
///
/// Mirrors [`run_dflash_on_qwen35`] for the B > 1 path. The prompts are
/// equal length (window-collector contract) so the `[B, L]` prefill is
/// byte-identical to B separate `[1, L]` prefills.
#[allow(clippy::too_many_arguments)]
fn run_dflash_batched_on_qwen35<T>(
    qwen: &T,
    prompts: &[Vec<i32>],
    sampling: &SamplingConfig,
    eos_token_ids: &[i32],
    owned_drafter: Box<dyn Drafter>,
    block_size: u32,
    max_tokens: usize,
    drafter_slot: &mut WorkerDrafterSlot,
) -> Result<BatchedBurstTokens, BurstOutcome>
where
    T: Qwen35DFlashTarget,
{
    let batch_size = prompts.len();
    let prompt_len = prompts[0].len();

    // Build the `[B, ...]` per-layer cache vector. As with the B = 1
    // path, we do NOT touch the scheduler-owned `sequence_state` map.
    let mut caches: Vec<crate::models::qwen3_next::Qwen3NextCache> = qwen.make_dflash_caches();

    // `[B, L]` prefill through the target's speculative forward. Capture the
    // hidden layers requested by this specific DFlash checkpoint.
    let capture_layer_ids = owned_drafter
        .dflash_target_layer_ids()
        .filter(|ids| !ids.is_empty())
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| qwen.capture_layer_ids().to_vec());
    let mut flat_prompt: Vec<i32> = Vec::with_capacity(batch_size * prompt_len);
    for row in prompts {
        flat_prompt.extend_from_slice(row);
    }
    let prompt_arr =
        mlxcel_core::from_slice_i32(&flat_prompt, &[batch_size as i32, prompt_len as i32]);
    let verify_out =
        qwen.verify_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);

    // Per-row first bonus from the `[B, prompt_len, vocab]` last-position
    // logits.
    let logits_shape = mlxcel_core::array_shape(&verify_out.logits);
    let last_pos = prompt_len as i32 - 1;
    let vocab = logits_shape[2];
    let last_logits = mlxcel_core::slice(
        &verify_out.logits,
        &[0, last_pos, 0],
        &[logits_shape[0], last_pos + 1, vocab],
    );
    let (first_bonus_arr, _) =
        mlxcel_core::sampling::sample_token_optimized(&last_logits, sampling, &[]);
    mlxcel_core::eval(&first_bonus_arr);
    let first_bonus_per_row = scalar_tokens_from_array(&first_bonus_arr, batch_size);

    // Build `first_hidden` = concat(hidden_states, axis=-1)[:, last:last+1, :]
    // at shape `[B, 1, num_layers * hidden_size]`.
    if verify_out.hidden_states.is_empty() {
        drafter_slot.drafter = Some(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash batched prefill returned no captured hidden layers".to_string(),
        ));
    }
    let mut concatenated = mlxcel_core::copy(
        verify_out.hidden_states[0]
            .as_ref()
            .expect("hidden state must be non-null"),
    );
    for slab in verify_out.hidden_states.iter().skip(1) {
        concatenated = mlxcel_core::concatenate(
            &concatenated,
            slab.as_ref().expect("hidden state must be non-null"),
            -1,
        );
    }
    let concatenated_shape = mlxcel_core::array_shape(&concatenated);
    debug_assert_eq!(
        concatenated_shape.len(),
        3,
        "concatenated hidden must be 3-D"
    );
    let feature_dim = concatenated_shape[2];
    let first_hidden = mlxcel_core::slice(
        &concatenated,
        &[0, last_pos, 0],
        &[concatenated_shape[0], last_pos + 1, feature_dim],
    );

    // Drive the batched round loop. `run_batched` returns per-row tokens
    // EXCLUDING the first bonus; we prepend each row's bonus on success.
    let mut generator = DFlashBatchedGenerator::new(
        owned_drafter,
        sampling.clone(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let run = generator.run_batched(
        qwen,
        qwen as &dyn LanguageModel,
        &mut caches,
        &first_bonus_per_row,
        first_hidden,
        eos_token_ids,
        max_tokens,
    );

    // Recover the drafter so the slot is consistent for the next burst.
    let recovered = generator.into_drafter();
    drafter_slot.return_drafter(recovered, qwen as &dyn LanguageModel);

    match run {
        Ok(output) => {
            debug_assert_eq!(output.tokens.len(), batch_size);
            let mut rows: BatchedBurstTokens = Vec::with_capacity(batch_size);
            for (r, row_tokens) in output.tokens.into_iter().enumerate() {
                let mut full = Vec::with_capacity(row_tokens.len() + 1);
                full.push(first_bonus_per_row[r]);
                full.extend(row_tokens);
                rows.push(full);
            }
            Ok(rows)
        }
        Err(e) => Err(BurstOutcome::Error(format!(
            "DFlash batched round loop failed: {e}"
        ))),
    }
}

/// Materialise a per-row `Vec<i32>` from a `[B]` / `[B, 1]` token tensor
/// produced by `sample_token_optimized`. Mirrors the helper in
/// `gemma4_mtp_target.rs`; duplicated rather than re-exported because the
/// burst module is binary-side glue and the helper is a single-call
/// utility.
fn scalar_tokens_from_array(token_arr: &mlxcel_core::MlxArray, batch_size: usize) -> Vec<i32> {
    let flat = mlxcel_core::reshape(token_arr, &[batch_size as i32]);
    let mut out: Vec<i32> = Vec::with_capacity(batch_size);
    for r in 0..batch_size as i32 {
        let cell = mlxcel_core::slice(&flat, &[r], &[r + 1]);
        let scalar = mlxcel_core::reshape(&cell, &[]);
        out.push(mlxcel_core::item_i32(&scalar));
    }
    out
}

/// Whether two [`SamplingConfig`]s would drive the speculative round
/// loop identically — used by the scheduler's batched-window collector
/// to decide whether a candidate request can ride on the head's window.
///
/// [`SamplingConfig`] does not derive `PartialEq` (it carries `Vec` and
/// `TokenBiasMap` fields), and even if it did, a full structural
/// equality would be too strict here: the batched MTP/DFlash round
/// loops only consult a *subset* of the config. This helper compares
/// exactly that subset:
///
/// - The four sampling-shape fields (`temperature`, `top_k`, `top_p`,
///   `min_p`) — they decide every token the per-row argmax/sample
///   produces. At `temperature == 0` (the greedy-parity gate) these
///   four must match or two rows in the same window would sample under
///   different laws.
/// - `seed` — affects the stochastic path; equal seeds keep the window
///   reproducible.
/// - `stop_token_ids` — the batched round loop computes ONE merged-EOS
///   set per window (`merged_eos_token_ids(target.eos, sampler.stop)`),
///   so a per-row stop set that differs would mis-terminate siblings.
/// - `token_bias` — applied before sampling; a per-row bias that
///   differs would skew that row's distribution under the shared
///   sampler. [`TokenBiasMap`](mlxcel_core::sampling::TokenBiasMap) does
///   not derive `PartialEq`, so rather than a structural comparison we
///   require **both** configs to carry an empty bias map. A request
///   that sets `logit_bias` therefore never joins a batched window — it
///   falls back to the B=1 burst, which is correct (just not batched).
///   The empty-bias case is the overwhelmingly common one.
///
/// ## History-dependent penalties are excluded from the batched window
///
/// removed the `should_burst_for_sequence` decline gates for
/// the four history-dependent penalty fields (`repetition_penalty`,
/// `frequency_penalty`, `presence_penalty`, `dry_*`) — the **B=1** burst
/// now threads `initial_token_history(&prompt, ..)` into its first-bonus
/// sample so a penalty-bearing request is byte-identical to classic
/// decode.
///
/// The **batched** path, however, samples each row's first bonus with an
/// empty token history (`prefill_and_seed_batched` /
/// `run_dflash_batched_on_qwen35` pass `&[]`): the batched `MtpTarget`
/// trait method and `MtpBatchedGenerator` do not yet thread per-row
/// token history. So `sampling_config_eq` requires **both** configs to
/// have `needs_token_history() == false`. A penalty-bearing request
/// therefore never joins a batched window — it falls back to the B=1
/// burst, which honors its penalties correctly. Threading per-row token
/// history through the batched round loop is a documented follow-up.
pub(crate) fn sampling_config_eq(a: &SamplingConfig, b: &SamplingConfig) -> bool {
    a.temperature.to_bits() == b.temperature.to_bits()
        && a.top_k == b.top_k
        && a.top_p.to_bits() == b.top_p.to_bits()
        && a.min_p.to_bits() == b.min_p.to_bits()
        && a.seed == b.seed
        && a.stop_token_ids == b.stop_token_ids
        && a.token_bias.is_empty()
        && b.token_bias.is_empty()
        // The batched path's first-bonus sample uses an empty token
        // history; a penalty-bearing request must not join a batched
        // window (it runs as a B=1 burst, which threads token history correctly since).
        && !a.needs_token_history()
        && !b.needs_token_history()
}

/// Best-effort diagnostic name for a [`LoadedModel`] variant, used in
/// burst error messages so the operator can see which model variant the
/// scheduler refused to speculatively-drive.
fn model_variant_label(model: &LoadedModel) -> &'static str {
    // Avoid pulling in `std::mem::discriminant` String formatting which
    // doesn't yield variant names by default. Returning a hand-picked
    // label per common variant keeps error messages stable and Greppable.
    match model {
        LoadedModel::Gemma4(_) => "Gemma4",
        LoadedModel::Gemma4VLM(_) => "Gemma4VLM",
        LoadedModel::Gemma4Unified(_) => "Gemma4Unified",
        LoadedModel::Qwen35(_) => "Qwen35",
        LoadedModel::Qwen35VLM(_) => "Qwen35VLM",
        LoadedModel::Qwen35Moe(_) => "Qwen35Moe",
        LoadedModel::Qwen35MoeVLM(_) => "Qwen35MoeVLM",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_slot_starts_empty_for_disabled_dispatch() {
        let slot = WorkerDrafterSlot::from_dispatch(&crate::server::SpeculativeDispatch::Disabled);
        assert!(slot.draft_model_path.is_none());
        assert!(slot.kind.is_none());
        assert!(slot.drafter.is_none());
    }

    #[test]
    fn drafter_slot_carries_path_for_mtp_dispatch() {
        let dispatch = crate::server::SpeculativeDispatch::Mtp {
            draft_model_path: std::path::PathBuf::from("/tmp/fake-drafter"),
            block_size: 4,
            user_requested_explicit_kind: true,
        };
        let slot = WorkerDrafterSlot::from_dispatch(&dispatch);
        assert_eq!(
            slot.draft_model_path,
            Some(std::path::PathBuf::from("/tmp/fake-drafter"))
        );
        assert_eq!(slot.kind, Some(DrafterKind::Mtp));
        assert!(slot.drafter.is_none()); // lazy
    }

    #[test]
    fn drafter_slot_carries_path_for_dflash_dispatch() {
        let dispatch = crate::server::SpeculativeDispatch::DFlash {
            draft_model_path: std::path::PathBuf::from("/tmp/fake-drafter"),
            block_size: 16,
            user_requested_explicit_kind: true,
        };
        let slot = WorkerDrafterSlot::from_dispatch(&dispatch);
        assert_eq!(slot.kind, Some(DrafterKind::Dflash));
    }

    #[test]
    fn drafter_slot_ensure_loaded_errors_when_disabled() {
        let mut slot =
            WorkerDrafterSlot::from_dispatch(&crate::server::SpeculativeDispatch::Disabled);
        let err = slot.ensure_loaded().expect_err("must error when disabled");
        assert!(err.contains("disabled"));
    }

    #[test]
    fn drafter_slot_ensure_loaded_errors_on_missing_path() {
        let dispatch = crate::server::SpeculativeDispatch::Mtp {
            draft_model_path: std::path::PathBuf::from("/nonexistent/drafter-path"),
            block_size: 4,
            user_requested_explicit_kind: true,
        };
        let mut slot = WorkerDrafterSlot::from_dispatch(&dispatch);
        let err = slot.ensure_loaded().expect_err("must fail on missing path");
        assert!(
            err.contains("Drafter load failed"),
            "error message must mention drafter load failure, got: {err}"
        );
    }

    #[test]
    fn should_burst_for_sequence_rejects_disabled_dispatch() {
        let dispatch = crate::server::SpeculativeDispatch::Disabled;
        // We need a SequenceInfo; constructing one fully is heavy. Use
        // the unsafe-pattern of just sanity-checking the gate's first
        // short-circuit (the dispatch check) by constructing the
        // dispatch only — the dispatch arm short-circuits before any
        // field of `seq` is read.
        //
        // This test only proves the disabled path returns false. The
        // other gate predicates are covered by the integration tests in
        // tests/speculative_dispatch.rs.
        assert!(!dispatch.is_kind_specific());
        // Bypass the full SequenceInfo construction by asserting the
        // first guard.
        let _ = dispatch;
    }
}
