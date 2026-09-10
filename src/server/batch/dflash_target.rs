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

//! Target contract of the server-side DFlash bursts, and the generic drivers
//! that run on it (issue #1339).
//!
//! `DFlashGenerator` already accepts any
//! [`mlxcel_core::drafter::dflash::SpeculativeTarget`], but the server side
//! of a burst also has to allocate the target's heterogeneous cache vector,
//! read logits and captured hidden states off the target's own verify-output
//! type, and decide how much of the prompt's captured hidden the first draft
//! sees. [`DFlashTargetModel`] holds exactly those three things, so a new
//! family plugs in with one `impl` block here plus a match arm in the burst
//! gate:
//!
//! - Qwen 3.5 (DFlash): fresh `Qwen3NextCache`s, the last prompt position's
//!   hidden as the first draft input.
//! - LFM2 / LFM2.5 (DSpark): fresh `Lfm2LayerCache`s, EVERY prompt row as the
//!   first draft input (the drafter's own context cache holds the whole
//!   prompt), and the block-versus-chain exactness probe as a gate.
//!
//! The VLM wrappers of both families implement the trait by delegating to
//! their text backbone, which is what lets a text-only request against a
//! VLM checkpoint run the burst.
//!
//! What the trait deliberately does NOT carry: a per-round block-size
//! override. `DFlashGenerator` holds one `block_size` for the whole run and
//! the burst hands it in once, so a family that wants a narrower first round
//! (to amortize a cold cache) has to change the round loop, not this trait.
//! Adding a hook here that the round loop cannot honour would read as
//! supported and do nothing.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use mlxcel_core::drafter::Drafter;
use mlxcel_core::drafter::DrafterKind;
use mlxcel_core::drafter::dflash::{DFlashBatchedGenerator, DFlashGenerator, SpeculativeTarget};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::TokenLogprobData;
use mlxcel_core::{MlxArray, UniquePtr};

use super::speculative_burst::{BurstOutcome, WorkerDrafterSlot};
use crate::server::model_provider::SpeculativeStats;

/// Uniform access to what the burst reads off a target's verify output.
///
/// Each family's `VerifyOut` is its own struct (Qwen's carries GDN
/// snapshots, LFM2's carries short-conv snapshots); the burst only ever
/// needs the logits and the captured hidden slabs.
pub(crate) trait DFlashVerifyOutput {
    /// `[B, L, vocab]` logits for every verify row.
    fn logits(&self) -> &MlxArray;
    /// One `[B, L, hidden]` slab per captured layer, in capture order.
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>];
}

impl DFlashVerifyOutput for crate::models::qwen3_5::VerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("verify logits must be non-null")
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

impl DFlashVerifyOutput for crate::models::lfm2::VerifyOutput {
    fn logits(&self) -> &MlxArray {
        self.logits
            .as_ref()
            .expect("verify logits must be non-null")
    }
    fn hidden_states(&self) -> &[UniquePtr<MlxArray>] {
        &self.hidden_states
    }
}

/// How much of the prompt's captured hidden the first draft round consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FirstHiddenRows {
    /// Only the last prompt position (`[B, 1, dim]`). The Qwen 3.5 DFlash
    /// drafter starts its context cache from the bonus token onwards.
    LastPromptPosition,
    /// Every prompt row (`[B, S, dim]`). The LFM2 DSpark drafter appends the
    /// whole prompt to its context cache before its first proposal.
    EveryPromptRow,
}

/// Server-side target contract of the DFlash bursts.
///
/// `SpeculativeTarget` is what the round loop drives; this trait adds what
/// the burst around it needs. The associated-type bound lets the generic
/// drivers below read logits and hidden states without naming any family's
/// `VerifyOut`.
pub(crate) trait DFlashTargetModel:
    LanguageModel + SpeculativeTarget<VerifyOut: DFlashVerifyOutput>
{
    /// A fresh per-layer cache vector for one burst. Independent of the
    /// scheduler-owned per-sequence slots, which is why an adopted
    /// prompt-cache prefix declines to classic decode on this path.
    fn make_dflash_caches(&self) -> Vec<Self::Cache>;

    /// Which prompt rows the first draft sees. Qwen 3.5 keeps the last
    /// position; DSpark wants every row.
    ///
    /// An associated function rather than a method: the policy is a
    /// property of the drafter family the target is paired with, not of a
    /// loaded instance, and taking no `self` is what lets the unit test
    /// below pin each family's answer without a checkpoint on disk.
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::LastPromptPosition
    }

    /// Hook for a target whose caches need arming for a multi-row verify
    /// (for example a rotating buffer sized to the block). No-op by
    /// default; runs on the fresh caches before the prefill.
    fn enable_speculative_buffers(&self, _caches: &mut [Self::Cache], _block_size: usize) {}

    /// Whether a `block_size`-row verify block is byte-identical to the
    /// single-token decode chain on this host, and the burst may therefore
    /// promise the temperature-0 contract. Qwen 3.5 DFlash has never gated on
    /// this and keeps the permissive default; LFM2 runs the measured probe.
    fn exactness_allows(&self, _block_size: usize) -> bool {
        true
    }
}

impl DFlashTargetModel for crate::models::Qwen35Model {
    fn make_dflash_caches(&self) -> Vec<crate::models::qwen3_next::Qwen3NextCache> {
        self.make_speculative_caches()
    }
}

impl DFlashTargetModel for crate::vision::Qwen35VLModel {
    fn make_dflash_caches(&self) -> Vec<crate::models::qwen3_next::Qwen3NextCache> {
        self.text_model.make_speculative_caches()
    }
}

impl DFlashTargetModel for crate::models::Lfm2Model {
    fn make_dflash_caches(&self) -> Vec<crate::models::lfm2::Lfm2LayerCache> {
        self.make_speculative_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.dflash_exactness_allows(block_size)
    }
}

impl DFlashTargetModel for crate::vision::Lfm2VlModel {
    fn make_dflash_caches(&self) -> Vec<crate::models::lfm2::Lfm2LayerCache> {
        self.text_model.make_speculative_caches()
    }
    fn first_hidden_rows() -> FirstHiddenRows {
        FirstHiddenRows::EveryPromptRow
    }
    fn exactness_allows(&self, block_size: usize) -> bool {
        self.text_model.dflash_exactness_allows(block_size)
    }
}

/// What one DFlash target run hands back to `run_dflash_burst`.
///
/// A struct rather than a tuple because the run grew a fourth member with
/// issue #1314 and a four-element tuple of two vectors, a float and an option
/// is read once and then guessed at forever.
pub(crate) struct DFlashTargetRun {
    /// Every token the run emitted, the first bonus included.
    pub(crate) tokens: Vec<i32>,
    /// Per-token logprob payloads, index-aligned with [`Self::tokens`]; empty
    /// when the request did not ask for logprobs.
    pub(crate) logprobs: Vec<Option<TokenLogprobData>>,
    /// Round-loop decode wall clock in milliseconds.
    pub(crate) decode_time_ms: f64,
    /// Client-facing drafter acceptance counters (issue #1314); `None` when
    /// the run executed no verify round.
    pub(crate) speculative: Option<SpeculativeStats>,
}

/// Concatenate the captured hidden slabs along the feature axis:
/// `[B, L, num_layers * hidden]`.
fn concat_captured_hidden(slabs: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let refs: Vec<&MlxArray> = slabs
        .iter()
        .map(|slab| slab.as_ref().expect("hidden state must be non-null"))
        .collect();
    mlxcel_core::concatenate_many(&refs, -1)
}

/// The prompt rows the first draft round consumes, per
/// [`DFlashTargetModel::first_hidden_rows`].
///
/// Takes the policy rather than the target so the two branches are
/// testable without a loaded checkpoint; the call sites read it off
/// [`DFlashTargetModel::first_hidden_rows`].
fn first_hidden_for(
    rows: FirstHiddenRows,
    concatenated: &MlxArray,
    last_pos: i32,
) -> UniquePtr<MlxArray> {
    match rows {
        FirstHiddenRows::EveryPromptRow => mlxcel_core::copy(concatenated),
        FirstHiddenRows::LastPromptPosition => {
            let shape = mlxcel_core::array_shape(concatenated);
            debug_assert_eq!(shape.len(), 3, "concatenated hidden must be 3-D");
            mlxcel_core::slice(
                concatenated,
                &[0, last_pos, 0],
                &[shape[0], last_pos + 1, shape[2]],
            )
        }
    }
}

/// DFlash B = 1 burst on any [`DFlashTargetModel`]: prefill through the
/// target's speculative verify hook, first-bonus and first-hidden
/// extraction, then `DFlashGenerator::run`.
///
/// `token_history` is the history-dependent-penalty context for the
/// first-bonus sample (repetition / frequency / presence / DRY); it is
/// forwarded to `sample_token_optimized` so a penalty-bearing request's first
/// bonus is byte-identical to the classic decode path. The round loop itself
/// runs greedy at temp=0, so the per-round target argmax is unaffected; only
/// the first bonus reads the history.
///
/// `cancel` is checked once per round so a disconnected client's burst stops
/// occupying the worker thread. `logprobs_config` controls per-token
/// log-probability capture: the first-bonus logprob is computed here from
/// the same penalty-adjusted logits the bonus was sampled from, and the
/// round-loop tokens' logprobs come back in `DFlashRunOutput::logprobs`.
///
/// **Drafter bind happens inside the round loop.** `DFlashGenerator::run`
/// binds and resets the drafter on every invocation; do not bind here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dflash_on_target<T>(
    target: &T,
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
) -> Result<DFlashTargetRun, BurstOutcome>
where
    T: DFlashTargetModel,
{
    // Fresh caches for this request; the scheduler-owned `sequence_state`
    // map is never touched, which is why `run_dflash_burst` declines an
    // adopted prompt-cache prefix (`prefill_start_offset > 0`) to classic
    // decode before reaching here.
    // Compatibility gate BEFORE the round loop binds. `DFlashGenerator::run`
    // binds internally, and a bind that succeeds on a mismatched pairing
    // fails later as a shape error deep inside the drafter forward: a DSpark
    // drafter borrows the target's embedding table as both its input
    // embedding and its LM head, and its Markov head is a transition table
    // over the target's vocabulary. The default trait impl is a no-op, so the
    // Qwen 3.5 DFlash pairing is unaffected.
    if let Err(e) = owned_drafter.validate_target_compat(target as &dyn LanguageModel) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {e}"
        )));
    }

    let mut caches: Vec<T::Cache> = target.make_dflash_caches();
    target.enable_speculative_buffers(&mut caches, block_size as usize);

    // Prefill through the speculative verify hook, capturing the layers the
    // drafter checkpoint asks for (`target_layer_ids`).
    let capture_layer_ids = owned_drafter
        .dflash_target_layer_ids()
        .filter(|ids| !ids.is_empty())
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| target.capture_layer_ids().to_vec());
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let prefill_verify_start = Instant::now();
    let verify_out =
        target.verify_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);
    let prefill_verify_ms = prefill_verify_start.elapsed().as_secs_f64() * 1000.0;

    // First bonus from the last-position logits, under the request's
    // sampler and penalty history so it matches the classic path's first
    // token byte for byte.
    let first_bonus_start = Instant::now();
    let last_pos = prompt_tokens.len() as i32 - 1;
    let logits = verify_out.logits();
    let logits_shape = mlxcel_core::array_shape(logits);
    let vocab = logits_shape[2];
    let last_logits = mlxcel_core::slice(
        logits,
        &[0, last_pos, 0],
        &[logits_shape[0], last_pos + 1, vocab],
    );
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

    // First draft input: the captured hidden concatenated along the feature
    // axis, then either the last prompt position or every prompt row.
    let first_hidden_start = Instant::now();
    let hidden_states = verify_out.hidden_states();
    if hidden_states.is_empty() {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash prefill returned no captured hidden layers".to_string(),
        ));
    }
    let concatenated = concat_captured_hidden(hidden_states);
    let first_hidden = first_hidden_for(T::first_hidden_rows(), &concatenated, last_pos);
    let first_hidden_ms = first_hidden_start.elapsed().as_secs_f64() * 1000.0;

    // Release the prefill's rollback snapshots before the round loop starts.
    // They are sized by the PROMPT, not by a verify block (LFM2 keeps one
    // `[1, S, hidden]` gated conv input per short-conv layer, Qwen 3.5 its GDN
    // states), and nothing past this point reads them: a prefill is never
    // rolled back. Holding `verify_out` to end of scope would pin that for the
    // whole generation. The captured hidden slabs survive regardless, because
    // the lazy `first_hidden` graph references them.
    drop(verify_out);

    let mut generator = DFlashGenerator::new(
        owned_drafter,
        sampling.clone(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let result = generator.run(
        target,
        target as &dyn LanguageModel,
        &mut caches,
        first_bonus,
        first_hidden,
        eos_token_ids,
        max_tokens,
        cancel,
        logprobs_config,
    );

    // Whatever the round loop's outcome, recover the drafter so the slot is
    // consistent for the next request.
    let recovered = generator.into_drafter();
    drafter_slot.return_drafter(recovered, target as &dyn LanguageModel);

    match result {
        Ok(output) => {
            let diagnostics = output.diagnostics.clone();
            // b10621 spec_decode_* Prometheus counters (#1440).
            super::observability::spec_counters::record(
                diagnostics.proposed_tokens,
                diagnostics.accepted_tokens,
                diagnostics.rounds,
            );
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
            // Same assembly as `tokens`: the first-bonus logprob prepended to
            // the round loop's. Stays empty when logprobs are disabled so
            // `finalize_burst_success` emits plain `Token` events.
            let logprobs: Vec<Option<TokenLogprobData>> = if logprobs_config.enabled {
                let mut lp = Vec::with_capacity(output.logprobs.len() + 1);
                lp.push(first_bonus_lp);
                lp.extend(output.logprobs);
                lp
            } else {
                Vec::new()
            };
            Ok(DFlashTargetRun {
                tokens,
                logprobs,
                decode_time_ms: output.stats.decode_time_ms,
                // Client-facing acceptance counters (issue #1314) from the
                // same diagnostics the log line reports. The kind is `Dflash`
                // for DSpark too: it is the DFlash round loop and dispatch
                // variant that served the request.
                speculative: SpeculativeStats::from_counts(
                    DrafterKind::Dflash,
                    diagnostics.rounds,
                    diagnostics.proposed_tokens,
                    diagnostics.accepted_tokens,
                ),
            })
        }
        Err(e) => Err(BurstOutcome::Error(format!(
            "DFlash round loop failed: {e}"
        ))),
    }
}

/// Per-row emitted-token output of a batched burst dispatch arm.
pub(crate) type BatchedBurstTokens = Vec<Vec<i32>>;

/// DFlash batched burst (B > 1) on any [`DFlashTargetModel`]: `[B, L]`
/// prefill, per-row first-bonus and first-hidden extraction, then
/// `DFlashBatchedGenerator::run_batched`.
///
/// Mirrors [`run_dflash_on_target`] for the B > 1 path. The prompts are
/// equal length (window-collector contract) so the `[B, L]` prefill is
/// byte-identical to B separate `[1, L]` prefills. The drafter binds inside
/// `run_batched`, as on the B = 1 path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dflash_batched_on_target<T>(
    target: &T,
    prompts: &[Vec<i32>],
    sampling: &SamplingConfig,
    eos_token_ids: &[i32],
    owned_drafter: Box<dyn Drafter>,
    block_size: u32,
    max_tokens: usize,
    drafter_slot: &mut WorkerDrafterSlot,
) -> Result<(BatchedBurstTokens, Instant), BurstOutcome>
where
    T: DFlashTargetModel,
{
    let batch_size = prompts.len();
    let prompt_len = prompts[0].len();

    // Compatibility gate BEFORE the round loop binds. `DFlashGenerator::run`
    // binds internally, and a bind that succeeds on a mismatched pairing
    // fails later as a shape error deep inside the drafter forward: a DSpark
    // drafter borrows the target's embedding table as both its input
    // embedding and its LM head, and its Markov head is a transition table
    // over the target's vocabulary. The default trait impl is a no-op, so the
    // Qwen 3.5 DFlash pairing is unaffected.
    if let Err(e) = owned_drafter.validate_target_compat(target as &dyn LanguageModel) {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(format!(
            "DFlash drafter incompatible with target: {e}"
        )));
    }

    let mut caches: Vec<T::Cache> = target.make_dflash_caches();
    target.enable_speculative_buffers(&mut caches, block_size as usize);

    let capture_layer_ids = owned_drafter
        .dflash_target_layer_ids()
        .filter(|ids| !ids.is_empty())
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| target.capture_layer_ids().to_vec());
    let mut flat_prompt: Vec<i32> = Vec::with_capacity(batch_size * prompt_len);
    for row in prompts {
        flat_prompt.extend_from_slice(row);
    }
    let prompt_arr =
        mlxcel_core::from_slice_i32(&flat_prompt, &[batch_size as i32, prompt_len as i32]);
    let verify_out =
        target.verify_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);

    // Per-row first bonus from the `[B, prompt_len, vocab]` last-position
    // logits.
    let logits = verify_out.logits();
    let logits_shape = mlxcel_core::array_shape(logits);
    let last_pos = prompt_len as i32 - 1;
    let vocab = logits_shape[2];
    let last_logits = mlxcel_core::slice(
        logits,
        &[0, last_pos, 0],
        &[logits_shape[0], last_pos + 1, vocab],
    );
    let (first_bonus_arr, _) =
        mlxcel_core::sampling::sample_token_optimized(&last_logits, sampling, &[]);
    mlxcel_core::eval(&first_bonus_arr);
    let first_bonus_per_row = scalar_tokens_from_array(&first_bonus_arr, batch_size);
    // Target prefill done, first bonus sampled: round 0 starts here.
    let prefill_end = Instant::now();

    let hidden_states = verify_out.hidden_states();
    if hidden_states.is_empty() {
        drafter_slot.restore_unused(owned_drafter);
        return Err(BurstOutcome::Error(
            "DFlash batched prefill returned no captured hidden layers".to_string(),
        ));
    }
    let concatenated = concat_captured_hidden(hidden_states);
    let first_hidden = first_hidden_for(T::first_hidden_rows(), &concatenated, last_pos);
    // Same reason as the B = 1 arm: the prefill's rollback snapshots are
    // prompt-sized and a prefill is never rolled back.
    drop(verify_out);

    let mut generator = DFlashBatchedGenerator::new(
        owned_drafter,
        sampling.clone(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let run = generator.run_batched(
        target,
        target as &dyn LanguageModel,
        &mut caches,
        &first_bonus_per_row,
        first_hidden,
        eos_token_ids,
        max_tokens,
    );

    let recovered = generator.into_drafter();
    drafter_slot.return_drafter(recovered, target as &dyn LanguageModel);

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
            Ok((rows, prefill_end))
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
fn scalar_tokens_from_array(token_arr: &MlxArray, batch_size: usize) -> Vec<i32> {
    let flat = mlxcel_core::reshape(token_arr, &[batch_size as i32]);
    let mut out: Vec<i32> = Vec::with_capacity(batch_size);
    for r in 0..batch_size as i32 {
        let cell = mlxcel_core::slice(&flat, &[r], &[r + 1]);
        let scalar = mlxcel_core::reshape(&cell, &[]);
        out.push(mlxcel_core::item_i32(&scalar));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one policy split this trait exists to carry (issue #1339): a Qwen
    /// 3.5 DFlash drafter starts from the last prompt position, an LFM2
    /// DSpark drafter from the whole prompt. Both branches of
    /// [`first_hidden_for`] run here, because feeding a DSpark drafter one
    /// row instead of the prompt is not a shape error anywhere: it silently
    /// leaves the drafter's context cache holding one row and degrades
    /// acceptance.
    #[test]
    fn first_hidden_for_slices_only_under_the_last_position_policy() {
        let _runtime = crate::initialize_runtime();
        // `[1, 4, 6]`: four prompt rows of six features.
        let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let concatenated = mlxcel_core::from_slice_f32(&values, &[1, 4, 6]);
        let last_pos = 3;

        let whole = first_hidden_for(FirstHiddenRows::EveryPromptRow, &concatenated, last_pos);
        assert_eq!(mlxcel_core::array_shape(&whole), vec![1, 4, 6]);
        assert_eq!(mlxcel_core::utils::array_to_vec_f32(&whole), values);

        let last = first_hidden_for(FirstHiddenRows::LastPromptPosition, &concatenated, last_pos);
        assert_eq!(mlxcel_core::array_shape(&last), vec![1, 1, 6]);
        assert_eq!(
            mlxcel_core::utils::array_to_vec_f32(&last),
            vec![18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
        );
    }

    /// The per-family policies, so a future `impl` that forgets
    /// `first_hidden_rows` is caught here rather than as a quiet acceptance
    /// regression on the LFM2 pairing.
    #[test]
    fn first_hidden_rows_defaults_to_the_qwen_policy() {
        assert_eq!(
            FirstHiddenRows::LastPromptPosition,
            <crate::models::Qwen35Model as DFlashTargetModel>::first_hidden_rows()
        );
        assert_eq!(
            FirstHiddenRows::EveryPromptRow,
            <crate::models::Lfm2Model as DFlashTargetModel>::first_hidden_rows()
        );
    }
}
