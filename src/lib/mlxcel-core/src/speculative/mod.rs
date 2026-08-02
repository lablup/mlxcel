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

//! Speculative decoding for accelerated inference
//!
//! Uses a small "draft" model to generate candidate tokens, then verifies
//! them in batch with the main model. Accepted tokens skip individual
//! forward passes, improving throughput when the draft model's predictions
//! match the main model's.
//!
//! Algorithm:
//! 1. Prefill prompt through both models
//! 2. Sample first token from main model
//! 3. Loop:
//!    a. Draft: generate `num_draft` tokens with draft model
//!    b. Verify: forward [current + draft tokens] through main model
//!    c. Accept matching prefix, rewind caches for rejected tokens
//!    d. Continue from the divergence point
//!
//! Step 3c is the acceptance rule. At `temperature == 0` it is the argmax
//! comparison described above. At `temperature > 0` it is modified rejection
//! sampling, which preserves the target model's distribution exactly; see
//! [`stochastic_accept`].
//!
//! ## Cache invariant
//!
//! Both models must condition on the same prefix, or the drafter proposes
//! against a context the target has already moved past and acceptance decays.
//! At every round boundary the main and draft KV caches therefore hold the
//! prompt plus every emitted token except the pending `current_token`, which
//! the next round forwards itself. The two caches rewind by different amounts
//! to get there: the verify pass forwards one token more than the draft loop
//! does, so on a round that rejects, the draft trim is one *less* than the main
//! trim. The one entry that cannot be reached by trimming at all is the last
//! proposal on a fully-accepted round, which the draft loop never forwarded;
//! it is carried in [`SpeculativeGenerator::pending_draft_context`] and
//! replayed at the head of the next round's first draft forward.
//!
//! ## Sibling modules
//!
//! - [`mtp`] — Multi-Token Prediction (MTP) round-loop generator for the
//!   Gemma 4 assistant drafter family. Peer code path to
//!   [`SpeculativeGenerator`] with fundamentally different semantics
//!   (drafter has no own KV cache; verify is a single forward over the
//!   whole draft block; rollback uses per-row tail-zero rather than
//!   `trim_caches`). See [`mtp::MtpGenerator`].
//! - [`stochastic_accept`] — the distribution-preserving acceptance rule and
//!   its residual resample, shared by every speculative verify path.

pub mod mtp;
pub mod stochastic_accept;

use crate::cache::can_trim_prompt_cache;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::ffi::MlxThreadLocalStream;
use crate::generate::{GenerationStats, LanguageModel, SamplingConfig};
use crate::generation_policy::{initial_token_history, merged_eos_token_ids};
use crate::hardware;
use crate::layers::KVCache;
use crate::sampling::{
    TokenBiasMap, effective_token_distribution, sample_token_optimized,
    sample_token_with_distribution,
};
use crate::streams::{install_thread_local_default_stream, new_thread_local_generation_stream};
use crate::utils::{align_to_na_tile, create_padded_prefill_mask};
use cxx::UniquePtr;
use std::borrow::Cow;
use std::time::Instant;
use stochastic_accept::{AcceptanceRule, DraftVerdict};

/// Default chunk size for chunked prefill in speculative decoding.
///
/// Mirrors the `prefill_step_size` default used by upstream mlx-lm
/// `speculative_generate_step` (512 tokens). Processing the prompt in
/// chunks reduces peak memory pressure for long prompts and ensures the
/// loop can correctly reserve the last token for the first speculation step.
const PREFILL_STEP_SIZE: usize = 512;

/// Returns true when the current hardware is M5+ with a Neural Accelerator
/// and tile-aligned verification batching should be applied.
#[inline]
fn should_align_verification() -> bool {
    let hw = hardware::get_hardware();
    hw.has_neural_accelerator && hw.macos_supports_na
}

/// Acceptance accounting for one [`SpeculativeGenerator::generate`] call.
///
/// The two ratios are the quantities the acceptance change is supposed to
/// move: [`Self::acceptance_rate`] should rise from about `sum_x p(x) q(x)`
/// under the old rule to about `sum_x min(p(x), q(x))` under modified
/// rejection sampling, and [`Self::mean_accepted_len`] is the per-round
/// version that drives end-to-end throughput.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpeculativeAcceptanceStats {
    /// Verify rounds executed (one batched target forward each).
    pub rounds: usize,
    /// Draft tokens proposed across all rounds.
    pub proposed_draft_tokens: usize,
    /// Draft tokens the verify rule accepted.
    pub accepted_draft_tokens: usize,
    /// The acceptance rule the run actually used.
    pub rule: Option<AcceptanceRule>,
    /// Verify positions the accept test actually ran on. Differs from
    /// `proposed_draft_tokens` because a chain stops at its first rejection, so
    /// positions after it are proposed but never tested.
    pub positions_tested: usize,
    /// Running sum of `sum_x min(p(x), q(x))` over tested positions, the
    /// closed-form probability modified rejection sampling accepts. Zero unless
    /// [`stochastic_accept::accept_diagnostics_enabled`].
    pub closed_form_sum_min: f64,
    /// Running sum of `sum_x p(x) q(x)` over tested positions, the closed-form
    /// probability the pre-#902 rule accepts. Zero unless diagnostics are on.
    pub closed_form_sum_prod: f64,
}

impl SpeculativeAcceptanceStats {
    /// Accepted draft tokens per *proposed* draft token.
    ///
    /// Not the per-position acceptance probability, and not comparable to
    /// `sum_x min(p(x), q(x))`: a chain stops at its first rejection, so the
    /// block positions behind it are counted as proposed but never tested. Use
    /// [`Self::per_position_acceptance`] to compare against theory or across an
    /// A/B; this one is the "how much of each drafted block survived" figure,
    /// which also moves with the block length.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed_draft_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.proposed_draft_tokens as f64
        }
    }

    /// Accepted draft tokens per verify position the accept test actually ran
    /// on.
    ///
    /// This is the quantity the theory predicts: modified rejection sampling
    /// converges to `sum_x min(p(x), q(x))` and the pre-#902 rule to
    /// `sum_x p(x) q(x)`, both per position. Counted identically on both
    /// acceptance rules.
    pub fn per_position_acceptance(&self) -> f64 {
        if self.positions_tested == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.positions_tested as f64
        }
    }

    /// Mean accepted draft tokens per verify round. This is the figure the
    /// issue's performance gate names ("mean accepted draft length").
    pub fn mean_accepted_len(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.rounds as f64
        }
    }

    /// One-line summary for the CLI, printed on **stdout** after a speculative
    /// run.
    ///
    /// This exists because the `mlxcel` CLI installs no `tracing` subscriber,
    /// so the info-level instrumentation in this module is unreachable on the
    /// only binary that runs [`SpeculativeGenerator`]. An acceptance-rate A/B
    /// that cannot name the rule it measured is not a measurement, so the two
    /// facts a reviewer needs (which rule ran, and the mean accepted draft
    /// length) are printed unconditionally rather than behind a log level.
    ///
    /// The `rule=` token is [`AcceptanceRule::id`], which is stable and
    /// greppable. `rule=unknown` means [`SpeculativeGenerator::generate`] never
    /// reached its decode loop.
    ///
    /// Used by: `commands::generate` (offline `mlxcel generate --draft-model`)
    pub fn summary_line(&self) -> String {
        let rule = self.rule.map_or("unknown", |r| r.id());
        let label = self.rule.map_or("no speculative round ran", |r| r.label());
        let mut line = format!(
            "[Speculative acceptance] rule={rule} rounds={} proposed={} positions_tested={} \
             accepted={} per_position_acceptance={:.4} acceptance_rate={:.4} \
             mean_accepted_len={:.4} ({label})",
            self.rounds,
            self.proposed_draft_tokens,
            self.positions_tested,
            self.accepted_draft_tokens,
            self.per_position_acceptance(),
            self.acceptance_rate(),
            self.mean_accepted_len(),
        );
        if self.positions_tested > 0 && self.closed_form_sum_min > 0.0 {
            let n = self.positions_tested as f64;
            line.push_str(&format!(
                "\n[Speculative acceptance diagnostic] closed_form_sum_min={:.4} \
                 closed_form_sum_prod={:.4} (measured per-position acceptance must sit at \
                 sum_min, and sum_min >= sum_prod always)",
                self.closed_form_sum_min / n,
                self.closed_form_sum_prod / n,
            ));
        }
        line
    }
}

/// Speculative decoding generator
///
/// Uses a draft model to propose candidate tokens and a main model to verify them.
/// When the draft model's predictions match, multiple tokens are accepted per
/// main model forward pass, improving throughput.
pub struct SpeculativeGenerator {
    main_caches: Vec<KVCache>,
    draft_caches: Vec<KVCache>,
    generated_tokens: Vec<i32>,
    /// Thread-local generation stream — see `mlxcel_core::streams`
    /// Resolves to a per-thread `MlxStream` on the
    /// worker thread that calls `generate`, so dispatch and
    /// synchronization stay paired even when the generator is moved
    /// across threads after construction.
    generation_stream: Option<UniquePtr<MlxThreadLocalStream>>,
    /// Acceptance accounting for the most recent [`Self::generate`] call.
    ///
    /// Reset at the top of every call by [`Self::reset`]. Read through
    /// [`Self::acceptance_stats`] and summarized once per call at info level,
    /// which is what makes an acceptance-rate A/B measurable from a log rather
    /// than only from a stopwatch.
    acceptance: SpeculativeAcceptanceStats,
    /// Per-generator override of the `MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT`
    /// default. `None` follows the env; `Some(v)` forces it.
    stochastic_acceptance: Option<bool>,
    /// Cached per-generator `TokenBiasMap` resolved from a `LangBiasConfig`.
    ///
    /// **Axis B invariant**: the bias is applied **only** to the target
    /// (main) model's sampler. The draft model must keep seeing the
    /// unmodified policy so its candidate distribution stays aligned with
    /// its own weights; otherwise the accept/reject comparison becomes
    /// biased on two different policies and speculative acceptance rate
    /// collapses. See [`Self::compose_target_sampling`] and
    /// [`Self::draft_sampling`] — only the former injects the cached bias.
    token_bias: TokenBiasMap,
    /// The one emitted token the draft model's KV cache does not hold yet.
    ///
    /// A round's draft loop forwards `current_token, d_0, ..., d_{k-2}` for
    /// `k` proposals: the last proposal `d_{k-1}` is *sampled from* the
    /// forward of `d_{k-2}` and is never itself forwarded. When verification
    /// accepts and emits it, it joins the sequence while the draft cache has
    /// no entry for it, and no amount of trimming can put it there. It is
    /// parked here and replayed at the head of the next round's first draft
    /// forward, which costs no extra forward pass because that forward simply
    /// becomes two tokens wide.
    ///
    /// `None` at every other round boundary. Reset by [`Self::reset`].
    ///
    /// Mirrors the `draft_y` concatenation in upstream mlx-lm
    /// `speculative_generate_step`
    /// (<https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py>).
    pending_draft_context: Option<i32>,
}

impl SpeculativeGenerator {
    /// Create a new speculative generator
    pub fn new(main_num_layers: usize, draft_num_layers: usize) -> Self {
        Self {
            main_caches: (0..main_num_layers).map(|_| KVCache::new()).collect(),
            draft_caches: (0..draft_num_layers).map(|_| KVCache::new()).collect(),
            generated_tokens: Vec::new(),
            generation_stream: new_thread_local_generation_stream(),
            acceptance: SpeculativeAcceptanceStats::default(),
            stochastic_acceptance: None,
            token_bias: TokenBiasMap::default(),
            pending_draft_context: None,
        }
    }

    /// Acceptance accounting for the most recent [`Self::generate`] call.
    pub fn acceptance_stats(&self) -> SpeculativeAcceptanceStats {
        self.acceptance
    }

    /// Force the acceptance-optimal (modified rejection sampling) rule on or
    /// off for this generator, overriding the
    /// `MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT` process default.
    ///
    /// The feature is opt-in because the rule it replaces was already
    /// distribution-preserving on this path (see
    /// [`stochastic_accept`]), so it trades measurable per-position cost for an
    /// acceptance gain that is near zero against a confident drafter. This
    /// setter is how a caller that has established the gain is worth taking,
    /// or a test that must exercise the rule regardless of the process
    /// default, turns it on.
    pub fn with_stochastic_acceptance(mut self, enabled: bool) -> Self {
        self.stochastic_acceptance = Some(enabled);
        self
    }

    /// Attach a pre-resolved `TokenBiasMap` to this speculative generator.
    ///
    /// The bias is cached for the generator's lifetime and applied **only** to
    /// the target model's sampling during verification (and the first-token
    /// prefill). The draft model's sampling is left untouched to preserve
    /// speculative acceptance behavior.
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// Returns a reference to the cached target-only token-bias map.
    ///
    /// Used by tests to assert that the bias was wired in correctly and that
    /// the draft model never observes it.
    pub fn token_bias(&self) -> &TokenBiasMap {
        &self.token_bias
    }

    /// Compose the effective **target-model** sampling config from the cached
    /// `token_bias` and the caller's [`SamplingConfig`].
    ///
    /// Empty cached bias => borrowed unchanged (bit-exact baseline). Non-empty
    /// bias but caller already set `sampling.token_bias` => caller wins.
    /// Otherwise the caller's config is cloned and the cached bias is injected.
    fn compose_target_sampling<'a>(&self, sampling: &'a SamplingConfig) -> Cow<'a, SamplingConfig> {
        if self.token_bias.is_empty() || !sampling.token_bias.is_empty() {
            Cow::Borrowed(sampling)
        } else {
            let mut cloned = sampling.clone();
            cloned.token_bias = self.token_bias.clone();
            Cow::Owned(cloned)
        }
    }

    /// Returns the sampling config used by the **draft** model.
    ///
    /// **Axis B**: by design this ignores the generator's cached
    /// `token_bias`. Biasing the draft sampler would skew candidate
    /// distribution away from the draft model's trained distribution and
    /// collapse speculative acceptance rates (the target's accept/reject
    /// comparison already reflects the bias on the verification side).
    #[inline]
    fn draft_sampling<'a>(&self, sampling: &'a SamplingConfig) -> &'a SamplingConfig {
        sampling
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        for cache in &mut self.main_caches {
            *cache = KVCache::new();
        }
        for cache in &mut self.draft_caches {
            *cache = KVCache::new();
        }
        self.generated_tokens.clear();
        self.acceptance = SpeculativeAcceptanceStats::default();
        self.pending_draft_context = None;
    }

    /// Get the generated tokens
    pub fn tokens(&self) -> &[i32] {
        &self.generated_tokens
    }

    /// Generate tokens using speculative decoding
    ///
    /// # Arguments
    /// * `main_model` - The main (target) model for verification
    /// * `draft_model` - The smaller draft model for candidate generation
    /// * `prompt_tokens` - Input prompt token IDs
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `num_draft` - Number of draft tokens to generate per speculation step
    /// * `sampling` - Sampling configuration
    ///
    /// # Panics
    ///
    /// Panics if any KV cache in the main model's caches is not trimmable,
    /// since speculative decoding requires cache rewind on draft rejection.
    /// All current `KVCache` variants are trimmable; this guard future-proofs
    /// the code against non-trimmable cache types (mirrors upstream mlx-lm
    /// `can_trim_prompt_cache` validation added in PR #1109 / commit `f56d997`).
    pub fn generate<M: LanguageModel, D: LanguageModel>(
        &mut self,
        main_model: &M,
        draft_model: &D,
        prompt_tokens: &[i32],
        max_tokens: usize,
        num_draft: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        self.reset();

        // Validate that all KV cache entries support trimming before we begin.
        // Speculative decoding rewrites the cache on every rejected draft token,
        // so a non-trimmable cache type would silently corrupt the state.
        // Mirrors upstream mlx-lm `can_trim_prompt_cache` check added in
        // PR #1109 / commit f56d997. All current KVCache mode variants are
        // trimmable; this assertion fires only when a new non-trimmable type
        // is introduced (fail fast rather than silent corruption).
        assert!(
            can_trim_prompt_cache(&self.main_caches),
            "speculative decoding requires a trimmable prompt cache (main model). \
             At least one KV cache entry does not support trimming. \
             Use a standard (non-speculative) generation path or switch to a \
             trimmable cache type."
        );
        assert!(
            can_trim_prompt_cache(&self.draft_caches),
            "speculative decoding requires a trimmable prompt cache (draft model). \
             At least one KV cache entry does not support trimming."
        );

        // Guard against empty prompts: speculative decoding requires at least
        // one token so the final forward pass (which produces the first
        // generated token's logits) can run. An empty prompt would cause a
        // `[1, 0]` tensor forward with undefined behaviour.
        assert!(
            !prompt_tokens.is_empty(),
            "speculative generate requires at least one prompt token"
        );

        // Axis B: compose target-only sampling once; draft sampling stays raw.
        // `target_cow` owns the merged config when a bias is active, otherwise
        // it borrows the caller's. `draft_sampling` always returns `sampling`
        // unchanged — biasing the draft would collapse acceptance rate.
        let target_cow = self.compose_target_sampling(sampling);
        let target_sampling: &SamplingConfig = target_cow.as_ref();
        let draft_sampling: &SamplingConfig = self.draft_sampling(sampling);

        // Set generation stream
        install_thread_local_default_stream(self.generation_stream.as_ref());

        // History + EOS handling inherit the caller's policy; history-based
        // penalties apply to both models so we read flags from the caller's
        // raw config (same shape as `target_sampling` except for `token_bias`).
        let eos_tokens = merged_eos_token_ids(main_model.eos_token_ids(), &sampling.stop_token_ids);
        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);

        // PREFILL PHASE.
        //
        // Process the prompt in chunks of `PREFILL_STEP_SIZE`, always reserving
        // the last token for the first speculation step. This mirrors the
        // upstream mlx-lm fix in PR #1109 / commit f56d997:
        //
        //   while y.size > 1:
        //       n_to_process = min(prefill_step_size, y.size - 1)
        //
        // The old single-shot prefill (`forward` over all tokens at once) worked
        // for short prompts but could process every token, leaving none for the
        // speculation bootstrap — causing output corruption when prompt length
        // was an exact multiple of the step size.
        let prefill_start = Instant::now();

        // Chunked prefill: process all but the last token in step-sized blocks,
        // evaluating and clearing the memory cache between steps to bound peak
        // memory usage for long prompts.
        let n = prompt_tokens.len();
        let mut consumed = 0usize;
        while n - consumed > 1 {
            let step = PREFILL_STEP_SIZE.min(n - consumed - 1);
            let chunk = &prompt_tokens[consumed..consumed + step];
            let chunk_input = ffi::from_slice_i32(chunk, &[1, step as i32]);
            // Prefill both models with the chunk (logits discarded — we only
            // need the KV cache state for the continuation).
            let _main_chunk_logits = main_model.forward(&chunk_input, &mut self.main_caches, None);
            let _draft_chunk_logits =
                draft_model.forward(&chunk_input, &mut self.draft_caches, None);
            // Evaluate only the KV cache state so it is materialised before
            // the next chunk. This mirrors the upstream mlx-lm pattern:
            //   mx.eval([c.state for c in cache])
            // Evaluating the full logit tensors (_main_chunk_logits /
            // _draft_chunk_logits) would force the LM-head matmul and a
            // peak allocation of ~3.5 GB per chunk for Llama-3 class models,
            // defeating the memory-bounding goal of chunked prefill.
            for cache in &self.main_caches {
                cache.eval_state();
            }
            for cache in &self.draft_caches {
                cache.eval_state();
            }
            consumed += step;
            ffi::clear_memory_cache();
        }

        // Forward the final token through both models to get the logits used
        // for sampling the first generated token. By construction `consumed < n`
        // so there is always at least one remaining token here.
        let last_chunk = &prompt_tokens[consumed..];
        let last_input = ffi::from_slice_i32(last_chunk, &[1, last_chunk.len() as i32]);
        let main_logits = main_model.forward(&last_input, &mut self.main_caches, None);
        let _draft_logits = draft_model.forward(&last_input, &mut self.draft_caches, None);

        // Sample first token from main model (target: bias applied).
        let (first_token_arr, _) =
            sample_token_optimized(&main_logits, target_sampling, &token_history);
        ffi::eval(&first_token_arr);
        let first_token = ffi::item_i32(&first_token_arr);
        let prefill_time = prefill_start.elapsed();

        if eos_tokens.contains(&first_token) || max_tokens == 0 {
            let stats = Self::build_stats(
                prompt_tokens.len(),
                self.generated_tokens.len(),
                prefill_time,
                std::time::Duration::ZERO,
            );
            return (self.generated_tokens.clone(), stats);
        }

        self.generated_tokens.push(first_token);
        if needs_history {
            token_history.push(first_token);
        }

        if max_tokens <= 1 {
            let stats = Self::build_stats(
                prompt_tokens.len(),
                self.generated_tokens.len(),
                prefill_time,
                std::time::Duration::ZERO,
            );
            return (self.generated_tokens.clone(), stats);
        }

        // DECODE PHASE.
        //
        // Acceptance policy for this call (issue #902). This generator owns
        // the draft sampler, so it can always report the distribution each
        // proposal was drawn from; `has_proposal_distribution` is
        // unconditionally true here. The remaining gates are the target
        // sampler being stochastic and the env kill switch. The chosen rule is
        // logged once per process per kind at info level, so a server log
        // proves which rule a run used.
        let rule = stochastic_accept::acceptance_rule_with_override(
            target_sampling,
            true,
            self.stochastic_acceptance,
        );
        stochastic_accept::note_rule(rule);
        let stochastic = rule == AcceptanceRule::Stochastic;
        self.acceptance.rule = Some(rule);

        // The closed-form diagnostic needs `q` on *both* arms, otherwise the
        // two halves of an A/B report different quantities and cannot be
        // compared. Off by default, so the argmax arm captures nothing and
        // stays exactly as it was.
        let diagnostics = stochastic_accept::accept_diagnostics_enabled();
        let capture_proposal_probs = stochastic || diagnostics;

        let decode_start = Instant::now();
        let mut current_token = first_token;
        let mut done = false;

        while self.generated_tokens.len() < max_tokens && !done {
            // Step 1: Generate draft tokens
            let mut draft_tokens = Vec::with_capacity(num_draft);
            // `q` per drafted position, in proposal order. Populated only on
            // the stochastic path; the argmax path allocates nothing and runs
            // exactly the pre-#902 code.
            let mut draft_probs: Vec<UniquePtr<MlxArray>> = Vec::new();
            let mut draft_token = current_token;

            for _ in 0..num_draft {
                // A previous round that accepted its last proposal left the
                // draft model one token behind the sequence (see
                // `pending_draft_context`). Replay that token at the head of
                // this round's first draft forward rather than spending a
                // separate pass on it: the sampler's `slice_last_logits`
                // pre-step already reads the final position, so the proposal
                // sampled here is the one conditioned on the full replayed
                // prefix, and the round still costs `num_draft` forwards. The
                // draft model sees a 2-token block with no explicit mask,
                // exactly as it does during chunked prefill and as the target
                // does during verification, so the same causal-prefill
                // contract covers it. `take` empties the slot, so only the
                // first iteration of the round can replay.
                let draft_input = match self.pending_draft_context.take() {
                    Some(owed) => ffi::from_slice_i32(&[owed, draft_token], &[1, 2]),
                    None => ffi::from_slice_i32(&[draft_token], &[1, 1]),
                };
                let draft_logits = draft_model.forward(&draft_input, &mut self.draft_caches, None);
                // Axis B: draft sampler MUST NOT see the bias. See
                // `draft_sampling` for the rationale.
                let tok_arr = if capture_proposal_probs {
                    // The token and `q` come out of one pre-step, which is what
                    // guarantees `q` is the distribution this very token was
                    // drawn from rather than a later reconstruction. `q` is
                    // left unevaluated: a chain that gets rejected at an
                    // earlier position never materializes the tail.
                    let (tok_arr, probs) = sample_token_with_distribution(
                        &draft_logits,
                        draft_sampling,
                        &token_history,
                    );
                    draft_probs.push(probs);
                    tok_arr
                } else {
                    sample_token_optimized(&draft_logits, draft_sampling, &token_history).0
                };
                ffi::eval(&tok_arr);
                draft_token = ffi::item_i32(&tok_arr);
                draft_tokens.push(draft_token);

                if eos_tokens.contains(&draft_token) {
                    break;
                }
            }

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Verify draft tokens with main model in a single batched forward pass.
            // Input: [current_token, draft_token_0, ..., draft_token_n-1] shape [1, N+1]
            // Output: logits shape [1, N+1, vocab_size]
            //
            // This is structurally identical to a prefill pass, converting N memory-bound
            // GEMV decode operations into one compute-bound GEMM. On M5+ Neural Accelerator
            // hardware, this yields 3-4x speedup via tile-aligned GEMM dispatch.
            let mut verify_tokens = vec![current_token];
            verify_tokens.extend_from_slice(&draft_tokens);
            let actual_verify_len = verify_tokens.len();

            // On M5+ hardware with Neural Accelerator, pad the verification sequence
            // to a 32-token tile boundary for optimal GEMM throughput. On other
            // hardware, no padding is needed (batching is still beneficial but
            // tile alignment does not apply).
            let main_logits = if should_align_verification() && main_model.supports_padded_prefill()
            {
                let padded_len = align_to_na_tile(actual_verify_len);
                // Capture the current KV cache offset before the verification pass
                // so the attention mask correctly spans [offset, offset + padded_len).
                let kv_offset = self.main_caches.first().map(|c| c.offset).unwrap_or(0);

                if padded_len > actual_verify_len {
                    // Pad with zeros up to the tile boundary
                    let mut padded_tokens = verify_tokens.clone();
                    padded_tokens.resize(padded_len, 0);
                    let verify_input = ffi::from_slice_i32(&padded_tokens, &[1, padded_len as i32]);
                    // Create attention mask so padding positions cannot attend to
                    // anything and real tokens cannot attend to padding keys.
                    let mask = create_padded_prefill_mask(
                        actual_verify_len as i32,
                        padded_len as i32,
                        kv_offset,
                    );
                    let raw_logits = main_model.forward(
                        &verify_input,
                        &mut self.main_caches,
                        Some(mask.as_ref().unwrap()),
                    );
                    // Trim padding positions from KV caches so subsequent decode
                    // steps see the correct cache offset (actual_verify_len tokens,
                    // not padded_len tokens).
                    let excess = (padded_len - actual_verify_len) as i32;
                    for cache in self.main_caches.iter_mut() {
                        cache.trim(excess);
                    }
                    main_model.trim_internal_caches(excess);
                    // Return only the logits for the actual (non-padded) positions,
                    // sliced to shape [1, actual_verify_len, vocab].
                    let vocab = ffi::array_shape(&raw_logits)[2];
                    ffi::slice(
                        &raw_logits,
                        &[0, 0, 0],
                        &[1, actual_verify_len as i32, vocab],
                    )
                } else {
                    // Sequence already aligns to a tile boundary; no padding needed.
                    let verify_input =
                        ffi::from_slice_i32(&verify_tokens, &[1, actual_verify_len as i32]);
                    main_model.forward(&verify_input, &mut self.main_caches, None)
                }
            } else {
                // Non-NA hardware: plain batched forward pass, no tile alignment.
                let verify_input =
                    ffi::from_slice_i32(&verify_tokens, &[1, actual_verify_len as i32]);
                main_model.forward(&verify_input, &mut self.main_caches, None)
            };

            // The main model returns logits for each position:
            // - Position 0 (current_token): logits that would produce draft_tokens[0]
            // - Position i: logits that would produce draft_tokens[i]
            // - Last position: logits for the token after all draft tokens

            // Step 3: Compare draft tokens with main model's choices
            let main_shape = ffi::array_shape(&main_logits);
            let seq_len = main_shape[1]; // Number of logit positions (actual, not padded)
            let mut accepted = 0;

            for (i, draft_token) in draft_tokens.iter().copied().enumerate() {
                if (i as i32) >= seq_len {
                    break;
                }

                // Get main model's logits at position i
                let pos_logits = ffi::slice(
                    &main_logits,
                    &[0, i as i32, 0],
                    &[1, (i as i32) + 1, main_shape[2]],
                );
                // Reshape to [1, 1, vocab] for sample_token_optimized
                let pos_logits = ffi::reshape(&pos_logits, &[1, 1, main_shape[2]]);

                // Axis B: target verification uses the bias-augmented sampling
                // on both arms.
                //
                // `accept_token` is the token kept when the draft is accepted;
                // `replacement` is the token emitted when it is not. Both arms
                // produce the same pair so the bookkeeping below is shared.
                // Counted on both arms: this is the denominator the per-position
                // acceptance probability is measured against, and the two arms
                // must report the same quantity for an A/B to mean anything.
                // `proposed_draft_tokens` is *not* that denominator, because a
                // chain stops at its first rejection and never tests the block
                // positions behind it.
                self.acceptance.positions_tested += 1;
                if diagnostics {
                    let target_probs =
                        effective_token_distribution(&pos_logits, target_sampling, &token_history);
                    let (sum_min, sum_prod) =
                        stochastic_accept::closed_form_acceptance(&target_probs, &draft_probs[i]);
                    self.acceptance.closed_form_sum_min += sum_min;
                    self.acceptance.closed_form_sum_prod += sum_prod;
                }

                let (accept_draft, replacement) = if stochastic {
                    // `p` at this position, conditioned on the tokens accepted
                    // so far in this round (`token_history` grows as we
                    // accept), which is the correct target conditional.
                    let target_probs =
                        effective_token_distribution(&pos_logits, target_sampling, &token_history);
                    match stochastic_accept::verify_draft_token(
                        &target_probs,
                        &draft_probs[i],
                        draft_token,
                    ) {
                        DraftVerdict::Accept => (true, draft_token),
                        DraftVerdict::Reject { replacement } => (false, replacement),
                    }
                } else {
                    let (main_tok_arr, _) =
                        sample_token_optimized(&pos_logits, target_sampling, &token_history);
                    ffi::eval(&main_tok_arr);
                    let main_token = ffi::item_i32(&main_tok_arr);
                    (main_token == draft_token, main_token)
                };

                if accept_draft {
                    // Accept draft token
                    accepted += 1;

                    if eos_tokens.contains(&draft_token) {
                        done = true;
                        break;
                    }

                    self.generated_tokens.push(draft_token);
                    if needs_history {
                        token_history.push(draft_token);
                    }

                    // `d_{k-1}` is the only proposal the draft loop never
                    // forwards, so emitting it leaves the draft cache one entry
                    // short of the sequence. Park it for the next round's first
                    // draft forward. Recorded here rather than beside the bonus
                    // token below so it is also right on the round that stops on
                    // `max_tokens`, and so an accepted-EOS proposal (which
                    // breaks above, before this push) never records a debt for a
                    // token that was never emitted.
                    if i + 1 == draft_tokens.len() {
                        self.pending_draft_context = Some(draft_token);
                    }

                    if self.generated_tokens.len() >= max_tokens {
                        done = true;
                        break;
                    }
                } else {
                    // Reject: emit the target-side replacement instead. On the
                    // argmax rule that is the target sampler's own draw; on
                    // the stochastic rule it is a draw from the normalized
                    // residual `relu(p - q)`, which is what makes the emitted
                    // stream distributed as `p`.
                    if eos_tokens.contains(&replacement) {
                        done = true;
                    } else {
                        self.generated_tokens.push(replacement);
                        if needs_history {
                            token_history.push(replacement);
                        }
                    }
                    break;
                }
            }

            // If all draft tokens were accepted and we're not done,
            // sample one more token from the main model's last logit position
            if accepted == draft_tokens.len() && !done && self.generated_tokens.len() < max_tokens {
                let last_pos = seq_len - 1;
                let last_logits = ffi::slice(
                    &main_logits,
                    &[0, last_pos, 0],
                    &[1, last_pos + 1, main_shape[2]],
                );
                let last_logits = ffi::reshape(&last_logits, &[1, 1, main_shape[2]]);
                // Axis B: bonus token comes from the main model → target bias.
                let (bonus_tok_arr, _) =
                    sample_token_optimized(&last_logits, target_sampling, &token_history);
                ffi::eval(&bonus_tok_arr);
                let bonus_token = ffi::item_i32(&bonus_tok_arr);

                if eos_tokens.contains(&bonus_token) {
                    done = true;
                } else {
                    self.generated_tokens.push(bonus_token);
                    if needs_history {
                        token_history.push(bonus_token);
                    }
                }

                current_token = bonus_token;
            } else if !done {
                current_token = *self.generated_tokens.last().unwrap();
            }

            self.acceptance.rounds += 1;
            self.acceptance.proposed_draft_tokens += draft_tokens.len();
            self.acceptance.accepted_draft_tokens += accepted;

            // Step 4: rewind whatever this round forwarded past the tokens it
            // actually kept.
            //
            // Write `k` for `draft_tokens.len()` and `a` for `accepted`. Both
            // caches must end the round holding the prompt plus every emitted
            // token except the next round's `current_token`, which the next
            // round forwards itself.
            let rejected = draft_tokens.len() - accepted;
            if rejected > 0 {
                // Main: the verify forward appended `current_token, d_0..d_{k-1}`.
                // The round keeps `d_0..d_{a-1}` and emits one replacement that
                // becomes the next `current_token`, so the cache must end at
                // `d_{a-1}`. Trim the rejected tail, `k - a == rejected`.
                trim_caches(&mut self.main_caches, rejected as i32);

                // Draft: after its loop the draft cache holds the prompt, every
                // emitted token, and `d_0..d_{k-2}`: the loop feeds
                // `current_token` and then every proposal except the last, plus
                // any replayed `pending_draft_context`. It must also end at
                // `d_{a-1}`, so the trim is `(k - 1) - a == rejected - 1`, one
                // less than the main cache's and never negative here because
                // `rejected >= 1`. Trimming `rejected + 1` instead, as this did
                // before, left the drafter conditioning on a prefix that fell
                // one to two tokens further behind on every single round.
                trim_caches(&mut self.draft_caches, (rejected - 1) as i32);
            }

            // Periodic cache clearing, backend-aware cadence (#627): disabled by
            // default on CUDA (clear churns the pool, defeats CUDA-graph reuse,
            // mlx#2358), 256 on Metal/CPU, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
            if crate::memory::should_clear_cache_at(
                self.generated_tokens.len(),
                crate::memory::cache_clear_interval(),
            ) {
                ffi::clear_memory_cache();
            }
        }

        let decode_time = decode_start.elapsed();

        // One summary line per call at info level. Together with the one-shot
        // rule line from `note_rule`, this is what lets an acceptance-rate A/B
        // be read off a log instead of inferred from wall-clock alone, and it
        // names the rule so a run can never be attributed to the wrong arm.
        tracing::info!(
            rule = rule.id(),
            rounds = self.acceptance.rounds,
            proposed_draft_tokens = self.acceptance.proposed_draft_tokens,
            positions_tested = self.acceptance.positions_tested,
            accepted_draft_tokens = self.acceptance.accepted_draft_tokens,
            per_position_acceptance = self.acceptance.per_position_acceptance(),
            acceptance_rate = self.acceptance.acceptance_rate(),
            mean_accepted_len = self.acceptance.mean_accepted_len(),
            generated_tokens = self.generated_tokens.len(),
            "speculative decode finished"
        );

        let stats = Self::build_stats(
            prompt_tokens.len(),
            self.generated_tokens.len(),
            prefill_time,
            decode_time,
        );

        (self.generated_tokens.clone(), stats)
    }

    fn build_stats(
        prompt_count: usize,
        gen_count: usize,
        prefill_time: std::time::Duration,
        decode_time: std::time::Duration,
    ) -> GenerationStats {
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;

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

/// Trim the last `n` entries from all caches in the slice
/// Returns the number of entries actually trimmed (from the first cache)
fn trim_caches(caches: &mut [KVCache], n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut trimmed = 0;
    for cache in caches.iter_mut() {
        trimmed = cache.trim(n);
    }
    trimmed
}

#[cfg(test)]
#[path = "distribution_tests.rs"]
mod distribution_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    #[test]
    fn test_kv_cache_trim_basic() {
        let mut cache = KVCache::new();

        // Add some data: [batch=1, heads=2, seq_len=5, head_dim=4]
        let keys = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
        cache.update(keys, values);
        assert_eq!(cache.offset, 5);

        // Trim 2
        let trimmed = cache.trim(2);
        assert_eq!(trimmed, 2);
        assert_eq!(cache.offset, 3);

        // Verify shapes
        let k_shape = ffi::array_shape(cache.keys.as_ref().unwrap());
        assert_eq!(k_shape, vec![1, 2, 3, 4]);
        let v_shape = ffi::array_shape(cache.values.as_ref().unwrap());
        assert_eq!(v_shape, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_kv_cache_trim_all() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        // Trim all
        let trimmed = cache.trim(3);
        assert_eq!(trimmed, 3);
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_none());
        assert!(cache.values.is_none());
    }

    #[test]
    fn test_kv_cache_trim_zero() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        let trimmed = cache.trim(0);
        assert_eq!(trimmed, 0);
        assert_eq!(cache.offset, 3);
    }

    #[test]
    fn test_kv_cache_trim_more_than_available() {
        let mut cache = KVCache::new();
        let keys = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        let values = ffi::ones(&[1, 2, 3, 4], dtype::FLOAT32);
        cache.update(keys, values);

        // Trim more than available - should trim only what's available
        let trimmed = cache.trim(10);
        assert_eq!(trimmed, 3);
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_none());
    }

    #[test]
    fn test_trim_caches_helper() {
        let mut caches = vec![KVCache::new(), KVCache::new()];
        for cache in caches.iter_mut() {
            let keys = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
            let values = ffi::ones(&[1, 2, 5, 4], dtype::FLOAT32);
            cache.update(keys, values);
        }

        let trimmed = trim_caches(&mut caches, 2);
        assert_eq!(trimmed, 2);
        for cache in &caches {
            assert_eq!(cache.offset, 3);
        }
    }

    // ------------------------------------------------------------------
    // B8 — token-bias wiring (target-only)
    // ------------------------------------------------------------------

    fn make_bias(entries: &[(i32, f32)]) -> TokenBiasMap {
        let mut m = TokenBiasMap::new();
        for &(id, b) in entries {
            m.insert(id, b);
        }
        m
    }

    /// Default construction yields an empty token-bias cache.
    #[test]
    fn speculative_generator_default_bias_is_empty() {
        let g = SpeculativeGenerator::new(4, 2);
        assert!(g.token_bias().is_empty());
    }

    /// `with_token_bias` caches the supplied map and exposes it via the
    /// inspector — the target path sees this map, the draft path never does.
    #[test]
    fn speculative_generator_passes_bias_to_target_only() {
        let bias = make_bias(&[(7, f32::NEG_INFINITY), (11, 2.0)]);
        let g = SpeculativeGenerator::new(4, 2).with_token_bias(bias.clone());

        // Target-side composition must inject the cached bias into a caller
        // config that lacks one.
        let caller = SamplingConfig::default();
        let target = g.compose_target_sampling(&caller);
        assert_eq!(
            target.token_bias.len(),
            2,
            "target sampler must carry the cached bias"
        );
        assert!(
            target.token_bias.contains(7),
            "target bias must contain id=7"
        );

        // Draft-side composition MUST remain unbiased regardless of the cached
        // map — this is the core speculative-acceptance invariant.
        let draft = g.draft_sampling(&caller);
        assert!(
            draft.token_bias.is_empty(),
            "draft sampler must NEVER carry the cached bias (got {} entries): \
             speculative acceptance is computed by comparing draft candidates \
             against target sampling, and biasing the draft collapses the \
             accept ratio",
            draft.token_bias.len()
        );
    }

    /// Caller-supplied bias wins over the generator-cached bias (explicit
    /// per-call override).
    #[test]
    fn speculative_generator_caller_bias_wins() {
        let cached = make_bias(&[(1, 1.0)]);
        let caller_bias = make_bias(&[(42, f32::NEG_INFINITY)]);
        let g = SpeculativeGenerator::new(2, 1).with_token_bias(cached);

        let caller = SamplingConfig {
            token_bias: caller_bias,
            ..SamplingConfig::default()
        };
        let target = g.compose_target_sampling(&caller);

        assert_eq!(
            target.token_bias.len(),
            1,
            "caller's explicit token_bias wins"
        );
        assert!(target.token_bias.contains(42));
    }

    /// Empty cached bias + empty caller bias yields the caller config
    /// unchanged (bit-exact baseline — `Cow::Borrowed`).
    #[test]
    fn speculative_generator_empty_bias_is_bit_exact() {
        let g = SpeculativeGenerator::new(2, 1);
        let caller = SamplingConfig::default();
        let target = g.compose_target_sampling(&caller);
        assert!(matches!(target, Cow::Borrowed(_)));
        assert!(target.token_bias.is_empty());
    }

    // ------------------------------------------------------------------
    // trimmable cache validation and last-token reservation
    // ------------------------------------------------------------------

    /// All freshly-constructed KVCache entries must report `is_trimmable()`.
    /// This is the per-entry predicate consumed by `can_trim_prompt_cache`.
    #[test]
    fn kv_cache_is_trimmable_always_true() {
        // Empty cache
        let c = KVCache::new();
        assert!(c.is_trimmable());

        // Cache with accumulated state
        let mut c = KVCache::new();
        let k = ffi::ones(&[1, 2, 4, 4], dtype::FLOAT32);
        let v = ffi::ones(&[1, 2, 4, 4], dtype::FLOAT32);
        c.update(k, v);
        assert!(c.is_trimmable());
    }

    /// `can_trim_prompt_cache` returns `true` for a slice of standard KVCaches.
    #[test]
    fn can_trim_prompt_cache_all_standard() {
        use crate::cache::can_trim_prompt_cache;

        let caches: Vec<KVCache> = (0..4).map(|_| KVCache::new()).collect();
        assert!(can_trim_prompt_cache(&caches));
    }

    /// `can_trim_prompt_cache` returns `true` even for an empty slice
    /// (vacuously: all members of the empty set satisfy the predicate).
    #[test]
    fn can_trim_prompt_cache_empty_slice() {
        use crate::cache::can_trim_prompt_cache;

        let caches: Vec<KVCache> = Vec::new();
        assert!(can_trim_prompt_cache(&caches));
    }

    /// Verify that `PREFILL_STEP_SIZE` matches the upstream mlx-lm default.
    /// If this constant is changed, the test must be updated deliberately so
    /// reviewers are aware of the deviation from upstream behavior.
    #[test]
    fn prefill_step_size_matches_upstream_default() {
        assert_eq!(
            PREFILL_STEP_SIZE, 512,
            "PREFILL_STEP_SIZE must match upstream mlx-lm default (512). \
             Update this test if you intentionally deviate."
        );
    }

    struct FixedLogitModel {
        preferred_token: usize,
        eos_tokens: Vec<i32>,
    }

    impl LanguageModel for FixedLogitModel {
        fn forward(
            &self,
            input_ids: &crate::ffi::MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&crate::ffi::MlxArray>,
        ) -> UniquePtr<crate::ffi::MlxArray> {
            let shape = ffi::array_shape(input_ids);
            let batch = shape[0] as usize;
            let seq_len = shape[1] as usize;
            let vocab = 4usize;
            let mut logits = vec![-10.0f32; batch * seq_len * vocab];
            for b in 0..batch {
                for s in 0..seq_len {
                    logits[(b * seq_len + s) * vocab + self.preferred_token] = 10.0;
                }
            }
            ffi::from_slice_f32(&logits, &[shape[0], shape[1], vocab as i32])
        }

        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }

        fn num_layers(&self) -> usize {
            1
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            self.eos_tokens.clone()
        }
    }

    #[test]
    fn speculative_generate_max_tokens_one_emits_first_non_eos_token() {
        let main = FixedLogitModel {
            preferred_token: 2,
            eos_tokens: vec![3],
        };
        let draft = FixedLogitModel {
            preferred_token: 2,
            eos_tokens: vec![3],
        };
        let mut generator = SpeculativeGenerator::new(main.num_layers(), draft.num_layers());

        let (tokens, stats) =
            generator.generate(&main, &draft, &[42], 1, 1, &SamplingConfig::greedy());

        assert_eq!(
            tokens,
            vec![2],
            "max_tokens=1 must still return the first sampled non-EOS token"
        );
        assert_eq!(stats.generated_tokens, 1);
    }
}
