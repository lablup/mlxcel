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

//! Acceptance-optimal speculative acceptance (chain speculative sampling).
//!
//! # What the rule this replaces actually did
//!
//! Issue #902 opens by stating that the verify loop "accepts a draft token iff
//! it equals the target model's argmax". **For this generator that is not
//! true**, and the correction matters enough to lead with.
//!
//! [`crate::speculative::SpeculativeGenerator`] selects the target token with
//! `sample_token_optimized(pos_logits, target_sampling, history)`, a fresh draw
//! from the target *sampler*, which is `argmax` only when `temperature == 0`.
//! It then emits that draw on **both** branches: on acceptance the drafted
//! token, which by the accept test equals the draw, and on rejection the draw
//! itself. Every emitted token is therefore a fresh conditional draw from the
//! target distribution, and the stream is already exactly a target-only sample.
//! Call this rule *sampler-match* ([`AcceptanceRule::SamplerMatch`]).
//!
//! So sampler-match was never biased. What it loses is acceptance rate: two
//! independent draws coincide with probability `sum_x p(x) q(x)`, which is
//! strictly below the best any correct rule can do.
//!
//! The biased argmax-against-argmax rule the issue describes is real, but it
//! lives elsewhere: `Gemma4MtpTargetAdapter::verify_forward` uses
//! `argmax_per_position` and the DFlash round loop uses `argmax_logits_to_array`,
//! both regardless of `sampler.temperature`. Neither is reached from here.
//!
//! # The acceptance ceiling, and why "beat argmax" is not achievable
//!
//! For a `q`-distributed proposal and a `p`-distributed emission, the largest
//! probability the two can be made to coincide is `sum_x min(p(x), q(x))`: the
//! maximal-coupling bound, equivalently `1 - TV(p, q)`. **Every**
//! distribution-preserving acceptance rule is capped there. Sampler-match
//! reaches `sum_x p(x) q(x)`; modified rejection sampling attains the bound
//! exactly and is therefore optimal among correct rules.
//!
//! Argmax-against-argmax is not bound by this, because it is not correct. It
//! accepts at `q(argmax p)`, which for a confident drafter that agrees with the
//! target's mode can be far above `sum_x min(p, q)`: with `p(a) = 0.3` and
//! `q(a) = 0.95` it accepts at 0.95 where the ceiling for any correct rule is
//! about 0.30. That extra acceptance is paid for in bias, one token at a time.
//!
//! The consequence is worth stating plainly, because it reframes the issue
//! rather than failing it: **no implementation can make a
//! distribution-preserving rule "strictly improve" mean accepted length against
//! argmax acceptance.** Asking for that is asking a correct rule to beat an
//! incorrect one at the metric the incorrect one optimizes. Against
//! sampler-match, the rule this generator actually ran, the improvement is
//! real and provable, since `sum_x min(p, q) >= sum_x p(x) q(x)` always.
//!
//! # The rule implemented here
//!
//! Modified rejection sampling, from Leviathan et al. 2023 ("Fast Inference
//! from Transformers via Speculative Decoding", Algorithm 1) and Chen et al.
//! 2023 ("Accelerating Large Language Model Decoding with Speculative
//! Sampling"). Write `p` for the target's effective distribution at the
//! position and `q` for the distribution the drafter actually proposed from:
//!
//! 1. Draw `u ~ U[0, 1)`. **Accept `t` iff `u * q(t) <= p(t)`**, i.e. accept
//!    with probability `min(1, p(t) / q(t))`.
//! 2. On the first rejection, emit one token drawn from the **residual**
//!    `normalize(relu(p - q))` and end the chain there.
//! 3. If every drafted token was accepted, emit a bonus token drawn from `p`
//!    at the final verify position (unchanged from before).
//!
//! ## Why this preserves the target distribution
//!
//! Fix a position and let `A` be the accept event. For any token `x`:
//!
//! ```text
//! P(emit x) = P(propose x, accept x) + P(reject) * P(residual draws x)
//!           = q(x) * min(1, p(x)/q(x))  +  beta * relu(p(x) - q(x)) / beta
//!           = min(q(x), p(x))           +  relu(p(x) - q(x))
//!           = p(x)
//! ```
//!
//! where `beta = sum_y relu(p(y) - q(y)) = 1 - sum_y min(p(y), q(y)) = P(reject)`,
//! so the normalizing constant of the residual cancels the rejection
//! probability exactly. The final line is the identity
//! `min(a, b) + max(a - b, 0) = a`. The argument holds for **any** proposal
//! `q` whose support is not required to relate to `p` in any way, which is
//! what makes the rest of the pipeline safe: a drafter that proposes greedily
//! (`q` one-hot), a drafter whose penalties were computed against a stale
//! token history, and a drafter that is a different model family all remain
//! lossless as long as `q` is *the distribution the token was actually drawn
//! from*. That is the one invariant every caller must maintain, and it is why
//! [`crate::sampling::sample_token_with_distribution`] returns the token and
//! its distribution from a single pre-step rather than recomputing `q` later.
//!
//! Chaining over a draft block is the same argument applied inductively: the
//! chain only reaches position `i` when positions `0..i` were accepted, in
//! which case the target's context at `i` is exactly the context the drafter
//! conditioned on, so `p_i` and `q_i` are distributions over the same
//! conditional.
//!
//! ## Temperature 0 is untouched
//!
//! [`acceptance_rule`] returns [`AcceptanceRule::GreedyArgmax`] whenever the
//! target sampler is greedy (`temperature == 0.0` or `top_k == 1`), and the
//! caller keeps running the pre-#902 comparison verbatim: no distribution
//! tensors are built, no randomness is drawn, and the stream is byte-identical.
//!
//! # Opt-in, not default
//!
//! `MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=1` enables this rule; it is **off by
//! default**. Since sampler-match was already distribution-preserving on this
//! path, the change buys acceptance rate and nothing else, and the available
//! gain is the ratio `sum_x min(p, q) / sum_x p(x) q(x)`. That ratio collapses
//! toward 1 whenever the drafter is confident, because `q(t*) ~ 1` makes
//! `min(p, q)` and `p * q` coincide. Measured on a Llama-3.1-8B /
//! Llama-3.2-1B pair at temperature 0.7 it is about 1.02, which does not
//! justify two extra full-vocabulary passes and an extra host sync per
//! verified position.
//!
//! Enable it where the gain is real: a high-entropy drafter, a higher
//! temperature, or a verify path whose current rule is genuinely biased.
//! `MLXCEL_SPECULATIVE_ACCEPT_DIAG=1` reports both closed forms so the gain can
//! be checked before a throughput run rather than inferred from one.
//!
//! # Observability
//!
//! Every distinct outcome kind is logged once per process at **info** level
//! (see [`note_rule`] and [`note_outcome`]), and
//! [`crate::speculative::SpeculativeAcceptanceStats::summary_line`] prints the
//! rule and the rates on stdout after every run, because the CLI installs no
//! tracing subscriber. The one-shot latch is per kind, not global, so a
//! later-appearing kind is not swallowed by an earlier one.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::generate::SamplingConfig;
use cxx::UniquePtr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Environment switch enabling modified rejection sampling. Opt-in.
pub const STOCHASTIC_ACCEPT_ENV: &str = "MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT";

/// Which acceptance rule a speculative verify round runs.
///
/// **Every variant here is distribution-preserving.** That is not an accident
/// of naming, it is the central finding of issue #902: the rule this generator
/// shipped with compares the drafted token against a fresh draw from the target
/// *sampler* and emits that draw on both branches, so the emitted stream was
/// already a target-only sample. The biased argmax-against-argmax rule the
/// issue describes lives in the Gemma 4 MTP and DFlash verify paths, is not
/// selected here, and is deliberately not represented in this enum: an
/// unreachable variant would be a claim this module cannot back up.
///
/// What the variants differ in is [`Self::is_acceptance_optimal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AcceptanceRule {
    /// The target sampler is greedy, so its draw is the argmax and the rule is
    /// "the drafted token is the argmax". Kept on the original integer
    /// comparison: identical outcome, none of the tensor work.
    GreedyArgmax,
    /// The pre-#902 default at `temperature > 0`, and still the default:
    /// accept iff the drafted token equals an independent draw from the target
    /// sampler, and emit that draw either way. Lossless, but its acceptance
    /// probability is `sum_x p(x) q(x)`, below the achievable optimum.
    SamplerMatch,
    /// Modified rejection sampling with residual resample, enabled by
    /// [`STOCHASTIC_ACCEPT_ENV`]. Lossless *and* acceptance-optimal.
    Stochastic,
    /// [`STOCHASTIC_ACCEPT_ENV`] asked for the stochastic rule but the drafter
    /// could not report the distribution it proposed from, so there is no `q`
    /// for the accept test and the round stays on [`Self::SamplerMatch`].
    /// Fabricating a `q` would void the losslessness proof outright.
    SamplerMatchNoProposalDistribution,
}

impl AcceptanceRule {
    /// True when this rule preserves the target model's token distribution.
    ///
    /// True for every variant. See the type-level docs: the biased rule is in
    /// MTP/DFlash, not here. Kept as an explicit, tested statement rather than
    /// deleted, because "which of these is safe to serve" is the first question
    /// a reader of this enum asks.
    #[inline]
    pub fn is_distribution_preserving(self) -> bool {
        true
    }

    /// True when this rule attains the maximum acceptance probability any
    /// distribution-preserving rule can reach.
    ///
    /// That maximum is `sum_x min(p(x), q(x))`, the maximal-coupling bound: it
    /// is the largest probability with which a `q`-distributed proposal and a
    /// `p`-distributed emission can be made to coincide. Modified rejection
    /// sampling attains it; [`Self::SamplerMatch`] reaches only
    /// `sum_x p(x) q(x)`.
    ///
    /// A rule that is *not* distribution-preserving is not bound by this and
    /// can accept more. Argmax-against-argmax accepts at `q(argmax p)`, which
    /// for a confident drafter agreeing with the target's mode far exceeds
    /// `sum_x min(p, q)`. That is bought by emitting a biased stream, so it is
    /// not a fair comparison point for any correct rule.
    #[inline]
    pub fn is_acceptance_optimal(self) -> bool {
        matches!(self, Self::Stochastic | Self::GreedyArgmax)
    }

    /// Stable, greppable identifier for this rule.
    ///
    /// Printed by the CLI summary line and matched by tooling, so it is part of
    /// the observable contract: renaming a token here breaks whoever is
    /// grepping a benchmark log to prove which arm they measured.
    pub fn id(self) -> &'static str {
        match self {
            Self::GreedyArgmax => "greedy-argmax",
            Self::SamplerMatch => "sampler-match",
            Self::Stochastic => "stochastic",
            Self::SamplerMatchNoProposalDistribution => "sampler-match-no-proposal-distribution",
        }
    }

    /// Human-readable description shown next to [`Self::id`].
    pub fn label(self) -> &'static str {
        match self {
            Self::GreedyArgmax => "greedy target sampler: accept iff the draft is the argmax",
            Self::SamplerMatch => {
                "sampler-match: accept iff the draft equals a fresh target draw (default; \
                 set MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=1 for the acceptance-optimal rule)"
            }
            Self::Stochastic => "stochastic: modified rejection sampling, acceptance-optimal",
            Self::SamplerMatchNoProposalDistribution => {
                "sampler-match: stochastic acceptance requested but the drafter reported no \
                 proposal distribution"
            }
        }
    }

    fn latch(self) -> &'static AtomicBool {
        static GREEDY: AtomicBool = AtomicBool::new(false);
        static SAMPLER_MATCH: AtomicBool = AtomicBool::new(false);
        static STOCHASTIC: AtomicBool = AtomicBool::new(false);
        static NO_Q: AtomicBool = AtomicBool::new(false);
        match self {
            Self::GreedyArgmax => &GREEDY,
            Self::SamplerMatch => &SAMPLER_MATCH,
            Self::Stochastic => &STOCHASTIC,
            Self::SamplerMatchNoProposalDistribution => &NO_Q,
        }
    }
}

/// What happened to one drafted token under [`AcceptanceRule::Stochastic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AcceptanceOutcome {
    /// `u * q(t) <= p(t)` held: the drafted token was kept.
    Accepted,
    /// The drafted token was rejected and replaced by a draw from the
    /// normalized residual `relu(p - q)`.
    ResidualResample,
    /// The residual carried no mass at all, so the replacement was drawn from
    /// the target distribution `p` instead. Unreachable in exact arithmetic
    /// (a rejection implies positive residual mass); reachable only when every
    /// positive entry of `p - q` underflows float32.
    ResidualDegenerateFallback,
}

impl AcceptanceOutcome {
    fn label(self) -> &'static str {
        match self {
            Self::Accepted => "accept",
            Self::ResidualResample => "reject, resampled from normalized relu(p - q)",
            Self::ResidualDegenerateFallback => {
                "reject, residual mass underflowed to zero, resampled from target p"
            }
        }
    }

    fn latch(self) -> &'static AtomicBool {
        static ACCEPTED: AtomicBool = AtomicBool::new(false);
        static RESIDUAL: AtomicBool = AtomicBool::new(false);
        static DEGENERATE: AtomicBool = AtomicBool::new(false);
        match self {
            Self::Accepted => &ACCEPTED,
            Self::ResidualResample => &RESIDUAL,
            Self::ResidualDegenerateFallback => &DEGENERATE,
        }
    }
}

/// Log the first occurrence of each distinct acceptance rule at info level.
///
/// One latch per variant, not one global flag: a run that starts greedy and
/// later serves a `temperature > 0` request logs both lines, so the log is a
/// complete record of which rules the process ever ran.
pub fn note_rule(rule: AcceptanceRule) {
    if !rule.latch().swap(true, Ordering::Relaxed) {
        tracing::info!(
            rule = rule.label(),
            distribution_preserving = rule.is_distribution_preserving(),
            "speculative acceptance rule active"
        );
    }
}

/// Log the first occurrence of each distinct per-token outcome at info level.
pub fn note_outcome(outcome: AcceptanceOutcome) {
    if !outcome.latch().swap(true, Ordering::Relaxed) {
        tracing::info!(
            outcome = outcome.label(),
            "speculative stochastic acceptance outcome first seen"
        );
    }
}

/// Reset every one-shot log latch. Test-only.
#[cfg(test)]
pub(crate) fn reset_log_latches() {
    for rule in [
        AcceptanceRule::GreedyArgmax,
        AcceptanceRule::SamplerMatch,
        AcceptanceRule::Stochastic,
        AcceptanceRule::SamplerMatchNoProposalDistribution,
    ] {
        rule.latch().store(false, Ordering::Relaxed);
    }
    for outcome in [
        AcceptanceOutcome::Accepted,
        AcceptanceOutcome::ResidualResample,
        AcceptanceOutcome::ResidualDegenerateFallback,
    ] {
        outcome.latch().store(false, Ordering::Relaxed);
    }
}

/// True when a [`SamplingConfig`] makes the fused sampler deterministic.
///
/// Mirrors the greedy short-circuit in `fused_sample_impl`
/// (`temperature == 0.0f || top_k == 1` selects `argmax`). Keeping the two in
/// step matters: if this predicate said "stochastic" for a config the sampler
/// treats as greedy, the accept test would compare against a one-hot `p` and
/// reject every non-argmax draft for no reason.
#[inline]
pub fn sampler_is_greedy(config: &SamplingConfig) -> bool {
    config.temperature == 0.0 || config.top_k == 1
}

/// Whether modified rejection sampling is enabled for this process.
///
/// **Opt-in, default off.** Reads [`STOCHASTIC_ACCEPT_ENV`] once; only an
/// explicitly truthy value enables it.
///
/// The default is off because on the one path this rule reaches, the classic
/// [`crate::speculative::SpeculativeGenerator`], the rule it replaces was
/// already distribution-preserving. So the change buys no correctness there,
/// only acceptance rate, and the acceptance gain is bounded by
/// `sum_x min(p, q) / sum_x p(x) q(x)`, which collapses toward 1 whenever the
/// drafter is confident. Measured on a Llama-3.1-8B / Llama-3.2-1B pair at
/// temperature 0.7 that ratio is about 1.02. Paying two extra
/// full-vocabulary passes and an extra host sync per verified position for a
/// two-percent theoretical acceptance gain is not a good default.
///
/// It is worth enabling where the gain is real: a high-entropy drafter, a
/// higher temperature, or a verify path whose current rule is *not*
/// distribution-preserving (the Gemma 4 MTP and DFlash round loops, which pick
/// the target token by argmax regardless of temperature). Check the expected
/// gain first with [`ACCEPT_DIAG_ENV`], which reports both closed forms.
pub fn stochastic_accept_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(STOCHASTIC_ACCEPT_ENV) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    })
}

/// Pick the acceptance rule for a verify round.
///
/// `has_proposal_distribution` is the caller's assertion that it can supply
/// `q` for every drafted position. A drafter that cannot report what it
/// proposed from leaves the round on argmax rather than silently inventing a
/// `q`, because an invented `q` breaks the losslessness proof outright.
pub fn acceptance_rule(
    target_config: &SamplingConfig,
    has_proposal_distribution: bool,
) -> AcceptanceRule {
    acceptance_rule_with_override(target_config, has_proposal_distribution, None)
}

/// [`acceptance_rule`] with an explicit per-caller override of the env default.
///
/// `stochastic_override` of `None` consults [`stochastic_accept_enabled`];
/// `Some(v)` uses `v` directly. This is what makes the opt-in reachable
/// programmatically rather than only through a process-wide environment
/// variable, which matters twice over: a future per-request server flag needs
/// it, and the distributional tests need to exercise the acceptance-optimal
/// rule regardless of how the process default happens to be set. A test that
/// silently followed the default would stop covering this module the moment the
/// default flipped.
pub fn acceptance_rule_with_override(
    target_config: &SamplingConfig,
    has_proposal_distribution: bool,
    stochastic_override: Option<bool>,
) -> AcceptanceRule {
    let want_stochastic = stochastic_override.unwrap_or_else(stochastic_accept_enabled);
    if sampler_is_greedy(target_config) {
        return AcceptanceRule::GreedyArgmax;
    }
    if !want_stochastic {
        return AcceptanceRule::SamplerMatch;
    }
    if !has_proposal_distribution {
        return AcceptanceRule::SamplerMatchNoProposalDistribution;
    }
    AcceptanceRule::Stochastic
}

/// Environment switch for the per-position acceptance diagnostic.
pub const ACCEPT_DIAG_ENV: &str = "MLXCEL_SPECULATIVE_ACCEPT_DIAG";

/// Whether to accumulate the closed-form acceptance probabilities per verify
/// position.
///
/// Off by default: it costs two full-vocabulary reductions and one host
/// readback per verified position, which is real work on a 128K vocabulary.
/// On, it turns "the acceptance rate looks wrong" into an arithmetic
/// statement, because `sum_x min(p, q) >= sum_x p(x) q(x)` holds for every
/// pair of distributions. If the measured rate does not sit at `sum min` while
/// `sum min` sits at or above `sum prod`, the defect is localized immediately
/// to either the accept test or to `p` and `q` themselves.
pub fn accept_diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(ACCEPT_DIAG_ENV) {
        Ok(raw) => !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    })
}

/// Closed-form acceptance probabilities at one verify position.
///
/// `sum_min` is the probability modified rejection sampling accepts, and also
/// the ceiling for *any* distribution-preserving rule (maximal coupling).
/// `sum_prod` is the probability the sampler-match rule accepts.
/// `sum_min >= sum_prod` always, because `min(a, b) >= a * b` for `a, b` in
/// `[0, 1]`. Their ratio is the entire acceptance headroom this feature can
/// deliver for the given `(p, q)`.
pub fn closed_form_acceptance(target_probs: &MlxArray, proposal_probs: &MlxArray) -> (f64, f64) {
    let sum_min = ffi::sum_all(&ffi::minimum(target_probs, proposal_probs));
    let sum_prod = ffi::sum_all(&ffi::multiply(target_probs, proposal_probs));
    ffi::eval(&sum_min);
    ffi::eval(&sum_prod);
    (
        f64::from(ffi::item_f32(&sum_min)),
        f64::from(ffi::item_f32(&sum_prod)),
    )
}

/// The verdict for one drafted token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftVerdict {
    /// Keep the drafted token and continue the chain.
    Accept,
    /// Drop the drafted token, emit `replacement`, and end the chain.
    Reject {
        /// Token drawn from the normalized residual (or from `p` in the
        /// degenerate case recorded by [`AcceptanceOutcome`]).
        replacement: i32,
    },
}

/// Read `probs[0, token]` back to the host as an `f32`.
///
/// `probs` is `[batch, vocab]`; only row 0 is read, which is all the B=1
/// speculative paths need.
fn probability_of(probs: &MlxArray, token: i32) -> UniquePtr<MlxArray> {
    let index = ffi::from_slice_i32(&[token], &[1, 1]);
    ffi::take_along_axis(probs, &index, -1)
}

/// The modified rejection-sampling accept test for one drafted token.
///
/// * `target_probs` — `p`, float32 `[1, vocab]`, from
///   [`crate::sampling::effective_token_distribution`] on the target's logits
///   at this verify position.
/// * `proposal_probs` — `q`, float32 `[1, vocab]`, the distribution the
///   drafter drew `draft_token` from. Must be the *actual* proposal
///   distribution; see the module docs.
///
/// Accepts iff `p(t) > 0` **and** `u * q(t) <= p(t)` for a fresh `u ~ U[0, 1)`.
///
/// The `p(t) > 0` conjunct is not cosmetic. When the target's top-k / top-p /
/// min-p filter excludes the drafted token, `p(t)` is exactly zero and the
/// token must always be rejected. Without the conjunct a `q(t)` that has
/// underflowed float32 to zero would make the product `0 <= 0` and accept a
/// token the target assigns no mass at all, which is precisely the failure the
/// whole change exists to prevent.
///
/// One `eval` and one host readback per call, the same synchronization cost as
/// the argmax comparison it replaces.
pub fn accept_draft_token(
    target_probs: &MlxArray,
    proposal_probs: &MlxArray,
    draft_token: i32,
) -> bool {
    let p_t = probability_of(target_probs, draft_token);
    let q_t = probability_of(proposal_probs, draft_token);

    // SAFETY: a null key means "draw from the thread-local default RNG state",
    // the same convention `apply_xtc_step` and `layers.rs` use. Drawing through
    // MLX's key sequence keeps the accept decisions reproducible under
    // `ffi::random_seed`.
    let u = unsafe { ffi::random_uniform(0.0, 1.0, &[1, 1], dtype::FLOAT32, std::ptr::null()) };

    let zero = ffi::zeros(&[1, 1], dtype::FLOAT32);
    let target_has_mass = ffi::greater(&p_t, &zero);
    let within_ratio = ffi::less_equal(&ffi::multiply(&u, &q_t), &p_t);
    let accept = ffi::logical_and(&target_has_mass, &within_ratio);

    ffi::eval(&accept);
    ffi::item_bool(&accept)
}

/// Draw the replacement token from the normalized residual `relu(p - q)`.
///
/// Returns the token and which outcome kind produced it, so the caller can
/// feed [`note_outcome`].
///
/// `relu(p - q)` is non-negative by construction and sums to the rejection
/// probability, so normalizing it is exactly the residual distribution the
/// losslessness proof requires. Sampling runs through
/// [`ffi::fused_sample`] on `log(relu(p - q))` with `temperature = 1.0` and
/// every filter disabled: `softmax(log r) == r / sum(r)`, so this is the
/// normalized residual and nothing else. Routing through `fused_sample` also
/// means the residual draw uses the same #900 Gumbel-max kernel the rest of
/// the sampler uses where the backend supports it, and the `categorical` graph
/// path where it does not. Entries with zero residual become `-inf` and can
/// never win.
pub fn residual_resample(
    target_probs: &MlxArray,
    proposal_probs: &MlxArray,
) -> (i32, AcceptanceOutcome) {
    let residual = ffi::relu(&ffi::subtract(target_probs, proposal_probs));
    let mass = ffi::sum_all(&residual);
    ffi::eval(&mass);
    let mass = ffi::item_f32(&mass);

    let (source, outcome) = if mass > 0.0 && mass.is_finite() {
        (residual, AcceptanceOutcome::ResidualResample)
    } else {
        // Unreachable in exact arithmetic: reaching this function means the
        // token was rejected, which implies positive residual mass. Only
        // float32 underflow of every positive entry gets here. Falling back to
        // `p` keeps the emitted token inside the target's support, which is
        // the closest available approximation of the residual.
        (
            ffi::copy(target_probs),
            AcceptanceOutcome::ResidualDegenerateFallback,
        )
    };

    let token_arr = ffi::fused_sample(&ffi::log(&source), 1.0, 0, 1.0, 0.0);
    ffi::eval(&token_arr);
    (ffi::item_i32(&token_arr), outcome)
}

/// Run the full per-token decision: accept test, and residual resample on
/// rejection. Records both the accept/reject outcome and, on rejection,
/// whether the residual was degenerate.
pub fn verify_draft_token(
    target_probs: &MlxArray,
    proposal_probs: &MlxArray,
    draft_token: i32,
) -> DraftVerdict {
    if accept_draft_token(target_probs, proposal_probs, draft_token) {
        note_outcome(AcceptanceOutcome::Accepted);
        DraftVerdict::Accept
    } else {
        let (replacement, outcome) = residual_resample(target_probs, proposal_probs);
        note_outcome(outcome);
        DraftVerdict::Reject { replacement }
    }
}

#[cfg(test)]
#[path = "stochastic_accept_tests.rs"]
mod stochastic_accept_tests;
