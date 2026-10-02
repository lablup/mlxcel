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

//! Prompt-lookup (n-gram) speculative decoding.
//!
//! A drafter that needs no second model. When the last few tokens of the
//! sequence also appear earlier in the context (prompt plus everything emitted
//! so far), the tokens that followed that earlier occurrence are proposed as
//! the draft, and the target verifies the whole block in one forward, exactly
//! as [`super::SpeculativeGenerator`] verifies a draft model's proposals.
//!
//! It pays off on replies that copy long runs of the prompt: editing a
//! function, fixing typos, adding a field to a JSON document, turning a CSV
//! into a table. Those replies repeat the prompt word for word for most of
//! their length, so most proposals are accepted and a verify forward (whose
//! cost is dominated by reading the weights once) emits several tokens.
//!
//! ## Round shape
//!
//! Each round forwards `[current_token, d_0, ..., d_{k-1}]` through the target,
//! keeps the longest prefix of the proposals the target agrees with, emits one
//! more token from the target (the replacement at the first disagreement, or
//! the bonus token after a fully accepted block), and trims the rejected tail
//! off the KV caches. With no match the round is `k = 0`: a plain one-token
//! decode step.
//!
//! ## Pipelining
//!
//! A round without a proposal is pipelined the way plain decoding is: its
//! one-token forward is submitted from the previous round's still-lazy sample,
//! before the host reads that sample, so the GPU goes straight from one step to
//! the next. Waiting on each token before encoding the next step cost 12% of
//! decode time on an M4 Pro. Pipelining starts only after a few rounds in a
//! row found no proposal, because a pipelined step cannot carry one. When
//! lookup finds a proposal while a step is in flight, the host reads the
//! in-flight token and verifies from there synchronously; that switch costs no
//! extra forward.
//!
//! ## Adaptive drafting
//!
//! A verify block is not free: on an M4 Pro an eight-token block costs about
//! two and a half one-token steps. On prose, where lookup matches a common
//! token and the target rarely agrees, proposing every round made decoding a
//! quarter slower than plain decode. [`DraftGovernor`] keeps the block short
//! while recent rounds land few tokens and stops proposing after repeated
//! misses. Edits keep landing five or six tokens a round, so they keep the full
//! block. How it shortens and when it resumes is a [`DraftPolicy`], because the
//! cost of a verify width depends on the backend's kernels: the
//! [`DraftPolicy::Graded`] rules were tuned on an M4 Pro, the
//! [`DraftPolicy::Gated`] rules on GB10.
//!
//! ## Acceptance
//!
//! A lookup proposal is deterministic, so its proposal distribution `q` is one
//! hot on the proposed token. Under that `q` the acceptance-optimal rule of
//! modified rejection sampling accepts with probability `p(d)` and resamples
//! from `p` restricted to tokens other than `d`, which is exactly what the
//! sampler-match rule does: draw `t ~ p` and accept iff `t == d`. So one rule
//! serves every sampler configuration and the emitted stream is distributed as
//! the target's own.
//!
//! ## Greedy exactness
//!
//! The prompt is prefilled by the same routine plain decoding uses
//! ([`crate::generate::prefill_prompt_last_logits`]), so the caches and the
//! first token match plain decoding exactly, and a round without a proposal
//! is the same one-token forward. A round with proposals runs a multi-token
//! forward, whose kernels may round differently from the one-token path; where
//! the target's top two logits are that close, greedy output can take the
//! other token from there on. Measured on Qwen3-1.7B-4bit (M4 Pro), edit
//! replies stayed identical to plain decoding. This is the same jitter class as
//! the draft-model path; nothing here gates on it the way the MTP exactness
//! probe does.
//!
//! ## Cache contract
//!
//! Rollback is `KVCache::trim`, so the target must keep all of its sequence
//! state in the external KV caches. Recurrent and hybrid families (Mamba,
//! GatedDeltaNet, NemotronH, ...) fold every consumed token into a fixed-size
//! state that cannot be trimmed; callers must refuse them, see
//! [`prompt_lookup_unsupported_reason`].

use crate::cache::{KVCacheMode, SequenceStateBackend, can_trim_prompt_cache};
use crate::drafter::dflash::materialize_argmax_i32_vec;
use crate::ffi;
use crate::ffi::MlxThreadLocalStream;
use crate::generate::{
    GenerationStats, LanguageModel, SamplingConfig, prefill_prompt_last_logits,
    resolve_kv_cache_layer_modes,
};
use crate::generation_policy::{initial_token_history, merged_eos_token_ids, seed_rng_if_needed};
use crate::layers::KVCache;
use crate::loop_detection::detect_repetition_loop;
use crate::sampling::{
    SamplerState, TokenBiasMap, sample_token_optimized, sample_token_optimized_with_state,
};
use crate::streams::{install_thread_local_default_stream, shared_thread_local_generation_stream};
use cxx::UniquePtr;
use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Instant;

use super::trim_caches;

/// Default longest n-gram matched against the context.
pub const DEFAULT_NGRAM_MAX: usize = 3;
/// Default shortest n-gram matched against the context.
///
/// A one-token match is usually a common token (a space, `the`, a comma), and
/// what followed it earlier rarely comes next. Measured on Qwen3-1.7B-4bit, a
/// minimum of 1 slowed prose by a quarter while a minimum of 2 kept edits just
/// as fast.
pub const DEFAULT_NGRAM_MIN: usize = 2;
/// Default maximum number of tokens proposed per round (verify block of 8).
pub const DEFAULT_MAX_DRAFT: usize = 7;
/// Largest accepted `ngram_max`.
///
/// [`NgramIndex`] keeps one map per n-gram length in `ngram_min..=ngram_max`,
/// and each map holds a boxed `n`-token key for every context position, so its
/// host memory grows with the square of the range times the context length.
/// Without a bound, a mistyped `--prompt-lookup-ngram-max 1000` over a long
/// document asks for hundreds of gigabytes before the first decode step.
pub const NGRAM_MAX_LIMIT: usize = 16;
/// Largest accepted `max_draft`. The verify block is one wider, and every
/// proposal is a forward position the target computes.
pub const MAX_DRAFT_LIMIT: usize = 64;

/// Tunables for [`PromptLookupGenerator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLookupConfig {
    /// Longest suffix n-gram tried first.
    pub ngram_max: usize,
    /// Shortest suffix n-gram tried before giving up on a round.
    pub ngram_min: usize,
    /// Maximum proposals per round. The verify block is one wider.
    pub max_draft: usize,
    /// Shorten or pause proposals while they stop landing, see
    /// [`DraftGovernor`]. `false` proposes up to `max_draft` every round.
    pub adaptive: bool,
    /// How the adaptive governor sizes blocks and decides when to resume.
    /// Ignored when `adaptive` is `false`.
    pub policy: DraftPolicy,
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        Self {
            ngram_max: DEFAULT_NGRAM_MAX,
            ngram_min: DEFAULT_NGRAM_MIN,
            max_draft: DEFAULT_MAX_DRAFT,
            adaptive: true,
            policy: DraftPolicy::default_for_backend(),
        }
    }
}

/// How [`DraftGovernor`] sizes verify blocks and decides when to resume
/// proposing after a run of misses.
///
/// The two policies answer one question differently: how much a drafted round
/// that lands nothing costs against a plain step. On an M4 Pro a wide block
/// is cheap, so [`Self::Graded`] grades the block with the recent acceptance
/// and probes again after a short pause. On GB10 (CUDA, affine 4-bit, MLX pin
/// `81ba1c6a`, release build) a synchronous verify forward measured, in
/// pipelined one-token steps for Qwen3-1.7B / Qwen3-8B: width 2 at
/// 1.37 / 1.15, width 3 at 1.76 / 1.59, width 4 at 2.43 / 2.10, width 7 at
/// 3.84 / 3.85, and width 8 and up at 3.91 / 3.30
/// (`examples/verify_width_cost.rs`; all four benchmarked models in
/// `docs/benchmark_results/data/prompt-lookup-governor-gb10-2026-10-02/`).
/// Below 8 rows the affine path runs `qmv`'s multirow kernel, instantiated at
/// 2, 4 and 8 accumulator rows, so 5 to 7 rows pay for the 8-row
/// instantiation; from 8 rows it switches to `qmm_sm80`. A block between
/// narrow and full therefore buys a few more proposals for close to a full
/// block's price (on the 8B models, more than a full block's), and a miss
/// also drains the pipeline. So [`Self::Gated`] budgets only a narrow or a
/// full block (a lookup near the end of the context, or the `max_tokens`
/// limit, can still return fewer tokens) and spends no verify forward to find
/// out whether a copy has resumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DraftPolicy {
    /// PR #2074's rules: the block is twice the recent accepted average plus
    /// two, and a pause after `MISSES_BEFORE_COOLDOWN` (3) misses lasts
    /// `MIN_COOLDOWN` (4) rounds, doubling up to `MAX_COOLDOWN` (32), before the
    /// next drafted round probes again.
    Graded,
    /// Narrow (`GATED_NARROW_DRAFT`, 2 proposals) until a narrow block lands
    /// whole, then full (`max_draft`). While paused it keeps looking up and
    /// checks each proposal against the tokens decoding emits next; proposing
    /// resumes only once `SHADOW_CONFIRM` (2) proposed tokens in a row came
    /// true, and until a round lands a whole narrow block, a round that lands
    /// nothing pauses again at once.
    Gated,
}

impl DraftPolicy {
    /// [`Self::Gated`] on CUDA builds, where it was measured, and
    /// [`Self::Graded`] everywhere else, where it was tuned.
    pub const fn default_for_backend() -> Self {
        if cfg!(feature = "cuda") {
            Self::Gated
        } else {
            Self::Graded
        }
    }
}

impl PromptLookupConfig {
    /// Reject configurations that could never propose anything.
    pub fn validate(&self) -> Result<(), String> {
        if self.ngram_min == 0 {
            return Err("prompt-lookup ngram-min must be at least 1".to_string());
        }
        if self.ngram_max < self.ngram_min {
            return Err(format!(
                "prompt-lookup ngram-max ({}) must be >= ngram-min ({})",
                self.ngram_max, self.ngram_min
            ));
        }
        if self.ngram_max > NGRAM_MAX_LIMIT {
            return Err(format!(
                "prompt-lookup ngram-max ({}) must be at most {NGRAM_MAX_LIMIT}",
                self.ngram_max
            ));
        }
        if self.max_draft == 0 {
            return Err("prompt-lookup max-draft must be at least 1".to_string());
        }
        if self.max_draft > MAX_DRAFT_LIMIT {
            return Err(format!(
                "prompt-lookup max-draft ({}) must be at most {MAX_DRAFT_LIMIT}",
                self.max_draft
            ));
        }
        Ok(())
    }
}

/// Why `model` cannot run under [`PromptLookupGenerator`], or `None` when it
/// can.
///
/// Verification rolls rejected tokens back with `KVCache::trim`, which is only
/// sound when every piece of per-sequence state lives in the external caches.
/// `supports_batching` is the existing marker for that, and a
/// [`SequenceStateBackend::ModelOwned`] layout is the direct statement of the
/// opposite: such a family keeps its caches inside the model, the external
/// caches are empty placeholders, and trimming them rolls nothing back.
/// `supports_padded_prefill` is `false` for the families whose state trailing
/// tokens corrupt past what a trim undoes, which is the same hazard a rejected
/// draft block poses.
///
/// The layout check keeps the refusal from resting on the other two flags:
/// Gemma 4 and Llama 4 report `supports_batching() == true` and were refused
/// only because they opt out of padded prefill, a flag that answers a
/// different question (the scheduler's post-pad trim, issue #1335) and may
/// change once that is solved.
pub fn prompt_lookup_unsupported_reason<M: LanguageModel>(model: &M) -> Option<&'static str> {
    if !model.supports_batching() {
        return Some(
            "this model keeps sequence state outside the external KV caches (recurrent or \
             hybrid layers, or model-owned sliding-window caches), so a rejected draft \
             cannot be rolled back",
        );
    }
    if model.sequence_state_layout().backend == SequenceStateBackend::ModelOwned {
        return Some(
            "this model owns its attention caches internally rather than in the external KV \
             caches the generator trims, so a rejected draft cannot be rolled back",
        );
    }
    if !model.supports_padded_prefill() {
        return Some(
            "this model's state is corrupted by trailing tokens that a cache trim cannot \
             undo (the same reason it opts out of padded prefill), so a rejected draft \
             cannot be rolled back",
        );
    }
    None
}

/// Whether `model` can run under [`PromptLookupGenerator`]. See
/// [`prompt_lookup_unsupported_reason`] for why not.
pub fn supports_prompt_lookup<M: LanguageModel>(model: &M) -> bool {
    prompt_lookup_unsupported_reason(model).is_none()
}

/// Smallest pause, in rounds, after proposals stop landing ([`DraftPolicy::Graded`]).
const MIN_COOLDOWN: usize = 4;
/// Largest pause ([`DraftPolicy::Graded`]). Kept short so an edit that writes
/// a new passage and then resumes copying is back on full blocks within a line
/// or two.
const MAX_COOLDOWN: usize = 32;
/// Consecutive drafted rounds that land nothing before a pause.
const MISSES_BEFORE_COOLDOWN: usize = 3;
/// Proposals in a [`DraftPolicy::Gated`] narrow block (verify width 3, the
/// widest block below the 4-row multirow instantiation on CUDA).
const GATED_NARROW_DRAFT: usize = 2;
/// Accepted-proposal average at which [`DraftPolicy::Gated`] switches from the
/// narrow to the full block. A narrow block that lands whole lifts the average
/// from its starting value to exactly this, so one clean narrow round is
/// enough, and the first full block that lands nothing drops it back below;
/// once full blocks have landed several tokens, it takes a run of misses.
const GATED_FULL_AT: f64 = 1.5;
/// Accepted-proposal average a [`DraftPolicy::Gated`] governor starts from and
/// resumes with: below [`GATED_FULL_AT`], so the first block is narrow.
const GATED_START_EMA: f64 = 1.0;
/// Proposed tokens in a row a paused [`DraftPolicy::Gated`] governor must see
/// come true before it proposes again. One is not enough on prose: the token
/// after a two-token match is often a common one that happens to recur.
pub(crate) const SHADOW_CONFIRM: usize = 2;
/// Consecutive rounds without a proposal before the decode loop pipelines.
///
/// A pipelined step is submitted before the host knows the token it follows,
/// so it cannot carry a proposal: the first proposal after a pipelined run
/// waits one step. Edits that miss for a token or two at each changed field
/// (a JSON id, a renamed identifier) would pay that step at every field, so
/// only a run of plain rounds switches to pipelining. A paused
/// [`DraftPolicy::Gated`] governor pipelines at once: it cannot propose until
/// a shadow probe settles, which takes at least [`SHADOW_CONFIRM`] emitted
/// tokens, so there is no proposal for a pipelined step to delay.
const PLAIN_ROUNDS_BEFORE_PIPELINE: usize = 2;

/// Decides, round by round, how many lookup tokens to propose.
///
/// Tracks an exponential moving average of the proposals each drafted round
/// landed. Under [`DraftPolicy::Graded`] it caps the next block at twice that
/// plus two, so a run of full matches keeps the full block and a run of
/// one-token landings verifies five-wide blocks instead of eight (lookup
/// landings are all-or-nothing more often than not, so the cap leaves room
/// past the average); after [`MISSES_BEFORE_COOLDOWN`] drafted rounds in a row
/// land nothing, it stops proposing for a cooldown that doubles on every repeat
/// up to [`MAX_COOLDOWN`] and resets once a round lands a token. Under
/// [`DraftPolicy::Gated`] the block is narrow or full, and a pause has no
/// length: it ends when [`Self::shadow_confirmed`] reports a proposal that came
/// true without being verified.
#[derive(Debug, Clone)]
pub(crate) struct DraftGovernor {
    max_draft: usize,
    adaptive: bool,
    policy: DraftPolicy,
    accepted_ema: f64,
    misses: usize,
    /// [`DraftPolicy::Graded`]: paused rounds left, and the next pause length.
    cooldown: usize,
    next_cooldown: usize,
    /// [`DraftPolicy::Gated`]: proposing at all, and whether the next miss
    /// pauses outright because no round since the last resume has landed a
    /// whole narrow block.
    drafting: bool,
    probation: bool,
}

impl DraftGovernor {
    pub(crate) fn new(config: &PromptLookupConfig) -> Self {
        let accepted_ema = match config.policy {
            // Optimistic start: the first match in an edit should get the
            // whole block.
            DraftPolicy::Graded => config.max_draft as f64,
            // A first block that misses on prose costs a narrow verify, not
            // a full one; an edit earns the full block one round later.
            DraftPolicy::Gated => GATED_START_EMA,
        };
        Self {
            max_draft: config.max_draft,
            adaptive: config.adaptive,
            policy: config.policy,
            accepted_ema,
            misses: 0,
            cooldown: 0,
            next_cooldown: MIN_COOLDOWN,
            drafting: true,
            // The first drafted round of a reply has no evidence behind it.
            probation: true,
        }
    }

    /// Proposals allowed this round; `0` means run a plain decode step. A
    /// paused [`DraftPolicy::Graded`] round counts down the pause.
    pub(crate) fn budget(&mut self) -> usize {
        if !self.adaptive {
            return self.max_draft;
        }
        match self.policy {
            DraftPolicy::Graded => {
                if self.cooldown > 0 {
                    self.cooldown -= 1;
                    return 0;
                }
                ((2.0 * self.accepted_ema).round() as usize + 2).clamp(1, self.max_draft)
            }
            DraftPolicy::Gated => {
                if !self.drafting {
                    0
                } else if self.accepted_ema >= GATED_FULL_AT {
                    self.max_draft
                } else {
                    GATED_NARROW_DRAFT.min(self.max_draft)
                }
            }
        }
    }

    /// Record a drafted round that landed `accepted` proposals.
    pub(crate) fn record(&mut self, accepted: usize) {
        if !self.adaptive {
            return;
        }
        self.accepted_ema = 0.5 * self.accepted_ema + 0.5 * accepted as f64;
        match self.policy {
            DraftPolicy::Graded => {
                if accepted > 0 {
                    self.misses = 0;
                    self.next_cooldown = MIN_COOLDOWN;
                    return;
                }
                self.misses += 1;
                if self.misses >= MISSES_BEFORE_COOLDOWN {
                    self.misses = 0;
                    self.cooldown = self.next_cooldown;
                    self.next_cooldown = (self.next_cooldown * 2).min(MAX_COOLDOWN);
                }
            }
            DraftPolicy::Gated => {
                // Probation ends on a round that lands a whole narrow block:
                // one stray token is what prose lands too, and on GB10 a run
                // of such rounds (8B email: 6 drafted, 3 tokens landed) cost
                // more than it emitted.
                if accepted >= GATED_NARROW_DRAFT.min(self.max_draft) {
                    self.probation = false;
                }
                if accepted > 0 {
                    self.misses = 0;
                    return;
                }
                self.misses += 1;
                if self.probation || self.misses >= MISSES_BEFORE_COOLDOWN {
                    self.drafting = false;
                    self.misses = 0;
                }
            }
        }
    }

    /// Whether the decode loop should keep looking up while this governor is
    /// paused and report proposals that come true via
    /// [`Self::shadow_confirmed`].
    pub(crate) fn probes_while_paused(&self) -> bool {
        self.adaptive && self.policy == DraftPolicy::Gated && !self.drafting
    }

    /// A paused [`DraftPolicy::Gated`] governor saw a lookup proposal come
    /// true for [`SHADOW_CONFIRM`] tokens: resume with a narrow block, on
    /// probation.
    pub(crate) fn shadow_confirmed(&mut self) {
        if self.probes_while_paused() {
            self.drafting = true;
            self.probation = true;
            self.misses = 0;
            self.accepted_ema = GATED_START_EMA;
        }
    }
}

/// A lookup proposal made while proposals are paused, tested against the
/// tokens decoding goes on to emit instead of against a verify forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShadowProbe {
    /// Context length when the proposal was made: `proposal[i]` claims the
    /// token that lands at `context[start + i]`.
    pub(crate) start: usize,
    pub(crate) proposal: Vec<i32>,
}

impl ShadowProbe {
    /// `Some(true)` once the first [`SHADOW_CONFIRM`] proposed tokens all came
    /// true, `Some(false)` at the first one that did not, and `None` while
    /// decoding has not emitted that far yet.
    pub(crate) fn settle(&self, context: &[i32]) -> Option<bool> {
        for (i, &claimed) in self.proposal.iter().take(SHADOW_CONFIRM).enumerate() {
            match context.get(self.start + i) {
                None => return None,
                Some(&emitted) if emitted != claimed => return Some(false),
                Some(_) => {}
            }
        }
        Some(true)
    }
}

/// Propose up to `max_draft` tokens by matching the tail of `context` against
/// an earlier position in `context`.
///
/// Tries suffix lengths from `ngram_max` down to `ngram_min` and, for each,
/// takes the most recent earlier occurrence that has at least one token after
/// it. The most recent occurrence is preferred because a reply that copies the
/// prompt advances through it front to back, so the latest match is the one
/// the copy is currently walking.
///
/// Returns an empty vector when nothing matches.
pub fn find_draft(context: &[i32], config: &PromptLookupConfig) -> Vec<i32> {
    let len = context.len();
    if config.max_draft == 0 || config.ngram_min == 0 {
        return Vec::new();
    }
    let ngram_max = config.ngram_max.min(len.saturating_sub(1));
    if ngram_max < config.ngram_min {
        return Vec::new();
    }

    for n in (config.ngram_min..=ngram_max).rev() {
        let pattern = &context[len - n..];
        // `start` is where a candidate occurrence begins. It must end before
        // the suffix itself does (`start + n < len`) so at least one token
        // follows it; walk from the latest such position backwards.
        let mut start = len - n;
        while start > 0 {
            start -= 1;
            if &context[start..start + n] == pattern {
                let from = start + n;
                let to = (from + config.max_draft).min(len);
                return context[from..to].to_vec();
            }
        }
    }
    Vec::new()
}

/// Incremental index answering [`find_draft`] without scanning the context.
///
/// For each n-gram length in the configured range it maps an n-gram to the
/// start of its most recent occurrence that already has a token after it,
/// which is exactly the occurrence [`find_draft`] settles on. It is built once
/// from the prompt and extended by the tokens emitted since the last call, so
/// a lookup costs one hash probe per n-gram length however long the context
/// grows, where a scan costs a pass over the whole context every round.
#[derive(Debug)]
pub(crate) struct NgramIndex {
    ngram_min: usize,
    /// `latest[n - ngram_min]`: n-gram to the start of its latest occurrence
    /// that has a follower.
    latest: Vec<HashMap<Box<[i32]>, usize>>,
    /// Context length already indexed.
    indexed_len: usize,
}

impl NgramIndex {
    pub(crate) fn new(config: &PromptLookupConfig) -> Self {
        let lengths = (config.ngram_min..=config.ngram_max).count();
        Self {
            ngram_min: config.ngram_min,
            latest: (0..lengths).map(|_| HashMap::new()).collect(),
            indexed_len: 0,
        }
    }

    /// Index every n-gram that gained a follower since the last call.
    /// `context` must extend the context of the previous call.
    pub(crate) fn extend(&mut self, context: &[i32]) {
        // An n-gram ending at `end` has a follower once `end + 1 < len`. The
        // previous call indexed every end up to `indexed_len - 2`.
        for end in self.indexed_len.saturating_sub(1)..context.len().saturating_sub(1) {
            for (offset, map) in self.latest.iter_mut().enumerate() {
                let n = self.ngram_min + offset;
                if end + 1 < n {
                    break;
                }
                let start = end + 1 - n;
                map.insert(context[start..=end].into(), start);
            }
        }
        self.indexed_len = context.len();
    }

    /// Same proposal as [`find_draft`] on the indexed `context`.
    pub(crate) fn find(&self, context: &[i32], config: &PromptLookupConfig) -> Vec<i32> {
        debug_assert_eq!(self.indexed_len, context.len(), "index is stale");
        let len = context.len();
        if config.max_draft == 0 || config.ngram_min == 0 {
            return Vec::new();
        }
        let ngram_max = config.ngram_max.min(len.saturating_sub(1));
        if ngram_max < config.ngram_min {
            return Vec::new();
        }
        for n in (config.ngram_min..=ngram_max).rev() {
            if let Some(&start) = self.latest[n - self.ngram_min].get(&context[len - n..]) {
                let from = start + n;
                let to = (from + config.max_draft).min(len);
                return context[from..to].to_vec();
            }
        }
        Vec::new()
    }
}

/// Acceptance accounting for one [`PromptLookupGenerator::generate`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PromptLookupStats {
    /// Target forwards in the decode loop (one per round).
    pub rounds: usize,
    /// Rounds whose lookup found a match and verified a draft block.
    pub drafted_rounds: usize,
    /// One-token rounds that verified no proposal because the
    /// [`DraftGovernor`] paused proposals or `max_tokens` left no room. A
    /// paused [`DraftPolicy::Gated`] round still looks up, to test the
    /// proposal against what decoding emits next.
    pub paused_rounds: usize,
    /// Tokens proposed across all rounds.
    pub proposed_draft_tokens: usize,
    /// Proposals the target accepted.
    pub accepted_draft_tokens: usize,
    /// Times a paused [`DraftPolicy::Gated`] governor resumed because a
    /// lookup proposal came true without being verified.
    pub shadow_confirmations: usize,
}

impl PromptLookupStats {
    /// Accepted proposals per proposed token.
    pub fn acceptance_rate(&self) -> f64 {
        ratio(self.accepted_draft_tokens, self.proposed_draft_tokens)
    }

    /// Emitted tokens per target forward in the decode loop. `1.0` is plain
    /// decoding; this is the multiplier lookup buys before verify overhead.
    pub fn tokens_per_forward(&self, generated_tokens: usize) -> f64 {
        // The first token comes out of the prefill forward, not a round.
        ratio(generated_tokens.saturating_sub(1), self.rounds)
    }

    /// One-line summary for the CLI, which installs no tracing subscriber.
    pub fn summary_line(&self, generated_tokens: usize) -> String {
        format!(
            "[Prompt lookup] rounds={} drafted_rounds={} paused_rounds={} proposed={} \
             accepted={} acceptance_rate={:.4} tokens_per_forward={:.4} \
             shadow_confirmations={}",
            self.rounds,
            self.drafted_rounds,
            self.paused_rounds,
            self.proposed_draft_tokens,
            self.accepted_draft_tokens,
            self.acceptance_rate(),
            self.tokens_per_forward(generated_tokens),
            self.shadow_confirmations,
        )
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// Speculative generator whose drafter is [`find_draft`], answered from an
/// [`NgramIndex`].
pub struct PromptLookupGenerator {
    caches: Vec<KVCache>,
    config: PromptLookupConfig,
    kv_cache_mode: KVCacheMode,
    /// Prompt followed by every emitted token. The lookup searches this.
    context: Vec<i32>,
    generated_tokens: Vec<i32>,
    generation_stream: Option<UniquePtr<MlxThreadLocalStream>>,
    token_bias: TokenBiasMap,
    stats: PromptLookupStats,
}

impl PromptLookupGenerator {
    /// Create a generator. Caches are built from the model on every
    /// [`Self::generate`] call.
    pub fn new(config: PromptLookupConfig) -> Self {
        Self {
            caches: Vec::new(),
            config,
            kv_cache_mode: KVCacheMode::Fp16,
            context: Vec::new(),
            generated_tokens: Vec::new(),
            generation_stream: shared_thread_local_generation_stream(),
            token_bias: TokenBiasMap::default(),
            stats: PromptLookupStats::default(),
        }
    }

    /// Store the KV caches in `mode`, with the same Boundary-V policy plain
    /// decoding applies.
    pub fn with_kv_cache_mode(mut self, mode: KVCacheMode) -> Self {
        self.kv_cache_mode = mode;
        self
    }

    /// Attach a pre-resolved token bias, applied to every target sample.
    pub fn with_token_bias(mut self, bias: TokenBiasMap) -> Self {
        self.token_bias = bias;
        self
    }

    /// Acceptance accounting for the most recent [`Self::generate`] call.
    pub fn stats(&self) -> PromptLookupStats {
        self.stats
    }

    /// Prefill `prompt_tokens` and run one verify forward at every block
    /// width [`Self::generate`] can use (2 through `max_draft + 1`), so a
    /// later call does not pay first-use costs in the middle of its decode.
    ///
    /// The first forward at a width this process has not run yet builds or
    /// traces kernels for it, which on GB10 cost more than the rest of a
    /// short reply: a 72-token Qwen3-1.7B answer with two drafted rounds ran
    /// at 0.65x plain decoding without this warmup and at 1.07x with it. A
    /// warmup generation only reaches the widths its own proposals happen to
    /// use, which on a reply with nothing to copy is none of them.
    ///
    /// # Panics
    ///
    /// Same contract as [`Self::generate`] for the model and the prompt.
    pub fn warm_up_verify_widths<M: LanguageModel>(&mut self, model: &M, prompt_tokens: &[i32]) {
        self.reset(model);
        assert!(
            !self.caches.is_empty() && can_trim_prompt_cache(&self.caches),
            "prompt-lookup decoding requires trimmable external KV caches"
        );
        let last = *prompt_tokens
            .last()
            .expect("prompt-lookup warmup requires at least one prompt token");
        install_thread_local_default_stream(self.generation_stream.as_ref());
        let logits = prefill_prompt_last_logits(model, &mut self.caches, prompt_tokens);
        ffi::eval(&logits);
        for width in 2..=self.config.max_draft + 1 {
            let tokens = vec![last; width];
            let input = ffi::from_slice_i32(&tokens, &[1, width as i32]);
            let logits = model.forward(&input, &mut self.caches, None);
            let argmax = ffi::argmax_last_axis(&logits);
            ffi::eval(&argmax);
            trim_caches(&mut self.caches, width as i32);
        }
        ffi::clear_memory_cache();
    }

    /// Fresh caches from the model, in the configured mode, as
    /// `CxxGenerator::reset_with_model` builds them.
    fn reset<M: LanguageModel>(&mut self, model: &M) {
        model.reset_runtime_state();
        self.caches = model.make_caches();
        let modes = resolve_kv_cache_layer_modes(self.kv_cache_mode, self.caches.len());
        for (cache, mode) in self.caches.iter_mut().zip(modes) {
            cache.mode = mode;
        }
        model.set_kv_cache_layer_modes(resolve_kv_cache_layer_modes(
            self.kv_cache_mode,
            model.num_layers(),
        ));
        self.context.clear();
        self.generated_tokens.clear();
        self.stats = PromptLookupStats::default();
    }

    /// Same composition rule as `SpeculativeGenerator::compose_target_sampling`:
    /// the caller's own bias wins, otherwise the cached one is injected.
    fn compose_sampling<'a>(&self, sampling: &'a SamplingConfig) -> Cow<'a, SamplingConfig> {
        if self.token_bias.is_empty() || !sampling.token_bias.is_empty() {
            Cow::Borrowed(sampling)
        } else {
            let mut cloned = sampling.clone();
            cloned.token_bias = self.token_bias.clone();
            Cow::Owned(cloned)
        }
    }

    /// Generate up to `max_tokens` tokens after `prompt_tokens`.
    ///
    /// # Panics
    ///
    /// Panics on an empty prompt, on a model with no external KV caches or a
    /// non-trimmable one (callers must check [`supports_prompt_lookup`]
    /// first), and on a sampler that carries feedback state across tokens
    /// (mirostat, adaptive-p), which this loop does not thread through its
    /// rounds; the server's speculative burst declines the same samplers.
    /// `mlxcel generate` cannot configure either one.
    pub fn generate<M: LanguageModel>(
        &mut self,
        model: &M,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> (Vec<i32>, GenerationStats) {
        self.reset(model);

        // `can_trim_prompt_cache` is vacuously true on an empty slice, which
        // is exactly what a family that keeps its caches internally returns.
        assert!(
            !self.caches.is_empty() && can_trim_prompt_cache(&self.caches),
            "prompt-lookup decoding requires trimmable external KV caches"
        );
        assert!(
            !prompt_tokens.is_empty(),
            "prompt-lookup generate requires at least one prompt token"
        );

        let sampling_cow = self.compose_sampling(sampling);
        let sampling: &SamplingConfig = sampling_cow.as_ref();
        assert!(
            !sampling.needs_sampler_feedback_state(),
            "prompt-lookup decoding does not carry mirostat or adaptive-p sampler state"
        );
        seed_rng_if_needed(sampling);
        install_thread_local_default_stream(self.generation_stream.as_ref());

        let eos_tokens = merged_eos_token_ids(model.eos_token_ids(), &sampling.stop_token_ids);
        let needs_history = sampling.needs_token_history();
        let mut token_history = initial_token_history(prompt_tokens, needs_history);
        // Incremental penalty state, as `CxxGenerator` keeps it: byte-identical
        // logits to rebuilding the penalties from `token_history` on every
        // sample, without the per-sample pass over the whole history. History
        // only grows here (rejected proposals are never sampled into it), so
        // the state's append-only fast path always applies.
        let mut sampler_state: Option<SamplerState> = None;
        // Pure argmax with nothing that depends on the emitted prefix: every
        // verify position can be decided from one batched argmax and a single
        // host read. Anything else goes position by position through the
        // sampler, as the draft-model path does.
        let batched_argmax =
            sampling.is_greedy_path() && !needs_history && sampling.token_bias.is_empty();

        self.context.extend_from_slice(prompt_tokens);

        // PREFILL through the plain-decode routine, so the caches and the
        // first token are exactly what plain decoding produces.
        let prefill_start = Instant::now();
        let logits = prefill_prompt_last_logits(model, &mut self.caches, prompt_tokens);
        ffi::clear_memory_cache();
        let (first_arr, _) = if needs_history {
            sample_token_optimized_with_state(&logits, sampling, &token_history, &mut sampler_state)
        } else {
            sample_token_optimized(&logits, sampling, &token_history)
        };
        ffi::async_eval(&first_arr);
        // Index the prompt while the GPU finishes the prefill.
        let mut index = NgramIndex::new(&self.config);
        index.extend(&self.context);
        let first_token = ffi::item_i32(&first_arr);
        let prefill_time = prefill_start.elapsed();

        if eos_tokens.contains(&first_token) || max_tokens == 0 {
            return self.finish(prompt_tokens.len(), prefill_time, Default::default());
        }
        let mut done = self.emit(
            first_token,
            needs_history.then_some(&mut token_history),
            max_tokens,
            sampling,
        );
        if max_tokens > 1 {
            for cache in &mut self.caches {
                cache.prepare_turbo4_delegated_for_decode();
            }
        }

        // DECODE.
        let decode_start = Instant::now();
        let mut current_token = first_token;
        let mut governor = DraftGovernor::new(&self.config);
        // Rounds without a proposal are pipelined the way plain decoding is:
        // the next step is encoded and submitted from the still-lazy sample
        // before the host reads it, so the GPU does not idle while the host
        // reads a token and builds the next graph (measured 12% of decode time
        // on an M4 Pro). A sampler that reads the emitted history needs each
        // token on the host first, so those runs stay synchronous.
        let pipeline = !needs_history && std::env::var("MLXCEL_FORCE_SYNC").is_err();
        // `Some(next)` while a one-token forward of `current_token` is in
        // flight: the cache already holds `current_token` and `next` is the
        // lazy sample of the token after it. `None` means the cache holds
        // every emitted token except `current_token`.
        let mut in_flight: Option<UniquePtr<ffi::MlxArray>> = None;
        // Rounds in a row, this one included, whose lookup proposed nothing.
        let mut plain_streak = 0usize;
        // A proposal looked up while a `DraftPolicy::Gated` governor is
        // paused, waiting for decoding to show whether it comes true. Exactly
        // as long as settling reads, whatever `max_draft` is.
        let mut shadow: Option<ShadowProbe> = None;
        let shadow_config = PromptLookupConfig {
            max_draft: SHADOW_CONFIRM,
            ..self.config
        };

        while !done {
            if let Some(confirmed) = shadow
                .as_ref()
                .and_then(|probe| probe.settle(&self.context))
            {
                shadow = None;
                if confirmed {
                    governor.shadow_confirmed();
                    self.stats.shadow_confirmations += 1;
                }
            }
            // Never propose past `max_tokens`: every accepted proposal is
            // emitted, and the round also emits one target token.
            let remaining = max_tokens - self.generated_tokens.len();
            let budget = governor.budget().min(remaining.saturating_sub(1));
            let draft = if budget == 0 {
                // Paused: look up anyway and let the next tokens decoding
                // emits say whether proposing would have paid, at the cost of
                // a host-side hash probe instead of a verify forward.
                if shadow.is_none() && governor.probes_while_paused() {
                    index.extend(&self.context);
                    let proposal = index.find(&self.context, &shadow_config);
                    if proposal.len() >= SHADOW_CONFIRM {
                        shadow = Some(ShadowProbe {
                            start: self.context.len(),
                            proposal,
                        });
                    }
                }
                Vec::new()
            } else {
                index.extend(&self.context);
                let mut draft = index.find(&self.context, &self.config);
                draft.truncate(budget);
                draft
            };

            if draft.is_empty() {
                plain_streak += 1;
            } else {
                plain_streak = 0;
            }
            if let Some(next) = in_flight.take() {
                // Keep the pipeline full unless lookup has something to verify
                // or `next` is the last token allowed.
                if draft.is_empty() && remaining > 1 {
                    let next_input = ffi::reshape_token_for_forward(&next);
                    let logits = model.forward(&next_input, &mut self.caches, None);
                    let (after_next, _) = sample_token_optimized(&logits, sampling, &token_history);
                    ffi::async_eval(&after_next);
                    in_flight = Some(after_next);
                    self.count_plain_round(budget);
                }
                let token = ffi::item_i32(&next);
                if eos_tokens.contains(&token) {
                    break;
                }
                done = self.emit(token, None, max_tokens, sampling);
                current_token = token;
                // With nothing in flight now, the next round verifies from
                // `current_token` synchronously; the proposal found this round
                // was for the token before it and is looked up again there.
                continue;
            }

            if draft.is_empty()
                && pipeline
                && remaining > 1
                && (plain_streak > PLAIN_ROUNDS_BEFORE_PIPELINE || governor.probes_while_paused())
            {
                let input = ffi::from_slice_i32(&[current_token], &[1, 1]);
                let logits = model.forward(&input, &mut self.caches, None);
                let (next, _) = sample_token_optimized(&logits, sampling, &token_history);
                ffi::async_eval(&next);
                in_flight = Some(next);
                self.count_plain_round(budget);
                continue;
            }

            let mut verify_tokens = Vec::with_capacity(draft.len() + 1);
            verify_tokens.push(current_token);
            verify_tokens.extend_from_slice(&draft);
            let verify_len = verify_tokens.len();
            let verify_input = ffi::from_slice_i32(&verify_tokens, &[1, verify_len as i32]);
            let logits = model.forward(&verify_input, &mut self.caches, None);

            // Target token at every verify position, in order. Position `i`
            // is the target's choice for the slot draft[i] claims; position
            // `draft.len()` is the bonus token after a fully accepted block.
            let vocab = ffi::array_shape(&logits)[2];
            let greedy_targets = batched_argmax.then(|| {
                let argmax = ffi::argmax_last_axis(&logits);
                ffi::eval(&argmax);
                materialize_argmax_i32_vec(&argmax, verify_len)
            });

            let mut accepted = 0usize;
            for pos in 0..verify_len {
                let target = match &greedy_targets {
                    Some(targets) => targets[pos],
                    None => {
                        let pos_logits =
                            ffi::slice(&logits, &[0, pos as i32, 0], &[1, pos as i32 + 1, vocab]);
                        let (arr, _) = if needs_history {
                            sample_token_optimized_with_state(
                                &pos_logits,
                                sampling,
                                &token_history,
                                &mut sampler_state,
                            )
                        } else {
                            sample_token_optimized(&pos_logits, sampling, &token_history)
                        };
                        ffi::eval(&arr);
                        ffi::item_i32(&arr)
                    }
                };

                if eos_tokens.contains(&target) {
                    done = true;
                    break;
                }
                done = self.emit(
                    target,
                    needs_history.then_some(&mut token_history),
                    max_tokens,
                    sampling,
                );
                current_token = target;
                if done {
                    break;
                }
                // `target` is emitted either way; it only extends the round
                // when it confirms the proposal in the same slot.
                if pos < draft.len() && target == draft[pos] {
                    accepted += 1;
                } else {
                    break;
                }
            }

            if draft.is_empty() {
                self.count_plain_round(budget);
            } else {
                self.stats.rounds += 1;
                self.stats.drafted_rounds += 1;
                governor.record(accepted);
            }
            self.stats.proposed_draft_tokens += draft.len();
            self.stats.accepted_draft_tokens += accepted;

            // The verify forward appended `current, d_0..d_{k-1}`. The cache
            // must end holding every emitted token except the next round's
            // `current_token`, i.e. through `d_{a-1}`: trim `k - a`.
            trim_caches(&mut self.caches, (draft.len() - accepted) as i32);

            if crate::memory::should_clear_cache_at(
                self.generated_tokens.len(),
                crate::memory::cache_clear_interval(),
            ) {
                ffi::clear_memory_cache();
            }
        }

        let decode_time = decode_start.elapsed();
        tracing::info!(
            rounds = self.stats.rounds,
            drafted_rounds = self.stats.drafted_rounds,
            proposed_draft_tokens = self.stats.proposed_draft_tokens,
            accepted_draft_tokens = self.stats.accepted_draft_tokens,
            shadow_confirmations = self.stats.shadow_confirmations,
            generated_tokens = self.generated_tokens.len(),
            "prompt-lookup decode finished"
        );
        self.finish(prompt_tokens.len(), prefill_time, decode_time)
    }

    /// Count a one-token forward. `budget == 0` means the governor paused
    /// proposals (or `max_tokens` left room for none) rather than lookup
    /// finding no match.
    fn count_plain_round(&mut self, budget: usize) {
        self.stats.rounds += 1;
        if budget == 0 {
            self.stats.paused_rounds += 1;
        }
    }

    /// Emit `token` and report whether generation stops after it: at
    /// `max_tokens`, or when the repetition-loop guard fires, which plain
    /// decoding checks after every token too.
    fn emit(
        &mut self,
        token: i32,
        history: Option<&mut Vec<i32>>,
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> bool {
        self.generated_tokens.push(token);
        self.context.push(token);
        if let Some(history) = history {
            history.push(token);
        }
        self.generated_tokens.len() >= max_tokens
            || detect_repetition_loop(&self.generated_tokens, &sampling.loop_detection)
    }

    fn finish(
        &self,
        prompt_count: usize,
        prefill_time: std::time::Duration,
        decode_time: std::time::Duration,
    ) -> (Vec<i32>, GenerationStats) {
        let prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        let decode_ms = decode_time.as_secs_f64() * 1000.0;
        let gen_count = self.generated_tokens.len();
        // The first token comes out of the prefill forward; the decode rate
        // counts only what the decode loop produced in `decode_time`.
        let decode_count = gen_count.saturating_sub(1);
        let stats = GenerationStats {
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
                decode_count as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
        };
        (self.generated_tokens.clone(), stats)
    }
}

#[cfg(test)]
#[path = "prompt_lookup_tests.rs"]
mod tests;
