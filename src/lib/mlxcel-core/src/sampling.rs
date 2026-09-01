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

//! Sampling and token-penalty helpers for generation.
//!
//! `generate.rs` and `speculative.rs` both rely on the same penalty pipeline.
//! Keeping those helpers here isolates the token-selection policy from the
//! pipelined decode loops and makes low-level sampling invariants easier to
//! test without touching model forward math.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::generate::SamplingConfig;
use cxx::UniquePtr;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// B9 — Observability: global Prometheus-compatible counters
//
// Process-wide atomics so the `/metrics` HTTP handler can read them from
// `mlxcel_core::sampling` without threading an extra struct through every
// call site.  All accesses use `Ordering::Relaxed`: exactness is not
// required for monitoring — slight staleness is acceptable and avoids
// unnecessary memory barriers on the hot decode path.
// ---------------------------------------------------------------------------

/// Total sampling calls where `token_bias` was non-empty.
///
/// Exposed via `/metrics` as `mlxcel_lang_bias_applied_total`.
/// Incremented once per `sample_token_optimized` call with a non-empty
/// `TokenBiasMap`; zero overhead when the map is empty (baseline path).
pub static LANG_BIAS_APPLIED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Total sampling calls where the pre-bias top-1 token was `-inf`-suppressed.
///
/// Exposed via `/metrics` as `mlxcel_lang_bias_tokens_suppressed_total`.
/// Incremented when the argmax of the original (pre-bias) logits is a token
/// that has `f32::NEG_INFINITY` bias in the map — signalling the bias
/// actively overrode the model's most probable token at that step.
pub static LANG_BIAS_TOKENS_SUPPRESSED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Total sampling calls where the pre-bias top-1 token was both
/// `-inf`-suppressed AND was a byte-fragment entry.
///
/// Exposed via `/metrics` as
/// `mlxcel_lang_bias_byte_fragment_suppressions_total`. This counter is a
/// strict subset of `LANG_BIAS_TOKENS_SUPPRESSED_TOTAL`: it increments only
/// when the suppressed token was classified via UTF-8 start-byte analysis and
/// participated in the bias decision. Operators use the counter to observe
/// how much of their suppression traffic comes from byte-fragment entries
/// versus merged whole-character tokens, which matters because start-byte
/// classification is an approximation and over-suppression is possible.
///
/// Populated via the bias-metadata channel wired in
/// `apply_token_bias` below.
pub static LANG_BIAS_BYTE_FRAGMENT_SUPPRESSIONS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Read the current value of `mlxcel_lang_bias_applied_total`.
#[inline]
pub fn lang_bias_applied_total() -> u64 {
    LANG_BIAS_APPLIED_TOTAL.load(Ordering::Relaxed)
}

/// Read the current value of `mlxcel_lang_bias_tokens_suppressed_total`.
#[inline]
pub fn lang_bias_tokens_suppressed_total() -> u64 {
    LANG_BIAS_TOKENS_SUPPRESSED_TOTAL.load(Ordering::Relaxed)
}

/// Read the current value of
/// `mlxcel_lang_bias_byte_fragment_suppressions_total`.
#[inline]
pub fn lang_bias_byte_fragment_suppressions_total() -> u64 {
    LANG_BIAS_BYTE_FRAGMENT_SUPPRESSIONS_TOTAL.load(Ordering::Relaxed)
}

/// Additive bias applied to specific token logits before any history-based penalty.
///
/// A positive bias makes a token more likely; a negative bias makes it less likely.
/// Use `f32::NEG_INFINITY` to permanently suppress a token (probability becomes 0).
///
/// When empty, `apply_token_bias` short-circuits without any array operations,
/// preserving bit-exact baseline behavior.
///
/// tokens that were classified via byte-fragment UTF-8 start-byte
/// analysis are tracked in a separate set so the observability path can count
/// how many suppressions originated from that opt-in classifier.
#[derive(Debug, Clone, Default)]
pub struct TokenBiasMap {
    entries: HashMap<i32, f32>,
    /// Token ids that were tagged as byte-fragment entries during vocab scan.
    /// Populated by `TokenLanguageIndex::to_token_bias` when
    /// `ExceptionConfig::include_byte_fragments` is enabled.
    byte_fragment_ids: std::collections::HashSet<i32>,
}

impl TokenBiasMap {
    /// Create an empty `TokenBiasMap`.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            byte_fragment_ids: std::collections::HashSet::new(),
        }
    }

    /// Insert or overwrite the bias for `token_id`.
    ///
    /// Negative token ids and ids outside the vocabulary range are accepted here
    /// but silently ignored when the bias is applied (see `apply_token_bias`).
    pub fn insert(&mut self, token_id: i32, bias: f32) {
        self.entries.insert(token_id, bias);
    }

    /// Insert a bias and tag the token as a byte-fragment entry.
    ///
    /// Used by `TokenLanguageIndex::to_token_bias` when the opt-in
    /// `include_byte_fragments` flag is set. Equivalent to [`Self::insert`]
    /// plus a side-channel annotation for the observability counter.
    pub fn insert_byte_fragment(&mut self, token_id: i32, bias: f32) {
        self.entries.insert(token_id, bias);
        self.byte_fragment_ids.insert(token_id);
    }

    /// Permanently suppress every token id in `ids` by forcing its bias to
    /// `f32::NEG_INFINITY` (sampled probability becomes 0).
    ///
    /// Suppression always wins: an existing finite bias for the same id is
    /// overwritten. This is the mechanism the generation paths use to mask a
    /// model's reserved output-illegal tokens (multimodal placeholder ids,
    /// issue #350) so they can never become the argmax at a near-tie decode
    /// step. Negative or out-of-range ids are stored but ignored when the
    /// bias is applied (see [`apply_token_bias`]).
    ///
    /// An empty slice is a no-op, so a non-multimodal model (whose
    /// suppressed set is empty) keeps the bit-exact zero-overhead baseline:
    /// the map stays empty and `apply_token_bias` short-circuits.
    ///
    /// Used by: CLI `generate` (`run_generation_mode`) and the server batch
    /// scheduler (`enqueue_request`).
    pub fn suppress_tokens(&mut self, ids: &[i32]) {
        for &id in ids {
            self.entries.insert(id, f32::NEG_INFINITY);
        }
    }

    /// Returns `true` when no bias entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of bias entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when a bias entry exists for `token_id`.
    ///
    /// Used by: `lang_analyzer::TokenLanguageIndex::to_token_bias` (B5) for
    /// first-language-wins conflict resolution.
    pub fn contains(&self, token_id: i32) -> bool {
        self.entries.contains_key(&token_id)
    }

    /// Returns the bias for `token_id`, or `None` if not present.
    ///
    /// Used by B9 observability to check whether the pre-bias argmax token
    /// was `-inf`-suppressed.
    pub fn get(&self, token_id: &i32) -> Option<&f32> {
        self.entries.get(token_id)
    }

    /// Returns `true` when `token_id` was tagged as a byte-fragment entry.
    pub fn is_byte_fragment(&self, token_id: i32) -> bool {
        self.byte_fragment_ids.contains(&token_id)
    }

    /// Number of byte-fragment entries currently in the map.
    ///
    /// Used by the tracing debug field `byte_fragment_entries` emitted
    /// alongside the B9 `lang_bias resolved` event.
    pub fn byte_fragment_len(&self) -> usize {
        self.byte_fragment_ids.len()
    }

    /// Iterate over `(&token_id, &bias)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&i32, &f32)> {
        self.entries.iter()
    }
}

/// Apply additive bias to token logits before repetition/frequency/presence penalties.
///
/// Zero-overhead when `bias.is_empty()`: returns a copy of the input without
/// any array arithmetic.
///
/// Invalid token ids (negative, or `>= vocab_size`) are silently ignored —
/// no panic, no error.
///
/// Used by: standard generation, speculative decoding, batch scheduler, MTP
/// verify (Gemma4 target adapter)
impl TokenBiasMap {
    /// Add `bias` onto the entry for `token_id`, creating it at zero first
    /// (#1485). b10621's logit-bias sampler applies every `(token, bias)`
    /// entry additively, so a token named twice (say, once by id and once
    /// through a string key that tokenizes to it) accumulates both biases;
    /// a plain insert would silently keep only the last. `-inf` saturates
    /// (`-inf + x == -inf`), so a ban composes with any further bias.
    pub fn accumulate(&mut self, token_id: i32, bias: f32) {
        let entry = self.entries.entry(token_id).or_insert(0.0);
        *entry += bias;
    }
}

pub fn apply_token_bias(logits: &MlxArray, bias: &TokenBiasMap) -> UniquePtr<MlxArray> {
    if bias.is_empty() {
        return ffi::copy(logits);
    }
    let shape = ffi::array_shape(logits);
    let vocab_size = *shape.last().unwrap() as usize;
    let mut bias_vec = vec![0.0f32; vocab_size];
    for (&tok, &b) in bias.iter() {
        if tok >= 0 && (tok as usize) < vocab_size {
            bias_vec[tok as usize] = b;
        }
    }
    let bias_arr = ffi::from_slice_f32(&bias_vec, &[1, vocab_size as i32]);
    let bias_broadcast = ffi::broadcast_to(&bias_arr, &shape);
    ffi::add(logits, &bias_broadcast)
}

/// Optimized sampling that returns arrays for pipelining.
///
/// Returns `(token_array, logits_array)` without forcing evaluation so the
/// caller can preserve async lookahead pipelining.
///
/// Uses fused C++ sampling in a single FFI call to minimize round-trip
/// overhead. Chain order (#1379, matching llama-server): top-k, top-p, and
/// min-p evaluate on the untempered distribution, XTC on the renormalised
/// filtered row, and the single temperature scaling comes last, applied only
/// to the draw.
///
/// **B9 observability**: when `config.token_bias` is non-empty this function
/// increments `LANG_BIAS_APPLIED_TOTAL` and, when the pre-bias top-1 token
/// was `-inf`-suppressed, `LANG_BIAS_TOKENS_SUPPRESSED_TOTAL`.  Both
/// increments are skipped entirely when the map is empty, preserving the
/// zero-overhead baseline path.
///
/// Used by: `CxxGenerator`, `SpeculativeGenerator`, `BatchScheduler`
pub fn sample_token_optimized(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    sample_token_optimized_core(logits, config, token_history, None)
}

/// Incremental-state variant of [`sample_token_optimized`].
///
/// Sampling behavior is identical, but the repetition and frequency/presence
/// penalties read from a per-sequence [`SamplerState`] that is maintained
/// incrementally instead of being rebuilt from `token_history` on every call.
/// The state is created lazily the first time a repetition/frequency/presence
/// penalty is active, so a config with none of those (the default no-penalty
/// path, and DRY-only configs) never allocates it and `state` stays `None`.
///
/// `token_history` is still passed: the deferred DRY path consumes it directly,
/// and the [`SamplerState`] synchronizes itself to it on entry (an append-only
/// fast path absorbs only the newly appended tail; a shorter or diverged
/// history triggers a rebuild, which keeps trim/restore correct without any
/// explicit reset).
///
/// Produces byte-identical logits to [`sample_token_optimized`] for the same
/// history, so penalty-adjusted greedy sampling selects identical token ids.
///
/// Used by: `BatchScheduler` decode steps, `CxxGenerator` decode loops
pub fn sample_token_optimized_with_state(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
    state: &mut Option<SamplerState>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // The incremental penalty state only serves the full-history window
    // (`penalty_last_n < 0`); a positive window rebuilds over its bounded
    // slice each step and a zero window disables the stage, so neither
    // allocates state (#1436). The #1485 feedback samplers (mirostat,
    // adaptive-p) also allocate here, carrying their `mu` / EMA across
    // steps.
    ensure_sampler_state(config, state);
    sample_token_optimized_core(logits, config, token_history, state.as_mut())
}

/// Shared implementation for [`sample_token_optimized`] (`state == None`) and
/// [`sample_token_optimized_with_state`] (`state == Some`).
///
/// With `state == None` every penalty takes the rebuild-every-token path, so
/// the output is bit-for-bit identical to the pre-incremental implementation.
/// The no-penalty baseline path is unchanged regardless of `state`: an empty
/// `token_history` skips every penalty block, and the only added work is one
/// already-cheap `Option` check.
fn sample_token_optimized_core(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
    state: Option<&mut SamplerState>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let (token, adjusted, _probs) =
        sample_token_optimized_core_full(logits, config, token_history, state, false);
    (token, adjusted)
}

/// Full-form core sampler shared by every entry point: samples one token and,
/// when `want_distribution` is set, also returns the float32 `[1, vocab]`
/// post-chain distribution the draw came from (b10621's `post_sampling_probs`
/// view). Routes to one of three paths (#1485):
///
/// - **mirostat** (`effective_mirostat() != 0`): b10621 replaces the whole
///   chain, so this bypasses penalties, DRY, and every truncation filter and
///   runs token-bias -> temperature -> mirostat-select with the per-sequence
///   `mu` from `state` ([`mirostat_sample_core`]).
/// - **extended chain** (`needs_extended_chain()`): dynamic temperature, a
///   `min_keep` floor, or adaptive-p require arithmetic the fused C++ chain
///   has no parameters for, so the whole b10621 filter order runs as Rust
///   graph ops ([`apply_extended_chain`]) ending in a plain categorical (or
///   the adaptive-p transform draw).
/// - **fused** (everything else): the pre-#1485 path, byte-identical.
fn sample_token_optimized_core_full(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
    mut state: Option<&mut SamplerState>,
    want_distribution: bool,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    Option<UniquePtr<MlxArray>>,
) {
    if config.effective_mirostat() != 0 {
        return mirostat_sample_core(logits, config, state, want_distribution);
    }

    if config.needs_extended_chain() {
        let adjusted =
            preprocess_penalty_stages(logits, config, token_history, state.as_deref_mut());
        let gate = xtc_gate_draw(config);
        let filtered = apply_extended_chain(&adjusted, config, gate.as_deref());
        let (token, probs) = if config.effective_adaptive_target() >= 0.0 {
            adaptive_p_sample(&filtered, config, state, want_distribution)
        } else {
            let token = ffi::fused_sample(&filtered, 1.0, 0, 1.0, 0.0);
            let probs = want_distribution.then(|| ffi::softmax(&filtered, -1));
            (token, probs)
        };
        crate::sampling_dispatch::report_sampling_dispatch();
        return (token, adjusted, probs);
    }

    let last_logits = preprocess_logits_for_sampling(logits, config, token_history, state);

    let gate = xtc_gate_draw(config);
    let token = fused_sample_dispatch(&last_logits, config, gate.as_deref());
    // Announce a newly-seen dispatch outcome at INFO. Costs one `u32` load per
    // step in steady state; see `sampling_dispatch` for why this is not `debug`.
    crate::sampling_dispatch::report_sampling_dispatch();
    let probs = want_distribution
        .then(|| fused_sample_probs_dispatch(&last_logits, config, gate.as_deref()));
    (token, last_logits, probs)
}

/// Stateful sampling with the post-chain distribution (#1485): the
/// [`sample_token_optimized_with_state`] behavior plus the float32
/// `[1, vocab]` distribution the draw came from, sharing one XTC gate draw
/// between the token and the distribution so both see the same gate outcome.
/// This is what the native `/completion` route's `post_sampling_probs` view
/// reports.
///
/// Used by: `BatchScheduler` decode steps for `post_sampling_probs` requests
pub fn sample_token_with_state_and_distribution(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
    state: &mut Option<SamplerState>,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    ensure_sampler_state(config, state);
    let (token, adjusted, probs) =
        sample_token_optimized_core_full(logits, config, token_history, state.as_mut(), true);
    let probs = probs.expect("want_distribution returns a distribution on every path");
    (token, adjusted, probs)
}

/// Create the per-sequence [`SamplerState`] on first use when `config` needs
/// one: an incremental full-history penalty state (the pre-#1485 condition),
/// or the #1485 sampler feedback state (mirostat `mu`, adaptive-p EMA).
fn ensure_sampler_state(config: &SamplingConfig, state: &mut Option<SamplerState>) {
    if state.is_none()
        && ((config.penalty_last_n < 0
            && (config.repetition_penalty != 1.0
                || config.frequency_penalty != 0.0
                || config.presence_penalty != 0.0))
            || config.needs_sampler_feedback_state())
    {
        *state = Some(SamplerState::for_config(config));
    }
}

/// One lazy `[1]` f32 uniform draw for the XTC gate, or `None` when XTC is
/// disabled (`xtc_probability <= 0.0`), in which case the per-request RNG
/// stream is not advanced and every existing request's token stream stays
/// byte-identical to before XTC existed.
///
/// The draw itself is the same one `apply_xtc_step` made before #1379 moved
/// XTC into the C++ chain: it comes from the thread-local default MLX key
/// sequence, split at graph-construction time BEFORE the categorical draw
/// inside the fused sampler, so a fixed seed reproduces the same gate outcome
/// followed by the same categorical sample. The comparison against
/// `xtc_probability` happens lazily inside the C++ chain, so nothing here
/// synchronizes.
///
/// Used by: [`sample_token_optimized_core`], [`effective_token_distribution`],
/// [`sample_token_with_distribution`]
fn xtc_gate_draw(config: &SamplingConfig) -> Option<UniquePtr<MlxArray>> {
    if config.xtc_probability > 0.0 {
        // SAFETY: `key` is documented to accept a null pointer, meaning "draw
        // from the current thread-local default RNG state" (mirrors the
        // existing `std::ptr::null()` "no explicit key" usage in `layers.rs`).
        Some(unsafe { ffi::random_uniform(0.0, 1.0, &[1], dtype::FLOAT32, std::ptr::null()) })
    } else {
        None
    }
}

/// Dispatch one sampling draw to [`ffi::fused_sample`], or to
/// [`ffi::fused_sample_xtc`] when an XTC gate draw is present. The XTC-less
/// call is byte-identical to the pre-#1379 call, so XTC-disabled configs pay
/// nothing.
fn fused_sample_dispatch(
    logits: &MlxArray,
    config: &SamplingConfig,
    gate: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    match gate {
        Some(gate) => ffi::fused_sample_xtc(
            logits,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            config.xtc_threshold,
            config.xtc_probability,
            &config.xtc_special_token_ids,
            gate,
        ),
        None => ffi::fused_sample(
            logits,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
        ),
    }
}

/// [`fused_sample_dispatch`]'s counterpart for the reported distribution
/// (#902): the same routing, into [`ffi::fused_sample_probs`] or
/// [`ffi::fused_sample_probs_xtc`]. Handing it the same `gate` array a
/// [`fused_sample_dispatch`] call used makes the reported distribution the
/// one that draw actually came from.
fn fused_sample_probs_dispatch(
    logits: &MlxArray,
    config: &SamplingConfig,
    gate: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    match gate {
        Some(gate) => ffi::fused_sample_probs_xtc(
            logits,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
            config.xtc_threshold,
            config.xtc_probability,
            &config.xtc_special_token_ids,
            gate,
        ),
        None => ffi::fused_sample_probs(
            logits,
            config.temperature,
            config.top_k,
            config.top_p,
            config.min_p,
        ),
    }
}

/// The slice of `token_history` the repetition / frequency / presence
/// penalties operate on, under b10621's `repeat_last_n` semantics (#1436):
/// `0` returns the empty slice (stage disabled), a negative value returns
/// the full history (mlxcel's pre-#1436 behavior and the CLI default), and
/// `N > 0` returns the last `N` tokens.
///
/// Used by: [`preprocess_logits_for_sampling`]
fn penalty_window(token_history: &[i32], penalty_last_n: i32) -> &[i32] {
    if penalty_last_n == 0 {
        &[]
    } else if penalty_last_n < 0 {
        token_history
    } else {
        let start = token_history.len().saturating_sub(penalty_last_n as usize);
        &token_history[start..]
    }
}

/// Everything [`sample_token_optimized`] does to the logits *before* the fused
/// top-k / top-p / min-p / XTC / temperature sampler: last-position slice,
/// token bias, repetition penalty, DRY, frequency/presence penalty. XTC is no
/// longer applied here: since #1379 it lives inside the C++ chain, after
/// min-p, on the renormalised filtered row (see [`fused_sample_dispatch`]).
///
/// Split out so the speculative acceptance path (issue #902) can obtain the
/// exact same pre-sampler logits and hand them to [`ffi::fused_sample_probs`],
/// which then applies the identical filter chain the sampler itself runs. The
/// effective categorical distribution is therefore derived from the sampler's
/// own code rather than reconstructed alongside it.
///
/// Used by: [`sample_token_optimized_core`], [`effective_token_distribution`],
/// [`sample_token_with_distribution`]
fn preprocess_logits_for_sampling(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
    state: Option<&mut SamplerState>,
) -> UniquePtr<MlxArray> {
    let last_logits = preprocess_penalty_stages(logits, config, token_history, state);

    // Row filters (top-n-sigma and typical_p; #1373 adds p_less) run on the
    // penalized, untempered logits: after every history-based penalty and
    // before the fused C++ chain (temperature / top_k / top_p / min_p /
    // XTC). Disabled filters return `last_logits` unchanged with no new
    // graph nodes, preserving the bit-exact baseline.
    apply_row_filters(last_logits, &FusedSampleParams::from_config(config))
}

/// The token-bias stage shared by [`preprocess_penalty_stages`] and the
/// mirostat bypass path (#1485): applies [`SamplingConfig::token_bias`] with
/// the B9 observability counters, or returns the input unchanged (no new
/// graph nodes) when the map is empty.
fn apply_token_bias_stage(
    last_logits: UniquePtr<MlxArray>,
    config: &SamplingConfig,
) -> UniquePtr<MlxArray> {
    if config.token_bias.is_empty() {
        return last_logits;
    }
    // B9 — increment applied counter (zero overhead when bias is empty).
    LANG_BIAS_APPLIED_TOTAL.fetch_add(1, Ordering::Relaxed);

    // B9 — check if the pre-bias argmax token was `-inf`-suppressed.
    // Evaluation is required to extract the integer id; the argmax is a
    // lightweight reduction over the last logits slice already in memory.
    let top_arr = ffi::argmax_last_axis(&last_logits);
    ffi::eval(&top_arr);
    let top_id = ffi::item_i32(&top_arr);
    if config
        .token_bias
        .get(&top_id)
        .is_some_and(|b| b.is_infinite() && b.is_sign_negative())
    {
        LANG_BIAS_TOKENS_SUPPRESSED_TOTAL.fetch_add(1, Ordering::Relaxed);
        // separate counter for suppressions that originated
        // from the opt-in byte-fragment classifier. Strict subset of the
        // total-suppressed counter above.
        if config.token_bias.is_byte_fragment(top_id) {
            LANG_BIAS_BYTE_FRAGMENT_SUPPRESSIONS_TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }

    apply_token_bias(&last_logits, &config.token_bias)
}

/// The bias/penalty half of [`preprocess_logits_for_sampling`]: last-position
/// slice, token bias, repetition penalty, DRY, frequency/presence penalty,
/// WITHOUT the row filters. The extended chain (#1485) consumes this
/// directly, because its filters need `min_keep` parameters the shared
/// [`apply_row_filters`] hook deliberately does not carry.
fn preprocess_penalty_stages(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
    mut state: Option<&mut SamplerState>,
) -> UniquePtr<MlxArray> {
    // Use optimized slice_last_logits: [batch, seq, vocab] -> [batch, vocab].
    let last_logits = ffi::slice_last_logits(logits);

    // Apply token bias first (before history-based penalties).
    // Language policy is an external decision, not history-based, so it takes
    // precedence. -inf composes correctly with downstream penalties:
    //   -inf × k == -inf,  -inf + f == -inf.
    let last_logits = apply_token_bias_stage(last_logits, config);

    // Synchronize the incremental state to the current history once, before any
    // penalty reads it. No-op when `state` is `None`.
    if let Some(s) = &mut state {
        s.sync(token_history);
    }

    // b10621 `repeat_last_n` window (#1436): the repetition and
    // frequency/presence penalties see only this slice. The incremental
    // `SamplerState` maintains full-history aggregates, so it serves only the
    // full-history form (`penalty_last_n < 0`, the pre-#1436 behavior it was
    // built for, byte-identical); a positive window takes the
    // rebuild-every-step path over at most `penalty_last_n` tokens, which is
    // bounded and cheap (the b10621 default window is 64).
    let penalty_history = penalty_window(token_history, config.penalty_last_n);
    let windowed = config.penalty_last_n > 0;

    let last_logits = if config.repetition_penalty != 1.0 && !penalty_history.is_empty() {
        match &mut state {
            Some(s) if !windowed => s.apply_repetition(&last_logits, config.repetition_penalty),
            _ => apply_repetition_penalty(&last_logits, penalty_history, config.repetition_penalty),
        }
    } else {
        last_logits
    };

    let last_logits = if config.dry_multiplier > 0.0
        && config.dry_penalty_last_n != 0
        && !token_history.is_empty()
    {
        apply_dry_penalty(&last_logits, token_history, config)
    } else {
        last_logits
    };

    if (config.frequency_penalty != 0.0 || config.presence_penalty != 0.0)
        && !penalty_history.is_empty()
    {
        match &mut state {
            Some(s) if !windowed => s.apply_frequency_presence(
                &last_logits,
                config.frequency_penalty,
                config.presence_penalty,
            ),
            _ => apply_frequency_presence_penalty(
                &last_logits,
                penalty_history,
                config.frequency_penalty,
                config.presence_penalty,
            ),
        }
    } else {
        last_logits
    }
}

/// The exact categorical distribution [`sample_token_optimized`] would draw
/// from for `logits` under `config`, as a float32 `[batch, vocab]`
/// row-normalized probability tensor.
///
/// This is the `p` and `q` of modified rejection sampling (issue #902). It is
/// built from the sampler's own pre-steps ([`preprocess_logits_for_sampling`])
/// and the sampler's own filter chain ([`ffi::fused_sample_probs`]), so a
/// change to either automatically moves the distribution with it.
///
/// A greedy config (`temperature == 0.0` or `top_k == 1`) yields the one-hot
/// indicator at the argmax, which is the correct degenerate proposal
/// distribution for a greedily-proposing drafter.
///
/// Consumes no randomness while XTC is disabled: safe to call without
/// perturbing the token stream of any other sampler on the same RNG key
/// sequence. An XTC-active config consumes exactly one uniform for the XTC
/// gate, the same one draw the pre-#1379 Rust XTC pre-step consumed.
///
/// Used by: `speculative::stochastic_accept`
pub fn effective_token_distribution(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
) -> UniquePtr<MlxArray> {
    debug_assert!(
        config.effective_mirostat() == 0 && !config.needs_extended_chain(),
        "speculative distribution paths exclude the #1485 feedback/extended sampler configs; admission must gate them out"
    );
    let processed = preprocess_logits_for_sampling(logits, config, token_history, None);
    let gate = xtc_gate_draw(config);
    fused_sample_probs_dispatch(&processed, config, gate.as_deref())
}

/// Sample one token *and* return the distribution it was drawn from.
///
/// Equivalent to [`sample_token_optimized`] followed by
/// [`effective_token_distribution`], but the (potentially expensive) bias and
/// penalty pre-steps run once instead of twice. The returned token is the same
/// array [`sample_token_optimized`] returns for the same inputs and RNG state:
/// the extra distribution tensor is a pure function of the pre-sampler logits
/// and draws nothing from the RNG stream.
///
/// Returns `(token, probs)` where `probs` is float32 `[batch, vocab]`.
///
/// Used by: `SpeculativeGenerator` draft loop (issue #902)
pub fn sample_token_with_distribution(
    logits: &MlxArray,
    config: &SamplingConfig,
    token_history: &[i32],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    debug_assert!(
        config.effective_mirostat() == 0 && !config.needs_extended_chain(),
        "speculative distribution paths exclude the #1485 feedback/extended sampler configs; admission must gate them out"
    );
    let processed = preprocess_logits_for_sampling(logits, config, token_history, None);
    // One gate draw for BOTH calls: the token and the reported distribution
    // must see the same XTC gate outcome, and only one uniform may leave the
    // per-request RNG stream per sampling step.
    let gate = xtc_gate_draw(config);
    let token = fused_sample_dispatch(&processed, config, gate.as_deref());
    // Announce a newly-seen dispatch outcome at INFO. Costs one `u32` load per
    // step in steady state; see `sampling_dispatch` for why this is not `debug`.
    crate::sampling_dispatch::report_sampling_dispatch();
    let probs = fused_sample_probs_dispatch(&processed, config, gate.as_deref());
    (token, probs)
}

/// Batch-parallel sampling: sample one token per sequence from batched logits.
///
/// `logits` has shape `[B, 1, vocab_size]`. Each sequence is sampled
/// independently using its own `SamplingConfig` and token history.
///
/// Returns a vector of B sampled token IDs.
///
/// Available for callers that need standalone batched sampling without
/// per-sequence state interleaving. The BatchScheduler currently inlines
/// equivalent logic to interleave sampling with EOS/state/streaming updates.
///
/// When every row shares one fused-eligible scalar config (see
/// [`uniform_fused_batch_params`]) the whole batch is sampled in a single
/// [`batched_fused_sample`] dispatch instead of the per-row loop. On the
/// no-filter stochastic path that dispatch is the batch-wide Gumbel-max kernel
/// (issue #900), which covers all `B` rows in one launch (grid.z = B); greedy
/// stays on the row-independent `argmax` and is byte-identical either way.
pub fn batched_sample(
    logits: &MlxArray,
    configs: &[&SamplingConfig],
    token_histories: &[&[i32]],
) -> Vec<i32> {
    let b = configs.len();
    debug_assert_eq!(b, token_histories.len());

    // Batch-wide single dispatch when the whole batch is uniform and needs no
    // per-row logit edits.
    if let Some(params) = uniform_fused_batch_params(configs) {
        return batched_fused_sample(logits, &params);
    }

    let mut tokens = Vec::with_capacity(b);
    for i in 0..b {
        // Slice [B, 1, vocab] -> [1, 1, vocab] for sequence i
        let seq_logits = ffi::slice(logits, &[i as i32, 0, 0], &[i as i32 + 1, 1, i32::MAX]);
        let (token_arr, _logprobs) =
            sample_token_optimized(&seq_logits, configs[i], token_histories[i]);
        ffi::eval(&token_arr);
        tokens.push(ffi::item_i32(&token_arr));
    }
    tokens
}

/// Shared scalar params for a batch, or `None` when the batch cannot take the
/// single-dispatch fused path.
///
/// Every row must be fused-eligible ([`config_supports_fused_batch`]: no
/// history-based penalty, no token bias, no XTC) and carry bit-identical
/// [`FusedSampleParams`]. Any divergence sends the whole batch back to the
/// per-row loop, which is the only place per-row logit edits can happen.
///
/// An empty batch returns `None` so the caller's loop returns an empty vector
/// instead of dispatching a zero-row kernel.
///
/// Used by: [`batched_sample`]
fn uniform_fused_batch_params(configs: &[&SamplingConfig]) -> Option<FusedSampleParams> {
    let first = configs.first()?;
    if !config_supports_fused_batch(first) {
        return None;
    }
    let params = FusedSampleParams::from_config(first);
    for config in &configs[1..] {
        if !config_supports_fused_batch(config) {
            return None;
        }
        if !params.matches(&FusedSampleParams::from_config(config)) {
            return None;
        }
    }
    Some(params)
}

/// Scalar sampling parameters consumed by [`ffi::fused_sample`].
///
/// Once [`config_supports_fused_batch`] has ruled out per-row penalties and
/// token bias, these four `Copy` fields are the entire sampler state the
/// batched fast path needs. Carrying them on their own lets the batch
/// scheduler gate compare rows and dispatch without cloning a full
/// [`SamplingConfig`] (with its penalty `Vec`s and bias maps) on every fused
/// decode step.
///
/// Used by: `BatchScheduler::execute_batched_decode` fast-path gate
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedSampleParams {
    /// Sampling temperature (`0.0` selects the greedy argmax path).
    pub temperature: f32,
    /// Top-k cutoff (`0` disables; `1` selects the greedy argmax path).
    pub top_k: i32,
    /// Top-p (nucleus) cutoff (`1.0` disables).
    pub top_p: f32,
    /// Min-p cutoff (`0.0` disables).
    pub min_p: f32,
    /// Top-n-sigma logit filter (`0.0` disables). Row-wise, history-free and
    /// RNG-free, so it stays fused-eligible: [`apply_row_filters`] applies it
    /// to the whole `[B, V]` batch before the single fused dispatch.
    pub top_n_sigma: f32,
    /// Locally typical sampling cutoff (`1.0` disables). Row-wise,
    /// history-free and RNG-free like `top_n_sigma`, and applied by the same
    /// [`apply_row_filters`] hook before the single fused dispatch.
    pub typical_p: f32,
}

impl FusedSampleParams {
    /// Extract the fused scalar params from a full [`SamplingConfig`].
    ///
    /// `top_n_sigma` is normalized through
    /// [`SamplingConfig::effective_top_n_sigma`]: a value the sampler would
    /// skip anyway (greedy config, non-positive, or non-finite) becomes
    /// `0.0` here, so [`FusedSampleParams::matches`] does not split a batch
    /// of rows whose sampled outputs are necessarily identical.
    pub fn from_config(config: &SamplingConfig) -> Self {
        Self {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
            top_n_sigma: config.effective_top_n_sigma(),
            typical_p: config.effective_typical_p(),
        }
    }

    /// Bitwise equality of the fused scalar params.
    ///
    /// Uses `f32::to_bits` so the comparison is exact (and clippy-clean): two
    /// rows that derive their config from the same request are bit-identical,
    /// and any difference forces the per-row fallback.
    pub fn matches(&self, other: &Self) -> bool {
        self.temperature.to_bits() == other.temperature.to_bits()
            && self.top_k == other.top_k
            && self.top_p.to_bits() == other.top_p.to_bits()
            && self.min_p.to_bits() == other.min_p.to_bits()
            && self.top_n_sigma.to_bits() == other.top_n_sigma.to_bits()
            && self.typical_p.to_bits() == other.typical_p.to_bits()
    }
}

/// Returns `true` when `config` can be sampled by the batched fused fast path.
///
/// The fast path applies one set of scalar parameters across the whole
/// `[B, vocab]` batch in a single [`ffi::fused_sample`] call. It cannot
/// represent per-row history-based penalties (repetition / DRY / frequency /
/// presence), a non-empty token bias, or XTC (`xtc_probability > 0.0`), all of
/// which require per-row logit edits before sampling. XTC, like a non-empty
/// token bias, is a per-row logit edit that must run through the per-row
/// sampler, so it disqualifies the fused fast path. When this returns `false`,
/// the caller must fall back to the per-row sampler.
///
/// `top_n_sigma` deliberately does NOT disqualify the fused path: it is
/// row-wise, history-free and RNG-free, and its scalar rides in
/// [`FusedSampleParams`], so a batch whose rows share one value applies the
/// filter to the whole `[B, V]` tensor ([`apply_row_filters`]) and still
/// dispatches once. Rows with different values diverge in
/// [`FusedSampleParams::matches`] and fall back per row, exactly like mixed
/// `top_p` values do.
///
/// Used by: `BatchScheduler::execute_batched_decode` fast-path gate
pub fn config_supports_fused_batch(config: &SamplingConfig) -> bool {
    !config.needs_token_history()
        && config.token_bias.is_empty()
        && config.xtc_probability <= 0.0
        // #1485: mirostat carries per-sequence feedback state and replaces
        // the chain; the extended chain (dynatemp / min_keep / adaptive-p)
        // needs per-row Rust filter arithmetic the fused dispatch has no
        // parameters for. Both must take the per-row sampler.
        && config.effective_mirostat() == 0
        && !config.needs_extended_chain()
}

/// Per-row eligibility for the batched fused fast path.
///
/// A row may join the single-dispatch `[B, vocab] -> [B]` fast path only when
/// its sampling config is fused-compatible ([`config_supports_fused_batch`])
/// and it imposes none of the per-row obligations that need the per-row
/// sampler:
///
/// - `needs_logit_mask`: a per-row logit mask, e.g. a structured-output
///   grammar mask.
/// - `needs_token_override`: a post-sample token override, e.g. a
///   thinking-budget forced `</think>`.
/// - `needs_per_token_payload`: a per-token output payload, e.g. logprobs.
///
/// Any of those returns `false` and sends the row to the per-row fallback.
///
/// Used by: `BatchScheduler::execute_batched_decode` fast-path gate
pub fn row_supports_fused_batch(
    config: &SamplingConfig,
    needs_logit_mask: bool,
    needs_token_override: bool,
    needs_per_token_payload: bool,
) -> bool {
    config_supports_fused_batch(config)
        && !needs_logit_mask
        && !needs_token_override
        && !needs_per_token_payload
}

/// Batched fused sampler: sample `[B]` token ids from `[B, vocab]` (or
/// `[B, 1, vocab]`) logits with a single eval/sync point.
///
/// All `B` rows are sampled with the same scalar parameters in ONE
/// [`ffi::fused_sample`] dispatch, then the `[B]` token array is evaluated
/// once and copied to host. This replaces the per-row slice + sample + eval +
/// `item_i32` round trips that [`batched_sample`] performs (one eval/sync per
/// row), collapsing `B` sync points into one.
///
/// Correctness: the caller must have confirmed every row is fused-eligible
/// (see [`row_supports_fused_batch`]) and shares these `params`. Greedy
/// (`temperature == 0` or `top_k == 1`) output is byte-identical to the
/// per-row path because `argmax` over the last axis is independent per row.
/// Stochastic sampling differs from the per-row path only in random-number
/// sequencing (the documented batched-vs-B=1 jitter class), not in the
/// sampled distribution.
///
/// Used by: `BatchScheduler::execute_batched_decode` fast-path dispatch
pub fn batched_fused_sample(logits: &MlxArray, params: &FusedSampleParams) -> Vec<i32> {
    // [B, 1, vocab] -> [B, vocab]; a 2-D input is returned unchanged.
    let last_logits = ffi::slice_last_logits(logits);
    // Row filters (top-n-sigma) apply to the whole [B, vocab] batch before the
    // single fused dispatch; a no-op when every filter is disabled.
    let last_logits = apply_row_filters(last_logits, params);
    let tokens = ffi::fused_sample(
        &last_logits,
        params.temperature,
        params.top_k,
        params.top_p,
        params.min_p,
    );
    crate::sampling_dispatch::report_sampling_dispatch();
    token_ids_to_host(&tokens)
}

/// Copy a 1-D `[B]` token-id array to host as `Vec<i32>` with a single
/// evaluation.
///
/// [`ffi::fused_sample`] returns `uint32` token ids (from `argmax` or
/// `categorical`); the raw bytes are reinterpreted as `i32`, which is exact
/// for any token id in `0..vocab_size` (well under `i32::MAX`). This mirrors
/// the raw-byte extraction already used by [`compute_logprobs`] for
/// argpartition indices and avoids adding an `astype` node to the sampling
/// graph. [`ffi::array_to_raw_bytes`] evaluates and makes the array
/// contiguous internally, so it is the single sync point for the batch.
fn token_ids_to_host(tokens: &MlxArray) -> Vec<i32> {
    let bytes = ffi::array_to_raw_bytes(tokens);
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Batch-safe row-filter hook: apply the enabled RNG-free, history-free
/// logit filters to a `[B, V]` (or `[V]`-broadcastable) logit tensor in the
/// fixed order pinned by #1375:
///
/// ```text
/// top_n_sigma -> p_less (#1373) -> typical_p (#1377)
/// ```
///
/// (`p_less` lands in its own issue; its slot in this function fixes the
/// order here so it cannot accidentally reorder the chain).
///
/// Returns the input pointer unchanged, adding NO graph nodes, when every
/// filter is disabled or when the config is greedy (`temperature == 0.0 ||
/// top_k == 1`), so the disabled baseline and the greedy path both stay
/// byte-identical. A non-finite or non-positive `top_n_sigma` counts as
/// disabled, which also makes the b10621 `-1.0` "disabled" sentinel inert
/// rather than a mask-everything threshold; a `typical_p` outside the open
/// interval `(0.0, 1.0)` (b10621 documents `1.0` as disabled and leaves the
/// low end of the range undeclared) counts as disabled the same way.
///
/// Used by: [`preprocess_logits_for_sampling`] (per-row sampler, and through
/// it [`effective_token_distribution`] / [`sample_token_with_distribution`]),
/// [`batched_fused_sample`], `BatchScheduler::prime_lookahead_with_input`
pub fn apply_row_filters(
    logits: UniquePtr<MlxArray>,
    params: &FusedSampleParams,
) -> UniquePtr<MlxArray> {
    // Greedy skips every row filter: the argmax of the unfiltered row is by
    // definition inside every "keep the top" mask, so filtering could only
    // add graph nodes without changing the sampled token.
    if params.temperature == 0.0 || params.top_k == 1 {
        return logits;
    }
    let mut logits = logits;
    if params.top_n_sigma > 0.0 && params.top_n_sigma.is_finite() {
        logits = top_n_sigma_filter(&logits, params.top_n_sigma);
    }
    // p_less (#1373) slot: runs here, after top_n_sigma, when it lands.
    if params.typical_p.is_finite() && params.typical_p > 0.0 && params.typical_p < 1.0 {
        // b10621's default sampler chain is `... top_n_sigma; top_k; typ_p;
        // top_p; min_p ...`: typical sampling runs on the RENORMALIZED top-k
        // survivors, so its entropy and typicality ranking are computed over
        // the truncated distribution, not the full vocabulary. mlxcel's
        // top_k lives in the C++ chain AFTER this hook, so to reproduce the
        // b10621 chain position the top-k mask is applied here first, ahead
        // of the typical filter. The C++ chain's own top_k then re-selects
        // the same set (masking to top-k is idempotent), so no stage runs
        // out of order and the fused dispatch is unchanged.
        if params.top_k > 1 {
            let vocab = ffi::array_shape(&logits).last().copied().unwrap_or(0);
            if params.top_k < vocab {
                logits = top_k_filter(&logits, params.top_k);
            }
        }
        logits = typical_p_filter(&logits, params.typical_p);
    }
    logits
}

/// The Rust extended sampler chain (#1485): the full b10621 filter order
///
/// ```text
/// top_n_sigma -> top_k -> typical_p -> top_p -> min_p -> xtc -> temperature
/// ```
///
/// as MLX graph ops on the penalized, untempered `[1, vocab]` logits, with
/// the `min_keep` floor on the four filters b10621 gives one to (top-p,
/// min-p, typical-p, XTC's skip rule) and the dynamic-temperature transform
/// in the temperature slot. Used only when
/// [`SamplingConfig::needs_extended_chain`] holds; every other config stays
/// on the fused C++ chain, byte-identical to before #1485.
///
/// Returns float32 logits already divided by the (possibly dynamic)
/// temperature, ready for a plain categorical draw (`ffi::fused_sample`
/// with `temperature 1.0` and every filter disabled).
///
/// `xtc_gate` is the same one-uniform gate draw the fused path consumes
/// ([`xtc_gate_draw`]); pass `None` when `xtc_probability <= 0.0`.
fn apply_extended_chain(
    logits: &MlxArray,
    config: &SamplingConfig,
    xtc_gate: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let min_keep = config.effective_min_keep();
    let mut x = ffi::astype(logits, dtype::FLOAT32);

    let top_n_sigma = config.effective_top_n_sigma();
    if top_n_sigma > 0.0 {
        x = top_n_sigma_filter(&x, top_n_sigma);
    }
    if config.top_k > 1 {
        let vocab = ffi::array_shape(&x).last().copied().unwrap_or(0);
        if config.top_k < vocab {
            x = top_k_filter(&x, config.top_k);
        }
    }
    let typical_p = config.effective_typical_p();
    if typical_p < 1.0 {
        x = typical_p_filter_min_keep(&x, typical_p, min_keep);
    }
    if config.top_p < 1.0 && config.top_p >= 0.0 {
        x = top_p_filter_min_keep(&x, config.top_p, min_keep);
    }
    if config.min_p > 0.0 {
        x = min_p_filter_min_keep(&x, config.min_p, min_keep);
    }
    if let Some(gate) = xtc_gate {
        let filtered = apply_xtc_filter_min_keep(
            &x,
            config.xtc_threshold,
            &config.xtc_special_token_ids,
            min_keep,
        );
        let probability = ffi::full_f32(&[1], config.xtc_probability, dtype::FLOAT32);
        let gate_hit = ffi::less(gate, &probability);
        x = ffi::where_cond(&gate_hit, &filtered, &x);
    }

    if config.effective_dynatemp_range() > 0.0 {
        dynatemp_transform(
            &x,
            config.temperature,
            config.effective_dynatemp_range(),
            config.dynatemp_exponent,
        )
    } else {
        // Plain temperature, divided exactly as the C++ chain divides it.
        // `needs_extended_chain` folds to `false` on the greedy path, so
        // `temperature > 0.0` holds here.
        let t = ffi::full_f32(&[1], config.temperature, dtype::FLOAT32);
        ffi::divide(&x, &t)
    }
}

/// b10621's `llama_sampler_temp_ext` (#1485): map the normalized entropy of
/// the surviving candidate distribution into
/// `[max(0, temp - range), temp + range]` through `norm_entropy ^ exponent`,
/// and scale the logits by that dynamic temperature.
///
/// The entropy is computed over the candidates that survive the preceding
/// filters (masked `-inf` entries carry zero probability and do not count),
/// normalized by `ln(candidate_count)`, exactly upstream's
/// `entropy / max_entropy`. A row with one (or zero) surviving candidates
/// passes through unscaled, upstream's `size <= 1` early return. The dynamic
/// temperature is floored at `1e-6` so an entropy of exactly zero degrades
/// to an argmax-equivalent draw instead of a division by zero (upstream
/// reaches its greedy `temp <= 0` branch in that case; the sampled token is
/// the same argmax either way).
fn dynatemp_transform(
    logits: &MlxArray,
    temperature: f32,
    range: f32,
    exponent: f32,
) -> UniquePtr<MlxArray> {
    let f = ffi::astype(logits, dtype::FLOAT32);
    let finite = ffi::isfinite(&f);
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    let sanitized = ffi::where_cond(&finite, &f, &neg_inf);

    let n = ffi::sum_axis(&ffi::astype(&finite, dtype::FLOAT32), -1, true);
    let logp = ffi::log_softmax(&sanitized, -1);
    let p = ffi::exp(&logp);
    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    let plogp = ffi::where_cond(&ffi::greater(&p, &zero), &ffi::multiply(&p, &logp), &zero);
    let entropy = ffi::negative(&ffi::sum_axis(&plogp, -1, true));
    let max_entropy = ffi::log(&n);
    let norm = ffi::divide(&entropy, &max_entropy);

    let min_temp = (temperature - range).max(0.0);
    let max_temp = temperature + range;
    let exp_arr = ffi::full_f32(&[1], exponent, dtype::FLOAT32);
    let scaled_norm = ffi::power(&norm, &exp_arr);
    let span = ffi::full_f32(&[1], max_temp - min_temp, dtype::FLOAT32);
    let base = ffi::full_f32(&[1], min_temp, dtype::FLOAT32);
    let dyn_temp = ffi::add(&base, &ffi::multiply(&span, &scaled_norm));
    let floor = ffi::full_f32(&[1], 1e-6, dtype::FLOAT32);
    let dyn_temp = ffi::maximum(&dyn_temp, &floor);

    let scaled = ffi::divide(&sanitized, &dyn_temp);
    // Upstream returns without touching a single-candidate row.
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
    let multi = ffi::greater(&n, &one);
    ffi::where_cond(&multi, &scaled, &sanitized)
}

/// b10621's adaptive-p draw (#1485, upstream `llama_sampler_adaptive_p`):
/// replaces the final categorical over the post-chain distribution. The
/// pre-transform probabilities are `softmax` of the filtered, tempered
/// logits; the adapted target is derived from the per-sequence EMA
/// (`2 * target - weighted_sum / total_weight`, clamped to `[0, 1]`); each
/// finite logit is rewritten to `5 - 10 d^2 / (1 + d)` with
/// `d = |p - adapted_target| / 0.3`; and the token is drawn from the softmax
/// of the transformed row. The selected token's ORIGINAL probability is read
/// back and parked as `pending` on the state; the caller confirms it with
/// [`SamplerState::accept_token`] once the token is final (a post-sample
/// override, e.g. a thinking-budget forced close, must NOT update the EMA,
/// mirroring upstream's accept-time id check).
///
/// A `state` of `None` (a stateless caller) uses the initial EMA, which is
/// upstream's freshly-reset sampler; production decode paths always pass
/// state.
fn adaptive_p_sample(
    filtered: &MlxArray,
    config: &SamplingConfig,
    state: Option<&mut SamplerState>,
    want_distribution: bool,
) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
    let target = config.effective_adaptive_target();
    let decay = config.adaptive_decay.clamp(0.0, 0.99);

    let f = ffi::astype(filtered, dtype::FLOAT32);
    let p = ffi::softmax(&f, -1);

    let adaptive = state.map(|s| {
        s.adaptive
            .get_or_insert_with(|| AdaptivePState::new(target, decay))
    });
    let (weighted_sum, total_weight) = adaptive
        .as_ref()
        .map(|a| (a.weighted_sum, a.total_weight))
        .unwrap_or_else(|| AdaptivePState::initial_ema(target, decay));

    let adapted = if total_weight == 0.0 {
        target.clamp(0.0, 1.0)
    } else {
        (2.0 * target.clamp(0.0, 1.0) - weighted_sum / total_weight).clamp(0.0, 1.0)
    };

    // Adaptive probability transform: quadratic near the target, linear
    // decay in the tails (upstream's DISTRIBUTION_WIDTH 0.3, PEAK 5.0,
    // SHARPNESS 10.0 constants).
    let adapted_arr = ffi::full_f32(&[1], adapted, dtype::FLOAT32);
    let inv_width = ffi::full_f32(&[1], 1.0 / 0.3, dtype::FLOAT32);
    let d = ffi::multiply(&ffi::abs(&ffi::subtract(&p, &adapted_arr)), &inv_width);
    let d2 = ffi::multiply(&d, &d);
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
    let sharp = ffi::full_f32(&[1], 10.0, dtype::FLOAT32);
    let peak = ffi::full_f32(&[1], 5.0, dtype::FLOAT32);
    let decayed = ffi::divide(&ffi::multiply(&sharp, &d2), &ffi::add(&one, &d));
    let transformed_vals = ffi::subtract(&peak, &decayed);
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    let transformed = ffi::where_cond(&ffi::isfinite(&f), &transformed_vals, &neg_inf);

    let token = ffi::fused_sample(&transformed, 1.0, 0, 1.0, 0.0);

    // Read back the selected token's ORIGINAL probability for the EMA.
    let idx = ffi::reshape(&ffi::astype(&token, dtype::INT32), &[1, 1]);
    let sel_p = ffi::take_along_axis(&p, &idx, -1);
    ffi::eval(&sel_p);
    let orig_p = ffi::item_f32(&sel_p);
    let token_id = token_ids_to_host(&token)[0];
    if let Some(a) = adaptive {
        a.pending = Some((token_id, orig_p));
    }

    let probs = want_distribution.then(|| ffi::softmax(&transformed, -1));
    (token, probs)
}

/// b10621's mirostat bypass path (#1485, upstream `common_sampler_init` with
/// `mirostat != 0`): token-bias -> plain temperature -> mirostat-select.
/// Penalties, DRY, and every truncation filter are skipped, exactly as
/// upstream skips its whole `params.samplers` chain.
///
/// Mirostat v2 truncates to the tokens whose surprise `-log2 p` is at most
/// `mu` (always keeping at least the argmax), renormalizes, draws, and
/// updates `mu -= eta * (observed_surprise - tau)`. Mirostat v1 first
/// estimates the Zipf exponent `s_hat` from the top `m = 100` sorted
/// probabilities, derives the truncation size
/// `k = ((s_hat - 1) * 2^mu / (1 - N^-(s_hat - 1)))^(1 / s_hat)`, truncates
/// to top-k, then draws and updates `mu` the same way. The observed surprise
/// is measured on the truncated, renormalized distribution, as upstream
/// measures it.
///
/// `temperature <= 0.0` degrades to the argmax with surprise `0` (upstream's
/// plain `llama_sampler_init_temp(temp)` stage collapses the candidate set
/// to the argmax before mirostat runs), still updating `mu`.
///
/// The second return value is the biased, untempered logits, the same
/// "adjusted logits" view the fused path returns for logprobs.
fn mirostat_sample_core(
    logits: &MlxArray,
    config: &SamplingConfig,
    state: Option<&mut SamplerState>,
    want_distribution: bool,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    Option<UniquePtr<MlxArray>>,
) {
    let version = config.effective_mirostat();
    let tau = config.mirostat_tau;
    let eta = config.mirostat_eta;

    let mirostat = state.map(|s| s.mirostat.get_or_insert_with(|| MirostatState::new(tau)));
    let mut mu = mirostat.as_ref().map_or(2.0 * tau, |m| m.mu);

    let last = ffi::slice_last_logits(logits);
    let biased = apply_token_bias_stage(last, config);

    const LN_2: f32 = std::f32::consts::LN_2;

    let (token, probs) = if config.temperature <= 0.0 {
        // The plain temperature stage collapses to the argmax; mirostat then
        // draws the single candidate with observed surprise 0.
        let token = ffi::fused_sample(&biased, 0.0, 0, 1.0, 0.0);
        ffi::eval(&token);
        mu -= eta * (0.0 - tau);
        let probs = want_distribution.then(|| ffi::fused_sample_probs(&biased, 0.0, 0, 1.0, 0.0));
        (token, probs)
    } else {
        let t = ffi::full_f32(&[1], config.temperature, dtype::FLOAT32);
        let x = ffi::divide(&ffi::astype(&biased, dtype::FLOAT32), &t);
        let logp = ffi::log_softmax(&x, -1);

        let masked = if version == 2 {
            // Surprise -log2 p <= mu, with the argmax always surviving: the
            // row-minimum surprise is the argmax's, so flooring the threshold
            // there reproduces upstream's keep-at-least-one truncation.
            let surprise = crate::ops::multiply_scalar(&ffi::negative(&logp), 1.0 / LN_2);
            let mu_arr = ffi::full_f32(&[1], mu, dtype::FLOAT32);
            let row_min = ffi::min_axis(&surprise, -1, true);
            let thresh = ffi::maximum(&mu_arr, &row_min);
            let keep = ffi::less_equal(&surprise, &thresh);
            let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
            ffi::where_cond(&keep, &x, &neg_inf)
        } else {
            // v1: estimate s_hat from the sorted top-m probabilities, derive
            // the truncation size k, and top-k the row.
            let vocab = ffi::array_shape(&x).last().copied().unwrap_or(0);
            let m = 100.min(vocab);
            let p = ffi::exp(&logp);
            let top = ffi::topk(&p, m, -1);
            ffi::eval(&top);
            let bytes = ffi::array_to_raw_bytes(&ffi::astype(&top, dtype::FLOAT32));
            let mut top_probs: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            top_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let mut sum_ti_bi = 0.0f64;
            let mut sum_ti_sq = 0.0f64;
            for i in 0..top_probs.len().saturating_sub(1) {
                if top_probs[i + 1] <= 0.0 {
                    break;
                }
                let t_i = (((i + 2) as f64) / ((i + 1) as f64)).ln();
                let b_i = ((top_probs[i] as f64) / (top_probs[i + 1] as f64)).ln();
                sum_ti_bi += t_i * b_i;
                sum_ti_sq += t_i * t_i;
            }
            let s_hat = if sum_ti_sq > 0.0 {
                sum_ti_bi / sum_ti_sq
            } else {
                f64::NAN
            };
            let epsilon_hat = s_hat - 1.0;
            let k = ((epsilon_hat * (mu as f64).exp2())
                / (1.0 - (vocab as f64).powf(-epsilon_hat)))
            .powf(1.0 / s_hat);
            let k = if k.is_finite() {
                (k as i64).clamp(1, vocab as i64) as i32
            } else {
                // Degenerate distribution (upstream feeds the same estimate
                // into an unchecked int cast): keep the whole row.
                vocab
            };
            if k < vocab {
                top_k_filter(&x, k)
            } else {
                ffi::copy(&x)
            }
        };

        let token = ffi::fused_sample(&masked, 1.0, 0, 1.0, 0.0);

        // Observed surprise of the drawn token within the truncated,
        // renormalized distribution.
        let idx = ffi::reshape(&ffi::astype(&token, dtype::INT32), &[1, 1]);
        let sel_lp = ffi::take_along_axis(&ffi::log_softmax(&masked, -1), &idx, -1);
        ffi::eval(&sel_lp);
        let observed_surprise = -ffi::item_f32(&sel_lp) / LN_2;
        mu -= eta * (observed_surprise - tau);

        let probs = want_distribution.then(|| ffi::softmax(&masked, -1));
        (token, probs)
    };

    if let Some(m) = mirostat {
        m.mu = mu;
    }
    crate::sampling_dispatch::report_sampling_dispatch();
    (token, biased, probs)
}

/// Top-n-sigma logit filter: keep only the tokens whose logit lies within
/// `n_sigma` standard deviations of the row maximum
/// (`logit >= max - n_sigma * std`), masking the rest to `-inf`.
/// Statistics (mean, population std with
/// `ddof = 0`) are taken over the FINITE entries of each vocabulary row, so
/// tokens already masked by token bias or penalties neither perturb the
/// threshold nor come back: `-inf >= thresh` is false and NaN compares false,
/// so masked entries stay masked.
///
/// All reductions are computed in float32 regardless of the logit dtype: a
/// float16 sum over a 150k-entry vocabulary overflows to `inf`, which would
/// drive the threshold to `-inf`/NaN and silently disable the filter.
///
/// Rows are independent (`axis = -1` reductions with `keepdims`), so a
/// `[B, V]` batch filters each row against its own statistics.
///
/// Used by: [`apply_row_filters`], unit tests
pub(crate) fn top_n_sigma_filter(logits: &MlxArray, n_sigma: f32) -> UniquePtr<MlxArray> {
    let f = ffi::astype(logits, dtype::FLOAT32);
    let finite = ffi::isfinite(&f);
    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    // Finite count per row. An all-masked row divides by zero into NaN,
    // which keeps every comparison false and leaves the row fully masked,
    // identical to its input.
    let n = ffi::sum_axis(&ffi::astype(&finite, dtype::FLOAT32), -1, true);
    let masked = ffi::where_cond(&finite, &f, &zero);
    let mean = ffi::divide(&ffi::sum_axis(&masked, -1, true), &n);
    let d = ffi::where_cond(&finite, &ffi::subtract(&f, &mean), &zero);
    let var = ffi::divide(&ffi::sum_axis(&ffi::multiply(&d, &d), -1, true), &n);
    let std = ffi::sqrt(&var);
    // Row maximum over the FINITE entries only: MLX's Max reducer propagates
    // NaN, so taking the max over the raw row would let a single NaN drive
    // `thresh` to NaN and silently mask the whole row to `-inf`. Replacing
    // every non-finite entry with `-inf` first keeps NaN (and `-inf`) out of
    // the reduction; a `+inf` entry is likewise excluded from the statistics
    // but still survives the `keep` comparison below.
    let neg_inf_f32 = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    let top = ffi::max_axis(&ffi::where_cond(&finite, &f, &neg_inf_f32), -1, true);
    let thresh = ffi::subtract(&top, &crate::ops::multiply_scalar(&std, n_sigma));
    let keep = ffi::greater_equal(&f, &thresh);
    // The mask fill carries the ORIGINAL logits dtype: MLX's `where`
    // promotes to `promote_types(x, y)`, so an f32 fill would silently
    // promote f16/bf16 logits to f32, changing the precision the C++ chain
    // runs in and doubling the `[B, V]` tensor. Same masking convention as
    // `min_p_filter` / `top_k_filter` otherwise: kept entries pass through
    // from the original logits.
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, ffi::array_dtype(logits));
    ffi::where_cond(&keep, logits, &neg_inf)
}

/// Locally typical sampling filter (Meister et al., "Locally Typical
/// Sampling"): keep the tokens whose surprisal `-log p` is closest to the
/// row entropy `H`, accumulating probability mass in that typicality order
/// until it reaches `typical_p`, and mask the rest to `-inf`.
///
/// All arithmetic runs in float32 on the log-softmax of the row, so the
/// entropy is computed on the penalized, untempered distribution and an f16
/// row cannot overflow. When `top_k` is active, [`apply_row_filters`] masks
/// the row to the top-k set BEFORE calling this filter, so the log-softmax
/// here renormalizes over the top-k survivors exactly as b10621's
/// `top_k -> typ_p` chain order does. Entries that are not finite in the
/// input (`-inf` masking from token bias / penalties / top-k, or a NaN from
/// a broken forward) are mapped to `-inf` BEFORE the softmax, so they carry
/// zero probability, contribute nothing to the entropy, sort to the end of
/// the typicality order, and stay masked in the output. Note this includes
/// `+inf`: a softmax cannot represent an infinite logit, so unlike
/// [`top_n_sigma_filter`] (which keeps `+inf` entries), this filter treats
/// them as masked. Both inputs are model failures; the divergence is
/// documented rather than papered over.
///
/// The most typical token always has an exclusive cumulative mass of
/// `0 < typical_p`, so at least one token survives. Unlike top-p, the
/// argmax CAN be dropped when it is less typical than the mid-probability
/// tokens, which is why [`apply_row_filters`] skips this filter on the
/// greedy path.
///
/// Rows are independent (`axis = -1` throughout), and the output carries the
/// original logits dtype (the mask fill is cast to it, mirroring
/// [`top_n_sigma_filter`]).
///
/// Used by: [`apply_row_filters`], unit tests
pub(crate) fn typical_p_filter(logits: &MlxArray, typical_p: f32) -> UniquePtr<MlxArray> {
    typical_p_filter_min_keep(logits, typical_p, 0)
}

/// [`typical_p_filter`] with b10621's `min_keep` floor (#1485): at least
/// `min_keep` tokens survive, taken in TYPICALITY order (upstream's
/// `llama_sampler_typical` keeps its first `min_keep` sorted candidates
/// regardless of the accumulated mass). `min_keep <= 1` adds no graph nodes
/// over the plain filter.
pub(crate) fn typical_p_filter_min_keep(
    logits: &MlxArray,
    typical_p: f32,
    min_keep: usize,
) -> UniquePtr<MlxArray> {
    let neg_inf_f32 = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    // Sanitize in f32: every non-finite entry (already-masked -inf, NaN,
    // +inf) becomes -inf so it cannot poison the softmax statistics.
    let raw = ffi::astype(logits, dtype::FLOAT32);
    let f = ffi::where_cond(&ffi::isfinite(&raw), &raw, &neg_inf_f32);
    let logp = ffi::log_softmax(&f, -1); // -inf entries stay -inf
    let p = ffi::exp(&logp); // masked entries are exactly 0
    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    // p * log p with the 0 * -inf = NaN case forced to 0.
    let plogp = ffi::where_cond(&ffi::greater(&p, &zero), &ffi::multiply(&p, &logp), &zero);
    // Entropy per row, [B, 1].
    let entropy = ffi::negative(&ffi::sum_axis(&plogp, -1, true));
    let finite_lp = ffi::isfinite(&logp);
    // |(-log p) - H|: distance from typicality. Masked entries sort last.
    let dev = ffi::abs(&ffi::subtract(&ffi::negative(&logp), &entropy));
    let pos_inf = ffi::full_f32(&[1], f32::INFINITY, dtype::FLOAT32);
    let shifted = ffi::where_cond(&finite_lp, &dev, &pos_inf);
    // Ascending sort: most typical first.
    let order = ffi::argsort(&shifted, -1);
    let p_sorted = ffi::take_along_axis(&p, &order, -1);
    // Exclusive cumulative mass strictly before each token in typicality
    // order: inclusive cumsum minus the entry itself.
    let cum_incl = ffi::cumsum(&p_sorted, -1, false, true);
    let cum_excl = ffi::subtract(&cum_incl, &p_sorted);
    // Inverse permutation back to vocabulary order.
    let inv = ffi::argsort(&order, -1);
    let cum_excl_orig = ffi::take_along_axis(&cum_excl, &inv, -1);
    let tp = ffi::full_f32(&[1], typical_p, dtype::FLOAT32);
    let mut keep = ffi::logical_and(&ffi::less(&cum_excl_orig, &tp), &finite_lp);
    if min_keep >= 2 {
        // b10621 `min_keep` (#1485): force-keep the first `min_keep` tokens
        // in typicality order. `inv` maps each vocab position to its
        // typicality rank, so `rank < min_keep` is the forced set; the
        // finite gate keeps masked entries out even when the row has fewer
        // finite candidates than the floor.
        let rank = ffi::astype(&inv, dtype::FLOAT32);
        let floor = ffi::full_f32(&[1], min_keep as f32, dtype::FLOAT32);
        let forced = ffi::logical_and(&ffi::less(&rank, &floor), &finite_lp);
        keep = ffi::logical_or(&keep, &forced);
    }
    // Mask fill in the ORIGINAL dtype (see `top_n_sigma_filter` on why an
    // f32 fill would silently promote f16/bf16 logits).
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, ffi::array_dtype(logits));
    ffi::where_cond(&keep, logits, &neg_inf)
}

/// Apply repetition penalty to logits.
///
/// For tokens in history:
/// - If logit > 0: divide by penalty
/// - If logit < 0: multiply by penalty
///
/// Used by: standard generation, speculative decoding
pub(crate) fn apply_repetition_penalty(
    logits: &MlxArray,
    token_history: &[i32],
    penalty: f32,
) -> UniquePtr<MlxArray> {
    let mut seen: Vec<i32> = token_history.to_vec();
    seen.sort_unstable();
    seen.dedup();
    apply_repetition_penalty_sorted(logits, &seen, penalty)
}

/// Core repetition-penalty application over an already sorted-and-deduped set
/// of seen token ids.
///
/// [`apply_repetition_penalty`] is the rebuild-every-token entry point: it
/// sorts and deduplicates `token_history` and then calls this. [`SamplerState`]
/// keeps its `seen_sorted` set incrementally maintained (sorted, deduped) and
/// feeds it here directly. Both produce byte-identical logits for the same
/// history because `take_along_axis`/`put_along_axis` over the same unique
/// index set apply the identical per-element ops, and the set of unique ids is
/// independent of how it was assembled.
///
/// Used by: apply_repetition_penalty (rebuild path), SamplerState::apply_repetition (incremental path)
pub(crate) fn apply_repetition_penalty_sorted(
    logits: &MlxArray,
    seen_sorted: &[i32],
    penalty: f32,
) -> UniquePtr<MlxArray> {
    if seen_sorted.is_empty() {
        return ffi::copy(logits);
    }

    let indices = ffi::from_slice_i32(seen_sorted, &[1, seen_sorted.len() as i32]);
    let selected = ffi::take_along_axis(logits, &indices, -1);

    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    let pen = ffi::full_f32(&[1], penalty, dtype::FLOAT32);

    let pos_mask = ffi::greater(&selected, &zero);
    let penalized_pos = ffi::divide(&selected, &pen);
    let penalized_neg = ffi::multiply(&selected, &pen);
    let penalized = ffi::where_cond(&pos_mask, &penalized_pos, &penalized_neg);

    ffi::put_along_axis(logits, &indices, &penalized, -1)
}

/// Apply OpenAI-style frequency and presence penalties to logits.
///
/// Used by: standard generation, speculative decoding
pub(crate) fn apply_frequency_presence_penalty(
    logits: &MlxArray,
    token_history: &[i32],
    frequency_penalty: f32,
    presence_penalty: f32,
) -> UniquePtr<MlxArray> {
    let mut token_counts: HashMap<i32, usize> = HashMap::new();
    for &tok in token_history {
        *token_counts.entry(tok).or_insert(0) += 1;
    }

    if token_counts.is_empty() {
        return ffi::copy(logits);
    }

    let shape = ffi::array_shape(logits);
    let vocab_size = *shape.last().unwrap() as usize;

    let mut penalties = vec![0.0f32; vocab_size];
    for (&token_id, &count) in &token_counts {
        if token_id >= 0 && (token_id as usize) < vocab_size {
            penalties[token_id as usize] = frequency_penalty * count as f32 + presence_penalty;
        }
    }

    let penalty_array = ffi::from_slice_f32(&penalties, &[1, vocab_size as i32]);
    let penalty_broadcast = ffi::broadcast_to(&penalty_array, &shape);
    ffi::subtract(logits, &penalty_broadcast)
}

/// Whether DRY matching breaks at `window[pos]` (#1485).
///
/// Two breaker sources compose:
/// - `dry_sequence_breakers`: exact token ids (the mlxcel-native surfaces),
///   the pre-#1485 semantics unchanged.
/// - `dry_breaker_heads`: head-token entries derived from b10621's breaker
///   STRINGS by scanning the vocabulary (upstream
///   `get_overlapping_token_sequences`). A head with an empty tail breaks on
///   its own (the token's decoded text contains the breaker string, the
///   overwhelmingly common case; for a single-character breaker this is the
///   only form). A head with a non-empty tail breaks only when the window
///   tokens after `pos` spell the tail out in full, which is how a
///   multi-character breaker split across token boundaries is recognized.
fn dry_breaks_at(config: &SamplingConfig, window: &[i32], pos: usize) -> bool {
    if config.dry_sequence_breakers.contains(&window[pos]) {
        return true;
    }
    if config.dry_breaker_heads.is_empty() {
        return false;
    }
    config
        .dry_breaker_heads
        .get(&window[pos])
        .is_some_and(|tails| {
            tails.iter().any(|tail| {
                tail.is_empty()
                    || (pos + tail.len() < window.len()
                        && tail
                            .iter()
                            .enumerate()
                            .all(|(j, &t)| window[pos + 1 + j] == t))
            })
        })
}

/// Apply DRY (Don't Repeat Yourself) penalty to logits.
///
/// This runs on CPU as sequential pattern matching, which keeps the matching
/// invariant explicit and mirrors the upstream llama.cpp style algorithm.
///
/// Used by: standard generation, speculative decoding
pub(crate) fn apply_dry_penalty(
    logits: &MlxArray,
    token_history: &[i32],
    config: &SamplingConfig,
) -> UniquePtr<MlxArray> {
    let history_len = token_history.len();
    if history_len < 2 {
        return ffi::copy(logits);
    }
    // b10621's sampler disables DRY outright when dry_base < 1.0 (its
    // llama-sampler init early-outs); the request layers already sanitize,
    // so this is the defense-in-depth mirror of that early-out.
    if config.dry_base < 1.0 {
        return ffi::copy(logits);
    }

    // b10621 sentinel semantics (#1436): `0` disables DRY (the caller gates
    // on it, and the empty-window guard below is the backstop);
    // `DRY_FULL_HISTORY` scans everything (the explicit successor of the
    // pre-#1436 `0`); any other value is a recent-token window.
    let window = if config.dry_penalty_last_n == crate::generate::DRY_FULL_HISTORY {
        token_history
    } else if config.dry_penalty_last_n == 0 {
        &token_history[history_len..]
    } else {
        let start = history_len.saturating_sub(config.dry_penalty_last_n);
        &token_history[start..]
    };

    let window_len = window.len();
    if window_len < 2 {
        return ffi::copy(logits);
    }

    let mut token_positions: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &tok) in window.iter().enumerate() {
        token_positions.entry(tok).or_default().push(i);
    }

    let last_token = window[window_len - 1];
    let mut penalties: HashMap<i32, f32> = HashMap::new();

    if let Some(positions) = token_positions.get(&last_token) {
        for &pos in positions {
            if pos >= window_len - 1 {
                continue;
            }

            let mut match_len = 1;
            let mut p1 = pos;
            let mut p2 = window_len - 1;

            while p1 > 0 && p2 > 0 {
                p1 -= 1;
                p2 -= 1;

                if dry_breaks_at(config, window, p1) {
                    break;
                }

                if window[p1] == window[p2] {
                    match_len += 1;
                } else {
                    break;
                }
            }

            // b10621 penalizes AT the allowed length (`>=`), so a repeat
            // exactly `dry_allowed_length` long gets the `base^0` tier;
            // pre-#1436 mlxcel used `>` and never emitted that tier.
            if match_len >= config.dry_allowed_length {
                let next_pos = pos + 1;
                if next_pos < window_len {
                    let next_token = window[next_pos];
                    // Upstream caps the exponent at FLOAT_MAX_LOG / ln(base)
                    // (~158 at base 1.75) so a very long full-history match
                    // cannot push the penalty to infinity; mirror the cap.
                    let mut exponent = (match_len - config.dry_allowed_length) as f32;
                    if config.dry_base > 1.0 {
                        let max_exponent = 88.722_84_f32 / config.dry_base.ln();
                        if exponent > max_exponent {
                            exponent = max_exponent;
                        }
                    }
                    let penalty = config.dry_multiplier * config.dry_base.powf(exponent);
                    let entry = penalties.entry(next_token).or_insert(0.0);
                    if penalty > *entry {
                        *entry = penalty;
                    }
                }
            }
        }
    }

    if penalties.is_empty() {
        return ffi::copy(logits);
    }

    let logits_shape = ffi::array_shape(logits);
    let vocab_size = *logits_shape.last().unwrap();
    let batch_size = if logits_shape.len() > 1 {
        logits_shape[0]
    } else {
        1
    };
    let total = (batch_size * vocab_size) as usize;
    let mut penalty_data = vec![0.0f32; total];

    for (token_id, penalty) in &penalties {
        let idx = *token_id as usize;
        if idx < vocab_size as usize {
            for b in 0..batch_size as usize {
                penalty_data[b * vocab_size as usize + idx] = -penalty;
            }
        }
    }

    let penalty_arr = ffi::from_slice_f32(&penalty_data, &logits_shape);
    ffi::add(logits, &penalty_arr)
}

/// Mirostat per-sequence feedback state (#1485): the surprise target `mu`,
/// initialized to `2 * tau` (upstream `llama_sampler_init_mirostat[_v2]`)
/// and updated `mu -= eta * (observed_surprise - tau)` after every draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirostatState {
    /// Current surprise target, in bits.
    pub mu: f32,
}

impl MirostatState {
    /// Fresh state for a target entropy `tau` (upstream's `2 * tau` init).
    pub fn new(tau: f32) -> Self {
        Self { mu: 2.0 * tau }
    }
}

/// Adaptive-p per-sequence feedback state (#1485): the EMA over the original
/// probabilities of the accepted tokens (upstream
/// `llama_sampler_adaptive_p`). `weighted_sum / total_weight` is the running
/// mean the adapted target is derived from; both are seeded so that the mean
/// starts exactly at `target`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptivePState {
    /// EMA decay captured at creation (clamped to `0.0..=0.99` upstream).
    pub decay: f32,
    /// `sum(p_i * decay^i)` over accepted tokens, seeded `target / (1 - decay)`.
    pub weighted_sum: f32,
    /// `sum(decay^i)`, converging to `1 / (1 - decay)`, seeded there.
    pub total_weight: f32,
    /// The (token id, original probability) pair the last draw parked,
    /// awaiting [`SamplerState::accept_token`]. Cleared on accept whether or
    /// not the ids matched, mirroring upstream's accept hook.
    pub pending: Option<(i32, f32)>,
}

impl AdaptivePState {
    /// Fresh state (upstream `llama_sampler_init_adaptive_p` seeding).
    pub fn new(target: f32, decay: f32) -> Self {
        let (weighted_sum, total_weight) = Self::initial_ema(target, decay);
        Self {
            decay,
            weighted_sum,
            total_weight,
            pending: None,
        }
    }

    /// The seed EMA values: `target / (1 - decay)` and `1 / (1 - decay)`.
    pub fn initial_ema(target: f32, decay: f32) -> (f32, f32) {
        let decay = decay.clamp(0.0, 0.99);
        (target.max(0.0) / (1.0 - decay), 1.0 / (1.0 - decay))
    }
}

/// Per-sequence incremental sampler state for history-based penalties.
///
/// Long generations re-derive the same penalty inputs on every decode step:
/// the rebuild-every-token [`apply_repetition_penalty`] clones, sorts, and
/// deduplicates the entire token history, and [`apply_frequency_presence_penalty`]
/// rebuilds a token-count map and allocates a fresh full-vocabulary penalty
/// vector. This state maintains those inputs incrementally per sequence so each
/// decode step only absorbs the newly appended token(s):
///
/// - `seen_sorted`: the sorted, deduplicated set of seen token ids for the
///   repetition penalty (binary-search insert per new token).
/// - `counts`: per-token occurrence counts for the frequency/presence penalty.
/// - `sparse_idx` / `sparse_val`: reusable scratch buffers that hold only the
///   touched token ids and their penalty deltas, so the frequency/presence
///   penalty never allocates a full-vocab vector.
///
/// The state is created lazily (only when a repetition/frequency/presence
/// penalty is active) and lives on the owning sequence, so the default
/// no-penalty path never allocates it. DRY is intentionally not state-backed
/// (its sliding window would need fragile position rebasing); it keeps using
/// `token_history` directly with unchanged behavior.
///
/// Results are byte-identical to the rebuild-every-token path (see the
/// `sampler_state_*` parity tests), so the incremental state is purely an
/// optimization and never changes which token is sampled.
///
/// Used by: [`sample_token_optimized_with_state`]
#[derive(Debug, Clone, Default)]
pub struct SamplerState {
    /// Mirostat per-sequence state (#1485): present exactly while the config
    /// runs a mirostat mode. Carries the surprise target `mu` across steps.
    pub mirostat: Option<MirostatState>,
    /// Adaptive-p per-sequence state (#1485): present exactly while the
    /// config enables adaptive-p. Carries the EMA and the pending
    /// (token, original probability) pair awaiting
    /// [`SamplerState::accept_token`].
    pub adaptive: Option<AdaptivePState>,
    /// Maintain `seen_sorted` (repetition penalty is active).
    track_seen: bool,
    /// Maintain `counts` (frequency or presence penalty is active).
    track_counts: bool,
    /// Sorted, deduplicated seen token ids (repetition penalty input).
    seen_sorted: Vec<i32>,
    /// Per-token occurrence counts (frequency/presence penalty input).
    counts: HashMap<i32, usize>,
    /// Reusable scratch buffer: touched token ids for the sparse penalty.
    sparse_idx: Vec<i32>,
    /// Reusable scratch buffer: per-id penalty deltas aligned with `sparse_idx`.
    sparse_val: Vec<f32>,
    /// Number of leading `token_history` entries already absorbed.
    absorbed_len: usize,
    /// Last absorbed token id, used for the O(1) append/divergence check.
    tip_token: i32,
}

impl SamplerState {
    /// Create state sized to the penalties `config` actually enables. Only the
    /// structures a penalty needs are maintained, so a repetition-only config
    /// never touches the count map and a frequency-only config never touches
    /// the sorted set.
    pub fn for_config(config: &SamplingConfig) -> Self {
        Self {
            track_seen: config.repetition_penalty != 1.0,
            track_counts: config.frequency_penalty != 0.0 || config.presence_penalty != 0.0,
            ..Self::default()
        }
    }

    /// Absorb a single newly appended token into the tracked structures.
    fn absorb_one(&mut self, token: i32) {
        if self.track_seen
            && let Err(pos) = self.seen_sorted.binary_search(&token)
        {
            self.seen_sorted.insert(pos, token);
        }
        if self.track_counts {
            *self.counts.entry(token).or_insert(0) += 1;
        }
    }

    /// Discard the incremental state and re-absorb `history` from scratch. Used
    /// when the history shrank or diverged (the append-only invariant no longer
    /// holds), which is always correct.
    fn rebuild(&mut self, history: &[i32]) {
        self.seen_sorted.clear();
        self.counts.clear();
        for &t in history {
            self.absorb_one(t);
        }
        self.absorbed_len = history.len();
        self.tip_token = history.last().copied().unwrap_or(0);
    }

    /// Synchronize the incremental state to `history`.
    ///
    /// Append-only growth (the decode common case) absorbs just the new tail in
    /// O(new tokens). A shorter or diverged history (speculative rollback, KV
    /// cache trim/restore) falls back to a full [`Self::rebuild`]. The O(1) tip
    /// check detects divergence without an O(n) prefix comparison; the decode
    /// model only ever appends or truncates a suffix, so a matching length and
    /// tip imply an unchanged prefix.
    fn sync(&mut self, history: &[i32]) {
        let n = history.len();
        let tip_matches = self.absorbed_len == 0
            || (self.absorbed_len <= n && history[self.absorbed_len - 1] == self.tip_token);
        if n < self.absorbed_len || !tip_matches {
            self.rebuild(history);
            return;
        }
        for &t in &history[self.absorbed_len..] {
            self.absorb_one(t);
        }
        self.absorbed_len = n;
        if n > 0 {
            self.tip_token = history[n - 1];
        }
    }

    /// Repetition penalty over the incrementally maintained `seen_sorted` set.
    /// Byte-identical to [`apply_repetition_penalty`] for the same history.
    fn apply_repetition(&self, logits: &MlxArray, penalty: f32) -> UniquePtr<MlxArray> {
        apply_repetition_penalty_sorted(logits, &self.seen_sorted, penalty)
    }

    /// Frequency/presence penalty using the reusable sparse scratch buffers
    /// (touched tokens only); never allocates a full-vocabulary vector.
    ///
    /// Byte-identical to [`apply_frequency_presence_penalty`]: the rebuild path
    /// computes `subtract(logits, penalty_f32)`, which promotes the whole array
    /// to f32 (f16/bf16 -> f32 is lossless). This path promotes first, then
    /// applies `logits[id] - penalty[id]` to exactly the touched ids via
    /// take/put. Untouched ids keep their promoted value, which equals the
    /// rebuild path's `logits[id] - 0.0` for every finite logit.
    fn apply_frequency_presence(
        &mut self,
        logits: &MlxArray,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> UniquePtr<MlxArray> {
        if self.counts.is_empty() {
            return ffi::copy(logits);
        }

        let shape = ffi::array_shape(logits);
        let vocab_size = *shape.last().unwrap() as usize;

        // Move the scratch buffers out so the loop can read `self.counts` and
        // fill them without a borrow conflict; put them back before returning
        // so their capacity is reused next step.
        let mut idx = std::mem::take(&mut self.sparse_idx);
        let mut val = std::mem::take(&mut self.sparse_val);
        idx.clear();
        val.clear();
        for (&token_id, &count) in &self.counts {
            if token_id >= 0 && (token_id as usize) < vocab_size {
                idx.push(token_id);
                val.push(frequency_penalty * count as f32 + presence_penalty);
            }
        }

        let result = if idx.is_empty() {
            // No in-range tokens: matches the rebuild path's empty early return.
            ffi::copy(logits)
        } else {
            let promoted = ffi::astype(logits, dtype::FLOAT32);
            let k = idx.len() as i32;
            let indices = ffi::from_slice_i32(&idx, &[1, k]);
            let selected = ffi::take_along_axis(&promoted, &indices, -1);
            let values = ffi::from_slice_f32(&val, &[1, k]);
            let penalized = ffi::subtract(&selected, &values);
            ffi::put_along_axis(&promoted, &indices, &penalized, -1)
        };

        self.sparse_idx = idx;
        self.sparse_val = val;
        result
    }

    /// Confirm the token a draw finally emitted (#1485). Adaptive-p parks the
    /// sampled token and its ORIGINAL probability as `pending`; upstream's
    /// accept hook folds that probability into the EMA only when the accepted
    /// token IS the sampled one, so a post-sample override (a thinking-budget
    /// forced close, for example) leaves the EMA untouched. Call this once
    /// per emitted token on every decode path that can carry adaptive-p
    /// state; a no-op for every other config (no `adaptive` state, or no
    /// pending pair).
    pub fn accept_token(&mut self, token: i32) {
        if let Some(a) = &mut self.adaptive
            && let Some((pending_token, orig_p)) = a.pending.take()
            && pending_token == token
        {
            a.weighted_sum = orig_p + a.decay * a.weighted_sum;
            a.total_weight = 1.0 + a.decay * a.total_weight;
        }
    }
}

/// Configuration for log probability computation during generation.
///
/// When `enabled` is false, no logprobs are computed and zero overhead is incurred.
#[derive(Debug, Clone, Default)]
pub struct LogprobsConfig {
    /// Whether to compute log probabilities at all
    pub enabled: bool,
    /// Number of top alternative tokens to return (0 = only the selected token)
    pub top_k: usize,
    /// Which distribution the report is taken from (#1485).
    pub source: LogprobSource,
}

/// The distribution a per-token probability report is computed over (#1485).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogprobSource {
    /// The penalty-adjusted, pre-chain logits (`adjusted_logits` from
    /// [`sample_token_optimized`]): mlxcel's OpenAI-shaped `logprobs`
    /// behavior since #340, the default.
    #[default]
    Adjusted,
    /// The raw model logits, before token bias and penalties: b10621's
    /// pre-sampling `n_probs` view (`get_token_probabilities` reads the
    /// context logits directly).
    RawModel,
    /// The post-sampling-chain distribution the draw came from: b10621's
    /// `post_sampling_probs` view, reported as LINEAR probabilities.
    PostSampling,
}

/// Log probability data for a single generated token.
#[derive(Debug, Clone)]
pub struct TokenLogprobData {
    /// Token ID of the selected token
    pub token_id: i32,
    /// Log probability of the selected token
    pub logprob: f32,
    /// Top-k alternative (token_id, logprob) pairs, sorted descending by logprob
    pub top_alternatives: Vec<(i32, f32)>,
}

/// Compute log probabilities for the selected token from penalty-adjusted logits.
///
/// `adjusted_logits` should have shape `[1, vocab]` (output of `sample_token_optimized`).
/// Returns `TokenLogprobData` containing the selected token's log-probability and
/// optionally the top-k alternatives.
///
/// Zero-overhead when `config.enabled` is false.
pub fn compute_logprobs(
    adjusted_logits: &MlxArray,
    selected_token: i32,
    config: &LogprobsConfig,
) -> Option<TokenLogprobData> {
    if !config.enabled {
        return None;
    }

    // Selected-token-only fast path (`top_k == 0`). Avoid materializing the
    // full-vocabulary log-softmax array: `log_softmax(x)[s] == x[s] -
    // logsumexp(x)`, where `logsumexp` is a reduction (no full-vocab output)
    // and the selected logit is a single gather. The dtype regime matches the
    // full path (compute in the logit dtype, read out as f32), and the only
    // numerical difference is the order of the final subtraction (<= 1 ULP),
    // which is OpenAI-compatible. Token selection already happened upstream, so
    // this never changes which token is emitted. Every logprob caller (classic
    // decode plus the dflash / MTP `per_position_logprobs` helpers) funnels
    // through here, so all paths stay mutually consistent.
    //
    // The `top_k > 0` path below is unchanged and keeps its own (issue #340)
    // dtype-aware top-k extraction.
    if config.top_k == 0 {
        let idx = ffi::from_slice_i32(&[selected_token], &[1, 1]);
        let selected_logit = ffi::take_along_axis(adjusted_logits, &idx, -1);
        let lse = ffi::logsumexp_axis(adjusted_logits, -1, true);
        let selected_lp = ffi::subtract(&selected_logit, &lse);
        let selected_lp_f32 = ffi::astype(&selected_lp, dtype::FLOAT32);
        ffi::eval(&selected_lp_f32);
        return Some(TokenLogprobData {
            token_id: selected_token,
            logprob: ffi::item_f32(&selected_lp_f32),
            top_alternatives: Vec::new(),
        });
    }

    // Apply log-softmax to get per-token log probabilities.
    let log_probs = ffi::log_softmax(adjusted_logits, -1);
    ffi::eval(&log_probs);

    // Extract the log probability of the selected token. `selected_lp_arr`
    // inherits the model logit dtype (f16/bf16 for quantized models post-#289),
    // and `item_f32` reads the element's raw bytes via MLX `item<float>()`
    // without dtype conversion, so a 2-byte f16/bf16 element would be
    // reinterpreted as garbage. Cast the single value to f32 first. Casting
    // only this 1-element array (not the full-vocab `log_probs`) keeps the
    // decode hot path cheap, matching the top-k boundary below.
    let idx = ffi::from_slice_i32(&[selected_token], &[1, 1]);
    let selected_lp_arr = ffi::take_along_axis(&log_probs, &idx, -1);
    let selected_lp_f32 = ffi::astype(&selected_lp_arr, dtype::FLOAT32);
    ffi::eval(&selected_lp_f32);
    let selected_logprob = ffi::item_f32(&selected_lp_f32);

    // Compute top-k alternatives if requested.
    let top_alternatives = if config.top_k > 0 {
        let vocab_size = ffi::array_shape(&log_probs).last().copied().unwrap_or(0);
        // Clamp k to vocab_size to satisfy argpartition's requirement that kth < array_size.
        let k = (config.top_k as i32).min(vocab_size);
        // negate log_probs so argpartition gives us the top-k (smallest negated = largest)
        let neg_log_probs = ffi::negative(&log_probs);
        let partition_idx = ffi::argpartition(&neg_log_probs, k - 1, -1);
        ffi::eval(&partition_idx);

        // Slice only the first k elements from the partitioned result.
        // argpartition guarantees that indices 0..k contain the k smallest
        // values of the negated log_probs (= the k largest log_probs),
        // so we avoid materializing the full vocabulary into host memory.
        let shape = ffi::array_shape(&partition_idx);
        let ndim = shape.len();
        let starts = vec![0i32; ndim];
        let mut stops = shape.clone();
        stops[ndim - 1] = k.min(stops[ndim - 1]);
        let top_idx = ffi::slice(&partition_idx, &starts, &stops);

        // Gather the log_probs for the top-k partitioned indices.
        let top_lp = ffi::take_along_axis(&log_probs, &top_idx, -1);
        ffi::eval(&top_idx);
        ffi::eval(&top_lp);

        let k_usize = k as usize;

        // Use raw bytes to extract i32 token IDs from top_idx.
        let idx_bytes = ffi::array_to_raw_bytes(&top_idx);
        // `top_lp` inherits the model logit dtype, which is f16/bf16 for
        // quantized models (post-#289). `array_to_raw_bytes` dumps the buffer
        // verbatim with no dtype conversion, so a hardcoded 4-byte stride
        // overruns a 2-byte-per-element buffer and reinterprets the bytes as
        // garbage. Cast to f32 first so the stride is valid and the values are
        // correct, mirroring the dtype-aware selected-token path (`item_f32`).
        let top_lp_f32 = ffi::astype(&top_lp, dtype::FLOAT32);
        ffi::eval(&top_lp_f32);
        let lp_bytes = ffi::array_to_raw_bytes(&top_lp_f32);

        // Build (token_id, logprob) pairs for only the top-k partition.
        let mut pairs: Vec<(i32, f32)> = (0..k_usize.min(idx_bytes.len() / 4))
            .filter_map(|i| {
                let tok_bytes: [u8; 4] = idx_bytes[i * 4..(i + 1) * 4].try_into().ok()?;
                let lp_bytes4: [u8; 4] = lp_bytes[i * 4..(i + 1) * 4].try_into().ok()?;
                Some((i32::from_ne_bytes(tok_bytes), f32::from_ne_bytes(lp_bytes4)))
            })
            .collect();

        // Sort the k elements descending by logprob.
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    } else {
        Vec::new()
    };

    Some(TokenLogprobData {
        token_id: selected_token,
        logprob: selected_logprob,
        top_alternatives,
    })
}

/// Post-sampling probability report (#1485, b10621's
/// `post_sampling_probs`): the selected token's probability and the top
/// `top_n` (token, probability) pairs, taken from the float32 `[1, vocab]`
/// POST-CHAIN distribution the draw actually came from (the second return of
/// [`sample_token_with_state_and_distribution`]). Values are LINEAR
/// probabilities, not logs, carried in [`TokenLogprobData`]'s fields;
/// zero-probability entries are dropped from the top list, matching
/// upstream's `populate_token_probs` post-sampling arm, which breaks at the
/// first `p == 0` candidate.
pub fn compute_post_sampling_probs(
    probs: &MlxArray,
    selected_token: i32,
    top_n: usize,
) -> TokenLogprobData {
    let idx = ffi::from_slice_i32(&[selected_token], &[1, 1]);
    let sel = ffi::take_along_axis(probs, &idx, -1);
    ffi::eval(&sel);
    let selected_prob = ffi::item_f32(&sel);

    let top_alternatives = if top_n > 0 {
        let vocab = ffi::array_shape(probs).last().copied().unwrap_or(0);
        let k = (top_n as i32).min(vocab).max(1);
        let neg = ffi::negative(probs);
        let part = ffi::argpartition(&neg, k - 1, -1);
        let shape = ffi::array_shape(&part);
        let ndim = shape.len();
        let starts = vec![0i32; ndim];
        let mut stops = shape.clone();
        stops[ndim - 1] = k.min(stops[ndim - 1]);
        let top_idx = ffi::slice(&part, &starts, &stops);
        let top_p = ffi::take_along_axis(probs, &top_idx, -1);
        ffi::eval(&top_idx);
        ffi::eval(&top_p);
        let idx_bytes = ffi::array_to_raw_bytes(&top_idx);
        let p_bytes = ffi::array_to_raw_bytes(&top_p);
        let mut pairs: Vec<(i32, f32)> = (0..(k as usize).min(idx_bytes.len() / 4))
            .filter_map(|i| {
                let t: [u8; 4] = idx_bytes[i * 4..(i + 1) * 4].try_into().ok()?;
                let p: [u8; 4] = p_bytes[i * 4..(i + 1) * 4].try_into().ok()?;
                Some((i32::from_ne_bytes(t), f32::from_ne_bytes(p)))
            })
            .filter(|&(_, p)| p > 0.0)
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    } else {
        Vec::new()
    };

    TokenLogprobData {
        token_id: selected_token,
        logprob: selected_prob,
        top_alternatives,
    }
}

/// Apply min-p filtering to logits.
///
/// Production runs min-p inside the C++ chain (`fused_sample_filter_logits`)
/// or inside the rejection kernel; this is the reference copy the tests hold
/// those against.
///
/// Used by: unit tests (`fused_sample_probs_equals_softmax_of_filtered_over_t`
/// in `sampling::tests`)
#[allow(dead_code)]
pub(crate) fn min_p_filter(logits: &MlxArray, min_p: f32) -> UniquePtr<MlxArray> {
    let probs = ffi::softmax(logits, -1);
    let max_prob = ffi::max_axis(&probs, -1, true);
    let min_p_scalar = ffi::full_f32(&[1], min_p, dtype::FLOAT32);
    let threshold = ffi::multiply(&max_prob, &min_p_scalar);
    let mask = ffi::greater_equal(&probs, &threshold);
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    ffi::where_cond(&mask, logits, &neg_inf)
}

/// Apply top-k filtering to logits.
///
/// Production runs top-k inside the C++ chain or inside the rejection kernel;
/// this is the reference copy the tests hold those against.
///
/// Used by: [`apply_row_filters`] (the idempotent pre-`typical_p` top-k
/// mask that pins b10621's `top_k -> typ_p` chain position), unit tests
/// (`fused_sample_probs_equals_softmax_of_filtered_over_t` in
/// `sampling::tests`)
pub(crate) fn top_k_filter(logits: &MlxArray, k: i32) -> UniquePtr<MlxArray> {
    let neg_logits = ffi::negative(logits);
    let indices = ffi::argpartition(&neg_logits, k - 1, -1);

    let shape = ffi::array_shape(&indices);
    let ndim = shape.len();
    let mut start = vec![0i32; ndim];
    let mut stop: Vec<i32> = shape.clone();
    start[ndim - 1] = k - 1;
    stop[ndim - 1] = k;

    let kth_idx = ffi::slice(&indices, &start, &stop);
    let threshold = ffi::take_along_axis(logits, &kth_idx, -1);

    let mask = ffi::greater_equal(logits, &threshold);
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    ffi::where_cond(&mask, logits, &neg_inf)
}

/// Apply top-p (nucleus) filtering to logits.
///
/// Operates per-row on `[B, V]` logits tensors: argsort, cumsum, and mask
/// construction all use axis=-1 so each batch row is filtered independently.
///
/// Algorithm (mirrors upstream mlx-vlm PR #1094 / commit c7aaf2d):
///   1. softmax(logits, axis=-1)  → probs per row
///   2. argsort(-probs, axis=-1)  → indices that sort each row descending
///   3. take_along_axis(probs, sorted_indices, axis=-1) → sorted_probs per row
///   4. exclusive cumsum per row: computed as inclusive_cumsum(sorted_probs, axis=-1) − sorted_probs
///      → cumulative probability strictly *before* each token position
///   5. mask = cumsum_before <= p  (include tokens up to the nucleus boundary)
///   6. apply mask to sorted logits; masked positions get -inf
///   7. argsort(sorted_indices, axis=-1) → indices to undo the sort per row
///   8. take_along_axis(filtered_sorted_logits, unsort_indices, axis=-1) → result
///
/// Note: production generation routes through the C++ `fused_sample`, which
/// since issue #901 resolves top-p inside the dual-pivot rejection kernel and
/// reaches the C++ `top_p_filter` at `cpp/mlx_cxx_bridge.cpp` only under
/// `MLXCEL_SAMPLING_REJECTION=0` or after a convergence-cap fallback. This Rust
/// implementation is a reference/test-parity copy used to validate the
/// algorithm and in unit tests for batched correctness.
///
/// Used by: unit tests (`top_p_filter_*`,
/// `fused_sample_probs_equals_softmax_of_filtered_over_t` in
/// `sampling::tests`)
#[allow(dead_code)]
pub(crate) fn top_p_filter(logits: &MlxArray, p: f32) -> UniquePtr<MlxArray> {
    // Step 1: per-row softmax probabilities.
    let probs = ffi::softmax(logits, -1);

    // Step 2: per-row descending sort via ascending argsort of negated probs.
    let neg_probs = ffi::negative(&probs);
    let sorted_indices = ffi::argsort(&neg_probs, -1);

    // Step 3: gather sorted probabilities per row.
    let sorted_probs = ffi::take_along_axis(&probs, &sorted_indices, -1);

    // Step 4: exclusive cumulative sum along vocab axis.
    // cumsum(..., reverse=false, inclusive=true) gives inclusive cumsum;
    // subtracting sorted_probs yields the cumulative probability *before*
    // each token, which is the exclusive (shifted) form needed for nucleus.
    let cum_probs = ffi::cumsum(&sorted_probs, -1, false, true);
    let shifted_cum = ffi::subtract(&cum_probs, &sorted_probs);

    // Step 5: mask — keep tokens whose cumulative-before-prob is <= p.
    let p_scalar = ffi::full_f32(&[1], p, dtype::FLOAT32);
    let mask = ffi::less_equal(&shifted_cum, &p_scalar);

    // Step 6: apply mask to sorted logits; excluded positions become -inf.
    let sorted_logits = ffi::take_along_axis(logits, &sorted_indices, -1);
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    let filtered_sorted = ffi::where_cond(&mask, &sorted_logits, &neg_inf);

    // Step 7: per-row inverse permutation — argsort of the sort indices undoes
    // the sort (stable property of argsort on a permutation).
    let unsort_indices = ffi::argsort(&sorted_indices, -1);

    // Step 8: scatter filtered logits back to original vocab order per row.
    ffi::take_along_axis(&filtered_sorted, &unsort_indices, -1)
}

/// Production top-p (nucleus) filter for the Rust extended chain (#1485),
/// with b10621's `min_keep` floor: keep tokens in probability-descending
/// order while the exclusive cumulative mass is `<= p` (the same keep rule
/// as the reference [`top_p_filter`]), and force-keep at least `min_keep`
/// candidates in that order regardless of the mass, upstream
/// `llama_sampler_top_p`'s `i + 1 >= min_keep` continuation. Non-finite
/// entries are sanitized to `-inf` first so they carry zero probability,
/// sort to the tail, and can never be resurrected by the floor (the forced
/// set is gated on finiteness). Output is float32.
pub(crate) fn top_p_filter_min_keep(
    logits: &MlxArray,
    p: f32,
    min_keep: usize,
) -> UniquePtr<MlxArray> {
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    let raw = ffi::astype(logits, dtype::FLOAT32);
    let f = ffi::where_cond(&ffi::isfinite(&raw), &raw, &neg_inf);

    let probs = ffi::softmax(&f, -1);
    let order = ffi::argsort(&ffi::negative(&probs), -1);
    let sorted_probs = ffi::take_along_axis(&probs, &order, -1);
    let cum_incl = ffi::cumsum(&sorted_probs, -1, false, true);
    let cum_excl = ffi::subtract(&cum_incl, &sorted_probs);

    let sorted_f = ffi::take_along_axis(&f, &order, -1);
    let finite_sorted = ffi::isfinite(&sorted_f);
    let p_arr = ffi::full_f32(&[1], p, dtype::FLOAT32);
    let mut keep = ffi::logical_and(&ffi::less_equal(&cum_excl, &p_arr), &finite_sorted);
    if min_keep >= 2 {
        // Position index 0..V-1 within the sorted row: inclusive cumsum of
        // ones, minus one.
        let ones = ffi::ones(&ffi::array_shape(&f), dtype::FLOAT32);
        let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
        let pos = ffi::subtract(&ffi::cumsum(&ones, -1, false, true), &one);
        let floor = ffi::full_f32(&[1], min_keep as f32, dtype::FLOAT32);
        let forced = ffi::logical_and(&ffi::less(&pos, &floor), &finite_sorted);
        keep = ffi::logical_or(&keep, &forced);
    }

    let filtered_sorted = ffi::where_cond(&keep, &sorted_f, &neg_inf);
    let unsort = ffi::argsort(&order, -1);
    ffi::take_along_axis(&filtered_sorted, &unsort, -1)
}

/// Production min-p filter for the Rust extended chain (#1485), with
/// b10621's `min_keep` floor: keep tokens whose probability is at least
/// `p * max_probability` (the same threshold as the reference
/// [`min_p_filter`], which equals upstream's `logit >= max_logit + ln(p)`),
/// and force-keep at least `min_keep` candidates in probability-descending
/// order, upstream `llama_sampler_min_p`'s `i >= min_keep` continuation.
/// Output is float32; non-finite entries are sanitized to `-inf` and stay
/// masked.
pub(crate) fn min_p_filter_min_keep(
    logits: &MlxArray,
    p: f32,
    min_keep: usize,
) -> UniquePtr<MlxArray> {
    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    let raw = ffi::astype(logits, dtype::FLOAT32);
    let finite = ffi::isfinite(&raw);
    let f = ffi::where_cond(&finite, &raw, &neg_inf);

    let probs = ffi::softmax(&f, -1);
    let max_prob = ffi::max_axis(&probs, -1, true);
    let p_arr = ffi::full_f32(&[1], p, dtype::FLOAT32);
    let threshold = ffi::multiply(&max_prob, &p_arr);
    let mut keep = ffi::greater_equal(&probs, &threshold);
    if min_keep >= 2 {
        let order = ffi::argsort(&ffi::negative(&probs), -1);
        let rank = ffi::astype(&ffi::argsort(&order, -1), dtype::FLOAT32);
        let floor = ffi::full_f32(&[1], min_keep as f32, dtype::FLOAT32);
        let forced = ffi::logical_and(&ffi::less(&rank, &floor), &finite);
        keep = ffi::logical_or(&keep, &forced);
    }
    ffi::where_cond(&keep, &f, &neg_inf)
}

/// Apply XTC (Exclude Top Choices) filtering to logits.
///
/// Among the tokens whose probability exceeds `threshold`, if two or more
/// exist, this removes (sets to `-inf`) all of them except the single
/// least-probable one — suppressing the dominant choices promotes lexical
/// diversity. If fewer than two tokens exceed the threshold, this is a
/// no-op. Token ids in `allowlist` are never removed even when selected by
/// that rule; callers pass the tokenizer's newline token id(s) plus the full
/// merged end-of-sequence set (see `BatchScheduler::enqueue_request`) so XTC
/// can never suppress a token needed to end a line or the sequence.
///
/// Since #1379 the production XTC filter lives inside the C++ chain
/// (`apply_xtc_to_filtered_logits` in `mlx_cxx_bridge.cpp`), after min-p, on
/// the renormalised filtered row. This function is the Rust REFERENCE
/// implementation the C++ port is held against, element for element, by the
/// unit tests below; it is no longer on any production path.
///
/// Algorithm (all lazy MLX array ops, no host round-trip):
/// 1. `probs = softmax(logits)`; `above = probs > threshold`.
/// 2. No-op guard: `count(above) < 2` disables removal for that row.
/// 3. Mask every non-`above` position to `+inf` in a scratch copy of
///    `probs`, then `argmin` finds the single least-probable `above` token
///    (the one to keep).
/// 4. Removal candidates = `above` AND NOT the kept index AND NOT
///    `allowlist`, gated by the no-op guard from step 2.
/// 5. `where(remove, -inf, logits)`.
///
/// Used by: [`apply_xtc_step`], unit tests (`apply_xtc_filter_*`,
/// `xtc_runs_on_renormalised_filtered_row`,
/// `cpp_xtc_matches_the_rust_reference_filter` in `sampling::tests`)
#[allow(dead_code)]
pub(crate) fn apply_xtc_filter(
    logits: &MlxArray,
    threshold: f32,
    allowlist: &[i32],
) -> UniquePtr<MlxArray> {
    apply_xtc_filter_min_keep(logits, threshold, allowlist, 0)
}

/// [`apply_xtc_filter`] with b10621's `min_keep` skip rule (#1485): when the
/// removal would leave fewer than `min_keep` survivors
/// (`candidates - removed < min_keep`, upstream `llama_sample_xtc_apply`'s
/// `cur_p->size - pos_last >= ctx->min_keep` guard), the whole removal is
/// skipped for that row. `min_keep <= 1` adds no graph nodes over the plain
/// filter (a removal always leaves at least the least-probable
/// above-threshold token). This is the production XTC used by the Rust
/// extended chain; the fused C++ chain keeps its own XTC for every
/// `min_keep`-less config.
pub(crate) fn apply_xtc_filter_min_keep(
    logits: &MlxArray,
    threshold: f32,
    allowlist: &[i32],
    min_keep: usize,
) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(logits);
    let vocab_size = *shape.last().unwrap() as usize;

    let probs = ffi::softmax(logits, -1);

    // Tokens whose probability exceeds the threshold.
    let threshold_arr = ffi::full_f32(&[1], threshold, dtype::FLOAT32);
    let above = ffi::greater(&probs, &threshold_arr);

    // No-op guard: fewer than two above-threshold tokens in this row.
    let above_f32 = ffi::astype(&above, dtype::FLOAT32);
    let count = ffi::sum_axis(&above_f32, -1, true);
    let two = ffi::full_f32(&[1], 2.0, dtype::FLOAT32);
    let has_two_or_more = ffi::greater_equal(&count, &two);

    // Identify the single least-probable above-threshold token: mask every
    // other position to +inf so it can never win the row-wise argmin.
    let pos_inf = ffi::full_f32(&[1], f32::INFINITY, dtype::FLOAT32);
    let masked_probs = ffi::where_cond(&above, &probs, &pos_inf);
    let least_idx = ffi::argmin(&masked_probs, -1, true);

    // One-hot mark of the least-probable token via scatter, so it can be
    // excluded from the removal set below.
    let zeros_full = ffi::zeros(&shape, dtype::FLOAT32);
    let least_idx_shape = ffi::array_shape(&least_idx);
    let ones_col = ffi::ones(&least_idx_shape, dtype::FLOAT32);
    let is_least_f32 = ffi::put_along_axis(&zeros_full, &least_idx, &ones_col, -1);
    let zero_scalar = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    let is_least = ffi::greater(&is_least_f32, &zero_scalar);

    // Removal candidates: above-threshold, excluding the least-probable one.
    let not_least = ffi::logical_not(&is_least);
    let remove_candidate = ffi::logical_and(&above, &not_least);

    // Never remove allowlisted special tokens (newline + merged EOS ids).
    let remove_candidate = if allowlist.is_empty() {
        remove_candidate
    } else {
        let mut allow_vec = vec![0.0f32; vocab_size];
        for &id in allowlist {
            if id >= 0 && (id as usize) < vocab_size {
                allow_vec[id as usize] = 1.0;
            }
        }
        let allow_arr = ffi::from_slice_f32(&allow_vec, &[1, vocab_size as i32]);
        let allow_broadcast = ffi::broadcast_to(&allow_arr, &shape);
        let is_allowed = ffi::greater(&allow_broadcast, &zero_scalar);
        let not_allowed = ffi::logical_not(&is_allowed);
        ffi::logical_and(&remove_candidate, &not_allowed)
    };

    // Gate the whole filter on having >= 2 above-threshold candidates.
    let mut remove_mask = ffi::logical_and(&remove_candidate, &has_two_or_more);

    if min_keep >= 2 {
        // b10621 `min_keep` skip rule (#1485): survivors after removal are
        // the finite candidates minus the removed above-threshold set (all
        // but its least-probable member). If that leaves fewer than
        // `min_keep`, the removal does not run for this row.
        let finite = ffi::astype(&ffi::isfinite(logits), dtype::FLOAT32);
        let n_finite = ffi::sum_axis(&finite, -1, true);
        let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
        let survivors = ffi::add(&ffi::subtract(&n_finite, &count), &one);
        let floor = ffi::full_f32(&[1], min_keep as f32, dtype::FLOAT32);
        let enough = ffi::greater_equal(&survivors, &floor);
        remove_mask = ffi::logical_and(&remove_mask, &enough);
    }

    let neg_inf = ffi::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
    ffi::where_cond(&remove_mask, &neg_inf, logits)
}

/// Per-step probability gate for [`apply_xtc_filter`].
///
/// Draws exactly one uniform sample from the same per-request seeded global
/// MLX random stream the fused categorical sampler consumes at the end of
/// [`sample_token_optimized_core`] (`ffi::random_seed`, called once per
/// generation via `generation_policy::seed_rng_if_needed`, seeds this
/// stream). MLX's default random key is a thread-local sequence that is
/// split synchronously at graph-*construction* time, not at `eval` time —
/// so the order these calls are made in Rust (not the order their results
/// are later evaluated) determines which slice of the stream each one
/// consumes. Drawing here, before the categorical draw inside
/// [`ffi::fused_sample`], keeps the whole decode step reproducible for a
/// fixed seed: the same seed always produces the same gate outcome followed
/// by the same categorical sample.
///
/// Since #1379 production draws the gate in [`xtc_gate_draw`] and applies the
/// filter inside the C++ chain; this function remains as the Rust reference
/// for the gate semantics (one uniform, compared lazily, whole-row
/// where-select), exercised by the `apply_xtc_step_*` unit tests.
#[allow(dead_code)]
fn apply_xtc_step(logits: &MlxArray, config: &SamplingConfig) -> UniquePtr<MlxArray> {
    // SAFETY: `key` is documented to accept a null pointer, meaning "draw
    // from the current thread-local default RNG state" (mirrors the
    // existing `std::ptr::null()` "no explicit key" usage in `layers.rs`).
    let gate_draw =
        unsafe { ffi::random_uniform(0.0, 1.0, &[1], dtype::FLOAT32, std::ptr::null()) };
    let probability = ffi::full_f32(&[1], config.xtc_probability, dtype::FLOAT32);
    let gate = ffi::less(&gate_draw, &probability);

    let filtered = apply_xtc_filter(logits, config.xtc_threshold, &config.xtc_special_token_ids);
    ffi::where_cond(&gate, &filtered, logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logit_at(logits: &MlxArray, token_id: i32) -> f32 {
        let index = ffi::from_slice_i32(&[token_id], &[1, 1]);
        let taken = ffi::take_along_axis(logits, &index, -1);
        ffi::eval(&taken);
        ffi::item_f32(&taken)
    }

    #[test]
    fn apply_repetition_penalty_modifies_selected_logits() {
        let logits = ffi::from_slice_f32(&[1.0, 2.0, -1.0, 3.0, -2.0], &[1, 5]);
        let result = apply_repetition_penalty(&logits, &[1, 3], 2.0);

        assert_eq!(logit_at(&result, 0), 1.0);
        assert_eq!(logit_at(&result, 1), 1.0);
        assert_eq!(logit_at(&result, 3), 1.5);
    }

    #[test]
    fn apply_frequency_presence_penalty_accumulates_by_token_count() {
        let logits = ffi::from_slice_f32(&[0.0, 0.0, 0.0], &[1, 3]);
        let result = apply_frequency_presence_penalty(&logits, &[1, 1, 2], 0.5, 0.25);

        assert_eq!(logit_at(&result, 0), 0.0);
        assert_eq!(logit_at(&result, 1), -1.25);
        assert_eq!(logit_at(&result, 2), -0.75);
    }

    #[test]
    fn apply_dry_penalty_penalizes_followup_token_after_suffix_match() {
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0], &[1, 3]);
        let config = SamplingConfig {
            dry_multiplier: 1.0,
            dry_base: 2.0,
            dry_allowed_length: 1,
            ..Default::default()
        };

        let result = apply_dry_penalty(&logits, &[0, 1, 2, 0, 1], &config);

        assert_eq!(logit_at(&result, 0), 1.0);
        assert_eq!(logit_at(&result, 1), 1.0);
        assert_eq!(logit_at(&result, 2), -1.0);
    }

    #[test]
    fn sample_token_optimized_respects_greedy_argmax_path() {
        let logits = ffi::from_slice_f32(&[0.1, 0.9, 1.2], &[1, 1, 3]);
        let config = SamplingConfig::greedy();
        let (token, processed_logits) = sample_token_optimized(&logits, &config, &[]);

        ffi::eval(&token);
        assert_eq!(ffi::item_i32(&token), 2);
        assert_eq!(ffi::array_shape(&processed_logits), vec![1, 3]);
    }

    #[test]
    fn batched_sample_greedy_selects_argmax_per_sequence() {
        // Two sequences with different argmax positions
        // Seq 0: logits [0.1, 0.9, 1.2] -> argmax = 2
        // Seq 1: logits [2.0, 0.5, 0.1] -> argmax = 0
        let logits = ffi::from_slice_f32(&[0.1, 0.9, 1.2, 2.0, 0.5, 0.1], &[2, 1, 3]);

        let config0 = SamplingConfig::greedy();
        let config1 = SamplingConfig::greedy();
        let configs: Vec<&SamplingConfig> = vec![&config0, &config1];
        let histories: Vec<&[i32]> = vec![&[], &[]];

        let tokens = batched_sample(&logits, &configs, &histories);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], 2);
        assert_eq!(tokens[1], 0);
    }

    #[test]
    fn batched_sample_single_sequence_matches_unbatched() {
        let logits = ffi::from_slice_f32(&[0.5, 1.5, 0.3], &[1, 1, 3]);
        let config = SamplingConfig::greedy();
        let configs: Vec<&SamplingConfig> = vec![&config];
        let histories: Vec<&[i32]> = vec![&[]];

        let tokens = batched_sample(&logits, &configs, &histories);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], 1); // argmax of [0.5, 1.5, 0.3] is index 1
    }

    // -- batched fused sampler ([B, vocab] -> [B]) --

    #[test]
    fn batched_fused_sample_greedy_matches_per_row() {
        // Four rows with distinct argmax positions (no ties), shape [B, 1, V].
        // Row 0: argmax 2, Row 1: argmax 0, Row 2: argmax 4, Row 3: argmax 1.
        #[rustfmt::skip]
        let flat = [
            0.1f32, 0.9, 1.2, 0.3, 0.0,
            2.0,    0.5, 0.1, 0.2, 0.3,
            0.0,    0.1, 0.2, 0.3, 1.5,
            0.4,    1.8, 0.2, 0.1, 0.0,
        ];
        let logits = ffi::from_slice_f32(&flat, &[4, 1, 5]);
        let greedy = SamplingConfig::greedy();

        // New fused path: one fused_sample dispatch + one host copy for all B.
        let params = FusedSampleParams::from_config(&greedy);
        let fused = batched_fused_sample(&logits, &params);

        // Per-row reference path (one eval/sync per row).
        let configs: Vec<&SamplingConfig> = vec![&greedy; 4];
        let histories: Vec<&[i32]> = vec![&[]; 4];
        let per_row = batched_sample(&logits, &configs, &histories);

        // Greedy output must be byte-identical to the per-row path.
        assert_eq!(fused, per_row);
        assert_eq!(fused, vec![2, 0, 4, 1]);
    }

    #[test]
    fn batched_fused_sample_single_row_no_regression() {
        // B=1 must match the per-row path exactly (argmax of [0.5,1.5,0.3] = 1).
        let logits = ffi::from_slice_f32(&[0.5, 1.5, 0.3], &[1, 1, 3]);
        let greedy = SamplingConfig::greedy();
        let params = FusedSampleParams::from_config(&greedy);
        let fused = batched_fused_sample(&logits, &params);
        assert_eq!(fused, vec![1]);
    }

    #[test]
    fn batched_fused_sample_accepts_2d_logits() {
        // A 2-D [B, V] input (already last-sliced) must work unchanged.
        let logits = ffi::from_slice_f32(&[0.1, 0.2, 0.9, 1.0, 0.0, 0.0], &[2, 3]);
        let greedy = SamplingConfig::greedy();
        let params = FusedSampleParams::from_config(&greedy);
        let fused = batched_fused_sample(&logits, &params);
        assert_eq!(fused, vec![2, 0]);
    }

    #[test]
    fn config_supports_fused_batch_true_for_plain_configs() {
        assert!(config_supports_fused_batch(&SamplingConfig::greedy()));
        assert!(config_supports_fused_batch(&SamplingConfig::default()));
        assert!(config_supports_fused_batch(
            &SamplingConfig::with_temperature(0.7)
        ));
    }

    #[test]
    fn config_supports_fused_batch_false_for_penalties() {
        let rep = SamplingConfig {
            repetition_penalty: 1.1,
            ..Default::default()
        };
        assert!(!config_supports_fused_batch(&rep));

        let freq = SamplingConfig {
            frequency_penalty: 0.5,
            ..Default::default()
        };
        assert!(!config_supports_fused_batch(&freq));

        let pres = SamplingConfig {
            presence_penalty: 0.5,
            ..Default::default()
        };
        assert!(!config_supports_fused_batch(&pres));

        let dry = SamplingConfig {
            dry_multiplier: 0.8,
            ..Default::default()
        };
        assert!(!config_supports_fused_batch(&dry));
    }

    #[test]
    fn config_supports_fused_batch_false_for_xtc() {
        let xtc = SamplingConfig {
            xtc_probability: 1.0,
            ..Default::default()
        };
        assert!(!config_supports_fused_batch(&xtc));
    }

    #[test]
    fn config_supports_fused_batch_false_for_token_bias() {
        let mut bias = TokenBiasMap::new();
        bias.insert(7, -1.0);
        let cfg = SamplingConfig {
            token_bias: bias,
            ..Default::default()
        };
        assert!(!config_supports_fused_batch(&cfg));
    }

    #[test]
    fn fused_sample_params_match_detects_each_difference() {
        let base = FusedSampleParams::from_config(&SamplingConfig::with_temperature(0.7));
        assert!(base.matches(&base));

        let diff_temp = FusedSampleParams {
            temperature: 0.8,
            ..base
        };
        assert!(!base.matches(&diff_temp));

        let diff_topk = FusedSampleParams { top_k: 40, ..base };
        assert!(!base.matches(&diff_topk));

        let diff_topp = FusedSampleParams { top_p: 0.9, ..base };
        assert!(!base.matches(&diff_topp));

        let diff_minp = FusedSampleParams {
            min_p: 0.05,
            ..base
        };
        assert!(!base.matches(&diff_minp));
    }

    #[test]
    fn row_supports_fused_batch_gate_on_for_plain_row() {
        // Plain greedy row with no per-row obligations joins the fast path.
        assert!(row_supports_fused_batch(
            &SamplingConfig::greedy(),
            false, // no logit mask
            false, // no token override
            false, // no per-token payload
        ));
    }

    #[test]
    fn row_supports_fused_batch_gate_off_for_per_row_obligations() {
        let greedy = SamplingConfig::greedy();
        // Structured-output mask forces the per-row fallback.
        assert!(!row_supports_fused_batch(&greedy, true, false, false));
        // Thinking-budget override forces the per-row fallback.
        assert!(!row_supports_fused_batch(&greedy, false, true, false));
        // Per-token logprobs payload forces the per-row fallback.
        assert!(!row_supports_fused_batch(&greedy, false, false, true));
    }

    #[test]
    fn row_supports_fused_batch_gate_off_for_incompatible_config() {
        // Even with no per-row obligations, a penalty config is not fusible.
        let rep = SamplingConfig {
            repetition_penalty: 1.2,
            ..Default::default()
        };
        assert!(!row_supports_fused_batch(&rep, false, false, false));
    }

    #[test]
    fn compute_logprobs_returns_none_when_disabled() {
        let logits = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let config = LogprobsConfig {
            enabled: false,
            top_k: 0,
            source: Default::default(),
        };
        let result = compute_logprobs(&logits, 2, &config);
        assert!(result.is_none());
    }

    #[test]
    fn compute_logprobs_returns_selected_token_logprob() {
        // Uniform logits -> log-softmax produces equal log-probs for all tokens
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0, 1.0], &[1, 4]);
        let config = LogprobsConfig {
            enabled: true,
            top_k: 0,
            source: Default::default(),
        };
        let result = compute_logprobs(&logits, 2, &config).expect("should return Some");
        assert_eq!(result.token_id, 2);
        // log(1/4) ≈ -1.386
        assert!((result.logprob - (-1.386_f32)).abs() < 0.01);
        assert!(result.top_alternatives.is_empty());
    }

    #[test]
    fn compute_logprobs_returns_top_k_alternatives_sorted_descending() {
        // logits: token 0 has highest, token 2 next, token 1 lowest
        let logits = ffi::from_slice_f32(&[3.0, 0.0, 2.0], &[1, 3]);
        let config = LogprobsConfig {
            enabled: true,
            top_k: 2,
            source: Default::default(),
        };
        // Select token 1 (low logprob) so top-k will include better alternatives
        let result = compute_logprobs(&logits, 1, &config).expect("should return Some");
        assert_eq!(result.token_id, 1);
        assert_eq!(result.top_alternatives.len(), 2);
        // Alternatives must be sorted descending by logprob
        assert!(result.top_alternatives[0].1 >= result.top_alternatives[1].1);
    }

    #[test]
    fn compute_logprobs_top_k_capped_at_vocab_size() {
        // Vocab of 3 tokens, top_k larger than vocab; k is clamped to 3
        let logits = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let config = LogprobsConfig {
            enabled: true,
            top_k: 10,
            source: Default::default(),
        };
        let result = compute_logprobs(&logits, 2, &config).expect("should return Some");
        // top_k is clamped to vocab size (3), so at most 3 alternatives
        assert_eq!(result.top_alternatives.len(), 3);
    }

    // -- f16 / bf16 logprobs regression coverage (issue #340) --
    //
    // Quantized models keep bf16 (and sometimes f16) logits post-#289, so the
    // arrays reaching `compute_logprobs` are 2 bytes per element, not 4. These
    // tests build the logit array at f16/bf16 from the SAME underlying f32
    // values used for an f32 reference run, then assert the top-k and the
    // selected-token logprobs come back as correct f32 values. The pre-fix code
    // read the 2-byte top-k buffer with a hardcoded 4-byte stride, which either
    // overran the slice (the reported server panic) or reinterpreted the bytes
    // as garbage, so even a loose tolerance separates correct from broken.

    // Build a `[1, vocab]` logit array at `target_dtype` from shared f32 values
    // and run `compute_logprobs`. `target_dtype == dtype::FLOAT32` skips the
    // cast so it doubles as the reference run.
    fn logprobs_for_dtype(
        values: &[f32],
        selected_token: i32,
        top_k: usize,
        target_dtype: i32,
    ) -> TokenLogprobData {
        let f32_logits = ffi::from_slice_f32(values, &[1, values.len() as i32]);
        let config = LogprobsConfig {
            enabled: true,
            top_k,
            source: Default::default(),
        };
        if target_dtype == dtype::FLOAT32 {
            compute_logprobs(&f32_logits, selected_token, &config).expect("should return Some")
        } else {
            let logits = ffi::astype(&f32_logits, target_dtype);
            compute_logprobs(&logits, selected_token, &config).expect("should return Some")
        }
    }

    // Shared logit row for the dtype tests: 6 distinct logits. Descending
    // logprob order by logit value is token 1 (3.0) > 3 (2.0) > 2 (1.0) >
    // 0 (0.5) > 5 (0.0) > 4 (-1.0). Selecting token 4 (the lowest) lets the
    // top-5 alternatives be the 5 highest, reproducing the `logprobs: 5`
    // request that crashed the server.
    const DTYPE_LOGITS: [f32; 6] = [0.5, 3.0, 1.0, 2.0, -1.0, 0.0];
    const SELECTED_LOWEST: i32 = 4; // token with logit -1.0

    // Assert a top-k run matches the f32 reference: same count, sorted
    // descending, same top-token id, and per-value agreement within `tol`.
    fn assert_top_k_matches_reference(
        result: &TokenLogprobData,
        reference: &TokenLogprobData,
        tol: f32,
    ) {
        // (b) count matches the reference (and the requested k).
        assert_eq!(
            result.top_alternatives.len(),
            reference.top_alternatives.len(),
            "alternative count must match the f32 reference"
        );
        // (c) alternatives are sorted descending by logprob.
        for w in result.top_alternatives.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "alternatives must be sorted descending: {:?}",
                result.top_alternatives
            );
        }
        // Identity of the top token matches the f32 reference.
        assert_eq!(
            result.top_alternatives[0].0, reference.top_alternatives[0].0,
            "top alternative token id must match the f32 reference"
        );
        // (d) each logprob value matches the reference within tolerance. Match
        // by token id rather than by position to stay robust to tie ordering.
        for &(tok, lp) in &result.top_alternatives {
            let ref_lp = reference
                .top_alternatives
                .iter()
                .find(|&&(t, _)| t == tok)
                .map(|&(_, lp)| lp)
                .unwrap_or_else(|| panic!("token {tok} missing from f32 reference set"));
            assert!(
                (lp - ref_lp).abs() <= tol,
                "logprob for token {tok} = {lp} differs from reference {ref_lp} by more than {tol}"
            );
        }
    }

    #[test]
    fn compute_logprobs_top_k_bf16_no_panic_matches_f32() {
        // Unit-level reproduction of the server crash: `top_k = 5` on bf16
        // logits drives the identical top-k path the server hits. The pre-fix
        // code panicked here ("range end index 12 out of range for slice of
        // length 10"); the fix must return correct f32 values instead.
        let reference = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 5, dtype::FLOAT32);
        let result = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 5, dtype::BFLOAT16);
        assert_eq!(result.top_alternatives.len(), 5);
        // bf16 has ~8 mantissa bits, so use a loose absolute tolerance.
        assert_top_k_matches_reference(&result, &reference, 0.1);
        // Highest-logprob alternative is token 1 (logit 3.0).
        assert_eq!(result.top_alternatives[0].0, 1);
    }

    #[test]
    fn compute_logprobs_top_k_f16_matches_f32() {
        let reference = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 5, dtype::FLOAT32);
        let result = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 5, dtype::FLOAT16);
        assert_eq!(result.top_alternatives.len(), 5);
        // f16 has ~10 mantissa bits, so a tighter tolerance still holds.
        assert_top_k_matches_reference(&result, &reference, 0.03);
        assert_eq!(result.top_alternatives[0].0, 1);
    }

    #[test]
    fn compute_logprobs_top_k_f32_values_correct() {
        // f32 reference path: the same top_k = 5 request must return exact
        // values (the dtype cast is a no-op here). Guards that the shared
        // helper and the f32 path agree before comparing dtype runs to it.
        let reference = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 5, dtype::FLOAT32);
        assert_eq!(reference.top_alternatives.len(), 5);
        assert_top_k_matches_reference(&reference, &reference, 1e-5);
        assert_eq!(reference.top_alternatives[0].0, 1);
    }

    #[test]
    fn compute_logprobs_selected_token_bf16_matches_f32() {
        // Selected-token path (top_k = 0) must stay correct on bf16. This path
        // already uses `item_f32`; the test guards it against future refactors.
        let reference = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 0, dtype::FLOAT32);
        let result = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 0, dtype::BFLOAT16);
        assert!(result.top_alternatives.is_empty());
        assert_eq!(result.token_id, SELECTED_LOWEST);
        assert!(
            (result.logprob - reference.logprob).abs() <= 0.1,
            "bf16 selected-token logprob {} differs from f32 reference {} by more than 0.1",
            result.logprob,
            reference.logprob
        );
    }

    #[test]
    fn compute_logprobs_selected_token_f16_matches_f32() {
        let reference = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 0, dtype::FLOAT32);
        let result = logprobs_for_dtype(&DTYPE_LOGITS, SELECTED_LOWEST, 0, dtype::FLOAT16);
        assert!(result.top_alternatives.is_empty());
        assert_eq!(result.token_id, SELECTED_LOWEST);
        assert!(
            (result.logprob - reference.logprob).abs() <= 0.03,
            "f16 selected-token logprob {} differs from f32 reference {} by more than 0.03",
            result.logprob,
            reference.logprob
        );
    }

    // -- TokenBiasMap and apply_token_bias --

    #[test]
    fn apply_token_bias_empty_noop() {
        // Empty bias map must produce bit-exact equal output.
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let logits = ffi::from_slice_f32(&data, &[1, 5]);
        let bias = TokenBiasMap::new();
        let result = apply_token_bias(&logits, &bias);
        ffi::eval(&result);
        for i in 0..5i32 {
            assert_eq!(
                logit_at(&result, i),
                data[i as usize],
                "token {i} should be unchanged"
            );
        }
    }

    #[test]
    fn apply_token_bias_positive_adds() {
        // {5: +2.0} -> logit[5] += 2.0, all others unchanged.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 6]);
        let mut bias = TokenBiasMap::new();
        bias.insert(5, 2.0);
        let result = apply_token_bias(&logits, &bias);
        ffi::eval(&result);
        for i in 0..5i32 {
            assert_eq!(
                logit_at(&result, i),
                i as f32,
                "token {i} should be unchanged"
            );
        }
        assert_eq!(
            logit_at(&result, 5),
            7.0,
            "token 5 should be 5.0 + 2.0 = 7.0"
        );
    }

    #[test]
    fn apply_token_bias_neg_inf_forces_zero_prob() {
        // {3: -inf} -> after softmax, probability at index 3 must be 0.
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0, 1.0, 1.0], &[1, 5]);
        let mut bias = TokenBiasMap::new();
        bias.insert(3, f32::NEG_INFINITY);
        let biased = apply_token_bias(&logits, &bias);
        ffi::eval(&biased);
        let probs = ffi::softmax(&biased, -1);
        ffi::eval(&probs);
        let prob_at_3 = logit_at(&probs, 3);
        assert_eq!(
            prob_at_3, 0.0,
            "probability at suppressed token must be 0.0"
        );
    }

    #[test]
    fn suppress_tokens_forces_neg_inf_and_drives_probability_to_zero() {
        // issue #350: suppress_tokens masks each id to -inf so it can never be
        // sampled, while leaving other tokens (e.g. real EOS) untouched.
        let mut bias = TokenBiasMap::new();
        // Simulate a model's reserved multimodal placeholder ids alongside an
        // existing finite bias that suppression must override.
        bias.insert(2, 5.0);
        bias.suppress_tokens(&[1, 2, 4]);

        for id in [1, 2, 4] {
            let b = *bias.get(&id).expect("suppressed id present");
            assert!(
                b.is_infinite() && b.is_sign_negative(),
                "token {id} must be -inf, got {b}"
            );
        }
        // An id that was never suppressed stays absent (not silenced).
        assert!(bias.get(&3).is_none(), "untouched token must stay absent");

        // After softmax the suppressed indices carry zero probability.
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0, 1.0, 1.0], &[1, 5]);
        let biased = apply_token_bias(&logits, &bias);
        let probs = ffi::softmax(&biased, -1);
        ffi::eval(&probs);
        for id in [1, 2, 4] {
            assert_eq!(
                logit_at(&probs, id),
                0.0,
                "probability at suppressed token {id} must be 0.0"
            );
        }
        // The non-suppressed token 3 keeps positive probability.
        assert!(logit_at(&probs, 3) > 0.0, "token 3 must remain reachable");
    }

    #[test]
    fn suppress_tokens_empty_slice_is_noop() {
        let mut bias = TokenBiasMap::new();
        bias.suppress_tokens(&[]);
        assert!(bias.is_empty(), "empty suppression keeps the baseline path");
    }

    #[test]
    fn apply_token_bias_multiple_entries() {
        // Multiple entries are applied independently and correctly.
        let logits = ffi::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 4]);
        let mut bias = TokenBiasMap::new();
        bias.insert(0, 1.0);
        bias.insert(2, -3.0);
        let result = apply_token_bias(&logits, &bias);
        ffi::eval(&result);
        assert_eq!(logit_at(&result, 0), 1.0, "token 0 should be 0.0 + 1.0");
        assert_eq!(logit_at(&result, 1), 0.0, "token 1 should be unchanged");
        assert_eq!(logit_at(&result, 2), -3.0, "token 2 should be 0.0 - 3.0");
        assert_eq!(logit_at(&result, 3), 0.0, "token 3 should be unchanged");
    }

    #[test]
    fn apply_token_bias_out_of_range_ignored() {
        // Token id >= vocab_size must be silently ignored — no panic.
        let logits = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let mut bias = TokenBiasMap::new();
        bias.insert(100, 99.0); // way beyond vocab_size = 3
        bias.insert(3, 5.0); // exactly vocab_size (off-by-one boundary)
        let result = apply_token_bias(&logits, &bias);
        ffi::eval(&result);
        // Original values unchanged
        assert_eq!(logit_at(&result, 0), 1.0);
        assert_eq!(logit_at(&result, 1), 2.0);
        assert_eq!(logit_at(&result, 2), 3.0);
    }

    #[test]
    fn apply_token_bias_negative_index_ignored() {
        // Negative token ids must be silently ignored — no panic.
        let logits = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let mut bias = TokenBiasMap::new();
        bias.insert(-1, 99.0);
        bias.insert(-100, -50.0);
        let result = apply_token_bias(&logits, &bias);
        ffi::eval(&result);
        // Original values unchanged
        assert_eq!(logit_at(&result, 0), 1.0);
        assert_eq!(logit_at(&result, 1), 2.0);
        assert_eq!(logit_at(&result, 2), 3.0);
    }

    // Helper: extract a flat f32 vector from an MlxArray.
    fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
        ffi::eval(a);
        let bytes = ffi::array_to_raw_bytes(a);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }

    // Helper: extract row `row` of a 2-D [B, V] MlxArray as Vec<f32>.
    fn row_vec(a: &MlxArray, row: i32, v: i32) -> Vec<f32> {
        let row_arr = ffi::slice(a, &[row, 0], &[row + 1, v]);
        to_vec_f32(&row_arr)
    }

    #[test]
    fn top_p_filter_single_row_nucleus_boundary() {
        // Row with 5 tokens. Logits are large enough that softmax concentrates
        // probability on the first two tokens. With p=0.7 the nucleus should
        // include the top-1 and top-2 tokens and exclude the rest.
        //
        // logits: [10.0, 8.0, 1.0, 1.0, 1.0]
        // After softmax (approx): token0 ≈ 0.88, token1 ≈ 0.12, others ≈ 0
        // Sorted descending: [0.88, 0.12, ~0, ~0, ~0]
        // Exclusive cumsum:  [0.00, 0.88, ~1,  ~1,  ~1 ]
        // Mask (<=0.7):      [true, false, ...]
        // Only token0 should survive with p=0.7.
        let logits = ffi::from_slice_f32(&[10.0, 8.0, 1.0, 1.0, 1.0], &[1, 5]);
        let result = top_p_filter(&logits, 0.7);
        ffi::eval(&result);

        // token0 (argmax) must survive; all others must be -inf.
        let v = to_vec_f32(&result);
        assert!(
            v[0].is_finite(),
            "top token should survive nucleus (got {})",
            v[0]
        );
        for (i, val) in v.iter().enumerate().take(5).skip(1) {
            assert!(
                val.is_infinite() && *val < 0.0,
                "token {i} should be filtered to -inf (got {val})",
            );
        }
    }

    #[test]
    fn top_p_filter_single_row_all_pass_at_one() {
        // With p=1.0 all tokens should survive (no filtering).
        let logits = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let result = top_p_filter(&logits, 1.0);
        ffi::eval(&result);
        let v = to_vec_f32(&result);
        for (i, &x) in v.iter().enumerate() {
            assert!(
                x.is_finite(),
                "all tokens should survive with p=1.0 (token {i} got {x})"
            );
        }
    }

    #[test]
    fn top_p_filter_batched_equals_per_row() {
        // Regression test (upstream mlx-vlm PR #1094, commit c7aaf2d).
        //
        // Construct a [2, 6] logits tensor where the two rows have deliberately
        // different distributions so a buggy global sort would give wrong results.
        //
        // Row 0: token 2 is the most probable; token 0 dominates next.
        // Row 1: token 5 is the most probable; token 3 dominates next.
        //
        // Running top_p_filter on the full [2, 6] batch must produce the same
        // filtered logits as running it on each [1, 6] row independently.
        let row0 = [1.0f32, 2.0, 10.0, 1.0, 1.0, 1.0]; // token2 dominates
        let row1 = [1.0f32, 1.0, 1.0, 2.0, 1.0, 10.0]; // token5 dominates
        let flat: Vec<f32> = row0.iter().chain(row1.iter()).copied().collect();

        // Batched call.
        let batch_logits = ffi::from_slice_f32(&flat, &[2, 6]);
        let p = 0.8_f32;
        let batch_result = top_p_filter(&batch_logits, p);
        ffi::eval(&batch_result);

        // Per-row calls.
        let logits0 = ffi::from_slice_f32(&row0, &[1, 6]);
        let logits1 = ffi::from_slice_f32(&row1, &[1, 6]);
        let result0 = top_p_filter(&logits0, p);
        let result1 = top_p_filter(&logits1, p);
        ffi::eval(&result0);
        ffi::eval(&result1);

        let batch_row0 = row_vec(&batch_result, 0, 6);
        let batch_row1 = row_vec(&batch_result, 1, 6);
        let solo_row0 = to_vec_f32(&result0);
        let solo_row1 = to_vec_f32(&result1);

        // Each batched row must match its corresponding solo result within
        // floating-point tolerance (1e-5).
        for i in 0..6 {
            let b0 = batch_row0[i];
            let s0 = solo_row0[i];
            if b0.is_infinite() && s0.is_infinite() {
                // Both filtered to -inf — correct.
            } else {
                assert!(
                    (b0 - s0).abs() < 1e-5,
                    "row0 token {i}: batched={b0} vs per-row={s0} mismatch"
                );
            }

            let b1 = batch_row1[i];
            let s1 = solo_row1[i];
            if b1.is_infinite() && s1.is_infinite() {
                // Both filtered to -inf — correct.
            } else {
                assert!(
                    (b1 - s1).abs() < 1e-5,
                    "row1 token {i}: batched={b1} vs per-row={s1} mismatch"
                );
            }
        }
    }

    // -- incremental SamplerState parity (issue #328) --

    // Astype to f32 and pull the values to host, so f16/bf16 penalty outputs can
    // be compared regardless of their native dtype.
    fn logits_to_f32_vec(a: &MlxArray) -> Vec<f32> {
        let f = ffi::astype(a, dtype::FLOAT32);
        to_vec_f32(&f)
    }

    // Bit-for-bit equality of two logit arrays (compared as f32). The
    // incremental path must reproduce the rebuild path exactly so penalty-
    // adjusted greedy sampling picks identical tokens.
    fn assert_logits_bit_identical(a: &MlxArray, b: &MlxArray, ctx: &str) {
        let va = logits_to_f32_vec(a);
        let vb = logits_to_f32_vec(b);
        assert_eq!(va.len(), vb.len(), "{ctx}: length mismatch");
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{ctx}: element {i} differs: {x} vs {y}"
            );
        }
    }

    // Distinct, mostly-nonzero logits used by the parity sweeps.
    const PARITY_LOGITS: [f32; 10] = [0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -0.5, 0.25, 1.1, 0.9];

    // A history with repeats so both repetition (unique set) and
    // frequency/presence (counts) inputs are exercised.
    const PARITY_HISTORY: [i32; 12] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 1];

    #[test]
    fn sampler_state_repetition_matches_rebuild_over_sequence() {
        let penalty = 1.3_f32;
        let cfg = SamplingConfig {
            repetition_penalty: penalty,
            ..Default::default()
        };
        let mut state = SamplerState::for_config(&cfg);
        for len in 1..=PARITY_HISTORY.len() {
            let h = &PARITY_HISTORY[..len];
            let rebuilt = apply_repetition_penalty(
                &ffi::from_slice_f32(&PARITY_LOGITS, &[1, 10]),
                h,
                penalty,
            );
            state.sync(h);
            let incremental =
                state.apply_repetition(&ffi::from_slice_f32(&PARITY_LOGITS, &[1, 10]), penalty);
            assert_logits_bit_identical(&rebuilt, &incremental, &format!("repetition len {len}"));
        }
    }

    #[test]
    fn sampler_state_repetition_matches_rebuild_f16() {
        // f16 logits exercise the same `put_along_axis` dtype path through both
        // routes; the shared core keeps them bit-identical.
        let penalty = 1.4_f32;
        let cfg = SamplingConfig {
            repetition_penalty: penalty,
            ..Default::default()
        };
        let mut state = SamplerState::for_config(&cfg);
        let h = &PARITY_HISTORY[..];
        state.sync(h);
        let f16 = ffi::astype(
            &ffi::from_slice_f32(&PARITY_LOGITS, &[1, 10]),
            dtype::FLOAT16,
        );
        let rebuilt = apply_repetition_penalty(&f16, h, penalty);
        let incremental = state.apply_repetition(&f16, penalty);
        assert_logits_bit_identical(&rebuilt, &incremental, "repetition f16");
    }

    #[test]
    fn sampler_state_frequency_presence_matches_rebuild_over_sequence() {
        let (freq, pres) = (0.7_f32, 0.3_f32);
        let cfg = SamplingConfig {
            frequency_penalty: freq,
            presence_penalty: pres,
            ..Default::default()
        };
        let mut state = SamplerState::for_config(&cfg);
        for len in 1..=PARITY_HISTORY.len() {
            let h = &PARITY_HISTORY[..len];
            let rebuilt = apply_frequency_presence_penalty(
                &ffi::from_slice_f32(&PARITY_LOGITS, &[1, 10]),
                h,
                freq,
                pres,
            );
            state.sync(h);
            let incremental = state.apply_frequency_presence(
                &ffi::from_slice_f32(&PARITY_LOGITS, &[1, 10]),
                freq,
                pres,
            );
            assert_logits_bit_identical(
                &rebuilt,
                &incremental,
                &format!("frequency/presence len {len}"),
            );
        }
    }

    #[test]
    fn sampler_state_frequency_presence_matches_rebuild_f16() {
        // The rebuild path's broadcast subtract promotes f16 logits to f32. The
        // sparse path must promote first to stay bit-identical.
        let (freq, pres) = (0.6_f32, 0.25_f32);
        let cfg = SamplingConfig {
            frequency_penalty: freq,
            presence_penalty: pres,
            ..Default::default()
        };
        let mut state = SamplerState::for_config(&cfg);
        let h = &PARITY_HISTORY[..];
        state.sync(h);
        let f16 = ffi::astype(
            &ffi::from_slice_f32(&PARITY_LOGITS, &[1, 10]),
            dtype::FLOAT16,
        );
        let rebuilt = apply_frequency_presence_penalty(&f16, h, freq, pres);
        let incremental = state.apply_frequency_presence(&f16, freq, pres);
        assert_logits_bit_identical(&rebuilt, &incremental, "frequency/presence f16");
    }

    #[test]
    fn sampler_state_sync_handles_append_shrink_and_divergence() {
        let cfg = SamplingConfig {
            repetition_penalty: 1.2,
            frequency_penalty: 0.5,
            ..Default::default()
        };
        let mut s = SamplerState::for_config(&cfg);

        // Append-only growth.
        s.sync(&[1, 2, 2, 3]);
        assert_eq!(s.absorbed_len, 4);
        assert_eq!(s.seen_sorted, vec![1, 2, 3]);
        assert_eq!(s.counts.get(&2), Some(&2));

        // Further append reuses the state (tip matches).
        s.sync(&[1, 2, 2, 3, 3, 1]);
        assert_eq!(s.absorbed_len, 6);
        assert_eq!(s.seen_sorted, vec![1, 2, 3]);
        assert_eq!(s.counts.get(&1), Some(&2));
        assert_eq!(s.counts.get(&3), Some(&2));

        // Shrink (cache trim / rollback) rebuilds to the shorter history.
        s.sync(&[1, 2]);
        assert_eq!(s.absorbed_len, 2);
        assert_eq!(s.seen_sorted, vec![1, 2]);
        assert_eq!(s.counts.get(&2), Some(&1));
        assert_eq!(s.counts.get(&3), None);

        // Same length but a diverged tip also rebuilds.
        s.sync(&[7, 8]);
        assert_eq!(s.seen_sorted, vec![7, 8]);
        assert_eq!(s.counts.get(&1), None);
        assert_eq!(s.counts.get(&7), Some(&1));
    }

    #[test]
    fn sample_token_optimized_with_state_greedy_parity_over_sequence() {
        // Greedy sampling with every history-based penalty active (repetition +
        // DRY + frequency + presence). The state-backed path must select the
        // same token id as the rebuild path at every step.
        let cfg = SamplingConfig {
            repetition_penalty: 1.5,
            dry_multiplier: 0.8,
            dry_base: 1.75,
            dry_allowed_length: 2,
            frequency_penalty: 0.8,
            presence_penalty: 0.5,
            ..SamplingConfig::greedy()
        };
        let vocab = 12usize;
        let mut history: Vec<i32> = vec![2, 5, 2, 7, 5];
        let mut state: Option<SamplerState> = None;

        for step in 0..40i32 {
            let vals: Vec<f32> = (0..vocab as i32)
                .map(|i| ((step * 31 + i * 17) % 23) as f32 * 0.3 - 3.0)
                .collect();
            let logits_a = ffi::from_slice_f32(&vals, &[1, 1, vocab as i32]);
            let logits_b = ffi::from_slice_f32(&vals, &[1, 1, vocab as i32]);

            let (tok_a, _) = sample_token_optimized(&logits_a, &cfg, &history);
            let (tok_b, _) =
                sample_token_optimized_with_state(&logits_b, &cfg, &history, &mut state);
            ffi::eval(&tok_a);
            ffi::eval(&tok_b);
            let a = ffi::item_i32(&tok_a);
            let b = ffi::item_i32(&tok_b);
            assert_eq!(a, b, "token mismatch at step {step}");
            history.push(a);
        }
        // State was created because penalties are active.
        assert!(state.is_some());
    }

    #[test]
    fn sample_token_optimized_with_state_no_penalty_allocates_no_state() {
        // The default no-penalty path must take the original fast path and never
        // allocate per-sequence state.
        let logits = ffi::from_slice_f32(&[0.1, 0.9, 1.2], &[1, 1, 3]);
        let cfg = SamplingConfig::greedy();
        let mut state: Option<SamplerState> = None;
        let (token, _) = sample_token_optimized_with_state(&logits, &cfg, &[], &mut state);
        ffi::eval(&token);
        assert_eq!(ffi::item_i32(&token), 2);
        assert!(
            state.is_none(),
            "no-penalty path must not allocate sampler state"
        );

        // And it agrees with the plain entry point.
        let logits2 = ffi::from_slice_f32(&[0.1, 0.9, 1.2], &[1, 1, 3]);
        let (token2, _) = sample_token_optimized(&logits2, &cfg, &[]);
        ffi::eval(&token2);
        assert_eq!(ffi::item_i32(&token2), 2);
    }

    #[test]
    fn sample_token_optimized_with_state_dry_only_allocates_no_state() {
        // DRY is intentionally not state-backed, so a DRY-only config must not
        // allocate state, and its output must match the rebuild path exactly.
        let cfg = SamplingConfig {
            dry_multiplier: 1.0,
            dry_base: 2.0,
            dry_allowed_length: 1,
            ..SamplingConfig::greedy()
        };
        let history = [0, 1, 2, 0, 1];
        let logits_a = ffi::from_slice_f32(&[1.0, 1.0, 1.0], &[1, 1, 3]);
        let logits_b = ffi::from_slice_f32(&[1.0, 1.0, 1.0], &[1, 1, 3]);
        let mut state: Option<SamplerState> = None;
        let (tok_a, _) = sample_token_optimized(&logits_a, &cfg, &history);
        let (tok_b, _) = sample_token_optimized_with_state(&logits_b, &cfg, &history, &mut state);
        ffi::eval(&tok_a);
        ffi::eval(&tok_b);
        assert_eq!(ffi::item_i32(&tok_a), ffi::item_i32(&tok_b));
        assert!(
            state.is_none(),
            "DRY-only path is not state-backed and must not allocate state"
        );
    }

    #[test]
    fn compute_logprobs_top_k_zero_fast_path_matches_full_softmax() {
        // The `top_k == 0` fast path (logit - logsumexp) must match the full
        // log-softmax gather within tight floating-point tolerance.
        let logits = ffi::from_slice_f32(&DTYPE_LOGITS, &[1, DTYPE_LOGITS.len() as i32]);
        let cfg = LogprobsConfig {
            enabled: true,
            top_k: 0,
            source: Default::default(),
        };
        let fast = compute_logprobs(&logits, SELECTED_LOWEST, &cfg).expect("should return Some");
        assert!(fast.top_alternatives.is_empty());

        // Reference: full log-softmax then gather the selected token.
        let log_probs = ffi::log_softmax(&logits, -1);
        let idx = ffi::from_slice_i32(&[SELECTED_LOWEST], &[1, 1]);
        let sel = ffi::take_along_axis(&log_probs, &idx, -1);
        ffi::eval(&sel);
        let full = ffi::item_f32(&sel);

        assert!(
            (fast.logprob - full).abs() < 1e-5,
            "fast-path logprob {} differs from full log-softmax {}",
            fast.logprob,
            full
        );
    }

    #[test]
    fn compute_logprobs_top_k_zero_fast_path_f16_matches_full_softmax() {
        // Same parity check as above but with f16 logits. Quantized models
        // produce f16 logits post-#289, so this guards the fast path (gather +
        // logsumexp) against the full log-softmax on the same f16 input. Both
        // operate in f16 arithmetic; the result is cast to f32 only at the read
        // boundary, so the two paths must agree within f16 precision (~0.01).
        let f32_logits = ffi::from_slice_f32(&DTYPE_LOGITS, &[1, DTYPE_LOGITS.len() as i32]);
        let f16_logits = ffi::astype(&f32_logits, dtype::FLOAT16);
        let cfg = LogprobsConfig {
            enabled: true,
            top_k: 0,
            source: Default::default(),
        };
        let fast =
            compute_logprobs(&f16_logits, SELECTED_LOWEST, &cfg).expect("should return Some");
        assert!(fast.top_alternatives.is_empty());

        // Reference: full log-softmax on the same f16 logits, then cast to f32 for reading.
        let log_probs = ffi::log_softmax(&f16_logits, -1);
        let idx = ffi::from_slice_i32(&[SELECTED_LOWEST], &[1, 1]);
        let sel = ffi::take_along_axis(&log_probs, &idx, -1);
        let sel_f32 = ffi::astype(&sel, dtype::FLOAT32);
        ffi::eval(&sel_f32);
        let full = ffi::item_f32(&sel_f32);

        assert!(
            (fast.logprob - full).abs() < 0.01,
            "f16 fast-path logprob {} differs from full log-softmax {} by more than 0.01",
            fast.logprob,
            full
        );
    }

    // -- XTC (Exclude Top Choices) filter --

    /// Logits whose softmax probabilities are roughly: token 0 ~0.665,
    /// token 1 ~0.245, token 2 ~0.090, token 3 ~1.4e-9. Tokens 0-2 clearly
    /// exceed a 0.01 threshold; token 3 clearly does not. Only token 0
    /// exceeds a 0.5 threshold.
    fn xtc_test_logits() -> UniquePtr<MlxArray> {
        ffi::from_slice_f32(&[10.0, 9.0, 8.0, -10.0], &[1, 4])
    }

    #[test]
    fn apply_xtc_filter_keeps_least_probable_above_threshold_token() {
        let logits = xtc_test_logits();
        let result = apply_xtc_filter(&logits, 0.01, &[]);

        // Tokens 0 and 1 are above threshold and not the least-probable of
        // the three, so both are removed. Token 2 is the least-probable
        // above-threshold token and is kept. Token 3 never exceeded the
        // threshold and is untouched either way.
        assert_eq!(logit_at(&result, 0), f32::NEG_INFINITY);
        assert_eq!(logit_at(&result, 1), f32::NEG_INFINITY);
        assert_eq!(logit_at(&result, 2), 8.0);
        assert_eq!(logit_at(&result, 3), -10.0);
    }

    #[test]
    fn apply_xtc_filter_allowlist_tokens_survive_removal() {
        let logits = xtc_test_logits();
        // Token 0 would otherwise be removed (above threshold, not the
        // least-probable); the allowlist must keep it intact.
        let result = apply_xtc_filter(&logits, 0.01, &[0]);

        assert_eq!(logit_at(&result, 0), 10.0);
        assert_eq!(logit_at(&result, 1), f32::NEG_INFINITY);
        assert_eq!(logit_at(&result, 2), 8.0);
        assert_eq!(logit_at(&result, 3), -10.0);
    }

    #[test]
    fn apply_xtc_filter_is_noop_with_fewer_than_two_candidates() {
        let logits = xtc_test_logits();
        // threshold 0.5: only token 0 (~0.665) exceeds it, so the filter
        // must not remove anything.
        let result = apply_xtc_filter(&logits, 0.5, &[]);

        assert_eq!(logit_at(&result, 0), 10.0);
        assert_eq!(logit_at(&result, 1), 9.0);
        assert_eq!(logit_at(&result, 2), 8.0);
        assert_eq!(logit_at(&result, 3), -10.0);
    }

    #[test]
    fn apply_xtc_filter_is_noop_with_zero_candidates() {
        let logits = xtc_test_logits();
        // threshold 0.9: no token exceeds it.
        let result = apply_xtc_filter(&logits, 0.9, &[]);

        assert_eq!(logit_at(&result, 0), 10.0);
        assert_eq!(logit_at(&result, 1), 9.0);
        assert_eq!(logit_at(&result, 2), 8.0);
        assert_eq!(logit_at(&result, 3), -10.0);
    }

    #[test]
    fn apply_xtc_step_gate_at_zero_never_fires() {
        let config = SamplingConfig {
            xtc_probability: 0.0,
            xtc_threshold: 0.01,
            ..Default::default()
        };
        // Try several seeds: a probability-0.0 gate must never fire
        // regardless of the drawn uniform sample.
        for seed in [1u64, 2, 3] {
            ffi::random_seed(seed);
            let logits = xtc_test_logits();
            let result = apply_xtc_step(&logits, &config);
            assert_eq!(logit_at(&result, 0), 10.0);
            assert_eq!(logit_at(&result, 1), 9.0);
        }
    }

    #[test]
    fn apply_xtc_step_gate_at_one_always_fires() {
        let config = SamplingConfig {
            xtc_probability: 1.0,
            xtc_threshold: 0.01,
            ..Default::default()
        };
        // Uniform samples are drawn from [0, 1), so a probability-1.0 gate
        // must always fire regardless of the drawn value.
        for seed in [11u64, 12, 13] {
            ffi::random_seed(seed);
            let logits = xtc_test_logits();
            let result = apply_xtc_step(&logits, &config);
            assert_eq!(logit_at(&result, 0), f32::NEG_INFINITY);
            assert_eq!(logit_at(&result, 1), f32::NEG_INFINITY);
            assert_eq!(logit_at(&result, 2), 8.0);
        }
    }

    #[test]
    fn apply_xtc_step_mid_probability_is_reproducible_for_the_same_seed() {
        let config = SamplingConfig {
            xtc_probability: 0.5,
            xtc_threshold: 0.01,
            ..Default::default()
        };

        ffi::random_seed(42);
        let result_a = apply_xtc_step(&xtc_test_logits(), &config);
        let a0 = logit_at(&result_a, 0);
        let a1 = logit_at(&result_a, 1);

        // Re-seeding with the same value must reproduce the same gate
        // outcome (and therefore the same resulting logits) deterministically.
        ffi::random_seed(42);
        let result_b = apply_xtc_step(&xtc_test_logits(), &config);
        let b0 = logit_at(&result_b, 0);
        let b1 = logit_at(&result_b, 1);

        assert_eq!(a0, b0);
        assert_eq!(a1, b1);
    }

    // -- #1379: filters on the untempered distribution, temperature last --

    /// f32 values of a `[1, n]` probability tensor.
    fn probs_row(arr: &MlxArray) -> Vec<f32> {
        ffi::array_to_raw_bytes(arr)
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Indices with non-zero mass.
    fn support_of(probs: &[f32]) -> Vec<usize> {
        (0..probs.len()).filter(|&i| probs[i] > 0.0).collect()
    }

    /// Snapshot comparison for probability rows on backends that can move a
    /// last-ULP reduction bit without changing the sampler's effective
    /// distribution.
    fn assert_probs_match_snapshot_within_ulp(got: &[f32], expected_bits: &[u32], ctx: &str) {
        assert_eq!(got.len(), expected_bits.len(), "{ctx}: length mismatch");

        let expected: Vec<f32> = expected_bits.iter().copied().map(f32::from_bits).collect();
        assert_eq!(
            support_of(got),
            support_of(&expected),
            "{ctx}: support changed"
        );

        for (i, (&observed, &want_bits)) in got.iter().zip(expected_bits).enumerate() {
            if want_bits == 0 {
                assert_eq!(observed.to_bits(), 0, "{ctx}: token {i} left zero support");
                continue;
            }
            assert!(
                observed.is_finite() && observed > 0.0,
                "{ctx}: token {i} left the positive finite support: {observed}"
            );
            let ulp = observed.to_bits().abs_diff(want_bits);
            assert!(
                ulp <= 2,
                "{ctx}: token {i} moved by {ulp} ulp ({observed} vs {})",
                f32::from_bits(want_bits)
            );
        }
    }

    /// `[1, 4]` logits whose softmax is `[0.50, 0.30, 0.15, 0.05]`.
    fn nucleus_test_logits() -> UniquePtr<MlxArray> {
        let row: Vec<f32> = [0.5f32, 0.3, 0.15, 0.05].iter().map(|p| p.ln()).collect();
        ffi::from_slice_f32(&row, &[1, 4])
    }

    #[test]
    fn top_p_support_is_temperature_invariant() {
        // The untempered nucleus at top_p = 0.9 keeps {0, 1, 2} (exclusive
        // cumulative mass 0.95 > 0.9 excludes index 3). Before #1379 the
        // filter saw the TEMPERED distribution, so T = 0.5 sharpened the row
        // and dropped index 2 while T = 2.0 flattened it and admitted index 3.
        let logits = nucleus_test_logits();
        for t in [0.5f32, 1.0, 2.0] {
            let probs = probs_row(&ffi::fused_sample_probs(&logits, t, 0, 0.9, 0.0));
            assert_eq!(
                support_of(&probs),
                vec![0, 1, 2],
                "top-p support moved with the temperature at T={t}: {probs:?}"
            );
        }
    }

    #[test]
    fn min_p_support_is_temperature_invariant() {
        // min-p 0.2 on the untempered row: threshold 0.2 * 0.5 = 0.1, so
        // {0, 1, 2} survive (0.15 >= 0.1) and index 3 does not. The test is
        // renormalisation-invariant but temperature-variant, which is why it
        // must read the untempered row like the others.
        let logits = nucleus_test_logits();
        for t in [0.5f32, 1.0, 2.0] {
            let probs = probs_row(&ffi::fused_sample_probs(&logits, t, 0, 1.0, 0.2));
            assert_eq!(
                support_of(&probs),
                vec![0, 1, 2],
                "min-p support moved with the temperature at T={t}: {probs:?}"
            );
        }
    }

    /// Deterministic pseudo-random `[1, n]` logits row over ~12 nats.
    fn lcg_logits(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed;
        let mut row = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 33) as f64) / f64::from(u32::MAX >> 1);
            row.push((unit * 12.0 - 6.0) as f32);
        }
        row
    }

    #[test]
    fn fused_sample_probs_equals_softmax_of_filtered_over_t() {
        // The reported distribution must be softmax(where(support, x, -inf)/T)
        // where `support` comes from the Rust reference filters applied to the
        // UNTEMPERED row in chain order. Configurations avoid the joint
        // top-k + top-p combination so the stock and rejection-kernel
        // semantics coincide and the reference is backend-independent.
        let t = 0.7f32;
        for (seed, top_k, top_p, min_p) in [
            (0xA11CEu64, 0i32, 0.9f32, 0.0f32),
            (0xB0B5Eu64, 0, 0.7, 0.2),
            (0xC4A7u64, 40, 1.0, 0.05),
        ] {
            let row = lcg_logits(seed, 64);
            let logits = ffi::from_slice_f32(&row, &[1, row.len() as i32]);

            // Reference support: chain order on the raw row.
            let mut filtered = ffi::astype(&logits, dtype::FLOAT32);
            if top_k > 0 {
                filtered = top_k_filter(&filtered, top_k);
            }
            if top_p > 0.0 && top_p < 1.0 {
                filtered = top_p_filter(&filtered, top_p);
            }
            if min_p > 0.0 && min_p < 1.0 {
                filtered = min_p_filter(&filtered, min_p);
            }
            let kept: Vec<bool> = probs_row(&filtered).iter().map(|v| v.is_finite()).collect();

            // Expected distribution in f64: softmax(x_kept / T).
            let scaled: Vec<f64> = row.iter().map(|&x| f64::from(x) / f64::from(t)).collect();
            let max = scaled
                .iter()
                .zip(&kept)
                .filter(|(_, k)| **k)
                .map(|(&s, _)| s)
                .fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = scaled
                .iter()
                .zip(&kept)
                .map(|(&s, &k)| if k { (s - max).exp() } else { 0.0 })
                .collect();
            let z: f64 = exp.iter().sum();
            let expected: Vec<f64> = exp.iter().map(|&e| e / z).collect();

            let probs = probs_row(&ffi::fused_sample_probs(&logits, t, top_k, top_p, min_p));
            for (i, (&got, &want)) in probs.iter().zip(&expected).enumerate() {
                assert!(
                    (f64::from(got) - want).abs() < 1e-5,
                    "top_k={top_k} top_p={top_p} min_p={min_p}: token {i} reported {got}, \
                     reference {want}"
                );
            }
        }
    }

    #[test]
    fn greedy_path_unchanged() {
        // T == 0 and top_k == 1 both return the argmax of the penalized
        // logits, untouched by #1379, for any filter combination.
        for seed in [0x9EED1u64, 0x9EED2, 0x9EED3] {
            let row = lcg_logits(seed, 128);
            let logits = ffi::from_slice_f32(&row, &[1, row.len() as i32]);
            let host_argmax = (0..row.len())
                .max_by(|&a, &b| row[a].partial_cmp(&row[b]).expect("finite"))
                .expect("non-empty") as i32;

            for (t, top_k) in [(0.0f32, 40i32), (0.0, 0), (0.7, 1)] {
                let tok = ffi::fused_sample(&logits, t, top_k, 0.9, 0.1);
                let b = ffi::array_to_raw_bytes(&tok);
                let id = i32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
                assert_eq!(
                    id, host_argmax,
                    "greedy config (T={t}, top_k={top_k}) diverged from argmax"
                );
            }
        }
    }

    /// Pre-#1379 `fused_sample_probs` probability rows at `T == 1.0`, captured
    /// from commit e989e45da on the Metal backend for min-p-free
    /// configurations. `T == 1.0` applies no temperature division on either
    /// side of #1379, so support and probability mass should stay fixed across
    /// the reorder.
    ///
    /// min-p-active configurations are deliberately absent: through #1378 the
    /// stock chain's compiled min-p filter was a silent no-op on Metal (see
    /// the min-p block in `fused_sample_filter_logits`), so their pre-change
    /// bits pin a defect, not a contract. `min_p_filters_the_stock_chain` in
    /// sampling_rejection_tests.rs pins the fixed behavior instead.
    ///
    /// The two top-p cases are guarded on the rejection backend: they route to
    /// the kernel, whose intersection semantics the snapshot was captured
    /// under, so they only mean anything where the kernel exists. The third
    /// case `(top_k = 40, top_p = 1.0)` is NOT guarded. Routing requires an
    /// active top-p (`rejection_sample_applies`), so that case takes the stock
    /// argpartition chain on every backend and is a valid regression check
    /// everywhere. Guarding the whole test on the kernel, as it was first
    /// written, left the single committed stock-chain check for the reorder
    /// running nowhere at all on a CPU-only build.
    ///
    /// Exact bits are too strong on Metal: MLX can converge the same softmax
    /// row to a different final ULP without changing the support or any
    /// decision boundary the sampler observes, and that has shown up
    /// intermittently in nightly `make verify-test` runs. This snapshot
    /// therefore holds the exact support and keeps every positive probability
    /// within two f32 ULPs of the saved row.
    #[test]
    fn temperature_one_support_unchanged() {
        let row = lcg_logits(0x1379_5EED, 64);
        let logits = ffi::from_slice_f32(&row, &[1, row.len() as i32]);

        // `routed`: true when the configuration reaches the rejection kernel.
        let cases: [(i32, f32, bool, Vec<u32>); 3] = [
            // The joint (40, 0.9) row equals the (0, 0.9) row bit for bit in
            // the capture: on this 64-token row the 0.9 nucleus holds 15
            // tokens, a strict subset of the top-40 set, so the kernel's
            // intersection semantics reduce to the plain nucleus.
            (
                40,
                0.9,
                true,
                vec![
                    0x00000000, 0x3e507f8c, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x3ce40600, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3d08d9cb, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x3d3f3109, 0x3db9baff, 0x00000000, 0x3cc3616a,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x3dabc508, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x3e2d97e5, 0x00000000, 0x3cc2352f,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3e334131, 0x00000000,
                    0x3d44a814, 0x00000000, 0x3d06223f, 0x00000000, 0x00000000, 0x00000000,
                    0x3d27f9f0, 0x00000000, 0x00000000, 0x00000000,
                ],
            ),
            (
                0,
                0.9,
                true,
                vec![
                    0x00000000, 0x3e507f8c, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x3ce40600, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3d08d9cb, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x3d3f3109, 0x3db9baff, 0x00000000, 0x3cc3616a,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x3dabc508, 0x00000000, 0x00000000, 0x00000000,
                    0x00000000, 0x00000000, 0x00000000, 0x3e2d97e5, 0x00000000, 0x3cc2352f,
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3e334131, 0x00000000,
                    0x3d44a814, 0x00000000, 0x3d06223f, 0x00000000, 0x00000000, 0x00000000,
                    0x3d27f9f0, 0x00000000, 0x00000000, 0x00000000,
                ],
            ),
            (
                40,
                1.0,
                false,
                vec![
                    0x3bbec528, 0x3e3c000c, 0x00000000, 0x00000000, 0x00000000, 0x38d87b5b,
                    0x00000000, 0x3c045883, 0x3c83b0c8, 0x3ccd9b17, 0x00000000, 0x38ccbd8e,
                    0x3ad63cff, 0x00000000, 0x00000000, 0x3ba91865, 0x3cf6cb0b, 0x3aa825ec,
                    0x3a28f9fa, 0x00000000, 0x3a5b3f4b, 0x3b9e73a3, 0x00000000, 0x3a3249b7,
                    0x00000000, 0x00000000, 0x3d2c651d, 0x3da77885, 0x00000000, 0x3cb02c10,
                    0x00000000, 0x3a01edd8, 0x00000000, 0x00000000, 0x3b492638, 0x00000000,
                    0x00000000, 0x00000000, 0x3d9ae1eb, 0x3a08a211, 0x3b203923, 0x00000000,
                    0x3ca0964f, 0x3b985ac4, 0x00000000, 0x3e1c86e2, 0x3b1c3010, 0x3caf1d59,
                    0x00000000, 0x3b9c3542, 0x3aee9c92, 0x3b5b3bc3, 0x3e21a1b2, 0x3b774dee,
                    0x3d31529c, 0x38ff1229, 0x3cf1e4b6, 0x395dc1b0, 0x00000000, 0x3bbc077d,
                    0x3d17764c, 0x38f7d749, 0x00000000, 0x00000000,
                ],
            ),
        ];
        for (top_k, top_p, routed, expected) in cases {
            if routed && !ffi::sampling_rejection_available() {
                continue;
            }
            let probs = ffi::fused_sample_probs(&logits, 1.0, top_k, top_p, 0.0);
            let got: Vec<f32> = ffi::array_to_raw_bytes(&probs)
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_probs_match_snapshot_within_ulp(
                &got,
                &expected,
                &format!("T=1.0 fused_sample_probs drifted for top_k={top_k} top_p={top_p}"),
            );
        }
    }

    #[test]
    fn xtc_runs_on_renormalised_filtered_row() {
        // Two ways the post-filter placement is observable.
        //
        // (a) A token that top-p removes must not count towards XTC's "at
        //     least two above threshold" test. Raw row [0.70, 0.15, 0.14,
        //     0.01] at top_p = 0.65 keeps only index 0; on the raw row three
        //     tokens exceed the 0.10 threshold and the pre-#1379 placement
        //     removed index 0, while the renormalised filtered row holds one
        //     token at probability 1.0, so in-chain XTC is a no-op.
        let gate = ffi::from_slice_f32(&[0.5], &[1]);
        let row_a: Vec<f32> = [0.70f32, 0.15, 0.14, 0.01].iter().map(|p| p.ln()).collect();
        let logits_a = ffi::from_slice_f32(&row_a, &[1, 4]);
        let probs_a = probs_row(&ffi::fused_sample_probs_xtc(
            &logits_a,
            1.0,
            0,
            0.65,
            0.0,
            0.10,
            1.0,
            &[],
            &gate,
        ));
        assert!(
            probs_a[0] > 0.99,
            "XTC counted a token top-p had removed: {probs_a:?}"
        );

        // (b) Renormalisation must raise a surviving token above the
        //     threshold. Raw row [0.50, 0.30, 0.15, 0.05] at top_p = 0.9
        //     keeps {0, 1, 2}; renormalised, index 2 sits at 0.158 > 0.155
        //     while its raw probability 0.15 is below, so all three survivors
        //     are above the threshold and XTC keeps exactly the least
        //     probable of them, index 2.
        let logits_b = nucleus_test_logits();
        let probs_b = probs_row(&ffi::fused_sample_probs_xtc(
            &logits_b,
            1.0,
            0,
            0.9,
            0.0,
            0.155,
            1.0,
            &[],
            &gate,
        ));
        assert_eq!(
            support_of(&probs_b),
            vec![2],
            "XTC did not evaluate the renormalised filtered row: {probs_b:?}"
        );
        assert!((probs_b[2] - 1.0).abs() < 1e-6, "{probs_b:?}");
    }

    /// The C++ in-chain XTC (`apply_xtc_to_filtered_logits` in
    /// `cpp/mlx_cxx_bridge.cpp`) against the Rust reference
    /// [`apply_xtc_filter`], element for element, on the four rows the
    /// `apply_xtc_filter_*` tests pin.
    ///
    /// Every truncation filter is off (`top_k = 0`, `top_p = 1.0`,
    /// `min_p = 0.0`) and `T == 1.0`, so the C++ XTC sees the raw row and the
    /// reported distribution is exactly `softmax(apply_xtc_filter(row))`,
    /// which is the Rust reference's own domain. `xtc_probability = 1.0`
    /// against a gate of 0.5 fires the removal unconditionally, so this test
    /// isolates the filter from the gate; `cpp_xtc_gate_decides_the_removal`
    /// covers the gate.
    ///
    /// The allowlist row is the case that earns this test. Production always
    /// populates `xtc_special_tokens` (newline plus the merged EOS set, built
    /// in `src/server/model_worker.rs`), and a defect in that branch would let
    /// XTC remove EOS and run generation away, which no other test would
    /// catch.
    #[test]
    fn cpp_xtc_matches_the_rust_reference_filter() {
        let gate = ffi::from_slice_f32(&[0.5], &[1]);
        let cases: [(f32, &[i32]); 4] = [
            // Tokens 0, 1, 2 above threshold: 0 and 1 are removed, 2 (the
            // least probable of them) survives.
            (0.01, &[]),
            // Same row, but token 0 is held by the allowlist.
            (0.01, &[0]),
            // One candidate above threshold: no-op.
            (0.5, &[]),
            // Zero candidates above threshold: no-op.
            (0.9, &[]),
        ];
        for (threshold, allowlist) in cases {
            let logits = xtc_test_logits();
            let reference = apply_xtc_filter(&logits, threshold, allowlist);
            let expected = probs_row(&ffi::softmax(&reference, -1));
            let got = probs_row(&ffi::fused_sample_probs_xtc(
                &logits, 1.0, 0, 1.0, 0.0, threshold, 1.0, allowlist, &gate,
            ));
            assert_eq!(got.len(), expected.len());
            for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() < 1e-6,
                    "C++ XTC diverged from apply_xtc_filter at token {i} \
                     (threshold {threshold}, allowlist {allowlist:?}): \
                     {got:?} vs {expected:?}"
                );
            }
        }
    }

    /// The gate compare (`gate < xtc_probability`) in its deciding regime.
    ///
    /// The production setting is `0 < xtc_probability < 1`, where that compare
    /// picks between a filtered and an unfiltered row on every step. The other
    /// two C++ XTC tests pass `xtc_probability = 1.0`, where it always fires,
    /// so neither reaches it. Here `probability = 0.5` is held against a gate
    /// on either side of it, so a compare that was inverted, dropped, or wired
    /// to the wrong operand fails.
    #[test]
    fn cpp_xtc_gate_decides_the_removal() {
        let logits = xtc_test_logits();
        let unfiltered = probs_row(&ffi::softmax(&logits, -1));

        // 0.9 is not below 0.5: no removal, the row is the plain softmax.
        let gate_off = ffi::from_slice_f32(&[0.9], &[1]);
        let off = probs_row(&ffi::fused_sample_probs_xtc(
            &logits,
            1.0,
            0,
            1.0,
            0.0,
            0.01,
            0.5,
            &[],
            &gate_off,
        ));
        for (i, (&g, &e)) in off.iter().zip(&unfiltered).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "XTC fired with gate 0.9 above probability 0.5 at token {i}: \
                 {off:?} vs {unfiltered:?}"
            );
        }

        // 0.1 is below 0.5: the removal fires and tokens 0 and 1 leave the
        // support.
        let gate_on = ffi::from_slice_f32(&[0.1], &[1]);
        let on = probs_row(&ffi::fused_sample_probs_xtc(
            &logits,
            1.0,
            0,
            1.0,
            0.0,
            0.01,
            0.5,
            &[],
            &gate_on,
        ));
        assert_eq!(
            support_of(&on),
            vec![2, 3],
            "XTC did not fire with gate 0.1 below probability 0.5: {on:?}"
        );
    }

    /// [`ffi::fused_sample_xtc`], the token-drawing entry point, at
    /// `temperature == 0.0`: the draw is the argmax of the XTC-filtered row,
    /// not of the raw row. This is the one deliberate edge change #1379 makes
    /// to a greedy configuration, and it is the only test that calls
    /// `fused_sample_xtc` at all.
    ///
    /// The second case runs the same draw with the allowlist populated, so the
    /// allowlist branch is exercised through the drawing path as well as
    /// through the reported distribution.
    #[test]
    fn cpp_xtc_greedy_draw_is_the_argmax_of_the_filtered_row() {
        let gate = ffi::from_slice_f32(&[0.5], &[1]);

        // Raw argmax is token 0 (logit 10.0). XTC at threshold 0.01 removes
        // tokens 0 and 1, leaving token 2 (logit 8.0) as the argmax.
        let logits = xtc_test_logits();
        let tok = ffi::fused_sample_xtc(&logits, 0.0, 0, 1.0, 0.0, 0.01, 1.0, &[], &gate);
        let b = ffi::array_to_raw_bytes(&tok);
        assert_eq!(
            i32::from_ne_bytes([b[0], b[1], b[2], b[3]]),
            2,
            "greedy XTC draw is not the argmax of the filtered row"
        );

        // With token 0 allowlisted it survives the removal and wins the
        // argmax again.
        let tok_allow = ffi::fused_sample_xtc(&logits, 0.0, 0, 1.0, 0.0, 0.01, 1.0, &[0], &gate);
        let ba = ffi::array_to_raw_bytes(&tok_allow);
        assert_eq!(
            i32::from_ne_bytes([ba[0], ba[1], ba[2], ba[3]]),
            0,
            "allowlisted token was removed by the C++ XTC draw path"
        );
    }

    // -- top-n-sigma row filter (#1375) --

    /// Indices of the finite (kept) entries of a filtered row.
    fn kept_indices(v: &[f32]) -> Vec<usize> {
        v.iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn top_n_sigma_filter_keeps_within_n_std() {
        // Row [0,1,2,3,4]: mean 2, population std sqrt(2) = 1.414.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 5]);
        // n = 1: threshold 4 - 1.414 = 2.586 -> keeps {3, 4}.
        let v = to_vec_f32(&top_n_sigma_filter(&logits, 1.0));
        assert_eq!(kept_indices(&v), vec![3, 4]);
        // n = 2: threshold 4 - 2.828 = 1.172 -> keeps {2, 3, 4}.
        let v = to_vec_f32(&top_n_sigma_filter(&logits, 2.0));
        assert_eq!(kept_indices(&v), vec![2, 3, 4]);
        // n = 0: threshold is the max itself -> keeps only the argmax.
        let v = to_vec_f32(&top_n_sigma_filter(&logits, 0.0));
        assert_eq!(kept_indices(&v), vec![4]);
        // n = 100: threshold far below the min -> keeps everything.
        let v = to_vec_f32(&top_n_sigma_filter(&logits, 100.0));
        assert_eq!(kept_indices(&v), vec![0, 1, 2, 3, 4]);
        // Kept entries pass through unchanged.
        assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn top_n_sigma_filter_rows_independent() {
        #[rustfmt::skip]
        let flat = [
            0.0f32, 1.0, 2.0, 3.0, 4.0,
            4.0,    3.0, 2.0, 1.0, 0.0,
        ];
        let logits = ffi::from_slice_f32(&flat, &[2, 5]);
        let filtered = top_n_sigma_filter(&logits, 1.0);
        assert_eq!(kept_indices(&row_vec(&filtered, 0, 5)), vec![3, 4]);
        assert_eq!(kept_indices(&row_vec(&filtered, 1, 5)), vec![0, 1]);
    }

    #[test]
    fn top_n_sigma_filter_ignores_neg_inf_entries() {
        // The -inf entry must neither perturb the row statistics nor come
        // back: kept set matches the 5-entry row above and index 5 stays -inf.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, f32::NEG_INFINITY], &[1, 6]);
        let v = to_vec_f32(&top_n_sigma_filter(&logits, 1.0));
        assert_eq!(kept_indices(&v), vec![3, 4]);
        assert_eq!(v[5], f32::NEG_INFINITY);
    }

    #[test]
    fn top_n_sigma_filter_nan_entry_does_not_mask_the_row() {
        // MLX's Max reducer propagates NaN; the filter must exclude the NaN
        // from the row maximum so one NaN entry cannot drive the threshold
        // to NaN and collapse the whole row to -inf. The NaN entry itself
        // stays masked (NaN >= thresh is false).
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, f32::NAN], &[1, 6]);
        let v = to_vec_f32(&top_n_sigma_filter(&logits, 1.0));
        assert_eq!(kept_indices(&v), vec![3, 4]);
        assert_eq!(v[5], f32::NEG_INFINITY);
    }

    #[test]
    fn top_n_sigma_filter_f16_large_vocab_does_not_overflow() {
        // 152k float16 logits in [0, 3]: an f16 row sum overflows to inf
        // (152000 * ~1.5 >> 65504), which would drive the mean/std to
        // inf/NaN and silently disable the filter. The float32 reduction
        // path must keep a strict subset: 0 < kept < V.
        const V: usize = 152_000;
        let vals: Vec<f32> = (0..V).map(|i| (i % 1000) as f32 * 0.003).collect();
        let logits_f32 = ffi::from_slice_f32(&vals, &[1, V as i32]);
        let logits_f16 = ffi::astype(&logits_f32, dtype::FLOAT16);
        let filtered = top_n_sigma_filter(&logits_f16, 1.0);
        // The filter must hand back the ORIGINAL dtype, not the f32 the
        // statistics were computed in.
        assert_eq!(ffi::array_dtype(&filtered), dtype::FLOAT16);
        let as_f32 = ffi::astype(&filtered, dtype::FLOAT32);
        let kept = kept_indices(&to_vec_f32(&as_f32)).len();
        assert!(kept > 0, "filter masked the whole row (overflow symptom)");
        assert!(kept < V, "filter kept the whole row (overflow symptom)");
    }

    #[test]
    fn apply_row_filters_disabled_returns_input_pointer_unchanged() {
        // Disabled filters must add zero graph nodes: the exact input
        // pointer comes back.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0], &[1, 3]);
        let ptr_before = &*logits as *const MlxArray;
        let params = FusedSampleParams::from_config(&SamplingConfig::default());
        let out = apply_row_filters(logits, &params);
        assert_eq!(ptr_before, &*out as *const MlxArray);
    }

    #[test]
    fn apply_row_filters_greedy_skips_active_filter() {
        // Greedy (temperature 0 / top_k 1) skips the filter even when
        // enabled: same pointer, no new nodes.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0], &[1, 3]);
        let ptr_before = &*logits as *const MlxArray;
        let mut config = SamplingConfig::greedy();
        config.top_n_sigma = 1.0;
        let out = apply_row_filters(logits, &FusedSampleParams::from_config(&config));
        assert_eq!(ptr_before, &*out as *const MlxArray);
    }

    #[test]
    fn apply_row_filters_treats_negative_sentinel_as_disabled() {
        // The b10621 `-1.0` "disabled" sentinel must be inert, not a
        // mask-everything threshold.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0], &[1, 3]);
        let ptr_before = &*logits as *const MlxArray;
        let config = SamplingConfig {
            top_n_sigma: -1.0,
            ..Default::default()
        };
        let out = apply_row_filters(logits, &FusedSampleParams::from_config(&config));
        assert_eq!(ptr_before, &*out as *const MlxArray);
    }

    #[test]
    fn top_n_sigma_skipped_on_greedy() {
        // temperature = 0 with the filter enabled returns the argmax of the
        // unfiltered row, byte-identical to the baseline greedy config.
        let logits = ffi::from_slice_f32(&[0.1, 0.9, 1.2], &[1, 1, 3]);
        let baseline = SamplingConfig::greedy();
        let mut with_filter = SamplingConfig::greedy();
        with_filter.top_n_sigma = 1.0;

        let (token_a, logits_a) = sample_token_optimized(&logits, &baseline, &[]);
        let (token_b, logits_b) = sample_token_optimized(&logits, &with_filter, &[]);
        ffi::eval(&token_a);
        ffi::eval(&token_b);
        assert_eq!(ffi::item_i32(&token_a), ffi::item_i32(&token_b));
        assert_eq!(to_vec_f32(&logits_a), to_vec_f32(&logits_b));
    }

    #[test]
    fn config_supports_fused_batch_true_for_top_n_sigma() {
        // The filter is row-wise, history-free and RNG-free, so it stays on
        // the single-dispatch fused batch path.
        let cfg = SamplingConfig {
            top_n_sigma: 1.0,
            ..Default::default()
        };
        assert!(config_supports_fused_batch(&cfg));
    }

    #[test]
    fn fused_sample_params_normalizes_inert_top_n_sigma() {
        // Greedy rows differing only in an inert top_n_sigma sample
        // identically, so from_config must normalize the field to 0.0 and
        // keep them on the shared fused batch / lookahead paths.
        let mut greedy_a = SamplingConfig::greedy();
        greedy_a.top_n_sigma = 0.0;
        let mut greedy_b = SamplingConfig::greedy();
        greedy_b.top_n_sigma = 1.5;
        let pa = FusedSampleParams::from_config(&greedy_a);
        let pb = FusedSampleParams::from_config(&greedy_b);
        assert_eq!(pa.top_n_sigma, 0.0);
        assert_eq!(pb.top_n_sigma, 0.0);
        assert!(pa.matches(&pb));

        // The non-positive / non-finite "disabled" forms normalize too.
        let sentinel = SamplingConfig {
            top_n_sigma: -1.0,
            ..Default::default()
        };
        assert_eq!(FusedSampleParams::from_config(&sentinel).top_n_sigma, 0.0);
        let nan = SamplingConfig {
            top_n_sigma: f32::NAN,
            ..Default::default()
        };
        assert_eq!(FusedSampleParams::from_config(&nan).top_n_sigma, 0.0);
    }

    #[test]
    fn fused_sample_params_matches_compares_top_n_sigma() {
        let base = FusedSampleParams::from_config(&SamplingConfig::with_temperature(0.7));
        assert!(base.matches(&base));
        let diff = FusedSampleParams {
            top_n_sigma: 1.0,
            ..base
        };
        assert!(!base.matches(&diff));
        assert!(diff.matches(&diff));
    }

    #[test]
    fn batched_fused_sample_honors_top_n_sigma() {
        // 64 identical rows [0,1,2,3,4], temperature 1.0, n = 1.0: the
        // filter keeps {3, 4}, so every sampled id must land there.
        const B: usize = 64;
        let mut flat = Vec::with_capacity(B * 5);
        for _ in 0..B {
            flat.extend_from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0]);
        }
        let logits = ffi::from_slice_f32(&flat, &[B as i32, 1, 5]);
        let cfg = SamplingConfig {
            top_n_sigma: 1.0,
            ..Default::default()
        };
        let params = FusedSampleParams::from_config(&cfg);
        let tokens = batched_fused_sample(&logits, &params);
        assert_eq!(tokens.len(), B);
        for (i, t) in tokens.iter().enumerate() {
            assert!(
                *t == 3 || *t == 4,
                "row {i} sampled masked token {t}; filter not applied on the fused batch path"
            );
        }
    }

    #[test]
    fn effective_token_distribution_zeroes_top_n_sigma_masked_tokens() {
        // The #902 proposal/target distribution must carry zero mass outside
        // the kept set {3, 4}, keeping speculative acceptance consistent
        // with the sampler.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5]);
        let cfg = SamplingConfig {
            top_n_sigma: 1.0,
            ..Default::default()
        };
        let probs = effective_token_distribution(&logits, &cfg, &[]);
        let v = to_vec_f32(&probs);
        assert_eq!(v.len(), 5);
        for i in [0usize, 1, 2] {
            assert_eq!(v[i], 0.0, "masked token {i} kept probability {}", v[i]);
        }
        for i in [3usize, 4] {
            assert!(v[i] > 0.0, "kept token {i} lost its probability");
        }
        let total: f32 = v.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "distribution not normalized: {total}"
        );
    }

    // -- typical_p row filter (#1377) --

    /// Host-side f64 reference of `typical_p_filter`'s kept set.
    fn typical_p_reference_keep(logits: &[f32], typical_p: f64) -> Vec<bool> {
        let finite: Vec<bool> = logits.iter().map(|x| x.is_finite()).collect();
        let max = logits
            .iter()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
        let z: f64 = logits
            .iter()
            .filter(|x| x.is_finite())
            .map(|&x| ((x as f64) - max).exp())
            .sum();
        let p: Vec<f64> = logits
            .iter()
            .map(|&x| {
                if x.is_finite() {
                    ((x as f64) - max).exp() / z
                } else {
                    0.0
                }
            })
            .collect();
        let entropy: f64 = -p
            .iter()
            .filter(|&&pi| pi > 0.0)
            .map(|&pi| pi * pi.ln())
            .sum::<f64>();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        let dev: Vec<f64> = p
            .iter()
            .zip(&finite)
            .map(|(&pi, &fin)| {
                if fin && pi > 0.0 {
                    ((-pi.ln()) - entropy).abs()
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        order.sort_by(|&a, &b| dev[a].partial_cmp(&dev[b]).unwrap());
        let mut keep = vec![false; logits.len()];
        let mut cum = 0.0f64;
        for &i in &order {
            if !finite[i] {
                break;
            }
            if cum < typical_p {
                keep[i] = true;
            }
            cum += p[i];
        }
        keep
    }

    /// Deterministic LCG so the reference test needs no rand dependency.
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 24 bits -> [0, 1)
        ((*state >> 40) as f32) / (1u64 << 24) as f32
    }

    #[test]
    fn typical_p_filter_matches_host_reference() {
        let mut rng = 0x1377_u64;
        for case in 0..40 {
            let v = 8 + (case * 5) % 193;
            let logits_host: Vec<f32> = (0..v).map(|_| (lcg_next(&mut rng) - 0.5) * 6.0).collect();
            let tp = [0.2f32, 0.5, 0.9, 0.95][case % 4];
            let logits = ffi::from_slice_f32(&logits_host, &[1, v as i32]);
            let out = to_vec_f32(&typical_p_filter(&logits, tp));
            let reference = typical_p_reference_keep(&logits_host, tp as f64);
            // Compare only tokens whose exclusive cumulative mass is not
            // within 1e-4 of the boundary, where f32-vs-f64 rounding can
            // legitimately flip the strict comparison.
            let max = logits_host
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = logits_host.iter().map(|&x| ((x - max) as f64).exp()).sum();
            let p: Vec<f64> = logits_host
                .iter()
                .map(|&x| ((x - max) as f64).exp() / z)
                .collect();
            let entropy: f64 = -p.iter().map(|&pi| pi * pi.ln()).sum::<f64>();
            let mut order: Vec<usize> = (0..v).collect();
            order.sort_by(|&a, &b| {
                ((-p[a].ln()) - entropy)
                    .abs()
                    .partial_cmp(&((-p[b].ln()) - entropy).abs())
                    .unwrap()
            });
            let mut cum_excl = vec![0.0f64; v];
            let mut cum = 0.0f64;
            for &i in &order {
                cum_excl[i] = cum;
                cum += p[i];
            }
            for i in 0..v {
                if (cum_excl[i] - tp as f64).abs() <= 1e-4 {
                    continue;
                }
                assert_eq!(
                    out[i].is_finite(),
                    reference[i],
                    "case {case} (V={v}, typical_p={tp}): token {i} diverges from the host reference"
                );
            }
        }
    }

    #[test]
    fn typical_p_one_disables_via_hook() {
        // typical_p = 1.0 must add zero graph nodes: the exact input pointer
        // comes back from the hook.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0], &[1, 3]);
        let ptr_before = &*logits as *const MlxArray;
        let params = FusedSampleParams::from_config(&SamplingConfig::default());
        assert_eq!(params.typical_p, 1.0);
        let out = apply_row_filters(logits, &params);
        assert_eq!(ptr_before, &*out as *const MlxArray);

        // And with a DIFFERENT filter active, typical_p = 1.0 must add
        // nothing on top of it: the hook output equals the direct
        // top-n-sigma filter byte for byte.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 5]);
        let cfg = SamplingConfig {
            top_n_sigma: 1.0,
            ..Default::default()
        };
        let via_hook = apply_row_filters(
            ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 5]),
            &FusedSampleParams::from_config(&cfg),
        );
        let direct = top_n_sigma_filter(&logits, 1.0);
        assert_eq!(to_vec_f32(&via_hook), to_vec_f32(&direct));
    }

    #[test]
    fn typical_p_runs_on_the_renormalized_top_k_survivors() {
        // b10621's chain order is `top_k -> typ_p`, so the typicality
        // statistics come from the RENORMALIZED top-k distribution. This row
        // is built so the two orders disagree: logits [5.0, 4.9, 0 x 50].
        //
        // Full-vocabulary entropy (~1.60 nats) is inflated by the 50 tail
        // tokens, making token 1 more typical than token 0, so typ_p = 0.3
        // over the full row would keep ONLY token 1. Over the renormalized
        // top-2 survivors (p = [0.525, 0.475], H = 0.692) token 0 is the
        // more typical one and typ_p = 0.3 keeps ONLY token 0.
        let mut row = vec![0.0f32; 52];
        row[0] = 5.0;
        row[1] = 4.9;
        let logits = ffi::from_slice_f32(&row, &[1, 52]);
        let cfg = SamplingConfig {
            top_k: 2,
            typical_p: 0.3,
            ..Default::default()
        };
        let out = apply_row_filters(logits, &FusedSampleParams::from_config(&cfg));
        let v = to_vec_f32(&out);
        assert!(
            v[0].is_finite(),
            "token 0 must survive: typicality must be computed over the \
             renormalized top-k set (b10621 order), not the full vocabulary"
        );
        assert_eq!(
            v[1],
            f32::NEG_INFINITY,
            "token 1 is less typical within the top-2 set"
        );
        for (i, x) in v.iter().enumerate().skip(2) {
            assert_eq!(
                *x,
                f32::NEG_INFINITY,
                "tail token {i} escaped the top-k mask"
            );
        }
    }

    #[test]
    fn typical_p_top_k_premask_skips_oversized_k() {
        // top_k >= vocab is a no-op upstream; the pre-mask must skip it
        // rather than hand argpartition an out-of-range kth index.
        let logits = ffi::from_slice_f32(&[0.45f32.ln(), 0.275f32.ln(), 0.275f32.ln()], &[1, 3]);
        let cfg = SamplingConfig {
            top_k: 64,
            typical_p: 0.3,
            ..Default::default()
        };
        let out = apply_row_filters(logits, &FusedSampleParams::from_config(&cfg));
        let v = to_vec_f32(&out);
        // Same result as the no-top_k case: the less-typical argmax drops.
        assert_eq!(v[0], f32::NEG_INFINITY);
        assert!(v[1].is_finite() || v[2].is_finite());
    }

    #[test]
    fn typical_p_filter_smaller_keeps_fewer() {
        let logits_host: Vec<f32> = (0..50).map(|i| ((i * 7) % 13) as f32 * 0.3).collect();
        let logits = ffi::from_slice_f32(&logits_host, &[1, 50]);
        let mut prev = 0usize;
        for tp in [0.2f32, 0.5, 0.9, 0.99] {
            let kept = to_vec_f32(&typical_p_filter(&logits, tp))
                .iter()
                .filter(|x| x.is_finite())
                .count();
            assert!(
                kept >= prev,
                "kept count decreased from {prev} to {kept} at typical_p={tp}"
            );
            assert!(
                kept > 0,
                "at least one token must survive at typical_p={tp}"
            );
            prev = kept;
        }
    }

    #[test]
    fn typical_p_filter_can_drop_argmax() {
        // p = [0.45, 0.275, 0.275]: H = 1.069 nats, surprisals
        // [0.799, 1.291, 1.291], deviations [0.271, 0.222, 0.222]. The two
        // 0.275 tokens are MORE typical than the argmax and their mass 0.55
        // accumulates first, so typical_p = 0.3 masks index 0.
        let logits = ffi::from_slice_f32(&[0.45f32.ln(), 0.275f32.ln(), 0.275f32.ln()], &[1, 3]);
        let v = to_vec_f32(&typical_p_filter(&logits, 0.3));
        assert_eq!(
            v[0],
            f32::NEG_INFINITY,
            "the less-typical argmax must be dropped"
        );
        assert!(v[1].is_finite() || v[2].is_finite());
    }

    #[test]
    fn typical_p_filter_ignores_neg_inf_entries() {
        let base = [0.45f32.ln(), 0.275f32.ln(), 0.275f32.ln()];
        let with_masked = [
            0.45f32.ln(),
            0.275f32.ln(),
            0.275f32.ln(),
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let a = to_vec_f32(&typical_p_filter(&ffi::from_slice_f32(&base, &[1, 3]), 0.3));
        let b = to_vec_f32(&typical_p_filter(
            &ffi::from_slice_f32(&with_masked, &[1, 5]),
            0.3,
        ));
        for i in 0..3 {
            assert_eq!(
                a[i].is_finite(),
                b[i].is_finite(),
                "appending -inf entries changed the kept set at index {i}"
            );
        }
        assert_eq!(b[3], f32::NEG_INFINITY);
        assert_eq!(b[4], f32::NEG_INFINITY);
    }

    #[test]
    fn typical_p_filter_nan_entry_does_not_mask_the_row() {
        // A NaN logit is sanitized to -inf before the softmax, so it cannot
        // poison the entropy or the kept set.
        let base = [0.45f32.ln(), 0.275f32.ln(), 0.275f32.ln()];
        let with_nan = [0.45f32.ln(), 0.275f32.ln(), 0.275f32.ln(), f32::NAN];
        let a = to_vec_f32(&typical_p_filter(&ffi::from_slice_f32(&base, &[1, 3]), 0.3));
        let b = to_vec_f32(&typical_p_filter(
            &ffi::from_slice_f32(&with_nan, &[1, 4]),
            0.3,
        ));
        for i in 0..3 {
            assert_eq!(a[i].is_finite(), b[i].is_finite());
        }
        assert_eq!(b[3], f32::NEG_INFINITY);
    }

    #[test]
    fn typical_p_filter_preserves_dtype() {
        let logits_f32 = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[1, 4]);
        let logits_f16 = ffi::astype(&logits_f32, dtype::FLOAT16);
        let filtered = typical_p_filter(&logits_f16, 0.5);
        assert_eq!(ffi::array_dtype(&filtered), dtype::FLOAT16);
    }

    #[test]
    fn typical_p_skipped_on_greedy() {
        // Greedy must still return the argmax even when typical_p would
        // drop it (the can_drop_argmax row above).
        let logits = ffi::from_slice_f32(&[0.45f32.ln(), 0.275f32.ln(), 0.275f32.ln()], &[1, 1, 3]);
        let mut config = SamplingConfig::greedy();
        config.typical_p = 0.3;
        let (token, _) = sample_token_optimized(&logits, &config, &[]);
        ffi::eval(&token);
        assert_eq!(ffi::item_i32(&token), 0);
    }

    #[test]
    fn batched_fused_sample_honors_typical_p() {
        // 64 rows [10, 0, 0, 0, 0]: the argmax holds ~99.98% of the mass, so
        // it is by far the most typical token and typical_p = 0.3 keeps only
        // it; every draw must be index 0.
        const B: usize = 64;
        let mut flat = Vec::with_capacity(B * 5);
        for _ in 0..B {
            flat.extend_from_slice(&[10.0f32, 0.0, 0.0, 0.0, 0.0]);
        }
        let logits = ffi::from_slice_f32(&flat, &[B as i32, 1, 5]);
        let cfg = SamplingConfig {
            typical_p: 0.3,
            ..Default::default()
        };
        let params = FusedSampleParams::from_config(&cfg);
        assert_eq!(params.typical_p, 0.3);
        let tokens = batched_fused_sample(&logits, &params);
        assert_eq!(tokens, vec![0; B]);
    }

    #[test]
    fn fused_sample_params_matches_compares_typical_p() {
        let base = FusedSampleParams::from_config(&SamplingConfig::with_temperature(0.7));
        assert!(base.matches(&base));
        let diff = FusedSampleParams {
            typical_p: 0.5,
            ..base
        };
        assert!(!base.matches(&diff));
        assert!(diff.matches(&diff));
    }

    #[test]
    fn fused_sample_params_normalizes_inert_typical_p() {
        // Greedy rows and out-of-domain values normalize to the disabled 1.0
        // so necessarily-identical rows stay on the shared fused paths.
        let mut greedy = SamplingConfig::greedy();
        greedy.typical_p = 0.5;
        assert_eq!(FusedSampleParams::from_config(&greedy).typical_p, 1.0);
        for bad in [0.0f32, -0.5, 1.5, f32::NAN, f32::INFINITY] {
            let cfg = SamplingConfig {
                typical_p: bad,
                ..Default::default()
            };
            assert_eq!(
                FusedSampleParams::from_config(&cfg).typical_p,
                1.0,
                "typical_p={bad} must normalize to the disabled form"
            );
        }
    }

    #[test]
    fn effective_token_distribution_zeroes_typical_p_masked_tokens() {
        // The #902 distribution must carry zero mass on the dropped argmax.
        let logits = ffi::from_slice_f32(&[0.45f32.ln(), 0.275f32.ln(), 0.275f32.ln()], &[1, 1, 3]);
        let cfg = SamplingConfig {
            typical_p: 0.3,
            ..Default::default()
        };
        let probs = effective_token_distribution(&logits, &cfg, &[]);
        let v = to_vec_f32(&probs);
        assert_eq!(v[0], 0.0, "dropped argmax kept probability {}", v[0]);
        let total: f32 = v.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    // -- b10621 penalty window and DRY sentinels (#1436) --

    #[test]
    fn penalty_window_sentinels() {
        let history = [1, 2, 3, 4, 5];
        assert_eq!(penalty_window(&history, -1), &history[..]);
        assert_eq!(penalty_window(&history, 0), &[] as &[i32]);
        assert_eq!(penalty_window(&history, 2), &[4, 5][..]);
        assert_eq!(penalty_window(&history, 64), &history[..]);
    }

    #[test]
    fn repetition_penalty_windowed_ignores_tokens_outside_the_window() {
        // History [0, 1]; window 1 sees only token 1, so token 0 keeps its
        // raw logit while the full-history form penalizes both.
        let logits = ffi::from_slice_f32(&[2.0, 2.0, 2.0], &[1, 1, 3]);
        let mut windowed = SamplingConfig::greedy();
        windowed.repetition_penalty = 2.0;
        windowed.penalty_last_n = 1;
        let (_, processed) = sample_token_optimized(&logits, &windowed, &[0, 1]);
        let v = to_vec_f32(&processed);
        assert_eq!(
            v[0], 2.0,
            "token 0 is outside the window and must not be penalized"
        );
        assert_eq!(v[1], 1.0, "token 1 is inside the window");
        assert_eq!(v[2], 2.0);

        let mut full = SamplingConfig::greedy();
        full.repetition_penalty = 2.0;
        full.penalty_last_n = -1;
        let (_, processed) = sample_token_optimized(&logits, &full, &[0, 1]);
        let v = to_vec_f32(&processed);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
    }

    #[test]
    fn penalty_last_n_zero_disables_the_stage_and_stays_fused_eligible() {
        let logits = ffi::from_slice_f32(&[2.0, 2.0, 2.0], &[1, 1, 3]);
        let mut cfg = SamplingConfig::greedy();
        cfg.repetition_penalty = 2.0;
        cfg.frequency_penalty = 0.5;
        cfg.presence_penalty = 0.5;
        cfg.penalty_last_n = 0;
        let (_, processed) = sample_token_optimized(&logits, &cfg, &[0, 1, 2]);
        assert_eq!(
            to_vec_f32(&processed),
            vec![2.0, 2.0, 2.0],
            "a zero window makes every history penalty inert"
        );
        assert!(!cfg.needs_token_history());
        assert!(config_supports_fused_batch(&cfg));
    }

    #[test]
    fn windowed_penalties_match_full_history_when_the_window_covers_it() {
        let logits = ffi::from_slice_f32(&[2.0, 2.0, -1.0, 2.0], &[1, 1, 4]);
        let history = [0, 2, 1];
        let mut covering = SamplingConfig::greedy();
        covering.repetition_penalty = 1.5;
        covering.frequency_penalty = 0.25;
        covering.penalty_last_n = 64;
        let mut full = covering.clone();
        full.penalty_last_n = -1;
        let (_, a) = sample_token_optimized(&logits, &covering, &history);
        let (_, b) = sample_token_optimized(&logits, &full, &history);
        assert_eq!(
            to_vec_f32(&a),
            to_vec_f32(&b),
            "a window covering the whole history must be byte-identical to the full-history form"
        );
    }

    #[test]
    fn dry_penalty_last_n_zero_disables_dry() {
        // A strongly repeating history that WOULD be penalized under any
        // scanning window produces untouched logits at the 0 sentinel.
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0], &[1, 1, 3]);
        let mut cfg = SamplingConfig::greedy();
        cfg.dry_multiplier = 1.0;
        cfg.dry_base = 2.0;
        cfg.dry_allowed_length = 1;
        cfg.dry_penalty_last_n = 0;
        let history = [0, 1, 2, 0, 1];
        let (_, processed) = sample_token_optimized(&logits, &cfg, &history);
        assert_eq!(to_vec_f32(&processed), vec![1.0, 1.0, 1.0]);
        assert!(!cfg.needs_token_history(), "disabled DRY needs no history");
    }

    #[test]
    fn dry_full_history_sentinel_scans_everything() {
        // History [0,1,2,0,1]: the suffix [0,1] repeats, so token 2 (the
        // continuation of the earlier occurrence) is penalized by
        // multiplier * base^(match_len - allowed) = 1.0 * 2^(2-1) = 2.0.
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0], &[1, 1, 3]);
        let mut cfg = SamplingConfig::greedy();
        cfg.dry_multiplier = 1.0;
        cfg.dry_base = 2.0;
        cfg.dry_allowed_length = 1;
        cfg.dry_penalty_last_n = crate::generate::DRY_FULL_HISTORY;
        let history = [0, 1, 2, 0, 1];
        let (_, processed) = sample_token_optimized(&logits, &cfg, &history);
        let v = to_vec_f32(&processed);
        assert!(
            (v[2] - -1.0).abs() < 1e-6,
            "token 2 must carry the -2.0 DRY penalty, got {v:?}"
        );
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
    }

    #[test]
    fn dry_penalizes_at_exactly_the_allowed_length() {
        // b10621's >= comparison: a match exactly dry_allowed_length long
        // gets the base^0 tier (= the bare multiplier). Pre-#1436 mlxcel
        // used > and skipped this tier entirely.
        let logits = ffi::from_slice_f32(&[1.0, 1.0, 1.0], &[1, 1, 3]);
        let mut cfg = SamplingConfig::greedy();
        cfg.dry_multiplier = 0.75;
        cfg.dry_base = 2.0;
        cfg.dry_allowed_length = 2;
        cfg.dry_penalty_last_n = crate::generate::DRY_FULL_HISTORY;
        let history = [0, 1, 2, 0, 1];
        let (_, processed) = sample_token_optimized(&logits, &cfg, &history);
        let v = to_vec_f32(&processed);
        assert!(
            (v[2] - (1.0 - 0.75)).abs() < 1e-6,
            "a match of exactly the allowed length must be penalized by multiplier * base^0, got {v:?}"
        );
    }

    // -- #1485: min_keep floors --

    #[test]
    fn top_p_min_keep_forces_floor_in_probability_order() {
        // Softmax of [0,1,2,3,8] puts ~0.99 on index 4, so top_p 0.5
        // naturally keeps only {4}; a floor of 3 forces the top three by
        // probability: {2, 3, 4}.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 8.0], &[1, 5]);
        let natural = to_vec_f32(&top_p_filter_min_keep(&logits, 0.5, 0));
        assert_eq!(kept_indices(&natural), vec![4]);
        let floored = to_vec_f32(&top_p_filter_min_keep(&logits, 0.5, 3));
        assert_eq!(kept_indices(&floored), vec![2, 3, 4]);
        // min_keep 1 adds nothing over the natural cut.
        let one = to_vec_f32(&top_p_filter_min_keep(&logits, 0.5, 1));
        assert_eq!(kept_indices(&one), vec![4]);
    }

    #[test]
    fn top_p_min_keep_floor_never_resurrects_masked_entries() {
        // Two finite candidates, three -inf-masked ones. A floor of 4 keeps
        // only the finite pair: masked entries carry zero probability and
        // the forced set is gated on finiteness.
        let neg = f32::NEG_INFINITY;
        let logits = ffi::from_slice_f32(&[neg, 1.0, neg, 2.0, neg], &[1, 5]);
        let v = to_vec_f32(&top_p_filter_min_keep(&logits, 0.1, 4));
        assert_eq!(kept_indices(&v), vec![1, 3]);
    }

    #[test]
    fn min_p_min_keep_forces_floor_in_probability_order() {
        // min_p 0.9 keeps only tokens with p >= 0.9 * p_max: just the argmax
        // here; a floor of 2 forces the runner-up back in.
        let logits = ffi::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 8.0], &[1, 5]);
        let natural = to_vec_f32(&min_p_filter_min_keep(&logits, 0.9, 0));
        assert_eq!(kept_indices(&natural), vec![4]);
        let floored = to_vec_f32(&min_p_filter_min_keep(&logits, 0.9, 2));
        assert_eq!(kept_indices(&floored), vec![3, 4]);
    }

    #[test]
    fn typical_p_min_keep_forces_floor_in_typicality_order() {
        let n = 16;
        let raw = lcg_logits(0x1485, n);
        let logits = ffi::from_slice_f32(&raw, &[1, n as i32]);
        // Tiny typical_p keeps exactly one token (the most typical).
        let natural = kept_indices(&to_vec_f32(&typical_p_filter_min_keep(&logits, 1e-6, 0)));
        assert_eq!(natural.len(), 1);
        let floored = kept_indices(&to_vec_f32(&typical_p_filter_min_keep(&logits, 1e-6, 5)));
        assert_eq!(
            floored.len(),
            5,
            "the floor forces exactly min_keep survivors"
        );
        assert!(
            floored.contains(&natural[0]),
            "the naturally kept most-typical token stays in the forced set"
        );
        // The forced set is the five most typical by the host reference
        // ordering: |(-log p) - H| ascending.
        let probs: Vec<f64> = {
            let mx = raw.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let exps: Vec<f64> = raw.iter().map(|&x| ((x as f64) - mx).exp()).collect();
            let z: f64 = exps.iter().sum();
            exps.iter().map(|e| e / z).collect()
        };
        let h: f64 = -probs.iter().map(|p| p * p.ln()).sum::<f64>();
        let mut by_typicality: Vec<usize> = (0..n).collect();
        by_typicality.sort_by(|&a, &b| {
            let da = ((-probs[a].ln()) - h).abs();
            let db = ((-probs[b].ln()) - h).abs();
            da.partial_cmp(&db).unwrap()
        });
        let mut expected: Vec<usize> = by_typicality[..5].to_vec();
        expected.sort_unstable();
        assert_eq!(floored, expected);
    }

    #[test]
    fn xtc_min_keep_skips_removal_when_too_few_would_survive() {
        // Row of 4 candidates, three above the 0.2 threshold. Removal keeps
        // the least-probable above-threshold token plus the below-threshold
        // one: 2 survivors. min_keep 3 > 2 must skip the removal entirely.
        let logits = ffi::from_slice_f32(&[2.0, 1.8, 1.6, -2.0], &[1, 4]);
        let removed = to_vec_f32(&apply_xtc_filter_min_keep(&logits, 0.2, &[], 0));
        assert_eq!(kept_indices(&removed), vec![2, 3]);
        let skipped = to_vec_f32(&apply_xtc_filter_min_keep(&logits, 0.2, &[], 3));
        assert_eq!(kept_indices(&skipped), vec![0, 1, 2, 3]);
        // A floor the survivors satisfy leaves the removal in place.
        let kept2 = to_vec_f32(&apply_xtc_filter_min_keep(&logits, 0.2, &[], 2));
        assert_eq!(kept_indices(&kept2), vec![2, 3]);
    }

    // -- #1485: dynamic temperature --

    #[test]
    fn dynatemp_uniform_row_scales_by_max_temp() {
        // A uniform distribution has normalized entropy 1, so the dynamic
        // temperature is exactly temp + range.
        let logits = ffi::from_slice_f32(&[1.5; 8], &[1, 8]);
        let out = to_vec_f32(&dynatemp_transform(&logits, 0.8, 0.5, 1.0));
        for x in &out {
            assert!((x - 1.5 / 1.3).abs() < 1e-5, "expected 1.5/1.3, got {x}");
        }
    }

    #[test]
    fn dynatemp_matches_host_reference_on_peaked_row() {
        let raw = [4.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let (temp, range, exponent) = (1.0f32, 0.6f32, 1.7f32);
        let logits = ffi::from_slice_f32(&raw, &[1, raw.len() as i32]);
        let out = to_vec_f32(&dynatemp_transform(&logits, temp, range, exponent));
        // Host reference in f64.
        let mx = raw.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let exps: Vec<f64> = raw.iter().map(|&x| ((x as f64) - mx).exp()).collect();
        let z: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|e| e / z).collect();
        let h: f64 = -probs.iter().map(|p| p * p.ln()).sum::<f64>();
        let norm = h / (raw.len() as f64).ln();
        let min_t = (temp - range).max(0.0) as f64;
        let max_t = (temp + range) as f64;
        let dyn_t = min_t + (max_t - min_t) * norm.powf(exponent as f64);
        for (o, r) in out.iter().zip(raw.iter()) {
            let expect = (*r as f64) / dyn_t;
            assert!(
                ((*o as f64) - expect).abs() < 1e-4,
                "expected {expect}, got {o} (dyn_t {dyn_t})"
            );
        }
    }

    #[test]
    fn dynatemp_single_candidate_passes_through() {
        let neg = f32::NEG_INFINITY;
        let logits = ffi::from_slice_f32(&[neg, 3.0, neg], &[1, 3]);
        let out = to_vec_f32(&dynatemp_transform(&logits, 0.0, 0.5, 1.0));
        assert_eq!(out[1], 3.0, "a single-candidate row is not rescaled");
        assert!(out[0].is_infinite() && out[2].is_infinite());
    }

    #[test]
    fn dynatemp_keeps_temperature_zero_config_off_the_greedy_path() {
        let mut config = SamplingConfig::with_temperature(0.0);
        assert!(config.is_greedy_path());
        config.dynatemp_range = 0.5;
        assert!(
            !config.is_greedy_path(),
            "a positive range re-widens temp 0"
        );
        assert!(config.needs_extended_chain());
        assert!(!config_supports_fused_batch(&config));
        config.top_k = 1;
        assert!(config.is_greedy_path(), "top_k 1 stays greedy regardless");
    }

    // -- #1485: mirostat --

    fn mirostat_config(version: i32, tau: f32, eta: f32, temperature: f32) -> SamplingConfig {
        SamplingConfig {
            mirostat: version,
            mirostat_tau: tau,
            mirostat_eta: eta,
            temperature,
            // Deliberately hostile settings that mirostat must ignore:
            top_k: 2,
            top_p: 0.1,
            min_p: 0.5,
            repetition_penalty: 1.5,
            ..Default::default()
        }
    }

    #[test]
    fn mirostat_v2_tiny_mu_selects_the_argmax_and_updates_mu() {
        let config = mirostat_config(2, 1e-4, 0.25, 1.0);
        let logits = ffi::from_slice_f32(&[0.0, 5.0, 1.0, -2.0], &[1, 1, 4]);
        let mut state = None;
        let (tok, _adj) = sample_token_optimized_with_state(&logits, &config, &[], &mut state);
        ffi::eval(&tok);
        assert_eq!(ffi::item_i32(&tok), 1, "mu ~= 2e-4 truncates to the argmax");
        let st = state.expect("mirostat allocates the feedback state");
        let m = st.mirostat.expect("mirostat state present");
        // Only the argmax survives, so its renormalized probability is 1 and
        // the observed surprise 0: mu <- mu - eta * (0 - tau) = mu + eta*tau.
        let expected = 2.0 * 1e-4 + 0.25 * 1e-4;
        assert!(
            (m.mu - expected).abs() < 1e-6,
            "mu update mismatch: got {}, expected {expected}",
            m.mu
        );
    }

    #[test]
    fn mirostat_ignores_penalties_and_truncation_filters() {
        let config = mirostat_config(2, 1e-4, 0.1, 1.0);
        assert!(
            !config.needs_token_history(),
            "mirostat skips the penalty stages"
        );
        assert_eq!(config.effective_top_n_sigma(), 0.0);
        assert_eq!(config.effective_typical_p(), 1.0);
        assert!(!config_supports_fused_batch(&config));
        // The hostile top_k/top_p/min_p above must not steer the draw: with a
        // huge mu nothing truncates and the post-chain distribution is the
        // plain softmax of the logits.
        let config = mirostat_config(2, 1e6, 0.1, 1.0);
        let raw = [0.0f32, 2.0, 1.0, -1.0];
        let logits = ffi::from_slice_f32(&raw, &[1, 1, 4]);
        let mut state = None;
        let (_tok, _adj, probs) =
            sample_token_with_state_and_distribution(&logits, &config, &[], &mut state);
        let p = to_vec_f32(&probs);
        let mx = raw.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = raw.iter().map(|&x| (x - mx).exp()).collect();
        let z: f32 = exps.iter().sum();
        for (got, e) in p.iter().zip(exps.iter()) {
            assert!(
                (got - e / z).abs() < 1e-5,
                "mirostat with huge mu must sample the untruncated softmax; got {p:?}"
            );
        }
    }

    #[test]
    fn mirostat_v1_peaked_row_truncates_to_the_argmax() {
        let config = mirostat_config(1, 0.1, 0.3, 1.0);
        let logits = ffi::from_slice_f32(&[10.0, 0.0, 0.0, 0.0], &[1, 1, 4]);
        let mut state = None;
        let (tok, _adj) = sample_token_optimized_with_state(&logits, &config, &[], &mut state);
        ffi::eval(&tok);
        assert_eq!(
            ffi::item_i32(&tok),
            0,
            "the Zipf estimate on a near-delta distribution yields k = 1"
        );
        let m = state.unwrap().mirostat.unwrap();
        let expected = 2.0 * 0.1 + 0.3 * 0.1; // observed surprise 0
        assert!((m.mu - expected).abs() < 1e-6, "got {}", m.mu);
    }

    #[test]
    fn mirostat_temperature_zero_draws_the_argmax() {
        let config = mirostat_config(2, 5.0, 0.1, 0.0);
        let logits = ffi::from_slice_f32(&[1.0, 0.0, 4.0], &[1, 1, 3]);
        let mut state = None;
        let (tok, _adj) = sample_token_optimized_with_state(&logits, &config, &[], &mut state);
        ffi::eval(&tok);
        assert_eq!(ffi::item_i32(&tok), 2);
        let m = state.unwrap().mirostat.unwrap();
        assert!((m.mu - (10.0 + 0.1 * 5.0)).abs() < 1e-5);
    }

    // -- #1485: adaptive-p --

    #[test]
    fn adaptive_p_state_accept_updates_ema_only_on_matching_token() {
        let mut st = SamplerState {
            adaptive: Some(AdaptivePState::new(0.5, 0.9)),
            ..Default::default()
        };
        let (ws0, tw0) = AdaptivePState::initial_ema(0.5, 0.9);
        st.adaptive.as_mut().unwrap().pending = Some((7, 0.42));
        st.accept_token(3); // overridden token: EMA untouched, pending cleared
        {
            let a = st.adaptive.as_ref().unwrap();
            assert_eq!(a.weighted_sum, ws0);
            assert_eq!(a.total_weight, tw0);
            assert!(a.pending.is_none());
        }
        st.adaptive.as_mut().unwrap().pending = Some((7, 0.42));
        st.accept_token(7);
        let a = st.adaptive.as_ref().unwrap();
        assert!((a.weighted_sum - (0.42 + 0.9 * ws0)).abs() < 1e-6);
        assert!((a.total_weight - (1.0 + 0.9 * tw0)).abs() < 1e-6);
    }

    #[test]
    fn adaptive_p_transform_prefers_tokens_near_the_target() {
        // Original probabilities roughly [0.70, 0.26, 0.02, 0.02]; with a
        // target of 0.25 the transformed argmax must be index 1.
        let mut config = SamplingConfig {
            adaptive_target: 0.25,
            adaptive_decay: 0.9,
            temperature: 1.0,
            ..Default::default()
        };
        config.top_k = 0;
        assert!(config.needs_extended_chain());
        let logits = ffi::from_slice_f32(&[4.0, 3.0, 0.5, 0.5], &[1, 1, 4]);
        let mut state = None;
        let (tok, _adj, probs) =
            sample_token_with_state_and_distribution(&logits, &config, &[], &mut state);
        ffi::eval(&tok);
        let p = to_vec_f32(&probs);
        let argmax = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            argmax, 1,
            "post-transform mass concentrates near the target: {p:?}"
        );
        let st = state.expect("adaptive-p allocates the feedback state");
        let a = st.adaptive.as_ref().expect("adaptive state present");
        let (tok_id, orig_p) = a.pending.expect("draw parks the pending pair");
        assert_eq!(tok_id, ffi::item_i32(&tok));
        assert!(orig_p > 0.0 && orig_p < 1.0);
    }

    #[test]
    fn adaptive_p_disabled_target_stays_on_the_fused_path() {
        let config = SamplingConfig {
            adaptive_target: -1.0,
            ..Default::default()
        };
        assert!(!config.needs_extended_chain());
        assert!(config_supports_fused_batch(&config));
        assert!(!config.needs_sampler_feedback_state());
    }

    // -- #1485: DRY breaker heads --

    #[test]
    fn dry_breaker_head_with_empty_tail_breaks_matching() {
        // History "1 2 3 9 1 2 3" repeats "1 2 3"; without breakers the next
        // token 9 is penalized. Marking 2 as a breaker head cuts the match
        // below the allowed length.
        let history = [1, 2, 3, 9, 1, 2, 3];
        let mut config = SamplingConfig {
            dry_multiplier: 0.75,
            dry_base: 2.0,
            dry_allowed_length: 2,
            ..Default::default()
        };
        let logits = ffi::from_slice_f32(&[0.0; 12], &[1, 12]);
        let v = to_vec_f32(&apply_dry_penalty(&logits, &history, &config));
        assert!(
            v[9] < 0.0,
            "baseline: the repeated continuation is penalized"
        );
        config.dry_breaker_heads =
            std::sync::Arc::new(std::collections::HashMap::from([(2, vec![Vec::new()])]));
        let v = to_vec_f32(&apply_dry_penalty(&logits, &history, &config));
        assert_eq!(v[9], 0.0, "a breaker head stops the backward match");
    }

    #[test]
    fn dry_breaker_head_with_tail_requires_the_full_sequence() {
        let history = [1, 2, 3, 9, 1, 2, 3];
        let mut config = SamplingConfig {
            dry_multiplier: 0.75,
            dry_base: 2.0,
            dry_allowed_length: 2,
            ..Default::default()
        };
        // Head 2 with tail [8]: "2 8" never occurs, so matching proceeds.
        config.dry_breaker_heads =
            std::sync::Arc::new(std::collections::HashMap::from([(2, vec![vec![8]])]));
        let logits = ffi::from_slice_f32(&[0.0; 12], &[1, 12]);
        let v = to_vec_f32(&apply_dry_penalty(&logits, &history, &config));
        assert!(v[9] < 0.0, "an unmatched tail does not break");
        // Head 2 with tail [3]: "2 3" occurs at the matched position.
        config.dry_breaker_heads =
            std::sync::Arc::new(std::collections::HashMap::from([(2, vec![vec![3]])]));
        let v = to_vec_f32(&apply_dry_penalty(&logits, &history, &config));
        assert_eq!(v[9], 0.0, "a matched head+tail sequence breaks");
    }

    // -- #1485: post-sampling probability report --

    #[test]
    fn post_sampling_probs_reports_linear_probabilities_and_drops_zeros() {
        let probs = ffi::from_slice_f32(&[0.5, 0.3, 0.2, 0.0], &[1, 4]);
        let data = compute_post_sampling_probs(&probs, 1, 4);
        assert_eq!(data.token_id, 1);
        assert!(
            (data.logprob - 0.3).abs() < 1e-6,
            "linear probability, not log"
        );
        let tops: Vec<(i32, f32)> = data.top_alternatives.clone();
        assert_eq!(tops.len(), 3, "the zero-probability candidate is dropped");
        assert_eq!(tops[0].0, 0);
        assert_eq!(tops[1].0, 1);
        assert_eq!(tops[2].0, 2);
    }

    // -- #1485: control arms (disabled features change nothing) --

    #[test]
    fn inert_new_fields_keep_the_fused_path_and_identical_bytes() {
        let baseline = SamplingConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            ..Default::default()
        };
        let mut inert = baseline.clone();
        inert.min_keep = 1;
        inert.dynatemp_range = 0.0;
        inert.dynatemp_exponent = 2.5;
        inert.mirostat = 0;
        inert.mirostat_tau = 9.0;
        inert.adaptive_target = -1.0;
        inert.adaptive_decay = 0.5;
        assert!(config_supports_fused_batch(&baseline));
        assert!(config_supports_fused_batch(&inert));
        assert!(
            FusedSampleParams::from_config(&baseline)
                .matches(&FusedSampleParams::from_config(&inert))
        );
        let logits = ffi::from_slice_f32(&lcg_logits(0xC0DE, 64), &[1, 1, 64]);
        let (_t1, a1) = sample_token_optimized(&logits, &baseline, &[]);
        let (_t2, a2) = sample_token_optimized(&logits, &inert, &[]);
        assert_logits_bit_identical(&a1, &a2, "inert #1485 fields");
    }
}
