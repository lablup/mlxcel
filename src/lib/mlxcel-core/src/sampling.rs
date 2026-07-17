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
/// Uses fused C++ sampling (temperature + top-k + top-p + min-p + categorical
/// in a single FFI call) to minimize round-trip overhead.
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
    if state.is_none()
        && (config.repetition_penalty != 1.0
            || config.frequency_penalty != 0.0
            || config.presence_penalty != 0.0)
    {
        *state = Some(SamplerState::for_config(config));
    }
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
    mut state: Option<&mut SamplerState>,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Use optimized slice_last_logits: [batch, seq, vocab] -> [batch, vocab].
    let last_logits = ffi::slice_last_logits(logits);

    // Apply token bias first (before history-based penalties).
    // Language policy is an external decision, not history-based, so it takes
    // precedence. -inf composes correctly with downstream penalties:
    //   -inf × k == -inf,  -inf + f == -inf.
    let last_logits = if !config.token_bias.is_empty() {
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
    } else {
        last_logits
    };

    // Synchronize the incremental state to the current history once, before any
    // penalty reads it. No-op when `state` is `None`.
    if let Some(s) = &mut state {
        s.sync(token_history);
    }

    let last_logits = if config.repetition_penalty != 1.0 && !token_history.is_empty() {
        match &mut state {
            Some(s) => s.apply_repetition(&last_logits, config.repetition_penalty),
            None => {
                apply_repetition_penalty(&last_logits, token_history, config.repetition_penalty)
            }
        }
    } else {
        last_logits
    };

    let last_logits = if config.dry_multiplier > 0.0 && !token_history.is_empty() {
        apply_dry_penalty(&last_logits, token_history, config)
    } else {
        last_logits
    };

    let last_logits = if (config.frequency_penalty != 0.0 || config.presence_penalty != 0.0)
        && !token_history.is_empty()
    {
        match &mut state {
            Some(s) => s.apply_frequency_presence(
                &last_logits,
                config.frequency_penalty,
                config.presence_penalty,
            ),
            None => apply_frequency_presence_penalty(
                &last_logits,
                token_history,
                config.frequency_penalty,
                config.presence_penalty,
            ),
        }
    } else {
        last_logits
    };

    // XTC (Exclude Top Choices): a logits pre-processing step, applied here
    // (before the fused temperature/top-k/top-p/min-p/categorical sampler)
    // the same way the penalties above are. `xtc_probability <= 0.0` is the
    // default disabled state and skips this entirely — no array ops, and no
    // draw from the per-request RNG stream, which keeps every existing
    // request's token stream byte-identical to before this feature existed.
    let last_logits = if config.xtc_probability > 0.0 {
        apply_xtc_step(&last_logits, config)
    } else {
        last_logits
    };

    let token = ffi::fused_sample(
        &last_logits,
        config.temperature,
        config.top_k,
        config.top_p,
        config.min_p,
    );
    (token, last_logits)
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
pub fn batched_sample(
    logits: &MlxArray,
    configs: &[&SamplingConfig],
    token_histories: &[&[i32]],
) -> Vec<i32> {
    let b = configs.len();
    debug_assert_eq!(b, token_histories.len());

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
}

impl FusedSampleParams {
    /// Extract the fused scalar params from a full [`SamplingConfig`].
    pub fn from_config(config: &SamplingConfig) -> Self {
        Self {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
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
/// Used by: `BatchScheduler::execute_batched_decode` fast-path gate
pub fn config_supports_fused_batch(config: &SamplingConfig) -> bool {
    !config.needs_token_history() && config.token_bias.is_empty() && config.xtc_probability <= 0.0
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
    let tokens = ffi::fused_sample(
        &last_logits,
        params.temperature,
        params.top_k,
        params.top_p,
        params.min_p,
    );
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

    let window = if config.dry_penalty_last_n == 0 {
        token_history
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

                if config.dry_sequence_breakers.contains(&window[p1]) {
                    break;
                }

                if window[p1] == window[p2] {
                    match_len += 1;
                } else {
                    break;
                }
            }

            if match_len > config.dry_allowed_length {
                let next_pos = pos + 1;
                if next_pos < window_len {
                    let next_token = window[next_pos];
                    let penalty = config.dry_multiplier
                        * config
                            .dry_base
                            .powi((match_len - config.dry_allowed_length) as i32);
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

/// Apply min-p filtering to logits.
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
#[allow(dead_code)]
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
/// Note: production generation routes through the C++ `fused_sample` → C++ `top_p_filter`
/// at `cpp/mlx_cxx_bridge.cpp`. This Rust implementation is a reference/test-parity
/// copy used to validate the algorithm and in unit tests for batched correctness.
///
/// Used by: unit tests (`top_p_filter_*` in `sampling::tests`)
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
/// This is a logits pre-processing step, applied the same way
/// [`apply_dry_penalty`] and the repetition/frequency/presence penalties
/// are: before the fused C++ temperature/top-k/top-p/min-p/categorical
/// sampler ([`ffi::fused_sample`]), not inside it.
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
/// Used by: [`apply_xtc_step`]
pub(crate) fn apply_xtc_filter(
    logits: &MlxArray,
    threshold: f32,
    allowlist: &[i32],
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
    let remove_mask = ffi::logical_and(&remove_candidate, &has_two_or_more);

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
/// Only called when `config.xtc_probability > 0.0` (see
/// [`sample_token_optimized_core`]); a request that leaves XTC disabled
/// never advances the RNG stream here, preserving the pre-XTC token stream
/// byte-for-byte.
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
}
