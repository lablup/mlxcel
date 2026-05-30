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
/// Used by: standard generation, speculative decoding, batch scheduler
pub(crate) fn apply_token_bias(logits: &MlxArray, bias: &TokenBiasMap) -> UniquePtr<MlxArray> {
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

    let last_logits = if config.repetition_penalty != 1.0 && !token_history.is_empty() {
        apply_repetition_penalty(&last_logits, token_history, config.repetition_penalty)
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
        apply_frequency_presence_penalty(
            &last_logits,
            token_history,
            config.frequency_penalty,
            config.presence_penalty,
        )
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

    if seen.is_empty() {
        return ffi::copy(logits);
    }

    let indices = ffi::from_slice_i32(&seen, &[1, seen.len() as i32]);
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

    // Apply log-softmax to get per-token log probabilities.
    let log_probs = ffi::log_softmax(adjusted_logits, -1);
    ffi::eval(&log_probs);

    // Extract the log probability of the selected token.
    let idx = ffi::from_slice_i32(&[selected_token], &[1, 1]);
    let selected_lp_arr = ffi::take_along_axis(&log_probs, &idx, -1);
    ffi::eval(&selected_lp_arr);
    let selected_logprob = ffi::item_f32(&selected_lp_arr);

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
        let lp_bytes = ffi::array_to_raw_bytes(&top_lp);

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
}
