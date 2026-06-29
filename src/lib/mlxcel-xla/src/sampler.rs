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

//! Host-side token sampler for the OpenXLA backend (issue #449 M3 Stage 2d).
//!
//! The continuous-batching engine reads the per-row logits back to the host (the
//! logits graph variant) and samples here, so the serve path can honor sampling
//! parameters instead of only greedy argmax. Greedy (`temperature == 0`) is plain
//! argmax, identical to the on-device argmax the greedy graphs produced (the same
//! logits, the same max), so greedy output is token-exact across the switch.
//!
//! Supported: history-based penalties (repetition, frequency, presence, DRY)
//! applied to the logits first, then temperature, top-k, top-p (nucleus), min-p,
//! and a seeded PRNG for reproducibility. The penalty math mirrors the MLX path
//! (`mlxcel-core::sampling`): same order (repetition, then DRY, then
//! frequency/presence), same per-token formulas, so the two backends penalize
//! identically for the same history.
//!
//! Cost note: a non-greedy step softmaxes and sorts the full vocabulary per active
//! row on the host. That is acceptable for this reference backend (it is a small
//! fraction of the decode matmuls), not a tuned sampler.

use std::collections::HashMap;

/// Sampling parameters for one request. Greedy is the default
/// ([`SampleParams::greedy`]); a non-zero `temperature` enables stochastic
/// sampling. `top_k == 0`, `top_p >= 1.0`, and `min_p == 0.0` each disable that
/// filter. The history-based penalties are off at their identity values
/// (`repetition_penalty == 1.0`, the rest `0`), matching the MLX defaults.
///
/// Not `Copy` because `dry_sequence_breakers` owns a `Vec`; each slot keeps its
/// own clone.
#[derive(Debug, Clone)]
pub struct SampleParams {
    /// Softmax temperature; `<= 0.0` means greedy (argmax).
    pub temperature: f32,
    /// Keep only the `top_k` highest-probability tokens (`0` = disabled).
    pub top_k: usize,
    /// Nucleus: keep the smallest set whose cumulative probability reaches
    /// `top_p` (`>= 1.0` = disabled).
    pub top_p: f32,
    /// Keep tokens with probability `>= min_p * max_probability` (`0.0` = disabled).
    pub min_p: f32,
    /// PRNG seed for reproducibility; `None` lets the engine derive a
    /// deterministic per-request seed.
    pub seed: Option<u64>,
    /// Repetition penalty over the unique seen tokens (`1.0` = disabled): a seen
    /// token's logit is divided by this when positive, multiplied when negative.
    pub repetition_penalty: f32,
    /// OpenAI-style frequency penalty: subtract `frequency_penalty * count` from a
    /// token's logit (`0.0` = disabled).
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty: subtract this once from any token that
    /// appeared at all (`0.0` = disabled).
    pub presence_penalty: f32,
    /// DRY multiplier (`0.0` = disabled). When positive, tokens that would extend
    /// a repeated suffix are penalized.
    pub dry_multiplier: f32,
    /// DRY exponential base (penalty grows as `dry_base^(match_len - allowed)`).
    pub dry_base: f32,
    /// DRY minimum match length before a penalty applies.
    pub dry_allowed_length: usize,
    /// DRY lookback window in tokens (`0` = the whole history).
    pub dry_penalty_last_n: usize,
    /// Token ids that break DRY suffix matching (e.g. newlines, punctuation).
    pub dry_sequence_breakers: Vec<i32>,
}

impl SampleParams {
    /// Greedy (argmax) sampling: temperature 0, no filters, no penalties.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 0,
            dry_sequence_breakers: Vec::new(),
        }
    }

    /// Whether this samples greedily (argmax), i.e. temperature is non-positive.
    /// Note: greedy still applies penalties first when any are enabled, so the
    /// argmax is taken over the penalized logits.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Whether any history-based penalty is enabled. When `false`, sampling needs
    /// no token history and runs exactly as it did before penalties existed.
    /// Mirrors `mlxcel-core`'s `needs_token_history` gate (any non-identity
    /// penalty value; `dry_multiplier` only counts when strictly positive).
    #[must_use]
    pub fn needs_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
            || self.dry_multiplier > 0.0
    }
}

/// splitmix64: a tiny, dependency-free, well-distributed PRNG. The engine threads
/// one `u64` state per slot; this advances it and returns the next draw.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform draw in `[0, 1)` from the top 24 bits of the next PRNG output.
fn next_unit(state: &mut u64) -> f32 {
    ((next_u64(state) >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
}

/// Index of the maximum logit (greedy). Matches the C shim's on-device argmax.
fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as i32
}

/// Divide the logit of each unique seen token by `penalty` when positive,
/// multiply when negative. Mirrors `mlxcel-core`'s `apply_repetition_penalty`
/// (a deduplicated set of seen ids, one op per unique token).
fn apply_repetition(logits: &mut [f32], history: &[i32], penalty: f32) {
    let mut seen: Vec<i32> = history.to_vec();
    seen.sort_unstable();
    seen.dedup();
    for t in seen {
        if t >= 0 && (t as usize) < logits.len() {
            let l = logits[t as usize];
            logits[t as usize] = if l > 0.0 { l / penalty } else { l * penalty };
        }
    }
}

/// Subtract `frequency * count + presence` from each seen token's logit. Mirrors
/// `mlxcel-core`'s `apply_frequency_presence_penalty`.
fn apply_frequency_presence(logits: &mut [f32], history: &[i32], frequency: f32, presence: f32) {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &t in history {
        *counts.entry(t).or_insert(0) += 1;
    }
    for (t, c) in counts {
        if t >= 0 && (t as usize) < logits.len() {
            logits[t as usize] -= frequency * c as f32 + presence;
        }
    }
}

/// Penalize tokens that would extend a repeated suffix (DRY). For each earlier
/// occurrence of the last token, extend the backward match (stopping at a sequence
/// breaker); when the match exceeds `dry_allowed_length`, penalize the token that
/// followed that occurrence by `dry_multiplier * dry_base^(match_len - allowed)`,
/// keeping the largest penalty per token. Mirrors `mlxcel-core`'s
/// `apply_dry_penalty`.
fn apply_dry(logits: &mut [f32], history: &[i32], params: &SampleParams) {
    let hlen = history.len();
    if hlen < 2 {
        return;
    }
    let window: &[i32] = if params.dry_penalty_last_n == 0 {
        history
    } else {
        &history[hlen.saturating_sub(params.dry_penalty_last_n)..]
    };
    let wlen = window.len();
    if wlen < 2 {
        return;
    }

    let mut positions: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &t) in window.iter().enumerate() {
        positions.entry(t).or_default().push(i);
    }

    let last = window[wlen - 1];
    let mut penalties: HashMap<i32, f32> = HashMap::new();
    if let Some(pos_list) = positions.get(&last) {
        for &pos in pos_list {
            if pos >= wlen - 1 {
                continue;
            }
            let mut match_len = 1usize;
            let mut p1 = pos;
            let mut p2 = wlen - 1;
            while p1 > 0 && p2 > 0 {
                p1 -= 1;
                p2 -= 1;
                if params.dry_sequence_breakers.contains(&window[p1]) {
                    break;
                }
                if window[p1] == window[p2] {
                    match_len += 1;
                } else {
                    break;
                }
            }
            if match_len > params.dry_allowed_length {
                let next_pos = pos + 1;
                if next_pos < wlen {
                    let next_token = window[next_pos];
                    let penalty = params.dry_multiplier
                        * params
                            .dry_base
                            .powi((match_len - params.dry_allowed_length) as i32);
                    let entry = penalties.entry(next_token).or_insert(0.0);
                    if penalty > *entry {
                        *entry = penalty;
                    }
                }
            }
        }
    }
    for (t, pen) in penalties {
        if t >= 0 && (t as usize) < logits.len() {
            logits[t as usize] -= pen;
        }
    }
}

/// Apply every enabled history-based penalty to `logits` in place, in the same
/// order as the MLX path: repetition, then DRY, then frequency/presence.
fn apply_penalties(logits: &mut [f32], params: &SampleParams, history: &[i32]) {
    if history.is_empty() {
        return;
    }
    if params.repetition_penalty != 1.0 {
        apply_repetition(logits, history, params.repetition_penalty);
    }
    if params.dry_multiplier > 0.0 {
        apply_dry(logits, history, params);
    }
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        apply_frequency_presence(
            logits,
            history,
            params.frequency_penalty,
            params.presence_penalty,
        );
    }
}

/// Sample a token id from `logits` under `params` given the request's prior
/// `history` (prompt + tokens generated so far), advancing the PRNG `rng`.
///
/// History-based penalties are applied to a local copy of the logits first, so
/// greedy too picks the argmax of the penalized logits. Then greedy
/// short-circuits to argmax; otherwise temperature-scaled stable softmax, then
/// top-k, top-p, and min-p filtering, then a categorical draw. Falls back to
/// argmax if filtering leaves nothing with positive mass. When no penalty is
/// enabled (or the history is empty) the logits are used as-is with no copy, so
/// greedy stays token-exact with the on-device argmax.
pub(crate) fn sample(logits: &[f32], params: &SampleParams, history: &[i32], rng: &mut u64) -> i32 {
    if logits.is_empty() {
        return 0;
    }
    let penalized: Vec<f32>;
    let logits: &[f32] = if params.needs_penalties() && !history.is_empty() {
        let mut v = logits.to_vec();
        apply_penalties(&mut v, params, history);
        penalized = v;
        &penalized
    } else {
        logits
    };
    if params.is_greedy() {
        return argmax(logits);
    }

    let t = params.temperature.max(1e-6);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // (index, unnormalized prob) via a numerically stable exp.
    let mut cand: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, ((l - max_logit) / t).exp()))
        .collect();

    // Sort by probability descending. top-k and top-p both need the ordering; a
    // single sort serves both.
    cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // top-k: keep the k highest.
    if params.top_k > 0 && params.top_k < cand.len() {
        cand.truncate(params.top_k);
    }

    // top-p (nucleus): smallest prefix whose cumulative (normalized) prob reaches p.
    if params.top_p < 1.0 {
        let total: f32 = cand.iter().map(|x| x.1).sum();
        if total > 0.0 {
            let mut cum = 0.0;
            let mut keep = cand.len();
            for (i, &(_, p)) in cand.iter().enumerate() {
                cum += p / total;
                if cum >= params.top_p {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep);
        }
    }

    // min-p: keep tokens within min_p of the top probability (cand[0] is the max).
    if params.min_p > 0.0 && !cand.is_empty() {
        let thresh = params.min_p * cand[0].1;
        cand.retain(|&(_, p)| p >= thresh);
    }

    // `total` is a sum of finite non-negative exponentials, so `<= 0.0` (no draw
    // possible) is the only degenerate case.
    let total: f32 = cand.iter().map(|x| x.1).sum();
    if cand.is_empty() || total <= 0.0 {
        return argmax(logits);
    }
    let r = next_unit(rng) * total;
    let mut acc = 0.0;
    for &(idx, p) in &cand {
        acc += p;
        if r < acc {
            return idx as i32;
        }
    }
    // Floating-point slack: fall back to the last kept candidate.
    cand.last()
        .map_or_else(|| argmax(logits), |&(idx, _)| idx as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sampling params with the given temperature / top-k / top-p and no penalties.
    fn params(temperature: f32, top_k: usize, top_p: f32) -> SampleParams {
        SampleParams {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            seed: Some(42),
            ..SampleParams::greedy()
        }
    }

    /// Greedy params with the given history-based penalties (deterministic, so the
    /// penalized argmax is exactly assertable).
    fn penalized(repetition: f32, frequency: f32, presence: f32) -> SampleParams {
        SampleParams {
            repetition_penalty: repetition,
            frequency_penalty: frequency,
            presence_penalty: presence,
            ..SampleParams::greedy()
        }
    }

    #[test]
    fn greedy_is_argmax() {
        let logits = [0.1, 0.5, 0.3, 9.0, 0.2];
        let mut rng = 1;
        assert_eq!(sample(&logits, &SampleParams::greedy(), &[], &mut rng), 3);
        // temperature 0 short-circuits regardless of filters
        assert_eq!(sample(&logits, &params(0.0, 0, 1.0), &[], &mut rng), 3);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = [1.0, 4.0, 2.0, 3.0];
        let mut rng = 7;
        // top_k=1 leaves only the max, so sampling is deterministic = argmax.
        for _ in 0..16 {
            assert_eq!(sample(&logits, &params(1.0, 1, 1.0), &[], &mut rng), 1);
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits = [2.0, 2.0, 2.0, 2.0, 2.0]; // uniform -> spread of outcomes
        let run = || {
            let mut rng = 12345u64;
            (0..32)
                .map(|_| sample(&logits, &params(1.0, 0, 1.0), &[], &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn top_p_restricts_to_the_nucleus() {
        // token 0 dominates; a tight nucleus must always pick it.
        let logits = [10.0, 0.0, 0.0, 0.0];
        let mut rng = 99;
        for _ in 0..16 {
            assert_eq!(sample(&logits, &params(1.0, 0, 0.5), &[], &mut rng), 0);
        }
    }

    #[test]
    fn empty_logits_is_safe() {
        let mut rng = 1;
        assert_eq!(sample(&[], &SampleParams::greedy(), &[], &mut rng), 0);
    }

    #[test]
    fn needs_penalties_matches_identity_values() {
        assert!(!SampleParams::greedy().needs_penalties());
        assert!(penalized(1.1, 0.0, 0.0).needs_penalties());
        assert!(penalized(0.8, 0.0, 0.0).needs_penalties());
        assert!(penalized(1.0, 0.5, 0.0).needs_penalties());
        assert!(penalized(1.0, 0.0, 0.5).needs_penalties());
        let mut d = SampleParams::greedy();
        d.dry_multiplier = 0.8;
        assert!(d.needs_penalties());
    }

    #[test]
    fn repetition_penalty_divides_positive_seen_logit() {
        let mut rng = 1;
        let logits = [2.0, 1.4, 0.5];
        // Empty history: nothing to penalize, argmax stays token 0.
        assert_eq!(sample(&logits, &penalized(2.0, 0.0, 0.0), &[], &mut rng), 0);
        // Token 0 seen: its 2.0 logit halves to 1.0 < 1.4, so greedy moves to token 1.
        assert_eq!(
            sample(&logits, &penalized(2.0, 0.0, 0.0), &[0], &mut rng),
            1
        );
    }

    #[test]
    fn repetition_penalty_multiplies_negative_seen_logit() {
        let mut v = [-1.0f32, 0.5];
        apply_repetition(&mut v, &[0, 0], 2.0); // dedup -> one application
        assert_eq!(v[0], -2.0); // negative logit is multiplied
        assert_eq!(v[1], 0.5);
    }

    #[test]
    fn frequency_presence_penalty_subtracts_count_and_presence() {
        let mut v = [0.0f32, 0.0, 0.0];
        // token 1 appears twice, token 2 once.
        apply_frequency_presence(&mut v, &[1, 1, 2], 0.5, 0.25);
        assert!((v[1] - -(0.5 * 2.0 + 0.25)).abs() < 1e-6); // -1.25
        assert!((v[2] - -(0.5 * 1.0 + 0.25)).abs() < 1e-6); // -0.75
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn dry_penalizes_followup_after_suffix_match() {
        // history a b c a b (0 1 2 0 1); last token b's earlier occurrence at idx 1
        // back-matches "a b" (len 2) > allowed 1, so the follow token c is penalized
        // by dry_multiplier * dry_base^(2 - 1) = 1.0 * 2.0 = 2.0.
        let mut v = [0.0f32; 3];
        let history = [0, 1, 2, 0, 1];
        let p = SampleParams {
            dry_multiplier: 1.0,
            dry_base: 2.0,
            dry_allowed_length: 1,
            ..SampleParams::greedy()
        };
        apply_dry(&mut v, &history, &p);
        assert!((v[2] - -2.0).abs() < 1e-6, "v={v:?}");
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
    }

    #[test]
    fn dry_sequence_breaker_stops_the_match() {
        // Same history, but token a (0) is a sequence breaker, so the back-match
        // breaks before reaching length 2 and nothing is penalized.
        let mut v = [0.0f32; 3];
        let history = [0, 1, 2, 0, 1];
        let p = SampleParams {
            dry_multiplier: 1.0,
            dry_base: 2.0,
            dry_allowed_length: 1,
            dry_sequence_breakers: vec![0],
            ..SampleParams::greedy()
        };
        apply_dry(&mut v, &history, &p);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn greedy_argmax_is_taken_over_penalized_logits() {
        let logits = [3.0, 2.0];
        let mut rng = 1;
        // Greedy with no penalty keeps token 0 even if it is in history.
        assert_eq!(sample(&logits, &SampleParams::greedy(), &[0], &mut rng), 0);
        // Repetition penalty halves token 0 (3.0 -> 1.5 < 2.0), flipping greedy.
        assert_eq!(
            sample(&logits, &penalized(2.0, 0.0, 0.0), &[0], &mut rng),
            1
        );
    }

    #[test]
    fn no_penalty_ignores_history() {
        // With every penalty at its identity value, history must not change output.
        let logits = [3.0, 2.0, 1.0];
        let mut rng = 1;
        assert_eq!(
            sample(&logits, &params(0.0, 0, 1.0), &[0, 0, 1, 2], &mut rng),
            0
        );
    }
}
