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
//! Supported: temperature, top-k, top-p (nucleus), min-p, and a seeded PRNG for
//! reproducibility. Repetition / frequency / presence penalties and DRY are not
//! applied here (the server warns once when requested).
//!
//! Cost note: a non-greedy step softmaxes and sorts the full vocabulary per active
//! row on the host. That is acceptable for this reference backend (it is a small
//! fraction of the decode matmuls), not a tuned sampler.

/// Sampling parameters for one request. Greedy is the default
/// ([`SampleParams::greedy`]); a non-zero `temperature` enables stochastic
/// sampling. `top_k == 0`, `top_p >= 1.0`, and `min_p == 0.0` each disable that
/// filter.
#[derive(Debug, Clone, Copy)]
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
}

impl SampleParams {
    /// Greedy (argmax) sampling: temperature 0, no filters.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
        }
    }

    /// Whether this samples greedily (argmax), i.e. temperature is non-positive.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
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

/// Sample a token id from `logits` under `params`, advancing the PRNG `rng`.
///
/// Greedy short-circuits to argmax. Otherwise: temperature-scaled stable softmax,
/// then top-k, top-p, and min-p filtering, then a categorical draw. Falls back to
/// argmax if filtering leaves nothing with positive mass.
pub(crate) fn sample(logits: &[f32], params: &SampleParams, rng: &mut u64) -> i32 {
    if logits.is_empty() {
        return 0;
    }
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

    fn params(temperature: f32, top_k: usize, top_p: f32) -> SampleParams {
        SampleParams {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            seed: Some(42),
        }
    }

    #[test]
    fn greedy_is_argmax() {
        let logits = [0.1, 0.5, 0.3, 9.0, 0.2];
        let mut rng = 1;
        assert_eq!(sample(&logits, &SampleParams::greedy(), &mut rng), 3);
        // temperature 0 short-circuits regardless of filters
        assert_eq!(sample(&logits, &params(0.0, 0, 1.0), &mut rng), 3);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = [1.0, 4.0, 2.0, 3.0];
        let mut rng = 7;
        // top_k=1 leaves only the max, so sampling is deterministic = argmax.
        for _ in 0..16 {
            assert_eq!(sample(&logits, &params(1.0, 1, 1.0), &mut rng), 1);
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits = [2.0, 2.0, 2.0, 2.0, 2.0]; // uniform -> spread of outcomes
        let run = || {
            let mut rng = 12345u64;
            (0..32)
                .map(|_| sample(&logits, &params(1.0, 0, 1.0), &mut rng))
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
            assert_eq!(sample(&logits, &params(1.0, 0, 0.5), &mut rng), 0);
        }
    }

    #[test]
    fn empty_logits_is_safe() {
        let mut rng = 1;
        assert_eq!(sample(&[], &SampleParams::greedy(), &mut rng), 0);
    }
}
