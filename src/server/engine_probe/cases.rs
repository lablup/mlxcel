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

//! The sampling configurations the parity harness runs, and the description
//! of a server prefill partition.
//!
//! Every configuration goes through [`build_sampling_config`], the helper both
//! the CLI and the server resolve their request parameters through, so the
//! one `SamplingConfig` built here is what each path receives.

use mlxcel_core::generate::{DRY_FULL_HISTORY, SamplingConfig};

use crate::execution::sampling::{ResolvedSamplingParams, build_sampling_config};
use crate::tokenizer::MlxcelTokenizer;

/// b10621's default DRY sequence breakers (`\n`, `:`, `"`, `*`), the set the
/// server applies when `--dry-sequence-breaker` is absent.
pub const DEFAULT_DRY_BREAKER_STRINGS: [&str; 4] = ["\n", ":", "\"", "*"];

/// A parity case: a name and the sampling configuration every path runs.
#[derive(Debug, Clone)]
pub struct ParityCase {
    pub name: &'static str,
    pub sampling: SamplingConfig,
}

/// Neutral request parameters: greedy, every penalty and optional sampler
/// off, the given stop ids.
#[must_use]
pub fn neutral_params(stop_token_ids: Vec<i32>) -> ResolvedSamplingParams {
    ResolvedSamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: None,
        repetition_penalty: 1.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: DRY_FULL_HISTORY,
        dry_sequence_breakers: Vec::new(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        top_n_sigma: 0.0,
        typical_p: 1.0,
        p_less: false,
        penalty_last_n: -1,
        stop_token_ids,
        mirostat: 0,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        dynatemp_range: 0.0,
        dynatemp_exponent: 1.0,
        adaptive_target: -1.0,
        adaptive_decay: 0.9,
        min_keep: 0,
    }
}

/// Greedy decoding with no penalties.
#[must_use]
pub fn greedy_case(stop_token_ids: Vec<i32>) -> ParityCase {
    ParityCase {
        name: "greedy",
        sampling: build_sampling_config(neutral_params(stop_token_ids)),
    }
}

/// Seeded stochastic sampling at the server's sampler defaults (temperature
/// 0.8, top-k 40, top-p 0.95, min-p 0.05) with repetition, frequency and
/// presence penalties over a 64-token window and DRY with the default
/// breakers, so every history-dependent part of the sampling step is live.
#[must_use]
pub fn seeded_penalties_case(
    stop_token_ids: Vec<i32>,
    dry_breaker_ids: Vec<i32>,
    seed: u64,
) -> ParityCase {
    ParityCase {
        name: "seeded-penalties-dry",
        sampling: build_sampling_config(ResolvedSamplingParams {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: Some(seed),
            repetition_penalty: 1.1,
            frequency_penalty: 0.1,
            presence_penalty: 0.1,
            penalty_last_n: 64,
            dry_multiplier: 0.8,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            dry_sequence_breakers: dry_breaker_ids,
            ..neutral_params(stop_token_ids)
        }),
    }
}

/// Token ids for the default DRY breakers that encode to exactly one token.
///
/// `SamplingConfig::dry_sequence_breakers` holds single-token ids, the form
/// both paths consume directly. A breaker that spans several tokens has no
/// single-id form and is skipped, so the case stays identical on both paths
/// (the server's string-derived breaker heads are not used here).
#[must_use]
pub fn default_dry_breaker_ids(tokenizer: &MlxcelTokenizer) -> Vec<i32> {
    let mut ids = Vec::new();
    for breaker in DEFAULT_DRY_BREAKER_STRINGS {
        if let Ok(encoded) = tokenizer.encode(breaker, false)
            && let [id] = encoded.as_slice()
        {
            let id = *id as i32;
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

/// How many leading prompt tokens the parity harness stores in the prompt
/// cache before its "hit" request: the history prefix (the part of the
/// prompt a follow-up chat turn shares with the stored conversation) when the
/// prompt has one, else three quarters of the prompt. Always leaves at least
/// one token to prefill.
#[must_use]
pub fn cache_prime_len(prompt: &[i32], history: Option<&[i32]>) -> usize {
    let shared = history.map_or(0, |h| {
        h.iter().zip(prompt).take_while(|(a, b)| a == b).count()
    });
    let len = if shared > 0 {
        shared
    } else {
        prompt.len() * 3 / 4
    };
    len.min(prompt.len().saturating_sub(1))
}

/// Describe how the server prefills a prompt of `prompt_len` tokens: the
/// adopted prompt-cache prefix, the history-boundary split, and the chunk.
#[must_use]
pub fn describe_prefill_partition(
    prompt_len: usize,
    cached_tokens: usize,
    history_boundary: Option<usize>,
    prefill_chunk: usize,
) -> String {
    let start = cached_tokens.min(prompt_len);
    let mut parts = Vec::new();
    if start > 0 {
        parts.push(format!("adopted[0..{start})"));
    }
    match history_boundary {
        Some(boundary) if boundary > start && boundary < prompt_len => {
            parts.push(format!("prefill[{start}..{boundary})"));
            parts.push(format!("prefill[{boundary}..{prompt_len})"));
        }
        _ => parts.push(format!("prefill[{start}..{prompt_len})")),
    }
    let mut out = parts.join("+");
    if prefill_chunk > 0 && prompt_len - start > prefill_chunk {
        out.push_str(&format!(" chunk={prefill_chunk}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_case_is_greedy_without_penalties() {
        let case = greedy_case(vec![2]);
        assert_eq!(case.sampling.temperature, 0.0);
        assert!(!case.sampling.needs_token_history());
        assert_eq!(case.sampling.stop_token_ids, vec![2]);
    }

    #[test]
    fn seeded_case_activates_history_dependent_sampling() {
        let case = seeded_penalties_case(vec![2], vec![198], 7);
        let s = &case.sampling;
        assert_eq!(s.seed, Some(7));
        assert!(s.temperature > 0.0);
        assert!(s.needs_token_history());
        assert_eq!(s.penalty_last_n, 64);
        assert_eq!(s.dry_penalty_last_n, 64);
        assert_eq!(s.dry_sequence_breakers, vec![198]);
    }

    #[test]
    fn cache_prime_len_prefers_the_history_prefix() {
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(cache_prime_len(&prompt, Some(&[1, 2, 3, 4, 5, 9])), 5);
        assert_eq!(cache_prime_len(&prompt, None), 6);
        // A history equal to the whole prompt still leaves one token.
        assert_eq!(cache_prime_len(&prompt, Some(&prompt)), 7);
        assert_eq!(cache_prime_len(&[], None), 0);
    }

    #[test]
    fn partition_cold_unsplit() {
        assert_eq!(
            describe_prefill_partition(100, 0, None, 512),
            "prefill[0..100)"
        );
    }

    #[test]
    fn partition_cold_history_split() {
        assert_eq!(
            describe_prefill_partition(100, 0, Some(80), 512),
            "prefill[0..80)+prefill[80..100)"
        );
    }

    #[test]
    fn partition_adopted_prefix_skips_covered_boundary() {
        assert_eq!(
            describe_prefill_partition(100, 99, Some(80), 512),
            "adopted[0..99)+prefill[99..100)"
        );
    }

    #[test]
    fn partition_reports_chunking() {
        assert_eq!(
            describe_prefill_partition(2000, 0, None, 512),
            "prefill[0..2000) chunk=512"
        );
    }
}
