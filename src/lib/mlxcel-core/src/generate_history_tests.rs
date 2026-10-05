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

//! Every `CxxGenerator` decode loop must sample each step against the history
//! that already holds the token it just read (#2090), and must stay exactly
//! plain sequential decoding when the sampler reads no history.

use super::{CxxGenerator, SamplingConfig};
use crate::test_support::induction::{
    INDUCTION_MODELS, INDUCTION_VOCAB, InductionModel, lcg_tokens, sequential_reference,
};

const MAX_TOKENS: usize = 96;

/// The four public decode loops, each run once on a fresh generator; the
/// embeddings variants run without embeddings, which falls back to the token
/// forward.
fn every_entry_point(
    model: &InductionModel,
    prompt: &[i32],
    sampling: &SamplingConfig,
) -> [(&'static str, Vec<i32>); 4] {
    [
        (
            "generate",
            CxxGenerator::new(1).generate(model, prompt, MAX_TOKENS, sampling),
        ),
        (
            "generate_streaming_with_embeddings",
            CxxGenerator::new(1).generate_streaming_with_embeddings(
                model,
                prompt,
                None,
                None,
                MAX_TOKENS,
                sampling,
                |_| true,
            ),
        ),
        (
            "generate_with_stats_and_embeddings",
            CxxGenerator::new(1)
                .generate_with_stats_and_embeddings(model, prompt, None, None, MAX_TOKENS, sampling)
                .0,
        ),
        (
            "generate_with_stats",
            CxxGenerator::new(1)
                .generate_with_stats(model, prompt, MAX_TOKENS, sampling)
                .0,
        ),
    ]
}

/// `prompt_alphabet` below [`INDUCTION_VOCAB`] leaves tokens the prompt never
/// uses, so the reply introduces them and a penalty set that misses the
/// newest token visibly differs even over the full history.
fn assert_every_entry_point_is_the_reference(sampling: &SamplingConfig, prompt_alphabet: u64) {
    for model in &INDUCTION_MODELS {
        for seed in 1..=6 {
            let prompt = lcg_tokens(seed, 48, prompt_alphabet);
            let expected = sequential_reference(model, &prompt, MAX_TOKENS, sampling);
            for (entry, tokens) in every_entry_point(model, &prompt, sampling) {
                assert_eq!(
                    tokens, expected,
                    "{entry}, earliest={}, seed {seed}, {sampling:?}",
                    model.earliest
                );
            }
        }
    }
}

/// The issue's reproduction: with a three-token window, a penalty that missed
/// the token just read let `InductionModel` repeat it (`5, 5, 5, ...`) where
/// the reference emits `5, 4, 0, 3, ...`.
#[test]
fn every_decode_loop_penalizes_the_token_it_just_read() {
    let sampling = SamplingConfig {
        repetition_penalty: 1.3,
        penalty_last_n: 3,
        ..SamplingConfig::greedy()
    };
    assert!(sampling.needs_token_history());
    assert_every_entry_point_is_the_reference(&sampling, INDUCTION_VOCAB as u64);
}

/// The full-history window goes through the incremental `SamplerState`,
/// which syncs to the history it is handed: it must be handed the token too.
/// Counts rather than a set, because over the full history a repetition
/// penalty's set rarely changes once the reply has used each token once,
/// while every occurrence moves a frequency penalty.
#[test]
fn every_decode_loop_feeds_the_incremental_penalty_state_the_token_it_just_read() {
    let sampling = SamplingConfig {
        repetition_penalty: 1.1,
        frequency_penalty: 0.6,
        presence_penalty: 0.4,
        penalty_last_n: -1,
        ..SamplingConfig::greedy()
    };
    assert!(sampling.needs_token_history());
    assert_every_entry_point_is_the_reference(&sampling, INDUCTION_VOCAB as u64 / 2);
}

#[test]
fn every_decode_loop_counts_the_token_it_just_read_for_frequency_and_presence() {
    let sampling = SamplingConfig {
        frequency_penalty: 0.6,
        presence_penalty: 0.4,
        penalty_last_n: 8,
        ..SamplingConfig::greedy()
    };
    assert!(sampling.needs_token_history());
    assert_every_entry_point_is_the_reference(&sampling, INDUCTION_VOCAB as u64);
}

#[test]
fn every_decode_loop_extends_the_dry_match_with_the_token_it_just_read() {
    let sampling = SamplingConfig {
        dry_multiplier: 0.8,
        ..SamplingConfig::greedy()
    };
    assert!(sampling.needs_token_history());
    assert_every_entry_point_is_the_reference(&sampling, INDUCTION_VOCAB as u64);
}

/// Without a history-reading sampler the loops keep sampling from the lazy
/// graph ahead of the host read, and their output is plain decoding's.
#[test]
fn every_decode_loop_is_plain_decoding_without_a_penalty() {
    let sampling = SamplingConfig::greedy();
    assert!(!sampling.needs_token_history());
    assert_every_entry_point_is_the_reference(&sampling, INDUCTION_VOCAB as u64);
}
