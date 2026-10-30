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

//! The stream guards of `speculative_distribution_tests` run with the
//! full-model drafter (#2757) over the same stub targets: a second stub model
//! drafts from its own caches through [`FullModelDrafter`], so the drafter's
//! own cache rollback is under the same chi-square, Markov and temperature-0
//! guards as the engine's verify round. The hypotheses and thresholds are the
//! ones that module states.

use super::direct::DirectRequest;
use super::speculative_distribution_tests::{
    CHI2_DOF5_ALPHA_1E4, CHI2_DOF30_ALPHA_1E4, GREEDY_TARGET, MIXED_DRAFT, MarkovModel, Orbit,
    REJECTING_DRAFT, TARGET_LOGITS, TEMPERATURE, VOCAB, chi_square, markov_logits, sample_budget,
    softmax, stochastic_config,
};
use super::*;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::speculative::full_model_drafter::{DraftModelState, FullModelDrafter};

/// One run on the engine's drafter loop with `draft` drafting for `target`.
fn run_full_model<T: LanguageModel, D: LanguageModel>(
    target: T,
    draft: &D,
    state: &mut DraftModelState,
    max_tokens: usize,
    block: usize,
    config: &SamplingConfig,
) -> SpeculativeRun {
    let mut drafter = FullModelDrafter::new(draft, state);
    DirectEngine::new(target, 512)
        .generate_with_drafter(
            &DirectRequest::text(&[2], max_tokens, config),
            &mut drafter,
            block,
            |_| true,
        )
        .expect("the stub models run token-only verify rounds")
}

/// H0 on the i.i.d. stream (`speculative_stream_matches_the_target_distribution`)
/// with the orbit draft model proposing: its proposals depend on the current
/// token, so a round that kept a rejected proposal in the drafter's caches
/// would skew what it proposes next, and the target's draws stay what the
/// statistic sees.
#[test]
fn full_model_stream_matches_the_target_distribution() {
    use super::speculative_distribution_tests::ContextFreeModel;
    let config = stochastic_config();
    let expected = softmax(&TARGET_LOGITS, TEMPERATURE);
    let mut state = DraftModelState::new(&MIXED_DRAFT).unwrap();
    let mut counts = vec![0u32; VOCAB];
    let mut collected = 0usize;
    let budget = sample_budget();
    while collected < budget {
        let per_call = 64.min(budget - collected);
        let run = run_full_model(
            &ContextFreeModel,
            &MIXED_DRAFT,
            &mut state,
            per_call,
            4,
            &config,
        );
        assert!(!run.run.tokens.is_empty(), "every call must make progress");
        for &t in &run.run.tokens {
            counts[t as usize] += 1;
        }
        collected += run.run.tokens.len();
    }
    let chi2 = chi_square(&counts, &expected);
    assert!(
        chi2 < CHI2_DOF5_ALPHA_1E4,
        "stream is distinguishable from target-only sampling: chi-square {chi2:.3} exceeds \
         {CHI2_DOF5_ALPHA_1E4}. counts={counts:?}, expected={expected:?}"
    );
}

/// The conditional guard (`markov_stream_keeps_the_target_conditional`) with
/// the full-model drafter: the accept decision and both rollbacks under one
/// statistic that sees the token before each draw.
#[test]
fn full_model_markov_stream_keeps_the_target_conditional() {
    const PROMPT_TOKEN: i32 = 2;
    let config = stochastic_config();
    let mut state = DraftModelState::new(&MIXED_DRAFT).unwrap();
    let mut counts = [[0u32; VOCAB]; VOCAB];
    let mut collected = 0usize;
    let budget = sample_budget();
    while collected < budget {
        let per_call = 64.min(budget - collected);
        let run = run_full_model(MarkovModel, &MIXED_DRAFT, &mut state, per_call, 4, &config);
        assert!(!run.run.tokens.is_empty(), "every call must make progress");
        let mut previous = PROMPT_TOKEN;
        for &t in &run.run.tokens {
            counts[previous as usize][t as usize] += 1;
            previous = t;
        }
        collected += run.run.tokens.len();
    }
    let chi2: f64 = (0..VOCAB as i32)
        .map(|prev| {
            let expected = softmax(&markov_logits(prev), TEMPERATURE);
            chi_square(&counts[prev as usize], &expected)
        })
        .sum();
    assert!(
        chi2 < CHI2_DOF30_ALPHA_1E4,
        "stream breaks the target's conditional: chi-square {chi2:.3} exceeds \
         {CHI2_DOF30_ALPHA_1E4}. counts={counts:?}"
    );
}

/// Temperature 0 is byte-identical to the greedy target alone, whatever the
/// draft model proposes and at every block width, and the acceptance counters
/// say how often the two orbits agreed.
#[test]
fn full_model_temperature_zero_stream_is_byte_identical_to_greedy_target_only() {
    const N: usize = 48;
    let mut tok = 2;
    let expected: Vec<i32> = (0..N)
        .map(|_| {
            tok = GREEDY_TARGET.next(tok);
            tok
        })
        .collect();
    for (draft, agrees) in [
        (MIXED_DRAFT, "sometimes"),
        (REJECTING_DRAFT, "never"),
        (GREEDY_TARGET, "always"),
    ] {
        for block in [1usize, 2, 4] {
            let mut state = DraftModelState::new(&draft).unwrap();
            let run = run_full_model(
                GREEDY_TARGET,
                &draft,
                &mut state,
                N,
                block,
                &SamplingConfig::greedy(),
            );
            assert_eq!(run.run.tokens, expected, "block={block}, agrees {agrees}");
            let rounds = run.rounds;
            assert!(rounds.proposed_draft_tokens > 0, "block={block}");
            match agrees {
                "never" => assert_eq!(rounds.accepted_draft_tokens, 0, "block={block}"),
                "always" => assert_eq!(
                    rounds.accepted_draft_tokens, rounds.proposed_draft_tokens,
                    "block={block}"
                ),
                _ => assert!(
                    rounds.accepted_draft_tokens < rounds.proposed_draft_tokens,
                    "block={block}"
                ),
            }
            // The drafter's caches end a prefix of the context, as the
            // target's state does; `N` tokens plus the one-token prompt.
            assert!(state.cached_positions() <= N + 1, "block={block}");
        }
    }
}

/// A draft model that disagrees with the target on every token costs its
/// forwards and gains nothing; the stream is still exact. Pins the shape the
/// PR's "drafter does not help" measurement has.
#[test]
fn a_rejecting_draft_model_costs_forwards_and_changes_nothing() {
    const N: usize = 24;
    let draft: Orbit = REJECTING_DRAFT;
    let mut state = DraftModelState::new(&draft).unwrap();
    let run = run_full_model(
        GREEDY_TARGET,
        &draft,
        &mut state,
        N,
        3,
        &SamplingConfig::greedy(),
    );
    assert_eq!(run.run.tokens.len(), N);
    assert_eq!(run.rounds.accepted_draft_tokens, 0);
    // One prefill chunk, then one forward per proposal (the last rounds
    // propose fewer than the block, capped under `max_tokens`).
    assert_eq!(
        state.stats().draft_forwards,
        1 + run.rounds.proposed_draft_tokens
    );
    assert!(run.rounds.proposed_draft_tokens >= 3 * (run.rounds.drafted_rounds - 1));
}
