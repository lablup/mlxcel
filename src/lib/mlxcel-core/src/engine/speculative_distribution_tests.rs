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

//! Distribution and rollback guards for the engine's token-only verify loop
//! ([`DirectEngine::generate_with_drafter`], [`Engine::verify_round`]).
//!
//! **Null hypothesis (H0):** the emitted stream at `temperature > 0` is
//! distributed as sampling the target alone. The target's logits ignore its
//! input, so target-only sampling emits i.i.d. draws from one known categorical
//! `p`, and a Pearson chi-square goodness-of-fit test against `p` tests H0.
//! The statistic has `dof = 5`; the threshold `CHI2_DOF5_ALPHA_1E4` is the
//! upper-tail quantile at `alpha = 1e-4`.
//!
//! A context-free target cannot see which position a token was sampled at or
//! what preceded it, and the round emits the target's draw whether or not it
//! confirms the proposal. So the two i.i.d. stream tests guard the sampler path
//! into the stream, not the accept/reject decision: a round that ignores a
//! rejection passes them. `markov_stream_keeps_the_target_conditional` is the
//! stochastic guard on that decision, because its target's distribution depends
//! on the token before it.
//!
//! `chi_square_rejects_known_wrong_acceptance_rules` measures the statistic's
//! power on the same sample count, on host draws rather than on the engine.
//! Chi-square cannot see one wrong token, so the support and temperature-0 tests
//! below are zero-tolerance. The cache rollback guards on the same stubs are in
//! `verify_round_rollback_tests`.

use super::direct::DirectRequest;
use super::*;
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr, dtype, ffi};

/// Upper-tail chi-square quantile at alpha = 1e-4, 5 degrees of freedom.
pub(super) const CHI2_DOF5_ALPHA_1E4: f64 = 25.7448;
/// Upper-tail chi-square quantile at alpha = 1e-4, 30 degrees of freedom: six
/// conditional rows of five each.
pub(super) const CHI2_DOF30_ALPHA_1E4: f64 = 67.6326;
/// Tokens per distributional run; `MLXCEL_SPEC_DIST_SAMPLES` overrides it.
const SAMPLE_TOKENS: usize = 12_000;
pub(super) const VOCAB: usize = 6;
pub(super) const TEMPERATURE: f32 = 0.7;
/// Deliberately not flat, so a bias in either direction reaches the statistic.
pub(super) const TARGET_LOGITS: [f32; VOCAB] = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0];
/// Agrees with the target often enough to be a realistic drafter, but disagrees
/// on the ordering of the top tokens.
const DRAFT_LOGITS: [f32; VOCAB] = [1.4, 1.3, 0.2, 0.6, -0.3, -0.8];

pub(super) fn sample_budget() -> usize {
    std::env::var("MLXCEL_SPEC_DIST_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SAMPLE_TOKENS)
}

pub(super) fn softmax(logits: &[f32], temperature: f32) -> Vec<f64> {
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&x| f64::from(x) / f64::from(temperature))
        .collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

pub(super) fn chi_square(observed: &[u32], expected: &[f64]) -> f64 {
    let n = f64::from(observed.iter().sum::<u32>());
    observed
        .iter()
        .zip(expected)
        .map(|(&o, &e)| {
            let e = n * e;
            let d = f64::from(o) - e;
            d * d / e
        })
        .sum()
}

pub(super) fn stochastic_config() -> SamplingConfig {
    SamplingConfig {
        temperature: TEMPERATURE,
        ..SamplingConfig::default()
    }
}

/// Host-side categorical draws for the proposals, independent of the MLX key
/// sequence the target's sampler consumes.
pub(super) struct Categorical {
    cdf: Vec<f64>,
    state: u64,
}

impl Categorical {
    fn new(probs: &[f64], seed: u64) -> Self {
        let mut acc = 0.0;
        let cdf = probs
            .iter()
            .map(|p| {
                acc += p;
                acc
            })
            .collect();
        Self { cdf, state: seed }
    }

    fn draw(&mut self) -> i32 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let word = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let u = (word >> 11) as f64 / (1u64 << 53) as f64;
        self.cdf
            .iter()
            .position(|&c| u < c)
            .unwrap_or(self.cdf.len() - 1) as i32
    }
}

/// Fills the pool caches as a real model would, so the rewind on rejection runs.
fn advance_caches(input_ids: &MlxArray, caches: &mut [KVCache]) -> Vec<i32> {
    let shape = ffi::array_shape(input_ids);
    for cache in caches.iter_mut() {
        let k = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
        let v = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
        cache.update(k, v);
    }
    shape
}

/// Same logits at every position whatever the input, on two cache layers so a
/// per-layer trim mismatch is visible.
pub(super) struct ContextFreeModel;

impl LanguageModel for ContextFreeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = advance_caches(input_ids, caches);
        let n = (shape[0] * shape[1]) as usize;
        let out: Vec<f32> = TARGET_LOGITS
            .iter()
            .copied()
            .cycle()
            .take(n * VOCAB)
            .collect();
        ffi::from_slice_f32(&out, &[shape[0], shape[1], VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new(), KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn supports_padded_prefill(&self) -> bool {
        true
    }
}

/// The next-token logits are [`TARGET_LOGITS`] rotated by the input token, so
/// a position's distribution depends on the token actually fed there.
pub(super) struct MarkovModel;

pub(super) fn markov_logits(token: i32) -> [f32; VOCAB] {
    let mut row = TARGET_LOGITS;
    row.rotate_right(token as usize);
    row
}

impl LanguageModel for MarkovModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = advance_caches(input_ids, caches);
        ffi::eval(input_ids);
        let out: Vec<f32> = ffi::array_to_raw_bytes(input_ids)
            .chunks_exact(4)
            .flat_map(|c| markov_logits(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        ffi::from_slice_f32(&out, &[shape[0], shape[1], VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn supports_padded_prefill(&self) -> bool {
        true
    }
}

/// Greedy decoding walks `next(t) = t * multiplier + offset (mod 6)`, so any
/// wrong acceptance decision is visible in the token stream.
#[derive(Clone, Copy)]
pub(super) struct Orbit {
    multiplier: i32,
    offset: i32,
}

impl Orbit {
    pub(super) fn next(self, token: i32) -> i32 {
        (token * self.multiplier + self.offset).rem_euclid(VOCAB as i32)
    }
}

impl LanguageModel for Orbit {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = advance_caches(input_ids, caches);
        ffi::eval(input_ids);
        let tokens: Vec<i32> = ffi::array_to_raw_bytes(input_ids)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = vec![0.0f32; tokens.len() * VOCAB];
        for (i, &tok) in tokens.iter().enumerate() {
            out[i * VOCAB + self.next(tok) as usize] = 8.0;
        }
        ffi::from_slice_f32(&out, &[shape[0], shape[1], VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn supports_padded_prefill(&self) -> bool {
        true
    }
}

/// `next(t) = t + 1`, a full 6-cycle.
pub(super) const GREEDY_TARGET: Orbit = Orbit {
    multiplier: 1,
    offset: 1,
};
/// Agrees with [`GREEDY_TARGET`] only at token 0 (`2t + 1 == t + 1` iff `t == 0`).
pub(super) const MIXED_DRAFT: Orbit = Orbit {
    multiplier: 2,
    offset: 1,
};
/// Never agrees with [`GREEDY_TARGET`].
pub(super) const REJECTING_DRAFT: Orbit = Orbit {
    multiplier: 1,
    offset: 4,
};

/// Where a [`ScriptedDrafter`]'s proposals come from.
pub(super) enum Proposals {
    Orbit(Orbit),
    Sampled(Categorical),
}

pub(super) struct ScriptedDrafter(pub(super) Proposals);

impl ScriptedDrafter {
    pub(super) fn sampled(seed: u64) -> Self {
        Self(Proposals::Sampled(Categorical::new(
            &softmax(&DRAFT_LOGITS, TEMPERATURE),
            seed,
        )))
    }

    pub(super) fn propose(&mut self, current: i32, n: usize) -> Vec<i32> {
        match &mut self.0 {
            Proposals::Sampled(q) => (0..n).map(|_| q.draw()).collect(),
            Proposals::Orbit(orbit) => {
                let mut tok = current;
                (0..n)
                    .map(|_| {
                        tok = orbit.next(tok);
                        tok
                    })
                    .collect()
            }
        }
    }
}

impl Drafter for ScriptedDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        Ok(self.propose(last_bonus, block_size))
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::PromptLookup
    }

    fn drafts_from_tokens_only(&self) -> bool {
        true
    }
}

/// Tokens of one run on the engine's drafter loop.
fn run_drafter<M: LanguageModel>(
    model: M,
    drafter: &mut ScriptedDrafter,
    max_tokens: usize,
    block: usize,
    config: &SamplingConfig,
) -> SpeculativeRun {
    DirectEngine::new(model, 512)
        .generate_with_drafter(
            &DirectRequest::text(&[2], max_tokens, config),
            drafter,
            block,
            |_| true,
        )
        .expect("the stub models run token-only verify rounds")
}

fn collect_counts(budget: usize, block: usize, seed: u64) -> Vec<u32> {
    let config = stochastic_config();
    let mut drafter = ScriptedDrafter::sampled(seed);
    let mut counts = vec![0u32; VOCAB];
    let mut collected = 0usize;
    while collected < budget {
        let per_call = 64.min(budget - collected);
        let run = run_drafter(&ContextFreeModel, &mut drafter, per_call, block, &config);
        assert!(!run.run.tokens.is_empty(), "every call must make progress");
        for &t in &run.run.tokens {
            counts[t as usize] += 1;
        }
        collected += run.run.tokens.len();
    }
    counts
}

#[test]
fn speculative_stream_matches_the_target_distribution() {
    let expected = softmax(&TARGET_LOGITS, TEMPERATURE);
    let counts = collect_counts(sample_budget(), 4, 0x9E37_79B9_7F4A_7C15);
    let chi2 = chi_square(&counts, &expected);
    assert!(
        chi2 < CHI2_DOF5_ALPHA_1E4,
        "stream is distinguishable from target-only sampling: chi-square {chi2:.3} exceeds \
         {CHI2_DOF5_ALPHA_1E4}. counts={counts:?}, expected={expected:?}"
    );
}

/// A one-token block reaches only the first verify position, so a bookkeeping
/// error that depends on block depth cannot cancel out here.
#[test]
fn speculative_stream_matches_the_target_distribution_at_block_size_one() {
    let expected = softmax(&TARGET_LOGITS, TEMPERATURE);
    let counts = collect_counts(sample_budget() / 2, 1, 0xD1B5_4A32_D192_ED03);
    let chi2 = chi_square(&counts, &expected);
    assert!(
        chi2 < CHI2_DOF5_ALPHA_1E4,
        "block=1 stream is distinguishable from the target: chi-square {chi2:.3}, {counts:?}"
    );
}

/// Pearson statistic over the conditional rows `(previous, next)` of the stream:
/// every emitted token must be a draw from the target given the token before it.
/// A round that runs on past a rejection samples from a context holding the
/// rejected proposal, and a round reading the wrong position samples from the
/// wrong token's row; both fail here and neither fails the i.i.d. tests above.
#[test]
fn markov_stream_keeps_the_target_conditional() {
    const PROMPT_TOKEN: i32 = 2;
    let config = stochastic_config();
    let mut drafter = ScriptedDrafter::sampled(0x2545_F491_4F6C_DD1D);
    let mut counts = [[0u32; VOCAB]; VOCAB];
    let mut collected = 0usize;
    let budget = sample_budget();
    while collected < budget {
        let per_call = 64.min(budget - collected);
        let run = run_drafter(MarkovModel, &mut drafter, per_call, 4, &config);
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

/// Replays the wrong rules the statistic must reject, with host draws from the
/// same `p` and `q` and the same sample count: emitting the proposal without
/// testing it, and argmax-against-argmax (the rule of the MTP and DFlash paths
/// at `temperature > 0`). A sampler-match arm pins that the harness itself is
/// not what fails.
#[test]
fn chi_square_rejects_known_wrong_acceptance_rules() {
    let p = softmax(&TARGET_LOGITS, TEMPERATURE);
    let q = softmax(&DRAFT_LOGITS, TEMPERATURE);
    let trials = sample_budget();
    let chi2_of = |emit: &dyn Fn(i32, i32) -> i32| {
        let mut target = Categorical::new(&p, 0x1234_5678_9ABC_DEF1);
        let mut draft = Categorical::new(&q, 0x0FED_CBA9_8765_4321);
        let mut counts = vec![0u32; VOCAB];
        for _ in 0..trials {
            counts[emit(target.draw(), draft.draw()) as usize] += 1;
        }
        chi_square(&counts, &p)
    };

    let sampler_match = chi2_of(&|t, _d| t);
    assert!(
        sampler_match < CHI2_DOF5_ALPHA_1E4,
        "sampler-match arm: {sampler_match:.3}"
    );
    let unconditional = chi2_of(&|_t, d| d);
    assert!(
        unconditional > CHI2_DOF5_ALPHA_1E4,
        "emit-the-draft arm: {unconditional:.3}"
    );
    // The argmax of TARGET_LOGITS is token 0.
    let argmax = chi2_of(&|_t, d| if d == 0 { d } else { 0 });
    assert!(argmax > CHI2_DOF5_ALPHA_1E4, "argmax arm: {argmax:.3}");
}

/// Zero tolerance: one token outside the target's top-2 support fails, however
/// the proposal that was accepted or rejected there was drawn.
#[test]
fn filtered_target_support_is_never_violated() {
    let config = SamplingConfig {
        temperature: TEMPERATURE,
        top_k: 2,
        ..SamplingConfig::default()
    };
    let mut drafter = ScriptedDrafter::sampled(0xA5A5_5A5A_1234_4321);
    let mut emitted = 0usize;
    while emitted < 1500 {
        let run = run_drafter(&ContextFreeModel, &mut drafter, 64, 4, &config);
        for t in &run.run.tokens {
            assert!(
                [0, 1].contains(t),
                "token {t} lies outside the top-2 support"
            );
        }
        emitted += run.run.tokens.len();
    }
}

#[test]
fn the_mixed_draft_agrees_on_exactly_one_token_in_six() {
    let agreements = (0..VOCAB as i32)
        .filter(|&t| GREEDY_TARGET.next(t) == MIXED_DRAFT.next(t))
        .count();
    assert_eq!(agreements, 1);
    assert!((0..VOCAB as i32).all(|t| GREEDY_TARGET.next(t) != REJECTING_DRAFT.next(t)));
}

/// At temperature 0 the stream equals the target's greedy continuation, token
/// for token, through both the accept and the reject-plus-rewind branches.
#[test]
fn temperature_zero_stream_is_byte_identical_to_greedy_target_only() {
    const N: usize = 48;
    let mut tok = 2;
    let expected: Vec<i32> = (0..N)
        .map(|_| {
            tok = GREEDY_TARGET.next(tok);
            tok
        })
        .collect();
    for draft in [MIXED_DRAFT, REJECTING_DRAFT, GREEDY_TARGET] {
        for block in [1usize, 2, 4] {
            let mut drafter = ScriptedDrafter(Proposals::Orbit(draft));
            let run = run_drafter(
                GREEDY_TARGET,
                &mut drafter,
                N,
                block,
                &SamplingConfig::greedy(),
            );
            assert_eq!(
                run.run.tokens, expected,
                "block={block}, offset {}",
                draft.offset
            );
        }
    }
}

/// Acceptance accounting in the two deterministic regimes.
#[test]
fn acceptance_accounting_is_exact_in_both_deterministic_regimes() {
    const N: usize = 24;
    let greedy = SamplingConfig::greedy();
    for block in [1usize, 3] {
        let mut rejecting = ScriptedDrafter(Proposals::Orbit(REJECTING_DRAFT));
        let rounds = run_drafter(GREEDY_TARGET, &mut rejecting, N, block, &greedy).rounds;
        assert_eq!(rounds.accepted_draft_tokens, 0);
        assert!(rounds.proposed_draft_tokens >= rounds.drafted_rounds);

        let mut agreeing = ScriptedDrafter(Proposals::Orbit(GREEDY_TARGET));
        let run = run_drafter(GREEDY_TARGET, &mut agreeing, N, block, &greedy);
        let rounds = run.rounds;
        assert_eq!(run.run.tokens.len(), N);
        assert!(rounds.proposed_draft_tokens > 0);
        assert_eq!(rounds.accepted_draft_tokens, rounds.proposed_draft_tokens);
    }
}
