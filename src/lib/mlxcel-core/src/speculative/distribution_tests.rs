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

//! End-to-end distributional regression guard for speculative decoding
//! (issue #902).
//!
//! # What is being tested
//!
//! **Null hypothesis (H0):** the token stream emitted by
//! [`SpeculativeGenerator::generate`] at `temperature > 0` is distributed
//! identically to sampling the target model alone at the same temperature.
//!
//! The test models make H0 checkable in closed form. The target's logits do
//! not depend on its input, so target-only sampling emits i.i.d. draws from a
//! single known categorical `p = softmax(target_logits / temperature)`. Under
//! H0 every token the speculative generator emits, whatever mixture of
//! accepted drafts, residual resamples and bonus tokens produced it, is also
//! an i.i.d. draw from that same `p`. A Pearson chi-square goodness-of-fit
//! test against `p` therefore tests H0 directly.
//!
//! **Statistic:** `sum_x (O_x - N p_x)^2 / (N p_x)` over the `V = 6` vocabulary
//! entries, `dof = V - 1 = 5`.
//!
//! **Sample count:** `SAMPLE_TOKENS` tokens, which the committed default sets
//! to 12000 (override with `MLXCEL_SPEC_DIST_SAMPLES`). Every emitted token is
//! one sample; a run collects them across as many `generate` calls as needed.
//!
//! **Threshold:** `CHI2_DOF5_ALPHA_1E4 = 25.7448`, the upper-tail chi-square
//! quantile at `alpha = 1e-4` for 5 degrees of freedom. A correct
//! implementation fails once in ten thousand runs.
//!
//! # Why this test can fail
//!
//! A goodness-of-fit test is worthless without demonstrated power, so the
//! power is measured in-tree rather than asserted. `chi_square_rejects_*`
//! replays the identical harness with two deliberately wrong acceptance rules
//! and requires the statistic to exceed the same threshold:
//!
//! * `residual_replaced_by_target_resample` — the single most common way to
//!   get this algorithm wrong: on rejection, draw from `p` instead of from
//!   `normalize(relu(p - q))`. Expected statistic at N = 12000 is roughly 360,
//!   fourteen times the threshold.
//! * `argmax_target` — the rule the MTP and DFlash verify paths still run
//!   (`Gemma4MtpTargetAdapter::verify_forward` calls `argmax_per_position`
//!   regardless of temperature). With a context-independent target it collapses
//!   the stream onto the argmax; the statistic runs into the thousands. This is
//!   the bias the issue describes, recorded here rather than gated.
//!
//! Chi-square is blind to a single wrong token, so the zero-tolerance
//! assertions live beside it: `filtered_target_support_is_never_violated`
//! fails on one emitted token outside the target's top-k support, and the
//! temperature-0 test fails on one differing token.

use super::*;
use crate::dtype;
use crate::sampling::effective_token_distribution;

/// Upper-tail chi-square quantile at alpha = 1e-4, 5 degrees of freedom.
const CHI2_DOF5_ALPHA_1E4: f64 = 25.7448;

/// Tokens collected by the committed default run of the distributional test.
///
/// Sized from the measured power of the two wrong-rule arms: at 12000 the
/// residual bug lands near 360 against a threshold of 25.74. Raise it with
/// `MLXCEL_SPEC_DIST_SAMPLES` when investigating.
const SAMPLE_TOKENS: usize = 12_000;

const VOCAB: usize = 6;
const TEMPERATURE: f32 = 0.7;

/// Target logits. Deliberately not flat, so the chi-square has cells with very
/// different expected counts and a bias in either direction shows up.
const TARGET_LOGITS: [f32; VOCAB] = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0];
/// Draft logits: a different model that agrees with the target often enough to
/// be a realistic drafter (`sum min(p, q)` is about 0.71) but disagrees on the
/// ordering of the top two tokens.
const DRAFT_LOGITS: [f32; VOCAB] = [1.4, 1.3, 0.2, 0.6, -0.3, -0.8];

fn sample_budget() -> usize {
    std::env::var("MLXCEL_SPEC_DIST_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SAMPLE_TOKENS)
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f64> {
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&x| f64::from(x) / f64::from(temperature))
        .collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

fn chi_square(observed: &[u32], expected: &[f64]) -> f64 {
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

fn stochastic_config() -> SamplingConfig {
    SamplingConfig {
        temperature: TEMPERATURE,
        ..SamplingConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Test models
// ---------------------------------------------------------------------------

/// A model whose logits are the same at every position and independent of its
/// input, so its next-token distribution is one fixed categorical.
///
/// Keeping the target context-independent is what makes H0 a plain i.i.d.
/// goodness-of-fit statement instead of a statement about a sequence model.
struct ContextFreeModel {
    logits: Vec<f32>,
    eos_tokens: Vec<i32>,
}

impl ContextFreeModel {
    fn new(logits: &[f32]) -> Self {
        Self {
            logits: logits.to_vec(),
            eos_tokens: Vec::new(),
        }
    }
}

impl LanguageModel for ContextFreeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(input_ids);
        let (batch, seq_len) = (shape[0] as usize, shape[1] as usize);
        // Advance the KV caches so cache-offset bookkeeping and the rewind on
        // rejection are exercised exactly as they are with a real model.
        for cache in caches.iter_mut() {
            let k = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
            let v = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
            cache.update(k, v);
        }
        let mut out = Vec::with_capacity(batch * seq_len * self.logits.len());
        for _ in 0..(batch * seq_len) {
            out.extend_from_slice(&self.logits);
        }
        ffi::from_slice_f32(&out, &[shape[0], shape[1], self.logits.len() as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_tokens.clone()
    }
}

/// A deterministic sequence model: the logit peak at each position is a
/// function of the token *at* that position, so greedy decoding walks a fixed
/// orbit. Used for the temperature-0 byte-identity test, where a
/// context-independent model would make any acceptance rule trivially right.
struct OrbitModel {
    multiplier: i32,
    offset: i32,
    peak: f32,
}

impl OrbitModel {
    fn next(&self, token: i32) -> i32 {
        (token * self.multiplier + self.offset).rem_euclid(VOCAB as i32)
    }
}

impl LanguageModel for OrbitModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(input_ids);
        ffi::eval(input_ids);
        let bytes = ffi::array_to_raw_bytes(input_ids);
        let tokens: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for cache in caches.iter_mut() {
            let k = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
            let v = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
            cache.update(k, v);
        }
        let mut out = vec![0.0f32; tokens.len() * VOCAB];
        for (i, &tok) in tokens.iter().enumerate() {
            out[i * VOCAB + self.next(tok) as usize] = self.peak;
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
}

// ---------------------------------------------------------------------------
// The main distributional test
// ---------------------------------------------------------------------------

/// Collect emitted tokens from repeated speculative generations until the
/// sample budget is met. The first token of each call comes from the prefill
/// sample, which is a plain draw from `p` and therefore also a valid sample.
fn collect_speculative_tokens(budget: usize, num_draft: usize) -> Vec<u32> {
    let target = ContextFreeModel::new(&TARGET_LOGITS);
    let draft = ContextFreeModel::new(&DRAFT_LOGITS);
    let config = stochastic_config();
    let mut counts = vec![0u32; VOCAB];
    let mut collected = 0usize;

    while collected < budget {
        let mut generator = SpeculativeGenerator::new(target.num_layers(), draft.num_layers());
        let per_call = 64.min(budget - collected);
        let (tokens, _stats) =
            generator.generate(&target, &draft, &[0], per_call, num_draft, &config);
        assert!(
            !tokens.is_empty(),
            "the generator must make progress on every call"
        );
        for t in &tokens {
            assert!(
                (0..VOCAB as i32).contains(t),
                "emitted token {t} is outside the vocabulary"
            );
            counts[*t as usize] += 1;
        }
        collected += tokens.len();
    }
    counts
}

/// H0: the speculative stream at `temperature > 0` is distributed as
/// target-only sampling. Rejected above the alpha = 1e-4 chi-square threshold.
///
/// See the module docs for the full statement, the sample count and the power
/// calibration that makes this assertion meaningful.
#[test]
fn speculative_stream_matches_the_target_distribution() {
    let expected = softmax(&TARGET_LOGITS, TEMPERATURE);
    let budget = sample_budget();
    let counts = collect_speculative_tokens(budget, 4);
    let n: u32 = counts.iter().sum();
    let chi2 = chi_square(&counts, &expected);

    let empirical: Vec<f64> = counts.iter().map(|&c| f64::from(c) / f64::from(n)).collect();
    assert!(
        chi2 < CHI2_DOF5_ALPHA_1E4,
        "speculative stream is distinguishable from target-only sampling: \
         chi-square {chi2:.3} exceeds the alpha=1e-4 dof=5 threshold {CHI2_DOF5_ALPHA_1E4}. \
         n={n}, empirical={empirical:?}, expected={expected:?}"
    );
}

/// The guarantee must not depend on the draft block length: a longer chain
/// runs the accept test more times per verify forward and reaches deeper
/// positions, where a context-bookkeeping error would surface.
#[test]
fn speculative_stream_matches_the_target_distribution_at_block_size_one() {
    let expected = softmax(&TARGET_LOGITS, TEMPERATURE);
    let counts = collect_speculative_tokens(sample_budget() / 2, 1);
    let chi2 = chi_square(&counts, &expected);
    assert!(
        chi2 < CHI2_DOF5_ALPHA_1E4,
        "num_draft=1 stream is distinguishable from the target: chi-square {chi2:.3}, \
         counts {counts:?}"
    );
}

// ---------------------------------------------------------------------------
// Power calibration: the same statistic must reject known-wrong rules
// ---------------------------------------------------------------------------

/// Replay one verify position under a chosen acceptance rule, returning the
/// emitted token. `p` and `q` are the fixed context-free distributions, so a
/// full chain is not needed to reproduce the emitted-token distribution: every
/// position of every chain sees the same pair.
fn emit_one_token(
    rule: &str,
    target_probs: &MlxArray,
    draft_probs: &MlxArray,
    draft_token: i32,
    argmax_token: i32,
) -> i32 {
    match rule {
        "stochastic" => match stochastic_accept::verify_draft_token(
            target_probs,
            draft_probs,
            draft_token,
        ) {
            DraftVerdict::Accept => draft_token,
            DraftVerdict::Reject { replacement } => replacement,
        },
        // The common implementation mistake: reject correctly, then draw the
        // replacement from `p` instead of from the residual.
        "residual_replaced_by_target_resample" => {
            if stochastic_accept::accept_draft_token(target_probs, draft_probs, draft_token) {
                draft_token
            } else {
                let tok = ffi::fused_sample(&ffi::log(target_probs), 1.0, 0, 1.0, 0.0);
                ffi::eval(&tok);
                ffi::item_i32(&tok)
            }
        }
        // The rule the MTP and DFlash verify paths still run: compare against
        // the target's argmax and emit the argmax on mismatch.
        "argmax_target" => argmax_token,
        other => panic!("unknown rule {other}"),
    }
}

/// Draw one token from a `[1, vocab]` probability row.
fn draw_from(probs: &MlxArray) -> i32 {
    let tok = ffi::fused_sample(&ffi::log(probs), 1.0, 0, 1.0, 0.0);
    ffi::eval(&tok);
    ffi::item_i32(&tok)
}

fn power_arm_chi_square(rule: &str, trials: usize) -> f64 {
    let target_logits = ffi::from_slice_f32(&TARGET_LOGITS, &[1, 1, VOCAB as i32]);
    let draft_logits = ffi::from_slice_f32(&DRAFT_LOGITS, &[1, 1, VOCAB as i32]);
    let config = stochastic_config();
    let target_probs = effective_token_distribution(&target_logits, &config, &[]);
    let draft_probs = effective_token_distribution(&draft_logits, &config, &[]);
    ffi::eval(&target_probs);
    ffi::eval(&draft_probs);

    let argmax_token = TARGET_LOGITS
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap();

    let mut counts = vec![0u32; VOCAB];
    for _ in 0..trials {
        let draft_token = draw_from(&draft_probs);
        let emitted = emit_one_token(
            rule,
            &target_probs,
            &draft_probs,
            draft_token,
            argmax_token,
        );
        counts[emitted as usize] += 1;
    }
    chi_square(&counts, &softmax(&TARGET_LOGITS, TEMPERATURE))
}

/// Power calibration. The committed acceptance rule must pass the same
/// statistic that rejects both wrong rules; otherwise the goodness-of-fit test
/// above proves nothing.
///
/// Uses a smaller trial count than the end-to-end test because each trial here
/// is one verify position rather than a whole `generate` call. The margins are
/// large enough that the conclusion is not sensitive to the exact count.
#[test]
fn chi_square_rejects_known_wrong_acceptance_rules() {
    const TRIALS: usize = 4000;

    let correct = power_arm_chi_square("stochastic", TRIALS);
    assert!(
        correct < CHI2_DOF5_ALPHA_1E4,
        "the committed rule must pass its own statistic: chi-square {correct:.3}"
    );

    let residual_bug = power_arm_chi_square("residual_replaced_by_target_resample", TRIALS);
    assert!(
        residual_bug > CHI2_DOF5_ALPHA_1E4,
        "resampling from p instead of the residual must be detected, but the \
         chi-square was only {residual_bug:.3} against a {CHI2_DOF5_ALPHA_1E4} threshold: \
         the goodness-of-fit test above has no power and cannot be trusted"
    );

    let argmax_bug = power_arm_chi_square("argmax_target", TRIALS);
    assert!(
        argmax_bug > CHI2_DOF5_ALPHA_1E4,
        "argmax-target acceptance must be detected, but the chi-square was only \
         {argmax_bug:.3}"
    );

    // Recorded, not gated: the margin by which each wrong rule is rejected.
    println!(
        "power calibration at n={TRIALS}: correct={correct:.2}, \
         residual-bug={residual_bug:.2}, argmax-target={argmax_bug:.2}, \
         threshold={CHI2_DOF5_ALPHA_1E4}"
    );
}

// ---------------------------------------------------------------------------
// Zero-tolerance guards (chi-square cannot see a single wrong token)
// ---------------------------------------------------------------------------

/// One emitted token outside the target's filtered support fails this test.
///
/// With `top_k = 2` the target assigns exactly zero mass to four of the six
/// tokens, while the unfiltered drafter proposes them roughly half the time.
/// A missing `p(t) > 0` conjunct in the accept test, or a residual that leaked
/// mass outside `p`'s support, shows up here on the first offending token
/// rather than as a shifted histogram.
#[test]
fn filtered_target_support_is_never_violated() {
    let target = ContextFreeModel::new(&TARGET_LOGITS);
    let draft = ContextFreeModel::new(&DRAFT_LOGITS);
    let config = SamplingConfig {
        temperature: TEMPERATURE,
        top_k: 2,
        ..SamplingConfig::default()
    };
    // Top-2 of TARGET_LOGITS.
    let allowed = [0i32, 1];

    let mut emitted = 0usize;
    while emitted < 1500 {
        let mut generator = SpeculativeGenerator::new(target.num_layers(), draft.num_layers());
        let (tokens, _) = generator.generate(&target, &draft, &[0], 64, 4, &config);
        for t in &tokens {
            assert!(
                allowed.contains(t),
                "emitted token {t} lies outside the target's top-2 support {allowed:?}; \
                 the target sampler could never have produced it"
            );
        }
        emitted += tokens.len();
    }
}

// ---------------------------------------------------------------------------
// Temperature 0 and cache rewind
// ---------------------------------------------------------------------------

/// The target model used by the temperature-0 and cache-rewind tests:
/// `next(t) = t + 1 (mod 6)`, a full 6-cycle rather than a short orbit.
fn greedy_target() -> OrbitModel {
    OrbitModel {
        multiplier: 1,
        offset: 1,
        peak: 8.0,
    }
}

/// A drafter that agrees with [`greedy_target`] exactly when the current token
/// is 0 (`2t + 1 == t + 1 (mod 6)` iff `t == 0`), so a run over the target's
/// 6-cycle produces a fixed 1-in-6 mix of accepted and rejected positions.
fn mixed_draft() -> OrbitModel {
    OrbitModel {
        multiplier: 2,
        offset: 1,
        peak: 8.0,
    }
}

/// A drafter that never agrees with [`greedy_target`]: `t + 4` and `t + 1`
/// differ for every `t`.
fn always_rejecting_draft() -> OrbitModel {
    OrbitModel {
        multiplier: 1,
        offset: 4,
        peak: 8.0,
    }
}

/// At temperature 0 the speculative stream must equal the target model's own
/// greedy continuation, token for token. One differing token fails.
///
/// The target is a deterministic 6-cycle, so the reference stream is computable
/// without a second generator. The drafter agrees on exactly one token in six,
/// which forces both the accept and the reject-plus-rewind branches to run
/// many times inside a single generation.
#[test]
fn temperature_zero_stream_is_byte_identical_to_greedy_target_only() {
    let target = greedy_target();
    const PROMPT: i32 = 2;
    const N: usize = 48;

    let mut expected = Vec::with_capacity(N);
    let mut tok = PROMPT;
    for _ in 0..N {
        tok = target.next(tok);
        expected.push(tok);
    }

    for draft in [mixed_draft(), always_rejecting_draft()] {
        for num_draft in [1usize, 2, 4] {
            let mut generator = SpeculativeGenerator::new(target.num_layers(), draft.num_layers());
            let (tokens, _) = generator.generate(
                &target,
                &draft,
                &[PROMPT],
                N,
                num_draft,
                &SamplingConfig::greedy(),
            );
            assert_eq!(
                tokens, expected,
                "num_draft={num_draft}, draft offset {}: temperature-0 speculative output \
                 diverged from the target's greedy continuation",
                draft.offset
            );
        }
    }

    assert_eq!(
        stochastic_accept::acceptance_rule(&SamplingConfig::greedy(), true),
        AcceptanceRule::Argmax,
        "temperature 0 must stay on the argmax rule"
    );
}

/// The mixed drafter really does produce both outcomes, so the byte-identity
/// test above is not silently exercising a single branch. Pins the arithmetic
/// of the agreement condition rather than trusting the comment.
#[test]
fn the_mixed_draft_model_agrees_on_exactly_one_token_in_six() {
    let target = greedy_target();
    let mixed = mixed_draft();
    let rejecting = always_rejecting_draft();
    let agreements = (0..VOCAB as i32)
        .filter(|&t| target.next(t) == mixed.next(t))
        .count();
    assert_eq!(agreements, 1, "mixed_draft must agree on exactly one token");
    for t in 0..VOCAB as i32 {
        assert_ne!(
            target.next(t),
            rejecting.next(t),
            "always_rejecting_draft must never agree, but it does at token {t}"
        );
    }
}

/// The main KV cache length after a run, in the two deterministic regimes.
///
/// The invariant is that the main cache holds the prompt plus every emitted
/// token *except* the pending `current_token`, which the next round will
/// forward. The one exception is a run that stops inside the accept branch on
/// `max_tokens`: there the final emitted token was a draft token that had
/// already been forwarded as part of the verify input, so it is legitimately
/// still in the cache. Both regimes are pinned exactly here by choosing model
/// pairs whose acceptance is deterministic at temperature 0.
#[test]
fn kv_cache_length_is_exact_in_both_termination_regimes() {
    const PROMPT: i32 = 2;
    const N: usize = 24;

    // Regime 1: total disagreement. The draft walks a different orbit, so
    // every round rejects at position 0, emits one replacement token and ends
    // the round. The loop always exits through the `while` condition, so the
    // last emitted token is always pending.
    let target = greedy_target();
    let disagreeing_draft = always_rejecting_draft();
    for num_draft in [1usize, 3, 5] {
        let mut generator = SpeculativeGenerator::new(1, 1);
        let (tokens, _) = generator.generate(
            &target,
            &disagreeing_draft,
            &[PROMPT],
            N,
            num_draft,
            &SamplingConfig::greedy(),
        );
        assert_eq!(tokens.len(), N);
        let expected = 1 + tokens.len() - 1;
        for (layer, cache) in generator.main_caches.iter().enumerate() {
            assert_eq!(
                cache.offset as usize, expected,
                "all-reject num_draft={num_draft} layer {layer}: main cache holds {} \
                 entries, expected {expected} (prompt 1 + {} emitted - 1 pending). \
                 A wrong trim on rejection lands here.",
                cache.offset,
                tokens.len()
            );
        }
    }

    // Regime 2: total agreement. The draft walks the target's own orbit, so
    // every draft position is accepted and the run stops inside the accept
    // branch on `max_tokens`, leaving the final token already forwarded.
    let agreeing_draft = greedy_target();
    for num_draft in [2usize, 4] {
        let mut generator = SpeculativeGenerator::new(1, 1);
        let (tokens, _) = generator.generate(
            &target,
            &agreeing_draft,
            &[PROMPT],
            N,
            num_draft,
            &SamplingConfig::greedy(),
        );
        assert_eq!(tokens.len(), N);
        let expected = 1 + tokens.len();
        for (layer, cache) in generator.main_caches.iter().enumerate() {
            assert_eq!(
                cache.offset as usize, expected,
                "all-accept num_draft={num_draft} layer {layer}: main cache holds {} \
                 entries, expected {expected} (prompt 1 + {} emitted, none pending). \
                 A spurious trim on full acceptance lands here.",
                cache.offset,
                tokens.len()
            );
        }
    }
}

/// The stochastic acceptance path must leave the caches in one of the same two
/// regimes: mixed accept/reject rounds change *which* regime the run ends in,
/// never the arithmetic. Every layer must also agree with every other layer; a
/// per-layer trim mismatch corrupts attention silently and produces no other
/// visible symptom.
#[test]
fn kv_cache_stays_consistent_across_mixed_accept_reject_rounds() {
    let target = ContextFreeModel::new(&TARGET_LOGITS);
    let draft = ContextFreeModel::new(&DRAFT_LOGITS);
    let config = stochastic_config();

    for num_draft in [1usize, 3, 5] {
        for trial in 0..8 {
            let mut generator = SpeculativeGenerator::new(target.num_layers(), draft.num_layers());
            let (tokens, _) = generator.generate(&target, &draft, &[0], 40, num_draft, &config);

            let pending = 1 + tokens.len() - 1;
            let forwarded = 1 + tokens.len();
            let offset = generator.main_caches[0].offset as usize;
            assert!(
                offset == pending || offset == forwarded,
                "num_draft={num_draft} trial={trial}: main cache offset {offset} is \
                 neither {pending} (last token pending) nor {forwarded} (run stopped \
                 inside the accept branch); the rewind on rejection is wrong"
            );
            for (layer, cache) in generator.main_caches.iter().enumerate() {
                assert_eq!(
                    cache.offset as usize, offset,
                    "num_draft={num_draft} trial={trial}: main cache layer {layer} \
                     disagrees with layer 0"
                );
            }
            let draft_offset = generator.draft_caches[0].offset;
            for (layer, cache) in generator.draft_caches.iter().enumerate() {
                assert_eq!(
                    cache.offset, draft_offset,
                    "num_draft={num_draft} trial={trial}: draft cache layer {layer} \
                     disagrees with layer 0"
                );
            }
        }
    }
}
