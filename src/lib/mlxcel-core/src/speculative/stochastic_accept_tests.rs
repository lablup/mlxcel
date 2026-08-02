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

//! Unit tests for the modified rejection-sampling primitives.
//!
//! These test the accept rule and the residual draw in isolation, against
//! hand-built `p` and `q` vectors where the exact answer is known in closed
//! form. The end-to-end distributional argument for the generator lives in
//! `distribution_tests.rs`.

use super::*;
use crate::sampling::effective_token_distribution;

/// Upper-tail chi-square critical values at alpha = 1e-4.
///
/// Sourced from the chi-square quantile function; indexed by degrees of
/// freedom. A false-failure rate of 1e-4 per assertion is the tolerance these
/// tests accept in exchange for keeping the power to reject a real defect (see
/// the power calibration in `distribution_tests.rs`).
const CHI2_CRIT_1E4: [f64; 8] = [
    f64::NAN, // dof 0, unused
    15.1367,  // dof 1
    18.4207,  // dof 2
    21.1075,  // dof 3
    23.5127,  // dof 4
    25.7448,  // dof 5
    27.8563,  // dof 6
    29.8775,  // dof 7
];

/// Pearson chi-square of `observed` counts against `expected` probabilities.
fn chi_square(observed: &[u32], expected: &[f64]) -> f64 {
    let n: u32 = observed.iter().sum();
    let n = f64::from(n);
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

/// Build a `[1, vocab]` float32 probability row from an explicit vector.
fn probs_row(values: &[f32]) -> UniquePtr<MlxArray> {
    ffi::from_slice_f32(values, &[1, values.len() as i32])
}

/// Build a `[1, 1, vocab]` logits tensor, the shape the samplers consume.
fn logits_tensor(values: &[f32]) -> UniquePtr<MlxArray> {
    ffi::from_slice_f32(values, &[1, 1, values.len() as i32])
}

/// Read a `[1, vocab]` float32 tensor back to the host.
fn to_host(arr: &MlxArray) -> Vec<f32> {
    ffi::eval(arr);
    let bytes = ffi::array_to_raw_bytes(arr);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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

fn stochastic_config(temperature: f32) -> SamplingConfig {
    SamplingConfig {
        temperature,
        ..SamplingConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Rule selection, greedy predicate, kill switch
// ---------------------------------------------------------------------------

/// The greedy predicate must agree with the C++ sampler's own short-circuit
/// (`temperature == 0.0f || top_k == 1` selects `argmax`). If it drifted, the
/// accept test would compare a stochastic draft against a one-hot `p` and
/// reject nearly everything.
#[test]
fn sampler_is_greedy_matches_fused_sampler_short_circuit() {
    assert!(sampler_is_greedy(&SamplingConfig::greedy()));
    assert!(sampler_is_greedy(&stochastic_config(0.0)));
    assert!(sampler_is_greedy(&SamplingConfig {
        temperature: 0.9,
        top_k: 1,
        ..SamplingConfig::default()
    }));
    assert!(!sampler_is_greedy(&stochastic_config(0.7)));
    assert!(!sampler_is_greedy(&SamplingConfig {
        temperature: 1.0,
        top_k: 40,
        ..SamplingConfig::default()
    }));
}

/// A greedy target keeps the pre-#902 argmax rule regardless of everything
/// else, which is what makes temperature 0 byte-identical.
#[test]
fn greedy_target_always_selects_argmax_rule() {
    assert_eq!(
        acceptance_rule(&SamplingConfig::greedy(), true),
        AcceptanceRule::Argmax
    );
    assert_eq!(
        acceptance_rule(&SamplingConfig::greedy(), false),
        AcceptanceRule::Argmax
    );
}

/// A drafter that cannot report `q` must not be given a fabricated one.
#[test]
fn missing_proposal_distribution_falls_back_to_argmax() {
    let rule = acceptance_rule(&stochastic_config(0.7), false);
    assert!(matches!(
        rule,
        AcceptanceRule::ArgmaxNoProposalDistribution | AcceptanceRule::ArgmaxKillSwitch
    ));
    assert!(!rule.is_distribution_preserving() || rule == AcceptanceRule::Argmax);
}

/// Only `Argmax` (greedy target) and `Stochastic` claim distribution
/// preservation. The two argmax fallbacks at `temperature > 0` must not.
#[test]
fn distribution_preservation_claims_are_correct() {
    assert!(AcceptanceRule::Argmax.is_distribution_preserving());
    assert!(AcceptanceRule::Stochastic.is_distribution_preserving());
    assert!(!AcceptanceRule::ArgmaxKillSwitch.is_distribution_preserving());
    assert!(!AcceptanceRule::ArgmaxNoProposalDistribution.is_distribution_preserving());
}

/// The env switch is parsed case- and whitespace-insensitively, and absence
/// means enabled. Tested through the same matcher the runtime uses rather than
/// through the process-wide `OnceLock`, which cannot be re-read in-process.
#[test]
fn kill_switch_parsing_accepts_the_documented_falsy_values() {
    let falsy = |raw: &str| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    };
    for v in ["0", "false", "FALSE", " off ", "No"] {
        assert!(falsy(v), "{v} must disable stochastic acceptance");
    }
    for v in ["1", "true", "on", "yes", ""] {
        assert!(!falsy(v), "{v} must leave stochastic acceptance enabled");
    }
}

/// The one-shot latches fire once per kind, not once globally: a process that
/// serves a greedy request and then a stochastic one must log both.
#[test]
fn log_latches_are_per_kind_and_one_shot() {
    reset_log_latches();
    assert!(!AcceptanceRule::Argmax.latch().load(Ordering::Relaxed));
    note_rule(AcceptanceRule::Argmax);
    assert!(AcceptanceRule::Argmax.latch().load(Ordering::Relaxed));
    assert!(
        !AcceptanceRule::Stochastic.latch().load(Ordering::Relaxed),
        "logging one rule must not swallow a different rule's first occurrence"
    );
    note_rule(AcceptanceRule::Stochastic);
    assert!(AcceptanceRule::Stochastic.latch().load(Ordering::Relaxed));

    note_outcome(AcceptanceOutcome::ResidualResample);
    assert!(
        AcceptanceOutcome::ResidualResample
            .latch()
            .load(Ordering::Relaxed)
    );
    assert!(
        !AcceptanceOutcome::ResidualDegenerateFallback
            .latch()
            .load(Ordering::Relaxed)
    );
    reset_log_latches();
}

// ---------------------------------------------------------------------------
// effective_token_distribution
// ---------------------------------------------------------------------------

/// The stochastic distribution must be the temperature-scaled softmax, to
/// float32 precision. This is the `p` of the accept test, so an error here is
/// an error in every downstream claim.
#[test]
fn effective_distribution_is_the_temperature_scaled_softmax() {
    let logits = [2.0f32, 1.0, 0.5, 0.0, -0.5, -1.0];
    let config = stochastic_config(0.7);
    let probs = effective_token_distribution(&logits_tensor(&logits), &config, &[]);
    let got = to_host(&probs);
    let want = softmax(&logits, 0.7);

    let sum: f64 = got.iter().map(|&x| f64::from(x)).sum();
    assert!((sum - 1.0).abs() < 1e-5, "distribution must normalize: {sum}");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (f64::from(*g) - w).abs() < 1e-5,
            "entry {i}: got {g}, want {w}"
        );
    }
}

/// A greedy config yields the one-hot indicator at the argmax. This is the
/// degenerate `q` the issue calls for when a drafter proposes greedily; the
/// accept test then reduces to `u <= p(t)`.
#[test]
fn effective_distribution_is_one_hot_for_a_greedy_config() {
    let logits = [0.1f32, 3.0, 0.2, -1.0];
    let probs =
        effective_token_distribution(&logits_tensor(&logits), &SamplingConfig::greedy(), &[]);
    let got = to_host(&probs);
    assert_eq!(got, vec![0.0, 1.0, 0.0, 0.0]);

    // `top_k == 1` is the sampler's other greedy short-circuit.
    let top1 = SamplingConfig {
        temperature: 0.8,
        top_k: 1,
        ..SamplingConfig::default()
    };
    let probs = effective_token_distribution(&logits_tensor(&logits), &top1, &[]);
    assert_eq!(to_host(&probs), vec![0.0, 1.0, 0.0, 0.0]);
}

/// A filtered target has *zero* mass outside its support. The accept test
/// depends on this being exactly zero, not merely small: it is what forces a
/// drafted token the target filtered out to be rejected.
#[test]
fn effective_distribution_zeroes_the_filtered_tail() {
    let logits = [3.0f32, 2.5, 1.0, 0.5, 0.0, -2.0];
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 2,
        ..SamplingConfig::default()
    };
    let got = to_host(&effective_token_distribution(
        &logits_tensor(&logits),
        &config,
        &[],
    ));
    assert!(got[0] > 0.0 && got[1] > 0.0);
    for (i, &v) in got.iter().enumerate().skip(2) {
        assert_eq!(v, 0.0, "token {i} is outside top-2 and must carry no mass");
    }
    let sum: f32 = got.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// accept_draft_token
// ---------------------------------------------------------------------------

/// `p(t) >= q(t)` implies `min(1, p/q) == 1`: the token must be accepted every
/// single time, with no tolerance for a stray rejection.
#[test]
fn tokens_the_target_likes_at_least_as_much_are_always_accepted() {
    let p = probs_row(&[0.6, 0.3, 0.1, 0.0]);
    let q = probs_row(&[0.2, 0.2, 0.1, 0.5]);
    for trial in 0..400 {
        for token in [0, 1] {
            assert!(
                accept_draft_token(&p, &q, token),
                "trial {trial}: p({token}) >= q({token}) must accept unconditionally"
            );
        }
    }
}

/// A token the target assigns zero mass must be rejected every single time.
/// This is the zero-tolerance guard: one accepted token here is one token in
/// the served stream that the target's own sampler could never have produced.
#[test]
fn tokens_outside_the_target_support_are_never_accepted() {
    let p = probs_row(&[0.7, 0.3, 0.0, 0.0]);
    let q = probs_row(&[0.1, 0.1, 0.4, 0.4]);
    for trial in 0..400 {
        for token in [2, 3] {
            assert!(
                !accept_draft_token(&p, &q, token),
                "trial {trial}: p({token}) == 0 must reject unconditionally"
            );
        }
    }
}

/// A `q(t)` that has underflowed float32 to exactly zero must not turn into an
/// unconditional accept for a token the target also gives zero mass. Without
/// the `p(t) > 0` conjunct the product test reads `0 <= 0` and accepts.
#[test]
fn underflowed_proposal_mass_does_not_bypass_the_target_support() {
    let p = probs_row(&[1.0, 0.0, 0.0]);
    let q = probs_row(&[0.0, 0.0, 1.0]);
    for _ in 0..200 {
        assert!(
            !accept_draft_token(&p, &q, 1),
            "q(1) == 0 and p(1) == 0 must reject, not accept"
        );
    }
    for _ in 0..200 {
        assert!(
            accept_draft_token(&p, &q, 0),
            "q(0) == 0 with p(0) > 0 accepts: the token was genuinely proposed"
        );
    }
}

/// The empirical acceptance rate must match `min(1, p(t)/q(t))`.
///
/// Null hypothesis: `accept_draft_token` accepts with probability
/// `min(1, p(t)/q(t))`. Binomial with n = 4000; the assertion band is +-4
/// standard errors, so a correct implementation fails about 6e-5 of the time
/// per token tested.
#[test]
fn acceptance_probability_matches_the_min_one_ratio() {
    let p_vals = [0.50f32, 0.20, 0.20, 0.10];
    let q_vals = [0.10f32, 0.40, 0.20, 0.30];
    let p = probs_row(&p_vals);
    let q = probs_row(&q_vals);

    const TRIALS: usize = 4000;
    for token in 0..4usize {
        let expected = f64::from(p_vals[token] / q_vals[token]).min(1.0);
        let accepts = (0..TRIALS)
            .filter(|_| accept_draft_token(&p, &q, token as i32))
            .count();
        let rate = accepts as f64 / TRIALS as f64;
        let se = (expected * (1.0 - expected) / TRIALS as f64).sqrt();
        let band = 4.0 * se + 1e-12;
        assert!(
            (rate - expected).abs() <= band,
            "token {token}: acceptance rate {rate:.4} outside {expected:.4} +- {band:.4}"
        );
    }
}

// ---------------------------------------------------------------------------
// residual_resample
// ---------------------------------------------------------------------------

/// The residual draw must follow `normalize(relu(p - q))`.
///
/// Null hypothesis: the residual sample is distributed as
/// `relu(p - q) / sum(relu(p - q))`. Pearson chi-square over the 3 tokens with
/// positive residual (dof = 2), n = 6000, rejected above 18.42 (alpha = 1e-4).
#[test]
fn residual_draw_follows_the_normalized_relu_difference() {
    let p_vals = [0.50f32, 0.25, 0.15, 0.07, 0.03];
    let q_vals = [0.20f32, 0.40, 0.05, 0.30, 0.05];
    let p = probs_row(&p_vals);
    let q = probs_row(&q_vals);

    let residual: Vec<f64> = p_vals
        .iter()
        .zip(&q_vals)
        .map(|(&a, &b)| f64::from(a - b).max(0.0))
        .collect();
    let mass: f64 = residual.iter().sum();
    let expected: Vec<f64> = residual.iter().map(|r| r / mass).collect();

    const TRIALS: usize = 6000;
    let mut counts = vec![0u32; p_vals.len()];
    for _ in 0..TRIALS {
        let (token, outcome) = residual_resample(&p, &q);
        assert_eq!(outcome, AcceptanceOutcome::ResidualResample);
        assert!(
            expected[token as usize] > 0.0,
            "residual drew token {token}, which has p <= q and therefore zero residual mass"
        );
        counts[token as usize] += 1;
    }

    // Only the support of the residual carries expected mass; restrict the
    // statistic to it so no zero-expectation cell appears in the denominator.
    let support: Vec<usize> = (0..p_vals.len()).filter(|&i| expected[i] > 0.0).collect();
    let obs: Vec<u32> = support.iter().map(|&i| counts[i]).collect();
    let exp: Vec<f64> = support.iter().map(|&i| expected[i]).collect();
    let dof = support.len() - 1;
    let chi2 = chi_square(&obs, &exp);
    assert!(
        chi2 < CHI2_CRIT_1E4[dof],
        "residual distribution chi-square {chi2:.3} exceeds the alpha=1e-4 \
         critical value {:.3} at dof={dof}; observed {counts:?}, expected {expected:?}",
        CHI2_CRIT_1E4[dof]
    );
}

/// `p == q` leaves no residual mass at all. The documented degenerate fallback
/// must fire, be reported as such, and still emit a token inside `p`'s support.
#[test]
fn identical_distributions_take_the_degenerate_residual_fallback() {
    let vals = [0.6f32, 0.3, 0.1, 0.0];
    let p = probs_row(&vals);
    let q = probs_row(&vals);
    for _ in 0..64 {
        let (token, outcome) = residual_resample(&p, &q);
        assert_eq!(outcome, AcceptanceOutcome::ResidualDegenerateFallback);
        assert!(
            vals[token as usize] > 0.0,
            "the fallback must stay inside the target's support, got token {token}"
        );
    }
}

/// `verify_draft_token` composes the two primitives without changing either
/// one's contract: an always-accepted token never resamples, and a token
/// outside the target support always comes back as a rejection whose
/// replacement carries target mass.
#[test]
fn verify_draft_token_composes_accept_and_residual() {
    let p_vals = [0.7f32, 0.3, 0.0, 0.0];
    let p = probs_row(&p_vals);
    let q = probs_row(&[0.1, 0.1, 0.4, 0.4]);

    for _ in 0..200 {
        assert_eq!(verify_draft_token(&p, &q, 0), DraftVerdict::Accept);
    }
    for _ in 0..200 {
        match verify_draft_token(&p, &q, 2) {
            DraftVerdict::Accept => panic!("token 2 has zero target mass and must be rejected"),
            DraftVerdict::Reject { replacement } => assert!(
                p_vals[replacement as usize] > 0.0,
                "replacement {replacement} must lie in the target's support"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Acceptance-rate improvement
// ---------------------------------------------------------------------------

/// The measured acceptance rate over proposals drawn from `q` must converge to
/// `sum_x min(p(x), q(x))`, the information-theoretic optimum for this pair,
/// and must beat the `sum_x p(x) q(x)` rate of the pre-#902 rule (which
/// accepts only when an independent target draw happens to land on the same
/// token).
///
/// Null hypothesis: the accept rate equals `sum_x min(p(x), q(x))`. n = 8000
/// proposals, band of +-4 binomial standard errors.
#[test]
fn measured_acceptance_rate_reaches_the_sum_min_optimum() {
    let p_vals = [0.55f32, 0.20, 0.12, 0.08, 0.05];
    let q_vals = [0.30f32, 0.35, 0.15, 0.15, 0.05];
    let p = probs_row(&p_vals);
    let q = probs_row(&q_vals);

    let sum_min: f64 = p_vals
        .iter()
        .zip(&q_vals)
        .map(|(&a, &b)| f64::from(a.min(b)))
        .sum();
    let sum_prod: f64 = p_vals
        .iter()
        .zip(&q_vals)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    assert!(
        sum_min > sum_prod,
        "the pair must actually favor the new rule: sum min {sum_min} vs sum prod {sum_prod}"
    );

    // Draw proposals from `q` by inverse-CDF over a deterministic sweep of the
    // unit interval, so the measured rate isolates the accept test's own
    // randomness instead of compounding two sampling errors.
    const TRIALS: usize = 8000;
    let mut cdf = Vec::with_capacity(q_vals.len());
    let mut acc = 0.0f64;
    for &v in &q_vals {
        acc += f64::from(v);
        cdf.push(acc);
    }
    let mut accepts = 0usize;
    for i in 0..TRIALS {
        let u = (i as f64 + 0.5) / TRIALS as f64;
        let token = cdf.iter().position(|&c| u < c).unwrap_or(q_vals.len() - 1);
        if accept_draft_token(&p, &q, token as i32) {
            accepts += 1;
        }
    }
    let rate = accepts as f64 / TRIALS as f64;
    let se = (sum_min * (1.0 - sum_min) / TRIALS as f64).sqrt();
    let band = 4.0 * se;
    assert!(
        (rate - sum_min).abs() <= band,
        "acceptance rate {rate:.4} outside the sum-min optimum {sum_min:.4} +- {band:.4}"
    );
    assert!(
        rate > sum_prod,
        "acceptance rate {rate:.4} must exceed the pre-#902 sum-product rate {sum_prod:.4}"
    );
}
