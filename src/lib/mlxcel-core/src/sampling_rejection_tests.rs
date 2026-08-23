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

//! Statistical regression guards for the dual-pivot rejection sampler (#901).
//!
//! The kernel replaces the `argpartition` / `argsort` / `cumsum` filter chain
//! with a shrinking probability interval. That is an *exact support* claim plus
//! a *distributional* claim, and neither is a bitwise comparison, so this file
//! tests both directly:
//!
//! 1. **Support equality.** The set of tokens the kernel can draw must equal the
//!    set the stock chain leaves unmasked, ties included. Tested by drawing
//!    enough samples to cover the support and asserting set equality against a
//!    host reference that reimplements the stock chain's masks in f64.
//! 2. **Goodness of fit.** Frequencies inside the support must match the
//!    truncated, renormalised distribution. Chi-square at upper-tail p = 1e-6,
//!    so a correct implementation fails about once in a million runs per test.
//! 3. **Convergence.** The round cap is reached only by genuinely pathological
//!    rows, the fallback produces valid tokens, and the event is counted.
//! 4. **Hardening.** Subnormal-scale probabilities do not break the shrinking
//!    interval, `-inf`-masked entries (token bias, XTC) are never drawn, and a
//!    fixed seed reproduces a stream.
//! 5. **No synchronization.** The production sampling call must return before a
//!    queue of outstanding GPU work drains. Both decode drivers are software
//!    pipelines that build step n+1 and `async_eval` it before reading step n,
//!    so a sampler that evaluates anything collapses the pipeline. The first
//!    cut of this issue did exactly that and lost 1.7x of end-to-end decode
//!    throughput while the op-level benchmark reported a win, because an
//!    op-level harness synchronizes around every iteration and cannot see a
//!    sync it has already paid for. See
//!    [`the_production_sampling_call_never_synchronizes`], which is a structural
//!    assertion rather than a timing threshold.
//!
//! ## One documented semantic difference
//!
//! When top-k AND top-p are both active, the stock chain masks to the top-k set
//! and then **renormalises** before applying top-p, so its mass target is
//! `top_p * Z_k` with `Z_k` the top-k mass. The kernel applies both tests to the
//! untruncated distribution, so its target is `top_p * total`. Because
//! `Z_k <= total`, the kernel's support is a superset of the stock chain's, and
//! the two agree unless some token's exclusive cumulative mass falls in
//! `(top_p * Z_k, top_p * total]`. Resolving `Z_k` exactly would require
//! pinning `tau_k` to a single float before sampling, which costs the
//! bisection rounds the algorithm exists to avoid. Every other configuration
//! (top-k alone, top-p alone, min-p alone, top-k+min-p, top-p+min-p) is exact,
//! because `Z = 1` for standalone top-p and the min-p threshold is invariant to
//! renormalisation. [`joint_top_k_top_p_support_is_a_superset_of_the_stock_chain`]
//! pins the relationship.
//!
//! ## Correctness is tested for every filter, routing is not
//!
//! The kernel is correct for top-k, top-p, min-p and every combination, and the
//! support and goodness-of-fit tests below exercise all of them through
//! `sampling_rejection_probe`, which bypasses routing. Production only *routes*
//! the combinations that measured faster; see
//! [`the_routing_policy_matches_the_measured_matrix`] for the table and
//! [`fused_sample_routes_only_the_configurations_that_measured_faster`] for the
//! end-to-end check. Keeping the two separate is deliberate: if the occupancy
//! limit that makes top-k slow is ever lifted, widening the policy is a
//! one-constant change against tests that already prove the kernel right.
//!
//! GPU-only: the kernel JITs through `mx.fast.metal_kernel` /
//! `mx.fast.cuda_kernel`, so every test returns early on a CPU-only build,
//! matching the convention in `sampling_gumbel_tests.rs`.
//!
//! Run on Apple Silicon:
//!   cargo test --release -p mlxcel-core --lib --features metal,accelerate \
//!     sampling_rejection_tests::

use super::*;
use std::collections::BTreeSet;

/// Every dispatch outcome description recorded since the last reset.
///
/// Reads the non-destructive channel on purpose. The INFO logger's
/// `sampling_dispatch_drain_report` pops what it reports, and the sampling
/// tests in `sampling.rs` call `report_sampling_dispatch` while these tests run,
/// so a test that drained would race them for its own record.
fn recorded_dispatch() -> Vec<String> {
    let report = sampling_dispatch_recorded_report();
    if report.is_empty() {
        return Vec::new();
    }
    report.lines().map(str::to_string).collect()
}

/// Draws per goodness-of-fit test, the count issue #901 asks for.
/// `MLXCEL_TEST_REJECTION_SAMPLES` overrides it, which is the knob for trading
/// runtime against tightness on a slower host.
const DEFAULT_SAMPLE_COUNT: usize = 1_000_000;

/// Draws for the support-equality tests. Support equality is a set-cover
/// property, not a frequency property: it needs enough draws that the least
/// likely surviving token is hit with overwhelming probability, and the shapes
/// below are built so that bound is far below this count.
const SUPPORT_SAMPLE_COUNT: usize = 200_000;

/// Rows per kernel launch while accumulating a sample. Every row carries the
/// same probabilities and gets its own Philox stream, so one launch yields this
/// many independent draws.
const ROWS_PER_LAUNCH: usize = 16_384;

/// Cap on `rows * vocab` for one launch, so a large-vocabulary shape does not
/// turn [`ROWS_PER_LAUNCH`] into a multi-hundred-megabyte staging buffer.
const MAX_ELEMENTS_PER_LAUNCH: usize = 4 << 20;

/// Minimum expected count per chi-square bin; cells below it are pooled.
const MIN_EXPECTED_PER_BIN: f64 = 10.0;

/// Upper-tail standard normal quantile for p = 1e-6.
const CRITICAL_Z: f64 = 4.7534;

fn sample_count() -> usize {
    std::env::var("MLXCEL_TEST_REJECTION_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_SAMPLE_COUNT)
}

/// True when the rejection kernel can run here. CPU-only builds skip.
fn gpu_backend() -> bool {
    sampling_rejection_available()
}

// -- numeric helpers (same construction as `sampling_gumbel_tests.rs`) --

/// Wilson-Hilferty upper critical value of the chi-square distribution.
/// `(chi2 / df)^(1/3)` is approximately normal with mean `1 - 2/(9 df)` and
/// variance `2/(9 df)`.
fn chi_square_upper_critical(df: f64, z: f64) -> f64 {
    let t = 2.0 / (9.0 * df);
    let x = 1.0 - t + z * t.sqrt();
    df * x * x * x
}

/// Exact `softmax(logits / temperature)` in f64, with the usual max shift.
/// A `-inf` logit yields exactly `0.0`.
fn softmax_reference(logits: &[f32], temperature: f32) -> Vec<f64> {
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&l| f64::from(l) / f64::from(temperature))
        .collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled
        .iter()
        .map(|&s| {
            if s.is_infinite() && s < 0.0 {
                0.0
            } else {
                (s - max).exp()
            }
        })
        .collect();
    let total: f64 = exp.iter().sum();
    exp.iter().map(|&e| e / total).collect()
}

// -- host reference for the stock filter chain's masks --

/// The support the stock `argpartition` top-k mask leaves, ties included.
///
/// The chain takes the k-th largest logit as a threshold and keeps
/// `x >= threshold`, so every token tied with the k-th largest survives and the
/// support can be larger than `k`. That is the behaviour the kernel's
/// `count(p > v) < k` test reproduces exactly.
fn stock_top_k_support(probs: &[f64], top_k: usize) -> BTreeSet<usize> {
    if top_k == 0 || top_k >= probs.len() {
        return (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
    }
    let mut sorted: Vec<f64> = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("finite probabilities"));
    let threshold = sorted[top_k - 1];
    (0..probs.len())
        .filter(|&i| probs[i] >= threshold && probs[i] > 0.0)
        .collect()
}

/// The support the stock top-p filter leaves: the descending prefix whose
/// EXCLUSIVE cumulative mass is at most `top_p`.
///
/// Ties are resolved by value here rather than by the sort's index order, so
/// the callers below use shapes without exact ties at the cutoff. See the
/// module docs.
fn stock_top_p_support(probs: &[f64], top_p: f64) -> BTreeSet<usize> {
    if !(top_p > 0.0 && top_p < 1.0) {
        return (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
    }
    let total: f64 = probs.iter().sum();
    let mut order: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).expect("finite"));
    let mut kept = BTreeSet::new();
    let mut cumulative = 0.0f64;
    for &i in &order {
        if cumulative <= top_p * total {
            kept.insert(i);
        }
        cumulative += probs[i];
    }
    kept
}

/// The support the stock min-p filter leaves: `p >= min_p * max(p)`. Invariant
/// to renormalisation, so where it sits in the chain does not matter.
fn stock_min_p_support(probs: &[f64], min_p: f64) -> BTreeSet<usize> {
    if !(min_p > 0.0 && min_p < 1.0) {
        return (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
    }
    let max = probs.iter().cloned().fold(0.0f64, f64::max);
    (0..probs.len())
        .filter(|&i| probs[i] >= min_p * max && probs[i] > 0.0)
        .collect()
}

fn intersect(a: &BTreeSet<usize>, b: &BTreeSet<usize>) -> BTreeSet<usize> {
    a.intersection(b).copied().collect()
}

// -- kernel drivers --

/// Read a `[..]` uint32 array back to host.
fn u32_values(arr: &MlxArray) -> Vec<u32> {
    array_to_raw_bytes(arr)
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One raw kernel launch: `(ids, converged, rounds)` per row, straight from the
/// kernel with no host fallback in the way.
fn probe(
    logits: &MlxArray,
    rows: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    max_rounds: i32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let stacked = sampling_rejection_probe(logits, temperature, top_k, top_p, min_p, max_rounds);
    let flat = u32_values(&stacked);
    assert_eq!(flat.len(), 3 * rows, "probe returned {} words", flat.len());
    (
        flat[0..rows].to_vec(),
        flat[rows..2 * rows].to_vec(),
        flat[2 * rows..3 * rows].to_vec(),
    )
}

/// Tile one logits row into a `[rows, vocab]` device array.
fn tiled(logits: &[f32], rows: usize) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity(rows * logits.len());
    for _ in 0..rows {
        data.extend_from_slice(logits);
    }
    from_slice_f32(&data, &[rows as i32, logits.len() as i32])
}

/// Draw `n` samples from one logits row through the rejection kernel and return
/// the per-token histogram.
///
/// Asserts along the way that every row converged inside the production round
/// cap; a histogram silently built from cap-overflow rows (which fall back to
/// the row argmax) would look like a badly skewed distribution rather than a
/// convergence failure, so it is caught here instead.
fn histogram(
    logits: &[f32],
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    n: usize,
) -> Vec<u64> {
    let vocab = logits.len();
    let rows = ROWS_PER_LAUNCH
        .min(n.max(1))
        .min((MAX_ELEMENTS_PER_LAUNCH / vocab).max(1));
    let batched = tiled(logits, rows);
    let cap = sampling_rejection_max_rounds();

    let mut counts = vec![0u64; vocab];
    let mut drawn = 0usize;
    while drawn < n {
        let (ids, ok, _rounds) = probe(&batched, rows, temperature, top_k, top_p, min_p, cap);
        assert!(
            ok.iter().all(|&flag| flag == 1),
            "{} of {rows} rows exhausted the {cap}-round cap while building a histogram",
            ok.iter().filter(|&&f| f == 0).count()
        );
        for id in ids {
            if drawn >= n {
                break;
            }
            counts[id as usize] += 1;
            drawn += 1;
        }
    }
    counts
}

/// Chi-square goodness-of-fit of a rejection sample against the truncated,
/// renormalised reference distribution.
///
/// Cells outside `support` have probability exactly zero and are asserted to
/// have drawn nothing; the rest are pooled by increasing probability until each
/// bin's expected count reaches [`MIN_EXPECTED_PER_BIN`], and the statistic is
/// compared against the Wilson-Hilferty critical value at upper-tail p = 1e-6.
fn assert_matches_truncated(
    label: &str,
    probs: &[f64],
    support: &BTreeSet<usize>,
    counts: &[u64],
    n: usize,
) {
    let mass: f64 = support.iter().map(|&i| probs[i]).sum();
    assert!(mass > 0.0, "{label}: empty support");
    let total: u64 = counts.iter().sum();
    assert_eq!(
        total, n as u64,
        "{label}: histogram holds {total} draws, wanted {n}"
    );

    for (i, &count) in counts.iter().enumerate() {
        if !support.contains(&i) {
            assert_eq!(
                count, 0,
                "{label}: token {i} is outside the filtered support but was sampled {count} times"
            );
        }
    }

    let mut live: Vec<usize> = support.iter().copied().collect();
    live.sort_by(|&a, &b| probs[a].partial_cmp(&probs[b]).expect("finite"));

    let n_f = n as f64;
    let mut bins: Vec<(f64, f64)> = Vec::new();
    let mut acc_expected = 0.0f64;
    let mut acc_observed = 0.0f64;
    for &i in &live {
        acc_expected += probs[i] / mass * n_f;
        acc_observed += counts[i] as f64;
        if acc_expected >= MIN_EXPECTED_PER_BIN {
            bins.push((acc_expected, acc_observed));
            acc_expected = 0.0;
            acc_observed = 0.0;
        }
    }
    if acc_expected > 0.0 {
        match bins.last_mut() {
            Some(last) => {
                last.0 += acc_expected;
                last.1 += acc_observed;
            }
            None => bins.push((acc_expected, acc_observed)),
        }
    }

    assert!(
        bins.len() >= 3,
        "{label}: only {} bins survived pooling; raise the sample count",
        bins.len()
    );

    let chi2: f64 = bins
        .iter()
        .map(|&(expected, observed)| {
            let d = observed - expected;
            d * d / expected
        })
        .sum();
    let df = (bins.len() - 1) as f64;
    let critical = chi_square_upper_critical(df, CRITICAL_Z);
    assert!(
        chi2 < critical,
        "{label}: chi-square {chi2:.2} exceeds the p=1e-6 critical value {critical:.2} at \
         df={df} over {n} samples ({} bins)",
        bins.len()
    );
}

/// Assert that the tokens actually drawn are exactly the stock chain's support.
fn assert_support_equals(label: &str, counts: &[u64], expected: &BTreeSet<usize>) {
    let observed: BTreeSet<usize> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, _)| i)
        .collect();
    let extra: Vec<usize> = observed.difference(expected).copied().collect();
    let missing: Vec<usize> = expected.difference(&observed).copied().collect();
    assert!(
        extra.is_empty(),
        "{label}: sampled {extra:?}, which the stock filter chain masks out"
    );
    assert!(
        missing.is_empty(),
        "{label}: never sampled {missing:?}, which the stock filter chain keeps"
    );
}

// -- synthetic logit shapes --

/// A deterministic pseudo-random spread, the shape a real decode step has: a
/// broad low floor with a handful of dominant tokens. Distinct values, so the
/// top-p cutoff is not tie-dependent.
fn spread_logits(vocab: usize, seed: u64) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..vocab)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0;
            -6.0 + 14.0 * u * u * u
        })
        .collect()
}

/// Two separated modes with a low-probability valley between them.
fn bimodal_logits(vocab: usize) -> Vec<f32> {
    let a = vocab / 5;
    let b = (4 * vocab) / 5;
    (0..vocab)
        .map(|i| {
            let da = (i as f32 - a as f32) / 3.0;
            let db = (i as f32 - b as f32) / 3.0;
            let left = 4.0 - da * da;
            let right = 3.5 - db * db;
            left.max(right).max(-8.0)
        })
        .collect()
}

/// Near-uniform over a production-sized vocabulary: the adversarial shape for a
/// value-bisection top-k. No token dominates, so the top-40 boundary sits deep
/// inside a dense cluster and the data-driven pivot buys almost nothing; the
/// bracket has to be closed by bisection alone.
///
/// The band is 2 nats wide rather than the 1e-4 that would make the shape
/// "maximally" flat, and deliberately so. At 1e-4 the probabilities of 152064
/// entries collapse onto a few hundred distinct f32 values, so the k-th largest
/// is a tie group of hundreds of entries; the kernel handles that correctly
/// (ties at the boundary all survive, exactly as the stock `x >= kth` mask
/// intends) but the test could then no longer state its expectation in f64.
/// At 2 nats the order statistics near the top are separated by roughly a
/// hundred f32 ulps, so the support is well defined in either precision while
/// the clustering that stresses the bisection is unchanged.
fn near_uniform_logits(vocab: usize) -> Vec<f32> {
    let mut state = 0xDEAD_BEEF_1234_5678u64;
    (0..vocab)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0;
            2.0 * u - 1.0
        })
        .collect()
}

// -- top-k: support equality and goodness of fit --

#[test]
fn top_k_support_matches_the_argpartition_mask_including_ties() {
    if !gpu_backend() {
        return;
    }
    // 12 tokens tied exactly at the value that lands on the k-th rank, so the
    // stock `x >= kth_logit` mask keeps 8 + 12 = 20 tokens for k = 12. A kernel
    // that resolved top-k as "exactly k tokens" would fail here.
    let vocab = 64usize;
    let mut logits = vec![-9.0f32; vocab];
    for (i, logit) in logits.iter_mut().enumerate().take(8) {
        *logit = 5.0 - i as f32 * 0.5;
    }
    for logit in logits.iter_mut().take(20).skip(8) {
        *logit = 0.75;
    }
    let top_k = 12i32;

    let probs = softmax_reference(&logits, 1.0);
    let expected = stock_top_k_support(&probs, top_k as usize);
    assert_eq!(expected.len(), 20, "reference support {expected:?}");

    let counts = histogram(&logits, 1.0, top_k, 1.0, 0.0, SUPPORT_SAMPLE_COUNT);
    assert_support_equals("top-k ties", &counts, &expected);
}

#[test]
fn top_k_frequencies_match_the_renormalised_truncated_distribution() {
    if !gpu_backend() {
        return;
    }
    let vocab = 256usize;
    let logits = spread_logits(vocab, 0x901);
    let top_k = 40i32;
    let n = sample_count();

    let probs = softmax_reference(&logits, 1.0);
    let expected = stock_top_k_support(&probs, top_k as usize);
    let counts = histogram(&logits, 1.0, top_k, 1.0, 0.0, n);
    assert_support_equals("top-k support", &counts, &expected);
    assert_matches_truncated("top-k", &probs, &expected, &counts, n);
}

#[test]
fn top_k_holds_under_temperature_scaling() {
    if !gpu_backend() {
        return;
    }
    // Since #1379 the kernel resolves the support on the UNTEMPERED row and
    // draws from the tempered one. Top-k's support is invariant under the
    // monotone temperature map, so the support still matches the reference
    // computed at each temperature, and the frequencies must follow the
    // tempered truncated distribution.
    let logits = bimodal_logits(128);
    for &temperature in &[0.5f32, 1.5] {
        let probs = softmax_reference(&logits, temperature);
        let expected = stock_top_k_support(&probs, 16);
        let counts = histogram(&logits, temperature, 16, 1.0, 0.0, SUPPORT_SAMPLE_COUNT);
        assert_support_equals(&format!("top-k T={temperature}"), &counts, &expected);
        assert_matches_truncated(
            &format!("top-k T={temperature}"),
            &probs,
            &expected,
            &counts,
            SUPPORT_SAMPLE_COUNT,
        );
    }
}

// -- top-p --

#[test]
fn top_p_support_matches_the_stock_nucleus_filter() {
    if !gpu_backend() {
        return;
    }
    let vocab = 256usize;
    let logits = spread_logits(vocab, 0x902);
    let probs = softmax_reference(&logits, 1.0);
    let expected = stock_top_p_support(&probs, 0.9);
    assert!(
        expected.len() > 3 && expected.len() < vocab,
        "reference nucleus has {} tokens; the shape does not exercise the filter",
        expected.len()
    );

    let counts = histogram(&logits, 1.0, 0, 0.9, 0.0, SUPPORT_SAMPLE_COUNT);
    assert_support_equals("top-p support", &counts, &expected);
}

#[test]
fn top_p_frequencies_match_the_renormalised_truncated_distribution() {
    if !gpu_backend() {
        return;
    }
    let vocab = 256usize;
    let logits = spread_logits(vocab, 0x903);
    let n = sample_count();
    let probs = softmax_reference(&logits, 1.0);
    let expected = stock_top_p_support(&probs, 0.95);
    let counts = histogram(&logits, 1.0, 0, 0.95, 0.0, n);
    assert_support_equals("top-p", &counts, &expected);
    assert_matches_truncated("top-p", &probs, &expected, &counts, n);
}

// -- min-p --

#[test]
fn min_p_support_and_frequencies_match_the_stock_filter() {
    if !gpu_backend() {
        return;
    }
    let vocab = 256usize;
    let logits = spread_logits(vocab, 0x904);
    let n = sample_count();
    let probs = softmax_reference(&logits, 1.0);
    let expected = stock_min_p_support(&probs, 0.05);
    assert!(
        expected.len() > 3 && expected.len() < vocab,
        "reference min-p set has {} tokens; the shape does not exercise the filter",
        expected.len()
    );
    let counts = histogram(&logits, 1.0, 0, 1.0, 0.05, n);
    assert_support_equals("min-p", &counts, &expected);
    assert_matches_truncated("min-p", &probs, &expected, &counts, n);
}

#[test]
fn min_p_needs_no_rejection_round() {
    if !gpu_backend() {
        return;
    }
    // min-p is a plain threshold, folded into the kernel's initial interval, so
    // the first candidate is always accepted. This pins the "single pass, no
    // iteration" property the issue asks for; a regression that pushed min-p
    // into the rejection loop would show up here as rounds > 1.
    let logits = spread_logits(4096, 0x905);
    let rows = 64usize;
    let batched = tiled(&logits, rows);
    let (_ids, ok, rounds) = probe(&batched, rows, 1.0, 0, 1.0, 0.05, 32);
    assert!(ok.iter().all(|&f| f == 1), "min-p failed to converge");
    assert!(
        rounds.iter().all(|&r| r == 1),
        "min-p consumed more than one round: {:?}",
        rounds.iter().max()
    );
}

// -- joint top-k + top-p --

#[test]
fn joint_top_k_top_p_support_is_a_superset_of_the_stock_chain() {
    if !gpu_backend() {
        return;
    }
    // The one documented semantic difference; see the module docs. The stock
    // chain renormalises over the top-k set before applying top-p, so its mass
    // target is `top_p * Z_k <= top_p`. The kernel's support therefore contains
    // the stock chain's and is contained in the kernel's own top-k set.
    let vocab = 256usize;
    let logits = spread_logits(vocab, 0x906);
    let probs = softmax_reference(&logits, 1.0);

    let top_k = 40i32;
    let top_p = 0.9f64;
    let k_set = stock_top_k_support(&probs, top_k as usize);

    // Stock chain: top-p over the top-k-renormalised distribution.
    let z_k: f64 = k_set.iter().map(|&i| probs[i]).sum();
    let mut k_probs = vec![0.0f64; vocab];
    for &i in &k_set {
        k_probs[i] = probs[i] / z_k;
    }
    let stock = intersect(&k_set, &stock_top_p_support(&k_probs, top_p));

    // Kernel: both tests against the untruncated distribution.
    let kernel_support = intersect(&k_set, &stock_top_p_support(&probs, top_p));

    assert!(
        stock.is_subset(&kernel_support),
        "the renormalised nucleus is not contained in the kernel's; stock={stock:?} \
         kernel={kernel_support:?}"
    );

    let counts = histogram(&logits, 1.0, top_k, top_p as f32, 0.0, SUPPORT_SAMPLE_COUNT);
    assert_support_equals("top-k + top-p", &counts, &kernel_support);
    assert_matches_truncated(
        "top-k + top-p",
        &probs,
        &kernel_support,
        &counts,
        SUPPORT_SAMPLE_COUNT,
    );
}

#[test]
fn all_three_filters_compose_into_one_threshold() {
    if !gpu_backend() {
        return;
    }
    let vocab = 256usize;
    let logits = spread_logits(vocab, 0x907);
    let probs = softmax_reference(&logits, 1.0);
    let expected = intersect(
        &intersect(
            &stock_top_k_support(&probs, 40),
            &stock_top_p_support(&probs, 0.9),
        ),
        &stock_min_p_support(&probs, 0.05),
    );
    let counts = histogram(&logits, 1.0, 40, 0.9, 0.05, SUPPORT_SAMPLE_COUNT);
    assert_support_equals("top-k + top-p + min-p", &counts, &expected);
    assert_matches_truncated(
        "top-k + top-p + min-p",
        &probs,
        &expected,
        &counts,
        SUPPORT_SAMPLE_COUNT,
    );
}

// -- masking (token bias and XTC leave `-inf` logits) --

#[test]
fn masked_entries_are_never_sampled_and_do_not_shift_the_support() {
    if !gpu_backend() {
        return;
    }
    // XTC masks logits to `-inf` before the sampler runs (issue #901 task 4).
    // A masked entry softmaxes to exactly 0, so it must sit outside the
    // proposal set for every filter, and the surviving tokens must renormalise
    // among themselves.
    let vocab = 128usize;
    let mut logits = spread_logits(vocab, 0x908);
    for i in (0..vocab).step_by(3) {
        logits[i] = f32::NEG_INFINITY;
    }
    // Mask what would otherwise be the argmax, so the support is genuinely
    // reshaped rather than merely thinned.
    let argmax = (0..vocab)
        .filter(|&i| logits[i].is_finite())
        .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).expect("finite"))
        .expect("some finite logit");
    logits[argmax] = f32::NEG_INFINITY;

    let probs = softmax_reference(&logits, 1.0);
    for (top_k, top_p, min_p) in [(20i32, 1.0f32, 0.0f32), (0, 0.9, 0.0), (0, 1.0, 0.05)] {
        let expected = intersect(
            &intersect(
                &stock_top_k_support(&probs, top_k.max(0) as usize),
                &stock_top_p_support(&probs, f64::from(top_p)),
            ),
            &stock_min_p_support(&probs, f64::from(min_p)),
        );
        let counts = histogram(&logits, 1.0, top_k, top_p, min_p, SUPPORT_SAMPLE_COUNT);
        for (i, &count) in counts.iter().enumerate() {
            if logits[i].is_infinite() {
                assert_eq!(
                    count, 0,
                    "masked token {i} sampled {count} times at top_k={top_k} top_p={top_p} \
                     min_p={min_p}"
                );
            }
        }
        assert_support_equals(
            &format!("masked top_k={top_k} top_p={top_p} min_p={min_p}"),
            &counts,
            &expected,
        );
    }
}

// -- numerical hardening --

#[test]
fn subnormal_scale_probabilities_do_not_break_the_shrinking_interval() {
    if !gpu_backend() {
        return;
    }
    // A 220-nat spread puts the tail probabilities at ~1e-96 and then below the
    // f32 subnormal floor, which is exactly where a fast-math flush-to-zero
    // would corrupt an arithmetic-midpoint bisection: the interval would stop
    // shrinking and the row would burn its round budget. The kernel bisects the
    // bit patterns instead, so this must converge and stay exact.
    let vocab = 512usize;
    let mut logits = vec![0.0f32; vocab];
    for (i, l) in logits.iter_mut().enumerate() {
        *l = -(i as f32) * 0.43;
    }
    assert!(
        logits[vocab - 1] < -200.0,
        "the shape does not reach subnormal probabilities"
    );

    let rows = 256usize;
    let batched = tiled(&logits, rows);
    let cap = sampling_rejection_max_rounds();

    for (top_k, top_p, min_p) in [(40i32, 1.0f32, 0.0f32), (0, 0.9, 0.0), (0, 1.0, 0.05)] {
        let (ids, ok, rounds) = probe(&batched, rows, 1.0, top_k, top_p, min_p, cap);
        assert!(
            ok.iter().all(|&f| f == 1),
            "subnormal shape failed to converge at top_k={top_k} top_p={top_p} min_p={min_p}: \
             {} of {rows} rows hit the cap",
            ok.iter().filter(|&&f| f == 0).count()
        );
        let worst = rounds.iter().copied().max().unwrap_or(0);
        assert!(
            worst < cap as u32,
            "subnormal shape needed {worst} rounds of {cap} at top_k={top_k} top_p={top_p} \
             min_p={min_p}"
        );

        // And the support is still the stock chain's, not a flush-widened one.
        let probs = softmax_reference(&logits, 1.0);
        let expected = intersect(
            &intersect(
                &stock_top_k_support(&probs, top_k.max(0) as usize),
                &stock_top_p_support(&probs, f64::from(top_p)),
            ),
            &stock_min_p_support(&probs, f64::from(min_p)),
        );
        for id in ids {
            assert!(
                expected.contains(&(id as usize)),
                "subnormal shape sampled {id}, outside the stock support, at top_k={top_k} \
                 top_p={top_p} min_p={min_p}"
            );
        }
    }
}

// -- determinism --

#[test]
fn a_fixed_seed_reproduces_the_stream() {
    if !gpu_backend() {
        return;
    }
    let logits = spread_logits(1024, 0x909);
    let rows = 64usize;
    let batched = tiled(&logits, rows);

    // A stream, not one draw: successive calls consume successive keys, so this
    // also pins that the key sequence advances identically after reseeding.
    let run = || {
        random_seed(0x5EED_0901);
        let mut stream = Vec::new();
        for _ in 0..8 {
            let (ids, _, _) = probe(&batched, rows, 1.0, 40, 0.9, 0.0, 32);
            stream.extend(ids);
        }
        stream
    };

    let first = run();
    let second = run();
    assert_eq!(first, second, "same seed produced a different token stream");
    assert!(
        first.iter().any(|&id| id != first[0]),
        "degenerate stream: every draw returned the same id"
    );
}

// -- convergence cap --

#[test]
fn a_near_uniform_152k_row_converges_inside_the_round_cap() {
    if !gpu_backend() {
        return;
    }
    // The adversarial case the issue names: 152064 entries whose probabilities
    // differ in the sixth significant digit, so the top-40 boundary sits inside
    // a dense cluster and the data-driven pivot buys almost nothing. The
    // bit-space bisection still has to close it inside 32 rounds.
    let vocab = 152_064usize;
    let logits = near_uniform_logits(vocab);
    let rows = 8usize;
    let batched = tiled(&logits, rows);
    let cap = sampling_rejection_max_rounds();

    let probs = softmax_reference(&logits, 1.0);
    let mut descending = probs.clone();
    descending.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    let tau_40 = descending[39];
    // The kernel resolves the boundary in f32; the reference above is f64. Allow
    // the boundary itself to move by a few f32 ulps, which is the only
    // disagreement the two precisions can produce on this shape.
    let slack = tau_40 * 4.0 * f64::from(f32::EPSILON);

    for (label, top_p) in [("top-k only", 1.0f32), ("top-k + top-p", 0.9f32)] {
        let (ids, ok, rounds) = probe(&batched, rows, 1.0, 40, top_p, 0.0, cap);
        let worst = rounds.iter().copied().max().unwrap_or(0);
        assert!(
            ok.iter().all(|&f| f == 1),
            "{label}: near-uniform 152K exhausted the {cap}-round cap on {} of {rows} rows \
             (worst {worst})",
            ok.iter().filter(|&&f| f == 0).count()
        );
        assert!(
            worst > 1,
            "{label}: the near-uniform shape converged in one round; it is not exercising the loop"
        );
        for id in ids {
            assert!(
                probs[id as usize] >= tau_40 - slack,
                "{label}: near-uniform 152K sampled {id} at p={:e}, below the top-40 threshold \
                 {tau_40:e}",
                probs[id as usize]
            );
        }
    }
}

#[test]
fn a_starved_round_cap_falls_back_and_the_event_is_counted() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    // Same adversarial shape, one round instead of 32: the loop cannot close on
    // the boundary, so the kernel must report failure rather than return a
    // token from outside the support, the host must fall back to the stock
    // chain, and the cap-overflow metric must move.
    let vocab = 152_064usize;
    let logits = near_uniform_logits(vocab);
    let rows = 4usize;
    let batched = tiled(&logits, rows);

    let (_ids, ok, rounds) = probe(&batched, rows, 1.0, 40, 1.0, 0.0, 1);
    assert!(
        ok.contains(&0),
        "a one-round cap converged on the adversarial shape; the cap is not reachable and the \
         fallback is untestable"
    );
    assert!(rounds.iter().all(|&r| r == 1), "rounds {rounds:?}");

    reset_sampling_dispatch();
    assert_eq!(rejection_cap_overflow_rows(), 0);
    let tokens = u32_values(&fused_sample_rejection(&batched, 1.0, 40, 1.0, 0.0, 1));
    assert_eq!(tokens.len(), rows);
    assert!(
        tokens.iter().all(|&id| (id as usize) < vocab),
        "fallback produced an out-of-range token: {tokens:?}"
    );
    assert!(
        rejection_cap_overflow_rows() > 0,
        "the cap-overflow row counter did not move"
    );
    assert_eq!(
        rejection_cap_overflow_launches(),
        1,
        "the cap-overflow launch counter did not move exactly once"
    );

    // And the fallback announces itself, once, at a level a server actually
    // emits. This is the guard against a silent permanent fallback.
    let lines = recorded_dispatch();
    assert!(
        lines.iter().any(|l| l.contains("rejection cap")),
        "no cap-overflow line in the dispatch report: {lines:?}"
    );
}

// -- routing --

/// The measured matrix the routing policy encodes (M1 Ultra, three repetitions,
/// vocab {32K, 64K, 152K} x batch {1, 4, 8}, both arms timed in one run).
///
/// | configuration | measured speedup | routed |
/// |---|---|---|
/// | top-p alone | 1.28x - 2.35x, every cell | yes, every vocabulary |
/// | top-k alone | 0.31x - 0.97x, every cell | no |
/// | min-p alone | 0.47x - 0.88x at 152K | no |
/// | top-k + top-p | 1.27x - 1.64x at 32K, 0.71x - 0.83x at 152K | only at vocab <= 32768 |
///
/// The kernel replaces a sort, so it wins exactly where the stock chain sorts,
/// which is when top-p is active. The `rounds` column of the microbenchmark
/// shows why the joint case degrades with vocabulary: top-p accepts in one or
/// two rounds, top-k needs two to seven and the count grows with the
/// vocabulary, and every round is another full-row sweep on a single
/// threadgroup.
#[test]
fn the_routing_policy_matches_the_measured_matrix() {
    // Pure host arithmetic, so this runs on a CPU-only build too.
    const VOCABS: [i32; 4] = [4096, 32_768, 65_536, 152_064];

    for vocab in VOCABS {
        assert!(
            sampling_rejection_routes(vocab, 0, 0.9, 0.0),
            "top-p alone at vocab {vocab} measured a win at every cell and must route"
        );
        // min-p is folded into the kernel's initial interval and costs no round,
        // so it rides along with top-p. Extrapolated from the two filters
        // measured separately, not itself a measured cell.
        assert!(
            sampling_rejection_routes(vocab, 0, 0.9, 0.05),
            "top-p + min-p at vocab {vocab} must route"
        );
        assert!(
            !sampling_rejection_routes(vocab, 40, 1.0, 0.0),
            "top-k alone at vocab {vocab} measured 0.31x-0.97x and must not route"
        );
        assert!(
            !sampling_rejection_routes(vocab, 0, 1.0, 0.05),
            "min-p alone at vocab {vocab} measured 0.47x-0.88x and must not route"
        );
        assert!(
            !sampling_rejection_routes(vocab, 40, 1.0, 0.05),
            "top-k + min-p at vocab {vocab} runs no sort in the stock chain and must not route"
        );
    }

    // The joint case is capped at the vocabulary where it was measured to win.
    // 65536 is excluded because it has not been measured for this combination,
    // not because it was measured to lose.
    assert!(sampling_rejection_routes(4096, 40, 0.9, 0.0));
    assert!(sampling_rejection_routes(32_768, 40, 0.9, 0.0));
    assert!(!sampling_rejection_routes(65_536, 40, 0.9, 0.0));
    assert!(!sampling_rejection_routes(152_064, 40, 0.9, 0.0));
    assert!(sampling_rejection_routes(32_768, 40, 0.9, 0.05));
    assert!(!sampling_rejection_routes(152_064, 40, 0.9, 0.05));

    // A top-k that cannot bind is not a top-k: it leaves the chain's cost
    // profile at top-p alone, so it does not drag the joint ceiling in.
    assert!(sampling_rejection_routes(152_064, 152_064, 0.9, 0.0));
    // `top_k == 1` is the greedy spelling; `fused_sample` takes `argmax` long
    // before it reaches this policy, so the value here is not load-bearing.
    assert!(sampling_rejection_routes(152_064, 1, 0.9, 0.0));
}

/// Which path `fused_sample` took for one configuration, decided by comparing
/// its stream against both arms at one seed.
///
/// The dispatch record cannot answer this on its own: it is process-global and
/// one-shot per kind, so another test sampling concurrently in the same binary
/// can claim the "rejection kernel" slot, and a negative assertion on it would
/// be a coin flip. Both entry points consume exactly one RNG key per call, so at
/// a fixed seed the streams are an exact, race-free witness of which code ran.
fn took_the_kernel(batched: &MlxArray, top_k: i32, top_p: f32, min_p: f32) -> bool {
    let cap = sampling_rejection_max_rounds();
    random_seed(0x0901_0F0F);
    let routed = u32_values(&fused_sample(batched, 1.0, top_k, top_p, min_p));
    random_seed(0x0901_0F0F);
    let kernel = u32_values(&fused_sample_rejection(
        batched, 1.0, top_k, top_p, min_p, cap,
    ));
    random_seed(0x0901_0F0F);
    let chain = u32_values(&fused_sample_categorical(batched, 1.0, top_k, top_p, min_p));
    assert_ne!(
        kernel, chain,
        "the two arms produced the same stream for top_k={top_k} top_p={top_p} min_p={min_p}, \
         so this witness cannot tell them apart"
    );
    routed == kernel
}

#[test]
fn fused_sample_routes_only_the_configurations_that_measured_faster() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    // vocab 1024, below the joint ceiling, so top-p and top-k+top-p both route.
    let logits = spread_logits(1024, 0x90A);
    let rows = 32usize;
    let batched = tiled(&logits, rows);

    for (top_k, top_p, min_p, want_kernel) in [
        (0i32, 0.9f32, 0.0f32, true),
        (0, 0.9, 0.05, true),
        (40, 0.9, 0.0, true),
        (40, 0.9, 0.05, true),
        (40, 1.0, 0.0, false),
        (0, 1.0, 0.05, false),
        (40, 1.0, 0.05, false),
    ] {
        assert_eq!(
            took_the_kernel(&batched, top_k, top_p, min_p),
            want_kernel,
            "top_k={top_k} top_p={top_p} min_p={min_p}: expected kernel={want_kernel}"
        );

        if !want_kernel {
            reset_sampling_dispatch();
            let tokens = u32_values(&fused_sample(&batched, 1.0, top_k, top_p, min_p));
            assert_eq!(tokens.len(), rows);
            let lines = recorded_dispatch();
            assert!(
                lines.iter().any(|l| l.contains("not routed")),
                "top_k={top_k} top_p={top_p} min_p={min_p} declined without saying why: \
                 {lines:?}"
            );
        }
    }
}

#[test]
fn a_large_vocabulary_sends_the_joint_config_back_to_the_stock_chain() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    // The llama-server default (top-k 40 + top-p 0.9) above the measured
    // ceiling. This is the cell that measured 0.71x-0.83x, so it must decline,
    // and it must say the vocabulary is the reason.
    let logits = spread_logits(65_536, 0x90F);
    let batched = tiled(&logits, 4);

    assert!(
        !took_the_kernel(&batched, 40, 0.9, 0.0),
        "a 65536-vocabulary top-k + top-p launch reached the kernel"
    );

    reset_sampling_dispatch();
    let tokens = u32_values(&fused_sample(&batched, 1.0, 40, 0.9, 0.0));
    assert_eq!(tokens.len(), 4);
    let lines = recorded_dispatch();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("not routed") && l.contains("65536")),
        "the decline did not name the vocabulary that caused it: {lines:?}"
    );

    // top-p alone at the same vocabulary still routes, which is the whole point
    // of splitting the policy by filter rather than by vocabulary alone.
    assert!(
        took_the_kernel(&batched, 0, 0.9, 0.0),
        "top-p alone at vocab 65536 measured a win at every batch and must still route"
    );
}

#[test]
fn greedy_and_no_filter_paths_report_that_they_bypass_the_rejection_kernel() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    let logits = spread_logits(1024, 0x90B);
    let rows = 8usize;
    let batched = tiled(&logits, rows);

    reset_sampling_dispatch();
    let greedy = u32_values(&fused_sample(&batched, 0.0, 0, 1.0, 0.0));
    let reference = u32_values(&argmax_last_axis(&batched));
    assert_eq!(greedy, reference, "temperature 0 diverged from argmax");

    reset_sampling_dispatch();
    let _ = u32_values(&fused_sample(&batched, 1.0, 0, 1.0, 0.0));
    let lines = recorded_dispatch();
    assert!(
        lines.iter().any(|l| l.contains("Gumbel-max kernel")),
        "the no-filter path did not report the #900 kernel: {lines:?}"
    );
}

#[test]
fn the_reference_arm_stays_on_the_argpartition_chain() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    // `fused_sample_categorical` is the A/B baseline the microbenchmark times.
    // If it ever started taking a kernel, the benchmark would compare a path
    // against itself, which is exactly the failure #899 shipped.
    let logits = spread_logits(512, 0x90C);
    let batched = tiled(&logits, 8);

    reset_sampling_dispatch();
    let _ = u32_values(&fused_sample_categorical(&batched, 1.0, 40, 0.9, 0.0));
    let lines = recorded_dispatch();
    assert!(
        lines.iter().any(|l| l.contains("fused_sample_categorical")),
        "the reference arm did not identify itself: {lines:?}"
    );
    // And it really is a different path. A negative assertion on the shared
    // dispatch record would be racy here (another test in this binary can set
    // the rejection bit between the reset and the read), so compare the streams
    // instead: at one seed the two arms agree only if they are the same code.
    random_seed(0x0901_ABCD);
    let chain = u32_values(&fused_sample_categorical(&batched, 1.0, 40, 0.9, 0.0));
    random_seed(0x0901_ABCD);
    let kernel = u32_values(&fused_sample_rejection(&batched, 1.0, 40, 0.9, 0.0, 32));
    assert_ne!(
        chain, kernel,
        "the reference arm produced the rejection kernel's stream; the A/B is measuring one path"
    );
}

#[test]
fn per_row_parameters_reach_the_kernel_from_one_launch() {
    if !gpu_backend() {
        return;
    }
    // The kernel reads `{top_k, top_p, min_p}` per row, so one launch can serve
    // a batch whose rows disagree. `fused_sample` broadcasts a single config
    // today, so this checks the property the kernel guarantees rather than the
    // caller's use of it: two launches with different k over the same
    // probabilities must produce supports of different size.
    let vocab = 256usize;
    let logits = spread_logits(vocab, 0x90D);
    let probs = softmax_reference(&logits, 1.0);

    let narrow = histogram(&logits, 1.0, 4, 1.0, 0.0, 20_000);
    let wide = histogram(&logits, 1.0, 40, 1.0, 0.0, 20_000);
    let narrow_seen = narrow.iter().filter(|&&c| c > 0).count();
    let wide_seen = wide.iter().filter(|&&c| c > 0).count();
    assert_eq!(narrow_seen, stock_top_k_support(&probs, 4).len());
    assert!(
        wide_seen > narrow_seen,
        "k=40 covered {wide_seen} tokens, k=4 covered {narrow_seen}"
    );
}

// -- the sampler must not synchronize --

/// A chain of large matmuls, long enough that draining it is unmistakably
/// measurable. Returns the tail of the chain, unevaluated.
fn pending_gpu_work(links: usize) -> UniquePtr<MlxArray> {
    let n = 1024i32;
    let side = (n * n) as usize;
    let a: Vec<f32> = (0..side).map(|i| ((i % 251) as f32) * 0.001).collect();
    let lhs = from_slice_f32(&a, &[n, n]);
    let mut acc = from_slice_f32(&a, &[n, n]);
    for _ in 0..links {
        acc = matmul(&lhs, &acc);
    }
    acc
}

/// Time how long `f` takes while `links` matmuls are still outstanding on the
/// GPU, and how long the outstanding work then takes to drain.
///
/// A call that only builds a graph returns in microseconds and leaves the drain
/// to be paid afterwards. A call that evaluates anything has to wait for the
/// queue ahead of it first, so it absorbs the drain and the second number
/// collapses. The ratio is therefore a direct, hardware-speed-independent
/// witness of whether the call synchronizes.
fn build_and_drain<F>(links: usize, mut f: F) -> (std::time::Duration, std::time::Duration)
where
    F: FnMut() -> UniquePtr<MlxArray>,
{
    let pending = pending_gpu_work(links);
    async_eval(&pending);

    let t0 = std::time::Instant::now();
    let out = f();
    let build = t0.elapsed();

    let t1 = std::time::Instant::now();
    eval(&pending);
    synchronize_default();
    let drain = t1.elapsed();

    eval(&out);
    synchronize_default();
    (build, drain)
}

/// The production sampling call must not block on the GPU.
///
/// This is the guard for the failure this issue actually shipped once. The CLI
/// decode loop and the batch scheduler are both software pipelines: they build
/// step n+1 and `async_eval` it, then read step n. A sampler that evaluates
/// anything inside that build collapses the pipeline, and the cost is invisible
/// to an op-level benchmark because the benchmark synchronizes around every
/// iteration anyway, which is the one regime where a forced sync is free.
/// Measured end to end on Qwen3-0.6B the difference was 1.7x, against an
/// op-level matrix that scored the same configuration at 1.14x to 1.17x faster.
///
/// So the property is tested structurally rather than by timing the op: with a
/// large chain of matmuls outstanding, the sampler must return before that
/// chain drains.
#[test]
fn the_production_sampling_call_never_synchronizes() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    let vocab = 152_064usize;
    let logits = spread_logits(vocab, 0x910);
    let batched = tiled(&logits, 1);

    // Calibrate: enough links that the drain dominates any graph-build cost.
    let links = 24usize;

    for (label, top_k, top_p, min_p) in [
        ("top-p (routed)", 0i32, 0.9f32, 0.0f32),
        ("top-k (declined)", 40, 1.0, 0.0),
        ("greedy", 0, 1.0, 0.0),
    ] {
        let (build, drain) =
            build_and_drain(links, || fused_sample(&batched, 1.0, top_k, top_p, min_p));
        assert!(
            build * 4 < drain,
            "{label}: fused_sample took {build:?} to build while {drain:?} of GPU work was \
             outstanding, so it waited for the queue instead of enqueueing behind it. A \
             synchronizing sampler collapses the decode pipeline."
        );
    }
}

/// The forced entry point is allowed to synchronize, and does.
///
/// It is what the microbenchmark and the cap-overflow tests call, because they
/// need the per-row converged flag on the host. Pinning the difference keeps the
/// two entry points from quietly converging: if this ever stopped blocking, the
/// cap-overflow fallback would have stopped running.
#[test]
fn the_forced_entry_point_is_the_one_that_synchronizes() {
    if !gpu_backend() {
        return;
    }
    let vocab = 152_064usize;
    let logits = spread_logits(vocab, 0x911);
    let batched = tiled(&logits, 1);
    let cap = sampling_rejection_max_rounds();

    let (build, drain) = build_and_drain(24, || {
        fused_sample_rejection(&batched, 1.0, 0, 0.9, 0.0, cap)
    });
    assert!(
        build > drain,
        "fused_sample_rejection built in {build:?} against a {drain:?} drain, so it no longer \
         reads the converged flags back and the cap-overflow fallback cannot be running"
    );
}

#[test]
fn cap_overflow_is_detected_without_waiting_on_the_production_path() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    // The production path cannot afford to read the converged flags at launch
    // time, so it parks them and inspects them later, once `is_available()`
    // says the launch has landed. What this test pins is the COUNTING RULE and
    // its report, through the same `count_and_report_overflow` the deferred
    // drain uses: a one-round cap on the adversarial near-uniform shape
    // guarantees rows that both fail and consume the whole budget, which is the
    // conjunction the rule requires.
    //
    // It deliberately does not test the delivery ring. That ring is best-effort
    // by design (production drains it on the next sampler call, and a burst from
    // other threads can evict an entry that has not landed), which is the right
    // trade for a hot-path diagnostic and the wrong basis for an assertion.
    // `the_production_sampling_call_never_synchronizes` covers the property that
    // made the deferral necessary in the first place.
    //
    // Note what is deliberately not claimed. The overflowing launch is not
    // re-sampled; its token was already emitted. The rows returned their row
    // argmax, which the kernel guarantees is inside the filtered support, so the
    // degradation is one greedy draw rather than an invalid token, and the point
    // of the check is that it is reported rather than silent.
    let vocab = 152_064usize;
    let logits = near_uniform_logits(vocab);
    let rows = 4usize;
    let batched = tiled(&logits, rows);

    reset_sampling_dispatch();
    assert_eq!(rejection_cap_overflow_rows(), 0);

    // top-k must be active. top-p alone accepts in the FIRST round, because a
    // probability-weighted draw lands in the high-mass head and `mass(> p)` is
    // immediately under the target, so a one-round cap converges and there is
    // nothing to count. Requiring the draw to land in the top forty of 152064
    // entries is what makes one round hopeless.
    let ids = u32_values(&fused_sample_rejection_deferred(
        &batched, 1.0, 40, 0.9, 0.0, 1,
    ));
    assert_eq!(ids.len(), rows);
    assert!(
        ids.iter().all(|&id| (id as usize) < vocab),
        "the unconverged rows returned something outside the vocabulary: {ids:?}"
    );
    // Not `== rows`: a single round can still accept, because the first draw is
    // probability-weighted and can land inside the top-40 by luck. What the rule
    // must do is count the rows that did not.
    let counted = rejection_cap_overflow_rows();
    assert!(
        counted > 0,
        "the overflow counting rule counted nothing on a one-round cap over a near-uniform \
         152064-entry row, where convergence in one round is a coin toss at best"
    );
    assert!(
        counted <= rows as u64,
        "the rule counted {counted} overflowing rows out of {rows}"
    );
    assert_eq!(
        rejection_cap_overflow_launches(),
        1,
        "the launch counter did not move exactly once"
    );

    let lines = recorded_dispatch();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("exhausted the round cap on an earlier launch")),
        "the deferred overflow was counted but not announced: {lines:?}"
    );
}

// -- agreement with the distribution #902 reports --

/// Row 0 of `fused_sample_probs` as host floats.
fn reported_probs(
    batched: &MlxArray,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    vocab: usize,
) -> Vec<f32> {
    let probs = fused_sample_probs(batched, 1.0, top_k, top_p, min_p);
    array_to_raw_bytes(&probs)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .take(vocab)
        .collect()
}

/// `fused_sample_probs` must describe the distribution `fused_sample` draws
/// from, for routed and declined configurations alike.
///
/// Issue #902 built that function on the sampler's own filter chain precisely so
/// the two could not drift. Issue #901 then gave the sampler a second support:
/// the rejection kernel evaluates top-k and top-p against the untruncated row
/// while the stock chain renormalises between them, so for that one combination
/// the kernel's support is a superset. `fused_sample_probs` therefore takes the
/// same routing decision the sampler took, and this test is what holds the two
/// together: it draws through `fused_sample` and requires every sampled token to
/// carry mass in the reported distribution, and every well-supported token in
/// that distribution to be reachable.
///
/// Getting this wrong would not fail any other test. It would quietly hand the
/// speculative accept test (#902) a `p` that is missing part of the target's
/// support, which rejects tokens the target could genuinely have produced.
#[test]
fn the_reported_distribution_matches_what_the_sampler_draws() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    let vocab = 512usize;
    let logits = spread_logits(vocab, 0x912);
    let single = tiled(&logits, 1);
    let rows = 4096usize;
    let batched = tiled(&logits, rows);

    for (top_k, top_p, min_p) in [
        (0i32, 0.9f32, 0.0f32), // routed
        (0, 0.9, 0.05),         // routed
        (40, 0.9, 0.0),         // routed, and the one divergent combination
        (40, 0.9, 0.05),        // routed
        (40, 1.0, 0.0),         // declined: stock chain
        (0, 1.0, 0.05),         // declined: stock chain
    ] {
        let routed = sampling_rejection_routes(vocab as i32, top_k, top_p, min_p);
        let probs = reported_probs(&single, top_k, top_p, min_p, vocab);
        let total: f64 = probs.iter().map(|&v| f64::from(v)).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "top_k={top_k} top_p={top_p} min_p={min_p}: reported probs sum to {total}"
        );

        let mut observed = BTreeSet::new();
        let mut drawn = 0usize;
        while drawn < 200_000 {
            for id in u32_values(&fused_sample(&batched, 1.0, top_k, top_p, min_p)) {
                observed.insert(id as usize);
                drawn += 1;
            }
        }

        for &id in &observed {
            assert!(
                probs[id] > 0.0,
                "top_k={top_k} top_p={top_p} min_p={min_p} (routed={routed}): the sampler drew \
                 token {id}, which the reported distribution gives zero mass"
            );
        }
        for (id, &pr) in probs.iter().enumerate() {
            if pr > 1e-4 {
                assert!(
                    observed.contains(&id),
                    "top_k={top_k} top_p={top_p} min_p={min_p} (routed={routed}): the reported \
                     distribution gives token {id} mass {pr}, but 200000 draws never produced it"
                );
            }
        }
    }
}

/// The divergent combination is genuinely divergent, so the test above is not
/// passing vacuously.
///
/// With top-k and top-p both active the rejection kernel's support is a strict
/// superset of the stock chain's on a distribution built to straddle the
/// boundary. If this ever stops holding, the two orderings have converged and
/// the `rejection_semantics` flag can be deleted; until then, deleting it would
/// silently narrow the reported target support.
#[test]
fn the_two_filter_orderings_really_do_differ_for_top_k_plus_top_p() {
    if !gpu_backend() {
        return;
    }
    let vocab = 512usize;
    let logits = spread_logits(vocab, 0x913);
    let probs = softmax_reference(&logits, 1.0);

    let k_set = stock_top_k_support(&probs, 40);
    let z_k: f64 = k_set.iter().map(|&i| probs[i]).sum();
    assert!(
        z_k < 1.0,
        "the top-40 set carries all the mass; shape is unusable"
    );

    let mut renormalised = vec![0.0f64; vocab];
    for &i in &k_set {
        renormalised[i] = probs[i] / z_k;
    }
    let stock = intersect(&k_set, &stock_top_p_support(&renormalised, 0.9));
    let kernel = intersect(&k_set, &stock_top_p_support(&probs, 0.9));

    assert!(
        stock.is_subset(&kernel),
        "the renormalised nucleus escaped the untruncated one: stock={stock:?} kernel={kernel:?}"
    );
    assert!(
        stock.len() < kernel.len(),
        "the two orderings agree on this shape ({} tokens each), so the divergence test has no \
         teeth; pick a distribution that straddles the top_p boundary",
        stock.len()
    );
}

/// #1379: the support comes from the UNTEMPERED distribution while the draw
/// follows the tempered one.
///
/// At `T = 0.5` and `top_p = 0.9` the untempered nucleus of this row keeps
/// seven tokens (exclusive cumulative mass 0.88 <= 0.9 at the seventh) while
/// the tempered nucleus the pre-#1379 kernel resolved kept four, so a kernel
/// that still filtered the tempered row fails the support half, and a kernel
/// that drew from the untempered row fails the frequency half. 100k draws,
/// each in-support frequency within 3 sigma of the tempered truncated
/// reference, and not one draw outside the untempered nucleus.
///
/// The stock-chain arm is exercised through `fused_sample_categorical`, the
/// committed in-process equivalent of `MLXCEL_SAMPLING_REJECTION=0` (the env
/// switch is read once per process, so the kill switch itself cannot be
/// toggled inside a test); its support must equal the kernel's.
#[test]
fn rejection_kernel_filter_on_untempered_draw_on_tempered() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !gpu_backend() {
        return;
    }
    let base: [f64; 16] = [
        0.30, 0.20, 0.14, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01, 0.004, 0.003, 0.002, 0.0005,
        0.0003, 0.0002,
    ];
    let logits: Vec<f32> = base.iter().map(|&p| p.ln() as f32).collect();
    let temperature = 0.5f32;
    let top_p = 0.9f32;
    let n = 100_000usize;

    // Untempered nucleus: descending prefix whose exclusive cumulative mass
    // stays within top_p, on the raw probabilities.
    let raw = softmax_reference(&logits, 1.0);
    let support = stock_top_p_support(&raw, f64::from(top_p));
    assert_eq!(
        support,
        (0..7usize).collect::<BTreeSet<_>>(),
        "the row no longer produces the intended untempered nucleus"
    );

    // Tempered draw distribution, truncated to that support.
    let tempered = softmax_reference(&logits, temperature);
    let mass: f64 = support.iter().map(|&i| tempered[i]).sum();

    random_seed(0x1379_F117);
    let counts = histogram(&logits, temperature, 0, top_p, 0.0, n);

    for (i, &count) in counts.iter().enumerate() {
        if !support.contains(&i) {
            assert_eq!(
                count, 0,
                "token {i} lies outside the untempered nucleus and was drawn {count} times"
            );
            continue;
        }
        let q = tempered[i] / mass;
        let expect = q * n as f64;
        let sigma = (n as f64 * q * (1.0 - q)).sqrt();
        let dev = (count as f64 - expect).abs();
        assert!(
            dev <= 3.0 * sigma,
            "token {i}: {count} draws against expectation {expect:.1} (tempered truncated), \
             deviation {dev:.1} > 3 sigma ({sigma:.1})"
        );
    }

    // Stock chain (reference arm): same support.
    let rows = 512usize;
    let batched = tiled(&logits, rows);
    random_seed(0x1379_F118);
    let mut stock_support = BTreeSet::new();
    for _ in 0..40 {
        for id in u32_values(&fused_sample_categorical(
            &batched,
            temperature,
            0,
            top_p,
            0.0,
        )) {
            stock_support.insert(id as usize);
        }
    }
    assert_eq!(
        stock_support, support,
        "the stock chain and the rejection kernel resolve different supports"
    );
}

/// Regression guard for the pre-#1379 stock-chain min-p no-op.
///
/// Through #1378 the chain routed min-p through a `mlx::core::compile`d
/// callable that returned its input UNFILTERED on the Metal backend: the
/// reported distribution did not move across min_p values from 0.05 to 0.9,
/// and the sampler drew tokens below `min_p * p_max` on both the production
/// stock chain and this reference arm. The rejection kernel applied min-p
/// correctly, which kept the defect invisible on routed configurations, and
/// `the_reported_distribution_matches_what_the_sampler_draws` compared the
/// broken chain against its own broken report, so it passed. This pins the
/// straight-line replacement, on both the production entry point (min-p alone
/// is never routed, so `fused_sample` takes the stock chain here) and the
/// reference arm.
#[test]
fn min_p_filters_the_stock_chain() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    // p = [0.5, 0.3, 0.15, 0.05]; min_p = 0.5 puts the threshold at 0.25, so
    // only {0, 1} survive and renormalise to [0.625, 0.375].
    let row: Vec<f32> = [0.5f32, 0.3, 0.15, 0.05].iter().map(|p| p.ln()).collect();
    let single = from_slice_f32(&row, &[1, 4]);
    let probs: Vec<f32> = array_to_raw_bytes(&fused_sample_probs(&single, 1.0, 0, 1.0, 0.5))
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected = [0.625f32, 0.375, 0.0, 0.0];
    for (i, (&got, &want)) in probs.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "reported min-p distribution wrong at token {i}: got {got}, want {want} \
             (full row {probs:?})"
        );
    }

    let rows = 64usize;
    let batched = tiled(&row, rows);
    for (label, arm) in [("fused_sample", true), ("fused_sample_categorical", false)] {
        random_seed(0x1379_313B);
        let mut drawn = 0usize;
        while drawn < 2000 {
            let ids = if arm {
                u32_values(&fused_sample(&batched, 1.0, 0, 1.0, 0.5))
            } else {
                u32_values(&fused_sample_categorical(&batched, 1.0, 0, 1.0, 0.5))
            };
            for id in ids {
                assert!(
                    id < 2,
                    "{label} drew token {id}, which min-p 0.5 must exclude"
                );
                drawn += 1;
            }
        }
    }
}
