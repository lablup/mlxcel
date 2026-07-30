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

//! Statistical correctness and routing tests for the Gumbel-max sampling
//! kernel (issue #900).
//!
//! The kernel replaces `mlx::core::random::categorical` on the no-filter
//! sampling path. That is a *distributional* equivalence, not a bit-for-bit
//! one: the two consume different amounts of the same RNG stream, so the only
//! way to show the replacement is correct is to draw a large sample and test
//! the empirical frequencies against the exact softmax probabilities.
//!
//! What each group pins:
//!
//! 1. **Goodness of fit**: chi-square against `softmax(logits / T)` on four
//!    logit shapes (peaked, flat, bimodal, `-inf`-masked) and three
//!    temperatures. See [`assert_matches_softmax`] for the binning rule and
//!    the acceptance threshold.
//! 2. **Masking**: a `-inf` logit (what token bias and the XTC pre-step leave
//!    behind) is never sampled, at any temperature.
//! 3. **Determinism**: the same seed reproduces the same token stream, and
//!    the sampled id does not depend on the launch shape the split heuristic
//!    picks.
//! 4. **Routing**: `fused_sample` reaches the kernel exactly when no filter is
//!    active, greedy stays byte-identical to `argmax`, and every filtered
//!    configuration stays bit-identical to the pre-#900 categorical path.
//!
//! GPU-only: the kernel JITs through `mx.fast.metal_kernel` /
//! `mx.fast.cuda_kernel`, so every test returns early on a CPU-only build,
//! matching the convention in `fused_moe_parity_tests.rs`.
//!
//! Run on Apple Silicon:
//!   cargo test --release -p mlxcel-core --lib --features metal,accelerate \
//!     sampling_gumbel_tests::

use super::*;

/// Draws per goodness-of-fit test. `MLXCEL_TEST_GUMBEL_SAMPLES` overrides it,
/// which is the knob for trading runtime against tightness on a slower host.
const DEFAULT_SAMPLE_COUNT: usize = 1_000_000;

/// Rows per kernel launch while accumulating a sample. Every row carries the
/// same logits and gets its own Philox stream, so one launch yields this many
/// independent draws.
const ROWS_PER_LAUNCH: usize = 16_384;

/// Cap on `rows * vocab` for one launch, so a large-vocabulary shape does not
/// turn [`ROWS_PER_LAUNCH`] into a multi-hundred-megabyte staging buffer.
/// 4M f32 is 16 MiB, small enough to stay resident and large enough that the
/// per-launch fixed cost is amortized.
const MAX_ELEMENTS_PER_LAUNCH: usize = 4 << 20;

/// Minimum expected count per chi-square bin. Cells below it are pooled (see
/// [`assert_matches_softmax`]) so the asymptotic chi-square approximation
/// stays valid on a peaked distribution whose tail cells are near-zero.
const MIN_EXPECTED_PER_BIN: f64 = 10.0;

/// Upper-tail standard normal quantile for p = 1e-6. Used with the
/// Wilson-Hilferty transform to derive the chi-square critical value, so a
/// correct implementation fails roughly once in a million runs per test.
const CRITICAL_Z: f64 = 4.7534;

fn sample_count() -> usize {
    std::env::var("MLXCEL_TEST_GUMBEL_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_SAMPLE_COUNT)
}

/// True when the Gumbel-max kernel can run here. CPU-only builds skip.
fn gpu_backend() -> bool {
    sampling_gumbel_available()
}

/// Wilson-Hilferty upper critical value of the chi-square distribution.
///
/// `(chi2 / df)^(1/3)` is approximately normal with mean `1 - 2/(9 df)` and
/// variance `2/(9 df)`, so the critical value at upper-tail probability `p` is
/// `df * (1 - t + z_p * sqrt(t))^3` with `t = 2/(9 df)`. The approximation is
/// accurate to well under a percent for the `df >= 10` this file produces, and
/// it keeps the threshold explicit in the test instead of hiding it in a
/// hand-copied table.
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

/// Draw `n` samples from one logits row through the Gumbel-max kernel and
/// return the per-token histogram.
///
/// The `[rows, vocab]` input is built once and reused across launches: each
/// launch draws a fresh Philox key from MLX's key sequence, so the rows of one
/// launch and the rows across launches are all independent draws. Reusing the
/// buffer keeps the test dominated by kernel time rather than by host-to-device
/// copies of the same logits.
fn histogram(logits: &[f32], temperature: f32, n: usize) -> Vec<u64> {
    let vocab = logits.len();
    let rows = ROWS_PER_LAUNCH
        .min(n.max(1))
        .min((MAX_ELEMENTS_PER_LAUNCH / vocab).max(1));
    let mut tiled = Vec::with_capacity(rows * vocab);
    for _ in 0..rows {
        tiled.extend_from_slice(logits);
    }
    let batched = from_slice_f32(&tiled, &[rows as i32, vocab as i32]);

    let mut counts = vec![0u64; vocab];
    let mut drawn = 0usize;
    while drawn < n {
        let tokens = gumbel_max_sample(&batched, temperature);
        for id in token_ids(&tokens) {
            if drawn >= n {
                break;
            }
            counts[id as usize] += 1;
            drawn += 1;
        }
    }
    counts
}

/// Read a `[B]` uint32 token-id array back to host.
fn token_ids(tokens: &MlxArray) -> Vec<u32> {
    array_to_raw_bytes(tokens)
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Chi-square goodness-of-fit of a Gumbel-max sample against exact softmax.
///
/// Cells whose logit is `-inf` have probability exactly zero; they are asserted
/// to have drawn zero samples and are then excluded, because a zero-expectation
/// cell has no chi-square term. The remaining cells are sorted by probability
/// and pooled greedily until each bin's expected count reaches
/// [`MIN_EXPECTED_PER_BIN`], which is the standard fix for the sparse tail of a
/// peaked distribution; the trailing remainder folds into the last bin. The
/// statistic is then compared against the Wilson-Hilferty critical value at
/// upper-tail p = 1e-6 (see [`chi_square_upper_critical`]), so a correct
/// implementation is expected to fail about once in a million runs.
fn assert_matches_softmax(label: &str, logits: &[f32], temperature: f32, n: usize) {
    let probs = softmax_reference(logits, temperature);
    let counts = histogram(logits, temperature, n);
    let total: u64 = counts.iter().sum();
    assert_eq!(total, n as u64, "{label}: drew {total} samples, wanted {n}");

    // Masked cells must never be selected.
    for (i, &p) in probs.iter().enumerate() {
        if p == 0.0 {
            assert_eq!(
                counts[i], 0,
                "{label}: masked token {i} was sampled {} times",
                counts[i]
            );
        }
    }

    let mut live: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
    live.sort_by(|&a, &b| {
        probs[a]
            .partial_cmp(&probs[b])
            .expect("finite probabilities")
    });

    let n_f = n as f64;
    let mut bins: Vec<(f64, f64)> = Vec::new(); // (expected, observed)
    let mut acc_expected = 0.0f64;
    let mut acc_observed = 0.0f64;
    for &i in &live {
        acc_expected += probs[i] * n_f;
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
        "{label}: chi-square {chi2:.2} exceeds the p=1e-6 critical value \
         {critical:.2} at df={df} over {n} samples ({} bins)",
        bins.len()
    );
}

// -- synthetic logit shapes --

/// Sharply peaked: one dominant token, geometric decay elsewhere.
fn peaked_logits(vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|i| {
            if i == vocab / 3 {
                6.0
            } else {
                -0.35 * i as f32
            }
        })
        .collect()
}

/// Perfectly flat: every token equally likely. The hardest shape for a
/// counter-based RNG, because any correlation between neighbouring element
/// streams shows up directly as a frequency imbalance.
fn flat_logits(vocab: usize) -> Vec<f32> {
    vec![0.0; vocab]
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

/// Peaked logits with a scattered `-inf` mask, matching what token bias and the
/// XTC pre-step leave in the logits before the sampler runs. The mask covers
/// the top token so the surviving support is genuinely reshaped.
fn masked_logits(vocab: usize) -> Vec<f32> {
    let mut logits = peaked_logits(vocab);
    for i in (0..vocab).step_by(3) {
        logits[i] = f32::NEG_INFINITY;
    }
    logits[vocab / 3] = f32::NEG_INFINITY;
    logits
}

// -- goodness of fit --

#[test]
fn gumbel_sample_matches_softmax_on_peaked_logits() {
    if !gpu_backend() {
        return;
    }
    assert_matches_softmax("peaked", &peaked_logits(64), 1.0, sample_count());
}

#[test]
fn gumbel_sample_matches_softmax_on_flat_logits() {
    if !gpu_backend() {
        return;
    }
    assert_matches_softmax("flat", &flat_logits(64), 1.0, sample_count());
}

#[test]
fn gumbel_sample_matches_softmax_on_bimodal_logits() {
    if !gpu_backend() {
        return;
    }
    assert_matches_softmax("bimodal", &bimodal_logits(64), 1.0, sample_count());
}

#[test]
fn gumbel_sample_matches_softmax_with_masked_entries() {
    if !gpu_backend() {
        return;
    }
    assert_matches_softmax("masked", &masked_logits(96), 1.0, sample_count());
}

#[test]
fn gumbel_sample_matches_softmax_over_a_large_vocabulary() {
    if !gpu_backend() {
        return;
    }
    // 4096 entries is 4 sweeps of the 1024-entry-per-threadgroup chunk, so this
    // is the shape that exercises the strided multi-iteration read loop rather
    // than a single-pass row.
    assert_matches_softmax("large-vocab", &bimodal_logits(4096), 1.0, sample_count());
}

#[test]
fn gumbel_noise_stays_inside_its_finite_range() {
    if !gpu_backend() {
        return;
    }
    // Guards the uniform's open-interval construction, which a chi-square test
    // is structurally blind to.
    //
    // The kernel builds `u` on a 2^-23 grid offset by half a step, so
    // `g = -log(-log(u))` is bounded by roughly [-2.9, 16.6]. A `u` of exactly
    // 1.0 would make `g` positive infinity and hand that element the argmax no
    // matter how small its logit is, and a `u` of exactly 0.0 would make it
    // negative infinity. Either is a rare per-element event, so it perturbs an
    // empirical frequency far below chi-square's resolution: at a 24-bit grid,
    // where `x + 0.5f` ties up to 2^24 for the top value and `u` does reach
    // 1.0, a 64-entry goodness-of-fit test over a million draws sees about four
    // stray samples and passes comfortably.
    //
    // Separating the logits by 100 makes the argmax unambiguous under any noise
    // inside that finite range, so a single wrong token is a hard failure
    // rather than a statistical wobble. Over a 4096-entry vocabulary a
    // million draws is 4.1e9 element draws, which at the 24-bit spelling would
    // hit the degenerate value about 244 times.
    let vocab = 4096usize;
    let winner = 1234usize;
    let mut logits = vec![0.0f32; vocab];
    logits[winner] = 100.0;

    let n = sample_count();
    let counts = histogram(&logits, 1.0, n);
    assert_eq!(
        counts[winner],
        n as u64,
        "{} of {n} draws escaped the dominant token; the Gumbel noise left its \
         finite range",
        n as u64 - counts[winner]
    );
}

// -- temperature --

#[test]
fn gumbel_sample_temperature_scaling_matches_reference() {
    if !gpu_backend() {
        return;
    }
    // Bimodal rather than peaked: dividing an already sharp peak by 0.5 leaves
    // a distribution with one cell at p > 0.99999, which has no testable
    // structure left. The bimodal shape keeps tens of cells above the pooling
    // floor at both temperatures, so the test actually constrains the scaling.
    let logits = bimodal_logits(64);
    let n = sample_count();
    // 0.5 sharpens, 1.5 flattens. Both are compared against the exact softmax
    // at that temperature, which is the same reference the categorical path
    // samples from after its own `x / temperature` scaling.
    assert_matches_softmax("temp-0.5", &logits, 0.5, n);
    assert_matches_softmax("temp-1.5", &logits, 1.5, n);
}

// -- masking --

#[test]
fn gumbel_sample_never_selects_masked_tokens() {
    if !gpu_backend() {
        return;
    }
    // A dense mask over a small support: only 4 of 64 tokens survive, and the
    // masked set includes what would otherwise be the argmax.
    let mut logits = vec![f32::NEG_INFINITY; 64];
    for (rank, &id) in [7usize, 19, 40, 63].iter().enumerate() {
        logits[id] = 2.0 - rank as f32;
    }

    for &temperature in &[0.5f32, 1.0, 1.5] {
        let counts = histogram(&logits, temperature, 131_072);
        for (id, &count) in counts.iter().enumerate() {
            if logits[id].is_infinite() {
                assert_eq!(
                    count, 0,
                    "masked token {id} sampled {count} times at temperature {temperature}"
                );
            }
        }
        for &id in &[7usize, 19, 40, 63] {
            assert!(
                counts[id] > 0,
                "surviving token {id} never sampled at temperature {temperature}"
            );
        }
    }
}

// -- determinism --

#[test]
fn gumbel_sample_is_reproducible_for_a_fixed_seed() {
    if !gpu_backend() {
        return;
    }
    let logits = bimodal_logits(512);
    let rows = 64usize;
    let mut tiled = Vec::with_capacity(rows * logits.len());
    for _ in 0..rows {
        tiled.extend_from_slice(&logits);
    }
    let batched = from_slice_f32(&tiled, &[rows as i32, logits.len() as i32]);

    // A stream, not one draw: successive calls consume successive keys, so this
    // also pins that the key sequence advances identically after reseeding.
    let run = || {
        random_seed(0x5EED_0900);
        let mut stream = Vec::new();
        for _ in 0..8 {
            stream.extend(token_ids(&gumbel_max_sample(&batched, 1.0)));
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

#[test]
fn gumbel_sample_index_does_not_depend_on_the_launch_split() {
    if !gpu_backend() {
        return;
    }
    // The split heuristic widens a small batch across cooperating threadgroups.
    // Row 0's Philox counters do not mention the split, so its sampled id must
    // be identical under both launch shapes for the same key.
    let vocab = 32_768usize;
    let logits = bimodal_logits(vocab);

    let narrow_splits = gumbel_sample_num_splits(1, vocab as i32);
    let wide_splits = gumbel_sample_num_splits(128, vocab as i32);
    assert_ne!(
        narrow_splits, wide_splits,
        "shapes chosen for this test do not actually produce different splits"
    );

    let single = from_slice_f32(&logits, &[1, vocab as i32]);
    let mut tiled = Vec::with_capacity(128 * vocab);
    for _ in 0..128 {
        tiled.extend_from_slice(&logits);
    }
    let batch = from_slice_f32(&tiled, &[128, vocab as i32]);

    random_seed(0xC0FFEE);
    let from_single = token_ids(&gumbel_max_sample(&single, 1.0));
    random_seed(0xC0FFEE);
    let from_batch = token_ids(&gumbel_max_sample(&batch, 1.0));

    assert_eq!(
        from_single[0], from_batch[0],
        "row 0 moved between a {narrow_splits}-split and a {wide_splits}-split launch"
    );
}

// -- routing --

#[test]
fn fused_sample_routes_the_no_filter_path_to_the_gumbel_kernel() {
    if !gpu_backend() {
        return;
    }
    let logits = bimodal_logits(1024);
    let rows = 32usize;
    let mut tiled = Vec::with_capacity(rows * logits.len());
    for _ in 0..rows {
        tiled.extend_from_slice(&logits);
    }
    let batched = from_slice_f32(&tiled, &[rows as i32, logits.len() as i32]);

    random_seed(0xA11CE);
    let routed = token_ids(&fused_sample(&batched, 1.0, 0, 1.0, 0.0));
    random_seed(0xA11CE);
    let direct = token_ids(&gumbel_max_sample(&batched, 1.0));
    assert_eq!(
        routed, direct,
        "fused_sample did not take the Gumbel-max path with no filter set"
    );

    random_seed(0xA11CE);
    let categorical = token_ids(&fused_sample_categorical(&batched, 1.0, 0, 1.0, 0.0));
    assert_ne!(
        routed, categorical,
        "Gumbel and categorical produced the same {rows}-token stream; \
         the routing gate is probably inert"
    );
}

#[test]
fn filtered_configs_stay_bit_identical_to_the_categorical_path() {
    if !gpu_backend() {
        return;
    }
    let logits = bimodal_logits(1024);
    let rows = 32usize;
    let mut tiled = Vec::with_capacity(rows * logits.len());
    for _ in 0..rows {
        tiled.extend_from_slice(&logits);
    }
    let batched = from_slice_f32(&tiled, &[rows as i32, logits.len() as i32]);

    // (top_k, top_p, min_p): every per-request filter, alone and combined.
    let filtered = [
        (40i32, 1.0f32, 0.0f32),
        (0, 0.9, 0.0),
        (0, 1.0, 0.1),
        (40, 0.9, 0.1),
    ];
    for (top_k, top_p, min_p) in filtered {
        random_seed(0xBEEF);
        let through_fused = token_ids(&fused_sample(&batched, 1.0, top_k, top_p, min_p));
        random_seed(0xBEEF);
        let reference = token_ids(&fused_sample_categorical(
            &batched, 1.0, top_k, top_p, min_p,
        ));
        assert_eq!(
            through_fused, reference,
            "top_k={top_k} top_p={top_p} min_p={min_p} diverged from the \
             categorical path; a filtered config must never reach the kernel"
        );
    }
}

#[test]
fn greedy_sampling_is_byte_identical_to_argmax() {
    if !gpu_backend() {
        return;
    }
    let logits = bimodal_logits(1024);
    let rows = 32usize;
    let mut tiled = Vec::with_capacity(rows * logits.len());
    for (r, _) in (0..rows).enumerate() {
        // Rotate each row so the argmax lands somewhere different per row.
        let shift = (r * 37) % logits.len();
        tiled.extend(
            logits[shift..]
                .iter()
                .chain(logits[..shift].iter())
                .copied(),
        );
    }
    let batched = from_slice_f32(&tiled, &[rows as i32, logits.len() as i32]);

    let greedy = token_ids(&fused_sample(&batched, 0.0, 0, 1.0, 0.0));
    let reference = token_ids(&argmax_last_axis(&batched));
    assert_eq!(greedy, reference, "temperature 0 diverged from argmax");

    // `top_k == 1` is the other greedy spelling and must behave the same.
    let top_one = token_ids(&fused_sample(&batched, 1.0, 1, 1.0, 0.0));
    assert_eq!(top_one, reference, "top_k=1 diverged from argmax");
}

#[test]
fn split_heuristic_stays_within_its_documented_range() {
    // Pure host arithmetic, so this runs on a CPU-only build too.
    for batch in [1i32, 2, 4, 8, 16, 64, 256] {
        for vocab in [64i32, 4096, 32_768, 65_536, 152_064] {
            let splits = gumbel_sample_num_splits(batch, vocab);
            assert!(
                (1..=64).contains(&splits),
                "batch={batch} vocab={vocab} produced {splits} splits"
            );
            assert!(
                (splits as u32).is_power_of_two(),
                "batch={batch} vocab={vocab} produced a non-power-of-two \
                 split count {splits}; the halving-tree reduction requires one"
            );
        }
    }
    // A batch that already fills the machine must not pay a second pass.
    assert_eq!(gumbel_sample_num_splits(256, 152_064), 1);
    // A tiny vocabulary has nothing to split.
    assert_eq!(gumbel_sample_num_splits(1, 64), 1);
    // A batch-1 long-vocabulary decode is the case the split exists for.
    assert!(gumbel_sample_num_splits(1, 152_064) > 1);
}
