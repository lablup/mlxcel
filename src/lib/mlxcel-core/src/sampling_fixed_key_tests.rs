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

//! Port-against-graph tests for the fused sampler kernels (issue #2064).
//!
//! `sampling_gumbel_tests.rs` and `sampling_rejection_tests.rs` hold each
//! kernel to the exact distribution it should draw from. This file adds the
//! two comparisons a new backend port needs on top of that, and runs on every
//! backend whose port table has an entry (Metal, CUDA, ROCm):
//!
//! 1. **Fixed key, deterministic.** Both kernels draw one Philox-4x32-10 key
//!    from MLX's default key sequence per call, and given that key their
//!    output is a pure function of the input. The key is recovered here by
//!    reseeding and calling `random::bits` exactly as the launcher does, the
//!    kernel's draw is recomputed on the host from that key, and for the
//!    Gumbel-max kernel the same noise is also pushed through the MLX graph
//!    (`argmax(logits / T + g)`). A port that changed the counter layout, the
//!    key words, the uniform mapping, the partition order or a dtype read
//!    disagrees on almost every row. Rows whose decision sits within rounding
//!    of a tie are skipped and counted, since `logf` and summation order are
//!    allowed to differ from the host in the last bits.
//! 2. **Statistical, against the graph sampler.** Where the kernel and the
//!    graph consume the RNG differently (`random::categorical`, the stock
//!    filter chain), a fixed key cannot line them up, so a two-sample
//!    chi-square homogeneity test compares the kernel's histogram with
//!    `fused_sample_categorical`'s, the explicit graph arm, on the same input.
//!
//! GPU-only: each test returns early when the kernel's support predicate says
//! this backend has no port, the convention of the two files above.

use super::*;
use crate::dtype;

/// Upper-tail standard normal quantile for p = 1e-6, as in the sibling files.
const CRITICAL_Z: f64 = 4.7534;

/// Minimum pooled count per bin for the two-sample test.
const MIN_POOLED_PER_BIN: u64 = 20;

/// Draws per arm for the two-sample tests.
const TWO_SAMPLE_DRAWS: usize = 400_000;

/// Smallest gap, between the winning and the runner-up Gumbel score, at which
/// a row counts as decided. GPU `logf` and the host's f64 `ln` may differ in
/// the last bits of a score of order ten, far below this.
const GUMBEL_DECIDED_GAP: f64 = 1e-4;

/// Smallest distance, relative to the proposal mass, between the rejection
/// kernel's draw target and the nearest cumulative boundary at which a row
/// counts as decided. The kernel sums in f32, the host in f64.
const REJECTION_DECIDED_GAP: f64 = 1e-5;

// -- host reference of the kernels' RNG --

/// Philox-4x32-10 exactly as both kernel sources spell it.
fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = counter;
    let [mut k0, mut k1] = key;
    for _ in 0..10 {
        let p0 = u64::from(0xD251_1F53u32) * u64::from(c0);
        let p1 = u64::from(0xCD9E_8D57u32) * u64::from(c2);
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        let n0 = hi1 ^ c1 ^ k0;
        let n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
        k0 = k0.wrapping_add(0x9E37_79B9);
        k1 = k1.wrapping_add(0xBB67_AE85);
    }
    [c0, c1, c2, c3]
}

/// The kernels' uniform on the open interval (0, 1): the top 23 bits plus a
/// half step, computed in f32 as the kernels do (both steps are exact).
fn uniform_open(word: u32) -> f32 {
    ((word >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

/// The key the next kernel launch will draw after `random_seed(seed)`.
///
/// The launchers call `random::bits({2}, 4)` with no key, which takes the next
/// key from the default sequence; this makes the same call, so reseeding with
/// the same value afterwards hands the launch the same key.
fn key_after_seed(seed: u64) -> [u32; 2] {
    random_seed(seed);
    // SAFETY: a null key pointer asks for MLX's default key sequence; the
    // shape slice outlives the call.
    let bits = unsafe { random_bits(&[2], 4, std::ptr::null()) };
    let words = u32_values(&bits);
    assert_eq!(words.len(), 2, "random_bits returned {} words", words.len());
    [words[0], words[1]]
}

// -- small helpers --

fn u32_values(arr: &MlxArray) -> Vec<u32> {
    array_to_raw_bytes(arr)
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_values(arr: &MlxArray) -> Vec<f32> {
    let as_f32 = astype(arr, dtype::FLOAT32);
    array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic pseudo-random logits in `[-span, span]`, distinct per row,
/// with a handful of raised entries per row so a filter keeps a support of
/// tens of tokens spread over the whole row.
fn row_logits(rows: usize, vocab: usize, span: f32, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let mut out = Vec::with_capacity(rows * vocab);
    for _ in 0..rows {
        let start = out.len();
        for _ in 0..vocab {
            out.push((next() * 2.0 - 1.0) * span);
        }
        for _ in 0..24 {
            let i = (next() * vocab as f32) as usize % vocab;
            out[start + i] = span + 2.0 + next() * 3.0;
        }
    }
    out
}

/// Two-sample chi-square homogeneity statistic for equal draw counts, with
/// cells pooled (largest first) until each bin holds [`MIN_POOLED_PER_BIN`]
/// draws across both arms. Returns `(statistic, degrees of freedom)`.
fn two_sample_chi_square(a: &[u64], b: &[u64]) -> (f64, f64) {
    let mut cells: Vec<(u64, u64)> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x, y))
        .filter(|&(x, y)| x + y > 0)
        .collect();
    cells.sort_by_key(|&(x, y)| std::cmp::Reverse(x + y));
    let mut bins: Vec<(u64, u64)> = Vec::new();
    let mut acc = (0u64, 0u64);
    for (x, y) in cells {
        acc = (acc.0 + x, acc.1 + y);
        if acc.0 + acc.1 >= MIN_POOLED_PER_BIN {
            bins.push(acc);
            acc = (0, 0);
        }
    }
    if acc.0 + acc.1 > 0 {
        match bins.last_mut() {
            Some(last) => *last = (last.0 + acc.0, last.1 + acc.1),
            None => bins.push(acc),
        }
    }
    let stat = bins
        .iter()
        .map(|&(x, y)| {
            let d = x as f64 - y as f64;
            d * d / (x + y) as f64
        })
        .sum();
    (stat, (bins.len().max(2) - 1) as f64)
}

fn chi_square_upper_critical(df: f64, z: f64) -> f64 {
    let t = 2.0 / (9.0 * df);
    let x = 1.0 - t + z * t.sqrt();
    df * x * x * x
}

fn histogram_of(vocab: usize, n: usize, mut draw: impl FnMut() -> Vec<u32>) -> Vec<u64> {
    let mut counts = vec![0u64; vocab];
    let mut drawn = 0usize;
    while drawn < n {
        for id in draw() {
            if drawn >= n {
                break;
            }
            assert!(
                (id as usize) < vocab,
                "sampler returned id {id} for vocab {vocab}"
            );
            counts[id as usize] += 1;
            drawn += 1;
        }
    }
    counts
}

fn tiled(row: &[f32], rows: usize) -> UniquePtr<MlxArray> {
    let mut data = Vec::with_capacity(rows * row.len());
    for _ in 0..rows {
        data.extend_from_slice(row);
    }
    from_slice_f32(&data, &[rows as i32, row.len() as i32])
}

// -- 1. fixed key --

#[test]
fn gumbel_kernel_matches_the_keyed_graph_argmax() {
    if !sampling_gumbel_available() {
        return;
    }
    // 5003 is not a multiple of four, so the last Philox block is partial, and
    // spans several 1024-entry sweeps; 48 rows take two splits per row.
    let rows = 48usize;
    let vocab = 5003usize;
    let cases = [
        (dtype::FLOAT32, 1.0f32),
        (dtype::FLOAT32, 0.7),
        (dtype::BFLOAT16, 1.3),
        (dtype::FLOAT16, 0.9),
    ];
    for (case, &(dt, temperature)) in cases.iter().enumerate() {
        let host = row_logits(rows, vocab, 4.0, 0x2064 + case as u64);
        let mut logits = from_slice_f32(&host, &[rows as i32, vocab as i32]);
        if dt != dtype::FLOAT32 {
            logits = astype(&logits, dt);
        }
        // The values the kernel reads, after any rounding to `dt`.
        let seen = f32_values(&logits);

        let seed = 0x5EED_2064_0000 + case as u64;
        let key = key_after_seed(seed);
        random_seed(seed);
        let kernel_ids = u32_values(
            &gumbel_max_sample(&logits, temperature)
                .expect("sampling_gumbel_available() reported a port"),
        );

        // Host scores in f64 from the kernel's own key, and the same noise
        // pushed through the MLX graph.
        let mut noise = vec![0f32; rows * vocab];
        let mut expected = vec![0usize; rows];
        let mut decided = vec![false; rows];
        for row in 0..rows {
            let mut best = (f64::NEG_INFINITY, 0usize);
            let mut second = f64::NEG_INFINITY;
            for block in 0..vocab.div_ceil(4) {
                let words = philox4x32_10([block as u32, 0, row as u32, 0], key);
                for (j, &word) in words.iter().enumerate() {
                    let idx = block * 4 + j;
                    if idx >= vocab {
                        break;
                    }
                    let u = f64::from(uniform_open(word));
                    let g = -(-u.ln()).ln();
                    noise[row * vocab + idx] = g as f32;
                    let scaled = f64::from(seen[row * vocab + idx] / temperature);
                    let score = scaled + g;
                    if score > best.0 {
                        second = best.0;
                        best = (score, idx);
                    } else if score > second {
                        second = score;
                    }
                }
            }
            expected[row] = best.1;
            decided[row] = best.0 - second >= GUMBEL_DECIDED_GAP;
        }
        let noise = from_slice_f32(&noise, &[rows as i32, vocab as i32]);
        let t = from_slice_f32(&[temperature], &[1]);
        let scaled = divide(&astype(&logits, dtype::FLOAT32), &t);
        let graph_ids = u32_values(&argmax(&add(&scaled, &noise), -1, false));

        let label = format!("dtype code {dt}, T={temperature}");
        let mut checked = 0usize;
        for row in 0..rows {
            if !decided[row] {
                continue;
            }
            checked += 1;
            assert_eq!(
                kernel_ids[row] as usize, expected[row],
                "{label}: row {row} kernel drew {} but the keyed host reference drew {}",
                kernel_ids[row], expected[row]
            );
            assert_eq!(
                graph_ids[row] as usize, expected[row],
                "{label}: row {row} graph argmax {} disagrees with the host reference {}",
                graph_ids[row], expected[row]
            );
        }
        assert!(
            checked * 10 >= rows * 9,
            "{label}: only {checked} of {rows} rows were decided; the test lost its power"
        );
    }
}

#[test]
fn rejection_kernel_draws_the_keyed_token_under_min_p() {
    // The probe does not park flags in the deferred ring, but the shared guard
    // keeps this test from interleaving with one that inspects it.
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !sampling_rejection_available() {
        return;
    }
    // min-p alone resolves its whole threshold before the first draw, so the
    // first round always accepts and the sampled token is fully determined by
    // the round-0 Philox word and the kernel's partition order: thread `t` of
    // the row's block owns entries `t, t + tg, ...`, and the block scans
    // threads in order. That is what the host reproduces below.
    let tg = sampling_rejection_threadgroup_size() as usize;
    let rows = 40usize;
    let vocab = 3001usize;
    let min_p = 0.05f32;
    for (case, &temperature) in [1.0f32, 0.7].iter().enumerate() {
        let host = row_logits(rows, vocab, 3.0, 0x0901 + case as u64);
        let logits = from_slice_f32(&host, &[rows as i32, vocab as i32]);
        // The kernel's two inputs, built with the bridge's own ops: the filter
        // row is the untempered softmax, the draw row the tempered one.
        let filter = f32_values(&softmax(&logits, -1));
        let draw = if temperature == 1.0 {
            filter.clone()
        } else {
            let t = from_slice_f32(&[temperature], &[1]);
            f32_values(&softmax(&divide(&logits, &t), -1))
        };

        let seed = 0x5EED_2064_0100 + case as u64;
        let key = key_after_seed(seed);
        random_seed(seed);
        let stacked = sampling_rejection_probe(&logits, temperature, 0, 1.0, min_p, 32);
        let flat = u32_values(&stacked);
        let (ids, ok, used) = (&flat[..rows], &flat[rows..2 * rows], &flat[2 * rows..]);

        let label = format!("min_p={min_p}, T={temperature}");
        let mut checked = 0usize;
        for row in 0..rows {
            assert_eq!(ok[row], 1, "{label}: row {row} did not converge");
            assert_eq!(used[row], 1, "{label}: row {row} took {} rounds", used[row]);

            let f = &filter[row * vocab..(row + 1) * vocab];
            let d = &draw[row * vocab..(row + 1) * vocab];
            let p_max = f.iter().cloned().fold(-1.0f32, f32::max);
            let low = f32::from_bits((min_p * p_max).to_bits() - 1);

            let order: Vec<usize> = (0..tg)
                .flat_map(|t| (t..vocab).step_by(tg))
                .filter(|&i| f[i] > low)
                .collect();
            let mass: f64 = order.iter().map(|&i| f64::from(d[i])).sum();
            let u = uniform_open(philox4x32_10([0, 0, row as u32, 0], key)[0]);
            let target = f64::from(u) * mass;

            let mut run = 0.0f64;
            let mut pick = None;
            let mut gap = f64::INFINITY;
            for &i in &order {
                gap = gap.min((run - target).abs());
                run += f64::from(d[i]);
                if pick.is_none() && run > target {
                    pick = Some(i);
                }
            }
            gap = gap.min((run - target).abs());
            if gap < REJECTION_DECIDED_GAP * mass {
                continue;
            }
            checked += 1;
            let expected = pick.expect("target lies below the proposal mass");
            assert_eq!(
                ids[row] as usize, expected,
                "{label}: row {row} kernel drew {} but the keyed host reference drew {expected}",
                ids[row]
            );
        }
        assert!(
            checked * 10 >= rows * 9,
            "{label}: only {checked} of {rows} rows were decided; the test lost its power"
        );
    }
}

#[test]
fn rejection_kernel_draws_stay_inside_the_filtered_support() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !sampling_rejection_available() {
        return;
    }
    // Rounds past the first (the pivot and bisection logic) are not
    // reproducible on the host bit for bit, because the kernel's f32 sums may
    // differ in their last bits. What is deterministic whatever the round
    // count: every row converges and every drawn token is inside the filter's
    // support, computed here from the same probabilities the kernel reads. The
    // host side is a slight superset (a relative 1e-5 slack on the top-p
    // cutoff), so rounding at the boundary cannot fail a correct kernel.
    let rows = 40usize;
    let vocab = 3001usize;
    let host = row_logits(rows, vocab, 3.0, 0x0902);
    let logits = from_slice_f32(&host, &[rows as i32, vocab as i32]);
    let filter = f32_values(&softmax(&logits, -1));
    let mut multi_round = 0usize;
    for (top_k, top_p) in [(0i32, 0.9f32), (40, 1.0), (0, 0.5)] {
        for launch in 0..4u64 {
            random_seed(0x5EED_2064_0200 + launch);
            let stacked = sampling_rejection_probe(&logits, 1.0, top_k, top_p, 0.0, 32);
            let flat = u32_values(&stacked);
            let (ids, ok, used) = (&flat[..rows], &flat[rows..2 * rows], &flat[2 * rows..]);
            for row in 0..rows {
                let label = format!("top_k={top_k} top_p={top_p} launch {launch} row {row}");
                assert_eq!(ok[row], 1, "{label}: did not converge");
                multi_round += usize::from(used[row] > 1);
                let f = &filter[row * vocab..(row + 1) * vocab];
                let id = ids[row] as usize;
                assert!(id < vocab, "{label}: id {id} out of range");
                let p = f[id];
                if top_k > 0 {
                    let above = f.iter().filter(|&&q| q > p).count();
                    assert!(
                        above < top_k as usize,
                        "{label}: id {id} has {above} tokens above it"
                    );
                }
                if top_p < 1.0 {
                    let total: f64 = f.iter().map(|&q| f64::from(q)).sum();
                    let exclusive: f64 = f.iter().filter(|&&q| q > p).map(|&q| f64::from(q)).sum();
                    assert!(
                        exclusive <= f64::from(top_p) * total * (1.0 + 1e-5),
                        "{label}: id {id} starts at mass {exclusive}, past top_p {top_p}"
                    );
                }
            }
        }
    }
    assert!(
        multi_round > 0,
        "no row needed a second round; the bisection went untested"
    );
}

// -- 2. statistical, against the graph sampler --

#[test]
fn gumbel_kernel_and_graph_categorical_draw_one_distribution() {
    if !sampling_gumbel_available() {
        return;
    }
    let vocab = 2048usize;
    let row = row_logits(1, vocab, 2.0, 0x900);
    let rows = 2048usize;
    let batched = tiled(&row, rows);
    let temperature = 0.8f32;

    random_seed(0x2064_0900);
    let kernel = histogram_of(vocab, TWO_SAMPLE_DRAWS, || {
        u32_values(
            &gumbel_max_sample(&batched, temperature)
                .expect("sampling_gumbel_available() reported a port"),
        )
    });
    let graph = histogram_of(vocab, TWO_SAMPLE_DRAWS, || {
        u32_values(&fused_sample_categorical(
            &batched,
            temperature,
            0,
            1.0,
            0.0,
        ))
    });

    let (stat, df) = two_sample_chi_square(&kernel, &graph);
    let critical = chi_square_upper_critical(df, CRITICAL_Z);
    assert!(
        stat <= critical,
        "Gumbel-max kernel and random::categorical disagree: chi2 {stat:.1} > {critical:.1} (df {df})"
    );
}

#[test]
fn rejection_kernel_and_graph_chain_draw_one_distribution() {
    let _dispatch = crate::sampling_dispatch::dispatch_test_guard();
    if !sampling_rejection_available() {
        return;
    }
    let vocab = 4096usize;
    let row = row_logits(1, vocab, 2.0, 0x901);
    let rows = 1024usize;
    let batched = tiled(&row, rows);
    let temperature = 0.8f32;
    // (top_k, top_p, min_p): top-p alone (the routed configuration) and
    // top-p with min-p. Both are exact in the kernel, unlike top-k with top-p
    // (see `sampling_rejection_tests.rs`), so the two arms must agree.
    for (top_k, top_p, min_p) in [(0i32, 0.9f32, 0.0f32), (0, 0.95, 0.02)] {
        random_seed(0x2064_0901);
        let kernel = histogram_of(vocab, TWO_SAMPLE_DRAWS, || {
            let stacked = sampling_rejection_probe(&batched, temperature, top_k, top_p, min_p, 32);
            let flat = u32_values(&stacked);
            assert!(
                flat[rows..2 * rows].iter().all(|&ok| ok == 1),
                "a row exhausted the round cap"
            );
            flat[..rows].to_vec()
        });
        let graph = histogram_of(vocab, TWO_SAMPLE_DRAWS, || {
            u32_values(&fused_sample_categorical(
                &batched,
                temperature,
                top_k,
                top_p,
                min_p,
            ))
        });

        let (stat, df) = two_sample_chi_square(&kernel, &graph);
        let critical = chi_square_upper_critical(df, CRITICAL_Z);
        assert!(
            stat <= critical,
            "top_p={top_p} min_p={min_p}: rejection kernel and the stock chain disagree: \
             chi2 {stat:.1} > {critical:.1} (df {df})"
        );
    }
}
