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

//! A `top_k` at or above the vocabulary is "no top-k" on every sampler entry
//! point (#2247).
//!
//! The stock chain's top-k filter calls `argpartition(-x, top_k - 1)`, which
//! throws for `top_k > vocab`. `fused_sample`, `fused_sample_xtc`,
//! `fused_sample_categorical` and the two `fused_sample_probs` variants are
//! not `Result` bridge functions, so that throw reached cxx's noexcept
//! boundary and aborted the process. A server forwards a request's `top_k`
//! unchanged, so one request could take the server down. Before the fix these
//! tests abort the test binary instead of failing.
//!
//! Each case compares against the same call with `top_k = 0`, the explicit
//! "no top-k" spelling, and checks that the drawn token is a valid id. The
//! configurations with top-p active also cover the rejection-kernel routing,
//! where top-k at or above the vocabulary already counted as inactive.
//! Runs on whatever device is the default, CPU or GPU.

use crate::{dtype, ffi};

const VOCAB: i32 = 64;
const TEMPERATURE: f32 = 0.7;

/// `top_k` values at or above the vocabulary: equal (a valid partition that
/// filters nothing), one above, and far above, up to `i32::MAX`.
const AT_OR_ABOVE_VOCAB: [i32; 4] = [VOCAB, VOCAB + 1, 1_000_000, i32::MAX];

fn lcg_logits(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut row = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((state >> 33) as f64) / f64::from(u32::MAX >> 1);
        row.push((unit * 12.0 - 6.0) as f32);
    }
    row
}

fn probs_row(arr: &ffi::MlxArray) -> Vec<f32> {
    let arr = ffi::astype(arr, dtype::FLOAT32);
    ffi::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_same_distribution(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6,
            "{what}: token {i} reported {g}, the top_k = 0 distribution has {w}"
        );
    }
}

fn assert_valid_token(token: &ffi::MlxArray, what: &str) {
    let id = ffi::item_i32(&ffi::astype(token, dtype::INT32));
    assert!(
        (0..VOCAB).contains(&id),
        "{what}: drew token {id} outside 0..{VOCAB}"
    );
}

#[test]
fn top_k_at_or_above_vocab_is_no_top_k() {
    let row = lcg_logits(0x2247, VOCAB as usize);
    let logits = ffi::from_slice_f32(&row, &[1, VOCAB]);

    // top-p inactive (1.0) is the issue's default-path case: the rejection
    // kernel is not routed, so the stock chain runs the top-k filter. 0.9
    // adds the routed configuration.
    for top_p in [1.0f32, 0.9] {
        let reference = probs_row(&ffi::fused_sample_probs(
            &logits,
            TEMPERATURE,
            0,
            top_p,
            0.0,
        ));
        for top_k in AT_OR_ABOVE_VOCAB {
            let what = format!("top_k={top_k} top_p={top_p}");
            let probs = probs_row(&ffi::fused_sample_probs(
                &logits,
                TEMPERATURE,
                top_k,
                top_p,
                0.0,
            ));
            assert_same_distribution(&probs, &reference, &format!("fused_sample_probs {what}"));

            let token = ffi::fused_sample(&logits, TEMPERATURE, top_k, top_p, 0.0);
            assert_valid_token(&token, &format!("fused_sample {what}"));

            let token = ffi::fused_sample_categorical(&logits, TEMPERATURE, top_k, top_p, 0.0);
            assert_valid_token(&token, &format!("fused_sample_categorical {what}"));
        }
    }
}

#[test]
fn top_k_at_or_above_vocab_is_no_top_k_with_xtc_active() {
    let row = lcg_logits(0x0022_470C, VOCAB as usize);
    let logits = ffi::from_slice_f32(&row, &[1, VOCAB]);
    // A gate of 0 is below any positive probability, so XTC fires on this
    // step; XTC-active configurations always take the stock chain.
    let gate = ffi::from_slice_f32(&[0.0], &[1]);
    let (threshold, probability) = (0.01f32, 1.0f32);

    for top_p in [1.0f32, 0.9] {
        let reference = probs_row(&ffi::fused_sample_probs_xtc(
            &logits,
            TEMPERATURE,
            0,
            top_p,
            0.0,
            threshold,
            probability,
            &[],
            &gate,
        ));
        for top_k in AT_OR_ABOVE_VOCAB {
            let what = format!("top_k={top_k} top_p={top_p}");
            let probs = probs_row(&ffi::fused_sample_probs_xtc(
                &logits,
                TEMPERATURE,
                top_k,
                top_p,
                0.0,
                threshold,
                probability,
                &[],
                &gate,
            ));
            assert_same_distribution(
                &probs,
                &reference,
                &format!("fused_sample_probs_xtc {what}"),
            );

            let token = ffi::fused_sample_xtc(
                &logits,
                TEMPERATURE,
                top_k,
                top_p,
                0.0,
                threshold,
                probability,
                &[],
                &gate,
            );
            assert_valid_token(&token, &format!("fused_sample_xtc {what}"));
        }
    }
}
