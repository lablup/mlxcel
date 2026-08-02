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

//! Tests for Stage 2 split-KV absorbed decode (issue #907).
//!
//! Two things are under test and they fail for different reasons:
//!
//! 1. The **decomposition**: splitting the latent range and merging the partial
//!    states must reproduce the unsplit answer for any chunk length.
//! 2. The **reuse of issue #898's merge kernel**: its contract is honoured
//!    rather than approximated. The load-bearing clause is that `lse_in` is in
//!    **log2** units, and a natural-log LSE does not fail loudly, it produces a
//!    plausible wrong weighted average. `merge_rejects_natural_log_lse_units`
//!    is the negative control that pins it.

use super::*;
use crate::dtype;
use crate::mla::absorb::MlaAbsorbedProjections;
use crate::mla::testkit::{MlaFixture, TINY, max_rel_error, serial, to_vec_f32};

const F32_TOL: f32 = 3e-5;

fn run_split(fx: &MlaFixture, chunk_len: i32) -> Vec<f32> {
    let proj =
        MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dtype::FLOAT32), fx.geometry).unwrap();
    let plan = MlaSplitPlan::with_chunk_len(fx.batch, fx.kv_len as i32, chunk_len);
    let out = absorbed_decode_split_kv(
        &fx.q_nope_array(dtype::FLOAT32),
        &fx.q_pe_array(dtype::FLOAT32),
        &fx.ckv_array(dtype::FLOAT32),
        &fx.kpe_array(dtype::FLOAT32),
        &proj,
        fx.scale,
        &plan,
    )
    .unwrap();
    to_vec_f32(&out)
}

#[test]
fn split_kv_matches_the_decompressed_reference_at_every_chunk_length() {
    let _guard = serial();
    // 37 rows: 8 gives a ragged last chunk of 5, 16 gives 5, 37 gives exactly
    // one chunk (the no-merge case), 64 exceeds the range entirely.
    let fx = MlaFixture::new(TINY, 2, 1, 37, 0x5911);
    let want = fx.decompressed_reference(false);
    for chunk_len in [8, 16, 37, 64] {
        let got = run_split(&fx, chunk_len);
        let err = max_rel_error(&got, &want);
        assert!(err < F32_TOL, "chunk_len {chunk_len} drifted by {err}");
    }
}

#[test]
fn split_kv_agrees_with_the_unsplit_stage_1_path() {
    let _guard = serial();
    // The regrouping property `paged_v2::launch_tests::merge_is_associative_across_regroupings`
    // pins for the kernel, restated at the MLA caller: how the range was cut
    // must not change the answer.
    let fx = MlaFixture::new(TINY, 3, 1, 40, 0x9E77);
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dtype::FLOAT32), TINY).unwrap();
    let unsplit = to_vec_f32(&crate::mla::decode::absorbed_decode(
        &fx.q_nope_array(dtype::FLOAT32),
        &fx.q_pe_array(dtype::FLOAT32),
        &fx.ckv_array(dtype::FLOAT32),
        &fx.kpe_array(dtype::FLOAT32),
        &proj,
        fx.scale,
        None,
    ));
    for chunk_len in [5, 10, 13] {
        let got = run_split(&fx, chunk_len);
        let err = max_rel_error(&got, &unsplit);
        assert!(err < F32_TOL, "chunk_len {chunk_len} drifted by {err}");
    }
}

#[test]
fn split_kv_records_its_own_path() {
    let _guard = serial();
    let _ = crate::mla::stats::take();
    let fx = MlaFixture::new(TINY, 1, 1, 20, 0x4242);
    run_split(&fx, 8);
    let counts = crate::mla::stats::take();
    assert_eq!(counts.absorbed_split_kv, 1, "{}", counts.summary());
    assert_eq!(counts.absorbed_composed, 0, "{}", counts.summary());
}

#[test]
fn split_kv_declines_a_multi_token_step() {
    let _guard = serial();
    let fx = MlaFixture::new(TINY, 1, 3, 20, 0x1234);
    let proj = MlaAbsorbedProjections::from_dense(&fx.kv_b_array(dtype::FLOAT32), TINY).unwrap();
    let plan = MlaSplitPlan::with_chunk_len(1, 20, 8);
    let err = match absorbed_decode_split_kv(
        &fx.q_nope_array(dtype::FLOAT32),
        &fx.q_pe_array(dtype::FLOAT32),
        &fx.ckv_array(dtype::FLOAT32),
        &fx.kpe_array(dtype::FLOAT32),
        &proj,
        fx.scale,
        &plan,
    ) {
        Ok(_) => panic!("split-KV accepted a 3-token step; it has no per-chunk causal mask"),
        Err(e) => e,
    };
    assert!(err.contains("single-token"), "{err}");
}

/// Directly exercise issue #898's merge kernel with MLA-shaped partials, and
/// show that the log2 unit conversion is the thing making it correct.
#[test]
fn merge_rejects_natural_log_lse_units() {
    let _guard = serial();
    const HEADS: usize = 3;
    const DIM: usize = 5;
    const A: usize = 4;
    const B: usize = 6;

    let mut rng = crate::mla::testkit::Rng::new(0xFEED);
    // Scores deliberately biased so the two chunks have very different
    // denominators; with equal denominators a wrong weighting would cancel.
    let scores_a: Vec<f64> = (0..HEADS * A)
        .map(|_| rng.next_f32() as f64 * 2.0)
        .collect();
    let scores_b: Vec<f64> = (0..HEADS * B)
        .map(|_| rng.next_f32() as f64 * 2.0 + 3.0)
        .collect();
    let vals_a: Vec<f64> = (0..A * DIM).map(|_| rng.next_f32() as f64).collect();
    let vals_b: Vec<f64> = (0..B * DIM).map(|_| rng.next_f32() as f64).collect();

    // Host truth: one softmax over the concatenated range.
    let mut truth = vec![0.0f32; HEADS * DIM];
    for h in 0..HEADS {
        let all: Vec<f64> = scores_a[h * A..(h + 1) * A]
            .iter()
            .chain(&scores_b[h * B..(h + 1) * B])
            .copied()
            .collect();
        let m = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = all.iter().map(|s| (s - m).exp()).collect();
        let denom: f64 = exps.iter().sum();
        for d in 0..DIM {
            let acc: f64 = (0..A).map(|t| exps[t] * vals_a[t * DIM + d]).sum::<f64>()
                + (0..B)
                    .map(|t| exps[A + t] * vals_b[t * DIM + d])
                    .sum::<f64>();
            truth[h * DIM + d] = (acc / denom) as f32;
        }
    }

    // Two partials, each normalized by its own denominator, plus its LSE.
    let mut v_in = vec![0.0f32; 2 * HEADS * DIM];
    let mut lse_ln = vec![0.0f32; 2 * HEADS];
    for (chunk, (scores, vals, n)) in [(&scores_a, &vals_a, A), (&scores_b, &vals_b, B)]
        .into_iter()
        .enumerate()
    {
        for h in 0..HEADS {
            let s = &scores[h * n..(h + 1) * n];
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let denom: f64 = exps.iter().sum();
            for d in 0..DIM {
                let acc: f64 = (0..n).map(|t| exps[t] * vals[t * DIM + d]).sum();
                v_in[(chunk * HEADS + h) * DIM + d] = (acc / denom) as f32;
            }
            lse_ln[chunk * HEADS + h] = (m + denom.ln()) as f32;
        }
    }

    let v_arr = crate::ffi::from_slice_f32(&v_in, &[2, HEADS as i32, DIM as i32]);
    let indptr = crate::ffi::from_slice_i32(&[0, 2], &[2]);

    let merge = |lse: &[f32]| {
        let lse_arr = crate::ffi::from_slice_f32(lse, &[2, HEADS as i32]);
        let mut out_v = cxx::UniquePtr::null();
        let mut out_lse = cxx::UniquePtr::null();
        crate::ffi::paged_attention_merge_states(
            &v_arr,
            &lse_arr,
            &indptr,
            &mut out_v,
            &mut out_lse,
        );
        to_vec_f32(&out_v)
    };

    let lse_log2: Vec<f32> = lse_ln.iter().map(|x| x * LOG2_E).collect();
    let good = merge(&lse_log2);
    assert!(
        max_rel_error(&good, &truth) < 1e-5,
        "log2 LSE merge drifted by {}",
        max_rel_error(&good, &truth)
    );

    // The negative control. A natural-log LSE is silently accepted by the
    // kernel and produces a different, plausible answer, which is exactly why
    // the conversion is a named constant with a comment rather than an inline
    // multiply.
    let bad = merge(&lse_ln);
    assert!(
        max_rel_error(&bad, &truth) > 1e-3,
        "natural-log LSE happened to agree, so this control proves nothing"
    );
}

#[test]
fn plan_covers_the_range_and_groups_request_major() {
    let plan = MlaSplitPlan::with_chunk_len(3, 37, 16);
    assert_eq!(plan.num_chunks, 3);
    assert_eq!(plan.num_partials(), 9);
    assert!(plan.needs_merge());
    assert_eq!(plan.o_indptr(), vec![0, 3, 6, 9]);
    assert_eq!(plan.chunk_range(0), (0, 16));
    assert_eq!(plan.chunk_range(1), (16, 32));
    assert_eq!(plan.chunk_range(2), (32, 37));
    plan.validate().unwrap();
}

#[test]
fn plan_with_one_chunk_skips_the_merge() {
    let plan = MlaSplitPlan::with_chunk_len(2, 100, 128);
    assert_eq!(plan.num_chunks, 1);
    assert!(!plan.needs_merge());
    assert_eq!(plan.chunk_range(0), (0, 100));
    plan.validate().unwrap();
}

#[test]
fn plan_rejects_an_empty_range() {
    let plan = MlaSplitPlan::with_chunk_len(1, 0, 16);
    let err = plan.validate().unwrap_err();
    assert!(err.contains("no visible latent rows"), "{err}");
}

#[test]
fn heuristic_stops_splitting_once_the_batch_fills_the_device() {
    // 8 requests * 128 heads already exceeds any plausible CTA target, so the
    // split must not fragment the range for nothing.
    let saturated = MlaSplitPlan::heuristic(8, 128, 32768, 512);
    assert_eq!(saturated.num_chunks, 1, "{saturated:?}");

    // Batch 1, 16 heads, 32K context: the shape a single-CTA-per-request decode
    // cannot fill, so the split must actually cut.
    let starved = MlaSplitPlan::heuristic(1, 16, 32768, 512);
    assert!(starved.num_chunks > 1, "{starved:?}");
    assert!(starved.chunk_len >= MIN_CHUNK_LEN, "{starved:?}");
    starved.validate().unwrap();

    // Never below the floor, however starved.
    let tiny = MlaSplitPlan::heuristic(1, 1, 64, 4096);
    assert!(tiny.chunk_len >= MIN_CHUNK_LEN, "{tiny:?}");
    assert_eq!(tiny.num_chunks, 1, "{tiny:?}");
}
