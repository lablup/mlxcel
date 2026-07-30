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

//! Autotuner outcome gate for the v1 paged-decode `NumSplits` op (issue #906).
//!
//! The mandatory guard from the issue: on the tuning matrix, the tactic the
//! autotuner selects must never be slower than the configuration the runtime
//! would have used without it. This runs a reduced matrix (two batches, two
//! context lengths) of real GPU launches and asserts that property directly on
//! the measured medians, plus the invariants that make the selection safe to
//! apply: the default is always among the candidates, the selection is always
//! feasible, and every candidate is inside the C++ launcher's budget ceiling.
//!
//! It deliberately does not touch the tactic cache. Persistence and
//! invalidation are covered by the `mlxcel-core` unit tests; what needs a GPU
//! is the measurement itself.
//!
//! Skips on a build with no GPU backend, where the fused kernel cannot launch.

use mlxcel_core::autotune::ops::paged_decode_splits::{
    DecodeShape, PagedDecodeSplitsOp, num_splits_cap, split_candidates,
};
use mlxcel_core::autotune::{ProfileConfig, TunableOp, profile};
use mlxcel_core::{MlxArray, UniquePtr, eval, from_slice_i32, synchronize_default, zeros};

const F16: i32 = 9;
const F32: i32 = 10;

/// Reduced tuning matrix: enough shapes to cover both the short-context and
/// long-context regimes without turning a unit-test run into a benchmark.
const BATCHES: [usize; 2] = [1, 4];
const CONTEXTS: [usize; 2] = [512, 8192];
const BLOCK_SIZE: usize = 32;
const HEAD_DIM: i32 = 128;
const Q_HEADS: i32 = 8;
const KV_HEADS: i32 = 2;

fn gpu_available() -> bool {
    mlxcel_core::metal_is_available() || mlxcel_core::cuda_is_available()
}

struct Workload {
    q: UniquePtr<MlxArray>,
    pool_k: UniquePtr<MlxArray>,
    pool_v: UniquePtr<MlxArray>,
    rows: UniquePtr<MlxArray>,
    row_offsets: UniquePtr<MlxArray>,
    logical_starts: UniquePtr<MlxArray>,
    visible_lens: UniquePtr<MlxArray>,
}

/// Synthetic pool and block table for one launch shape. Values do not matter
/// (the split count is a launch-shape decision), but the scatter pattern does,
/// so physical rows run in reverse order over a pool with 2x slack.
fn build(batch: usize, context: usize) -> Workload {
    let blocks_per_seq = context.div_ceil(BLOCK_SIZE);
    let total_rows = batch * blocks_per_seq;
    let pool_blocks = (total_rows * 2).max(1) as i32;
    let bs = BLOCK_SIZE as i32;

    let q = zeros(&[batch as i32, Q_HEADS, 1, HEAD_DIM], F32);
    let pool_k = zeros(&[pool_blocks, bs, KV_HEADS, HEAD_DIM], F16);
    let pool_v = zeros(&[pool_blocks, bs, KV_HEADS, HEAD_DIM], F16);

    let rows_vec: Vec<i32> = (0..total_rows)
        .map(|i| (pool_blocks as usize - 1 - i) as i32)
        .collect();
    let rows = from_slice_i32(&rows_vec, &[rows_vec.len() as i32]);
    let offsets: Vec<i32> = (0..=batch).map(|b| (b * blocks_per_seq) as i32).collect();
    let row_offsets = from_slice_i32(&offsets, &[offsets.len() as i32]);
    let starts = vec![0i32; batch];
    let logical_starts = from_slice_i32(&starts, &[batch as i32]);
    let lens = vec![context as i32; batch];
    let visible_lens = from_slice_i32(&lens, &[batch as i32]);

    for arr in [
        &q,
        &pool_k,
        &pool_v,
        &rows,
        &row_offsets,
        &logical_starts,
        &visible_lens,
    ] {
        eval(arr);
    }
    synchronize_default();

    Workload {
        q,
        pool_k,
        pool_v,
        rows,
        row_offsets,
        logical_starts,
        visible_lens,
    }
}

#[test]
fn tuned_num_splits_is_never_slower_than_the_default() {
    if !gpu_available() {
        eprintln!("skipping: no GPU backend in this build");
        return;
    }
    let cfg = ProfileConfig::default();

    for &batch in &BATCHES {
        for &context in &CONTEXTS {
            let w = build(batch, context);
            let shape = DecodeShape {
                batch,
                q_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                context,
            };
            let op = PagedDecodeSplitsOp::new(
                &w.q,
                &w.pool_k,
                &w.pool_v,
                &w.rows,
                &w.row_offsets,
                &w.logical_starts,
                &w.visible_lens,
                1.0 / (HEAD_DIM as f32).sqrt(),
                shape,
            );

            let bucket = op.bucket();
            let candidates = op.candidates(&bucket);
            let default = op.default_tactic(&bucket);
            assert!(
                candidates.contains(&default),
                "batch={batch} ctx={context}: the default {default} must be measured, otherwise the no-regression rule cannot be enforced"
            );

            let result = profile(&op, cfg)
                .unwrap_or_else(|| panic!("batch={batch} ctx={context}: no candidate ran"));

            assert!(
                candidates.contains(&result.best),
                "batch={batch} ctx={context}: selected {} is not a feasible candidate",
                result.best
            );
            assert!(
                result.reps >= 5,
                "determinism guard wants median-of-5 or more"
            );

            let Some(default_us) = result.default_us else {
                panic!("batch={batch} ctx={context}: the default candidate failed to run");
            };
            assert!(
                result.best_us <= default_us,
                "batch={batch} ctx={context}: selected {} at {:.1}us is slower than the default {default} at {default_us:.1}us",
                result.best,
                result.best_us
            );
            if result.changed {
                // The harness only switches away from the default when the win
                // clears the noise margin, so a change implies a real win.
                assert!(
                    result.best_us < default_us,
                    "batch={batch} ctx={context}: switched away from the default without a win"
                );
            }
        }
    }
}

#[test]
fn split_candidates_stay_inside_the_launcher_budget() {
    if !gpu_available() {
        eprintln!("skipping: no GPU backend in this build");
        return;
    }
    // The candidate set has to come from the C++ launcher's own ceiling,
    // otherwise a tuned tactic could ask for a launch the kernel cannot make.
    for head_dim in [64, 96, 128, 192, 256] {
        let cap = num_splits_cap(head_dim);
        assert!(
            (1..=32).contains(&cap),
            "head_dim={head_dim}: ceiling {cap} is outside the thread-count bound"
        );
        let candidates = split_candidates(cap);
        assert!(!candidates.is_empty());
        assert_eq!(candidates.first().copied(), Some(1));
        assert_eq!(
            candidates.last().copied(),
            Some(i64::from(cap)),
            "head_dim={head_dim}: the ceiling is the pre-#906 default and must be a candidate"
        );
        for c in &candidates {
            assert!(*c >= 1 && *c <= i64::from(cap));
        }
    }
}
