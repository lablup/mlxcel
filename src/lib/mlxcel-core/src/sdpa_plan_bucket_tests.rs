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

//! [#1820] Numbers out of the bucketed cuDNN SDPA path.
//!
//! The patched `sdpa_cudnn` in
//! `src/lib/mlx-cpp/patches/mlx/backend/cuda/scaled_dot_product_attention.cpp`
//! answers a small array-masked multi-row call over a KV cache by handing cuDNN
//! the whole cache buffer instead of the live prefix, with the mask widened to
//! the same width and the true lengths carried by `set_seq_len_kv`. That points
//! the kernel past the keys the caller wrote: the columns between `k_len` and
//! the buffer extent hold whatever the allocator last left there. Two
//! mechanisms keep them out of the result, cuDNN's padding mask and the `-inf`
//! columns the widened bias carries, and if both failed the symptom would be a
//! `NaN` or a quietly wrong row rather than an error. These tests pin that: the
//! shape is built the way a verify round builds it (a `KVCache` prefilled and
//! then appended to in a block), and the output is compared against an explicit
//! f32 score-matrix reference.
//!
//! Which dispatch a given call took is not observable from Rust, so these tests
//! do not assert it. That evidence is the per-call trace behind
//! `MLXCEL_SDPA_PLAN_DEBUG=1`, recorded in
//! `docs/benchmark_results/sdpa-plan-cache-bucket-gb10-2026-09-12.md` beside
//! the end-to-end token-id identity, which is the real gate.
//!
//! Run:
//!   MLX_CUDA_ARCHITECTURES=121 cargo test --profile test-fast --features cuda \
//!     -p mlxcel-core sdpa_plan_bucket
//!
//! GPU-only; they return early on a CPU-only build.

use super::*;
use crate::layers::{self, KVCache};

/// Deterministic pseudo-random f32 tensor, the generator the layers tests use.
fn test_tensor(shape: &[i32]) -> UniquePtr<MlxArray> {
    let len: i32 = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|i| ((i as f32) * 0.7371 + 0.13).sin())
        .collect();
    from_slice_f32(&data, shape)
}

fn bf16_tensor(shape: &[i32]) -> UniquePtr<MlxArray> {
    astype(&test_tensor(shape), crate::dtype::BFLOAT16)
}

fn scalar_f32(a: &MlxArray) -> f32 {
    let f = astype(a, crate::dtype::FLOAT32);
    eval(&f);
    item_f32(&f)
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    scalar_f32(&max_all(&abs(&subtract(a, b))))
}

fn has_nan(a: &MlxArray) -> bool {
    scalar_f32(&any_all(&isnan(a))) != 0.0
}

/// Attention from an explicit score matrix in f32, independent of any fused
/// kernel: `softmax(q k^T * scale + mask) v`.
fn reference_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: &MlxArray,
) -> UniquePtr<MlxArray> {
    let qf = astype(q, crate::dtype::FLOAT32);
    let kf = astype(k, crate::dtype::FLOAT32);
    let vf = astype(v, crate::dtype::FLOAT32);
    let scores = matmul(&qf, &transpose_axes(&kf, &[0, 1, 3, 2]));
    let scaled = multiply(&scores, &from_slice_f32(&[scale], &[1, 1, 1, 1]));
    let biased = add(&scaled, mask);
    matmul(&softmax_precise(&biased, 3), &vf)
}

/// Prefill `prior` positions and then append a `block`-row verify block, the
/// way a speculative round drives the cache, and return the live window.
fn cache_window(
    heads: i32,
    head_dim: i32,
    prior: i32,
    block: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let mut cache = KVCache::new();
    let _ = cache.update_and_fetch(
        bf16_tensor(&[1, heads, prior, head_dim]),
        bf16_tensor(&[1, heads, prior, head_dim]),
    );
    cache.update_and_fetch(
        bf16_tensor(&[1, heads, block, head_dim]),
        bf16_tensor(&[1, heads, block, head_dim]),
    )
}

/// A verify block over a prefilled cache must agree with an explicit f32 score
/// matrix, at every block width and prior length the bucketing covers.
///
/// The tolerance is loose on purpose. Both sides read the same bf16 keys, but
/// the fused kernel accumulates in its own order and MLX leaves
/// `MLX_ENABLE_TF32` on by default, so agreement to better than a few percent
/// is not the property under test. Reading a padded column would move a row by
/// far more than this, or produce `NaN`, which is checked separately.
#[test]
fn sdpa_plan_bucket_verify_block_matches_explicit_score_matrix() {
    if !cuda_is_available() {
        return;
    }
    let (heads, head_dim) = (8, 128);
    let scale = 1.0 / (head_dim as f32).sqrt();
    for (prior, block) in [(60, 2), (152, 4), (300, 4), (500, 16), (700, 8)] {
        let (ck, cv) = cache_window(heads, head_dim, prior, block);
        assert_eq!(array_shape(&ck)[2], prior + block);
        let q = bf16_tensor(&[1, heads, block, head_dim]);
        let mask = crate::utils::create_causal_mask(block, prior);
        let mask = mask.as_ref().unwrap();

        let out = layers::attention(&q, &ck, &cv, scale, Some(mask), 0.0, 0);
        let reference = reference_attention(&q, &ck, &cv, scale, mask);

        let out_f32 = astype(&out, crate::dtype::FLOAT32);
        assert!(
            !has_nan(&out_f32),
            "prior={prior} block={block}: attention produced NaN, which is what \
             reading past the live keys looks like"
        );
        let diff = max_abs_diff(&out_f32, &reference);
        assert!(
            diff < 5e-2,
            "prior={prior} block={block}: fused attention diverged from the \
             explicit f32 score matrix by {diff}"
        );
    }
}

/// The block's last row attends every live key and nothing else, so widening
/// the tensors must not change it. This is the row a stale padded column would
/// reach first: it is the only row whose causal mask blocks no live column, so
/// any column the kernel reads beyond `k_len` enters its softmax unopposed by a
/// `-inf` from the caller's own mask.
#[test]
fn sdpa_plan_bucket_last_verify_row_is_unaffected_by_the_padded_tail() {
    if !cuda_is_available() {
        return;
    }
    let (heads, head_dim) = (8, 128);
    let scale = 1.0 / (head_dim as f32).sqrt();
    for (prior, block) in [(152, 4), (300, 8)] {
        let (ck, cv) = cache_window(heads, head_dim, prior, block);
        let k_len = prior + block;
        let q = bf16_tensor(&[1, heads, block, head_dim]);
        let mask = crate::utils::create_causal_mask(block, prior);
        let mask = mask.as_ref().unwrap();
        let out = layers::attention(&q, &ck, &cv, scale, Some(mask), 0.0, 0);

        // The same last row computed on its own, as a one-row step over the
        // identical keys. A one-row call never takes the bucketed path.
        let last = slice(&q, &[0, 0, block - 1, 0], &[1, heads, block, head_dim]);
        let open = from_slice_f32(&vec![0.0; k_len as usize], &[1, k_len]);
        let decode = layers::attention(&last, &ck, &cv, scale, Some(&open), 0.0, 0);

        let got = astype(
            &slice(&out, &[0, 0, block - 1, 0], &[1, heads, block, head_dim]),
            crate::dtype::FLOAT32,
        );
        let want = astype(&decode, crate::dtype::FLOAT32);
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff < 5e-2,
            "prior={prior} block={block}: the fully open last verify row \
             diverged from the same row computed alone by {diff}"
        );
    }
}
