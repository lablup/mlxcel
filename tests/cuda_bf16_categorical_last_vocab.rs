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

//! Regression tests for CUDA `bfloat16` categorical sampling at large vocabulary
//! sizes (issue #2958).
//!
//! `overlays/cuda/mlx/dtype.cpp` maps `promote_types(bfloat16, float32) -> bfloat16`
//! so elementwise transformer ops avoid `copy_v` round-trips. Upstream MLX's
//! `mlx/random.cpp` assumes `promote_types(logits.dtype(), float32)` yields at
//! least `float32`:
//!
//! 1. Single-row decode (`categorical_inverse_cdf`): when `logits` is `bfloat16`
//!    (as `fused_sample_filter_logits` keeps it on CUDA when `vocab = 262144`
//!    exceeds `REJECTION_JOINT_VOCAB_MAX = 32768`), `w`, the exclusive `cumsum`
//!    `cdf`, `sum(w)`, and `u = uniform_f32 * sum(w)` all ran in `bfloat16`.
//!    Any uniform draw in `[1 - 2^-8, 1)` rounded `u` up to `sum(w)` (and `bf16`
//!    scan rounding could leave `cdf[V - 1] < sum(w)`), making `u >= cdf[V - 1]`
//!    so `searchsorted(cdf, u, "right") - 1` returned `V - 1` (`262143`,
//!    `<unused6226>` on Gemma 4) ~0.4% to 1% of steps even though row `262143`
//!    had been masked to `-inf` (`w[262143] == 0`).
//! 2. Batched decode (`categorical_impl` Gumbel-max path for `B >= 2`):
//!    `add(g /*float32*/, logits /*bfloat16*/)` demoted the `float32` Gumbel
//!    noise to `bfloat16`, collapsing fine logit differences and shift-invariance
//!    into `argmax` index-0 ties.

#![cfg(feature = "cuda")]

use std::sync::Mutex;

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const VOCAB: usize = 262_144;
const LAST_VOCAB: u32 = (VOCAB - 1) as u32;
const TOP_K: usize = 64;

static GPU: Mutex<()> = Mutex::new(());

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn on_cuda() -> bool {
    gpu_backend_kind() == GpuBackendKind::Cuda
}

fn read_u32_tokens(arr: &MlxArray) -> Vec<u32> {
    let as_u32 = mlxcel_core::astype(arr, dtype::UINT32);
    mlxcel_core::eval(&as_u32);
    let bytes = mlxcel_core::array_to_raw_bytes(&as_u32);
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Build a `[1, 262144]` `bfloat16` logit row matching Gemma 4 post-softcap
/// decode: indices `0..64` lie in `[12.0, 24.0]` and indices `64..262144`
/// (including `262143`) sit at `-20.0`, strictly outside the top-64 support.
fn gemma4_like_bf16_logits() -> UniquePtr<MlxArray> {
    let _gpu = DefaultDeviceGuard::gpu();
    let mut row = vec![-20.0_f32; VOCAB];
    for (i, slot) in row.iter_mut().take(TOP_K).enumerate() {
        *slot = 24.0 - (i as f32) * (12.0 / TOP_K as f32);
    }
    let f32_arr = mlxcel_core::from_slice_f32(&row, &[1, VOCAB as i32]);
    let bf16_arr = mlxcel_core::astype(&f32_arr, dtype::BFLOAT16);
    mlxcel_core::eval(&bf16_arr);
    bf16_arr
}

#[test]
fn fused_sample_bf16_large_vocab_never_draws_masked_last_token() {
    let _gpu = gpu_guard();
    if !on_cuda() {
        eprintln!("skip: not a CUDA build/device");
        return;
    }

    let _dev = DefaultDeviceGuard::gpu();
    let logits = gemma4_like_bf16_logits();
    mlxcel_core::random_seed(0x2958_0001);

    const DRAWS: usize = 2_048;
    let mut out_of_support = Vec::new();
    let mut last_vocab_hits = 0usize;

    for step in 0..DRAWS {
        let tok = mlxcel_core::fused_sample(&logits, 0.8, TOP_K as i32, 0.95, 0.0);
        let id = read_u32_tokens(&tok)[0];
        if id == LAST_VOCAB {
            last_vocab_hits += 1;
        }
        if id as usize >= TOP_K {
            out_of_support.push((step, id));
        }
    }

    // Exercise `fused_sample_categorical` directly as well.
    for step in 0..512 {
        let tok = mlxcel_core::fused_sample_categorical(&logits, 0.8, TOP_K as i32, 0.95, 0.0);
        let id = read_u32_tokens(&tok)[0];
        if id == LAST_VOCAB {
            last_vocab_hits += 1;
        }
        if id as usize >= TOP_K {
            out_of_support.push((DRAWS + step, id));
        }
    }

    assert!(
        out_of_support.is_empty(),
        "bfloat16 fused_sample drew {} masked token(s) outside 0..{TOP_K} ({} hit(s) on \
         vocab-1 = {LAST_VOCAB}); first occurrences: {:?}",
        out_of_support.len(),
        last_vocab_hits,
        &out_of_support[..out_of_support.len().min(10)]
    );
}

#[test]
fn random_categorical_bf16_single_row_never_draws_neg_inf_tail() {
    let _gpu = gpu_guard();
    if !on_cuda() {
        eprintln!("skip: not a CUDA build/device");
        return;
    }

    let _dev = DefaultDeviceGuard::gpu();
    let mut row = vec![f32::NEG_INFINITY; VOCAB];
    for (i, slot) in row.iter_mut().take(12).enumerate() {
        *slot = (24.0 - (i as f32) * (12.0 / TOP_K as f32)) / 0.8;
    }
    let logits_f32 = mlxcel_core::from_slice_f32(&row, &[1, VOCAB as i32]);
    let logits = mlxcel_core::astype(&logits_f32, dtype::BFLOAT16);
    mlxcel_core::eval(&logits);

    mlxcel_core::random_seed(0x2958_0002);
    const DRAWS: usize = 2_048;
    let mut bad = Vec::new();
    let mut last_vocab_hits = 0usize;
    for step in 0..DRAWS {
        let sampled = mlxcel_core::random_categorical(&logits, -1);
        let id = read_u32_tokens(&sampled)[0];
        if id == LAST_VOCAB {
            last_vocab_hits += 1;
        }
        if id as usize >= TOP_K {
            bad.push((step, id));
        }
    }

    assert!(
        bad.is_empty(),
        "bfloat16 random_categorical drew {} -inf tail token(s) over {DRAWS} draws \
         ({} hit(s) on vocab-1 = {LAST_VOCAB}); first occurrences: {:?}",
        bad.len(),
        last_vocab_hits,
        &bad[..bad.len().min(10)]
    );
}

#[test]
fn random_categorical_bf16_batched_preserves_f32_gumbel_and_masks_neg_inf() {
    let _gpu = gpu_guard();
    if !on_cuda() {
        eprintln!("skip: not a CUDA build/device");
        return;
    }

    let _dev = DefaultDeviceGuard::gpu();
    mlxcel_core::random_seed(0x2958_0003);

    // Part A: batched [8, 262144] with -inf tail must never sample masked indices.
    const BATCH: usize = 8;
    let mut single = vec![f32::NEG_INFINITY; VOCAB];
    for (i, slot) in single.iter_mut().take(32).enumerate() {
        *slot = -(i as f32) * 0.25;
    }
    let mut batched_host = Vec::with_capacity(BATCH * VOCAB);
    for _ in 0..BATCH {
        batched_host.extend_from_slice(&single);
    }
    let batched_f32 = mlxcel_core::from_slice_f32(&batched_host, &[BATCH as i32, VOCAB as i32]);
    let batched_bf16 = mlxcel_core::astype(&batched_f32, dtype::BFLOAT16);
    mlxcel_core::eval(&batched_bf16);

    for step in 0..128 {
        let sampled = mlxcel_core::random_categorical(&batched_bf16, -1);
        for (b, id) in read_u32_tokens(&sampled).into_iter().enumerate() {
            assert!(
                (id as usize) < 32,
                "batched bfloat16 random_categorical drew -inf token {id} at step {step} batch {b}"
            );
        }
    }

    // Part B: shift-invariant equal logits [1024.0, 1024.0] across 1024 batch rows.
    // At 1024.0, 1 bf16 ULP is 8.0, so demoting `add(g_f32, logits_bf16)` to bf16
    // rounds >98% of Gumbel draws into a tie that `argmax` breaks to index 0.
    const ROWS: usize = 1_024;
    let equal_large = vec![1024.0_f32; ROWS * 2];
    let eq_f32 = mlxcel_core::from_slice_f32(&equal_large, &[ROWS as i32, 2]);
    let eq_bf16 = mlxcel_core::astype(&eq_f32, dtype::BFLOAT16);
    mlxcel_core::eval(&eq_bf16);

    let draws = read_u32_tokens(&mlxcel_core::random_categorical(&eq_bf16, -1));
    let ones = draws.iter().filter(|&&id| id == 1).count();
    assert!(
        (420..=604).contains(&ones),
        "batched bfloat16 random_categorical on [1024.0, 1024.0] chose index 1 in {ones}/{ROWS} \
         rows (expected ~512; bf16 Gumbel demotion collapses draws to index 0)"
    );
}

#[test]
fn fused_sample_bf16_top_p_only_masks_tail_without_cumsum_swamping() {
    let _gpu = gpu_guard();
    if !on_cuda() {
        eprintln!("skip: not a CUDA build/device");
        return;
    }

    let _dev = DefaultDeviceGuard::gpu();
    // p_0 ~= 0.601, p_{1..=10000} ~= 2.73e-5 each (cumulative ~0.874 > top_p = 0.80),
    // and indices 10001..262144 hold the remaining ~12.6% tail mass. In bf16,
    // 0.601 + 2.73e-5 == 0.601 (ULP at 0.5 is 2^-8 = 3.9e-3), which would leave
    // cumsum stuck below 0.80 and leak ~12.6% of draws into 10001..262144.
    let mut row = vec![-2.0_f32; VOCAB];
    row[0] = 12.0;
    for slot in &mut row[1..=10_000] {
        *slot = 2.0;
    }
    let logits_f32 = mlxcel_core::from_slice_f32(&row, &[1, VOCAB as i32]);
    let logits_bf16 = mlxcel_core::astype(&logits_f32, dtype::BFLOAT16);
    mlxcel_core::eval(&logits_bf16);

    mlxcel_core::random_seed(0x2958_0004);
    const DRAWS: usize = 512;
    let mut leaked_tail = Vec::new();
    for step in 0..DRAWS {
        let tok = mlxcel_core::fused_sample(&logits_bf16, 1.0, 0, 0.80, 0.0);
        let id = read_u32_tokens(&tok)[0];
        if id as usize > 10_000 {
            leaked_tail.push((step, id));
        }
    }

    assert!(
        leaked_tail.is_empty(),
        "bfloat16 top_p=0.80 (top_k=0) leaked {} tail token(s) past index 10000 over {DRAWS} \
         draws; first occurrences: {:?}",
        leaked_tail.len(),
        &leaked_tail[..leaked_tail.len().min(10)]
    );
}
