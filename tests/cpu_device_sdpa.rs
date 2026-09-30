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

//! Scaled dot-product attention runs on the CPU device of a GPU build
//! (issue #1807).
//!
//! `MLXCEL_DEVICE=cpu` moves the default device to the CPU, and every
//! attention call then asks the GPU backend's
//! `ScaledDotProductAttention::use_fallback` whether to build the fused
//! primitive for a CPU stream. The ROCm overlay answered from the shape alone,
//! so a decode-shaped query got the fused primitive, whose `eval_cpu` throws
//! "NYI" across the cxx bridge and terminates the process: `mlxcel generate`
//! and `examples/logit_trace` aborted at the first token on the CPU device.
//! `patches-rocm/LOCAL_FIXES.md` item 23 adds the GPU-stream check upstream
//! CUDA makes first. On the pre-fix overlay this binary aborts at the decode
//! case (checked on gfx1151).
//!
//! Not backend-gated: every backend must take the fallback on a CPU stream.
//! The test moves the process-global default device, so it holds
//! `streams::lock_default_device` for its whole body.

use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, dtype};

fn to_f32_vec(arr: &MlxArray) -> Vec<f32> {
    let as_f32 = mlxcel_core::astype(arr, dtype::FLOAT32);
    mlxcel_core::eval(&as_f32);
    mlxcel_core::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 23) as f32 - 11.0) / 13.0)
        .collect()
}

/// Plain softmax attention over `[heads, len, dim]`, no mask.
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    q_len: usize,
    kv_len: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(heads * q_len * dim);
    for h in 0..heads {
        for i in 0..q_len {
            let q_row = &q[(h * q_len + i) * dim..][..dim];
            let scores: Vec<f32> = (0..kv_len)
                .map(|j| {
                    let k_row = &k[(h * kv_len + j) * dim..][..dim];
                    q_row.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for d in 0..dim {
                let acc: f32 = (0..kv_len)
                    .map(|j| weights[j] * v[(h * kv_len + j) * dim + d])
                    .sum();
                out.push(acc / total);
            }
        }
    }
    out
}

#[test]
fn sdpa_takes_the_fallback_on_the_cpu_device() {
    let _device = lock_default_device();
    let _cpu = DefaultDeviceGuard::cpu();

    let (heads, kv_len, dim) = (2usize, 16usize, 64usize);
    let scale = 1.0 / (dim as f32).sqrt();
    // Decode (one query row, the shape the vector kernel claims) and a short
    // prefill block, in f32 and in the bf16 the models run. The reference
    // reads the inputs back after the cast, so only the attention itself and
    // the output rounding separate the two.
    for (act_dtype, tolerance) in [(dtype::FLOAT32, 1e-5f32), (dtype::BFLOAT16, 2e-2)] {
        for q_len in [1usize, 4] {
            let shape_q = [1, heads as i32, q_len as i32, dim as i32];
            let shape_kv = [1, heads as i32, kv_len as i32, dim as i32];
            let cast = |len: usize, seed: usize, shape: &[i32]| {
                let arr = mlxcel_core::from_slice_f32(&values(len, seed), shape);
                mlxcel_core::astype(&arr, act_dtype)
            };
            let q = cast(heads * q_len * dim, 1, &shape_q);
            let k = cast(heads * kv_len * dim, 2, &shape_kv);
            let v = cast(heads * kv_len * dim, 3, &shape_kv);
            // The entry point every attention layer calls, which reaches
            // `fast::scaled_dot_product_attention` and so `use_fallback`.
            // SAFETY: `q`, `k` and `v` are live arrays built above, and a null
            // mask pointer is the documented "no mask" argument.
            let out = unsafe {
                mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, scale, std::ptr::null())
            };
            let got = to_f32_vec(&out);
            let want = reference(
                &to_f32_vec(&q),
                &to_f32_vec(&k),
                &to_f32_vec(&v),
                heads,
                q_len,
                kv_len,
                dim,
                scale,
            );
            assert_eq!(got.len(), want.len());
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < tolerance,
                "dtype {act_dtype} q_len {q_len}: max abs error {worst:e}"
            );
        }
    }
}
