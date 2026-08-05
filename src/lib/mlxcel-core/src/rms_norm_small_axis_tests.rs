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

//! Numeric regression tests for `fast::rms_norm` on the small-axis dispatch
//! band (#830, #831).
//!
//! mlxcel used to overlay the pre-#3792 upstream CUDA RMSNorm kernel. In the
//! dispatch band `axis_size in (N_READS * 32, N_READS * 32 * 2]`, that kernel
//! launched 128 threads against shared scratch sized for 64, so the two-stage
//! reduction read two floats past the end of its `__shared__` array: with two
//! rows per block it folded the sibling row's partials into the normalizer
//! (deterministic, 0.55 to 0.94 relative error), and with one row it read
//! whatever the previous kernel left in shared memory (intermittent, which is
//! what made DeepSeek-V2 decode flaky). `kv_a_layernorm` over
//! `kv_lora_rank == 512` at bf16 lands exactly on that band, 27 times per
//! DeepSeek-V2-Lite forward. The overlay was deleted in #831; the full
//! analysis is docs/upstream/mlx-cuda-rmsnorm-small-axis-regression.md.
//!
//! These tests pin the numerics across the band against a float64 host
//! reference computed from the dtype-rounded inputs, so a reintroduced broken
//! kernel fails here loudly instead of surfacing as flaky greedy decode. The
//! band is dtype-dependent (`N_READS == 16 / sizeof(dtype)`): `(256, 512]`
//! for 16-bit dtypes, `(128, 256]` for float32. Row counts above one are the
//! deterministic gate; the single-row case is kept last in the sweep so
//! earlier launches have dirtied shared memory, which is the state in which
//! the old kernel's uninitialized read was observable.
//!
//! GPU-only: on CPU-only builds `fast::rms_norm` takes the fallback path and
//! these tests return early, matching `fused_norm_parity_tests.rs`.

use super::*;

fn gpu_available() -> bool {
    crate::metal_is_available() || crate::cuda_is_available()
}

fn flatten_f32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Per-row float64 RMSNorm of the dtype-rounded inputs:
/// `x * w / sqrt(mean(x^2) + eps)`.
fn reference_rows(x: &[f32], w: &[f32], rows: usize, axis: usize, eps: f64) -> Vec<f64> {
    let mut out = vec![0f64; rows * axis];
    for r in 0..rows {
        let row = &x[r * axis..(r + 1) * axis];
        let mean_sq = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / axis as f64;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..axis {
            out[r * axis + i] = (row[i] as f64) * scale * (w[i] as f64);
        }
    }
    out
}

/// Worst per-row relative L2 error, the doc's repro metric:
/// `sqrt(sum((got - ref)^2) / sum(ref^2))` maximised over rows.
fn worst_row_rel_err(got: &[f32], reference: &[f64], rows: usize, axis: usize) -> f64 {
    let mut worst = 0f64;
    for r in 0..rows {
        let mut diff_sq = 0f64;
        let mut ref_sq = 0f64;
        for i in 0..axis {
            let g = got[r * axis + i] as f64;
            let want = reference[r * axis + i];
            diff_sq += (g - want) * (g - want);
            ref_sq += want * want;
        }
        worst = worst.max((diff_sq / ref_sq.max(1e-20)).sqrt());
    }
    worst
}

const EPS: f32 = 1e-6;

/// Sweep one dtype across the given axis sizes and row counts, asserting the
/// worst-row relative error against the float64 reference stays at the
/// dtype's own rounding scale. The broken overlay landed at 0.55 and above.
fn sweep_band(dtype_id: i32, dtype_name: &str, axes: &[i32], tol: f64) {
    for &axis in axes {
        // Multi-row cases first: they are the deterministic failure mode of
        // the old kernel (cross-row fold). The final single-row case then
        // runs after shared memory has been dirtied by the earlier launches.
        for rows in [8i32, 3, 2, 1] {
            random_seed(0x0831 + axis as u64 + rows as u64);
            let x_f32 = unsafe { random_normal(&[rows, axis], dtype::FLOAT32, std::ptr::null()) };
            let w_noise = unsafe { random_normal(&[axis], dtype::FLOAT32, std::ptr::null()) };
            let ones = full_f32(&[axis], 1.0, dtype::FLOAT32);
            let w_f32 = add(
                &ones,
                &multiply(&w_noise, &full_f32(&[1], 0.1, dtype::FLOAT32)),
            );
            let x = astype(&x_f32, dtype_id);
            let w = astype(&w_f32, dtype_id);
            eval(&x);
            eval(&w);

            let y = fast_rms_norm(&x, &w, EPS);
            let got = flatten_f32(&y);
            // Reference from the dtype-rounded bytes the kernel actually read.
            let xs = flatten_f32(&x);
            let ws = flatten_f32(&w);
            let reference = reference_rows(&xs, &ws, rows as usize, axis as usize, EPS as f64);
            let err = worst_row_rel_err(&got, &reference, rows as usize, axis as usize);
            assert!(
                err < tol,
                "rms_norm {dtype_name} axis {axis} rows {rows}: worst-row relative error \
                 {err:.3e} exceeds {tol:.0e}; the small-axis RMSNorm dispatch band has \
                 regressed again, see docs/upstream/mlx-cuda-rmsnorm-small-axis-regression.md"
            );
        }
    }
}

/// 16-bit dtypes: `N_READS == 8`, broken band `(256, 512]`. 512 is the
/// DeepSeek-V2 `kv_a_layernorm` axis; 320 is an interior point; 256 and 576
/// bracket the band from the neighbouring dispatch configs.
#[test]
fn rms_norm_small_axis_band_16bit_matches_f64_reference() {
    if !gpu_available() {
        return;
    }
    sweep_band(dtype::BFLOAT16, "bf16", &[256, 320, 512, 576], 2e-2);
    sweep_band(dtype::FLOAT16, "f16", &[256, 320, 512, 576], 5e-3);
}

/// float32: `N_READS == 4` puts the same dispatch config at `(128, 256]`.
#[test]
fn rms_norm_small_axis_band_f32_matches_f64_reference() {
    if !gpu_available() {
        return;
    }
    sweep_band(dtype::FLOAT32, "f32", &[128, 192, 256, 288], 1e-5);
}
