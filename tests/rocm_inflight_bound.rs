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

//! The ROCm backend bounds what in-flight command batches pin (issue #2062,
//! `patches-rocm/LOCAL_FIXES.md` item 28).
//!
//! A command batch keeps every buffer its operations allocate until the batch
//! finishes on the GPU. Before the bound, the eager path let the host encode
//! far ahead of the GPU, so the transients of a whole prefill were live at
//! once: Llama-3.1-8B-4bit peaked at 20.60 GB for 4.75 GB of weights, most of
//! it the f16 weight copies `QuantizedMatmul` allocates on its
//! dequantize-and-GEMM path. This test builds that pattern directly: 40
//! quantized 8192x8192 f16 matmuls over a 512-row activation in one
//! evaluation, each allocating a 128 MiB f16 weight copy, 5 GiB of transients
//! in all. (It has to be f16: bf16 affine 4-bit matmuls take the fused WMMA
//! kernel, which allocates no copy.) Under the default
//! `MLX_ROCM_MAX_INFLIGHT_MB` (1024) the allocator's peak rose by 1.75 GiB on
//! gfx1151 and must stay under 3 GiB; with the bound off
//! (`MLX_ROCM_MAX_INFLIGHT_MB=0`) it rose by 6.25 GiB and the test fails.
//!
//! Skips on any other backend. Run on a ROCm host with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_inflight_bound -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const DIM: i32 = 8192;
const ROWS: i32 = 512;
const MATMULS: usize = 40;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

fn eval_ok(label: &str, a: &MlxArray) {
    if let Err(err) = mlxcel_core::try_eval(a) {
        panic!("{label}: evaluation failed: {err}");
    }
}

fn normal_f16(shape: &[i32]) -> UniquePtr<MlxArray> {
    let f32 = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    let out = mlxcel_core::astype(&f32, dtype::FLOAT16);
    eval_ok("random input", &out);
    out
}

#[test]
fn prefill_transients_stay_within_the_inflight_budget() {
    if gpu_backend_kind() != GpuBackendKind::Rocm {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _gpu = DefaultDeviceGuard::gpu();
    mlxcel_core::random_seed(2062);

    // Distinct quantized weights, so no evaluation can reuse another's
    // dequantized copy.
    let base = normal_f16(&[DIM, DIM]);
    let mut weights = Vec::with_capacity(MATMULS);
    for i in 0..MATMULS {
        let shift = mlxcel_core::full_f32(&[1], 0.01 * i as f32, dtype::FLOAT16);
        let w = mlxcel_core::add(&base, &shift);
        let q = mlxcel_core::quantize_weights_with_mode(&w, GROUP_SIZE, BITS, "affine");
        let packed = mlxcel_core::quantized_weights_w(&q);
        let scales = mlxcel_core::quantized_weights_scales(&q);
        let biases = mlxcel_core::quantized_weights_biases(&q);
        eval_ok("quantize", &packed);
        eval_ok("quantize", &scales);
        eval_ok("quantize", &biases);
        weights.push((packed, scales, biases));
    }
    drop(base);
    let x = normal_f16(&[ROWS, DIM]);
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let baseline = mlxcel_core::memory::active_memory();
    mlxcel_core::memory::reset_peak_memory();

    let outputs: Vec<UniquePtr<MlxArray>> = weights
        .iter()
        .map(|(packed, scales, biases)| unsafe {
            mlxcel_core::quantized_matmul(
                &x,
                packed,
                scales,
                &**biases as *const MlxArray,
                true,
                GROUP_SIZE,
                BITS,
                "affine",
            )
        })
        .collect();
    let refs: Vec<&MlxArray> = outputs.iter().map(|o| &**o).collect();
    let total = mlxcel_core::sum_all(&mlxcel_core::concatenate_many(&refs, 0));
    eval_ok("40 quantized matmuls", &total);
    mlxcel_core::synchronize_default();

    let growth = mlxcel_core::memory::peak_memory().saturating_sub(baseline);
    let gib = 1024.0 * 1024.0 * 1024.0;
    eprintln!(
        "peak rose {:.2} GiB over {:.2} GiB of weights and input",
        growth as f64 / gib,
        baseline as f64 / gib
    );
    assert!(
        (growth as f64) < 3.0 * gib,
        "the allocator's peak rose by {:.2} GiB for {MATMULS} matmuls whose \
         transients the in-flight bound should keep near 1 GiB",
        growth as f64 / gib
    );
}
