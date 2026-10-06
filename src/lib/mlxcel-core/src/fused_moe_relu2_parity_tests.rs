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

//! Parity for `fused_moe_forward`'s opt-in squared-ReLU branch
//! (`MLXCEL_FUSED_MOE_RELU2`, kernels from #268, ROCm port in #2069).
//!
//! The branch launches the fc1 + relu² kernel and the shared down kernel in
//! place of the default `gather_qmm` routed experts. These tests run the real
//! entry point both ways on the same inputs and hold the kernel branch to:
//!
//! 1. an all-f32 dense reference (dequantize + matmul + relu² + matmul,
//!    score-weighted, rounded to bf16 once), the sharp gate: a dropped lane,
//!    a wrong group or a wrong expert row lands orders of magnitude above it;
//! 2. the `gather_qmm` branch it replaces, within that branch's own measured
//!    distance from the same dense reference.
//!
//! Shapes are NVIDIA-Nemotron-3-Nano-30B-A3B's (hidden 2688, expert
//! intermediate 1856, top-k 6, group size 64) with the expert count cut from
//! 128 to 16, which changes only the range of the expert index.
//!
//! GPU-only, and only where `fused_moe_relu2_kernels_available()` is true
//! (Metal, ROCm). On CUDA the flag declines to `gather_qmm`, so there is no
//! kernel to compare and the tests skip.
//!
//! Run on ROCm (gfx1151 etc.):
//!   cargo test -p mlxcel-core --release --features rocm --lib \
//!     fused_moe_relu2_parity_tests -- --test-threads=1

use super::*;
use crate::fused_moe_parity_tests::{
    flatten_f32, normalized_deviation, random_quantized_expert_stack,
};

const DIN: i32 = 2688;
const DFF: i32 = 1856;
const NUM_EXPERTS: i32 = 16;
const TOP_K: i32 = 6;
const GROUP_SIZE: i32 = 64;
const SCALING_FACTOR: f32 = 2.5;

struct Relu2Case {
    bits: i32,
    x: UniquePtr<MlxArray>,               // [1, din] bf16
    gate_weight: UniquePtr<MlxArray>,     // [E, din] bf16
    correction_bias: UniquePtr<MlxArray>, // [E] f32
    fc1: (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ),
    fc2: (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ),
}

fn build_case(seed: u64, bits: i32) -> Relu2Case {
    random_seed(seed);
    let bf16 = |shape: &[i32]| {
        let f = unsafe { random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
        let a = astype(&f, dtype::BFLOAT16);
        eval(&a);
        a
    };
    let x = bf16(&[1, DIN]);
    let gate_weight = bf16(&[NUM_EXPERTS, DIN]);
    let correction_bias =
        unsafe { random_normal(&[NUM_EXPERTS], dtype::FLOAT32, std::ptr::null()) };
    eval(&correction_bias);
    let fc1 = random_quantized_expert_stack(NUM_EXPERTS, DFF, DIN, GROUP_SIZE, bits);
    let fc2 = random_quantized_expert_stack(NUM_EXPERTS, DIN, DFF, GROUP_SIZE, bits);
    Relu2Case {
        bits,
        x,
        gate_weight,
        correction_bias,
        fc1,
        fc2,
    }
}

/// `fused_moe_forward` with no shared expert, as Nemotron-H's MoE calls it.
/// The caller holds the env lock and sets or clears `MLXCEL_FUSED_MOE_RELU2`.
fn run_forward(case: &Relu2Case) -> UniquePtr<MlxArray> {
    let out = unsafe {
        fused_moe_forward(
            &case.x,
            &case.gate_weight,
            &case.correction_bias,
            &case.fc1.0,
            &case.fc1.1,
            &case.fc1.2,
            &case.fc2.0,
            &case.fc2.1,
            &case.fc2.2,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            TOP_K,
            SCALING_FACTOR,
            true,
            GROUP_SIZE,
            case.bits,
        )
    }
    .expect("fused_moe_forward must not refuse on a backend with both relu2 ports");
    eval(&out);
    out
}

/// All-f32 dense routed-expert reference on the identical bf16 inputs, using
/// the router's own top-k (the same compiled gate `fused_moe_forward` runs).
fn reference_dense_f32(case: &Relu2Case) -> UniquePtr<MlxArray> {
    let gates = matmul(&case.x, &transpose(&case.gate_weight)); // [1, E]
    let mut indices = UniquePtr::null();
    let mut scores = UniquePtr::null();
    compiled_moe_gate(
        &gates,
        &case.correction_bias,
        TOP_K,
        SCALING_FACTOR,
        true,
        &mut indices,
        &mut scores,
    );
    let idx = reshape(&indices, &[TOP_K]);
    let dq = |w: &MlxArray, s: &MlxArray, b: &MlxArray| -> UniquePtr<MlxArray> {
        // f32 scales and biases keep the dequantized weights in f32, as the
        // kernel computes `w * scale + bias` in f32 registers.
        let s32 = astype(s, dtype::FLOAT32);
        let b32 = astype(b, dtype::FLOAT32);
        unsafe {
            dequantize(
                w,
                &s32,
                &b32 as &MlxArray as *const MlxArray,
                GROUP_SIZE,
                case.bits,
                "affine",
            )
        }
    };
    let fc1_sel = take(&dq(&case.fc1.0, &case.fc1.1, &case.fc1.2), &idx, 0); // [k, dff, din]
    let fc2_sel = take(&dq(&case.fc2.0, &case.fc2.1, &case.fc2.2), &idx, 0); // [k, din, dff]
    let x_col = reshape(&astype(&case.x, dtype::FLOAT32), &[DIN, 1]);
    let h = matmul(&fc1_sel, &x_col); // [k, dff, 1]
    let zero = zeros(&[1], dtype::FLOAT32);
    let act = square(&maximum(&h, &zero));
    let per_expert = squeeze_axis(&matmul(&fc2_sel, &act), -1); // [k, din]
    let w_col = reshape(&astype(&scores, dtype::FLOAT32), &[TOP_K, 1]);
    let out = reshape(
        &sum_axis(&multiply(&per_expert, &w_col), 0, false),
        &[1, DIN],
    );
    eval(&out);
    out
}

/// Runs the forward with the flag unset (the `gather_qmm` branch) and set
/// (the kernel branch), restoring the caller's environment afterwards.
fn run_both_branches(case: &Relu2Case) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let _guard = crate::test_support::env_lock::env_lock();
    let saved = std::env::var("MLXCEL_FUSED_MOE_RELU2").ok();
    unsafe { std::env::remove_var("MLXCEL_FUSED_MOE_RELU2") };
    let graph = run_forward(case);
    unsafe { std::env::set_var("MLXCEL_FUSED_MOE_RELU2", "1") };
    let kernel = run_forward(case);
    match saved {
        Some(v) => unsafe { std::env::set_var("MLXCEL_FUSED_MOE_RELU2", v) },
        None => unsafe { std::env::remove_var("MLXCEL_FUSED_MOE_RELU2") },
    }
    (graph, kernel)
}

/// The backend's name when it has both relu2 kernel ports, else `None`.
fn relu2_backend_or_skip() -> Option<&'static str> {
    use crate::hardware::{GpuBackendKind, gpu_backend_kind};
    let name = match gpu_backend_kind() {
        GpuBackendKind::Metal => "metal",
        GpuBackendKind::Rocm => "rocm",
        GpuBackendKind::Cuda => {
            assert!(
                !crate::fused_moe_relu2_kernels_available(),
                "CUDA has no fc1_relu2 port, so the relu2 predicate must be false"
            );
            eprintln!("skipping: CUDA has no fc1_relu2 port, the flag declines to gather_qmm");
            return None;
        }
        GpuBackendKind::None => {
            eprintln!("skipping: no GPU backend, so no fused MoE kernel port");
            return None;
        }
    };
    assert!(
        crate::fused_moe_relu2_kernels_available(),
        "{name} has both relu2 kernel ports, so fused_moe_relu2_kernels_available() must be true"
    );
    Some(name)
}

/// Kernel branch against the dense f32 reference (sharp) and against the
/// `gather_qmm` branch it replaces (within that branch's own jitter), at 4
/// and 8 bits, the two widths the branch accepts.
#[test]
fn fused_moe_relu2_kernel_matches_references_nemotron_shape() {
    let Some(backend) = relu2_backend_or_skip() else {
        return;
    };
    for bits in [4, 8] {
        for seed in [2069u64, 2070] {
            let case = build_case(seed, bits);
            let (graph_raw, kernel_raw) = run_both_branches(&case);
            assert_eq!(array_dtype(&kernel_raw), dtype::BFLOAT16);
            let kernel = flatten_f32(&kernel_raw);
            let graph = flatten_f32(&graph_raw);
            let dense_raw = reference_dense_f32(&case);
            let dense = flatten_f32(&dense_raw);
            let dense_bf16 = flatten_f32(&astype(&dense_raw, dtype::BFLOAT16));

            let (nrms_dense, nmax_dense) = normalized_deviation(&kernel, &dense_bf16);
            let (nrms_graph, nmax_graph) = normalized_deviation(&kernel, &graph);
            let (nrms_jitter, nmax_jitter) = normalized_deviation(&graph, &dense);
            println!(
                "relu2 parity [{backend}, {bits}-bit, seed {seed}]: kernel vs dense f32 \
                 (bf16-rounded) nrms={nrms_dense:e} nmax={nmax_dense:e}; kernel vs gather_qmm \
                 nrms={nrms_graph:e} nmax={nmax_graph:e}; gather_qmm vs dense \
                 nrms={nrms_jitter:e} nmax={nmax_jitter:e}"
            );
            // The kernel rounds once, at the end, like the bf16-rounded
            // reference; the bound is the one `fused_moe_parity_tests` holds
            // the SwiGLU and GeGLU kernels to.
            assert!(
                nrms_dense < 5e-4 && nmax_dense < 2e-2,
                "relu2 kernel deviates from the dense f32 reference ({bits}-bit, seed {seed}): \
                 nrms={nrms_dense:e} nmax={nmax_dense:e}"
            );
            // Against the branch it replaces: within 1.5x / 2x that branch's
            // own distance from the truth, never tighter than 5e-3 / 5e-2
            // (the margins `fused_moe_parity_tests` uses).
            let nrms_limit = (1.5 * nrms_jitter).max(5e-3);
            let nmax_limit = (2.0 * nmax_jitter).max(5e-2);
            assert!(
                nrms_graph < nrms_limit && nmax_graph < nmax_limit,
                "relu2 kernel and gather_qmm disagree beyond gather_qmm's own jitter \
                 ({bits}-bit, seed {seed}): nrms={nrms_graph:e} (limit {nrms_limit:e}) \
                 nmax={nmax_graph:e} (limit {nmax_limit:e})"
            );
        }
    }
}
