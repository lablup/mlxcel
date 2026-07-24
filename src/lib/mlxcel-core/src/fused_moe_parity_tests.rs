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

//! Numeric-parity and determinism tests for the fused single-token decode-MoE
//! GeGLU kernel (#886, kernel introduced in #268/#282).
//!
//! `gemma-4-26b-a4b` produced corrupted tokens during long multi-turn decode
//! with the fused kernel enabled, and the corruption was non-deterministic
//! run-to-run at a fixed config (see issue #886). These tests pin the three
//! properties the kernel must hold so a regression of that class can never
//! land silently again:
//!
//! 1. **Numeric parity**: the fused kernel must match an all-f32 dense
//!    reference (dequantize + matmul + tanh-approx GeGLU) built from the very
//!    same bf16 quantized weights, and the production `gather_qmm` fallback,
//!    within a tolerance far below the greedy-argmax-flip scale.
//! 2. **Bitwise run-to-run determinism**: repeated invocations on identical
//!    inputs must produce byte-identical outputs, including with allocator
//!    churn between calls (dirty recycled buffers are what turn an
//!    uninitialized-read or race bug into visible corruption).
//! 3. **SGY invariance**: each output row is computed by one 32-lane
//!    warp/simdgroup whose math never depends on `MLXCEL_FUSED_MOE_SGY`
//!    (rows-per-threadgroup packing), so the output must be byte-identical
//!    across every SGY setting. Any divergence proves a launch-geometry race.
//!
//! Shapes mirror gemma-4-26b-a4b exactly where it matters for kernel geometry
//! (Din=2816, Dff=704, K=8, 4-bit, group_size=64, bf16); the expert count is
//! reduced 128 -> 16 to keep the synthetic weight stacks small, which does not
//! change per-row addressing (`row = e * Dff + f`), only the range of `e`.
//!
//! GPU-only: the kernels JIT through `mx.fast.metal_kernel` /
//! `mx.fast.cuda_kernel`, so the tests skip (return early) on CPU-only builds,
//! matching the convention of the fused paged-decode tests in `ffi_tests.rs`.
//!
//! Run on CUDA (GB10 etc.):
//!   MLX_CUDA_ARCHITECTURES=121 cargo test -p mlxcel-core --release \
//!     --features cuda fused_moe_geglu
//! Run on Apple Silicon:
//!   cargo test -p mlxcel-core --release fused_moe_geglu

use super::*;

/// Quantize a random `[e * rows, cols]` bf16 matrix into a per-expert stack
/// `([e, rows, cols/pack] u32, [e, rows, cols/gs] bf16, [e, rows, cols/gs]
/// bf16)`. Quantization groups run along `cols`, so quantizing the flattened
/// 2-D matrix is exactly equivalent to quantizing each expert separately.
fn random_quantized_expert_stack(
    e: i32,
    rows: i32,
    cols: i32,
    group_size: i32,
    bits: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let full_f32 = unsafe { random_normal(&[e * rows, cols], dtype::FLOAT32, std::ptr::null()) };
    // bf16 weights like the real checkpoint, so the kernel-side astype(s, T)
    // and the dense reference read the very same values.
    let full = astype(&full_f32, dtype::BFLOAT16);
    let w2 = quantize_weights_w(&full, group_size, bits);
    let s2 = quantize_weights_scales(&full, group_size, bits);
    let b2 = quantize_weights_biases(&full, group_size, bits);
    let packed_cols = *array_shape(&w2).last().unwrap();
    let scale_cols = *array_shape(&s2).last().unwrap();
    let w = reshape(&w2, &[e, rows, packed_cols]);
    let s = reshape(&s2, &[e, rows, scale_cols]);
    let b = reshape(&b2, &[e, rows, scale_cols]);
    eval(&w);
    eval(&s);
    eval(&b);
    (w, s, b)
}

/// One randomized GeGLU MoE decode case: bf16 activations/scales/biases,
/// 4-bit affine weights, K distinct experts, positive normalized scores.
struct GegluMoeCase {
    din: i32,
    dff: i32,
    k: i32,
    group_size: i32,
    bits: i32,
    x: UniquePtr<MlxArray>,       // [din] bf16
    indices: UniquePtr<MlxArray>, // [k] u32, distinct
    scores: UniquePtr<MlxArray>,  // [k] bf16 (pre-rounded so both paths see identical values)
    gate_w: UniquePtr<MlxArray>,
    gate_s: UniquePtr<MlxArray>,
    gate_b: UniquePtr<MlxArray>,
    up_w: UniquePtr<MlxArray>,
    up_s: UniquePtr<MlxArray>,
    up_b: UniquePtr<MlxArray>,
    down_w: UniquePtr<MlxArray>,
    down_s: UniquePtr<MlxArray>,
    down_b: UniquePtr<MlxArray>,
}

fn build_geglu_case(seed: u64, din: i32, dff: i32, num_experts: i32, k: i32) -> GegluMoeCase {
    let group_size = 64;
    let bits = 4;
    random_seed(seed);

    let x_f32 = unsafe { random_normal(&[din], dtype::FLOAT32, std::ptr::null()) };
    let x = astype(&x_f32, dtype::BFLOAT16);
    eval(&x);

    // Distinct expert ids spread across the stack (stride keeps them unique
    // as long as k <= num_experts, which every case here satisfies).
    assert!(k <= num_experts);
    let stride = (num_experts / k).max(1) as u32;
    let idx_vals: Vec<u32> = (0..k as u32)
        .map(|i| (seed as u32 + i * stride) % num_experts as u32)
        .collect();
    {
        let mut sorted = idx_vals.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), k as usize, "expert indices must be distinct");
    }
    let indices = from_slice_u32(&idx_vals, &[k]);

    // Positive, roughly softmax-like scores, pre-rounded to bf16 so the fused
    // kernel (which casts scores to the activation dtype) and the reference
    // consume bit-identical values.
    let raw: Vec<f32> = (0..k)
        .map(|i| 1.0f32 / (1.5f32 + (i as f32) * 0.37f32 + ((seed % 7) as f32) * 0.11f32))
        .collect();
    let total: f32 = raw.iter().sum();
    let normalized: Vec<f32> = raw.iter().map(|v| v / total).collect();
    let scores = astype(&from_slice_f32(&normalized, &[k]), dtype::BFLOAT16);
    eval(&scores);

    let (gate_w, gate_s, gate_b) =
        random_quantized_expert_stack(num_experts, dff, din, group_size, bits);
    let (up_w, up_s, up_b) = random_quantized_expert_stack(num_experts, dff, din, group_size, bits);
    let (down_w, down_s, down_b) =
        random_quantized_expert_stack(num_experts, din, dff, group_size, bits);

    GegluMoeCase {
        din,
        dff,
        k,
        group_size,
        bits,
        x,
        indices,
        scores,
        gate_w,
        gate_s,
        gate_b,
        up_w,
        up_s,
        up_b,
        down_w,
        down_s,
        down_b,
    }
}

/// Invoke the fused two-dispatch GeGLU kernel exactly as
/// `SwitchGeGLU::forward_fused_kernel` does and force evaluation.
fn run_fused(case: &GegluMoeCase) -> UniquePtr<MlxArray> {
    let out = fused_moe_geglu_kernel(
        &case.x,
        &case.indices,
        &case.gate_w,
        &case.gate_s,
        &case.gate_b,
        &case.up_w,
        &case.up_s,
        &case.up_b,
        &case.down_w,
        &case.down_s,
        &case.down_b,
        &case.scores,
        case.din,
        case.dff,
        case.k,
        case.bits,
        case.bits,
        case.group_size,
    );
    eval(&out);
    out
}

/// All-f32 dense reference on the identical bf16 inputs:
/// `sum_k scores[k] * down_k( gelu_tanh_approx(gate_k(x)) * up_k(x) )` with
/// dequantized weights and f32 matmuls, isolating the fused kernel's own
/// numeric error from input-quantization rounding.
fn reference_dense_f32(case: &GegluMoeCase) -> UniquePtr<MlxArray> {
    // Dequantize with f32 scales/biases so the dequantized weights stay in
    // f32. `dequantize` emits the scales dtype, and both the fused kernel and
    // `gather_qmm` compute `w * scale + bias` inline in f32 registers, so a
    // bf16-dequantized reference would round every weight to bf16 and carry
    // MORE error than either production path (bf16 -> f32 is exact, so the
    // upcast does not change the stored values).
    let dq = |w: &MlxArray, s: &MlxArray, b: &MlxArray| -> UniquePtr<MlxArray> {
        let s32 = astype(s, dtype::FLOAT32);
        let b32 = astype(b, dtype::FLOAT32);
        unsafe {
            dequantize(
                w,
                &s32,
                &b32 as &MlxArray as *const MlxArray,
                case.group_size,
                case.bits,
                "affine",
            )
        }
    };
    let gate_dq = dq(&case.gate_w, &case.gate_s, &case.gate_b); // [E, dff, din]
    let up_dq = dq(&case.up_w, &case.up_s, &case.up_b); // [E, dff, din]
    let down_dq = dq(&case.down_w, &case.down_s, &case.down_b); // [E, din, dff]

    let gate_sel = take(&gate_dq, &case.indices, 0); // [k, dff, din]
    let up_sel = take(&up_dq, &case.indices, 0);
    let down_sel = take(&down_dq, &case.indices, 0); // [k, din, dff]

    let x_col = reshape(&astype(&case.x, dtype::FLOAT32), &[case.din, 1]);
    let g = squeeze_axis(&matmul(&gate_sel, &x_col), -1); // [k, dff]
    let u = squeeze_axis(&matmul(&up_sel, &x_col), -1); // [k, dff]
    let act = compiled_geglu_approx_activation(&g, &u); // [k, dff] f32

    let act_col = reshape(&act, &[case.k, case.dff, 1]);
    let per_expert = squeeze_axis(&matmul(&down_sel, &act_col), -1); // [k, din]

    let w_col = reshape(&astype(&case.scores, dtype::FLOAT32), &[case.k, 1]);
    let weighted = multiply(&per_expert, &w_col);
    let out = sum_axis(&weighted, 0, false); // [din]
    eval(&out);
    out
}

/// Production fallback reference: three `gather_qmm` calls + tanh-approx GeGLU
/// on the bf16 pipeline, shaped like `SwitchGeGLU::forward` decode
/// (`x [1, 1, 1, din]`, `rhs_indices [1, k]`), then score-weighted and summed.
fn reference_gather_qmm(case: &GegluMoeCase) -> UniquePtr<MlxArray> {
    let x4 = reshape(&case.x, &[1, 1, 1, case.din]);
    let idx2 = reshape(&case.indices, &[1, case.k]);
    let gq = |w: &MlxArray, s: &MlxArray, b: &MlxArray, x: &MlxArray| -> UniquePtr<MlxArray> {
        unsafe {
            gather_qmm(
                x,
                w,
                s,
                b as *const MlxArray,
                std::ptr::null(),
                &idx2 as &MlxArray as *const MlxArray,
                true,
                case.group_size,
                case.bits,
                false,
                "affine",
            )
        }
    };
    let gate = gq(&case.gate_w, &case.gate_s, &case.gate_b, &x4); // [1, 1, k, dff]
    let up = gq(&case.up_w, &case.up_s, &case.up_b, &x4);
    let act = compiled_geglu_approx_activation(&gate, &up);
    let down = gq(&case.down_w, &case.down_s, &case.down_b, &act); // [1, 1, k, din]

    let per_expert = reshape(&down, &[case.k, case.din]);
    let w_col = reshape(&astype(&case.scores, dtype::FLOAT32), &[case.k, 1]);
    let weighted = multiply(&astype(&per_expert, dtype::FLOAT32), &w_col);
    let out = sum_axis(&weighted, 0, false);
    eval(&out);
    out
}

fn flatten_f32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn raw_bytes(arr: &MlxArray) -> Vec<u8> {
    eval(arr);
    array_to_raw_bytes(arr)
}

/// RMS of (a - b) normalized by the RMS of b, plus the max absolute deviation
/// normalized the same way. Returns (nrms, nmax).
fn normalized_deviation(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut max_abs = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        diff_sq += d * d;
        ref_sq += (*y as f64) * (*y as f64);
        max_abs = max_abs.max(d.abs());
    }
    let ref_rms = (ref_sq / b.len() as f64).sqrt().max(1e-20);
    (
        (diff_sq / a.len() as f64).sqrt() / ref_rms,
        max_abs / ref_rms,
    )
}

fn gpu_backend_or_skip() -> Option<&'static str> {
    if crate::metal_is_available() {
        Some("metal")
    } else if crate::cuda_is_available() {
        Some("cuda")
    } else {
        // The fused kernels JIT through mx.fast.metal_kernel /
        // mx.fast.cuda_kernel; a CPU-only build cannot launch either body.
        None
    }
}

/// Gemma-4-26b-a4b decode-shape parity: fused kernel vs the all-f32 dense
/// reference and vs the production `gather_qmm` fallback.
///
/// The fused kernel accumulates every stage in f32 and (post-#886) rounds to
/// the activation dtype exactly once, at the very end. Against the dense f32
/// reference rounded the same single time, the only legitimate differences
/// are f32 summation order (~1e-6 relative) and sub-ulp rounding ties, so the
/// bound is tight: any real math defect (a dropped lane, a wrong group, an
/// extra rounding stage) shows up orders of magnitude above it.
#[test]
fn fused_moe_geglu_kernel_matches_references_gemma4_shape() {
    let Some(backend) = gpu_backend_or_skip() else {
        return;
    };
    // Din/Dff/K/bits/group_size exactly as gemma-4-26b-a4b; E cut to 16.
    for seed in [886u64, 887, 888] {
        let case = build_geglu_case(seed, 2816, 704, 16, 8);
        let fused = flatten_f32(&run_fused(&case));
        let dense_raw = reference_dense_f32(&case);
        let dense = flatten_f32(&dense_raw);
        // The fused output has been rounded to bf16 once; apply the identical
        // rounding to the dense reference before comparing so the assertion
        // measures kernel math, not the unavoidable output-dtype quantum.
        let dense_bf16 = flatten_f32(&astype(&dense_raw, dtype::BFLOAT16));
        let gather = flatten_f32(&reference_gather_qmm(&case));

        let (nrms_dense, nmax_dense) = normalized_deviation(&fused, &dense_bf16);
        let (nrms_gather, nmax_gather) = normalized_deviation(&fused, &gather);
        let (nrms_ref_jitter, _) = normalized_deviation(&gather, &dense);
        println!(
            "fused_moe_geglu parity [{backend}, seed {seed}]: vs dense f32 (bf16-rounded) \
             nrms={nrms_dense:e} nmax={nmax_dense:e}; vs gather_qmm nrms={nrms_gather:e} \
             nmax={nmax_gather:e}; gather-vs-dense baseline nrms={nrms_ref_jitter:e}"
        );
        assert!(
            nrms_dense < 5e-4 && nmax_dense < 2e-2,
            "fused vs bf16-rounded dense f32 reference deviates (seed {seed}): \
             nrms={nrms_dense:e} nmax={nmax_dense:e}"
        );
        // gather_qmm rounds gate/up/activation through bf16 between GEMMs, so
        // the mutual deviation sits in the documented fp16-jitter class.
        assert!(
            nrms_gather < 5e-3 && nmax_gather < 5e-2,
            "fused vs gather_qmm reference deviates (seed {seed}): nrms={nrms_gather:e} nmax={nmax_gather:e}"
        );
    }
}

/// CPU-side structural guard on the kernel sources themselves (no GPU
/// needed): the warp reductions must keep full-warp masks and the complete
/// 16..1 offset ladder, and the down kernel must emit f32 partials (the #886
/// faithfulness fix). Catches silent regressions of the reduction/rounding
/// structure in review, long before a GPU parity run.
#[test]
fn fused_moe_kernel_source_structure() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/cpp/mlx_cxx_kernels.cpp"
    ))
    .expect("kernel source file must be readable");

    // Every CUDA warp shuffle in this file must use the full-warp mask; a
    // partial mask under warp-uniform control flow is exactly the class of
    // bug #886 told us to guard against.
    let mut shfl_count = 0usize;
    for (off, _) in src.match_indices("__shfl_down_sync(") {
        shfl_count += 1;
        let tail = &src[off..off + 64];
        assert!(
            tail.starts_with("__shfl_down_sync(0xffffffffu"),
            "__shfl_down_sync without a full-warp mask at byte {off}: {tail}"
        );
    }
    assert!(
        shfl_count >= 3,
        "expected the fused-MoE CUDA reductions to use __shfl_down_sync (found {shfl_count})"
    );
    // The reduction ladder must start at offset 16 (covers all 32 lanes).
    assert!(
        src.contains("for (int o = 16; o > 0; o >>= 1)"),
        "warp reduction ladder must cover offsets 16..1"
    );
    // The down kernels (CUDA and Metal) must write f32 partials, not round
    // per-expert outputs to the activation dtype before the host K-sum.
    assert_eq!(
        src.matches("out[eslot * Din + h] = (float)scores[eslot] * d;")
            .count(),
        2,
        "both MOE_DOWN kernel twins must emit f32 partials (score in f32)"
    );
    assert!(
        !src.contains("out[eslot * Din + h] = (T)("),
        "MOE_DOWN must not round per-expert partials to the activation dtype"
    );
}

/// Repeated invocations on identical inputs must be byte-identical, including
/// with allocator churn in between (recycled dirty buffers are what expose
/// uninitialized reads or missing synchronization as visible corruption).
#[test]
fn fused_moe_geglu_kernel_bitwise_deterministic_across_reruns() {
    let Some(backend) = gpu_backend_or_skip() else {
        return;
    };
    let case = build_geglu_case(4242, 2816, 704, 16, 8);
    let baseline = raw_bytes(&run_fused(&case));
    for iter in 0..24 {
        // Dirty the allocator pool with a large transient allocation so a
        // kernel bug that reads unwritten memory sees garbage, not zeros.
        let churn = unsafe { random_normal(&[4 * 1024 * 1024], dtype::FLOAT32, std::ptr::null()) };
        eval(&churn);
        drop(churn);

        let out = raw_bytes(&run_fused(&case));
        assert_eq!(
            out, baseline,
            "fused GeGLU kernel output diverged from baseline on rerun {iter} [{backend}]: \
             run-to-run non-determinism (issue #886)"
        );
    }
}

/// `MLXCEL_FUSED_MOE_SGY` only regroups rows into threadgroups; each row's
/// 32-lane reduction is identical at every setting, so outputs must be
/// byte-identical across the full SGY range (including non-divisor values
/// that force `round_up` padding rows). Divergence proves a launch-geometry
/// race, precision has nothing to do with it.
#[test]
fn fused_moe_geglu_kernel_bitwise_invariant_across_sgy() {
    let Some(backend) = gpu_backend_or_skip() else {
        return;
    };
    let _guard = crate::test_support::env_lock::env_lock();
    let saved = std::env::var("MLXCEL_FUSED_MOE_SGY").ok();

    // Gemma4 decode shape plus a smaller shape whose Dff (192) is not a
    // multiple of several SGY values, exercising the guard rows added by
    // round_up in the dispatch.
    let cases = [
        build_geglu_case(886, 2816, 704, 16, 8),
        build_geglu_case(31, 320, 192, 16, 4),
    ];
    for (ci, case) in cases.iter().enumerate() {
        unsafe { std::env::remove_var("MLXCEL_FUSED_MOE_SGY") };
        let baseline = raw_bytes(&run_fused(case));
        for sgy in [1, 2, 3, 5, 8, 16, 32] {
            unsafe { std::env::set_var("MLXCEL_FUSED_MOE_SGY", sgy.to_string()) };
            for rep in 0..3 {
                let out = raw_bytes(&run_fused(case));
                assert_eq!(
                    out, baseline,
                    "fused GeGLU kernel output changed with SGY={sgy} (case {ci}, rep {rep}) \
                     [{backend}]: launch-geometry-dependent result (issue #886)"
                );
            }
        }
    }

    match saved {
        Some(v) => unsafe { std::env::set_var("MLXCEL_FUSED_MOE_SGY", v) },
        None => unsafe { std::env::remove_var("MLXCEL_FUSED_MOE_SGY") },
    }
}
