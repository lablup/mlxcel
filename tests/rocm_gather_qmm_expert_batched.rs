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

//! The expert-batched `gather_qmm` kernel on ROCm matches the unsorted path
//! and a dequantized f32 reference (issue #2066, `patches-rocm/LOCAL_FIXES.md`
//! item 9).
//!
//! `GatherQMM::eval_gpu` in `patches-rocm/mlx/backend/rocm/quantized/qmm.hip`
//! sends a sorted, transposed, affine, group-size-64, 4- or 8-bit call with
//! `M == 1`, `B >= 64`, `E <= 64` and `B / E >= 4` to
//! `gather_qmv_expert_batched_kernel`, which loads each expert's weights once
//! for all of that expert's rows. Every case below satisfies that gate. The
//! sorted call runs that kernel, and the same inputs with
//! `sorted_indices = false` run the kernel the unsorted path uses (the wide
//! or warp-shared gather qmv).
//!
//! The kernel used to read `lhs_indices[b]` and `rhs_indices[b]` as flat
//! arrays. That holds for the `[B]` indices `SwitchGLU`'s sorted path builds,
//! but not when MLX broadcasts the indices against the activation's batch
//! shape: `x` as `[T, 1, K]` with sorted `rhs_indices` of shape `[T, 1]`
//! broadcasts to a `[T, T]` batch whose rhs strides are `(1, 0)` and whose
//! implicit lhs strides are `(0, 1)`, and the kernel read both arrays past
//! their end. That is the call that found item 9 (bf16 only because only bf16
//! reached the kernel). The `Broadcast*` cases below are those shapes.
//!
//! Each case checks:
//!
//! * the sorted result's relative L2 error against `x @ dequantize(w).T` in f32
//!   stays within the unsorted path's own error against that reference (with
//!   a little slack for the different summation order), the bar the issue sets
//!   for enabling the kernel by default, and both stay under a bound that
//!   rounding to the activation dtype alone meets;
//! * two sorted runs agree bit for bit.
//!
//! Shapes: a Mixtral-like layer (8 experts, top 2, `K = 4096`, with an output
//! width of 516 so the last column block is partial) and a 64-expert layer at
//! `granite-4.0-h-tiny`'s widths (top 6, its `K = 1536` gate/up projection and
//! its `K = 512` down projection), each with 4- and 8-bit weights and bf16 and
//! f16 activations. The reference is computed per expert with dense f32
//! matmuls on the GPU from weights dequantized to f32, so it shares no kernel
//! with the quantized paths under test.
//!
//! ```sh
//! cargo test --release --features rocm --test rocm_gather_qmm_expert_batched -- --test-threads=1
//! ```
//!
//! The kernel is forced on through `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=1`,
//! which `GatherQMM::eval_gpu` reads on every call. Every test moves the
//! process-global default device, so each holds
//! `streams::lock_default_device` for its whole body. Skips on any other
//! backend.

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

const GROUP_SIZE: i32 = 64;
const MODE: &str = "affine";
const ENV: &str = "MLX_ROCM_GATHER_QMV_EXPERT_BATCHED";

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn force_expert_batched(on: bool) {
    // SAFETY: `set_var` mutates the process-global environment. Every test in
    // this binary holds `lock_default_device` while it sets the variable and
    // while MLX reads it during evaluation, and nothing else in the process
    // touches the environment concurrently.
    unsafe { std::env::set_var(ENV, if on { "1" } else { "0" }) };
}

fn eval_ok(label: &str, a: &MlxArray) {
    if let Err(err) = mlxcel_core::try_eval(a) {
        panic!("{label}: evaluation failed: {err}");
    }
}

/// Standard normal samples of `shape` in `dt`, generated on the GPU.
fn normal(shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let f32 = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, std::ptr::null()) };
    let out = mlxcel_core::astype(&f32, dt);
    eval_ok("random input", &out);
    out
}

fn dtype_name(dt: i32) -> &'static str {
    match dt {
        dtype::FLOAT16 => "f16",
        dtype::BFLOAT16 => "bf16",
        _ => "?",
    }
}

/// Relative L2 error a correct kernel stays under from rounding the output
/// (and the activations) to `dt`: unit roundoff 2^-9 for bf16 and 2^-11 for
/// f16, with room for where the partial sums are rounded. Measured on gfx1151:
/// about 2.3e-3 (bf16) and 2.8e-4 (f16) on both paths.
fn absolute_bound(dt: i32) -> f32 {
    match dt {
        dtype::FLOAT16 => 2e-3,
        dtype::BFLOAT16 => 1e-2,
        _ => unreachable!("untested dtype {dt}"),
    }
}

/// `||got - want|| / ||want||`, in f32.
fn relative_l2(got: &MlxArray, want: &MlxArray) -> f32 {
    let got = mlxcel_core::astype(got, dtype::FLOAT32);
    let want = mlxcel_core::astype(want, dtype::FLOAT32);
    let diff = mlxcel_core::sum_all(&mlxcel_core::square(&mlxcel_core::subtract(&got, &want)));
    let norm = mlxcel_core::sum_all(&mlxcel_core::square(&want));
    eval_ok("error metric", &diff);
    eval_ok("error metric", &norm);
    let (diff, norm) = (mlxcel_core::item_f32(&diff), mlxcel_core::item_f32(&norm));
    assert!(norm > 0.0, "reference is all zeros");
    (diff / norm).sqrt()
}

/// One MoE projection shape.
#[derive(Debug, Clone, Copy)]
struct Shape {
    name: &'static str,
    experts: i32,
    top_k: i32,
    tokens: i32,
    k: i32,
    n: i32,
}

const SHAPES: [Shape; 3] = [
    Shape {
        name: "mixtral-like",
        experts: 8,
        top_k: 2,
        tokens: 64,
        k: 4096,
        n: 516,
    },
    Shape {
        name: "64-expert gate",
        experts: 64,
        top_k: 6,
        tokens: 64,
        k: 1536,
        n: 512,
    },
    Shape {
        name: "64-expert down",
        experts: 64,
        top_k: 6,
        tokens: 64,
        k: 512,
        n: 1536,
    },
];

/// How the activation rows and the sorted expert indices reach the kernel.
#[derive(Debug, Clone, Copy)]
enum Layout {
    /// `SwitchGLU`'s sorted path: `x` gathered by expert, `[B, 1, K]`, and
    /// `rhs` `[B]`.
    Gathered,
    /// One activation row shared by every batch element: `x` `[1, 1, K]`,
    /// `rhs` `[B]` (`x` broadcasts, stride 0).
    Shared,
    /// `x` `[T, 1, 1, K]` with `rhs` `[T, top_k]` sorted in flat order; the
    /// implicit `lhs` broadcasts with strides `(1, 0)`.
    BroadcastRows,
    /// `x` `[T, 1, K]` with `rhs` `[T, 1]` sorted; the batch broadcasts to
    /// `[T, T]`, with rhs strides `(1, 0)` and implicit lhs strides `(0, 1)`.
    BroadcastIndices,
}

/// Deterministic, uneven routing: `count` expert ids in `[0, experts)`, sorted.
fn sorted_experts(count: i32, experts: i32, seed: u32) -> Vec<i32> {
    let mut state = seed;
    let mut out: Vec<i32> = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Squaring the uniform draw skews routing toward low ids, so runs
            // have uneven lengths and some experts get no rows.
            let u = (state >> 8) as f32 / (1u32 << 24) as f32;
            ((u * u) * experts as f32) as i32
        })
        .collect();
    out.sort_unstable();
    out
}

fn u32_array(values: &[i32], shape: &[i32]) -> UniquePtr<MlxArray> {
    let a = mlxcel_core::from_slice_i32(values, shape);
    let a = mlxcel_core::astype(&a, dtype::UINT32);
    eval_ok("indices", &a);
    a
}

/// One case's inputs and, for the reference, which `x` row and which expert
/// every flat batch element uses.
struct Inputs {
    x: UniquePtr<MlxArray>,
    rhs: UniquePtr<MlxArray>,
    x_rows: Vec<i32>,
    experts: Vec<i32>,
    out_shape: Vec<i32>,
}

fn inputs(shape: Shape, layout: Layout, dt: i32, seed: u32) -> Inputs {
    let Shape {
        experts,
        top_k,
        tokens,
        k,
        n,
        ..
    } = shape;
    match layout {
        Layout::Gathered => {
            let b = tokens * top_k;
            let e = sorted_experts(b, experts, seed);
            Inputs {
                x: normal(&[b, 1, k], dt),
                rhs: u32_array(&e, &[b]),
                x_rows: (0..b).collect(),
                experts: e,
                out_shape: vec![b, 1, n],
            }
        }
        Layout::Shared => {
            let b = tokens * top_k;
            let e = sorted_experts(b, experts, seed);
            Inputs {
                x: normal(&[1, 1, k], dt),
                rhs: u32_array(&e, &[b]),
                x_rows: vec![0; b as usize],
                experts: e,
                out_shape: vec![b, 1, n],
            }
        }
        Layout::BroadcastRows => {
            let b = tokens * top_k;
            let e = sorted_experts(b, experts, seed);
            Inputs {
                x: normal(&[tokens, 1, 1, k], dt),
                rhs: u32_array(&e, &[tokens, top_k]),
                x_rows: (0..b).map(|i| i / top_k).collect(),
                experts: e,
                out_shape: vec![tokens, top_k, 1, n],
            }
        }
        Layout::BroadcastIndices => {
            // T * T batch elements; T = 16 gives B = 256, which keeps
            // B / E >= 4 for 64 experts.
            let t = 16;
            let per_row = sorted_experts(t, experts, seed);
            let mut x_rows = Vec::with_capacity((t * t) as usize);
            let mut e = Vec::with_capacity((t * t) as usize);
            for &expert in &per_row {
                for j in 0..t {
                    x_rows.push(j);
                    e.push(expert);
                }
            }
            Inputs {
                x: normal(&[t, 1, k], dt),
                rhs: u32_array(&per_row, &[t, 1]),
                x_rows,
                experts: e,
                out_shape: vec![t, t, 1, n],
            }
        }
    }
}

type Quantized = (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
);

fn gather_qmm(
    label: &str,
    inp: &Inputs,
    quant: &Quantized,
    bits: i32,
    sorted: bool,
) -> UniquePtr<MlxArray> {
    let (packed, scales, biases) = quant;
    let y = unsafe {
        mlxcel_core::gather_qmm(
            &inp.x,
            packed,
            scales,
            &**biases as *const MlxArray,
            std::ptr::null(),
            &*inp.rhs as *const MlxArray,
            true,
            GROUP_SIZE,
            bits,
            sorted,
            MODE,
        )
    };
    eval_ok(label, &y);
    y
}

/// `x[x_rows[b]] @ dense[experts[b]].T` for every flat `b`, in f32, as one
/// dense matmul per run of equal experts (`experts` is sorted).
fn reference(inp: &Inputs, dense: &MlxArray, k: i32, n: i32) -> UniquePtr<MlxArray> {
    let x_flat = mlxcel_core::reshape(&mlxcel_core::astype(&inp.x, dtype::FLOAT32), &[-1, k]);
    let rows = mlxcel_core::from_slice_i32(&inp.x_rows, &[inp.x_rows.len() as i32]);
    let x_ref = mlxcel_core::take(&x_flat, &rows, 0);
    let mut parts: Vec<UniquePtr<MlxArray>> = Vec::new();
    let mut start = 0usize;
    while start < inp.experts.len() {
        let e = inp.experts[start];
        let mut end = start;
        while end < inp.experts.len() && inp.experts[end] == e {
            end += 1;
        }
        let x_run = mlxcel_core::slice(&x_ref, &[start as i32, 0], &[end as i32, k]);
        let w_e = mlxcel_core::reshape(
            &mlxcel_core::slice(dense, &[e, 0, 0], &[e + 1, n, k]),
            &[n, k],
        );
        parts.push(mlxcel_core::matmul(&x_run, &mlxcel_core::transpose(&w_e)));
        start = end;
    }
    let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
    let y = mlxcel_core::concatenate_many(&refs, 0);
    let y = mlxcel_core::reshape(&y, &inp.out_shape);
    eval_ok("reference", &y);
    y
}

fn check_shape(shape: Shape, bits: i32, dt: i32) {
    let Shape { experts, k, n, .. } = shape;
    // Weights quantized from activation-dtype values, so scales and biases
    // carry the activation dtype as they do in a checkpoint.
    let w = normal(&[experts, n, k], dt);
    let quant: Quantized = {
        let q = mlxcel_core::quantize_weights_with_mode(&w, GROUP_SIZE, bits, MODE);
        let quant = (
            mlxcel_core::quantized_weights_w(&q),
            mlxcel_core::quantized_weights_scales(&q),
            mlxcel_core::quantized_weights_biases(&q),
        );
        eval_ok("quantize", &quant.0);
        eval_ok("quantize", &quant.1);
        eval_ok("quantize", &quant.2);
        quant
    };
    let dense = {
        let d = unsafe {
            mlxcel_core::dequantize(&quant.0, &quant.1, &*quant.2, GROUP_SIZE, bits, MODE)
        };
        let d = mlxcel_core::astype(&d, dtype::FLOAT32);
        eval_ok("dequantize", &d);
        d
    };

    let layouts = [
        Layout::Gathered,
        Layout::Shared,
        Layout::BroadcastRows,
        Layout::BroadcastIndices,
    ];
    for (i, layout) in layouts.into_iter().enumerate() {
        let inp = inputs(shape, layout, dt, 0x2066 + i as u32);
        let b = inp.experts.len() as i32;
        assert!(
            b >= 64 && experts <= 64 && b / experts >= 4,
            "{}: {layout:?} must satisfy the expert-batched gate",
            shape.name
        );
        let label = format!(
            "{} {bits}-bit {} {layout:?} (B={b}, E={experts}, K={k}, N={n})",
            shape.name,
            dtype_name(dt)
        );

        force_expert_batched(true);
        let batched = gather_qmm(&format!("{label} sorted"), &inp, &quant, bits, true);
        let again = gather_qmm(&format!("{label} sorted rerun"), &inp, &quant, bits, true);
        let unsorted = gather_qmm(&format!("{label} unsorted"), &inp, &quant, bits, false);
        assert_eq!(
            mlxcel_core::array_shape(&batched),
            inp.out_shape,
            "{label}: shape"
        );
        assert_eq!(mlxcel_core::array_dtype(&batched), dt, "{label}: dtype");
        assert!(
            mlxcel_core::array_to_raw_bytes(&batched) == mlxcel_core::array_to_raw_bytes(&again),
            "{label}: two sorted runs differ"
        );

        let want = reference(&inp, &dense, k, n);
        let err_batched = relative_l2(&batched, &want);
        let err_unsorted = relative_l2(&unsorted, &want);
        eprintln!("{label}: expert-batched {err_batched:.3e}, unsorted {err_unsorted:.3e}");
        assert!(
            err_unsorted <= absolute_bound(dt),
            "{label}: the unsorted path's relative error {err_unsorted} exceeds {}",
            absolute_bound(dt)
        );
        assert!(
            err_batched <= absolute_bound(dt),
            "{label}: expert-batched relative error {err_batched} exceeds {}",
            absolute_bound(dt)
        );
        // Both round an f32 accumulation to `dt`; only the summation order
        // differs, so the sorted kernel must be as close to the reference as
        // the unsorted one, up to that order.
        assert!(
            err_batched <= err_unsorted * 1.25,
            "{label}: expert-batched relative error {err_batched} is above the unsorted \
             path's {err_unsorted}"
        );
    }
}

#[test]
fn expert_batched_gather_qmm_matches_unsorted_and_reference() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _gpu = DefaultDeviceGuard::gpu();
    mlxcel_core::random_seed(2066);
    for shape in SHAPES {
        for bits in [4, 8] {
            for dt in [dtype::BFLOAT16, dtype::FLOAT16] {
                check_shape(shape, bits, dt);
            }
        }
    }
    force_expert_batched(false);
}
