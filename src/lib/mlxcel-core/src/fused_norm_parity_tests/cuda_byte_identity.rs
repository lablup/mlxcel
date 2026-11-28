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

//! The CUDA fused kernel is byte-identical to the CUDA graph (issue #3510).
//!
//! Before #3510 the kernel swept the row with `fused_norm_threads(dim)` lanes
//! of four elements, while MLX's `rms_norm` picks its block and vector width
//! per dim and dtype, so the fp32 sum of squares was reduced in another order
//! and a `normed` element could land one or two bf16 ulps away (#3501). The
//! kernel now takes MLX's geometry (`fused_norm_cuda_block` in
//! `kernels/fused_norm_cuda.h`), and this module pins the result: both outputs
//! equal `graph_add_rms_norm` byte for byte.
//!
//! The dims reach every branch of MLX's dispatch (`dispatch_num_chunks` and
//! the `eval_gpu` threshold in `mlx/backend/cuda/rms_norm.cu`) for both vector
//! widths, 8 elements for a 16-bit compute dtype and 4 for f32:
//!
//! - one 64-thread chunk: 1, 3, 100, 128, 512 (f16/bf16) and up to 256 (f32);
//! - 128 threads, whole chunks: 2048, 3072, 4096, 8192 and 512 at f32;
//! - 64 threads, several chunks: 1536, 2560, 3584 (N = 8) and 768, 1280,
//!   1792 (N = 4);
//! - 128 threads with a ragged tail: 513, 1000, 2049 and 4097 at 16 bits;
//! - the looped 1024-thread kernel: 4097 and wider at f32, 8193 and 12288 at
//!   16 bits.
//!
//! A mixed pair computes in `promote_types(activation, weight)`, as
//! `fast::rms_norm` does, and writes the activation dtype, so its reference is
//! the graph output cast to the activation dtype.

use super::*;

const F32: i32 = dtype::FLOAT32;
const F16: i32 = dtype::FLOAT16;
const BF16: i32 = dtype::BFLOAT16;

/// Every (activation, weight) pair the fused path accepts.
const PAIRS: &[(i32, i32)] = &[
    (F32, F32),
    (F16, F16),
    (BF16, BF16),
    (F16, F32),
    (BF16, F32),
    (F32, F16),
    (F32, BF16),
    (F16, BF16),
];

const DIMS: &[i32] = &[
    1, 3, 100, 128, 512, 513, 768, 1000, 1280, 1536, 1792, 2048, 2049, 2560, 3072, 3584, 4096,
    4097, 8192, 8193, 12288,
];

fn dtype_name(dt: i32) -> &'static str {
    match dt {
        F32 => "f32",
        F16 => "f16",
        _ => "bf16",
    }
}

fn element_bytes(dt: i32) -> usize {
    if dt == F32 { 4 } else { 2 }
}

/// Which weight tensor the graph reference normalizes with.
#[derive(Clone, Copy)]
enum Gain {
    /// `weight_bias = 0` against `RMSNorm`.
    Plain,
    /// `weight_bias = 1` against `GemmaRMSNorm`'s precomputed `(1 + w)`.
    Gemma,
}

/// Both paths on the same evaluated inputs: `(fused, graph)`, each as
/// `(normed, new_residual)`, with the graph `normed` cast to the activation
/// dtype (a no-op for a same-dtype pair).
fn both_paths(
    delta: &MlxArray,
    residual: &MlxArray,
    weight: &MlxArray,
    eps: f32,
    gain: Gain,
) -> [(UniquePtr<MlxArray>, UniquePtr<MlxArray>); 2] {
    let act = array_dtype(delta);
    let (fused, graph) = match gain {
        Gain::Plain => (
            run_fused(delta, residual, weight, eps, 0.0),
            crate::layers::graph_add_rms_norm(&RMSNorm::new(copy(weight), eps), delta, residual),
        ),
        Gain::Gemma => {
            let gemma = GemmaRMSNorm::new(copy(weight), eps);
            (
                run_fused(delta, residual, gemma.raw_norm_weight(), eps, 1.0),
                crate::layers::graph_add_rms_norm(&gemma, delta, residual),
            )
        }
    };
    let (graph_normed, graph_residual) = graph;
    let graph_normed = astype(&graph_normed, act);
    eval(&graph_normed);
    eval(&graph_residual);
    [fused, (graph_normed, graph_residual)]
}

/// Rows and elements of `got` whose bytes differ from `want`, or `None` when
/// the two are identical. A summary, not the buffers: a failing case is
/// megabytes wide.
fn byte_difference(got: &MlxArray, want: &MlxArray, dim: i32) -> Option<String> {
    if array_dtype(got) != array_dtype(want) {
        return Some(format!(
            "dtype {} against {}",
            array_dtype(got),
            array_dtype(want)
        ));
    }
    let (g, w) = (raw_bytes(got), raw_bytes(want));
    if g == w {
        return None;
    }
    if g.len() != w.len() {
        return Some(format!("{} bytes against {}", g.len(), w.len()));
    }
    let row_bytes = dim as usize * element_bytes(array_dtype(got));
    let elements = g
        .chunks_exact(element_bytes(array_dtype(got)))
        .zip(w.chunks_exact(element_bytes(array_dtype(got))))
        .filter(|(a, b)| a != b)
        .count();
    let rows = g
        .chunks(row_bytes)
        .zip(w.chunks(row_bytes))
        .filter(|(a, b)| a != b)
        .count();
    Some(format!(
        "{elements} elements in {rows} of {} rows differ",
        g.len() / row_bytes
    ))
}

/// Run one case and append a line to `failures` for each output that is not
/// byte-identical to the graph.
fn check(
    failures: &mut Vec<String>,
    label: &str,
    delta: &MlxArray,
    residual: &MlxArray,
    weight: &MlxArray,
    eps: f32,
    gain: Gain,
) {
    let dim = *array_shape(delta).last().expect("activations have a width");
    let [(normed, new_residual), (want_normed, want_residual)] =
        both_paths(delta, residual, weight, eps, gain);
    if let Some(d) = byte_difference(&new_residual, &want_residual, dim) {
        failures.push(format!("{label}: new_residual: {d}"));
    }
    if let Some(d) = byte_difference(&normed, &want_normed, dim) {
        failures.push(format!("{label}: normed: {d}"));
    }
}

fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} cases differ from the CUDA graph:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn cuda_or_skip() -> Option<crate::streams::DefaultDeviceLock> {
    use crate::hardware::{GpuBackendKind, gpu_backend_kind};
    if gpu_backend_kind() != GpuBackendKind::Cuda {
        return None;
    }
    gpu_kernel_or_skip()
}

#[test]
fn fused_add_rms_norm_is_byte_identical_to_the_cuda_graph() {
    let Some(_device) = cuda_or_skip() else {
        return;
    };
    let mut failures = Vec::new();

    // The #3501 inputs: 5 of these 24576 bf16 rows differed before #3510.
    for seed in 0..6u64 {
        let (delta, residual, weight) = random_case(7000 + seed, 4096, 2048, BF16);
        check(
            &mut failures,
            &format!("#3501 seed {seed} bf16 rows=4096 dim=2048"),
            &delta,
            &residual,
            &weight,
            EPS,
            Gain::Plain,
        );
    }

    for (p, &(act, w)) in PAIRS.iter().enumerate() {
        let pair = format!("{}/{}", dtype_name(act), dtype_name(w));
        for (i, &dim) in DIMS.iter().enumerate() {
            for &rows in &[1, 33] {
                let seed = 8000 + (p * 100 + i) as u64 * 2 + u64::from(rows == 33);
                let (delta, residual, weight) = random_case_with_weight(seed, rows, dim, act, w);
                check(
                    &mut failures,
                    &format!("{pair} rows={rows} dim={dim}"),
                    &delta,
                    &residual,
                    &weight,
                    EPS,
                    Gain::Plain,
                );
            }
        }
        // The Gemma fold and a second eps, on one dim of each dispatch branch.
        for (i, &dim) in [128, 1536, 2048, 2049, 4097, 8193].iter().enumerate() {
            let (delta, residual, weight) =
                random_case_with_weight(9000 + (p * 10 + i) as u64, 5, dim, act, w);
            check(
                &mut failures,
                &format!("{pair} gemma rows=5 dim={dim}"),
                &delta,
                &residual,
                &weight,
                EPS,
                Gain::Gemma,
            );
            check(
                &mut failures,
                &format!("{pair} eps=1e-5 rows=5 dim={dim}"),
                &delta,
                &residual,
                &weight,
                1e-5,
                Gain::Plain,
            );
        }
        // Wide dynamic range: per-row residual scales exp(2 * N(0, 1)), where
        // a normalizer one ulp off shows up on about 1% of rows.
        for &dim in &[2048, 4096] {
            let rows = 1024;
            let (delta, residual, weight) =
                random_case_with_weight(9500 + p as u64, rows, dim, act, w);
            random_seed(9600 + p as u64);
            let noise = unsafe { random_normal(&[rows, 1], F32, std::ptr::null()) };
            let row_scale = exp(&multiply(&noise, &full_f32(&[1], 2.0, F32)));
            let residual = astype(&multiply(&astype(&residual, F32), &row_scale), act);
            eval(&residual);
            check(
                &mut failures,
                &format!("{pair} spread row scales rows={rows} dim={dim}"),
                &delta,
                &residual,
                &weight,
                EPS,
                Gain::Plain,
            );
        }
    }
    assert_no_failures(&failures);
}

/// Signed zeros: an element that underflows in the activation dtype, and a
/// zero weight times a negative element. The graph multiplies in the compute
/// dtype, which keeps IEEE's sign of zero. Same inputs as the ROCm pin.
#[test]
fn fused_add_rms_norm_keeps_the_cuda_graph_sign_of_zero() {
    let Some(_device) = cuda_or_skip() else {
        return;
    };
    let dim = 128usize;
    let mut residual = vec![0f32; dim];
    let mut weight = vec![1f32; dim];
    // One large element sets a small normalizer, so the tiny ones underflow.
    residual[0] = 1000.0;
    let tiny = [-0.0f32, 0.0, -1e-6, -5e-8, -3e-8, -1e-4, 1e-4, -6e-5, -1e-7];
    for (i, t) in tiny.iter().enumerate() {
        residual[1 + i] = *t;
    }
    for i in 20..40 {
        weight[i] = 0.0;
        residual[i] = -0.5;
    }
    for i in 40..60 {
        weight[i] = -0.0;
        residual[i] = 0.5;
    }
    let mut failures = Vec::new();
    for &(act, w) in PAIRS {
        let r = astype(&from_slice_f32(&residual, &[1, dim as i32]), act);
        let d = astype(&from_slice_f32(&vec![0f32; dim], &[1, dim as i32]), act);
        let wt = astype(&from_slice_f32(&weight, &[dim as i32]), w);
        eval(&r);
        eval(&d);
        eval(&wt);
        check(
            &mut failures,
            &format!("{}/{} signed zeros", dtype_name(act), dtype_name(w)),
            &d,
            &r,
            &wt,
            EPS,
            Gain::Plain,
        );
    }
    assert_no_failures(&failures);
}

/// NaN with NaN, an infinity with the same-signed infinity, anything finite
/// bit for bit (a 16-bit value widens to f32 exactly, so f32 bits decide).
fn same_class_exact(got: f32, want: f32) -> bool {
    if got.is_nan() || want.is_nan() {
        got.is_nan() && want.is_nan()
    } else {
        got.to_bits() == want.to_bits()
    }
}

/// Non-finite inputs follow IEEE propagation on both paths: a NaN makes its
/// whole normalized row NaN, an infinity makes the sum of squares infinite
/// and its reciprocal root zero, so finite elements normalize to signed zero
/// and the infinite one to NaN. NaN payloads are not part of the contract, so
/// those rows compare by class; the finite row between them stays bit-exact.
#[test]
fn fused_add_rms_norm_propagates_non_finite_rows_like_the_cuda_graph() {
    let Some(_device) = cuda_or_skip() else {
        return;
    };
    let rows = 4usize;
    let mut failures = Vec::new();
    for (p, &(act, w)) in PAIRS.iter().enumerate() {
        for &dim in &[1024usize, 2049, 8193] {
            let (delta, residual, weight) =
                random_case_with_weight(9800 + p as u64, rows as i32, dim as i32, act, w);
            let mut x = flatten_f32(&delta);
            x[1] = f32::NAN;
            x[2 * dim] = f32::INFINITY;
            x[3 * dim + dim - 1] = f32::NEG_INFINITY;
            let delta = astype(&from_slice_f32(&x, &[rows as i32, dim as i32]), act);
            eval(&delta);
            let [(normed, new_residual), (want_normed, want_residual)] =
                both_paths(&delta, &residual, &weight, EPS, Gain::Plain);
            let label = format!("{}/{} dim={dim}", dtype_name(act), dtype_name(w));
            for (name, got, want) in [
                ("new_residual", &new_residual, &want_residual),
                ("normed", &normed, &want_normed),
            ] {
                if array_dtype(got) != array_dtype(want) {
                    failures.push(format!("{label}: {name}: dtypes differ"));
                    continue;
                }
                let (g, wv) = (flatten_f32(got), flatten_f32(want));
                let off = g
                    .iter()
                    .zip(&wv)
                    .filter(|(a, b)| !same_class_exact(**a, **b))
                    .count();
                if off != 0 {
                    failures.push(format!("{label}: {name}: {off} elements differ in class"));
                }
            }
            // The row without a non-finite input is an ordinary row.
            let finite = dim * element_bytes(act)..2 * dim * element_bytes(act);
            if raw_bytes(&normed)[finite.clone()] != raw_bytes(&want_normed)[finite] {
                failures.push(format!("{label}: normed: the finite row differs"));
            }
        }
    }
    assert_no_failures(&failures);
}
