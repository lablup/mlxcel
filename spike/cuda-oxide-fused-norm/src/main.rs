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

/*
 * Standalone cuda-oxide fused residual-add + RMSNorm (issue #2832).
 *
 * This crate has no MLX dependency. It compiles the fused kernel to PTX/cubin
 * with the pinned cuda-oxide toolchain and executes it directly through the
 * CUDA runtime. Three subcommands:
 *
 *   run     [--out DIR]      execute the case matrix on device 0, check every
 *                            element against a host model, write fixtures and
 *                            device outputs to DIR for the MLX oracle
 *   compare --fixtures DIR --oracle DIR --out results.json [--note k=v ...]
 *                            compare the device outputs against the MLX oracle
 *                            dumps (existing fused CUDA kernel + graph path)
 *                            under the frozen per-dtype tolerances
 *   bench   [--out bench.json] [--iters N]
 *                            cold and warm timings with event boundaries
 *
 * Reference semantics: `src/lib/mlxcelverse/kernels/fused_norm.cpp`
 * (`FUSED_ADD_RMS_NORM_CUDA_SOURCE`) and
 * `src/lib/mlxcel-core/src/fused_norm_parity_tests.rs`. The kernel keeps the
 * same block-per-row mapping, the same four-wide lane sweep, the same
 * rounding points (residual sum rounded to the activation dtype before it is
 * squared, `x * inv_mean` rounded before the gain multiply, the weight bias
 * folded in the weight dtype), f32 accumulation, and the same two-stage
 * butterfly reduction order, so the device output is expected to match the
 * existing fused CUDA kernel bit for bit. That identity is reported, not
 * assumed: the pass criterion is the parity budget from the tests above.
 */

#![feature(f16)]
#![allow(clippy::too_many_arguments)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{fma_rn_f32, rsqrt_approx_f32};
use cuda_device::{SharedArray, cuda_module, kernel, thread, warp};

// ---------------------------------------------------------------------------
// Launch geometry: the reference launcher's rule (`fused_norm_threads`).
// ---------------------------------------------------------------------------

/// Lanes per SIMD group (warp).
const SIMD_WIDTH: u32 = 32;
/// Contiguous elements each lane reads per sweep.
const N_READS: u32 = 4;
/// Largest block the kernel is written for (32 warps, `LOCAL_SUMS` has 32 slots).
const MAX_THREADS: u32 = 1024;

/// Threads per row: enough lanes to cover the row at `N_READS` each, rounded up
/// to a whole warp, clamped to `[SIMD_WIDTH, MAX_THREADS]`. Rows wider than
/// `MAX_THREADS * N_READS` loop. This is `fused_norm_threads` in
/// `kernels/fused_norm.cpp`, so a comparison against the existing kernel uses
/// the same block for the same width.
fn fused_norm_threads(dim: u32) -> u32 {
    if dim == 0 {
        return SIMD_WIDTH;
    }
    let lanes = dim.div_ceil(N_READS);
    let threads = lanes.div_ceil(SIMD_WIDTH) * SIMD_WIDTH;
    threads.clamp(SIMD_WIDTH, MAX_THREADS)
}

// ---------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------

/// Load/store policy for one lane type. The ABI carries 16-bit activations as
/// `u16` bit patterns (IEEE half or bfloat16); the marker selects the
/// conversion, so one generic row routine serves every dtype pair.
trait Conv<R: Copy> {
    fn load(raw: R) -> f32;
    fn store(value: f32) -> R;
}

struct F32Conv;
struct F16Conv;
struct Bf16Conv;

impl Conv<f32> for F32Conv {
    #[inline(always)]
    fn load(raw: f32) -> f32 {
        raw
    }
    #[inline(always)]
    fn store(value: f32) -> f32 {
        value
    }
}

impl Conv<u16> for F16Conv {
    #[inline(always)]
    fn load(raw: u16) -> f32 {
        f16::from_bits(raw) as f32
    }
    #[inline(always)]
    fn store(value: f32) -> u16 {
        (value as f16).to_bits()
    }
}

impl Conv<u16> for Bf16Conv {
    #[inline(always)]
    fn load(raw: u16) -> f32 {
        bf16_to_f32(raw)
    }
    #[inline(always)]
    fn store(value: f32) -> u16 {
        // Round-to-nearest-even, the same rounding `(T)s` performs for
        // `__nv_bfloat16` in the reference kernel.
        f32_to_bf16_rne(value)
    }
}

/// Butterfly all-reduce over the warp, in the reference kernel's order
/// (`for (o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(.., acc, o)`), so
/// every lane holds the same total and the summation order is identical.
#[inline(always)]
fn warp_sum(mut value: f32) -> f32 {
    let mut offset = 16u32;
    while offset > 0 {
        value += warp::shuffle_xor_f32(value, offset);
        offset >>= 1;
    }
    value
}

/// One block owns one row of `dim` elements. `blockDim.x` lanes sweep the row
/// in `N_READS`-wide contiguous chunks. Two passes, as in the reference:
///
///   1. read `x` and `residual`, sum in f32, round the sum to the activation
///      dtype, store it as `new_residual`, accumulate its square (f32 FMA);
///   2. re-read `new_residual`, scale by `rsqrt(mean + eps)`, round, apply
///      the `(weight_bias + weight)` gain folded in the weight dtype.
///
/// `sums` and `inv_mean` point at block shared memory declared by the kernel
/// wrapper (32 floats and 1 float). Every lane reaches every barrier: the
/// grid is exactly one block per row, so no block returns early.
#[inline(always)]
fn fused_add_rms_norm_row<A: Copy, W: Copy, CA: Conv<A>, CW: Conv<W>>(
    x: &[A],
    residual: &[A],
    weight: &[W],
    eps: f32,
    weight_bias: f32,
    dim: u32,
    normed: &mut [A],
    new_residual: &mut [A],
    sums: *mut f32,
    inv_mean: *mut f32,
) {
    let row = thread::blockIdx_x() as usize;
    let lid = thread::threadIdx_x();
    let lane = (lid % SIMD_WIDTH) as usize;
    let sg = (lid / SIMD_WIDTH) as usize;
    let stride = (thread::blockDim_x() * N_READS) as usize;
    let dim_u = dim as usize;
    let base = row * dim_u;

    // Stage 1: residual sum, store, sum of squares.
    let mut acc = 0.0f32;
    let mut r = 0usize;
    while r < dim_u {
        let d0 = r + (lid as usize) * (N_READS as usize);
        let mut j = 0usize;
        while j < N_READS as usize {
            let d = d0 + j;
            if d < dim_u {
                let s = CA::load(x[base + d]) + CA::load(residual[base + d]);
                let st = CA::store(s);
                new_residual[base + d] = st;
                let sr = CA::load(st);
                // `acc += sr * sr` in the reference; nvcc contracts it to an
                // FMA (`-fmad=true`), so the contraction is explicit here.
                acc = fma_rn_f32(sr, sr, acc);
            }
            j += 1;
        }
        r += stride;
    }

    acc = warp_sum(acc);
    // SAFETY: `sums` has 32 slots and `inv_mean` one; `sg < 32` because the
    // block is at most `MAX_THREADS` lanes, and every access is separated by a
    // block barrier from the writes it depends on.
    unsafe {
        if sg == 0 {
            *sums.add(lane) = 0.0;
        }
        thread::sync_threads();
        if lane == 0 {
            *sums.add(sg) = acc;
        }
        thread::sync_threads();
        if sg == 0 {
            let total = warp_sum(*sums.add(lane));
            if lane == 0 {
                *inv_mean = rsqrt_approx_f32(total / (dim as f32) + eps);
            }
        }
        thread::sync_threads();
    }

    // Stage 2: scale and apply the (weight_bias + weight) gain.
    // SAFETY: `inv_mean` was published by lane 0 of warp 0 before the barrier.
    let scale = unsafe { *inv_mean };
    let mut r = 0usize;
    while r < dim_u {
        let d0 = r + (lid as usize) * (N_READS as usize);
        let mut j = 0usize;
        while j < N_READS as usize {
            let d = d0 + j;
            if d < dim_u {
                let sr = CA::load(new_residual[base + d]);
                let wv = CW::load(weight[d]);
                // Fold the bias in the weight's own precision so
                // `weight_bias = 1` reproduces GemmaRMSNorm's precomputed
                // `(1 + w)` tensor exactly.
                let gain = if weight_bias == 0.0 {
                    wv
                } else {
                    CW::load(CW::store(weight_bias + wv))
                };
                // Round `x * inv_mean` to the activation dtype before the gain
                // multiply: the store order MLX's rms_norm kernel uses.
                let scaled = CA::load(CA::store(sr * scale));
                normed[base + d] = CA::store(gain * scaled);
            }
            j += 1;
        }
        r += stride;
    }
}

/// Five dtype pairs: activations f32 / f16 / bf16 with the weight in the same
/// dtype, plus the two mixed pairs (16-bit activations, f32 weight). Each
/// kernel declares its own block shared memory and hands raw pointers to the
/// shared row routine.
#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! shared_slots {
        () => {{
            static mut LOCAL_SUMS: SharedArray<f32, 32> = SharedArray::UNINIT;
            static mut LOCAL_INV_MEAN: SharedArray<f32, 1> = SharedArray::UNINIT;
            // SAFETY: raw pointers into this block's shared memory, taken
            // without forming a reference to the `static mut`.
            unsafe {
                (
                    SharedArray::as_raw_mut_ptr(&raw mut LOCAL_SUMS),
                    SharedArray::as_raw_mut_ptr(&raw mut LOCAL_INV_MEAN),
                )
            }
        }};
    }

    #[kernel]
    pub fn fused_add_rms_norm_f32(
        x: &[f32],
        residual: &[f32],
        weight: &[f32],
        eps: f32,
        weight_bias: f32,
        dim: u32,
        normed: &mut [f32],
        new_residual: &mut [f32],
    ) {
        let (sums, inv) = shared_slots!();
        fused_add_rms_norm_row::<f32, f32, F32Conv, F32Conv>(
            x,
            residual,
            weight,
            eps,
            weight_bias,
            dim,
            normed,
            new_residual,
            sums,
            inv,
        );
    }

    #[kernel]
    pub fn fused_add_rms_norm_f16(
        x: &[u16],
        residual: &[u16],
        weight: &[u16],
        eps: f32,
        weight_bias: f32,
        dim: u32,
        normed: &mut [u16],
        new_residual: &mut [u16],
    ) {
        let (sums, inv) = shared_slots!();
        fused_add_rms_norm_row::<u16, u16, F16Conv, F16Conv>(
            x,
            residual,
            weight,
            eps,
            weight_bias,
            dim,
            normed,
            new_residual,
            sums,
            inv,
        );
    }

    #[kernel]
    pub fn fused_add_rms_norm_bf16(
        x: &[u16],
        residual: &[u16],
        weight: &[u16],
        eps: f32,
        weight_bias: f32,
        dim: u32,
        normed: &mut [u16],
        new_residual: &mut [u16],
    ) {
        let (sums, inv) = shared_slots!();
        fused_add_rms_norm_row::<u16, u16, Bf16Conv, Bf16Conv>(
            x,
            residual,
            weight,
            eps,
            weight_bias,
            dim,
            normed,
            new_residual,
            sums,
            inv,
        );
    }

    #[kernel]
    pub fn fused_add_rms_norm_f16_wf32(
        x: &[u16],
        residual: &[u16],
        weight: &[f32],
        eps: f32,
        weight_bias: f32,
        dim: u32,
        normed: &mut [u16],
        new_residual: &mut [u16],
    ) {
        let (sums, inv) = shared_slots!();
        fused_add_rms_norm_row::<u16, f32, F16Conv, F32Conv>(
            x,
            residual,
            weight,
            eps,
            weight_bias,
            dim,
            normed,
            new_residual,
            sums,
            inv,
        );
    }

    #[kernel]
    pub fn fused_add_rms_norm_bf16_wf32(
        x: &[u16],
        residual: &[u16],
        weight: &[f32],
        eps: f32,
        weight_bias: f32,
        dim: u32,
        normed: &mut [u16],
        new_residual: &mut [u16],
    ) {
        let (sums, inv) = shared_slots!();
        fused_add_rms_norm_row::<u16, f32, Bf16Conv, F32Conv>(
            x,
            residual,
            weight,
            eps,
            weight_bias,
            dim,
            normed,
            new_residual,
            sums,
            inv,
        );
    }
}

// ---------------------------------------------------------------------------
// Host: dtypes and software conversions (no `f16` host arithmetic needed)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dt {
    F32,
    F16,
    Bf16,
}

impl Dt {
    fn name(self) -> &'static str {
        match self {
            Dt::F32 => "f32",
            Dt::F16 => "f16",
            Dt::Bf16 => "bf16",
        }
    }

    fn parse(s: &str) -> Option<Dt> {
        match s {
            "f32" => Some(Dt::F32),
            "f16" => Some(Dt::F16),
            "bf16" => Some(Dt::Bf16),
            _ => None,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Dt::F32 => 4,
            Dt::F16 | Dt::Bf16 => 2,
        }
    }

    /// Round an f32 to this dtype and back (round-to-nearest-even).
    fn round(self, v: f32) -> f32 {
        match self {
            Dt::F32 => v,
            Dt::F16 => f16_bits_to_f32(f32_to_f16_bits(v)),
            Dt::Bf16 => bf16_bits_to_f32(f32_to_bf16_bits(v)),
        }
    }

    /// Bit pattern of `v` in this dtype, widened to u32.
    fn bits(self, v: f32) -> u32 {
        match self {
            Dt::F32 => v.to_bits(),
            Dt::F16 => f32_to_f16_bits(v) as u32,
            Dt::Bf16 => f32_to_bf16_bits(v) as u32,
        }
    }

    fn decode(self, b: u32) -> f32 {
        match self {
            Dt::F32 => f32::from_bits(b),
            Dt::F16 => f16_bits_to_f32(b as u16),
            Dt::Bf16 => bf16_bits_to_f32(b as u16),
        }
    }

    /// Normalized (RMS, max) deviation budget, from
    /// `fused_norm_parity_tests::tolerance_for`: a small multiple of one ulp
    /// of the dtype relative to the reference RMS.
    fn budget(self) -> (f64, f64) {
        match self {
            Dt::F32 => (1e-6, 1e-5),
            Dt::F16 => (2e-3, 1.2e-2),
            Dt::Bf16 => (1.6e-2, 7e-2),
        }
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        // Quiet NaN with a non-zero payload.
        return sign | 0x7e00 | ((mant >> 13) as u16);
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // may carry into the exponent, including to infinity
    }
    sign | r as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e: i32 = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127 - 15) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    if v.is_nan() {
        return ((x >> 16) | 0x40) as u16;
    }
    let lsb = (x >> 16) & 1;
    ((x.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

// ---------------------------------------------------------------------------
// Host: deterministic inputs
// ---------------------------------------------------------------------------

/// 64-bit LCG, seeded per case so a case cannot pass by sharing data with its
/// neighbour. Uniforms feed a Box-Muller normal: activations are standard
/// normal (the parity tests' `random_normal`), weights sit near 1
/// (`1 + 0.1 * normal`), because a standard-normal weight would make the
/// reference RMS meaningless.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pattern {
    /// Standard-normal activations, near-one weights.
    Normal,
    /// `Normal`, plus a NaN and a +Inf injected into the activations.
    NonFinite,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Pattern::Normal => "normal",
            Pattern::NonFinite => "nonfinite",
        }
    }
}

#[derive(Clone, Debug)]
struct Case {
    name: String,
    act: Dt,
    w: Dt,
    rows: usize,
    dim: usize,
    eps: f32,
    weight_bias: f32,
    pattern: Pattern,
}

impl Case {
    fn new(
        act: Dt,
        w: Dt,
        rows: usize,
        dim: usize,
        eps: f32,
        weight_bias: f32,
        pattern: Pattern,
    ) -> Self {
        let eps_tag = if eps == 1e-6 { "e6" } else { "e5" };
        let bias_tag = if weight_bias == 0.0 { "b0" } else { "b1" };
        let name = format!(
            "{}_{}_r{}_d{}_{}_{}_{}",
            act.name(),
            w.name(),
            rows,
            dim,
            eps_tag,
            bias_tag,
            pattern.name()
        );
        Self {
            name,
            act,
            w,
            rows,
            dim,
            eps,
            weight_bias,
            pattern,
        }
    }

    fn threads(&self) -> u32 {
        fused_norm_threads(self.dim as u32)
    }

    fn elements(&self) -> usize {
        self.rows * self.dim
    }

    fn seed(&self, index: usize) -> u64 {
        (index as u64) * 7919 + (self.rows as u64) * 131 + self.dim as u64
    }
}

/// Activation dtype pairs: same-dtype weights, then the two mixed pairs.
const DTYPE_PAIRS: &[(Dt, Dt)] = &[
    (Dt::F32, Dt::F32),
    (Dt::F16, Dt::F16),
    (Dt::Bf16, Dt::Bf16),
    (Dt::F16, Dt::F32),
    (Dt::Bf16, Dt::F32),
];

/// Small, odd and tail widths, the representative widths 1024 / 4096 / 8192,
/// one and many rows, widths wider than one 1024-lane sweep (4097, 8191, 8192).
const SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (1, 3),
    (1, 31),
    (1, 32),
    (1, 33),
    (2, 100),
    (3, 127),
    (1, 128),
    (2, 129),
    (5, 1000),
    (1, 1024),
    (5, 1024),
    (33, 2048),
    (1, 3072),
    (5, 4096),
    (2, 4097),
    (1, 8192),
    (3, 8191),
    (16, 8192),
];

/// Shapes that also run with the Gemma `weight_bias = 1` fold and with
/// `eps = 1e-5`.
const VARIANT_SHAPES: &[(usize, usize)] = &[(1, 128), (5, 1024), (2, 4097), (16, 8192)];

fn case_matrix() -> Vec<Case> {
    let mut cases = Vec::new();
    for &(act, w) in DTYPE_PAIRS {
        for &(rows, dim) in SHAPES {
            cases.push(Case::new(act, w, rows, dim, 1e-6, 0.0, Pattern::Normal));
        }
        for &(rows, dim) in VARIANT_SHAPES {
            cases.push(Case::new(act, w, rows, dim, 1e-6, 1.0, Pattern::Normal));
        }
        for &(rows, dim) in &VARIANT_SHAPES[1..3] {
            cases.push(Case::new(act, w, rows, dim, 1e-5, 0.0, Pattern::Normal));
        }
        cases.push(Case::new(act, w, 3, 1024, 1e-6, 0.0, Pattern::NonFinite));
    }
    cases
}

struct Inputs {
    x: Vec<f32>,
    residual: Vec<f32>,
    weight: Vec<f32>,
}

/// Inputs already rounded to the case dtypes, so a consumer that stores them
/// as f32 and casts to the dtype (the MLX oracle) sees the exact same bits.
fn inputs(case: &Case, index: usize) -> Inputs {
    let mut rng = Lcg::new(case.seed(index));
    let n = case.elements();
    let mut x: Vec<f32> = (0..n)
        .map(|_| case.act.round(rng.normal() as f32))
        .collect();
    let residual: Vec<f32> = (0..n)
        .map(|_| case.act.round(rng.normal() as f32))
        .collect();
    let weight: Vec<f32> = (0..case.dim)
        .map(|_| case.w.round((1.0 + 0.1 * rng.normal()) as f32))
        .collect();
    if case.pattern == Pattern::NonFinite {
        // Row 0 carries a NaN, the last row a +Inf; rows in between stay finite.
        x[1.min(case.dim - 1)] = f32::NAN;
        let last = (case.rows - 1) * case.dim;
        x[last] = f32::INFINITY;
    }
    Inputs {
        x,
        residual,
        weight,
    }
}

// ---------------------------------------------------------------------------
// Host model (independent of MLX and of the device summation order)
// ---------------------------------------------------------------------------

struct Model {
    normed: Vec<f32>,
    new_residual: Vec<f32>,
    /// Rows whose activations hold a non-finite value: the whole normalized row
    /// is non-finite in every implementation, and `new_residual` is non-finite
    /// exactly where the input is.
    nonfinite_rows: Vec<bool>,
}

/// The same rounding points as the kernel, with the sum of squares and the
/// reciprocal square root in f64 and a sequential summation order, so it is a
/// check on the semantics rather than a copy of the device arithmetic.
fn host_model(case: &Case, inp: &Inputs) -> Model {
    let dim = case.dim;
    let mut normed = vec![0.0f32; case.elements()];
    let mut new_residual = vec![0.0f32; case.elements()];
    let mut nonfinite_rows = vec![false; case.rows];
    for (row, row_flag) in nonfinite_rows.iter_mut().enumerate() {
        let base = row * dim;
        let mut acc = 0.0f64;
        let mut nonfinite = false;
        for d in 0..dim {
            let s = case.act.round(inp.x[base + d] + inp.residual[base + d]);
            new_residual[base + d] = s;
            if !s.is_finite() {
                nonfinite = true;
            }
            acc += (s as f64) * (s as f64);
        }
        *row_flag = nonfinite;
        let inv = 1.0 / (acc / dim as f64 + case.eps as f64).sqrt();
        for d in 0..dim {
            let sr = new_residual[base + d];
            let wv = inp.weight[d];
            let gain = if case.weight_bias == 0.0 {
                wv
            } else {
                case.w.round(case.weight_bias + wv)
            };
            let scaled = case.act.round((sr as f64 * inv) as f32);
            normed[base + d] = case.act.round(gain * scaled);
        }
    }
    Model {
        normed,
        new_residual,
        nonfinite_rows,
    }
}

/// Element classes agree: NaN with NaN, an infinity with the same-signed
/// infinity, finite with finite bit for bit. Used on rows that carry a
/// non-finite input, where IEEE propagation rather than a tolerance decides.
fn same_class_exact(g: f32, w: f32) -> bool {
    if g.is_nan() || w.is_nan() {
        g.is_nan() && w.is_nan()
    } else if g.is_infinite() || w.is_infinite() {
        g == w
    } else {
        g.to_bits() == w.to_bits()
    }
}

/// RMS and max of `a - b`, both divided by the RMS of `b`, over finite rows.
fn normalized_deviation(a: &[f32], b: &[f32], dim: usize, skip_rows: &[bool]) -> (f64, f64, usize) {
    let mut sq = 0.0f64;
    let mut max = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut n = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if skip_rows[i / dim] {
            continue;
        }
        let d = (*x as f64) - (*y as f64);
        sq += d * d;
        max = max.max(d.abs());
        ref_sq += (*y as f64) * (*y as f64);
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let ref_rms = (ref_sq / n as f64).sqrt().max(1e-20);
    ((sq / n as f64).sqrt() / ref_rms, max / ref_rms, n)
}

// ---------------------------------------------------------------------------
// Host: device execution
// ---------------------------------------------------------------------------

struct DeviceOut {
    normed_bits: Vec<u32>,
    new_residual_bits: Vec<u32>,
}

fn to_u16_bits(dt: Dt, values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| dt.bits(*v) as u16).collect()
}

fn launch_config(case: &Case) -> Result<LaunchConfig, String> {
    let rows = u32::try_from(case.rows).map_err(|_| "rows exceed u32".to_string())?;
    let dim = u32::try_from(case.dim).map_err(|_| "dim exceeds u32".to_string())?;
    if rows == 0 || dim == 0 {
        return Err(format!("rows={rows} dim={dim}: both must be positive"));
    }
    if (rows as u64) * (dim as u64) > u32::MAX as u64 {
        return Err("rows * dim exceeds the 32-bit element budget of this harness".to_string());
    }
    Ok(LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (case.threads(), 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Host-side contract: shapes the kernel is not written for are refused with
/// a message before any launch, never narrowed silently.
fn validate_shapes(case: &Case, x: usize, residual: usize, weight: usize) -> Result<(), String> {
    let n = case
        .rows
        .checked_mul(case.dim)
        .ok_or_else(|| "rows * dim overflows".to_string())?;
    if case.rows == 0 || case.dim == 0 {
        return Err(format!(
            "rows={} dim={}: both must be positive",
            case.rows, case.dim
        ));
    }
    if x != n || residual != n {
        return Err(format!(
            "x has {x} and residual {residual} elements; both must be rows*dim={n}"
        ));
    }
    if weight != case.dim {
        return Err(format!(
            "weight has {weight} elements; must equal dim={}",
            case.dim
        ));
    }
    if !case.threads().is_multiple_of(SIMD_WIDTH) || case.threads() > MAX_THREADS {
        return Err(format!(
            "block {} is not a whole number of warps within {MAX_THREADS}",
            case.threads()
        ));
    }
    Ok(())
}

/// Prefix a driver or loader error with what the harness was doing.
fn fail<E: std::fmt::Display>(prefix: String) -> impl FnOnce(E) -> String {
    move |e| format!("{prefix}: {e}")
}

struct Gpu {
    ctx: std::sync::Arc<cuda_core::Context>,
    module: kernels::LoadedModule,
}

impl Gpu {
    fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("failed to create CUDA context: {e}"))?;
        let module = kernels::load(&ctx)
            .map_err(|e| format!("failed to load the cuda-oxide module: {e}"))?;
        Ok(Self { ctx, module })
    }

    /// Upload, launch once, download. Returns the raw output bits.
    fn run_case(&self, case: &Case, inp: &Inputs) -> Result<DeviceOut, String> {
        validate_shapes(case, inp.x.len(), inp.residual.len(), inp.weight.len())?;
        let config = launch_config(case)?;
        let stream = self.ctx.default_stream();
        let dim = case.dim as u32;
        let n = case.elements();

        macro_rules! run_u16_act {
            ($kernel:ident, $w_upload:expr) => {{
                let x = DeviceBuffer::from_host(&stream, &to_u16_bits(case.act, &inp.x))
                    .map_err(fail(format!("{}: upload x", case.name)))?;
                let r = DeviceBuffer::from_host(&stream, &to_u16_bits(case.act, &inp.residual))
                    .map_err(fail(format!("{}: upload residual", case.name)))?;
                let w = $w_upload;
                let mut normed = DeviceBuffer::<u16>::zeroed(&stream, n)
                    .map_err(fail(format!("{}: alloc normed", case.name)))?;
                let mut nr = DeviceBuffer::<u16>::zeroed(&stream, n)
                    .map_err(fail(format!("{}: alloc new_residual", case.name)))?;
                // SAFETY: one block per row with `rows` blocks, `dim` is the
                // row width of every buffer, and the kernel bounds every
                // access by `dim` and the slice lengths.
                unsafe {
                    self.module.$kernel(
                        &stream,
                        config,
                        &x,
                        &r,
                        &w,
                        case.eps,
                        case.weight_bias,
                        dim,
                        &mut normed,
                        &mut nr,
                    )
                }
                .map_err(fail(format!("{}: launch", case.name)))?;
                let normed_bits: Vec<u32> = normed
                    .to_host_vec(&stream)
                    .map_err(fail(format!("{}: download normed", case.name)))?
                    .into_iter()
                    .map(|b| b as u32)
                    .collect();
                let new_residual_bits: Vec<u32> = nr
                    .to_host_vec(&stream)
                    .map_err(fail(format!("{}: download new_residual", case.name)))?
                    .into_iter()
                    .map(|b| b as u32)
                    .collect();
                DeviceOut {
                    normed_bits,
                    new_residual_bits,
                }
            }};
        }

        let out = match (case.act, case.w) {
            (Dt::F32, Dt::F32) => {
                let x = DeviceBuffer::from_host(&stream, &inp.x)
                    .map_err(fail(format!("{}: upload x", case.name)))?;
                let r = DeviceBuffer::from_host(&stream, &inp.residual)
                    .map_err(fail(format!("{}: upload residual", case.name)))?;
                let w = DeviceBuffer::from_host(&stream, &inp.weight)
                    .map_err(fail(format!("{}: upload weight", case.name)))?;
                let mut normed = DeviceBuffer::<f32>::zeroed(&stream, n)
                    .map_err(fail(format!("{}: alloc normed", case.name)))?;
                let mut nr = DeviceBuffer::<f32>::zeroed(&stream, n)
                    .map_err(fail(format!("{}: alloc new_residual", case.name)))?;
                // SAFETY: see `run_u16_act!`.
                unsafe {
                    self.module.fused_add_rms_norm_f32(
                        &stream,
                        config,
                        &x,
                        &r,
                        &w,
                        case.eps,
                        case.weight_bias,
                        dim,
                        &mut normed,
                        &mut nr,
                    )
                }
                .map_err(fail(format!("{}: launch", case.name)))?;
                DeviceOut {
                    normed_bits: normed
                        .to_host_vec(&stream)
                        .map_err(fail(format!("{}: download normed", case.name)))?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect(),
                    new_residual_bits: nr
                        .to_host_vec(&stream)
                        .map_err(fail(format!("{}: download new_residual", case.name)))?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect(),
                }
            }
            (Dt::F16, Dt::F16) => run_u16_act!(
                fused_add_rms_norm_f16,
                DeviceBuffer::from_host(&stream, &to_u16_bits(Dt::F16, &inp.weight))
                    .map_err(fail(format!("{}: upload weight", case.name)))?
            ),
            (Dt::Bf16, Dt::Bf16) => run_u16_act!(
                fused_add_rms_norm_bf16,
                DeviceBuffer::from_host(&stream, &to_u16_bits(Dt::Bf16, &inp.weight))
                    .map_err(fail(format!("{}: upload weight", case.name)))?
            ),
            (Dt::F16, Dt::F32) => run_u16_act!(
                fused_add_rms_norm_f16_wf32,
                DeviceBuffer::from_host(&stream, &inp.weight)
                    .map_err(fail(format!("{}: upload weight", case.name)))?
            ),
            (Dt::Bf16, Dt::F32) => run_u16_act!(
                fused_add_rms_norm_bf16_wf32,
                DeviceBuffer::from_host(&stream, &inp.weight)
                    .map_err(fail(format!("{}: upload weight", case.name)))?
            ),
            (act, w) => {
                return Err(format!(
                    "{}: unsupported dtype pair {}/{}",
                    case.name,
                    act.name(),
                    w.name()
                ));
            }
        };
        if out.normed_bits.len() != n || out.new_residual_bits.len() != n {
            return Err(format!("{}: downloaded a wrong element count", case.name));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Fixture files
// ---------------------------------------------------------------------------

fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_bits(path: &Path, dt: Dt, bits: &[u32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(bits.len() * dt.bytes());
    for b in bits {
        match dt.bytes() {
            4 => bytes.extend_from_slice(&b.to_le_bytes()),
            _ => bytes.extend_from_slice(&(*b as u16).to_le_bytes()),
        }
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn read_bits(path: &Path, dt: Dt, expect: usize) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != expect * dt.bytes() {
        return Err(format!(
            "{}: {} bytes, expected {} ({} x {} bytes)",
            path.display(),
            bytes.len(),
            expect * dt.bytes(),
            expect,
            dt.bytes()
        ));
    }
    Ok(match dt.bytes() {
        4 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect(),
        _ => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c) as u32)
            .collect(),
    })
}

const CASES_TSV_HEADER: &str = "name\tact\tw\trows\tdim\teps\tweight_bias\tpattern\tthreads\tgrid";

fn cases_tsv_line(case: &Case) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{:e}\t{}\t{}\t{}\t{}",
        case.name,
        case.act.name(),
        case.w.name(),
        case.rows,
        case.dim,
        case.eps,
        case.weight_bias,
        case.pattern.name(),
        case.threads(),
        case.rows
    )
}

fn read_cases_tsv(path: &Path) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut cases = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            if line != CASES_TSV_HEADER {
                return Err(format!("{}: unexpected header {line:?}", path.display()));
            }
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 10 {
            return Err(format!(
                "{}: line {}: expected 10 fields",
                path.display(),
                i + 1
            ));
        }
        let bad = |what: &str| format!("{}: line {}: bad {what}", path.display(), i + 1);
        let act = Dt::parse(f[1]).ok_or_else(|| bad("act"))?;
        let w = Dt::parse(f[2]).ok_or_else(|| bad("w"))?;
        let rows: usize = f[3].parse().map_err(|_| bad("rows"))?;
        let dim: usize = f[4].parse().map_err(|_| bad("dim"))?;
        let eps: f32 = f[5].parse().map_err(|_| bad("eps"))?;
        let weight_bias: f32 = f[6].parse().map_err(|_| bad("weight_bias"))?;
        let pattern = match f[7] {
            "normal" => Pattern::Normal,
            "nonfinite" => Pattern::NonFinite,
            _ => return Err(bad("pattern")),
        };
        let case = Case::new(act, w, rows, dim, eps, weight_bias, pattern);
        if case.name != f[0] {
            return Err(format!(
                "{}: line {}: name {} does not match its fields ({})",
                path.display(),
                i + 1,
                f[0],
                case.name
            ));
        }
        cases.push(case);
    }
    Ok(cases)
}

// ---------------------------------------------------------------------------
// Subcommand: run
// ---------------------------------------------------------------------------

fn cmd_run(out_dir: Option<PathBuf>) -> Result<(), String> {
    let gpu = Gpu::new()?;
    let cases = case_matrix();
    let mut elements_checked = 0usize;
    let mut passed = 0usize;
    let mut tsv = String::from(CASES_TSV_HEADER);
    tsv.push('\n');

    for (index, case) in cases.iter().enumerate() {
        let inp = inputs(case, index);
        let model = host_model(case, &inp);
        let out = gpu.run_case(case, &inp)?;
        let got_normed: Vec<f32> = out
            .normed_bits
            .iter()
            .map(|b| case.act.decode(*b))
            .collect();
        let got_residual: Vec<f32> = out
            .new_residual_bits
            .iter()
            .map(|b| case.act.decode(*b))
            .collect();

        // new_residual is one f32 add and one rounding on both sides: exact.
        let mut residual_mismatch = 0usize;
        for (i, (g, w)) in got_residual
            .iter()
            .zip(model.new_residual.iter())
            .enumerate()
        {
            let same = if w.is_nan() {
                g.is_nan()
            } else {
                g.to_bits() == w.to_bits()
            };
            if !same {
                residual_mismatch += 1;
                if residual_mismatch == 1 {
                    eprintln!("{}: new_residual[{i}] device {g:?} model {w:?}", case.name);
                }
            }
        }
        if residual_mismatch != 0 {
            return Err(format!(
                "{}: {residual_mismatch} of {} new_residual elements differ from the host model",
                case.name,
                case.elements()
            ));
        }

        // Rows with a non-finite input follow IEEE propagation, not a
        // tolerance: a NaN anywhere makes the whole normalized row NaN; a +Inf
        // makes the sum of squares infinite, `rsqrt` of it zero, so the finite
        // elements normalize to (signed) zero and the Inf element to NaN. The
        // host model computes the same, so these rows compare class-exact.
        for (row, &nonfinite) in model.nonfinite_rows.iter().enumerate() {
            if !nonfinite {
                continue;
            }
            let base = row * case.dim;
            if let Some(d) = (0..case.dim)
                .find(|d| !same_class_exact(got_normed[base + d], model.normed[base + d]))
            {
                return Err(format!(
                    "{}: row {row} holds a non-finite input; normed[{}] device {:?} vs model {:?}",
                    case.name,
                    base + d,
                    got_normed[base + d],
                    model.normed[base + d]
                ));
            }
        }

        let (nrms, nmax, n) =
            normalized_deviation(&got_normed, &model.normed, case.dim, &model.nonfinite_rows);
        let (rms_tol, max_tol) = case.act.budget();
        if n > 0 && !(nrms < rms_tol && nmax < max_tol) {
            return Err(format!(
                "{}: normed vs host model: normalized rms {nrms:.3e} (tol {rms_tol:.1e}), max {nmax:.3e} (tol {max_tol:.1e})",
                case.name
            ));
        }
        for (i, g) in got_normed.iter().enumerate() {
            if !model.nonfinite_rows[i / case.dim] && !g.is_finite() {
                return Err(format!("{}: normed[{i}] = {g} on a finite row", case.name));
            }
        }

        if let Some(dir) = &out_dir {
            let case_dir = dir.join(&case.name);
            fs::create_dir_all(&case_dir)
                .map_err(|e| format!("mkdir {}: {e}", case_dir.display()))?;
            write_f32(&case_dir.join("x.f32"), &inp.x)?;
            write_f32(&case_dir.join("residual.f32"), &inp.residual)?;
            write_f32(&case_dir.join("weight.f32"), &inp.weight)?;
            write_bits(&case_dir.join("normed.bits"), case.act, &out.normed_bits)?;
            write_bits(
                &case_dir.join("new_residual.bits"),
                case.act,
                &out.new_residual_bits,
            )?;
        }
        tsv.push_str(&cases_tsv_line(case));
        tsv.push('\n');

        elements_checked += case.elements();
        passed += 1;
        println!(
            "PASS case={} act={} w={} rows={} dim={} grid={} block={} elements={} model_nrms={:.3e} model_nmax={:.3e}",
            case.name,
            case.act.name(),
            case.w.name(),
            case.rows,
            case.dim,
            case.rows,
            case.threads(),
            case.elements(),
            nrms,
            nmax
        );
    }

    // Shapes the contract refuses, checked against the same validator the
    // launcher runs, so a silently narrowed contract would show up here.
    let rejections: &[(&str, Case, usize, usize, usize)] = &[
        (
            "zero_dim",
            Case::new(Dt::F32, Dt::F32, 1, 0, 1e-6, 0.0, Pattern::Normal),
            0,
            0,
            0,
        ),
        (
            "zero_rows",
            Case::new(Dt::F32, Dt::F32, 0, 128, 1e-6, 0.0, Pattern::Normal),
            0,
            0,
            128,
        ),
        (
            "x_length",
            Case::new(Dt::F32, Dt::F32, 2, 128, 1e-6, 0.0, Pattern::Normal),
            255,
            256,
            128,
        ),
        (
            "residual_length",
            Case::new(Dt::F32, Dt::F32, 2, 128, 1e-6, 0.0, Pattern::Normal),
            256,
            257,
            128,
        ),
        (
            "weight_length",
            Case::new(Dt::F32, Dt::F32, 2, 128, 1e-6, 0.0, Pattern::Normal),
            256,
            256,
            127,
        ),
    ];
    let mut rejected = 0usize;
    for (label, case, x, r, w) in rejections {
        match validate_shapes(case, *x, *r, *w) {
            Err(reason) => {
                rejected += 1;
                println!("REJECT case={label} reason={reason:?}");
            }
            Ok(()) => {
                return Err(format!(
                    "rejection case {label} was accepted by the shape contract"
                ));
            }
        }
    }

    if let Some(dir) = &out_dir {
        fs::write(dir.join("cases.tsv"), tsv).map_err(|e| format!("write cases.tsv: {e}"))?;
    }
    println!(
        "RESULT standalone=1 mlx=0 cases_passed={passed} cases_total={} rejected={rejected} elements_checked={elements_checked} comparison=host-model",
        cases.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Subcommand: compare (device outputs vs the MLX oracle dumps)
// ---------------------------------------------------------------------------

struct Cmp {
    nrms: f64,
    nmax: f64,
    compared: usize,
    differing: usize,
    max_ulp: u32,
}

/// Element-wise comparison of two bit vectors of the same dtype over finite
/// rows: normalized deviation, count of differing bit patterns and the largest
/// distance in units of the dtype's last place.
fn compare_bits(got: &[u32], want: &[u32], dt: Dt, dim: usize, skip_rows: &[bool]) -> Cmp {
    let gv: Vec<f32> = got.iter().map(|b| dt.decode(*b)).collect();
    let wv: Vec<f32> = want.iter().map(|b| dt.decode(*b)).collect();
    let (nrms, nmax, compared) = normalized_deviation(&gv, &wv, dim, skip_rows);
    let mut differing = 0usize;
    let mut max_ulp = 0u32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if skip_rows[i / dim] || g == w {
            continue;
        }
        differing += 1;
        // Order-preserving integer image of the float bits.
        let key = |b: u32| -> i64 {
            let width = (dt.bytes() * 8) as u32;
            let sign = 1u32 << (width - 1);
            if b & sign != 0 {
                -((b & (sign - 1)) as i64)
            } else {
                b as i64
            }
        };
        let ulp = (key(*g) - key(*w)).unsigned_abs().min(u32::MAX as u64) as u32;
        max_ulp = max_ulp.max(ulp);
    }
    Cmp {
        nrms,
        nmax,
        compared,
        differing,
        max_ulp,
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn cmd_compare(
    fixtures: &Path,
    oracle: &Path,
    out: &Path,
    notes: &[(String, String)],
) -> Result<(), String> {
    let cases = read_cases_tsv(&fixtures.join("cases.tsv"))?;
    let oracle_meta = fs::read_to_string(oracle.join("meta.tsv")).map_err(|e| {
        format!(
            "read {}: {e} (run the mlxcel oracle example first)",
            oracle.join("meta.tsv").display()
        )
    })?;
    let mut entries = Vec::new();
    let mut all_pass = true;
    let mut bitwise_fused = 0usize;
    let mut elements = 0usize;

    for (index, case) in cases.iter().enumerate() {
        let n = case.elements();
        let inp = inputs(case, index);
        let model = host_model(case, &inp);
        let case_dir = fixtures.join(&case.name);
        let ocase = oracle.join(&case.name);
        let ours_normed = read_bits(&case_dir.join("normed.bits"), case.act, n)?;
        let ours_residual = read_bits(&case_dir.join("new_residual.bits"), case.act, n)?;
        let fused_normed = read_bits(&ocase.join("fused_normed.bits"), case.act, n)?;
        let fused_residual = read_bits(&ocase.join("fused_new_residual.bits"), case.act, n)?;
        let graph_normed = read_bits(&ocase.join("graph_normed.bits"), case.act, n)?;
        let graph_residual = read_bits(&ocase.join("graph_new_residual.bits"), case.act, n)?;
        let skip = &model.nonfinite_rows;

        let vs_fused = compare_bits(&ours_normed, &fused_normed, case.act, case.dim, skip);
        let vs_graph = compare_bits(&ours_normed, &graph_normed, case.act, case.dim, skip);
        let res_vs_fused = compare_bits(&ours_residual, &fused_residual, case.act, case.dim, skip);
        let res_vs_graph = compare_bits(&ours_residual, &graph_residual, case.act, case.dim, skip);
        // Reference-vs-reference, so a reader can see how far the existing
        // kernel itself sits from the graph path on the same inputs.
        let fused_vs_graph = compare_bits(&fused_normed, &graph_normed, case.act, case.dim, skip);

        // Rows with a non-finite input: IEEE propagation decides, so every
        // element must fall in the same class (NaN / same-signed Inf / finite
        // bit-exact) in the standalone kernel and in both references.
        let mut nonfinite_ok = true;
        for (row, &nf) in skip.iter().enumerate() {
            if !nf {
                continue;
            }
            let base = row * case.dim;
            for d in 0..case.dim {
                let ours = case.act.decode(ours_normed[base + d]);
                let fused = case.act.decode(fused_normed[base + d]);
                let graph = case.act.decode(graph_normed[base + d]);
                let ours_r = case.act.decode(ours_residual[base + d]);
                let fused_r = case.act.decode(fused_residual[base + d]);
                let graph_r = case.act.decode(graph_residual[base + d]);
                if !same_class_exact(ours, fused)
                    || !same_class_exact(ours, graph)
                    || !same_class_exact(ours_r, fused_r)
                    || !same_class_exact(ours_r, graph_r)
                {
                    if nonfinite_ok {
                        eprintln!(
                            "{}: non-finite row {row} element {d}: ours {ours:?} fused {fused:?} graph {graph:?}",
                            case.name
                        );
                    }
                    nonfinite_ok = false;
                }
            }
        }

        // Frozen criteria (recorded before any oracle run; see README):
        //   normed vs graph and vs fused: the parity budget of the activation dtype;
        //   new_residual vs graph: 0 differing elements for f32, <= 1% for 16-bit;
        //   new_residual vs fused: 0 differing elements (same single rounding);
        //   non-finite rows: class-exact (NaN / same Inf / finite bit-exact) in every path.
        let (rms_tol, max_tol) = case.act.budget();
        let residual_budget = if case.act == Dt::F32 {
            0
        } else {
            vs_graph.compared / 100
        };
        let pass = vs_graph.nrms < rms_tol
            && vs_graph.nmax < max_tol
            && vs_fused.nrms < rms_tol
            && vs_fused.nmax < max_tol
            && res_vs_graph.differing <= residual_budget
            && res_vs_fused.differing == 0
            && nonfinite_ok;
        if !pass {
            all_pass = false;
        }
        if vs_fused.differing == 0 && res_vs_fused.differing == 0 {
            bitwise_fused += 1;
        }
        elements += n;

        println!(
            "COMPARE case={} pass={} fused_bitwise={} fused_diff={} fused_max_ulp={} fused_nrms={:.3e} graph_nrms={:.3e} graph_nmax={:.3e} res_graph_diff={} refs_nrms={:.3e}",
            case.name,
            pass,
            vs_fused.differing == 0 && res_vs_fused.differing == 0,
            vs_fused.differing,
            vs_fused.max_ulp,
            vs_fused.nrms,
            vs_graph.nrms,
            vs_graph.nmax,
            res_vs_graph.differing,
            fused_vs_graph.nrms
        );

        let cmp_json = |c: &Cmp| {
            format!(
                "{{\"compared\": {}, \"differing\": {}, \"max_ulp\": {}, \"normalized_rms\": {:e}, \"normalized_max\": {:e}}}",
                c.compared, c.differing, c.max_ulp, c.nrms, c.nmax
            )
        };
        entries.push(format!(
            "    {{\"case\": {}, \"act\": \"{}\", \"weight\": \"{}\", \"rows\": {}, \"dim\": {}, \"eps\": {:e}, \"weight_bias\": {}, \"pattern\": \"{}\", \"block\": {}, \"grid\": {}, \"elements\": {}, \"pass\": {}, \"budget\": {{\"normalized_rms\": {:e}, \"normalized_max\": {:e}, \"residual_differing\": {}}}, \"normed_vs_fused_cuda\": {}, \"normed_vs_graph\": {}, \"new_residual_vs_fused_cuda\": {}, \"new_residual_vs_graph\": {}, \"fused_cuda_vs_graph\": {}, \"nonfinite_rows\": {}, \"nonfinite_rows_agree\": {}}}",
            json_str(&case.name),
            case.act.name(),
            case.w.name(),
            case.rows,
            case.dim,
            case.eps,
            case.weight_bias,
            case.pattern.name(),
            case.threads(),
            case.rows,
            n,
            pass,
            rms_tol,
            max_tol,
            residual_budget,
            cmp_json(&vs_fused),
            cmp_json(&vs_graph),
            cmp_json(&res_vs_fused),
            cmp_json(&res_vs_graph),
            cmp_json(&fused_vs_graph),
            skip.iter().filter(|b| **b).count(),
            nonfinite_ok
        ));
    }

    let notes_json: Vec<String> = notes
        .iter()
        .map(|(k, v)| format!("    {}: {}", json_str(k), json_str(v)))
        .collect();
    let meta_json: Vec<String> = oracle_meta
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(k, v)| format!("    {}: {}", json_str(k), json_str(v)))
        .collect();
    let json = format!(
        "{{\n  \"results\": \"mlxcel.cuda-oxide-fused-norm.compare/v1\",\n  \"pass\": {},\n  \"cases_total\": {},\n  \"cases_bitwise_identical_to_fused_cuda\": {},\n  \"elements_total\": {},\n  \"criteria\": \"normed within the activation dtype parity budget of both the existing fused CUDA kernel and the graph path; new_residual bit-identical to the fused kernel and within the graph add budget; non-finite rows class-exact in every path\",\n  \"notes\": {{\n{}\n  }},\n  \"oracle\": {{\n{}\n  }},\n  \"cases\": [\n{}\n  ]\n}}\n",
        all_pass,
        cases.len(),
        bitwise_fused,
        elements,
        notes_json.join(",\n"),
        meta_json.join(",\n"),
        entries.join(",\n")
    );
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::write(out, json).map_err(|e| format!("write {}: {e}", out.display()))?;
    println!(
        "RESULT compare pass={all_pass} cases_total={} bitwise_identical_to_fused_cuda={bitwise_fused} elements_total={elements} out={}",
        cases.len(),
        out.display()
    );
    if all_pass {
        Ok(())
    } else {
        Err("at least one case is outside its frozen budget".to_string())
    }
}

// ---------------------------------------------------------------------------
// Subcommand: bench
// ---------------------------------------------------------------------------

const BENCH_SHAPES: &[(usize, usize)] = &[
    (1, 1024),
    (1, 4096),
    (1, 8192),
    (5, 4096),
    (33, 2048),
    (64, 8192),
];

fn cmd_bench(out: Option<PathBuf>, iters: usize) -> Result<(), String> {
    let load_start = Instant::now();
    let gpu = Gpu::new()?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1e3;
    let stream = gpu.ctx.default_stream();
    let mut rows_json = Vec::new();
    let mut first_launch = true;

    for &(act, w) in &[(Dt::F32, Dt::F32), (Dt::Bf16, Dt::Bf16), (Dt::F16, Dt::F16)] {
        for &(rows, dim) in BENCH_SHAPES {
            let case = Case::new(act, w, rows, dim, 1e-6, 0.0, Pattern::Normal);
            let inp = inputs(&case, 0);
            let n = case.elements();
            let config = launch_config(&case)?;
            let dim_u32 = dim as u32;

            // Upload once; time only launches.
            let timings =
                |launch: &mut dyn FnMut() -> Result<(), String>| -> Result<(f64, f64), String> {
                    let start = gpu.ctx.new_event().map_err(fail(String::from("event")))?;
                    let end = gpu.ctx.new_event().map_err(fail(String::from("event")))?;
                    // Cold: the first launch of this kernel in this process (the
                    // module is loaded, nothing has run) when `first_launch`,
                    // otherwise the first launch at this shape.
                    start
                        .record(&stream)
                        .map_err(fail(String::from("record")))?;
                    launch()?;
                    end.record(&stream).map_err(fail(String::from("record")))?;
                    end.synchronize().map_err(fail(String::from("sync")))?;
                    let cold_ms = start
                        .elapsed_time(&end)
                        .map_err(fail(String::from("elapsed")))?
                        as f64;
                    for _ in 0..20 {
                        launch()?;
                    }
                    start
                        .record(&stream)
                        .map_err(fail(String::from("record")))?;
                    for _ in 0..iters {
                        launch()?;
                    }
                    end.record(&stream).map_err(fail(String::from("record")))?;
                    end.synchronize().map_err(fail(String::from("sync")))?;
                    let warm_ms = start
                        .elapsed_time(&end)
                        .map_err(fail(String::from("elapsed")))?
                        as f64
                        / iters as f64;
                    Ok((cold_ms, warm_ms))
                };

            let (cold_ms, warm_ms) = match act {
                Dt::F32 => {
                    let x = DeviceBuffer::from_host(&stream, &inp.x)
                        .map_err(fail(String::from("upload")))?;
                    let r = DeviceBuffer::from_host(&stream, &inp.residual)
                        .map_err(fail(String::from("upload")))?;
                    let wb = DeviceBuffer::from_host(&stream, &inp.weight)
                        .map_err(fail(String::from("upload")))?;
                    let mut normed = DeviceBuffer::<f32>::zeroed(&stream, n)
                        .map_err(fail(String::from("alloc")))?;
                    let mut nr = DeviceBuffer::<f32>::zeroed(&stream, n)
                        .map_err(fail(String::from("alloc")))?;
                    let mut launch = || -> Result<(), String> {
                        // SAFETY: as in `run_case`.
                        unsafe {
                            gpu.module.fused_add_rms_norm_f32(
                                &stream,
                                config,
                                &x,
                                &r,
                                &wb,
                                1e-6,
                                0.0,
                                dim_u32,
                                &mut normed,
                                &mut nr,
                            )
                        }
                        .map_err(fail(String::from("launch")))
                    };
                    timings(&mut launch)?
                }
                Dt::F16 | Dt::Bf16 => {
                    let x = DeviceBuffer::from_host(&stream, &to_u16_bits(act, &inp.x))
                        .map_err(fail(String::from("upload")))?;
                    let r = DeviceBuffer::from_host(&stream, &to_u16_bits(act, &inp.residual))
                        .map_err(fail(String::from("upload")))?;
                    let wb = DeviceBuffer::from_host(&stream, &to_u16_bits(w, &inp.weight))
                        .map_err(fail(String::from("upload")))?;
                    let mut normed = DeviceBuffer::<u16>::zeroed(&stream, n)
                        .map_err(fail(String::from("alloc")))?;
                    let mut nr = DeviceBuffer::<u16>::zeroed(&stream, n)
                        .map_err(fail(String::from("alloc")))?;
                    let mut launch = || -> Result<(), String> {
                        // SAFETY: as in `run_case`.
                        let result = unsafe {
                            if act == Dt::F16 {
                                gpu.module.fused_add_rms_norm_f16(
                                    &stream,
                                    config,
                                    &x,
                                    &r,
                                    &wb,
                                    1e-6,
                                    0.0,
                                    dim_u32,
                                    &mut normed,
                                    &mut nr,
                                )
                            } else {
                                gpu.module.fused_add_rms_norm_bf16(
                                    &stream,
                                    config,
                                    &x,
                                    &r,
                                    &wb,
                                    1e-6,
                                    0.0,
                                    dim_u32,
                                    &mut normed,
                                    &mut nr,
                                )
                            }
                        };
                        result.map_err(fail(String::from("launch")))
                    };
                    timings(&mut launch)?
                }
            };
            // Bytes the kernel moves: x, residual read; new_residual written
            // then re-read; normed written; weight read once per row.
            let bytes = (n * act.bytes() * 5 + rows * dim * w.bytes()) as f64;
            let gbps = bytes / (warm_ms * 1e-3) / 1e9;
            println!(
                "BENCH act={} w={} rows={} dim={} block={} grid={} cold_us={:.1} warm_us={:.2} iters={} effective_GBps={:.1}{}",
                act.name(),
                w.name(),
                rows,
                dim,
                case.threads(),
                rows,
                cold_ms * 1e3,
                warm_ms * 1e3,
                iters,
                gbps,
                if first_launch {
                    " first_launch_in_process=1"
                } else {
                    ""
                }
            );
            rows_json.push(format!(
                "    {{\"act\": \"{}\", \"weight\": \"{}\", \"rows\": {}, \"dim\": {}, \"block\": {}, \"grid\": {}, \"cold_us\": {:.3}, \"warm_us\": {:.3}, \"iters\": {}, \"effective_gbps\": {:.2}, \"first_launch_in_process\": {}}}",
                act.name(),
                w.name(),
                rows,
                dim,
                case.threads(),
                rows,
                cold_ms * 1e3,
                warm_ms * 1e3,
                iters,
                gbps,
                first_launch
            ));
            first_launch = false;
        }
    }
    println!(
        "RESULT bench module_load_ms={load_ms:.2} shapes={}",
        rows_json.len()
    );
    if let Some(path) = out {
        let json = format!(
            "{{\n  \"results\": \"mlxcel.cuda-oxide-fused-norm.bench/v1\",\n  \"module_load_ms\": {load_ms:.3},\n  \"timing\": \"CUDA events recorded on the launch stream around the launches; cold = first launch at the shape (the very first also pays the process's first launch), warm = mean of {iters} launches after 20 warm-up launches; inputs uploaded once per shape\",\n  \"shapes\": [\n{}\n  ]\n}}\n",
            rows_json.join(",\n")
        );
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn usage() -> String {
    "usage:\n  cuda-oxide-fused-norm run [--out DIR]\n  cuda-oxide-fused-norm compare --fixtures DIR --oracle DIR --out results.json [--note key=value ...]\n  cuda-oxide-fused-norm bench [--out bench.json] [--iters N]\n  cuda-oxide-fused-norm threads DIM   (print the block size for a width)".to_string()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        return Err(usage());
    };
    let mut it = args[1..].iter();
    match cmd.as_str() {
        "-h" | "--help" => {
            println!("{}", usage());
            Ok(())
        }
        "threads" => {
            let dim: u32 = it.next().ok_or_else(usage)?.parse().map_err(|_| usage())?;
            println!("{}", fused_norm_threads(dim));
            Ok(())
        }
        "run" => {
            let mut out = None;
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--out" => out = Some(PathBuf::from(it.next().ok_or_else(usage)?)),
                    _ => return Err(format!("unexpected argument {a}\n{}", usage())),
                }
            }
            if let Some(dir) = &out {
                fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
            }
            cmd_run(out)
        }
        "compare" => {
            let (mut fixtures, mut oracle, mut out) = (None, None, None);
            let mut notes = Vec::new();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--fixtures" => fixtures = Some(PathBuf::from(it.next().ok_or_else(usage)?)),
                    "--oracle" => oracle = Some(PathBuf::from(it.next().ok_or_else(usage)?)),
                    "--out" => out = Some(PathBuf::from(it.next().ok_or_else(usage)?)),
                    "--note" => {
                        let kv = it.next().ok_or_else(usage)?;
                        let (k, v) = kv
                            .split_once('=')
                            .ok_or_else(|| format!("--note expects key=value, got {kv}"))?;
                        notes.push((k.to_string(), v.to_string()));
                    }
                    _ => return Err(format!("unexpected argument {a}\n{}", usage())),
                }
            }
            cmd_compare(
                &fixtures.ok_or_else(usage)?,
                &oracle.ok_or_else(usage)?,
                &out.ok_or_else(usage)?,
                &notes,
            )
        }
        "bench" => {
            let mut out = None;
            let mut iters = 200usize;
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--out" => out = Some(PathBuf::from(it.next().ok_or_else(usage)?)),
                    "--iters" => {
                        iters = it.next().ok_or_else(usage)?.parse().map_err(|_| usage())?
                    }
                    _ => return Err(format!("unexpected argument {a}\n{}", usage())),
                }
            }
            if iters == 0 {
                return Err("--iters must be positive".to_string());
            }
            cmd_bench(out, iters)
        }
        other => Err(format!("unknown subcommand {other}\n{}", usage())),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAIL error={error}");
            ExitCode::FAILURE
        }
    }
}
