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

//! Accumulation precision of MLX's CPU quantized matmul (issue #2248).
//!
//! Without Accelerate's SIMD header (every Linux build, x86 macOS), MLX's
//! `simd::max_size` is 1 and every quantized matmul on the CPU stream runs the
//! scalar loops in `mlx/backend/cpu/quantized.cpp`. Upstream they keep the
//! running sum in the activation type, so a bf16 projection summed its 2048
//! products in an 8-bit mantissa. The overlay in
//! `src/lib/mlx-cpp/patches/mlx/backend/cpu/quantized.cpp` accumulates in
//! f32 and rounds once on store.
//!
//! Each case runs on the CPU stream against a host f64 reference built from
//! the exact values the device holds (activations, scales and biases read back
//! after the dtype cast). The bound for a 16-bit output is the error of
//! rounding that reference once to the output dtype: an f32 accumulator lands
//! within it, a 16-bit accumulator does not. f32 activations must stay below
//! 1e-5 relative L2.
//!
//!   cargo test -p mlxcel-core --lib cpu_qmm_accum_tests -- --nocapture

use std::time::Instant;

use super::*;
use crate::streams::{DefaultDeviceGuard, lock_default_device};

/// Deterministic pseudo-random stream (64-bit LCG), so the host reference and
/// the device inputs come from the same numbers without an RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn flatten_f32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn flatten_u8(arr: &MlxArray) -> Vec<u8> {
    eval(arr);
    array_to_raw_bytes(arr)
}

/// Code `i` of a packed row: bits `[bits * i, bits * i + bits)` of the row's
/// little-endian byte stream, which is how MLX packs every affine width and
/// the FP4 nibbles (low nibble first).
fn code(row: &[u8], bits: usize, i: usize) -> u32 {
    let start = bits * i;
    let mut v = 0u32;
    for b in 0..bits {
        let bit = start + b;
        v |= u32::from((row[bit / 8] >> (bit % 8)) & 1) << b;
    }
    v
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// `||got - want|| / ||want||`.
fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length");
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).powi(2))
        .sum();
    let den: f64 = want.iter().map(|w| w * w).sum();
    (num / den.max(1e-30)).sqrt()
}

/// The error of storing `want` once in `dtype`: the floor any correctly
/// accumulated result in that dtype sits at.
fn rounding_floor(want: &[f64], dtype: i32) -> f64 {
    let as_f32: Vec<f32> = want.iter().map(|v| *v as f32).collect();
    let stored = astype(&from_slice_f32(&as_f32, &[as_f32.len() as i32]), dtype);
    rel_l2(&flatten_f32(&stored), want)
}

fn dtype_name(dtype: i32) -> &'static str {
    match dtype {
        dtype::BFLOAT16 => "bf16",
        dtype::FLOAT16 => "f16",
        dtype::FLOAT32 => "f32",
        _ => "other",
    }
}

/// One `quantized_matmul` call's inputs: the prepared arrays and the tag
/// arguments, so `run_qmm` stays under the clippy argument cap.
struct Qmm<'a> {
    x: &'a MlxArray,
    w: &'a MlxArray,
    scales: &'a MlxArray,
    biases: Option<&'a MlxArray>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: &'a str,
}

/// Run `quantized_matmul` on prepared inputs, timing only the matmul.
fn run_qmm(q: &Qmm) -> (Vec<f32>, f64) {
    eval(q.x);
    eval(q.w);
    eval(q.scales);
    if let Some(b) = q.biases {
        eval(b);
    }
    let biases_ptr = q.biases.map_or(std::ptr::null(), |b| b as *const MlxArray);
    let start = Instant::now();
    // SAFETY: `biases_ptr` is null or points at an array borrowed for this call.
    let y = unsafe {
        quantized_matmul(
            q.x,
            q.w,
            q.scales,
            biases_ptr,
            q.transpose,
            q.group_size,
            q.bits,
            q.mode,
        )
    };
    eval(&y);
    let ms = start.elapsed().as_secs_f64() * 1e3;
    assert_eq!(array_dtype(&y), array_dtype(q.x), "output dtype follows x");
    (flatten_f32(&y), ms)
}

/// Check the measured error against the dtype's bound and print it. Returns
/// the failure, so a test reports every dtype before it fails.
fn check(what: &str, x_dtype: i32, got: &[f32], want: &[f64], ms: f64) -> Option<String> {
    let err = rel_l2(got, want);
    let name = dtype_name(x_dtype);
    if x_dtype == dtype::FLOAT32 {
        eprintln!("{what} {name}: rel L2 {err:.3e} ({ms:.1} ms)");
        (err >= 1e-5).then(|| format!("{what} {name}: rel L2 {err:.3e} >= 1e-5"))
    } else {
        let floor = rounding_floor(want, x_dtype);
        eprintln!(
            "{what} {name}: rel L2 {err:.3e}, store-rounding floor {floor:.3e}, ratio {:.2} ({ms:.1} ms)",
            err / floor
        );
        (err > 2.0 * floor).then(|| {
            format!(
                "{what} {name}: rel L2 {err:.3e} exceeds 2x the store-rounding floor {floor:.3e}; \
                 the CPU quantized matmul is accumulating in the activation dtype (#2248)"
            )
        })
    }
}

fn assert_all_within(failures: Vec<Option<String>>) {
    let failures: Vec<String> = failures.into_iter().flatten().collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// An affine weight with `rows` packed rows of `cols` codes each, plus the
/// host copies of what the device holds.
struct Affine {
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: UniquePtr<MlxArray>,
    bytes: Vec<u8>,
    scales_host: Vec<f32>,
    biases_host: Vec<f32>,
    cols: usize,
    bits: usize,
    g: usize,
}

impl Affine {
    fn new(rng: &mut Lcg, rows: usize, cols: usize, bits: usize, g: usize, dtype: i32) -> Self {
        let words: Vec<u32> = (0..rows * cols * bits / 32)
            .map(|_| rng.next_u32())
            .collect();
        let groups = cols / g;
        let levels = ((1u32 << bits) - 1) as f32;
        // Weights in about [-0.05, 0.05]: scale * code + bias with the code
        // range centred, like a real checkpoint's group.
        let s: Vec<f32> = (0..rows * groups)
            .map(|_| (0.06 + 0.04 * rng.next_f32()) / levels)
            .collect();
        let b: Vec<f32> = s
            .iter()
            .map(|v| -0.5 * levels * v + 0.005 * rng.next_f32())
            .collect();
        let w = from_slice_u32(&words, &[rows as i32, (cols * bits / 32) as i32]);
        let scales = astype(&from_slice_f32(&s, &[rows as i32, groups as i32]), dtype);
        let biases = astype(&from_slice_f32(&b, &[rows as i32, groups as i32]), dtype);
        let scales_host = flatten_f32(&scales);
        let biases_host = flatten_f32(&biases);
        Self {
            w,
            scales,
            biases,
            bytes: words_to_bytes(&words),
            scales_host,
            biases_host,
            cols,
            bits,
            g,
        }
    }

    /// Dequantized element `(r, c)` in f64.
    fn weight(&self, r: usize, c: usize) -> f64 {
        let row_bytes = self.cols * self.bits / 8;
        let row = &self.bytes[r * row_bytes..(r + 1) * row_bytes];
        let gi = r * (self.cols / self.g) + c / self.g;
        f64::from(code(row, self.bits, c)) * f64::from(self.scales_host[gi])
            + f64::from(self.biases_host[gi])
    }
}

fn activations(rng: &mut Lcg, m: usize, k: usize, dtype: i32) -> (UniquePtr<MlxArray>, Vec<f32>) {
    let v: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
    let x = astype(&from_slice_f32(&v, &[m as i32, k as i32]), dtype);
    let host = flatten_f32(&x);
    (x, host)
}

/// `x [m, k] @ W^T` with `W [n, k]`, the transposed layout every linear layer
/// uses (`_qmm_t` upstream).
fn affine_transposed_case(
    m: usize,
    k: usize,
    n: usize,
    bits: usize,
    g: usize,
    x_dtype: i32,
) -> Option<String> {
    let mut rng = Lcg(0x2248 + bits as u64);
    let t = Affine::new(&mut rng, n, k, bits, g, x_dtype);
    let (x, x_host) = activations(&mut rng, m, k, x_dtype);
    let mut want = vec![0f64; m * n];
    for o in 0..n {
        let row: Vec<f64> = (0..k).map(|i| t.weight(o, i)).collect();
        for r in 0..m {
            let xr = &x_host[r * k..(r + 1) * k];
            want[r * n + o] = row.iter().zip(xr).map(|(w, x)| w * f64::from(*x)).sum();
        }
    }
    let (got, ms) = run_qmm(&Qmm {
        x: &x,
        w: &t.w,
        scales: &t.scales,
        biases: Some(&t.biases),
        transpose: true,
        group_size: g as i32,
        bits: bits as i32,
        mode: "affine",
    });
    let what = format!("affine transposed [{m}x{k}]x[{n}x{k}]^T bits={bits} g={g}");
    check(&what, x_dtype, &got, &want, ms)
}

/// `x [m, k] @ W` with `W [k, n]` (`_qmm` upstream).
fn affine_plain_case(m: usize, k: usize, n: usize, x_dtype: i32) -> Option<String> {
    let (bits, g) = (4, 64);
    let mut rng = Lcg(0x2248_0001);
    let t = Affine::new(&mut rng, k, n, bits, g, x_dtype);
    let (x, x_host) = activations(&mut rng, m, k, x_dtype);
    let mut want = vec![0f64; m * n];
    for kk in 0..k {
        for o in 0..n {
            let w = t.weight(kk, o);
            for r in 0..m {
                want[r * n + o] += f64::from(x_host[r * k + kk]) * w;
            }
        }
    }
    let (got, ms) = run_qmm(&Qmm {
        x: &x,
        w: &t.w,
        scales: &t.scales,
        biases: Some(&t.biases),
        transpose: false,
        group_size: g as i32,
        bits: bits as i32,
        mode: "affine",
    });
    let what = format!("affine plain [{m}x{k}]x[{k}x{n}] bits={bits} g={g}");
    check(&what, x_dtype, &got, &want, ms)
}

/// The FP4 (E2M1) code table MLX decodes MXFP4 with.
const FP4_LUT: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// MXFP4 `x [m, k] @ W^T` (`fp_qmm_t` upstream): group 32, E8M0 scales.
fn mxfp4_transposed_case(m: usize, k: usize, n: usize, x_dtype: i32) -> Option<String> {
    let g = 32;
    let mut rng = Lcg(0x2248_0002);
    let words: Vec<u32> = (0..n * k / 8).map(|_| rng.next_u32()).collect();
    // E8M0 exponents around 2^-6, so weights land in about [-0.1, 0.1].
    let exps: Vec<u32> = (0..n * k / g).map(|_| 119 + rng.next_u32() % 4).collect();
    let w = from_slice_u32(&words, &[n as i32, (k / 8) as i32]);
    let scales = astype(
        &from_slice_u32(&exps, &[n as i32, (k / g) as i32]),
        dtype::UINT8,
    );
    let scale_bytes = flatten_u8(&scales);
    let bytes = words_to_bytes(&words);
    let (x, x_host) = activations(&mut rng, m, k, x_dtype);
    let mut want = vec![0f64; m * n];
    for o in 0..n {
        let row = &bytes[o * k / 2..(o + 1) * k / 2];
        let wrow: Vec<f64> = (0..k)
            .map(|i| {
                let e = i32::from(scale_bytes[o * (k / g) + i / g]) - 127;
                FP4_LUT[code(row, 4, i) as usize] * 2f64.powi(e)
            })
            .collect();
        for r in 0..m {
            let xr = &x_host[r * k..(r + 1) * k];
            want[r * n + o] = wrow.iter().zip(xr).map(|(w, x)| w * f64::from(*x)).sum();
        }
    }
    let (got, ms) = run_qmm(&Qmm {
        x: &x,
        w: &w,
        scales: &scales,
        biases: None,
        transpose: true,
        group_size: g as i32,
        bits: 4,
        mode: "mxfp4",
    });
    let what = format!("mxfp4 transposed [{m}x{k}]x[{n}x{k}]^T");
    check(&what, x_dtype, &got, &want, ms)
}

const DTYPES: [i32; 3] = [dtype::BFLOAT16, dtype::FLOAT16, dtype::FLOAT32];

/// The issue's shape: one decode row through a 2048x2048 4-bit, group-64
/// projection.
#[test]
fn cpu_affine_qmm_t_accumulates_in_f32() {
    let _lock = lock_default_device();
    let _cpu = DefaultDeviceGuard::cpu();
    assert_all_within(
        DTYPES
            .iter()
            .map(|&d| affine_transposed_case(1, 2048, 2048, 4, 64, d))
            .collect(),
    );
}

/// The 3/6-bit unpacking branch of the same loop, 8-bit, and M above 1.
#[test]
fn cpu_affine_qmm_t_odd_widths_accumulate_in_f32() {
    let _lock = lock_default_device();
    let _cpu = DefaultDeviceGuard::cpu();
    assert_all_within(vec![
        affine_transposed_case(4, 2048, 256, 3, 64, dtype::BFLOAT16),
        affine_transposed_case(4, 2048, 256, 6, 64, dtype::BFLOAT16),
        affine_transposed_case(8, 1024, 256, 8, 128, dtype::BFLOAT16),
    ]);
}

#[test]
fn cpu_affine_qmm_plain_accumulates_in_f32() {
    let _lock = lock_default_device();
    let _cpu = DefaultDeviceGuard::cpu();
    assert_all_within(
        DTYPES
            .iter()
            .map(|&d| affine_plain_case(2, 2048, 512, d))
            .collect(),
    );
}

#[test]
fn cpu_mxfp4_qmm_t_accumulates_in_f32() {
    let _lock = lock_default_device();
    let _cpu = DefaultDeviceGuard::cpu();
    assert_all_within(
        DTYPES
            .iter()
            .map(|&d| mxfp4_transposed_case(1, 2048, 1024, d))
            .collect(),
    );
}
