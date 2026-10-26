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

//! Gather QMV past the 65535-block grid-z limit and past 2^31 output
//! elements (issue #2620, `overlays/rocm/LOCAL_FIXES.md` item 61).
//!
//! `GatherQMM::eval_gpu` puts the batch `B = tokens * top_k` of an `M == 1`
//! call in grid z. The fast tests run `B = 65_537` through each unsorted
//! gather kernel the dispatch picks for a small `N = K = 64` layer of 128
//! experts and compare every output element with a reference computed on the
//! host from the dequantized weights:
//!
//! * bf16, affine 4-bit, group 64: `gather_qmv_wide_kernel`;
//! * bf16, affine 8-bit, group 64: `gather_qmv_warp_shared_kernel`;
//! * bf16, mxfp4, group 32: the warp-shared kernel's mxfp4 word path;
//! * f32, affine 4-bit, group 64: the per-row `gather_qmv_kernel`.
//!
//! The `#[ignore]`d test runs f32 activations with `N = 65_536` and
//! `B = 32_769`, an output of 2^31 + 2^16 elements, through
//! `gather_qmv_kernel`, and checks the rows on both sides of element 2^31.

use super::*;

/// Experts, output width and depth of the fast layer.
const E: i32 = 128;
const N: i32 = 64;
const K: i32 = 64;
/// One past the 65535-block limit of grid z.
const PAST_Z_CAP: i32 = 65_537;

/// An activation value, exact in bf16: multiples of 1/8 in [-1, 1].
fn x_value(i: usize) -> f32 {
    ((i * 7 + 3) % 17) as f32 / 8.0 - 1.0
}

/// A weight value before quantization.
fn w_value(i: usize) -> f32 {
    ((i * 13 + 5) % 23) as f32 / 16.0 - 0.6875
}

/// Unsorted expert ids in `[0, experts)` for `count` batch rows.
fn unsorted_experts(count: i32, experts: i32) -> Vec<i32> {
    let mut state: u32 = 0x2620;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) % experts as u32) as i32
        })
        .collect()
}

fn to_host_f32(a: &MlxArray) -> Vec<f32> {
    let as_f32 = mlxcel_core::astype(a, dtype::FLOAT32);
    mlxcel_core::try_eval(&as_f32).expect("reads back");
    mlxcel_core::array_to_raw_bytes(&as_f32)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

/// A quantized `[experts, n, k]` weight and its dequantized f32 values on the
/// host. The weight is quantized from `dt`, so affine scales and biases come out
/// in `dt`: `gather_qmm` promotes the activations to the affine scales' dtype,
/// and f32 scales would send bf16 activations to the f32 per-row kernel.
struct Weights {
    packed: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: Option<UniquePtr<MlxArray>>,
    dense: Vec<f32>,
    mode: &'static str,
    group_size: i32,
    bits: i32,
}

impl Weights {
    fn new(
        experts: i32,
        n: i32,
        k: i32,
        dt: i32,
        mode: &'static str,
        group_size: i32,
        bits: i32,
    ) -> Self {
        let len = (experts * n * k) as usize;
        // `w_value` is a multiple of 1/16 in [-0.6875, 0.6875], exact in bf16.
        let w = mlxcel_core::reshape(&typed_from(w_value, len, dt), &[experts, n, k]);
        let q = mlxcel_core::quantize_weights_with_mode(&w, group_size, bits, mode);
        let packed = mlxcel_core::quantized_weights_w(&q);
        let scales = mlxcel_core::quantized_weights_scales(&q);
        let biases = mlxcel_core::quantized_weights_has_biases(&q)
            .then(|| mlxcel_core::quantized_weights_biases(&q));
        let biases_ptr = biases
            .as_ref()
            .map_or(std::ptr::null(), |b| &**b as *const MlxArray);
        // SAFETY: every reference outlives the call; `biases_ptr` is null only
        // for mxfp4, where the bridge documents it as nullable.
        let dense = unsafe {
            mlxcel_core::dequantize(&packed, &scales, biases_ptr, group_size, bits, mode)
        };
        let dense = to_host_f32(&dense);
        Weights {
            packed,
            scales,
            biases,
            dense,
            mode,
            group_size,
            bits,
        }
    }

    /// Unsorted `gather_qmm(x, w, rhs_indices = experts)`, transposed.
    fn gather(&self, x: &MlxArray, experts: &MlxArray) -> UniquePtr<MlxArray> {
        let biases_ptr = self
            .biases
            .as_ref()
            .map_or(std::ptr::null(), |b| &**b as *const MlxArray);
        // SAFETY: as in `new`; a null `lhs_indices` selects the implicit lhs.
        unsafe {
            mlxcel_core::gather_qmm(
                x,
                &self.packed,
                &self.scales,
                biases_ptr,
                std::ptr::null(),
                experts as *const MlxArray,
                true,
                self.group_size,
                self.bits,
                false,
                self.mode,
                std::ptr::null(),
            )
        }
    }
}

/// `x[b] @ dense[experts[b]].T` for batch row `b`, with the sum of the terms'
/// magnitudes per element, in f32.
fn host_row(b: usize, experts: &[i32], dense: &[f32], n: usize, k: usize) -> Vec<(f32, f32)> {
    let e = experts[b] as usize;
    (0..n)
        .map(|col| {
            let w = &dense[(e * n + col) * k..(e * n + col + 1) * k];
            (0..k).fold((0.0f32, 0.0f32), |(sum, mag), i| {
                let t = x_value(b * k + i) * w[i];
                (sum + t, mag + t.abs())
            })
        })
        .collect()
}

/// Asserts `got` (row `b`) within rounding of the host reference: the bf16
/// output rounding is at most 2^-9 of the result, under 1% of the terms'
/// magnitude sum. A row the kernel skipped or read with the wrong expert or
/// activation misses by far more.
fn assert_row(what: &str, b: usize, got: &[f32], want: &[(f32, f32)]) {
    for (col, (g, (w, mag))) in got.iter().zip(want).enumerate() {
        let tol = 1e-2 * mag + 1e-6;
        assert!(
            (g - w).abs() <= tol,
            "{what}: row {b} col {col} is {g} (expected {w}, tolerance {tol})"
        );
    }
}

/// Runs the fast case in `dt` with the given scheme at `B = 65_537`.
fn past_z_cap(what: &str, dt: i32, mode: &'static str, group_size: i32, bits: i32) {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let b = PAST_Z_CAP;
    let weights = Weights::new(E, N, K, dt, mode, group_size, bits);
    let x = mlxcel_core::reshape(&typed_from(x_value, (b * K) as usize, dt), &[b, 1, K]);
    let experts = unsorted_experts(b, E);
    let rhs = mlxcel_core::astype(&mlxcel_core::from_slice_i32(&experts, &[b]), dtype::UINT32);
    let out = weights.gather(&x, &rhs);
    // The output dtype is the dtype the kernel ran in; a promotion to f32
    // would mean the dispatch picked the f32 per-row kernel instead.
    assert_eq!(mlxcel_core::array_dtype(&out), dt, "{what}: output dtype");
    if let Err(err) = mlxcel_core::try_eval(&out) {
        panic!("{what}: gather_qmm at B = {b} failed to run: {err}");
    }
    let got = to_host_f32(&out);
    assert_eq!(got.len(), (b * N) as usize, "{what}: output size");
    for row_b in 0..b as usize {
        let want = host_row(row_b, &experts, &weights.dense, N as usize, K as usize);
        let n = N as usize;
        assert_row(what, row_b, &got[row_b * n..(row_b + 1) * n], &want);
    }
}

/// The decisive case of issue #2620: the default bf16 4-bit route.
#[test]
fn gather_qmm_bf16_affine4_past_the_z_cap_matches_host() {
    past_z_cap("bf16 affine 4-bit", dtype::BFLOAT16, "affine", 64, 4);
}

#[test]
fn gather_qmm_bf16_affine8_past_the_z_cap_matches_host() {
    past_z_cap("bf16 affine 8-bit", dtype::BFLOAT16, "affine", 64, 8);
}

#[test]
fn gather_qmm_bf16_mxfp4_past_the_z_cap_matches_host() {
    past_z_cap("bf16 mxfp4", dtype::BFLOAT16, "mxfp4", 32, 4);
}

#[test]
fn gather_qmm_f32_affine4_past_the_z_cap_matches_host() {
    past_z_cap("f32 affine 4-bit", dtype::FLOAT32, "affine", 64, 4);
}

/// f32 output of `32_769 x 65_536` elements, 2^31 + 2^16: the per-row kernel's
/// output offset passes `i32::MAX` from row 32_768 on.
#[test]
#[ignore = "allocates an 8.6 GB f32 output; run with --ignored on a device with ~10 GB free"]
fn gather_qmm_f32_past_i32_output_elements_keeps_the_last_rows() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let _lock = lock_default_device();
    let _guard = DefaultDeviceGuard::gpu();
    let (experts_n, n, k, b): (i32, i32, i32, i32) = (2, 65_536, 64, 32_769);
    let weights = Weights::new(experts_n, n, k, dtype::FLOAT32, "affine", 64, 4);
    let x = mlxcel_core::reshape(
        &typed_from(x_value, (b * k) as usize, dtype::FLOAT32),
        &[b, 1, k],
    );
    let experts = unsorted_experts(b, experts_n);
    let rhs = mlxcel_core::astype(&mlxcel_core::from_slice_i32(&experts, &[b]), dtype::UINT32);
    let out = weights.gather(&x, &rhs);
    mlxcel_core::try_eval(&out).expect("the gather launches");
    for row_b in [0, 32_767, 32_768] {
        let got = row(&out, &[row_b, 0, 0], &[row_b + 1, 1, n], n as usize);
        let want = host_row(
            row_b as usize,
            &experts,
            &weights.dense,
            n as usize,
            k as usize,
        );
        assert_row("f32 past 2^31", row_b as usize, &got, &want);
    }
}
