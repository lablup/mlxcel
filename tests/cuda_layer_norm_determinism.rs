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

//! Regression tests for issue #2696: MLX's CUDA `fast::layer_norm` and its
//! VJP must be bitwise deterministic and match a composed f32 reference.
//!
//! Upstream `mlx/backend/cuda/layer_norm.cu` reduces the mean and then the
//! variance through one shared-memory buffer, and `BlockBroadcastReduce`
//! had no barrier after reading it. A fast warp could write its variance
//! partial into a slot a slower warp had not read its mean partial from yet,
//! so a few rows per call came out wrong (30 of 30 calls differed on the
//! Mage-VAE encoder's `norm_out` in #2688). The VJP kernel has the same
//! hazard through its `temp.f` / `temp.f3` union. The CUDA overlay
//! `overlays/cuda/mlx/backend/cuda/layer_norm.cu` adds the second barrier.
//!
//! Each case calls the kernel [`CALLS`] times on one fixed input. The input
//! varies within every row (an integer hash of the element index) and has a
//! per-row mean offset; alternating constants give every warp the same
//! partial sums and hide the race.
//!
//! ```sh
//! cargo test --profile test-fast --features cuda --test cuda_layer_norm_determinism -- --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Calls per case.
const CALLS: usize = 32;
/// The eps of the #2688 VAE norm and the DiT `FinalLayer`.
const EPS: f32 = 1e-6;

/// splitmix64 of `i` under `seed`, mapped to [-1, 1].
fn hash_unit(i: u64, seed: u64) -> f32 {
    let mut z = i
        .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 23) as f32) - 1.0
}

/// A `shape` array in `dt` whose element `i` is `scale * hash + offset(row)`.
fn hashed(shape: &[i32], dt: i32, seed: u64, scale: f32, row_offset: bool) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let last = i64::from(*shape.last().unwrap());
    let values: Vec<f32> = (0..i64::from(n))
        .map(|i| {
            let offset = if row_offset {
                0.25 * ((i / last) % 5 - 2) as f32
            } else {
                0.0
            };
            scale * hash_unit(i as u64, seed) + offset
        })
        .collect();
    let out = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&values, shape), dt);
    mlxcel_core::eval(&out);
    out
}

/// `weight` in [0.75, 1.25] and `bias` in [-0.25, 0.25], both `[dim]`.
fn affine(dim: i32, dt: i32, seed: u64) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let weight = mlxcel_core::add(
        &hashed(&[dim], dt, seed, 0.25, false),
        &mlxcel_core::full_f32(&[1], 1.0, dt),
    );
    mlxcel_core::eval(&weight);
    (weight, hashed(&[dim], dt, seed + 1, 0.25, false))
}

fn ptr(a: Option<&UniquePtr<MlxArray>>) -> *const MlxArray {
    a.and_then(|a| a.as_ref())
        .map_or(std::ptr::null(), |a| a as *const MlxArray)
}

fn fused(
    x: &MlxArray,
    weight: Option<&UniquePtr<MlxArray>>,
    bias: Option<&UniquePtr<MlxArray>>,
) -> UniquePtr<MlxArray> {
    // SAFETY: each pointer is null or borrows an array that outlives the call.
    let out = unsafe { mlxcel_core::fast_layer_norm(x, ptr(weight), ptr(bias), EPS) };
    mlxcel_core::eval(&out);
    out
}

/// LayerNorm composed from separate ops, as Mage-Flow's `LayerNorm::forward`
/// (#2688) computes it: statistics, normalization and affine in f32, one cast
/// back to the input dtype.
fn composed_reference(
    x: &MlxArray,
    weight: Option<&UniquePtr<MlxArray>>,
    bias: Option<&UniquePtr<MlxArray>>,
) -> Vec<f32> {
    let f32_of = |a: &MlxArray| mlxcel_core::astype(a, dtype::FLOAT32);
    let xf = f32_of(x);
    let centered = mlxcel_core::subtract(&xf, &mlxcel_core::mean_axis(&xf, -1, true));
    let var = mlxcel_core::mean_axis(&mlxcel_core::square(&centered), -1, true);
    let eps = mlxcel_core::full_f32(&[1], EPS, dtype::FLOAT32);
    let mut y = mlxcel_core::multiply(
        &centered,
        &mlxcel_core::rsqrt(&mlxcel_core::add(&var, &eps)),
    );
    if let Some(w) = weight {
        y = mlxcel_core::multiply(&y, &f32_of(w));
    }
    if let Some(b) = bias {
        y = mlxcel_core::add(&y, &f32_of(b));
    }
    array_to_vec_f32(&mlxcel_core::astype(&y, mlxcel_core::array_dtype(x)))
}

/// One bf16 ulp at `v`: bf16 keeps 8 significand bits.
fn bf16_ulp(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return 0.0;
    }
    let exponent = ((v.to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// The fused-vs-composed bound: f32 within 1e-5; bf16 within one bf16 ulp of
/// the reference plus 1e-2, since the kernel rounds the normalized value to
/// bf16 before the affine step.
fn tolerance(dt: i32, reference: f32) -> f32 {
    if dt == dtype::FLOAT32 {
        1e-5
    } else {
        bf16_ulp(reference) + 1e-2
    }
}

/// Elements of `out` outside the bound against `reference`, and the largest
/// excess over it.
fn out_of_bound(dt: i32, out: &MlxArray, reference: &[f32]) -> (usize, f32) {
    let got = array_to_vec_f32(out);
    assert_eq!(got.len(), reference.len());
    got.iter()
        .zip(reference)
        .fold((0, 0f32), |(count, worst), (&g, &r)| {
            let excess = (g - r).abs() - tolerance(dt, r);
            if excess > 0.0 || !g.is_finite() {
                (count + 1, worst.max(excess))
            } else {
                (count, worst)
            }
        })
}

/// Rows of `row_bytes` bytes that differ between `a` and `b`.
fn differing_rows(a: &[u8], b: &[u8], row_bytes: usize) -> usize {
    a.chunks(row_bytes)
        .zip(b.chunks(row_bytes))
        .filter(|(x, y)| x != y)
        .count()
}

/// What [`CALLS`] repeats of one kernel call produced.
#[derive(Debug, Default)]
struct Repeats {
    /// Distinct output byte patterns over all calls (1 when deterministic).
    distinct_outputs: usize,
    /// Calls whose output bytes differ from the first call's.
    differing_calls: usize,
    /// The most rows any one call differed from the first in.
    max_differing_rows: usize,
    /// Calls with at least one element outside the reference bound. Every
    /// call is covered: one identical to a checked first output passes or
    /// fails with it.
    out_of_bound_calls: usize,
    /// Out-of-bound elements summed over those calls.
    out_of_bound_elements: usize,
    /// The largest excess over the bound.
    worst_excess: f32,
}

/// Runs `call` [`CALLS`] times and compares every output with the first.
/// With a `reference`, the first output and every one that differs from it
/// are checked against the reference bound; a call identical to the first
/// shares the first's result.
fn repeat(
    call: impl Fn() -> UniquePtr<MlxArray>,
    reference: Option<&[f32]>,
    row_bytes: usize,
) -> Repeats {
    let mut r = Repeats::default();
    let mut first: Option<(Vec<u8>, (usize, f32))> = None;
    let mut distinct = HashSet::new();
    for _ in 0..CALLS {
        let out = call();
        let bytes = mlxcel_core::array_to_raw_bytes(&out);
        let mut h = DefaultHasher::new();
        bytes.hash(&mut h);
        distinct.insert(h.finish());
        let check = |out: &MlxArray| {
            reference.map_or((0, 0.0), |reference| {
                out_of_bound(mlxcel_core::array_dtype(out), out, reference)
            })
        };
        let (count, worst) = match &first {
            None => {
                let result = check(&out);
                first = Some((bytes, result));
                result
            }
            Some((first, _)) if *first != bytes => {
                r.differing_calls += 1;
                r.max_differing_rows = r
                    .max_differing_rows
                    .max(differing_rows(first, &bytes, row_bytes));
                check(&out)
            }
            Some((_, result)) => *result,
        };
        if count > 0 {
            r.out_of_bound_calls += 1;
            r.out_of_bound_elements += count;
            r.worst_excess = r.worst_excess.max(worst);
        }
    }
    r.distinct_outputs = distinct.len();
    r
}

fn assert_clean(case: &str, r: &Repeats) {
    eprintln!("{case}: {r:?}");
    assert!(
        r.distinct_outputs == 1 && r.differing_calls == 0 && r.out_of_bound_calls == 0,
        "{case}: {} distinct outputs over {CALLS} identical calls, {} calls differing from the \
         first (up to {} rows each); {} calls had {} elements outside the composed-reference \
         bound (worst excess {:.3e}) (issue #2696)",
        r.distinct_outputs,
        r.differing_calls,
        r.max_differing_rows,
        r.out_of_bound_calls,
        r.out_of_bound_elements,
        r.worst_excess
    );
}

fn element_bytes(dt: i32) -> usize {
    if dt == dtype::FLOAT32 { 4 } else { 2 }
}

/// Forward case: `shape` in `dt`, with or without weight and bias.
fn check_forward(case: &str, shape: &[i32], dt: i32, seed: u64, with_affine: bool) {
    let dim = *shape.last().unwrap();
    let x = hashed(shape, dt, seed, 1.0, true);
    let (weight, bias) = if with_affine {
        let (w, b) = affine(dim, dt, seed + 100);
        (Some(w), Some(b))
    } else {
        (None, None)
    };
    let reference = composed_reference(&x, weight.as_ref(), bias.as_ref());
    let r = repeat(
        || fused(&x, weight.as_ref(), bias.as_ref()),
        Some(&reference),
        dim as usize * element_bytes(dt),
    );
    assert_clean(case, &r);
}

/// The #2688 shape: the Mage-VAE encoder's `norm_out`.
#[test]
fn bf16_affine_vae_norm_out_is_deterministic_and_matches_reference() {
    check_forward(
        "bf16 [2, 32, 32, 384] affine",
        &[2, 32, 32, 384],
        dtype::BFLOAT16,
        1,
        true,
    );
}

#[test]
fn f32_affine_vae_norm_out_is_deterministic_and_matches_reference() {
    check_forward(
        "f32 [2, 32, 32, 384] affine",
        &[2, 32, 32, 384],
        dtype::FLOAT32,
        2,
        true,
    );
}

#[test]
fn bf16_affine_wide_rows_are_deterministic_and_match_reference() {
    check_forward(
        "bf16 [8, 4096] affine",
        &[8, 4096],
        dtype::BFLOAT16,
        3,
        true,
    );
}

/// The shared DiT `FinalLayer` (Z-Image, LLaDA-Image, Ming-Image): no weight
/// or bias, 3840-wide rows.
#[test]
fn bf16_affine_free_final_layer_is_deterministic_and_matches_reference() {
    check_forward(
        "bf16 [1, 4096, 3840] affine-free",
        &[1, 4096, 3840],
        dtype::BFLOAT16,
        4,
        false,
    );
}

/// `layer_norm_vjp` reduces a mean and then three factors through a union
/// of the two storages: the gradient with respect to x must not change
/// between identical calls.
#[test]
fn bf16_vjp_input_gradient_is_deterministic() {
    const ROWS: i32 = 2048;
    const DIM: i32 = 384;
    let dt = dtype::BFLOAT16;
    let x = hashed(&[ROWS, DIM], dt, 5, 1.0, true);
    let cotan = hashed(&[ROWS, DIM], dt, 6, 1.0, false);
    let (weight, bias) = affine(DIM, dt, 7);
    let r = repeat(
        || {
            // SAFETY: `weight` is a live array for the whole call.
            let gx = unsafe {
                mlxcel_core::fast_layer_norm_vjp_for_test(&x, &*weight, &bias, &cotan, EPS, 0)
            }
            .expect("layer_norm VJP with respect to x");
            mlxcel_core::eval(&gx);
            gx
        },
        None,
        DIM as usize * element_bytes(dt),
    );
    assert_clean("bf16 [2048, 384] VJP dx", &r);
}
