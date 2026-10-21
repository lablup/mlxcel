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

//! Shared helpers for the issue #2462 A1 sorted `gather_mm` tests
//! (`switch_layers_sorted_gather_history_tests.rs` and
//! `switch_layers_sorted_gather_layout_tests.rs`): deterministic inputs on
//! the Privacy Filter's expert geometry, an f64 host reference of the
//! uploaded bytes, and the predecessor workloads.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::SwitchLinear;
use crate::models::embedding_test_support::mlx_test_guard;

/// The Privacy Filter's expert geometry: 128 experts, hidden 640, top-4.
pub(super) const EXPERTS: i32 = 128;
pub(super) const HIDDEN: i32 = 640;
pub(super) const INTER: i32 = 640;
pub(super) const TOP_K: i32 = 4;

pub(super) struct Lcg(pub(super) u64);

impl Lcg {
    pub(super) fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Uniform in `[-scale, scale)`: the top 24 bits of the state, the same
    /// form as `openai_privacy_filter_tests::Lcg`.
    pub(super) fn next_f32(&mut self, scale: f32) -> f32 {
        self.next_u32();
        let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        (unit * 2.0 - 1.0) * scale
    }

    pub(super) fn f32_array(&mut self, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
        let count: i32 = shape.iter().product();
        let data: Vec<f32> = (0..count).map(|_| self.next_f32(scale)).collect();
        mlxcel_core::from_slice_f32(&data, shape)
    }
}

/// The dtype `gather_mm` computes an f32-activation x bf16-weight product in:
/// MLX casts both operands to `promote_types(a, b)` first, and `add` resolves
/// the same rule. Upstream MLX promotes `f32 + bf16` to f32 (Metal); the
/// CUDA overlay promotes it to bf16 (`overlays/cuda/mlx/dtype.cpp`). Probing
/// the rule keeps the reference tied to what the backend does rather than to
/// a backend name.
fn gather_mm_compute_dtype() -> i32 {
    let activation = mlxcel_core::zeros(&[1], dtype::FLOAT32);
    let weight = mlxcel_core::zeros(&[1], dtype::BFLOAT16);
    mlxcel_core::array_dtype(&mlxcel_core::add(&activation, &weight))
}

/// The activations as the sorted `gather_mm` consumes them. Where the backend
/// promotes f32 x bf16 to bf16 (CUDA) the kernel sees activations rounded to
/// bf16, and the f64 reference has to start from those rounded values. Where
/// it promotes to f32 (Metal) the kernel sees the f32 activations
/// unchanged, and rounding them here would put a bf16 ulp of reference error
/// on a correct result.
pub(super) fn kernel_activations(array: &MlxArray) -> Vec<f32> {
    let seen = if gather_mm_compute_dtype() == dtype::BFLOAT16 {
        mlxcel_core::astype(array, dtype::BFLOAT16)
    } else {
        mlxcel_core::astype(array, dtype::FLOAT32)
    };
    mlxcel_core::eval(&seen);
    to_vec(&seen)
}

pub(super) fn to_vec(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(array, dtype::FLOAT32))
}

pub(super) fn bytes(array: &MlxArray) -> Vec<u8> {
    mlxcel_core::eval(array);
    mlxcel_core::array_to_raw_bytes(array)
}

/// `tokens * TOP_K` routed rows: every token picks `TOP_K` distinct experts,
/// the pairs are flattened and sorted ascending, as `gather_sort` produces.
pub(super) fn sorted_expert_ids(rng: &mut Lcg, tokens: i32) -> Vec<u32> {
    let mut ids = Vec::with_capacity((tokens * TOP_K) as usize);
    for _ in 0..tokens {
        let mut picked = Vec::with_capacity(TOP_K as usize);
        while picked.len() < TOP_K as usize {
            let e = rng.next_u32() % EXPERTS as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids.sort_unstable();
    ids
}

/// A bf16 `[EXPERTS, out, in]` plane plus its exact values widened to f32.
pub(super) fn bf16_plane(rng: &mut Lcg, out: i32, inp: i32) -> (UniquePtr<MlxArray>, Vec<f32>) {
    let dense = rng.f32_array(&[EXPERTS, out, inp], 0.05);
    let plane = mlxcel_core::astype(&dense, dtype::BFLOAT16);
    mlxcel_core::eval(&plane);
    let exact = to_vec(&plane);
    (plane, exact)
}

/// `out[r] = x[r] @ w[ids[r]]^T` in f64 from the uploaded bytes.
pub(super) fn reference(x: &[f32], w: &[f32], ids: &[u32], out: usize, inp: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(ids.len() * out);
    for (row, &expert) in ids.iter().enumerate() {
        let xr = &x[row * inp..(row + 1) * inp];
        let base = expert as usize * out * inp;
        for n in 0..out {
            let wr = &w[base + n * inp..base + (n + 1) * inp];
            let acc: f64 = xr
                .iter()
                .zip(wr)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            result.push(acc as f32);
        }
    }
    result
}

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// An 8-bit affine plane of the Privacy Filter 8-bit export's layout
/// (`group_size` 64), for the predecessor workload.
pub(super) fn quantized_plane(rng: &mut Lcg, out: i32, inp: i32, prefix: &str) -> SwitchLinear {
    let dense = rng.f32_array(&[EXPERTS, out, inp], 0.05);
    let q = mlxcel_core::quantize_weights(&dense, 64, 8);
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::quantized_weights_w(&q),
    );
    weights.insert(
        format!("{prefix}.scales"),
        mlxcel_core::quantized_weights_scales(&q),
    );
    weights.insert(
        format!("{prefix}.biases"),
        mlxcel_core::quantized_weights_biases(&q),
    );
    SwitchLinear::from_weights(&weights, prefix, 64, 8).expect("8-bit plane loads")
}

/// The 8-bit Privacy Filter's MoE over `tokens` tokens: sorted `gather_qmm`
/// through gate, up and down planes, several times, with every result
/// evaluated so the allocator sees the whole workload.
pub(super) fn eight_bit_workload(rng: &mut Lcg, tokens: i32, rounds: usize) {
    let gate = quantized_plane(rng, INTER, HIDDEN, "pred.gate");
    let up = quantized_plane(rng, INTER, HIDDEN, "pred.up");
    let down = quantized_plane(rng, HIDDEN, INTER, "pred.down");
    for _ in 0..rounds {
        let ids = sorted_expert_ids(rng, tokens);
        let rows = ids.len() as i32;
        let x = rng.f32_array(&[rows, 1, HIDDEN], 1.0);
        let idx = mlxcel_core::from_slice_u32(&ids, &[rows]);
        let g = gate.forward(&x, &idx, true);
        let u = up.forward(&x, &idx, true);
        let h = mlxcel_core::multiply(&mlxcel_core::sigmoid(&g), &u);
        let out = down.forward(&h, &idx, true);
        mlxcel_core::eval(&out);
    }
}

pub(super) struct SortedCase {
    pub(super) plane: SwitchLinear,
    pub(super) exact_w: Vec<f32>,
    pub(super) x: UniquePtr<MlxArray>,
    pub(super) x_host: Vec<f32>,
    pub(super) idx: UniquePtr<MlxArray>,
    pub(super) ids: Vec<u32>,
}

impl SortedCase {
    pub(super) fn new(rng: &mut Lcg, tokens: i32) -> Self {
        let (weight, exact_w) = bf16_plane(rng, INTER, HIDDEN);
        let ids = sorted_expert_ids(rng, tokens);
        let rows = ids.len() as i32;
        let x = rng.f32_array(&[rows, 1, HIDDEN], 1.0);
        mlxcel_core::eval(&x);
        let x_host = kernel_activations(&x);
        let idx = mlxcel_core::from_slice_u32(&ids, &[rows]);
        Self {
            plane: SwitchLinear::Regular { weight },
            exact_w,
            x,
            x_host,
            idx,
            ids,
        }
    }

    pub(super) fn sorted(&self) -> UniquePtr<MlxArray> {
        self.plane.forward(&self.x, &self.idx, true)
    }

    pub(super) fn unsorted(&self) -> UniquePtr<MlxArray> {
        self.plane.forward(&self.x, &self.idx, false)
    }

    pub(super) fn reference(&self) -> Vec<f32> {
        reference(
            &self.x_host,
            &self.exact_w,
            &self.ids,
            INTER as usize,
            HIDDEN as usize,
        )
    }
}

/// Per-row max error and the row's expert, for finding out which rows a
/// kernel gets wrong.
pub(super) fn row_errors(
    got: &[f32],
    want: &[f32],
    ids: &[u32],
    out: usize,
) -> Vec<(usize, u32, f32)> {
    (0..ids.len())
        .map(|row| {
            let err = max_abs_diff(
                &got[row * out..(row + 1) * out],
                &want[row * out..(row + 1) * out],
            );
            (row, ids[row], err)
        })
        .collect()
}

pub(super) fn report(label: &str, got: &[f32], want: &[f32], ids: &[u32], out: usize) -> f32 {
    let rows = row_errors(got, want, ids, out);
    let worst = rows.iter().map(|r| r.2).fold(0.0, f32::max);
    let bad: Vec<_> = rows.iter().filter(|r| r.2 > 1e-4).collect();
    let mut counts = std::collections::BTreeMap::new();
    for &e in ids {
        *counts.entry(e).or_insert(0usize) += 1;
    }
    eprintln!(
        "{label}: max |err| {worst:.3e}, {} of {} rows off (> 1e-4); first bad rows (row, expert, err, group size): {:?}",
        bad.len(),
        rows.len(),
        bad.iter()
            .take(12)
            .map(|r| (r.0, r.1, r.2, counts[&r.1]))
            .collect::<Vec<_>>()
    );
    worst
}

/// `got` against the f64 reference built from `kernel_activations`: within one
/// bf16 ulp (2^-8 relative) plus slack for the f32 accumulation. That bound is
/// what a bf16 result (CUDA, which promotes f32 x bf16 to bf16) needs; an f32
/// result (Metal) lands orders of magnitude inside it, and still fails a
/// reference that rounds activations the kernel did not round.
pub(super) fn assert_matches_reference(label: &str, got: &[f32], want: &[f32]) {
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(0.25))
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1.5 * 2f32.powi(-8),
        "{label}: off the f64 reference by {worst:.3e} relative (bf16 ulp is {:.3e})",
        2f32.powi(-8)
    );
}

pub(super) fn describe(array: &MlxArray) -> String {
    format!(
        "dtype {} shape {:?}",
        mlxcel_core::array_dtype(array),
        mlxcel_core::array_shape(array)
    )
}

/// Large host-uploaded (unified-memory) arrays and device temporaries of
/// other sizes, to churn the allocator cache without any MoE kernel.
pub(super) fn allocator_churn(rng: &mut Lcg, rounds: usize) {
    for round in 0..rounds {
        let rows = 1500 + 37 * round as i32;
        let host = rng.f32_array(&[rows, HIDDEN], 1.0);
        let device = mlxcel_core::matmul(&host, &mlxcel_core::swap_axes(&host, 0, 1));
        let half = mlxcel_core::astype(&device, dtype::BFLOAT16);
        mlxcel_core::eval(&half);
    }
}

pub(super) fn history_test(label: &str, predecessor: impl FnOnce(&mut Lcg)) {
    if !mlxcel_core::gpu_backend_available() {
        eprintln!("no GPU backend; the sorted gather_mm branch is a GPU kernel, skipping");
        return;
    }
    let _guard = mlx_test_guard();
    let mut rng = Lcg(0x2462);
    // 24 tokens x top-4 = 96 routed rows over 128 experts, the regime the
    // Privacy Filter's short sentences run (most experts receive 0 or 1 row).
    let case = SortedCase::new(&mut rng, 24);
    let reference = case.reference();

    let fresh = case.sorted();
    let fresh_bytes = bytes(&fresh);
    let fresh_values = to_vec(&fresh);
    eprintln!("{label}: sorted gather_mm output {}", describe(&fresh));
    assert_matches_reference(&format!("{label} fresh sorted"), &fresh_values, &reference);
    let unsorted = to_vec(&case.unsorted());
    assert_matches_reference(&format!("{label} unsorted"), &unsorted, &reference);
    eprintln!(
        "{label}: fresh sorted vs unsorted max |diff| {:.3e}",
        max_abs_diff(&fresh_values, &unsorted)
    );

    predecessor(&mut rng);

    let again = case.sorted();
    let again_bytes = bytes(&again);
    let again_values = to_vec(&again);
    let drift = max_abs_diff(&again_values, &fresh_values);
    let differing = fresh_bytes
        .iter()
        .zip(&again_bytes)
        .filter(|(a, b)| a != b)
        .count();
    let rows = row_errors(&again_values, &fresh_values, &case.ids, INTER as usize);
    let bad: Vec<_> = rows.iter().filter(|r| r.2 > 0.0).take(12).collect();
    assert_eq!(
        fresh_bytes, again_bytes,
        "{label}: sorted gather_mm changed after the predecessor workload: max |drift| {drift} \
         ({differing} bytes differ); first drifted rows (row, expert, |drift|): {bad:?}"
    );
}
