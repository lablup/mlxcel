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

//! Shared fixtures for the cuda-oxide fused RMSNorm adapter tests (#2833).
//!
//! The case matrix and the input generator are a port of the standalone
//! harness in `spike/cuda-oxide-fused-norm/src/main.rs` (#2832): the same 26
//! cases per dtype pair, the same per-case LCG seed, the same Box-Muller
//! normals, already rounded to the case dtypes. The harness needs the
//! cuda-oxide toolchain to build; this port lets the adapter run the same
//! matrix from an ordinary mlxcel build with the committed artifact.

use crate::cuda_oxide_artifact::Lane;
use crate::cuda_oxide_fused_norm as oxide;
use crate::layers::{GemmaRMSNorm, RMSNorm, graph_add_rms_norm};
use crate::streams::{DefaultDeviceLock, lock_default_device};
use crate::{MlxArray, UniquePtr, array_to_raw_bytes, astype, copy, eval, from_slice_f32};

pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
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
        r += 1;
    }
    sign | r as u16
}

pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
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

pub(crate) fn f32_to_bf16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    if v.is_nan() {
        return ((x >> 16) | 0x40) as u16;
    }
    let lsb = (x >> 16) & 1;
    ((x.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

/// Round an f32 to `lane` and back (round-to-nearest-even).
pub(crate) fn round(lane: Lane, v: f32) -> f32 {
    match lane {
        Lane::F32 => v,
        Lane::F16 => f16_bits_to_f32(f32_to_f16_bits(v)),
        Lane::Bf16 => f32::from_bits((f32_to_bf16_bits(v) as u32) << 16),
    }
}

/// The value a `lane` bit pattern holds.
pub(crate) fn decode(lane: Lane, bits: u32) -> f32 {
    match lane {
        Lane::F32 => f32::from_bits(bits),
        Lane::F16 => f16_bits_to_f32(bits as u16),
        Lane::Bf16 => f32::from_bits(bits << 16),
    }
}

pub(crate) fn lane_name(lane: Lane) -> &'static str {
    match lane {
        Lane::F32 => "f32",
        Lane::F16 => "f16",
        Lane::Bf16 => "bf16",
    }
}

/// The harness's 64-bit LCG with Box-Muller normals.
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

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// One case of the #2832 matrix.
#[derive(Debug, Clone)]
pub(crate) struct Case {
    pub name: String,
    pub act: Lane,
    pub w: Lane,
    pub rows: usize,
    pub dim: usize,
    pub eps: f32,
    pub weight_bias: f32,
    pub nonfinite: bool,
}

impl Case {
    pub(crate) fn new(
        act: Lane,
        w: Lane,
        rows: usize,
        dim: usize,
        eps: f32,
        weight_bias: f32,
        nonfinite: bool,
    ) -> Self {
        let name = format!(
            "{}_{}_r{rows}_d{dim}_{}_{}_{}",
            lane_name(act),
            lane_name(w),
            if eps == 1e-6 { "e6" } else { "e5" },
            if weight_bias == 0.0 { "b0" } else { "b1" },
            if nonfinite { "nonfinite" } else { "normal" },
        );
        Self {
            name,
            act,
            w,
            rows,
            dim,
            eps,
            weight_bias,
            nonfinite,
        }
    }

    pub(crate) fn elements(&self) -> usize {
        self.rows * self.dim
    }

    fn seed(&self, index: usize) -> u64 {
        (index as u64) * 7919 + (self.rows as u64) * 131 + self.dim as u64
    }
}

/// Activation and weight dtype pairs, one per entry point of the module.
pub(crate) const DTYPE_PAIRS: [(Lane, Lane); 5] = [
    (Lane::F32, Lane::F32),
    (Lane::F16, Lane::F16),
    (Lane::Bf16, Lane::Bf16),
    (Lane::F16, Lane::F32),
    (Lane::Bf16, Lane::F32),
];

const SHAPES: [(usize, usize); 19] = [
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

const VARIANT_SHAPES: [(usize, usize); 4] = [(1, 128), (5, 1024), (2, 4097), (16, 8192)];

/// The harness's `case_matrix()`: 26 cases per dtype pair, 130 in all,
/// 2,181,390 elements.
pub(crate) fn case_matrix() -> Vec<Case> {
    let mut cases = Vec::new();
    for (act, w) in DTYPE_PAIRS {
        for (rows, dim) in SHAPES {
            cases.push(Case::new(act, w, rows, dim, 1e-6, 0.0, false));
        }
        for (rows, dim) in VARIANT_SHAPES {
            cases.push(Case::new(act, w, rows, dim, 1e-6, 1.0, false));
        }
        for (rows, dim) in &VARIANT_SHAPES[1..3] {
            cases.push(Case::new(act, w, *rows, *dim, 1e-5, 0.0, false));
        }
        cases.push(Case::new(act, w, 3, 1024, 1e-6, 0.0, true));
    }
    cases
}

/// Host-side inputs of a case, already rounded to its dtypes.
pub(crate) struct Inputs {
    pub x: Vec<f32>,
    pub residual: Vec<f32>,
    pub weight: Vec<f32>,
}

impl Inputs {
    /// Rows whose rounded residual sum is not finite. Such a row is non-finite
    /// in every implementation and is compared by class, not by tolerance.
    pub(crate) fn nonfinite_rows(&self, case: &Case) -> Vec<bool> {
        (0..case.rows)
            .map(|row| {
                (0..case.dim).any(|d| {
                    let i = row * case.dim + d;
                    !round(case.act, self.x[i] + self.residual[i]).is_finite()
                })
            })
            .collect()
    }
}

/// The harness's `inputs()`, `index` being the case's position in the matrix.
pub(crate) fn inputs(case: &Case, index: usize) -> Inputs {
    let mut rng = Lcg::new(case.seed(index));
    let n = case.elements();
    let mut x: Vec<f32> = (0..n)
        .map(|_| round(case.act, rng.normal() as f32))
        .collect();
    let residual: Vec<f32> = (0..n)
        .map(|_| round(case.act, rng.normal() as f32))
        .collect();
    let weight: Vec<f32> = (0..case.dim)
        .map(|_| round(case.w, (1.0 + 0.1 * rng.normal()) as f32))
        .collect();
    if case.nonfinite {
        // Row 0 carries a NaN, the last row a +Inf; rows in between stay finite.
        x[1.min(case.dim - 1)] = f32::NAN;
        x[(case.rows - 1) * case.dim] = f32::INFINITY;
    }
    Inputs {
        x,
        residual,
        weight,
    }
}

/// The MLX arrays of a case: the f32 fixtures cast to the case dtypes, which
/// is exact because the fixtures are already rounded.
pub(crate) struct CaseArrays {
    pub x: UniquePtr<MlxArray>,
    pub residual: UniquePtr<MlxArray>,
    pub weight: UniquePtr<MlxArray>,
}

pub(crate) fn case_arrays(case: &Case, inp: &Inputs) -> CaseArrays {
    let make = |values: &[f32], shape: &[i32], lane: Lane| {
        let a = astype(&from_slice_f32(values, shape), lane.mlx_dtype());
        eval(&a);
        a
    };
    let (rows, dim) = (case.rows as i32, case.dim as i32);
    CaseArrays {
        x: make(&inp.x, &[rows, dim], case.act),
        residual: make(&inp.residual, &[rows, dim], case.act),
        weight: make(&inp.weight, &[dim], case.w),
    }
}

/// `(normed, new_residual)` from the `fused_add_rms_norm` launcher, lazily.
/// Which implementation runs is decided by whether an artifact is installed.
pub(crate) fn fused_lazy(
    x: &MlxArray,
    residual: &MlxArray,
    weight: &MlxArray,
    eps: f32,
    weight_bias: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let mut normed = UniquePtr::null();
    let mut new_residual = UniquePtr::null();
    crate::fused_add_rms_norm(
        x,
        residual,
        weight,
        eps,
        weight_bias,
        &mut normed,
        &mut new_residual,
    )
    .expect("the CUDA build has a fused_add_rms_norm port");
    (normed, new_residual)
}

/// Raw output bits of one implementation.
pub(crate) struct OutBits {
    pub normed: Vec<u32>,
    pub new_residual: Vec<u32>,
}

/// The evaluated array's elements as `lane` bit patterns.
pub(crate) fn bits_of(arr: &MlxArray, lane: Lane) -> Vec<u32> {
    eval(arr);
    let bytes = array_to_raw_bytes(arr);
    match lane {
        Lane::F32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Lane::F16 | Lane::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    }
}

/// Run a case through the launcher and read both outputs back.
pub(crate) fn run_fused(case: &Case, arrays: &CaseArrays) -> OutBits {
    let (normed, new_residual) = fused_lazy(
        &arrays.x,
        &arrays.residual,
        &arrays.weight,
        case.eps,
        case.weight_bias,
    );
    OutBits {
        normed: bits_of(&normed, case.act),
        new_residual: bits_of(&new_residual, case.act),
    }
}

/// Run a case through the graph path the blocks used before the fused kernel.
///
/// MLX's `rms_norm` promotes to `result_type(x, weight)`, so an f16
/// activation with an f32 weight comes back as f32 (a bf16 one stays bf16
/// under this tree's promotion table); it is rounded to the activation dtype,
/// the dtype the kernels write, as the #2832 oracle did.
pub(crate) fn run_graph(case: &Case, arrays: &CaseArrays) -> OutBits {
    let (normed, new_residual) = if case.weight_bias == 0.0 {
        let norm = RMSNorm::new(copy(&arrays.weight), case.eps);
        graph_add_rms_norm(&norm, &arrays.x, &arrays.residual)
    } else {
        assert_eq!(case.weight_bias, 1.0, "only the Gemma fold has a layer");
        let norm = GemmaRMSNorm::new(copy(&arrays.weight), case.eps);
        graph_add_rms_norm(&norm, &arrays.x, &arrays.residual)
    };
    let dt = case.act.mlx_dtype();
    OutBits {
        normed: bits_of(&astype(&normed, dt), case.act),
        new_residual: bits_of(&astype(&new_residual, dt), case.act),
    }
}

/// Holds the device lock for a test that changes which implementation is
/// installed, and removes the artifact again when the test ends, pass or
/// fail, so no other test in the binary inherits the cuda-oxide route.
pub(crate) struct RouteGuard {
    _device: DefaultDeviceLock,
}

impl RouteGuard {
    /// The guard, or `None` (skip) when device 0 is not the artifact's target:
    /// the committed PTX is sm_121 only, so on any other CUDA device there is
    /// nothing these tests could run.
    pub(crate) fn on_target_device() -> Option<Self> {
        let device = crate::cuda_arch::cuda_compute_capability();
        if device != Some(crate::cuda_oxide_artifact::TARGET_CAPABILITY) {
            eprintln!("skipping: the cuda-oxide artifact targets sm_121, device 0 is {device:?}");
            return None;
        }
        let guard = Self {
            _device: lock_default_device(),
        };
        oxide::uninstall();
        Some(guard)
    }
}

impl Drop for RouteGuard {
    fn drop(&mut self) {
        oxide::uninstall();
    }
}

/// Counter movement since `before`.
pub(crate) fn delta(before: oxide::RouteCounters) -> oxide::RouteCounters {
    let now = oxide::route_counters();
    oxide::RouteCounters {
        oxide_routes: now.oxide_routes - before.oxide_routes,
        native_routes: now.native_routes - before.native_routes,
        launches: now.launches - before.launches,
    }
}
