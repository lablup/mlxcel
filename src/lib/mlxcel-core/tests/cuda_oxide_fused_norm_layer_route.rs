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

//! The cuda-oxide route as model code reaches it (#2833): through
//! `layers::fused_add_rms_norm`, with the two environment switches read from
//! the real process environment.
//!
//! Both switches are read once per process, so this is its own binary and one
//! test, and what it asserts depends on how it was launched:
//!
//! | `MLXCEL_FUSED_ADD_RMSNORM` | `MLXCEL_CUDA_OXIDE_FUSED_NORM` | expected route |
//! |---|---|---|
//! | unset or `0` | any | graph (`add` + `fast_rms_norm`), adapter untouched |
//! | `1` | unset or `0` | existing fused CUDA kernel |
//! | `1` | `1` | cuda-oxide node, launched once per join |
//!
//! Run it once per row; the evidence report lists the three commands. It is
//! compiled only into a build made with `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1`.

#![cfg(mlxcel_cuda_oxide_fused_norm)]

use mlxcel_core::cuda_oxide_artifact::TARGET_CAPABILITY;
use mlxcel_core::cuda_oxide_fused_norm::{self as oxide, Fallback};
use mlxcel_core::layers::{
    GemmaRMSNorm, RMSNorm, fused_add_rms_norm, fused_add_rmsnorm_enabled,
    fused_flag_explicit_value, graph_add_rms_norm,
};
use mlxcel_core::{MlxArray, array_to_raw_bytes, astype, dtype, eval, from_slice_f32};

const ROWS: i32 = 4;
const DIM: i32 = 2048;
const EPS: f32 = 1e-6;

fn values(seed: u32, len: usize, center: f32, spread: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            center + spread * ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
        })
        .collect()
}

fn f32_values(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Largest deviation of `got` from `want`, relative to the RMS of `want`.
fn normalized_max(got: &[f32], want: &[f32]) -> f64 {
    let ref_sq: f64 = want.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let ref_rms = (ref_sq / want.len() as f64).sqrt().max(1e-20);
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0f64, f64::max)
        / ref_rms
}

#[test]
fn the_layer_entry_point_takes_the_route_the_environment_selects() {
    if mlxcel_core::cuda_arch::cuda_compute_capability() != Some(TARGET_CAPABILITY) {
        eprintln!("skipping: the cuda-oxide artifact targets sm_121");
        return;
    }
    let fused = fused_add_rmsnorm_enabled();
    let opted_in =
        fused_flag_explicit_value(std::env::var(oxide::ENABLE_ENV).ok().as_deref()) == Some(true);

    let n = (ROWS * DIM) as usize;
    let make = |seed, len, center, spread, shape: &[i32]| {
        let a = astype(
            &from_slice_f32(&values(seed, len, center, spread), shape),
            dtype::FLOAT16,
        );
        eval(&a);
        a
    };
    let delta = make(1, n, 0.0, 4.0, &[ROWS, DIM]);
    let residual = make(2, n, 0.0, 4.0, &[ROWS, DIM]);
    let weight = make(3, DIM as usize, 1.0, 0.2, &[DIM]);

    // One standard and one Gemma join, as a decoder block issues them.
    let plain = RMSNorm::new(mlxcel_core::copy(&weight), EPS);
    let gemma = GemmaRMSNorm::new(mlxcel_core::copy(&weight), EPS);
    let before = oxide::route_counters();
    let (normed_a, residual_a) = fused_add_rms_norm(&plain, &delta, &residual);
    let (normed_b, residual_b) = fused_add_rms_norm(&gemma, &delta, &residual);
    let got = [&normed_a, &residual_a, &normed_b, &residual_b].map(|a| f32_values(a));
    let now = oxide::route_counters();
    let moved = (
        now.oxide_routes - before.oxide_routes,
        now.launches - before.launches,
        now.native_routes - before.native_routes,
    );

    let (want_a, want_ra) = graph_add_rms_norm(&plain, &delta, &residual);
    let (want_b, want_rb) = graph_add_rms_norm(&gemma, &delta, &residual);
    let want = [&want_a, &want_ra, &want_b, &want_rb].map(|a| f32_values(a));
    // The f16 budget of `fused_norm_parity_tests::tolerance_for`.
    for (g, w) in got.iter().zip(&want) {
        let deviation = normalized_max(g, w);
        assert!(deviation < 1.2e-2, "deviation {deviation:e} from the graph");
    }

    let expected = match (fused, opted_in) {
        (false, _) => (0, 0, 0),
        (true, false) => (0, 0, 2),
        (true, true) => (2, 2, 0),
    };
    println!(
        "RESULT layer-route fused_add_rmsnorm={fused} cuda_oxide_opt_in={opted_in} installed={} oxide_routes={} launches={} existing_kernel_routes={}",
        oxide::is_installed(),
        moved.0,
        moved.1,
        moved.2
    );
    assert_eq!(
        moved, expected,
        "(oxide routes, launches, existing-kernel routes)"
    );
    assert_eq!(oxide::is_installed(), fused && opted_in);
    if fused && opted_in {
        assert_eq!(oxide::last_fallback(), Fallback::Taken);
    }
}
