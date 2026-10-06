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

//! On a device that reports a 64-lane wavefront, every wave32-only ROCm port is
//! held back and its callers take the graph (issue #2147).
//!
//! No wave64 device is available to the ROCm gate (gfx1151 is wave32), so this
//! replaces the width the port tables are checked against through
//! `set_rocm_port_warp_size_for_tests`, the test-only seam in
//! `src/lib/mlx-cpp/turbo/gpu_backend.cpp`. Everything downstream of the width
//! is the production path: `port_for`, `has_kernel_port`, every C++
//! `*_available()` predicate, `select_kernel_port`, and the Rust callers.
//!
//! The seam must be set before the first port predicate is asked, because the
//! Rust gates cache a `true` answer, so this file is its own test binary and
//! holds one test that sets it first and runs every check in order (it also
//! sets an environment variable, which is not safe alongside other threads).

#![cfg(feature = "rocm")]

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::layers::{LayerNorm, RMSNorm};
use mlxcel_core::{MlxArray, UniquePtr};

/// The width a CDNA device reports.
const WAVE64: i32 = 64;

/// Predicates whose answer comes only from wave32-only tables, one or more per
/// table: `bitlinear`, `ssm`, `mamba1_scan`, `moe_gateup` (with `moe_down`),
/// `moe_down`, `moe_fc1_relu2`, `add3_layer_norm`, `fused_norm`,
/// `paged_attention` and `paged_v2_partial`. `moe_gateup` has no predicate of
/// its own; its launcher is checked below instead.
const WAVE32_ONLY: [(&str, fn() -> bool); 11] = [
    ("bitlinear", mlxcel_core::bitlinear_kernel_available),
    ("ssm", mlxcel_core::ssm_kernel_available),
    ("mamba1_scan", mlxcel_core::mamba1_scan_kernel_available),
    (
        "mamba1_scan (f32 state)",
        mlxcel_core::mamba1_scan_float_state_kernel_available,
    ),
    (
        "moe_gateup + moe_down",
        mlxcel_core::fused_moe_kernels_available,
    ),
    ("moe_down", mlxcel_core::moe_down_kernel_available),
    (
        "moe_fc1_relu2 + moe_down",
        mlxcel_core::fused_moe_relu2_kernels_available,
    ),
    (
        "add3_layer_norm",
        mlxcel_core::fused_add3_layer_norm_available,
    ),
    ("fused_norm", mlxcel_core::fused_add_rms_norm_available),
    (
        "paged_attention",
        mlxcel_core::paged_attention_decode_available,
    ),
    // `paged_attention_v2_available` is partial && merge; merge stays true
    // below, so a false here is the partial table's.
    (
        "paged_v2_partial + merge",
        mlxcel_core::paged_attention_v2_available,
    ),
];

/// The five tables marked `rocm_any_wave_size`: no lane-level operation, so
/// they stay selected at any width.
const ANY_WAVE: [(&str, fn() -> bool); 5] = [
    ("xielu", mlxcel_core::fused_xielu_kernel_available),
    ("paged_merge", mlxcel_core::paged_attention_merge_available),
    ("fused_rope", mlxcel_core::fused_rope_qk_append_available),
    ("gumbel", mlxcel_core::sampling_gumbel_backend_supported),
    (
        "rejection",
        mlxcel_core::sampling_rejection_backend_supported,
    ),
];

fn f32_array(values: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, shape)
}

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * scale + offset).sin())
        .collect()
}

fn bytes(a: &MlxArray) -> Vec<u8> {
    mlxcel_core::try_eval(a).expect("evaluate on the GPU");
    mlxcel_core::array_to_raw_bytes(a)
}

/// The refusal names the wavefront width, not a missing port.
fn assert_wave_refusal(what: &str, err: impl std::fmt::Display) {
    let msg = err.to_string();
    assert!(
        msg.contains("validated only on a 32-lane wavefront") && msg.contains("reports 64 lanes"),
        "{what}: the refusal must name the wavefront width, got: {msg}"
    );
}

fn check_predicates() {
    let still_selected: Vec<&str> = WAVE32_ONLY
        .iter()
        .filter(|(_, available)| available())
        .map(|(name, _)| *name)
        .collect();
    assert!(
        still_selected.is_empty(),
        "wave32-only ports still selected with a 64-lane report: {still_selected:?}"
    );
    let held_back: Vec<&str> = ANY_WAVE
        .iter()
        .filter(|(_, available)| !available())
        .map(|(name, _)| *name)
        .collect();
    assert!(
        held_back.is_empty(),
        "any-wave ports held back with a 64-lane report: {held_back:?}"
    );
}

/// A direct launch reaches `select_kernel_port`, which refuses from the same
/// table the predicates read, so a predicate and its launcher cannot disagree.
fn check_launchers_refuse() {
    // bitlinear: the only port with no graph fallback; the BitNet loader
    // refuses on the predicate above.
    let x = f32_array(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let packed =
        mlxcel_core::from_bytes(&[134u8, 137, 100, 97], &[1, 4], mlxcel_core::dtype::UINT8);
    let scale = f32_array(&[2.0], &[1]);
    match mlxcel_core::bitlinear_matmul(&x, &packed, &scale, 4, 4, false) {
        Ok(_) => panic!("bitlinear_matmul launched a wave32-only port on a 64-lane report"),
        Err(e) => assert_wave_refusal("bitlinear_matmul", e),
    }

    // fused_add_rms_norm (the `fused_norm` table).
    let d = 64;
    let a = f32_array(&ramp(d, 0.3, 0.1), &[1, d as i32]);
    let r = f32_array(&ramp(d, 0.7, 0.4), &[1, d as i32]);
    let w = f32_array(&ramp(d, 0.2, 1.3), &[d as i32]);
    let mut normed = UniquePtr::null();
    let mut new_residual = UniquePtr::null();
    match mlxcel_core::fused_add_rms_norm(&a, &r, &w, 1e-5, 0.0, &mut normed, &mut new_residual) {
        Ok(()) => panic!("fused_add_rms_norm launched a wave32-only port on a 64-lane report"),
        Err(e) => assert_wave_refusal("fused_add_rms_norm", e),
    }

    // moe_gateup: the first kernel `fused_moe_expert_kernel` selects, so the
    // refusal comes from that table before any argument is read.
    let z = f32_array(&[0.0], &[1]);
    match mlxcel_core::fused_moe_expert_kernel(
        &z, &z, &z, &z, &z, &z, &z, &z, &z, &z, &z, &z, 64, 64, 1, 4, 4, 64,
    ) {
        Ok(_) => panic!("fused_moe_expert_kernel launched a wave32-only port on a 64-lane report"),
        Err(e) => assert_wave_refusal("fused_moe_expert_kernel (moe_gateup)", e),
    }
}

/// Production callers gate on the predicates and take their graph path. Each
/// would reach the refusal above, and fail, if it tried the kernel instead.
fn check_callers_take_the_graph() {
    // The residual-add + RMSNorm join of every Llama/Qwen block, with the
    // fusion switched on (the default is off): the graph pair, byte for byte.
    let d = 128;
    let delta = f32_array(&ramp(2 * d, 0.37, 0.2), &[2, d as i32]);
    let residual = f32_array(&ramp(2 * d, 0.13, 1.1), &[2, d as i32]);
    let norm = RMSNorm::new(f32_array(&ramp(d, 0.05, 0.9), &[d as i32]), 1e-5);
    let (normed, new_residual) = mlxcel_core::layers::fused_add_rms_norm(&norm, &delta, &residual);
    let (g_normed, g_residual) = mlxcel_core::layers::graph_add_rms_norm(&norm, &delta, &residual);
    assert_eq!(bytes(&normed), bytes(&g_normed));
    assert_eq!(bytes(&new_residual), bytes(&g_residual));

    // Cohere2's parallel-residual join (fused by default on ROCm): the
    // unfused add3 + LayerNorm pair, byte for byte.
    let a = f32_array(&ramp(2 * d, 0.21, 0.0), &[2, d as i32]);
    let b = f32_array(&ramp(2 * d, 0.17, 0.5), &[2, d as i32]);
    let xr = f32_array(&ramp(2 * d, 0.09, 2.0), &[2, d as i32]);
    let ln = LayerNorm::new(
        f32_array(&ramp(d, 0.03, 0.7), &[d as i32]),
        Some(f32_array(&ramp(d, 0.04, 0.1), &[d as i32])),
        1e-5,
    );
    let (x_new, h) = mlxcel_core::layers::residual_add3_layer_norm(&a, &b, &xr, &ln);
    let x_ref = mlxcel_core::compiled_add3(&a, &b, &xr);
    let h_ref = ln.forward(&x_ref);
    assert_eq!(bytes(&x_new), bytes(&x_ref));
    assert_eq!(bytes(&h), bytes(&h_ref));
}

#[test]
fn wave64_report_holds_back_every_wave32_port() {
    if gpu_backend_kind() != GpuBackendKind::Rocm {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    // Before any predicate is asked (the Rust gates cache a `true`), and before
    // any other thread exists in this binary (it holds this one test).
    mlxcel_core::set_rocm_port_warp_size_for_tests(WAVE64);
    // SAFETY: this binary runs a single test, so no other thread reads the
    // environment concurrently.
    unsafe { std::env::set_var("MLXCEL_FUSED_ADD_RMSNORM", "1") };

    check_predicates();
    check_launchers_refuse();
    check_callers_take_the_graph();

    // The seam is the only cause: with the hardware width back, the same
    // predicates answer true again on a wave32 device.
    mlxcel_core::set_rocm_port_warp_size_for_tests(0);
    if mlxcel_core::rocm_device_warp_size() == 32 {
        let missing: Vec<&str> = WAVE32_ONLY
            .iter()
            .chain(ANY_WAVE.iter())
            .filter(|(_, available)| !available())
            .map(|(name, _)| *name)
            .collect();
        assert!(
            missing.is_empty(),
            "ports not restored with the hardware width: {missing:?}"
        );
    }
}
