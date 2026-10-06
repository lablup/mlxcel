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

//! The wavefront width the ROCm port tables are checked against, on the real
//! device (issue #2147, `patches-rocm/LOCAL_FIXES.md` item 32).
//!
//! Every shuffle-based HIP port is held to 32-lane devices, because its
//! `#error` guard on `__AMDGCN_WAVEFRONT_SIZE` is inert with AMD clang 23. The
//! hold reads `mlx::core::rocm::device_warp_size()`, the hardware value. These
//! tests pin the two properties that make it safe on the RDNA host:
//!
//! - it reads the hardware width, 32 on RDNA and 64 on CDNA, and is not moved
//!   by `MLX_ROCM_FORCE_WARP_SIZE`, the launch-width experiment that overrides
//!   `Device::warp_size()` (checked in a child process, after the device has
//!   bound and logged the override, so a version that read `Device` would
//!   report 64 there and fail);
//! - on a 32-lane device every ported kernel's predicate is still true, so the
//!   hold changes nothing on gfx1151.
//!
//! The refusal side, with a 64-lane report, is
//! `tests/rocm_wave64_port_refusal.rs`, in a process of its own.

#![cfg(feature = "rocm")]

use std::process::Command;

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::rocm_arch::device_gfx_target;

const CHILD_ENV: &str = "MLXCEL_ROCM_WAVE_SIZE_CHILD";
const CHILD_TEST: &str = "child_reports_warp_size_under_forced_launch_width";
const REPORT: &str = "ROCM_DEVICE_WARP_SIZE=";

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

/// The wavefront width the hardware of `gfx` has: RDNA (`gfx10`, `gfx11`,
/// `gfx12`) is wave32, GCN and CDNA (`gfx9`) wave64.
fn expected_warp_size(gfx: &str) -> Option<i32> {
    if gfx.starts_with("gfx10") || gfx.starts_with("gfx11") || gfx.starts_with("gfx12") {
        Some(32)
    } else if gfx.starts_with("gfx9") {
        Some(64)
    } else {
        None
    }
}

/// Bind the HIP device the way any inference does, by evaluating an array on
/// the GPU, so `Device` has been constructed (and has applied
/// `MLX_ROCM_FORCE_WARP_SIZE`, if set) before the width is read.
fn bind_device() {
    let ones = mlxcel_core::ones(&[8], mlxcel_core::dtype::FLOAT32);
    mlxcel_core::try_eval(&ones).expect("evaluate on the GPU");
}

#[test]
fn device_warp_size_matches_the_gfx_family() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    bind_device();
    let gfx = device_gfx_target().expect("a ROCm device has a gfx target");
    let w = mlxcel_core::rocm_device_warp_size();
    match expected_warp_size(gfx) {
        Some(expected) => assert_eq!(w, expected, "{gfx} reported a {w}-lane wavefront"),
        None => assert!(w == 32 || w == 64, "{gfx} reported a {w}-lane wavefront"),
    }
}

#[test]
#[ignore = "run by force_warp_size_does_not_move_the_device_warp_size in a child process"]
fn child_reports_warp_size_under_forced_launch_width() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    bind_device();
    println!("{REPORT}{}", mlxcel_core::rocm_device_warp_size());
}

#[test]
fn force_warp_size_does_not_move_the_device_warp_size() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let real = mlxcel_core::rocm_device_warp_size();
    assert!(
        real == 32 || real == 64,
        "the device reported a {real}-lane wavefront"
    );
    // Force the other width, so the child's answer can only equal `real` if the
    // override is ignored.
    let forced = if real == 32 { 64 } else { 32 };

    let exe = std::env::current_exe().expect("path of this test binary");
    let out = Command::new(exe)
        .args([
            CHILD_TEST,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env("MLX_ROCM_FORCE_WARP_SIZE", forced.to_string())
        .output()
        .expect("run the child test process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child failed:\n{stdout}\n{stderr}");

    // The override was live in the child: the device logged applying it. Without
    // this the test could pass because the variable never reached MLX.
    let applied = format!("MLX_ROCM_FORCE_WARP_SIZE={forced} overrides device warp={real}");
    assert!(
        stderr.contains(&applied),
        "the child's device did not apply the override (`{applied}` missing):\n{stderr}"
    );

    let reported: i32 = stdout
        .lines()
        .find_map(|line| line.split_once(REPORT).map(|(_, v)| v.trim().to_string()))
        .unwrap_or_else(|| panic!("no `{REPORT}` line in the child's stdout:\n{stdout}"))
        .parse()
        .expect("an integer width");
    assert_eq!(
        reported, real,
        "MLX_ROCM_FORCE_WARP_SIZE={forced} moved the hardware width the port tables read"
    );
}

#[test]
fn every_rocm_port_is_still_selected_on_a_wave32_device() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    if mlxcel_core::rocm_device_warp_size() != 32 {
        eprintln!("skipping: this device is not wave32; the refusal side is covered elsewhere");
        return;
    }
    // One predicate per port table with a `.rocm` entry, wave32-only and
    // any-wave alike. On gfx1151 each was true before #2147 and must stay so.
    let predicates: [(&str, fn() -> bool); 16] = [
        ("bitlinear", mlxcel_core::bitlinear_kernel_available),
        ("xielu", mlxcel_core::fused_xielu_kernel_available),
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
        (
            "paged_v2_partial + merge",
            mlxcel_core::paged_attention_v2_available,
        ),
        ("paged_merge", mlxcel_core::paged_attention_merge_available),
        ("fused_rope", mlxcel_core::fused_rope_qk_append_available),
        ("gumbel", mlxcel_core::sampling_gumbel_backend_supported),
        (
            "rejection",
            mlxcel_core::sampling_rejection_backend_supported,
        ),
    ];
    let missing: Vec<&str> = predicates
        .iter()
        .filter(|(_, available)| !available())
        .map(|(name, _)| *name)
        .collect();
    assert!(
        missing.is_empty(),
        "ports not selected on a wave32 ROCm device: {missing:?}"
    );
}
