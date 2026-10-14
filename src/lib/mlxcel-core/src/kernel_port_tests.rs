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

//! The wavefront rule `port_for` applies to every ROCm port (issue #2147).
//!
//! `rocm_port_allowed` is pure, so it runs on every backend, including a
//! CPU-only build. The process-level behaviour on ROCm is covered by
//! `tests/rocm_wave_size.rs` (the real gfx1151 device) and
//! `tests/rocm_wave64_port_refusal.rs` (a simulated wave64 report).

use crate::ffi;
use crate::hardware::{GpuBackendKind, gpu_backend_kind};

#[test]
fn rocm_port_allowed_holds_wave32_only_ports_to_32_lanes() {
    // A wave32-only port (the default) runs only at 32 lanes.
    assert!(ffi::rocm_port_allowed(false, 32));
    assert!(!ffi::rocm_port_allowed(false, 64));
    // An unreadable width (0) is not 32.
    assert!(!ffi::rocm_port_allowed(false, 0));
    // A port with no lane-level operation runs at any width.
    assert!(ffi::rocm_port_allowed(true, 64));
    assert!(ffi::rocm_port_allowed(true, 32));
    assert!(ffi::rocm_port_allowed(true, 0));
    // Nothing between or beyond counts as 32.
    assert!(!ffi::rocm_port_allowed(false, 16));
    assert!(!ffi::rocm_port_allowed(false, -1));
}

#[test]
fn rocm_device_warp_size_is_zero_off_rocm() {
    // Off ROCm there is no HIP device to ask; the overlay call is not even
    // compiled in, and the bridge answers 0, which `rocm_port_allowed` treats
    // as "not 32". Never consulted there: `port_for` reads it only on ROCm.
    if gpu_backend_kind() == GpuBackendKind::Rocm {
        let w = ffi::rocm_device_warp_size();
        assert!(
            w == 32 || w == 64,
            "a ROCm device reported a {w}-lane wavefront"
        );
        return;
    }
    assert_eq!(ffi::rocm_device_warp_size(), 0);
}
