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

//! The one place a GPU test of a fused kernel may skip because the backend has
//! no port of that kernel (lablup/mlxcel#1809).
//!
//! The ROCm test gate (`make verify-test-rocm`) runs every GPU test on gfx1151.
//! The paged-attention kernels (v1 decode, v2 partial, merge) have Metal and
//! CUDA ports and no HIP port yet; that port is lablup/mlxcel#1814. Their tests
//! call the launchers directly, get the launcher's refusal, and fail, which
//! reads as a correctness failure when it is a missing port.
//!
//! A skip here is narrow on purpose, so it cannot hide a real defect:
//!
//! - It asks the kernel's own support predicate, which reads the same
//!   `KernelPorts` table the launcher dispatches through (`has_kernel_port`).
//!   When #1814 fills the `.rocm` entry, the predicate turns true and every
//!   skipped test runs again with no edit to the tests.
//! - It skips only on ROCm. On Metal or CUDA a `false` predicate is itself a
//!   defect (those ports exist), so the test runs and fails there rather than
//!   passing silently.
//! - It prints the skip, with the issue number, so the gate log says which
//!   tests did not run and why.
//!
//! Every paged-attention test that needs a port starts with
//! [`require_paged_attention_port!`] or [`require_paged_merge_port!`]; `grep`
//! for either to list them.

use std::io::Write;

use crate::hardware::{GpuBackendKind, gpu_backend_kind};

/// The issue that ports the kernels these skips wait on.
pub(crate) const PAGED_ATTENTION_ROCM_PORT_ISSUE: &str = "lablup/mlxcel#1814";

/// True, after printing why, when the calling test should return early: the
/// backend is ROCm and `available` (the kernel's port predicate) is false.
pub(crate) fn skip_for_missing_rocm_port(kernel: &str, available: bool, test: &str) -> bool {
    if available || gpu_backend_kind() != GpuBackendKind::Rocm {
        return false;
    }
    // Written to the process's stderr rather than through `eprintln!`, which
    // libtest captures and discards for a test that passes, so the skip shows
    // in the gate log instead of reading as a pass.
    let _ = writeln!(
        std::io::stderr(),
        "skipping {test}: ROCm has no {kernel} kernel port yet ({PAGED_ATTENTION_ROCM_PORT_ISSUE}); \
         the test runs again once the port table has a .rocm entry"
    );
    true
}

/// Return early from a test that launches the paged-attention kernels (v1
/// decode, v2 partial, merge) when ROCm has no port of them. See the module
/// docs for why this is the only such skip and when it goes away.
macro_rules! require_paged_attention_port {
    () => {
        if $crate::test_support::kernel_ports::skip_for_missing_rocm_port(
            "paged-attention",
            $crate::ffi::paged_attention_kernels_available(),
            concat!(module_path!(), ":", line!()),
        ) {
            return;
        }
    };
}

/// As [`require_paged_attention_port!`], for a test that launches only the
/// merge kernel (MLA split-KV).
macro_rules! require_paged_merge_port {
    () => {
        if $crate::test_support::kernel_ports::skip_for_missing_rocm_port(
            "paged-attention merge",
            $crate::ffi::paged_attention_merge_available(),
            concat!(module_path!(), ":", line!()),
        ) {
            return;
        }
    };
}

pub(crate) use require_paged_attention_port;
pub(crate) use require_paged_merge_port;
