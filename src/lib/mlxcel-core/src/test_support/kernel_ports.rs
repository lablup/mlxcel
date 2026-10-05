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
//! The paged-attention kernels (v1 decode, v2 partial, merge) had Metal and
//! CUDA ports and no HIP port until lablup/mlxcel#2068 (part of #1814), and
//! these skips were written for that gap. Without a port the tests either
//! launch the kernels directly and get the launcher's refusal, or go through a
//! production path that declines on the missing port before it reaches the
//! decision the test checks (the batched paged decode's floor and multi-slab
//! reports, for example). Either way they fail for a missing port, which reads
//! as a correctness failure.
//!
//! With the HIP ports in the tables every predicate below is true on ROCm, so
//! nothing skips there any more. The macros stay, so that a kernel added to
//! this family, or a backend added later, skips visibly here too instead of
//! failing.
//!
//! A skip here is narrow on purpose, so it cannot hide a real defect:
//!
//! - It asks the support predicate of exactly the kernels the test needs, which
//!   reads the same `KernelPorts` table the launcher dispatches through
//!   (`has_kernel_port`). When #1814 fills a `.rocm` entry, the predicates that
//!   depend on it turn true and those tests run again with no edit, even if the
//!   kernels are ported one at a time.
//! - It skips only on ROCm. On Metal or CUDA a `false` predicate is itself a
//!   defect (those ports exist), so the test runs and fails there rather than
//!   passing silently.
//! - It prints the skip, as `skipping <module>:<line>: ROCm has no <kernels>
//!   kernel port yet (lablup/mlxcel#1814) ...`, so the gate log says which tests
//!   did not run and why.
//!
//! Every such test starts with one of the `require_paged_*_port!` macros below;
//! `grep` for `require_paged_` to list them.

use std::io::Write;

use crate::hardware::{GpuBackendKind, gpu_backend_kind};

/// The issue that ports the kernels these skips wait on.
pub(crate) const PAGED_ATTENTION_ROCM_PORT_ISSUE: &str = "lablup/mlxcel#1814";

/// True, after printing why, when the calling test should return early: the
/// backend is ROCm and `available` (the kernels' port predicate) is false.
pub(crate) fn skip_for_missing_rocm_port(kernels: &str, available: bool, test: &str) -> bool {
    if available || gpu_backend_kind() != GpuBackendKind::Rocm {
        return false;
    }
    // Written to the process's stderr rather than through `eprintln!`, which
    // libtest captures and discards for a test that passes, so the skip shows
    // in the gate log instead of reading as a pass.
    let _ = writeln!(
        std::io::stderr(),
        "skipping {test}: ROCm has no {kernels} kernel port yet ({PAGED_ATTENTION_ROCM_PORT_ISSUE}); \
         the test runs again once the port table has a .rocm entry"
    );
    true
}

/// Shared body of the `require_paged_*_port!` macros.
macro_rules! require_port {
    ($kernels:literal, $predicate:ident) => {
        if $crate::test_support::kernel_ports::skip_for_missing_rocm_port(
            $kernels,
            $crate::ffi::$predicate(),
            concat!(module_path!(), ":", line!()),
        ) {
            return;
        }
    };
}

/// Return early on ROCm without all three paged-attention kernels (v1 decode,
/// v2 partial, merge): the batched paged decode, which may take either path.
macro_rules! require_paged_attention_port {
    () => {
        $crate::test_support::kernel_ports::require_port!(
            "paged-attention (v1 decode, v2 partial, merge)",
            paged_attention_kernels_available
        )
    };
}

/// Return early on ROCm without the v1 paged decode kernel.
macro_rules! require_paged_decode_port {
    () => {
        $crate::test_support::kernel_ports::require_port!(
            "paged-attention v1 decode",
            paged_attention_decode_available
        )
    };
}

/// Return early on ROCm without the v2 partial and merge kernels: the flat,
/// cascade and sparse v2 launches.
macro_rules! require_paged_v2_port {
    () => {
        $crate::test_support::kernel_ports::require_port!(
            "paged-attention v2 (partial, merge)",
            paged_attention_v2_available
        )
    };
}

/// Return early on ROCm without the merge kernel: MLA split-KV and the merge
/// kernel's own tests.
macro_rules! require_paged_merge_port {
    () => {
        $crate::test_support::kernel_ports::require_port!(
            "paged-attention merge",
            paged_attention_merge_available
        )
    };
}

pub(crate) use require_paged_attention_port;
pub(crate) use require_paged_decode_port;
pub(crate) use require_paged_merge_port;
pub(crate) use require_paged_v2_port;
pub(crate) use require_port;
