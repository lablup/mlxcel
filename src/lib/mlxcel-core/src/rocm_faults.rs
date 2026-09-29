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

//! Deliberate ROCm GPU failures, for tests (issue #1804).
//!
//! A GPU failure on ROCm used to end as NaN output or a hang: the HIP runtime
//! does not abort the process on a queue fault, it fails every later HIP call
//! and never runs the host callbacks the ROCm backend signals completion
//! with. The backend now reports such failures through `Event::error`, so an
//! [`crate::try_eval`] that touches the failed work returns `Err`. Nothing in
//! a normal run provokes a failure on purpose, so this module builds arrays
//! whose evaluation fails in a chosen way, through the same custom-kernel path
//! the fused kernel ports use. The launch, the wait and the error attach are
//! the production ones; only the kernel bodies are contrived.
//!
//! An [`RocmFaultKind::OutOfBoundsWrite`] leaves the HIP device context
//! unusable for the rest of the process, so a test that evaluates one must
//! run in a process of its own (`tests/rocm_gpu_faults.rs` spawns itself).

use cxx::UniquePtr;

use crate::MlxArray;

/// How the evaluation should fail.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RocmFaultKind {
    /// A kernel launched with a 2048-thread block. HIP allows 1024, so the
    /// launch is rejected synchronously with `hipErrorInvalidConfiguration`
    /// and the device stays usable.
    OversizedBlock,
    /// A kernel that writes 2^40 bytes past its output. The GPU raises a
    /// memory aperture violation, the runtime retires the queue, and every
    /// later HIP call in the process returns `hipErrorIllegalAddress`.
    OutOfBoundsWrite,
}

impl RocmFaultKind {
    fn bridge_code(self) -> i32 {
        match self {
            RocmFaultKind::OversizedBlock => 0,
            RocmFaultKind::OutOfBoundsWrite => 1,
        }
    }
}

/// A lazy array whose evaluation fails as `kind` describes.
///
/// Building it is cheap and does nothing on the GPU; the failure happens in
/// the [`crate::try_eval`] (or any other evaluation) that consumes it, which
/// is the point: the test observes the production error path, not a fixture.
///
/// # Errors
///
/// Returns the bridge's error on backends other than ROCm, where the probe
/// kernels cannot be built.
pub fn fault_probe_array(kind: RocmFaultKind) -> Result<UniquePtr<MlxArray>, cxx::Exception> {
    crate::ffi::rocm_fault_probe_array(kind.bridge_code())
}
