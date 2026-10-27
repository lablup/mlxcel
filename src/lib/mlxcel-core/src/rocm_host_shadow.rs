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

//! Host-shadow accounting of the ROCm allocator (issue #2643).
//!
//! On Windows the ROCm allocator cannot hand the CPU a device pointer (WDDM
//! does not map device memory), so every host access to a GPU buffer goes
//! through a pinned host copy, the buffer's shadow (`overlays/rocm` item 55).
//! A shadow is `hipHostMalloc` memory, which is capped by the RAM Windows sees
//! rather than by the device limit, and it stays with its buffer until the
//! buffer is released, cached buffers included. These counters say how much is
//! pinned right now and how often the shadows move data: a download is a
//! device-to-host copy into a shadow (a host read of a buffer whose shadow is
//! stale), an upload is a host-to-device flush of a shadow the CPU wrote.
//!
//! [`crate::memory::snapshot`] carries `bytes` next to the active memory, and
//! `tests/rocm_host_shadow.rs` checks the counters against the read and write
//! paths. All zero on Linux ROCm (the CPU dereferences device memory there)
//! and on every other backend.

use crate::MlxArray;

/// A snapshot of the allocator's host-shadow counters. Each field is one
/// relaxed atomic read; the set is not read under a lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HostShadowStats {
    /// Pinned bytes the live shadows hold now: the sum of the shadowed
    /// buffers' allocator sizes (the rounded request, not the array's
    /// `nbytes`). Hidden page rounding inside `hipHostMalloc` is not counted.
    pub bytes: u64,
    /// Device-to-host copies into a shadow since process start.
    pub downloads: u64,
    /// Host-to-device flushes of a dirty shadow since process start.
    pub uploads: u64,
}

/// Reads the counters. Zero everywhere except Windows ROCm.
pub fn host_shadow_stats() -> HostShadowStats {
    let raw = crate::ffi::rocm_host_shadow_stats();
    let at = |i: usize| raw.get(i).copied().unwrap_or(0);
    HostShadowStats {
        bytes: at(0),
        downloads: at(1),
        uploads: at(2),
    }
}

/// Test-only: element `index` of the evaluated float32 array `arr`, read
/// through the bridge's read-only host path (the one `item_f32` and the byte
/// readers use). On Windows ROCm this refreshes the shadow when it is stale
/// and leaves the buffer clean, so the next kernel reading it flushes nothing.
///
/// # Errors
///
/// Returns the bridge's error when `arr` is not float32 or `index` is out of
/// range.
#[doc(hidden)]
pub fn host_read_f32_for_test(arr: &MlxArray, index: usize) -> Result<f32, cxx::Exception> {
    crate::ffi::host_read_f32_for_test(arr, index)
}

/// Test-only: writes `value` at element `index` of the evaluated float32 array
/// `arr` through `array::data<T>()`, the mutable host path every CPU writer in
/// MLX takes. On Windows ROCm this marks the shadow dirty, so the next kernel
/// reading the buffer flushes it first.
///
/// # Errors
///
/// Returns the bridge's error when `arr` is not float32 or `index` is out of
/// range.
#[doc(hidden)]
pub fn host_write_f32_for_test(
    arr: &MlxArray,
    index: usize,
    value: f32,
) -> Result<(), cxx::Exception> {
    crate::ffi::host_write_f32_for_test(arr, index, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuBackendKind, gpu_backend_kind};

    #[test]
    fn stats_are_zero_off_windows_rocm() {
        let stats = host_shadow_stats();
        if gpu_backend_kind() == GpuBackendKind::Rocm && cfg!(windows) {
            // The counters are live on this host; the integration test checks
            // their movement.
            return;
        }
        assert_eq!(stats, HostShadowStats::default());
    }
}
