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
//! What macOS can hand this process right now, and what must stay with it.
//!
//! Total RAM and Metal's recommended working set say what a machine could
//! hold, not what it can give at this moment without compressing or swapping.
//! Issue #2883 adds that figure and the reserve left for macOS:
//!
//! - **Available now** is `free + purgeable + min(inactive + speculative,
//!   file-backed)` pages from `host_statistics64`. Inactive anonymous memory
//!   of other programs is excluded, because macOS frees it only by compressing
//!   or swapping it. See [`available_now_pages`].
//! - **The reserve** is `max(7 GiB, 15% of RAM)`. See [`memory_reserve_bytes`].
//!
//! Anything that fills memory in steps should re-check before each step and
//! stop rather than push macOS below the reserve. [`AvailabilityGate`] is that
//! check, with the query rate-limited so a hot path can poll it. The
//! expert-streaming (#2882) and prompt-cache snapshot (#2885) stores are
//! expected to use it alongside the row cache of #2881.
//!
//! The arithmetic is plain integer code that runs and is tested on every
//! platform. The only platform-specific parts are the two FFI queries, which
//! report `None` off macOS, so callers fall back to their previous behaviour.

use std::sync::Mutex;
use std::time::{Duration, Instant};

const GIB: u64 = 1 << 30;

/// Lower bound of the reserve left for macOS.
pub const MIN_RESERVE_BYTES: u64 = 7 * GIB;

/// Share of RAM, in percent, left for macOS once that exceeds
/// [`MIN_RESERVE_BYTES`].
pub const RESERVE_PERCENT_OF_RAM: u64 = 15;

/// How long [`AvailabilityGate`] reuses one availability query.
pub const AVAILABILITY_REFRESH_INTERVAL: Duration = Duration::from_millis(250);

/// The page counts [`available_now_pages`] reads, from `vm_statistics64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VmPageCounts {
    /// `free_count`: pages on the free list.
    pub free: u64,
    /// `purgeable_count`: pages the owner marked as discardable.
    pub purgeable: u64,
    /// `inactive_count`: pages not recently used, anonymous or file-backed.
    pub inactive: u64,
    /// `speculative_count`: read-ahead pages not yet used.
    pub speculative: u64,
    /// `external_page_count`: pages backed by a file.
    pub file_backed: u64,
    /// `wire_count`: pages that cannot be paged out or compressed.
    pub wired: u64,
}

/// Pages macOS can give up without compressing or swapping anything.
///
/// Reclaimable file cache is capped at the number of file-backed pages, so
/// inactive anonymous pages of other programs are not counted.
#[must_use]
pub fn available_now_pages(counts: &VmPageCounts) -> u64 {
    let reclaimable_cache = counts
        .inactive
        .saturating_add(counts.speculative)
        .min(counts.file_backed);
    counts
        .free
        .saturating_add(counts.purgeable)
        .saturating_add(reclaimable_cache)
}

/// Bytes to leave to macOS on a machine with `total_ram_bytes` of memory:
/// `max(7 GiB, 15% of RAM)`.
#[must_use]
pub fn memory_reserve_bytes(total_ram_bytes: u64) -> u64 {
    let share = u128::from(total_ram_bytes) * u128::from(RESERVE_PERCENT_OF_RAM) / 100;
    MIN_RESERVE_BYTES.max(share as u64)
}

/// One reading of host memory: total, available now, reserve, and wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct HostMemoryStatus {
    /// Physical RAM (`hw.memsize`).
    pub total_bytes: u64,
    /// See [`available_now_pages`].
    pub available_now_bytes: u64,
    /// See [`memory_reserve_bytes`].
    pub reserve_bytes: u64,
    /// Memory the system has wired at this moment, this process's included.
    pub wired_bytes: u64,
}

impl HostMemoryStatus {
    /// Build a status from raw page counts. Pure, so it is testable anywhere.
    #[must_use]
    pub fn from_counts(total_bytes: u64, page_size: u64, counts: &VmPageCounts) -> Self {
        Self {
            total_bytes,
            available_now_bytes: available_now_pages(counts).saturating_mul(page_size),
            reserve_bytes: memory_reserve_bytes(total_bytes),
            wired_bytes: counts.wired.saturating_mul(page_size),
        }
    }

    /// Available-now minus the reserve: what can still be taken without
    /// pushing macOS below its reserve.
    #[inline]
    #[must_use]
    pub fn usable_bytes(&self) -> u64 {
        self.available_now_bytes.saturating_sub(self.reserve_bytes)
    }

    /// Bytes by which `demand_bytes` exceeds [`Self::usable_bytes`], or `0`
    /// when it fits.
    #[inline]
    #[must_use]
    pub fn shortfall_bytes(&self, demand_bytes: u64) -> u64 {
        demand_bytes.saturating_sub(self.usable_bytes())
    }

    /// Most that the system's wired total may reach while leaving the reserve.
    #[inline]
    #[must_use]
    pub fn wired_ceiling_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.reserve_bytes)
    }
}

/// Read the current host memory figures. `None` off macOS or if the kernel
/// query fails; callers must treat that as "unknown", never as "no memory".
#[must_use]
pub fn host_memory_status() -> Option<HostMemoryStatus> {
    platform::host_memory_status()
}

/// This process's memory as macOS accounts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ProcessFootprint {
    /// `phys_footprint`: what Activity Monitor calls Memory, compressed pages
    /// included.
    pub phys_footprint_bytes: u64,
    /// Bytes of this process currently held in the compressor or swap. A value
    /// that grows while the server idles is the signature of idle compression.
    pub compressed_bytes: u64,
}

/// Read this process's footprint. `None` off macOS or if the query fails.
#[must_use]
pub fn process_footprint() -> Option<ProcessFootprint> {
    platform::process_footprint()
}

type StatusProbe = Box<dyn Fn() -> Option<HostMemoryStatus> + Send + Sync>;

#[derive(Debug, Default)]
struct GateState {
    probed_at: Option<Instant>,
    status: Option<HostMemoryStatus>,
    /// Bytes approved since `status` was read, which it cannot reflect yet.
    granted_since_probe: u64,
}

/// Decides whether a store may take one more growth step.
///
/// A step is approved when available-now minus what this gate already approved
/// since its last query still covers the step plus the reserve. The query is
/// reused for [`AVAILABILITY_REFRESH_INTERVAL`], so polling on every insert
/// costs a mutex and a clock read. When availability is unknown (off macOS, or
/// a failed query) every step is approved, which is the behaviour without a
/// gate.
pub struct AvailabilityGate {
    probe: StatusProbe,
    refresh: Duration,
    state: Mutex<GateState>,
}

impl std::fmt::Debug for AvailabilityGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AvailabilityGate")
            .field("refresh", &self.refresh)
            .finish_non_exhaustive()
    }
}

impl Default for AvailabilityGate {
    fn default() -> Self {
        Self::new()
    }
}

impl AvailabilityGate {
    /// A gate over [`host_memory_status`], refreshed every
    /// [`AVAILABILITY_REFRESH_INTERVAL`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_probe(AVAILABILITY_REFRESH_INTERVAL, host_memory_status)
    }

    /// A gate over any status source, for tests and alternative policies.
    #[must_use]
    pub fn with_probe(
        refresh: Duration,
        probe: impl Fn() -> Option<HostMemoryStatus> + Send + Sync + 'static,
    ) -> Self {
        Self {
            probe: Box::new(probe),
            refresh,
            state: Mutex::new(GateState::default()),
        }
    }

    /// Whether a step of `step_bytes` may be taken now.
    #[must_use]
    pub fn allows_step(&self, step_bytes: u64) -> bool {
        self.allows_step_at(step_bytes, Instant::now())
    }

    /// [`Self::allows_step`] at an explicit time, so tests need no sleeping.
    #[must_use]
    pub fn allows_step_at(&self, step_bytes: u64, now: Instant) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let fresh = state
            .probed_at
            .is_some_and(|at| now.saturating_duration_since(at) < self.refresh);
        if !fresh {
            state.status = (self.probe)();
            state.probed_at = Some(now);
            state.granted_since_probe = 0;
        }
        let Some(status) = state.status else {
            return true;
        };
        let needed = step_bytes.saturating_add(state.granted_since_probe);
        if status.shortfall_bytes(needed) > 0 {
            return false;
        }
        state.granted_since_probe = needed;
        true
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{HostMemoryStatus, ProcessFootprint, VmPageCounts};
    use std::mem::{MaybeUninit, size_of};
    use std::sync::OnceLock;

    /// `TASK_VM_INFO` from `<mach/task_info.h>`.
    const TASK_VM_INFO: libc::task_flavor_t = 22;

    /// The leading fields of `struct task_vm_info` through `phys_footprint`,
    /// which is `TASK_VM_INFO_REV1_COUNT` in `<mach/task_info.h>`. The kernel
    /// fills only as much as the count passed to `task_info` allows.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TaskVmInfoRev1 {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }

    fn total_ram_bytes() -> Option<u64> {
        static TOTAL: OnceLock<Option<u64>> = OnceLock::new();
        *TOTAL.get_or_init(|| {
            let mut value: u64 = 0;
            let mut len = size_of::<u64>();
            // SAFETY: the name is NUL-terminated, `value` and `len` are live
            // for the call, and `len` holds the size of `value`.
            let rc = unsafe {
                libc::sysctlbyname(
                    c"hw.memsize".as_ptr(),
                    (&raw mut value).cast(),
                    &raw mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            (rc == 0 && value > 0).then_some(value)
        })
    }

    /// The host port, taken once: each `mach_host_self` call adds a send right
    /// that would otherwise accumulate across polls.
    #[allow(deprecated)] // libc suggests the mach2 crate; one call is not worth it.
    fn host_port() -> libc::mach_port_t {
        static PORT: OnceLock<libc::mach_port_t> = OnceLock::new();
        // SAFETY: no preconditions.
        *PORT.get_or_init(|| unsafe { libc::mach_host_self() })
    }

    pub(super) fn host_memory_status() -> Option<HostMemoryStatus> {
        let total = total_ram_bytes()?;
        // SAFETY: `sysconf` has no memory preconditions.
        let page_size = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
        let mut stats = MaybeUninit::<libc::vm_statistics64>::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `stats` has room for `count` natural_t units, the host port
        // is this task's own, and the kernel writes at most `count` units.
        let rc = unsafe {
            libc::host_statistics64(
                host_port(),
                libc::HOST_VM_INFO64,
                stats.as_mut_ptr().cast(),
                &raw mut count,
            )
        };
        if rc != libc::KERN_SUCCESS {
            return None;
        }
        // SAFETY: zero is valid for every integer field and the call succeeded.
        let stats = unsafe { stats.assume_init() };
        let counts = VmPageCounts {
            free: u64::from(stats.free_count),
            purgeable: u64::from(stats.purgeable_count),
            inactive: u64::from(stats.inactive_count),
            speculative: u64::from(stats.speculative_count),
            file_backed: u64::from(stats.external_page_count),
            wired: u64::from(stats.wire_count),
        };
        Some(HostMemoryStatus::from_counts(total, page_size, &counts))
    }

    #[allow(deprecated)] // libc suggests the mach2 crate; one call is not worth it.
    pub(super) fn process_footprint() -> Option<ProcessFootprint> {
        let mut info = MaybeUninit::<TaskVmInfoRev1>::zeroed();
        let full_count = (size_of::<TaskVmInfoRev1>() / size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        let mut count = full_count;
        // SAFETY: `info` has room for `count` natural_t units and the task
        // port names this process; the kernel writes at most `count` units.
        let rc = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                TASK_VM_INFO,
                info.as_mut_ptr().cast(),
                &raw mut count,
            )
        };
        // A shorter answer would leave `phys_footprint` unfilled.
        if rc != libc::KERN_SUCCESS || count < full_count {
            return None;
        }
        // SAFETY: zero is valid for every integer field and the call succeeded.
        let info = unsafe { info.assume_init() };
        Some(ProcessFootprint {
            phys_footprint_bytes: info.phys_footprint,
            compressed_bytes: info.compressed,
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{HostMemoryStatus, ProcessFootprint};

    pub(super) fn host_memory_status() -> Option<HostMemoryStatus> {
        None
    }

    pub(super) fn process_footprint() -> Option<ProcessFootprint> {
        None
    }
}

#[cfg(test)]
#[path = "host_memory_tests.rs"]
mod tests;
