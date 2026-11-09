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
//! Optional `mlock` of evaluated weight buffers on macOS (issue #2883).
//!
//! GPU wiring of a buffer ends a few seconds after its last command buffer, so
//! macOS may compress or swap an idle server's weights. `MLXCEL_WIRE_WEIGHTS=1`
//! wires them for the life of the mapping, off by default until the soak in
//! `scripts/macos_memory_soak.sh` shows it helps.
//!
//! Wiring runs at the end of [`crate::weight_materialize`] over the tensors it
//! just evaluated, in 1 GiB chunks. Before the first chunk and after each one
//! the system wired total is compared with RAM minus the reserve, and a chunk
//! that crosses it is unwired again. The comparison reads the real wired count
//! rather than predicting it, because pages Metal has wired for the GPU are
//! already in that count and `mlock` adds nothing for them.
//!
//! Scope limits: multimodal towers stay lazy and are not wired, a strided view
//! has no single range to wire and is skipped, and a repack
//! that runs at the first token (expert stacking, dtype conversion) replaces
//! wired sources with unwired results.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::MlxArray;
#[cfg(target_os = "macos")]
use crate::ffi;

/// Environment variable that turns weight wiring on.
pub const WIRE_WEIGHTS_ENV: &str = "MLXCEL_WIRE_WEIGHTS";

/// Bytes wired between two checks of the system wired total.
pub const WIRE_CHUNK_BYTES: usize = 1 << 30;

static WIRED_WEIGHT_BYTES: AtomicU64 = AtomicU64::new(0);

/// Bytes of weights wired by the most recent load; `0` when wiring is off.
#[must_use]
pub fn wired_weight_bytes() -> u64 {
    WIRED_WEIGHT_BYTES.load(Ordering::Relaxed)
}

/// Whether a raw `MLXCEL_WIRE_WEIGHTS` value turns wiring on.
#[must_use]
pub fn parse_wire_weights(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

/// Whether this process was asked to wire weights.
#[must_use]
pub fn wire_weights_requested() -> bool {
    parse_wire_weights(std::env::var(WIRE_WEIGHTS_ENV).ok().as_deref())
}

/// What one wiring pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct WireReport {
    /// Bytes of page-aligned ranges the pass tried to wire.
    pub requested_bytes: u64,
    /// Bytes it left wired.
    pub wired_bytes: u64,
    /// Ranges in which `mlock` failed.
    pub failed_ranges: usize,
    /// `errno` of the first failure.
    pub first_errno: Option<i32>,
    /// The system wired total reached the ceiling, or could not be read, and
    /// the pass stopped early.
    pub stopped_at_ceiling: bool,
}

/// Round `(address, length)` ranges out to whole pages, then sort and merge
/// them so that a page shared by two small buffers is wired once.
#[must_use]
pub fn merge_page_ranges(ranges: &[(usize, usize)], page_size: usize) -> Vec<(usize, usize)> {
    let mask = page_size.max(1) - 1;
    let mut spans: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|&&(start, len)| start != 0 && len != 0)
        .map(|&(start, len)| {
            let end = start.saturating_add(len).saturating_add(mask) & !mask;
            (start & !mask, end)
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged.into_iter().map(|(s, e)| (s, e - s)).collect()
}

/// Wire `ranges` chunk by chunk while the system wired total stays at or below
/// `ceiling_bytes`. The kernel calls are parameters so the policy is testable
/// with fixed inputs: `wired_now` reads the system total, `lock` and `unlock`
/// act on one `(address, length)`.
pub fn wire_ranges(
    ranges: &[(usize, usize)],
    ceiling_bytes: u64,
    chunk_bytes: usize,
    mut wired_now: impl FnMut() -> Option<u64>,
    mut lock: impl FnMut(usize, usize) -> Result<(), i32>,
    mut unlock: impl FnMut(usize, usize),
) -> WireReport {
    let mut report = WireReport {
        requested_bytes: ranges.iter().map(|&(_, len)| len as u64).sum(),
        ..WireReport::default()
    };
    if wired_now().is_none_or(|wired| wired >= ceiling_bytes) {
        report.stopped_at_ceiling = true;
        return report;
    }
    for &(start, len) in ranges {
        let mut offset = 0;
        while offset < len {
            let chunk = chunk_bytes.max(1).min(len - offset);
            let address = start + offset;
            if let Err(errno) = lock(address, chunk) {
                report.failed_ranges += 1;
                report.first_errno.get_or_insert(errno);
                break;
            }
            if wired_now().is_none_or(|wired| wired > ceiling_bytes) {
                unlock(address, chunk);
                report.stopped_at_ceiling = true;
                return report;
            }
            report.wired_bytes += chunk as u64;
            offset += chunk;
        }
    }
    report
}

/// Wire the evaluated buffers of `arrays` when `MLXCEL_WIRE_WEIGHTS` asks for
/// it and log the outcome. Returns `None` when wiring did not run.
///
/// # Safety
///
/// Every pointer must point to a live `MlxArray` for the duration of the call.
pub(crate) unsafe fn wire_evaluated_buffers(arrays: &[*const MlxArray]) -> Option<WireReport> {
    if !wire_weights_requested() {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        let ranges: Vec<(usize, usize)> = arrays
            .iter()
            // SAFETY: the caller guarantees each pointer is live.
            .map(|&array| unsafe { &*array })
            .map(|array| (ffi::array_data_address(array), ffi::array_nbytes(array)))
            .collect();
        let report = macos::wire(&ranges)?;
        WIRED_WEIGHT_BYTES.store(report.wired_bytes, Ordering::Relaxed);
        log_report(&report);
        Some(report)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = arrays;
        tracing::warn!("{WIRE_WEIGHTS_ENV} is macOS only and is ignored on this platform");
        None
    }
}

#[cfg(target_os = "macos")]
fn log_report(report: &WireReport) {
    const GIB: f64 = (1u64 << 30) as f64;
    let wired = report.wired_bytes as f64 / GIB;
    let requested = report.requested_bytes as f64 / GIB;
    if report.failed_ranges == 0 && !report.stopped_at_ceiling {
        tracing::info!(
            wired_bytes = report.wired_bytes,
            "{WIRE_WEIGHTS_ENV}: wired {wired:.2} GiB of weights"
        );
    } else {
        tracing::warn!(
            wired_bytes = report.wired_bytes,
            requested_bytes = report.requested_bytes,
            failed_ranges = report.failed_ranges,
            first_errno = report.first_errno,
            stopped_at_ceiling = report.stopped_at_ceiling,
            "{WIRE_WEIGHTS_ENV}: wired {wired:.2} of {requested:.2} GiB of weights"
        );
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{WIRE_CHUNK_BYTES, WireReport, merge_page_ranges, wire_ranges};
    use crate::host_memory::host_memory_status;

    fn page_size() -> Option<usize> {
        // SAFETY: `sysconf` has no memory preconditions.
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()
    }

    /// `mlock` one range, as `Err(errno)` on failure.
    pub(super) fn mlock(address: usize, len: usize) -> Result<(), i32> {
        // SAFETY: `mlock` only changes the pageability of the range and reads
        // or writes no bytes through the pointer. An unmapped range is
        // reported as an error.
        if unsafe { libc::mlock(address as *const libc::c_void, len) } == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
        }
    }

    pub(super) fn munlock(address: usize, len: usize) {
        // SAFETY: as for `mlock`. A failure leaves the range wired, which only
        // makes the reported figure conservative.
        let _ = unsafe { libc::munlock(address as *const libc::c_void, len) };
    }

    pub(super) fn wire(ranges: &[(usize, usize)]) -> Option<WireReport> {
        let page_size = page_size()?;
        let ceiling = host_memory_status()?.wired_ceiling_bytes();
        let merged = merge_page_ranges(ranges, page_size);
        Some(wire_ranges(
            &merged,
            ceiling,
            WIRE_CHUNK_BYTES,
            || host_memory_status().map(|status| status.wired_bytes),
            mlock,
            munlock,
        ))
    }
}

#[cfg(test)]
#[path = "weight_wiring_tests.rs"]
mod tests;
