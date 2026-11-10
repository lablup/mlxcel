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
//! The macOS "available now" section of the memory estimate (issue #2883).
//!
//! `format_estimate` compares a load against total RAM or the allocator cap.
//! This section adds what macOS can give at this moment and warns when the
//! load would push it below its reserve. It only warns: `fits` and the
//! `--estimate-memory` verdict are unchanged.

use std::fmt::Write as _;

use mlxcel_core::host_memory::{HostMemoryStatus, host_memory_status};

use super::memory_estimate::{MemoryEstimate, format_bytes};

/// Marker that starts the warning line, for scripts and tests.
pub const AVAILABILITY_WARNING_MARKER: &str = "WARNING: memory available now";

/// The bytes the warning compares: weights, KV cache and activations.
#[must_use]
pub fn estimated_demand_bytes(est: &MemoryEstimate) -> u64 {
    est.weights_bytes
        .saturating_add(est.kv_cache_bytes)
        .saturating_add(est.activation_bytes)
}

/// Render the section for fixed inputs. `metal_working_set_bytes` of `0` means
/// the device did not report one.
#[must_use]
pub fn render_availability(
    demand_bytes: u64,
    status: &HostMemoryStatus,
    metal_working_set_bytes: u64,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "=== macOS Memory ===");
    let _ = writeln!(
        out,
        "  Total RAM:         {}",
        format_bytes(status.total_bytes)
    );
    let working_set = if metal_working_set_bytes > 0 {
        format_bytes(metal_working_set_bytes)
    } else {
        "unknown".to_string()
    };
    let _ = writeln!(out, "  Metal working set: {working_set}");
    let _ = writeln!(
        out,
        "  Available now:     {}  (free + purgeable + reclaimable file cache)",
        format_bytes(status.available_now_bytes),
    );
    let _ = writeln!(
        out,
        "  macOS reserve:     {}  (max(7 GiB, 15% of RAM))",
        format_bytes(status.reserve_bytes),
    );
    let _ = writeln!(
        out,
        "  This load:         {}  (weights + KV + activations)",
        format_bytes(demand_bytes),
    );
    let shortfall = status.shortfall_bytes(demand_bytes);
    if shortfall > 0 {
        let _ = writeln!(
            out,
            "{AVAILABILITY_WARNING_MARKER} is too low for this load: it needs {} but {} available minus the {} reserve leaves {}, short by {}. \
             macOS may compress or swap model memory. \
             Close other programs or use a smaller context, batch or model.",
            format_bytes(demand_bytes),
            format_bytes(status.available_now_bytes),
            format_bytes(status.reserve_bytes),
            format_bytes(status.usable_bytes()),
            format_bytes(shortfall),
        );
    }
    out
}

/// The section for this estimate on this host, or `None` off macOS.
#[must_use]
pub fn format_host_availability(est: &MemoryEstimate) -> Option<String> {
    let status = host_memory_status()?;
    let working_set = mlxcel_core::gpu_max_memory_size() as u64;
    Some(render_availability(
        estimated_demand_bytes(est),
        &status,
        working_set,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn status(available_gib: u64) -> HostMemoryStatus {
        HostMemoryStatus {
            total_bytes: 128 * GIB,
            available_now_bytes: available_gib * GIB,
            reserve_bytes: 19 * GIB,
            wired_bytes: 0,
        }
    }

    #[test]
    fn section_prints_every_figure_without_a_warning_when_the_load_fits() {
        let text = render_availability(50 * GIB, &status(100), 107 * GIB);
        for needle in [
            "Total RAM:         128.00 GiB",
            "Metal working set: 107.00 GiB",
            "Available now:     100.00 GiB",
            "macOS reserve:     19.00 GiB",
            "This load:         50.00 GiB",
        ] {
            assert!(text.contains(needle), "{needle} missing from {text}");
        }
        assert!(!text.contains(AVAILABILITY_WARNING_MARKER));
    }

    #[test]
    fn warning_gives_the_numbers_when_the_load_exceeds_available_minus_reserve() {
        let text = render_availability(90 * GIB, &status(100), 107 * GIB);
        assert!(text.contains(AVAILABILITY_WARNING_MARKER), "{text}");
        assert!(text.contains("needs 90.00 GiB"), "{text}");
        assert!(text.contains("leaves 81.00 GiB"), "{text}");
        assert!(text.contains("short by 9.00 GiB"), "{text}");
    }

    #[test]
    fn a_load_exactly_at_the_limit_does_not_warn() {
        let text = render_availability(81 * GIB, &status(100), 0);
        assert!(!text.contains(AVAILABILITY_WARNING_MARKER), "{text}");
        assert!(text.contains("Metal working set: unknown"), "{text}");
    }

    #[test]
    fn availability_below_the_reserve_warns_for_any_load() {
        let text = render_availability(GIB, &status(10), 0);
        assert!(text.contains("leaves 0"), "{text}");
        assert!(text.contains(AVAILABILITY_WARNING_MARKER), "{text}");
    }
}
