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
//! Process-footprint gauges for `/metrics` (issue #2883).
//!
//! Off macOS the footprint query reports `None` and these families are absent.

use std::fmt::Write as _;

use mlxcel_core::host_memory::{ProcessFootprint, process_footprint};
use mlxcel_core::memory::{MemorySnapshot, snapshot};
use mlxcel_core::weight_wiring::wired_weight_bytes;

fn gauge(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = write!(
        out,
        "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"
    );
}

/// Render the gauges from fixed inputs.
pub(super) fn render(footprint: ProcessFootprint, mlx: MemorySnapshot, wired: u64) -> String {
    let mut out = String::new();
    gauge(
        &mut out,
        "mlxcel_process_phys_footprint_bytes",
        "Process memory footprint as macOS accounts it (phys_footprint)",
        footprint.phys_footprint_bytes,
    );
    gauge(
        &mut out,
        "mlxcel_process_compressed_bytes",
        "Bytes of this process held in the compressor or swap",
        footprint.compressed_bytes,
    );
    gauge(
        &mut out,
        "mlxcel_mlx_active_bytes",
        "Bytes actively allocated by the MLX allocator",
        mlx.active_bytes,
    );
    gauge(
        &mut out,
        "mlxcel_mlx_cache_bytes",
        "Bytes held in the MLX allocator free-buffer cache",
        mlx.cache_bytes,
    );
    gauge(
        &mut out,
        "mlxcel_weights_wired_bytes",
        "Weight bytes wired with mlock by MLXCEL_WIRE_WEIGHTS at the last load",
        wired,
    );
    out
}

/// The gauges for this process now, or an empty string off macOS.
pub(super) fn render_current() -> String {
    process_footprint()
        .map(|footprint| render(footprint, snapshot(), wired_weight_bytes()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_is_rendered_with_type_and_value() {
        let text = render(
            ProcessFootprint {
                phys_footprint_bytes: 11,
                compressed_bytes: 22,
            },
            MemorySnapshot {
                active_bytes: 33,
                cache_bytes: 44,
                ..MemorySnapshot::default()
            },
            55,
        );
        for (name, value) in [
            ("mlxcel_process_phys_footprint_bytes", 11),
            ("mlxcel_process_compressed_bytes", 22),
            ("mlxcel_mlx_active_bytes", 33),
            ("mlxcel_mlx_cache_bytes", 44),
            ("mlxcel_weights_wired_bytes", 55),
        ] {
            assert!(text.contains(&format!("# TYPE {name} gauge\n")), "{name}");
            assert!(text.contains(&format!("\n{name} {value}\n")), "{name}");
        }
    }
}
