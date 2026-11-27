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

//! Route marker of the fused residual-add + RMSNorm kernel, readable from any
//! build (issue #2834).
//!
//! The counters themselves live in the cuda-oxide adapter
//! ([`crate::cuda_oxide_fused_norm`]), which exists only in a build made with
//! `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1`. That cfg is set by this crate's
//! build script, so a dependent crate cannot test for it. Benchmark binaries in
//! the root crate still have to record which implementation served a run, and
//! "no refusal line on stderr" is not proof that the cuda-oxide route ran.
//! [`fused_norm_route_snapshot`] is the cfg-independent reader: `None` in a
//! build without the adapter, the counters otherwise. It reads atomics and
//! changes nothing.
//!
//! Used by: `examples/oxide_fused_norm_checkpoint_bench.rs` and
//! `examples/logit_trace.rs` in the root crate.

/// The adapter's process-wide route counters at one instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FusedNormRouteSnapshot {
    /// Whether a validated cuda-oxide artifact is installed.
    pub installed: bool,
    /// Fused joins routed to the cuda-oxide node (counted when the node is
    /// built).
    pub oxide_routes: u64,
    /// Fused joins that took the existing CUDA kernel.
    pub native_routes: u64,
    /// cuda-oxide kernels enqueued at evaluation.
    pub launches: u64,
    /// The most recent routing decision, as a stable lowercase name.
    pub last_fallback: &'static str,
}

impl FusedNormRouteSnapshot {
    /// Which implementation the counters say served the process so far:
    /// `"graph"` when no fused join was built, `"fused"` when only the
    /// existing kernel was, `"oxide"` when only the cuda-oxide node was and
    /// every one of them launched or is still lazy, and `"mixed"` otherwise.
    #[must_use]
    pub fn route(&self) -> &'static str {
        match (self.oxide_routes, self.native_routes) {
            (0, 0) => "graph",
            (0, _) => "fused",
            (_, 0) if self.launches > 0 => "oxide",
            _ => "mixed",
        }
    }
}

/// The route counters, or `None` in a build without the cuda-oxide adapter.
#[must_use]
pub fn fused_norm_route_snapshot() -> Option<FusedNormRouteSnapshot> {
    #[cfg(mlxcel_cuda_oxide_fused_norm)]
    {
        use crate::cuda_oxide_fused_norm::{self as oxide, Fallback};
        let counters = oxide::route_counters();
        let last_fallback = match oxide::last_fallback() {
            Fallback::Taken => "taken",
            Fallback::NotInstalled => "not_installed",
            Fallback::DeviceNotGpu => "device_not_gpu",
            Fallback::DtypePair => "dtype_pair",
            Fallback::ElementLimit => "element_limit",
            Fallback::Shape => "shape",
            Fallback::BlockDim => "block_dim",
            Fallback::GridLimit => "grid_limit",
            _ => "unknown",
        };
        Some(FusedNormRouteSnapshot {
            installed: oxide::is_installed(),
            oxide_routes: counters.oxide_routes,
            native_routes: counters.native_routes,
            launches: counters.launches,
            last_fallback,
        })
    }
    #[cfg(not(mlxcel_cuda_oxide_fused_norm))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(oxide_routes: u64, native_routes: u64, launches: u64) -> FusedNormRouteSnapshot {
        FusedNormRouteSnapshot {
            installed: oxide_routes > 0,
            oxide_routes,
            native_routes,
            launches,
            last_fallback: "taken",
        }
    }

    #[test]
    fn the_route_name_follows_the_counters() {
        assert_eq!(snapshot(0, 0, 0).route(), "graph");
        assert_eq!(snapshot(0, 16, 0).route(), "fused");
        assert_eq!(snapshot(16, 0, 16).route(), "oxide");
        // Routed but never launched, or both implementations in one process:
        // neither is a clean arm.
        assert_eq!(snapshot(16, 0, 0).route(), "mixed");
        assert_eq!(snapshot(16, 2, 16).route(), "mixed");
    }

    #[test]
    fn a_build_without_the_adapter_reports_nothing() {
        assert_eq!(
            fused_norm_route_snapshot().is_some(),
            cfg!(mlxcel_cuda_oxide_fused_norm)
        );
    }
}
