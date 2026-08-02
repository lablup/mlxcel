// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Which MLA attention path each step actually took (issue #907).
//!
//! Issue #899 shipped a fused decode path that never activated, and the
//! before/after benchmark compared the fallback against itself and produced a
//! clean-looking null. The counters here exist so that cannot repeat: every MLA
//! attention call records the path it took, and the benchmark harness prints
//! the snapshot for each arm. An arm that claims "absorbed" while
//! `absorbed_composed` is 0 is visibly a measurement of the fallback.
//!
//! `tracing` is deliberately not used. The `mlxcel` CLI installs no tracing
//! subscriber at all (only `src/server/startup.rs` does), so a `tracing::info!`
//! from a CLI-reachable path emits nothing no matter what `RUST_LOG` says.
//! Counters plus an explicit print in the harness are observable everywhere.

use std::sync::atomic::{AtomicU64, Ordering};

/// The MLA attention paths a step can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MlaDecodePath {
    /// Cache holds decompressed per-head K/V; dense SDPA. The pre-#907 path and
    /// the default.
    Decompressed,
    /// Cache holds `(ckv, kpe)`; prefill up-projects the cached latent and runs
    /// dense SDPA. Absorption is a decode transform, so multi-token steps land
    /// here even with the flag on.
    AbsorbedPrefill,
    /// Stage 1: absorbed single-token decode built from composed MLX ops.
    AbsorbedComposed,
    /// Stage 2: absorbed decode split across latent chunks, merged with issue
    /// #898's `paged_attention_merge_states`.
    AbsorbedSplitKv,
}

impl MlaDecodePath {
    /// Stable label for printed output and test assertions.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Decompressed => "decompressed",
            Self::AbsorbedPrefill => "absorbed_prefill",
            Self::AbsorbedComposed => "absorbed_composed",
            Self::AbsorbedSplitKv => "absorbed_split_kv",
        }
    }
}

/// Every path, in report order.
pub const MLA_DECODE_PATHS: [MlaDecodePath; 4] = [
    MlaDecodePath::Decompressed,
    MlaDecodePath::AbsorbedPrefill,
    MlaDecodePath::AbsorbedComposed,
    MlaDecodePath::AbsorbedSplitKv,
];

static DECOMPRESSED: AtomicU64 = AtomicU64::new(0);
static ABSORBED_PREFILL: AtomicU64 = AtomicU64::new(0);
static ABSORBED_COMPOSED: AtomicU64 = AtomicU64::new(0);
static ABSORBED_SPLIT_KV: AtomicU64 = AtomicU64::new(0);

fn counter(path: MlaDecodePath) -> &'static AtomicU64 {
    match path {
        MlaDecodePath::Decompressed => &DECOMPRESSED,
        MlaDecodePath::AbsorbedPrefill => &ABSORBED_PREFILL,
        MlaDecodePath::AbsorbedComposed => &ABSORBED_COMPOSED,
        MlaDecodePath::AbsorbedSplitKv => &ABSORBED_SPLIT_KV,
    }
}

/// Record one MLA attention call on `path`.
///
/// `Relaxed` because the counters are diagnostics: they are read after the
/// measured region, never used to order anything.
#[inline]
pub fn record(path: MlaDecodePath) {
    counter(path).fetch_add(1, Ordering::Relaxed);
}

/// Per-path call counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlaDispatchCounts {
    pub decompressed: u64,
    pub absorbed_prefill: u64,
    pub absorbed_composed: u64,
    pub absorbed_split_kv: u64,
}

impl MlaDispatchCounts {
    /// Count for one path.
    #[must_use]
    pub const fn get(&self, path: MlaDecodePath) -> u64 {
        match path {
            MlaDecodePath::Decompressed => self.decompressed,
            MlaDecodePath::AbsorbedPrefill => self.absorbed_prefill,
            MlaDecodePath::AbsorbedComposed => self.absorbed_composed,
            MlaDecodePath::AbsorbedSplitKv => self.absorbed_split_kv,
        }
    }

    /// Total recorded calls.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.decompressed + self.absorbed_prefill + self.absorbed_composed + self.absorbed_split_kv
    }

    /// One-line `path=count` summary, in [`MLA_DECODE_PATHS`] order.
    ///
    /// This is what a benchmark arm prints so its executed path is provable
    /// from the transcript rather than inferred from the flag it was given.
    #[must_use]
    pub fn summary(&self) -> String {
        MLA_DECODE_PATHS
            .iter()
            .map(|p| format!("{}={}", p.label(), self.get(*p)))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Read the counters without clearing them.
#[must_use]
pub fn snapshot() -> MlaDispatchCounts {
    MlaDispatchCounts {
        decompressed: DECOMPRESSED.load(Ordering::Relaxed),
        absorbed_prefill: ABSORBED_PREFILL.load(Ordering::Relaxed),
        absorbed_composed: ABSORBED_COMPOSED.load(Ordering::Relaxed),
        absorbed_split_kv: ABSORBED_SPLIT_KV.load(Ordering::Relaxed),
    }
}

/// Read the counters and reset them to zero.
///
/// A benchmark calls this between arms so each arm's summary describes only
/// that arm.
#[must_use]
pub fn take() -> MlaDispatchCounts {
    MlaDispatchCounts {
        decompressed: DECOMPRESSED.swap(0, Ordering::Relaxed),
        absorbed_prefill: ABSORBED_PREFILL.swap(0, Ordering::Relaxed),
        absorbed_composed: ABSORBED_COMPOSED.swap(0, Ordering::Relaxed),
        absorbed_split_kv: ABSORBED_SPLIT_KV.swap(0, Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_distinct_and_stable() {
        let mut seen = std::collections::HashSet::new();
        for p in MLA_DECODE_PATHS {
            assert!(seen.insert(p.label()), "duplicate label {}", p.label());
        }
        assert_eq!(MlaDecodePath::AbsorbedComposed.label(), "absorbed_composed");
        assert_eq!(MlaDecodePath::AbsorbedSplitKv.label(), "absorbed_split_kv");
    }

    #[test]
    fn summary_names_every_path_so_a_zero_is_visible() {
        let counts = MlaDispatchCounts {
            decompressed: 7,
            ..Default::default()
        };
        let s = counts.summary();
        for p in MLA_DECODE_PATHS {
            assert!(s.contains(p.label()), "{s} is missing {}", p.label());
        }
        // The trap this exists to catch: an "absorbed" arm whose absorbed
        // counter is zero must read as zero, not be omitted.
        assert!(s.contains("absorbed_composed=0"), "{s}");
        assert_eq!(counts.total(), 7);
        assert_eq!(counts.get(MlaDecodePath::Decompressed), 7);
    }
}
