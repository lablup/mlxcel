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

//! HIP `gfx` targets: what this binary was compiled for, what it is running
//! on, and whether those two agree (issue #1805).
//!
//! The ROCm counterpart of [`crate::cuda_arch`], and deliberately not a reuse
//! of it. CUDA's coverage rule is a numeric one: a cubin for a lower compute
//! capability runs on a higher minor revision, and PTX JITs forward across
//! majors. HIP has neither property. A code object is built for one `gfx`
//! target and runs on that target, so coverage here is set membership, not an
//! ordering. Modelling it with `cuda_arch`'s `ArchVariant` ladder would invent
//! a compatibility rule the runtime does not have.
//!
//! The comparison is string equality after normalization, because that is what
//! the loader does. `gfx1151` and `gfx1150` are different code objects even
//! though the numbers are adjacent.
//!
//! One exception exists and it is handled by not deciding. ROCm 6.3 added
//! *generic* targets (`gfx11-generic`, `gfx10-3-generic`), whose code objects
//! do load on several specific targets, so for those the relation is
//! subsumption after all. Rather than reimplement LLVM's generic-target table
//! and get it subtly wrong, an architecture list containing any entry this
//! module does not recognize as a plain target makes the check report nothing.
//! That is the same direction an unrecorded list takes: a build that cannot be
//! judged must still start, since the cost of a wrong refusal is a binary that
//! will not run at all and the cost of a missed one is the opaque HIP error
//! this check was going to improve on.

use std::fmt;
use std::sync::OnceLock;

use crate::hardware::{GpuBackendKind, gpu_backend_kind};

// ── Compiled architecture list ────────────────────────────────────────────────

/// The `MLX_ROCM_ARCHITECTURES` list this binary's MLX device code was compiled
/// for, as a CMake-style semicolon-separated string (`"gfx1151"`).
///
/// Empty on any build without the `rocm` feature. Callers treat empty as
/// "unknown", never as "covers nothing", so a build whose script predates this
/// record degrades to skipping the check rather than refusing to start.
#[must_use]
#[inline]
pub fn compiled_rocm_architectures() -> &'static str {
    option_env!("MLXCEL_ROCM_ARCHITECTURES").unwrap_or("")
}

/// Split a `MLX_ROCM_ARCHITECTURES` string into normalized `gfx` targets.
///
/// Accepts the `;` CMake uses plus the `,` and whitespace an operator is likely
/// to type (`--offload-arch` habits), drops empty entries, lowercases, and
/// strips the ROCm target-feature suffixes
/// (`gfx90a:xnack+` becomes `gfx90a`). Those suffixes select code-object
/// features on one target rather than naming a different one, so keeping them
/// would make a build that does cover the device look like one that does not.
#[must_use]
pub fn parse_rocm_arch_list(list: &str) -> Vec<String> {
    list.split([';', ',', ' ', '\t'])
        .map(normalize_gfx_target)
        .filter(|entry| !entry.is_empty())
        .collect()
}

/// True for a plain `gfx` target whose coverage is decided by equality.
///
/// False for a generic target (`gfx11-generic`) and for anything that does not
/// look like a target at all. Both make [`covers`] decline to answer rather
/// than answer "not covered", which is what keeps an unrecognized spelling from
/// manufacturing a startup refusal.
#[must_use]
pub fn is_specific_gfx_target(entry: &str) -> bool {
    entry.starts_with("gfx")
        && entry.len() > 3
        && entry[3..].chars().all(|c| c.is_ascii_alphanumeric())
}

/// Whether `entries` covers `device`, or `None` when this module cannot say.
///
/// `Some(true)` and `Some(false)` are only returned when every entry is a plain
/// `gfx` target, since that is the case where equality is the whole rule. An
/// empty list, a generic target, or an entry of an unfamiliar shape yields
/// `None`, and every caller reads `None` as "nothing to report".
#[must_use]
pub fn covers(entries: &[String], device: &str) -> Option<bool> {
    if entries.is_empty() || !is_specific_gfx_target(device) {
        return None;
    }
    if !entries.iter().all(|entry| is_specific_gfx_target(entry)) {
        return None;
    }
    Some(entries.iter().any(|entry| entry == device))
}

/// Lowercase, trim, and drop any `:feature` suffix.
#[must_use]
pub fn normalize_gfx_target(target: &str) -> String {
    target
        .trim()
        .split(':')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

// ── Running target ────────────────────────────────────────────────────────────

static DEVICE_GFX: OnceLock<Option<String>> = OnceLock::new();

/// The `gfx` target of the running device, normalized, or `None` when there is
/// none to report.
///
/// `Some("gfx1151")` on the Radeon 8060S. `None` on Metal, CUDA and CPU-only
/// builds. Note that every backend publishes `device_info()["architecture"]`,
/// each in its own vocabulary (`sm_89` on CUDA), so what makes this ROCm-only
/// is the backend gate, not the key's absence. The value cannot change within a
/// process, so it is probed at most once.
///
/// Device 0, matching [`crate::cuda_arch::cuda_compute_capability`]. On a
/// heterogeneous multi-GPU host that is a real limitation, and a sharper one
/// here than on CUDA because this feeds a startup refusal: a binary could
/// refuse over a device it would never have used. No such host exists in the
/// supported matrix; widening both probes to the selected device is its own
/// change.
#[must_use]
pub fn device_gfx_target() -> Option<&'static str> {
    DEVICE_GFX
        .get_or_init(|| {
            if gpu_backend_kind() != GpuBackendKind::Rocm {
                return None;
            }
            let raw = crate::ffi::gpu_architecture(0);
            let normalized = normalize_gfx_target(&raw);
            (!normalized.is_empty()).then_some(normalized)
        })
        .as_deref()
}

// ── Mismatch ──────────────────────────────────────────────────────────────────

/// The running GPU's `gfx` target is not among the architectures this binary
/// was compiled for.
///
/// Every kernel launch on this host would fail to find a code object, so this
/// is reported at device init rather than surfacing as an opaque HIP error
/// inside the first forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmArchMismatch {
    /// The `gfx` target of the device that is actually present.
    pub device: String,
    /// The architecture list this binary was compiled for, verbatim.
    pub compiled: &'static str,
}

impl fmt::Display for RocmArchMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this mlxcel binary was compiled for HIP architectures [{}] and cannot run on the GPU \
             in this host, which is {}. HIP code objects are built per target and do not carry to \
             another one, so every kernel launch would fail to find one. Rebuild for this device \
             with MLX_ROCM_ARCHITECTURES={} (see docs/installation.md, \"ROCm architecture \
             selection\"), or install a build whose architecture list covers it. To run this \
             binary as it is, set MLXCEL_DEVICE=cpu.",
            self.compiled, self.device, self.device
        )
    }
}

impl std::error::Error for RocmArchMismatch {}

/// Describe the mismatch between the running device and the compiled list, or
/// `None` when there is nothing to report.
///
/// Returns `None` whenever either side is unknown: off ROCm, with no visible
/// device, and when the build recorded no architecture list. An empty list is
/// deliberately "unknown" rather than "covers nothing", so a build script that
/// did not emit the record can never manufacture a startup failure.
#[must_use]
pub fn rocm_arch_mismatch() -> Option<RocmArchMismatch> {
    let device = device_gfx_target()?;
    let compiled = compiled_rocm_architectures();
    if covers(&parse_rocm_arch_list(compiled), device) != Some(false) {
        return None;
    }
    Some(RocmArchMismatch {
        device: device.to_owned(),
        compiled,
    })
}

/// Fail when the running device is not covered by the compiled architectures.
///
/// Inert on every build where [`rocm_arch_mismatch`] has nothing to report,
/// which includes all Metal, CUDA and CPU-only builds.
pub fn enforce_rocm_arch_compatibility() -> Result<(), RocmArchMismatch> {
    match rocm_arch_mismatch() {
        Some(mismatch) => Err(mismatch),
        None => Ok(()),
    }
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

/// One-line architecture summary for the startup device report, or `None` when
/// there is no `gfx` target to report (Metal, CUDA, CPU-only).
#[must_use]
pub fn rocm_arch_startup_summary() -> Option<String> {
    let device = device_gfx_target()?;
    let compiled = compiled_rocm_architectures();
    let mut summary = format!("HIP architecture {device}");
    if compiled.is_empty() {
        return Some(summary);
    }
    summary.push_str(&format!("; compiled for [{compiled}]"));
    match covers(&parse_rocm_arch_list(compiled), device) {
        Some(false) => summary.push_str(" (not covered)"),
        // `None` is "this module will not judge this list", not "covered".
        None => summary.push_str(" (coverage undetermined)"),
        Some(true) => {}
    }
    Some(summary)
}

/// Print the resolved `gfx` target and the compiled list once per process,
/// when `MLXCEL_TRACE_ARCH` is set.
///
/// Shares the variable with [`crate::cuda_arch::trace_arch_once`] and its own
/// latch, so a ROCm build traces the ROCm picture and a CUDA build the CUDA
/// one without either needing to know which backend is live.
pub fn trace_rocm_arch_once() {
    if !crate::cuda_arch::trace_arch_enabled() {
        return;
    }
    static EMITTED: OnceLock<()> = OnceLock::new();
    if EMITTED.set(()).is_err() {
        return;
    }
    let compiled = compiled_rocm_architectures();
    let compiled = if compiled.is_empty() {
        "<none: non-ROCm build>".to_owned()
    } else {
        format!("[{compiled}]")
    };
    match device_gfx_target() {
        Some(device) => {
            let coverage = if rocm_arch_mismatch().is_some() {
                "not covered"
            } else {
                "covered"
            };
            eprintln!(
                "[mlxcel arch] HIP architecture {device}; compiled for {compiled}; \
                 coverage: {coverage}"
            );
        }
        None => {
            eprintln!(
                "[mlxcel arch] HIP architecture unavailable (no ROCm device); \
                 compiled for {compiled}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_parser_accepts_both_separators_and_normalizes() {
        assert_eq!(parse_rocm_arch_list("gfx1151"), vec!["gfx1151"]);
        assert_eq!(
            parse_rocm_arch_list("gfx1100;gfx1151"),
            vec!["gfx1100", "gfx1151"]
        );
        assert_eq!(
            parse_rocm_arch_list("gfx1100, GFX1151"),
            vec!["gfx1100", "gfx1151"]
        );
        assert_eq!(parse_rocm_arch_list(";;gfx1151;;"), vec!["gfx1151"]);
        assert!(parse_rocm_arch_list("").is_empty());
        assert!(parse_rocm_arch_list("  ;  ").is_empty());
    }

    #[test]
    fn target_features_are_stripped_from_the_target_name() {
        // `gfx90a:xnack+` selects a code-object feature on gfx90a; it is not a
        // different target. Keeping the suffix would make a build that does
        // cover the device look like one that does not.
        assert_eq!(normalize_gfx_target("gfx90a:xnack+"), "gfx90a");
        assert_eq!(normalize_gfx_target("gfx90a:sramecc-:xnack+"), "gfx90a");
        assert_eq!(normalize_gfx_target("  GFX1151  "), "gfx1151");
        assert_eq!(normalize_gfx_target(""), "");
    }

    #[test]
    fn coverage_is_set_membership_and_not_an_ordering() {
        // The property that separates this from `cuda_arch`: adjacent numbers
        // are unrelated targets, and a higher one does not subsume a lower one.
        let list = parse_rocm_arch_list("gfx1100;gfx1151");
        assert_eq!(covers(&list, "gfx1151"), Some(true));
        assert_eq!(covers(&list, "gfx1100"), Some(true));
        assert_eq!(
            covers(&list, "gfx1150"),
            Some(false),
            "adjacent is unrelated"
        );
        assert_eq!(
            covers(&list, "gfx1103"),
            Some(false),
            "lower is not subsumed"
        );
        assert_eq!(
            covers(&list, "gfx1200"),
            Some(false),
            "higher is not subsumed"
        );
        // A feature suffix on either side names the same target.
        assert_eq!(
            covers(&parse_rocm_arch_list("gfx90a:xnack+"), "gfx90a"),
            Some(true)
        );
    }

    #[test]
    fn a_generic_target_makes_the_check_decline_rather_than_refuse() {
        // ROCm 6.3 generic targets (`gfx11-generic`) do load on several
        // specific targets, so equality would report "not covered" for a build
        // that runs. Declining to judge is the direction that cannot turn a
        // working binary into one that refuses to start.
        let generic = parse_rocm_arch_list("gfx11-generic");
        assert_eq!(covers(&generic, "gfx1151"), None);
        // Mixed lists decline too: one unjudgeable entry is enough.
        let mixed = parse_rocm_arch_list("gfx1100;gfx11-generic");
        assert_eq!(covers(&mixed, "gfx1151"), None);
        // And a list of only plain targets still answers.
        assert_eq!(
            covers(&parse_rocm_arch_list("gfx1100;gfx1151"), "gfx1151"),
            Some(true)
        );
    }

    #[test]
    fn an_empty_or_unrecognized_side_never_answers_not_covered() {
        // Every "cannot judge" case must be `None`, never `Some(false)`:
        // `rocm_arch_mismatch` only refuses on `Some(false)`.
        assert_eq!(covers(&[], "gfx1151"), None);
        assert_eq!(covers(&parse_rocm_arch_list("gfx1151"), ""), None);
        assert_eq!(covers(&parse_rocm_arch_list("gfx1151"), "sm_89"), None);
        assert_eq!(covers(&parse_rocm_arch_list("sm_89"), "gfx1151"), None);
        assert_eq!(covers(&parse_rocm_arch_list("gfx"), "gfx1151"), None);
    }

    #[test]
    fn whitespace_separated_lists_do_not_manufacture_a_refusal() {
        // `--offload-arch` habits produce space-separated lists. Parsed as one
        // entry, "gfx1100 gfx1151" would match nothing and refuse a build that
        // covers the device.
        let list = parse_rocm_arch_list("gfx1100 gfx1151");
        assert_eq!(list, vec!["gfx1100", "gfx1151"]);
        assert_eq!(covers(&list, "gfx1151"), Some(true));
    }

    #[test]
    fn a_mismatch_names_both_targets_and_the_cpu_bypass() {
        let mismatch = RocmArchMismatch {
            device: "gfx1151".to_owned(),
            compiled: "gfx1100",
        };
        let message = mismatch.to_string();
        assert!(
            message.contains("gfx1151"),
            "must name the device: {message}"
        );
        assert!(
            message.contains("gfx1100"),
            "must name the compiled list: {message}"
        );
        assert!(
            message.contains("MLX_ROCM_ARCHITECTURES=gfx1151"),
            "must say how to rebuild for this device: {message}"
        );
        assert!(
            message.contains("MLXCEL_DEVICE=cpu"),
            "must name the bypass: {message}"
        );
    }

    #[test]
    fn the_probe_agrees_with_the_backend() {
        // Every backend publishes an `architecture`, so the gate on the
        // resolved backend is the whole of what makes this ROCm-only: a CUDA
        // host's `sm_89` must not come back from here.
        let target = device_gfx_target();
        if gpu_backend_kind() != GpuBackendKind::Rocm {
            let raw = crate::ffi::gpu_architecture(0);
            assert!(
                target.is_none(),
                "backend {:?} publishes architecture {raw:?}; the gate must suppress it",
                gpu_backend_kind()
            );
        }
        if gpu_backend_kind() == GpuBackendKind::Rocm {
            // A ROCm build with no visible device legitimately reports `None`.
            if let Some(target) = target {
                assert!(
                    target.starts_with("gfx"),
                    "implausible HIP target {target:?} from the device probe"
                );
                assert_eq!(target, normalize_gfx_target(target), "must be normalized");
            }
        } else {
            assert_eq!(
                target,
                None,
                "only ROCm has a gfx target; backend is {:?}",
                gpu_backend_kind()
            );
        }
        // The summary exists exactly when a target does.
        assert_eq!(rocm_arch_startup_summary().is_some(), target.is_some());
        // And nothing to compare means nothing to refuse.
        if target.is_none() {
            assert_eq!(rocm_arch_mismatch(), None);
            assert!(enforce_rocm_arch_compatibility().is_ok());
        }
    }

    #[test]
    fn an_unrecorded_architecture_list_never_manufactures_a_refusal() {
        // The build script may not have emitted the record. Empty is "unknown",
        // never "covers nothing": a build that cannot tell must still start.
        if compiled_rocm_architectures().is_empty() {
            assert_eq!(rocm_arch_mismatch(), None);
        }
    }

    #[test]
    fn this_rocm_build_can_run_on_the_host_running_its_tests() {
        // A ROCm binary that cannot drive its own build host is the failure
        // this module exists to name. Off ROCm, or with no recorded list, there
        // is nothing to check and the assertion below holds trivially; the name
        // says ROCm so a green run elsewhere is not read as coverage.
        assert_eq!(
            rocm_arch_mismatch(),
            None,
            "this build's HIP architectures do not cover the GPU running its tests"
        );
        if gpu_backend_kind() == GpuBackendKind::Rocm && !compiled_rocm_architectures().is_empty() {
            // On the host this issue was measured on, the check is live rather
            // than short-circuited: both sides are known and they agree.
            let device = device_gfx_target().expect("a ROCm device reports a gfx target");
            assert_eq!(
                covers(&parse_rocm_arch_list(compiled_rocm_architectures()), device),
                Some(true),
                "the coverage check must actually decide on a ROCm build"
            );
        }
    }
}
