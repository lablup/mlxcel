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

//! CUDA compute capability: what this binary was compiled for, what it is
//! running on, and whether those two agree (issue #1537).
//!
//! Three values meet here.
//!
//! - The **compiled architecture list** is the `MLX_CUDA_ARCHITECTURES` string
//!   that `build.rs` handed to CMake, recorded back into the crate as
//!   `MLXCEL_CUDA_ARCHITECTURES` so the binary knows what device code it
//!   actually carries. Empty off CUDA.
//! - The **running compute capability** comes from MLX's own cached device
//!   attributes through the `gpu_compute_capability` bridge function. `None`
//!   on Metal, on a CPU-only build, and on a CUDA build with no usable device.
//! - The **coverage verdict** is the predicate over those two that decides
//!   whether the running device can execute this binary's kernels at all.
//!
//! Until this module existed, mlxcel had no notion of compute capability
//! anywhere in its own code: every architecture decision was delegated to
//! MLX's C++ gates, so a binary whose cubin set did not cover the host failed
//! with an opaque CUDA load error at the first kernel launch rather than
//! saying which architectures it was built for. The published x86_64 release
//! matrix starts at `80`, so a released archive cannot run on a V100 at all,
//! and `build.rs` falls back to `90a` when `nvidia-smi` is missing, which
//! yields a binary that cannot run on its own build host. Both cases are now
//! a named error at device init.
//!
//! This module is plumbing and diagnostics only. Nothing here changes which
//! kernel runs; consumers that act on the capability are separate work.

use std::fmt;
use std::sync::OnceLock;

// ── Compiled architecture list ────────────────────────────────────────────────

/// The `MLX_CUDA_ARCHITECTURES` list this binary's MLX device code was
/// compiled for, as a CMake-style semicolon-separated string
/// (`"80;86;89;90a;100;120"`).
///
/// Empty on any build without the `cuda` feature, and empty on a CUDA build
/// whose build script predates this record. Callers treat empty as "unknown",
/// never as "covers nothing".
#[must_use]
#[inline]
pub fn compiled_cuda_architectures() -> &'static str {
    // `option_env!` rather than `env!` so the crate still compiles if the
    // build script did not emit the variable; the runtime then degrades to
    // skipping the architecture check instead of failing to build.
    option_env!("MLXCEL_CUDA_ARCHITECTURES").unwrap_or("")
}

// ── Runtime compute capability ────────────────────────────────────────────────

static COMPUTE_CAPABILITY: OnceLock<Option<(u32, u32)>> = OnceLock::new();

/// The CUDA compute capability of GPU 0 as `(major, minor)`, or `None` when
/// there is none to report.
///
/// `Some((7, 0))` on a V100, `Some((12, 1))` on GB10. `None` on a Metal build,
/// on a CPU-only build, and on a CUDA build running where no device is
/// visible. The value cannot change within a process, so it is probed at most
/// once and cached.
///
/// The probe reads the compute-capability attributes MLX already caches per
/// device rather than issuing a second CUDA call, and reading device
/// properties never loads a cubin. That matters: this stays callable on a
/// binary whose compiled architectures do not cover the running device, which
/// is the only reason [`cuda_arch_mismatch`] can report anything useful
/// instead of dying first.
#[must_use]
pub fn cuda_compute_capability() -> Option<(u32, u32)> {
    *COMPUTE_CAPABILITY
        .get_or_init(|| unpack_compute_capability(crate::ffi::gpu_compute_capability(0)))
}

/// Split the bridge's `major * 1000 + minor` packing, mapping the `-1`
/// sentinel (and any other negative value) to `None`.
fn unpack_compute_capability(packed: i32) -> Option<(u32, u32)> {
    if packed < 0 {
        return None;
    }
    let packed = u32::try_from(packed).ok()?;
    Some((packed / 1000, packed % 1000))
}

// ── Architecture list parsing ─────────────────────────────────────────────────

/// Which code objects a single architecture-list entry contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArchVariant {
    /// A plain entry such as `80`. Its cubin follows CUDA's minor-revision
    /// binary compatibility rule and its PTX can be JIT-compiled forward.
    Generic,
    /// An `a`-suffixed entry such as `90a`. Architecture-specific code objects
    /// use instructions that exist only on that exact target, so neither the
    /// cubin nor the PTX carries to any other compute capability.
    ArchSpecific,
    /// An `f`-suffixed entry such as `100f`. Family-specific code objects
    /// carry forward within the same major version but not across one.
    FamilySpecific,
}

/// One entry of a CMake `CUDA_ARCHITECTURES` list, parsed.
///
/// CMake spells an entry as `<sm>[a|f][-real|-virtual]`: the number packs the
/// compute capability with the minor version as its last digit (`70` is 7.0,
/// `121` is 12.1), the letter selects the architecture- or family-specific
/// variant, and the trailing word restricts the entry to cubin (`-real`) or
/// PTX (`-virtual`) only. A bare entry emits both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaArchEntry {
    /// Compute capability major version (`8` for `86`).
    pub major: u32,
    /// Compute capability minor version (`6` for `86`).
    pub minor: u32,
    /// Architecture-specific / family-specific qualifier.
    pub variant: ArchVariant,
    /// Whether the entry contributes a cubin (false for `-virtual`).
    pub emits_cubin: bool,
    /// Whether the entry contributes PTX (false for `-real`).
    pub emits_ptx: bool,
}

impl fmt::Display for CudaArchEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.major, self.minor)?;
        match self.variant {
            ArchVariant::Generic => {}
            ArchVariant::ArchSpecific => f.write_str("a")?,
            ArchVariant::FamilySpecific => f.write_str("f")?,
        }
        match (self.emits_cubin, self.emits_ptx) {
            (true, false) => f.write_str("-real"),
            (false, true) => f.write_str("-virtual"),
            _ => Ok(()),
        }
    }
}

/// How a compiled architecture entry covers a running device, if at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArchCoverage {
    /// The device runs JIT-compiled PTX. Correct, but pays a compile at first
    /// launch.
    Ptx,
    /// The device runs a precompiled cubin directly.
    Cubin,
}

impl fmt::Display for ArchCoverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchCoverage::Cubin => f.write_str("cubin"),
            ArchCoverage::Ptx => f.write_str("PTX JIT"),
        }
    }
}

/// Parse one CMake architecture-list entry, or `None` if it is not one.
///
/// Unrecognised entries are dropped rather than rejected: the list can be set
/// verbatim by an operator, and one entry this parser does not know must not
/// be able to turn into a spurious startup failure.
#[must_use]
pub fn parse_cuda_arch_entry(entry: &str) -> Option<CudaArchEntry> {
    let entry = entry.trim();
    if entry.is_empty() {
        return None;
    }

    let (head, emits_cubin, emits_ptx) = match entry {
        e if e.ends_with("-real") => (&e[..e.len() - "-real".len()], true, false),
        e if e.ends_with("-virtual") => (&e[..e.len() - "-virtual".len()], false, true),
        e => (e, true, true),
    };

    let (digits, variant) = match head.as_bytes().last() {
        Some(b'a') => (&head[..head.len() - 1], ArchVariant::ArchSpecific),
        Some(b'f') => (&head[..head.len() - 1], ArchVariant::FamilySpecific),
        _ => (head, ArchVariant::Generic),
    };

    // The minor version is the last digit; everything before it is the major.
    // Two digits is the minimum that can carry both (`70`).
    if digits.len() < 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (major, minor) = digits.split_at(digits.len() - 1);
    Some(CudaArchEntry {
        major: major.parse().ok()?,
        minor: minor.parse().ok()?,
        variant,
        emits_cubin,
        emits_ptx,
    })
}

/// Parse a whole `MLX_CUDA_ARCHITECTURES` list, dropping entries this parser
/// does not recognise.
///
/// Accepts both the CMake separator (`;`) and a comma, since operators write
/// both.
#[must_use]
pub fn parse_cuda_arch_list(list: &str) -> Vec<CudaArchEntry> {
    list.split([';', ','])
        .filter_map(parse_cuda_arch_entry)
        .collect()
}

// ── Coverage predicate ────────────────────────────────────────────────────────

/// How, if at all, one compiled entry covers a device of compute capability
/// `device`.
///
/// The rules are CUDA's own:
///
/// - A cubin is binary compatible across minor revisions upward within one
///   major version, never downward and never across majors: `sm_120` code runs
///   on a 12.1 device, `sm_121` code does not run on a 12.0 device, and neither
///   runs on any 9.x device.
/// - PTX is forward compatible: the driver can JIT `compute_80` PTX for any
///   device at 8.0 or newer, including later major versions.
/// - Architecture-specific (`a`) code objects opt out of both rules. `90a`
///   uses Hopper-only instructions, so it covers 9.0 and nothing else, which
///   is exactly why the release matrices spell Hopper as `90a` and why a
///   `90a`-only binary is stranded on every other card.
/// - Family-specific (`f`) code objects carry forward within their major
///   version only.
#[must_use]
pub fn entry_coverage(entry: CudaArchEntry, device: (u32, u32)) -> Option<ArchCoverage> {
    let (device_major, device_minor) = device;
    match entry.variant {
        ArchVariant::ArchSpecific => {
            if entry.major == device_major && entry.minor == device_minor {
                emitted_coverage(entry)
            } else {
                None
            }
        }
        ArchVariant::FamilySpecific => {
            if entry.major == device_major && entry.minor <= device_minor {
                emitted_coverage(entry)
            } else {
                None
            }
        }
        ArchVariant::Generic => {
            if entry.emits_cubin && entry.major == device_major && entry.minor <= device_minor {
                Some(ArchCoverage::Cubin)
            } else if entry.emits_ptx && (entry.major, entry.minor) <= (device_major, device_minor)
            {
                Some(ArchCoverage::Ptx)
            } else {
                None
            }
        }
    }
}

/// The better of the code objects an entry emits, for a target the entry is
/// already known to apply to.
fn emitted_coverage(entry: CudaArchEntry) -> Option<ArchCoverage> {
    if entry.emits_cubin {
        Some(ArchCoverage::Cubin)
    } else if entry.emits_ptx {
        Some(ArchCoverage::Ptx)
    } else {
        None
    }
}

/// The best coverage a whole architecture list offers a device, or `None` when
/// no entry covers it.
///
/// "Best" is cubin over PTX: both run, but a cubin match means no JIT at first
/// launch.
#[must_use]
pub fn arch_list_coverage(list: &str, device: (u32, u32)) -> Option<ArchCoverage> {
    parse_cuda_arch_list(list)
        .into_iter()
        .filter_map(|entry| entry_coverage(entry, device))
        .max()
}

// ── Mismatch ──────────────────────────────────────────────────────────────────

/// The running GPU's compute capability is not covered by any architecture
/// this binary was compiled for.
///
/// Every kernel launch on this host would fail, so this is reported at device
/// init instead of being discovered as an opaque CUDA load error deep inside
/// the first forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaArchMismatch {
    /// Compute capability of the device that is actually present.
    pub device: (u32, u32),
    /// The architecture list this binary was compiled for, verbatim.
    pub compiled: &'static str,
}

impl CudaArchMismatch {
    /// The `MLX_CUDA_ARCHITECTURES` value that would build for this device.
    ///
    /// Applies the same rule `build.rs`'s `sm_arch_with_suffix` applies when it
    /// auto-detects: CUDA's architecture-specific `a` suffix from SM 90 up. The
    /// suffix is load-bearing on Hopper and newer, where MLX compiles its
    /// dedicated quantized kernel only when the list says `90a` rather than
    /// `90`. Suggesting anything else here would send an operator to a rebuild
    /// that differs from the one auto-detection would have produced.
    #[must_use]
    pub fn suggested_architecture(&self) -> String {
        let (major, minor) = self.device;
        let sm = major * 10 + minor;
        let suffix = if sm >= 90 { "a" } else { "" };
        format!("{sm}{suffix}")
    }
}

impl fmt::Display for CudaArchMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (major, minor) = self.device;
        write!(
            f,
            "this mlxcel binary was compiled for CUDA architectures [{}] and cannot run on the \
             GPU in this host, which is compute capability {major}.{minor} (sm_{major}{minor}). \
             No compiled architecture provides a cubin for sm_{major}{minor}, and none is old \
             enough for the driver to JIT its PTX forward, so every kernel launch would fail. \
             Rebuild for this device with MLX_CUDA_ARCHITECTURES={} (see docs/installation.md, \
             \"CUDA architecture selection\"), or install a build whose architecture list covers \
             it. To run this binary as it is, set MLXCEL_DEVICE=cpu.",
            self.compiled,
            self.suggested_architecture()
        )
    }
}

impl std::error::Error for CudaArchMismatch {}

/// Check the running device against the compiled architecture list.
///
/// Returns `None` (meaning "nothing to report") whenever either side is
/// unknown: off CUDA, with no visible device, when the build recorded no
/// architecture list, and when no entry in a non-empty list could be parsed.
/// An unparseable list is deliberately treated as unknown rather than as
/// "covers nothing" so an architecture spelling this parser has not learned
/// yet can never manufacture a startup failure.
#[must_use]
pub fn cuda_arch_mismatch() -> Option<CudaArchMismatch> {
    let device = cuda_compute_capability()?;
    let compiled = compiled_cuda_architectures();
    let entries = parse_cuda_arch_list(compiled);
    if entries.is_empty() {
        return None;
    }
    if entries
        .iter()
        .any(|entry| entry_coverage(*entry, device).is_some())
    {
        return None;
    }
    Some(CudaArchMismatch { device, compiled })
}

/// Fail when the running device is not covered by the compiled architectures.
///
/// The `Result` form binaries call at device init, so an incompatible build
/// exits with the named error above rather than crawling into a CUDA load
/// failure. Inert on every build where [`cuda_arch_mismatch`] has nothing to
/// report, which includes all Metal and CPU-only builds.
pub fn enforce_cuda_arch_compatibility() -> Result<(), CudaArchMismatch> {
    match cuda_arch_mismatch() {
        Some(mismatch) => Err(mismatch),
        None => Ok(()),
    }
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

/// One-line architecture summary for the startup device report, or `None` when
/// there is no compute capability to report (Metal, CPU-only).
///
/// Sits next to the `Detected N GPU(s)` line so an operator reading a log can
/// see the running capability, the architectures the binary carries, and
/// whether the device is served by a cubin or by JIT-compiled PTX.
#[must_use]
pub fn cuda_arch_startup_summary() -> Option<String> {
    let (major, minor) = cuda_compute_capability()?;
    let compiled = compiled_cuda_architectures();
    let mut summary = format!("CUDA compute capability {major}.{minor} (sm_{major}{minor})");
    if compiled.is_empty() {
        return Some(summary);
    }
    summary.push_str(&format!("; compiled for [{compiled}]"));
    match arch_list_coverage(compiled, (major, minor)) {
        Some(coverage) => summary.push_str(&format!(" ({coverage})")),
        None => summary.push_str(" (not covered)"),
    }
    Some(summary)
}

/// Environment variable that enables the one-shot architecture trace.
pub const TRACE_ARCH_ENV: &str = "MLXCEL_TRACE_ARCH";

static TRACE_ARCH_EMITTED: OnceLock<()> = OnceLock::new();

/// True when the architecture trace is enabled for this process.
///
/// Presence-enabled, matching the other diagnostic switches in
/// `docs/environment-variables.md`: any value, including the empty string,
/// turns it on. Read on each call, which costs nothing here because the only
/// caller runs once per process; the CUDA `QuantizedMatmul` overlay reads the
/// same variable but caches it, since it sits on the decode path.
#[must_use]
#[inline]
pub fn trace_arch_enabled() -> bool {
    std::env::var_os(TRACE_ARCH_ENV).is_some()
}

/// Print the resolved capability and the compiled architecture list once per
/// process, when `MLXCEL_TRACE_ARCH` is set.
///
/// Goes to stderr so it never mixes into generated text on stdout. Repeated
/// calls after the first are a no-op, which is what lets every device-init
/// path call it without coordinating.
pub fn trace_arch_once() {
    if !trace_arch_enabled() {
        return;
    }
    if TRACE_ARCH_EMITTED.set(()).is_err() {
        return;
    }
    let compiled = compiled_cuda_architectures();
    let compiled = if compiled.is_empty() {
        "<none: non-CUDA build>".to_owned()
    } else {
        format!("[{compiled}]")
    };
    match cuda_compute_capability() {
        Some((major, minor)) => {
            let coverage = arch_list_coverage(compiled_cuda_architectures(), (major, minor))
                .map_or_else(|| "not covered".to_owned(), |c| c.to_string());
            eprintln!(
                "[mlxcel arch] compute capability {major}.{minor} (sm_{major}{minor}); \
                 compiled for {compiled}; coverage: {coverage}"
            );
        }
        None => {
            eprintln!(
                "[mlxcel arch] compute capability unavailable (no CUDA device); \
                 compiled for {compiled}"
            );
        }
    }
    // The quantized-matmul path picked on the first call is emitted by the
    // CUDA `QuantizedMatmul::eval_gpu` overlay, which reads the same variable
    // directly; it cannot be reported from here because the choice is not made
    // until a quantized matmul actually runs.
}
