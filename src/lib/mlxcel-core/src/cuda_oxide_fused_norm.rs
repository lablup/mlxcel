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

//! Experimental cuda-oxide route behind the fused residual-add + RMSNorm
//! kernel (issue #2833).
//!
//! This module exists only in a build made with
//! `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1` on Linux with the `cuda` feature.
//! Such a build embeds the PTX module that `spike/cuda-oxide-fused-norm`
//! produced and validated on GB10 (#2832), and still does nothing with it
//! until the process sets `MLXCEL_CUDA_OXIDE_FUSED_NORM=1`.
//!
//! When it is set, the first fused residual join validates the artifact
//! ([`crate::cuda_oxide_artifact`]), checks that device 0 is the artifact's
//! target, and installs it into the C++ adapter
//! (`mlxcelverse/kernels/fused_norm_oxide.cpp`). From then on
//! `fused_add_rms_norm` builds the cuda-oxide node for the dtype pairs the
//! module has an entry for and the existing CUDA kernel for everything else.
//! The choice is made when the graph node is built, before anything is
//! enqueued. A refused artifact, a device of another architecture, or an unset
//! switch all leave the existing kernel serving every call.
//!
//! The switch selects the implementation behind `fused_add_rms_norm`; it does
//! not turn the fusion on. Model code reaches that kernel only with
//! `MLXCEL_FUSED_ADD_RMSNORM=1` on CUDA.
//!
//! [`route_counters`] is the route marker: a test that claims parity for the
//! cuda-oxide route asserts that the route and launch counters moved and the
//! existing-kernel counter did not, so it cannot pass on a silent fallback.
//!
//! Used by: [`crate::layers::fused_add_rms_norm`] (`ensure_from_env`).

use std::ffi::OsStr;
use std::path::Path;
use std::sync::OnceLock;

use crate::cuda_oxide_artifact::{
    self as artifact, ArtifactError, EntryDocuments, TARGET_CAPABILITY, ValidatedArtifact,
};

/// Runtime opt-in. `1`/`true`/`on`/`yes` enable; anything else leaves the
/// existing CUDA kernel in place. Read once, at the first fused residual join.
pub const ENABLE_ENV: &str = "MLXCEL_CUDA_OXIDE_FUSED_NORM";

/// Optional directory to load the artifact from instead of the embedded copy,
/// laid out like `spike/cuda-oxide-fused-norm` (`artifacts/`, `abi/`). It goes
/// through the same validation; a directory that fails it disables the route
/// instead of falling back to the embedded artifact.
pub const ARTIFACT_DIR_ENV: &str = "MLXCEL_CUDA_OXIDE_FUSED_NORM_DIR";

// `mlxcelverse/kernels/fused_norm_oxide.cpp`.
unsafe extern "C" {
    fn mlxcel_oxide_fused_norm_install(
        payload: *const u8,
        payload_len: usize,
        module_name: *const u8,
        module_name_len: usize,
        target_major: i32,
        target_minor: i32,
        symbols: *const u8,
        symbols_len: usize,
        act_dtypes: *const i32,
        weight_dtypes: *const i32,
        roles: *const u8,
        entry_count: usize,
    ) -> i32;
    fn mlxcel_oxide_fused_norm_uninstall();
    fn mlxcel_oxide_fused_norm_installed() -> bool;
    fn mlxcel_oxide_fused_norm_route_count() -> u64;
    fn mlxcel_oxide_fused_norm_native_route_count() -> u64;
    fn mlxcel_oxide_fused_norm_launch_count() -> u64;
    fn mlxcel_oxide_fused_norm_last_fallback() -> i32;
}

/// Why the artifact was not installed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InstallError {
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error(
        "the cuda-oxide artifact targets sm_{}{}, but device 0 is {device}",
        TARGET_CAPABILITY.0,
        TARGET_CAPABILITY.1
    )]
    Target { device: String },
    #[error("the cuda-oxide adapter rejected the artifact description (code {code})")]
    Rejected { code: i32 },
}

/// Why a call took the existing CUDA kernel. Mirrors `Fallback` in
/// `fused_norm_oxide.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Fallback {
    /// The most recent call took the cuda-oxide route.
    Taken,
    /// No validated artifact is installed.
    NotInstalled,
    /// The default device is the CPU.
    DeviceNotGpu,
    /// The module has no entry for this activation and weight dtype pair.
    DtypePair,
    /// More than 2^32-1 elements, which the artifact was never validated for.
    ElementLimit,
    /// A zero row count or width.
    Shape,
    /// The block width is not a whole number of warps within 1024 lanes.
    BlockDim,
    /// More rows than a CUDA grid's x extent holds (2^31-1); one block runs
    /// per row.
    GridLimit,
    /// A code this build does not know.
    Unknown,
}

/// Monotonic process-wide counters of the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RouteCounters {
    /// Calls routed to the cuda-oxide node (counted at graph construction).
    pub oxide_routes: u64,
    /// Calls that took the existing CUDA kernel instead.
    pub native_routes: u64,
    /// cuda-oxide kernels enqueued by `eval_gpu`: proof of GPU execution.
    pub launches: u64,
}

/// The adapter's counters now.
#[must_use]
pub fn route_counters() -> RouteCounters {
    // SAFETY: the three functions only load an atomic.
    unsafe {
        RouteCounters {
            oxide_routes: mlxcel_oxide_fused_norm_route_count(),
            native_routes: mlxcel_oxide_fused_norm_native_route_count(),
            launches: mlxcel_oxide_fused_norm_launch_count(),
        }
    }
}

/// The outcome of the most recent routing decision.
#[must_use]
pub fn last_fallback() -> Fallback {
    // SAFETY: loads an atomic.
    match unsafe { mlxcel_oxide_fused_norm_last_fallback() } {
        0 => Fallback::Taken,
        1 => Fallback::NotInstalled,
        2 => Fallback::DeviceNotGpu,
        3 => Fallback::DtypePair,
        4 => Fallback::ElementLimit,
        5 => Fallback::Shape,
        6 => Fallback::BlockDim,
        7 => Fallback::GridLimit,
        _ => Fallback::Unknown,
    }
}

/// Whether a validated artifact is installed, so eligible calls take the
/// cuda-oxide route.
#[must_use]
pub fn is_installed() -> bool {
    // SAFETY: loads an atomic.
    unsafe { mlxcel_oxide_fused_norm_installed() }
}

/// Remove the installed artifact; later calls take the existing CUDA kernel.
/// Lazy nodes already built keep the artifact alive and still evaluate.
pub fn uninstall() {
    // SAFETY: takes the adapter's lock and drops its reference.
    unsafe { mlxcel_oxide_fused_norm_uninstall() }
}

/// The embedded artifact: the committed PTX, manifests and schemas, staged by
/// the build script only after their digests matched.
fn embedded() -> (&'static [u8], Vec<EntryDocuments>) {
    macro_rules! staged {
        ($name:expr) => {
            concat!(env!("OUT_DIR"), "/cuda_oxide_fused_norm/", $name)
        };
    }
    macro_rules! entry {
        ($entry:literal) => {
            EntryDocuments {
                entry: $entry.to_string(),
                manifest: include_str!(staged!(concat!($entry, ".sm_121.manifest.json")))
                    .to_string(),
                schema: include_str!(staged!(concat!($entry, ".schema.json"))).to_string(),
            }
        };
    }
    (
        include_bytes!(staged!("fused_add_rms_norm.sm_121.ptx")),
        vec![
            entry!("fused_add_rms_norm_bf16"),
            entry!("fused_add_rms_norm_f16"),
            entry!("fused_add_rms_norm_f32"),
            entry!("fused_add_rms_norm_f16_wf32"),
            entry!("fused_add_rms_norm_bf16_wf32"),
        ],
    )
}

/// Validate the artifact embedded in this build.
pub fn validate_embedded() -> Result<ValidatedArtifact, ArtifactError> {
    let (ptx, documents) = embedded();
    artifact::validate(ptx, &documents)
}

/// Validate the embedded artifact and install it.
pub fn install_embedded() -> Result<(), InstallError> {
    install(&validate_embedded()?)
}

/// Validate the artifact under `dir` and install it. On any failure nothing is
/// installed, including whatever was installed before.
pub fn install_from_dir(dir: &Path) -> Result<(), InstallError> {
    uninstall();
    let (ptx, documents) = artifact::read_dir(dir)?;
    install(&artifact::validate(&ptx, &documents)?)
}

/// Install a validated artifact, provided device 0 is its target.
///
/// The capability check is what keeps PTX built for one architecture off
/// another: the decision is made here, before any graph node exists, and a
/// device that does not match leaves the existing kernel in place.
pub fn install(validated: &ValidatedArtifact) -> Result<(), InstallError> {
    uninstall();
    check_target(crate::cuda_arch::cuda_compute_capability())?;
    install_unchecked(
        &validated.ptx,
        &validated.module_name,
        TARGET_CAPABILITY,
        &validated.entries,
    )
}

/// Whether a device of compute capability `device` (`None` when device 0 is
/// not a CUDA device) can run the artifact.
pub(crate) fn check_target(device: Option<(u32, u32)>) -> Result<(), InstallError> {
    if device == Some(TARGET_CAPABILITY) {
        return Ok(());
    }
    Err(InstallError::Target {
        device: device.map_or_else(
            || "not a CUDA device".to_string(),
            |(major, minor)| format!("sm_{major}{minor}"),
        ),
    })
}

/// Hand an artifact description to the C++ adapter without validating it.
///
/// [`install`] is the only production caller. Tests also use it to install a
/// payload that cannot load, which is how they show that an evaluation-time
/// failure surfaces as an error instead of a retry on the existing kernel.
pub(crate) fn install_unchecked(
    payload: &[u8],
    module_name: &str,
    target: (u32, u32),
    entries: &[artifact::EntrySpec],
) -> Result<(), InstallError> {
    let symbols = entries
        .iter()
        .map(|e| e.symbol.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let act: Vec<i32> = entries.iter().map(|e| e.activation.code()).collect();
    let weight: Vec<i32> = entries.iter().map(|e| e.weight.code()).collect();
    let roles: Vec<u8> = entries
        .iter()
        .flat_map(|e| e.roles.iter().map(|role| *role as u8))
        .collect();
    // SAFETY: every pointer is to a live slice whose length is passed with it
    // (`roles` holds `ARG_COUNT` bytes per entry, `act` and `weight` one code
    // per entry), and the callee copies what it keeps before returning.
    let code = unsafe {
        mlxcel_oxide_fused_norm_install(
            payload.as_ptr(),
            payload.len(),
            module_name.as_ptr(),
            module_name.len(),
            target.0 as i32,
            target.1 as i32,
            symbols.as_ptr(),
            symbols.len(),
            act.as_ptr(),
            weight.as_ptr(),
            roles.as_ptr(),
            entries.len(),
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(InstallError::Rejected { code })
    }
}

/// Install the artifact if the process opted in, once.
///
/// Called at every fused residual join, so after the first call it is one
/// atomic load. A failure is reported once on stderr and leaves the existing
/// kernel in place: an operator who set the switch should learn that it did
/// nothing, and the model must still run.
pub fn ensure_from_env() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        let enable = std::env::var(ENABLE_ENV).ok();
        let dir = std::env::var_os(ARTIFACT_DIR_ENV);
        match install_from_switches(enable.as_deref(), dir.as_deref()) {
            None => {}
            Some(Ok(())) => tracing::info!(
                "cuda-oxide fused RMSNorm route installed ({ENABLE_ENV}=1, experimental)"
            ),
            Some(Err(error)) => eprintln!(
                "mlxcel: {ENABLE_ENV} is set but the cuda-oxide fused RMSNorm route is disabled, using the existing CUDA kernel: {error}"
            ),
        }
    });
}

/// The decision behind [`ensure_from_env`], on explicit values so it can be
/// tested without touching the process environment. `None` when `enable` does
/// not opt in, in which case nothing is installed or removed.
pub(crate) fn install_from_switches(
    enable: Option<&str>,
    dir: Option<&OsStr>,
) -> Option<Result<(), InstallError>> {
    if crate::layers::fused_flag_explicit_value(enable) != Some(true) {
        return None;
    }
    Some(match dir.filter(|dir| !dir.is_empty()) {
        Some(dir) => install_from_dir(Path::new(dir)),
        None => install_embedded(),
    })
}
