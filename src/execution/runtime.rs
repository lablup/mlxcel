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

//! Runtime/device selection helpers shared by all inference entry points.
//!
//! CLI generation and the HTTP server both rely on the same environment-based
//! device resolution so CPU overrides and GPU wired-limit behavior stay
//! consistent regardless of how inference is entered.

use std::fmt;

const RUNTIME_DEVICE_ENV: &str = "MLXCEL_DEVICE";
const WIRED_LIMIT_ENV: &str = "MLXCEL_WIRED_LIMIT";
/// Issue #55: optional soft cap on the MLX allocator. When set, the
/// runtime calls `mlxcel_core::memory::set_memory_limit(...)` at startup
/// so MLX raises an exception once allocations would push the working
/// set past this value, instead of thrashing or OOM-killing the process.
/// Used by the future preflight capstone (#56). Accepts the shared size
/// grammar of [`parse_memory_size`]: plain bytes, `NK`/`NKB`, `NM`/`NMB`
/// or `NG`/`NGB`. Unset means "do not override MLX's default limit".
const MEMORY_LIMIT_ENV: &str = "MLXCEL_MEMORY_LIMIT";
/// Issue #627: optional bound on MLX's buffer cache. When set, the runtime
/// calls `mlxcel_core::memory::set_cache_limit(...)` at startup so the CUDA
/// memory pool stays bounded without the per-decode `clear_memory_cache`
/// churn that defeats CUDA-graph reuse (ml-explore/mlx#2358). Accepts the
/// shared size grammar of [`parse_memory_size`]: plain bytes, `NK`/`NKB`,
/// `NM`/`NMB` or `NG`/`NGB`. Unset means "do not override MLX's default
/// cache behavior".
const CACHE_LIMIT_ENV: &str = "MLXCEL_CACHE_LIMIT";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDevice {
    Cpu,
    Gpu,
}

impl RuntimeDevice {
    const fn uses_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

impl fmt::Display for RuntimeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => {
                #[cfg(feature = "cuda")]
                return write!(f, "NVIDIA GPU (CUDA)");
                #[cfg(target_os = "macos")]
                return write!(f, "Apple GPU (Metal)");
                #[cfg(not(any(feature = "cuda", target_os = "macos")))]
                write!(f, "GPU")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSetup {
    /// The device the runtime resolved to. `Gpu` when a GPU backend is
    /// available (`mlxcel_core::gpu_backend_available()`) and nothing asked
    /// for the CPU; `Cpu` either because no GPU backend exists or because
    /// `MLXCEL_DEVICE=cpu` asked for it, which [`Self::cpu_override`] tells
    /// apart. This is what was resolved, not the MLX default device read
    /// back afterwards: the field used to be filled from
    /// `is_gpu_available()` after the CPU override had already moved the
    /// default device, so it echoed the request instead of reporting
    /// availability (issue #1421).
    pub device: RuntimeDevice,
    /// `true` when `MLXCEL_DEVICE=cpu` moved the MLX default device to the
    /// CPU. Together with [`Self::device`] this separates "CPU by request"
    /// from "CPU because no GPU backend exists", so the startup lines can say
    /// which one applies instead of printing `CPU` for both.
    pub cpu_override: bool,
    pub wired_limit_bytes: Option<usize>,
    /// Soft MLX allocator memory limit applied via `MLXCEL_MEMORY_LIMIT`
    /// (issue #55). `None` when the env var was unset or invalid and
    /// MLX's default limit is in effect.
    pub memory_limit_bytes: Option<usize>,
    /// Buffer-cache bound applied via `MLXCEL_CACHE_LIMIT` (issue #627).
    /// `None` when the env var was unset/invalid and MLX's default cache
    /// behavior is in effect.
    pub cache_limit_bytes: Option<usize>,
    pub invalid_device_override: Option<String>,
}

/// Initialize the runtime, refusing to start when this binary's compiled CUDA
/// architectures do not cover the GPU in this host (issue #1537).
///
/// This is the entry point every binary-facing caller uses. The check runs
/// before any device state is touched, so an incompatible build reports which
/// architectures it carries and which one it found, instead of continuing to
/// the first kernel launch and dying there on a CUDA load error that names
/// neither. It is inert on Metal, on CPU-only builds, and on any CUDA build
/// whose architecture list does cover the device, which is every supported
/// configuration.
///
/// `MLXCEL_DEVICE=cpu` bypasses the refusal: a binary that cannot drive this
/// host's GPU can still run on its CPU, and taking that away would remove the
/// one workaround available to someone holding the wrong archive.
///
/// Tests and other in-process callers that do not own the process exit path
/// keep using [`initialize_runtime`], which performs the same setup without the
/// refusal.
pub fn initialize_runtime_checked() -> Result<RuntimeSetup, mlxcel_core::hardware::CudaArchMismatch>
{
    // Trace before the refusal, so `MLXCEL_TRACE_ARCH` still explains an
    // incompatible build rather than going quiet on the one path where the
    // architecture picture matters most.
    mlxcel_core::hardware::trace_arch_once();
    let (requested_device, _) =
        resolve_runtime_device(std::env::var(RUNTIME_DEVICE_ENV).ok().as_deref());
    if requested_device.uses_gpu() {
        mlxcel_core::hardware::enforce_cuda_arch_compatibility()?;
    }
    Ok(initialize_runtime())
}

pub fn initialize_runtime() -> RuntimeSetup {
    // Device init is the once-per-process point where the architecture trace
    // has something to say, so emit it here; it is a no-op unless
    // `MLXCEL_TRACE_ARCH` is set, and a no-op after the first call.
    mlxcel_core::hardware::trace_arch_once();

    let (requested_device, invalid_device_override) =
        resolve_runtime_device(std::env::var(RUNTIME_DEVICE_ENV).ok().as_deref());

    // Availability is a backend fact and the override is an operator request;
    // resolve them separately so the setup can say which one put the runtime
    // on the CPU (issue #1421). The override moves the MLX default device;
    // the availability query never does.
    let (device, cpu_override) =
        resolve_device(requested_device, mlxcel_core::gpu_backend_available());
    if cpu_override {
        mlxcel_core::set_default_device(false);
    }

    // Footgun guard: a default `cargo build --release` on Linux omits the `cuda`
    // feature and silently runs MLX on the CPU. If the user wanted the GPU but
    // this CPU-only binary fell back to CPU on a host that has an NVIDIA GPU,
    // say so loudly instead of crawling at a fraction of GPU speed. The OpenXLA
    // backend (issue #449) is the exception: it drives inference through IREE,
    // not MLX, and can run on the GPU via its own device (MLXCEL_XLA_DEVICE),
    // so the MLX-CPU-fallback message would be misleading there.
    let xla_backend_selected =
        cfg!(feature = "xla-backend") && std::env::var("MLXCEL_BACKEND").as_deref() == Ok("xla");
    if !xla_backend_selected
        && should_warn_cpu_only_on_nvidia_host(
            requested_device,
            device,
            cfg!(feature = "cuda"),
            nvidia_host_present(),
        )
    {
        warn_cpu_only_on_nvidia_host();
    }

    let wired_limit_bytes = if device.uses_gpu() {
        resolve_wired_limit()
    } else {
        None
    };

    // Issue #55: apply optional soft allocator cap regardless of device.
    // The MLX no-gpu CPU allocator also honours `set_memory_limit()`, so
    // the preflight (#56) can use this on Linux/CI just as on Apple
    // Silicon.
    let memory_limit_bytes = resolve_memory_limit();

    // Issue #627: apply optional buffer-cache bound. Meaningful mainly on
    // CUDA, where the periodic decode-loop clear is disabled by default and
    // this cap is the intended mechanism for bounding cache growth instead.
    let cache_limit_bytes = resolve_cache_limit();

    RuntimeSetup {
        device,
        cpu_override,
        wired_limit_bytes,
        memory_limit_bytes,
        cache_limit_bytes,
        invalid_device_override,
    }
}

fn resolve_runtime_device(value: Option<&str>) -> (RuntimeDevice, Option<String>) {
    match value {
        Some(raw) => match parse_runtime_device(raw) {
            Some(device) => (device, None),
            None => (RuntimeDevice::Gpu, Some(raw.to_owned())),
        },
        None => (RuntimeDevice::Gpu, None),
    }
}

/// Resolve the runtime device from the operator's request and the backend's
/// availability answer, returning `(device, cpu_override)`.
///
/// The two inputs are independent facts and stay separate in the result: a
/// CPU request yields `(Cpu, true)` whether or not a GPU backend exists, and
/// a GPU request yields `Gpu` only when a backend exists, `(Cpu, false)`
/// otherwise. Pure, so all four combinations are unit-tested without an
/// environment or a device.
fn resolve_device(requested: RuntimeDevice, gpu_backend_available: bool) -> (RuntimeDevice, bool) {
    let cpu_override = requested == RuntimeDevice::Cpu;
    let device = if !cpu_override && gpu_backend_available {
        RuntimeDevice::Gpu
    } else {
        RuntimeDevice::Cpu
    };
    (device, cpu_override)
}

/// Whether `MLXCEL_DEVICE` currently requests the CPU. Test support reads
/// this to tell an operator's CPU request, which `initialize_runtime` honors
/// by moving the default device on purpose, from a default device that a
/// test moved and never restored (`mlx_test_guard`, issue #1421).
#[cfg(test)]
pub(crate) fn cpu_override_requested() -> bool {
    resolve_runtime_device(std::env::var(RUNTIME_DEVICE_ENV).ok().as_deref()).0
        == RuntimeDevice::Cpu
}

/// Resolve wired memory limit from MLXCEL_WIRED_LIMIT env var.
///
/// Default: set to gpu_max_memory_size (matches Python mlx-lm's wired_limit context manager).
/// This is critical for large models (>50% of GPU memory) to avoid weight eviction.
///
/// - Not set or "max": set to gpu_max_memory_size (default, matches Python mlx-lm)
/// - "0" or "none": disable wired limit
/// - Any size accepted by [`parse_memory_size`]: explicit limit
fn resolve_wired_limit() -> Option<usize> {
    let raw = std::env::var(WIRED_LIMIT_ENV).ok();
    let limit = match raw.as_deref() {
        Some("0") | Some("none") | Some("NONE") => return None,
        None | Some("") | Some("max") | Some("MAX") => mlxcel_core::gpu_max_memory_size(),
        Some(s) => parse_memory_size(s)
            .map(clamp_to_usize)
            .unwrap_or(mlxcel_core::gpu_max_memory_size()),
    };
    if limit > 0 {
        mlxcel_core::set_wired_limit(limit);
        Some(limit)
    } else {
        None
    }
}

/// Resolve the MLX allocator soft limit from MLXCEL_MEMORY_LIMIT (issue #55).
///
/// Returns the limit actually applied to MLX, or `None` when the env var
/// is unset / explicitly disabled. This is the hook the capstone preflight
/// (#56) drives when a model is too large to fit comfortably — calling
/// `mlxcel_core::memory::set_memory_limit` makes MLX raise an exception
/// during evaluation instead of thrashing the system allocator.
fn resolve_memory_limit() -> Option<usize> {
    let raw = std::env::var(MEMORY_LIMIT_ENV).ok();
    let bytes = match raw.as_deref() {
        Some("0") | Some("none") | Some("NONE") | None | Some("") => return None,
        Some(s) => parse_memory_size(s)?,
    };
    if bytes == 0 {
        return None;
    }
    mlxcel_core::memory::set_memory_limit(bytes);
    Some(clamp_to_usize(bytes))
}

/// Resolve the MLX buffer-cache bound from MLXCEL_CACHE_LIMIT (issue #627).
///
/// Returns the limit applied, or `None` when unset/disabled. On CUDA this is
/// the intended replacement for the periodic decode-loop `clear_memory_cache`
/// (disabled by default there): it keeps the memory pool bounded without the
/// per-step churn that defeats CUDA-graph reuse (ml-explore/mlx#2358).
fn resolve_cache_limit() -> Option<usize> {
    let raw = std::env::var(CACHE_LIMIT_ENV).ok();
    let bytes = match raw.as_deref() {
        Some("0") | Some("none") | Some("NONE") | None | Some("") => return None,
        Some(s) => parse_memory_size(s)?,
    };
    if bytes == 0 {
        return None;
    }
    mlxcel_core::memory::set_cache_limit(bytes);
    Some(clamp_to_usize(bytes))
}

/// The one size grammar behind every size-valued mlxcel environment variable
/// (issue #1317).
///
/// Accepted input is trimmed and case-insensitive. A `K`/`KB`, `M`/`MB` or
/// `G`/`GB` suffix scales the numeric part by the matching power of 1024 and
/// may carry a fraction (`1.5GB`); no suffix means plain bytes. A scaled
/// result is floored and saturates at `u64::MAX`, and a negative, `NaN` or
/// infinite numeric part is rejected. `"0"` parses to `Some(0)`: mapping that
/// to "unset" belongs to the caller, next to its own `none` and empty-string
/// handling.
///
/// Multiplying by a power of two is exact in binary floating point, so a
/// fractional value loses nothing before the floor: `4.1GB` is 4402341478
/// bytes here and in any other caller.
///
/// This is `pub(crate)` because the memory-estimation preflight in
/// [`crate::execution::memory_estimate`] reads the same variables before
/// runtime bring-up and must reach the same number. It used to carry its own
/// parser that accepted only `GB` and `MB`, so `MLXCEL_MEMORY_LIMIT=4G` capped
/// the allocator while the preflight silently ignored it and reported the
/// machine's total memory instead.
pub(crate) fn parse_memory_size(s: &str) -> Option<u64> {
    let s = s.trim().to_ascii_uppercase();
    let (number, scale) = if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix('G')) {
        (n, 1024.0 * 1024.0 * 1024.0)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix('M')) {
        (n, 1024.0 * 1024.0)
    } else if let Some(n) = s.strip_suffix("KB").or_else(|| s.strip_suffix('K')) {
        (n, 1024.0)
    } else {
        // No suffix: plain bytes, and deliberately integer-only. A bare
        // `1.5` was never a byte count and stays rejected.
        return s.parse::<u64>().ok();
    };

    let value = number.trim().parse::<f64>().ok()?;
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    // `value >= 0` and `scale > 0`, so the product is either a finite
    // non-negative number or `+inf`; both compare correctly against the
    // saturation bound.
    let bytes = (value * scale).floor();
    if bytes >= u64::MAX as f64 {
        return Some(u64::MAX);
    }
    Some(bytes as u64)
}

/// Narrow a parsed byte count to the `usize` the MLX C++ setters and
/// [`RuntimeSetup`] use. Lossless on every 64-bit target mlxcel builds for;
/// the saturation is there so a 32-bit build clamps instead of wrapping.
fn clamp_to_usize(bytes: u64) -> usize {
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

fn parse_runtime_device(value: &str) -> Option<RuntimeDevice> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(RuntimeDevice::Cpu),
        "gpu" | "metal" => Some(RuntimeDevice::Gpu),
        _ => None,
    }
}

/// Detect an NVIDIA GPU without linking CUDA. The kernel driver exposes these
/// paths whether or not this binary was built with the `cuda` feature, so a
/// CPU-only build can still tell it is sitting on an NVIDIA host.
fn nvidia_host_present() -> bool {
    std::path::Path::new("/dev/nvidiactl").exists()
        || std::path::Path::new("/proc/driver/nvidia/version").exists()
}

/// Whether to warn that a CPU-only build is wasting an NVIDIA GPU. True only
/// when the GPU was wanted, the runtime fell back to CPU, this binary lacks the
/// `cuda` feature (so it can never use the GPU), and an NVIDIA host is present.
/// An explicit `MLXCEL_DEVICE=cpu` (`requested == Cpu`) suppresses the warning,
/// and a `cuda`-capable build that fell back to CPU is a genuine no-GPU host,
/// not the footgun.
fn should_warn_cpu_only_on_nvidia_host(
    requested: RuntimeDevice,
    resolved: RuntimeDevice,
    cuda_build: bool,
    nvidia_host: bool,
) -> bool {
    requested == RuntimeDevice::Gpu && resolved == RuntimeDevice::Cpu && !cuda_build && nvidia_host
}

/// Loud one-time startup warning for the CPU-only-build-on-NVIDIA-host footgun.
fn warn_cpu_only_on_nvidia_host() {
    eprintln!(
        "warning: an NVIDIA GPU is present but this mlxcel binary was built \
         without CUDA support, so it is running on the CPU (orders of magnitude \
         slower).\n         \
         Rebuild with the `cuda` feature: \
         `MLX_CUDA_ARCHITECTURES=<arch> cargo build --release --features cuda` \
         (or `cargo cuda`).\n         \
         See docs/installation.md (Linux with CUDA)."
    );
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
