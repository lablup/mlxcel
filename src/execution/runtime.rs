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
/// Used by the future preflight capstone (#56). Accepts the same syntax
/// as `MLXCEL_WIRED_LIMIT`: plain bytes, `NGB`, or `NMB`. Unset means
/// "do not override MLX's default limit".
const MEMORY_LIMIT_ENV: &str = "MLXCEL_MEMORY_LIMIT";
/// Issue #627: optional bound on MLX's buffer cache. When set, the runtime
/// calls `mlxcel_core::memory::set_cache_limit(...)` at startup so the CUDA
/// memory pool stays bounded without the per-decode `clear_memory_cache`
/// churn that defeats CUDA-graph reuse (ml-explore/mlx#2358). Same syntax as
/// `MLXCEL_WIRED_LIMIT`: plain bytes, `NG`/`NGB`, or `NM`/`NMB`. Unset means
/// "do not override MLX's default cache behavior".
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
    pub device: RuntimeDevice,
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

pub fn initialize_runtime() -> RuntimeSetup {
    let (requested_device, invalid_device_override) =
        resolve_runtime_device(std::env::var(RUNTIME_DEVICE_ENV).ok().as_deref());

    if requested_device == RuntimeDevice::Cpu {
        mlxcel_core::set_default_device(false);
    }

    let device = if mlxcel_core::is_gpu_available() {
        RuntimeDevice::Gpu
    } else {
        RuntimeDevice::Cpu
    };

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

/// Resolve wired memory limit from MLXCEL_WIRED_LIMIT env var.
///
/// Default: set to gpu_max_memory_size (matches Python mlx-lm's wired_limit context manager).
/// This is critical for large models (>50% of GPU memory) to avoid weight eviction.
///
/// - Not set or "max": set to gpu_max_memory_size (default, matches Python mlx-lm)
/// - "0" or "none": disable wired limit
/// - Number (bytes) or "NGB"/"NMB": explicit limit
fn resolve_wired_limit() -> Option<usize> {
    let raw = std::env::var(WIRED_LIMIT_ENV).ok();
    let limit = match raw.as_deref() {
        Some("0") | Some("none") | Some("NONE") => return None,
        None | Some("") | Some("max") | Some("MAX") => mlxcel_core::gpu_max_memory_size(),
        Some(s) => parse_memory_size(s).unwrap_or(mlxcel_core::gpu_max_memory_size()),
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
    mlxcel_core::memory::set_memory_limit(bytes as u64);
    Some(bytes)
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
    mlxcel_core::memory::set_cache_limit(bytes as u64);
    Some(bytes)
}

/// Parse a memory size string: plain bytes, "NG"/"NGB", or "NM"/"NMB".
fn parse_memory_size(s: &str) -> Option<usize> {
    let s = s.trim().to_ascii_uppercase();
    if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix('G')) {
        n.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix('M')) {
        n.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 1024.0 * 1024.0) as usize)
    } else {
        s.parse::<usize>().ok()
    }
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
