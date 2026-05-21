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

    RuntimeSetup {
        device,
        wired_limit_bytes,
        memory_limit_bytes,
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

/// Parse a memory size string: plain bytes, "NGB", or "NMB".
fn parse_memory_size(s: &str) -> Option<usize> {
    let s = s.trim().to_ascii_uppercase();
    if let Some(n) = s.strip_suffix("GB") {
        n.trim()
            .parse::<f64>()
            .ok()
            .map(|v| (v * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(n) = s.strip_suffix("MB") {
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

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
