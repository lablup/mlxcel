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

use std::sync::{Mutex, OnceLock};

use super::{
    RuntimeDevice, parse_memory_size, parse_runtime_device, resolve_runtime_device,
    should_warn_cpu_only_on_nvidia_host,
};

/// Serialisation lock for tests that mutate `MLXCEL_CACHE_LIMIT`.
/// Env var changes are process-global and would race when run in parallel.
fn cache_limit_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Helper: run `f` with the env var set, then restore.
fn with_cache_limit<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _guard = cache_limit_lock().lock().unwrap();
    let prev = std::env::var("MLXCEL_CACHE_LIMIT").ok();
    // SAFETY: single-threaded behind a mutex; env var changes are
    // restored to the previous value before the lock is released.
    unsafe {
        if let Some(v) = value {
            std::env::set_var("MLXCEL_CACHE_LIMIT", v);
        } else {
            std::env::remove_var("MLXCEL_CACHE_LIMIT");
        }
    }
    let result = f();
    // SAFETY: same mutex guard still held.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("MLXCEL_CACHE_LIMIT", v),
            None => std::env::remove_var("MLXCEL_CACHE_LIMIT"),
        }
    }
    result
}

#[test]
fn parse_runtime_device_accepts_cpu() {
    assert_eq!(parse_runtime_device("cpu"), Some(RuntimeDevice::Cpu));
}

#[test]
fn parse_runtime_device_accepts_gpu_aliases() {
    assert_eq!(parse_runtime_device("gpu"), Some(RuntimeDevice::Gpu));
    assert_eq!(parse_runtime_device("Metal"), Some(RuntimeDevice::Gpu));
}

#[test]
fn parse_runtime_device_rejects_unknown_values() {
    assert_eq!(parse_runtime_device("tpu"), None);
}

#[test]
fn resolve_runtime_device_defaults_to_gpu() {
    assert_eq!(resolve_runtime_device(None), (RuntimeDevice::Gpu, None));
}

#[test]
fn resolve_runtime_device_preserves_invalid_override() {
    assert_eq!(
        resolve_runtime_device(Some("mps")),
        (RuntimeDevice::Gpu, Some("mps".to_string()))
    );
}

#[test]
fn parse_memory_size_gb() {
    assert_eq!(parse_memory_size("64GB"), Some(64 * 1024 * 1024 * 1024));
    assert_eq!(parse_memory_size("128gb"), Some(128 * 1024 * 1024 * 1024));
}

#[test]
fn parse_memory_size_mb() {
    assert_eq!(parse_memory_size("512MB"), Some(512 * 1024 * 1024));
}

#[test]
fn parse_memory_size_bytes() {
    assert_eq!(parse_memory_size("1073741824"), Some(1073741824));
}

#[test]
fn parse_memory_size_fractional_gb() {
    // 1.5 GB
    assert_eq!(
        parse_memory_size("1.5GB"),
        Some((1.5 * 1024.0 * 1024.0 * 1024.0) as usize)
    );
}

#[test]
fn parse_memory_size_invalid() {
    assert_eq!(parse_memory_size("abc"), None);
}

#[test]
fn warns_only_for_cpu_fallback_on_nvidia_host_without_cuda() {
    use RuntimeDevice::{Cpu, Gpu};
    // Footgun: wanted GPU, fell back to CPU, no cuda feature, NVIDIA host present.
    assert!(should_warn_cpu_only_on_nvidia_host(Gpu, Cpu, false, true));
    // cuda-capable build that fell back to CPU is a genuine no-GPU host, not the footgun.
    assert!(!should_warn_cpu_only_on_nvidia_host(Gpu, Cpu, true, true));
    // Genuine CPU-only Linux box (no NVIDIA device node): no nag.
    assert!(!should_warn_cpu_only_on_nvidia_host(Gpu, Cpu, false, false));
    // Explicit MLXCEL_DEVICE=cpu (requested == Cpu): respect the override.
    assert!(!should_warn_cpu_only_on_nvidia_host(Cpu, Cpu, false, true));
    // Already running on the GPU: nothing to warn about.
    assert!(!should_warn_cpu_only_on_nvidia_host(Gpu, Gpu, false, true));
}

// ── apply_default_cache_limit_if_unset ───────────────────────────────────────

/// Unset `MLXCEL_CACHE_LIMIT` → helper returns `true` (default applied).
#[test]
fn apply_default_cache_limit_if_unset_when_unset() {
    with_cache_limit(None, || {
        assert!(super::apply_default_cache_limit_if_unset(42));
    });
}

/// `MLXCEL_CACHE_LIMIT=1GB` → returns `false` (operator wins).
#[test]
fn apply_default_cache_limit_if_unset_operator_1gb() {
    with_cache_limit(Some("1GB"), || {
        assert!(!super::apply_default_cache_limit_if_unset(42));
    });
}

/// `MLXCEL_CACHE_LIMIT=0` → returns `false`. The operator explicitly
/// disabled the cap; the default must not override them.
#[test]
fn apply_default_cache_limit_if_unset_operator_zero() {
    with_cache_limit(Some("0"), || {
        assert!(!super::apply_default_cache_limit_if_unset(42));
    });
}

/// `MLXCEL_CACHE_LIMIT=""` → returns `true` (empty is indistinguishable
/// from unset, matching `resolve_cache_limit` semantics).
#[test]
fn apply_default_cache_limit_if_unset_empty_string() {
    with_cache_limit(Some(""), || {
        assert!(super::apply_default_cache_limit_if_unset(42));
    });
}
