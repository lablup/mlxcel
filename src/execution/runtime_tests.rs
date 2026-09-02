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

use super::{
    RuntimeDevice, cpu_override_requested, parse_memory_size, parse_runtime_device, resolve_device,
    resolve_runtime_device, should_warn_cpu_only_on_nvidia_host,
};

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
        Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64)
    );
}

#[test]
fn parse_memory_size_invalid() {
    assert_eq!(parse_memory_size("abc"), None);
}

/// Issue #1317: every spelling of a suffix is one value. The preflight used to
/// carry a parser that took `GB` but not `G`, so `MLXCEL_MEMORY_LIMIT=4G`
/// capped the allocator and was silently dropped by the estimate.
#[test]
fn parse_memory_size_accepts_every_suffix_spelling() {
    let four_gib = Some(4 * 1024 * 1024 * 1024);
    assert_eq!(parse_memory_size("4G"), four_gib);
    assert_eq!(parse_memory_size("4GB"), four_gib);
    assert_eq!(parse_memory_size("4gb"), four_gib);
    assert_eq!(parse_memory_size(" 4 GB "), four_gib);

    let five_hundred_twelve_mib = Some(512 * 1024 * 1024);
    assert_eq!(parse_memory_size("512M"), five_hundred_twelve_mib);
    assert_eq!(parse_memory_size("512MB"), five_hundred_twelve_mib);
    assert_eq!(parse_memory_size("512mb"), five_hundred_twelve_mib);

    assert_eq!(parse_memory_size("8K"), Some(8192));
    assert_eq!(parse_memory_size("8KB"), Some(8192));

    assert_eq!(parse_memory_size("1024"), Some(1024));
}

/// Every scale is a power of two, so the multiply is exact in binary floating
/// point and only the floor removes anything.
#[test]
fn parse_memory_size_fractional_is_exact_floor() {
    assert_eq!(parse_memory_size("1.5GB"), Some(1_610_612_736));
    assert_eq!(parse_memory_size("4.1GB"), Some(4_402_341_478));
    assert_eq!(parse_memory_size("0.5M"), Some(524_288));
}

#[test]
fn parse_memory_size_rejects_garbage() {
    assert_eq!(parse_memory_size("-1GB"), None);
    assert_eq!(parse_memory_size("NaNGB"), None);
    assert_eq!(parse_memory_size("infGB"), None);
    assert_eq!(parse_memory_size("abc"), None);
    assert_eq!(parse_memory_size("GB"), None);
    // A bare number is a byte count, so a fraction of a byte is not a size.
    assert_eq!(parse_memory_size("1.5"), None);
    // `0` parses; each resolver is what maps it to "unset".
    assert_eq!(parse_memory_size("0"), Some(0));
}

/// A value large enough to overflow the multiply saturates instead of
/// wrapping into a small cap that would refuse every allocation.
#[test]
fn parse_memory_size_saturates_instead_of_wrapping() {
    assert_eq!(parse_memory_size("1e30GB"), Some(u64::MAX));
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

// ── Backend availability vs. the CPU override (issue #1421) ──────────────────

/// `RuntimeSetup.device` comes from the backend's availability answer and the
/// override is reported next to it, so the two reasons for `Cpu` stay apart.
#[test]
fn resolve_device_separates_backend_availability_from_the_cpu_override() {
    use RuntimeDevice::{Cpu, Gpu};
    // GPU wanted and a backend exists: the GPU, no override.
    assert_eq!(resolve_device(Gpu, true), (Gpu, false));
    // GPU wanted but no backend: the CPU is the only option, not an override.
    assert_eq!(resolve_device(Gpu, false), (Cpu, false));
    // CPU asked for: an override, whether or not a backend exists.
    assert_eq!(resolve_device(Cpu, true), (Cpu, true));
    assert_eq!(resolve_device(Cpu, false), (Cpu, true));
}

/// Run both with and without `MLXCEL_DEVICE=cpu`: the setup reports the
/// override, the resolved device follows the backend answer when nothing asked
/// for the CPU, and the backend answer is the same before and after, even
/// though the override moves the MLX default device. A guard puts the default
/// device back so the override does not leak into later tests in this process.
#[test]
fn initialize_runtime_reports_the_cpu_override_without_hiding_the_backend() {
    let _device = mlxcel_core::streams::DefaultDeviceGuard::capture();
    let backend_before = mlxcel_core::gpu_backend_available();

    let setup = super::initialize_runtime();
    let cpu_requested = cpu_override_requested();

    assert_eq!(setup.cpu_override, cpu_requested);
    if setup.device == RuntimeDevice::Cpu {
        // Every CPU resolution is applied, not just the operator's override:
        // a CUDA build on a host without a driver starts with MLX defaulting
        // to a GPU that `gpu_backend_available()` calls unusable, and
        // reporting `Cpu` while leaving MLX dispatching there would make the
        // field a claim rather than a fact (issue #1421).
        assert!(
            !mlxcel_core::default_device_is_gpu(),
            "a runtime that resolved to the CPU must leave MLX's default device on the CPU"
        );
    }
    if cpu_requested {
        assert_eq!(setup.device, RuntimeDevice::Cpu);
    } else {
        assert_eq!(
            setup.device == RuntimeDevice::Gpu,
            backend_before,
            "without an override the device is the GPU exactly when a backend exists"
        );
    }
    assert_eq!(
        mlxcel_core::gpu_backend_available(),
        backend_before,
        "the override must not change what the backend reports"
    );
}

// ── CUDA architecture refusal (issue #1537) ──────────────────────────────────

#[test]
fn checked_init_agrees_with_the_architecture_check_on_this_host() {
    // `initialize_runtime_checked` refuses exactly when the compiled CUDA
    // architectures do not cover the host GPU. On every supported build that
    // is never, and the two must not be able to drift apart: a refusal with no
    // mismatch, or a mismatch that starts anyway, are both bugs.
    let mismatch = mlxcel_core::hardware::cuda_arch_mismatch();
    assert_eq!(
        super::initialize_runtime_checked().is_err(),
        mismatch.is_some(),
        "refusal must track cuda_arch_mismatch(), which reported {mismatch:?}"
    );
}

#[test]
fn the_architecture_refusal_only_applies_to_gpu_runs() {
    // The bypass is deliberate: `MLXCEL_DEVICE=cpu` is the one workaround left
    // to someone holding an archive built for the wrong architecture, so the
    // refusal must be reachable only when the GPU was actually requested.
    assert!(!resolve_runtime_device(Some("cpu")).0.uses_gpu());
    assert!(resolve_runtime_device(None).0.uses_gpu());
    assert!(resolve_runtime_device(Some("gpu")).0.uses_gpu());
    // An unparseable override still resolves to the GPU, so it stays checked.
    assert!(resolve_runtime_device(Some("tpu")).0.uses_gpu());
}
