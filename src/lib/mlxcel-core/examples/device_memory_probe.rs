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

//! Print every figure the memory estimator and the device report read, so the
//! numbers in a bug report can be reproduced instead of retyped (issue #1805).
//!
//! This started as a throwaway probe used to measure the UMA carve-out host for
//! #1805, whose issue body quotes its output. Keeping it in the tree is the
//! difference between a measurement anyone can re-run on their own host and a
//! table nobody can check.
//!
//! Reads only. It applies no limit, allocates nothing on the device, and
//! touches no environment variable, so running it cannot change what a
//! subsequent `mlxcel` run does.
//!
//! ```bash
//! cargo run --release --features rocm --example device_memory_probe
//! ```
//!
//! What each line means:
//!
//! - `memory_limit()` is MLX's allocator soft cap, step 2 of
//!   `resolve_available_memory`. Nonzero on ROCm, which is why that walk stops
//!   there and never reaches `/proc/meminfo` on an AMD host.
//! - `gpu_max_memory_size()` prefers Metal's recommended working-set size and
//!   falls back to `total_memory`, so on ROCm it reports the full carve-out.
//! - `get_wired_limit()` reads `max_recommended_working_set_size` only, so it
//!   is 0 on every backend but Metal. That is the same no-op it is on CUDA,
//!   not a ROCm defect.
//! - `unified_memory_gb` is Apple-only by construction; read
//!   `device_memory_bytes` for device memory off Apple.

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn main() {
    let backend = mlxcel_core::hardware::gpu_backend_kind();
    println!("resolved GPU backend:        {backend:?}");
    println!(
        "gpu_device_count():          {}",
        mlxcel_core::gpu_device_count()
    );

    let hw = mlxcel_core::hardware::get_hardware();
    println!();
    println!("── HardwareCapabilities ──");
    println!("vendor:                      {:?}", hw.vendor);
    println!("device_name:                 {:?}", hw.device_name);
    println!(
        "device_architecture:         {:?}",
        hw.device_architecture.as_deref().unwrap_or("<none>")
    );
    println!(
        "device_memory_bytes:         {} ({})",
        hw.device_memory_bytes,
        gib(hw.device_memory_bytes)
    );
    println!("silicon_gen:                 {}", hw.silicon_gen);
    println!("unified_memory_gb:           {}", hw.unified_memory_gb);
    println!("gpu_core_count:              {}", hw.gpu_core_count);
    println!("memory_bandwidth_gbps:       {}", hw.memory_bandwidth_gbps);
    println!("is_apple_silicon():          {}", hw.is_apple_silicon());

    println!();
    println!("── MLX memory surface ──");
    let limit = mlxcel_core::memory::memory_limit();
    println!("memory_limit():              {limit} ({})", gib(limit));
    let max_size = mlxcel_core::gpu_max_memory_size() as u64;
    println!(
        "gpu_max_memory_size():       {max_size} ({})",
        gib(max_size)
    );
    let wired = mlxcel_core::get_wired_limit() as u64;
    println!("get_wired_limit():           {wired} ({})", gib(wired));

    println!();
    println!("── Compiled architectures ──");
    println!(
        "CUDA compute capability:     {:?}",
        mlxcel_core::hardware::cuda_compute_capability()
    );
    println!(
        "MLX_CUDA_ARCHITECTURES:      {:?}",
        mlxcel_core::hardware::compiled_cuda_architectures()
    );
    println!(
        "HIP gfx target:              {:?}",
        mlxcel_core::rocm_arch::device_gfx_target()
    );
    println!(
        "MLX_ROCM_ARCHITECTURES:      {:?}",
        mlxcel_core::rocm_arch::compiled_rocm_architectures()
    );
    match mlxcel_core::rocm_arch::rocm_arch_mismatch() {
        Some(mismatch) => println!("gfx coverage:                MISMATCH: {mismatch}"),
        None => println!("gfx coverage:                nothing to report"),
    }
}
