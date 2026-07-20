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

// Build script for mlx-cxx
// This builds the MLX C++ library and the cxx bridge

use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Expose the pinned MLX commit to the crate so the runtime can scope the
    // persistent CUDA PTX cache directory by it (see ensure_persistent_ptx_cache).
    println!("cargo:rustc-env=MLXCEL_MLX_COMMIT={MLX_EXPECTED_COMMIT}");

    // Build MLX using cmake
    let mlx_dst = build_mlx();
    mark_mlx_cache_valid(&out_dir);
    let mlx_include = mlx_dst.join("build/include");
    let mlx_lib = mlx_dst.join("build/lib");

    // Build the cxx bridge with optimization flags
    let mut bridge = cxx_build::bridge("src/lib.rs");
    bridge
        .file("cpp/mlx_cxx_bridge.cpp")
        // Bridge implementation split out of mlx_cxx_bridge.cpp by domain
        // (shared helpers in cpp/mlx_cxx_internal.h): fused decode Metal
        // kernels, the NemotronH full-forward path, and safetensors loading +
        // Metal 4 / turbo / paged attention launchers.
        .file("cpp/mlx_cxx_kernels.cpp")
        .file("cpp/mlx_cxx_nemotron.cpp")
        .file("cpp/mlx_cxx_ext.cpp")
        // Fused Sparse-V SDPA kernel launcher. Lives under
        // `src/lib/mlx-cpp/turbo/` so the MLX-upstream-commit upgrade
        // checklist (CLAUDE.md) treats this directory as in-scope.
        .file("../mlx-cpp/turbo/sparse_v_sdpa.cpp")
        // Fused Turbo4Delegated cold-V weighted-sum kernel
        // launcher. Reads the packed cold V directly so the dequantised
        // FP16 cold body never materialises in global memory; the host
        // pairs this with a hot-V matmul to produce the final SDPA output.
        .file("../mlx-cpp/turbo/turbo4_delegated_sdpa.cpp")
        // Fused paged-attention decode kernel launcher (epic #116 Phase 6,
        // #123). Reads scattered KV blocks out of the global pool via a block
        // table with no separate gather copy; the gather-then-SDPA path stays
        // the correctness reference and fallback.
        .file("../mlx-cpp/turbo/paged_attention.cpp")
        .include(&mlx_include)
        .include("cpp")
        .include("../mlx-cpp/turbo")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        // MLX v0.31.0 triggers a Clang deprecated-copy warning from bf16.h when
        // included by the generated cxx bridge, so suppress it for all profiles.
        .flag_if_supported("-Wno-deprecated-copy");

    // Add optimization flags for release builds
    #[cfg(not(debug_assertions))]
    {
        bridge
            .flag_if_supported("-O3")
            .flag_if_supported("-DNDEBUG")
            .flag_if_supported("-ffast-math");
        // ISA baseline for the bridge C++. Defaults to the build host's ISA
        // (-march=native), which is correct for builds that run where they
        // are built (developer machines, the per-machine gb10/gh200 release
        // assets). Redistributable release builds must override this with a
        // portable baseline via MLXCEL_CXX_MARCH (e.g. "x86-64-v3" for the
        // generic Linux x86-64 asset), otherwise the binary inherits the
        // build runner's ISA (possibly AVX-512) and SIGILLs on older CPUs.
        // Set MLXCEL_CXX_MARCH=none to omit the flag entirely.
        match env::var("MLXCEL_CXX_MARCH").as_deref() {
            Err(_) => {
                bridge.flag_if_supported("-march=native");
            }
            Ok("none") => {}
            Ok(march) => {
                bridge.flag_if_supported(format!("-march={march}"));
            }
        }
        // On macOS, Clang produces LLVM bitcode with -flto, which is compatible
        // with Rust's LLVM LTO. On Linux with GCC, -flto produces GIMPLE IR
        // objects that are incompatible, causing undefined-reference linker errors.
        #[cfg(target_os = "macos")]
        bridge.flag_if_supported("-flto");
    }

    bridge.compile("mlx_cxx_bridge");

    // Link against MLX
    println!("cargo:rustc-link-search=native={}", mlx_lib.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // Platform-specific system libraries
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=objc");
        println!("cargo:rustc-link-lib=framework=Foundation");

        // Link clang compiler-rt for ___isPlatformVersionAtLeast
        // (required by MLX C++ @available() runtime checks)
        if let Ok(output) = std::process::Command::new("clang")
            .arg("--print-runtime-dir")
            .output()
        {
            let runtime_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !runtime_dir.is_empty() {
                println!("cargo:rustc-link-search=native={runtime_dir}");
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
        // CPU backend needs BLAS/LAPACK on Linux
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=lapack");
    }

    // Backend-specific linking
    #[cfg(target_os = "macos")]
    {
        // Metal and Accelerate are always linked on macOS
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(feature = "cuda")]
    {
        link_cuda();
    }

    // Rerun if bridge files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.h");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_internal.h");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_kernels.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_nemotron.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_ext.cpp");
    println!("cargo:rerun-if-changed=metal/fused_attention_metal4.metal");
    println!("cargo:rerun-if-changed=../mlx-cpp/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../mlx-cpp/patches");
    println!("cargo:rerun-if-changed=../mlx-cpp/patches-cuda");
    // Sparse-V fused-skip Metal kernel launchers.
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sparse_v_sdpa.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sparse_v_sdpa.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sparse_v_sdpa.metal");
    // Turbo4Delegated cold-V fused weighted-sum kernel launcher.
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/turbo4_delegated_sdpa.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/turbo4_delegated_sdpa.cpp");
    // Fused paged-attention decode kernel launcher (#123).
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention.metal");
    println!("cargo:rerun-if-env-changed=MLX_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_METAL");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_ACCELERATE");
    println!("cargo:rerun-if-env-changed=MLXCEL_CXX_MARCH");
}

/// Expected MLX git commit — must match GIT_TAG in mlx-cpp/CMakeLists.txt.
const MLX_EXPECTED_COMMIT: &str = "b7c3dd6d27f45b5365b08a840310187dc503f1db";

/// Purge stale cached MLX build artifacts before CMake runs.
///
/// CI caches may restore `_deps/` from a previous build. Even when the git
/// source checkout is correct, stale CMake build artifacts (object files in
/// `_deps/mlx-build/`) can cause compilation to succeed using outdated `.o`
/// files because make skips recompilation when timestamps look current.
///
/// Instead of fragile git-based validation, we use a simple marker file:
/// after a successful build, `_deps/.mlx-build-commit` records the commit.
/// If the marker is missing or doesn't match, we purge the entire `_deps/`.
fn purge_stale_mlx_cache(out_dir: &std::path::Path) {
    let deps_dir = out_dir.join("build/_deps");
    if !deps_dir.exists() {
        return;
    }

    let marker = deps_dir.join(".mlx-build-commit");
    let cached_commit = std::fs::read_to_string(&marker)
        .ok()
        .map(|s| s.trim().to_string());

    if cached_commit.as_deref() == Some(MLX_EXPECTED_COMMIT) {
        return; // Cache is valid
    }

    eprintln!(
        "mlxcel-core: MLX build cache stale (cached={}, expected={}), purging _deps/",
        cached_commit.as_deref().unwrap_or("none"),
        MLX_EXPECTED_COMMIT
    );
    let _ = std::fs::remove_dir_all(&deps_dir);
}

/// Write a marker after successful MLX build so future runs can validate the cache.
fn mark_mlx_cache_valid(out_dir: &std::path::Path) {
    let marker = out_dir.join("build/_deps/.mlx-build-commit");
    let _ = std::fs::write(marker, MLX_EXPECTED_COMMIT);
}

fn build_mlx() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    purge_stale_mlx_cache(&out_dir);

    let mut config = Config::new("../mlx-cpp");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Platform features
    // On macOS: Metal and Accelerate are always available and enabled by default.
    // Feature flags can still override (e.g. for CPU-only testing).
    // On Linux: CPU-only by default, CUDA opt-in via feature flag.
    config.define("MLX_BUILD_CUDA", "OFF");

    #[cfg(target_os = "macos")]
    {
        let build_metal = cmake_bool_from_env("MLXCEL_BUILD_METAL").unwrap_or("ON");
        let build_accelerate = cmake_bool_from_env("MLXCEL_BUILD_ACCELERATE").unwrap_or("ON");

        // Default to Metal + Accelerate on macOS, but allow CPU-only rebuilds
        // for environments where Metal device enumeration is unavailable.
        config.define("MLX_BUILD_METAL", build_metal);
        config.define("MLX_BUILD_ACCELERATE", build_accelerate);
    }

    #[cfg(not(target_os = "macos"))]
    {
        config.define("MLX_BUILD_METAL", "OFF");
        config.define("MLX_BUILD_ACCELERATE", "OFF");
    }

    #[cfg(feature = "cuda")]
    {
        config.define("MLX_BUILD_CUDA", "ON");

        // Help CMake find CUDA toolkit
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            config.define("CMAKE_CUDA_COMPILER", format!("{cuda_path}/bin/nvcc"));
        } else if PathBuf::from("/usr/local/cuda/bin/nvcc").exists() {
            config.define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc");
        }

        // CUDA architecture selection. An explicitly set MLX_CUDA_ARCHITECTURES is
        // honored verbatim (escape hatch); otherwise we auto-detect via nvidia-smi
        // and fall back to Hopper's sm_90a.
        //
        // The `a` suffix is load-bearing: MLX only defines MLX_CUDA_SM90A_ENABLED
        // (which compiles the dedicated Hopper `qmm_sm90` quantized kernel) when
        // "90a" is in the arch list. MLX's own CMake appends that suffix for
        // cc >= 90, but only inside its `if(NOT DEFINED MLX_CUDA_ARCHITECTURES)`
        // branch. Because we always pass MLX_CUDA_ARCHITECTURES explicitly, that
        // branch never runs, so we apply the same rule ourselves here and in
        // detect_cuda_arch. See docs/installation.md (CUDA architecture selection).
        let cuda_arch = env::var("MLX_CUDA_ARCHITECTURES")
            .unwrap_or_else(|_| detect_cuda_arch().unwrap_or_else(|| "90a".to_string()));
        config.define("MLX_CUDA_ARCHITECTURES", &cuda_arch);
    }

    config.build()
}

// Used by the macOS configuration block above; dead on non-macOS targets.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn cmake_bool_from_env(name: &str) -> Option<&'static str> {
    let value = env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some("ON"),
        "0" | "off" | "false" | "no" => Some("OFF"),
        _ => panic!(
            "Invalid {name} value {:?}. Expected one of: 1/0, on/off, true/false, yes/no.",
            value
        ),
    }
}

#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> Option<String> {
    use std::process::Command;
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let caps = String::from_utf8_lossy(&output.stdout);
    // Parse "X.Y" compute capabilities, convert to SM number (e.g. "9.0" -> "90"),
    // and append the architecture-specific "a" suffix for cc >= 90 (e.g. "90a").
    let archs: Vec<String> = caps
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let parts: Vec<&str> = line.split('.').collect();
            if parts.len() == 2 {
                Some(sm_arch_with_suffix(&format!("{}{}", parts[0], parts[1])))
            } else {
                None
            }
        })
        .collect();
    if archs.is_empty() {
        None
    } else {
        // Deduplicate
        let mut unique = archs;
        unique.sort();
        unique.dedup();
        Some(unique.join(";"))
    }
}

/// Append CUDA's architecture-specific `a` suffix for SM >= 90, mirroring MLX's
/// own CMake logic (`MLX_CUDA_ARCHITECTURES GREATER_EQUAL 90` -> append `a`).
///
/// The `a` suffix enables architecture-specific features (e.g. Hopper wgmma/TMA)
/// the dedicated quantized kernels rely on, and it gates MLX_CUDA_SM90A_ENABLED on
/// "90a" (not "90"). SM < 90 (e.g. Ampere sm_80/sm_86) has no `a` variant and is
/// returned unchanged.
#[cfg(feature = "cuda")]
fn sm_arch_with_suffix(sm: &str) -> String {
    match sm.parse::<u32>() {
        Ok(n) if n >= 90 => format!("{sm}a"),
        _ => sm.to_string(),
    }
}

#[cfg(feature = "cuda")]
fn link_cuda() {
    // Find CUDA lib directory
    let cuda_lib = if let Ok(cuda_home) = env::var("CUDA_HOME") {
        PathBuf::from(cuda_home).join("lib64")
    } else if PathBuf::from("/usr/local/cuda/lib64").exists() {
        PathBuf::from("/usr/local/cuda/lib64")
    } else {
        panic!("Cannot find CUDA library directory. Set CUDA_HOME environment variable.");
    };

    println!("cargo:rustc-link-search=native={}", cuda_lib.display());

    // CUDA runtime and math libraries
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cufft");

    // CUDA driver API (cuLaunchKernel, cuModuleLoad, etc.)
    println!("cargo:rustc-link-lib=dylib=cuda");

    // cuDNN
    println!("cargo:rustc-link-lib=dylib=cudnn");

    // NVRTC for runtime compilation (JIT kernels)
    println!("cargo:rustc-link-lib=dylib=nvrtc");

    // CUDA stubs directory (for driver API on systems without GPU driver in lib path)
    let cuda_stubs = cuda_lib.join("stubs");
    if cuda_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_stubs.display());
    }
}
