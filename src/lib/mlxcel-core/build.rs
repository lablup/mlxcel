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

// Build script for mlx-cxx
// This builds the MLX C++ library and the cxx bridge

use cmake::Config;
use std::{env, path::PathBuf};

// Single-source-of-truth resolution and verification of the pinned MLX commit.
// Shared by path (not by dependency) with `mlxcel-mlx-pin`, which unit-tests it
// without dragging in an MLX build; see that crate's manifest for the reason.
#[path = "build_support/mlx_pin.rs"]
mod mlx_pin;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // The pinned MLX commit has exactly one home: GIT_TAG in
    // ../mlx-cpp/CMakeLists.txt, which is what CMake actually fetches. Resolve
    // it once, here, before anything compares against it. Everything below
    // (the _deps/ purge decision, the MLXCEL_MLX_COMMIT export, the post-build
    // HEAD verification and the cache marker) consumes this one value, so they
    // cannot disagree with each other or with the tag that was fetched (#1047).
    let mlx_commit = match mlx_pin::read_pinned_commit(&manifest_dir) {
        Ok(commit) => commit,
        Err(err) => panic!("mlxcel-core: {err}"),
    };

    // Expose the pinned MLX commit to the crate so the runtime can scope the
    // persistent CUDA PTX cache directory by it (see ensure_persistent_ptx_cache).
    println!("cargo:rustc-env=MLXCEL_MLX_COMMIT={mlx_commit}");

    // Resolve the CUDA architecture list once, before CMake runs, and record it
    // in the crate so the binary knows what it was compiled for and not only
    // what it is running on (#1537). Empty off CUDA. `hardware.rs` compares this
    // against the running device's compute capability at device init, which is
    // what turns "released x86_64 archive silently fails to load on a V100" into
    // a named error naming both the device and this list.
    let cuda_arch = resolve_cuda_architectures();
    println!("cargo:rustc-env=MLXCEL_CUDA_ARCHITECTURES={cuda_arch}");

    // The ROCm counterpart (#1802): the HIP `gfx` targets this build compiles
    // MLX device code for, recorded the same way so the runtime can name both
    // the device and this list when they disagree. Empty off ROCm.
    reject_conflicting_gpu_features();
    let rocm_arch = resolve_rocm_architectures();
    println!("cargo:rustc-env=MLXCEL_ROCM_ARCHITECTURES={rocm_arch}");

    // Build MLX using cmake
    let mlx_dst = build_mlx(&mlx_commit, &cuda_arch, &rocm_arch);
    // Verify what actually landed on disk before blessing it. CMake reuses an
    // already-populated _deps/mlx-src rather than re-running FetchContent, so a
    // checkout restored from a CI cache or seeded by hand can disagree with the
    // pin without anything upstream of here noticing.
    verify_fetched_mlx_head(&out_dir, &mlx_commit);
    mark_mlx_cache_valid(&out_dir, &mlx_commit);
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
        // checklist ("Bumping the MLX upstream pin" in CONTRIBUTING.md)
        // treats this directory as in-scope.
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
        // Paged-attention decode v2 (#898): CSR page table plus cross-CTA
        // split-KV in `paged_attention_v2.cpp`, and the variable-length
        // attention-state merge kernel in `paged_attention_v2_merge.cpp`. The
        // merge half is split out because the cascade issue #903 reuses it
        // unchanged. v1 above stays intact and remains the default path.
        .file("../mlx-cpp/turbo/paged_attention_v2.cpp")
        .file("../mlx-cpp/turbo/paged_attention_v2_merge.cpp")
        // Softmax-free Gumbel-max categorical sampling kernel launcher (#900).
        // Replaces the `random::categorical` normalization pass plus the
        // `argpartition` sort on the no-filter sampling path with one
        // index-carrying max reduction over the vocabulary.
        .file("../mlx-cpp/turbo/sampling.cpp")
        // Sorting-free top-k / top-p / min-p sampling by dual-pivot rejection
        // (#901). Replaces the `argpartition` + `argsort` + `cumsum` filter
        // chain with a shrinking probability interval resolved by two pivots
        // per vocabulary sweep.
        .file("../mlx-cpp/turbo/sampling_rejection.cpp")
        // Fused residual-add + RMSNorm and fused q/k RoPE + KV-append-layout
        // decode kernel launchers (#905). Both are two/three-output custom
        // kernels: MLX arrays are immutable from the graph's perspective, so
        // the in-place residual update and the append payload are returned as
        // values rather than written through.
        .file("../mlx-cpp/turbo/fused_norm.cpp")
        .file("../mlx-cpp/turbo/fused_rope_append.cpp")
        // C shim over the `qmm_naive` CTA tile selector (#1541), so
        // `qmm_naive_tile_tests.rs` can sweep the shipped selection function on
        // the host. The selector is pure integer arithmetic with no CUDA in it,
        // and the sm_80+ non-regression claim it carries has to be checkable on
        // a machine with no NVIDIA hardware, so this is not gated on `cuda`.
        .file("cpp/qmm_naive_tile_probe.cpp")
        // C shim over the grouped GEMM CUTLASS architecture tag selection
        // (#1544), so `grouped_gemm_arch_tests.rs` can enumerate the shipped
        // decision on the host. Same reasoning as the selector above: the
        // decision is a pure function of the compute capability major version,
        // and its "sm_80 and later are untouched" claim has to be checkable on
        // a machine with no Ampere-or-later part, so this is not gated on
        // `cuda`.
        .file("cpp/grouped_gemm_arch_probe.cpp")
        .include(&mlx_include)
        .include("cpp")
        .include("../mlx-cpp/turbo")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        // MLX v0.31.0 triggers a Clang deprecated-copy warning from bf16.h when
        // included by the generated cxx bridge, so suppress it for all profiles.
        .flag_if_supported("-Wno-deprecated-copy");

    // The qmv_wide off-switch (issue #1187) lives in the
    // mlx/backend/metal/quantized.cpp overlay, so its symbol exists only when
    // the Metal backend is actually compiled. `__APPLE__` is not that
    // condition: a macOS build without the `metal` feature sets
    // MLX_BUILD_METAL=OFF and would link against a symbol that was never
    // emitted. Gate on the feature that decides whether the file is built.
    if std::env::var("CARGO_FEATURE_METAL").is_ok() {
        bridge.define("MLXCEL_BRIDGE_METAL_BACKEND", None);
    }

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

    #[cfg(feature = "rocm")]
    {
        link_rocm(&out_dir);
    }

    // Rerun if bridge files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.h");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_internal.h");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_kernels.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_nemotron.cpp");
    println!("cargo:rerun-if-changed=cpp/mlx_cxx_ext.cpp");
    println!("cargo:rerun-if-changed=cpp/qmm_naive_tile_probe.cpp");
    println!("cargo:rerun-if-changed=cpp/grouped_gemm_arch_probe.cpp");
    println!("cargo:rerun-if-changed=metal/fused_attention_metal4.metal");
    println!("cargo:rerun-if-changed=../mlx-cpp/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../mlx-cpp/patches");
    println!("cargo:rerun-if-changed=../mlx-cpp/patches-cuda");
    println!("cargo:rerun-if-changed=../mlx-cpp/patches-rocm");
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

    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention_v2.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention_v2.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/paged_attention_v2_merge.cpp");
    // Gumbel-max categorical sampling kernel launcher (#900).
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling_rejection.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/sampling_rejection.cpp");
    // Fused residual-add RMSNorm and fused RoPE + KV-append kernel launchers (#905).
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_norm.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_norm.cpp");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_rope_append.h");
    println!("cargo:rerun-if-changed=../mlx-cpp/turbo/fused_rope_append.cpp");
    println!("cargo:rerun-if-env-changed=MLX_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=MLX_ROCM_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_METAL");
    println!("cargo:rerun-if-env-changed=MLXCEL_BUILD_ACCELERATE");
    println!("cargo:rerun-if-env-changed=MLXCEL_CXX_MARCH");
}

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
///
/// `expected_commit` is the value resolved from `../mlx-cpp/CMakeLists.txt` in
/// `main`, so this decision is made against the tag CMake would actually fetch.
fn purge_stale_mlx_cache(out_dir: &std::path::Path, expected_commit: &str) {
    let deps_dir = out_dir.join("build/_deps");
    if !deps_dir.exists() {
        return;
    }

    let marker = deps_dir.join(".mlx-build-commit");
    let marker_contents = std::fs::read_to_string(&marker).ok();

    match mlx_pin::cache_state(marker_contents.as_deref(), expected_commit) {
        mlx_pin::CacheState::Valid => {}
        mlx_pin::CacheState::Stale { cached } => {
            eprintln!(
                "mlxcel-core: MLX build cache stale (cached={}, expected={expected_commit}), purging _deps/",
                cached.as_deref().unwrap_or("none")
            );
            let _ = std::fs::remove_dir_all(&deps_dir);
        }
    }
}

/// Fail the build when the MLX checkout CMake used is not the pinned commit.
///
/// Runs after the CMake build returns and before the cache marker is written,
/// so a tree that fails verification is never blessed as valid. A source tree
/// with no git metadata, or a host without `git`, warns and skips: vendored and
/// offline checkouts are legitimate and prove nothing either way. Only a
/// readable HEAD that disagrees is an error.
fn verify_fetched_mlx_head(out_dir: &std::path::Path, expected_commit: &str) {
    let mlx_src = out_dir.join("build/_deps/mlx-src");
    match mlx_pin::check_fetched_head(&mlx_src, expected_commit) {
        mlx_pin::HeadCheck::Match => {}
        mlx_pin::HeadCheck::Unavailable { reason } => {
            println!(
                "cargo:warning=mlxcel-core: skipped the fetched-MLX-commit check against \
                 {expected_commit} ({reason})"
            );
        }
        mlx_pin::HeadCheck::Mismatch { found } => {
            panic!(
                "mlxcel-core: {}",
                mlx_pin::head_mismatch_message(&mlx_src, expected_commit, &found)
            );
        }
    }
}

/// Write a marker after successful MLX build so future runs can validate the cache.
fn mark_mlx_cache_valid(out_dir: &std::path::Path, expected_commit: &str) {
    let marker = out_dir.join("build/_deps/.mlx-build-commit");
    let _ = std::fs::write(marker, expected_commit);
}

// Each architecture list is consumed only by its own backend's branch below,
// and at most one of the two backends is ever enabled.
#[allow(unused_variables)]
fn build_mlx(expected_commit: &str, cuda_architectures: &str, rocm_architectures: &str) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    purge_stale_mlx_cache(&out_dir, expected_commit);

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
        //
        // The value is resolved in `main` by `resolve_cuda_architectures` and
        // passed in, so the list CMake compiles for and the list recorded in
        // `MLXCEL_CUDA_ARCHITECTURES` are the same string by construction.
        config.define("MLX_CUDA_ARCHITECTURES", cuda_architectures);
    }

    #[cfg(feature = "rocm")]
    {
        // Turns on the ROCm overlay in ../mlx-cpp/CMakeLists.txt and the
        // backend's own CMake in mlx/backend/rocm/. The backend passes one
        // `--offload-arch` per entry of CMAKE_HIP_ARCHITECTURES to hipcc.
        let rocm = rocm_path();
        config.define("MLX_BUILD_ROCM", "ON");
        config.define("CMAKE_HIP_ARCHITECTURES", rocm_architectures);
        config.define("MLX_ROCM_ARCHITECTURES", rocm_architectures);
        // find_package(hip|rocblas|rocthrust|rocprim|hiprand|rocwmma CONFIG)
        // resolves through the prefix path, and the backend's find_library
        // calls for hiprtc and hipblaslt read ROCM_PATH.
        config.define("CMAKE_PREFIX_PATH", rocm.display().to_string());
        config.define("ROCM_PATH", rocm.display().to_string());
        let hipcc = rocm.join("bin/hipcc");
        if hipcc.exists() {
            config.define("CMAKE_HIP_COMPILER", hipcc.display().to_string());
        }
    }

    config.build()
}

/// Refuse feature combinations that cannot produce a working binary.
///
/// The ROCm overlay replaces MLX core files that the CUDA-only patches also
/// replace, and the `metal` feature makes the bridge reference symbols that
/// only the Metal overlay emits. Failing here names the conflict instead of
/// leaving it to a CMake error or an undefined symbol at link time.
fn reject_conflicting_gpu_features() {
    if env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        panic!("mlxcel-core: the `rocm` and `cuda` features cannot be enabled together");
    }
    if env::var_os("CARGO_FEATURE_METAL").is_some() {
        panic!("mlxcel-core: the `rocm` and `metal` features cannot be enabled together");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!("mlxcel-core: the `rocm` feature is supported on Linux only");
    }
}

/// The ROCm installation root: `ROCM_PATH` if set, else `/opt/rocm`.
#[cfg(feature = "rocm")]
fn rocm_path() -> PathBuf {
    env::var_os("ROCM_PATH")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"))
}

/// The HIP `gfx` target list this build compiles MLX device code for.
///
/// An explicitly set `MLX_ROCM_ARCHITECTURES` wins verbatim (`;`-separated,
/// for example `gfx1151` or `gfx1100;gfx1151`); otherwise the GPU agents that
/// `rocminfo` reports decide. Unlike CUDA there is no safe default target, so
/// a host with neither fails the build with a message naming the variable.
///
/// Empty string on a non-ROCm build.
fn resolve_rocm_architectures() -> String {
    #[cfg(feature = "rocm")]
    {
        if let Ok(value) = env::var("MLX_ROCM_ARCHITECTURES") {
            let value = value.trim();
            if !value.is_empty() {
                return value.to_string();
            }
        }
        detect_rocm_arch().unwrap_or_else(|| {
            panic!(
                "mlxcel-core: the `rocm` feature needs the HIP targets to compile for, and \
                 rocminfo reported no GPU agent. Set MLX_ROCM_ARCHITECTURES, for example \
                 MLX_ROCM_ARCHITECTURES=gfx1151."
            )
        })
    }
    #[cfg(not(feature = "rocm"))]
    {
        String::new()
    }
}

#[cfg(feature = "rocm")]
fn detect_rocm_arch() -> Option<String> {
    use std::process::Command;
    let bundled = rocm_path().join("bin/rocminfo");
    let program = if bundled.exists() {
        bundled
    } else {
        PathBuf::from("rocminfo")
    };
    let output = Command::new(program).output().ok()?;
    // GPU agents print `Name:  gfx1151`; the ISA lines print
    // `Name:  amdgcn-amd-amdhsa--gfx1151` and CPU agents print the CPU model,
    // so only a bare `gfx...` token after `Name:` is a target.
    let text = String::from_utf8_lossy(&output.stdout);
    let mut archs: Vec<String> = text
        .lines()
        .filter_map(|line| {
            let mut tokens = line.split_whitespace();
            match (tokens.next(), tokens.next()) {
                (Some("Name:"), Some(name)) if name.starts_with("gfx") => Some(name.to_string()),
                _ => None,
            }
        })
        .collect();
    archs.sort();
    archs.dedup();
    if archs.is_empty() {
        None
    } else {
        Some(archs.join(";"))
    }
}

#[cfg(feature = "rocm")]
fn link_rocm(out_dir: &std::path::Path) {
    let rocm_lib = rocm_path().join("lib");

    // The backend compiles its .hip sources into a separate static archive
    // that MLX's CMake links PRIVATE into libmlx. Cargo links libmlx.a directly
    // and never sees that edge, so the archive is named here, after `mlx`,
    // the same way link_cuda names cuSOLVER.
    let kernels_dir = out_dir.join("build/_deps/mlx-build/mlx/backend/rocm");
    println!("cargo:rustc-link-search=native={}", kernels_dir.display());
    println!("cargo:rustc-link-lib=static=mlx_rocm_kernels");

    // The shared libraries the backend's CMake links (target_link_libraries
    // in mlx/backend/rocm/CMakeLists.txt).
    println!("cargo:rustc-link-search=native={}", rocm_lib.display());
    for lib in ["amdhip64", "rocblas", "hiprand", "hiprtc", "hipblaslt"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    // ROCm installs do not always register their library directory with the
    // dynamic loader, so embed it. This covers this crate's own test and
    // example binaries; the root build script adds the same for mlxcel's,
    // because a dependency's link args do not reach the binary that links it.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rocm_lib.display());
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

/// The CUDA architecture list this build compiles MLX device code for.
///
/// An explicitly set `MLX_CUDA_ARCHITECTURES` wins verbatim (the documented
/// escape hatch); otherwise `nvidia-smi` detection decides, and `90a` is the
/// last resort. That fallback is why the runtime mismatch check exists: on a
/// host without `nvidia-smi` it produces a binary that cannot run on its own
/// build machine, and without the check the only symptom is an opaque CUDA
/// load failure at the first kernel launch.
///
/// Empty string on a non-CUDA build, which the runtime reads as "no compiled
/// architecture list" and skips the check entirely.
fn resolve_cuda_architectures() -> String {
    #[cfg(feature = "cuda")]
    {
        env::var("MLX_CUDA_ARCHITECTURES")
            .unwrap_or_else(|_| detect_cuda_arch().unwrap_or_else(|| "90a".to_string()))
    }
    #[cfg(not(feature = "cuda"))]
    {
        String::new()
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
    // cuSOLVER, since MLX 81ba1c6a: ml-explore/mlx#4208 moved Cholesky onto it
    // and `gpu::init()` now creates its handle cache on every CUDA start, so a
    // link without it fails on `cusolverDnCreate`. MLX's own CMake links it
    // PRIVATE, which cargo never sees, so it has to be named here like the rest.
    println!("cargo:rustc-link-lib=dylib=cusolver");

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
