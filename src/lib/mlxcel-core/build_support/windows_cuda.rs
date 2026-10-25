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

//! Native Windows (MSVC) layout of the CUDA toolkit and cuDNN, for the `cuda`
//! feature on `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc` (#58).
//!
//! Shared by path, like `rocm_rpath.rs`: `mlxcel-core/build.rs` uses it to
//! point CMake at the toolkit and to name the import libraries, and both that
//! script and the root `build.rs` call [`emit_delayload_link_args`], because a
//! dependency's `rustc-link-arg` does not reach the binary that links it. The
//! pure helpers are unit-tested in `mlxcel-mlx-pin`.
//!
//! Windows differs from the Linux layout `link_cuda` assumes in three ways:
//! import libraries live in `lib\<arch>` (`x64` or `arm64`) instead of `lib64`,
//! the DLLs live in `bin\<arch>`, and there is no `stubs\` directory. CMake's
//! `FindCUDAToolkit` and MLX's `FindCUDNN` only search `lib\x64`, so on arm64
//! the library directory has to be handed to them explicitly.

use std::env;
use std::path::{Path, PathBuf};

/// The `bin\` and `lib\` subdirectory a Windows CUDA toolkit and cuDNN use for
/// a Rust target architecture, or `None` for an architecture NVIDIA ships no
/// Windows CUDA for.
pub fn arch_subdir(target_arch: &str) -> Option<&'static str> {
    match target_arch {
        "x86_64" => Some("x64"),
        "aarch64" => Some("arm64"),
        _ => None,
    }
}

/// The toolkit root: `CUDA_HOME` if set, else `CUDA_PATH`, which is what the
/// Windows installer sets. Empty values count as unset.
pub fn resolve_cuda_root(cuda_home: Option<&str>, cuda_path: Option<&str>) -> Option<PathBuf> {
    [cuda_home, cuda_path]
        .into_iter()
        .flatten()
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Where the build finds each part of the toolkit for one target architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayout {
    pub nvcc: PathBuf,
    /// Import libraries (`cudart.lib`, `cublas.lib`, ...).
    pub lib_dir: PathBuf,
    /// Runtime DLLs (`cudart64_13.dll`, ...), which MLX's delay-load hook
    /// resolves at run time.
    pub dll_dir: PathBuf,
    pub cudnn_include_dir: PathBuf,
    pub cudnn_lib_dir: PathBuf,
    pub cudnn_dll_dir: PathBuf,
}

/// The layout under `cuda_root`, with cuDNN under `cudnn_root` when it is a
/// separate install (`CUDNN_PATH`) and inside the toolkit otherwise. Both use
/// the NVIDIA redistributable layout: `include\`, `lib\<arch>\`, `bin\<arch>\`.
pub fn layout(cuda_root: &Path, cudnn_root: Option<&Path>, arch_subdir: &str) -> CudaLayout {
    let cudnn_root = cudnn_root.unwrap_or(cuda_root);
    CudaLayout {
        nvcc: cuda_root.join("bin").join("nvcc.exe"),
        lib_dir: cuda_root.join("lib").join(arch_subdir),
        dll_dir: cuda_root.join("bin").join(arch_subdir),
        cudnn_include_dir: cudnn_root.join("include"),
        cudnn_lib_dir: cudnn_root.join("lib").join(arch_subdir),
        cudnn_dll_dir: cudnn_root.join("bin").join(arch_subdir),
    }
}

/// `path` spelled for CMake: forward slashes only. CMake itself accepts either
/// separator, but MLX forwards `MLX_CUDA_BIN_DIR` / `MLX_CUDNN_BIN_DIR` into C++
/// string-literal defines, where `C:\Users\me\tools\bin` would read `\t`, `\b`
/// and friends as escapes and the delay-load hook would look for the DLLs in a
/// directory that does not exist.
pub fn cmake_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}

/// The DLLs behind the import libraries `link_cuda` names, by file-name prefix.
/// `cuda.lib` is left out on purpose: it imports the driver's `nvcuda.dll`,
/// which the display driver installs in System32 and MLX does not delay-load
/// either. Everything else is resolved by MLX's hook from the toolkit and
/// cuDNN directories baked in at build time.
pub const DELAYLOAD_DLL_PREFIXES: &[&str] = &[
    "cudart64_",
    "cublas64_",
    "cublaslt64_",
    "cufft64_",
    "cusolver64_",
    "nvrtc64_",
    "cudnn64_",
];

/// The subset of `file_names` to pass to `/DELAYLOAD`, sorted and deduplicated.
///
/// Matching is case-insensitive, as Windows file names are. The rest of the
/// toolkit's DLLs are skipped because the linker warns (LNK4199) for every
/// delay-loaded name nothing imports, and `.alt.dll` side-by-side variants are
/// not what the import libraries name. At the current MLX pin `cudart64_` and
/// `cublas64_` still draw that warning (nvcc links the static cudart and MLX
/// calls only cuBLASLt); they stay listed so a later pin that imports them
/// delay-loads them too instead of failing to start without `PATH`.
pub fn select_delayload_dlls<I, S>(file_names: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut selected: Vec<String> = file_names
        .into_iter()
        .map(|name| name.as_ref().to_string())
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            lower.ends_with(".dll")
                && !lower.contains(".alt.")
                && DELAYLOAD_DLL_PREFIXES
                    .iter()
                    .any(|prefix| lower.starts_with(prefix))
        })
        .collect();
    selected.sort();
    selected.dedup();
    selected
}

/// The layout for the target being built, or `None` when the target is not
/// Windows MSVC. Panics with an actionable message when it is, but the toolkit
/// cannot be located, since the CMake configure would fail later and less
/// clearly.
pub fn layout_from_env() -> Option<CudaLayout> {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDNN_PATH");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc")
    {
        return None;
    }
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let Some(arch) = arch_subdir(&target_arch) else {
        panic!(
            "mlxcel-core: the `cuda` feature on Windows supports x86_64 and aarch64, not {target_arch}"
        );
    };
    let cuda_home = env::var("CUDA_HOME").ok();
    let cuda_path = env::var("CUDA_PATH").ok();
    let Some(root) = resolve_cuda_root(cuda_home.as_deref(), cuda_path.as_deref()) else {
        panic!(
            "mlxcel-core: the `cuda` feature needs the CUDA toolkit. Set CUDA_PATH (or \
             CUDA_HOME) to its root, the directory holding bin\\nvcc.exe and lib\\{arch}."
        );
    };
    let cudnn_root = env::var_os("CUDNN_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let layout = layout(&root, cudnn_root.as_deref(), arch);
    if !layout.nvcc.exists() {
        panic!(
            "mlxcel-core: {} does not exist. CUDA_PATH / CUDA_HOME must name the toolkit root.",
            layout.nvcc.display()
        );
    }
    if !layout.cudnn_lib_dir.join("cudnn.lib").exists() {
        panic!(
            "mlxcel-core: cuDNN was not found at {}. Install cuDNN 9 into the toolkit, or set \
             CUDNN_PATH to a cuDNN root with include\\, lib\\{arch}\\ and bin\\{arch}\\.",
            layout.cudnn_lib_dir.display()
        );
    }
    Some(layout)
}

/// Link arguments that make the final binary delay-load the CUDA and cuDNN
/// DLLs: `/DELAYLOAD` per DLL, plus `/INCLUDE` of MLX's notify hook. MLX
/// compiles its `delayload.cpp` (mlxcel's overlay of it) into `mlx.lib`, but
/// nothing references the hook, so without `/INCLUDE` the linker would never
/// pull that member and the null default in `delayimp.lib` would win. `mlx` is
/// linked ahead of `delayimp`, so the forced symbol resolves from `mlx.lib`.
///
/// MLX's CMake attaches the same options to the `mlx` target as PUBLIC link
/// options, which only CMake consumers inherit; cargo links `mlx.lib` directly.
/// Without them the binary imports the DLLs eagerly and fails to start unless
/// the toolkit's `bin\<arch>` directory is on `PATH`.
pub fn emit_delayload_link_args(layout: &CudaLayout) {
    let mut names = Vec::new();
    for dir in [&layout.dll_dir, &layout.cudnn_dll_dir] {
        if let Ok(entries) = std::fs::read_dir(dir) {
            names.extend(
                entries
                    .flatten()
                    .filter_map(|entry| entry.file_name().into_string().ok()),
            );
        }
    }
    for dll in select_delayload_dlls(names) {
        println!("cargo:rustc-link-arg=/DELAYLOAD:{dll}");
    }
    println!("cargo:rustc-link-arg=/INCLUDE:__pfnDliNotifyHook2");
}

#[cfg(test)]
#[path = "windows_cuda_tests.rs"]
mod tests;
