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

//! Native Windows (MSVC) layout of a ROCm SDK, for the `rocm` feature on
//! `x86_64-pc-windows-msvc` (#2601).
//!
//! The Windows counterpart of the Linux `/opt/rocm` assumptions in
//! `mlxcel-core/build.rs`, shaped like `windows_cuda.rs`, whose `cmake_path`
//! it reuses. The pure helpers are unit-tested in `mlxcel-mlx-pin`.
//!
//! A ROCm 10 TheRock tarball (or the pip wheels' SDK) keeps the HIP compiler
//! driver at `bin\hipcc.exe`, `amdgpu-arch.exe` (the stand-in for `rocminfo`,
//! which Windows SDKs do not ship) in `bin\` or `lib\llvm\bin\`, the runtime
//! DLLs in `bin\`, and
//! the MSVC import libraries in `lib\`. hipBLASLt's import library is named
//! `libhipblaslt.dll.a` but is a COFF import library that link.exe reads.

use std::env;
use std::path::{Path, PathBuf};

pub use crate::windows_cuda::cmake_path;

/// The SDK root: `ROCM_PATH` if set, else `HIP_PATH`, which the HIP SDK
/// installer sets. Empty values count as unset.
pub fn resolve_rocm_root(rocm_path: Option<&str>, hip_path: Option<&str>) -> Option<PathBuf> {
    [rocm_path, hip_path]
        .into_iter()
        .flatten()
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Where the build finds each part of the SDK.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmLayout {
    pub root: PathBuf,
    /// Tools (`hipcc.exe`, `amdgpu-arch.exe`) and runtime DLLs
    /// (`amdhip64_7.dll`, `rocblas.dll`, ...).
    pub bin_dir: PathBuf,
    /// Import libraries (`amdhip64.lib`, `rocblas.lib`, `libhipblaslt.dll.a`).
    pub lib_dir: PathBuf,
    pub hipcc: PathBuf,
    pub amdgpu_arch: PathBuf,
}

/// The layout under `root`. `exists` decides between the two places SDKs keep
/// `amdgpu-arch.exe`: `bin\` (HIP SDK installs) or `lib\llvm\bin\` (the ROCm
/// 10.1 TheRock tarball). `bin\` is used when neither has it, so the
/// detection error names the conventional path.
pub fn layout(root: &Path, exists: impl Fn(&Path) -> bool) -> RocmLayout {
    let bin_dir = root.join("bin");
    let llvm_arch = root
        .join("lib")
        .join("llvm")
        .join("bin")
        .join("amdgpu-arch.exe");
    let bin_arch = bin_dir.join("amdgpu-arch.exe");
    let amdgpu_arch = if !exists(&bin_arch) && exists(&llvm_arch) {
        llvm_arch
    } else {
        bin_arch
    };
    RocmLayout {
        root: root.to_path_buf(),
        lib_dir: root.join("lib"),
        hipcc: bin_dir.join("hipcc.exe"),
        amdgpu_arch,
        bin_dir,
    }
}

/// The HIP target list from `amdgpu-arch` output: one `gfx...` per line,
/// trimmed (the tool ends lines with `\r\n` on Windows), sorted, deduplicated
/// and `;`-joined, the format `MLX_ROCM_ARCHITECTURES` takes. Lines that are
/// not a `gfx` target are ignored; `None` when no target remains.
pub fn parse_amdgpu_arch(stdout: &str) -> Option<String> {
    let mut archs: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("gfx"))
        .collect();
    archs.sort_unstable();
    archs.dedup();
    if archs.is_empty() {
        None
    } else {
        Some(archs.join(";"))
    }
}

/// The DLLs behind the import libraries `link_rocm` names, by file-name
/// prefix. `hiprand` is absent because the linker drops it (nothing calls it),
/// so the binaries never import `hiprand.dll`. The overlay's `delayload.cpp`
/// resolves each one from the SDK `bin` directory baked in at build time.
pub const DELAYLOAD_DLL_PREFIXES: &[&str] =
    &["amdhip64_", "rocblas", "libhipblaslt", "hipfft", "hiprtc"];

/// SDK DLLs that match a prefix above but that no import library names, so
/// `/DELAYLOAD` on them would only draw LNK4199: hipRTC's builtins (loaded at
/// run time by name, which the overlay hook preloads instead) and hipFFT's
/// FFTW-compatible front end.
pub const DELAYLOAD_EXCLUDED_PREFIXES: &[&str] = &["hiprtc-builtins", "hipfftw"];

/// The subset of `file_names` to pass to `/DELAYLOAD`, sorted and deduplicated,
/// with the semantics of `windows_cuda::select_delayload_dlls`: matching is
/// case-insensitive, as Windows file names are, and only `.dll` files count.
/// The original spelling is kept, since it is what link.exe matches against
/// the import names.
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
                && DELAYLOAD_DLL_PREFIXES
                    .iter()
                    .any(|prefix| lower.starts_with(prefix))
                && !DELAYLOAD_EXCLUDED_PREFIXES
                    .iter()
                    .any(|prefix| lower.starts_with(prefix))
        })
        .collect();
    selected.sort();
    selected.dedup();
    selected
}

/// Link arguments that make the final binary delay-load the ROCm DLLs from
/// `layout.bin_dir`: `/DELAYLOAD` per DLL, plus `/INCLUDE` of the overlay's
/// notify hook. `link_rocm` links `delayimp.lib` after `mlx`, as `link_cuda`
/// does, so the forced symbol resolves from `mlx.lib`. As with CUDA,
/// `delayload.cpp` is compiled into `mlx.lib` but
/// nothing references the hook, so without `/INCLUDE` the linker would never
/// pull that member and the null default in `delayimp.lib` would win.
///
/// Without these the binary imports the DLLs eagerly by name: it fails to
/// start unless the SDK `bin` is on `PATH`, and a same-named DLL in the
/// current directory wins over `PATH` (SafeDllSearchMode puts the current
/// directory first).
pub fn emit_delayload_link_args(layout: &RocmLayout) {
    let names: Vec<String> = std::fs::read_dir(&layout.bin_dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    for dll in select_delayload_dlls(names) {
        println!("cargo:rustc-link-arg=/DELAYLOAD:{dll}");
    }
    println!("cargo:rustc-link-arg=/INCLUDE:__pfnDliNotifyHook2");
}

/// The layout for the target being built, or `None` when the target is not
/// Windows MSVC. Panics with an actionable message when it is but the SDK
/// cannot be located, since the CMake configure would fail later and less
/// clearly. The feature gate in `build.rs` has already rejected every Windows
/// target other than x86_64 MSVC.
pub fn layout_from_env() -> Option<RocmLayout> {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc")
    {
        return None;
    }
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    let rocm_path = env::var("ROCM_PATH").ok();
    let hip_path = env::var("HIP_PATH").ok();
    let Some(root) = resolve_rocm_root(rocm_path.as_deref(), hip_path.as_deref()) else {
        panic!(
            "mlxcel-core: the `rocm` feature needs a ROCm SDK. Set ROCM_PATH (or HIP_PATH) to \
             its root, the directory holding bin\\hipcc.exe and lib\\amdhip64.lib."
        );
    };
    let layout = layout(&root, Path::exists);
    if !layout.hipcc.exists() {
        panic!(
            "mlxcel-core: {} does not exist. Set ROCM_PATH (or HIP_PATH) to the ROCm SDK root.",
            layout.hipcc.display()
        );
    }
    let amdhip = layout.lib_dir.join("amdhip64.lib");
    if !amdhip.exists() {
        panic!(
            "mlxcel-core: {} does not exist. Set ROCM_PATH (or HIP_PATH) to the ROCm SDK root.",
            amdhip.display()
        );
    }
    Some(layout)
}

#[cfg(test)]
#[path = "windows_rocm_tests.rs"]
mod tests;
