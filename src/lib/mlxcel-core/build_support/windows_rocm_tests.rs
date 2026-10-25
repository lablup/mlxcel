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

use super::{
    cmake_path, layout, parse_amdgpu_arch, require_absolute_root, require_delayload_dlls,
    resolve_rocm_root, select_delayload_dlls,
};
use std::path::{Path, PathBuf};

#[test]
fn rocm_path_wins_over_hip_path() {
    assert_eq!(
        resolve_rocm_root(Some("C:/rocm/10.1.0"), Some("C:/hip")),
        Some(PathBuf::from("C:/rocm/10.1.0"))
    );
}

#[test]
fn hip_path_is_the_fallback_when_rocm_path_is_unset_or_empty() {
    assert_eq!(
        resolve_rocm_root(None, Some("C:/hip")),
        Some(PathBuf::from("C:/hip"))
    );
    assert_eq!(
        resolve_rocm_root(Some(""), Some("C:/hip")),
        Some(PathBuf::from("C:/hip"))
    );
    assert_eq!(
        resolve_rocm_root(Some("   "), Some("C:/hip")),
        Some(PathBuf::from("C:/hip"))
    );
}

#[test]
fn no_root_when_both_variables_are_unset_or_empty() {
    assert_eq!(resolve_rocm_root(None, None), None);
    assert_eq!(resolve_rocm_root(Some(""), Some("")), None);
    assert_eq!(resolve_rocm_root(None, Some("")), None);
}

fn llvm_arch(root: &Path) -> PathBuf {
    root.join("lib")
        .join("llvm")
        .join("bin")
        .join("amdgpu-arch.exe")
}

#[test]
fn layout_names_the_tools_and_library_directories_under_the_root() {
    let root = Path::new("C:/rocm/10.1.0");
    let layout = layout(root, |_| true);
    assert_eq!(layout.root, root);
    assert_eq!(layout.bin_dir, root.join("bin"));
    assert_eq!(layout.lib_dir, root.join("lib"));
    assert_eq!(layout.hipcc, root.join("bin").join("hipcc.exe"));
    assert_eq!(layout.amdgpu_arch, root.join("bin").join("amdgpu-arch.exe"));
}

#[test]
fn amdgpu_arch_falls_back_to_the_llvm_bin_directory_of_a_tarball_sdk() {
    let root = Path::new("C:/rocm/10.1.0");
    let only_llvm = llvm_arch(root);
    let layout = layout(root, |path| path == only_llvm);
    assert_eq!(layout.amdgpu_arch, only_llvm);
}

#[test]
fn amdgpu_arch_defaults_to_bin_when_no_candidate_exists() {
    let root = Path::new("C:/rocm/10.1.0");
    let layout = layout(root, |_| false);
    assert_eq!(layout.amdgpu_arch, root.join("bin").join("amdgpu-arch.exe"));
}

#[test]
fn layout_paths_spell_with_forward_slashes_for_cmake() {
    let layout = layout(Path::new(r"C:\rocm\10.1.0"), |_| true);
    assert_eq!(cmake_path(&layout.hipcc), "C:/rocm/10.1.0/bin/hipcc.exe");
}

#[test]
fn amdgpu_arch_single_gpu() {
    assert_eq!(parse_amdgpu_arch("gfx1151\n"), Some("gfx1151".to_string()));
}

#[test]
fn amdgpu_arch_crlf_lines_are_trimmed_sorted_and_deduplicated() {
    assert_eq!(
        parse_amdgpu_arch("gfx1100\r\ngfx1151\r\ngfx1100\r\n"),
        Some("gfx1100;gfx1151".to_string())
    );
}

#[test]
fn amdgpu_arch_without_a_gfx_target_gives_none() {
    assert_eq!(parse_amdgpu_arch(""), None);
    assert_eq!(parse_amdgpu_arch("\r\n"), None);
    assert_eq!(
        parse_amdgpu_arch("error: no AMD GPU found\nAMD Radeon(TM) 8060S\n"),
        None
    );
}

#[test]
fn amdgpu_arch_ignores_non_gfx_lines_around_targets() {
    assert_eq!(
        parse_amdgpu_arch("warning: something\n  gfx1151  \nnot-a-target\n"),
        Some("gfx1151".to_string())
    );
}

#[test]
fn delayload_selects_exactly_the_five_imported_rocm_dlls() {
    // The DLLs of the ROCm 10.1.0 TheRock tarball's `bin\` that matter here,
    // plus the near misses: hiprand (linked but dropped by the linker), the
    // SDK's other DLLs, hipRTC's builtins and hipFFT's FFTW front end (no
    // import library names them) and files that are not DLLs.
    let listing = [
        "amd_comgr.dll",
        "amdhip64_7.dll",
        "HIPFFT.DLL",
        "hipfftw.dll",
        "hiprand.dll",
        "hiprtc-builtins0716.dll",
        "hiprtc0716.dll",
        "libhipblaslt.dll",
        "origami.dll",
        "rocblas",
        "rocblas.dll",
        "rocblas_clients_readme.txt",
        "rocfft.dll",
        "rocm_kpack.dll",
        "hipcc.exe",
        "amdhip64.lib",
    ];
    assert_eq!(
        select_delayload_dlls(listing),
        vec![
            "HIPFFT.DLL",
            "amdhip64_7.dll",
            "hiprtc0716.dll",
            "libhipblaslt.dll",
            "rocblas.dll",
        ]
    );
}

#[test]
fn delayload_matching_ignores_case_sorts_and_deduplicates() {
    assert_eq!(
        select_delayload_dlls([
            "rocblas.dll",
            "AMDHIP64_7.DLL",
            "rocblas.dll",
            "HipRtc-Builtins0716.dll",
            "hipfft.lib",
        ]),
        vec!["AMDHIP64_7.DLL", "rocblas.dll"]
    );
}

#[test]
fn relative_root_is_rejected_with_the_value_named() {
    for relative in ["rocm", "..\\rocm", ".", "bin\\sdk"] {
        let err = require_absolute_root(PathBuf::from(relative)).unwrap_err();
        assert!(err.contains("absolute"), "{err}");
        assert!(err.contains(relative), "{err}");
    }
}

#[cfg(windows)]
#[test]
fn absolute_root_is_accepted_unchanged() {
    let root = PathBuf::from(r"C:\rocm\10.1.0");
    assert_eq!(require_absolute_root(root.clone()), Ok(root));
}

#[test]
fn delayload_requires_the_hip_runtime_dll() {
    let bin = Path::new("C:/rocm/bin");
    let err = require_delayload_dlls(["rocblas.dll", "hipfft.dll"], bin).unwrap_err();
    assert!(err.contains("amdhip64_"), "{err}");
    assert!(err.contains("bin"), "{err}");
    assert!(require_delayload_dlls(Vec::<String>::new(), bin).is_err());
    assert_eq!(
        require_delayload_dlls(["AMDHIP64_7.DLL", "rocblas.dll"], bin).unwrap(),
        vec!["AMDHIP64_7.DLL", "rocblas.dll"]
    );
}
