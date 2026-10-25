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

use super::{arch_subdir, cmake_path, layout, resolve_cuda_root, select_delayload_dlls};
use std::path::{Path, PathBuf};

#[test]
fn arch_subdir_names_the_toolkit_directories_for_both_windows_targets() {
    assert_eq!(arch_subdir("x86_64"), Some("x64"));
    assert_eq!(arch_subdir("aarch64"), Some("arm64"));
    assert_eq!(arch_subdir("x86"), None);
    assert_eq!(arch_subdir("arm"), None);
}

#[test]
fn cuda_home_wins_over_cuda_path_and_empty_values_are_unset() {
    assert_eq!(
        resolve_cuda_root(Some("C:/cuda-home"), Some("C:/cuda-path")),
        Some(PathBuf::from("C:/cuda-home"))
    );
    assert_eq!(
        resolve_cuda_root(None, Some("C:/cuda-path")),
        Some(PathBuf::from("C:/cuda-path"))
    );
    assert_eq!(
        resolve_cuda_root(Some("  "), Some("C:/cuda-path")),
        Some(PathBuf::from("C:/cuda-path"))
    );
    assert_eq!(resolve_cuda_root(Some(""), None), None);
    assert_eq!(resolve_cuda_root(None, None), None);
}

#[test]
fn layout_uses_the_per_arch_redistributable_directories() {
    let root = Path::new("C:/cuda");
    let l = layout(root, None, "arm64");
    assert_eq!(l.nvcc, root.join("bin").join("nvcc.exe"));
    assert_eq!(l.lib_dir, root.join("lib").join("arm64"));
    assert_eq!(l.dll_dir, root.join("bin").join("arm64"));
    // cuDNN merged into the toolkit shares its directories.
    assert_eq!(l.cudnn_include_dir, root.join("include"));
    assert_eq!(l.cudnn_lib_dir, l.lib_dir);
    assert_eq!(l.cudnn_dll_dir, l.dll_dir);
}

#[test]
fn a_separate_cudnn_root_moves_only_the_cudnn_directories() {
    let root = Path::new("C:/cuda");
    let cudnn = Path::new("D:/cudnn");
    let l = layout(root, Some(cudnn), "x64");
    assert_eq!(l.lib_dir, root.join("lib").join("x64"));
    assert_eq!(l.dll_dir, root.join("bin").join("x64"));
    assert_eq!(l.cudnn_include_dir, cudnn.join("include"));
    assert_eq!(l.cudnn_lib_dir, cudnn.join("lib").join("x64"));
    assert_eq!(l.cudnn_dll_dir, cudnn.join("bin").join("x64"));
}

#[test]
fn delayload_selects_exactly_the_dlls_behind_the_linked_import_libraries() {
    // The CUDA 13.4 + cuDNN 9.27 Windows arm64 `bin\arm64` listing.
    let listing = [
        "cublas64_13.dll",
        "cublasLt64_13.dll",
        "cudart64_13.dll",
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll",
        "cufft64_12.dll",
        "cufftw64_12.dll",
        "curand64_10.dll",
        "cusolver64_12.dll",
        "cusolverMg64_12.dll",
        "cusparse64_12.dll",
        "nvJitLink_130_0.dll",
        "nvblas64_13.dll",
        "nvrtc-builtins64_134.dll",
        "nvrtc64_130_0.dll",
        "nvrtc64_130_0.alt.dll",
        "cupti64_2026.3.1.dll",
        "README.txt",
    ];
    assert_eq!(
        select_delayload_dlls(listing),
        vec![
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "cudart64_13.dll",
            "cudnn64_9.dll",
            "cufft64_12.dll",
            "cusolver64_12.dll",
            "nvrtc64_130_0.dll",
        ]
    );
}

#[test]
fn cmake_path_leaves_no_backslash_for_a_string_literal_to_misread() {
    // `\t`, `\b` and `\a` in this path are what broke the delay-load hook when
    // the bin dir reached MLX's string-literal define with backslashes.
    assert_eq!(
        cmake_path(Path::new(r"C:\Users\jshin\tools\cuda\13.4\bin\arm64")),
        "C:/Users/jshin/tools/cuda/13.4/bin/arm64"
    );
    assert_eq!(
        cmake_path(Path::new("C:/already/forward")),
        "C:/already/forward"
    );
}

#[test]
fn delayload_matching_ignores_case_and_deduplicates() {
    assert_eq!(
        select_delayload_dlls(["CUDART64_13.DLL", "CUDART64_13.DLL", "cudart64_13.lib"]),
        vec!["CUDART64_13.DLL"]
    );
}
