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
// New file for mlxcel (no upstream MLX counterpart)

// Choice of the one directory the Windows CUDA runtime is loaded from
// (lablup/mlxcel#2640), mirroring the ROCm overlay's runtime_dir.h
// (lablup/mlxcel#2606). delayload.cpp loads every CUDA, cuDNN and NVRTC DLL
// from it and jit_module.cpp compiles against its include/ when present. The
// functions are pure and header-only so scripts/mlxcelverse/
// cuda_runtime_dir_test.sh can test them on any host without CUDA or Windows.

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mlx::core::cu {

enum class RuntimeDirSource {
  None,
  Env, // MLXCEL_CUDA_RUNTIME_DIR
  ExeDir, // the directory of the running executable
  Baked, // the toolkit and cuDNN bin directories recorded at build time
};

enum class RuntimeDirError {
  None,
  EnvNotAbsolute, // MLXCEL_CUDA_RUNTIME_DIR is a relative path
  EnvMissingRuntime, // MLXCEL_CUDA_RUNTIME_DIR lacks the CUDA runtime DLL
};

struct RuntimeDirChoice {
  std::filesystem::path dir;
  RuntimeDirSource source = RuntimeDirSource::None;
  RuntimeDirError error = RuntimeDirError::None;
};

// Pick the CUDA runtime directory from, in order:
//   1. `env_dir` (MLXCEL_CUDA_RUNTIME_DIR; empty means unset). An explicit
//      choice: it must be absolute and hold the CUDA runtime, otherwise the
//      result is an error rather than a fall back to another candidate, so a
//      mistyped override never silently runs a different CUDA.
//   2. `exe_dir`, when it holds the CUDA runtime (a bundle extracted beside
//      mlxcel.exe).
//   3. `baked_dir`, the build-time layout: the toolkit and cuDNN `bin\<arch>`
//      directories, or the pip-wheel `../nvidia/...` directories when none was
//      recorded. It is returned unchecked, because a developer build keeps
//      today's per-library lookup there (delayload.cpp), including its
//      fallback to the standard search when a DLL is not found.
// `has_cuda_runtime(dir)` reports whether `dir` contains the DLL the binary
// delay-loads that marks a CUDA runtime directory (cuBLASLt at the current
// pin; see cuda_runtime_marker()). Only one directory is ever returned for a
// bundle, so DLLs from two CUDA installations are never mixed.
inline RuntimeDirChoice choose_cuda_runtime_dir(
    const std::filesystem::path& env_dir,
    const std::filesystem::path& exe_dir,
    const std::filesystem::path& baked_dir,
    const std::function<bool(const std::filesystem::path&)>& has_cuda_runtime) {
  RuntimeDirChoice choice;
  if (!env_dir.empty()) {
    choice.dir = env_dir;
    choice.source = RuntimeDirSource::Env;
    if (!env_dir.is_absolute()) {
      choice.error = RuntimeDirError::EnvNotAbsolute;
    } else if (!has_cuda_runtime(env_dir)) {
      choice.error = RuntimeDirError::EnvMissingRuntime;
    }
    return choice;
  }
  if (!exe_dir.empty() && has_cuda_runtime(exe_dir)) {
    choice.dir = exe_dir;
    choice.source = RuntimeDirSource::ExeDir;
    return choice;
  }
  choice.dir = baked_dir;
  choice.source = RuntimeDirSource::Baked;
  return choice;
}

// Whether the runtime directory is a bundle (every DLL from that one
// directory, missing files are errors) rather than the build-time layout.
inline bool is_runtime_bundle(RuntimeDirSource source) {
  return source == RuntimeDirSource::Env || source == RuntimeDirSource::ExeDir;
}

inline const char* runtime_dir_source_name(RuntimeDirSource source) {
  switch (source) {
    case RuntimeDirSource::Env:
      return "MLXCEL_CUDA_RUNTIME_DIR";
    case RuntimeDirSource::ExeDir:
      return "executable directory";
    case RuntimeDirSource::Baked:
      return "build-time CUDA toolkit";
    case RuntimeDirSource::None:
      break;
  }
  return "none";
}

// Case-insensitive ASCII prefix test (`prefix` must be lower case).
inline bool starts_with_ci(std::string_view name, std::string_view prefix) {
  if (name.size() < prefix.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.size(); ++i) {
    char c = name[i];
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
    if (c != prefix[i]) {
      return false;
    }
  }
  return true;
}

// The DLLs mlxcel's build scripts delay-load, by lower-case file-name prefix;
// windows_cuda::DELAYLOAD_DLL_PREFIXES on the Rust side.
inline constexpr std::string_view kCudaDelayloadPrefixes[] = {
    "cudart64_",
    "cublas64_",
    "cublaslt64_",
    "cufft64_",
    "cusolver64_",
    "nvrtc64_",
    "cudnn64_",
};

inline bool is_cuda_dll(std::string_view name) {
  for (std::string_view prefix : kCudaDelayloadPrefixes) {
    if (starts_with_ci(name, prefix)) {
      return true;
    }
  }
  return false;
}

// The delay-loaded DLL whose presence makes a directory a CUDA runtime
// directory, picked from `delay_imports` (the binary's delay-import table):
// the CUDA runtime when the binary imports it, else cuBLASLt (which every
// matmul uses; nvcc links the static cudart at the current pin), else NVRTC.
// Returns "" when the table holds none of them.
inline std::string cuda_runtime_marker(
    const std::vector<std::string>& delay_imports) {
  for (std::string_view prefix : {"cudart64_", "cublaslt64_", "nvrtc64_"}) {
    for (const std::string& name : delay_imports) {
      if (starts_with_ci(name, prefix)) {
        return name;
      }
    }
  }
  return "";
}

// DLLs that the CUDA libraries open by plain name at run time, which a
// LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR load of their importer does not reach. In
// a bundle delayload.cpp preloads them by full path from the runtime
// directory, so a later plain-name LoadLibrary finds them in the loaded-module
// list instead of searching the current directory and PATH. Each prefix must
// match at least one `<prefix>*.dll` in the runtime directory, or the binary
// stops before loading any CUDA DLL.
//   - kCorePreloadPrefixes, preloaded before the first CUDA DLL: NVRTC and its
//     builtins (cuBLASLt and cuDNN open nvrtc64_* by name, NVRTC opens
//     nvrtc-builtins64_* at its first compile), cuBLAS and cuBLASLt (cuDNN
//     opens both by name), and nvJitLink (cuSPARSE, imported by cuSOLVER,
//     opens it by name).
//   - kCudnnPreloadPrefixes, preloaded when cuDNN is first loaded, graph
//     first: the sub-libraries cudnn64_9 and cudnn_graph open by name. Any
//     other cudnn_*.dll in the directory (cudnn_ext and
//     cudnn_engines_tensor_ir in cuDNN 9.27) is preloaded too but not
//     required, since earlier cuDNN 9 releases do not ship it.
//   - kImportOnlyPrefixes, required but not preloaded: DLLs only a bundle DLL
//     imports (cuSPARSE, imported by cuSOLVER). The loader resolves them from
//     the runtime directory, but when one is missing the failed load names
//     only its importer, so the check names the missing file instead.
inline constexpr std::string_view kCorePreloadPrefixes[] = {
    "nvrtc64_",
    "nvrtc-builtins",
    "cublaslt64_",
    "cublas64_",
    "nvjitlink_",
};

inline constexpr std::string_view kImportOnlyPrefixes[] = {
    "cusparse64_",
};

inline constexpr std::string_view kCudnnPreloadPrefixes[] = {
    "cudnn_graph",
    "cudnn_ops",
    "cudnn_cnn",
    "cudnn_adv",
    "cudnn_heuristic",
    "cudnn_engines_precompiled",
    "cudnn_engines_runtime_compiled",
};

// Whether `filename` is `<prefix>*.dll` (case-insensitive). `.alt.dll`
// side-by-side variants do not count.
inline bool is_preload_dll(std::string_view filename, std::string_view prefix) {
  constexpr std::string_view ext = ".dll";
  constexpr std::string_view alt = ".alt.dll";
  if (filename.size() < prefix.size() + ext.size() ||
      !starts_with_ci(filename, prefix) ||
      !starts_with_ci(filename.substr(filename.size() - ext.size()), ext)) {
    return false;
  }
  return filename.size() < alt.size() ||
      !starts_with_ci(filename.substr(filename.size() - alt.size()), alt);
}

// Whether `filenames` (the entries of the executable's directory) hold CUDA
// DLLs of a bundle although the directory did not qualify as one, i.e. the
// bundle lacks its marker DLL. delayload.cpp then names the marker's expected
// path when the build-time lookup fails, instead of failing silently.
inline bool looks_like_runtime_bundle(const std::vector<std::string>& filenames) {
  for (const std::string& name : filenames) {
    if ((is_cuda_dll(name) && is_preload_dll(name, "")) ||
        is_preload_dll(name, "cudnn_") || is_preload_dll(name, "nvrtc-builtins")) {
      return true;
    }
  }
  return false;
}

// The prefixes of kCorePreloadPrefixes, kImportOnlyPrefixes and
// kCudnnPreloadPrefixes with no matching DLL among `filenames` (the entries
// of the runtime directory). Empty when the directory can serve every
// plain-name load and import-only dependency itself.
inline std::vector<std::string_view> missing_runtime_preloads(
    const std::vector<std::string>& filenames) {
  std::vector<std::string_view> missing;
  auto scan = [&](std::string_view prefix) {
    for (const std::string& name : filenames) {
      if (is_preload_dll(name, prefix)) {
        return;
      }
    }
    missing.push_back(prefix);
  };
  for (std::string_view prefix : kCorePreloadPrefixes) {
    scan(prefix);
  }
  for (std::string_view prefix : kImportOnlyPrefixes) {
    scan(prefix);
  }
  for (std::string_view prefix : kCudnnPreloadPrefixes) {
    scan(prefix);
  }
  return missing;
}

// The include directory NVRTC takes the CUDA headers from, from the
// candidates in order: CUDA_HOME (an explicit override, `<root>/include`);
// then the include/ of the CUDA runtime directory when that directory is a
// bundle (`runtime_is_bundle`, lablup/mlxcel#2640) and holds the headers;
// then CUDA_PATH, which the CUDA installer sets machine-wide and which a
// bundle therefore outranks (the lesson of lablup/mlxcel#2639 for HIP_PATH);
// then the toolkit include directory recorded at build time when it still
// exists (Windows, #58); then `default_include`, the installer's
// Program Files toolkit or /usr/local/cuda (empty when none). Pure: callers
// pass the getenv results (nullptr or "" is unset). Returns "" when nothing
// qualifies. Without a bundle the order is the one jit_module.cpp used
// before: CUDA_HOME, CUDA_PATH, the baked directory, the default.
inline std::string choose_cuda_include_dir(
    const char* cuda_home,
    const char* cuda_path,
    const std::string& runtime_include_dir,
    bool runtime_is_bundle,
    const char* baked_include_dir,
    bool baked_include_dir_exists,
    const std::string& default_include) {
  if (cuda_home && *cuda_home) {
    return std::string(cuda_home) + "/include";
  }
  if (runtime_is_bundle && !runtime_include_dir.empty()) {
    return runtime_include_dir;
  }
  if (cuda_path && *cuda_path) {
    return std::string(cuda_path) + "/include";
  }
  if (baked_include_dir && *baked_include_dir && baked_include_dir_exists) {
    return baked_include_dir;
  }
  return default_include;
}

#if defined(_WIN32)
// The directory chosen for this process by delayload.cpp: a bundle
// directory, the build-time toolkit bin directory, or an empty path when the
// MLXCEL_CUDA_RUNTIME_DIR override is invalid.
const std::filesystem::path& cuda_runtime_dir();

// Where cuda_runtime_dir() came from: RuntimeDirSource::None when the override
// is invalid. jit_module.cpp lets a bundle's headers (Env or ExeDir) outrank
// CUDA_PATH, but not the toolkit the binary was built against (Baked).
RuntimeDirSource cuda_runtime_dir_source();
#endif

} // namespace mlx::core::cu
