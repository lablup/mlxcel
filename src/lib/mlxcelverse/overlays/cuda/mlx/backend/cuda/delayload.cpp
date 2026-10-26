// Copyright © 2026 Apple Inc.
// Modified from upstream MLX 7ca40033132dae444d45f7fc3daae57e24c13f6c mlx/backend/cuda/delayload.cpp
// Modified by mlxcel (#58) so a statically linked mlxcel.exe runs without the
// CUDA toolkit on PATH:
// - Every DLL the hook resolves is loaded with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
//   | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, so its own imports (cuSOLVER's
//   cuSPARSE and nvJitLink, cuDNN's sub-libraries) are found next to it, and
//   neither the current directory nor PATH is searched for them. A plain
//   LoadLibraryW by full path resolves dependencies by module name through the
//   default search order, which AddDllDirectory does not extend.
// - load_cudnn() preloads cuBLAS, cuBLASLt, NVRTC and every cudnn_*.dll
//   sub-library, not only cudnn_graph, because cuDNN opens all of them by
//   plain name on first use.
// - load_nvrtc() preloads nvrtc-builtins*.dll from the NVRTC directory by full
//   path. NVRTC loads it by name at its first compile, which otherwise fails
//   with "failed to open nvrtc-builtins64_<ver>.dll".

#include "mlx/backend/common/utils.h"

// clang-format off
#include <windows.h> // must be included first
#include <delayimp.h>
// clang-format on

#include <filesystem>
#include <string_view>
#include <system_error>
#include <vector>
#include <cuda_runtime.h>

namespace mlx::core::cu {

// Defined in dirs.cpp to avoid invalidating compile cache.
const char* cuda_bin_dir();
const char* cudnn_bin_dir();

namespace fs = std::filesystem;

// Load `path` so that its own imports resolve from its directory, the
// application directory, System32 and AddDllDirectory entries, but never from
// the current directory or PATH.
inline HMODULE load_beside(const fs::path& path) {
  return ::LoadLibraryExW(
      path.c_str(),
      nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

// Load every `<prefix>*.dll` in `dir` with load_beside.
inline void preload_prefixed(const fs::path& dir, std::string_view prefix) {
  std::error_code ec;
  for (auto it = fs::directory_iterator(dir, ec);
       !ec && it != fs::directory_iterator();
       it.increment(ec)) {
    if (it->path().filename().string().starts_with(prefix) &&
        it->path().extension() == ".dll") {
      load_beside(it->path());
    }
  }
}

#if CUDART_VERSION == 0
#error "CUDART_VERSION not defined"
#elif CUDART_VERSION < 13000
#define BIN_DIR(name) "../nvidia/" name "/bin"
#elif CUDART_VERSION < 14000
#define BIN_DIR(name) "../nvidia/cu13/bin/x86_64"
#else
#define BIN_DIR(name) "../nvidia/cu14/bin/x86_64"
#endif

inline fs::path relative_to_current_binary(const char* relative) {
  return fs::absolute(current_binary_dir() / relative);
}

inline fs::path cublas_dir() {
  if (const char* dir = cuda_bin_dir()) {
    return fs::path(dir);
  }
  return relative_to_current_binary(BIN_DIR("cublas"));
}

fs::path load_nvrtc() {
  fs::path nvrtc_dir;
  if (const char* dir = cuda_bin_dir()) {
    nvrtc_dir = fs::path(dir);
  } else {
    nvrtc_dir = relative_to_current_binary(BIN_DIR("cuda_nvrtc"));
  }
  // Internally nvrtc loads some libs dynamically, add to search dirs.
  ::AddDllDirectory(nvrtc_dir.c_str());
  // AddDllDirectory only reaches LoadLibraryEx calls that ask for user dirs;
  // NVRTC loads its builtins by plain name, so put them in the process first.
  preload_prefixed(nvrtc_dir, "nvrtc-builtins");
  return nvrtc_dir;
}

fs::path load_cudnn() {
  fs::path cudnn_dir;
  if (const char* dir = cudnn_bin_dir()) {
    cudnn_dir = fs::path(dir);
  } else {
    cudnn_dir = relative_to_current_binary("../nvidia/cudnn/bin");
  }
  // Must load cudnn_graph64_9.dll before locating symbols, otherwise We would
  // get errors like "Invalid handle. Cannot load symbol cudnnCreate".
  // cuDNN opens its other sub-libraries (engines, heuristic, ops, ...) and
  // cuBLAS / cuBLASLt / NVRTC by plain name on first use, which AddDllDirectory
  // below does not reach, so they are put in the process here too: cuBLAS and
  // NVRTC first, then graph, then the rest. Without this cuDNN prints "Could
  // not locate cudnn_engines_runtime_compiled64_9.dll" or, once the engines
  // load, "Could not locate cublasLt64_13.dll" and fails fast.
  load_nvrtc();
  preload_prefixed(cublas_dir(), "cublasLt64_");
  preload_prefixed(cublas_dir(), "cublas64_");
  std::vector<fs::path> sub_libraries;
  std::error_code ec;
  for (auto it = fs::directory_iterator(cudnn_dir, ec);
       !ec && it != fs::directory_iterator();
       it.increment(ec)) {
    auto name = it->path().filename().string();
    if (name.starts_with("cudnn_") && it->path().extension() == ".dll") {
      if (name.starts_with("cudnn_graph")) {
        load_beside(it->path());
      } else {
        sub_libraries.push_back(it->path());
      }
    }
  }
  for (const auto& dll : sub_libraries) {
    load_beside(dll);
  }
  // Internally cuDNN loads some libs dynamically, add to search dirs.
  ::AddDllDirectory(cudnn_dir.c_str());
  ::AddDllDirectory(cublas_dir().c_str());
  return cudnn_dir;
}

fs::path load_cusolver() {
  fs::path cusolver_dir;
  if (const char* dir = cuda_bin_dir()) {
    cusolver_dir = dir;
  } else {
    cusolver_dir = relative_to_current_binary(BIN_DIR("cusolver"));
    ::AddDllDirectory(cublas_dir().c_str());
    ::AddDllDirectory(relative_to_current_binary(BIN_DIR("cusparse")).c_str());
    ::AddDllDirectory(relative_to_current_binary(BIN_DIR("nvjitlink")).c_str());
  }
  ::AddDllDirectory(cusolver_dir.c_str());
  return cusolver_dir;
}

// Called by system before loading a lazy-loaded DLL.
FARPROC WINAPI delayload_helper(unsigned dliNotify, PDelayLoadInfo pdli) {
  HMODULE mod = NULL;
  try {
    if (dliNotify == dliNotePreLoadLibrary) {
      std::string dll = pdli->szDll;
      if (dll.starts_with("cudnn")) {
        static auto cudnn_dir = load_cudnn();
        mod = load_beside((cudnn_dir / dll).c_str());
      } else if (dll.starts_with("cublas")) {
        mod = load_beside((cublas_dir() / dll).c_str());
      } else if (dll.starts_with("cusolver")) {
        static auto cusolver_dir = load_cusolver();
        mod = load_beside((cusolver_dir / dll).c_str());
      } else if (dll.starts_with("nvrtc")) {
        static auto nvrtc_dir = load_nvrtc();
        mod = load_beside((nvrtc_dir / dll).c_str());
      } else if (const char* dir = cuda_bin_dir()) {
        mod = load_beside((fs::path(dir) / dll).c_str());
      }
    }
  } catch (...) {
    // Return control to system on error.
  }
  return reinterpret_cast<FARPROC>(mod);
}

} // namespace mlx::core::cu

extern "C" const PfnDliHook __pfnDliNotifyHook2 =
    mlx::core::cu::delayload_helper;
