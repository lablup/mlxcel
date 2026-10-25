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

// Delay-load notify hook for the ROCm DLLs on Windows (lablup/mlxcel#2602),
// modeled on the CUDA backend's delayload.cpp. mlxcel's build scripts pass
// /DELAYLOAD for amdhip64_7, rocblas, libhipblaslt, hipfft and hiprtc and
// /INCLUDE this hook, so the binaries start without the SDK on PATH and every
// ROCm DLL comes from MLXCEL_ROCM_BIN_DIR, the SDK bin directory recorded at
// build time.
//
// Search order: each DLL is loaded by full path with
// LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, so its
// own imports (rocm_kpack, rocfft, origami, ...) resolve beside it, then from
// the application directory, AddDllDirectory entries and System32, and never
// from the current directory or PATH. Two DLLs are not imports but are loaded
// at run time by plain name, which those flags do not reach: amd_comgr.dll
// (by amdhip64 at initialization and by hiprtc) and hiprtc-builtins*.dll (by
// amdhip64). They are preloaded from the same directory before the first ROCm
// DLL loads, so a later plain-name LoadLibrary finds them in the loaded-module
// list, which Windows consults before any directory. Without this, amdhip64
// picks up the display driver's older System32\amd_comgr.dll and finds no
// device, and a same-named DLL in the current directory could be loaded.
// SetDefaultDllDirectories was not used: it changes every later LoadLibrary
// in the process, including the display driver's own loads.

// clang-format off
#include <windows.h> // must be included first
#include <delayimp.h>
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#ifndef MLXCEL_ROCM_BIN_DIR
#error "MLXCEL_ROCM_BIN_DIR must name the ROCm SDK bin directory"
#endif

namespace mlx::core::rocm {

namespace {

namespace fs = std::filesystem;

// The build-time SDK bin directory. CMake passes it as UTF-8 with forward
// slashes; convert explicitly rather than through the ANSI code page.
const fs::path& rocm_bin_dir() {
  static const fs::path dir = [] {
    std::string_view utf8 = MLXCEL_ROCM_BIN_DIR;
    int n = ::MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring wide(n > 0 ? n : 0, L'\0');
    if (n > 0) {
      ::MultiByteToWideChar(
          CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), n);
    }
    return fs::path(wide).make_preferred();
  }();
  return dir;
}

// Load `path` so that its own imports resolve from its directory, the
// application directory, System32 and AddDllDirectory entries, but never from
// the current directory or PATH.
HMODULE load_beside(const fs::path& path) {
  return ::LoadLibraryExW(
      path.c_str(),
      nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

bool starts_with_ci(std::string_view name, std::string_view prefix) {
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

// The DLLs the build scripts delay-load; windows_rocm::DELAYLOAD_DLL_PREFIXES
// on the Rust side.
bool is_rocm_dll(std::string_view name) {
  for (std::string_view prefix :
       {"amdhip64_", "rocblas", "libhipblaslt", "hipfft", "hiprtc"}) {
    if (starts_with_ci(name, prefix)) {
      return true;
    }
  }
  return false;
}

// Load every `<prefix>*.dll` in `dir` with load_beside.
void preload_prefixed(const fs::path& dir, std::string_view prefix) {
  std::error_code ec;
  for (auto it = fs::directory_iterator(dir, ec);
       !ec && it != fs::directory_iterator();
       it.increment(ec)) {
    auto name = it->path().filename().string();
    if (starts_with_ci(name, prefix) && starts_with_ci(
            it->path().extension().string(), ".dll")) {
      load_beside(it->path());
    }
  }
}

// Runs once, before the first ROCm DLL is loaded.
void prepare_rocm_bin() {
  static const bool done = [] {
    const fs::path& dir = rocm_bin_dir();
    // rocBLAS and hipBLASLt locate their kernel libraries beside their DLLs;
    // this also serves any LoadLibraryEx that asks for user directories.
    ::AddDllDirectory(dir.c_str());
    preload_prefixed(dir, "amd_comgr");
    preload_prefixed(dir, "hiprtc-builtins");
    return true;
  }();
  (void)done;
}

// Raise the exception the delay-load helper raises when a DLL cannot be
// loaded, after saying where the DLL was expected. Returning null instead
// would make the helper retry with a plain LoadLibrary, whose search includes
// the current directory and PATH.
[[noreturn]] void fail_load(PDelayLoadInfo pdli, const fs::path& path) {
  DWORD err = ::GetLastError();
  std::fprintf(
      stderr,
      "mlxcel: cannot load %s from the ROCm SDK recorded at build time "
      "(%ls), Windows error %lu. Install the SDK at that path or rebuild "
      "with ROCM_PATH naming it.\n",
      pdli->szDll,
      path.c_str(),
      static_cast<unsigned long>(err));
  std::fflush(stderr);
  pdli->dwLastError = err;
  ULONG_PTR args[] = {reinterpret_cast<ULONG_PTR>(pdli)};
  ::RaiseException(
      VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND), 0, 1, args);
  std::abort();
}

FARPROC WINAPI delayload_helper(unsigned dliNotify, PDelayLoadInfo pdli) {
  if (dliNotify != dliNotePreLoadLibrary || !is_rocm_dll(pdli->szDll)) {
    return nullptr;
  }
  prepare_rocm_bin();
  fs::path path = rocm_bin_dir() / pdli->szDll;
  HMODULE mod = load_beside(path);
  if (!mod) {
    fail_load(pdli, path);
  }
  return reinterpret_cast<FARPROC>(mod);
}

} // namespace

} // namespace mlx::core::rocm

extern "C" const PfnDliHook __pfnDliNotifyHook2 =
    mlx::core::rocm::delayload_helper;
