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
// /INCLUDE this hook, so the binaries start without the SDK on PATH.
//
// Runtime directory (lablup/mlxcel#2606): every ROCm DLL comes from one
// directory, chosen once per process by choose_rocm_runtime_dir() in
// runtime_dir.h: MLXCEL_ROCM_RUNTIME_DIR if set, else the directory of the
// binary when it holds the HIP runtime (a runtime bundle extracted beside
// mlxcel.exe), else MLXCEL_ROCM_BIN_DIR, the SDK bin directory recorded at
// build time. A DLL missing from the chosen directory is an error, never a
// reason to try the next candidate, so DLLs of two ROCm installations are
// never mixed. The choice is printed once as "[mlx-rocm] ROCm runtime: ...".
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
// list, which Windows consults before any directory. A runtime directory that
// lacks either is an error, reported before any ROCm DLL is loaded. Without
// the preload, amdhip64 picks up the display driver's older
// System32\amd_comgr.dll and finds no device, and a same-named DLL in the
// current directory could be loaded.
// SetDefaultDllDirectories was not used: it changes every later LoadLibrary
// in the process, including the display driver's own loads.

// clang-format off
#include <windows.h> // must be included first
#include <delayimp.h>
// clang-format on

#include "mlx/backend/rocm/runtime_dir.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef MLXCEL_ROCM_BIN_DIR
#error "MLXCEL_ROCM_BIN_DIR must name the ROCm SDK bin directory"
#endif

// The image this file is linked into (the linker defines it).
extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace mlx::core::rocm {

namespace fs = std::filesystem;

namespace {

// Convert UTF-8 explicitly rather than through the ANSI code page.
std::wstring utf8_to_wide(std::string_view utf8) {
  int n = ::MultiByteToWideChar(
      CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
  std::wstring wide(n > 0 ? n : 0, L'\0');
  if (n > 0) {
    ::MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), n);
  }
  return wide;
}

// UTF-8 for diagnostics. MSVC's printf %ls converts through the C locale,
// which stops at the first non-ASCII character and drops the rest of the
// message, so a runtime directory under a non-ASCII user name would lose the
// path and the advice after it. Unpaired surrogates become U+FFFD; never
// throws, so it is safe inside the delay-load hook.
std::string wide_to_utf8(std::wstring_view wide) {
  int n = ::WideCharToMultiByte(
      CP_UTF8,
      0,
      wide.data(),
      static_cast<int>(wide.size()),
      nullptr,
      0,
      nullptr,
      nullptr);
  std::string out(n > 0 ? n : 0, '\0');
  if (n > 0) {
    ::WideCharToMultiByte(
        CP_UTF8,
        0,
        wide.data(),
        static_cast<int>(wide.size()),
        out.data(),
        n,
        nullptr,
        nullptr);
  }
  return out;
}

std::string to_utf8(const fs::path& path) {
  return wide_to_utf8(path.native());
}

// The build-time SDK bin directory. CMake passes it as UTF-8 with forward
// slashes.
const fs::path& baked_bin_dir() {
  static const fs::path dir =
      fs::path(utf8_to_wide(MLXCEL_ROCM_BIN_DIR)).make_preferred();
  return dir;
}

// The directory of the module this hook is linked into. mlx is a static
// library, so for mlxcel.exe and mlxcel-server.exe it is the executable's
// directory. Empty if Windows cannot report it.
const fs::path& binary_dir() {
  static const fs::path dir = []() -> fs::path {
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
      DWORD n = ::GetModuleFileNameW(
          reinterpret_cast<HMODULE>(&__ImageBase),
          buf.data(),
          static_cast<DWORD>(buf.size()));
      if (n == 0) {
        return {};
      }
      if (n < buf.size()) {
        buf.resize(n);
        return fs::path(buf).parent_path();
      }
      if (buf.size() >= 32768) {
        return {};
      }
      buf.resize(buf.size() * 2);
    }
  }();
  return dir;
}

// MLXCEL_ROCM_RUNTIME_DIR, read as UTF-16 so any path works; empty if unset
// or empty.
fs::path env_runtime_dir() {
  const wchar_t* value = ::_wgetenv(L"MLXCEL_ROCM_RUNTIME_DIR");
  return (value && *value) ? fs::path(value).make_preferred() : fs::path();
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

// The HIP runtime DLL this binary delay-loads (amdhip64_7.dll for ROCm 10),
// read from the image's delay-import table, so a candidate directory is
// judged by the exact name the binary will ask for. Empty if there is none.
std::string delay_loaded_hip_runtime() {
  auto* base = reinterpret_cast<const unsigned char*>(&__ImageBase);
  auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(
      base + __ImageBase.e_lfanew);
  const IMAGE_DATA_DIRECTORY& entry =
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
  if (entry.VirtualAddress == 0) {
    return {};
  }
  for (auto* d = reinterpret_cast<const IMAGE_DELAYLOAD_DESCRIPTOR*>(
           base + entry.VirtualAddress);
       d->DllNameRVA != 0;
       ++d) {
    if (!d->Attributes.RvaBased) {
      continue;
    }
    const char* name = reinterpret_cast<const char*>(base + d->DllNameRVA);
    if (starts_with_ci(name, "amdhip64_")) {
      return name;
    }
  }
  return {};
}

const std::string& hip_runtime_name() {
  static const std::string name = [] {
    std::string n = delay_loaded_hip_runtime();
    return n.empty() ? std::string("amdhip64_7.dll") : n;
  }();
  return name;
}

bool has_hip_runtime(const fs::path& dir) {
  std::error_code ec;
  return fs::is_regular_file(dir / hip_runtime_name(), ec);
}

// The runtime directory for this process, chosen on first use.
const RuntimeDirChoice& runtime_choice() {
  static const RuntimeDirChoice choice = [] {
    RuntimeDirChoice c = choose_rocm_runtime_dir(
        env_runtime_dir(), binary_dir(), baked_bin_dir(), has_hip_runtime);
    if (c.error == RuntimeDirError::None) {
      std::fprintf(
          stderr,
          "[mlx-rocm] ROCm runtime: %s (%s)\n",
          to_utf8(c.dir).c_str(),
          runtime_dir_source_name(c.source));
      // rocBLAS and hipBLASLt open kernel files up to 128 characters below
      // this directory (ROCm 10.1) with MAX_PATH-limited calls; past 260
      // characters the first GEMM crashed with an access violation.
      constexpr size_t kMaxRuntimeDirChars = MAX_PATH - 2 - 128;
      if (c.dir.native().size() > kMaxRuntimeDirChars) {
        std::fprintf(
            stderr,
            "[mlx-rocm] warning: the ROCm runtime path is %zu characters "
            "long; rocBLAS and hipBLASLt cannot open kernel files whose full "
            "path exceeds %d characters. Move the runtime to a path of at "
            "most %zu characters.\n",
            c.dir.native().size(),
            MAX_PATH,
            kMaxRuntimeDirChars);
      }
      std::fflush(stderr);
    }
    return c;
  }();
  return choice;
}

// Outcome of preparing the runtime directory: the preload prefixes with no
// DLL in it, or the preload that Windows refused (with its error).
struct PreloadResult {
  std::vector<std::string_view> missing;
  fs::path failed;
  DWORD error = 0;
  bool ok() const {
    return missing.empty() && failed.empty();
  }
};

// Runs once, before the first ROCm DLL is loaded: checks that `dir` holds
// every DLL loaded by plain name at run time (kRuntimePreloadPrefixes) and
// preloads them by full path. A missing one is reported before anything is
// loaded, because the plain-name load would otherwise take another
// installation's copy (the driver's System32\amd_comgr.dll).
const PreloadResult& prepare_rocm_runtime(const fs::path& dir) {
  static const PreloadResult result = [&dir] {
    PreloadResult r;
    std::vector<fs::path> paths;
    std::vector<std::string> names;
    std::error_code ec;
    for (auto it = fs::directory_iterator(dir, ec);
         !ec && it != fs::directory_iterator();
         it.increment(ec)) {
      // UTF-8 without exceptions, so no file name in the directory (not even
      // an unpaired surrogate) can throw inside the delay-load hook.
      names.push_back(wide_to_utf8(it->path().filename().native()));
      paths.push_back(it->path());
    }
    r.missing = missing_runtime_preloads(names);
    if (!r.missing.empty()) {
      return r;
    }
    // rocBLAS and hipBLASLt locate their kernel libraries beside their DLLs;
    // this also serves any LoadLibraryEx that asks for user directories.
    ::AddDllDirectory(dir.c_str());
    for (std::string_view prefix : kRuntimePreloadPrefixes) {
      for (size_t i = 0; i < names.size(); ++i) {
        if (is_preload_dll(names[i], prefix) && !load_beside(paths[i])) {
          r.error = ::GetLastError();
          r.failed = paths[i];
          return r;
        }
      }
    }
    return r;
  }();
  return result;
}

// Raise the exception the delay-load helper raises when a DLL cannot be
// loaded. Returning null instead would make the helper retry with a plain
// LoadLibrary, whose search includes the current directory and PATH.
[[noreturn]] void raise_load_failure(PDelayLoadInfo pdli, DWORD err) {
  std::fflush(stderr);
  pdli->dwLastError = err;
  ULONG_PTR args[] = {reinterpret_cast<ULONG_PTR>(pdli)};
  ::RaiseException(
      VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND), 0, 1, args);
  std::abort();
}

// No candidate directory qualified: say why, then fail as for a missing DLL.
[[noreturn]] void fail_choice(PDelayLoadInfo pdli, const RuntimeDirChoice& c) {
  const char* hip = hip_runtime_name().c_str();
  switch (c.error) {
    case RuntimeDirError::EnvNotAbsolute:
      std::fprintf(
          stderr,
          "mlxcel: MLXCEL_ROCM_RUNTIME_DIR must be an absolute path, got "
          "\"%s\".\n",
          to_utf8(c.dir).c_str());
      break;
    case RuntimeDirError::EnvMissingRuntime:
      std::fprintf(
          stderr,
          "mlxcel: MLXCEL_ROCM_RUNTIME_DIR names %s, which holds no %s. "
          "Point it at the bin folder of an extracted mlxcel ROCm runtime "
          "bundle.\n",
          to_utf8(c.dir).c_str(),
          hip);
      break;
    default:
      std::fprintf(
          stderr,
          "mlxcel: cannot find the ROCm runtime (%s). Looked beside the "
          "executable (%s) and in the ROCm SDK recorded at build time "
          "(%s). Put the executable in the bin folder of an extracted "
          "mlxcel ROCm runtime bundle, or set MLXCEL_ROCM_RUNTIME_DIR to "
          "that bin folder.\n",
          hip,
          to_utf8(binary_dir()).c_str(),
          to_utf8(baked_bin_dir()).c_str());
      break;
  }
  raise_load_failure(pdli, ERROR_MOD_NOT_FOUND);
}

// A DLL is missing from the chosen directory: name the expected path.
[[noreturn]] void fail_load(
    PDelayLoadInfo pdli,
    const fs::path& path,
    const RuntimeDirChoice& c) {
  DWORD err = ::GetLastError();
  std::fprintf(
      stderr,
      "mlxcel: cannot load %s from the ROCm runtime directory chosen from "
      "%s (%s), Windows error %lu. Other ROCm installations are not "
      "searched; restore the file or use a complete runtime directory.\n",
      pdli->szDll,
      runtime_dir_source_name(c.source),
      to_utf8(path).c_str(),
      static_cast<unsigned long>(err));
  raise_load_failure(pdli, err);
}

// The chosen directory lacks, or cannot load, a DLL that is opened by plain
// name at run time: name the directory and the file pattern.
[[noreturn]] void fail_preload(
    PDelayLoadInfo pdli,
    const RuntimeDirChoice& c,
    const PreloadResult& r) {
  if (!r.missing.empty()) {
    std::string patterns;
    for (std::string_view prefix : r.missing) {
      if (!patterns.empty()) {
        patterns += ", ";
      }
      patterns.append(prefix.data(), prefix.size());
      patterns += "*.dll";
    }
    std::fprintf(
        stderr,
        "mlxcel: the ROCm runtime directory chosen from %s (%s) holds no "
        "%s. The HIP runtime loads these files by name, and other ROCm "
        "installations (such as the display driver's copy in System32) are "
        "not used; restore the files or use a complete runtime directory.\n",
        runtime_dir_source_name(c.source),
        to_utf8(c.dir).c_str(),
        patterns.c_str());
    raise_load_failure(pdli, ERROR_MOD_NOT_FOUND);
  }
  std::fprintf(
      stderr,
      "mlxcel: cannot load %s from the ROCm runtime directory chosen from "
      "%s, Windows error %lu. Other ROCm installations are not searched; "
      "restore the file or use a complete runtime directory.\n",
      to_utf8(r.failed).c_str(),
      runtime_dir_source_name(c.source),
      static_cast<unsigned long>(r.error));
  raise_load_failure(pdli, r.error);
}

FARPROC WINAPI delayload_helper(unsigned dliNotify, PDelayLoadInfo pdli) {
  if (dliNotify != dliNotePreLoadLibrary || !is_rocm_dll(pdli->szDll)) {
    return nullptr;
  }
  const RuntimeDirChoice& choice = runtime_choice();
  if (choice.error != RuntimeDirError::None) {
    fail_choice(pdli, choice);
  }
  const PreloadResult& preload = prepare_rocm_runtime(choice.dir);
  if (!preload.ok()) {
    fail_preload(pdli, choice, preload);
  }
  fs::path path = choice.dir / pdli->szDll;
  HMODULE mod = load_beside(path);
  if (!mod) {
    fail_load(pdli, path, choice);
  }
  return reinterpret_cast<FARPROC>(mod);
}

} // namespace

const fs::path& rocm_runtime_dir() {
  static const fs::path empty;
  const RuntimeDirChoice& choice = runtime_choice();
  return choice.error == RuntimeDirError::None ? choice.dir : empty;
}

RuntimeDirSource rocm_runtime_dir_source() {
  const RuntimeDirChoice& choice = runtime_choice();
  return choice.error == RuntimeDirError::None ? choice.source
                                               : RuntimeDirSource::None;
}

} // namespace mlx::core::rocm

extern "C" const PfnDliHook __pfnDliNotifyHook2 =
    mlx::core::rocm::delayload_helper;
