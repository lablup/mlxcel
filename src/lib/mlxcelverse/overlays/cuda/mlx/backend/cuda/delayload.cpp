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
// Modified by mlxcel (lablup/mlxcel#2640): one runtime directory per process,
// as the ROCm overlay does (lablup/mlxcel#2606). choose_cuda_runtime_dir() in
// runtime_dir.h picks MLXCEL_CUDA_RUNTIME_DIR, else the directory of the
// binary when it holds the CUDA runtime DLL the binary delay-loads (a runtime
// bundle beside mlxcel.exe), else the build-time layout. In a bundle every
// CUDA, cuDNN and NVRTC DLL comes from that one directory: the DLLs the
// libraries open by plain name are preloaded from it, a directory that lacks
// one of them stops the binary before any CUDA DLL loads, and a DLL that
// cannot be loaded raises the delay-load helper's exception with the expected
// full path instead of returning null, which would make the helper retry with
// a plain LoadLibrary that searches the current directory and PATH. The
// build-time layout keeps the upstream per-library lookup above unchanged.
// The choice is printed once as "[mlx-cuda] CUDA runtime: <dir> (<source>)".
// SetDefaultDllDirectories is not used: it changes every later LoadLibrary in
// the process, including the display driver's own loads.

#include "mlx/backend/common/utils.h"

// clang-format off
#include <windows.h> // must be included first
#include <delayimp.h>
// clang-format on

#include "mlx/backend/cuda/runtime_dir.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <cuda_runtime.h>

// The image this file is linked into (the linker defines it).
extern "C" IMAGE_DOS_HEADER __ImageBase;

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

// The build-time layout (no bundle): upstream's per-library lookup. A DLL
// that cannot be loaded returns null, so the helper falls back to its own
// LoadLibrary as before.
HMODULE load_from_build_layout(const std::string& dll) {
  HMODULE mod = NULL;
  try {
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
  } catch (...) {
    // Return control to system on error.
  }
  return mod;
}

namespace {

// UTF-8 for diagnostics. MSVC's printf %ls converts through the C locale,
// which stops at the first non-ASCII character and drops the rest of the
// message. Unpaired surrogates become U+FFFD; never throws, so it is safe
// inside the delay-load hook.
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

// MLXCEL_CUDA_RUNTIME_DIR, read as UTF-16 so any path works; empty if unset
// or empty.
fs::path env_runtime_dir() {
  const wchar_t* value = ::_wgetenv(L"MLXCEL_CUDA_RUNTIME_DIR");
  return (value && *value) ? fs::path(value).make_preferred() : fs::path();
}

// The names in this image's delay-import table.
std::vector<std::string> delay_imports() {
  std::vector<std::string> names;
  auto* base = reinterpret_cast<const unsigned char*>(&__ImageBase);
  auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(
      base + __ImageBase.e_lfanew);
  const IMAGE_DATA_DIRECTORY& entry =
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
  if (entry.VirtualAddress == 0) {
    return names;
  }
  for (auto* d = reinterpret_cast<const IMAGE_DELAYLOAD_DESCRIPTOR*>(
           base + entry.VirtualAddress);
       d->DllNameRVA != 0;
       ++d) {
    if (d->Attributes.RvaBased) {
      names.emplace_back(reinterpret_cast<const char*>(base + d->DllNameRVA));
    }
  }
  return names;
}

// The DLL that marks a CUDA runtime directory, read from the image's
// delay-import table (cublasLt64_13.dll at the current pin), so a candidate
// directory is judged by the exact name the binary will ask for.
const std::string& runtime_marker() {
  static const std::string name = [] {
    std::string n = cuda_runtime_marker(delay_imports());
    if (n.empty()) {
      n = "cublasLt64_" + std::to_string(CUDART_VERSION / 1000) + ".dll";
    }
    return n;
  }();
  return name;
}

bool has_cuda_runtime(const fs::path& dir) {
  std::error_code ec;
  return fs::is_regular_file(dir / runtime_marker(), ec);
}

// The build-time directory reported for a developer build: the toolkit DLL
// directory, or the pip-wheel cuBLAS directory when none was recorded.
fs::path baked_runtime_dir() {
  try {
    return cublas_dir();
  } catch (...) {
    return {};
  }
}

// The runtime directory for this process, chosen on first use.
const RuntimeDirChoice& runtime_choice() {
  static const RuntimeDirChoice choice = [] {
    RuntimeDirChoice c = choose_cuda_runtime_dir(
        env_runtime_dir(), binary_dir(), baked_runtime_dir(), has_cuda_runtime);
    if (c.error == RuntimeDirError::None) {
      std::fprintf(
          stderr,
          "[mlx-cuda] CUDA runtime: %s (%s)\n",
          to_utf8(c.dir).c_str(),
          runtime_dir_source_name(c.source));
      // fs::path(const char*) converts through the ANSI code page and can
      // throw; nothing may leave the delay-load hook as a C++ exception.
      try {
        if (c.source == RuntimeDirSource::Baked && cudnn_bin_dir() &&
            fs::path(cudnn_bin_dir()) != c.dir) {
          std::fprintf(
              stderr,
              "[mlx-cuda] cuDNN: %s (build-time cuDNN)\n",
              cudnn_bin_dir());
        }
      } catch (...) {
      }
      std::fflush(stderr);
    }
    return c;
  }();
  return choice;
}

// The entries of the bundle directory, listed once.
struct BundleListing {
  std::vector<fs::path> paths;
  std::vector<std::string> names; // UTF-8 file names, same order
  std::vector<std::string_view> missing; // preload prefixes with no DLL
};

const BundleListing& bundle_listing(const fs::path& dir) {
  static const BundleListing listing = [&dir] {
    BundleListing l;
    std::error_code ec;
    for (auto it = fs::directory_iterator(dir, ec);
         !ec && it != fs::directory_iterator();
         it.increment(ec)) {
      // UTF-8 without exceptions, so no file name in the directory (not even
      // an unpaired surrogate) can throw inside the delay-load hook.
      l.names.push_back(wide_to_utf8(it->path().filename().native()));
      l.paths.push_back(it->path());
    }
    l.missing = missing_runtime_preloads(l.names);
    return l;
  }();
  return listing;
}

// Outcome of a preload step: the DLL that Windows refused and its error.
struct PreloadFailure {
  fs::path path;
  DWORD error = 0;
  bool failed() const {
    return !path.empty();
  }
};

// Preload every `<prefix>*.dll` of the listing for each prefix, in order.
template <typename Prefixes>
PreloadFailure preload(const BundleListing& l, const Prefixes& prefixes) {
  for (std::string_view prefix : prefixes) {
    for (size_t i = 0; i < l.names.size(); ++i) {
      if (is_preload_dll(l.names[i], prefix) && !load_beside(l.paths[i])) {
        return {l.paths[i], ::GetLastError()};
      }
    }
  }
  return {};
}

// Before the first CUDA DLL: the DLLs other libraries open by plain name.
const PreloadFailure& preload_core(const fs::path& dir) {
  static const PreloadFailure result = [&dir] {
    // Also serves LoadLibraryEx calls that ask for user directories.
    ::AddDllDirectory(dir.c_str());
    const BundleListing& l = bundle_listing(dir);
    return preload(l, kCorePreloadPrefixes);
  }();
  return result;
}

// Before cuDNN: cudnn_graph, then every other cudnn_*.dll sub-library.
const PreloadFailure& preload_cudnn(const fs::path& dir) {
  static const PreloadFailure result = [&dir] {
    const BundleListing& l = bundle_listing(dir);
    constexpr std::string_view graph[] = {"cudnn_graph"};
    PreloadFailure f = preload(l, graph);
    if (f.failed()) {
      return f;
    }
    for (size_t i = 0; i < l.names.size(); ++i) {
      if (is_preload_dll(l.names[i], "cudnn_") &&
          !is_preload_dll(l.names[i], "cudnn_graph") &&
          !load_beside(l.paths[i])) {
        return PreloadFailure{l.paths[i], ::GetLastError()};
      }
    }
    return PreloadFailure{};
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

// The MLXCEL_CUDA_RUNTIME_DIR override is invalid: say why, then fail as for
// a missing DLL.
[[noreturn]] void fail_choice(PDelayLoadInfo pdli, const RuntimeDirChoice& c) {
  if (c.error == RuntimeDirError::EnvNotAbsolute) {
    std::fprintf(
        stderr,
        "mlxcel: MLXCEL_CUDA_RUNTIME_DIR must be an absolute path, got "
        "\"%s\".\n",
        to_utf8(c.dir).c_str());
  } else {
    std::fprintf(
        stderr,
        "mlxcel: MLXCEL_CUDA_RUNTIME_DIR names %s, which holds no %s. Point "
        "it at the bin folder of an extracted mlxcel CUDA runtime bundle.\n",
        to_utf8(c.dir).c_str(),
        runtime_marker().c_str());
  }
  raise_load_failure(pdli, ERROR_MOD_NOT_FOUND);
}

// The bundle lacks DLLs that are opened by plain name at run time, or that
// only another bundle DLL imports: name the expected paths.
[[noreturn]] void fail_missing_preloads(
    PDelayLoadInfo pdli,
    const RuntimeDirChoice& c,
    const std::vector<std::string_view>& missing) {
  std::string paths;
  for (std::string_view prefix : missing) {
    if (!paths.empty()) {
      paths += ", ";
    }
    paths += to_utf8(c.dir / std::string(prefix));
    paths += "*.dll";
  }
  std::fprintf(
      stderr,
      "mlxcel: the CUDA runtime directory chosen from %s (%s) is missing %s. "
      "The CUDA libraries load these files at run time, and other CUDA "
      "installations are not searched; restore the files or use a complete "
      "runtime directory.\n",
      runtime_dir_source_name(c.source),
      to_utf8(c.dir).c_str(),
      paths.c_str());
  raise_load_failure(pdli, ERROR_MOD_NOT_FOUND);
}

// A DLL is missing from, or cannot be loaded from, the bundle: name the
// expected path.
[[noreturn]] void fail_load(
    PDelayLoadInfo pdli,
    const RuntimeDirChoice& c,
    const fs::path& path,
    DWORD err) {
  std::error_code ec;
  bool exists = fs::is_regular_file(path, ec);
  std::fprintf(
      stderr,
      "mlxcel: cannot load %s from the CUDA runtime directory chosen from %s, "
      "Windows error %lu%s. Other CUDA installations are not searched; "
      "restore the file or use a complete runtime directory.\n",
      to_utf8(path).c_str(),
      runtime_dir_source_name(c.source),
      static_cast<unsigned long>(err),
      exists ? " (the file exists; a DLL it imports may be missing from the "
               "same directory)"
             : " (the file does not exist)");
  raise_load_failure(pdli, err);
}

// A bundle: every CUDA DLL from `c.dir`, failures are fatal.
HMODULE load_from_bundle(PDelayLoadInfo pdli, const RuntimeDirChoice& c) {
  const BundleListing& listing = bundle_listing(c.dir);
  if (!listing.missing.empty()) {
    fail_missing_preloads(pdli, c, listing.missing);
  }
  if (const PreloadFailure& f = preload_core(c.dir); f.failed()) {
    fail_load(pdli, c, f.path, f.error);
  }
  if (starts_with_ci(pdli->szDll, "cudnn")) {
    if (const PreloadFailure& f = preload_cudnn(c.dir); f.failed()) {
      fail_load(pdli, c, f.path, f.error);
    }
  }
  fs::path path = c.dir / pdli->szDll;
  HMODULE mod = load_beside(path);
  if (!mod) {
    fail_load(pdli, c, path, ::GetLastError());
  }
  return mod;
}

// The build-time lookup failed. When the executable's directory holds CUDA
// DLLs but not the marker DLL, it is a bundle missing that file: name the
// expected path. The helper still falls back to its own LoadLibrary, as
// before, so a developer setup with the toolkit on PATH is unaffected.
void report_unmarked_bundle(const char* dll, const RuntimeDirChoice& c) {
  const fs::path& exe = binary_dir();
  if (exe.empty()) {
    return;
  }
  std::vector<std::string> names;
  std::error_code ec;
  for (auto it = fs::directory_iterator(exe, ec);
       !ec && it != fs::directory_iterator();
       it.increment(ec)) {
    names.push_back(wide_to_utf8(it->path().filename().native()));
  }
  if (!looks_like_runtime_bundle(names)) {
    return;
  }
  std::fprintf(
      stderr,
      "mlxcel: cannot load %s from the %s (%s). %s holds CUDA DLLs but "
      "not %s, so it was not used as a CUDA runtime bundle; restore %s.\n",
      dll,
      runtime_dir_source_name(c.source),
      to_utf8(c.dir).c_str(),
      to_utf8(exe).c_str(),
      runtime_marker().c_str(),
      to_utf8(exe / runtime_marker()).c_str());
  std::fflush(stderr);
}

} // namespace

// Called by system before loading a lazy-loaded DLL.
FARPROC WINAPI delayload_helper(unsigned dliNotify, PDelayLoadInfo pdli) {
  if (dliNotify != dliNotePreLoadLibrary || !is_cuda_dll(pdli->szDll)) {
    return nullptr;
  }
  const RuntimeDirChoice& choice = runtime_choice();
  if (choice.error != RuntimeDirError::None) {
    fail_choice(pdli, choice);
  }
  if (is_runtime_bundle(choice.source)) {
    return reinterpret_cast<FARPROC>(load_from_bundle(pdli, choice));
  }
  HMODULE mod = load_from_build_layout(pdli->szDll);
  if (!mod) {
    report_unmarked_bundle(pdli->szDll, choice);
  }
  return reinterpret_cast<FARPROC>(mod);
}

const fs::path& cuda_runtime_dir() {
  static const fs::path empty;
  const RuntimeDirChoice& choice = runtime_choice();
  return choice.error == RuntimeDirError::None ? choice.dir : empty;
}

RuntimeDirSource cuda_runtime_dir_source() {
  const RuntimeDirChoice& choice = runtime_choice();
  return choice.error == RuntimeDirError::None ? choice.source
                                               : RuntimeDirSource::None;
}

} // namespace mlx::core::cu

extern "C" const PfnDliHook __pfnDliNotifyHook2 =
    mlx::core::cu::delayload_helper;
