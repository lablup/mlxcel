// Copyright © 2025 Apple Inc.
// Modified from upstream MLX 7ca40033132dae444d45f7fc3daae57e24c13f6c mlx/backend/cuda/jit_module.cpp
// Modified by mlxcel (rebased onto 9a795735 in #1198):
// - CCCL include resolution honors the MLXCEL_CCCL_DIR override and resolves
//   the executable directory from /proc/self/exe, so the bundled headers are
//   found regardless of how the process is launched (a relative ./mlxcel from
//   inside bin/ otherwise yields a relative dli_fname from dladdr and the
//   lookup misses). Upstream's cccl_dir() (dirs.cpp, MLX_CCCL_DIR) remains the
//   final fallback.
// - A one-time notice when CUDA kernels are JIT-compiled on a cold cache, so
//   the slower first response is not a mystery (suppress with MLXCEL_QUIET_JIT).
// - Custom kernels (mx.fast.cuda_kernel) are written to the disk cache like the
//   built-in ones, and cache files are published by rename so a concurrent
//   reader never loads a partial file (issue #2492). The cache heals itself:
//   an empty or unparseable entry is a miss, and an entry that fails to load
//   is discarded, recompiled and republished once.
// - Windows CUDA header lookup (#58): a missing default toolkit directory is
//   "not found" instead of an uncaught directory_iterator exception, and the
//   build-time toolkit include dir (MLXCEL_CUDA_INCLUDE_DIR) is tried after
//   CUDA_HOME / CUDA_PATH, so a dev binary built against a toolkit outside
//   Program Files runs without either variable set.
// - Windows CUDA runtime bundle (lablup/mlxcel#2640): when delayload.cpp
//   chose a bundle directory (MLXCEL_CUDA_RUNTIME_DIR or the executable's
//   directory, runtime_dir.h), the include/ in or beside it is added to the
//   NVRTC include path (CUTLASS/CuTe), its cccl/ comes before the build-time
//   CCCL, and its CUDA headers come after CUDA_HOME but before CUDA_PATH,
//   which the CUDA installer sets machine-wide (choose_cuda_include_dir()).
//   With CUDA_HOME set the bundle's include/ is the last include path, so
//   CUDA_HOME's CUDA headers are the ones NVRTC finds.
// - Disk cache trust and keying (lablup/mlxcel#2539, POSIX parts only): the
//   cache directory is used only when every component from the leaf up to /
//   is owned by the current user or by root and is not group- or
//   other-writable (a root-owned sticky directory such as /tmp excepted), and
//   one notice says when it is refused. The temp fallback is a per-user
//   $TMPDIR/mlx-<euid>/ root, entries are keyed by an sm<cc>-nvrtc<version>
//   subdirectory, stale publish temporaries are swept, and temporaries are
//   opened with O_CREAT|O_EXCL|O_NOFOLLOW so a planted symlink is never
//   written through.

#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/runtime_dir.h"
#include "mlx/version.h"

#include "cuda_jit_sources.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>

#if defined(_WIN32)
#include <process.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <fmt/format.h>
#include <nvrtc.h>

namespace mlx::core::cu {

// Defined in dirs.cpp to avoid invalidating compile cache.
const char* cccl_dir();

namespace {

#define CHECK_NVRTC_ERROR(cmd) check_nvrtc_error(#cmd, (cmd))

void check_nvrtc_error(const char* name, nvrtcResult err) {
  if (err != NVRTC_SUCCESS) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, nvrtcGetErrorString(err)));
  }
}

// Return the default path to CUDA toolkit.
const std::filesystem::path& default_cuda_toolkit_path() {
#if defined(_WIN32)
  static auto cached_path = []() -> std::filesystem::path {
    std::filesystem::path root(
        LR"(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA)");
    // No installer toolkit (for example a redistributable toolkit unpacked
    // elsewhere) is "not found", not an exception out of the JIT.
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator(root, ec);
         !ec && it != std::filesystem::directory_iterator();
         it.increment(ec)) {
      if (std::filesystem::exists(it->path() / "include" / "cuda.h")) {
        return it->path();
      }
    }
    return {};
  }();
#else
  static std::filesystem::path cached_path = "/usr/local/cuda";
#endif
  return cached_path;
}

// Get the dirname of nvidia python package that contains CUDA headers.
inline const char* cudart_dirname() {
#if CUDART_VERSION == 0
#error "CUDART_VERSION not defined"
#elif CUDART_VERSION < 13000
  return "cuda_runtime";
#elif CUDART_VERSION < 14000
  return "cu13";
#else
  return "cu14";
#endif
}

// The headers of the CUDA runtime bundle delayload.cpp loaded the DLLs from:
// <runtime dir>/include, or the include/ beside it (the bundle keeps bin/ and
// include/ under one root), whichever holds cuda_runtime_api.h. Empty when the
// runtime directory is not a bundle or ships no headers. Windows only.
std::filesystem::path bundle_include_dir() {
#if defined(_WIN32)
  if (!is_runtime_bundle(cuda_runtime_dir_source())) {
    return {};
  }
  const std::filesystem::path& runtime = cuda_runtime_dir();
  for (const std::filesystem::path& include :
       {runtime / "include", runtime.parent_path() / "include"}) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(
            include / "cuda_runtime_api.h", ec)) {
      return include;
    }
  }
#endif
  return {};
}

// `path` for an NVRTC argument, or "" when it has no narrow spelling (a
// character outside the ANSI code page), instead of an exception out of the
// JIT.
std::string narrow_path(const std::filesystem::path& path) {
  try {
    return path.string();
  } catch (const std::exception&) {
    return "";
  }
}

// Return the --include-path args used for invoking NVRTC.
const std::vector<std::string>& include_path_args() {
  static std::vector<std::string> cached_args = []() {
    std::vector<std::string> args;
    // Resolve the directory of the running executable. MLX's current_binary_dir
    // uses dladdr, whose dli_fname can be relative when the program is launched
    // by a relative argv[0] (e.g. ./mlxcel from inside bin/), which breaks the
    // bundled-header lookup below. Prefer the canonical absolute path from
    // /proc/self/exe on Linux so resolution does not depend on the launch path.
    std::filesystem::path root_dir;
#if defined(__linux__)
    {
      std::error_code exe_ec;
      auto exe_path = std::filesystem::read_symlink("/proc/self/exe", exe_ec);
      root_dir = exe_ec ? current_binary_dir() : exe_path.parent_path();
    }
#else
    root_dir = current_binary_dir();
#endif
#if !defined(_WIN32)
    root_dir = root_dir.parent_path();
#endif
    // Add path to bundled headers.
    auto path = root_dir / "include";
    if (std::filesystem::exists(path)) {
      args.push_back(fmt::format("--include-path={}", path.string()));
    }
    // A Windows runtime bundle's include/ (CUTLASS/CuTe and the CUDA
    // headers), when it is not the directory just added. NVRTC takes each
    // header from the first include path that holds it, so with CUDA_HOME
    // set, which outranks a bundle's CUDA headers (choose_cuda_include_dir()),
    // the bundle's include/ goes last instead, after CUDA_HOME's include/,
    // and supplies only what that lacks (CUTLASS/CuTe).
    const std::string bundle_include = narrow_path(bundle_include_dir());
    const char* cuda_home = std::getenv("CUDA_HOME");
    const bool bundle_include_last =
        !bundle_include.empty() && cuda_home && *cuda_home;
    const bool bundle_include_added = !bundle_include.empty() &&
        !bundle_include_last && bundle_include != path.string();
    if (bundle_include_added) {
      args.push_back(fmt::format("--include-path={}", bundle_include));
    }
    // CCCL headers: an explicit MLXCEL_CCCL_DIR override wins (it lets an
    // embedder such as Backend.AI:GO point at the headers regardless of the
    // on-disk layout), then the bundled layout next to the executable
    // (bin/ + include/cccl/), then upstream's cccl_dir() fallback compiled
    // into dirs.cpp via MLX_CCCL_DIR.
    if (const char* cccl_override = std::getenv("MLXCEL_CCCL_DIR");
        cccl_override != nullptr && *cccl_override != '\0' &&
        std::filesystem::exists(cccl_override)) {
      path = cccl_override;
    } else {
      path = path / "cccl";
      if (!std::filesystem::exists(path) && !bundle_include.empty() &&
          std::filesystem::exists(
              std::filesystem::path(bundle_include) / "cccl")) {
        path = std::filesystem::path(bundle_include) / "cccl";
      }
      if (!std::filesystem::exists(path) && cccl_dir()) {
        path = cccl_dir();
      }
    }
    if (std::filesystem::exists(path)) {
      args.push_back(fmt::format("--include-path={}", path.string()));
    }
    // CUTLASS/CuTe headers for the runtime-JIT'd quantized kernels (upstream
    // #3706/#3576 compile qmm_sm80/sm90/naive and gather_gemm with NVRTC, and
    // those kernels include <cute/...> / <cutlass/...>). Installed layouts are
    // already covered by the bundled include/ path added above (upstream
    // installs cute/ and cutlass/ directly into include/); here an explicit
    // MLXCEL_CUTLASS_DIR override wins, then the build-tree fallback compiled
    // in via MLX_CUTLASS_DIR covers uninstalled dev binaries.
    if (const char* cutlass_override = std::getenv("MLXCEL_CUTLASS_DIR");
        cutlass_override != nullptr && *cutlass_override != '\0' &&
        std::filesystem::exists(cutlass_override)) {
      args.push_back(fmt::format("--include-path={}", cutlass_override));
    }
#if defined(MLX_CUTLASS_DIR)
    else if (std::filesystem::exists(MLX_CUTLASS_DIR)) {
      args.push_back(fmt::format("--include-path={}", MLX_CUTLASS_DIR));
    }
#endif
    // Add path to CUDA runtime headers, try local-installed python package
    // first and then system-installed headers.
    path = root_dir.parent_path() / "nvidia" / cudart_dirname() / "include";
    if (!std::filesystem::exists(path)) {
      // CUDA_HOME, a runtime bundle's include/, CUDA_PATH, the toolkit this
      // binary was built against (Windows, #58; ahead of the Program Files
      // scan, so Linux keeps /usr/local/cuda), then the default toolkit.
#if defined(_WIN32) && defined(MLXCEL_CUDA_INCLUDE_DIR)
      const char* baked = MLXCEL_CUDA_INCLUDE_DIR;
      bool baked_exists = std::filesystem::exists(baked);
#else
      const char* baked = nullptr;
      bool baked_exists = false;
#endif
      const std::filesystem::path& default_root = default_cuda_toolkit_path();
      std::string default_include =
          default_root.empty() ? "" : narrow_path(default_root / "include");
      path = choose_cuda_include_dir(
          cuda_home,
          std::getenv("CUDA_PATH"),
          bundle_include,
          !bundle_include.empty(),
          baked,
          baked_exists,
          default_include);
      if (path.empty() || !std::filesystem::exists(path)) {
        throw std::runtime_error(
            "Can not find locations of CUDA headers, please set environment "
            "variable CUDA_HOME or CUDA_PATH.");
      }
    }
    if (!bundle_include_added || path.string() != bundle_include) {
      args.push_back(fmt::format("--include-path={}", path.string()));
    }
    if (bundle_include_last && path.string() != bundle_include) {
      args.push_back(fmt::format("--include-path={}", bundle_include));
    }
    return args;
  }();
  return cached_args;
}

// [mlxcel #2539] Say once per process that the disk cache is off and why, so
// a user can see why every run compiles again and fix the directory. Not
// suppressed by MLXCEL_QUIET_JIT: a refused cache is a configuration problem,
// not progress noise.
void report_cache_disabled(const std::string& detail) {
  static std::once_flag once;
  std::call_once(once, [&]() {
    std::fprintf(
        stderr, "mlxcel: CUDA kernel disk cache disabled: %s\n", detail.c_str());
    std::fflush(stderr);
  });
}

#if !defined(_WIN32)
// [mlxcel #2539] Create each missing component of the absolute |dir| with mode
// 0700. Existing components are left as they are, never chmod-ed:
// cache_dir_is_trusted decides whether they are acceptable. False when a
// component can neither be created nor found as a directory.
bool make_private_directories(const std::filesystem::path& dir) {
  std::filesystem::path partial;
  for (const auto& part : dir) {
    partial /= part;
    if (::mkdir(partial.c_str(), 0700) == 0 || errno == EEXIST) {
      continue;
    }
    struct stat st;
    if (::stat(partial.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
      return false;
    }
  }
  return true;
}

// [mlxcel #2539] Whether |dir| is safe to load compiled modules from. On a
// multi-user host, whoever can write the cache directory, or replace it
// through a parent they can write, can plant modules under derivable names
// that load into this process. So every component of the canonical path,
// from the leaf up to /, must be owned by the current user or by root and
// must not be group- or other-writable, unless it is root-owned with the
// sticky bit set (/tmp). Trusting the leaf alone is not enough: the owner of
// a parent can rename the leaf away and put their own in its place. On
// success |resolved| holds the canonical path, which callers use from then on
// so a symlink component swapped in after the check cannot redirect them. On
// failure |reason| names the offending component.
bool cache_dir_is_trusted(
    const std::filesystem::path& dir,
    std::filesystem::path& resolved,
    std::string& reason) {
  std::error_code error;
  auto canonical = std::filesystem::canonical(dir, error);
  if (error) {
    reason = fmt::format("{} cannot be resolved ({})", dir.string(), error.message());
    return false;
  }
  const uid_t euid = ::geteuid();
  bool leaf = true;
  for (auto component = canonical;; component = component.parent_path()) {
    struct stat st;
    if (::lstat(component.c_str(), &st) != 0) {
      reason = fmt::format("{} cannot be inspected", component.string());
      return false;
    }
    // canonical() resolved every link; one here was swapped in since.
    if (S_ISLNK(st.st_mode)) {
      reason = fmt::format("{} is a symbolic link", component.string());
      return false;
    }
    if (leaf && !S_ISDIR(st.st_mode)) {
      reason = fmt::format("{} is not a directory", component.string());
      return false;
    }
    if (st.st_uid != euid && st.st_uid != 0) {
      reason = fmt::format(
          "{} is not owned by the current user", component.string());
      return false;
    }
    const bool shared_write = (st.st_mode & (S_IWGRP | S_IWOTH)) != 0;
    const bool root_sticky = st.st_uid == 0 && (st.st_mode & S_ISVTX) != 0;
    if (shared_write && !root_sticky) {
      reason = fmt::format(
          "{} is group- or other-writable", component.string());
      return false;
    }
    leaf = false;
    if (component == component.root_path()) {
      break;
    }
  }
  resolved = std::move(canonical);
  return true;
}
#endif

// Get the cache directory for storing compiled results.
const std::filesystem::path& ptx_cache_dir() {
  static std::filesystem::path cache = []() -> std::filesystem::path {
    std::filesystem::path cache;
    if (auto c = std::getenv("MLX_PTX_CACHE_DIR"); c) {
      cache = c;
    } else {
#if defined(_WIN32)
      cache =
          std::filesystem::temp_directory_path() / "mlx" / version() / "ptx";
#else
      // [mlxcel #2539] A per-user root. Upstream's shared <tmp>/mlx was owned
      // by whoever created it first, who could then plant modules for
      // everyone else.
      std::error_code tmp_error;
      auto tmp = std::filesystem::temp_directory_path(tmp_error);
      if (tmp_error) {
        return std::filesystem::path();
      }
      cache = tmp / ("mlx-" + std::to_string(::geteuid())) / version() / "ptx";
#endif
    }

#if defined(_WIN32)
    // Add "\\?\" prefix to support long file path.
    const wchar_t* long_path_prefix = L"\\\\?\\";
    if (cache.is_relative()) {
      cache = std::filesystem::absolute(cache);
    }
    if (!cache.native().starts_with(long_path_prefix)) {
      cache = long_path_prefix + cache.native();
    }

    if (!std::filesystem::exists(cache)) {
      std::error_code error;
      if (!std::filesystem::create_directories(cache, error)) {
        return std::filesystem::path();
      }
    }
    return cache;
#else
    // [mlxcel #2539] Missing components are created private, and the result
    // is used only if it passes the trust check, whatever its source
    // (MLX_PTX_CACHE_DIR, the persistent default mlxcel's binaries set
    // through it, or the temp fallback). A refused or uncreatable directory
    // turns the cache off, which read_cached_ptx and write_cached_ptx honor;
    // compilation still works.
    if (cache.empty()) {
      return std::filesystem::path();
    }
    if (cache.is_relative()) {
      std::error_code abs_error;
      cache = std::filesystem::absolute(cache, abs_error);
      if (abs_error) {
        return std::filesystem::path();
      }
    }
    if (!make_private_directories(cache)) {
      return std::filesystem::path();
    }
    std::filesystem::path trusted;
    std::string reason;
    if (!cache_dir_is_trusted(cache, trusted, reason)) {
      report_cache_disabled(reason);
      return std::filesystem::path();
    }
    return trusted;
#endif
  }();
  return cache;
}

std::filesystem::path get_ptx_path(
    const std::filesystem::path& cache_dir,
    const std::string& module_name) {
  constexpr int max_file_name_length = 245;
  if (module_name.size() <= max_file_name_length) {
    return cache_dir / (module_name + ".ptx");
  }

  auto ptx_path = cache_dir;
  int offset = 0;
  while (module_name.size() - offset > max_file_name_length) {
    ptx_path /= module_name.substr(offset, max_file_name_length);
    offset += max_file_name_length;
  }
  ptx_path /= module_name.substr(offset) + ".ptx";

  return ptx_path;
}

// Try to read the cached |ptx| and |ptx_kernels| from |cache_dir|.
bool read_cached_ptx(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    std::string& ptx,
    std::vector<std::pair<std::string, std::string>>& ptx_kernels) {
  if (cache_dir.empty()) {
    return false;
  }

  auto ptx_path = get_ptx_path(cache_dir, module_name);
  std::error_code error;
  auto ptx_size = std::filesystem::file_size(ptx_path, error);
  if (error) {
    return false;
  }
  std::ifstream ptx_file(ptx_path, std::ios::binary);
  if (!ptx_file.good()) {
    return false;
  }
  // [mlxcel #2492] An entry that cannot be trusted is a miss, so the caller
  // recompiles and republishes it. Upstream accepted an empty .ptx or an empty
  // kernel map, which fails at load time on every launch; with custom kernels
  // now cached, that would break decode permanently.
  if (ptx_size == 0) {
    return false;
  }
  ptx.resize(ptx_size);
  ptx_file.read(ptx.data(), ptx_size);
  if (static_cast<std::uintmax_t>(ptx_file.gcount()) != ptx_size) {
    ptx.clear();
    return false;
  }

  std::ifstream txt_file(ptx_path.replace_extension(".txt"), std::ios::binary);
  std::string line;
  bool valid = true;
  while (std::getline(txt_file, line)) {
    auto tab = line.find('\t');
    if (tab == std::string::npos || tab == 0 || tab + 1 >= line.size()) {
      valid = false;
      break;
    }
    ptx_kernels.emplace_back(line.substr(0, tab), line.substr(tab + 1));
  }
  if (!valid || ptx_kernels.empty()) {
    ptx.clear();
    ptx_kernels.clear();
    return false;
  }
  return true;
}

// [mlxcel #2539] Whether |s| has the form of a publish temporary's suffix,
// `.tmp<hex>_<hex>` (process id and clock tick, see write_cached_ptx).
bool is_temp_suffix(std::string_view s) {
  constexpr std::string_view prefix = ".tmp";
  if (!s.starts_with(prefix)) {
    return false;
  }
  s.remove_prefix(prefix.size());
  auto underscore = s.find('_');
  if (underscore == std::string_view::npos) {
    return false;
  }
  auto is_hex = [](std::string_view v) {
    return !v.empty() && std::all_of(v.begin(), v.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
  };
  return is_hex(s.substr(0, underscore)) && is_hex(s.substr(underscore + 1));
}

// [mlxcel #2539] Whether |name| is a publish temporary: the suffix followed
// by the extension of the file it publishes (.txt, .ptx, or the .cu source).
bool is_publish_temporary(std::string_view name) {
  for (std::string_view ext : {".ptx", ".txt", ".cu"}) {
    if (name.size() > ext.size() && name.ends_with(ext) &&
        is_temp_suffix(name.substr(0, name.size() - ext.size()))) {
      return true;
    }
  }
  return false;
}

// [mlxcel #2539] Test-only hook: MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX replaces the
// generated temporary suffix, so a test can plant a file or symlink at the
// exact path a publish will open. Honored only when it has the generated form
// `.tmp<hex>_<hex>`, so it can never carry a path separator; any other value
// is ignored. Empty when unset or ignored.
const std::string& test_temp_suffix() {
  static const std::string suffix = []() -> std::string {
    const char* value = std::getenv("MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX");
    if (value != nullptr && is_temp_suffix(value)) {
      return value;
    }
    return {};
  }();
  return suffix;
}

#if !defined(_WIN32)
// Write all of |data| to |fd|, retrying short writes and EINTR.
bool write_all(int fd, const char* data, size_t size) {
  while (size > 0) {
    ssize_t n = ::write(fd, data, size);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (n == 0) {
      return false;
    }
    data += n;
    size -= static_cast<size_t>(n);
  }
  return true;
}
#endif

// Write the |ptx| and |ptx_kernels| to |cache_dir| with |name|.
void write_cached_ptx(
    const std::filesystem::path& cache_dir,
    const std::string& module_name,
    const std::string& ptx,
    const std::vector<std::pair<std::string, std::string>>& ptx_kernels,
    const std::string& source_code) {
  if (cache_dir.empty()) {
    return;
  }

  auto ptx_path = get_ptx_path(cache_dir, module_name);

  // Ensure that the directory exists. A failure only leaves the cache cold.
  auto parent = ptx_path.parent_path();
  if (parent != cache_dir) {
#if defined(_WIN32)
    std::error_code dir_error;
    std::filesystem::create_directories(parent, dir_error);
    if (dir_error) {
      return;
    }
#else
    // [mlxcel #2539] Nested directories (module names over 245 characters)
    // are private like the cache directory itself.
    if (!make_private_directories(parent)) {
      return;
    }
#endif
  }

  // [mlxcel #2492] Publish each file by writing a process-unique temporary and
  // renaming it into place. Upstream wrote the final paths directly, so a
  // second process (a parallel test binary, a server next to a CLI run) could
  // read a truncated .ptx and fail in cuModuleLoadDataEx. The .txt kernel map
  // is published before the .ptx, and the .ptx only when the map made it,
  // because read_cached_ptx keys the hit on the .ptx: once it exists, its map
  // is complete. Each temporary is closed before its state is checked, so a
  // failed final flush (a full disk) is caught before the rename. A failed
  // write or rename only leaves the cache cold, which the next process
  // repairs by recompiling. Within a process get_jit_module serializes
  // writers, so the process id plus a clock tick names the temporary.
#if defined(_WIN32)
  const auto process_id = static_cast<unsigned long long>(_getpid());
#else
  const auto process_id = static_cast<unsigned long long>(getpid());
#endif
  const auto unique_suffix = !test_temp_suffix().empty()
      ? test_temp_suffix()
      : fmt::format(
            ".tmp{:x}_{:x}",
            process_id,
            static_cast<unsigned long long>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
  // [mlxcel #2423] The temporary is named from the suffix and the extension
  // only, not appended to the final name: `get_ptx_path` keeps a module name
  // of up to 245 characters in one path component, so `<name>.txt` plus the
  // suffix could pass the 255-byte NAME_MAX. That open failed with
  // ENAMETOOLONG, the entry was never published, and every process
  // NVRTC-compiled such a kernel again (a 234-character fused tanh-GELU
  // kernel cost about 1.7 s per process). The rename stays within one
  // directory, so publishing is still atomic.
  auto publish = [&](const std::filesystem::path& path,
                     const char* data,
                     size_t size) -> bool {
    auto tmp = path.parent_path() /
        (unique_suffix + path.extension().string());
    bool written = false;
#if defined(_WIN32)
    {
      std::ofstream file(tmp, std::ios::binary);
      if (size > 0) {
        file.write(data, size);
      }
      file.close();
      written = !file.fail();
    }
#else
    // [mlxcel #2539] Create the temporary exclusively and without following
    // a symlink. Upstream's std::ofstream opened an existing path, so a
    // symlink planted at a predicted temporary name was written through,
    // to wherever it pointed. Any failure here (EEXIST from a planted file
    // or symlink, ELOOP) leaves this entry unpublished and the cache cold
    // for this kernel, whose compiled code is still used; the path is not
    // ours, so it is not removed either.
    int fd = ::open(
        tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {
      return false;
    }
    written = write_all(fd, data, size);
    if (::close(fd) != 0) {
      written = false;
    }
#endif
    std::error_code error;
    if (written) {
      std::filesystem::rename(tmp, path, error);
      if (!error) {
        return true;
      }
    }
    std::filesystem::remove(tmp, error);
    return false;
  };

  // Write the mangled names, then the compiled code
  std::string kernel_map;
  for (const auto& [name, mangled] : ptx_kernels) {
    kernel_map += name;
    kernel_map += '\t';
    kernel_map += mangled;
    kernel_map += '\n';
  }
  auto txt_path = ptx_path;
  txt_path.replace_extension(".txt");
  if (!publish(txt_path, kernel_map.data(), kernel_map.size()) ||
      !publish(ptx_path, ptx.data(), ptx.size())) {
    return;
  }

  // Write the generated code
  auto source_path = ptx_path;
  source_path.replace_extension(".cu");
  publish(source_path, source_code.data(), source_code.size());
}

// [mlxcel #2539] A live publish renames its temporary within milliseconds, so
// one older than this was orphaned by a process killed mid-publish.
constexpr auto kStaleTempAge = std::chrono::hours(1);

// [mlxcel #2539] Remove orphaned publish temporaries from the top level of
// |dir|. Only regular files are removed, judged without following symlinks;
// errors are ignored (the sweep is housekeeping, not correctness).
void sweep_stale_temporaries(const std::filesystem::path& dir) {
  std::error_code error;
  const auto now = std::filesystem::file_time_type::clock::now();
  for (std::filesystem::directory_iterator it(dir, error), end;
       !error && it != end;
       it.increment(error)) {
    const auto& entry = *it;
    if (!is_publish_temporary(entry.path().filename().string())) {
      continue;
    }
    std::error_code entry_error;
    auto status = entry.symlink_status(entry_error);
    if (entry_error || !std::filesystem::is_regular_file(status)) {
      continue;
    }
    auto mtime = std::filesystem::last_write_time(entry.path(), entry_error);
    if (entry_error || now - mtime < kStaleTempAge) {
      continue;
    }
    std::filesystem::remove(entry.path(), entry_error);
  }
}

// [mlxcel #2539] The cache directory for |device|'s modules:
// <ptx_cache_dir()>/sm<cc major><cc minor>-nvrtc<major>.<minor>. compile()
// targets the device's architecture and may emit SASS, so a module built for
// one device or compiler must not be found by another; upstream's flat
// directory shared one name across both. The directory is created private
// and trust-checked like the root, and stale temporaries in it are swept,
// once per process; empty (cache off) when the root is off or either step
// fails. Entries from the flat layout are no longer read.
std::filesystem::path device_cache_dir(Device& device) {
  const auto& root = ptx_cache_dir();
  if (root.empty()) {
    return {};
  }
  int nvrtc_major = 0;
  int nvrtc_minor = 0;
  if (nvrtcVersion(&nvrtc_major, &nvrtc_minor) != NVRTC_SUCCESS) {
    return {};
  }
  const auto key = fmt::format(
      "sm{}{}-nvrtc{}.{}",
      device.compute_capability_major(),
      device.compute_capability_minor(),
      nvrtc_major,
      nvrtc_minor);

  // Leaked like get_jit_module's cache: modules may still be built during
  // static teardown.
  static auto* mtx = new std::mutex;
  static auto* resolved =
      new std::unordered_map<std::string, std::filesystem::path>;
  std::lock_guard<std::mutex> lock(*mtx);
  if (auto it = resolved->find(key); it != resolved->end()) {
    return it->second;
  }

  auto dir = root / key;
#if defined(_WIN32)
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  if (!std::filesystem::is_directory(dir, error)) {
    dir.clear();
  }
#else
  if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
    dir.clear();
  } else {
    std::filesystem::path trusted;
    std::string reason;
    if (cache_dir_is_trusted(dir, trusted, reason)) {
      dir = std::move(trusted);
    } else {
      report_cache_disabled(reason);
      dir.clear();
    }
  }
#endif
  if (!dir.empty()) {
    sweep_stale_temporaries(dir);
  }
  resolved->emplace(key, dir);
  return dir;
}

// Return if |device|'s version is not newer than |major|.|minor| version.
inline bool version_lower_equal(Device& device, int major, int minor) {
  if (device.compute_capability_major() < major) {
    return true;
  } else if (device.compute_capability_major() == major) {
    return device.compute_capability_minor() <= minor;
  } else {
    return false;
  }
}

// Return whether NVRTC supports compiling to |device|'s SASS code.
bool compiler_supports_device_sass(Device& device) {
  int nvrtc_major, nvrtc_minor;
  CHECK_NVRTC_ERROR(nvrtcVersion(&nvrtc_major, &nvrtc_minor));
  if (nvrtc_major < 9) {
    return false;
  } else if (nvrtc_major == 9) {
    return version_lower_equal(device, 7, 2);
  } else if (nvrtc_major == 10) {
    return version_lower_equal(device, 7, 5);
  } else if (nvrtc_major == 11 && nvrtc_minor == 0) {
    return version_lower_equal(device, 8, 0);
  } else if (nvrtc_major == 11 && nvrtc_minor < 8) {
    return version_lower_equal(device, 8, 6);
  } else {
    return true;
  }
}

#define INCLUDE_PREFIX "mlx/backend/cuda/device/"

constexpr const char* g_include_names[] = {
    INCLUDE_PREFIX "atomic_ops.cuh",
    INCLUDE_PREFIX "binary_ops.cuh",
    INCLUDE_PREFIX "cast_op.cuh",
    INCLUDE_PREFIX "config.h",
    INCLUDE_PREFIX "complex.cuh",
    INCLUDE_PREFIX "cute_dequant.cuh",
    INCLUDE_PREFIX "fp16_math.cuh",
    INCLUDE_PREFIX "gemm_sm70.cuh",
    INCLUDE_PREFIX "hadamard.cuh",
    INCLUDE_PREFIX "indexing.cuh",
    INCLUDE_PREFIX "qmm_naive.cuh",
    INCLUDE_PREFIX "qmm_sm80.cuh",
    INCLUDE_PREFIX "qmm_sm90.cuh",
    INCLUDE_PREFIX "scatter_ops.cuh",
    INCLUDE_PREFIX "unary_ops.cuh",
    INCLUDE_PREFIX "ternary_ops.cuh",
    INCLUDE_PREFIX "utils.cuh",
};

#undef INCLUDE_PREFIX

constexpr const char* g_headers[] = {
    jit_source_atomic_ops,
    jit_source_binary_ops,
    jit_source_cast_op,
    jit_source_config,
    jit_source_complex,
    jit_source_cute_dequant,
    jit_source_fp16_math,
    jit_source_gemm_sm70,
    jit_source_hadamard,
    jit_source_indexing,
    jit_source_qmm_naive,
    jit_source_qmm_sm80,
    jit_source_qmm_sm90,
    jit_source_scatter_ops,
    jit_source_unary_ops,
    jit_source_ternary_ops,
    jit_source_utils,
};

void compile(
    Device& device,
    const std::string& module_name,
    const std::string& source,
    const std::vector<std::string>& kernel_names,
    std::string& ptx,
    std::vector<std::pair<std::string, std::string>>& ptx_kernels) {
  // Create the program
  nvrtcProgram prog;
  CHECK_NVRTC_ERROR(nvrtcCreateProgram(
      &prog,
      source.c_str(),
      (module_name + ".cu").c_str(),
      std::size(g_headers),
      g_headers,
      g_include_names));
  std::unique_ptr<nvrtcProgram, void (*)(nvrtcProgram*)> prog_freer(
      &prog,
      [](nvrtcProgram* p) { CHECK_NVRTC_ERROR(nvrtcDestroyProgram(p)); });
  for (const auto& name : kernel_names) {
    CHECK_NVRTC_ERROR(nvrtcAddNameExpression(prog, name.c_str()));
  }

  // Required for compiling CUTLASS code.
  std::vector<const char*> args;
  args.push_back("--device-as-default-execution-space");

  // Target current device.
  bool use_sass = compiler_supports_device_sass(device);
  auto cc = device.compute_capability_major();
  std::string arch_tag = (cc >= 9) ? "a" : "";
  std::string compute = fmt::format(
      "--gpu-architecture={}_{}{}{}",
      use_sass ? "sm" : "compute",
      cc,
      device.compute_capability_minor(),
      arch_tag);
  args.push_back(compute.c_str());
  for (const auto& include : include_path_args()) {
    args.push_back(include.c_str());
  }

  // Compile program.
  nvrtcResult compile_result =
      nvrtcCompileProgram(prog, args.size(), args.data());
  if (compile_result != NVRTC_SUCCESS) {
    size_t log_size;
    CHECK_NVRTC_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    std::vector<char> log(log_size + 1, 0);
    CHECK_NVRTC_ERROR(nvrtcGetProgramLog(prog, log.data()));
    throw std::runtime_error(
        fmt::format("Failed to compile kernel: {}.", log.data()));
  }

  // Get mangled names of kernel names.
  for (const auto& name : kernel_names) {
    const char* mangled;
    CHECK_NVRTC_ERROR(nvrtcGetLoweredName(prog, name.c_str(), &mangled));
    ptx_kernels.emplace_back(name, mangled);
  }

  // Get ptx data.
  size_t ptx_size;
  if (use_sass) {
    CHECK_NVRTC_ERROR(nvrtcGetCUBINSize(prog, &ptx_size));
  } else {
    CHECK_NVRTC_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
  }
  ptx.resize(ptx_size);
  if (use_sass) {
    CHECK_NVRTC_ERROR(nvrtcGetCUBIN(prog, ptx.data()));
  } else {
    CHECK_NVRTC_ERROR(nvrtcGetPTX(prog, ptx.data()));
  }
}

void load_module(
    const std::string& module_name,
    const std::string& ptx,
    const std::vector<std::pair<std::string, std::string>>& ptx_kernels,
    CUmodule& module_,
    std::unordered_map<std::string, std::tuple<CUfunction, bool, uint32_t>>&
        kernels) {
  // Load module.
  char jit_log[4089] = {};
  CUjit_option options[] = {
      CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* values[] = {jit_log, reinterpret_cast<void*>(std::size(jit_log) - 1)};
  CUresult jit_result = cuModuleLoadDataEx(
      &module_, ptx.data(), std::size(options), options, values);
  if (jit_result != CUDA_SUCCESS) {
    throw std::runtime_error(
        fmt::format(
            "Failed to load compiled {} kernel: {}.", module_name, jit_log));
  }

  // Load kernels.
  for (const auto& [name, mangled] : ptx_kernels) {
    CUfunction kernel;
    CHECK_CUDA_ERROR(cuModuleGetFunction(&kernel, module_, mangled.c_str()));
    kernels[name] = std::make_tuple(kernel, false, 0);
  }
}

} // namespace

JitModule::JitModule(
    Device& device,
    const std::string& module_name,
    const KernelBuilder& builder,
    bool use_disk_cache) {
  // Will hold the actual device executable source code and kernel names
  std::string ptx;
  std::vector<std::pair<std::string, std::string>> ptx_kernels;

  // [mlxcel #2539] Keyed by device and compiler, trust-checked; empty when
  // the disk cache is off.
  const std::filesystem::path cache_dir = device_cache_dir(device);

  // Compile from source (or take the precompiled blob) and publish the result
  // to the file cache.
  auto compile_and_publish = [&]() {
    ptx.clear();
    ptx_kernels.clear();
    auto [precompiled, source_code, kernel_names] = builder();

    // Get the PTX or cubin
    if (precompiled) {
      ptx = std::move(source_code);
      for (auto& name : kernel_names) {
        ptx_kernels.emplace_back(name, name);
      }
    } else {
      // The first NVRTC compile on a cold cache makes the first response
      // noticeably slower (gather/indexing kernels are JIT-built here). Emit a
      // one-time notice so the latency is not a mystery; a function-local static
      // initializes exactly once and is thread-safe. Suppress with
      // MLXCEL_QUIET_JIT.
      [[maybe_unused]] static const bool mlxcel_jit_notice = []() {
        if (std::getenv("MLXCEL_QUIET_JIT") == nullptr) {
          std::fputs(
              "Compiling CUDA kernels (first run on this host; cached for later runs)...\n",
              stderr);
          std::fflush(stderr);
        }
        return true;
      }();
      compile(device, module_name, source_code, kernel_names, ptx, ptx_kernels);
    }

    // Save them in the file cache for the next launch.
    //
    // [mlxcel #2492] Upstream skips the write when use_disk_cache is false,
    // which only custom_kernel.cpp passes, so every mx.fast.cuda_kernel was
    // NVRTC-compiled again in every process (the read above is attempted
    // either way, it just never hit). mlxcel's fused decode kernels (MoE,
    // SSM, ...) are custom kernels, and their first launch lands inside the
    // first decode step: on GB10 the fused MoE pair alone added about 1.2 s
    // to each `mlxcel generate`, which made granite-4.0-h decode at -n 256
    // read 20% slower with the kernel than without it although the kernel
    // is faster per token (docs/benchmark_results/
    // fused-moe-jit-cache-gb10-2026-10-22.md). The module name already keys
    // on the template-argument hash and on a hash of the full source
    // (custom_kernel.cpp), so a hit can only return code compiled from the
    // same source, which makes caching them as safe as caching the built-in
    // kernels. The parameter stays for upstream signature parity.
    (void)use_disk_cache;
    write_cached_ptx(cache_dir, module_name, ptx, ptx_kernels, source_code);
  };

  // Try to load them from the file cache
  const bool cache_hit =
      read_cached_ptx(cache_dir, module_name, ptx, ptx_kernels);
  if (!cache_hit) {
    compile_and_publish();
  }

  // Load the module
  if (!cache_hit) {
    load_module(module_name, ptx, ptx_kernels, module_, kernels_);
  } else {
    // [mlxcel #2492] A cached entry that fails to load (a truncated or
    // corrupt file, or code built for another device) is discarded and
    // rebuilt once, so a bad entry cannot break the kernel permanently. A
    // second failure takes the original error path.
    try {
      load_module(module_name, ptx, ptx_kernels, module_, kernels_);
    } catch (const std::exception&) {
      kernels_.clear();
      // A failure in cuModuleGetFunction leaves the module loaded; unload it
      // so the retry does not leak the first handle.
      if (module_ != nullptr) {
        cuModuleUnload(module_);
        module_ = nullptr;
      }
      auto bad_path = get_ptx_path(cache_dir, module_name);
      std::error_code remove_error;
      std::filesystem::remove(bad_path, remove_error);
      compile_and_publish();
      load_module(module_name, ptx, ptx_kernels, module_, kernels_);
    }
  }
}

JitModule::~JitModule() {
  CHECK_CUDA_ERROR(cuModuleUnload(module_));
}

std::pair<CUfunction, uint32_t> JitModule::get_kernel_and_dims(
    const std::string& kernel_name,
    std::function<void(CUfunction)> configure_kernel) {
  auto it = kernels_.find(kernel_name);
  if (it == kernels_.end()) {
    throw std::runtime_error(
        fmt::format("There is no kernel named {}.", kernel_name));
  }

  // [mlxcel #1566] Serialize the lazy fill below. Upstream published the
  // configured flag before max_block_dim, so a second thread could observe the
  // flag set, skip the block, and read max_block_dim while it was still zero.
  // That zero reaches get_launch_args as max_block_dim, which computes
  // block_dim = min(max_block_dim, nthreads) = 0 and then divides nthreads by
  // it, raising SIGFPE. Reproduced on a Tesla V100 with
  // `mlxcel_core ffi_tests::compiled --test-threads=4`, which dies in the grid
  // computation for a compiled kernel; the same fault predates this repository's
  // Volta work and is upstream's, not the overlay's.
  //
  // The lock is held across configure_kernel, which is safe: every call site
  // passes a lambda that only calls cuFuncSetAttribute and never re-enters this
  // function. kernels_.find above needs no lock because the map is filled once
  // in the constructor, under the write lock in get_jit_module, and is never
  // restructured afterwards; only the two mapped fields are mutated here.
  std::lock_guard<std::mutex> lock(kernels_mtx_);

  // If it is the first time we run this kernel then configure it. Do it only
  // once!
  auto kernel = std::get<0>(it->second);
  if (!std::get<1>(it->second)) {
    if (configure_kernel) {
      configure_kernel(kernel);
    }
    // Publish max_block_dim before the flag, so that even without the lock a
    // reader that sees the flag set cannot observe a zero dimension.
    std::get<2>(it->second) = max_occupancy_block_dim(kernel);
    std::get<1>(it->second) = true;
  }

  return {kernel, std::get<2>(it->second)};
}

CUfunction JitModule::get_kernel(
    const std::string& kernel_name,
    std::function<void(CUfunction)> configure_kernel) {
  return get_kernel_and_dims(kernel_name, std::move(configure_kernel)).first;
}

JitModule& get_jit_module(
    Device& device,
    const std::string& name,
    const KernelBuilder& builder,
    bool use_disk_cache) {
  // The cache are leak intentionally as user code may still be running JIT
  // compiled code after main thread teardown.
  static auto* cache = new std::unordered_map<std::string, JitModule>;
  static auto* mtx = new std::shared_mutex;

  {
    std::shared_lock rlock(*mtx);
    if (auto it = cache->find(name); it != cache->end()) {
      return it->second;
    }
  }

  std::unique_lock wlock(*mtx);
  auto it = cache->find(name);
  if (it == cache->end()) {
    it = cache->try_emplace(name, device, name, builder, use_disk_cache).first;
  }
  return it->second;
}

} // namespace mlx::core::cu
