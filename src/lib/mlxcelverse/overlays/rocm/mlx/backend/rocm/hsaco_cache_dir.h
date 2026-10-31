// Copyright 2026 Lablup Inc.
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

// Location and trust of the JIT kernel disk cache (lablup/mlxcel#2641).
// jit_module.cpp loads what it finds in this cache into the GPU, so a directory
// another user can write is a code-planting path. The rules are the CUDA
// overlay's (make_owned_directory, cache_dir_is_trusted and
// read_trusted_cache_entry in its jit_module.cpp, lablup/mlxcel#2539 and #2765).
// They live in this header, which has no HIP dependency, so
// scripts/mlxcelverse/rocm_hsaco_cache_test.sh can test them on any host.
//
// The default is per-user, ${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/hsaco/
// <version>/<arch>, resolved here rather than by mlxcel's binaries so that every
// program linking MLX gets it, not only mlxcel and mlxcel-server.
//
// Windows: the home directory and %TEMP% are per-user, and this header does not
// evaluate ACLs, so the ownership and mode checks are POSIX only. An explicit
// MLX_HSACO_CACHE_DIR on Windows is used as given.

#pragma once

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace mlx::core::rocm {

// The environment the cache location depends on. An empty path means unset: an
// empty variable must not turn into a directory relative to the current one.
struct HsacoCacheEnv {
  std::filesystem::path override_dir; // MLX_HSACO_CACHE_DIR
  std::filesystem::path cache_root; // MLXCEL_CACHE_DIR
  std::filesystem::path home; // the home directory
};

// The directory to use, or an empty path with `refusal` saying why the cache
// is off (a component and a reason, for the one stderr line).
struct HsacoCacheDir {
  std::filesystem::path dir;
  std::string refusal;
};

// One path component, as the trust check sees it.
struct CacheDirComponent {
  bool is_symlink;
  bool is_directory;
  std::uint32_t uid;
  std::uint32_t mode;
};

// The candidate directory: MLX_HSACO_CACHE_DIR/<arch> when set, else
// <cache root>/hsaco/<version>/<arch>, where the cache root is MLXCEL_CACHE_DIR
// or <home>/.cache/mlxcel. Empty when nothing names a root. Pure.
inline std::filesystem::path hsaco_cache_path(
    const HsacoCacheEnv& env,
    std::string_view version,
    std::string_view arch) {
  if (!env.override_dir.empty()) {
    return env.override_dir / std::string(arch);
  }
  std::filesystem::path root = env.cache_root;
  if (root.empty()) {
    if (env.home.empty()) {
      return {};
    }
    root = env.home / ".cache" / "mlxcel";
  }
  return root / "hsaco" / std::string(version) / std::string(arch);
}

// Why `c` makes a cache directory untrustworthy, or nullptr when it does not.
// Whoever can write a component, or replace it through a parent they can write,
// can plant kernels, so each component must be owned by `euid` or by root and
// must not be group- or other-writable, unless it is root-owned with the sticky
// bit set (/tmp). A symlink is refused: canonicalizing resolved every link, so
// one seen here was swapped in since. `leaf` is the cache directory itself,
// which must be a directory. Pure, so a different owner is testable without a
// second account.
inline const char* cache_component_refusal(
    const CacheDirComponent& c,
    std::uint32_t euid,
    bool leaf) {
  if (c.is_symlink) {
    return "is a symbolic link";
  }
  if (leaf && !c.is_directory) {
    return "is not a directory";
  }
  if (c.uid != euid && c.uid != 0) {
    return "is not owned by the current user";
  }
  const bool shared_write = (c.mode & 0022) != 0;
  const bool root_sticky = c.uid == 0 && (c.mode & 01000) != 0;
  if (shared_write && !root_sticky) {
    return "is group- or other-writable";
  }
  return nullptr;
}

// A path as UTF-8 for a diagnostic; never throws.
inline std::string cache_path_for_log(const std::filesystem::path& path) {
  try {
    auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
  } catch (const std::exception&) {
    return "<path>";
  }
}

#if defined(_WIN32)

inline std::uint32_t cache_current_euid() {
  return 0;
}

inline HsacoCacheEnv hsaco_cache_env_from_process() {
  auto read = [](const wchar_t* name) -> std::filesystem::path {
    const wchar_t* value = ::_wgetenv(name);
    return (value && *value) ? std::filesystem::path(value)
                             : std::filesystem::path();
  };
  return {
      read(L"MLX_HSACO_CACHE_DIR"),
      read(L"MLXCEL_CACHE_DIR"),
      read(L"USERPROFILE")};
}

inline bool make_private_directories(const std::filesystem::path& dir) {
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  return !error && std::filesystem::is_directory(dir, error);
}

// `euid` exists for the POSIX signature; see the note at the top of this file.
inline HsacoCacheDir resolve_hsaco_cache_dir(
    const HsacoCacheEnv& env,
    std::string_view version,
    std::string_view arch,
    std::uint32_t /*euid*/) {
  HsacoCacheDir result;
  std::filesystem::path cache = hsaco_cache_path(env, version, arch);
  if (cache.empty()) {
    result.refusal = "no directory: MLXCEL_CACHE_DIR and USERPROFILE are unset";
    return result;
  }
  std::error_code error;
  auto absolute = std::filesystem::absolute(cache, error);
  if (error || !make_private_directories(absolute)) {
    result.refusal = cache_path_for_log(cache) + " cannot be created";
    return result;
  }
  result.dir = std::move(absolute);
  return result;
}

inline bool read_trusted_cache_entry(
    const std::filesystem::path& path,
    std::string& contents,
    std::uint32_t /*euid*/ = 0) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    return false;
  }
  contents.assign(
      std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  return !file.bad();
}

#else

inline std::uint32_t cache_current_euid() {
  return static_cast<std::uint32_t>(::geteuid());
}

// The password database's home directory, as Rust's home_dir does when HOME is
// unset. Empty if there is none.
inline std::filesystem::path passwd_home_dir() {
  long hint = ::sysconf(_SC_GETPW_R_SIZE_MAX);
  std::string buffer(hint > 0 ? static_cast<size_t>(hint) : 16384, '\0');
  struct passwd entry;
  struct passwd* found = nullptr;
  if (::getpwuid_r(::geteuid(), &entry, buffer.data(), buffer.size(), &found) !=
          0 ||
      found == nullptr || found->pw_dir == nullptr) {
    return {};
  }
  return std::filesystem::path(found->pw_dir);
}

inline HsacoCacheEnv hsaco_cache_env_from_process() {
  auto read = [](const char* name) -> std::filesystem::path {
    const char* value = std::getenv(name);
    return (value && *value) ? std::filesystem::path(value)
                             : std::filesystem::path();
  };
  std::filesystem::path home = read("HOME");
  if (home.empty()) {
    home = passwd_home_dir();
  }
  return {read("MLX_HSACO_CACHE_DIR"), read("MLXCEL_CACHE_DIR"), home};
}

// Create each missing component of the absolute `dir` with mode 0700. Existing
// components are left as they are: cache_dir_is_trusted decides whether they
// are acceptable. False when a component can neither be created nor found as a
// directory.
inline bool make_private_directories(const std::filesystem::path& dir) {
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

// Clear the group- and other-write bits of `path`, when it is a real directory
// owned by `euid`, as ptx_cache_dirs.rs does for the CUDA cache. The mode is
// changed through a descriptor, only for the directory the lstat approved, so a
// symlink swapped in meanwhile cannot redirect the chmod. Errors are ignored: a
// directory left as it was is refused with a notice that names it.
inline void remove_shared_write(
    const std::filesystem::path& path,
    std::uint32_t euid) {
  struct stat checked;
  if (::lstat(path.c_str(), &checked) != 0 || !S_ISDIR(checked.st_mode) ||
      static_cast<std::uint32_t>(checked.st_uid) != euid ||
      (checked.st_mode & 0022) == 0) {
    return;
  }
  int fd = ::open(
      path.c_str(),
      O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    return;
  }
  struct stat opened;
  if (::fstat(fd, &opened) == 0 && opened.st_dev == checked.st_dev &&
      opened.st_ino == checked.st_ino &&
      static_cast<std::uint32_t>(opened.st_uid) == euid) {
    (void)::fchmod(fd, (opened.st_mode & 07777) & ~static_cast<mode_t>(0022));
  }
  ::close(fd);
}

// Whether `dir` is safe to load compiled kernels from: every component of the
// canonical path, from the leaf up to /, must pass cache_component_refusal
// (trusting the leaf alone is not enough, since the owner of a parent can
// rename the leaf away and put their own in its place). On success `resolved`
// holds the canonical path, which callers use from then on. On failure `reason`
// names the offending component.
inline bool cache_dir_is_trusted(
    const std::filesystem::path& dir,
    std::uint32_t euid,
    std::filesystem::path& resolved,
    std::string& reason) {
  std::error_code error;
  auto canonical = std::filesystem::canonical(dir, error);
  if (error) {
    reason = dir.string() + " cannot be resolved (" + error.message() + ")";
    return false;
  }
  bool leaf = true;
  for (auto component = canonical;; component = component.parent_path()) {
    struct stat st;
    if (::lstat(component.c_str(), &st) != 0) {
      reason = component.string() + " cannot be inspected";
      return false;
    }
    const CacheDirComponent view{
        S_ISLNK(st.st_mode),
        S_ISDIR(st.st_mode),
        static_cast<std::uint32_t>(st.st_uid),
        static_cast<std::uint32_t>(st.st_mode)};
    if (const char* refusal = cache_component_refusal(view, euid, leaf)) {
      reason = component.string() + " " + refusal;
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

// The cache directory for this process, or an empty path with the reason. A
// `..` is resolved before anything is created. Missing components get mode
// 0700. For the default location, directories earlier mlxcel code created under
// a umask of 002 are repaired (<arch>, <version>, hsaco, and the cache root only
// when MLXCEL_CACHE_DIR is unset); an explicit directory is the user's choice
// and is never changed, and nothing above the root is touched. Whatever the
// source, the result is used only if it passes the trust check.
inline HsacoCacheDir resolve_hsaco_cache_dir(
    const HsacoCacheEnv& env,
    std::string_view version,
    std::string_view arch,
    std::uint32_t euid) {
  HsacoCacheDir result;
  std::filesystem::path cache = hsaco_cache_path(env, version, arch);
  if (cache.empty()) {
    result.refusal = "no directory: MLXCEL_CACHE_DIR and the home directory "
                     "are unset";
    return result;
  }
  std::error_code error;
  cache = std::filesystem::absolute(cache, error);
  std::filesystem::path normalized;
  if (!error) {
    normalized = std::filesystem::weakly_canonical(cache, error);
  }
  if (error || normalized.empty()) {
    result.refusal = cache.string() + " cannot be resolved";
    return result;
  }
  if (!make_private_directories(normalized)) {
    result.refusal = normalized.string() + " cannot be created";
    return result;
  }
  if (env.override_dir.empty()) {
    const auto version_dir = cache.parent_path();
    const auto hsaco_dir = version_dir.parent_path();
    remove_shared_write(cache, euid);
    remove_shared_write(version_dir, euid);
    remove_shared_write(hsaco_dir, euid);
    if (env.cache_root.empty()) {
      remove_shared_write(hsaco_dir.parent_path(), euid);
    }
  }
  std::filesystem::path trusted;
  std::string reason;
  if (!cache_dir_is_trusted(normalized, euid, trusted, reason)) {
    result.refusal = std::move(reason);
    return result;
  }
  result.dir = std::move(trusted);
  return result;
}

// Read the cache entry at `path` into `contents`. False (a miss) unless it is a
// regular file owned by `euid` or by root that neither group nor others can
// write. A file can be writable by others inside a trusted directory, from a
// restored backup or an earlier umask. The file is opened without following a
// symlink (O_NONBLOCK so a FIFO cannot block the open) and checked through the
// descriptor, so the file read is the file checked. A short read is a miss.
inline bool read_trusted_cache_entry(
    const std::filesystem::path& path,
    std::string& contents,
    std::uint32_t euid = cache_current_euid()) {
  contents.clear();
  struct FdCloser {
    int fd;
    ~FdCloser() {
      if (fd >= 0) {
        ::close(fd);
      }
    }
  } guard{::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)};
  if (guard.fd < 0) {
    return false;
  }
  struct stat st;
  if (::fstat(guard.fd, &st) != 0 || !S_ISREG(st.st_mode) ||
      (static_cast<std::uint32_t>(st.st_uid) != euid && st.st_uid != 0) ||
      (st.st_mode & 0022) != 0) {
    return false;
  }
  const auto size = static_cast<size_t>(st.st_size);
  try {
    contents.resize(size);
  } catch (const std::exception&) {
    contents.clear();
    return false;
  }
  size_t done = 0;
  while (done < size) {
    ssize_t n = ::read(guard.fd, contents.data() + done, size - done);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      contents.clear();
      return false;
    }
    done += static_cast<size_t>(n);
  }
  return true;
}

#endif // _WIN32

// Publish `size` bytes at `path` atomically: write a temporary in the same
// directory, then rename it over the entry. On POSIX the temporary is created
// exclusively, without following a symlink, with mode 0600, so a planted link
// is never written through and an entry never comes out group-writable under a
// umask of 002, which read_trusted_cache_entry would refuse on every later run.
// An older entry with a wider mode or a symlink at `path` is replaced, not
// followed. False, with nothing published, on any failure; the kernel is still
// used and the cache stays cold for it. The temporary is named from the process
// and a counter, not from `path`, so a long entry name cannot overflow NAME_MAX.
inline bool publish_cache_entry(
    const std::filesystem::path& path,
    const char* data,
    size_t size) {
  static std::atomic<std::uint64_t> counter{0};
#if defined(_WIN32)
  const long pid = static_cast<long>(::_getpid());
#else
  const long pid = static_cast<long>(::getpid());
#endif
  const auto tmp = path.parent_path() /
      (".hsaco-tmp-" + std::to_string(pid) + "-" +
       std::to_string(counter.fetch_add(1)) + path.extension().string());
  bool written = false;
#if defined(_WIN32)
  {
    std::ofstream file(tmp, std::ios::binary);
    if (size > 0) {
      file.write(data, static_cast<std::streamsize>(size));
    }
    file.close();
    written = !file.fail();
  }
#else
  int fd = ::open(
      tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) {
    return false;
  }
  written = true;
  size_t done = 0;
  while (done < size) {
    ssize_t n = ::write(fd, data + done, size - done);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      written = false;
      break;
    }
    done += static_cast<size_t>(n);
  }
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
}

} // namespace mlx::core::rocm
