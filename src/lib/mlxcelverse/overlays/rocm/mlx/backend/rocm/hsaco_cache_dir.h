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
// another account can write is a code-planting path. The POSIX rules are the
// CUDA overlay's (make_owned_directory, cache_dir_is_trusted and
// read_trusted_cache_entry in its jit_module.cpp, lablup/mlxcel#2539 and #2765);
// Windows applies the same idea to the owner and DACL. They live in this header,
// which has no HIP dependency, so scripts/mlxcelverse/rocm_hsaco_cache_test.sh
// can test them on any host.
//
// The default is per-user, ${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/hsaco/
// <version>/<arch>, resolved here rather than by mlxcel's binaries so that every
// program linking MLX gets it, not only mlxcel and mlxcel-server. Include this
// header after the backend's own headers: on Windows it brings in <windows.h>
// through hsaco_cache_dir_win.h, which holds the Windows half of these functions.

#pragma once

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace mlx::core::rocm {

// The environment the cache location depends on. An empty path means unset.
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

// One path component, as the POSIX trust check sees it.
struct CacheDirComponent {
  bool is_symlink;
  bool is_directory;
  std::uint32_t uid;
  std::uint32_t mode;
};

// Where the cache goes: `base` is the directory the user or the home directory
// names, and `tail` the components below it that only mlxcel creates.
struct HsacoCacheLayout {
  std::filesystem::path base;
  std::vector<std::string> tail;
  bool explicit_override = false; // MLX_HSACO_CACHE_DIR is set
  bool base_is_home_default = false; // base is <home>/.cache/mlxcel
};

// Pure: MLX_HSACO_CACHE_DIR/<arch> when set, else <cache root>/hsaco/<version>/
// <arch>, where the cache root is MLXCEL_CACHE_DIR or <home>/.cache/mlxcel. An
// empty base when nothing names a root.
inline HsacoCacheLayout hsaco_cache_layout(
    const HsacoCacheEnv& env,
    std::string_view version,
    std::string_view arch) {
  HsacoCacheLayout layout;
  if (!env.override_dir.empty()) {
    layout.base = env.override_dir;
    layout.tail = {std::string(arch)};
    layout.explicit_override = true;
    return layout;
  }
  layout.tail = {"hsaco", std::string(version), std::string(arch)};
  if (!env.cache_root.empty()) {
    layout.base = env.cache_root;
  } else if (!env.home.empty()) {
    layout.base = env.home / ".cache" / "mlxcel";
    layout.base_is_home_default = true;
  }
  return layout;
}

// Pure: the full candidate directory, or an empty path.
inline std::filesystem::path hsaco_cache_path(
    const HsacoCacheEnv& env,
    std::string_view version,
    std::string_view arch) {
  const HsacoCacheLayout layout = hsaco_cache_layout(env, version, arch);
  if (layout.base.empty()) {
    return {};
  }
  std::filesystem::path path = layout.base;
  for (const std::string& name : layout.tail) {
    path /= name;
  }
  return path;
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

// Why the environment cannot name a cache, or "" when it can. A relative path
// resolves against the current directory, which anyone can pick for a victim,
// so it is refused as MLXCEL_ROCM_RUNTIME_DIR is. On Windows a path such as
// \cache or C:cache is relative too: it depends on the current drive or the
// drive's current directory.
inline std::string hsaco_env_refusal(const HsacoCacheEnv& env) {
  const auto relative = [](const std::filesystem::path& path, const char* name) {
    return name + std::string(" must be an absolute path, got \"") +
        cache_path_for_log(path) + "\"";
  };
  if (!env.override_dir.empty()) {
    return env.override_dir.is_absolute()
        ? std::string()
        : relative(env.override_dir, "MLX_HSACO_CACHE_DIR");
  }
  if (!env.cache_root.empty()) {
    return env.cache_root.is_absolute()
        ? std::string()
        : relative(env.cache_root, "MLXCEL_CACHE_DIR");
  }
  if (env.home.empty()) {
    return "no directory: MLXCEL_CACHE_DIR and the home directory are unset";
  }
  return env.home.is_absolute() ? std::string()
                                : relative(env.home, "the home directory");
}

// The deepest component of `path` that exists, following symlinks. Trust is
// checked on it before anything is created, so a refused run leaves nothing
// behind, whoever runs it.
inline std::filesystem::path deepest_existing_ancestor(
    const std::filesystem::path& path) {
  std::filesystem::path current = path;
  for (;;) {
    std::error_code error;
    if (std::filesystem::exists(current, error) ||
        current == current.root_path() || !current.has_parent_path()) {
      return current;
    }
    current = current.parent_path();
  }
}

// Pure: why `c` makes a cache directory untrustworthy, or nullptr when it does
// not. Whoever can write a component, or replace it through a parent they can
// write, can plant kernels, so each component must be owned by `euid` or by root
// and must not be group- or other-writable, unless it is root-owned with the
// sticky bit set (/tmp). A symlink is refused: canonicalizing resolved every
// link, so one seen here was swapped in since. `leaf` is the cache directory
// itself, which must be a directory. Pure, so a different owner is testable
// without a second account.
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

#if defined(_WIN32)

} // namespace mlx::core::rocm
#include "mlx/backend/rocm/hsaco_cache_dir_win.h"
namespace mlx::core::rocm {

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

// Closes a descriptor on every path.
struct CacheFd {
  int fd = -1;
  CacheFd() = default;
  explicit CacheFd(int value) : fd(value) {}
  CacheFd(const CacheFd&) = delete;
  CacheFd(CacheFd&& other) noexcept : fd(other.fd) {
    other.fd = -1;
  }
  CacheFd& operator=(const CacheFd&) = delete;
  CacheFd& operator=(CacheFd&& other) noexcept {
    std::swap(fd, other.fd);
    return *this;
  }
  ~CacheFd() {
    if (fd >= 0) {
      ::close(fd);
    }
  }
};

// Create `name` under `parent` with mode 0700 if it is missing, open it without
// following a symlink, and check it. This is make_owned_directory's rule from
// the CUDA overlay: for a component only mlxcel creates, a symlink or other file
// was planted by someone else, and following it would put the cache wherever the
// link points, so it is refused even when the target is private. With `repair`
// the group-write bit is cleared through the descriptor on a directory `euid`
// owns, which is what a umask of 002 leaves behind. Other-write is not something
// a umask explains: that directory is left as it is and refused. Returns an
// invalid descriptor with `reason` set on refusal.
inline CacheFd open_owned_component(
    int parent,
    const std::string& name,
    const std::filesystem::path& shown,
    bool repair,
    std::uint32_t euid,
    std::string& reason) {
  if (::mkdirat(parent, name.c_str(), 0700) != 0 && errno != EEXIST) {
    reason = shown.string() + " cannot be created";
    return CacheFd();
  }
  CacheFd fd(::openat(
      parent,
      name.c_str(),
      O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
  struct stat st;
  if (fd.fd < 0) {
    if (::fstatat(parent, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
      reason = shown.string() + " cannot be inspected";
    } else if (S_ISLNK(st.st_mode)) {
      reason = shown.string() + " is a symbolic link";
    } else if (!S_ISDIR(st.st_mode)) {
      reason = shown.string() + " is not a directory";
    } else {
      reason = shown.string() + " cannot be inspected";
    }
    return CacheFd();
  }
  if (::fstat(fd.fd, &st) != 0) {
    reason = shown.string() + " cannot be inspected";
    return CacheFd();
  }
  if (repair && static_cast<std::uint32_t>(st.st_uid) == euid &&
      (st.st_mode & S_IWGRP) != 0 && (st.st_mode & S_IWOTH) == 0) {
    const mode_t narrowed = (st.st_mode & 07777) & ~static_cast<mode_t>(S_IWGRP);
    if (::fchmod(fd.fd, narrowed) == 0) {
      st.st_mode = (st.st_mode & ~static_cast<mode_t>(07777)) | narrowed;
    }
  }
  const CacheDirComponent view{
      false,
      S_ISDIR(st.st_mode),
      static_cast<std::uint32_t>(st.st_uid),
      static_cast<std::uint32_t>(st.st_mode)};
  if (const char* refusal = cache_component_refusal(view, euid, true)) {
    reason = shown.string() + " " + refusal;
    return CacheFd();
  }
  return fd;
}

// The cache directory for this process, or an empty path with the reason.
//
// A `..` is resolved first, and the existing part of the path is checked before
// anything is created, so a refused run leaves no directories behind. Only then
// are missing components created (mode 0700) and repairs made, which therefore
// never run under an ancestor that failed the check. The components below `base`
// are walked with openat(O_NOFOLLOW), so a symlink in any of them stops the walk
// and nothing is chmodded through one.
//
// For the default location, group write left by a umask of 002 is cleared on
// hsaco, <version> and <arch>, and on the cache root when MLXCEL_CACHE_DIR is
// unset. An explicit directory is the user's choice and is never changed, and
// nothing above the root is touched. ~/.cache/mlxcel may itself be a symlink (to
// a larger disk, say): the Rust side creates that directory too, so the link is
// judged by its target, which is not repaired.
inline HsacoCacheDir resolve_hsaco_cache_dir(
    const HsacoCacheEnv& env,
    std::string_view version,
    std::string_view arch,
    std::uint32_t euid) {
  HsacoCacheDir result;
  result.refusal = hsaco_env_refusal(env);
  if (!result.refusal.empty()) {
    return result;
  }
  const HsacoCacheLayout layout = hsaco_cache_layout(env, version, arch);

  // Everything a walk starts from; for the home default, the root itself is
  // examined without following a link, so normalize only its parent.
  std::error_code error;
  std::filesystem::path start = std::filesystem::weakly_canonical(
      layout.base_is_home_default ? layout.base.parent_path() : layout.base,
      error);
  if (error || start.empty()) {
    result.refusal = layout.base.string() + " cannot be resolved";
    return result;
  }
  std::filesystem::path base = layout.base_is_home_default
      ? start / layout.base.filename()
      : start;

  std::filesystem::path resolved;
  if (!cache_dir_is_trusted(
          deepest_existing_ancestor(start), euid, resolved, result.refusal)) {
    return result;
  }
  if (!make_private_directories(start)) {
    result.refusal = start.string() + " cannot be created";
    return result;
  }
  if (layout.base_is_home_default) {
    struct stat st;
    const bool is_link =
        ::lstat(base.c_str(), &st) == 0 && S_ISLNK(st.st_mode);
    if (!is_link) {
      CacheFd parent(
          ::open(start.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
      if (parent.fd < 0) {
        result.refusal = start.string() + " cannot be inspected";
        return result;
      }
      CacheFd root = open_owned_component(
          parent.fd,
          layout.base.filename().string(),
          base,
          /*repair=*/true,
          euid,
          result.refusal);
      if (root.fd < 0) {
        return result;
      }
    }
  }
  if (!cache_dir_is_trusted(base, euid, resolved, result.refusal)) {
    return result;
  }

  CacheFd current(::open(resolved.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (current.fd < 0) {
    result.refusal = resolved.string() + " cannot be inspected";
    return result;
  }
  std::filesystem::path shown = resolved;
  for (const std::string& name : layout.tail) {
    shown /= name;
    CacheFd next = open_owned_component(
        current.fd,
        name,
        shown,
        /*repair=*/!layout.explicit_override,
        euid,
        result.refusal);
    if (next.fd < 0) {
      return result;
    }
    current = std::move(next);
  }
  if (!cache_dir_is_trusted(shown, euid, resolved, result.refusal)) {
    return result;
  }
  result.dir = std::move(resolved);
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
  CacheFd guard(
      ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
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

// The counter behind publish temporaries' names, one per process.
inline std::atomic<std::uint64_t>& cache_publish_counter() {
  static std::atomic<std::uint64_t> counter{0};
  return counter;
}

// Publish `size` bytes at `path` atomically: write a temporary in the same
// directory, then rename it over the entry. On POSIX the temporary is created
// exclusively, without following a symlink, with mode 0600, so a planted link
// is never written through and an entry never comes out group-writable under a
// umask of 002, which read_trusted_cache_entry would refuse on every later run.
// An older entry with a wider mode or a symlink at `path` is replaced, not
// followed. False, with nothing published, on any failure; the kernel is still
// used and the cache stays cold for it. The temporary is named from the process
// and a counter, not from `path`, so a long entry name cannot overflow NAME_MAX.
// A name already taken moves on to the next counter: a process killed mid-publish
// leaves an orphan, and a later process with the same pid would otherwise fail on
// it and recompile the module on every run. The orphan itself is left in place.
inline bool publish_cache_entry(
    const std::filesystem::path& path,
    const char* data,
    size_t size) {
#if defined(_WIN32)
  const long pid = static_cast<long>(::_getpid());
#else
  const long pid = static_cast<long>(::getpid());
#endif
  const auto temporary = [&] {
    return path.parent_path() /
        (".hsaco-tmp-" + std::to_string(pid) + "-" +
         std::to_string(cache_publish_counter().fetch_add(1)) + path.extension().string());
  };
  std::filesystem::path tmp;
  bool written = false;
#if defined(_WIN32)
  // std::ofstream replaces an orphan at this name, so nothing is retried here.
  tmp = temporary();
  {
    std::ofstream file(tmp, std::ios::binary);
    if (size > 0) {
      file.write(data, static_cast<std::streamsize>(size));
    }
    file.close();
    written = !file.fail();
  }
#else
  constexpr int kMaxNameAttempts = 64;
  int fd = -1;
  for (int attempt = 0; attempt < kMaxNameAttempts; ++attempt) {
    tmp = temporary();
    fd = ::open(
        tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd >= 0 || errno != EEXIST) {
      break;
    }
  }
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
