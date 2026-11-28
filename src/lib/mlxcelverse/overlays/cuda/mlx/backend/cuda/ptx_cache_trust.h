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
// New file for mlxcel (no upstream MLX counterpart)

// POSIX trust rules of the PTX disk cache directory (lablup/mlxcel#2539,
// #3503). jit_module.cpp supplies the process's euid, the trusted group and
// MLXCEL_CACHE_TRUSTED_ROOT and reports the outcome; the rules take them as
// parameters and have no CUDA dependency, so
// scripts/mlxcelverse/cuda_runtime_dir_test.sh runs them against real
// directories on any POSIX host. The threat model is in jit_module.cpp's
// header comment.
//
// Two ways to accept a directory:
//
// - The walk (ptx_cache_dir_is_trusted). Every component of the canonical
//   path, from the leaf up to /, is owned by the current user or by root and
//   is not group- or other-writable. Two exceptions: a root-owned sticky
//   directory (/tmp), and a component that only its group can write besides
//   the owner when that group is the trusted one (the process's effective
//   gid, handed over only when it is provably private to the user).
// - The descent (open_ptx_cache_below_trusted_root), used instead of the walk
//   when MLXCEL_CACHE_TRUSTED_ROOT names an ancestor of the cache directory.
//   The root and everything above it are not judged by owner or mode. Each
//   component below it is opened with openat(O_NOFOLLOW) from its parent's
//   descriptor and must be a real directory owned by the current user (root
//   does not count here) that neither group nor others can write. The
//   returned descriptor pins the leaf, so entries read through it come from
//   the directory that was checked whatever happens to the path afterwards
//   (ptx_cache_entry_ref maps a cache path to the descriptor that holds it).

#pragma once

#if !defined(_WIN32)

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "mlx/backend/cuda/ptx_cache_dir.h"

namespace mlx::core::cu {

// "No trusted group": the walk then refuses every group-writable component.
inline constexpr std::int64_t kNoPtxCacheTrustedGroup = -1;

// Why the walk refuses a component with this owner, group and mode, or
// nullptr when it passes. |trusted_gid| is kNoPtxCacheTrustedGroup or the gid
// whose write access counts as the user's own.
inline const char* ptx_cache_component_refusal(
    mode_t mode,
    uid_t uid,
    gid_t gid,
    uid_t euid,
    std::int64_t trusted_gid) {
  if (uid != euid && uid != 0) {
    return "is not owned by the current user";
  }
  if (uid == 0 && (mode & S_ISVTX) != 0) {
    return nullptr;
  }
  const bool group_trusted =
      trusted_gid >= 0 && static_cast<std::int64_t>(gid) == trusted_gid;
  if ((mode & S_IWOTH) != 0 || ((mode & S_IWGRP) != 0 && !group_trusted)) {
    return "is group- or other-writable";
  }
  return nullptr;
}

// Whether |dir| is safe to load compiled modules from, by the walk. On
// success |resolved| holds the canonical path, which callers use from then
// on. On failure |reason| names the offending component.
inline bool ptx_cache_dir_is_trusted(
    const std::filesystem::path& dir,
    uid_t euid,
    std::int64_t trusted_gid,
    std::filesystem::path& resolved,
    std::string& reason) {
  std::error_code error;
  auto canonical = std::filesystem::canonical(dir, error);
  if (error) {
    reason = ptx_cache_path_for_log(dir) + " cannot be resolved (" +
        error.message() + ")";
    return false;
  }
  bool leaf = true;
  for (auto component = canonical;; component = component.parent_path()) {
    struct stat st;
    if (::lstat(component.c_str(), &st) != 0) {
      reason = ptx_cache_path_for_log(component) + " cannot be inspected";
      return false;
    }
    // canonical() resolved every link; one here was swapped in since.
    if (S_ISLNK(st.st_mode)) {
      reason = ptx_cache_path_for_log(component) + " is a symbolic link";
      return false;
    }
    if (leaf && !S_ISDIR(st.st_mode)) {
      reason = ptx_cache_path_for_log(component) + " is not a directory";
      return false;
    }
    if (const char* refusal = ptx_cache_component_refusal(
            st.st_mode, st.st_uid, st.st_gid, euid, trusted_gid)) {
      reason = ptx_cache_path_for_log(component) + " " + refusal;
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

// A valid MLXCEL_CACHE_TRUSTED_ROOT: the canonical root and the cache
// directory relative to it (one or more plain names).
struct PtxTrustedRoot {
  std::filesystem::path root;
  std::filesystem::path below;
};

// Resolve MLXCEL_CACHE_TRUSTED_ROOT |value| against the absolute cache
// directory |cache_dir| as it was given, before any symlink in it is
// resolved. True with |out| set when the value names an existing directory
// that a proper prefix of |cache_dir| resolves to. Otherwise false, with
// |ignored| holding the reason for the "ignored" notice, or left empty for an
// empty (unset) value.
//
// The split is taken on the path as given so that the names below the root
// are the ones the caller asked for: the descent opens exactly those without
// following a symlink. Splitting the canonical path instead would accept
// <root>/cuda-ptx after someone who can write the root replaced it with a
// symlink to another directory of this user. The shortest matching prefix is
// used, so a symlink below the root that points back at the root is refused
// by the descent as well. |cache_dir| is normalized lexically; a `..` in it
// is not resolved through symlinks as the walk's caller does.
inline bool resolve_ptx_trusted_root(
    const std::filesystem::path& value,
    const std::filesystem::path& cache_dir,
    PtxTrustedRoot& out,
    std::string& ignored) {
  if (value.empty()) {
    return false;
  }
  if (!value.is_absolute()) {
    ignored = "must be an absolute path, got \"" +
        ptx_cache_path_for_log(value) + "\"";
    return false;
  }
  std::error_code error;
  auto root = std::filesystem::canonical(value, error);
  if (error) {
    ignored = ptx_cache_path_for_log(value) + " cannot be resolved (" +
        error.message() + ")";
    return false;
  }
  if (!std::filesystem::is_directory(root, error) || error) {
    ignored = ptx_cache_path_for_log(root) + " is not a directory";
    return false;
  }
  std::filesystem::path prefix;
  std::filesystem::path below;
  bool found = false;
  for (const auto& part : cache_dir.lexically_normal()) {
    if (part.empty()) {
      continue; // a trailing separator
    }
    if (found) {
      below /= part;
      continue;
    }
    prefix /= part;
    auto resolved = std::filesystem::canonical(prefix, error);
    if (error) {
      break; // nothing that does not exist yet can be the root
    }
    found = resolved == root;
  }
  if (!found || below.empty()) {
    ignored = ptx_cache_path_for_log(root) + " is not an ancestor of " +
        ptx_cache_path_for_log(cache_dir);
    return false;
  }
  out.root = std::move(root);
  out.below = std::move(below);
  return true;
}

// Open the directories named by |below| (one or more plain names), starting
// in the directory |start_fd| refers to, creating missing ones with mode
// 0700. Returns a descriptor of the last one, which the caller owns, or -1
// with |reason| naming the offending component; |start_fd| stays open either
// way. |start_path| is where |start_fd| was opened, for the reason only.
//
// Each name is created if missing and opened from its parent's descriptor
// without following a symlink, then checked on the descriptor: a directory
// owned by |euid| that neither group nor others can write. A name that sits
// in a directory others may write can be a symlink, their own directory or
// nothing, each of which is refused; once a component is held and private,
// nobody else can change what is below it.
inline int open_ptx_cache_components(
    int start_fd,
    const std::filesystem::path& start_path,
    const std::filesystem::path& below,
    uid_t euid,
    std::string& reason) {
  const int flags = O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
  int fd = start_fd;
  auto path = start_path;
  for (const auto& part : below) {
    path /= part;
    const char* refusal = nullptr;
    std::string detail;
    int next = -1;
    if (::mkdirat(fd, part.c_str(), 0700) != 0 && errno != EEXIST) {
      refusal = "cannot be created";
      detail = std::string(" (") + std::strerror(errno) + ")";
    } else if ((next = ::openat(fd, part.c_str(), flags)) < 0) {
      // ELOOP or ENOTDIR for a symlink, depending on the platform: ask.
      struct stat st;
      if (::fstatat(fd, part.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
        refusal = "cannot be inspected";
      } else if (S_ISLNK(st.st_mode)) {
        refusal = "is a symbolic link";
      } else if (!S_ISDIR(st.st_mode)) {
        refusal = "is not a directory";
      } else {
        refusal = "cannot be inspected";
      }
    } else {
      struct stat st;
      if (::fstat(next, &st) != 0) {
        refusal = "cannot be inspected";
      } else if (!S_ISDIR(st.st_mode)) {
        refusal = "is not a directory";
      } else if (st.st_uid != euid) {
        refusal = "is not owned by the current user";
      } else if ((st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        refusal = "is group- or other-writable";
      }
    }
    if (fd != start_fd) {
      ::close(fd);
    }
    if (refusal != nullptr) {
      if (next >= 0) {
        ::close(next);
      }
      reason = ptx_cache_path_for_log(path) + " " + refusal + detail;
      return -1;
    }
    fd = next;
  }
  if (fd == start_fd) {
    reason = ptx_cache_path_for_log(start_path) + " cannot be inspected";
    return -1;
  }
  return fd;
}

// Open |root|/|below| as a cache directory by the descent, creating missing
// components with mode 0700. Returns a descriptor the caller owns, or -1 with
// |reason| naming the offending component.
//
// |root| and each directory above it must exist and not be a symlink
// (canonical() resolved them; one found now was swapped in since); nothing
// else about them is checked. The names of |below| follow the rule of
// open_ptx_cache_components.
inline int open_ptx_cache_below_trusted_root(
    const std::filesystem::path& root,
    const std::filesystem::path& below,
    uid_t euid,
    std::string& reason) {
  for (auto component = root;; component = component.parent_path()) {
    struct stat st;
    if (::lstat(component.c_str(), &st) != 0) {
      reason = ptx_cache_path_for_log(component) + " cannot be inspected";
      return -1;
    }
    if (S_ISLNK(st.st_mode)) {
      reason = ptx_cache_path_for_log(component) + " is a symbolic link";
      return -1;
    }
    if (component == component.root_path()) {
      break;
    }
  }
  const int root_fd =
      ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (root_fd < 0) {
    reason = ptx_cache_path_for_log(root) + " cannot be inspected";
    return -1;
  }
  const int fd = open_ptx_cache_components(root_fd, root, below, euid, reason);
  ::close(root_fd);
  return fd;
}

// A cache directory the descent opened, with the descriptor that pins it.
struct PtxPinnedDir {
  std::string prefix; // the directory's path with a trailing separator
  int fd;
};

// How to reach a cache file or directory: a descriptor and a name relative
// to it, or AT_FDCWD and the path itself.
struct PtxCacheEntryRef {
  int dir_fd;
  std::string name;
};

// How to reach |path|: through the pinned directory that holds it, or by the
// path itself when none does, which is every case without a trusted root.
// Under a trusted root the path's first component below the root can be
// replaced at any time by whoever can write the root; the descriptor still
// refers to the directory the descent checked. The pinned directory's own
// path is not "held" by it and resolves by path.
inline PtxCacheEntryRef ptx_cache_entry_ref(
    const std::vector<PtxPinnedDir>& pinned,
    const std::filesystem::path& path) {
  std::string name = path.string();
  for (const auto& dir : pinned) {
    if (name.size() > dir.prefix.size() &&
        name.compare(0, dir.prefix.size(), dir.prefix) == 0) {
      name.erase(0, dir.prefix.size());
      return {dir.fd, std::move(name)};
    }
  }
  return {AT_FDCWD, std::move(name)};
}

// Create each missing component of |relative| below |dir_fd| with mode 0700,
// for the nested directories of a long module name. False when a component
// can neither be created nor found as a directory.
inline bool make_private_directories_at(
    int dir_fd,
    const std::filesystem::path& relative) {
  std::filesystem::path partial;
  for (const auto& part : relative) {
    partial /= part;
    if (::mkdirat(dir_fd, partial.c_str(), 0700) == 0 || errno == EEXIST) {
      continue;
    }
    struct stat st;
    if (::fstatat(dir_fd, partial.c_str(), &st, 0) != 0 ||
        !S_ISDIR(st.st_mode)) {
      return false;
    }
  }
  return true;
}

} // namespace mlx::core::cu

#endif // !_WIN32
