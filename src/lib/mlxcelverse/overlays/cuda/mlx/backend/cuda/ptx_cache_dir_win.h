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

// The Windows half of ptx_cache_dir.h (lablup/mlxcel#2797), which includes this
// file after declaring the types it uses. Do not include it on its own. The rules
// are those of hsaco_cache_dir_win.h in the ROCm overlay (the overlays share no
// code); what differs is that jit_module.cpp keeps this cache's path with the
// \\?\ prefix, so that a path past MAX_PATH works.

#pragma once

#include "mlx/backend/cuda/ptx_cache_acl_win.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cwchar>
#include <system_error>
#include <utility>

namespace mlx::core::cu {

// Read a variable from the process environment block; empty when it is unset or
// empty. ensure_persistent_ptx_cache sets MLX_PTX_CACHE_DIR through set_native_var,
// which reaches the C runtime too (lablup/mlxcel#2803), but getenv returns the
// active code page and a path may hold a character outside it. A variable set by
// plain Rust set_var (a test, for one) is in this block alone.
inline std::filesystem::path windows_env_value(const wchar_t* name) {
  const DWORD needed = ::GetEnvironmentVariableW(name, nullptr, 0);
  if (needed == 0) {
    return {};
  }
  std::wstring text(needed, L'\0');
  const DWORD got = ::GetEnvironmentVariableW(name, text.data(), needed);
  if (got == 0 || got >= needed) {
    return {};
  }
  text.resize(got);
  return text;
}

inline PtxCacheEnv ptx_cache_env_from_process() {
  PtxCacheEnv env;
  env.override_dir = windows_env_value(L"MLX_PTX_CACHE_DIR");
  env.cache_root = windows_env_value(L"MLXCEL_CACHE_DIR");
  return env;
}

// `path` without a \\?\ prefix: \\?\C:\x is C:\x and \\?\UNC\host\x is \\host\x.
// std::filesystem takes \\? for the root name of a prefixed path, so its root is
// not the drive and its parents run past the drive root; the walk below works on
// this form.
inline std::filesystem::path without_long_path_prefix(
    const std::filesystem::path& path) {
  const std::wstring& text = path.native();
  if (text.size() >= 8 && ::_wcsnicmp(text.c_str(), LR"(\\?\UNC\)", 8) == 0) {
    return std::wstring(LR"(\\)") + text.substr(8);
  }
  if (text.size() >= 4 && text.compare(0, 4, LR"(\\?\)") == 0) {
    return text.substr(4);
  }
  return path;
}

// `path` as the file system calls below take it, past MAX_PATH too. The prefix
// turns off the parsing of /, . and .., so `path` must be normal already.
inline std::filesystem::path with_long_path_prefix(
    const std::filesystem::path& path) {
  const std::wstring& text = path.native();
  if (text.compare(0, 4, LR"(\\?\)") == 0 ||
      text.compare(0, 4, LR"(\\.\)") == 0) {
    return path;
  }
  if (text.compare(0, 2, LR"(\\)") == 0) {
    return std::wstring(LR"(\\?\UNC\)") + text.substr(2);
  }
  return std::wstring(LR"(\\?\)") + text;
}

// Why the owner or DACL of `component` makes it untrustworthy, or "". `component`
// has no \\?\ prefix, so the notice shows the path the user knows.
inline std::string windows_component_refusal(
    const std::filesystem::path& component,
    WindowsTrustedSids& trusted,
    bool leaf) {
  const std::string shown = ptx_cache_path_for_log(component);
  const std::filesystem::path target = with_long_path_prefix(component);
  PSID owner = nullptr;
  PACL dacl = nullptr;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  const DWORD status = ::GetNamedSecurityInfoW(
      target.c_str(),
      SE_FILE_OBJECT,
      OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
      &owner,
      nullptr,
      &dacl,
      nullptr,
      &descriptor);
  if (status != ERROR_SUCCESS) {
    return shown + " cannot be inspected";
  }
  struct Release {
    PSECURITY_DESCRIPTOR descriptor;
    ~Release() {
      ::LocalFree(descriptor);
    }
  } release{descriptor};
  const std::string refusal =
      windows_security_refusal(owner, dacl, trusted, leaf);
  return refusal.empty() ? refusal : shown + " " + refusal;
}

// Whether `dir` is safe to load compiled modules from: it is a directory, and
// every component of the canonical path below the drive root passes
// windows_component_refusal, since the owner of a parent can rename a child away
// and put their own in its place. On success `resolved` holds the canonical path
// without a \\?\ prefix. On failure `reason` names the offending component.
//
// The drive root is excluded. By default it grants Authenticated Users the right
// to create subdirectories on every drive, so checking it would turn the cache
// off everywhere, and it is not the right that matters: those defaults do not
// include deleting or renaming another account's child. What a root grants to
// inherit reaches a first component that keeps inheritance, where this walk sees
// it. A first component with inheritance disabled does not show the root's
// grants, so a root that gives another account delete-child, which no default
// does, could replace it unseen.
inline bool ptx_cache_dir_is_trusted(
    const std::filesystem::path& dir,
    std::filesystem::path& resolved,
    std::string& reason) {
  std::error_code error;
  auto canonical = without_long_path_prefix(
      std::filesystem::canonical(with_long_path_prefix(dir), error));
  if (error || !canonical.is_absolute()) {
    reason = ptx_cache_path_for_log(without_long_path_prefix(dir)) +
        " cannot be resolved" + (error ? " (" + error.message() + ")" : "");
    return false;
  }
  if (!std::filesystem::is_directory(with_long_path_prefix(canonical), error)) {
    reason = ptx_cache_path_for_log(canonical) + " is not a directory";
    return false;
  }
  WindowsTrustedSids trusted;
  if (!trusted.ok) {
    reason = ptx_cache_path_for_log(canonical) +
        " cannot be inspected (the current account's SID is not available)";
    return false;
  }
  const auto root = canonical.root_path();
  bool leaf = true;
  for (auto component = canonical; component != root;
       component = component.parent_path()) {
    if (!component.has_relative_path()) {
      break;
    }
    reason = windows_component_refusal(component, trusted, leaf);
    if (!reason.empty()) {
      return false;
    }
    leaf = false;
  }
  resolved = std::move(canonical);
  return true;
}

// Create `name` in the trusted `parent`, a name only mlxcel creates, and confirm
// it is a real directory: a symlink or junction there was planted by someone else
// and is refused even when it points at a private directory. Returns the path
// with a \\?\ prefix, or an empty path; `reason` is set when it is refused.
inline std::filesystem::path make_ptx_cache_subdir(
    const std::filesystem::path& parent,
    const std::filesystem::path& name,
    std::string& reason) {
  const auto path = with_long_path_prefix(parent) / name;
  const std::string shown = ptx_cache_path_for_log(without_long_path_prefix(path));
  std::error_code error;
  std::filesystem::create_directory(path, error);
  const auto status = std::filesystem::symlink_status(path, error);
  if (error) {
    reason = shown + " cannot be created";
    return {};
  }
  if (std::filesystem::is_symlink(status) ||
      status.type() == std::filesystem::file_type::junction) {
    reason = shown + " is a symbolic link";
    return {};
  }
  if (!std::filesystem::is_directory(status)) {
    reason = shown + " is not a directory";
    return {};
  }
  return path;
}

// The directory `base`, with the components of `tail` below it, or an empty path
// with `reason` set. `base` is absolute and may hold a \\?\ prefix. The existing
// part is checked before anything is created, so nothing is created under a
// component that fails. A missing first component below the drive root has only
// the root above it, which is not checked, so it can be created and then refused
// once it shows what it inherited, and it stays. The result is checked once it
// exists. Nothing is repaired: an ACL is the user's to change. The result has the
// \\?\ prefix.
inline std::filesystem::path resolve_ptx_cache_dir(
    const std::filesystem::path& base,
    const std::vector<std::filesystem::path>& tail,
    std::string& reason) {
  // What Win32 does to a path without the prefix, so .. and / stay usable:
  // mlxcel_core joins the default root with a forward slash.
  auto full = without_long_path_prefix(base).lexically_normal();
  if (!full.has_filename() && full.has_relative_path()) {
    full = full.parent_path();
  }
  if (!full.is_absolute()) {
    reason = ptx_cache_path_for_log(full) + " cannot be resolved";
    return {};
  }
  auto existing = full;
  for (;;) {
    std::error_code error;
    if (std::filesystem::exists(with_long_path_prefix(existing), error) ||
        !existing.has_relative_path()) {
      break;
    }
    existing = existing.parent_path();
  }
  std::filesystem::path trusted;
  if (!ptx_cache_dir_is_trusted(existing, trusted, reason)) {
    return {};
  }
  std::error_code error;
  std::filesystem::create_directories(with_long_path_prefix(full), error);
  if (!std::filesystem::is_directory(with_long_path_prefix(full), error)) {
    reason = ptx_cache_path_for_log(full) + " cannot be created";
    return {};
  }
  auto current = with_long_path_prefix(full);
  for (const std::filesystem::path& name : tail) {
    current = make_ptx_cache_subdir(current, name, reason);
    if (current.empty()) {
      return {};
    }
  }
  if (!ptx_cache_dir_is_trusted(current, trusted, reason)) {
    return {};
  }
  return with_long_path_prefix(trusted);
}

// Read the entry at `path`: a regular file, not a reparse point, that passes
// windows_security_refusal. A directory that was shared and later made private
// passes its own check and can still hold another account's entries, so each one
// is checked. Everything goes through one handle, opened without following a link
// and without sharing write, so the file read is the file checked. A refusal, a
// short read or a file too large to hold is a miss, which the caller recompiles
// and republishes: a published entry is renamed within the directory, so it holds
// the owner of this process and the directory's inherited ACL, and passes.
inline bool read_trusted_cache_entry(
    const std::filesystem::path& path,
    std::string& contents) {
  contents.clear();
  const std::filesystem::path target = with_long_path_prefix(path);
  struct File {
    HANDLE handle;
    ~File() {
      if (handle != INVALID_HANDLE_VALUE) {
        ::CloseHandle(handle);
      }
    }
  } file{::CreateFileW(
      target.c_str(),
      GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_DELETE,
      nullptr,
      OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
      nullptr)};
  BY_HANDLE_FILE_INFORMATION info;
  if (file.handle == INVALID_HANDLE_VALUE ||
      !::GetFileInformationByHandle(file.handle, &info) ||
      (info.dwFileAttributes &
       (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0) {
    return false;
  }
  PSID owner = nullptr;
  PACL dacl = nullptr;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  if (::GetSecurityInfo(
          file.handle,
          SE_FILE_OBJECT,
          OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
          &owner,
          nullptr,
          &dacl,
          nullptr,
          &descriptor) != ERROR_SUCCESS) {
    return false;
  }
  struct Release {
    PSECURITY_DESCRIPTOR descriptor;
    ~Release() {
      ::LocalFree(descriptor);
    }
  } release{descriptor};
  WindowsTrustedSids trusted;
  if (!trusted.ok ||
      !windows_security_refusal(owner, dacl, trusted, false).empty()) {
    return false;
  }
  const std::uint64_t size =
      (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
  try {
    contents.resize(static_cast<std::size_t>(size));
  } catch (const std::exception&) {
    contents.clear();
    return false;
  }
  std::size_t done = 0;
  while (done < contents.size()) {
    const DWORD chunk = static_cast<DWORD>(
        std::min<std::size_t>(contents.size() - done, 1u << 30));
    DWORD got = 0;
    if (!::ReadFile(file.handle, contents.data() + done, chunk, &got, nullptr) ||
        got == 0) {
      contents.clear();
      return false;
    }
    done += got;
  }
  return true;
}

// The device subdirectory `name` of the trusted cache `root`, or an empty path
// with `reason` set when it is refused. The result has a \\?\ prefix.
inline std::filesystem::path resolve_ptx_cache_subdir(
    const std::filesystem::path& root,
    const std::filesystem::path& name,
    std::string& reason) {
  const auto dir = make_ptx_cache_subdir(root, name, reason);
  if (dir.empty()) {
    return {};
  }
  std::filesystem::path trusted;
  if (!ptx_cache_dir_is_trusted(dir, trusted, reason)) {
    return {};
  }
  return with_long_path_prefix(trusted);
}

} // namespace mlx::core::cu
