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

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// clang-format off
#include <windows.h> // must be included first
#include <aclapi.h>
#include <sddl.h>
// clang-format on
#include <cwchar>
#include <system_error>
#include <utility>
#include <vector>
#pragma comment(lib, "advapi32")

namespace mlx::core::cu {

// Read a variable from the process environment block. Not getenv or _wgetenv:
// they read the C runtime's copy, taken at startup, and
// ensure_persistent_ptx_cache sets MLX_PTX_CACHE_DIR afterwards through
// SetEnvironmentVariableW (Rust's set_var), which does not update that copy. False
// when the variable is not set.
inline bool windows_env_value(const wchar_t* name, std::filesystem::path& value) {
  value.clear();
  ::SetLastError(ERROR_SUCCESS);
  const DWORD needed = ::GetEnvironmentVariableW(name, nullptr, 0);
  if (needed == 0) {
    return ::GetLastError() != ERROR_ENVVAR_NOT_FOUND;
  }
  std::wstring text(needed, L'\0');
  const DWORD got = ::GetEnvironmentVariableW(name, text.data(), needed);
  if (got < needed) {
    text.resize(got);
    value = text;
  }
  return true;
}

inline PtxCacheEnv ptx_cache_env_from_process() {
  PtxCacheEnv env;
  env.override_set = windows_env_value(L"MLX_PTX_CACHE_DIR", env.override_dir);
  windows_env_value(L"MLXCEL_CACHE_DIR", env.cache_root);
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

// The SIDs that may hold write access to a cache directory, besides its owner.
struct WindowsTrustedSids {
  std::vector<BYTE> user;
  std::vector<BYTE> system;
  std::vector<BYTE> administrators;
  std::vector<BYTE> creator_owner;
  bool ok = false;

  WindowsTrustedSids() {
    HANDLE token = nullptr;
    if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &token)) {
      return;
    }
    DWORD size = 0;
    ::GetTokenInformation(token, TokenUser, nullptr, 0, &size);
    std::vector<BYTE> buffer(size);
    const bool have_user = size != 0 &&
        ::GetTokenInformation(token, TokenUser, buffer.data(), size, &size);
    ::CloseHandle(token);
    if (!have_user) {
      return;
    }
    PSID user_sid = reinterpret_cast<TOKEN_USER*>(buffer.data())->User.Sid;
    const auto* user_bytes = static_cast<const BYTE*>(user_sid);
    user.assign(user_bytes, user_bytes + ::GetLengthSid(user_sid));
    ok = make(WinLocalSystemSid, system) &&
        make(WinBuiltinAdministratorsSid, administrators) &&
        make(WinCreatorOwnerSid, creator_owner);
  }

 private:
  static bool make(WELL_KNOWN_SID_TYPE type, std::vector<BYTE>& sid) {
    DWORD size = SECURITY_MAX_SID_SIZE;
    sid.resize(size);
    return ::CreateWellKnownSid(type, nullptr, sid.data(), &size) != FALSE;
  }
};

inline bool sid_is(PSID sid, std::vector<BYTE>& known) {
  return ::EqualSid(sid, known.data()) != FALSE;
}

inline std::string windows_sid_text(PSID sid) {
  LPSTR text = nullptr;
  if (!::ConvertSidToStringSidA(sid, &text)) {
    return "another account";
  }
  std::string result(text);
  ::LocalFree(text);
  return result;
}

// The access that lets an account plant or replace something in a directory:
// add a file or a subdirectory, delete a child, delete or rename the directory
// itself, or take over its ACL or owner. Generic rights count, since an ACE can
// hold them unmapped.
inline constexpr DWORD kPtxCacheWriteClassMask = FILE_WRITE_DATA |
    FILE_APPEND_DATA | FILE_DELETE_CHILD | DELETE | WRITE_DAC | WRITE_OWNER |
    GENERIC_WRITE | GENERIC_ALL;

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
  // Whoever owns a directory can rewrite its DACL, so an owner outside the
  // trusted set is a refusal even if the DACL looks private now.
  if (owner == nullptr ||
      !(sid_is(owner, trusted.user) || sid_is(owner, trusted.system) ||
        sid_is(owner, trusted.administrators))) {
    return shown + " is not owned by the current user";
  }
  // A null DACL grants everyone full access, and a volume without ACLs (FAT,
  // exFAT) reports one too.
  if (dacl == nullptr) {
    return shown + " has no access control list";
  }
  for (DWORD i = 0; i < dacl->AceCount; ++i) {
    void* raw = nullptr;
    if (!::GetAce(dacl, i, &raw)) {
      return shown + " cannot be inspected";
    }
    const auto* header = static_cast<const ACE_HEADER*>(raw);
    const bool plain_allow = header->AceType == ACCESS_ALLOWED_ACE_TYPE ||
        header->AceType == ACCESS_ALLOWED_CALLBACK_ACE_TYPE;
    const bool object_allow =
        header->AceType == ACCESS_ALLOWED_OBJECT_ACE_TYPE ||
        header->AceType == ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE;
    if (!plain_allow && !object_allow) {
      continue;
    }
    const DWORD mask = static_cast<const ACCESS_ALLOWED_ACE*>(raw)->Mask;
    if ((mask & kPtxCacheWriteClassMask) == 0) {
      continue;
    }
    // Object ACEs hold the SID at a variable offset and do not occur on files.
    if (object_allow) {
      return shown + " has an access entry this check cannot read";
    }
    PSID sid = const_cast<DWORD*>(
        &static_cast<const ACCESS_ALLOWED_ACE*>(raw)->SidStart);
    // An inherit-only entry does not apply to this directory. On a parent it
    // reaches the next component, which is checked itself. On the leaf it
    // reaches the entries written there, except that CREATOR OWNER turns into
    // their owner.
    if ((header->AceFlags & INHERIT_ONLY_ACE) != 0 &&
        (!leaf || sid_is(sid, trusted.creator_owner))) {
      continue;
    }
    if (::EqualSid(sid, owner) || sid_is(sid, trusted.user) ||
        sid_is(sid, trusted.system) || sid_is(sid, trusted.administrators)) {
      continue;
    }
    return shown + " is writable by " + windows_sid_text(sid);
  }
  return "";
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
// include deleting or renaming another account's child. Whatever a root does
// grant reaches its children by inheritance, so the first component below it
// catches an inherited write entry.
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
// part is checked before anything is created, so a refused run leaves nothing
// behind in a directory it would refuse, and the result is checked once it exists.
// Nothing is repaired: an ACL is the user's to change. The result has the
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
