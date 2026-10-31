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

// The Windows half of hsaco_cache_dir.h (lablup/mlxcel#2641), which includes this
// file inside no namespace after declaring the types it uses. Do not include it
// on its own.

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
#include <shlobj_core.h>
// clang-format on
#include <process.h>
#pragma comment(lib, "advapi32")
#pragma comment(lib, "ole32")
#pragma comment(lib, "shell32")

namespace mlx::core::rocm {

inline std::uint32_t cache_current_euid() {
  return 0;
}

// The home directory as Rust's dirs::home_dir reads it (dirs-sys'
// known_folder_profile, which is what cache_root() in mlxcel-core uses), so both
// sides agree on the cache root. It comes from the Known Folder API and not from
// USERPROFILE, so overriding that variable moves neither. Empty on failure.
inline std::filesystem::path windows_profile_dir() {
  static const GUID kFolderIdProfile = {
      0x5E6C858F,
      0x0E22,
      0x4760,
      {0x9A, 0xFE, 0xEA, 0x33, 0x17, 0xB6, 0x71, 0x73}};
  PWSTR raw = nullptr;
  const HRESULT result =
      ::SHGetKnownFolderPath(kFolderIdProfile, 0, nullptr, &raw);
  std::filesystem::path path;
  if (SUCCEEDED(result) && raw != nullptr) {
    path = raw;
  }
  ::CoTaskMemFree(raw);
  return path;
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
      windows_profile_dir()};
}

inline bool make_private_directories(const std::filesystem::path& dir) {
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  return !error && std::filesystem::is_directory(dir, error);
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
    const bool have_user =
        size != 0 &&
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
inline constexpr DWORD kCacheWriteClassMask = FILE_WRITE_DATA |
    FILE_APPEND_DATA | FILE_DELETE_CHILD | DELETE | WRITE_DAC | WRITE_OWNER |
    GENERIC_WRITE | GENERIC_ALL;

// Why the owner or DACL of `component` makes it untrustworthy, or "".
inline std::string windows_component_refusal(
    const std::filesystem::path& component,
    WindowsTrustedSids& trusted,
    bool leaf) {
  const std::string shown = cache_path_for_log(component);
  PSID owner = nullptr;
  PACL dacl = nullptr;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  const DWORD status = ::GetNamedSecurityInfoW(
      component.c_str(),
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
    if ((mask & kCacheWriteClassMask) == 0) {
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

// Whether `dir` is safe to load compiled kernels from: every component of the
// canonical path below the drive root must pass windows_component_refusal, since
// the owner of a parent can rename a child away and put their own in its place.
//
// The drive root is excluded. By default it grants Authenticated Users the right
// to create subdirectories on every drive, so checking it would turn the cache
// off everywhere, and it is not the right that matters: those defaults do not
// include deleting or renaming another account's child. Whatever a root does
// grant reaches its children by inheritance, so the first component below it
// catches an inherited write entry. `euid` exists for the POSIX signature.
inline bool cache_dir_is_trusted(
    const std::filesystem::path& dir,
    std::uint32_t /*euid*/,
    std::filesystem::path& resolved,
    std::string& reason) {
  std::error_code error;
  auto canonical = std::filesystem::canonical(dir, error);
  if (error) {
    reason = cache_path_for_log(dir) + " cannot be resolved (" +
        error.message() + ")";
    return false;
  }
  WindowsTrustedSids trusted;
  if (!trusted.ok) {
    reason = cache_path_for_log(canonical) +
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

// The cache directory for this process, or an empty path with the reason. The
// existing part of the path is checked before anything is created. A symlink or
// junction at a component only mlxcel creates was planted by someone else, and
// is refused even when it points at a private directory. Nothing is repaired:
// an ACL is the user's to change.
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
  std::error_code error;
  std::filesystem::path current =
      std::filesystem::weakly_canonical(layout.base, error);
  if (error || current.empty()) {
    result.refusal = cache_path_for_log(layout.base) + " cannot be resolved";
    return result;
  }
  std::filesystem::path trusted;
  if (!cache_dir_is_trusted(
          deepest_existing_ancestor(current), euid, trusted, result.refusal)) {
    return result;
  }
  if (!make_private_directories(current)) {
    result.refusal = cache_path_for_log(current) + " cannot be created";
    return result;
  }
  for (const std::string& name : layout.tail) {
    current /= name;
    std::filesystem::create_directory(current, error);
    const auto status = std::filesystem::symlink_status(current, error);
    if (error) {
      result.refusal = cache_path_for_log(current) + " cannot be created";
      return result;
    }
    if (std::filesystem::is_symlink(status) ||
        status.type() == std::filesystem::file_type::junction) {
      result.refusal = cache_path_for_log(current) + " is a symbolic link";
      return result;
    }
    if (!std::filesystem::is_directory(status)) {
      result.refusal = cache_path_for_log(current) + " is not a directory";
      return result;
    }
  }
  if (!cache_dir_is_trusted(current, euid, trusted, result.refusal)) {
    return result;
  }
  result.dir = std::move(trusted);
  return result;
}

// Read the entry at `path`, sized from the file. On Windows the entry's trust
// comes from the directory check; a short read or a file too large to hold is a
// miss.
inline bool read_trusted_cache_entry(
    const std::filesystem::path& path,
    std::string& contents,
    std::uint32_t /*euid*/ = 0) {
  contents.clear();
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return false;
  }
  const auto size = std::filesystem::file_size(path, error);
  if (error) {
    return false;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    return false;
  }
  try {
    contents.resize(static_cast<size_t>(size));
  } catch (const std::exception&) {
    contents.clear();
    return false;
  }
  file.read(contents.data(), static_cast<std::streamsize>(size));
  if (static_cast<std::uintmax_t>(file.gcount()) != size) {
    contents.clear();
    return false;
  }
  return true;
}

} // namespace mlx::core::rocm
