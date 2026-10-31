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

// The ownership and ACL rule of the PTX disk cache (lablup/mlxcel#2797), which
// ptx_cache_dir_win.h applies to each path component and each cache entry. It takes
// the owner and the ACL as input, so scripts/mlxcelverse/cuda_ptx_cache_win_test.cpp
// tests it without a second account or a FAT volume.

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
#include <string>
#include <vector>
#pragma comment(lib, "advapi32")

namespace mlx::core::cu {

// The SIDs that may hold write access to a cache directory, besides its owner.
struct WindowsTrustedSids {
  std::vector<BYTE> user;
  std::vector<BYTE> system;
  std::vector<BYTE> administrators;
  std::vector<BYTE> trusted_installer;
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
        make(WinCreatorOwnerSid, creator_owner) &&
        make_trusted_installer();
  }

  // The accounts whose write access costs this account nothing: itself, and
  // those more privileged than it. TrustedInstaller owns C:\Windows, which holds
  // the profile of a service running as SYSTEM or NetworkService, and without it
  // such a server would compile every kernel on every start.
  bool holds(PSID sid) {
    return sid_equals(sid, user) || sid_equals(sid, system) ||
        sid_equals(sid, administrators) || sid_equals(sid, trusted_installer);
  }

  static bool sid_equals(PSID sid, std::vector<BYTE>& known) {
    return ::EqualSid(sid, known.data()) != FALSE;
  }

 private:
  bool make_trusted_installer() {
    PSID raw = nullptr;
    if (!::ConvertStringSidToSidW(
            L"S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464",
            &raw)) {
      return false;
    }
    const auto* bytes = static_cast<const BYTE*>(raw);
    trusted_installer.assign(bytes, bytes + ::GetLengthSid(raw));
    ::LocalFree(raw);
    return true;
  }

  static bool make(WELL_KNOWN_SID_TYPE type, std::vector<BYTE>& sid) {
    DWORD size = SECURITY_MAX_SID_SIZE;
    sid.resize(size);
    return ::CreateWellKnownSid(type, nullptr, sid.data(), &size) != FALSE;
  }
};

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
    FILE_APPEND_DATA | FILE_WRITE_ATTRIBUTES | FILE_DELETE_CHILD | DELETE |
    WRITE_DAC | WRITE_OWNER | GENERIC_WRITE | GENERIC_ALL;

// Pure: why this owner and DACL make a directory or an entry untrustworthy, as
// the text after its path, or "". Takes both as input, so a foreign owner and a
// null DACL are testable without a second account or a FAT volume.
inline std::string windows_security_refusal(
    PSID owner,
    PACL dacl,
    WindowsTrustedSids& trusted,
    bool leaf) {
  // Whoever owns a directory can rewrite its DACL, so an owner outside the
  // trusted set is a refusal even if the DACL looks private now.
  if (owner == nullptr || !trusted.holds(owner)) {
    return "is not owned by the current user";
  }
  // A null DACL grants everyone full access, and a volume without ACLs (FAT,
  // exFAT) reports one too.
  if (dacl == nullptr) {
    return "has no access control list";
  }
  for (DWORD i = 0; i < dacl->AceCount; ++i) {
    void* raw = nullptr;
    if (!::GetAce(dacl, i, &raw)) {
      return "cannot be inspected";
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
      return "has an access entry this check cannot read";
    }
    PSID sid = const_cast<DWORD*>(
        &static_cast<const ACCESS_ALLOWED_ACE*>(raw)->SidStart);
    // An inherit-only entry does not apply to this directory. On a parent it
    // reaches the next component, which is checked itself. On the leaf it
    // reaches the entries written there, except that CREATOR OWNER turns into
    // their owner.
    if ((header->AceFlags & INHERIT_ONLY_ACE) != 0 &&
        (!leaf ||
         WindowsTrustedSids::sid_equals(sid, trusted.creator_owner))) {
      continue;
    }
    if (::EqualSid(sid, owner) || trusted.holds(sid)) {
      continue;
    }
    return "is writable by " + windows_sid_text(sid);
  }
  return "";
}

} // namespace mlx::core::cu
