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

// The ownership and ACL rule of cuda_ptx_cache_win_test.cpp, on ACLs built in
// memory, so a foreign owner, a null DACL and unmapped generic rights need no second
// account or FAT volume.

std::vector<BYTE> well_known(WELL_KNOWN_SID_TYPE type) {
  std::vector<BYTE> sid(SECURITY_MAX_SID_SIZE);
  DWORD size = SECURITY_MAX_SID_SIZE;
  check(::CreateWellKnownSid(type, nullptr, sid.data(), &size) != FALSE, "SID");
  return sid;
}

// An ACL holding one entry, in `storage`.
PACL acl_with(
    std::vector<BYTE>& storage,
    bool allow,
    PSID sid,
    DWORD mask,
    DWORD flags = 0) {
  storage.assign(1024, 0);
  auto* acl = reinterpret_cast<PACL>(storage.data());
  check(::InitializeAcl(acl, 1024, ACL_REVISION) != FALSE, "InitializeAcl");
  const BOOL added = allow
      ? ::AddAccessAllowedAceEx(acl, ACL_REVISION, flags, mask, sid)
      : ::AddAccessDeniedAceEx(acl, ACL_REVISION, flags, mask, sid);
  check(added != FALSE, "add ACE");
  return acl;
}

void test_security_rule() {
  WindowsTrustedSids trusted;
  check(trusted.ok, "the trusted SIDs resolve");
  auto users = well_known(WinBuiltinUsersSid);
  auto everyone = well_known(WinWorldSid);
  PSID me = trusted.user.data();
  std::vector<BYTE> storage;
  auto* empty = acl_with(storage, false, users.data(), 0);

  // Owner.
  check(
      windows_security_refusal(nullptr, empty, trusted, true) ==
          "is not owned by the current user",
      "no owner");
  check(
      windows_security_refusal(users.data(), empty, trusted, true) ==
          "is not owned by the current user",
      "a foreign owner, whatever the ACL says");
  for (std::vector<BYTE>* owner :
       {&trusted.user,
        &trusted.system,
        &trusted.administrators,
        &trusted.trusted_installer}) {
    check(
        windows_security_refusal(owner->data(), empty, trusted, true).empty(),
        "a trusted owner");
  }

  // A null DACL grants everyone full access.
  check(
      windows_security_refusal(me, nullptr, trusted, true) ==
          "has no access control list",
      "a null DACL");

  // Each right alone, for another account, on a leaf and on a parent.
  for (const auto& right : kWriteRights) {
    const std::string what = std::string("the right ") + right.icacls;
    auto* acl = acl_with(storage, true, users.data(), right.mask);
    const std::string refusal = windows_security_refusal(me, acl, trusted, true);
    check(contains(refusal, kUsers), what + " on a leaf, got: " + refusal);
    check(
        contains(windows_security_refusal(me, acl, trusted, false), kUsers),
        what + " on a parent");
    // For an account the check trusts, nothing is wrong.
    acl = acl_with(storage, true, trusted.trusted_installer.data(), right.mask);
    check(
        windows_security_refusal(me, acl, trusted, true).empty(),
        what + " for TrustedInstaller");
    acl = acl_with(storage, true, me, right.mask);
    check(
        windows_security_refusal(me, acl, trusted, true).empty(),
        what + " for this account");
  }

  // Rights that plant nothing, and a deny entry.
  auto* acl = acl_with(
      storage, true, everyone.data(), FILE_GENERIC_READ | FILE_GENERIC_EXECUTE);
  check(windows_security_refusal(me, acl, trusted, true).empty(), "read access");
  acl = acl_with(storage, true, everyone.data(), GENERIC_READ | GENERIC_EXECUTE);
  check(
      windows_security_refusal(me, acl, trusted, true).empty(),
      "generic read access");
  acl = acl_with(storage, false, everyone.data(), GENERIC_ALL);
  check(windows_security_refusal(me, acl, trusted, true).empty(), "a deny entry");

  // An inherit-only entry reaches what is written in a leaf, and on a parent
  // reaches the next component, which is checked itself.
  acl = acl_with(
      storage,
      true,
      users.data(),
      GENERIC_WRITE,
      OBJECT_INHERIT_ACE | INHERIT_ONLY_ACE);
  check(
      contains(windows_security_refusal(me, acl, trusted, true), kUsers),
      "an inherit-only entry on a leaf");
  check(
      windows_security_refusal(me, acl, trusted, false).empty(),
      "an inherit-only entry on a parent");
  acl = acl_with(
      storage,
      true,
      trusted.creator_owner.data(),
      GENERIC_ALL,
      OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE | INHERIT_ONLY_ACE);
  check(
      windows_security_refusal(me, acl, trusted, true).empty(),
      "CREATOR OWNER turns into the entry's owner");
}
