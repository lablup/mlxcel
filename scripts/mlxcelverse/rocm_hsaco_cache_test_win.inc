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

// The Windows cases of rocm_hsaco_cache_test.cpp: real directories whose ACLs
// are set with icacls, as an administrator or the inheritance from a drive would
// leave them.

// Run a command through icacls or cmd; a failed setup fails the test, or a
// refusal could come from anything.
void run_or_fail(const std::string& command) {
  const std::string quiet = "\"" + command + " >NUL 2>&1\"";
  check(std::system(quiet.c_str()) == 0, "setup command: " + command);
}

std::string q(const fs::path& path) {
  return "\"" + path.string() + "\"";
}

fs::path make_dir(const fs::path& path) {
  fs::create_directories(path);
  return path;
}

void plant_entry(const fs::path& dir, const std::string& source) {
  fs::create_directories(dir);
  const auto write = [&](const char* ext, const std::string& text) {
    std::ofstream file(dir / (std::string(kModule) + ext), std::ios::binary);
    file << text;
  };
  write(".hsaco", "PLANTED");
  write(".txt", "kernel\tmangled\n");
  write(".hip", source);
}

void test_acls(const fs::path& base) {
  const auto euid = cache_current_euid();
  const std::string source = "src";

  // Control: the same planted entry in a private directory does load, so the
  // refusals below come from the check rather than from the test setup.
  {
    const auto over = make_dir(base / "control");
    plant_entry(over / kArch, source);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.refusal.empty(), "a private directory passes: " + got.refusal);
    check(blob_loaded(got.dir, &source), "control blob is loaded");
  }

  // S-1-5-32-545 is BUILTIN\Users, S-1-1-0 is Everyone, S-1-5-11 is Authenticated
  // Users. Well-known SIDs keep this independent of the display language.
  const struct {
    const char* name;
    const char* sid;
    const char* right;
  } grants[] = {
      {"users-modify", "S-1-5-32-545", "(OI)(CI)M"},
      {"users-write", "S-1-5-32-545", "(W)"},
      {"everyone-full", "S-1-1-0", "(OI)(CI)F"},
      {"auth-users-modify", "S-1-5-11", "(OI)(CI)M"},
      {"users-add-subdir", "S-1-5-32-545", "(AD)"},
      {"users-delete-child", "S-1-5-32-545", "(DC)"},
      {"users-write-dac", "S-1-5-32-545", "(WDAC)"},
      {"users-write-data", "S-1-5-32-545", "(WD)"},
      {"users-write-attributes", "S-1-5-32-545", "(WA)"},
      {"users-delete", "S-1-5-32-545", "(D)"},
      {"users-write-owner", "S-1-5-32-545", "(WO)"},
      {"users-generic-all", "S-1-5-32-545", "(GA)"},
      {"users-generic-write", "S-1-5-32-545", "(GW)"},
  };
  for (const auto& grant : grants) {
    const std::string acl = std::string("*") + grant.sid + ":" + grant.right;
    // On the leaf, then on the directory the user named, then one level above it.
    for (int level = 0; level < 3; ++level) {
      const std::string what =
          std::string(grant.name) + " at level " + std::to_string(level);
      const auto top =
          make_dir(base / (std::string(grant.name) + std::to_string(level)));
      const auto over = make_dir(top / "named");
      const auto leaf = make_dir(over / kArch);
      plant_entry(leaf, source);
      const fs::path& target = level == 0 ? leaf : level == 1 ? over : top;
      run_or_fail("icacls " + q(target) + " /grant " + acl);
      const auto got =
          resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
      check(got.dir.empty(), what + ": refused");
      check(
          got.refusal.find("is writable by") != std::string::npos,
          what + ": names the entry, got: " + got.refusal);
      check(
          got.refusal.find(grant.sid) != std::string::npos,
          what + ": names the SID, got: " + got.refusal);
      check(!blob_loaded(got.dir, &source), what + ": no blob loaded");
    }
  }

  // Read access for others is not a reason to refuse.
  {
    const auto over = make_dir(base / "users-read");
    const auto leaf = make_dir(over / kArch);
    run_or_fail("icacls " + q(leaf) + " /grant *S-1-5-32-545:(OI)(CI)RX");
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.refusal.empty(), "a read entry passes: " + got.refusal);
  }

  // An inherit-only write entry on the leaf reaches the entries written there.
  {
    const auto over = make_dir(base / "inherit-only");
    const auto leaf = make_dir(over / kArch);
    run_or_fail("icacls " + q(leaf) + " /grant *S-1-5-32-545:(OI)(IO)M");
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "an inherit-only write entry on the leaf: refused");
  }

  // A refused run creates nothing under the refused ancestor.
  {
    const auto top = make_dir(base / "no-leftovers");
    run_or_fail("icacls " + q(top) + " /grant *S-1-5-32-545:(OI)(CI)M");
    const auto got = resolve_hsaco_cache_dir(
        {top / "named" / "dirs", {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "a shared ancestor of a missing path: refused");
    check(!fs::exists(top / "named"), "a refused run leaves no directories");
  }

  // A junction at a component only mlxcel creates is refused, even to a private
  // target. (mklink /J needs no privilege, unlike a symlink.)
  for (const char* tail : {"arch", "hsaco"}) {
    const std::string tag = tail;
    const auto target = make_dir(base / ("junction-target-" + tag));
    plant_entry(target / kVersion / kArch, source);
    plant_entry(target / kArch, source);
    const auto root = make_dir(base / ("junction-root-" + tag));
    fs::path link;
    HsacoCacheEnv env;
    if (tag == "arch") {
      env = {root, {}, {}};
      link = root / kArch;
      run_or_fail("mklink /J " + q(link) + " " + q(target / kArch));
    } else {
      env = {{}, root, {}};
      link = root / "hsaco";
      run_or_fail("mklink /J " + q(link) + " " + q(target));
    }
    const auto got = resolve_hsaco_cache_dir(env, kVersion, kArch, euid);
    check(got.dir.empty(), "a junction at " + tag + ": refused");
    check(
        got.refusal.find("is a symbolic link") != std::string::npos,
        "a junction at " + tag + ": says so, got: " + got.refusal);
    check(!blob_loaded(got.dir, &source), "a junction at " + tag + ": no blob");
  }

  // A named directory that is a junction is judged by its target.
  {
    const auto target = make_dir(base / "named-junction-target");
    const auto link = base / "named-junction";
    run_or_fail("mklink /J " + q(link) + " " + q(target));
    const auto got =
        resolve_hsaco_cache_dir({link, {}, {}}, kVersion, kArch, euid);
    check(got.refusal.empty(), "a named junction passes: " + got.refusal);
  }

  // A file too small or missing reads as a miss; a sized read returns the bytes.
  {
    const auto dir = make_dir(base / "sized-read");
    std::string contents;
    check(
        !read_trusted_cache_entry(dir / "missing", contents),
        "a missing entry is a miss");
    check(
        !read_trusted_cache_entry(dir, contents),
        "a directory is not an entry");
    std::string binary("a\0b\r\nc", 6);
    {
      std::ofstream file(dir / "entry", std::ios::binary);
      file << binary;
    }
    check(
        read_trusted_cache_entry(dir / "entry", contents) && contents == binary,
        "a binary entry reads back exactly");
  }
}

std::vector<BYTE> well_known(WELL_KNOWN_SID_TYPE type) {
  std::vector<BYTE> sid(SECURITY_MAX_SID_SIZE);
  DWORD size = SECURITY_MAX_SID_SIZE;
  check(::CreateWellKnownSid(type, nullptr, sid.data(), &size) != FALSE, "SID");
  return sid;
}

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

// The ownership and ACL rule on ACLs built in memory, so a foreign owner, a null
// DACL and unmapped generic rights need no second account or FAT volume.
void test_security_rule() {
  WindowsTrustedSids trusted;
  check(trusted.ok, "the trusted SIDs resolve");
  auto users = well_known(WinBuiltinUsersSid);
  PSID me = trusted.user.data();
  std::vector<BYTE> storage;
  auto* empty = acl_with(storage, false, users.data(), 0);
  check(
      windows_security_refusal(users.data(), empty, trusted, true) ==
          "is not owned by the current user",
      "a foreign owner, whatever the ACL says");
  check(
      windows_security_refusal(nullptr, empty, trusted, true) ==
          "is not owned by the current user",
      "no owner");
  for (std::vector<BYTE>* owner :
       {&trusted.user,
        &trusted.system,
        &trusted.administrators,
        &trusted.trusted_installer}) {
    check(
        windows_security_refusal(owner->data(), empty, trusted, true).empty(),
        "a trusted owner");
  }
  check(
      windows_security_refusal(me, nullptr, trusted, true) ==
          "has no access control list",
      "a null DACL");
  const struct {
    const char* name;
    DWORD mask;
  } rights[] = {
      {"WD", FILE_WRITE_DATA},
      {"AD", FILE_APPEND_DATA},
      {"WA", FILE_WRITE_ATTRIBUTES},
      {"DC", FILE_DELETE_CHILD},
      {"D", DELETE},
      {"WDAC", WRITE_DAC},
      {"WO", WRITE_OWNER},
      {"GA", GENERIC_ALL},
      {"GW", GENERIC_WRITE},
  };
  for (const auto& right : rights) {
    auto* acl = acl_with(storage, true, users.data(), right.mask);
    check(
        windows_security_refusal(me, acl, trusted, true).find("S-1-5-32-545") !=
            std::string::npos,
        std::string("the right ") + right.name);
    acl = acl_with(storage, true, trusted.trusted_installer.data(), right.mask);
    check(
        windows_security_refusal(me, acl, trusted, true).empty(),
        std::string("TrustedInstaller holding ") + right.name);
  }
  auto* acl = acl_with(
      storage, true, users.data(), FILE_GENERIC_READ | FILE_GENERIC_EXECUTE);
  check(windows_security_refusal(me, acl, trusted, true).empty(), "read access");
}

// An entry is checked on its own: a directory that was shared and later made
// private passes its check and can still hold another account's entries.
void test_entry_acls(const fs::path& base) {
  const auto euid = cache_current_euid();
  const std::string source = "src";
  const auto over = make_dir(base / "entry-acls");
  const auto leaf = over / kArch;
  // Written and renamed as write_cached_hsaco does, so it inherits a private ACL.
  plant_entry(leaf, source);
  const auto got = resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
  check(got.refusal.empty(), "the directory passes: " + got.refusal);
  check(blob_loaded(got.dir, &source), "control: a private entry loads");

  for (const char* ext : {".hsaco", ".txt", ".hip"}) {
    for (const char* right : {"(WD)", "(M)", "(WDAC)", "(WA)"}) {
      const std::string what = std::string(ext) + " with " + right;
      plant_entry(leaf, source);
      const auto file = leaf / (std::string(kModule) + ext);
      run_or_fail("icacls " + q(file) + " /grant *S-1-5-32-545:" + right);
      const auto again =
          resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
      check(again.refusal.empty(), what + ": the directory passes: " + again.refusal);
      check(!blob_loaded(again.dir, &source), what + ": no blob loaded");
      // Republishing replaces the file, and the ACL with it.
      fs::remove(file);
      plant_entry(leaf, source);
      check(blob_loaded(again.dir, &source), what + ": a republished entry loads");
    }
  }
  std::string contents;
  run_or_fail("mklink /J " + q(leaf / "link") + " " + q(base));
  check(!read_trusted_cache_entry(leaf / "link", contents), "a junction is no entry");
  // A symlink to a private file: creating one needs a privilege or Developer Mode,
  // which the elevated CI runner has.
  if (::CreateSymbolicLinkW(
          (leaf / "link-to-file").c_str(),
          (leaf / "planted_module.hsaco").c_str(),
          SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
    check(
        !read_trusted_cache_entry(leaf / "link-to-file", contents),
        "a symlink is no entry, whatever it points at");
  } else {
    std::fprintf(stderr, "skipping the symlink case: no privilege to create one\n");
  }
}

// The profile of a service running as SYSTEM or NetworkService is under
// C:\Windows, owned by TrustedInstaller.
void test_windows_directory() {
  wchar_t windows[MAX_PATH];
  const UINT size = ::GetWindowsDirectoryW(windows, MAX_PATH);
  check(size != 0 && size < MAX_PATH, "the Windows directory");
  fs::path resolved;
  std::string reason;
  check(
      cache_dir_is_trusted(windows, cache_current_euid(), resolved, reason),
      "the Windows directory passes: " + reason);
}

// The cache root must agree with Rust's cache_root(): dirs::home_dir reads the
// Known Folder profile, not the USERPROFILE variable.
void test_home_directory() {
  const auto profile = windows_profile_dir();
  check(profile.is_absolute(), "the Known Folder profile is absolute");
  const std::wstring saved = _wgetenv(L"USERPROFILE") ? _wgetenv(L"USERPROFILE") : L"";
  _wputenv_s(L"USERPROFILE", L"C:\\mlxcel-overridden-profile");
  const auto env = hsaco_cache_env_from_process();
  check(
      env.home == windows_profile_dir(),
      "the home directory comes from the Known Folder API");
  check(
      env.home != fs::path(L"C:\\mlxcel-overridden-profile"),
      "overriding USERPROFILE moves neither side");
  _wputenv_s(L"USERPROFILE", saved.c_str());
}

void test_platform(const fs::path& base) {
  test_acls(base);
  test_security_rule();
  test_entry_acls(base);
  test_windows_directory();
  test_home_directory();
}
