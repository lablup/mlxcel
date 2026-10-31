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
  test_home_directory();
}
