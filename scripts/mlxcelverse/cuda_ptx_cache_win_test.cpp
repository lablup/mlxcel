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

// Windows host test of the CUDA PTX cache's trust check (lablup/mlxcel#2797):
// ptx_cache_dir_win.h, with no CUDA and no GPU. The ownership and ACL rule is
// tested on ACLs built in memory, so a foreign owner, a null DACL and generic
// rights need no second account or FAT volume; the walk and the entry read run on
// real directories under argv[1], which must be private, with ACLs set by icacls.
// tests/cuda_ptx_cache_trust_windows.rs covers the same paths through a real GPU
// run. Run by cuda_ptx_cache_win_test.sh.
#include "mlx/backend/cuda/ptx_cache_dir.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace mlx::core::cu;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

bool contains(const std::string& text, const std::string& piece) {
  return text.find(piece) != std::string::npos;
}

constexpr const char* kUsers = "S-1-5-32-545";

// Each right that lets another account plant or replace something, alone.
const struct {
  const char* icacls;
  DWORD mask;
} kWriteRights[] = {
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

#include "cuda_ptx_cache_win_test_rule.inc"

void test_environment() {
  ::SetEnvironmentVariableW(L"MLX_PTX_CACHE_DIR", LR"(C:\set\by\rust)");
  ::SetEnvironmentVariableW(L"MLXCEL_CACHE_DIR", LR"(C:\root)");
  auto env = ptx_cache_env_from_process();
  check(
      env.override_dir == fs::path(LR"(C:\set\by\rust)"),
      "a variable set after startup is read");
  check(env.cache_root == fs::path(LR"(C:\root)"), "the root is read");
  ::SetEnvironmentVariableW(L"MLX_PTX_CACHE_DIR", L"");
  env = ptx_cache_env_from_process();
  check(env.override_dir.empty(), "an empty variable reads as unset");
  ::SetEnvironmentVariableW(L"MLX_PTX_CACHE_DIR", nullptr);
  ::SetEnvironmentVariableW(L"MLXCEL_CACHE_DIR", nullptr);
  env = ptx_cache_env_from_process();
  check(env.override_dir.empty() && env.cache_root.empty(), "unset");
}

void test_long_path_prefix() {
  check(
      without_long_path_prefix(LR"(\\?\C:\x)") == fs::path(LR"(C:\x)"),
      "drive path");
  check(
      without_long_path_prefix(LR"(\\?\UNC\host\share\x)") ==
          fs::path(LR"(\\host\share\x)"),
      "UNC path");
  check(without_long_path_prefix(LR"(C:\x)") == fs::path(LR"(C:\x)"), "plain");
  check(
      with_long_path_prefix(LR"(C:\x)") == fs::path(LR"(\\?\C:\x)"), "add drive");
  check(
      with_long_path_prefix(LR"(\\host\share)") ==
          fs::path(LR"(\\?\UNC\host\share)"),
      "add UNC");
  check(
      with_long_path_prefix(LR"(\\?\C:\x)") == fs::path(LR"(\\?\C:\x)"),
      "already prefixed");
}

void run_or_fail(const std::string& command) {
  const std::string quiet = "\"" + command + " >NUL 2>&1\"";
  check(std::system(quiet.c_str()) == 0, "setup command: " + command);
}

std::string q(const fs::path& path) {
  return "\"" + path.string() + "\"";
}

void grant(const fs::path& path, const std::string& rights) {
  run_or_fail(
      "icacls " + q(path) + " /grant *" + std::string(kUsers) + ":" + rights);
}

fs::path make_dir(const fs::path& path) {
  fs::create_directories(path);
  return path;
}

bool trusted_dir(const fs::path& dir, std::string& reason) {
  fs::path resolved;
  reason.clear();
  return ptx_cache_dir_is_trusted(dir, resolved, reason);
}

void test_walk(const fs::path& base) {
  make_dir(base);
  std::string reason;
  check(trusted_dir(base, reason), "the private base passes: " + reason);

  // Each right on the leaf and on a parent. icacls stores GA and GW mapped to
  // specific rights unless the entry is inherit-only, so those go in that way too;
  // the rule's own test above has them unmapped.
  for (const auto& right : kWriteRights) {
    for (int level = 0; level < 2; ++level) {
      const std::string what = std::string("the right ") + right.icacls +
          " at level " + std::to_string(level);
      const auto top =
          make_dir(base / (std::string("r-") + right.icacls + std::to_string(level)));
      const auto leaf = make_dir(top / "leaf");
      const fs::path& target = level == 0 ? leaf : top;
      grant(target, std::string("(") + right.icacls + ")");
      check(!trusted_dir(leaf, reason), what + ": refused");
      check(contains(reason, kUsers), what + ": names the SID, got: " + reason);
      check(!contains(reason, R"(\\?\)"), what + ": the plain path, got: " + reason);
    }
  }
  for (const char* right : {"GA", "GW"}) {
    const auto leaf = make_dir(base / (std::string("io-") + right));
    grant(leaf, std::string("(OI)(IO)(") + right + ")");
    check(!trusted_dir(leaf, reason), std::string("inherit-only ") + right);
  }

  // Read access, as C:\Users gives everyone, does not refuse.
  {
    const auto leaf = make_dir(base / "read-only");
    grant(leaf, "(OI)(CI)(RX)");
    check(trusted_dir(leaf, reason), "read access passes: " + reason);
  }

  // The service accounts' profiles sit under C:\Windows, owned by TrustedInstaller.
  wchar_t windows[MAX_PATH];
  const UINT size = ::GetWindowsDirectoryW(windows, MAX_PATH);
  check(size != 0 && size < MAX_PATH, "the Windows directory");
  check(trusted_dir(windows, reason), "the Windows directory: " + reason);
  const fs::path profiles = fs::path(windows) / "ServiceProfiles";
  std::error_code error;
  if (fs::exists(profiles, error)) {
    check(trusted_dir(profiles, reason), "ServiceProfiles: " + reason);
  }
}

void test_resolve(const fs::path& base) {
  std::string reason;

  // A private directory, created when missing, and the spellings that reach it.
  {
    const auto named = base / "named" / "cache";
    auto dir = resolve_ptx_cache_dir(named, {}, reason);
    check(!dir.empty(), "a private directory: " + reason);
    check(fs::is_directory(named), "it is created");
    check(contains(dir.string(), R"(\\?\)"), "the result keeps the prefix");
    const auto spelled = fs::path(base.native() + LR"(/named/missing/../cache)");
    check(
        resolve_ptx_cache_dir(spelled, {}, reason) == dir,
        "/ and .. are normalized: " + reason);
    check(!fs::exists(base / "named" / "missing"), ".. creates nothing");
    check(
        resolve_ptx_cache_dir(with_long_path_prefix(named), {}, reason) == dir,
        "a prefixed spelling: " + reason);
  }

  // The tail is created, and a junction at a name in it is refused.
  {
    const auto temp = make_dir(base / "temp");
    const auto dir = resolve_ptx_cache_dir(temp, {"mlx", "1.0", "ptx"}, reason);
    check(!dir.empty(), "the temp fallback: " + reason);
    check(fs::is_directory(temp / "mlx" / "1.0" / "ptx"), "the tail is created");

    const auto other = make_dir(base / "temp-junction");
    const auto target = make_dir(base / "temp-junction-target");
    run_or_fail("mklink /J " + q(other / "mlx") + " " + q(target));
    reason.clear();
    check(
        resolve_ptx_cache_dir(other, {"mlx", "1.0", "ptx"}, reason).empty(),
        "a junction in the tail");
    check(contains(reason, "is a symbolic link"), "says so, got: " + reason);
    check(fs::is_empty(target), "nothing is created through it");
  }

  // Nothing is created under a component that fails.
  {
    const auto top = make_dir(base / "shared");
    grant(top, "(M)");
    reason.clear();
    check(
        resolve_ptx_cache_dir(top / "new" / "cache", {}, reason).empty(),
        "a missing directory under a shared parent");
    check(contains(reason, kUsers), "names the SID, got: " + reason);
    check(!fs::exists(top / "new"), "it is not created");
  }

  // A file in place of the directory.
  {
    const auto file = base / "a-file";
    std::ofstream(file) << "x";
    reason.clear();
    check(resolve_ptx_cache_dir(file, {}, reason).empty(), "a file");
    check(contains(reason, "is not a directory"), "says so, got: " + reason);
  }

  // Past MAX_PATH.
  {
    auto long_dir = base;
    while (long_dir.native().size() < 300) {
      long_dir /= "a-directory-name-of-some-length";
    }
    auto dir = resolve_ptx_cache_dir(long_dir, {}, reason);
    check(!dir.empty(), "past MAX_PATH: " + reason);
    grant(base / "a-directory-name-of-some-length", "(M)");
    reason.clear();
    check(
        resolve_ptx_cache_dir(long_dir, {}, reason).empty(),
        "a shared parent past MAX_PATH");
    check(contains(reason, kUsers), "names the SID, got: " + reason);
  }
}

// The device subdirectory, under a root that passes.
void test_device_dir(const fs::path& base) {
  std::string reason;
  const auto root = make_dir(base / "device-root");
  check(
      !resolve_ptx_cache_subdir(root, "sm121", reason).empty(),
      "a private device directory: " + reason);
  check(fs::is_directory(root / "sm121"), "it is created");

  // A grant on the device directory alone: the root still passes.
  const auto shared = make_dir(root / "shared-device");
  grant(shared, "(M)");
  fs::path resolved;
  reason.clear();
  check(ptx_cache_dir_is_trusted(root, resolved, reason), "the root: " + reason);
  reason.clear();
  check(
      resolve_ptx_cache_subdir(root, "shared-device", reason).empty(),
      "a grant on the device directory itself");
  check(contains(reason, kUsers), "names the SID, got: " + reason);

  const auto target = make_dir(base / "device-target");
  run_or_fail("mklink /J " + q(root / "junction") + " " + q(target));
  reason.clear();
  check(
      resolve_ptx_cache_subdir(root, "junction", reason).empty(),
      "a junction at the device directory");
  check(contains(reason, "is a symbolic link"), "says so, got: " + reason);

  std::ofstream(root / "a-file") << "x";
  reason.clear();
  check(
      resolve_ptx_cache_subdir(root, "a-file", reason).empty(),
      "a file at the device directory");
  check(contains(reason, "is not a directory"), "says so, got: " + reason);
}

void write_file(const fs::path& path, const std::string& text) {
  std::ofstream file(path, std::ios::binary);
  file << text;
}

void test_entries(const fs::path& base) {
  const auto dir = make_dir(base / "entries");
  std::string reason;
  check(trusted_dir(dir, reason), "the directory passes: " + reason);
  std::string contents;

  // As write_cached_ptx publishes: a temporary renamed into place, which holds
  // this process's owner and the directory's inherited ACL.
  const std::string binary("a\0b\r\nc", 6);
  const auto entry = dir / "module.ptx";
  write_file(dir / ".tmp1_1.ptx", binary);
  fs::rename(dir / ".tmp1_1.ptx", entry);
  check(
      read_trusted_cache_entry(entry, contents) && contents == binary,
      "a published entry reads back exactly");
  check(
      read_trusted_cache_entry(with_long_path_prefix(entry), contents) &&
          contents == binary,
      "with the prefix the cache uses");

  const auto empty = dir / "empty.ptx";
  write_file(empty, "");
  check(
      read_trusted_cache_entry(empty, contents) && contents.empty(),
      "an empty file reads as empty, which the caller treats as a miss");

  check(!read_trusted_cache_entry(dir / "missing", contents), "missing");
  check(contents.empty(), "a miss leaves nothing");
  check(!read_trusted_cache_entry(dir, contents), "a directory is not an entry");

  const auto junction_target = make_dir(base / "entry-junction-target");
  run_or_fail("mklink /J " + q(dir / "link.ptx") + " " + q(junction_target));
  check(!read_trusted_cache_entry(dir / "link.ptx", contents), "a junction");
  // A symlink to a private file: creating one needs a privilege or Developer Mode,
  // which the elevated CI runner has.
  if (::CreateSymbolicLinkW(
          (dir / "link-to-file").c_str(),
          (dir / "module.ptx").c_str(),
          SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
    check(
        !read_trusted_cache_entry(dir / "link-to-file", contents),
        "a symlink is no entry, whatever it points at");
  } else {
    std::fprintf(stderr, "skipping the symlink case: no privilege to create one\n");
  }

  // A foreign write entry on the file, in a directory that passes.
  for (const auto& right : kWriteRights) {
    const std::string name = std::string("planted-") + right.icacls;
    write_file(dir / name, "PLANTED");
    check(read_trusted_cache_entry(dir / name, contents), name + " before");
    grant(dir / name, std::string("(") + right.icacls + ")");
    check(trusted_dir(dir, reason), name + ": the directory passes: " + reason);
    check(!read_trusted_cache_entry(dir / name, contents), name + " is refused");
    check(contents.empty(), name + " returns nothing");
  }

  // Inherited from a directory shared when it was written, then made private.
  {
    const auto shared = make_dir(base / "was-shared");
    grant(shared, "(OI)(CI)(M)");
    write_file(shared / "planted.ptx", "PLANTED");
    run_or_fail("icacls " + q(shared) + " /remove *" + kUsers);
    check(trusted_dir(shared, reason), "now private: " + reason);
    // Removing the entry from the directory rewrites what its children inherit,
    // so set the entry the file kept when it was written.
    grant(shared / "planted.ptx", "(M)");
    check(
        !read_trusted_cache_entry(shared / "planted.ptx", contents),
        "an entry from when the directory was shared");

    // Republishing replaces it with one that passes.
    write_file(shared / ".tmp2_2.ptx", binary);
    fs::rename(shared / ".tmp2_2.ptx", shared / "planted.ptx");
    check(
        read_trusted_cache_entry(shared / "planted.ptx", contents) &&
            contents == binary,
        "the republished entry");
  }

  // Read access for others does not refuse.
  write_file(dir / "readable", "x");
  grant(dir / "readable", "(RX)");
  check(read_trusted_cache_entry(dir / "readable", contents), "readable by others");
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <private directory>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const fs::path base = fs::canonical(argv[1]);
  test_security_rule();
  test_environment();
  test_long_path_prefix();
  test_walk(base / "walk");
  test_resolve(base / "resolve");
  test_device_dir(base / "device");
  test_entries(base / "read");
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
