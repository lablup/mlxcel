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

// Host-only tests for the ROCm JIT kernel disk cache (lablup/mlxcel#2641): the
// location and trust logic of the overlay's hsaco_cache_dir.h, and the read and
// write of entries, get_hsaco_path(), read_cached_hsaco() and
// write_cached_hsaco(), copied out of jit_module.cpp by rocm_hsaco_cache_test.sh.
//
// A different owner needs no second account: the trust check takes the euid, so
// passing another uid makes a real directory owned by this user "owned by
// someone else" (as root, the directory is chowned instead, since root-owned
// components are trusted). Everything else is a real directory or file under
// the base directory given as argv[1], which must be private.
#include "mlx/backend/rocm/hsaco_cache_dir.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mlx::core::rocm {
#include "hsaco_cache_read.inc"
} // namespace mlx::core::rocm

namespace fs = std::filesystem;
using namespace mlx::core::rocm;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

constexpr const char* kVersion = "0.0.0-test";
constexpr const char* kArch = "gfx1151";
constexpr const char* kModule = "planted_module";
constexpr std::uint32_t kMe = 1000;

void test_path_choice() {
  const fs::path over = "over";
  const fs::path root = "root";
  const fs::path home = "home";
  const fs::path tail = fs::path("hsaco") / kVersion / kArch;
  check(
      hsaco_cache_path({over, root, home}, kVersion, kArch) == over / kArch,
      "MLX_HSACO_CACHE_DIR wins and gets the arch");
  check(
      hsaco_cache_path({{}, root, home}, kVersion, kArch) == root / tail,
      "MLXCEL_CACHE_DIR is the root of the default");
  check(
      hsaco_cache_path({{}, {}, home}, kVersion, kArch) ==
          home / ".cache" / "mlxcel" / tail,
      "the home directory gives the per-user default");
  check(
      hsaco_cache_path({{}, {}, {}}, kVersion, kArch).empty(),
      "nothing names a root: no candidate, never the temp directory");
}

void test_component_rules() {
  auto refusal = [](bool link, bool dir, std::uint32_t uid, std::uint32_t mode) {
    return cache_component_refusal({link, dir, uid, mode}, kMe, false);
  };
  check(refusal(false, true, kMe, 040700) == nullptr, "own private directory");
  check(refusal(false, true, 0, 040755) == nullptr, "root-owned parent");
  check(
      refusal(false, true, 0, 041777) == nullptr,
      "root-owned sticky directory (/tmp)");
  check(
      refusal(false, true, 0, 040777) != nullptr,
      "root-owned but world-writable and not sticky");
  check(
      refusal(false, true, kMe + 1, 040700) != nullptr,
      "owned by another user, even private");
  check(
      refusal(false, true, kMe + 1, 041777) != nullptr,
      "sticky does not excuse a non-root owner");
  check(
      refusal(false, true, kMe, 041777) != nullptr,
      "sticky does not excuse the current user's own shared directory");
  check(refusal(false, true, kMe, 040770) != nullptr, "group-writable");
  check(refusal(false, true, kMe, 040702) != nullptr, "other-writable");
  check(refusal(true, false, kMe, 0120700) != nullptr, "a symlink component");
  check(
      cache_component_refusal({false, false, kMe, 0100700}, kMe, true) !=
          nullptr,
      "the leaf must be a directory");
}

bool blob_loaded(const fs::path& dir, const std::string* expected_source) {
  std::string hsaco;
  std::vector<std::pair<std::string, std::string>> kernels;
  return read_cached_hsaco(dir, kModule, hsaco, kernels, expected_source);
}

// Publishing a cache round trip works wherever the platform allows it.
void test_write_then_read(const fs::path& base) {
  const HsacoCacheEnv env{base / "roundtrip", {}, {}};
  const auto resolved =
      resolve_hsaco_cache_dir(env, kVersion, kArch, cache_current_euid());
  check(!resolved.dir.empty(), "round trip dir: " + resolved.refusal);
  if (resolved.dir.empty()) {
    return;
  }
  const std::string source = "kernel source";
  const std::vector<std::pair<std::string, std::string>> names = {
      {"kernel", "mangled"}, {"other", "mangled_other"}};
  write_cached_hsaco(resolved.dir, kModule, "BLOB", names, source);
  std::string hsaco;
  std::vector<std::pair<std::string, std::string>> kernels;
  check(
      read_cached_hsaco(resolved.dir, kModule, hsaco, kernels, &source),
      "a written entry reads back");
  check(hsaco == "BLOB", "the written blob comes back");
  check(kernels == names, "the kernel map comes back");
  const std::string changed = "changed source";
  check(
      !blob_loaded(resolved.dir, &changed),
      "a source change is still a miss");
  check(
      !blob_loaded(fs::path(), &source),
      "an empty cache directory (the cache is off) never loads anything");
  fs::remove(resolved.dir / (std::string(kModule) + ".txt"));
  check(!blob_loaded(resolved.dir, &source), "an entry without .txt is a miss");
}

#if !defined(_WIN32)

void write_file(const fs::path& path, const std::string& contents) {
  std::ofstream file(path, std::ios::binary);
  file << contents;
}

// Plant a complete entry for kModule, as an attacker would.
void plant_entry(const fs::path& dir, const std::string& source) {
  fs::create_directories(dir);
  write_file(dir / (std::string(kModule) + ".hsaco"), "PLANTED");
  write_file(dir / (std::string(kModule) + ".txt"), "kernel\tmangled\n");
  write_file(dir / (std::string(kModule) + ".hip"), source);
}

unsigned mode_of(const fs::path& path) {
  struct stat st;
  return ::lstat(path.c_str(), &st) == 0 ? (st.st_mode & 07777) : 0;
}

fs::path private_dir(const fs::path& path, unsigned mode = 0700) {
  fs::create_directories(path);
  ::chmod(path.c_str(), mode);
  return path;
}

// Under the private base: refused directories, nothing created or loaded.
void test_refused_directories(const fs::path& base) {
  const auto euid = cache_current_euid();
  const std::string source = "src";

  // Control: the same planted entry in a private directory does load, so the
  // refusals below come from the trust check rather than from the test setup.
  {
    const auto over = private_dir(base / "control");
    private_dir(over / kArch);
    plant_entry(over / kArch, source);
    ::chmod((over / kArch).c_str(), 0700);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(!got.dir.empty(), "control dir is used: " + got.refusal);
    check(got.refusal.empty(), "control dir has no refusal");
    check(blob_loaded(got.dir, &source), "control blob is loaded");
  }

  const struct {
    const char* name;
    unsigned leaf_mode;
    unsigned parent_mode;
  } modes[] = {
      {"world-writable leaf", 0777, 0700},
      {"group-writable leaf", 0770, 0700},
      {"world-writable override root", 0700, 0777},
      {"group-writable override root", 0700, 0770},
      {"sticky world-writable leaf (not root-owned)", 01777, 0700},
  };
  int index = 0;
  for (const auto& c : modes) {
    if (euid == 0 && (c.leaf_mode & 01000) != 0) {
      continue; // root-owned and sticky is the /tmp case, which is trusted
    }
    const auto over = private_dir(base / ("mode" + std::to_string(index++)));
    const auto leaf = private_dir(over / kArch);
    plant_entry(leaf, source);
    ::chmod(leaf.c_str(), c.leaf_mode);
    ::chmod(over.c_str(), c.parent_mode);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    const std::string what = c.name;
    check(got.dir.empty(), what + ": refused");
    check(!got.refusal.empty(), what + ": says why");
    check(
        got.refusal.find("group- or other-writable") != std::string::npos,
        what + ": names the mode, got: " + got.refusal);
    check(!blob_loaded(got.dir, &source), what + ": no blob loaded");
    check(mode_of(leaf) == c.leaf_mode, what + ": leaf mode not changed");
    check(mode_of(over) == c.parent_mode, what + ": parent mode not changed");
    ::chmod(leaf.c_str(), 0700);
    ::chmod(over.c_str(), 0700);
  }

  // A pre-created directory owned by another user. Owner rules are the real
  // walk with real lstat results: only the identity asked about differs.
  {
    const auto over = private_dir(base / "foreign");
    const auto leaf = private_dir(over / kArch);
    plant_entry(leaf, source);
    std::uint32_t asking = euid + 1;
    if (euid == 0) {
      // Root-owned components are trusted, so as root the directory itself
      // has to change hands.
      check(::chown(leaf.c_str(), 65534, 65534) == 0, "chown as root");
      asking = 0;
    }
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, asking);
    check(got.dir.empty(), "foreign owner: refused");
    check(
        got.refusal.find("is not owned by the current user") !=
            std::string::npos,
        "foreign owner: names the owner, got: " + got.refusal);
    check(!blob_loaded(got.dir, &source), "foreign owner: no blob loaded");
    if (euid == 0) {
      check(::chown(leaf.c_str(), 0, 0) == 0, "chown back as root");
    }
  }

  // A symlink to a world-writable directory is judged by its target.
  {
    const auto target = private_dir(base / "link-target");
    private_dir(target / kArch);
    ::chmod((target / kArch).c_str(), 0777);
    const auto over = private_dir(base / "link-over");
    fs::create_directory_symlink(target / kArch, over / kArch);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "symlink to a shared directory: refused");
    ::chmod((target / kArch).c_str(), 0700);
  }

  // A regular file where the cache directory should be.
  {
    const auto over = private_dir(base / "file-leaf");
    write_file(over / kArch, "not a directory");
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "a file at the leaf: refused");
    check(
        got.refusal.find("is not a directory") != std::string::npos,
        "a file at the leaf: says so, got: " + got.refusal);
  }
}

// The per-user default, under a umask that would make everything group-writable.
void test_default_location(const fs::path& base) {
  const auto euid = cache_current_euid();
  const mode_t old_umask = ::umask(002);

  // Fresh: every component it creates is 0700, the path is under the home.
  {
    const auto home = private_dir(base / "home-fresh");
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    const auto expected = fs::canonical(home) / ".cache" / "mlxcel" / "hsaco" /
        kVersion / kArch;
    check(got.dir == expected, "default path: " + got.dir.string());
    check(got.refusal.empty(), "default path has no refusal: " + got.refusal);
    for (auto p = expected; p != fs::canonical(home); p = p.parent_path()) {
      check(mode_of(p) == 0700, "created private: " + p.string());
    }
  }

  // Directories an earlier release made under this umask are repaired, the
  // ones mlxcel manages only: the root, hsaco, <version> and <arch>.
  {
    const auto home = private_dir(base / "home-repair");
    const auto dot_cache = private_dir(home / ".cache", 0755);
    const auto root = private_dir(dot_cache / "mlxcel", 0775);
    const auto hsaco = private_dir(root / "hsaco", 0775);
    const auto ver = private_dir(hsaco / kVersion, 02775);
    const auto arch = private_dir(ver / kArch, 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(!got.dir.empty(), "repaired default is used: " + got.refusal);
    check(mode_of(root) == 0755, "root repaired");
    check(mode_of(hsaco) == 0755, "hsaco repaired");
    check(mode_of(ver) == 02755, "version repaired, other bits kept");
    check(mode_of(arch) == 0755, "arch repaired");
  }

  // Nothing above the root is touched, so a group-writable ~/.cache is refused.
  {
    const auto home = private_dir(base / "home-shared-cache");
    const auto dot_cache = private_dir(home / ".cache", 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, {}, home}, kVersion, kArch, euid);
    check(got.dir.empty(), "group-writable ~/.cache: refused");
    check(mode_of(dot_cache) == 0775, "~/.cache not changed");
  }

  // An explicit MLXCEL_CACHE_DIR is not repaired, but what mlxcel keeps under
  // it is.
  {
    const auto root = private_dir(base / "explicit-root", 0775);
    const auto hsaco = private_dir(root / "hsaco", 0775);
    const auto got =
        resolve_hsaco_cache_dir({{}, root, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "group-writable MLXCEL_CACHE_DIR: refused");
    check(mode_of(root) == 0775, "MLXCEL_CACHE_DIR not changed");
    check(mode_of(hsaco) == 0755, "hsaco under it repaired");
  }

  // An explicit override is never repaired.
  {
    const auto over = private_dir(base / "explicit-over");
    const auto leaf = private_dir(over / kArch, 0775);
    const auto got =
        resolve_hsaco_cache_dir({over, {}, {}}, kVersion, kArch, euid);
    check(got.dir.empty(), "group-writable override: refused");
    check(mode_of(leaf) == 0775, "override not changed");
  }

  // A `..` creates no stray directory.
  {
    const auto root = private_dir(base / "dotdot");
    const auto got = resolve_hsaco_cache_dir(
        {root / "missing" / ".." / "cache", {}, {}}, kVersion, kArch, euid);
    check(!got.dir.empty(), "`..` in the override: " + got.refusal);
    check(!fs::exists(root / "missing"), "`..` left no stray directory");
  }

  // A symlink replacing a checked directory is not chmodded through.
  {
    const auto target = private_dir(base / "chmod-target", 0775);
    const auto link = base / "chmod-link";
    fs::create_directory_symlink(target, link);
    remove_shared_write(link, euid);
    check(mode_of(target) == 0775, "repair does not follow a symlink");
    const auto foreign = private_dir(base / "chmod-foreign", 0775);
    remove_shared_write(foreign, euid + 1);
    check(mode_of(foreign) == 0775, "repair leaves another owner's directory");
  }

  ::umask(old_umask);
}

// Entries inside a trusted directory.
void test_entries(const fs::path& base) {
  const auto euid = cache_current_euid();
  const mode_t old_umask = ::umask(002);
  const std::string source = "src";
  const auto dir = private_dir(base / "entries");
  const auto entry = [&](const char* ext) {
    return dir / (std::string(kModule) + ext);
  };

  // Written entries are private whatever the umask, and read back.
  const std::vector<std::pair<std::string, std::string>> names = {{"k", "m"}};
  write_cached_hsaco(dir, kModule, "BLOB", names, source);
  for (const char* ext : {".hsaco", ".txt", ".hip"}) {
    check(mode_of(entry(ext)) == 0600, std::string(ext) + " is 0600");
  }
  check(blob_loaded(dir, &source), "a private entry loads");

  // Any shared-write entry is a miss: the blob, the map and the source.
  for (const char* ext : {".hsaco", ".txt", ".hip"}) {
    ::chmod(entry(ext).c_str(), 0664);
    check(!blob_loaded(dir, &source), std::string(ext) + " group-writable: miss");
    ::chmod(entry(ext).c_str(), 0666);
    check(!blob_loaded(dir, &source), std::string(ext) + " world-writable: miss");
    ::chmod(entry(ext).c_str(), 0600);
  }
  check(blob_loaded(dir, &source), "restored entry loads again");

  // An entry owned by someone else is a miss (euid is the only input).
  {
    std::string contents;
    check(
        !read_trusted_cache_entry(
            entry(".hsaco"), contents, euid == 0 ? 1 : euid + 1) ||
            euid == 0,
        "an entry owned by another user is a miss");
    check(
        read_trusted_cache_entry(entry(".hsaco"), contents, euid),
        "the owner's entry reads");
    check(contents == "BLOB", "the entry's contents");
  }

  // A symlink or a FIFO at the entry's name is a miss and does not block.
  {
    const auto outside = base / "outside.hsaco";
    write_file(outside, "PLANTED");
    ::chmod(outside.c_str(), 0600);
    fs::remove(entry(".hsaco"));
    fs::create_symlink(outside, entry(".hsaco"));
    check(!blob_loaded(dir, &source), "a symlinked entry: miss");

    // A publish replaces the link instead of writing through it.
    write_cached_hsaco(dir, kModule, "BLOB", names, source);
    check(!fs::is_symlink(entry(".hsaco")), "a publish replaces the symlink");
    std::string outside_after;
    {
      std::ifstream file(outside, std::ios::binary);
      std::getline(file, outside_after);
    }
    check(outside_after == "PLANTED", "a publish does not write through it");
    check(blob_loaded(dir, &source), "the replacement loads");

    fs::remove(entry(".hsaco"));
    check(::mkfifo(entry(".hsaco").c_str(), 0600) == 0, "mkfifo");
    check(!blob_loaded(dir, &source), "a FIFO entry: miss, no hang");
    fs::remove(entry(".hsaco"));
  }

  // An older, wider entry is replaced by the next publish, not kept.
  {
    write_file(entry(".hsaco"), "OLD");
    ::chmod(entry(".hsaco").c_str(), 0664);
    write_cached_hsaco(dir, kModule, "BLOB", names, source);
    check(mode_of(entry(".hsaco")) == 0600, "a publish narrows the entry");
    check(blob_loaded(dir, &source), "the republished entry loads");
  }

  // No publish temporaries are left behind.
  for (const auto& e : fs::directory_iterator(dir)) {
    check(
        e.path().filename().string().rfind(".hsaco-tmp-", 0) != 0,
        "stray temporary: " + e.path().string());
  }
  ::umask(old_umask);
}

#endif // !_WIN32

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <private base directory>\n", argv[0]);
    return 2;
  }
  const fs::path base = fs::canonical(argv[1]);
#if !defined(_WIN32)
  // Files the test plants must not come out group-writable by accident.
  ::umask(022);
#endif
  test_path_choice();
  test_component_rules();
#if !defined(_WIN32)
  fs::path trusted;
  std::string reason;
  if (!cache_dir_is_trusted(base, cache_current_euid(), trusted, reason)) {
    std::fprintf(
        stderr,
        "FAIL: the test base directory is itself untrusted, so no case below "
        "would mean anything: %s. Set TMPDIR to a private directory.\n",
        reason.c_str());
    return 1;
  }
  test_refused_directories(base);
  test_default_location(base);
  test_entries(base);
#endif
  test_write_then_read(base);
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  return 0;
}
