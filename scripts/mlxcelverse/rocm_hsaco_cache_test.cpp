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
// the base directory given as argv[1], which must be private. On Windows the
// ACL cases run icacls on real directories.
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
#include <sys/types.h>
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

  // A relative path would resolve against the current directory.
  const auto abs = fs::canonical(fs::current_path());
  const auto refused = [](const HsacoCacheEnv& env) {
    const auto got = resolve_hsaco_cache_dir(
        env, kVersion, kArch, cache_current_euid());
    return got.dir.empty() ? got.refusal : std::string();
  };
  check(
      refused({over, {}, abs}).find("MLX_HSACO_CACHE_DIR must be an absolute") !=
          std::string::npos,
      "a relative MLX_HSACO_CACHE_DIR is refused");
  check(
      refused({{}, root, abs}).find("MLXCEL_CACHE_DIR must be an absolute") !=
          std::string::npos,
      "a relative MLXCEL_CACHE_DIR is refused");
  check(
      refused({{}, {}, home}).find("the home directory must be an absolute") !=
          std::string::npos,
      "a relative home directory is refused");
  check(
      refused({{}, {}, {}}).rfind("no directory:", 0) == 0,
      "nothing names a root: refused with the reason");
  check(!fs::exists(over), "a refused relative path creates nothing");
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

  // Entries that would fail at load or launch are misses (#2492).
  const auto txt = resolved.dir / (std::string(kModule) + ".txt");
  const auto blob = resolved.dir / (std::string(kModule) + ".hsaco");
  const auto write_raw = [](const fs::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary);
    file << text;
  };
  for (const char* bad :
       {"", "no-tab\n", "\tno-name\n", "no-mangled\t\n", "ok\tok\nbad\n"}) {
    write_raw(txt, bad);
    check(
        !blob_loaded(resolved.dir, &source),
        std::string("a malformed .txt is a miss: ") + bad);
  }
  write_raw(txt, "kernel\tmangled\n");
  check(blob_loaded(resolved.dir, &source), "a repaired .txt loads");
  write_raw(blob, "");
  check(!blob_loaded(resolved.dir, &source), "an empty .hsaco is a miss");
  write_raw(blob, "BLOB");
  fs::remove(txt);
  check(!blob_loaded(resolved.dir, &source), "an entry without .txt is a miss");
}

#if defined(_WIN32)
#include "rocm_hsaco_cache_test_win.inc"
#else
#include "rocm_hsaco_cache_test_posix.inc"
#endif

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
  test_platform(base);
  test_write_then_read(base);
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  return 0;
}
