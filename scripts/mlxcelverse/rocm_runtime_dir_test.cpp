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

// Host-only tests for the Windows ROCm runtime directory choice
// (lablup/mlxcel#2606): choose_rocm_runtime_dir() from the overlay's
// runtime_dir.h, choose_rocm_include_dir() copied out of jit_module.cpp by
// rocm_runtime_dir_test.sh, and the hiprtc header notice
// (rocm_include_dir_notice(), lablup/mlxcel#2639). None touches the file
// system or HIP, so the test runs on any host.
#include "mlx/backend/rocm/runtime_dir.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using mlx::core::rocm::choose_rocm_runtime_dir;
using mlx::core::rocm::missing_runtime_preloads;
using mlx::core::rocm::rocm_include_dir_notice;
using mlx::core::rocm::RuntimeDirChoice;
using mlx::core::rocm::RuntimeDirError;
using mlx::core::rocm::RuntimeDirSource;

namespace mlx::core::rocm {
#include "rocm_include_dir.inc"
} // namespace mlx::core::rocm
using mlx::core::rocm::choose_rocm_include_dir;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

// Absolute paths in the host's own syntax.
const fs::path root = fs::current_path().root_path();
const fs::path env_dir = root / "bundle";
const fs::path exe_dir = root / "app";
const fs::path baked_dir = root / "rocm" / "bin";

// A predicate that reports the HIP runtime in exactly `dirs`, and records
// every directory it was asked about.
struct Runtimes {
  Runtimes(std::initializer_list<fs::path> with_runtime = {})
      : dirs(with_runtime) {}
  std::set<fs::path> dirs;
  mutable std::set<fs::path> asked;
  bool operator()(const fs::path& dir) const {
    asked.insert(dir);
    return dirs.count(dir) != 0;
  }
};

RuntimeDirChoice choose(
    const fs::path& env,
    const fs::path& exe,
    const fs::path& baked,
    const Runtimes& runtimes) {
  return choose_rocm_runtime_dir(env, exe, baked, std::cref(runtimes));
}

void test_runtime_dir() {
  // The override wins over a bundle beside the executable and the SDK.
  {
    Runtimes r{env_dir, exe_dir, baked_dir};
    auto c = choose(env_dir, exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::None, "env: no error");
    check(c.source == RuntimeDirSource::Env, "env: source");
    check(c.dir == env_dir, "env: dir");
  }
  // An override without the runtime is an error, not a fall back.
  {
    Runtimes r{exe_dir, baked_dir};
    auto c = choose(env_dir, exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::EnvMissingRuntime, "env missing: error");
    check(c.dir == env_dir, "env missing: names the override");
    check(r.asked.count(exe_dir) == 0, "env missing: exe dir not consulted");
    check(r.asked.count(baked_dir) == 0, "env missing: SDK not consulted");
  }
  // A relative override is rejected before the file system is consulted.
  {
    Runtimes r{fs::path("bundle"), exe_dir, baked_dir};
    auto c = choose(fs::path("bundle"), exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::EnvNotAbsolute, "env relative: error");
    check(r.asked.empty(), "env relative: nothing consulted");
  }
  // Unset override: a bundle beside the executable wins over the SDK.
  {
    Runtimes r{exe_dir, baked_dir};
    auto c = choose(fs::path(), exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::None, "exe: no error");
    check(c.source == RuntimeDirSource::ExeDir, "exe: source");
    check(c.dir == exe_dir, "exe: dir");
    check(r.asked.count(baked_dir) == 0, "exe: SDK not consulted");
  }
  // No bundle: the build-time SDK (a developer build).
  {
    Runtimes r{baked_dir};
    auto c = choose(fs::path(), exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::None, "baked: no error");
    check(c.source == RuntimeDirSource::Baked, "baked: source");
    check(c.dir == baked_dir, "baked: dir");
  }
  // An unknown executable directory is skipped.
  {
    Runtimes r{baked_dir};
    auto c = choose(fs::path(), fs::path(), baked_dir, r);
    check(c.source == RuntimeDirSource::Baked, "no exe dir: SDK chosen");
    check(r.asked.count(fs::path()) == 0, "no exe dir: empty path not asked");
  }
  // Nothing anywhere.
  {
    Runtimes r{};
    auto c = choose(fs::path(), exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::NotFound, "none: error");
    check(c.source == RuntimeDirSource::None, "none: source");
    check(c.dir.empty(), "none: no dir");
  }
}

void test_include_dir() {
  const std::string bundle_inc = (env_dir / "include").string();
  const std::string sdk_inc = (baked_dir.parent_path() / "include").string();
  // ROCM_HOME and ROCM_PATH are explicit overrides: they beat a bundle.
  check(
      choose_rocm_include_dir(
          "/h", "/p", "/hp", bundle_inc, true, "/baked", true, true) ==
          "/h/include",
      "include: ROCM_HOME beats the bundle");
  check(
      choose_rocm_include_dir(
          nullptr, "/p", "/hp", bundle_inc, true, "/baked", true, true) ==
          "/p/include",
      "include: ROCM_PATH beats the bundle");
  check(
      choose_rocm_include_dir(
          "", "", "/hp", bundle_inc, true, "/baked", true, true) == bundle_inc,
      "include: empty ROCM_HOME/ROCM_PATH are unset");
  // A bundle's headers beat HIP_PATH, which the HIP SDK installer sets
  // machine-wide (lablup/mlxcel#2639).
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, "/hp", bundle_inc, true, "/baked", true, true) ==
          bundle_inc,
      "include: bundle beats HIP_PATH");
  // A bundle without headers falls through to HIP_PATH.
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, "/hp", "", true, "/baked", true, true) ==
          "/hp/include",
      "include: bundle without include/ falls through to HIP_PATH");
  // The build-time SDK (a Baked runtime directory) keeps HIP_PATH first.
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, "/hp", sdk_inc, false, "/baked", true, true) ==
          "/hp/include",
      "include: Baked source keeps HIP_PATH first");
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, "", sdk_inc, false, "/baked", true, true) ==
          sdk_inc,
      "include: empty HIP_PATH is unset");
  // The runtime directory's headers come before the baked SDK and /opt/rocm.
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, nullptr, bundle_inc, true, "/baked", true, true) ==
          bundle_inc,
      "include: runtime dir before baked");
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, nullptr, "", false, "/baked", true, true) ==
          "/baked",
      "include: baked when the runtime dir has no headers");
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, nullptr, "", false, "/baked", false, true) ==
          "/opt/rocm/include",
      "include: /opt/rocm when the baked dir is gone");
  check(
      choose_rocm_include_dir(
          nullptr, nullptr, nullptr, "", false, nullptr, false, false)
          .empty(),
      "include: nothing qualifies");
  // Linux passes no HIP_PATH, no runtime include and no bundle: the -I string
  // is the one ROCM_PATH or /opt/rocm gave before.
  check(
      choose_rocm_include_dir(
          nullptr, "/opt/rocm-7", nullptr, "", false, nullptr, false, true) ==
          "/opt/rocm-7/include",
      "include: Linux ROCM_PATH unchanged");
}

void test_include_dir_notice() {
  const std::string bundle_inc = (env_dir / "include").string();
  const char* src = "executable directory";
  // ROCM_PATH overrides the bundle: name both.
  check(
      rocm_include_dir_notice(
          nullptr, "/p", "/hp", bundle_inc, true, src, "/p/include") ==
          "[mlx-rocm] hiprtc headers: /p/include (ROCM_PATH); ignoring " +
              bundle_inc + " from executable directory",
      "notice: ROCM_PATH over the bundle");
  check(
      rocm_include_dir_notice(
          "/h", "/p", nullptr, bundle_inc, true, src, "/h/include") ==
          "[mlx-rocm] hiprtc headers: /h/include (ROCM_HOME); ignoring " +
              bundle_inc + " from executable directory",
      "notice: ROCM_HOME over the bundle");
  // The bundle beats HIP_PATH: name both.
  check(
      rocm_include_dir_notice(
          nullptr,
          nullptr,
          "/hp",
          bundle_inc,
          true,
          "MLXCEL_ROCM_RUNTIME_DIR",
          bundle_inc) ==
          "[mlx-rocm] hiprtc headers: " + bundle_inc +
              " (MLXCEL_ROCM_RUNTIME_DIR); ignoring /hp/include from HIP_PATH",
      "notice: bundle over HIP_PATH");
  // Silent when nothing disagrees.
  check(
      rocm_include_dir_notice(
          nullptr, nullptr, nullptr, bundle_inc, true, src, bundle_inc)
          .empty(),
      "notice: bundle alone");
  check(
      rocm_include_dir_notice(
          nullptr, nullptr, "/hp", "/sdk/include", false, src, "/hp/include")
          .empty(),
      "notice: Baked source with HIP_PATH");
  check(
      rocm_include_dir_notice(
          nullptr, "/p", nullptr, "", true, src, "/p/include")
          .empty(),
      "notice: bundle without headers");
  check(
      rocm_include_dir_notice(
          nullptr, "/opt/rocm", nullptr, "", false, "none", "/opt/rocm/include")
          .empty(),
      "notice: Linux");
  // The same directory spelled differently is not a disagreement.
  check(
      rocm_include_dir_notice(
          nullptr,
          "C:\\rocm\\10.1.0",
          nullptr,
          "c:\\ROCM\\10.1.0\\include\\",
          true,
          "MLXCEL_ROCM_RUNTIME_DIR",
          "C:\\rocm\\10.1.0/include")
          .empty(),
      "notice: ROCM_PATH naming the bundle's own SDK");
  check(
      rocm_include_dir_notice(
          nullptr,
          nullptr,
          "D:/bundle/",
          "D:\\bundle\\include",
          true,
          src,
          "D:\\bundle\\include")
          .empty(),
      "notice: HIP_PATH naming the bundle");
}

using Names = std::vector<std::string>;
using Missing = std::vector<std::string_view>;

void test_runtime_preloads() {
  // A complete runtime directory (the ROCm 10.1.0 bundle's bin/).
  check(
      missing_runtime_preloads(
          {"amdhip64_7.dll",
           "amd_comgr.dll",
           "hiprtc0716.dll",
           "hiprtc-builtins0716.dll",
           "rocblas.dll"})
          .empty(),
      "preloads: complete directory");
  // No amd_comgr: the HIP runtime would load the driver's System32 copy.
  check(
      missing_runtime_preloads(
          {"amdhip64_7.dll", "hiprtc-builtins0716.dll", "amd_comgr.dll.off"}) ==
          Missing{"amd_comgr"},
      "preloads: amd_comgr missing (a renamed copy does not count)");
  check(
      missing_runtime_preloads({"amdhip64_7.dll", "AMD_COMGR_3.DLL"}) ==
          Missing{"hiprtc-builtins"},
      "preloads: case-insensitive, hiprtc-builtins missing");
  check(
      missing_runtime_preloads({"hiprtc0716.dll", "amd_comgr"}) ==
          Missing{"amd_comgr", "hiprtc-builtins"},
      "preloads: hiprtc is not hiprtc-builtins, a name without .dll is not a "
      "DLL");
  check(
      missing_runtime_preloads(Names{}) ==
          Missing{"amd_comgr", "hiprtc-builtins"},
      "preloads: empty or unreadable directory");
}

} // namespace

int main() {
  test_runtime_dir();
  test_runtime_preloads();
  test_include_dir();
  test_include_dir_notice();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
