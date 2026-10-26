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

// Host-only tests for the Windows CUDA runtime directory choice
// (lablup/mlxcel#2640): choose_cuda_runtime_dir(), cuda_runtime_marker(),
// missing_runtime_preloads() and choose_cuda_include_dir() from the CUDA
// overlay's runtime_dir.h. None touches the file system or CUDA, so the test
// runs on any host.
#include "mlx/backend/cuda/runtime_dir.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using mlx::core::cu::choose_cuda_include_dir;
using mlx::core::cu::choose_cuda_runtime_dir;
using mlx::core::cu::cuda_runtime_marker;
using mlx::core::cu::is_cuda_dll;
using mlx::core::cu::is_runtime_bundle;
using mlx::core::cu::missing_runtime_preloads;
using mlx::core::cu::RuntimeDirChoice;
using mlx::core::cu::RuntimeDirError;
using mlx::core::cu::RuntimeDirSource;

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
const fs::path env_dir = root / "bundle" / "bin";
const fs::path exe_dir = root / "app";
const fs::path baked_dir = root / "cuda" / "bin" / "x64";

// A predicate that reports the CUDA runtime in exactly `dirs`, and records
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
  return choose_cuda_runtime_dir(env, exe, baked, std::cref(runtimes));
}

void test_runtime_dir() {
  // The override wins over a bundle beside the executable and the toolkit.
  {
    Runtimes r{env_dir, exe_dir, baked_dir};
    auto c = choose(env_dir, exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::None, "env: no error");
    check(c.source == RuntimeDirSource::Env, "env: source");
    check(c.dir == env_dir, "env: dir");
    check(is_runtime_bundle(c.source), "env: a bundle");
  }
  // An override without the runtime is an error, not a fall back.
  {
    Runtimes r{exe_dir, baked_dir};
    auto c = choose(env_dir, exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::EnvMissingRuntime, "env missing: error");
    check(c.source == RuntimeDirSource::Env, "env missing: source");
    check(c.dir == env_dir, "env missing: names the override");
    check(r.asked.count(exe_dir) == 0, "env missing: exe dir not consulted");
    check(r.asked.count(baked_dir) == 0, "env missing: toolkit not consulted");
  }
  // A relative override is rejected before the file system is consulted.
  {
    Runtimes r{fs::path("bundle"), exe_dir, baked_dir};
    auto c = choose(fs::path("bundle"), exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::EnvNotAbsolute, "env relative: error");
    check(c.dir == fs::path("bundle"), "env relative: names the override");
    check(r.asked.empty(), "env relative: nothing consulted");
  }
  // Unset override: a bundle beside the executable wins over the toolkit.
  {
    Runtimes r{exe_dir, baked_dir};
    auto c = choose(fs::path(), exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::None, "exe: no error");
    check(c.source == RuntimeDirSource::ExeDir, "exe: source");
    check(c.dir == exe_dir, "exe: dir");
    check(is_runtime_bundle(c.source), "exe: a bundle");
    check(r.asked.count(baked_dir) == 0, "exe: toolkit not consulted");
  }
  // No bundle: the build-time layout, unchecked, so a developer build keeps
  // its per-library lookup (and its fallback) exactly as before.
  {
    Runtimes r{};
    auto c = choose(fs::path(), exe_dir, baked_dir, r);
    check(c.error == RuntimeDirError::None, "baked: no error");
    check(c.source == RuntimeDirSource::Baked, "baked: source");
    check(c.dir == baked_dir, "baked: dir");
    check(!is_runtime_bundle(c.source), "baked: not a bundle");
    check(r.asked.count(baked_dir) == 0, "baked: not checked for the runtime");
  }
  // An unknown executable directory is skipped.
  {
    Runtimes r{baked_dir};
    auto c = choose(fs::path(), fs::path(), baked_dir, r);
    check(c.source == RuntimeDirSource::Baked, "no exe dir: toolkit chosen");
    check(r.asked.count(fs::path()) == 0, "no exe dir: empty path not asked");
  }
}

void test_marker() {
  // The binaries at the current pin: static cudart, so cuBLASLt marks it.
  check(
      cuda_runtime_marker(
          {"cudnn64_9.dll",
           "cufft64_12.dll",
           "nvrtc64_130_0.dll",
           "cublasLt64_13.dll",
           "cusolver64_12.dll"}) == "cublasLt64_13.dll",
      "marker: cuBLASLt at the current pin");
  check(
      cuda_runtime_marker(
          {"cublasLt64_13.dll", "cudart64_13.dll", "nvrtc64_130_0.dll"}) ==
          "cudart64_13.dll",
      "marker: the CUDA runtime when imported");
  check(
      cuda_runtime_marker({"cudnn64_9.dll", "NVRTC64_130_0.DLL"}) ==
          "NVRTC64_130_0.DLL",
      "marker: NVRTC, case-insensitive");
  check(
      cuda_runtime_marker({"cublas64_13.dll", "dxgi.dll"}).empty(),
      "marker: cuBLAS is not cuBLASLt");
  check(
      is_cuda_dll("cublasLt64_13.dll") && is_cuda_dll("CUDNN64_9.DLL") &&
          !is_cuda_dll("cudnn_graph64_9.dll") && !is_cuda_dll("nvcuda.dll"),
      "delay-loaded CUDA DLLs");
}

using Names = std::vector<std::string>;
using Missing = std::vector<std::string_view>;

// The runtime directory of the CUDA 13.4 / cuDNN 9.27 bundle.
Names complete_bundle() {
  return {
      "cublas64_13.dll",
      "cublasLt64_13.dll",
      "cufft64_12.dll",
      "cusolver64_12.dll",
      "cusparse64_12.dll",
      "nvJitLink_130_0.dll",
      "nvrtc64_130_0.dll",
      "nvrtc-builtins64_134.dll",
      "cudnn64_9.dll",
      "cudnn_adv64_9.dll",
      "cudnn_cnn64_9.dll",
      "cudnn_engines_precompiled64_9.dll",
      "cudnn_engines_runtime_compiled64_9.dll",
      "cudnn_engines_tensor_ir64_9.dll",
      "cudnn_ext64_9.dll",
      "cudnn_graph64_9.dll",
      "cudnn_heuristic64_9.dll",
      "cudnn_ops64_9.dll",
      "openblas.dll",
      "mlxcel.exe",
  };
}

Names without(Names names, std::string_view drop) {
  Names out;
  for (auto& n : names) {
    if (n != drop) {
      out.push_back(n);
    }
  }
  return out;
}

void test_runtime_preloads() {
  check(
      missing_runtime_preloads(complete_bundle()).empty(),
      "preloads: complete bundle");
  check(
      missing_runtime_preloads(
          without(complete_bundle(), "nvrtc-builtins64_134.dll")) ==
          Missing{"nvrtc-builtins"},
      "preloads: nvrtc-builtins missing (nvrtc64 does not count)");
  check(
      missing_runtime_preloads(
          without(complete_bundle(), "cublasLt64_13.dll")) ==
          Missing{"cublaslt64_"},
      "preloads: cuBLASLt missing (cuBLAS does not count)");
  check(
      missing_runtime_preloads(
          without(complete_bundle(), "cudnn_engines_runtime_compiled64_9.dll")) ==
          Missing{"cudnn_engines_runtime_compiled"},
      "preloads: a cuDNN engine sub-library missing");
  check(
      missing_runtime_preloads(
          without(without(complete_bundle(), "cudnn_ext64_9.dll"),
                  "cudnn_engines_tensor_ir64_9.dll"))
          .empty(),
      "preloads: cudnn_ext and tensor_ir are optional (earlier cuDNN 9)");
  {
    Names n = without(complete_bundle(), "nvJitLink_130_0.dll");
    n.push_back("nvJitLink_130_0.dll.off");
    n.push_back("nvjitlink_130_0.alt.dll");
    check(
        missing_runtime_preloads(n) == Missing{"nvjitlink_"},
        "preloads: renamed and .alt.dll copies do not count");
  }
  {
    Names n = without(complete_bundle(), "cudnn_graph64_9.dll");
    n.push_back("CUDNN_GRAPH64_9.DLL");
    check(missing_runtime_preloads(n).empty(), "preloads: case-insensitive");
  }
  check(
      missing_runtime_preloads(Names{}).size() == 12,
      "preloads: empty or unreadable directory misses every prefix");
}

void test_include_dir() {
  const std::string bundle_inc = (env_dir.parent_path() / "include").string();
  const std::string def = "/usr/local/cuda/include";
  // CUDA_HOME is an explicit override: it beats a bundle.
  check(
      choose_cuda_include_dir("/h", "/p", bundle_inc, true, "/b", true, def) ==
          "/h/include",
      "include: CUDA_HOME beats the bundle");
  // A bundle's headers beat CUDA_PATH, which the installer sets machine-wide.
  check(
      choose_cuda_include_dir(
          nullptr, "/p", bundle_inc, true, "/b", true, def) == bundle_inc,
      "include: bundle beats CUDA_PATH");
  check(
      choose_cuda_include_dir("", "/p", bundle_inc, true, "/b", true, def) ==
          bundle_inc,
      "include: empty CUDA_HOME is unset");
  // A bundle without headers falls through to CUDA_PATH.
  check(
      choose_cuda_include_dir(nullptr, "/p", "", true, "/b", true, def) ==
          "/p/include",
      "include: bundle without include/ falls through to CUDA_PATH");
  // No bundle: the order jit_module.cpp used before lablup/mlxcel#2640.
  check(
      choose_cuda_include_dir(
          nullptr, "/p", bundle_inc, false, "/b", true, def) == "/p/include",
      "include: not a bundle, CUDA_PATH first");
  check(
      choose_cuda_include_dir(nullptr, nullptr, "", false, "/b", true, def) ==
          "/b",
      "include: the build-time toolkit when no variable is set");
  check(
      choose_cuda_include_dir(nullptr, "", "", false, "/b", false, def) == def,
      "include: the default toolkit when the baked dir is gone");
  check(
      choose_cuda_include_dir(nullptr, nullptr, "", false, nullptr, false, "")
          .empty(),
      "include: nothing qualifies");
}

} // namespace

int main() {
  test_runtime_dir();
  test_marker();
  test_runtime_preloads();
  test_include_dir();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
