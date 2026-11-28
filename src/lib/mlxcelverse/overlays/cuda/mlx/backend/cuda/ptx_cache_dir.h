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

// Location rules of the PTX disk cache that jit_module.cpp loads compiled modules
// from (lablup/mlxcel#2797). The POSIX trust check is in ptx_cache_trust.h;
// ptx_cache_dir_win.h is its Windows counterpart. This part has no CUDA
// dependency, so scripts/mlxcelverse/cuda_runtime_dir_test.sh tests it on any
// host. Include it after the backend's own headers: on Windows it brings in
// <windows.h>.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

namespace mlx::core::cu {

// The environment the cache location depends on.
struct PtxCacheEnv {
  std::filesystem::path override_dir; // MLX_PTX_CACHE_DIR; empty means unset
  std::filesystem::path cache_root; // MLXCEL_CACHE_DIR; empty means unset
  // MLXCEL_CACHE_TRUSTED_ROOT (lablup/mlxcel#3503); empty means unset. POSIX
  // only: the Windows reader leaves it empty.
  std::filesystem::path trusted_root;
};

// [mlxcel #3503] Hand over the gid whose write access on a cache path
// component counts as the user's own, or a negative value for none (the
// default). mlxcel's binaries call it at startup, through
// ensure_persistent_ptx_cache, with the effective gid when that group is
// provably private to the user (mlxcel-core's private_group.rs); the decision
// is made there, in one place, and the walk only compares. It is a call and
// not an environment variable so that a launcher cannot name a shared group.
// It is read at each walk (the cache directory, then each device
// subdirectory), so call it before the first kernel is compiled or loaded.
// POSIX only. Defined in jit_module.cpp.
void set_ptx_cache_trusted_group(std::int64_t gid);

// A path as UTF-8 for a diagnostic; never throws.
inline std::string ptx_cache_path_for_log(const std::filesystem::path& path) {
  try {
    auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
  } catch (const std::exception&) {
    return "<path>";
  }
}

// Why the environment cannot name a cache, or "" when it can. A relative path
// resolves against the current directory, which anyone can pick for a victim, so
// it is refused, as hsaco_env_refusal does in the ROCm overlay. On Windows a path
// such as \cache or C:cache is relative too: it depends on the current drive or
// that drive's current directory. MLXCEL_CACHE_DIR does not matter when
// MLX_PTX_CACHE_DIR is set, since nothing is placed under it.
inline std::string ptx_cache_env_refusal(const PtxCacheEnv& env) {
  const auto relative = [](const std::filesystem::path& path, const char* name) {
    return name + std::string(" must be an absolute path, got \"") +
        ptx_cache_path_for_log(path) + "\"";
  };
  if (!env.override_dir.empty()) {
    return env.override_dir.is_absolute()
        ? std::string()
        : relative(env.override_dir, "MLX_PTX_CACHE_DIR");
  }
  return env.cache_root.empty() || env.cache_root.is_absolute()
      ? std::string()
      : relative(env.cache_root, "MLXCEL_CACHE_DIR");
}

#if !defined(_WIN32)
inline PtxCacheEnv ptx_cache_env_from_process() {
  PtxCacheEnv env;
  if (const char* value = std::getenv("MLX_PTX_CACHE_DIR");
      value != nullptr && *value != '\0') {
    env.override_dir = value;
  }
  if (const char* value = std::getenv("MLXCEL_CACHE_DIR");
      value != nullptr && *value != '\0') {
    env.cache_root = value;
  }
  if (const char* value = std::getenv("MLXCEL_CACHE_TRUSTED_ROOT");
      value != nullptr && *value != '\0') {
    env.trusted_root = value;
  }
  return env;
}
#endif

} // namespace mlx::core::cu

#if defined(_WIN32)
#include "mlx/backend/cuda/ptx_cache_dir_win.h"
#endif
