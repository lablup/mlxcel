// Copyright 2025-2026 Lablup Inc.
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

// Choice of the one directory the Windows ROCm runtime is loaded from
// (lablup/mlxcel#2606). delayload.cpp loads every ROCm DLL from it and
// jit_module.cpp compiles against its include/ when present. The selection is
// a pure, header-only function so scripts/mlxcelverse/rocm_runtime_dir_test.sh
// can test it on any host without HIP or Windows.

#pragma once

#include <filesystem>
#include <functional>

namespace mlx::core::rocm {

enum class RuntimeDirSource {
  None,
  Env, // MLXCEL_ROCM_RUNTIME_DIR
  ExeDir, // the directory of the running executable
  Baked, // MLXCEL_ROCM_BIN_DIR, the SDK bin directory recorded at build time
};

enum class RuntimeDirError {
  None,
  EnvNotAbsolute, // MLXCEL_ROCM_RUNTIME_DIR is a relative path
  EnvMissingRuntime, // MLXCEL_ROCM_RUNTIME_DIR lacks the HIP runtime DLL
  NotFound, // neither the executable's directory nor the SDK has it
};

struct RuntimeDirChoice {
  std::filesystem::path dir;
  RuntimeDirSource source = RuntimeDirSource::None;
  RuntimeDirError error = RuntimeDirError::None;
};

// Pick the ROCm runtime directory from, in order:
//   1. `env_dir` (MLXCEL_ROCM_RUNTIME_DIR; empty means unset). An explicit
//      choice: it must be absolute and hold the HIP runtime, otherwise the
//      result is an error rather than a fall back to another candidate, so a
//      mistyped override never silently runs a different ROCm.
//   2. `exe_dir`, when it holds the HIP runtime (a bundle extracted beside
//      mlxcel.exe).
//   3. `baked_dir`, when it holds the HIP runtime (a developer build).
// `has_hip_runtime(dir)` reports whether `dir` contains the HIP runtime DLL
// the binary delay-loads. Only one directory is ever returned, so DLLs from
// two ROCm installations are never mixed.
inline RuntimeDirChoice choose_rocm_runtime_dir(
    const std::filesystem::path& env_dir,
    const std::filesystem::path& exe_dir,
    const std::filesystem::path& baked_dir,
    const std::function<bool(const std::filesystem::path&)>& has_hip_runtime) {
  RuntimeDirChoice choice;
  if (!env_dir.empty()) {
    choice.dir = env_dir;
    choice.source = RuntimeDirSource::Env;
    if (!env_dir.is_absolute()) {
      choice.error = RuntimeDirError::EnvNotAbsolute;
    } else if (!has_hip_runtime(env_dir)) {
      choice.error = RuntimeDirError::EnvMissingRuntime;
    }
    return choice;
  }
  if (!exe_dir.empty() && has_hip_runtime(exe_dir)) {
    choice.dir = exe_dir;
    choice.source = RuntimeDirSource::ExeDir;
    return choice;
  }
  if (!baked_dir.empty() && has_hip_runtime(baked_dir)) {
    choice.dir = baked_dir;
    choice.source = RuntimeDirSource::Baked;
    return choice;
  }
  choice.error = RuntimeDirError::NotFound;
  return choice;
}

inline const char* runtime_dir_source_name(RuntimeDirSource source) {
  switch (source) {
    case RuntimeDirSource::Env:
      return "MLXCEL_ROCM_RUNTIME_DIR";
    case RuntimeDirSource::ExeDir:
      return "executable directory";
    case RuntimeDirSource::Baked:
      return "build-time ROCm SDK";
    case RuntimeDirSource::None:
      break;
  }
  return "none";
}

#if defined(_WIN32)
// The directory chosen for this process by delayload.cpp, or an empty path
// when no candidate qualified (the first ROCm DLL load then fails).
const std::filesystem::path& rocm_runtime_dir();
#endif

} // namespace mlx::core::rocm
