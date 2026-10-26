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
#include <string>
#include <string_view>
#include <vector>

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

// Case-insensitive ASCII prefix test (`prefix` must be lower case).
inline bool starts_with_ci(std::string_view name, std::string_view prefix) {
  if (name.size() < prefix.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.size(); ++i) {
    char c = name[i];
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
    if (c != prefix[i]) {
      return false;
    }
  }
  return true;
}

// The DLLs the HIP runtime and hipRTC open by plain name at run time, which
// delayload.cpp preloads by full path from the runtime directory. Each prefix
// must match at least one `<prefix>*.dll` there: without a copy in the chosen
// directory the plain-name LoadLibrary would resolve through the standard
// search, which reaches System32 (the display driver's older amd_comgr.dll,
// with which the SDK's HIP runtime finds no device) and the current directory.
inline constexpr std::string_view kRuntimePreloadPrefixes[] = {
    "amd_comgr",
    "hiprtc-builtins",
};

// Whether `filename` is `<prefix>*.dll` (case-insensitive).
inline bool is_preload_dll(std::string_view filename, std::string_view prefix) {
  constexpr std::string_view ext = ".dll";
  return filename.size() >= prefix.size() + ext.size() &&
      starts_with_ci(filename, prefix) &&
      starts_with_ci(filename.substr(filename.size() - ext.size()), ext);
}

// The prefixes of kRuntimePreloadPrefixes with no matching DLL among
// `filenames` (the entries of the runtime directory). Empty when the
// directory can serve every plain-name load itself.
inline std::vector<std::string_view> missing_runtime_preloads(
    const std::vector<std::string>& filenames) {
  std::vector<std::string_view> missing;
  for (std::string_view prefix : kRuntimePreloadPrefixes) {
    bool found = false;
    for (const std::string& name : filenames) {
      if (is_preload_dll(name, prefix)) {
        found = true;
        break;
      }
    }
    if (!found) {
      missing.push_back(prefix);
    }
  }
  return missing;
}

// Whether `a` and `b` name the same include directory as far as a string can
// tell: slash and backslash are the same separator, a run of separators
// counts as one, trailing separators are ignored and ASCII letters compare
// case-insensitively (Windows paths: only Windows has a runtime bundle, so
// only Windows ever compares two non-empty candidates).
inline bool same_include_dir(std::string_view a, std::string_view b) {
  auto normalize = [](std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      bool sep = c == '/' || c == '\\';
      if (sep) {
        if (out.empty() || out.back() != '/') {
          out.push_back('/');
        }
        continue;
      }
      if (c >= 'A' && c <= 'Z') {
        c = static_cast<char>(c - 'A' + 'a');
      }
      out.push_back(c);
    }
    if (out.size() > 1 && out.back() == '/') {
      out.pop_back();
    }
    return out;
  };
  return normalize(a) == normalize(b);
}

// The one stderr line rocm_include_dir() prints when the hiprtc headers it
// chose (`chosen`, from choose_rocm_include_dir() in jit_module.cpp with the
// same arguments) differ from headers a runtime bundle or HIP_PATH offered
// (lablup/mlxcel#2639):
//   - ROCM_HOME or ROCM_PATH overrode the headers of a runtime bundle
//     (`runtime_is_bundle`, a runtime directory from MLXCEL_ROCM_RUNTIME_DIR
//     or the executable's directory, whose include dir is
//     `runtime_include_dir`);
//   - a runtime bundle's headers won over HIP_PATH, which the HIP SDK
//     installer sets machine-wide.
// `runtime_source` names where the runtime directory came from
// (runtime_dir_source_name()). Returns "" when nothing was overridden or the
// two candidates are the same directory. Pure, like the choice itself.
inline std::string rocm_include_dir_notice(
    const char* rocm_home,
    const char* rocm_path,
    const char* hip_path,
    const std::string& runtime_include_dir,
    bool runtime_is_bundle,
    const char* runtime_source,
    const std::string& chosen) {
  if (chosen.empty() || !runtime_is_bundle || runtime_include_dir.empty()) {
    return "";
  }
  auto line = [](const std::string& dir,
                 const char* source,
                 const std::string& other,
                 const char* other_source) {
    return "[mlx-rocm] hiprtc headers: " + dir + " (" + source +
        "); ignoring " + other + " from " + other_source;
  };
  const char* env_var = nullptr;
  if (rocm_home && *rocm_home) {
    env_var = "ROCM_HOME";
  } else if (rocm_path && *rocm_path) {
    env_var = "ROCM_PATH";
  }
  if (env_var) {
    if (same_include_dir(chosen, runtime_include_dir)) {
      return "";
    }
    return line(chosen, env_var, runtime_include_dir, runtime_source);
  }
  if (hip_path && *hip_path && same_include_dir(chosen, runtime_include_dir)) {
    std::string hip_include = std::string(hip_path) + "/include";
    if (same_include_dir(hip_include, runtime_include_dir)) {
      return "";
    }
    return line(chosen, runtime_source, hip_include, "HIP_PATH");
  }
  return "";
}

#if defined(_WIN32)
// The directory chosen for this process by delayload.cpp, or an empty path
// when no candidate qualified (the first ROCm DLL load then fails).
const std::filesystem::path& rocm_runtime_dir();

// Where rocm_runtime_dir() came from: RuntimeDirSource::None when no
// candidate qualified. jit_module.cpp lets a bundle's headers (Env or ExeDir)
// outrank HIP_PATH, but not the SDK the binary was built against (Baked).
RuntimeDirSource rocm_runtime_dir_source();
#endif

} // namespace mlx::core::rocm
