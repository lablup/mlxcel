// Copyright © 2025 Apple Inc.

// mlxcelverse: integer environment knobs, range-checked (lablup/mlxcel#2152).
//
// The same rule as MLX_ROCM_GPU_WATCHDOG_SECS (gpu_watchdog_seconds() in
// device.cpp) and MLX_ROCM_FFT_CACHE_SIZE (capacity_from_env() in
// lru_cache.h): the whole value must be a base-10 integer inside the knob's
// range. The parsers this replaces took any `long` and cast it to `int`, so
// MLX_ROCM_WMMA_QMM_MAX_M=4294967297 silently became a row ceiling of 1.

#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace mlx::core::rocm {

// Reads `name` as a decimal integer from `min_value` to `max_value`. Unset or
// empty returns `default_value` silently. Anything else that is not such an
// integer (trailing junk such as "12abc", a value outside the range, or one
// past `long`) prints one stderr line and returns `default_value`:
//
//   [ROCm] ignoring invalid NAME="value" (expected <expected>); using <what>
//
// where <what> is `default_description` when given (for a default that is a
// sentinel, such as -1 for "use the per-device choice") and "the default N"
// otherwise. `default_value` itself is not range-checked, and `max_value`
// must not exceed INT_MAX. The function warns on every call that sees a bad
// value, so callers keep the result in a function-local static to read each
// variable, and warn, once per process.
inline int env_int_or_default(
    const char* name,
    int default_value,
    long min_value,
    long max_value,
    const char* expected,
    const char* default_description = nullptr) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }
  char* end = nullptr;
  errno = 0;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || errno == ERANGE || value < min_value ||
      value > max_value) {
    if (default_description != nullptr) {
      std::fprintf(
          stderr,
          "[ROCm] ignoring invalid %s=\"%s\" (expected %s); using %s\n",
          name,
          raw,
          expected,
          default_description);
    } else {
      std::fprintf(
          stderr,
          "[ROCm] ignoring invalid %s=\"%s\" (expected %s); "
          "using the default %d\n",
          name,
          raw,
          expected,
          default_value);
    }
    return default_value;
  }
  return static_cast<int>(value);
}

} // namespace mlx::core::rocm
