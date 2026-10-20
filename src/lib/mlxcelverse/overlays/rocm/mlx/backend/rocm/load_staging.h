// Copyright © 2025 Apple Inc.

// mlxcelverse: counters for the weight upload path in `Load::eval_gpu`
// (`load.cpp` in this directory, lablup/mlxcel#1564). Mirrors
// `mlx/backend/cuda/load_staging.h` for the ROCm backend so
// `weight_load_transfer_stats` in `src/lib/mlxcel-core/cpp/mlx_cxx_ext.cpp`
// reports the same host staging and copy back-pressure legs on ROCm builds.

#pragma once

#include <cstdint>

namespace mlx::core::rocm {

struct LoadTransferStats {
  // `Load` primitives evaluated on a HIP stream, and their bytes.
  uint64_t tensors;
  uint64_t bytes;
  // How many of those took the pinned staging path and how many the pageable
  // one (disabled by `MLXCEL_ROCM_PINNED_LOAD=0` / `MLXCEL_CUDA_PINNED_LOAD=0`,
  // or after a pinned allocation failed).
  uint64_t pinned_tensors;
  uint64_t pageable_tensors;
  // `hipMemcpyAsync` / `hipMemcpy` calls issued. Equal to `tensors` on the
  // pageable path; a pinned upload issues one per staging chunk, so a tensor
  // larger than one chunk adds more than one.
  uint64_t memcpy_calls;
  // Host staging leg: malloc (pageable path) plus the read into the staging
  // buffer, wall-clock nanoseconds on the evaluating thread.
  uint64_t read_ns;
  // Host time blocked waiting for a pinned staging slot to drain.
  uint64_t wait_ns;
  // Wall-clock nanoseconds inside `Load::eval_gpu`, both legs plus overhead.
  uint64_t total_ns;
};

// A consistent-enough snapshot of the process-wide counters (each field is
// read atomically; the set is not read under one lock).
LoadTransferStats load_transfer_stats();

} // namespace mlx::core::rocm
