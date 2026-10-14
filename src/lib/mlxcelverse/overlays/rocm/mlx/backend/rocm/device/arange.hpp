// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <typename T>
__global__ void arange_kernel(T* out, T start, T step, int64_t size) {
  // Grid capped by copy_grid_blocks (LOCAL_FIXES.md 46); the threads stride
  // the elements on a 64-bit counter.
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += stride) {
    out[idx] = start + static_cast<T>(idx) * step;
  }
}

} // namespace mlx::core::rocm
