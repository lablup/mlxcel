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

// New file for mlxcel (no upstream MLX counterpart)
// Added by mlxcel (lablup/mlxcel#1564).
//
// Counters for the weight upload path in `Load::eval_gpu` (`load.cpp` in this
// directory). Every `Load` primitive evaluated on a CUDA stream adds to them,
// so a caller that snapshots them around an evaluation learns how many
// checkpoint tensors that evaluation materialized and where the host time went.
//
// The two time counters are the two legs a CUDA profiler cannot tell apart:
// `read_ns` is the host staging leg (the `pread` out of the page cache into the
// staging buffer, plus the `malloc` on the pageable path), which makes no CUDA
// call at all and is invisible to nsys; `wait_ns` is the time the host spent
// blocked on a pinned staging slot whose host-to-device copy had not finished,
// which is how much of the copy leg the read failed to hide. The copy leg
// itself runs asynchronously and is measured on the device side by nsys
// (`cuda_gpu_mem_time_sum`, `[CUDA memcpy Host-to-Device]`).
//
// mlxcel reads them through `weight_load_transfer_stats` in
// `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`.

#pragma once

#include <cstdint>

namespace mlx::core::cu {

struct LoadTransferStats {
  // `Load` primitives evaluated on a CUDA stream, and their bytes.
  uint64_t tensors;
  uint64_t bytes;
  // How many of those took the pinned staging path and how many the pageable
  // one (disabled by `MLXCEL_CUDA_PINNED_LOAD=0`, or after a pinned
  // allocation failed).
  uint64_t pinned_tensors;
  uint64_t pageable_tensors;
  // `cudaMemcpyAsync` calls issued. Equal to `tensors` on the pageable path; a
  // pinned upload issues one per staging chunk, so a tensor larger than one
  // chunk adds more than one.
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

} // namespace mlx::core::cu
