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

#pragma once

// Experimental cuda-oxide route behind `fused_add_rms_norm` (issue #2833).
//
// The precompiled cuda-oxide fused residual-add + RMSNorm module from
// `spike/cuda-oxide-fused-norm` (#2832) is a second CUDA implementation of the
// kernel in `fused_norm.cpp`. This adapter runs it as an MLX primitive, so MLX
// keeps what it owns today: evaluation timing (the launch happens in
// `eval_gpu`, never at graph construction), output allocation, the command
// encoder's dependency tracking and CUDA graph capture, and buffer lifetimes.
//
// It is compiled only when mlxcel-core is built with the
// `cuda-oxide-fused-norm` cargo feature on a Linux CUDA build (the
// `MLXCEL_CUDA_OXIDE_FUSED_NORM` define); `fused_norm.cpp` references nothing
// here otherwise. Even then the route is taken only after the Rust side
// (`mlxcel_core::cuda_oxide_fused_norm`) has validated an artifact against its
// manifests and installed it, which happens when
// `MLXCEL_CUDA_OXIDE_FUSED_NORM=1` is set.
//
// Why a dedicated primitive rather than `fast::precompiled_cuda_kernel`: the
// emitted entry points take Rust slices as `(pointer, length)` pairs
// interleaved with the scalars (13 parameters, u64 lengths, a u32 dim), and
// `CustomKernel` packs inputs, outputs, then bool/int/float scalars. The
// primitive below packs in the order the artifact's manifest states.
//
// MLX internals this file depends on (re-check on an MLX pin bump):
// `cu::get_command_encoder`, `CommandEncoder::{set_input_array,
// set_output_array, add_temporary, add_kernel_node_raw, device}`,
// `cu::malloc_async`, `cu::get_jit_module`, `cu::KernelArgs`, and `copy_gpu`.
// The CUDA overlay is not modified.
//
// Used by: `fused_norm.cpp` (`fused_add_rms_norm`), which asks this adapter
// first and takes the existing port when it answers `nullopt`.

#include <mlx/array.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace mlxcel::turbo {

// The oxide route for one call, or `nullopt` when the existing CUDA port must
// run instead. Never throws for an unsupported configuration: no artifact
// installed, the default device is not the GPU, a dtype pair outside the
// artifact's entries, or more than 2^32-1 elements all answer `nullopt` and
// record the reason. The caller has already validated the shapes.
//
// `threads` is `fused_norm_threads(dim)` from `fused_norm.cpp`, the block
// width the artifact was validated with (#2832), passed in so the two routes
// cannot compute it differently.
std::optional<std::vector<mlx::core::array>> try_oxide_fused_add_rms_norm(
    const mlx::core::array& x,
    const mlx::core::array& residual,
    const mlx::core::array& weight,
    float eps,
    float weight_bias,
    std::size_t rows,
    int dim,
    int threads);

} // namespace mlxcel::turbo
