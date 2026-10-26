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

// mlxcelverse: native NVFP4 kernels for QuantizedMatmul and GatherQMM
// (lablup/mlxcel#2617). Included by qmm.hip, which owns the dispatch; not a
// translation unit of its own.
//
// NVFP4 packs eight E2M1 values per uint32 (value i in bits 4i..4i+3), one
// E4M3 block scale byte per 16 values, and optionally one float32 global
// scale per weight matrix (per expert for GatherQMM). A weight decodes as
//
//   w = e2m1(q) * e4m3(scale) * (global_scale / (448 * 6))
//
// with the last factor 1 when there is no global scale, which is MLX's
// `dequantize` for mode "nvfp4" (mlx/ops.cpp) and the CUDA backend's
// fp_gather_qmv. Both decodes are the branch-free half-bit forms in
// qdequant.hpp, each off by an exact power of two that is folded into one
// multiply per output.
//
// Lane use. Every kernel here is lane-free: a 16-thread group reduces one
// output column with width-16 shuffles, and blockDim.x is always 16, so the
// groups are 16-aligned inside any wavefront of 16, 32 or 64 lanes. Nothing
// reads WARP_SIZE, so the same code is correct on wave32 (RDNA) and wave64
// (CDNA) devices, and the dispatch needs no wavefront gate (compare item 52,
// whose WMMA kernels hard-code 32-lane warps).

#pragma once

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/quantized/qdequant.hpp"

#include <hip/hip_runtime.h>

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace mlx::core {

namespace rocm {

constexpr int kNvfp4GroupSize = 16;
constexpr int kNvfp4WordsPerGroup = kNvfp4GroupSize / 8;
// Threads that reduce one output column, and columns per block.
constexpr int kNvfp4LanesPerCol = 16;
constexpr int kNvfp4ColsPerBlock = 16;
// E4M3's largest finite value times E2M1's: MLX encodes the block scales
// relative to global_scale / (448 * 6).
constexpr float kNvfp4GlobalScaleDenom = 448.0f * 6.0f;

// The exact powers of two the half-bit decodes leave out, times the global
// scale factor of expert `e` (1 without a global scale).
__device__ __forceinline__ float nvfp4_output_factor(
    const float* __restrict__ global_scale,
    uint32_t e) {
  const float unscale = kFp4HalfBitsScale * kE4m3HalfBitsScale;
  return global_scale ? unscale * (global_scale[e] / kNvfp4GlobalScaleDenom)
                      : unscale;
}

// out[t, n] = factor(e_t) * sum_k x_t[k] * w_{e_t}[n, k] for ROWS consecutive
// rows t per block. A row is one (batch, m) pair, t = b * M + m.
//
// Gather (lhs_indices != nullptr): x_t is row m of x batch lhs[b] and e_t is
// rhs[b], both read through the collapsed index strides; an expert index at
// or past E writes zeros, as the other gather kernels do. Dense (null
// indices): x_t is row m of x batch b (stride x_batch_stride, 0 for a
// broadcast x) and e_t is b when w is batched, 0 otherwise.
//
// The rows of a block share one load of the weight group whenever their
// experts repeat (every dense call, and the expert runs of a sorted gather),
// so ROWS > 1 reads each weight byte once per ROWS rows of the same expert.
// Rows whose expert differs from the previous row's reload, which keeps any
// order correct.
//
// Grid: (ceil(rows / ROWS), ceil(N / kNvfp4ColsPerBlock)).
// Block: (kNvfp4LanesPerCol, kNvfp4ColsPerBlock).
template <typename T, int ROWS>
__global__ void __launch_bounds__(kNvfp4LanesPerCol * kNvfp4ColsPerBlock)
    nvfp4_qmv_kernel(
        const T* __restrict__ x,
        const uint32_t* __restrict__ w,
        const uint8_t* __restrict__ scales,
        const float* __restrict__ global_scale,
        const uint32_t* __restrict__ lhs_indices,
        const uint32_t* __restrict__ rhs_indices,
        const rocm::Shape batch_shape,
        const rocm::Strides lhs_idx_strides,
        const rocm::Strides rhs_idx_strides,
        int batch_ndim,
        T* __restrict__ out,
        int64_t rows,
        int M,
        int N,
        int K,
        int E,
        int64_t x_batch_stride,
        bool w_batched) {
  const int lane = threadIdx.x;
  const int col = blockIdx.y * kNvfp4ColsPerBlock + threadIdx.y;
  if (col >= N) {
    return; // the whole 16-lane group of this column leaves together
  }
  const int64_t words_per_row = K / 8;
  const int64_t groups_per_row = K / kNvfp4GroupSize;
  const int64_t w_expert_stride = static_cast<int64_t>(N) * words_per_row;
  const int64_t s_expert_stride = static_cast<int64_t>(N) * groups_per_row;

  const T* x_row[ROWS];
  uint32_t expert[ROWS];
  bool row_ok[ROWS];
  float acc[ROWS];
  const int64_t t0 = static_cast<int64_t>(blockIdx.x) * ROWS;
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int64_t t = t0 + r;
    acc[r] = 0.0f;
    row_ok[r] = t < rows;
    x_row[r] = x;
    expert[r] = 0;
    if (!row_ok[r]) {
      continue;
    }
    const int64_t b = t / M;
    const int64_t m = t - b * M;
    if (lhs_indices != nullptr) {
      int64_t lhs_loc = 0;
      int64_t rhs_loc = 0;
      if (batch_ndim == 1) {
        lhs_loc = b * lhs_idx_strides[0];
        rhs_loc = b * rhs_idx_strides[0];
      } else if (batch_ndim > 1) {
        int64_t elem = b;
        for (int i = batch_ndim - 1; i >= 0; --i) {
          const int64_t coord = elem % batch_shape[i];
          lhs_loc += coord * lhs_idx_strides[i];
          rhs_loc += coord * rhs_idx_strides[i];
          elem /= batch_shape[i];
        }
      }
      const uint32_t xi = lhs_indices[lhs_loc];
      expert[r] = rhs_indices[rhs_loc];
      x_row[r] = x + (static_cast<int64_t>(xi) * M + m) * K;
      row_ok[r] = expert[r] < static_cast<uint32_t>(E);
    } else {
      expert[r] = w_batched ? static_cast<uint32_t>(b) : 0u;
      x_row[r] = x + b * x_batch_stride + m * K;
    }
  }

  for (int64_t g = lane; g < groups_per_row; g += kNvfp4LanesPerCol) {
    float wv[kNvfp4GroupSize];
    float scale = 0.0f;
    uint32_t loaded = 0xFFFFFFFFu;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (!row_ok[r]) {
        continue;
      }
      if (expert[r] != loaded) {
        loaded = expert[r];
        const int64_t row_base = static_cast<int64_t>(loaded) *
                w_expert_stride +
            static_cast<int64_t>(col) * words_per_row +
            g * kNvfp4WordsPerGroup;
        const uint32_t w0 = w[row_base];
        const uint32_t w1 = w[row_base + 1];
        scale = e4m3_to_float_scaled(
            scales
                [static_cast<int64_t>(loaded) * s_expert_stride +
                 static_cast<int64_t>(col) * groups_per_row + g]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          wv[j] = fp4_e2m1_to_float_scaled((w0 >> (4 * j)) & 0xFu);
          wv[j + 8] = fp4_e2m1_to_float_scaled((w1 >> (4 * j)) & 0xFu);
        }
      }
      const T* xg = x_row[r] + g * kNvfp4GroupSize;
      float qdot = 0.0f;
#pragma unroll
      for (int j = 0; j < kNvfp4GroupSize; ++j) {
        qdot = fmaf(to_float(xg[j]), wv[j], qdot);
      }
      acc[r] = fmaf(scale, qdot, acc[r]);
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int offset = kNvfp4LanesPerCol / 2; offset > 0; offset /= 2) {
      v += __shfl_xor(v, offset, kNvfp4LanesPerCol);
    }
    const int64_t t = t0 + r;
    if (lane == 0 && t < rows) {
      // An out-of-range expert leaves row_ok false and its sum at zero.
      const float factor =
          row_ok[r] ? nvfp4_output_factor(global_scale, expert[r]) : 0.0f;
      out[t * N + col] = from_float<T>(v * factor);
    }
  }
}

// Dense dequantization of an NVFP4 plane into T, one 16-value group per
// thread per pass: out[16g + j] = e2m1(q_j) * e4m3(scale_g). Both factors
// have at most four significant bits and their product fits fp16 and bf16
// exactly, so the dequantized plane equals MLX's `dequantize`. Grid capped by
// copy_grid_blocks; the threads stride the groups on a 64-bit counter.
template <typename T>
__global__ void nvfp4_dequantize_kernel(
    const uint32_t* __restrict__ w,
    const uint8_t* __restrict__ scales,
    T* __restrict__ out,
    int64_t num_groups) {
  const float unscale = kFp4HalfBitsScale * kE4m3HalfBitsScale;
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       g < num_groups;
       g += stride) {
    const float s = e4m3_to_float_scaled(scales[g]) * unscale;
    const uint32_t w0 = w[g * kNvfp4WordsPerGroup];
    const uint32_t w1 = w[g * kNvfp4WordsPerGroup + 1];
    T* o = out + g * kNvfp4GroupSize;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[j] = from_float<T>(fp4_e2m1_to_float_scaled((w0 >> (4 * j)) & 0xFu) * s);
      o[j + 8] =
          from_float<T>(fp4_e2m1_to_float_scaled((w1 >> (4 * j)) & 0xFu) * s);
    }
  }
}

} // namespace rocm

namespace {

template <typename T>
struct nvfp4_type_tag {
  using type = T;
};

// Throws unless (bits, group_size) is NVFP4's 4-bit, group-16 layout and the
// quantized axis (K for a transposed weight, N otherwise) splits into whole
// groups.
inline void check_nvfp4_layout(
    const char* who,
    int bits,
    int group_size,
    int quantized_axis) {
  if (bits != 4 || group_size != rocm::kNvfp4GroupSize) {
    throw std::runtime_error(
        std::string(who) +
        " nvfp4 requires bits 4 and group_size 16; got bits=" +
        std::to_string(bits) + " group_size=" + std::to_string(group_size));
  }
  if (quantized_axis % rocm::kNvfp4GroupSize != 0) {
    throw std::runtime_error(
        std::string(who) +
        " nvfp4 requires the quantized axis to be a multiple of 16; got " +
        std::to_string(quantized_axis));
  }
}

// Dequantizes the NVFP4 plane (w, scales) into `out`, which has w's shape
// with the packed last axis expanded and the activation dtype.
inline void nvfp4_dequantize_dense(
    rocm::CommandEncoder& enc,
    const array& w,
    const array& scales,
    array& out) {
  const int64_t num_groups = scales.size();
  constexpr int block = 256;
  const uint32_t blocks = rocm::copy_grid_blocks(num_groups, block);
  auto launch = [&](auto type_tag) {
    using T = typename decltype(type_tag)::type;
    enc.add_kernel_node(
        &rocm::nvfp4_dequantize_kernel<T>,
        dim3(blocks),
        dim3(block),
        0u,
        gpu_ptr<const uint32_t>(w),
        gpu_ptr<const uint8_t>(scales),
        gpu_ptr<T>(out),
        num_groups);
  };
  switch (out.dtype()) {
    case float32:
      launch(nvfp4_type_tag<float>{});
      break;
    case float16:
      launch(nvfp4_type_tag<__half>{});
      break;
    case bfloat16:
      launch(nvfp4_type_tag<hip_bfloat16>{});
      break;
    default:
      throw std::runtime_error("[nvfp4] unsupported dequantize dtype");
  }
}

// Everything nvfp4_qmv_kernel reads besides the operands themselves.
struct Nvfp4QmvArgs {
  const uint32_t* lhs_indices = nullptr; // null for QuantizedMatmul
  const uint32_t* rhs_indices = nullptr;
  rocm::Shape batch_shape{};
  rocm::Strides lhs_idx_strides{};
  rocm::Strides rhs_idx_strides{};
  int batch_ndim = 0;
  int64_t rows = 0; // batch * M
  int M = 0;
  int N = 0;
  int K = 0;
  int E = 1;
  int64_t x_batch_stride = 0; // dense only
  bool w_batched = false; // dense only
  bool multi_row = false; // ROWS = 8 instead of 1
};

inline void nvfp4_qmv(
    rocm::CommandEncoder& enc,
    const array& x,
    const array& w,
    const array& scales,
    const float* global_scale,
    array& out,
    const Nvfp4QmvArgs& a) {
  if (a.rows == 0 || a.N == 0) {
    return;
  }
  const int rows_per_block = a.multi_row ? 8 : 1;
  const int64_t grid_x = (a.rows + rows_per_block - 1) / rows_per_block;
  const int64_t grid_y =
      (a.N + rocm::kNvfp4ColsPerBlock - 1) / rocm::kNvfp4ColsPerBlock;
  // HIP bounds the work items of a grid dimension (blocks times threads) by
  // UINT32_MAX, so grid x tops out at UINT32_MAX / kNvfp4LanesPerCol blocks,
  // not at INT32_MAX (the limit items 11 and 62 work around).
  constexpr int64_t kMaxGridX =
      std::numeric_limits<uint32_t>::max() / rocm::kNvfp4LanesPerCol;
  if (grid_x > kMaxGridX || grid_y > 65535) {
    throw std::runtime_error(
        "[nvfp4] output too large for one launch: rows=" +
        std::to_string(a.rows) + " N=" + std::to_string(a.N));
  }
  dim3 grid(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y));
  dim3 block(rocm::kNvfp4LanesPerCol, rocm::kNvfp4ColsPerBlock);
  auto launch = [&](auto type_tag, auto rows_tag) {
    using T = typename decltype(type_tag)::type;
    constexpr int ROWS = decltype(rows_tag)::value;
    enc.add_kernel_node(
        &rocm::nvfp4_qmv_kernel<T, ROWS>,
        grid,
        block,
        0u,
        gpu_ptr<const T>(x),
        gpu_ptr<const uint32_t>(w),
        gpu_ptr<const uint8_t>(scales),
        global_scale,
        a.lhs_indices,
        a.rhs_indices,
        a.batch_shape,
        a.lhs_idx_strides,
        a.rhs_idx_strides,
        a.batch_ndim,
        gpu_ptr<T>(out),
        a.rows,
        a.M,
        a.N,
        a.K,
        a.E,
        a.x_batch_stride,
        a.w_batched);
  };
  auto by_rows = [&](auto type_tag) {
    if (a.multi_row) {
      launch(type_tag, std::integral_constant<int, 8>{});
    } else {
      launch(type_tag, std::integral_constant<int, 1>{});
    }
  };
  switch (x.dtype()) {
    case float32:
      by_rows(nvfp4_type_tag<float>{});
      break;
    case float16:
      by_rows(nvfp4_type_tag<__half>{});
      break;
    case bfloat16:
      by_rows(nvfp4_type_tag<hip_bfloat16>{});
      break;
    default:
      throw std::runtime_error("[nvfp4] unsupported activation dtype");
  }
}

} // namespace

} // namespace mlx::core
