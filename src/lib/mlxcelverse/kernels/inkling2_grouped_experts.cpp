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

// Grouped native-quantized expert matvec for short Inkling 2 Mini verify
// forwards (issue #3359). See `inkling2_kernels.h` for the contract.
//
// Derived from mlx-vlm `mlx_vlm/models/inkling2/grouped_experts.py` (MIT,
// Copyright 2026 Neywa Labs Inc, Abdur Rahim), commit 8e80d434. The fp4 / fp8
// decoders and the scale and vector helpers in the header are MLX's
// `mlx/backend/metal/kernels/{fp4,fp8,fp_quantized}.h` (MIT, Copyright 2023
// Apple Inc.), inlined because the JIT preamble of `fast::metal_kernel` does
// not include them.
//
// One threadgroup per route; only the first route of each expert run works.
// It decodes four output channels of the expert per SIMD group once and dots
// them with up to four routed rows at a time, so a run of `n` routes reads the
// expert's weights `ceil(n / 4)` times instead of `n`. The lane assignment
// (`P` values per lane, `32 * P` per block), the 4-value dot grouping of
// NVFP4, the per-group scale product, the f32 accumulation order, the
// `simd_sum` and the global scale applied after it are those of MLX's
// `fp_qmv_fast_impl` (`P = 2 * 32 / bits`) and `fp_qmv_impl` (`P = 32 /
// bits`), which `gather_qmm` runs for one routed row, so each output equals
// the per-token path's bit for bit.

#include "inkling2_kernels.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "kernel_port.h"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mlxcel::turbo {
namespace {

constexpr const char* GROUPED_HEADER = R"(
constant constexpr float F8E4M3_MAX = 448.0f;
constant constexpr float F4E2M1_MAX = 6.0f;

inline float inkling2_fp4_e2m1(uint8_t bits) {
  half converted = as_type<half>(ushort((bits & 7) << 9));
  converted *= 16384.0;
  return static_cast<float>(bits & 8 ? -converted : converted);
}

inline float inkling2_fp8_e4m3(uint8_t bits) {
  uint16_t v = bits & 127;
  uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  half converted = as_type<half>(u);
  return static_cast<float>(half(converted * 256.0));
}

inline float inkling2_fp8_e8m0(uint8_t bits) {
  uint32_t out = (bits == 0)
      ? 0x400000
      : (bits == 0xFF ? 0x7FC00000 : (static_cast<uint16_t>(bits) << 23));
  return as_type<float>(out);
}

template <int group_size>
inline float inkling2_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return inkling2_fp8_e4m3(s);
  } else {
    return inkling2_fp8_e8m0(s);
  }
}

template <int V, int bits, int group_size, int P>
inline void inkling2_expert_tile(
    const device bfloat16_t* x, const device uint* weights, const device uchar* scales,
    device bfloat16_t* out, const thread int* src, const thread int* dst,
    int expert, int row, uint lane, int K, int N, float factor) {
  const size_t channel = size_t(expert) * N + row;
  const int weight_stride = K * bits / 8, scale_stride = K / group_size;
  const device uchar* packed = reinterpret_cast<const device uchar*>(weights)
      + channel * weight_stride;
  const device uchar* scale = scales + channel * scale_stride;
  float sum[V][4] = {};
  for (int block = 0; block < K; block += 32 * P) {
    int col = block + int(lane) * P;
    if (col >= K) break;
    float values[V][P];
#pragma unroll
    for (int v = 0; v < V; ++v) {
      const device bfloat16_t* xv = x + size_t(src[v]) * K + col;
#pragma unroll
      for (int i = 0; i < P; ++i) values[v][i] = xv[i];
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const device uchar* w = packed + r * weight_stride + col * bits / 8;
      float s = inkling2_scale<group_size>(scale[r * scale_stride + col / group_size]);
      float decoded[P];
#pragma unroll
      for (int i = 0; i < P; ++i) {
        if constexpr (bits == 4) decoded[i] = inkling2_fp4_e2m1(w[i / 2] >> (4 * (i % 2)));
        else decoded[i] = inkling2_fp8_e4m3(w[i]);
      }
#pragma unroll
      for (int v = 0; v < V; ++v) {
        float dot = 0;
        if constexpr (bits == 4) {
#pragma unroll
          for (int i = 0; i < P; i += 4) {
            dot += (values[v][i] * decoded[i] + values[v][i + 1] * decoded[i + 1]
                  + values[v][i + 2] * decoded[i + 2] + values[v][i + 3] * decoded[i + 3]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < P; ++i) dot += values[v][i] * decoded[i];
        }
        const float product = s * dot;
        sum[v][r] += product;
      }
    }
  }
#pragma unroll
  for (int v = 0; v < V; ++v) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      float result = simd_sum(sum[v][r]);
      if (lane == 0) out[size_t(dst[v]) * N + row + r] = bfloat16_t(result * factor);
    }
  }
}
)";

// Each run length has its own specialization, so a singleton route loads one
// activation row rather than four.
constexpr const char* GROUPED_SOURCE = R"(
    int first = threadgroup_position_in_grid.z;
    int expert = int(experts[first]);
    if (first > 0 && int(experts[first - 1]) == expert) return;
    int end = first + 1;
    while (end < int(experts_shape[0]) && int(experts[end]) == expert) ++end;
    int K = x_shape[1], N = w_shape[1];
    int row = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
    uint lane = thread_index_in_simdgroup;
    float factor = 1.0f;
    if constexpr (HAS_SCALE) factor = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
    for (int begin = first; begin < end; begin += 4) {
      int count = min(4, end - begin), src[4], dst[4];
      for (int i = 0; i < count; ++i) {
        src[i] = int(source[begin + i]);
        dst[i] = int(destination[begin + i]);
      }
      switch (count) {
        case 1: inkling2_expert_tile<1, BITS, GS, P>(x, w, s, out, src, dst, expert, row, lane, K, N, factor); break;
        case 2: inkling2_expert_tile<2, BITS, GS, P>(x, w, s, out, src, dst, expert, row, lane, K, N, factor); break;
        case 3: inkling2_expert_tile<3, BITS, GS, P>(x, w, s, out, src, dst, expert, row, lane, K, N, factor); break;
        case 4: inkling2_expert_tile<4, BITS, GS, P>(x, w, s, out, src, dst, expert, row, lane, K, N, factor); break;
      }
    }
)";

struct GroupedKernel {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag once;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(once, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_inkling2_grouped_expert_qmv",
                {"x", "w", "s", "g", "experts", "source", "destination"},
                {"out"},
                GROUPED_SOURCE,
                GROUPED_HEADER);
        });
        return *kernel;
    }
};

GroupedKernel& grouped_kernel() {
    static GroupedKernel holder;
    return holder;
}

// No CUDA or ROCm port: the gather path runs there.
const mlxcel::KernelPorts& grouped_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return grouped_kernel().get();
        },
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("[inkling2_grouped_expert_qmv] " + what);
}

} // namespace

mlx::core::array inkling2_grouped_expert_qmv(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array& scales,
    const std::optional<mlx::core::array>& global_scale,
    const mlx::core::array& experts,
    const mlx::core::array& source,
    const mlx::core::array& destination,
    int bits,
    int group_size) {
    using namespace mlx::core;
    auto& kernel = mlxcel::select_kernel_port(
        "inkling2_grouped_expert_qmv", "SwitchGLU gather fallback", grouped_ports());
    const bool nvfp4 = bits == 4 && group_size == 16;
    const bool mxfp8 = bits == 8 && group_size == 32;
    if (!nvfp4 && !mxfp8) fail("only NVFP4 (4 bits, group 16) and MXFP8 (8 bits, group 32)");
    if (global_scale && !nvfp4) fail("a global scale is NVFP4 only");
    if (x.ndim() != 2 || x.dtype() != bfloat16) fail("x must be a [rows, K] bf16 array");
    const int K = x.shape(1);
    if (K % 32 != 0) fail("K must be a multiple of 32");
    if (w.ndim() != 3 || w.dtype() != uint32 || w.shape(2) != K * bits / 32) {
        fail("w must be a uint32 [E, N, K * bits / 32] plane");
    }
    const int E = w.shape(0), N = w.shape(1);
    if (N % 8 != 0) fail("N must be a multiple of 8");
    if (scales.shape() != Shape{E, N, K / group_size} || scales.dtype() != uint8) {
        fail("scales must be a uint8 [E, N, K / group_size] plane");
    }
    if (global_scale && (global_scale->shape() != Shape{E} || global_scale->dtype() != float32)) {
        fail("the global scale must be an f32 [E] array");
    }
    const int routes = experts.size();
    if (experts.ndim() != 1 || source.shape() != experts.shape() ||
        destination.shape() != experts.shape()) {
        fail("experts, source and destination must be [routes]");
    }
    for (const auto* a : {&experts, &source, &destination}) {
        if (a->dtype() != uint32) fail("experts, source and destination must be uint32");
    }
    // MLX doubles the values per lane when two packs tile the input exactly
    // (its `qmv_fast` alignment); the tile then follows the same lanes.
    const int pack = 32 / bits;
    const int values_per_lane = K % (64 * pack) == 0 ? 2 * pack : pack;
    // The body reads `g` only with HAS_SCALE; the scales stand in otherwise.
    const array& g = global_scale ? *global_scale : scales;
    auto outputs = kernel(
        {x, w, scales, g, experts, source, destination},
        {Shape{routes, N}},
        {bfloat16},
        std::make_tuple(32, N / 8 * 2, routes),
        std::make_tuple(32, 2, 1),
        {{"BITS", bits}, {"GS", group_size}, {"P", values_per_lane}, {"HAS_SCALE", global_scale.has_value()}},
        std::nullopt,
        false,
        {});
    return outputs[0];
}

} // namespace mlxcel::turbo
