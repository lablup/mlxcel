// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <cuda_bf16.h>
#include <cuda/std/cmath>
#include <stdint.h>

// cuBLASLt FAST_TF32 uses round-to-nearest-even operand conversion. Keep this
// bit conversion explicit: sm_80 PTX only exposes cvt.rna.tf32.f32, while the
// cuBLAS contract rounds halfway values according to the retained mantissa LSB.
__device__ __forceinline__ float gemma3n_to_tf32_rne(float value) {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7F800000u) == 0x7F800000u) {
    return value;
  }
  const uint32_t retained_lsb = (bits >> 13) & 1u;
  return __uint_as_float((bits + 0x0FFFu + retained_lsb) & 0xFFFFE000u);
}

// Correctness-first Gemma3n Q4 prefill projection.
//
// IREE custom-dispatch ABI:
//   bindings: input f32[M,K], weight f32[N,K], output f32[M,N]
//   constants: M, N, K
//   workgroup: [32, 8, 1], grid: [ceildiv(N, 8), M, 1]
//
// The resident f32 buffers are BF16 carriers produced by the existing loader.
// Convert at the kernel boundary and preserve MLX CUDA's 16 independent BF16
// FMA chains, ordered f32 slot fold, and f32 warp-reduction schedule. Each warp
// owns one weight/output row in one M-row workgroup.
extern "C" __global__ void gemma3n_qmv(
    const float* __restrict__ input, const float* __restrict__ weight,
    float* __restrict__ output, int32_t m_rows, int32_t n_rows,
    int32_t k_width) {
  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t output_row =
      static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.y) +
      static_cast<int32_t>(threadIdx.y);
  const int32_t m = static_cast<int32_t>(blockIdx.y);
  if (output_row >= n_rows || m >= m_rows) {
    return;
  }

  const float* weight_row =
      weight + static_cast<int64_t>(output_row) * k_width;
  const float* input_row = input + static_cast<int64_t>(m) * k_width;
  __nv_bfloat16 sums[16] = {};

  // One lane owns 16 contiguous slots in each 512-element round. The loop
  // order is part of the numerical ABI: each i slot is an independent BF16
  // FMA chain across rounds.
  // K is host-validated as i32, but keep the 512-step loop arithmetic in i64
  // so the terminating multiplication cannot overflow at the i32 boundary.
  for (int64_t base = 0; base < k_width; base += 512) {
#pragma unroll
    for (int32_t i = 0; i < 16; ++i) {
      const int64_t k = lane * 16 + i + base;
      if (k < k_width) {
        const __nv_bfloat16 lhs = __float2bfloat16(input_row[k]);
        const __nv_bfloat16 rhs = __float2bfloat16(weight_row[k]);
        sums[i] = __hfma(lhs, rhs, sums[i]);
      }
    }
  }

  // Fold the 16 BF16 chains in ascending slot order using f32 additions.
  float sum = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < 16; ++i) {
    sum += __bfloat162float(sums[i]);
  }

  // Fixed f32 butterfly order matching the pinned MLX CUDA warp reduction.
  constexpr uint32_t kFullWarp = 0xFFFFFFFFu;
#pragma unroll
  for (int32_t mask = 16; mask >= 1; mask >>= 1) {
    sum += __shfl_xor_sync(kFullWarp, sum, mask);
  }

  if (lane == 0) {
    output[static_cast<int64_t>(m) * n_rows + output_row] =
        __bfloat162float(__float2bfloat16(sum));
  }
}

// Match MLX CUDA's f32 AltUp router nonlinearity exactly. MLX dispatches
// `cuda::std::tanh(float)` from its unary kernel; StableHLO/IREE's generic tanh
// lowering differs by an f32 ULP on the pinned checkpoint, which is enough to
// perturb the subsequent f32 4x16 coefficient projection and BF16 plane update.
extern "C" __global__ void gemma3n_tanh(
    const float* __restrict__ input, float* __restrict__ output,
    int32_t length) {
  const int32_t index = static_cast<int32_t>(blockIdx.x) *
                            static_cast<int32_t>(blockDim.x) +
                        static_cast<int32_t>(threadIdx.x);
  if (index < length) {
    output[index] = cuda::std::tanh(input[index]);
  }
}

// Match MLX CUDA's default cuBLASLt coefficient projection contract:
// CUBLAS_COMPUTE_32F_FAST_TF32 converts both f32 operands to TF32 before the
// f32 accumulation. AltUp's coefficient dimensions are tiny (K is normally 4),
// so one thread owns one output and pins the accumulation order.
extern "C" __global__ void gemma3n_altup_coeff(
    const float* __restrict__ input, const float* __restrict__ weight,
    float* __restrict__ output, int32_t rows, int32_t output_width,
    int32_t inner_width) {
  const int32_t index = static_cast<int32_t>(blockIdx.x) *
                            static_cast<int32_t>(blockDim.x) +
                        static_cast<int32_t>(threadIdx.x);
  // The host emitter proves rows * output_width <= INT32_MAX.
  const int32_t length = rows * output_width;
  if (index >= length) {
    return;
  }
  const int32_t row = index / output_width;
  const int32_t column = index - row * output_width;
  float sum = 0.0f;
  for (int32_t k = 0; k < inner_width; ++k) {
    const float lhs =
        gemma3n_to_tf32_rne(
            input[static_cast<int64_t>(row) * inner_width + k]);
    const float rhs =
        gemma3n_to_tf32_rne(
            weight[static_cast<int64_t>(column) * inner_width + k]);
    sum = fmaf(lhs, rhs, sum);
  }
  output[index] = sum;
}

// Match MLX CUDA's cuBLASLt FAST_TF32 batched AltUp prediction and the
// immediately observable BF16 activation boundary. `coefficients` has the
// logical [rows, source, target] layout produced by the reference transpose.
//
// One warp computes a [16, 8] tile from A[16, 4] and B[4, 8]. The logical
// AltUp target width is four; B columns 4..7 are zero padding and only D
// columns 0..3 are stored. This emits the same single m16n8k4 Tensor Core
// accumulation as the pinned cuBLASLt algorithm, then adds the residual in f32
// and rounds to BF16.
extern "C" __global__ void gemma3n_altup_predict(
    const float* __restrict__ planes,
    const float* __restrict__ coefficients, float* __restrict__ output,
    int32_t plane_count, int32_t rows, int32_t hidden) {
  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t group = lane >> 2;
  const int32_t thread_in_group = lane & 3;
  const int32_t row = static_cast<int32_t>(blockIdx.y);
  const int32_t feature_base = static_cast<int32_t>(blockIdx.x) * 16;
  const int32_t feature0 = feature_base + group;
  const int32_t feature1 = feature0 + 8;
  const int32_t plane_size = rows * hidden;

  const auto load_plane = [&](int32_t feature) {
    if (thread_in_group >= plane_count || feature >= hidden) {
      return 0.0f;
    }
    return planes[static_cast<int64_t>(thread_in_group) * plane_size +
                  static_cast<int64_t>(row) * hidden + feature];
  };
  const float a0_tf32 = gemma3n_to_tf32_rne(load_plane(feature0));
  const float a1_tf32 = gemma3n_to_tf32_rne(load_plane(feature1));

  float b_tf32 = 0.0f;
  if (thread_in_group < plane_count && group < plane_count) {
    b_tf32 = gemma3n_to_tf32_rne(
        coefficients[(static_cast<int64_t>(row) * plane_count +
                      thread_in_group) *
                         plane_count +
                     group]);
  }

  const uint32_t a0 = __float_as_uint(a0_tf32);
  const uint32_t a1 = __float_as_uint(a1_tf32);
  const uint32_t b0 = __float_as_uint(b_tf32);
  float d0;
  float d1;
  float d2;
  float d3;
  asm volatile(
      "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(b0), "f"(0.0f), "f"(0.0f), "f"(0.0f),
        "f"(0.0f));

  if (thread_in_group >= 2) {
    return;
  }
  const int32_t target0 = thread_in_group * 2;
  const int32_t target1 = target0 + 1;
  const auto store = [&](int32_t target, int32_t feature, float delta) {
    if (target >= plane_count || feature >= hidden) {
      return;
    }
    const int64_t index = static_cast<int64_t>(target) * plane_size +
                          static_cast<int64_t>(row) * hidden + feature;
    const float predicted = planes[index] + delta;
    output[index] = __bfloat162float(__float2bfloat16(predicted));
  };
  store(target0, feature0, d0);
  store(target1, feature0, d1);
  store(target0, feature1, d2);
  store(target1, feature1, d3);
}

// Reproduce MLX CUDA Compiled's generated GeGLU tape exactly. Its fused JIT
// declares every node in gelu_tanh_approx as __nv_bfloat16, including the
// dtype-authored scalar constants and every intermediate primitive result.
extern "C" __global__ void gemma3n_geglu_bf16(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, int32_t length) {
  const int32_t index = static_cast<int32_t>(blockIdx.x) *
                            static_cast<int32_t>(blockDim.x) +
                        static_cast<int32_t>(threadIdx.x);
  if (index >= length) {
    return;
  }

  const __nv_bfloat16 x = __float2bfloat16(gate[index]);
  const __nv_bfloat16 up_value = __float2bfloat16(up[index]);
  const __nv_bfloat16 half = __float2bfloat16(0.5f);
  const __nv_bfloat16 one = __float2bfloat16(1.0f);
  const __nv_bfloat16 root = __float2bfloat16(0.7978845608028654f);
  const __nv_bfloat16 cubic = __float2bfloat16(0.044715f);
  const __nv_bfloat16 x2 = x * x;
  const __nv_bfloat16 x3 = x2 * x;
  // MLX's generated CUDA is authored as separate BF16 multiply/add nodes, but
  // nvcc contracts them to HFMA2.BF16_V2. Use the scalar BF16 intrinsic so the
  // single-rounding behavior is explicit and stable across this AOT kernel.
  const __nv_bfloat16 inner_add = __hfma(x3, cubic, x);
  const __nv_bfloat16 inner = root * inner_add;
  const __nv_bfloat16 tanh_value = cuda::std::tanh(inner);
  const __nv_bfloat16 one_tanh = one + tanh_value;
  const __nv_bfloat16 cdf = half * one_tanh;
  const __nv_bfloat16 gelu = x * cdf;
  const __nv_bfloat16 product = gelu * up_value;
  output[index] = __bfloat162float(product);
}

// Token-exact decode attention schedule for the production CUDA bundle.
//
// This reproduces MLX CUDA's kernel_sdpav_1pass<bf16, false, 256> launch:
// one 1024-thread block per query head, one live key stream per warp, eight
// f32 FFMA lanes per thread, butterfly reductions, online base-2 softmax, and
// the shared-memory transpose used to combine all 32 warp accumulators.
//
// IREE custom-dispatch ABI:
//   bindings:
//     query  f32[query_heads, 256]       (BF16 carriers)
//     keys   f32[capacity, kv_heads, 256] (BF16 carriers)
//     values f32[capacity, kv_heads, 256] (BF16 carriers)
//     output f32[query_heads, 256]       (BF16 carriers)
//   constants: query_heads, kv_heads, capacity, position, window, scale_bits
//   workgroup: [1024, 1, 1], grid: [query_heads, 1, 1]
//
// `window == 0` or `window >= live` means full attention. Truncated windows
// require a mask/fallback and are rejected: MLX's vector kernel itself has no
// window input. The fixed-capacity time-major KV addressing intentionally
// differs from MLX's compact head-major view while preserving the same logical
// key order and arithmetic schedule.
__device__ __forceinline__ float gemma3n_sdpa_exp2(float value) {
  // MLX CUDA's instantiated exp2f path preserves subnormal-range results by
  // evaluating half the exponent and squaring. Pin that behavior explicitly
  // instead of relying on the target-specific libdevice lowering.
  const bool underflow_range = value < -126.0f;
  const float exponent = underflow_range ? value * 0.5f : value;
  const float result = exp2f(exponent);
  return underflow_range ? result * result : result;
}

extern "C" __global__ __launch_bounds__(1024)
void gemma3n_sdpa_vector(
    const float* __restrict__ query, const float* __restrict__ keys,
    const float* __restrict__ values, float* __restrict__ output,
    int32_t query_heads, int32_t kv_heads, int32_t capacity,
    int32_t position, int32_t window, int32_t scale_bits) {
  constexpr int32_t kWarpSize = 32;
  constexpr int32_t kHeadDim = 256;
  constexpr int32_t kValuesPerThread = kHeadDim / kWarpSize;
  constexpr uint32_t kFullWarp = 0xFFFFFFFFu;

  const int32_t head = static_cast<int32_t>(blockIdx.x);
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
  const int32_t live = position + 1;
  const float scale = __int_as_float(scale_bits);

  // Every predicate is block-uniform, so an invalid dispatch can
  // return before the barriers without creating partial-block deadlock.
  if (static_cast<int32_t>(blockDim.x) != 1024 || head >= query_heads ||
      query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 ||
      capacity <= 0 || live <= 0 || live > capacity || live > 1024 ||
      window < 0 || (window != 0 && window < live) || !(scale > 0.0f)) {
    return;
  }

  const int32_t group = query_heads / kv_heads;
  const int32_t kv_head = head / group;
  const float scale_log2 =
      static_cast<float>(static_cast<double>(scale) *
                         1.44269504088896340735992468100189214);

  float q[kValuesPerThread];
  float accumulated[kValuesPerThread];
#pragma unroll
  for (int32_t i = 0; i < kValuesPerThread; ++i) {
    q[i] =
        scale_log2 * query[static_cast<int64_t>(head) * kHeadDim +
                           kValuesPerThread * lane + i];
    accumulated[i] = 0.0f;
  }

  float max_score = -3.402823466e+38F;
  float sum_exp_score = 0.0f;

  // Warp `warp` owns keys first + warp, first + warp + 32, ... exactly as the
  // MLX BN=32 one-pass kernel owns kv_seq_idx + n*BN.
  for (int32_t key = warp; key < live; key += kWarpSize) {
    const int64_t kv_base =
        (static_cast<int64_t>(key) * kv_heads + kv_head) * kHeadDim;
    float score = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kValuesPerThread; ++i) {
      const float key_value =
          keys[kv_base + kValuesPerThread * lane + i];
      score = fmaf(q[i], key_value, score);
    }
#pragma unroll
    for (int32_t mask = 16; mask >= 1; mask >>= 1) {
      score += __shfl_xor_sync(kFullWarp, score, mask);
    }

    const float new_max = fmaxf(max_score, score);
    const float factor = gemma3n_sdpa_exp2(max_score - new_max);
    const float exp_score = gemma3n_sdpa_exp2(score - new_max);
    max_score = new_max;
    sum_exp_score = fmaf(sum_exp_score, factor, exp_score);
#pragma unroll
    for (int32_t i = 0; i < kValuesPerThread; ++i) {
      const float value =
          values[kv_base + kValuesPerThread * lane + i];
      accumulated[i] =
          fmaf(accumulated[i], factor, exp_score * value);
    }
  }

  __shared__ float outputs[kWarpSize][kWarpSize + 1];
  __shared__ float max_scores[kWarpSize];
  __shared__ float sum_exp_scores[kWarpSize];
  if (lane == 0) {
    max_scores[warp] = max_score;
    sum_exp_scores[warp] = sum_exp_score;
  }
  __syncthreads();

  // Each lane now represents one source warp. Preserve MLX's butterfly order
  // for both the global maximum and the rescaled denominator.
  max_score = max_scores[lane];
  float new_max = max_score;
#pragma unroll
  for (int32_t mask = 16; mask >= 1; mask >>= 1) {
    new_max = fmaxf(new_max, __shfl_xor_sync(kFullWarp, new_max, mask));
  }
  const float factor = gemma3n_sdpa_exp2(max_score - new_max);
  sum_exp_score = sum_exp_scores[lane] * factor;
#pragma unroll
  for (int32_t mask = 16; mask >= 1; mask >>= 1) {
    sum_exp_score +=
        __shfl_xor_sync(kFullWarp, sum_exp_score, mask);
  }
  sum_exp_score =
      sum_exp_score == 0.0f ? 0.0f : __frcp_rn(sum_exp_score);

  // Transpose [dimension-lane, source-warp]. Warp `warp` reduces the eight
  // output dimensions originally held by lane `warp` in every source warp.
#pragma unroll
  for (int32_t i = 0; i < kValuesPerThread; ++i) {
    outputs[lane][warp] = accumulated[i];
    __syncthreads();
    float combined = outputs[warp][lane] * factor;
#pragma unroll
    for (int32_t mask = 16; mask >= 1; mask >>= 1) {
      combined += __shfl_xor_sync(kFullWarp, combined, mask);
    }
    combined *= sum_exp_score;
    __syncthreads();
    if (lane == 0) {
      output[static_cast<int64_t>(head) * kHeadDim +
             kValuesPerThread * warp + i] =
          __bfloat162float(__float2bfloat16(combined));
    }
  }
}
