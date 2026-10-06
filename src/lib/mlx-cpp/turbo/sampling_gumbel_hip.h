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

// HIP body of the Gumbel-max sampling kernel (issue #2064).
//
// Kept apart from `sampling.cpp` only for size: the launch, and the
// `fast::hip_kernel` call that compiles this string, stay in `sampling.cpp`
// next to the CUDA launch, so `make verify-kernel-dtype-keys` keeps its pinned
// scope unchanged. This header holds data, not a launch.
//
// A line-for-line port of `GUMBEL_MAX_SAMPLE_CUDA_SOURCE`: same thread mapping,
// same grid (`(TgSize, NumSplits, B)` over `(TgSize, 1, 1)`, which the ROCm
// `CustomKernel::eval_gpu` ceil-divides into blocks `(1, NumSplits, B)` exactly
// as CUDA does), same Philox-4x32-10 counter `{base/4, 0, row, 0}` and key
// `{rng_key[0], rng_key[1]}`, same 23-bit open-interval uniform, same
// index-carrying halving reduction. A given key and logits row therefore pick
// the same index on Metal, CUDA and ROCm up to `logf` rounding, and
// `mlx::core::random::seed(...)` reproduces a ROCm stream the way it does on
// the other two. `__umulhi` is the HIP intrinsic with CUDA's meaning.
//
// Differences from the CUDA text, all spelling:
//
// - The initial best score is `-__builtin_huge_valf()`. hipRTC compiles this
//   body with only the headers `fast::hip_kernel` prepends, and `INFINITY` is a
//   `<cmath>` macro that is not guaranteed among them.
// - `temp[0]` gains a redundant `(float)` (it is float32). The logits read
//   keeps the CUDA text's explicit `(float)`, which is required here:
//   `hip_bfloat16`, the type MLX substitutes for bfloat16 on ROCm, converts to
//   float only through an `explicit` operator.
//
// No wave32 guard, deliberately. The guard that the shuffle-based #1814 ports
// carry protects a lane fold that starts at 16 and so silently drops half of a
// 64-lane wave. This kernel has no lane-level operation: every cross-thread
// step goes through `__shared__` memory with a `__syncthreads()` between
// steps, so it is correct for any wavefront size, and an `#error` on wave64
// would reject a correct kernel. (The guard idiom is also a no-op with ROCm
// 10's AMD clang, which defines neither wavefront macro; see #2067.)

namespace mlxcel::turbo {

inline constexpr const char* GUMBEL_MAX_SAMPLE_HIP_SOURCE = R"(
    uint32_t t = threadIdx.x;                     // 0 .. TgSize-1
    uint32_t split = blockIdx.y;                  // 0 .. NumSplits-1
    uint32_t row = blockIdx.z;                    // 0 .. B-1

    uint32_t vocab = (uint32_t)logits_shape[1];
    float temp_v = (float)temp[0];
    uint32_t key0_base = rng_key[0];
    uint32_t key1_base = rng_key[1];
    uint32_t row_off = row * vocab;

    float best = -__builtin_huge_valf();
    uint32_t best_idx = 0u;

    uint32_t chunk = (uint32_t)TgSize * 4u;
    uint32_t stride = chunk * (uint32_t)NumSplits;
    for (uint32_t base = split * chunk + t * 4u; base < vocab; base += stride) {
        // Philox-4x32-10 over counter {base/4, 0, row, 0}.
        uint32_t c0 = base >> 2;
        uint32_t c1 = 0u;
        uint32_t c2 = row;
        uint32_t c3 = 0u;
        uint32_t k0 = key0_base;
        uint32_t k1 = key1_base;
        for (uint32_t r = 0u; r < 10u; r++) {
            uint32_t hi0 = __umulhi(0xD2511F53u, c0);
            uint32_t lo0 = 0xD2511F53u * c0;
            uint32_t hi1 = __umulhi(0xCD9E8D57u, c2);
            uint32_t lo1 = 0xCD9E8D57u * c2;
            uint32_t n0 = hi1 ^ c1 ^ k0;
            uint32_t n1 = lo1;
            uint32_t n2 = hi0 ^ c3 ^ k1;
            uint32_t n3 = lo0;
            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }

        for (uint32_t j = 0u; j < 4u; j++) {
            uint32_t idx = base + j;
            if (idx >= vocab) {
                break;
            }
            uint32_t word =
                (j == 0u) ? c0 : ((j == 1u) ? c1 : ((j == 2u) ? c2 : c3));
            // Uniform on the OPEN interval (0, 1); see the Metal source for
            // why this takes 23 bits rather than 24.
            float u = ((float)(word >> 9) + 0.5f) * (1.0f / 8388608.0f);
            float g = -logf(-logf(u));
            float scaled = (float)logits[row_off + idx] / temp_v;
            float cand = scaled + g;
            if (cand > best || (cand == best && idx < best_idx)) {
                best = cand;
                best_idx = idx;
            }
        }
    }

    __shared__ float tg_val[TgSize];
    __shared__ uint32_t tg_idx[TgSize];
    tg_val[t] = best;
    tg_idx[t] = best_idx;
    __syncthreads();

    for (uint32_t s = (uint32_t)TgSize / 2u; s > 0u; s >>= 1) {
        if (t < s) {
            float other = tg_val[t + s];
            uint32_t other_idx = tg_idx[t + s];
            if (other > tg_val[t] ||
                (other == tg_val[t] && other_idx < tg_idx[t])) {
                tg_val[t] = other;
                tg_idx[t] = other_idx;
            }
        }
        __syncthreads();
    }

    if (t == 0u) {
        uint32_t out_off = row * (uint32_t)NumSplits + split;
        vals[out_off] = tg_val[0];
        idxs[out_off] = tg_idx[0];
    }
)";

} // namespace mlxcel::turbo
