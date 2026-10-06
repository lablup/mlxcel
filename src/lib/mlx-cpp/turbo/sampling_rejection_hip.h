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

// HIP body of the dual-pivot rejection sampler (issue #2064).
//
// Kept apart from `sampling_rejection.cpp` only for size: the launch, and the
// `fast::hip_kernel` call that compiles this string, stay in
// `sampling_rejection.cpp` next to the CUDA launch, so
// `make verify-kernel-dtype-keys` keeps its pinned scope unchanged. This
// header holds data, not a launch.
//
// A line-for-line port of `REJECTION_SAMPLE_CUDA_SOURCE`: same partition (one
// block of `TgSize` threads per row, thread `t` owning entries `t, t + TgSize,
// ...`), same Hillis-Steele scan, same Philox-4x32-10 counter
// `{round, 0, row, 0}` and key `{rng_key[0], rng_key[1]}`, same bit-space
// bisection. The grid `(TgSize, 1, B)` over `(TgSize, 1, 1)` becomes blocks
// `(1, 1, B)` in the ROCm `CustomKernel::eval_gpu` exactly as on CUDA. A given
// key and probability row therefore resolve the same support and draw the same
// index as the CUDA and Metal ports, and `mlx::core::random::seed(...)`
// reproduces a ROCm stream.
//
// Differences from the CUDA text, all spelling: every `probs`, `probs_draw`
// and `params` read is an explicit `(float)` conversion, because `hip_bfloat16`
// (MLX's bfloat16 type on ROCm) converts to float only through an `explicit`
// operator, and `rejection_sample_accepts` places no dtype restriction on the
// inputs. `__umulhi`, `__float_as_uint` and `__uint_as_float` are the HIP
// intrinsics with CUDA's meaning.
//
// No wave32 guard, deliberately. The guard that the shuffle-based #1814 ports
// carry protects a lane fold that starts at 16 and so silently drops half of a
// 64-lane wave. This kernel has no lane-level operation: the max, sum and
// owner reductions and the prefix scan all go through `__shared__` memory with
// a `__syncthreads()` between steps, so it is correct for any wavefront size,
// and an `#error` on wave64 would reject a correct kernel. (The guard idiom is
// also a no-op with ROCm 10's AMD clang, which defines neither wavefront
// macro; see #2067.) Every barrier is reached by every thread for the reason
// the CUDA source gives: the conditionals around them are block-uniform.

namespace mlxcel::turbo {

inline constexpr const char* REJECTION_SAMPLE_HIP_SOURCE = R"(
    uint32_t t = threadIdx.x;   // 0 .. TgSize-1
    uint32_t row = blockIdx.z;  // 0 .. B-1

    const uint32_t tg = (uint32_t)TgSize;
    uint32_t vocab = (uint32_t)probs_shape[1];
    uint32_t row_off = row * vocab;

    float top_k_f = (float)params[row * 3u + 0u];
    float top_p   = (float)params[row * 3u + 1u];
    float min_p   = (float)params[row * 3u + 2u];
    uint32_t top_k = (top_k_f >= 1.0f) ? (uint32_t)top_k_f : 0u;
    bool use_k  = (top_k > 0u) && (top_k < vocab);
    bool use_p  = (top_p > 0.0f) && (top_p < 1.0f);
    bool use_mp = (min_p > 0.0f) && (min_p < 1.0f);

    __shared__ float sh_max[TgSize];
    __shared__ uint32_t sh_arg[TgSize];
    __shared__ uint32_t sh_own[TgSize];
    __shared__ float sh_scan[TgSize];
    __shared__ float sh_c0[TgSize];
    __shared__ float sh_m0[TgSize];
    __shared__ float sh_c1[TgSize];
    __shared__ float sh_m1[TgSize];
    __shared__ uint32_t bcu[2];

    // See the Metal source: the draw-row sum is gated on filter membership so
    // the proposal mass never counts a token the pick loop cannot reach.
    float local_max = -1.0f;
    uint32_t local_arg = 0u;
    float local_sum = 0.0f;
    float local_draw = 0.0f;
    for (uint32_t i = t; i < vocab; i += tg) {
        float p = (float)probs[row_off + i];
        local_sum += p;
        if (p > 0.0f) {
            local_draw += (float)probs_draw[row_off + i];
        }
        if (p > local_max) {
            local_max = p;
            local_arg = i;
        }
    }

    sh_max[t] = local_max;
    sh_arg[t] = local_arg;
    __syncthreads();
    for (uint32_t s = tg / 2u; s > 0u; s >>= 1) {
        if (t < s) {
            float o = sh_max[t + s];
            uint32_t oi = sh_arg[t + s];
            if (o > sh_max[t] || (o == sh_max[t] && oi < sh_arg[t])) {
                sh_max[t] = o;
                sh_arg[t] = oi;
            }
        }
        __syncthreads();
    }
    float p_max = sh_max[0];
    uint32_t arg_max = sh_arg[0];
    __syncthreads();

    sh_scan[t] = local_sum;
    __syncthreads();
    for (uint32_t s = tg / 2u; s > 0u; s >>= 1) {
        if (t < s) {
            sh_scan[t] += sh_scan[t + s];
        }
        __syncthreads();
    }
    float total = sh_scan[0];
    __syncthreads();

    float low = 0.0f;
    float part = local_draw;
    if (use_mp) {
        float tau_min = min_p * p_max;
        uint32_t tau_bits = __float_as_uint(tau_min);
        if (tau_bits > 0u) {
            low = __uint_as_float(tau_bits - 1u);
            part = 0.0f;
            for (uint32_t i = t; i < vocab; i += tg) {
                float p = (float)probs[row_off + i];
                if (p > low) {
                    part += (float)probs_draw[row_off + i];
                }
            }
        }
    }

    float high = p_max;
    float pmass_target = use_p ? (top_p * total) : 0.0f;
    uint32_t sampled = arg_max;
    uint32_t ok_flag = 0u;
    uint32_t rounds_used = 0u;

    // See the Metal source: a fully -inf row softmaxes to NaN and is served by
    // its argmax without entering the loop.
    if (!(p_max > 0.0f)) {
        ok_flag = 1u;
    }

    for (uint32_t round = 0u; (p_max > 0.0f) && round < (uint32_t)MaxRounds;
         round++) {
        sh_scan[t] = part;
        __syncthreads();
        for (uint32_t off = 1u; off < tg; off <<= 1) {
            float add = (t >= off) ? sh_scan[t - off] : 0.0f;
            __syncthreads();
            sh_scan[t] += add;
            __syncthreads();
        }
        float incl = sh_scan[t];
        float mass_prop = sh_scan[tg - 1u];
        float excl = incl - part;
        __syncthreads();

        uint32_t c0 = round;
        uint32_t c1 = 0u;
        uint32_t c2 = row;
        uint32_t c3 = 0u;
        uint32_t k0 = rng_key[0];
        uint32_t k1 = rng_key[1];
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
        float u = ((float)(c0 >> 9) + 0.5f) * (1.0f / 8388608.0f);
        float target = u * mass_prop;

        uint32_t pick = 0xFFFFFFFFu;
        if (incl > target) {
            float run = excl;
            for (uint32_t i = t; i < vocab; i += tg) {
                float p = (float)probs[row_off + i];
                if (p > low) {
                    run += (float)probs_draw[row_off + i];
                    if (run > target) {
                        pick = i;
                        break;
                    }
                }
            }
        }

        sh_own[t] = (pick == 0xFFFFFFFFu) ? 0xFFFFFFFFu : t;
        __syncthreads();
        for (uint32_t s = tg / 2u; s > 0u; s >>= 1) {
            if (t < s) {
                uint32_t o = sh_own[t + s];
                if (o < sh_own[t]) {
                    sh_own[t] = o;
                }
            }
            __syncthreads();
        }
        uint32_t owner = sh_own[0];
        __syncthreads();

        if (t == 0u) {
            bcu[0] = arg_max;
        }
        __syncthreads();
        if (owner != 0xFFFFFFFFu && t == owner) {
            bcu[0] = pick;
        }
        __syncthreads();
        uint32_t cand = bcu[0];

        float pivot0 = (float)probs[row_off + cand];

        float hi_eff = (pivot0 > high) ? pivot0 : high;
        uint32_t b0 = __float_as_uint(pivot0);
        uint32_t bh = __float_as_uint(hi_eff);
        uint32_t bm = (b0 >> 1u) + (bh >> 1u) + (b0 & bh & 1u);
        float pivot1 = __uint_as_float(bm);

        float cnt0 = 0.0f;
        float mass0 = 0.0f;
        float cnt1 = 0.0f;
        float mass1 = 0.0f;
        float d0acc = 0.0f;
        float d1acc = 0.0f;
        if (use_k || use_p) {
            float c0acc = 0.0f;
            float c1acc = 0.0f;
            float m0acc = 0.0f;
            float m1acc = 0.0f;
            for (uint32_t i = t; i < vocab; i += tg) {
                float p = (float)probs[row_off + i];
                float pd = (float)probs_draw[row_off + i];
                if (p > pivot0) {
                    c0acc += 1.0f;
                    m0acc += p;
                    d0acc += pd;
                }
                if (p > pivot1) {
                    c1acc += 1.0f;
                    m1acc += p;
                    d1acc += pd;
                }
            }
            sh_c0[t] = c0acc;
            sh_m0[t] = m0acc;
            sh_c1[t] = c1acc;
            sh_m1[t] = m1acc;
            __syncthreads();
            for (uint32_t s = tg / 2u; s > 0u; s >>= 1) {
                if (t < s) {
                    sh_c0[t] += sh_c0[t + s];
                    sh_m0[t] += sh_m0[t + s];
                    sh_c1[t] += sh_c1[t + s];
                    sh_m1[t] += sh_m1[t + s];
                }
                __syncthreads();
            }
            cnt0 = sh_c0[0];
            mass0 = sh_m0[0];
            cnt1 = sh_c1[0];
            mass1 = sh_m1[0];
            __syncthreads();
        }

        bool fail0 = (use_k && cnt0 >= (float)top_k) ||
                     (use_p && mass0 > pmass_target);
        bool fail1 = (use_k && cnt1 >= (float)top_k) ||
                     (use_p && mass1 > pmass_target);

        rounds_used = round + 1u;
        if (!fail0) {
            sampled = cand;
            ok_flag = 1u;
            break;
        }

        if (fail1) {
            low = pivot1;
            part = d1acc;
        } else {
            low = pivot0;
            high = pivot1;
            part = d0acc;
        }
    }

    if (t == 0u) {
        ids[row] = sampled;
        ok[row] = ok_flag;
        rounds[row] = rounds_used;
    }
)";

} // namespace mlxcel::turbo
