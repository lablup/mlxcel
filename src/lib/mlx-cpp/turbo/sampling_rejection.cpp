// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

#include "sampling_rejection.h"

#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/random.h>

// Backend availability probes for the Metal-vs-CUDA kernel gate; both headers
// resolve on either backend (the absent one links its no_* stub), exactly as in
// `sampling.cpp`.
#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Threads per threadgroup. One threadgroup owns one row for the whole rejection
// loop, because the loop's shrinking interval is carried across vocabulary
// sweeps and MLX custom kernels have no grid-wide barrier to carry it across
// threadgroups. 256 matches the launch width validated for the Gumbel kernel in
// `sampling.cpp`; the block scan and the halving-tree reductions below need a
// power of two.
constexpr int REJECTION_TG_SIZE = 256;

// Metal body of the dual-pivot rejection sampler. The string is the kernel BODY
// only; `mlx::core::fast::metal_kernel` wraps it with the declaration and the
// buffer arguments.
//
// One threadgroup per row, `TgSize` threads, no split: the interval state
// (`low`, `high`, per-thread proposal mass) lives in registers and threadgroup
// memory across rounds, so every round's sweep must be executed by the same
// threadgroup.
//
// Vocabulary partition: thread `t` owns indices `t, t + TgSize, t + 2*TgSize,
// ...`. Coalesced, and it also fixes the CDF order used by the inverse-CDF
// draw (all of thread 0's elements, then thread 1's, and so on). Any fixed
// permutation samples the same distribution; fixing it is what makes the draw
// reproducible.
//
// Template constants:
//   TgSize    - threads per threadgroup (power of two).
//   MaxRounds - rejection rounds before the row is declared unconverged.
//
// Buffers (order matches the launcher's input vector):
//   probs   [B, V]   f32  softmax probabilities, one row per batch entry
//   params  [B, 3]   f32  {top_k, top_p, min_p} per row
//   rng_key [2]      u32  Philox key drawn from MLX's key sequence
//   ids     [B]      u32  sampled token id
//   ok      [B]      u32  1 when the loop accepted inside the round cap
//   rounds  [B]      u32  rounds the row consumed
constexpr const char* REJECTION_SAMPLE_SOURCE = R"(
    uint t = thread_position_in_threadgroup.x;   // 0 .. TgSize-1
    uint row = threadgroup_position_in_grid.z;   // 0 .. B-1

    const uint tg = (uint)TgSize;
    uint vocab = (uint)probs_shape[1];
    uint row_off = row * vocab;

    // Per-row filter parameters. Threadgroup-uniform, so every branch that
    // tests them is uniform and the barriers inside them are safe.
    float top_k_f = params[row * 3u + 0u];
    float top_p   = params[row * 3u + 1u];
    float min_p   = params[row * 3u + 2u];
    uint  top_k   = (top_k_f >= 1.0f) ? (uint)top_k_f : 0u;
    bool use_k  = (top_k > 0u) && (top_k < vocab);
    bool use_p  = (top_p > 0.0f) && (top_p < 1.0f);
    bool use_mp = (min_p > 0.0f) && (min_p < 1.0f);

    threadgroup float sh_max[TgSize];
    threadgroup uint  sh_arg[TgSize];
    threadgroup uint  sh_own[TgSize];
    threadgroup float sh_scan[TgSize];
    threadgroup float sh_c0[TgSize];
    threadgroup float sh_m0[TgSize];
    threadgroup float sh_c1[TgSize];
    threadgroup float sh_m1[TgSize];
    threadgroup uint  bcu[2];

    // ---- sweep 1: row maximum (index carrying) and row mass.
    float local_max = -1.0f;
    uint  local_arg = 0u;
    float local_sum = 0.0f;
    for (uint i = t; i < vocab; i += tg) {
        float p = probs[row_off + i];
        local_sum += p;
        if (p > local_max) {
            local_max = p;
            local_arg = i;
        }
    }

    sh_max[t] = local_max;
    sh_arg[t] = local_arg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1) {
        if (t < s) {
            float o = sh_max[t + s];
            uint oi = sh_arg[t + s];
            if (o > sh_max[t] || (o == sh_max[t] && oi < sh_arg[t])) {
                sh_max[t] = o;
                sh_arg[t] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float p_max = sh_max[0];
    uint  arg_max = sh_arg[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row mass, reduced with the same partition and the same halving tree the
    // pivot masses below use, so `mass(> v) == total` holds exactly (not just
    // to within rounding) when the pivot sits under every entry.
    sh_scan[t] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1) {
        if (t < s) {
            sh_scan[t] += sh_scan[t + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = sh_scan[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- min-p is a plain threshold, so it is folded into the initial
    // interval instead of being tested per round: the proposal starts as
    // { p : p >= min_p * p_max } and never has to shrink for min-p again.
    // `low` is an EXCLUSIVE bound, so the inclusive threshold is spelled as the
    // largest float strictly below it. That is one integer decrement of the bit
    // pattern, which a fast-math backend cannot perturb.
    float low = 0.0f;
    float part = local_sum;
    if (use_mp) {
        float tau_min = min_p * p_max;
        uint tau_bits = as_type<uint>(tau_min);
        if (tau_bits > 0u) {
            low = as_type<float>(tau_bits - 1u);
            part = 0.0f;
            for (uint i = t; i < vocab; i += tg) {
                float p = probs[row_off + i];
                if (p > low) {
                    part += p;
                }
            }
        }
    }

    float high = p_max;
    float pmass_target = use_p ? (top_p * total) : 0.0f;
    uint sampled = arg_max;
    uint ok_flag = 0u;
    uint rounds_used = 0u;

    // A row whose logits are entirely -inf softmaxes to NaN, and a NaN row has
    // no meaningful support. Guard the whole loop on a positive maximum (NaN
    // fails the test), report the row as converged, and leave `sampled` at the
    // row's argmax so the caller does not see a spurious cap overflow. The
    // branch is threadgroup-uniform, so the barriers inside stay well formed.
    if (!(p_max > 0.0f)) {
        ok_flag = 1u;
    }

    for (uint round = 0u; (p_max > 0.0f) && round < (uint)MaxRounds; round++) {
        // ---- proposal mass and per-thread exclusive CDF offsets. Hillis-
        // Steele inclusive scan over threadgroup memory: fixed step order, so
        // the summation order does not depend on scheduling.
        sh_scan[t] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < tg; off <<= 1) {
            float add = (t >= off) ? sh_scan[t - off] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sh_scan[t] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float incl = sh_scan[t];
        float mass_prop = sh_scan[tg - 1u];
        float excl = incl - part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- one uniform per (row, round). Philox-4x32-10 over counter
        // {round, 0, row, 0}: stateless, so the draw is a pure function of the
        // launch key and never depends on how the threadgroup is scheduled.
        uint c0 = round;
        uint c1 = 0u;
        uint c2 = row;
        uint c3 = 0u;
        uint k0 = rng_key[0];
        uint k1 = rng_key[1];
        for (uint r = 0u; r < 10u; r++) {
            uint hi0 = mulhi(0xD2511F53u, c0);
            uint lo0 = 0xD2511F53u * c0;
            uint hi1 = mulhi(0xCD9E8D57u, c2);
            uint lo1 = 0xCD9E8D57u * c2;
            uint n0 = hi1 ^ c1 ^ k0;
            uint n1 = lo1;
            uint n2 = hi0 ^ c3 ^ k1;
            uint n3 = lo0;
            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        // Uniform on the OPEN interval (0, 1) on a 2^-23 grid offset by half a
        // step; see `sampling.cpp` for why this takes 23 bits and not 24.
        float u = ((float)(c0 >> 9) + 0.5f) * (1.0f / 8388608.0f);
        float target = u * mass_prop;

        // ---- inverse-CDF draw over the proposal set. A thread can only hold
        // the crossing if its inclusive prefix passes the target; the lowest
        // such thread in scan order owns it. Reducing over "which thread has a
        // crossing" rather than testing `excl <= target < incl` makes the pick
        // immune to the last-bit disagreement between a Hillis-Steele prefix
        // and `incl - part`, which would otherwise leave a hairline gap
        // between adjacent threads that no thread claims.
        uint pick = 0xFFFFFFFFu;
        if (incl > target) {
            float run = excl;
            for (uint i = t; i < vocab; i += tg) {
                float p = probs[row_off + i];
                if (p > low) {
                    run += p;
                    if (run > target) {
                        pick = i;
                        break;
                    }
                }
            }
        }

        sh_own[t] = (pick == 0xFFFFFFFFu) ? 0xFFFFFFFFu : t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg / 2u; s > 0u; s >>= 1) {
            if (t < s) {
                uint o = sh_own[t + s];
                if (o < sh_own[t]) {
                    sh_own[t] = o;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint owner = sh_own[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // No thread claimed a crossing (only reachable when rounding leaves the
        // target past the last accumulated element). The row maximum is always
        // inside the filtered support, so it is the one fallback that cannot
        // produce an out-of-support token.
        if (t == 0u) {
            bcu[0] = arg_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (owner != 0xFFFFFFFFu && t == owner) {
            bcu[0] = pick;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint cand = bcu[0];

        float pivot0 = probs[row_off + cand];

        // Bisection pivot as the midpoint of the IEEE-754 BIT PATTERNS of
        // `pivot0` and `high`, not their arithmetic mean. The bit pattern of a
        // positive float is monotone in its value, so this is a valid
        // bisection, and it is pure integer arithmetic that flush-to-zero
        // cannot touch. It also isolates a single float in at most 32 rounds at
        // any magnitude, where arithmetic bisection of [0, 1] never resolves a
        // boundary that sits at 1e-6. `>> 1` on each side before adding keeps
        // the sum from overflowing.
        float hi_eff = (pivot0 > high) ? pivot0 : high;
        uint b0 = as_type<uint>(pivot0);
        uint bh = as_type<uint>(hi_eff);
        uint bm = (b0 >> 1u) + (bh >> 1u) + (b0 & bh & 1u);
        float pivot1 = as_type<float>(bm);

        // ---- (count, mass) above both pivots. Skipped outright when neither
        // top-k nor top-p is active (min-p alone needs no pivot at all), which
        // is threadgroup-uniform, so the barriers inside stay well formed.
        float cnt0 = 0.0f;
        float mass0 = 0.0f;
        float cnt1 = 0.0f;
        float mass1 = 0.0f;
        float m0acc = 0.0f;
        float m1acc = 0.0f;
        if (use_k || use_p) {
            float c0acc = 0.0f;
            float c1acc = 0.0f;
            for (uint i = t; i < vocab; i += tg) {
                float p = probs[row_off + i];
                if (p > pivot0) {
                    c0acc += 1.0f;
                    m0acc += p;
                }
                if (p > pivot1) {
                    c1acc += 1.0f;
                    m1acc += p;
                }
            }
            sh_c0[t] = c0acc;
            sh_m0[t] = m0acc;
            sh_c1[t] = c1acc;
            sh_m1[t] = m1acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tg / 2u; s > 0u; s >>= 1) {
                if (t < s) {
                    sh_c0[t] += sh_c0[t + s];
                    sh_m0[t] += sh_m0[t + s];
                    sh_c1[t] += sh_c1[t + s];
                    sh_m1[t] += sh_m1[t + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            cnt0 = sh_c0[0];
            mass0 = sh_m0[0];
            cnt1 = sh_c1[0];
            mass1 = sh_m1[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // A value fails when the filtered support lies strictly above it.
        // Monotone decreasing in the value, which is what makes the bisection
        // valid and what makes `!fail(candidate)` the exact membership test.
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

        // Shrink. `pivot1 >= pivot0 > low`, so both updates raise `low`
        // strictly and the bit-space bracket at least halves every round.
        if (fail1) {
            low = pivot1;
            part = m1acc;
        } else {
            low = pivot0;
            high = pivot1;
            part = m0acc;
        }
    }

    if (t == 0u) {
        ids[row] = sampled;
        ok[row] = ok_flag;
        rounds[row] = rounds_used;
    }
)";

// CUDA port of the dual-pivot rejection sampler. Structurally identical to the
// Metal source above: same partition, same scan, same Philox counters, same
// bit-space bisection, so both backends resolve the same support and draw the
// same index for the same key.
//
// Grid mapping (MLX passes Metal-style total threads and ceil-divides by the
// threadgroup tuple, cf. backend/cuda/custom_kernel.cpp): grid `(TgSize, 1, B)`
// over threadgroup `(TgSize, 1, 1)` yields blocks `(1, 1, B)` with
// `blockDim = (TgSize, 1, 1)`, an exact multiple in every field, so no padded
// threads are launched. Every `__syncthreads()` is reached by every thread:
// the sweeps carry no barriers, and each conditional that wraps one is
// block-uniform (`use_k`, `use_p`, `use_mp` come from the row's parameters, and
// the loop's `break` is taken on a value all threads reduced).
//
// `mulhi` becomes `__umulhi`, `as_type<uint>` / `as_type<float>` become
// `__float_as_uint` / `__uint_as_float`.
//
// UNVALIDATED: written from the Metal source for parity, but no CUDA hardware
// and no `nvcc` were available when this landed, so it has never been compiled
// or run. Same status as the CUDA strings in `sampling.cpp`,
// `fused_norm.cpp`, and `fused_rope_append.cpp`.
constexpr const char* REJECTION_SAMPLE_CUDA_SOURCE = R"(
    uint32_t t = threadIdx.x;   // 0 .. TgSize-1
    uint32_t row = blockIdx.z;  // 0 .. B-1

    const uint32_t tg = (uint32_t)TgSize;
    uint32_t vocab = (uint32_t)probs_shape[1];
    uint32_t row_off = row * vocab;

    float top_k_f = params[row * 3u + 0u];
    float top_p   = params[row * 3u + 1u];
    float min_p   = params[row * 3u + 2u];
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

    float local_max = -1.0f;
    uint32_t local_arg = 0u;
    float local_sum = 0.0f;
    for (uint32_t i = t; i < vocab; i += tg) {
        float p = probs[row_off + i];
        local_sum += p;
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
    float part = local_sum;
    if (use_mp) {
        float tau_min = min_p * p_max;
        uint32_t tau_bits = __float_as_uint(tau_min);
        if (tau_bits > 0u) {
            low = __uint_as_float(tau_bits - 1u);
            part = 0.0f;
            for (uint32_t i = t; i < vocab; i += tg) {
                float p = probs[row_off + i];
                if (p > low) {
                    part += p;
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
                float p = probs[row_off + i];
                if (p > low) {
                    run += p;
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

        float pivot0 = probs[row_off + cand];

        float hi_eff = (pivot0 > high) ? pivot0 : high;
        uint32_t b0 = __float_as_uint(pivot0);
        uint32_t bh = __float_as_uint(hi_eff);
        uint32_t bm = (b0 >> 1u) + (bh >> 1u) + (b0 & bh & 1u);
        float pivot1 = __uint_as_float(bm);

        float cnt0 = 0.0f;
        float mass0 = 0.0f;
        float cnt1 = 0.0f;
        float mass1 = 0.0f;
        float m0acc = 0.0f;
        float m1acc = 0.0f;
        if (use_k || use_p) {
            float c0acc = 0.0f;
            float c1acc = 0.0f;
            for (uint32_t i = t; i < vocab; i += tg) {
                float p = probs[row_off + i];
                if (p > pivot0) {
                    c0acc += 1.0f;
                    m0acc += p;
                }
                if (p > pivot1) {
                    c1acc += 1.0f;
                    m1acc += p;
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
            part = m1acc;
        } else {
            low = pivot0;
            high = pivot1;
            part = m0acc;
        }
    }

    if (t == 0u) {
        ids[row] = sampled;
        ok[row] = ok_flag;
        rounds[row] = rounds_used;
    }
)";

// Thread-safe lazy-initialised holders for the JIT-compiled kernels, matching
// the `std::call_once` pattern in `sampling.cpp` and `paged_attention.cpp`: the
// server reaches first use concurrently from per-request blocking workers, and
// `call_once` re-runs the initializer if MLX device lookup throws.
struct RejectionKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_rejection_sample",
                {"probs", "params", "rng_key"},
                {"ids", "ok", "rounds"},
                std::string(REJECTION_SAMPLE_SOURCE));
        });
        return *kernel;
    }
};

inline RejectionKernelHolder& get_rejection_kernel() {
    static RejectionKernelHolder holder;
    return holder;
}

struct RejectionKernelHolderCuda {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::cuda_kernel(
                "mlxcel_rejection_sample",
                {"probs", "params", "rng_key"},
                {"ids", "ok", "rounds"},
                std::string(REJECTION_SAMPLE_CUDA_SOURCE));
        });
        return *kernel;
    }
};

inline RejectionKernelHolderCuda& get_rejection_kernel_cuda() {
    static RejectionKernelHolderCuda holder;
    return holder;
}

} // namespace

bool rejection_sample_supported() {
    if (mlx::core::default_device() != mlx::core::Device::gpu) {
        return false;
    }
    return mlx::core::metal::is_available() || mlx::core::cu::is_available();
}

bool rejection_sample_accepts(const mlx::core::array& probs) {
    if (probs.ndim() != 2) {
        return false;
    }
    return probs.shape(0) > 0 && probs.shape(1) > 0;
}

int rejection_threadgroup_size() {
    return REJECTION_TG_SIZE;
}

RejectionSampleResult rejection_sample(
    const mlx::core::array& probs,
    const mlx::core::array& params,
    int max_rounds) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    const int batch = probs.shape(0);
    const int rounds = max_rounds > 0 ? max_rounds : 1;

    // Metal kernel on Apple, CUDA port elsewhere. `mx.fast.metal_kernel` throws
    // "[metal_kernel] No Metal back-end" on the CUDA backend, so dispatch the
    // `cuda_kernel` port there; both share the template args, grid, and buffer
    // contract.
    const bool use_cuda = !mlx::core::metal::is_available();
    auto& kernel = use_cuda ? get_rejection_kernel_cuda().get()
                            : get_rejection_kernel().get();

    // One key per call, drawn from MLX's default (thread-local) PRNG key
    // sequence, the same stream `random::categorical` consumes. A call
    // therefore advances the shared random state once and
    // `mlx::core::random::seed(...)` reproduces it. Two u32 words are the
    // Philox key; the counter carries the (round, row) pair.
    auto rng_key = mlx::core::random::bits(Shape{2}, 4);

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"TgSize", REJECTION_TG_SIZE},
        {"MaxRounds", rounds},
    };

    std::vector<mlx::core::array> inputs = {probs, params, rng_key};
    std::vector<Shape> output_shapes = {
        Shape{batch},
        Shape{batch},
        Shape{batch},
    };
    std::vector<Dtype> output_dtypes = {
        mlx::core::uint32,
        mlx::core::uint32,
        mlx::core::uint32,
    };

    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(REJECTION_TG_SIZE, 1, batch), // grid
        std::make_tuple(REJECTION_TG_SIZE, 1, 1),     // threadgroup
        template_args,
        std::nullopt,
        false,
        {});

    return RejectionSampleResult{results[0], results[1], results[2]};
}

} // namespace mlxcel::turbo
