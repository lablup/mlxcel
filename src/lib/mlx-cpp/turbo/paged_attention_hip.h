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

// HIP bodies of the three paged-attention kernels (issue #2068): the v1 split-K
// decode, the v2 partial and the v2 merge.
//
// Kept apart from the launchers only for size. Each launch, and the
// `fast::hip_kernel` call that compiles its string, stays in the `.cpp` that
// already holds the CUDA launch (`paged_attention.cpp`,
// `paged_attention_v2.cpp`, `paged_attention_v2_merge.cpp`), so
// `make verify-kernel-dtype-keys` keeps its pinned scope. This header holds
// data, not a launch.
//
// Each body is a line-for-line port of the CUDA body next to its launcher:
// same thread mapping, inputs, outputs, grid and template arguments. The ROCm
// `CustomKernel::eval_gpu` ceil-divides the Metal-style total-thread grid by
// the threadgroup tuple exactly as CUDA does, so every `blockIdx` / `threadIdx`
// below means what it means in the CUDA body. The differences are spelling:
//
// - The 32-lane all-reduce is `__shfl_xor(v, o, 32)`. `__shfl_xor_sync` exists
//   in HIP only as a compatibility shim that ignores its mask, so the native
//   form is used with the width stated.
// - Negative infinity is `-__builtin_huge_valf()`. hipRTC compiles these
//   bodies with only the headers `fast::hip_kernel` prepends, and `INFINITY` is
//   a `<cmath>` macro that is not guaranteed among them.
// - The KV reads keep the CUDA text's explicit `(float)`: `hip_bfloat16`, the
//   type MLX substitutes for bfloat16 on ROCm, converts to float only through
//   an `explicit` operator.
// - The geometry is read from `<input>_shape` exactly as the CUDA bodies read
//   it. That relies on the overlay passing shapes by value, which it does
//   since lablup/mlxcel#2100 (`patches-rocm/LOCAL_FIXES.md` item 30); before
//   that fix a HIP body that named `<input>_shape` faulted the queue.
//
// Wave32 guard. The v1 and v2-partial bodies fold a dot product across 32
// lanes with an XOR butterfly that starts at 16. They carry the `#error` guard
// every #1814 port carries, spelled as the bitlinear and SSM ports spell it
// (`static_assert(warpSize == 32)` does not compile in HIP). The guard is inert
// with ROCm 10's AMD clang, which defines neither macro for gfx1151, gfx942 or
// gfx90a (#2067). The fold is written not to need it: the 32 lanes that share a
// `threadIdx.y` are consecutive in the block's linear order, so on a wave64
// target the explicit width of 32 should keep each butterfly inside its own
// row (not run: this host has no wave64 device), which is why both tables stay
// wave32-only and `port_for` refuses them on a wider device (#2147). The
// merge body has no lane-level operation (one thread per output element, no
// shuffle and no barrier), so it is correct for any wavefront size, carries
// no guard (an `#error` there would only reject a correct kernel), and
// `paged_merge_ports()` is marked `rocm_any_wave_size`.

namespace mlxcel::turbo {

// v1 split-K flash-decoding (port of `PAGED_ATTENTION_DECODE_CUDA_SOURCE`).
// Grid `(32, NumSplits, B*Hq)` over threadgroup `(32, NumSplits, 1)`: one block
// per (batch, query head), 32 lanes over the head dimension, `NumSplits` warps
// over strided token stripes, merged through shared memory.
inline constexpr const char* PAGED_ATTENTION_DECODE_HIP_SOURCE = R"(
    #if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
    #error "mlxcel_paged_attention_decode assumes a 32-lane wavefront"
    #endif
    #if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
    #error "mlxcel_paged_attention_decode assumes a 32-lane wavefront"
    #endif
    const float neg_inf = -__builtin_huge_valf();

    uint32_t lane = threadIdx.x;                      // 0 .. 31 (within warp)
    uint32_t sg = threadIdx.y;                        // 0 .. NumSplits-1
    uint32_t bhq = blockIdx.z;                        // 0 .. B*Hq-1

    uint32_t hq_count = (uint32_t)q_shape[1];         // Hq
    uint32_t block_size = (uint32_t)k_pool_shape[1];  // tokens per block
    uint32_t hkv_count = (uint32_t)k_pool_shape[2];   // Hkv
    uint32_t dim = (uint32_t)Dim;
    uint32_t dpt = (uint32_t)DimsPerThread;           // dims this lane owns
    uint32_t d0 = lane * dpt;                         // first dim of this lane

    uint32_t b = bhq / hq_count;                      // batch index
    uint32_t h = bhq % hq_count;                      // query head
    uint32_t kv_head = h / (uint32_t)NRep;            // grouped-query KV head
    if (kv_head >= hkv_count) {
        kv_head = 0;                                  // defensive
    }

    int vlen_i = visible_lens[b];
    uint32_t vlen = vlen_i > 0 ? (uint32_t)vlen_i : 0u;
    uint32_t logical_start = (uint32_t)logical_starts[b];
    uint32_t row_off = (uint32_t)row_offsets[b];

    float q_reg[DimsPerThread];
    for (uint32_t j = 0; j < dpt; j++) {
        uint32_t d = d0 + j;
        q_reg[j] = (d < dim) ? (float)q[bhq * dim + d] : 0.0f;
    }

    __shared__ float tg_m[NumSplits];
    __shared__ float tg_l[NumSplits];
    __shared__ float tg_acc[NumSplits * Dim];

    // Empty window: block-uniform (a function of the batch index only), so the
    // whole block returns together and no warp is stranded at the barrier.
    if (vlen == 0u) {
        if (sg == 0u) {
            for (uint32_t j = 0; j < dpt; j++) {
                uint32_t d = d0 + j;
                if (d < dim) {
                    out[bhq * dim + d] = 0.0f;
                }
            }
        }
        return;
    }

    float scale_v = (float)scale[0];

    float m = neg_inf;
    float l = 0.0f;
    float acc[DimsPerThread];
    for (uint32_t j = 0; j < dpt; j++) {
        acc[j] = 0.0f;
    }

    uint32_t stride_kv = hkv_count * dim;             // elements per (block,slot)
    for (uint32_t t = sg; t < vlen; t += (uint32_t)NumSplits) {
        uint32_t abs_pos = logical_start + t;
        uint32_t block_idx = abs_pos / block_size;
        uint32_t slot = abs_pos - block_idx * block_size;
        uint32_t row = (uint32_t)rows[row_off + block_idx];
        // 64-bit: one slab can hold more than 2^32 elements (#2153).
        uint64_t base = ((uint64_t)row * block_size + slot) * (uint64_t)stride_kv + kv_head * dim;

        float partial = 0.0f;
        for (uint32_t j = 0; j < dpt; j++) {
            uint32_t d = d0 + j;
            float kd = (d < dim) ? (float)k_pool[base + d] : 0.0f;
            partial += q_reg[j] * kd;
        }
        // Butterfly all-reduce over the 32 lanes of this row; the trip count
        // is warp-uniform, so every lane reaches each shuffle.
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            partial += __shfl_xor(partial, o, 32);
        }
        float score = partial * scale_v;

        float m_new = fmaxf(m, score);
        float corr = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * corr + p;
        for (uint32_t j = 0; j < dpt; j++) {
            uint32_t d = d0 + j;
            float vd = (d < dim) ? (float)v_pool[base + d] : 0.0f;
            acc[j] = acc[j] * corr + p * vd;
        }
        m = m_new;
    }

    for (uint32_t j = 0; j < dpt; j++) {
        uint32_t d = d0 + j;
        if (d < dim) {
            tg_acc[sg * dim + d] = acc[j];
        }
    }
    if (lane == 0u) {
        tg_m[sg] = m;
        tg_l[sg] = l;
    }
    __syncthreads();

    if (sg == 0u) {
        float m_g = tg_m[0];
        for (uint32_t s = 1; s < (uint32_t)NumSplits; s++) {
            m_g = fmaxf(m_g, tg_m[s]);
        }
        float l_g = 0.0f;
        for (uint32_t s = 0; s < (uint32_t)NumSplits; s++) {
            l_g += tg_l[s] * expf(tg_m[s] - m_g);
        }
        float inv_l = l_g > 0.0f ? (1.0f / l_g) : 0.0f;
        for (uint32_t j = 0; j < dpt; j++) {
            uint32_t d = d0 + j;
            if (d < dim) {
                float a = 0.0f;
                for (uint32_t s = 0; s < (uint32_t)NumSplits; s++) {
                    a += tg_acc[s * dim + d] * expf(tg_m[s] - m_g);
                }
                out[bhq * dim + d] = a * inv_l;
            }
        }
    }
)";

// v2 partial (port of `PAGED_ATTENTION_V2_PARTIAL_CUDA_SOURCE`). Grid
// `(32, NumWarps * Hkv * QGroups, num_chunks)` over threadgroup
// `(32, NumWarps, 1)`: one block per (chunk, KV head, query-head group).
// Scores are in base 2, as in the Metal and CUDA bodies, so the emitted LSE is
// in the units the merge kernel consumes.
inline constexpr const char* PAGED_ATTENTION_V2_PARTIAL_HIP_SOURCE = R"(
    #if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
    #error "mlxcel_paged_attention_v2_partial assumes a 32-lane wavefront"
    #endif
    #if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
    #error "mlxcel_paged_attention_v2_partial assumes a 32-lane wavefront"
    #endif
    const float neg_inf = -__builtin_huge_valf();

    uint32_t lane = threadIdx.x;                      // 0 .. 31 (within warp)
    uint32_t sg = threadIdx.y;                        // 0 .. NumWarps-1
    uint32_t yblk = blockIdx.y;                       // kv_head * QGroups + qgrp
    uint32_t chunk = blockIdx.z;                      // flat (request, tile) id

    uint32_t dim = (uint32_t)Dim;
    uint32_t dpt = (uint32_t)DimsPerThread;
    uint32_t d0 = lane * dpt;
    uint32_t page_size = (uint32_t)PageSize;

    uint32_t hq_count = (uint32_t)q_shape[1];
    uint32_t hkv_count = (uint32_t)k_pool_shape[2];

    uint32_t kv_head = yblk / (uint32_t)QGroups;
    uint32_t q_group = yblk - kv_head * (uint32_t)QGroups;
    if (kv_head >= hkv_count) {
        kv_head = 0;
    }
    uint32_t q_base = kv_head * (uint32_t)NRep + q_group * (uint32_t)QHeads;

    int req_i = request_indices[chunk];
    int tile_i = kv_tile_indices[chunk];
    uint32_t r = req_i > 0 ? (uint32_t)req_i : 0u;
    uint32_t tile = tile_i > 0 ? (uint32_t)tile_i : 0u;

    uint32_t page_begin = (uint32_t)indptr[r];
    uint32_t page_end = (uint32_t)indptr[r + 1];
    uint32_t npages = page_end > page_begin ? page_end - page_begin : 0u;
    uint32_t fpo = (uint32_t)first_page_offset[r];
    uint32_t lpl = (uint32_t)last_page_len[r];
    uint32_t seq_len = 0u;
    if (npages > 0u) {
        uint32_t total = (npages - 1u) * page_size + lpl;
        seq_len = total > fpo ? total - fpo : 0u;
    }

    uint32_t ppc = (uint32_t)params[0];
    uint32_t chunk_page_end = page_begin + (tile + 1u) * ppc;
    if (chunk_page_end > page_end) {
        chunk_page_end = page_end;
    }
    uint32_t t_begin = tile * ppc * page_size;
    t_begin = t_begin > fpo ? t_begin - fpo : 0u;
    uint32_t t_end;
    if (chunk_page_end >= page_end) {
        t_end = seq_len;
    } else {
        uint32_t span = (chunk_page_end - page_begin) * page_size;
        t_end = span > fpo ? span - fpo : 0u;
    }
    if (t_end > seq_len) {
        t_end = seq_len;
    }

    float q_reg[QHeads * DimsPerThread];
    for (uint32_t g = 0; g < (uint32_t)QHeads; g++) {
        uint32_t h = q_base + g;
        for (uint32_t j = 0; j < dpt; j++) {
            uint32_t d = d0 + j;
            q_reg[g * dpt + j] = (h < hq_count && d < dim)
                ? (float)q[(r * hq_count + h) * dim + d]
                : 0.0f;
        }
    }

    float m[QHeads];
    float l[QHeads];
    float acc[QHeads * DimsPerThread];
    for (uint32_t g = 0; g < (uint32_t)QHeads; g++) {
        m[g] = neg_inf;
        l[g] = 0.0f;
        for (uint32_t j = 0; j < dpt; j++) {
            acc[g * dpt + j] = 0.0f;
        }
    }

    float scale_v = (float)scale[0] * 1.4426950408889634f;
    uint32_t stride_kv = hkv_count * dim;

    for (uint32_t t = t_begin + sg; t < t_end; t += (uint32_t)NumWarps) {
        uint32_t abs_pos = fpo + t;
        uint32_t page_off = abs_pos / page_size;
        uint32_t entry = abs_pos - page_off * page_size;
        uint32_t row = (uint32_t)indices[page_begin + page_off];
        // 64-bit: one slab can hold more than 2^32 elements (#2153).
        uint64_t base = ((uint64_t)row * page_size + entry) * (uint64_t)stride_kv + kv_head * dim;

        float k_reg[DimsPerThread];
        float v_reg[DimsPerThread];
        for (uint32_t j = 0; j < dpt; j++) {
            uint32_t d = d0 + j;
            k_reg[j] = (d < dim) ? (float)k_pool[base + d] : 0.0f;
            v_reg[j] = (d < dim) ? (float)v_pool[base + d] : 0.0f;
        }

        for (uint32_t g = 0; g < (uint32_t)QHeads; g++) {
            float partial = 0.0f;
            for (uint32_t j = 0; j < dpt; j++) {
                partial += q_reg[g * dpt + j] * k_reg[j];
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                partial += __shfl_xor(partial, o, 32);
            }
            float score = partial * scale_v;
            float m_new = fmaxf(m[g], score);
            float corr = exp2f(m[g] - m_new);
            float p = exp2f(score - m_new);
            l[g] = l[g] * corr + p;
            for (uint32_t j = 0; j < dpt; j++) {
                acc[g * dpt + j] = acc[g * dpt + j] * corr + p * v_reg[j];
            }
            m[g] = m_new;
        }
    }

    __shared__ float tg_m[NumWarps * QHeads];
    __shared__ float tg_l[NumWarps * QHeads];
    __shared__ float tg_acc[NumWarps * QHeads * Dim];

    for (uint32_t g = 0; g < (uint32_t)QHeads; g++) {
        for (uint32_t j = 0; j < dpt; j++) {
            uint32_t d = d0 + j;
            if (d < dim) {
                tg_acc[(sg * (uint32_t)QHeads + g) * dim + d] = acc[g * dpt + j];
            }
        }
        if (lane == 0u) {
            tg_m[sg * (uint32_t)QHeads + g] = m[g];
            tg_l[sg * (uint32_t)QHeads + g] = l[g];
        }
    }
    __syncthreads();

    if (sg == 0u) {
        for (uint32_t g = 0; g < (uint32_t)QHeads; g++) {
            uint32_t h = q_base + g;
            if (h >= hq_count) {
                continue;
            }
            float m_g = tg_m[g];
            for (uint32_t s = 1; s < (uint32_t)NumWarps; s++) {
                m_g = fmaxf(m_g, tg_m[s * (uint32_t)QHeads + g]);
            }
            float l_g = 0.0f;
            if (m_g > neg_inf) {
                for (uint32_t s = 0; s < (uint32_t)NumWarps; s++) {
                    uint32_t idx = s * (uint32_t)QHeads + g;
                    l_g += tg_l[idx] * exp2f(tg_m[idx] - m_g);
                }
            }
            float inv_l = l_g > 0.0f ? (1.0f / l_g) : 0.0f;
            uint32_t out_base = (chunk * hq_count + h) * dim;
            for (uint32_t j = 0; j < dpt; j++) {
                uint32_t d = d0 + j;
                if (d < dim) {
                    float a = 0.0f;
                    if (l_g > 0.0f) {
                        for (uint32_t s = 0; s < (uint32_t)NumWarps; s++) {
                            uint32_t idx = s * (uint32_t)QHeads + g;
                            a += tg_acc[idx * dim + d] * exp2f(tg_m[idx] - m_g);
                        }
                    }
                    out_v[out_base + d] = a * inv_l;
                }
            }
            if (lane == 0u) {
                out_lse[chunk * hq_count + h] =
                    l_g > 0.0f ? (m_g + log2f(l_g)) : neg_inf;
            }
        }
    }
)";

// v2 merge (port of `PAGED_ATTENTION_MERGE_CUDA_SOURCE`). Grid `(Dim, H, M)`
// over threadgroup `(Dim, 1, 1)` yields blocks `(1, H, M)`. No barrier, so the
// early `return` is safe.
inline constexpr const char* PAGED_ATTENTION_MERGE_HIP_SOURCE = R"(
    const float neg_inf = -__builtin_huge_valf();

    uint32_t d = threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t o = blockIdx.z;

    uint32_t dim = (uint32_t)Dim;
    uint32_t heads = (uint32_t)v_in_shape[1];
    if (d >= dim || h >= heads) {
        return;
    }

    uint32_t begin = (uint32_t)o_indptr[o];
    uint32_t end = (uint32_t)o_indptr[o + 1];

    float m = neg_inf;
    float l = 0.0f;
    float acc = 0.0f;
    for (uint32_t i = begin; i < end; i++) {
        float s = (float)lse_in[i * heads + h];
        // Skips -inf (an empty chunk) and any NaN, so an empty partial can
        // never poison the rescale with (-inf) - (-inf).
        if (!(s > neg_inf)) {
            continue;
        }
        float m_new = fmaxf(m, s);
        float corr = exp2f(m - m_new);
        float w = exp2f(s - m_new);
        l = l * corr + w;
        acc = acc * corr + w * (float)v_in[(i * heads + h) * dim + d];
        m = m_new;
    }

    out_v[(o * heads + h) * dim + d] = l > 0.0f ? (acc / l) : 0.0f;
    if (d == 0u) {
        out_lse[o * heads + h] = l > 0.0f ? (m + log2f(l)) : neg_inf;
    }
)";

} // namespace mlxcel::turbo
