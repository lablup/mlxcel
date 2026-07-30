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

#include "paged_attention_v2.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

// Backend availability probes for the Metal-vs-CUDA kernel gate, same contract
// as `paged_attention.cpp`: both headers are backend-agnostic, so
// `is_available()` always resolves and returns false for the absent backend.
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

// Body of the v2 partial kernel (Metal). `mlx::core::fast::metal_kernel` wraps
// this with the declaration, buffer arguments, and template substitution.
//
// One threadgroup owns one `(chunk, kv head, q-head group)` triple:
//
//   - grid z  : flat chunk index, i.e. one `(request, tile)` pair from the plan.
//   - grid y  : `kv_head * QGroups + q_group`, so the query heads of one KV head
//               are split into `QGroups` CTAs of `QHeads` heads each and the KV
//               element loaded by this CTA is reused across all of them.
//   - 32 lanes: partition the head dimension, `DimsPerThread` dims per lane, so
//               the QK dot product is a barrier-free `simd_sum`.
//   - NumWarps: stripe the chunk's tokens, merged at the end through
//               threadgroup memory (the same flash rescale v1 performs).
//
// Scores are computed in base 2: `log2(e)` is folded into the attention scale
// so the online softmax uses `exp2`, and the emitted LSE is
// `max_2 + log2(sum exp2(score_2 - max_2))`. The merge kernel consumes exactly
// those units.
//
// Template constants:
//   Dim           - head dimension D.
//   PageSize      - tokens per pool page (compile-time so `/` and `%` become
//                   shifts for the power-of-two page sizes in use).
//   NRep          - Hq / Hkv (grouped-query replication factor).
//   QHeads        - query heads this CTA owns; divides NRep.
//   QGroups       - NRep / QHeads.
//   DimsPerThread - ceil(Dim / 32).
//   NumWarps      - SIMD groups per threadgroup.
//
// Buffers (order matches the launcher's input vector):
//   q                 [B, Hq, 1, Dim]                    f32
//   k_pool            [num_blocks, PageSize, Hkv, Dim]   f16
//   v_pool            [num_blocks, PageSize, Hkv, Dim]   f16
//   indices           [total_pages]                      i32
//   indptr            [B + 1]                            i32
//   last_page_len     [B]                                i32
//   first_page_offset [B]                                i32
//   request_indices   [num_chunks]                       i32
//   kv_tile_indices   [num_chunks]                       i32
//   params            [1]                                i32 (pages_per_chunk)
//   scale             [1]                                f32
//   out_v             [num_chunks, Hq, Dim]              f32
//   out_lse           [num_chunks, Hq]                   f32
constexpr const char* PAGED_ATTENTION_V2_PARTIAL_SOURCE = R"(
    uint lane = thread_position_in_threadgroup.x;    // 0 .. 31 (within SIMD grp)
    uint sg = thread_position_in_threadgroup.y;      // 0 .. NumWarps-1
    uint yblk = threadgroup_position_in_grid.y;      // kv_head * QGroups + qgrp
    uint chunk = threadgroup_position_in_grid.z;     // flat (request, tile) id

    uint dim = (uint)Dim;
    uint dpt = (uint)DimsPerThread;
    uint d0 = lane * dpt;                            // first dim of this lane
    uint page_size = (uint)PageSize;

    uint hq_count = (uint)q_shape[1];                // Hq
    uint hkv_count = (uint)k_pool_shape[2];          // Hkv

    uint kv_head = yblk / (uint)QGroups;
    uint q_group = yblk - kv_head * (uint)QGroups;
    if (kv_head >= hkv_count) {
        kv_head = 0;                                 // defensive
    }
    uint q_base = kv_head * (uint)NRep + q_group * (uint)QHeads;

    int req_i = request_indices[chunk];
    int tile_i = kv_tile_indices[chunk];
    uint r = req_i > 0 ? (uint)req_i : 0u;
    uint tile = tile_i > 0 ? (uint)tile_i : 0u;

    // CSR page range and visible-token window of this request.
    uint page_begin = (uint)indptr[r];
    uint page_end = (uint)indptr[r + 1];
    uint npages = page_end > page_begin ? page_end - page_begin : 0u;
    uint fpo = (uint)first_page_offset[r];
    uint lpl = (uint)last_page_len[r];
    uint seq_len = 0u;
    if (npages > 0u) {
        uint total = (npages - 1u) * page_size + lpl;
        seq_len = total > fpo ? total - fpo : 0u;
    }

    // Token half-open range [t_begin, t_end) this chunk covers, derived from
    // its page range so the arithmetic matches the host plan exactly.
    uint ppc = (uint)params[0];
    uint chunk_page_end = page_begin + (tile + 1u) * ppc;
    if (chunk_page_end > page_end) {
        chunk_page_end = page_end;
    }
    uint t_begin = tile * ppc * page_size;
    t_begin = t_begin > fpo ? t_begin - fpo : 0u;
    uint t_end;
    if (chunk_page_end >= page_end) {
        t_end = seq_len;
    } else {
        uint span = (chunk_page_end - page_begin) * page_size;
        t_end = span > fpo ? span - fpo : 0u;
    }
    if (t_end > seq_len) {
        t_end = seq_len;
    }

    // Stage this lane's Q slice for every query head this CTA owns.
    float q_reg[QHeads * DimsPerThread];
    for (uint g = 0; g < (uint)QHeads; g++) {
        uint h = q_base + g;
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            q_reg[g * dpt + j] = (h < hq_count && d < dim)
                ? q[(r * hq_count + h) * dim + d]
                : 0.0f;
        }
    }

    // Per-head online softmax state for this warp's token stripe.
    float m[QHeads];
    float l[QHeads];
    float acc[QHeads * DimsPerThread];
    for (uint g = 0; g < (uint)QHeads; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        for (uint j = 0; j < dpt; j++) {
            acc[g * dpt + j] = 0.0f;
        }
    }

    // log2(e) folded into the scale: the whole softmax then runs on exp2.
    float scale_v = scale[0] * 1.4426950408889634f;
    uint stride_kv = hkv_count * dim;

    for (uint t = t_begin + sg; t < t_end; t += (uint)NumWarps) {
        uint abs_pos = fpo + t;
        uint page_off = abs_pos / page_size;
        uint entry = abs_pos - page_off * page_size;
        uint row = (uint)indices[page_begin + page_off];
        uint base = (row * page_size + entry) * stride_kv + kv_head * dim;

        float k_reg[DimsPerThread];
        float v_reg[DimsPerThread];
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            k_reg[j] = (d < dim) ? (float)k_pool[base + d] : 0.0f;
            v_reg[j] = (d < dim) ? (float)v_pool[base + d] : 0.0f;
        }

        for (uint g = 0; g < (uint)QHeads; g++) {
            float partial = 0.0f;
            for (uint j = 0; j < dpt; j++) {
                partial += q_reg[g * dpt + j] * k_reg[j];
            }
            float score = simd_sum(partial) * scale_v;  // full q . k_t, no barrier
            float m_new = fmax(m[g], score);
            float corr = exp2(m[g] - m_new);
            float p = exp2(score - m_new);
            l[g] = l[g] * corr + p;
            for (uint j = 0; j < dpt; j++) {
                acc[g * dpt + j] = acc[g * dpt + j] * corr + p * v_reg[j];
            }
            m[g] = m_new;
        }
    }

    // Publish every warp's partial, then warp 0 merges them and writes the
    // chunk's normalized output plus its LSE.
    threadgroup float tg_m[NumWarps * QHeads];
    threadgroup float tg_l[NumWarps * QHeads];
    threadgroup float tg_acc[NumWarps * QHeads * Dim];

    for (uint g = 0; g < (uint)QHeads; g++) {
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            if (d < dim) {
                tg_acc[(sg * (uint)QHeads + g) * dim + d] = acc[g * dpt + j];
            }
        }
        if (lane == 0u) {
            tg_m[sg * (uint)QHeads + g] = m[g];
            tg_l[sg * (uint)QHeads + g] = l[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        for (uint g = 0; g < (uint)QHeads; g++) {
            uint h = q_base + g;
            if (h >= hq_count) {
                continue;
            }
            float m_g = tg_m[g];
            for (uint s = 1; s < (uint)NumWarps; s++) {
                m_g = fmax(m_g, tg_m[s * (uint)QHeads + g]);
            }
            // An all-empty chunk leaves m_g at -inf; skip the rescale so no
            // (-inf) - (-inf) NaN can reach the output.
            float l_g = 0.0f;
            if (m_g > -INFINITY) {
                for (uint s = 0; s < (uint)NumWarps; s++) {
                    uint idx = s * (uint)QHeads + g;
                    l_g += tg_l[idx] * exp2(tg_m[idx] - m_g);
                }
            }
            float inv_l = l_g > 0.0f ? (1.0f / l_g) : 0.0f;
            uint out_base = (chunk * hq_count + h) * dim;
            for (uint j = 0; j < dpt; j++) {
                uint d = d0 + j;
                if (d < dim) {
                    float a = 0.0f;
                    if (l_g > 0.0f) {
                        for (uint s = 0; s < (uint)NumWarps; s++) {
                            uint idx = s * (uint)QHeads + g;
                            a += tg_acc[idx * dim + d] * exp2(tg_m[idx] - m_g);
                        }
                    }
                    out_v[out_base + d] = a * inv_l;
                }
            }
            if (lane == 0u) {
                out_lse[chunk * hq_count + h] =
                    l_g > 0.0f ? (m_g + log2(l_g)) : -INFINITY;
            }
        }
    }
)";

// CUDA port of the v2 partial kernel.
//
// **Unvalidated.** Issue #898 was implemented on an Apple Silicon host with no
// CUDA hardware and no `nvcc`, so this body has never been compiled or run. It
// is a structural transliteration of the Metal source above: same thread
// mapping, same chunk arithmetic, same accumulation order. The only substantive
// differences are the ones the two languages force:
//
//   - `simd_sum` becomes a butterfly `__shfl_xor_sync` all-reduce, so every
//     lane ends up with the full dot product exactly as `simd_sum` leaves it.
//   - `threadgroup` becomes `__shared__` and the barrier becomes
//     `__syncthreads()`. No thread returns early before that barrier, so no
//     warp can be stranded.
//   - `exp2`/`log2`/`fmax` become `exp2f`/`log2f`/`fmaxf`.
//
// Grid mapping: MLX ceil-divides the Metal-style total-thread grid by the
// threadgroup tuple (backend/cuda/custom_kernel.cpp), so grid
// `(32, NumWarps * Hkv * QGroups, num_chunks)` over threadgroup
// `(32, NumWarps, 1)` yields blocks `(1, Hkv * QGroups, num_chunks)` with
// `blockDim = (32, NumWarps, 1)`. Every field divides exactly, so no padded
// threads are launched.
constexpr const char* PAGED_ATTENTION_V2_PARTIAL_CUDA_SOURCE = R"(
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
                ? q[(r * hq_count + h) * dim + d]
                : 0.0f;
        }
    }

    float m[QHeads];
    float l[QHeads];
    float acc[QHeads * DimsPerThread];
    for (uint32_t g = 0; g < (uint32_t)QHeads; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        for (uint32_t j = 0; j < dpt; j++) {
            acc[g * dpt + j] = 0.0f;
        }
    }

    float scale_v = scale[0] * 1.4426950408889634f;
    uint32_t stride_kv = hkv_count * dim;

    for (uint32_t t = t_begin + sg; t < t_end; t += (uint32_t)NumWarps) {
        uint32_t abs_pos = fpo + t;
        uint32_t page_off = abs_pos / page_size;
        uint32_t entry = abs_pos - page_off * page_size;
        uint32_t row = (uint32_t)indices[page_begin + page_off];
        uint32_t base = (row * page_size + entry) * stride_kv + kv_head * dim;

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
                partial += __shfl_xor_sync(0xffffffffu, partial, o);
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
            if (m_g > -INFINITY) {
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
                    l_g > 0.0f ? (m_g + log2f(l_g)) : -INFINITY;
            }
        }
    }
)";

// Apple Silicon SIMD width; the warp width on CUDA.
constexpr int PAGED_V2_SIMD_WIDTH = 32;

// Threadgroup-memory budget for `tg_acc`, matching the ~28 KB v1 ceiling.
constexpr int PAGED_V2_TG_BUDGET_BYTES = 28672;

// Largest number of query heads one CTA may own. Beyond this the per-thread
// accumulator dominates the register file even at small head dims.
constexpr int PAGED_V2_MAX_Q_HEADS = 8;

// Register budget for the accumulator: `QHeads * DimsPerThread` floats.
constexpr int PAGED_V2_MAX_ACC_FLOATS = 32;

const std::vector<std::string>& partial_input_names() {
    static const std::vector<std::string> names = {
        "q",
        "k_pool",
        "v_pool",
        "indices",
        "indptr",
        "last_page_len",
        "first_page_offset",
        "request_indices",
        "kv_tile_indices",
        "params",
        "scale"};
    return names;
}

const std::vector<std::string>& partial_output_names() {
    static const std::vector<std::string> names = {"out_v", "out_lse"};
    return names;
}

// Thread-safe lazy-initialised holders for the two JIT-compiled bodies, using
// the `std::call_once` pattern of `paged_attention.cpp` / `sparse_v_sdpa.cpp`:
// the server reaches first use concurrently from per-request blocking workers,
// and `call_once` re-runs the initializer if MLX device lookup throws.
struct PagedV2PartialHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;
    bool cuda;

    explicit PagedV2PartialHolder(bool use_cuda) : cuda(use_cuda) {}

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = cuda
                ? mlx::core::fast::cuda_kernel(
                      "mlxcel_paged_attention_v2_partial",
                      partial_input_names(),
                      partial_output_names(),
                      std::string(PAGED_ATTENTION_V2_PARTIAL_CUDA_SOURCE))
                : mlx::core::fast::metal_kernel(
                      "mlxcel_paged_attention_v2_partial",
                      partial_input_names(),
                      partial_output_names(),
                      std::string(PAGED_ATTENTION_V2_PARTIAL_SOURCE));
        });
        return *kernel;
    }
};

inline PagedV2PartialHolder& get_partial_kernel(bool cuda) {
    static PagedV2PartialHolder metal_holder(false);
    static PagedV2PartialHolder cuda_holder(true);
    return cuda ? cuda_holder : metal_holder;
}

} // namespace

int paged_attention_v2_q_heads_per_cta(int dim, int n_rep) {
    if (n_rep < 1) {
        n_rep = 1;
    }
    int dims_per_thread = dim > 0
        ? (dim + PAGED_V2_SIMD_WIDTH - 1) / PAGED_V2_SIMD_WIDTH
        : 1;
    int reg_cap = PAGED_V2_MAX_ACC_FLOATS / dims_per_thread;
    if (reg_cap < 1) {
        reg_cap = 1;
    }
    int cap = reg_cap < PAGED_V2_MAX_Q_HEADS ? reg_cap : PAGED_V2_MAX_Q_HEADS;
    if (cap > n_rep) {
        cap = n_rep;
    }
    // Largest divisor of `n_rep` at or below the cap, so the query heads of one
    // KV head partition exactly and the kernel never launches a ragged group.
    for (int q = cap; q >= 1; q--) {
        if (n_rep % q == 0) {
            return q;
        }
    }
    return 1;
}

int paged_attention_v2_num_warps(int dim, int q_heads_per_cta) {
    if (dim <= 0 || q_heads_per_cta <= 0) {
        return 1;
    }
    int budget = PAGED_V2_TG_BUDGET_BYTES / (q_heads_per_cta * dim * 4);
    if (budget > 8) {
        budget = 8;
    }
    int warps = 1;
    while (warps * 2 <= budget) {
        warps *= 2;
    }
    return warps;
}

std::vector<mlx::core::array> paged_attention_decode_v2_partial(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& indices,
    const mlx::core::array& indptr,
    const mlx::core::array& last_page_len,
    const mlx::core::array& first_page_offset,
    const mlx::core::array& request_indices,
    const mlx::core::array& kv_tile_indices,
    const mlx::core::array& params,
    float scale) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    const auto& q_shape = q.shape();       // [B, Hq, 1, Dim]
    const auto& kp_shape = k_pool.shape(); // [num_blocks, PageSize, Hkv, Dim]

    int hq = q_shape[1];
    int dim = q_shape[3];
    int page_size = kp_shape[1];
    int hkv = kp_shape[2];
    int n_rep = hkv > 0 ? hq / hkv : 1;
    if (n_rep < 1) {
        n_rep = 1;
    }
    int num_chunks = static_cast<int>(request_indices.size());

    int q_heads = paged_attention_v2_q_heads_per_cta(dim, n_rep);
    int q_groups = n_rep / q_heads;
    int dims_per_thread = (dim + PAGED_V2_SIMD_WIDTH - 1) / PAGED_V2_SIMD_WIDTH;
    int num_warps = paged_attention_v2_num_warps(dim, q_heads);

    const bool use_cuda = !mlx::core::metal::is_available();
    auto& kernel = get_partial_kernel(use_cuda).get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"PageSize", page_size},
        {"NRep", n_rep},
        {"QHeads", q_heads},
        {"QGroups", q_groups},
        {"DimsPerThread", dims_per_thread},
        {"NumWarps", num_warps},
    };

    // Pack scale into a 1-element f32 array (metal_kernel inputs must be
    // arrays; ScalarArg is reserved for the precompiled-kernel path).
    auto scale_arr =
        mlx::core::full(mlx::core::Shape{1}, scale, mlx::core::float32);

    std::vector<mlx::core::array> inputs = {
        q,
        k_pool,
        v_pool,
        indices,
        indptr,
        last_page_len,
        first_page_offset,
        request_indices,
        kv_tile_indices,
        params,
        scale_arr,
    };
    std::vector<Shape> output_shapes = {
        Shape{num_chunks, hq, dim},
        Shape{num_chunks, hq},
    };
    std::vector<Dtype> output_dtypes = {mlx::core::float32, mlx::core::float32};

    return kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(
            PAGED_V2_SIMD_WIDTH, num_warps * hkv * q_groups, num_chunks), // grid
        std::make_tuple(PAGED_V2_SIMD_WIDTH, num_warps, 1), // threadgroup
        template_args,
        std::nullopt,
        false,
        {});
}

} // namespace mlxcel::turbo
