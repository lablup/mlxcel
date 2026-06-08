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

#include "paged_attention.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Body of the fused paged-attention decode Metal kernel. The string is the
// kernel BODY only; `mlx::core::fast::metal_kernel` wraps it with the
// declaration, buffer arguments, and template-argument substitution. See
// `paged_attention.metal` for an annotated standalone copy that the Metal
// compiler can lint.
//
// Split-K flash-decoding attention. One threadgroup handles one (batch * query
// head) slot (grid z). The threadgroup is `NumSplits` SIMD groups of 32 lanes
// (a 2D `(32, NumSplits)` layout). The two axes parallelise the two reductions
// attention needs:
//
//   - Across the 32 lanes of a SIMD group: each lane owns a `DimsPerThread =
//     ceil(Dim / 32)`-wide slice of the head dimension, so the per-token QK dot
//     product is a single barrier-free `simd_sum` and the weighted-V
//     accumulator `acc[DimsPerThread]` stays tiny (no register-spilling
//     `acc[Dim]`).
//   - Across the `NumSplits` SIMD groups: each SIMD group `sg` sweeps a strided
//     token stripe (t = sg, sg + NumSplits, ...) and keeps its own online
//     softmax partial. After the stripe, a small threadgroup combine merges the
//     `NumSplits` partials (the flash-attention rescale). This is the token
//     parallelism a single SIMD group lacks: without it a long context is one
//     serial token loop per slot, which dominates at >=4k.
//
// GQA maps query head `h` to KV head `h / NRep`.
//
// Template constants:
//   Dim           - head dimension D.
//   NRep          - Hq / Hkv (grouped-query replication factor).
//   DimsPerThread - ceil(Dim / 32); compile-time so `acc[]` lives in registers.
//   NumSplits     - SIMD groups per threadgroup (token-stripe count).
//
// Buffers (order matches the launcher's input vector):
//   q              [B, Hq, 1, Dim]                     f32
//   k_pool         [num_blocks, block_size, Hkv, Dim]  f16
//   v_pool         [num_blocks, block_size, Hkv, Dim]  f16
//   rows           [total_rows]                        i32  physical pool rows
//   row_offsets    [B + 1]                             i32  start of seq b's rows
//   logical_starts [B]                                 i32  first visible abs idx
//   visible_lens   [B]                                 i32  visible token count
//   scale          [1]                                 f32
//   out            [B, Hq, 1, Dim]                     f32
constexpr const char* PAGED_ATTENTION_DECODE_SOURCE = R"(
    uint lane = thread_position_in_threadgroup.x;    // 0 .. 31 (within SIMD grp)
    uint sg = thread_position_in_threadgroup.y;      // 0 .. NumSplits-1
    uint bhq = threadgroup_position_in_grid.z;       // 0 .. B*Hq-1

    uint hq_count = (uint)q_shape[1];                // Hq
    uint block_size = (uint)k_pool_shape[1];         // tokens per block
    uint hkv_count = (uint)k_pool_shape[2];          // Hkv
    uint dim = (uint)Dim;
    uint dpt = (uint)DimsPerThread;                  // dims this lane owns
    uint d0 = lane * dpt;                            // first dim of this lane

    uint b = bhq / hq_count;                          // batch index
    uint h = bhq % hq_count;                          // query head
    uint kv_head = h / (uint)NRep;                    // grouped-query KV head
    if (kv_head >= hkv_count) {
        kv_head = 0;                                  // defensive
    }

    int vlen_i = visible_lens[b];
    uint vlen = vlen_i > 0 ? (uint)vlen_i : 0u;
    uint logical_start = (uint)logical_starts[b];
    uint row_off = (uint)row_offsets[b];

    // Stage this lane's Q slice in registers.
    float q_reg[DimsPerThread];
    for (uint j = 0; j < dpt; j++) {
        uint d = d0 + j;
        q_reg[j] = (d < dim) ? q[bhq * dim + d] : 0.0f;
    }

    // Threadgroup scratch for the cross-SIMD-group flash combine.
    threadgroup float tg_m[NumSplits];
    threadgroup float tg_l[NumSplits];
    threadgroup float tg_acc[NumSplits * Dim];

    // Empty window: only SIMD group 0 emits zeros (uniform across the group).
    if (vlen == 0u) {
        if (sg == 0u) {
            for (uint j = 0; j < dpt; j++) {
                uint d = d0 + j;
                if (d < dim) {
                    out[bhq * dim + d] = 0.0f;
                }
            }
        }
        return;
    }

    float scale_v = scale[0];

    // This SIMD group's online softmax over its strided token stripe.
    float m = -INFINITY;
    float l = 0.0f;
    float acc[DimsPerThread];
    for (uint j = 0; j < dpt; j++) {
        acc[j] = 0.0f;
    }

    uint stride_kv = hkv_count * dim;                 // elements per (block,slot)
    for (uint t = sg; t < vlen; t += (uint)NumSplits) {
        uint abs_pos = logical_start + t;
        uint block_idx = abs_pos / block_size;
        uint slot = abs_pos - block_idx * block_size;
        uint row = (uint)rows[row_off + block_idx];
        uint base = (row * block_size + slot) * stride_kv + kv_head * dim;

        float partial = 0.0f;
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            float kd = (d < dim) ? (float)k_pool[base + d] : 0.0f;
            partial += q_reg[j] * kd;
        }
        float score = simd_sum(partial) * scale_v;    // full q . k_t, no barrier

        float m_new = fmax(m, score);
        float corr = fast::exp(m - m_new);
        float p = fast::exp(score - m_new);
        l = l * corr + p;
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            float vd = (d < dim) ? (float)v_pool[base + d] : 0.0f;
            acc[j] = acc[j] * corr + p * vd;
        }
        m = m_new;
    }

    // Publish this SIMD group's partial. Every lane stores its dim slice; lane 0
    // stores the scalar (max, denominator).
    for (uint j = 0; j < dpt; j++) {
        uint d = d0 + j;
        if (d < dim) {
            tg_acc[sg * dim + d] = acc[j];
        }
    }
    if (lane == 0u) {
        tg_m[sg] = m;
        tg_l[sg] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD group 0 merges the NumSplits partials (flash rescale) and writes out.
    if (sg == 0u) {
        float m_g = tg_m[0];
        for (uint s = 1; s < (uint)NumSplits; s++) {
            m_g = fmax(m_g, tg_m[s]);
        }
        float l_g = 0.0f;
        for (uint s = 0; s < (uint)NumSplits; s++) {
            l_g += tg_l[s] * fast::exp(tg_m[s] - m_g);
        }
        float inv_l = l_g > 0.0f ? (1.0f / l_g) : 0.0f;
        for (uint j = 0; j < dpt; j++) {
            uint d = d0 + j;
            if (d < dim) {
                float a = 0.0f;
                for (uint s = 0; s < (uint)NumSplits; s++) {
                    a += tg_acc[s * dim + d] * fast::exp(tg_m[s] - m_g);
                }
                out[bhq * dim + d] = a * inv_l;
            }
        }
    }
)";

// Apple Silicon SIMD width. Each SIMD group is 32 lanes that partition the head
// dimension; `NumSplits` SIMD groups split the token range.
constexpr int PAGED_ATTENTION_SIMD_WIDTH = 32;

// Thread-safe lazy-initialised holder for the JIT-compiled kernel. Mirrors the
// `std::call_once` pattern in `sparse_v_sdpa.cpp`: the server reaches first-use
// concurrently from per-request blocking workers, and `call_once` re-runs the
// initializer if MLX device lookup throws.
struct PagedAttentionKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_paged_attention_decode",
                {"q", "k_pool", "v_pool", "rows", "row_offsets", "logical_starts",
                 "visible_lens", "scale"},
                {"out"},
                std::string(PAGED_ATTENTION_DECODE_SOURCE));
        });
        return *kernel;
    }
};

inline PagedAttentionKernelHolder& get_paged_attention_kernel() {
    static PagedAttentionKernelHolder holder;
    return holder;
}

} // namespace

mlx::core::array paged_attention_decode(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& rows,
    const mlx::core::array& row_offsets,
    const mlx::core::array& logical_starts,
    const mlx::core::array& visible_lens,
    float scale) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    const auto& q_shape = q.shape();       // [B, Hq, 1, Dim]
    const auto& kp_shape = k_pool.shape(); // [num_blocks, block_size, Hkv, Dim]

    int batch = q_shape[0];
    int hq = q_shape[1];
    int dim = q_shape[3];
    int hkv = kp_shape[2];
    int n_rep = hkv > 0 ? hq / hkv : 1;
    if (n_rep < 1) {
        n_rep = 1;
    }

    auto& kernel = get_paged_attention_kernel().get();

    // Each of the 32 lanes owns a ceil(Dim/32)-wide slice of the head.
    int dims_per_thread = (dim + PAGED_ATTENTION_SIMD_WIDTH - 1) / PAGED_ATTENTION_SIMD_WIDTH;

    // Token-split count = SIMD groups per threadgroup. Bounded by the 1024
    // thread/threadgroup cap (32 * NumSplits <= 1024 => NumSplits <= 32) and by
    // the `tg_acc[NumSplits * Dim]` threadgroup-memory budget (kept under ~28 KB
    // of the 32 KB limit).
    int num_splits = 28672 / (dim * 4);
    if (num_splits > 32) {
        num_splits = 32;
    }
    if (num_splits < 1) {
        num_splits = 1;
    }

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"NRep", n_rep},
        {"DimsPerThread", dims_per_thread},
        {"NumSplits", num_splits},
    };

    // Pack scale into a 1-element f32 array (metal_kernel inputs must be
    // arrays; ScalarArg is reserved for the precompiled-kernel path).
    auto scale_arr =
        mlx::core::full(mlx::core::Shape{1}, scale, mlx::core::float32);

    std::vector<mlx::core::array> inputs = {
        q,              // [B, Hq, 1, Dim]                    f32
        k_pool,         // [num_blocks, block_size, Hkv, Dim] f16
        v_pool,         // [num_blocks, block_size, Hkv, Dim] f16
        rows,           // [total_rows]                       i32
        row_offsets,    // [B + 1]                            i32
        logical_starts, // [B]                                i32
        visible_lens,   // [B]                                i32
        scale_arr,      // [1]                                f32
    };
    std::vector<Shape> output_shapes = {Shape{batch, hq, 1, dim}};
    std::vector<Dtype> output_dtypes = {mlx::core::float32};

    // Grid: (32, NumSplits, B*Hq) with one (32, NumSplits, 1) threadgroup per
    // (batch, query head) slot: 32 lanes partition the head, NumSplits SIMD
    // groups split the token range.
    int bhq = batch * hq;
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(PAGED_ATTENTION_SIMD_WIDTH, num_splits, bhq), // grid
        std::make_tuple(PAGED_ATTENTION_SIMD_WIDTH, num_splits, 1),   // threadgroup
        template_args,
        std::nullopt,
        false,
        {});

    return results[0];
}

} // namespace mlxcel::turbo
