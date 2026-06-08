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

#pragma once

// Fused Metal paged-attention decode kernel launcher (epic #116 Phase 6, #123).
//
// This is strategy (B) from ADR 0001: a custom Metal kernel that reads
// scattered KV blocks directly out of the global pool via a per-sequence block
// table, with no separate gather copy. It replaces the gather-then-SDPA decode
// path (`paged_decode_attention_pooled_fallback`) on the >=16k or batched
// regime where the per-step gather cost is material. The gather path stays the
// correctness reference and the fallback.
//
// The kernel is a flash-decoding attention: one threadgroup per (batch * query
// head), threads split the visible tokens, each thread keeps a running online
// softmax (max, sum, weighted-V accumulator) over its token stripe, and a final
// threadgroup combine merges the partial softmaxes. GQA is handled by mapping a
// query head h to KV head h / n_rep.
//
// Pool layout (layout A, per ADR 0001):
//   k_pool, v_pool : [num_blocks, block_size, n_kv_heads, head_dim]  f16
//
// The block table is flattened: `rows` holds every sequence's physical pool
// rows concatenated in block-table order, `row_offsets[b]` is the start of
// sequence b's rows. A sequence's visible tokens are the absolute positions
// `[logical_starts[b], logical_starts[b] + visible_lens[b])` inside its
// concatenated blocks, so token i reads block `(logical_starts[b] + i) /
// block_size` (resolved through `rows`) at slot `(logical_starts[b] + i) %
// block_size`. This matches `PagedBlockPool::gather_visible`.

#include <mlx/array.h>

namespace mlxcel::turbo {

// Run the fused paged-attention decode kernel and return the attention output.
//
// Inputs:
// - `q`:              `[B, Hq, 1, head_dim]` f32.
// - `k_pool`:         `[num_blocks, block_size, Hkv, head_dim]` f16.
// - `v_pool`:         `[num_blocks, block_size, Hkv, head_dim]` f16.
// - `rows`:           `[total_rows]` i32 physical pool rows, per sequence
//                     concatenated in block-table order.
// - `row_offsets`:    `[B + 1]` i32 start of each sequence's rows in `rows`.
// - `logical_starts`: `[B]` i32 first visible absolute token index per sequence.
// - `visible_lens`:   `[B]` i32 visible token count per sequence.
// - `scale`:          attention scale applied to the QK dot product.
//
// Output:
// - `[B, Hq, 1, head_dim]` f32 attention output. The caller casts to the model
//   dtype to match the gather reference.
mlx::core::array paged_attention_decode(
    const mlx::core::array& q,
    const mlx::core::array& k_pool,
    const mlx::core::array& v_pool,
    const mlx::core::array& rows,
    const mlx::core::array& row_offsets,
    const mlx::core::array& logical_starts,
    const mlx::core::array& visible_lens,
    float scale);

} // namespace mlxcel::turbo
