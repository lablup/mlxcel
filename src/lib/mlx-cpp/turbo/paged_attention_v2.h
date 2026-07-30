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

// Fused paged-attention decode kernel v2: CSR page table, cross-CTA split-KV,
// and a variable-length merge kernel (issue #898).
//
// ## Why a second kernel
//
// The v1 kernel (`paged_attention.h` / `paged_attention.cpp`) splits the KV
// range *inside one threadgroup*: grid `(32, NumSplits, B*Hq)` over threadgroup
// `(32, NumSplits, 1)`, where `NumSplits` is bounded by the threadgroup-memory
// budget. One CTA therefore serves one `(batch, query head)` pair no matter how
// long the context is, so a small batch with a long context leaves most of the
// GPU idle and adding context adds no parallelism.
//
// v2 moves the KV split *across* CTAs. Each request's page range is cut into
// chunks of `pages_per_chunk` pages; each `(chunk, kv head, q-head group)`
// triple is one CTA that computes an online-softmax partial over its chunk and
// writes `(normalized partial V, LSE)` into a workspace. A second, deliberately
// generic kernel merges the partials of each output row with the closed-form
// flash rescale. Parallelism is now `num_chunks * Hkv * q_groups`, which the
// host-side plan (`PagedDecodePlan`, Rust side) sizes to the device.
//
// ## Layouts
//
// The pool layout is unchanged from v1 (layout A of ADR 0001):
//
//   k_pool, v_pool : [num_blocks, page_size, n_kv_heads, head_dim]  f16
//
// The block table is a CSR view of the batch instead of v1's
// `rows`/`row_offsets`/`logical_starts`/`visible_lens` quadruple:
//
//   indices           [total_pages] i32  flat physical pool rows
//   indptr            [B + 1]       i32  prefix sums delimiting each request
//   last_page_len     [B]           i32  valid entries in the final page
//   first_page_offset [B]           i32  first visible entry in the first page
//
// `first_page_offset` is the mlxcel extension to the standard three-array CSR
// layout: a sequence trimmed by a sliding window starts mid-page
// (`PagedLayerState::logical_start`), which the canonical layout cannot express.
// Request `r`'s visible length is therefore
// `(pages - 1) * page_size + last_page_len[r] - first_page_offset[r]`, and its
// token `i` lives at page `indptr[r] + (first_page_offset[r] + i) / page_size`,
// entry `(first_page_offset[r] + i) % page_size`.
//
// ## Merge-kernel contract (reused unchanged by the cascade issue #903)
//
// `paged_attention_merge_states` is intentionally agnostic to paging. It takes
//
//   v_in     [N, H, D] f32   partial outputs, each already normalized by its
//                            own softmax denominator
//   lse_in   [N, H]    f32   matching log-sum-exp values, in **log2 units**
//   o_indptr [M + 1]   i32   variable-length grouping: output row `o` merges
//                            partial rows `[o_indptr[o], o_indptr[o + 1])`
//
// and returns `(v_out [M, H, D] f32, lse_out [M, H] f32)` using
//
//   m = max(lse_a, lse_b); w_a = exp2(lse_a - m); w_b = exp2(lse_b - m)
//   out = (w_a * v_a + w_b * v_b) / (w_a + w_b)
//   lse_out = log2(w_a + w_b) + m
//
// generalized to a variable-length group. The algebra is associative and
// commutative, so a shared-prefix (cascade) decomposition can feed the same
// kernel with a different `o_indptr` and nothing else. An empty group, and a
// partial whose `lse` is `-inf` (an empty chunk), both contribute nothing;
// an output row with no finite partial yields zeros and `lse_out = -inf`.
//
// **LSE units are log2, not natural log.** The partial kernel folds `log2(e)`
// into the attention scale so its online softmax runs on `exp2`, and it emits
// `lse = max_2 + log2(sum exp2(score_2 - max_2))`. A consumer that wants the
// natural-log LSE multiplies by `ln(2)`.
//
// ## Backends
//
// Metal and CUDA JIT bodies live in one translation unit and are selected at
// runtime, following the dual-source pattern of `paged_attention.cpp`. The CUDA
// bodies are structurally parallel ports of the Metal ones and are
// **unvalidated**: issue #898 was implemented on an Apple Silicon host with no
// CUDA hardware and no `nvcc`, so they have never been compiled or run.

#include <vector>

#include <mlx/array.h>

namespace mlxcel::turbo {

// Query heads one CTA processes together, for a head dimension and GQA group
// size `n_rep = Hq / Hkv`.
//
// One CTA reads each KV element once and reuses it across every query head it
// owns, so a larger value amortizes the KV read; the accumulator
// `acc[QHeads * DimsPerThread]` lives in registers, so too large a value
// spills. The returned value always divides `n_rep`, so the query heads of one
// KV head partition exactly into `n_rep / result` CTA groups with no ragged
// remainder. Exposed so the Rust plan derives its CTA count from the same
// source of truth as the launcher.
int paged_attention_v2_q_heads_per_cta(int dim, int n_rep);

// SIMD groups (warps) per CTA for a head dimension and CTA query-head count.
//
// Bounded by the `tg_acc[NumWarps * QHeads * Dim]` threadgroup-memory budget
// (the same ~28 KB ceiling v1 uses) and capped at 8; always a power of two.
// Warps stripe the chunk's tokens, so this is a second, intra-CTA token split
// on top of the cross-CTA chunk split.
int paged_attention_v2_num_warps(int dim, int q_heads_per_cta);

// Run the v2 partial kernel over one batch's CSR page table.
//
// Inputs:
// - `q`:                 `[B, Hq, 1, head_dim]` f32.
// - `k_pool` / `v_pool`: `[num_blocks, page_size, Hkv, head_dim]` f16.
// - `indices`:           `[total_pages]` i32 flat physical pool rows.
// - `indptr`:            `[B + 1]` i32 per-request page-range prefix sums.
// - `last_page_len`:     `[B]` i32 valid entries in each request's last page.
// - `first_page_offset`: `[B]` i32 first visible entry in the first page.
// - `request_indices`:   `[num_chunks]` i32 request id of each chunk.
// - `kv_tile_indices`:   `[num_chunks]` i32 chunk index within its request.
// - `params`:            `[1]` i32, `params[0] = pages_per_chunk`.
// - `scale`:             attention scale applied to the QK dot product.
//
// Returns `{partial_v [num_chunks, Hq, head_dim] f32,
//           lse [num_chunks, Hq] f32}`.
//
// When the plan emitted exactly one chunk per request (in request order), the
// partial output *is* the final answer: `partial_v` reshapes to
// `[B, Hq, 1, head_dim]` and no merge launch is needed. That is the "write O
// directly" case, decided on the host so no output element is ever left
// unwritten.
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
    float scale);

// Variable-length merge over attention states. See the merge-kernel contract
// above; `v_in` is `[N, H, D]` f32, `lse_in` is `[N, H]` f32 in log2 units, and
// `o_indptr` is `[M + 1]` i32. Returns `{v_out [M, H, D] f32,
// lse_out [M, H] f32}`.
std::vector<mlx::core::array> paged_attention_merge_states(
    const mlx::core::array& v_in,
    const mlx::core::array& lse_in,
    const mlx::core::array& o_indptr);

} // namespace mlxcel::turbo
