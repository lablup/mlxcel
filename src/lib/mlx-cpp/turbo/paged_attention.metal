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
//
// Reference Metal source for the fused paged-attention decode kernel (epic
// #116 Phase 6, #123).
//
// This file is a readable copy of the kernel BODY embedded as a C++ string in
// `paged_attention.cpp` (`PAGED_ATTENTION_DECODE_SOURCE`). Keeping it as a
// standalone .metal file makes it possible to read and review the kernel
// without digging through a C++ raw-string literal, and to diff it against MLX
// upstream kernels during a commit-bump review.
//
// At runtime the kernel is JIT-compiled via `mlx::core::fast::metal_kernel` in
// `paged_attention.cpp`. The JIT path expects the body WITHOUT a `kernel`
// declaration: `metal_kernel` wraps the body with the kernel signature, the
// buffer arguments (in the order named in the launcher), the `*_shape` buffers
// for each input, and the template-argument substitution for `Dim`, `NRep`,
// and `Wthreads`. The dummy declaration below exists only so this file is
// syntactically a kernel; the body it contains is the authoritative copy.
//
// Algorithm: single-SIMD-group flash-decoding attention. One threadgroup of
// exactly 32 lanes (one SIMD group) handles one (batch * query head) slot. Each
// lane owns a `DimsPerThread = ceil(Dim / 32)`-wide contiguous slice of the head
// dimension, so the 32 lanes partition the whole head. The lane sweeps ALL
// visible tokens, keeping a running online softmax (max `m`, denominator `l`,
// weighted-V accumulator `acc[DimsPerThread]`), reading scattered pool rows
// through the block table. The per-token QK dot product is reduced across the 32
// lanes with `simd_sum` (a barrier-free SIMD shuffle), so there is no
// threadgroup memory, no barriers, and no register-spilling `acc[Dim]`. GQA maps
// query head `h` to KV head `h / NRep`.
//
// Why this layout: a thread-per-token stripe needs an `acc[Dim]` register array
// (spills at D=128) plus an O(Wthreads*Dim) threadgroup combine; thread-per-dim
// across multiple SIMD groups needs a per-token cross-group barrier. Folding the
// whole head into one SIMD group makes the QK reduction a single barrier-free
// `simd_sum` and keeps `acc[]` tiny (2 for D=64, 4 for D=128, 8 for D=256).
//
// Template constants (substituted per launch):
//   Dim           - head dimension D.
//   NRep          - Hq / Hkv (grouped-query replication factor).
//   DimsPerThread - ceil(Dim / 32); so `acc[]` lives in registers.
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

kernel void mlxcel_paged_attention_decode_reference(
    /* dummy declaration kept here so the file is syntactically a kernel even
       though the runtime JIT path strips the kernel/buffer wrapper and injects
       the `*_shape` buffers and the Dim/NRep/DimsPerThread template constants */ )
{
    // The authoritative kernel body (mirrors PAGED_ATTENTION_DECODE_SOURCE in
    // paged_attention.cpp):
    //
    //   uint lane = thread_position_in_threadgroup.x;  // 0..31
    //   uint bhq = threadgroup_position_in_grid.z;
    //   ... decode (b, h), map kv_head = h / NRep ...
    //   d0 = lane * DimsPerThread; stage q[bhq] dims [d0, d0+dpt) in q_reg[]
    //   if visible_lens[b] == 0: write zeros for this lane's dims; return
    //   for each visible token t:
    //     row = rows[row_offsets[b] + abs/block_size], slot = abs % block_size,
    //       abs = logical_starts[b] + t
    //     partial = sum over this lane's dims of q_reg * k_pool
    //     score = simd_sum(partial) * scale          // full q . k_t, no barrier
    //     online-softmax update of m, l, acc[] (identical scalars on every lane)
    //   out[bhq*Dim + d] = acc[j] / l   for this lane's dims
    //
    // See paged_attention.cpp for the exact body string.
}
