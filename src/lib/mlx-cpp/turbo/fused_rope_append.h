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

// Fused q/k RoPE + KV-append-layout kernel launcher (issue #905).
//
// The dense decode path between the QKV projection and the cache append runs
// as eight MLX graph nodes per layer per token:
//
//     q, k, v          = slice(qkv, ...)          x3   (strided views)
//     q, k, v          = transpose(reshape(...))  x3   (strided views)
//     q                = fast::rope(q, offset)         (one kernel)
//     k                = fast::rope(k, offset)         (one kernel)
//
// and each strided view forces a materializing copy at the next consumer that
// needs contiguity (the cache `slice_update`, and `fast::rope`'s own output).
// This launcher replaces all of it with one dispatch that reads the
// row-contiguous fused-QKV projection output directly, applies rotary embedding
// to q and k, and writes q, k and v out already in the layout their consumers
// want.
//
// # Why this emits the append payload instead of writing the cache in place
//
// The obvious shape for "fused RoPE + KV-append" is a kernel that writes k and
// v straight into the destination cache slab. MLX cannot express that. Outputs
// of `mlx::core::fast::metal_kernel` / `cuda_kernel` are always freshly
// allocated (`CustomKernel::eval_gpu` calls `allocator::malloc` per output and
// has no donation path), so taking the slab as an input and returning the
// updated slab as an output would copy the whole slab every step. The existing
// `slice_update` append does not: `copy_gpu` donates the input buffer when it
// is uniquely referenced, which the cache's slab always is, so the append is
// already an O(new tokens) write into a donated buffer.
//
// Fusing the write would therefore trade an O(1) donated update for an
// O(capacity) copy: at 4 K context, 8 KV heads and head_dim 128 that is 8 MiB
// per tensor per layer per token. So the kernel produces the append *payload*
// in the destination's own layout and leaves the O(1) donated `slice_update` to
// perform the actual store. `dest_layout` selects which destination:
//
//   0 - dense `KVCache` slab order `[B, Hkv, L, D]` (what `update_and_fetch`
//       splices into `[B, Hkv, capacity, D]`).
//   1 - paged block-pool row order `[B, L, Hkv, D]` (what the pool's
//       `write_prefill` consumes into `[num_blocks, block_size, Hkv, D]`).
//
// Layout 1 is implemented and covered by parity tests but is not wired into any
// caller yet: the batched paged decode path it belongs to is issue #899's, and
// restructuring it here would collide with that work.
//
// Used by: `cpp/mlx_cxx_ext.cpp` (`fused_rope_qk_append`), reached from
// `mlxcel_core::layers::FusedQKVLinear::forward_fused_rope_append` and the
// Llama3-family dense decode path behind `MLXCEL_FUSED_ROPE_APPEND`.

#include <mlx/array.h>

#include <vector>

namespace mlxcel::turbo {

// Run the fused RoPE + append-layout kernel.
//
// Inputs:
// - `qkv`: `[B, L, (Hq + 2 * Hkv) * D]` row-contiguous fused-QKV projection
//   output, q first, then k, then v along the trailing axis.
//
// Parameters:
// - `num_heads` / `num_kv_heads` / `head_dim`: Hq, Hkv, D.
// - `rope_dims`: rotated prefix of each head; `[rope_dims, D)` is copied
//   through untouched. Must be even and at most `head_dim`.
// - `rope_base` / `rope_scale` / `traditional`: the same rotary parameters
//   `mlx::core::fast::rope` takes, with the same meaning.
// - `positions_base`: absolute position of the first token in the window. Token
//   `t` uses position `positions_base + t`, which is what `KVCache::offset`
//   supplies and what `RingSlidingKVCache`'s absolute positions already are.
// - `dest_layout`: `0` for the dense slab order, `1` for paged pool row order.
//
// Outputs (in order):
// - `q_out`: `[B, Hq, L, D]`, always attention order.
// - `k_out`: `[B, Hkv, L, D]` (layout 0) or `[B, L, Hkv, D]` (layout 1).
// - `v_out`: same shape as `k_out`; copied through without rotation.
//
// Throws `std::invalid_argument` when the geometry does not match `qkv`'s
// trailing dimension, when `rope_dims` is odd or larger than `head_dim`, or
// when `dest_layout` is not 0 or 1.
std::vector<mlx::core::array> fused_rope_qk_append(
    const mlx::core::array& qkv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rope_dims,
    float rope_base,
    float rope_scale,
    bool traditional,
    int positions_base,
    int dest_layout);

// Whether the current backend has a fused RoPE + append kernel at all.
//
// False on a CPU-only build, where both `metal_kernel` and `cuda_kernel` throw.
bool fused_rope_qk_append_available();

} // namespace mlxcel::turbo
