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

// In-place KV cache row write (issue #1959).
//
// `slice_update(cache, rows, start)` is MLX's functional update: its GPU eval
// first copies the whole input into the output and then writes the rows,
// skipping the copy only when the input buffer is donatable (exactly one
// reference to its data). During pipelined decode it never is: MLX's Metal
// backend keeps every kernel's input buffers alive in the command buffer's
// completion handler, and the next step is encoded while the previous one is
// still in flight, so the previous step's attention still references the cache
// buffer. Every decode step therefore copied every layer's K and V cache in
// full, a cost that grows with context (about 268 MB read and written per
// token at 2048 tokens on command-r7b).
//
// `inplace_slice_write` returns an array that shares `dst`'s buffer and has
// `rows` written at `start`, with no copy of the rest. The GPU work is the row
// write alone. This breaks MLX's value semantics for `dst`: after evaluation,
// `dst`'s buffer holds the new rows too. It is sound only when no reader of
// `dst` depends on the written region keeping its old contents. The decode
// cache satisfies that: the previous step reads rows `< start` and the write
// covers `[start, start + rows)`. The caller must also be the only owner of the
// buffer; a cache buffer shared with another owner (prompt-cache snapshot,
// cloned or detached cache) must keep using `slice_update`.
//
// GPU only. `dst` and `rows` must share a dtype and rank, and the written
// region must lie inside `dst`.
//
// Used by: `cpp/mlx_cxx_ext.cpp` (`inplace_slice_write`), reached from the
// FP16 `KVCache` decode write in `mlxcel-core/src/cache.rs`.

#include <mlx/array.h>

#include <vector>

namespace mlxcel::turbo {

mlx::core::array inplace_slice_write(
    const mlx::core::array& dst,
    const mlx::core::array& rows,
    const std::vector<int>& start);

} // namespace mlxcel::turbo
