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
#include <mlx/array.h>

namespace mlxcel::turbo {
bool turbo_packed_attention_available();

// Used by: the single-token Turbo decode routes in KVCache::attend_dense.
// q, k and v_hot are in their respective codec bases. For dense K, k_norm
// is unused. When all V rows are packed, v_hot is unused.
mlx::core::array turbo_packed_attention(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& k_norm, const mlx::core::array& v_packed,
    const mlx::core::array& v_rescale, const mlx::core::array& v_hot,
    const mlx::core::array& centroids, float scale, bool packed_k);
} // namespace mlxcel::turbo
