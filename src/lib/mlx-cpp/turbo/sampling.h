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

// Softmax-free GPU categorical sampling via the Gumbel-max trick (issue #900).
//
// The stock fused sampler ends in `mlx::core::random::categorical`, which
// normalises over the whole vocabulary (32K-152K entries) per token per row.
// The Gumbel-max identity removes that pass entirely: adding i.i.d. Gumbel(0,1)
// noise to `logits / temperature` and taking the argmax draws exactly from
// `softmax(logits / temperature)`. No softmax, no sort, one index-carrying
// max-reduction over the row.
//
// The noise is generated in-kernel by a counter-based Philox-4x32-10 CBRNG, so
// the kernel is stateless: element `i` of row `r` always gets
// `philox(key, {i >> 2, 0, r, 0})[i & 3]`, a pure function of the launch key and
// the (row, element) pair. That makes the sampled index invariant to the launch
// shape (thread count, split count), which is what lets the launcher pick a
// split count from the batch size without perturbing the token stream.
//
// The key itself is drawn from MLX's own thread-local PRNG key sequence, the
// same stream `random::categorical` consumes, so `mlx::core::random::seed(...)`
// (mlxcel's `ffi::random_seed`) still reproduces a token stream exactly. The
// values differ from the pre-#900 `categorical` streams at equal seeds: this is
// a different RNG consumer, not a different distribution.
//
// Used by: `cpp/mlx_cxx_bridge.cpp` (`fused_sample`, `gumbel_max_sample`).

#include <mlx/array.h>

namespace mlxcel::turbo {

// True when the Gumbel-max kernel can run on the active backend: a GPU default
// device plus an available Metal or CUDA backend. A CPU-only build, or a
// process running with `MLXCEL_DEVICE=cpu`, returns false and the caller must
// keep the `categorical` graph path.
bool gumbel_max_sample_supported();

// True when `logits` has a shape and dtype the kernel accepts: 2-D `[B, V]`
// with `B > 0`, `V > 0`, and a float32 / float16 / bfloat16 element type.
bool gumbel_max_sample_accepts(const mlx::core::array& logits);

// Number of threadgroups that cooperate on one row for a `[batch, vocab]`
// launch. Always a power of two in `[1, 64]`. Exposed for tests: the sampled
// index must not depend on this value.
int gumbel_num_splits(int batch, int vocab);

// Draw one token id per row from `softmax(logits / temperature)`.
//
// - `logits`: `[B, V]`, float32 / float16 / bfloat16. `-inf` entries (the mask
//   produced by token bias and the XTC pre-step) stay `-inf` after the
//   temperature divide and the finite Gumbel add, so they can never win the
//   argmax while any finite logit remains in the row.
// - `temperature`: must be `> 0`. Greedy (`temperature == 0`) stays on `argmax`
//   in the caller and never reaches here.
//
// Returns `[B]` uint32 token ids, matching the dtype `argmax` and
// `random::categorical` return, so downstream host readback is unchanged.
//
// The launch requests a row-contiguous input, so a strided `logits` costs one
// `[B, V]` copy before the kernel runs. Decode never pays it: `slice_last_logits`
// on the `[B, 1, V]` decode shape is already contiguous, as is any tensor that
// came out of a penalty or bias pre-step. The strided case is the batched
// first-token sample off a `[B, S, V]` prefill output with `S > 1` and `B > 1`,
// where a `[B, V]` copy is negligible against the prefill that produced it.
//
// One RNG key is drawn from MLX's default key sequence per call, so a call
// advances the shared random state exactly once, the same way one
// `random::categorical` call does.
mlx::core::array gumbel_max_sample(
    const mlx::core::array& logits,
    float temperature);

} // namespace mlxcel::turbo
