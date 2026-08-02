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

// Sorting-free top-k / top-p / min-p sampling by dual-pivot rejection
// (issue #901). Companion to `sampling.h`, which covers the no-filter
// stochastic path with the Gumbel-max kernel (issue #900); this file covers the
// path where at least one of top-k, top-p, or min-p narrows the support.
//
// ## What it replaces
//
// The stock filtered sampler runs `argpartition` over the whole vocabulary for
// top-k, an `argsort` + `cumsum` + two `take_along_axis` chain for top-p, a
// compiled softmax/max/mask for min-p, and finally `random::categorical`. Every
// stage materialises a full `[B, V]` intermediate, and `argsort` alone is
// `O(V log V)` over 32K-152K entries per token.
//
// ## The algorithm
//
// All three filters are threshold filters on the probability value:
//
//   - top-k keeps `p_i >= tau_k`, where `tau_k` is the k-th largest
//     probability. `count(p > v) < k` holds exactly when `v >= tau_k`, which is
//     also what the stock `x >= kth_logit` mask does, ties included.
//   - top-p keeps the descending prefix whose exclusive cumulative mass is at
//     most `top_p`, i.e. `mass(p > v) <= top_p * total`.
//   - min-p keeps `p_i >= min_p * p_max`. That test is invariant to
//     renormalisation, so it does not care where in the chain it is applied.
//
// So the filtered support is `{ i : p_i > low }` for one scalar `low` per row,
// and the whole job is to find `low` without sorting. The kernel does that by
// rejection sampling on a shrinking interval, one threadgroup per row:
//
//   1. Draw a candidate from the current proposal set `{p > low}`, weighted by
//      probability, with a fixed-order block scan (no sort).
//   2. Set `pivot_0 = p[candidate]` and `pivot_1 = midpoint(pivot_0, high)`.
//   3. One vocabulary sweep reduces `(count, mass)` above each pivot.
//   4. Accept the candidate when it passes every active filter test; the
//      accepted draw is then distributed exactly as the truncated,
//      renormalised distribution (standard rejection sampling: the proposal
//      always contains the target support).
//   5. Otherwise raise `low` to whichever pivot still leaves the boundary
//      above it, lower `high` when the bisection pivot cleared, and repeat.
//
// `pivot_1` is the *bit-pattern* midpoint of `pivot_0` and `high`, not the
// arithmetic one. Two reasons. It is pure integer arithmetic, so a backend that
// JITs with fast-math and flush-to-zero cannot corrupt the shrink step for
// subnormal probabilities (the hardening the issue asks for). And because the
// IEEE-754 bit pattern of a positive float is monotone in its value, halving the
// bit distance isolates a single float in at most 32 rounds regardless of
// magnitude, where arithmetic bisection of `[0, 1]` stalls long before reaching
// a probability of 1e-6.
//
// ## Convergence, and why 32 rounds is a proof rather than a hope
//
// Let `d` be the distance between the bit patterns of `low` and `high`. Every
// round sets `pivot_1` to the bit midpoint of `[pivot_0, high]` with
// `low < pivot_0 <= high`, and then either raises `low` to `pivot_1` (leaving
// `high - pivot_1`, at most half of `d`) or lowers `high` to `pivot_1` while
// raising `low` to `pivot_0` (leaving `pivot_1 - pivot_0`, again at most half).
// So `d` at least halves every round.
//
// `low` starts at 0 (or just under the min-p threshold) and `high` starts at the
// row maximum, which is a probability and therefore at most 1.0, so `d` starts
// below `0x3F800000 < 2^30`. Thirty rounds bring `d` to 1, meaning `low` and
// `high` are adjacent floats. At that point the proposal `{p > low}` is exactly
// `{p >= high}`, and the loop invariant says `high` passes every filter test,
// so by monotonicity every element of the proposal passes: the next draw is
// accepted unconditionally. Convergence therefore takes at most 31 rounds from
// any starting bracket, and the data-driven `pivot_0` normally gets there in
// one or two.
//
// The cap is `REJECTION_MAX_ROUNDS = 32`, so under this arithmetic a row cannot
// exhaust it. The `ok` output and the caller's `argpartition` fallback are kept
// anyway, as the guard that catches a future change to the pivot arithmetic
// (an arithmetic midpoint, for instance, does not halve the bit distance and
// stalls outright on small probabilities). A test drives the fallback by
// lowering the cap explicitly.
//
// ## Determinism
//
// One threadgroup per row, a compile-time thread count, a fixed-order block
// scan, and a halving-tree reduction: nothing about the launch geometry varies
// with batch size, so the reduction order is fixed and the sampled id is a pure
// function of `(key, probs, params)`. The uniform for round `r` of row `b` is
// `philox(key, {r, 0, b, 0})[0]`, so rounds and rows never share a draw.
//
// Used by: `cpp/mlx_cxx_bridge.cpp` (`fused_sample`, `fused_sample_rejection`).

#include <mlx/array.h>

namespace mlxcel::turbo {

// Vocabulary sweeps the rejection loop is allowed before it gives up on a row
// and the host falls back to the `argpartition` chain. The interval halves in
// bit space every round, so 32 rounds isolates a single float from any starting
// bracket; overflow means a genuinely pathological row rather than slow
// convergence.
inline constexpr int REJECTION_MAX_ROUNDS = 32;

// True when the rejection kernel can run on the active backend: a GPU default
// device plus an available Metal or CUDA backend. A CPU-only build, or a
// process under `MLXCEL_DEVICE=cpu`, returns false and the caller keeps the
// `argpartition` chain.
bool rejection_sample_supported();

// True when `probs` has a shape the kernel accepts: 2-D `[B, V]` with `B > 0`
// and `V > 0`. The kernel reads f32 probabilities, which the caller produces
// with a single softmax, so there is no dtype variant to accept here.
bool rejection_sample_accepts(const mlx::core::array& probs);

// Threads per threadgroup the kernel launches with. Exposed so a test can
// reason about the fixed launch geometry the determinism argument rests on.
int rejection_threadgroup_size();

// Result of one batch-wide rejection sampling launch.
struct RejectionSampleResult {
    // `[B]` uint32 token ids. A row that failed to converge carries that row's
    // argmax, which is always inside the filtered support, so the value is
    // usable even before the caller applies its fallback.
    mlx::core::array ids;
    // `[B]` uint32, 1 when the row's rejection loop accepted within the round
    // cap and 0 when it ran out of rounds.
    mlx::core::array ok;
    // `[B]` uint32 rounds the row consumed, for the microbenchmark and for the
    // adversarial test that has to show the cap was actually reached.
    mlx::core::array rounds;
};

// Draw one token id per row from the top-k / top-p / min-p truncated,
// renormalised distribution.
//
// - `probs`: `[B, V]` float32 probabilities, one softmax row per batch entry.
//   Rows need not sum to exactly 1; the kernel renormalises by the mass it
//   measures, so a row that sums to 0.9999 samples the same distribution.
// - `params`: `[B, 3]` float32, `{top_k, top_p, min_p}` per row. `top_k < 1` or
//   `top_k >= V` disables top-k, `top_p` outside `(0, 1)` disables top-p,
//   `min_p` outside `(0, 1)` disables min-p. Rows in one launch may carry
//   different values; a single launch covers the whole batch either way.
// - `max_rounds`: rejection rounds before the row is declared unconverged.
//   Production passes `REJECTION_MAX_ROUNDS`; a test lowers it to force the
//   cap-overflow path.
//
// The launch requests a row-contiguous `probs`, so a strided input costs one
// `[B, V]` copy first. Decode never pays it: the caller's softmax output is
// contiguous.
//
// One RNG key is drawn from MLX's default key sequence per call, so a call
// advances the shared random state exactly once, the same way one
// `random::categorical` call does.
RejectionSampleResult rejection_sample(
    const mlx::core::array& probs,
    const mlx::core::array& params,
    int max_rounds);

} // namespace mlxcel::turbo
