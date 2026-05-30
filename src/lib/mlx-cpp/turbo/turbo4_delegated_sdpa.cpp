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

#include "turbo4_delegated_sdpa.h"

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

// Source for the Turbo4Delegated cold-V weighted-sum Metal kernel. The string
// is the kernel BODY only; `mlx::core::fast::metal_kernel` wraps it with the
// declaration, buffer arguments, and template-argument substitution.
//
// Algorithmic contract (matches `sparse_v_sdpa.cpp` for the cold path):
//   for each token t in [0, T_cold):
//     if all attn_weights[b, q, t] ≤ threshold: skip token (per-thread)
//     code  = codebook[unpack_nibble(packed[b, t, d])]
//     scaled = code * rescale[b, t]
//     for each q in 0..Tq:
//       out[b, q, d] += attn_weights[b, q, t] * scaled
//
// The kernel returns `out_cold_pre[b, q, d]` (unrotated). The host caller
// applies `signs1 · WHT(signs2 · out_cold_pre)` to produce the rotated cold
// contribution and adds the hot V matmul contribution before returning the
// final attention output.
//
// Why this is identical to `SPARSE_V_WEIGHTED_SUM_SOURCE`:
//
// The Turbo4Delegated cache stores cold V tokens with the same Turbo4 packing
// as Turbo4Asym (4-bit indices into a 16-centroid Lloyd-Max codebook plus a
// per-token fp16 rescale; see `cache::turbo::quant`). The fused weighted-sum
// kernel only operates on cold tokens; hot tokens are read directly from
// `hot_v` via a host-graph matmul. So the per-coordinate kernel body is the
// same — only the inputs are restricted to the cold range and the host-side
// orchestration is different.
//
// We deliberately keep this as a separate source string (rather than calling
// into the sparse-V launcher) for two reasons:
// 1. **Independent kernel name** — the JIT cache key is the function name. A
//    separate name avoids accidental cross-issue cache collisions when
//    `MLX_DEBUG_KERNELS=1` dumps kernel sources.
// 2. **Independent re-validation surface** — the CLAUDE.md MLX upgrade
//    checklist treats each kernel launcher as its own re-validation unit.
//    Two launchers, two test entries, no shared mutable state.
//
// If you change one kernel body, you must consider whether the other needs
// the matching change. As the bodies are bit-identical.
constexpr const char* TURBO4_DELEGATED_COLD_WEIGHTED_SUM_SOURCE = R"(
    auto d = thread_position_in_grid.x;
    auto n = thread_position_in_grid.z;

    // Decode `n` into (batch * Hq) and infer the matching Hkv slot. We pass
    // the layout sizes via the input shapes; `weights_shape` is
    // [B*Hq, Tq, T_cold], `rescale_shape` is [B*Hkv, T_cold]. From these we
    // recover B*Hq and B*Hkv directly without needing extra constants.
    auto bhq = n;                              // 0 .. B*Hq-1
    auto t_count = (uint)weights_shape[2];     // T_cold
    auto bhkv_count = (uint)rescale_shape[0];  // B*Hkv

    // Grouped attention: NRep = Hq / Hkv. NRep is given as a template
    // constant. Since bhq_count / NRep == bhkv_count holds for valid
    // inputs, the per-`n` Hkv slot is `n / NRep`.
    uint nrep_u = (uint)NRep;
    uint bhkv = bhq / nrep_u;
    if (bhkv >= bhkv_count) {
        bhkv = 0;  // defensive fallback for inconsistent shapes
    }

    auto wt = weights + bhq * (uint)RepeatCount * t_count;  // [Tq, T_cold]
    auto rs = rescale + bhkv * t_count;                      // [T_cold] f16
    auto pk = packed + bhkv * t_count * (uint)(Dim / 2);     // [T_cold, D/2]

    // Threshold passed as a 1-element array so MLX accepts it as a regular
    // input buffer. (TemplateArg only supports int/bool/Dtype; ScalarArg is
    // reserved for `precompiled_*_kernel` paths, not `metal_kernel`.)
    float thresh = threshold[0];

    // One accumulator per Q token. RepeatCount is a template constant so
    // the array size is known at compile time and lives in registers.
    float acc[RepeatCount];
    for (int r = 0; r < RepeatCount; r++) {
        acc[r] = 0.0f;
    }

    // Per-dim packed-byte coordinates: byte index = d/2; nibble = d%2.
    uint byte_idx = (uint)d >> 1u;
    uint nibble_shift = ((uint)d & 1u) * 4u;

    // Sweep over T_cold tokens. Each token: check aliveness, look up the
    // centroid for `(t, d)`, multiply by the precomputed per-token rescale,
    // and gate by per-Q attention weight.
    //
    // No threadgroup memory and no barriers in the inner loop — the same
    // discipline introduced for the Sparse-V kernel applies here.
    for (uint t = 0; t < t_count; t++) {
        bool any_alive = false;
        for (int r = 0; r < RepeatCount; r++) {
            if (wt[r * t_count + t] > thresh) {
                any_alive = true;
                break;
            }
        }
        if (!any_alive) {
            continue;
        }

        // Unpack nibble for this (t, d). Byte offset = byte_idx, nibble
        // selected by nibble_shift (0 for even d, 4 for odd d). Out-of-Dim
        // threads (d >= Dim) compute a benign 0 contribution; the `d < Dim`
        // bounds gate at the bottom prevents stray output writes.
        float code = 0.0f;
        if (d < (uint)Dim) {
            uint pb = (uint)pk[t * (uint)(Dim / 2) + byte_idx];
            uint idx = (pb >> nibble_shift) & 0x0Fu;
            code = codebook[idx];
        }

        // Per-token rescale `norm[t] / max(|y_hat[t]|, 1e-10)` was computed
        // on the host at quantize time. Each thread loads the
        // same fp16 scalar; Apple's L1 / per-threadgroup cache coalesces the
        // broadcast so the bandwidth cost is effectively one read per
        // threadgroup per token.
        float rescale_t = (float)rs[t];
        float scaled = code * rescale_t;

        // Accumulate into each Q slot's running sum, gated by its weight.
        for (int r = 0; r < RepeatCount; r++) {
            float w = wt[r * t_count + t];
            if (w > thresh) {
                acc[r] += w * scaled;
            }
        }
    }

    // Write D-aligned output. `out_shape` is [B*Hq, Tq, D].
    if (d < (uint)Dim) {
        for (int r = 0; r < RepeatCount; r++) {
            out[(bhq * (uint)RepeatCount + (uint)r) * (uint)Dim + d] = acc[r];
        }
    }
)";

// Thread-safe lazy-initialised holder for the JIT-compiled kernel. Same
// pattern as `SparseVKernelHolder` — the server runs each request on its own
// `tokio::task::spawn_blocking` worker, so concurrent first-use is reachable.
struct Turbo4DelegatedKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_turbo4_delegated_cold_weighted_sum",
                // Input names — must match the launcher's `inputs` vector
                // order in `turbo4_delegated_cold_weighted_sum`.
                {"weights", "rescale", "packed", "threshold", "codebook"},
                {"out"},
                std::string(TURBO4_DELEGATED_COLD_WEIGHTED_SUM_SOURCE));
        });
        return *kernel;
    }
};

inline Turbo4DelegatedKernelHolder& get_turbo4_delegated_kernel() {
    static Turbo4DelegatedKernelHolder holder;
    return holder;
}

// Bulk rotated dequant kernel for the Swift-LM-style dequant-first SDPA path.
// Each thread owns one output coordinate `(b, h, t, d)`, unpacks one 4-bit
// codebook index, applies the precomputed token rescale, and writes rotated V.
constexpr const char* TURBO4_BULK_DEQUANT_ROTATED_SOURCE = R"(
    auto d = thread_position_in_grid.x;
    auto n = thread_position_in_grid.z;

    auto h_count = (uint)packed_shape[1];
    auto t_count = (uint)packed_shape[2];
    auto packed_width = (uint)packed_shape[3];
    auto token_count = h_count * t_count;

    if (token_count == 0 || n >= (uint)packed_shape[0] * token_count) {
        return;
    }

    auto t = (uint)n % t_count;
    auto bh = (uint)n / t_count;
    auto byte_idx = (uint)d >> 1u;
    auto nibble_shift = ((uint)d & 1u) * 4u;

    float scaled = 0.0f;
    if (d < (uint)Dim) {
        uint packed_offset = (bh * t_count + t) * packed_width + byte_idx;
        uint pb = (uint)packed[packed_offset];
        uint idx = (pb >> nibble_shift) & 0x0Fu;
        scaled = codebook[idx] * (float)rescale[bh * t_count + t];
    }

    if (d < (uint)Dim) {
        out[(bh * t_count + t) * (uint)Dim + d] = scaled;
    }
)";

struct Turbo4BulkDequantRotatedKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_turbo4_bulk_dequant_rotated",
                {"packed", "rescale", "codebook"},
                {"out"},
                std::string(TURBO4_BULK_DEQUANT_ROTATED_SOURCE));
        });
        return *kernel;
    }
};

inline Turbo4BulkDequantRotatedKernelHolder& get_turbo4_bulk_dequant_rotated_kernel() {
    static Turbo4BulkDequantRotatedKernelHolder holder;
    return holder;
}

} // namespace (cold-only kernel internals)

namespace {  // anonymous namespace steel-envelope kernel internals

// =============================================================================
// Steel-attention-envelope fused SDPA kernel for Turbo4Delegated.
// =============================================================================
//
// The cold-V weighted-sum kernel above fixed the V-memory budget by
// reading packed 4-bit cold V directly inside the kernel. But the host-side
// composition in `attention_turbo4_delegated_fused` still issues 14+ MLX
// dispatches per decode step (Q·K, scale, mask add, softmax, slice cold,
// kernel, signs2 mul, WHT, signs1 mul, slice hot, hot repeat, hot matmul,
// add, cast to fp16). On M5 Max at 4K decode that pipeline runs at 18.90
// tok/s vs 101.76 tok/s for the FP16 baseline (0.186× FP16) — the FFI +
// per-dispatch overhead is the dominant cost.
//
// This kernel collapses the post-Q·K work into a single Metal dispatch:
//   softmax(scores) → cold-V dequant + accum → hot-V accum → per-Q normalise
//
// Q·K stays on the host because (a) MLX's matmul is steel-attention /
// NAX-accelerated and we do not want to reproduce a generic GEMM inside a
// JIT'd kernel; (b) the matmul is a single dispatch, so it does not
// contribute to the per-step dispatch count we are trying to reduce.
//
// Numerical stability: a 2-pass softmax over the full T_total range. Pass 1
// scans `scores[bhq, r, :]` for the per-(B*Hq, Tq) max and sum_exp; pass 2
// re-reads scores during cold + hot accumulation, computing `exp(score - max)`
// inline and dividing by the cached `sum_exp` at write time. Re-computing
// `exp` per accumulation step is cheap on Apple Silicon (fast::exp is single
// digit cycles), and avoids storing per-token `exp_score` in threadgroup
// memory (which would blow the 32KB budget at T_total ≥ 4096).
//
// Algorithmic contract (must stay numerically equivalent to the host
// composition this replaces — tracked by
// `delegated_fused_kernel_matches_reference_over_200_steps`):
//   for each token t in [0, T_cold):
//     // softmax-normalised attention weight for Q slot r
//     attn[r, t] = exp(scores[bhq, r, t] - max[r]) / sum_exp[r]
//     if attn[r, t] <= threshold for all r in [0, Tq): skip token (per-thread)
//     code   = codebook[unpack_nibble(packed[b, t, d])]
//     scaled = code * cold_rescale[b, t]
//     out_cold_pre[b, q, d] += attn[q, t] * scaled
//   for each token t in [0, T_hot):
//     attn[r, t] = exp(scores[bhq, r, T_cold + t] - max[r]) / sum_exp[r]
//     out_hot[b, q, d] += attn[q, t] * hot_v[b, t, d]
//
// Edge cases:
// - `T_cold == 0`: the cold loop is a no-op; `out_cold_pre` is all zeros.
//   The host's `signs1 · WHT(signs2 · 0)` is also zero, preserving the
//   "hot only" semantics. We still pass a 1-token zero placeholder for
//   `cold_packed` / `cold_rescale` because MLX `metal_kernel` rejects empty
//   inputs.
// - `T_hot == 0`: the hot loop is a no-op; `out_hot` is all zeros. This is
//   the immediately-post-fold case.
// - Both zero: the host guards against this by skipping the kernel call
//   entirely (no decode step ever sees zero context after `update`).
constexpr const char* TURBO4_DELEGATED_STEEL_SDPA_SOURCE = R"(
    auto d = thread_position_in_grid.x;
    auto n = thread_position_in_grid.z;

    // n indexes (batch * Hq). Recover the matching Hkv slot via the grouped
    // attention NRep template constant (Hq / Hkv).
    auto bhq = n;
    auto t_total = (uint)scores_shape[2];        // T_total
    auto t_cold = (uint)cold_offset[0];          // T_cold (passed as 1-elem buffer)
    auto t_hot = (uint)hot_offset[0];            // T_hot
    auto bhkv_count_cold = (uint)cold_rescale_shape[0]; // B*Hkv (cold side)
    auto bhkv_count_hot = (uint)hot_v_shape[0];         // B*Hkv (hot side)

    uint nrep_u = (uint)NRep;
    uint bhkv = bhq / nrep_u;
    if (bhkv >= bhkv_count_cold && bhkv_count_cold > 0u) {
        bhkv = 0;
    }
    if (bhkv >= bhkv_count_hot && bhkv_count_hot > 0u) {
        bhkv = 0;
    }

    // scores layout: [B*Hq, Tq, T_total] f32. Per-Q slice base.
    auto sc_base = scores + bhq * (uint)RepeatCount * t_total;
    // cold_rescale layout: [B*Hkv, T_cold] f16.
    auto rs = cold_rescale + bhkv * t_cold;
    // cold_packed layout: [B*Hkv, T_cold, D/2] u8.
    auto pk = cold_packed + bhkv * t_cold * (uint)(Dim / 2);
    // hot_v layout: [B*Hkv, T_hot, D] f16.
    auto hv = hot_v + bhkv * t_hot * (uint)Dim;

    float thresh = threshold[0];

    // -------------------------------------------------------------------------
    // Pass 1: per-Q max + sum_exp scan over the full T_total score range.
    //
    // History: ran this entire pass on thread 0 only and parked the
    // other D-1 threads at a single barrier, citing as precedent
    // for avoiding threadgroup tree-reduction barriers. That assumption held
    // on M1 Ultra at parity contexts but failed by 6×–25× on M5 Max because the M5 Max GPU has higher thread occupancy and each
    // idle thread now costs more in opportunity terms than the barrier
    // chain itself. inverts the choice for this kernel: split
    // both Pass 1a (max) and Pass 1b (sum_exp) across all D threads with a
    // tree reduction. With D=128 and T_total=16K each thread reads 128
    // positions instead of 16K — a 128× cut in per-thread serial work.
    //
    // Why's pattern still applies to `sparse_v_sdpa.cpp`: that
    // kernel does no pre-pass; it only does a single accumulation loop with
    // per-thread skip. There is no cross-thread dependency in that loop, so
    // there is no barrier to eliminate. Here we have an explicit reduction.
    //
    // Tree-reduction barriers cost ~10–50ns each on Apple Silicon. For
    // D=128 the depth is log2(128)=7 barriers per pass × 2 passes ×
    // RepeatCount ≈ 56 barriers per kernel launch for RepeatCount=4 — about
    // 1.7μs total, negligible against the multi-millisecond per-step cost.
    //
    // The numerical-stability guards are unchanged: still subtract max
    // before exp, still pin a degenerate `mx` of -inf to 0 and a degenerate
    // `sm` of 0 to 1, but applied after the reduction at the broadcast
    // point.
    // -------------------------------------------------------------------------
    threadgroup float tg_max[RepeatCount];
    threadgroup float tg_sum[RepeatCount];
    // `tg_alive_cutoff[r]` is the score-space equivalent of the sparse-V
    // threshold: `score > max + log(threshold * sum_exp)` iff the normalised
    // attention weight is `> threshold`. Pass 2 uses it to reject dead cold
    // tokens before paying the exp + dequant cost. When threshold is disabled
    // (`0.0`) the cutoff is `-inf` and Pass 2 falls back to the exact full
    // sweep.
    threadgroup float tg_alive_cutoff[RepeatCount];
    // Scratch buffer for the per-r tree reduction. Sized at the threadgroup
    // width (Dim, since the launcher uses tg_x = Dim for the production
    // power-of-2 head dims). Reused across Pass 1a and Pass 1b within each
    // r iteration. 4 bytes × 256 = 1KB worst case for D=256, well under the
    // 32KB threadgroup-memory budget.
    threadgroup float tg_red[Dim];

    for (int r = 0; r < RepeatCount; r++) {
        // -------- Pass 1a: parallel max over T_total. --------
        float local_max = -INFINITY;
        for (uint t = d; t < t_total; t += (uint)Dim) {
            float s = sc_base[r * t_total + t];
            if (s > local_max) {
                local_max = s;
            }
        }
        tg_red[d] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction across the D threads. Dim is a power of 2 for all
        // production models the launcher gates on, so this halving loop
        // terminates cleanly with stride=1 → 0.
        for (uint stride = (uint)Dim >> 1u; stride > 0u; stride >>= 1u) {
            if (d < stride) {
                float other = tg_red[d + stride];
                if (other > tg_red[d]) {
                    tg_red[d] = other;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // tg_red[0] now holds the per-r max. Apply the -inf guard at the
        // broadcast point so all threads see the same `mx` for Pass 1b.
        float mx = tg_red[0];
        if (!isfinite(mx)) {
            mx = 0.0f;
        }

        // -------- Pass 1b: parallel sum_exp over T_total. --------
        float local_sum = 0.0f;
        for (uint t = d; t < t_total; t += (uint)Dim) {
            float s = sc_base[r * t_total + t];
            local_sum += metal::fast::exp(s - mx);
        }
        // Reuse `tg_red` for the sum reduction. The previous max-reduction
        // result has already been read into the per-thread `mx` above, so
        // overwriting the buffer is safe after the next barrier.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_red[d] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = (uint)Dim >> 1u; stride > 0u; stride >>= 1u) {
            if (d < stride) {
                tg_red[d] = tg_red[d] + tg_red[d + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sm = tg_red[0];
        // Same degenerate-sum guard as the prior single-thread design: if
        // every score was -inf (so every exp(score - mx) underflowed to 0
        // or produced NaN), pin sum to 1 so per-token attn comes out 0
        // instead of NaN.
        if (!(sm > 0.0f) || !isfinite(sm)) {
            sm = 1.0f;
        }

        // Publish to the per-r broadcast slots. Only thread 0 needs to
        // write; the trailing barrier guarantees all threads see the
        // final values before Pass 2 reads them.
        if (d == 0u) {
            tg_max[r] = mx;
            tg_sum[r] = sm;
            float cutoff = -INFINITY;
            if (thresh > 0.0f && isfinite(thresh)) {
                float thresh_sum = thresh * sm;
                if (thresh_sum > 0.0f && isfinite(thresh_sum)) {
                    cutoff = mx + metal::fast::log(thresh_sum);
                }
            }
            tg_alive_cutoff[r] = cutoff;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // Pass 2: cold + hot weighted-sum into per-thread accumulators.
    //
    // Each thread owns output dim `d`. The accumulators `acc_cold[r]` and
    // `acc_hot[r]` collect `Σ_t attn[r, t] · V_*[t, d]` for the cold and hot
    // ranges respectively. The host applies the inverse Turbo4 rotation to
    // the cold output and adds the hot output before casting to FP16.
    // -------------------------------------------------------------------------
    float acc_cold[RepeatCount];
    float acc_hot[RepeatCount];
    for (int r = 0; r < RepeatCount; r++) {
        acc_cold[r] = 0.0f;
        acc_hot[r] = 0.0f;
    }
    float max_vals[RepeatCount];
    float sum_vals[RepeatCount];
    float cutoff_vals[RepeatCount];
    for (int r = 0; r < RepeatCount; r++) {
        max_vals[r] = tg_max[r];
        sum_vals[r] = tg_sum[r];
        cutoff_vals[r] = tg_alive_cutoff[r];
    }

    // Per-dim packed-byte coordinates for the cold V dequant.
    uint byte_idx = (uint)d >> 1u;
    uint nibble_shift = ((uint)d & 1u) * 4u;

    // Cold loop. Mirror the alive-skip discipline of the cold-only kernel
    // if the post-softmax weight for this token is below the
    // threshold for every Q slot, skip the dequant + rescale + accumulate
    // work entirely. We compare `exp_score > thresh * sum_exp` instead of
    // the equivalent `exp_score / sum_exp > thresh` to avoid a divide in the
    // hot path — `sum_exp` is loop-invariant per r. follow-up:
    // make that comparison in score-space first so fully-dead tokens also
    // avoid the exp calls.
    bool threshold_enabled = thresh > 0.0f && isfinite(thresh);
    if (threshold_enabled) {
        for (uint t = 0; t < t_cold; t++) {
            bool any_alive = false;
            // Cache the per-r score and exp(score - max) so we don't
            // recompute them inside the accumulation phase. RepeatCount is
            // a template constant, so these register arrays are sized at
            // compile time.
            float scores_local[RepeatCount];
            float exp_scores[RepeatCount];
            for (int r = 0; r < RepeatCount; r++) {
                float s = sc_base[r * t_total + t];
                scores_local[r] = s;
                if (s > cutoff_vals[r]) {
                    any_alive = true;
                }
            }
            if (!any_alive) {
                continue;
            }
            for (int r = 0; r < RepeatCount; r++) {
                exp_scores[r] = metal::fast::exp(scores_local[r] - max_vals[r]);
            }

            // Decode the per-(t, d) cold V centroid. Out-of-Dim threads
            // (d >= Dim) compute a benign 0 contribution; the bottom write
            // gate suppresses stray output.
            float code = 0.0f;
            if (d < (uint)Dim) {
                uint pb = (uint)pk[t * (uint)(Dim / 2) + byte_idx];
                uint idx = (pb >> nibble_shift) & 0x0Fu;
                code = codebook[idx];
            }
            float rescale_t = (float)rs[t];
            float scaled = code * rescale_t;

            for (int r = 0; r < RepeatCount; r++) {
                acc_cold[r] += exp_scores[r] * scaled;
            }
        }
    } else {
        for (uint t = 0; t < t_cold; t++) {
            bool any_alive = false;
            float exp_scores[RepeatCount];
            for (int r = 0; r < RepeatCount; r++) {
                float s = sc_base[r * t_total + t];
                float es = metal::fast::exp(s - max_vals[r]);
                exp_scores[r] = es;
                if (es > 0.0f) {
                    any_alive = true;
                }
            }
            if (!any_alive) {
                continue;
            }

            // Decode the per-(t, d) cold V centroid. Out-of-Dim threads
            // (d >= Dim) compute a benign 0 contribution; the bottom write
            // gate suppresses stray output.
            float code = 0.0f;
            if (d < (uint)Dim) {
                uint pb = (uint)pk[t * (uint)(Dim / 2) + byte_idx];
                uint idx = (pb >> nibble_shift) & 0x0Fu;
                code = codebook[idx];
            }
            float rescale_t = (float)rs[t];
            float scaled = code * rescale_t;

            for (int r = 0; r < RepeatCount; r++) {
                acc_cold[r] += exp_scores[r] * scaled;
            }
        }
    }

    // Hot loop. No threshold gating — hot tokens are recent and almost
    // always alive, so the per-thread skip would not pay back its branch
    // cost. The hot V is FP16 plain (no rotation) so we read it directly.
    for (uint t = 0; t < t_hot; t++) {
        // Hot scores live at offset `t_cold + t` in the score row.
        float exp_scores[RepeatCount];
        for (int r = 0; r < RepeatCount; r++) {
            float s = sc_base[r * t_total + t_cold + t];
            exp_scores[r] = metal::fast::exp(s - max_vals[r]);
        }
        float hv_d = 0.0f;
        if (d < (uint)Dim) {
            hv_d = (float)hv[t * (uint)Dim + d];
        }
        for (int r = 0; r < RepeatCount; r++) {
            acc_hot[r] += exp_scores[r] * hv_d;
        }
    }

    // -------------------------------------------------------------------------
    // Normalise by sum_exp and write to the two output buffers.
    //
    // Both outputs share the same per-Q sum_exp denominator. The host then
    // computes `final = signs1 · WHT(signs2 · out_cold_pre) + out_hot` and
    // casts to FP16. Bit-equivalence with `softmax(scores) @ V_full` requires
    // (linearity of WHT) that the rotation distributes across the sum — which
    // it does, because hot V is unrotated and the rotation is only the
    // inverse of the cold V quantize-time rotation.
    // -------------------------------------------------------------------------
    if (d < (uint)Dim) {
        for (int r = 0; r < RepeatCount; r++) {
            float inv_sum = 1.0f / sum_vals[r];
            uint out_idx = (bhq * (uint)RepeatCount + (uint)r) * (uint)Dim + d;
            out_cold_pre[out_idx] = acc_cold[r] * inv_sum;
            out_hot[out_idx]      = acc_hot[r]  * inv_sum;
        }
    }
)";

// Thread-safe lazy-initialised holder for the steel-envelope kernel. Same
// `std::call_once` pattern as the cold-only kernel above so concurrent
// first-use from `tokio::task::spawn_blocking` workers is safe.
struct Turbo4DelegatedSteelKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_turbo4_delegated_steel_sdpa",
                // Input names — must match the launcher's `inputs` vector
                // order in `turbo4_delegated_steel_sdpa`.
                {"scores", "cold_packed", "cold_rescale", "hot_v",
                 "codebook", "threshold", "cold_offset", "hot_offset"},
                {"out_cold_pre", "out_hot"},
                std::string(TURBO4_DELEGATED_STEEL_SDPA_SOURCE));
        });
        return *kernel;
    }
};

inline Turbo4DelegatedSteelKernelHolder& get_turbo4_delegated_steel_kernel() {
    static Turbo4DelegatedSteelKernelHolder holder;
    return holder;
}

} // namespace

mlx::core::array turbo4_delegated_cold_weighted_sum(
    const mlx::core::array& attn_weights_cold,
    const mlx::core::array& v_packed_cold,
    const mlx::core::array& v_rescale_cold,
    const mlx::core::array& codebook,
    int dim,
    int n_rep,
    float threshold) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    auto& w_shape = attn_weights_cold.shape();

    int bhq = w_shape[0];
    int tq = w_shape[1];

    auto& kernel = get_turbo4_delegated_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"RepeatCount", tq},
        {"NRep", n_rep},
    };

    auto threshold_arr = mlx::core::full(
        mlx::core::Shape{1}, threshold, mlx::core::float32);

    // Input order MUST match the buffer-name vector in `metal_kernel(...)`:
    // {"weights", "rescale", "packed", "threshold", "codebook"}.
    std::vector<mlx::core::array> inputs = {
        attn_weights_cold,  // [B*Hq, Tq, T_cold]    f32
        v_rescale_cold,     // [B*Hkv, T_cold]       f16
        v_packed_cold,      // [B*Hkv, T_cold, D/2]  u8
        threshold_arr,      // [1]                   f32
        codebook,           // [16]                  f32
    };
    std::vector<Shape> output_shapes = {Shape{bhq, tq, dim}};
    std::vector<Dtype> output_dtypes = {mlx::core::float32};

    // Threadgroup: (Dim, 1, 1). Grid: (Dim, 1, B*Hq). Apple Silicon caps
    // threadgroup width at 1024 — for D > 1024 (rare) the caller must split
    // along D, but typical head_dim is 64/128/256 so we ignore that path.
    int tg_x = dim;
    if (tg_x > 1024) {
        tg_x = 1024;
    }
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(dim, 1, bhq),    // grid
        std::make_tuple(tg_x, 1, 1),     // threadgroup
        template_args,
        std::nullopt,
        false,
        {});

    return results[0];
}

mlx::core::array turbo4_delegated_bulk_dequant_rotated(
    const mlx::core::array& v_packed,
    const mlx::core::array& v_rescale,
    const mlx::core::array& codebook,
    int dim) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    auto& p_shape = v_packed.shape();
    int b = p_shape[0];
    int h = p_shape[1];
    int t = p_shape[2];
    int bht = b * h * t;

    auto& kernel = get_turbo4_bulk_dequant_rotated_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
    };

    std::vector<mlx::core::array> inputs = {
        v_packed,   // [B, H, T, D/2] u8
        v_rescale,  // [B, H, T, 1]   f16
        codebook,   // [16]           f32
    };
    std::vector<Shape> output_shapes = {Shape{b, h, t, dim}};
    std::vector<Dtype> output_dtypes = {mlx::core::float16};

    int tg_x = dim;
    if (tg_x > 1024) {
        tg_x = 1024;
    }
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(dim, 1, bht),
        std::make_tuple(tg_x, 1, 1),
        template_args,
        std::nullopt,
        false,
        {});

    return results[0];
}

// =============================================================================
// Steel-attention-envelope fused SDPA launcher.
// =============================================================================
//
// Wraps the JIT-compiled `mlxcel_turbo4_delegated_steel_sdpa` kernel. The
// caller is responsible for slicing the cache state down to the visible
// (cold + hot) range, computing the FP32 `Q·K * scale + mask` score matrix on
// the host (one MLX matmul that benefits from steel attention / NAX), and
// applying the linear inverse Turbo4 rotation to the cold output.
//
// `cold_packed` and `cold_rescale` may be passed as a single-token zero-shape
// placeholder when `cold_offset == 0`; MLX `metal_kernel` rejects buffers with
// any zero-shape axis, so the host-side launcher in
// `mlxcel-core/src/cache/turbo/sparse_v.rs` substitutes 1-token zero buffers
// in that case (the kernel then takes the `t_cold == 0` early-out via the
// loop bound). Same convention applies to `hot_v` when `hot_offset == 0`.
std::vector<mlx::core::array> turbo4_delegated_steel_sdpa(
    const mlx::core::array& scores,
    const mlx::core::array& cold_packed,
    const mlx::core::array& cold_rescale,
    const mlx::core::array& hot_v,
    const mlx::core::array& codebook,
    int dim,
    int n_rep,
    int cold_offset,
    int hot_offset,
    float threshold) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    auto& s_shape = scores.shape();
    int bhq = s_shape[0];
    int tq = s_shape[1];

    auto& kernel = get_turbo4_delegated_steel_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"RepeatCount", tq},
        {"NRep", n_rep},
    };

    // Pack scalars as 1-element arrays. metal_kernel accepts only `array`
    // inputs (TemplateArg covers int/bool/Dtype but not float; ScalarArg is
    // for the precompiled-kernel path, not metal_kernel).
    auto threshold_arr = mlx::core::full(
        mlx::core::Shape{1}, threshold, mlx::core::float32);
    auto cold_offset_arr = mlx::core::full(
        mlx::core::Shape{1}, cold_offset, mlx::core::int32);
    auto hot_offset_arr = mlx::core::full(
        mlx::core::Shape{1}, hot_offset, mlx::core::int32);

    // Input order MUST match the buffer-name vector in `metal_kernel(...)`:
    // {"scores", "cold_packed", "cold_rescale", "hot_v", "codebook",
    //  "threshold", "cold_offset", "hot_offset"}.
    std::vector<mlx::core::array> inputs = {
        scores,           // [B*Hq, Tq, T_total]   f32
        cold_packed,      // [B*Hkv, T_cold, D/2]  u8 (or 1-token placeholder)
        cold_rescale,     // [B*Hkv, T_cold]       f16
        hot_v,            // [B*Hkv, T_hot, D]     f16
        codebook,         // [16]                  f32
        threshold_arr,    // [1]                   f32
        cold_offset_arr,  // [1]                   i32
        hot_offset_arr,   // [1]                   i32
    };
    std::vector<Shape> output_shapes = {
        Shape{bhq, tq, dim},  // out_cold_pre
        Shape{bhq, tq, dim},  // out_hot
    };
    std::vector<Dtype> output_dtypes = {
        mlx::core::float32,  // out_cold_pre
        mlx::core::float32,  // out_hot
    };

    // Threadgroup: (Dim, 1, 1). Grid: (Dim, 1, B*Hq). Same as the cold-only
    // launcher — D threads cooperate via threadgroup memory for the
    // per-(B*Hq) softmax max+sum scan, then accumulate a per-thread output
    // dim into registers.
    int tg_x = dim;
    if (tg_x > 1024) {
        tg_x = 1024;
    }
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(dim, 1, bhq),   // grid
        std::make_tuple(tg_x, 1, 1),    // threadgroup
        template_args,
        std::nullopt,
        false,
        {});

    return results;
}

} // namespace mlxcel::turbo
