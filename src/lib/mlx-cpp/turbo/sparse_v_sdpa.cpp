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

#include "sparse_v_sdpa.h"

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

// Source for the fused Sparse-V weighted-sum Metal kernel. The string is the
// kernel BODY only; `mlx::core::fast::metal_kernel` wraps it with the
// declaration, buffer arguments, and template-argument substitution.
//
// See `sparse_v_sdpa.metal` for an annotated reference copy.
//
// IMPORTANT: per-thread skipping is the WHOLE point of this kernel. The
// `if (any_alive == 0) continue;` block is what produces the speedup at long
// context — every skipped token saves the codebook gather, rescale fetch,
// and (D × RepeatCount) accumulator updates that the graph-level
// `where(alive, V_dq, 0)` path still pays.
//
// precomputed rescale.
//
// The previous implementation derived `|y_hat[t]| = sqrt(Σ_d codebook[idx]²)`
// per token via a `log2(Dim) + 2`-barrier threadgroup tree reduction over a
// `tg_y_hat[Dim]` shared scratch buffer. On M5 Max at 4 K decode for
// `turbo4-asym` this reduction dominated decode latency: ~9 barriers per
// cache token × 4 K tokens × 32 layers × ~32 threadgroups produced tens of
// millions of `threadgroup_barrier` calls per decode step, making the kernel
// 2.0× slower than the graph fallback (A/B).
//
// Because `|y_hat|` is a pure function of the packed indices (themselves
// fixed at quantize time), the entire rescale `rescale[t] = norm[t] / |y_hat[t]|`
// is now precomputed on the host inside `quantize_into_packed` and stored
// alongside the packed buffer. The kernel reads the scalar once per live
// token via a single fp16 load — no threadgroup memory, no barriers.
//
// Layout contract:
//   weights_shape = [B*Hq, Tq, Tk]   f32
//   rescale_shape = [B*Hkv, Tk]      f16  (precomputed `norm/|y_hat|`)
//   packed_shape  = [B*Hkv, Tk, D/2] u8   (low/high nibble = even/odd dim)
//   threshold     = [1]              f32  (alive cutoff scalar passed as array)
//   codebook      = [16]             f32  (Lloyd-Max centroids, 4-bit)
//   out_shape     = [B*Hq, Tq, D]    f32  (unrotated weighted sum)
//
// The caller applies the inverse Turbo4 rotation (`signs1 * WHT * signs2 *`)
// to the [B, Hq, Tq, D] output, so the per-cache-token rotation cost is
// eliminated.
//
// Threadgroup layout: (Dim, 1, 1). One threadgroup per (B*Hq) slot. The
// per-token `rescale` is small (one fp16 per token) and identical across
// all `Dim` threads in a threadgroup — Apple's L1 / per-threadgroup cache
// coalesces the broadcast load with no further effort.
constexpr const char* SPARSE_V_WEIGHTED_SUM_SOURCE = R"(
    auto d = thread_position_in_grid.x;
    auto n = thread_position_in_grid.z;

    // Decode `n` into (batch * Hq) and infer the matching Hkv slot. We pass
    // the layout sizes via the input shapes; `weights_shape` is
    // [B*Hq, Tq, Tk], `rescale_shape` is [B*Hkv, Tk]. From these we recover
    // B*Hq and B*Hkv directly without needing extra constants.
    auto bhq = n;                              // 0 .. B*Hq-1
    auto t_count = (uint)weights_shape[2];     // Tk
    auto bhkv_count = (uint)rescale_shape[0];  // B*Hkv

    // Grouped attention: NRep = Hq / Hkv. NRep is given as a template
    // constant. Since bhq_count / NRep == bhkv_count holds for valid
    // inputs, the per-`n` Hkv slot is `n / NRep`.
    uint nrep_u = (uint)NRep;
    uint bhkv = bhq / nrep_u;
    if (bhkv >= bhkv_count) {
        bhkv = 0;  // defensive fallback for inconsistent shapes
    }

    auto wt = weights + bhq * (uint)RepeatCount * t_count; // [Tq, Tk]
    auto rs = rescale + bhkv * t_count;                     // [Tk] f16
    auto pk = packed + bhkv * t_count * (uint)(Dim / 2);    // [Tk, D/2]

    // Thresholds passed as a 1-element array so MLX accepts them as a regular
    // input buffer. (TemplateArg only supports int/bool/Dtype; ScalarArg
    // is reserved for `precompiled_*_kernel` paths, not `metal_kernel`.)
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

    // Sweep over T tokens. Each token: check aliveness, look up the
    // centroid for `(t, d)`, multiply by the precomputed per-token rescale,
    // and gate by per-Q attention weight.
    //
    // No threadgroup memory and no barriers in the inner loop — this is the
    // entire point. The previous tree-reduction body lived
    // here; see git history for the earlier form.
    for (uint t = 0; t < t_count; t++) {
        // Aliveness check: scan RepeatCount Q slots and short-circuit if all
        // are below threshold for this (b, h, t). For RepeatCount=1 (decode
        // case) this is a single compare-and-skip — the dominant production
        // path on Apple Silicon at long context. For RepeatCount > 1 the
        // compiler unrolls the inner loop.
        bool any_alive = false;
        for (int r = 0; r < RepeatCount; r++) {
            if (wt[r * t_count + t] > thresh) {
                any_alive = true;
                break;
            }
        }
        if (!any_alive) {
            // Per-thread skip — the speed gate of this kernel. With's
            // precomputed rescale there is no in-flight threadgroup state to
            // synchronize against, so the skip is a clean continue without
            // any cross-thread divergence concerns.
            continue;
        }

        // Unpack nibble for this (t, d). Byte offset = byte_idx, nibble
        // selected by nibble_shift (0 for even d, 4 for odd d). Out-of-Dim
        // threads (d >= Dim) compute a benign 0 contribution — the `d < Dim`
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
        // Per-r aliveness check: even if any_alive=true above, only
        // contribute the weight-gated value. This keeps the result
        // deterministic regardless of the any_alive short-circuit.
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

// Thread-safe lazy-initialised holder for the JIT-compiled kernel.
//
// The server runs each request on its own `tokio::task::spawn_blocking`
// worker, so concurrent first-use of the kernel is reachable. The earlier
// implementation mutated `kernel` and `initialized` without synchronisation,
// which is C++ data-race UB. We use `std::call_once` here because:
// - the initializer can throw (MLX device lookup failure on a misconfigured
//   build), and `call_once` re-runs the initializer on the next call after a
//   throw, matching the documented MLX behaviour;
// - the post-init read of `kernel` is publication-safe since `call_once` is a
//   release/acquire fence.
struct SparseVKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_sparse_v_weighted_sum",
                // Input names — must match the launcher's `inputs` vector
                // order in `sparse_v_weighted_sum`. `rescale` is
                // the precomputed `norm[t] / |y_hat[t]|` fp16 sidecar that
                // replaced the in-kernel `norms` + threadgroup tree
                // reduction.
                {"weights", "rescale", "packed", "threshold", "codebook"},
                {"out"},
                std::string(SPARSE_V_WEIGHTED_SUM_SOURCE));
        });
        return *kernel;
    }
};

inline SparseVKernelHolder& get_sparse_v_kernel() {
    static SparseVKernelHolder holder;
    return holder;
}

} // namespace

mlx::core::array sparse_v_weighted_sum(
    const mlx::core::array& attn_weights,
    const mlx::core::array& v_packed,
    const mlx::core::array& v_rescale,
    const mlx::core::array& codebook,
    int dim,
    int n_rep,
    float threshold) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    auto& w_shape = attn_weights.shape();

    int bhq = w_shape[0];
    int tq = w_shape[1];

    auto& kernel = get_sparse_v_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
        {"RepeatCount", tq},
        {"NRep", n_rep},
    };

    // Pack threshold into a 1-element f32 array. metal_kernel only takes
    // arrays for inputs (ScalarArg is reserved for the precompiled path).
    // Use a 1-D shape so the kernel side gets `constant float* threshold`
    // (size < 8 → constant address space) and `threshold[0]` works.
    auto threshold_arr = mlx::core::full(
        mlx::core::Shape{1}, threshold, mlx::core::float32);

    // Input order MUST match the buffer-name vector in `metal_kernel(...)`
    // below: {"weights", "rescale", "packed", "threshold", "codebook"}.
    std::vector<mlx::core::array> inputs = {
        attn_weights,   // [B*Hq, Tq, Tk]    f32
        v_rescale,      // [B*Hkv, Tk]       f16  (precompute)
        v_packed,       // [B*Hkv, Tk, D/2]  u8
        threshold_arr,  // [1]               f32
        codebook,       // [16]              f32
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

} // namespace mlxcel::turbo
