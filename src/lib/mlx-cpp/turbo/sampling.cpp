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

#include "sampling.h"

#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/random.h>

// Backend availability probes for the Metal-vs-CUDA kernel gate. Both headers
// are backend-agnostic: an Apple build links the CUDA no_cuda stub, a CUDA
// build links the Metal no_metal stub, so `is_available()` always resolves and
// returns false for the absent backend.
#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Threads per threadgroup. Each thread owns 4 consecutive vocabulary entries
// per sweep, so one threadgroup covers 1024 entries per iteration. A power of
// two is required: the index-carrying reduction below is a halving tree.
constexpr int GUMBEL_TG_SIZE = 256;

// Vocabulary entries one threadgroup covers per sweep.
constexpr int GUMBEL_CHUNK = GUMBEL_TG_SIZE * 4;

// Threadgroup target for the split heuristic. A `[1, 152064]` decode row is a
// single reduction, so without row splitting a batch-1 launch would occupy one
// GPU core while the rest idle. Splitting to this many cooperating threadgroups
// keeps a small batch wide; a large batch already fills the machine and takes
// `NumSplits == 1` (no second reduction pass at all).
constexpr int GUMBEL_TARGET_THREADGROUPS = 64;

// Body of the Gumbel-max sampling Metal kernel. The string is the kernel BODY
// only; `mlx::core::fast::metal_kernel` wraps it with the declaration and the
// buffer arguments.
//
// One threadgroup handles one (row, split) pair: `GUMBEL_TG_SIZE` threads sweep
// a strided sequence of contiguous `4 * TgSize`-entry chunks of the row,
// keeping a per-thread running `(value, index)` maximum, and a halving-tree
// reduction over threadgroup memory merges them. When `NumSplits > 1` the host
// finishes the row with a tiny `argmax` + `take_along_axis` over the
// `[B, NumSplits]` partials.
//
// The Gumbel noise is Philox-4x32-10 (Salmon et al., Random123): a stateless
// counter-based bijection, so element `i` of row `r` always draws
// `philox(key, {i >> 2, 0, r, 0})[i & 3]` regardless of `TgSize`, `NumSplits`,
// or which thread happens to visit it. The sampled index is therefore a pure
// function of `(key, logits, temperature)` and does not move when the launcher
// changes the split count for a different batch size.
//
// Tie-breaks always keep the lower index (both in the per-thread scan and in
// the reduction), so a fully `-inf`-masked row yields index 0 deterministically
// instead of whatever thread happened to write last.
//
// Template constants:
//   TgSize    - threads per threadgroup (power of two).
//   NumSplits - threadgroups cooperating on one row (power of two).
//
// Buffers (order matches the launcher's input vector):
//   logits  [B, V]           f32 / f16 / bf16
//   rng_key [2]              u32   Philox key drawn from MLX's key sequence
//   temp    [1]              f32   sampling temperature, > 0
//   vals    [B, NumSplits]   f32   per-split maximum of logit/T + gumbel
//   idxs    [B, NumSplits]   u32   argmax index that produced `vals`
constexpr const char* GUMBEL_MAX_SAMPLE_SOURCE = R"(
    uint t = thread_position_in_threadgroup.x;    // 0 .. TgSize-1
    uint split = threadgroup_position_in_grid.y;  // 0 .. NumSplits-1
    uint row = threadgroup_position_in_grid.z;    // 0 .. B-1

    uint vocab = (uint)logits_shape[1];
    float temp_v = temp[0];
    uint key0_base = rng_key[0];
    uint key1_base = rng_key[1];
    uint row_off = row * vocab;

    float best = -INFINITY;
    uint best_idx = 0u;

    uint chunk = (uint)TgSize * 4u;
    uint stride = chunk * (uint)NumSplits;
    for (uint base = split * chunk + t * 4u; base < vocab; base += stride) {
        // Philox-4x32-10 over counter {base/4, 0, row, 0}.
        uint c0 = base >> 2;
        uint c1 = 0u;
        uint c2 = row;
        uint c3 = 0u;
        uint k0 = key0_base;
        uint k1 = key1_base;
        for (uint r = 0u; r < 10u; r++) {
            uint hi0 = mulhi(0xD2511F53u, c0);
            uint lo0 = 0xD2511F53u * c0;
            uint hi1 = mulhi(0xCD9E8D57u, c2);
            uint lo1 = 0xCD9E8D57u * c2;
            uint n0 = hi1 ^ c1 ^ k0;
            uint n1 = lo1;
            uint n2 = hi0 ^ c3 ^ k1;
            uint n3 = lo0;
            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }

        // Select by ternary chain rather than indexing a thread-local array:
        // the trip count is 4 but the early exit can stop the compiler from
        // unrolling, and a dynamically indexed thread array spills out of
        // registers into backing memory on both back-ends.
        for (uint j = 0u; j < 4u; j++) {
            uint idx = base + j;
            if (idx >= vocab) {
                break;
            }
            uint word = (j == 0u) ? c0 : ((j == 1u) ? c1 : ((j == 2u) ? c2 : c3));
            // Uniform on the OPEN interval (0, 1): a 2^-23 grid offset by half
            // a step, so `u` is never 0 or 1 and both logs stay finite. 23 bits,
            // not 24: an integer below 2^23 has an f32 ulp of at most 0.5, so
            // `x + 0.5f` is exact for every value this can produce. At 24 bits
            // the top value 2^24-1 sits in a binade with ulp 1, `x + 0.5f` ties
            // up to 2^24, and `u` lands on exactly 1.0 -- which makes the noise
            // +inf and hands that element the argmax regardless of its logit.
            // That is a 2^-24 chance per element, which over a 152K vocabulary
            // is roughly one uniformly-random token every 110 decode steps.
            float u = ((float)(word >> 9) + 0.5f) * (1.0f / 8388608.0f);
            float g = -log(-log(u));
            float scaled = (float)logits[row_off + idx] / temp_v;
            float cand = scaled + g;
            if (cand > best || (cand == best && idx < best_idx)) {
                best = cand;
                best_idx = idx;
            }
        }
    }

    threadgroup float tg_val[TgSize];
    threadgroup uint tg_idx[TgSize];
    tg_val[t] = best;
    tg_idx[t] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = (uint)TgSize / 2u; s > 0u; s >>= 1) {
        if (t < s) {
            float other = tg_val[t + s];
            uint other_idx = tg_idx[t + s];
            if (other > tg_val[t] ||
                (other == tg_val[t] && other_idx < tg_idx[t])) {
                tg_val[t] = other;
                tg_idx[t] = other_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0u) {
        uint out_off = row * (uint)NumSplits + split;
        vals[out_off] = tg_val[0];
        idxs[out_off] = tg_idx[0];
    }
)";

// CUDA port of the Gumbel-max sampling kernel. Structurally identical to the
// Metal source above: same thread mapping, same Philox counter derivation, same
// halving-tree reduction, so both backends select the same index for the same
// key and logits.
//
// Grid mapping (MLX passes Metal-style total threads and ceil-divides by the
// threadgroup tuple, cf. backend/cuda/custom_kernel.cpp): grid
// `(TgSize, NumSplits, B)` over threadgroup `(TgSize, 1, 1)` yields blocks
// `(1, NumSplits, B)` with `blockDim = (TgSize, 1, 1)`. Every field is an exact
// multiple, so no padded threads are launched. The loop bound is not
// block-uniform, but no thread returns early: every thread reaches every
// `__syncthreads()` in the reduction.
//
// `mulhi` becomes `__umulhi` and `log` becomes `logf`; `(float)logits[...]`
// uses the implicit float conversion on `__half` / `__nv_bfloat16` after MLX
// type substitution.
//
// UNVALIDATED: written from the Metal source for parity, but no CUDA hardware
// was available when this landed, so it has never been compiled or run.
constexpr const char* GUMBEL_MAX_SAMPLE_CUDA_SOURCE = R"(
    uint32_t t = threadIdx.x;                     // 0 .. TgSize-1
    uint32_t split = blockIdx.y;                  // 0 .. NumSplits-1
    uint32_t row = blockIdx.z;                    // 0 .. B-1

    uint32_t vocab = (uint32_t)logits_shape[1];
    float temp_v = temp[0];
    uint32_t key0_base = rng_key[0];
    uint32_t key1_base = rng_key[1];
    uint32_t row_off = row * vocab;

    float best = -INFINITY;
    uint32_t best_idx = 0u;

    uint32_t chunk = (uint32_t)TgSize * 4u;
    uint32_t stride = chunk * (uint32_t)NumSplits;
    for (uint32_t base = split * chunk + t * 4u; base < vocab; base += stride) {
        // Philox-4x32-10 over counter {base/4, 0, row, 0}.
        uint32_t c0 = base >> 2;
        uint32_t c1 = 0u;
        uint32_t c2 = row;
        uint32_t c3 = 0u;
        uint32_t k0 = key0_base;
        uint32_t k1 = key1_base;
        for (uint32_t r = 0u; r < 10u; r++) {
            uint32_t hi0 = __umulhi(0xD2511F53u, c0);
            uint32_t lo0 = 0xD2511F53u * c0;
            uint32_t hi1 = __umulhi(0xCD9E8D57u, c2);
            uint32_t lo1 = 0xCD9E8D57u * c2;
            uint32_t n0 = hi1 ^ c1 ^ k0;
            uint32_t n1 = lo1;
            uint32_t n2 = hi0 ^ c3 ^ k1;
            uint32_t n3 = lo0;
            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }

        // Ternary chain rather than a thread-local array; see the Metal source.
        for (uint32_t j = 0u; j < 4u; j++) {
            uint32_t idx = base + j;
            if (idx >= vocab) {
                break;
            }
            uint32_t word =
                (j == 0u) ? c0 : ((j == 1u) ? c1 : ((j == 2u) ? c2 : c3));
            // Uniform on the OPEN interval (0, 1); see the Metal source for
            // why this takes 23 bits rather than 24.
            float u = ((float)(word >> 9) + 0.5f) * (1.0f / 8388608.0f);
            float g = -logf(-logf(u));
            float scaled = (float)logits[row_off + idx] / temp_v;
            float cand = scaled + g;
            if (cand > best || (cand == best && idx < best_idx)) {
                best = cand;
                best_idx = idx;
            }
        }
    }

    __shared__ float tg_val[TgSize];
    __shared__ uint32_t tg_idx[TgSize];
    tg_val[t] = best;
    tg_idx[t] = best_idx;
    __syncthreads();

    for (uint32_t s = (uint32_t)TgSize / 2u; s > 0u; s >>= 1) {
        if (t < s) {
            float other = tg_val[t + s];
            uint32_t other_idx = tg_idx[t + s];
            if (other > tg_val[t] ||
                (other == tg_val[t] && other_idx < tg_idx[t])) {
                tg_val[t] = other;
                tg_idx[t] = other_idx;
            }
        }
        __syncthreads();
    }

    if (t == 0u) {
        uint32_t out_off = row * (uint32_t)NumSplits + split;
        vals[out_off] = tg_val[0];
        idxs[out_off] = tg_idx[0];
    }
)";

// Thread-safe lazy-initialised holder for the JIT-compiled Metal kernel.
// Mirrors the `std::call_once` pattern in `paged_attention.cpp`: the server
// reaches first use concurrently from per-request blocking workers, and
// `call_once` re-runs the initializer if MLX device lookup throws.
struct GumbelKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_gumbel_max_sample",
                {"logits", "rng_key", "temp"},
                {"vals", "idxs"},
                std::string(GUMBEL_MAX_SAMPLE_SOURCE));
        });
        return *kernel;
    }
};

inline GumbelKernelHolder& get_gumbel_kernel() {
    static GumbelKernelHolder holder;
    return holder;
}

// CUDA counterpart of `GumbelKernelHolder`, reached only on a CUDA backend
// where `metal::is_available()` is false.
struct GumbelKernelHolderCuda {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::cuda_kernel(
                "mlxcel_gumbel_max_sample",
                {"logits", "rng_key", "temp"},
                {"vals", "idxs"},
                std::string(GUMBEL_MAX_SAMPLE_CUDA_SOURCE));
        });
        return *kernel;
    }
};

inline GumbelKernelHolderCuda& get_gumbel_kernel_cuda() {
    static GumbelKernelHolderCuda holder;
    return holder;
}

} // namespace

bool gumbel_max_sample_supported() {
    if (mlx::core::default_device() != mlx::core::Device::gpu) {
        return false;
    }
    return mlx::core::metal::is_available() || mlx::core::cu::is_available();
}

bool gumbel_max_sample_accepts(const mlx::core::array& logits) {
    if (logits.ndim() != 2) {
        return false;
    }
    if (logits.shape(0) <= 0 || logits.shape(1) <= 0) {
        return false;
    }
    const auto dtype = logits.dtype();
    return dtype == mlx::core::float32 || dtype == mlx::core::float16 ||
        dtype == mlx::core::bfloat16;
}

int gumbel_num_splits(int batch, int vocab) {
    if (batch <= 0 || vocab <= 0) {
        return 1;
    }

    // Sweeps available in this row. Splitting past this leaves threadgroups
    // with nothing to read while still paying a launch and a reduction.
    const long long chunks =
        (static_cast<long long>(vocab) + GUMBEL_CHUNK - 1) / GUMBEL_CHUNK;
    int chunk_cap = 1;
    while (chunk_cap * 2 <= chunks && chunk_cap < GUMBEL_TARGET_THREADGROUPS) {
        chunk_cap <<= 1;
    }

    // Split just enough that `batch * splits` reaches the threadgroup target.
    int want = 1;
    while (want < GUMBEL_TARGET_THREADGROUPS &&
           static_cast<long long>(batch) * want < GUMBEL_TARGET_THREADGROUPS) {
        want <<= 1;
    }

    return want < chunk_cap ? want : chunk_cap;
}

mlx::core::array gumbel_max_sample(
    const mlx::core::array& logits,
    float temperature) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    const int batch = logits.shape(0);
    const int vocab = logits.shape(1);
    const int num_splits = gumbel_num_splits(batch, vocab);

    // Metal kernel on Apple, CUDA port elsewhere. `mx.fast.metal_kernel` throws
    // "[metal_kernel] No Metal back-end" on the CUDA backend, so dispatch the
    // `cuda_kernel` port there; `metal::is_available()` is false on a CUDA-only
    // build. Both kernels share the template args, grid, and buffer contract.
    const bool use_cuda = !mlx::core::metal::is_available();
    auto& kernel =
        use_cuda ? get_gumbel_kernel_cuda().get() : get_gumbel_kernel().get();

    // One key per call, drawn from MLX's default (thread-local) PRNG key
    // sequence, which is exactly what `random::categorical` consumes. A call
    // therefore advances the shared random state once, and
    // `mlx::core::random::seed(...)` reproduces the stream. Two u32 words are
    // the Philox key; the counter carries the (row, element) pair.
    auto rng_key = mlx::core::random::bits(Shape{2}, 4);
    auto temp_arr =
        mlx::core::full(Shape{1}, temperature, mlx::core::float32);

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"TgSize", GUMBEL_TG_SIZE},
        {"NumSplits", num_splits},
    };

    std::vector<mlx::core::array> inputs = {logits, rng_key, temp_arr};
    std::vector<Shape> output_shapes = {
        Shape{batch, num_splits},
        Shape{batch, num_splits},
    };
    std::vector<Dtype> output_dtypes = {mlx::core::float32, mlx::core::uint32};

    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(GUMBEL_TG_SIZE, num_splits, batch), // grid
        std::make_tuple(GUMBEL_TG_SIZE, 1, 1),              // threadgroup
        template_args,
        std::nullopt,
        false,
        {});

    if (num_splits == 1) {
        return mlx::core::reshape(results[1], Shape{batch});
    }

    // Merge the per-split partials. Tiny ([B, NumSplits] with NumSplits <= 64),
    // so a graph reduction costs less than a second JIT kernel.
    auto best = mlx::core::argmax(results[0], -1, true);
    auto picked = mlx::core::take_along_axis(
        results[1], mlx::core::astype(best, mlx::core::int32), -1);
    return mlx::core::reshape(picked, Shape{batch});
}

} // namespace mlxcel::turbo
