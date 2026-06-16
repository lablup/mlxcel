// Copyright 2025 mlx-lm-rs authors
// Native safetensors loading and Metal 4 / turbo / paged fused-attention
// launchers for the mlx_cxx bridge. Split out of mlx_cxx_bridge.cpp.

#include "mlx_cxx_internal.h"
#include "sparse_v_sdpa.h"          // fused Sparse-V SDPA kernel.
#include "turbo4_delegated_sdpa.h"  // fused Turbo4Delegated SDPA kernel.
#include "paged_attention.h"        // fused paged-attention decode kernel (#123).

namespace mlx_cxx {

// ── Native safetensors loading ────────────────────────────────────────

// ── Metal 4 fused attention kernel scaffolding ────────────────────────────────
//
// BACKGROUND: Metal 4 (available on M5 and later) introduces TensorOps —
// first-class tensor primitives built into the Metal Shading Language (MSL).
// Key capabilities relevant to attention fusion:
//
//   - MTLTensor: Native GPU tensor resource type with explicit layout control.
//     Allows keeping intermediate Q, K, V, scores, and context vectors
//     on-chip in registers between shader stages.
//
//   - Tensor matmul / reduction: MSL built-in operators for fusing QKV
//     projection, RoPE, score computation, softmax, and output projection
//     into a SINGLE GPU dispatch, eliminating all intermediate memory writes.
//
//   - MTL4MachineLearningCommandEncoder: Encodes full ML subgraphs onto the
//     GPU command timeline, enabling the Neural Accelerator to execute an
//     entire attention layer without returning to the CPU scheduler.
//
// CURRENT STATUS:
//   - For standard SDPA, this bridge delegates to upstream MLX
//     `fast::scaled_dot_product_attention()`, which can dispatch to the
//     backend Metal/NAX fused kernels when eligible.
//   - For softcap attention (Gemma family), this bridge currently keeps the
//     existing compiled softcap fallback.
//   - window_size metadata is currently represented by explicit masks from
//     Rust model code (no dedicated on-chip band-mask kernel here yet).
//
// WHEN TO IMPLEMENT THE FULL KERNEL:
//   Requirements:
//     - macOS 26.2 or later (first macOS release supporting Metal 4)
//     - Xcode with Metal 4 SDK (WWDC25 release cycle)
//     - M5 hardware for compilation and testing
//   Reference material:
//     - WWDC25 "Metal 4 TensorOps" session
//     - WWDC25 "Accelerate ML inference with Metal 4" session
//     - https://github.com/liuliu/example_matmul_metal4  (open-source matmul example)
//     - mlx/backend/metal/steel_attention.metal  (MLX baseline for reference)
//
std::unique_ptr<MlxArray> fused_metal4_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask,
    float softcap,
    int32_t window_size,
    bool use_metal4
) {
    // window_size is currently represented by an explicit mask from Rust.
    (void)window_size;

    std::optional<mlx::core::array> mask_opt = std::nullopt;
    std::string mask_mode = "";
    if (mask) {
        auto m = mask->inner;
        // Float masks are additive masks and should match q dtype.
        // Bool/int masks must preserve original semantics.
        if (mlx::core::issubdtype(m.dtype(), mlx::core::floating) &&
            m.dtype() != q.inner.dtype()) {
            m = mlx::core::astype(m, q.inner.dtype());
        }
        mask_opt = m;
        mask_mode = "array";
    }

    // Keep existing softcap semantics (including GQA) on the Metal4 bridge path
    // until upstream MLX provides a native fused softcap SDPA variant.
    if (softcap > 0.0f) {
        auto q_heads = q.inner.shape(1);
        auto kv_heads = k.inner.shape(1);
        std::unique_ptr<MlxArray> mask_holder = nullptr;
        if (mask_opt.has_value()) {
            mask_holder = std::make_unique<MlxArray>(mask_opt.value());
        }
        const MlxArray* mask_ptr = mask_holder ? mask_holder.get() : nullptr;

        if (q_heads > kv_heads && q_heads % kv_heads == 0) {
            int32_t n_rep = static_cast<int32_t>(q_heads / kv_heads);
            return compiled_softcap_sdpa_gqa(q, k, v, scale, softcap, n_rep, mask_ptr);
        }
        return compiled_softcap_sdpa(q, k, v, scale, softcap, mask_ptr);
    }

    // Standard path: delegate to upstream MLX fast SDPA so backend kernel
    // selection (including M5 NAX full/vector kernels) remains active.
    (void)use_metal4;
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, mask_mode, mask_opt
    ));
}

// Fused Sparse-V SDPA Metal kernel launcher. Implementation in
// `src/lib/mlx-cpp/turbo/sparse_v_sdpa.cpp`; we forward the call here so the
// new symbol shows up in the cxx-bridge ABI without bloating the bridge .cpp.
std::unique_ptr<MlxArray> turbo_sparse_v_weighted_sum(
    const MlxArray& attn_weights,
    const MlxArray& v_packed,
    const MlxArray& v_rescale,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    float threshold) {
    auto out = mlxcel::turbo::sparse_v_weighted_sum(
        attn_weights.inner,
        v_packed.inner,
        v_rescale.inner,
        codebook.inner,
        dim,
        n_rep,
        threshold);
    return std::make_unique<MlxArray>(std::move(out));
}

// Fused Turbo4Delegated cold-V weighted-sum kernel launcher.
// Implementation in `src/lib/mlx-cpp/turbo/turbo4_delegated_sdpa.cpp`; we
// forward the call here so the new symbol shows up in the cxx-bridge ABI
// without bloating the bridge .cpp. The unrotated cold weighted sum returned
// here is paired with a host-side hot V matmul; the dequantised cold V never
// materialises in global memory (the whole point).
std::unique_ptr<MlxArray> turbo4_delegated_cold_weighted_sum(
    const MlxArray& attn_weights_cold,
    const MlxArray& v_packed_cold,
    const MlxArray& v_rescale_cold,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    float threshold) {
    auto out = mlxcel::turbo::turbo4_delegated_cold_weighted_sum(
        attn_weights_cold.inner,
        v_packed_cold.inner,
        v_rescale_cold.inner,
        codebook.inner,
        dim,
        n_rep,
        threshold);
    return std::make_unique<MlxArray>(std::move(out));
}

std::unique_ptr<MlxArray> turbo4_delegated_bulk_dequant_rotated(
    const MlxArray& v_packed,
    const MlxArray& v_rescale,
    const MlxArray& codebook,
    int32_t dim) {
    auto out = mlxcel::turbo::turbo4_delegated_bulk_dequant_rotated(
        v_packed.inner,
        v_rescale.inner,
        codebook.inner,
        dim);
    return std::make_unique<MlxArray>(std::move(out));
}

// Steel-attention-envelope fused Turbo4Delegated SDPA bridge.
// Wraps the launcher in `src/lib/mlx-cpp/turbo/turbo4_delegated_sdpa.cpp` and
// repackages the std::vector<mlx::core::array> return into the `cxx`-friendly
// `Turbo4DelegatedSteelOutputs` struct (cxx does not directly model multiple
// return values, but it does model unique_ptr<struct>).
std::unique_ptr<Turbo4DelegatedSteelOutputs> turbo4_delegated_steel_sdpa(
    const MlxArray& scores,
    const MlxArray& cold_packed,
    const MlxArray& cold_rescale,
    const MlxArray& hot_v,
    const MlxArray& codebook,
    int32_t dim,
    int32_t n_rep,
    int32_t cold_offset,
    int32_t hot_offset,
    float threshold) {
    auto outs = mlxcel::turbo::turbo4_delegated_steel_sdpa(
        scores.inner,
        cold_packed.inner,
        cold_rescale.inner,
        hot_v.inner,
        codebook.inner,
        dim,
        n_rep,
        cold_offset,
        hot_offset,
        threshold);
    auto wrapper = std::make_unique<Turbo4DelegatedSteelOutputs>();
    wrapper->out_cold_pre = std::make_unique<MlxArray>(std::move(outs[0]));
    wrapper->out_hot      = std::make_unique<MlxArray>(std::move(outs[1]));
    return wrapper;
}

// Fused paged-attention decode kernel launcher (epic #116 Phase 6, #123).
// Implementation in `src/lib/mlx-cpp/turbo/paged_attention.cpp`; forwarded here
// so the new symbol shows up in the cxx-bridge ABI. Reads scattered KV blocks
// out of the global pool via the block table with no separate gather copy.
std::unique_ptr<MlxArray> paged_attention_decode(
    const MlxArray& q,
    const MlxArray& k_pool,
    const MlxArray& v_pool,
    const MlxArray& rows,
    const MlxArray& row_offsets,
    const MlxArray& logical_starts,
    const MlxArray& visible_lens,
    float scale) {
    auto out = mlxcel::turbo::paged_attention_decode(
        q.inner,
        k_pool.inner,
        v_pool.inner,
        rows.inner,
        row_offsets.inner,
        logical_starts.inner,
        visible_lens.inner,
        scale);
    return std::make_unique<MlxArray>(std::move(out));
}

std::unique_ptr<MlxLoadedWeights> mlx_load_safetensors(rust::Str path) {
    std::string path_str(path.data(), path.size());
    auto [weights_map, metadata] = mlx::core::load_safetensors(path_str);

    auto result = std::make_unique<MlxLoadedWeights>();
    result->names.reserve(weights_map.size());
    result->arrays.reserve(weights_map.size());

    for (auto& [name, arr] : weights_map) {
        result->names.push_back(std::move(name));
        result->arrays.push_back(std::make_unique<MlxArray>(std::move(arr)));
    }

    return result;
}

size_t loaded_weights_len(const MlxLoadedWeights& w) {
    return w.names.size();
}

rust::String loaded_weights_name(const MlxLoadedWeights& w, size_t index) {
    return rust::String(w.names.at(index));
}

std::unique_ptr<MlxArray> loaded_weights_take(MlxLoadedWeights& w, size_t index) {
    return std::move(w.arrays.at(index));
}

// steel SDPA output takers. The cxx bridge does not directly
// model destructuring a struct returned over the FFI boundary; we expose two
// simple `move-out` accessors that the Rust caller invokes once each. After
// both calls the struct's `unique_ptr` slots are empty (a second call would
// return a null `unique_ptr`, which is harmless because the Rust side drops
// the struct after the second take).
std::unique_ptr<MlxArray> steel_outputs_take_cold(Turbo4DelegatedSteelOutputs& o) {
    return std::move(o.out_cold_pre);
}

std::unique_ptr<MlxArray> steel_outputs_take_hot(Turbo4DelegatedSteelOutputs& o) {
    return std::move(o.out_hot);
}

}  // namespace mlx_cxx
