// Copyright 2025 mlx-lm-rs authors
// Fused single-token decode Metal kernels for the mlx_cxx bridge:
// Mamba2/SSM, GatedDeltaNet, and the decode-MoE expert path. Split out of
// mlx_cxx_bridge.cpp; see mlx_cxx_internal.h for the shared helpers.

#include "mlx_cxx_internal.h"

namespace mlx_cxx {

// ── BitLinear ternary matmul (BitNet b1.58) ────────────────────────────────
// Port of mlx-lm bitlinear_layers.py make_bitlinear_kernel(): multiply directly
// on 2-bit-packed ternary weights (4 output rows per uint8) without unpacking.
// packed_weights is [out_features/4, in_features] uint8; each byte holds the
// {-1,0,+1} weights (stored as {0,1,2}, value = bits - 1) for output rows
// {row, row+out/4, row+2*out/4, row+3*out/4} at one input column. One simdgroup
// (32 lanes) per (batch, row/4) reduces the in_features dim and writes 4 rows.
namespace {
    static const char* BITLINEAR_METAL_SOURCE = R"(
        constexpr int M = 4;
        uint tid       = thread_position_in_grid.y;   // batch * out/4
        uint in_offset = thread_position_in_grid.x;   // lane 0..31

        uint batch_idx = tid / (out_features / 4);
        uint row_idx   = tid % (out_features / 4);

        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = in_offset * M; i < in_features; i += 32u * M) {
            float v[M];
            for (int j = 0; j < M; j++) {
                v[j] = (float)x[batch_idx * in_features + i + j];
            }
            for (int j = 0; j < M; j++) {
                uint w = packed_weights[row_idx * in_features + i + j];
                sum[0] += v[j] * ((float)(w & 3u) - 1.0f);
                sum[1] += v[j] * ((float)((w >> 2) & 3u) - 1.0f);
                sum[2] += v[j] * ((float)((w >> 4) & 3u) - 1.0f);
                sum[3] += v[j] * ((float)((w >> 6) & 3u) - 1.0f);
            }
        }
        for (int j = 0; j < 4; j++) {
            sum[j] = simd_sum(sum[j]);
        }
        if (in_offset == 0) {
            float scale = invert_weight_scales ? 1.0f / (float)weight_scale[0]
                                               : (float)weight_scale[0];
            for (int i = 0; i < 4; i++) {
                out[batch_idx * out_features + row_idx + i * (out_features / 4)] =
                    (T)(sum[i] * scale);
            }
        }
    )";

    struct BitlinearKernelHolder {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::metal_kernel(
                    "bitlinear_matmul",
                    {"x", "packed_weights", "weight_scale"},
                    {"out"},
                    BITLINEAR_METAL_SOURCE);
                initialized = true;
            }
            return *kernel;
        }
    };
    static BitlinearKernelHolder& get_bitlinear_kernel() {
        static BitlinearKernelHolder holder;
        return holder;
    }

    // CUDA port of the BitLinear ternary matmul (the Metal source above is
    // mx.fast.metal_kernel, which throws "[metal_kernel] No Metal back-end" on
    // the CUDA backend). Same computation via mx.fast.cuda_kernel: one warp (32
    // lanes, threadIdx.x) per (batch, out/4) row group, striding in_features in
    // chunks of M=4 and reducing with __shfl_down_sync instead of simd_sum.
    // Selected at runtime by metal::is_available() in bitlinear_matmul.
    static const char* BITLINEAR_CUDA_SOURCE = R"(
        constexpr int Mc = 4;
        uint32_t tid       = blockIdx.y;     // batch * out/4
        uint32_t in_offset = threadIdx.x;    // lane 0..31

        uint32_t batch_idx = tid / (out_features / 4);
        uint32_t row_idx   = tid % (out_features / 4);

        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t i = in_offset * Mc; i < (uint32_t)in_features; i += 32u * Mc) {
            float v[Mc];
            for (int j = 0; j < Mc; j++) {
                v[j] = (float)x[batch_idx * in_features + i + j];
            }
            for (int j = 0; j < Mc; j++) {
                uint32_t w = packed_weights[row_idx * in_features + i + j];
                sum[0] += v[j] * ((float)(w & 3u) - 1.0f);
                sum[1] += v[j] * ((float)((w >> 2) & 3u) - 1.0f);
                sum[2] += v[j] * ((float)((w >> 4) & 3u) - 1.0f);
                sum[3] += v[j] * ((float)((w >> 6) & 3u) - 1.0f);
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum[0] += __shfl_down_sync(0xffffffffu, sum[0], o);
            sum[1] += __shfl_down_sync(0xffffffffu, sum[1], o);
            sum[2] += __shfl_down_sync(0xffffffffu, sum[2], o);
            sum[3] += __shfl_down_sync(0xffffffffu, sum[3], o);
        }
        if (in_offset == 0u) {
            float scale = invert_weight_scales ? 1.0f / (float)weight_scale[0]
                                               : (float)weight_scale[0];
            for (int i = 0; i < 4; i++) {
                out[batch_idx * out_features + row_idx + i * (out_features / 4)] =
                    (T)(sum[i] * scale);
            }
        }
    )";

    struct BitlinearKernelHolderCuda {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::cuda_kernel(
                    "bitlinear_matmul_cu",
                    {"x", "packed_weights", "weight_scale"},
                    {"out"},
                    BITLINEAR_CUDA_SOURCE);
                initialized = true;
            }
            return *kernel;
        }
    };
    static BitlinearKernelHolderCuda& get_bitlinear_kernel_cuda() {
        static BitlinearKernelHolderCuda holder;
        return holder;
    }
}

std::unique_ptr<MlxArray> bitlinear_matmul(
    const MlxArray& x,
    const MlxArray& packed_weights,
    const MlxArray& weight_scale,
    int32_t in_features,
    int32_t out_features,
    bool invert_weight_scales
) {
    using namespace mlx::core;
    auto T = x.inner.dtype();

    // Flatten leading dims to [total_batch, in_features].
    auto xs = x.inner.shape();
    int total_batch = 1;
    for (size_t i = 0; i + 1 < xs.size(); ++i) total_batch *= (int)xs[i];
    auto x2d = reshape(astype(x.inner, T), {total_batch, in_features});

    // mx.fast.metal_kernel throws on CUDA, so dispatch the cuda_kernel port
    // there. metal::is_available() is false on a CUDA-only build.
    const bool use_cuda = !mlx::core::metal::is_available();
    auto& kernel = use_cuda ? get_bitlinear_kernel_cuda().get()
                            : get_bitlinear_kernel().get();
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> ta = {
        {"T", T},
        {"in_features", in_features},
        {"out_features", out_features},
        {"invert_weight_scales", invert_weight_scales ? 1 : 0},
    };
    std::vector<array> inputs = {
        x2d, packed_weights.inner, astype(weight_scale.inner, T),
    };
    auto results = kernel(
        inputs, { Shape{total_batch, out_features} }, { T },
        std::make_tuple(32, total_batch * (out_features / 4), 1),
        std::make_tuple(32, 1, 1),
        ta, std::nullopt, false, {});

    Shape out_shape(xs.begin(), xs.end() - 1);
    out_shape.push_back(out_features);
    return std::make_unique<MlxArray>(reshape(results[0], out_shape));
}

// ── xIELU activation (Apertus) fused elementwise Metal kernel ───────────────
// Collapses the ~11 elementwise MLX ops in src/models/apertus.rs::apertus_xielu
// (square, multiply_scalar, full, minimum, expm1, subtract, multiply_scalar,
// greater, where, multiply_scalar, add) into a single launch over the MLP
// intermediate buffer. The element-wise formula, mirroring mlx-lm's XieLU
// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/activations.py):
//   x  > 0:  alpha_p * x^2 + beta * x
//   x <= 0:  (expm1(min(x, eps)) - x) * alpha_n + beta * x
//
// Every intermediate is held in T (the input dtype, bfloat16 for the native
// Apertus path), so each sub-expression rounds exactly where the elementwise
// reference rounds: bfloat arithmetic on Metal promotes to float, computes, and
// rounds back to bfloat per operation, matching MLX's per-op bf16 store. The
// scalars are cast f32->T once in-kernel, matching the reference's
// multiply_scalar/full helpers which materialize the scalar in the array dtype.
// expm1f((float)x) mirrors MLX's Expm1 unary op. The result is greedy-temp-0
// byte-identical to apertus_xielu on Apple Silicon (verified in #409).
namespace {
    // Device-side expm1f for the negative branch. The JIT metal_kernel preamble
    // (utils.h) does NOT pull in MLX's expm1f.h, and Metal's <metal_math> has no
    // expm1, so `mx.fast.metal_kernel` cannot see the same expm1f that the AOT
    // unary kernels use. To stay greedy-temp-0 byte-identical to the elementwise
    // path (whose expm1 routes through mlx::core::expm1 -> the AOT expm1f), this
    // header reproduces MLX's expm1f verbatim (renamed to avoid any clash). It
    // is itself Norbert Juffa's BSD-2-Clause routine carried in MLX; see the
    // top-level NOTICE.
    //
    // Derived from mlx/backend/metal/kernels/expm1f.h in ml-explore/mlx
    // (https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/expm1f.h),
    // Copyright (c) Apple Inc. (MIT) and Copyright (c) 2015-2023 Norbert Juffa
    // (BSD-2-Clause). See the top-level NOTICE file.
    static const char* XIELU_METAL_HEADER = R"(
        inline float xielu_expm1f_scaled_unchecked(float a, float b) {
            float f, j, r, s, t, u, v, x, y;
            int i;
            j = metal::fma(1.442695f, a, 12582912.f);
            j = j - 12582912.0f;
            i = (int)j;
            f = metal::fma(j, -6.93145752e-1f, a);
            s = f * f;
            if (a == 0.0f) s = a;
            r = 1.97350979e-4f;
            r = metal::fma(r, f, 1.39309070e-3f);
            r = metal::fma(r, f, 8.33343994e-3f);
            r = metal::fma(r, f, 4.16668020e-2f);
            r = metal::fma(r, f, 1.66666716e-1f);
            r = metal::fma(r, f, 4.99999970e-1f);
            u = (j == 1) ? (f + 0.5f) : f;
            v = metal::fma(r, s, u);
            s = 0.5f * b;
            t = metal::ldexp(s, i);
            y = t - s;
            x = (t - y) - s;
            r = metal::fma(v, t, x) + y;
            r = r + r;
            if (j == 0) r = v;
            if (j == 1) r = v + v;
            return r;
        }
        inline float xielu_expm1f(float a) {
            float r = xielu_expm1f_scaled_unchecked(a, 1.0f);
            if (metal::fabs(a - 1.0f) > 88.0f) {
                r = metal::pow(2.0f, a);
                r = metal::fma(r, r, -1.0f);
            }
            return r;
        }
    )";

    static const char* XIELU_METAL_SOURCE = R"(
        uint i = thread_position_in_grid.x;
        if (i >= (uint)n) { return; }
        T xx = x[i];
        T ap = (T)alpha_p[0];
        T an = (T)alpha_n[0];
        T bb = (T)beta[0];
        T ee = (T)eps[0];
        // Each named T (bfloat for the native Apertus path) intermediate forces
        // a round-to-bf16 at exactly the points the elementwise reference rounds
        // (one MLX op == one round). Keeping every sub-expression to a single
        // arithmetic op prevents the Metal compiler from contracting a multiply
        // and add into one FMA (which would round once instead of twice) and
        // makes the fused result greedy-temp-0 byte-identical to apertus_xielu.
        T pos_x_sq = xx * xx;        // square(x)
        T pos_core = ap * pos_x_sq;  // alpha_p * x^2
        T clamped = min(xx, ee);     // minimum(x, eps)
        T em = (T)xielu_expm1f((float)clamped);  // expm1(clamped)
        T neg_sub = em - xx;         // expm1(clamped) - x
        T neg_core = neg_sub * an;   // (expm1(clamped) - x) * alpha_n
        T selected = (xx > (T)0) ? pos_core : neg_core;  // where(x > 0, ...)
        T bx = xx * bb;              // beta * x (added once, outside the select)
        out[i] = selected + bx;      // selected + beta * x
    )";

    struct XieluKernelHolder {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::metal_kernel(
                    "xielu_fused",
                    {"x", "alpha_p", "alpha_n", "beta", "eps"},
                    {"out"},
                    XIELU_METAL_SOURCE,
                    XIELU_METAL_HEADER);
                initialized = true;
            }
            return *kernel;
        }
    };
    static XieluKernelHolder& get_xielu_kernel() {
        static XieluKernelHolder holder;
        return holder;
    }

    // Elementwise fallback mirroring src/models/apertus.rs::apertus_xielu. Used
    // when the Metal back-end is unavailable (e.g. a CUDA-only build, where
    // mx.fast.metal_kernel throws "[metal_kernel] No Metal back-end"). Keeps the
    // FFI entry total so the MLXCEL_FUSED_XIELU flag never crashes a non-Metal
    // build; the per-op result is identical to the Rust reference.
    static mlx::core::array xielu_elementwise(
        const mlx::core::array& x, float alpha_p, float alpha_n,
        float beta, float eps) {
        using namespace mlx::core;
        auto dt = x.dtype();
        auto ap = array(alpha_p, dt);
        auto an = array(alpha_n, dt);
        auto bb = array(beta, dt);
        auto ee = array(eps, dt);
        auto pos_core = multiply(ap, square(x));
        auto neg_core = multiply(an, subtract(expm1(minimum(x, ee)), x));
        auto cond = greater(x, array(0.0f, dt));
        auto selected = where(cond, pos_core, neg_core);
        return add(selected, multiply(x, bb));
    }
}

std::unique_ptr<MlxArray> fused_xielu(
    const MlxArray& x,
    float alpha_p,
    float alpha_n,
    float beta,
    float eps
) {
    using namespace mlx::core;
    auto T = x.inner.dtype();
    auto xs = x.inner.shape();

    // Non-Metal back-ends: mx.fast.metal_kernel throws, so use the elementwise
    // fallback (correct, just not fused). Apertus is a macOS/Metal target.
    if (!mlx::core::metal::is_available()) {
        return std::make_unique<MlxArray>(
            xielu_elementwise(x.inner, alpha_p, alpha_n, beta, eps));
    }

    int64_t n = 1;
    for (auto d : xs) n *= d;
    auto xflat = reshape(x.inner, {(int)n});

    // Per-layer scalars as 1-element f32 inputs; cast f32->T once in-kernel.
    auto ap = full({1}, alpha_p, float32);
    auto an = full({1}, alpha_n, float32);
    auto bb = full({1}, beta, float32);
    auto ee = full({1}, eps, float32);

    auto& kernel = get_xielu_kernel().get();
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> ta = {
        {"T", T},
        {"n", (int)n},
    };
    std::vector<array> inputs = {xflat, ap, an, bb, ee};
    const int tg = 256;
    const int grid = (int)(((n + tg - 1) / tg) * tg);
    auto results = kernel(
        inputs, {Shape{(int)n}}, {T},
        std::make_tuple(grid, 1, 1),
        std::make_tuple(tg, 1, 1),
        ta, std::nullopt, false, {});

    return std::make_unique<MlxArray>(reshape(results[0], xs));
}

// SSM (Mamba2) fused Metal kernel for single-token decode.
// Port of Python mlx-lm ssm.py make_ssm_kernel() + ssm_update_kernel()
namespace {
    static const char* SSM_METAL_SOURCE = R"(
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<U>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
    )";

    struct SsmKernelHolder {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;

        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::metal_kernel(
                    "ssm_kernel",
                    {"X", "A_log", "B", "C", "D", "dt", "state_in"},
                    {"out", "state_out"},
                    SSM_METAL_SOURCE
                );
                initialized = true;
            }
            return *kernel;
        }
    };

    static SsmKernelHolder& get_ssm_kernel() {
        static SsmKernelHolder holder;
        return holder;
    }

    // Compiled compute_dt: float32 promotion + softplus + clip → single fused kernel
    // Matches Python's @mx.compile compute_dt (casts dt to float32 before softplus for precision)
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_compute_dt() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            auto dt = mlx::core::astype(inputs[0], mlx::core::float32);
            const auto& dt_bias = inputs[1];
            const auto& lo = inputs[2];
            const auto& hi = inputs[3];
            auto result = mlx::core::add(dt, dt_bias);
            result = mlx::core::log1p(mlx::core::exp(result));
            return {mlx::core::clip(result, lo, hi)};
        };
        return mlx::core::compile(fn, true);
    }

    static array compute_dt_compiled(const array& dt, const array& dt_bias, float min_val, float max_val) {
        static auto compiled_fn = get_compiled_compute_dt();
        auto lo = mlx::core::array(min_val);
        auto hi = mlx::core::array(max_val);
        return compiled_fn({dt, dt_bias, lo, hi})[0];
    }
}

// GatedDeltaNet custom Metal kernel for single/multi-token decode.
// Port of Python mlx-lm gated_delta.py _make_gated_delta_kernel()
// Four variants: scalar/vec gate x mask/no-mask
namespace {
    // Variant 1: scalar gate, no mask
    static const char* GATED_DELTA_METAL_SOURCE = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
            if (true) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                    y[dv_idx] = static_cast<InT>(out);
                }
            }
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";

    // Variant 2: scalar gate, with mask
    static const char* GATED_DELTA_METAL_SOURCE_MASK = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
            if (mask[b_idx * T + t]) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                    y[dv_idx] = static_cast<InT>(out);
                }
            } else {
                y[dv_idx] = static_cast<InT>(0);
            }
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";

    // Variant 3: vectorized gate (per-dim), no mask
    static const char* GATED_DELTA_METAL_SOURCE_VEC = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv, Dk]
        auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
            if (true) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[s_idx];
                    kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                    y[dv_idx] = static_cast<InT>(out);
                }
            }
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv * Dk;
            beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";

    // Variant 4: vectorized gate, with mask
    static const char* GATED_DELTA_METAL_SOURCE_VEC_MASK = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv, Dk]
        auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
            if (mask[b_idx * T + t]) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[s_idx];
                    kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                    y[dv_idx] = static_cast<InT>(out);
                }
            } else {
                y[dv_idx] = static_cast<InT>(0);
            }
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv * Dk;
            beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";

    // Kernel holder structs (lazy init, one per variant)
    struct GatedDeltaKernelHolder {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;

        mlx::core::fast::CustomKernelFunction& get(const char* name,
                                                    const std::vector<std::string>& inputs,
                                                    const char* source) {
            if (!initialized) {
                kernel = mlx::core::fast::metal_kernel(
                    name, inputs,
                    {"y", "state_out"},
                    source
                );
                initialized = true;
            }
            return *kernel;
        }
    };

    static GatedDeltaKernelHolder& get_gd_kernel() {
        static GatedDeltaKernelHolder holder;
        return holder;
    }
    static GatedDeltaKernelHolder& get_gd_kernel_mask() {
        static GatedDeltaKernelHolder holder;
        return holder;
    }
    static GatedDeltaKernelHolder& get_gd_kernel_vec() {
        static GatedDeltaKernelHolder holder;
        return holder;
    }
    static GatedDeltaKernelHolder& get_gd_kernel_vec_mask() {
        static GatedDeltaKernelHolder holder;
        return holder;
    }
}

bool gated_delta_kernel_available() {
#ifdef __APPLE__
    return mlx::core::metal::is_available();
#else
    return false;
#endif
}

// Start a GPU trace capture and stop an active one. Mirrors the Python
// `mx.metal.start_capture` / `mx.metal.stop_capture` API so a mlxcel
// decode run can emit the same `.gputrace` files that `mlx-lm` produces
// and both be loaded into Xcode's Metal Debugger side by side.
// The process must have been launched with `MTL_CAPTURE_ENABLED=1`,
// otherwise Metal silently drops the capture.
void metal_start_capture(rust::Str path) {
    mlx::core::metal::start_capture(std::string(path.data(), path.size()));
}

void metal_stop_capture() {
    mlx::core::metal::stop_capture();
}

void metal_gated_delta_forward(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    const MlxArray& g,
    const MlxArray& beta,
    const MlxArray& state,
    const MlxArray* mask,
    std::unique_ptr<MlxArray>& output,
    std::unique_ptr<MlxArray>& new_state
) {
    using namespace mlx::core;

    // Extract dimensions from input shapes
    auto q_shape = q.inner.shape();
    int B = q_shape[0];
    int T_val = q_shape[1];
    int Hk = q_shape[2];
    int Dk = q_shape[3];
    auto v_shape = v.inner.shape();
    int Hv = v_shape[2];
    int Dv = v_shape[3];

    auto input_type = q.inner.dtype();

    // Cast inputs to input_type if needed (state may be float32 from Rust side)
    auto state_cast = (state.inner.dtype() != input_type)
        ? mlx::core::astype(state.inner, input_type) : state.inner;
    auto g_cast = (g.inner.dtype() != input_type)
        ? mlx::core::astype(g.inner, input_type) : g.inner;
    auto beta_cast = (beta.inner.dtype() != input_type)
        ? mlx::core::astype(beta.inner, input_type) : beta.inner;

    // Detect vectorized gate: g has 4 dims [B, T, Hv, Dk]
    bool vectorized = (g.inner.ndim() == 4);
    bool has_mask = (mask != nullptr);

    // Build T as a scalar array input
    auto T_arr = mlx::core::array(T_val);

    // Select kernel variant and build inputs
    std::vector<array> inputs;
    GatedDeltaKernelHolder* holder;
    const char* kernel_name;
    const char* kernel_source;
    std::vector<std::string> input_names;

    if (vectorized && has_mask) {
        input_names = {"q", "k", "v", "g", "beta", "state_in", "T", "mask"};
        inputs = {q.inner, k.inner, v.inner, g_cast, beta_cast, state_cast, T_arr, mask->inner};
        holder = &get_gd_kernel_vec_mask();
        kernel_name = "gated_delta_step_vec_mask";
        kernel_source = GATED_DELTA_METAL_SOURCE_VEC_MASK;
    } else if (vectorized) {
        input_names = {"q", "k", "v", "g", "beta", "state_in", "T"};
        inputs = {q.inner, k.inner, v.inner, g_cast, beta_cast, state_cast, T_arr};
        holder = &get_gd_kernel_vec();
        kernel_name = "gated_delta_step_vec";
        kernel_source = GATED_DELTA_METAL_SOURCE_VEC;
    } else if (has_mask) {
        input_names = {"q", "k", "v", "g", "beta", "state_in", "T", "mask"};
        inputs = {q.inner, k.inner, v.inner, g_cast, beta_cast, state_cast, T_arr, mask->inner};
        holder = &get_gd_kernel_mask();
        kernel_name = "gated_delta_step_mask";
        kernel_source = GATED_DELTA_METAL_SOURCE_MASK;
    } else {
        input_names = {"q", "k", "v", "g", "beta", "state_in", "T"};
        inputs = {q.inner, k.inner, v.inner, g_cast, beta_cast, state_cast, T_arr};
        holder = &get_gd_kernel();
        kernel_name = "gated_delta_step";
        kernel_source = GATED_DELTA_METAL_SOURCE;
    }

    auto& kernel = holder->get(kernel_name, input_names, kernel_source);

    // Template parameters: InT (dtype), Dk, Dv, Hk, Hv
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
        {"InT", input_type},
        {"Dk", Dk},
        {"Dv", Dv},
        {"Hk", Hk},
        {"Hv", Hv},
    };

    // Output shapes and dtypes (matching Python: both in input_type)
    std::vector<Shape> output_shapes = {
        Shape{B, T_val, Hv, Dv},   // y
        state.inner.shape(),        // state_out (same shape as state_in)
    };
    std::vector<Dtype> output_dtypes = {input_type, input_type};

    // Grid: (32, Dv, B * Hv), Threadgroup: (32, 4, 1)
    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(32, Dv, B * Hv),   // grid
        std::make_tuple(32, 4, 1),          // threadgroup
        template_args,
        std::nullopt,  // init_value
        false,         // verbose
        {}             // stream (default)
    );

    output = std::make_unique<MlxArray>(std::move(results[0]));
    new_state = std::make_unique<MlxArray>(std::move(results[1]));
}

// Compiled MoE gate: sigmoid + bias + topk + normalize + scale
// Uses mx::core::compile for kernel fusion matching Python's @mx.compile group_expert_select
namespace {
    // Pre-sigmoid + bias + negative: compiled into fused kernel
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_gate_scores() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& gates = inputs[0];
            const auto& bias = inputs[1];
            auto orig = mlx::core::sigmoid(mlx::core::astype(gates, mlx::core::float32));
            auto biased = mlx::core::add(orig, bias);
            auto neg = mlx::core::negative(biased);
            return {orig, neg};
        };
        return mlx::core::compile(fn, true);
    }

    // Post-topk normalize + scale: compiled into fused kernel
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_gate_normalize() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& scores = inputs[0];
            const auto& scale = inputs[1];
            auto denom = mlx::core::sum(scores, -1, true);
            auto normed = mlx::core::divide(scores, mlx::core::add(denom, mlx::core::array(1e-20f)));
            return {mlx::core::multiply(normed, scale)};
        };
        return mlx::core::compile(fn, true);
    }
}

void compiled_moe_gate(
    const MlxArray& gates,
    const MlxArray& correction_bias,
    int32_t top_k,
    float scaling_factor,
    bool norm_topk_prob,
    std::unique_ptr<MlxArray>& indices_out,
    std::unique_ptr<MlxArray>& scores_out
) {
    using namespace mlx::core;

    // Step 1: compiled sigmoid + bias + negative
    static auto gate_scores_fn = get_compiled_gate_scores();
    auto gate_results = gate_scores_fn({gates.inner, correction_bias.inner});
    auto orig_scores = gate_results[0];  // sigmoid(float32)
    auto neg_biased = gate_results[1];   // negative(sigmoid + bias)

    // Step 2: argpartition + slice (not compilable due to dynamic shapes)
    auto indices = argpartition(neg_biased, top_k - 1, -1);
    auto topk_indices = slice(indices, {0, 0}, {(int)indices.shape()[0], top_k});
    auto topk_scores = take_along_axis(orig_scores, topk_indices, -1);

    // Step 3: compiled normalize + scale
    if (top_k > 1 && norm_topk_prob) {
        static auto normalize_fn = get_compiled_gate_normalize();
        auto scale = array(scaling_factor);
        auto norm_results = normalize_fn({topk_scores, scale});
        topk_scores = norm_results[0];
    } else {
        topk_scores = multiply(topk_scores, array(scaling_factor));
    }

    indices_out = std::make_unique<MlxArray>(std::move(topk_indices));
    scores_out = std::make_unique<MlxArray>(std::move(topk_scores));
}

// Fused MoE forward: combines gate + switch_mlp + score weighting + shared expert
namespace {
// Defined alongside the MoE kernel holders below; forward-declared so the
// experimental squared-ReLU fast path in fused_moe_forward can reach them (the
// holders are defined further down, after this function).
mlx::core::fast::CustomKernelFunction& moe_fc1_relu2_kernel_fn();
mlx::core::fast::CustomKernelFunction& moe_down_kernel_fn();
}  // namespace

std::unique_ptr<MlxArray> fused_moe_forward(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& correction_bias,
    const MlxArray& fc1_weight,
    const MlxArray& fc1_scales,
    const MlxArray& fc1_biases,
    const MlxArray& fc2_weight,
    const MlxArray& fc2_scales,
    const MlxArray& fc2_biases,
    const MlxArray* shared_up_weight,
    const MlxArray* shared_up_scales,
    const MlxArray* shared_up_biases,
    const MlxArray* shared_down_weight,
    const MlxArray* shared_down_scales,
    const MlxArray* shared_down_biases,
    int32_t top_k,
    float scaling_factor,
    bool norm_topk_prob,
    int32_t group_size,
    int32_t bits
) {
    using namespace mlx::core;

    // 1. Gate: compiled sigmoid + topk + compiled normalize + scale
    auto gates = matmul(x.inner, transpose(gate_weight.inner));

    // Compiled: sigmoid(astype(gates, f32)) + add(bias) + negative
    static auto gate_scores_fn = get_compiled_gate_scores();
    auto gate_results = gate_scores_fn({gates, correction_bias.inner});
    auto orig_scores = gate_results[0];
    auto neg_biased = gate_results[1];

    auto all_indices = argpartition(neg_biased, top_k - 1, -1);
    auto topk_indices = slice(all_indices, {0, 0}, {(int)all_indices.shape()[0], top_k});
    auto topk_scores = take_along_axis(orig_scores, topk_indices, -1);

    if (top_k > 1 && norm_topk_prob) {
        // Compiled: normalize + scale
        static auto normalize_fn = get_compiled_gate_normalize();
        topk_scores = normalize_fn({topk_scores, array(scaling_factor)})[0];
    } else {
        topk_scores = multiply(topk_scores, array(scaling_factor));
    }

    // 2. SwitchMLP: expand + gather_qmm(fc1) + relu² + gather_qmm(fc2) + squeeze.
    auto x_shape = x.inner.shape();
    auto T = x.inner.dtype();

    // Experimental fused squared-ReLU decode path (#268), behind its own flag
    // MLXCEL_FUSED_MOE_RELU2 (NOT the default MLXCEL_FUSED_MOE): fc1 + relu² ->
    // act_g[K, Dff], then reuse moe_down for fc2 * score. Correct and
    // byte-identical, but measured performance-NEUTRAL on nemotron-h-30b (its
    // decode is dominated by Mamba2 + attention, so the MoE expert GEMV is a
    // small, already-efficient slice). Kept wired behind the dedicated flag so
    // the kernel stays referenceable for a future squared-ReLU model whose
    // decode is MoE-dominated; the default path below stays on gather_qmm.
    bool fused_relu2 = x_shape[0] == 1 && (bits == 4 || bits == 8) &&
        std::getenv("MLXCEL_FUSED_MOE_RELU2");
    array result = x.inner;  // placeholder; overwritten in both branches
    if (fused_relu2) {
        int din = (int)x_shape[1];
        int dff = (int)fc1_weight.inner.shape()[1];
        int k = top_k;
        int sgy = 8;
        if (const char* s = std::getenv("MLXCEL_FUSED_MOE_SGY")) {
            int v = std::atoi(s);
            if (v >= 1 && v <= 32) sgy = v;
        }
        auto round_up = [](int n, int m) { return ((n + m - 1) / m) * m; };
        std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> ta = {
            {"T", T}, {"K", k}, {"Din", din}, {"Dff", dff},
            {"bits", bits}, {"group_size", group_size},
        };
        auto idx_u32 = astype(reshape(topk_indices, {k}), uint32);

        // A) fc1 + relu² -> act_g[K, Dff] (f32 for the fc2 GEMV).
        auto& kA = moe_fc1_relu2_kernel_fn();
        std::vector<array> inA = {
            astype(reshape(x.inner, {din}), T), idx_u32,
            fc1_weight.inner, astype(fc1_scales.inner, T), astype(fc1_biases.inner, T),
        };
        auto rA = kA(
            inA, { Shape{k, dff} }, { float32 },
            std::make_tuple(32, round_up(dff, sgy), k),
            std::make_tuple(32, sgy, 1),
            ta, std::nullopt, false, {});

        // B) fc2 * score -> partial[K, Din]; summed over K into [1, Din].
        auto& kB = moe_down_kernel_fn();
        std::vector<array> inB = {
            idx_u32,
            fc2_weight.inner, astype(fc2_scales.inner, T), astype(fc2_biases.inner, T),
            rA[0], astype(reshape(topk_scores, {k}), T),
        };
        auto rB = kB(
            inB, { Shape{k, din} }, { T },
            std::make_tuple(32, round_up(din, sgy), k),
            std::make_tuple(32, sgy, 1),
            ta, std::nullopt, false, {});
        result = reshape(sum(rB[0], 0, false), {1, din});
    } else {
        auto x_exp = reshape(x.inner, {x_shape[0], 1, 1, x_shape[1]});

        auto h = gather_qmm(
            x_exp, fc1_weight.inner, fc1_scales.inner, fc1_biases.inner,
            std::nullopt, topk_indices,
            true, group_size, bits, "affine", false);
        // relu² = relu(x)²
        { MlxArray h_w{h}; h = compiled_relu_squared(h_w)->inner; }
        h = gather_qmm(
            h, fc2_weight.inner, fc2_scales.inner, fc2_biases.inner,
            std::nullopt, topk_indices,
            true, group_size, bits, "affine", false);
        h = squeeze(h, -2);  // [tokens, top_k, hidden]

        // 3. Score weighting: weighted sum over experts, cast back to input dtype
        // Cast scores to h's dtype to avoid mixed float32×float16 multiply
        // which produces NaN on M5 Max (Metal GPU Family 4) NAx broadcast kernel.
        auto scores_exp = reshape(topk_scores, {topk_scores.shape()[0], top_k, 1});
        auto scores_cast = astype(scores_exp, h.dtype());
        result = astype(sum(multiply(h, scores_cast), -2, false), x.inner.dtype());
    }

    // 4. Optional shared expert
    if (shared_up_weight && shared_down_weight) {
        auto shared_h = quantized_matmul(
            x.inner, shared_up_weight->inner, shared_up_scales->inner,
            shared_up_biases ? std::optional(shared_up_biases->inner) : std::nullopt,
            true, group_size, bits);
        { MlxArray sh_w{shared_h}; shared_h = compiled_relu_squared(sh_w)->inner; }
        shared_h = quantized_matmul(
            shared_h, shared_down_weight->inner, shared_down_scales->inner,
            shared_down_biases ? std::optional(shared_down_biases->inner) : std::nullopt,
            true, group_size, bits);
        result = add(result, shared_h);
    }

    return std::make_unique<MlxArray>(std::move(result));
}

bool ssm_kernel_available() {
#ifdef __APPLE__
    return mlx::core::metal::is_available();
#else
    return false;
#endif
}

void ssm_update_kernel(
    const MlxArray& hidden_states,
    const MlxArray& A_log,
    const MlxArray& B,
    const MlxArray& C,
    const MlxArray& D,
    const MlxArray& dt,
    const MlxArray& dt_bias,
    const MlxArray& state_in,
    float time_step_min,
    float time_step_max,
    std::unique_ptr<MlxArray>& output,
    std::unique_ptr<MlxArray>& next_state
) {
    using namespace mlx::core;

    auto shape = hidden_states.inner.shape();
    int n = shape[0];  // batch
    int h = shape[2];  // num_heads
    int dh = shape[3]; // head_dim
    auto b_shape = B.inner.shape();
    int hb = b_shape[2]; // n_groups
    int ds = b_shape[3]; // state_dim
    int g = h / hb;      // heads per group

    auto input_type = hidden_states.inner.dtype();
    auto state_type = state_in.inner.dtype();

    // Compute dt with softplus + clip (promoted to float32 internally)
    auto dt_result = compute_dt_compiled(dt.inner, dt_bias.inner, time_step_min, time_step_max);

    // Call the Metal kernel
    auto& kernel = get_ssm_kernel().get();

    // CustomKernelFunction signature:
    // (inputs, output_shapes, output_dtypes, grid, threadgroup, template_args, init_value, verbose, stream)
    // T = input type for x/out, U = state type for state_in/state_out (allows float32 state accumulation)
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
        {"T", input_type},
        {"U", state_type},
        {"Dh", dh},
        {"Ds", ds},
        {"H", h},
        {"G", g},
    };

    std::vector<array> inputs = {
        hidden_states.inner, A_log.inner, B.inner, C.inner,
        mlx::core::astype(D.inner, input_type), dt_result, state_in.inner
    };
    std::vector<Shape> output_shapes = {Shape{n, 1, h, dh}, state_in.inner.shape()};
    std::vector<Dtype> output_dtypes = {input_type, state_type};

    auto results = kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(32, dh, h * n),    // grid
        std::make_tuple(32, 8, 1),          // threadgroup
        template_args,
        std::nullopt,  // init_value
        false,         // verbose
        {}             // stream (default)
    );

    output = std::make_unique<MlxArray>(std::move(results[0]));
    next_state = std::make_unique<MlxArray>(std::move(results[1]));
}

// ── Fused MoE expert kernel (single-token decode, power-of-2 bits) ──────────
// Computes a decode token's routed-expert output:
//   out[h] = sum_k scores[k] * down_k( silu(gate_k(x)) * up_k(x) )[h]
// over the K selected experts (indices[k]), with affine-quantized gate/up/down,
// as two Metal dispatches that stream every active expert weight once.
//
// A single fused launch is bandwidth-bound the wrong way at batch=1: the
// gate/up -> down dependency goes through the per-expert activation, so keeping
// it in threadgroup memory confines each expert to one threadgroup and caps
// occupancy at K cores (measured ~0.46x of gather_qmm on qwen3-30b-a3b).
// Instead we break the barrier by staging the swiglu activation in global
// memory between two dispatches, so every GEMV output row runs as an
// independent simdgroup across all GPU cores:
//   A) gate/up GEMV + swiglu  -> act_g[K, Dff] (f32)
//   B) down GEMV * score       -> partial[K, Din]
// partial is summed over K by the host wrapper. This is non-redundant (each
// weight read once) and beats gather_qmm by ~3.5% on qwen3-30b-a3b, greedy
// output byte-identical. Power-of-2 bits only (4/8); callers fall back to
// SwitchGLU for 6-bit, non-affine, or oversized configs (#268 step 2b).
namespace {
    // SGY = simdgroups per threadgroup; each owns one output row (grid.y) and
    // its 32 lanes stride the contraction dim, reduced with simd_sum (cf.
    // ssm_update_kernel). A lane unpacks a whole 32-bit pack (vpw = 32/bits
    // weights) at a time: group_size is a multiple of vpw for bits in {4, 8},
    // so every weight in a pack shares one (scale, bias). The reduction order
    // differs from gather_qmm, which is fine for the RMS<5e-3 / greedy-identical
    // gate, not bitwise parity.
    static const char* MOE_GATEUP_METAL_SOURCE = R"(
        uint lane  = thread_position_in_threadgroup.x;   // 0..31 (one simdgroup)
        uint f     = thread_position_in_grid.y;          // output row 0..Dff-1
        uint eslot = thread_position_in_grid.z;          // 0..K-1
        if (f >= Dff) return;                            // uniform across the simdgroup
        uint e = indices[eslot];

        constexpr uint vpw   = 32u / bits;
        constexpr uint wmask = (1u << bits) - 1u;
        constexpr uint Din_p = Din / vpw;
        constexpr uint G     = Din / group_size;

        uint row = e * Dff + f;
        const device uint* gwr = gate_w + row * Din_p;
        const device T*    gsr = gate_s + row * G;
        const device T*    gbr = gate_b + row * G;
        const device uint* uwr = up_w   + row * Din_p;
        const device T*    usr = up_s   + row * G;
        const device T*    ubr = up_b   + row * G;
        float g = 0.0f, u = 0.0f;
        for (uint p = lane; p < Din_p; p += 32u) {
            uint base = p * vpw;
            uint grp  = base / group_size;
            float gs = (float)gsr[grp], gb = (float)gbr[grp];
            float us = (float)usr[grp], ub = (float)ubr[grp];
            uint gpk = gwr[p];
            uint upk = uwr[p];
            for (uint j = 0; j < vpw; ++j) {
                float xv = (float)x[base + j];
                g += xv * ((float)((gpk >> (j * bits)) & wmask) * gs + gb);
                u += xv * ((float)((upk >> (j * bits)) & wmask) * us + ub);
            }
        }
        g = simd_sum(g);
        u = simd_sum(u);
        if (lane == 0u) {
            if (act == 1) {
                // GeGLU (gelu tanh approx) * up — matches
                // compiled_geglu_approx_activation (gemma4 experts).
                float g3 = g * g * g;
                float inner = 0.7978845608028654f * (g + 0.044715f * g3);
                float gelu = 0.5f * g * (1.0f + precise::tanh(inner));
                act_g[eslot * Dff + f] = gelu * u;
            } else {
                // SwiGLU (silu) * up.
                act_g[eslot * Dff + f] = (g / (1.0f + fast::exp(-g))) * u;
            }
        }
    )";

    static const char* MOE_DOWN_METAL_SOURCE = R"(
        uint lane  = thread_position_in_threadgroup.x;   // 0..31 (one simdgroup)
        uint h     = thread_position_in_grid.y;          // output row 0..Din-1
        uint eslot = thread_position_in_grid.z;          // 0..K-1
        if (h >= Din) return;                            // uniform across the simdgroup
        uint e = indices[eslot];

        constexpr uint Gd = Dff / group_size;
        uint row = e * Din + h;
        const device T*     dsr = down_s + row * Gd;
        const device T*     dbr = down_b + row * Gd;
        const device float* a   = act_g  + eslot * Dff;
        float d = 0.0f;

        if (bits == 6) {
            // Non-power-of-2: MLX packs 4 weights into 3 bytes (pack_factor 4,
            // bytes_per_pack 3). Read the row as bytes; each lane owns whole
            // packs. The 6-bit value layout matches MLX quantized.h qdot:
            //   v0=b0&0x3f, v1=(b0>>6)|((b1&0x0f)<<2),
            //   v2=(b1>>4)|((b2&0x03)<<4), v3=b2>>2.
            constexpr uint packs = Dff / 4u;          // 4 values per pack
            constexpr uint row_u32 = Dff * 3u / 16u;  // 3 bytes/pack -> uint32 cols
            const device uchar* wb = (const device uchar*)(down_w + row * row_u32);
            for (uint p = lane; p < packs; p += 32u) {
                uint base = p * 4u;
                uint grp  = base / group_size;
                float ds = (float)dsr[grp], db = (float)dbr[grp];
                uint b0 = wb[p * 3u], b1 = wb[p * 3u + 1u], b2 = wb[p * 3u + 2u];
                uint v0 = b0 & 0x3fu;
                uint v1 = (b0 >> 6) | ((b1 & 0x0fu) << 2);
                uint v2 = (b1 >> 4) | ((b2 & 0x03u) << 4);
                uint v3 = (b2 >> 2);
                d += a[base + 0u] * ((float)v0 * ds + db);
                d += a[base + 1u] * ((float)v1 * ds + db);
                d += a[base + 2u] * ((float)v2 * ds + db);
                d += a[base + 3u] * ((float)v3 * ds + db);
            }
        } else {
            // Power-of-2 bits (4/8): vpw weights per 32-bit pack, clean shift.
            // (When bits==6 this branch is dead-eliminated; the constexprs stay
            // valid, just unused.)
            constexpr uint vpw   = 32u / bits;
            constexpr uint wmask = (1u << bits) - 1u;
            constexpr uint Dff_p = Dff / vpw;
            const device uint* dwr = down_w + row * Dff_p;
            for (uint p = lane; p < Dff_p; p += 32u) {
                uint base = p * vpw;
                uint grp  = base / group_size;
                float ds = (float)dsr[grp], db = (float)dbr[grp];
                uint dpk = dwr[p];
                for (uint j = 0; j < vpw; ++j) {
                    d += a[base + j] * ((float)((dpk >> (j * bits)) & wmask) * ds + db);
                }
            }
        }
        d = simd_sum(d);
        if (lane == 0u) {
            out[eslot * Din + h] = (T)((float)scores[eslot] * d);
        }
    )";

    struct MoeGateUpKernelHolder {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::metal_kernel(
                    "moe_gateup_kernel",
                    {"x", "indices", "gate_w", "gate_s", "gate_b",
                     "up_w", "up_s", "up_b"},
                    {"act_g"},
                    MOE_GATEUP_METAL_SOURCE
                );
                initialized = true;
            }
            return *kernel;
        }
    };
    static MoeGateUpKernelHolder& get_moe_gateup_kernel() {
        static MoeGateUpKernelHolder holder;
        return holder;
    }

    struct MoeDownKernelHolder {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::metal_kernel(
                    "moe_down_kernel",
                    {"indices", "down_w", "down_s", "down_b", "act_g", "scores"},
                    {"out"},
                    MOE_DOWN_METAL_SOURCE
                );
                initialized = true;
            }
            return *kernel;
        }
    };
    static MoeDownKernelHolder& get_moe_down_kernel() {
        static MoeDownKernelHolder holder;
        return holder;
    }

    // ---- CUDA ports of the two fused decode-MoE kernels (#268 step 2b). ----
    // The Metal sources above are mx.fast.metal_kernel, which throws on the CUDA
    // backend ("[metal_kernel] No Metal back-end"). These are the same
    // computation in CUDA C++ for mx.fast.cuda_kernel: one warp (32 lanes,
    // threadIdx.x) owns each output row (grid.y), striding the contraction dim
    // and reduced with __shfl_down_sync instead of simd_sum. grid.z selects the
    // expert slot. MLX injects the template args (T, K, Din, Dff, bits,
    // group_size, act) as constexpr/using, and wraps these bodies as a
    // __global__ with the named buffers as parameters, mirroring the Metal path.
    // Selected at runtime by metal::is_available() in run_fused_moe_two_kernel.
    static const char* MOE_GATEUP_CUDA_SOURCE = R"(
        uint32_t lane  = threadIdx.x;                          // 0..31 (one warp)
        uint32_t f     = blockIdx.y * blockDim.y + threadIdx.y; // output row 0..Dff-1
        uint32_t eslot = blockIdx.z;                           // 0..K-1
        if (f >= (uint32_t)Dff) return;                        // warp-uniform
        uint32_t e = indices[eslot];

        constexpr uint32_t vpw   = 32u / bits;
        constexpr uint32_t wmask = (1u << bits) - 1u;
        constexpr uint32_t Din_p = Din / vpw;
        constexpr uint32_t G     = Din / group_size;

        uint32_t row = e * Dff + f;
        const uint32_t* gwr = gate_w + row * Din_p;
        const T*        gsr = gate_s + row * G;
        const T*        gbr = gate_b + row * G;
        const uint32_t* uwr = up_w   + row * Din_p;
        const T*        usr = up_s   + row * G;
        const T*        ubr = up_b   + row * G;
        float g = 0.0f, u = 0.0f;
        for (uint32_t p = lane; p < Din_p; p += 32u) {
            uint32_t base = p * vpw;
            uint32_t grp  = base / group_size;
            float gs = (float)gsr[grp], gb = (float)gbr[grp];
            float us = (float)usr[grp], ub = (float)ubr[grp];
            uint32_t gpk = gwr[p];
            uint32_t upk = uwr[p];
            for (uint32_t j = 0; j < vpw; ++j) {
                float xv = (float)x[base + j];
                g += xv * ((float)((gpk >> (j * bits)) & wmask) * gs + gb);
                u += xv * ((float)((upk >> (j * bits)) & wmask) * us + ub);
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, o);
            u += __shfl_down_sync(0xffffffffu, u, o);
        }
        if (lane == 0u) {
            if (act == 1) {
                // GeGLU (gelu tanh approx) * up (gemma4 experts).
                float g3 = g * g * g;
                float inner = 0.7978845608028654f * (g + 0.044715f * g3);
                float gelu = 0.5f * g * (1.0f + tanhf(inner));
                act_g[eslot * Dff + f] = gelu * u;
            } else {
                // SwiGLU (silu) * up. Precise expf (not __expf) to match the
                // gather_qmm silu and hold greedy parity; it runs once per row
                // on lane 0, so the cost is negligible against the GEMV.
                act_g[eslot * Dff + f] = (g / (1.0f + expf(-g))) * u;
            }
        }
    )";

    static const char* MOE_DOWN_CUDA_SOURCE = R"(
        uint32_t lane  = threadIdx.x;                          // 0..31 (one warp)
        uint32_t h     = blockIdx.y * blockDim.y + threadIdx.y; // output row 0..Din-1
        uint32_t eslot = blockIdx.z;                           // 0..K-1
        if (h >= (uint32_t)Din) return;                        // warp-uniform
        uint32_t e = indices[eslot];

        constexpr uint32_t Gd = Dff / group_size;
        uint32_t row = e * Din + h;
        const T*     dsr = down_s + row * Gd;
        const T*     dbr = down_b + row * Gd;
        const float* a   = act_g  + eslot * Dff;
        float d = 0.0f;

        if (bits == 6) {
            // 6-bit: MLX packs 4 weights into 3 bytes; read the row as bytes.
            // Layout matches quantized.h qdot: v0=b0&0x3f,
            // v1=(b0>>6)|((b1&0x0f)<<2), v2=(b1>>4)|((b2&0x03)<<4), v3=b2>>2.
            constexpr uint32_t packs   = Dff / 4u;
            constexpr uint32_t row_u32 = Dff * 3u / 16u;
            const uint8_t* wb = reinterpret_cast<const uint8_t*>(down_w + row * row_u32);
            for (uint32_t p = lane; p < packs; p += 32u) {
                uint32_t base = p * 4u;
                uint32_t grp  = base / group_size;
                float ds = (float)dsr[grp], db = (float)dbr[grp];
                uint32_t b0 = wb[p * 3u], b1 = wb[p * 3u + 1u], b2 = wb[p * 3u + 2u];
                uint32_t v0 = b0 & 0x3fu;
                uint32_t v1 = (b0 >> 6) | ((b1 & 0x0fu) << 2);
                uint32_t v2 = (b1 >> 4) | ((b2 & 0x03u) << 4);
                uint32_t v3 = (b2 >> 2);
                d += a[base + 0u] * ((float)v0 * ds + db);
                d += a[base + 1u] * ((float)v1 * ds + db);
                d += a[base + 2u] * ((float)v2 * ds + db);
                d += a[base + 3u] * ((float)v3 * ds + db);
            }
        } else {
            // Power-of-2 bits (4/8): vpw weights per 32-bit pack.
            constexpr uint32_t vpw   = 32u / bits;
            constexpr uint32_t wmask = (1u << bits) - 1u;
            constexpr uint32_t Dff_p = Dff / vpw;
            const uint32_t* dwr = down_w + row * Dff_p;
            for (uint32_t p = lane; p < Dff_p; p += 32u) {
                uint32_t base = p * vpw;
                uint32_t grp  = base / group_size;
                float ds = (float)dsr[grp], db = (float)dbr[grp];
                uint32_t dpk = dwr[p];
                for (uint32_t j = 0; j < vpw; ++j) {
                    d += a[base + j] * ((float)((dpk >> (j * bits)) & wmask) * ds + db);
                }
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_down_sync(0xffffffffu, d, o);
        if (lane == 0u) {
            out[eslot * Din + h] = (T)((float)scores[eslot] * d);
        }
    )";

    struct MoeGateUpKernelHolderCuda {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::cuda_kernel(
                    "moe_gateup_kernel_cu",
                    {"x", "indices", "gate_w", "gate_s", "gate_b",
                     "up_w", "up_s", "up_b"},
                    {"act_g"},
                    MOE_GATEUP_CUDA_SOURCE
                );
                initialized = true;
            }
            return *kernel;
        }
    };
    static MoeGateUpKernelHolderCuda& get_moe_gateup_kernel_cuda() {
        static MoeGateUpKernelHolderCuda holder;
        return holder;
    }

    struct MoeDownKernelHolderCuda {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::cuda_kernel(
                    "moe_down_kernel_cu",
                    {"indices", "down_w", "down_s", "down_b", "act_g", "scores"},
                    {"out"},
                    MOE_DOWN_CUDA_SOURCE
                );
                initialized = true;
            }
            return *kernel;
        }
    };
    static MoeDownKernelHolderCuda& get_moe_down_kernel_cuda() {
        static MoeDownKernelHolderCuda holder;
        return holder;
    }

    // fc1 + relu² -> act_g[K, Dff] for the squared-ReLU MoE (nemotron-h: the
    // experts are fc1 -> relu² -> fc2, not SwiGLU). One simdgroup per output
    // row, simd_sum over the contraction dim; relu² = square(max(x, 0)) to
    // match compiled_relu_squared. The fc2 down-projection reuses
    // moe_down_kernel unchanged. Power-of-2 bits only (4/8). Wired only behind
    // MLXCEL_FUSED_MOE_RELU2 (see fused_moe_forward) — correct but measured
    // performance-neutral on nemotron-h, kept for a future MoE-dominated
    // squared-ReLU model.
    static const char* MOE_FC1_RELU2_METAL_SOURCE = R"(
        uint lane  = thread_position_in_threadgroup.x;   // 0..31 (one simdgroup)
        uint f     = thread_position_in_grid.y;          // output row 0..Dff-1
        uint eslot = thread_position_in_grid.z;          // 0..K-1
        if (f >= Dff) return;                            // uniform across the simdgroup
        uint e = indices[eslot];

        constexpr uint vpw   = 32u / bits;
        constexpr uint wmask = (1u << bits) - 1u;
        constexpr uint Din_p = Din / vpw;
        constexpr uint G     = Din / group_size;

        uint row = e * Dff + f;
        const device uint* wr = fc1_w + row * Din_p;
        const device T*    sr = fc1_s + row * G;
        const device T*    br = fc1_b + row * G;
        float acc = 0.0f;
        for (uint p = lane; p < Din_p; p += 32u) {
            uint base = p * vpw;
            uint grp  = base / group_size;
            float s = (float)sr[grp], b = (float)br[grp];
            uint pk = wr[p];
            for (uint j = 0; j < vpw; ++j) {
                acc += (float)x[base + j] * ((float)((pk >> (j * bits)) & wmask) * s + b);
            }
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            float r = acc > 0.0f ? acc : 0.0f;   // relu
            act_g[eslot * Dff + f] = r * r;       // ^2
        }
    )";

    struct MoeFc1Relu2KernelHolder {
        std::optional<mlx::core::fast::CustomKernelFunction> kernel;
        bool initialized = false;
        mlx::core::fast::CustomKernelFunction& get() {
            if (!initialized) {
                kernel = mlx::core::fast::metal_kernel(
                    "moe_fc1_relu2_kernel",
                    {"x", "indices", "fc1_w", "fc1_s", "fc1_b"},
                    {"act_g"},
                    MOE_FC1_RELU2_METAL_SOURCE
                );
                initialized = true;
            }
            return *kernel;
        }
    };
    static MoeFc1Relu2KernelHolder& get_moe_fc1_relu2_kernel() {
        static MoeFc1Relu2KernelHolder holder;
        return holder;
    }

    // Definitions for the forward declarations above fused_moe_forward.
    mlx::core::fast::CustomKernelFunction& moe_fc1_relu2_kernel_fn() {
        return get_moe_fc1_relu2_kernel().get();
    }
    mlx::core::fast::CustomKernelFunction& moe_down_kernel_fn() {
        return get_moe_down_kernel().get();
    }
}

namespace {
// Shared body for the two fused decode-MoE FFIs. `act` selects the gate/up
// activation: 0 = SwiGLU (silu), 1 = GeGLU (gelu tanh approx, gemma4). gate/up
// use `gu_bits` (4/8), down uses `d_bits` (4/6/8); group_size shared.
std::unique_ptr<MlxArray> run_fused_moe_two_kernel(
    const MlxArray& x, const MlxArray& indices,
    const MlxArray& gate_w, const MlxArray& gate_s, const MlxArray& gate_b,
    const MlxArray& up_w,   const MlxArray& up_s,   const MlxArray& up_b,
    const MlxArray& down_w, const MlxArray& down_s, const MlxArray& down_b,
    const MlxArray& scores,
    int32_t din, int32_t dff, int32_t k,
    int32_t gu_bits, int32_t d_bits, int32_t group_size, int act
) {
    using namespace mlx::core;
    auto T = x.inner.dtype();

    // SGY = simdgroups per threadgroup (one per output row). 8 is the measured
    // sweet spot on qwen3-30b-a3b (M1 Ultra); env-overridable for other
    // hardware. It feeds an MLX template arg, which JIT-specializes at runtime.
    int sgy = 8;
    if (const char* s = std::getenv("MLXCEL_FUSED_MOE_SGY")) {
        int v = std::atoi(s);
        if (v >= 1 && v <= 32) sgy = v;
    }
    auto round_up = [](int n, int m) { return ((n + m - 1) / m) * m; };
    // gate/up and down can carry different bit widths (e.g. dots.llm1: gate/up
    // 4-bit, down 6-bit), so each kernel gets its own `bits` template arg.
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> taA = {
        {"T", T}, {"K", k}, {"Din", din}, {"Dff", dff},
        {"bits", gu_bits}, {"group_size", group_size}, {"act", act},
    };
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> taB = {
        {"T", T}, {"K", k}, {"Din", din}, {"Dff", dff},
        {"bits", d_bits}, {"group_size", group_size},
    };

    // mx.fast.metal_kernel throws on CUDA ("No Metal back-end"), so dispatch the
    // cuda_kernel port there. metal::is_available() is false on a CUDA-only build.
    const bool use_cuda = !mlx::core::metal::is_available();

    // A) gate/up + activation -> act_g[K, Dff] (f32 for the down GEMV).
    auto& kA = use_cuda ? get_moe_gateup_kernel_cuda().get()
                        : get_moe_gateup_kernel().get();
    std::vector<array> inA = {
        astype(x.inner, T), astype(indices.inner, uint32),
        gate_w.inner, astype(gate_s.inner, T), astype(gate_b.inner, T),
        up_w.inner,   astype(up_s.inner, T),   astype(up_b.inner, T),
    };
    auto rA = kA(
        inA, { Shape{k, dff} }, { float32 },
        std::make_tuple(32, round_up(dff, sgy), k),
        std::make_tuple(32, sgy, 1),
        taA, std::nullopt, false, {}
    );

    // B) down * score -> partial[K, Din]; summed over K into [Din].
    auto& kB = use_cuda ? get_moe_down_kernel_cuda().get()
                        : get_moe_down_kernel().get();
    std::vector<array> inB = {
        astype(indices.inner, uint32),
        down_w.inner, astype(down_s.inner, T), astype(down_b.inner, T),
        rA[0], astype(scores.inner, T),
    };
    auto rB = kB(
        inB, { Shape{k, din} }, { T },
        std::make_tuple(32, round_up(din, sgy), k),
        std::make_tuple(32, sgy, 1),
        taB, std::nullopt, false, {}
    );
    auto summed = sum(rB[0], /*axis=*/0, /*keepdims=*/false);
    return std::make_unique<MlxArray>(std::move(summed));
}
}  // namespace

// SwiGLU experts (qwen3_moe, dots.llm1, qwen3_next, ...).
std::unique_ptr<MlxArray> fused_moe_expert_kernel(
    const MlxArray& x,
    const MlxArray& indices,
    const MlxArray& gate_w, const MlxArray& gate_s, const MlxArray& gate_b,
    const MlxArray& up_w,   const MlxArray& up_s,   const MlxArray& up_b,
    const MlxArray& down_w, const MlxArray& down_s, const MlxArray& down_b,
    const MlxArray& scores,
    int32_t din, int32_t dff, int32_t k,
    int32_t gu_bits, int32_t d_bits, int32_t group_size
) {
    return run_fused_moe_two_kernel(
        x, indices, gate_w, gate_s, gate_b, up_w, up_s, up_b,
        down_w, down_s, down_b, scores, din, dff, k, gu_bits, d_bits,
        group_size, /*act=*/0);
}

// GeGLU experts (gemma4): gelu-tanh-approx(gate) * up.
std::unique_ptr<MlxArray> fused_moe_geglu_kernel(
    const MlxArray& x,
    const MlxArray& indices,
    const MlxArray& gate_w, const MlxArray& gate_s, const MlxArray& gate_b,
    const MlxArray& up_w,   const MlxArray& up_s,   const MlxArray& up_b,
    const MlxArray& down_w, const MlxArray& down_s, const MlxArray& down_b,
    const MlxArray& scores,
    int32_t din, int32_t dff, int32_t k,
    int32_t gu_bits, int32_t d_bits, int32_t group_size
) {
    return run_fused_moe_two_kernel(
        x, indices, gate_w, gate_s, gate_b, up_w, up_s, up_b,
        down_w, down_s, down_b, scores, din, dff, k, gu_bits, d_bits,
        group_size, /*act=*/1);
}

// Fused Mamba2 mixer forward for single-token decode.
// Combines in_proj + conv1d + SSM kernel + MambaRMSNormGated + out_proj into one C++ call.
// Used by: NemotronH
void fused_mamba2_forward(
    const MlxArray& hidden_states,
    const MlxArray& in_proj_weight,
    const MlxArray& in_proj_scales,
    const MlxArray* in_proj_biases,
    const MlxArray& conv_weight,
    const MlxArray* conv_bias,
    const MlxArray& A_log,
    const MlxArray& D,
    const MlxArray& dt_bias,
    const MlxArray& norm_weight,
    const MlxArray& out_proj_weight,
    const MlxArray& out_proj_scales,
    const MlxArray* out_proj_biases,
    const MlxArray& conv_state_in,
    const MlxArray& ssm_state_in,
    int32_t intermediate_size,
    int32_t conv_dim,
    int32_t conv_kernel_size,
    int32_t num_heads,
    int32_t head_dim,
    int32_t n_groups,
    int32_t ssm_state_size,
    float time_step_min,
    float time_step_max,
    float norm_eps,
    int32_t group_size,
    int32_t bits,
    std::unique_ptr<MlxArray>& output,
    std::unique_ptr<MlxArray>& conv_state_out,
    std::unique_ptr<MlxArray>& ssm_state_out
) {
    using namespace mlx::core;

    // --- Shape extraction ---
    auto hs_shape = hidden_states.inner.shape();
    int batch    = (int)hs_shape[0];
    int seq_len  = (int)hs_shape[1];   // always 1 for decode
    int hidden   = (int)hs_shape[2];
    int bc_size  = n_groups * ssm_state_size;
    int proj_cols = intermediate_size + conv_dim + num_heads;

    // --- Step 1: in_proj (quantized matmul, affine mode) ---
    // Flatten to [batch*seq_len, hidden] for matmul
    auto x_flat = reshape(hidden_states.inner, {batch * seq_len, hidden});
    // Fast path: omit mode for affine, pass biases directly when present
    auto projected_flat = in_proj_biases
        ? quantized_matmul(x_flat, in_proj_weight.inner, in_proj_scales.inner,
                          in_proj_biases->inner, true, group_size, bits)
        : quantized_matmul(x_flat, in_proj_weight.inner, in_proj_scales.inner,
                          std::nullopt, true, group_size, bits);
    // Restore to [batch, seq_len, proj_cols]
    auto projected = reshape(projected_flat, {batch, seq_len, proj_cols});

    // Dtype trace (temporary)
    if (std::getenv("MLXCEL_TRACE_DTYPE")) {
        mlx::core::eval(projected);
        std::cerr << "[C++ DTYPE] after in_proj: " << projected.dtype() << std::endl;
    }
    // --- Step 2: Split projected into gate, conv_input, dt ---
    // gate:       [batch, seq_len, intermediate_size]
    // conv_input: [batch, seq_len, conv_dim]
    // dt:         [batch, seq_len, num_heads]
    auto gate       = slice(projected,
                            {0, 0, 0},
                            {batch, seq_len, intermediate_size});
    auto conv_input = slice(projected,
                            {0, 0, intermediate_size},
                            {batch, seq_len, intermediate_size + conv_dim});
    auto dt_raw     = slice(projected,
                            {0, 0, intermediate_size + conv_dim},
                            {batch, seq_len, proj_cols});

    // --- Step 3: Depthwise conv1d with sliding state ---
    // Concatenate conv_state_in [batch, k-1, conv_dim] with conv_input [batch, 1, conv_dim]
    // -> padded_input [batch, k, conv_dim]  (k = conv_kernel_size)
    auto padded_input = concatenate(std::vector<array>{conv_state_in.inner, conv_input}, 1);

    // New conv state: last (k-1) elements of padded_input on axis 1
    int padded_len      = conv_kernel_size - 1 + seq_len;
    int new_state_start = padded_len - (conv_kernel_size - 1);
    auto new_conv_state = slice(padded_input,
                                {0, new_state_start, 0},
                                {batch, padded_len, conv_dim});

    // Depthwise conv1d: stride=1, padding=0, dilation=1, groups=conv_dim
    auto conv_out = mlx::core::conv1d(
        padded_input, conv_weight.inner,
        /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/conv_dim);

    // Optional bias: reshape [conv_dim] -> [1, 1, conv_dim] for broadcast
    if (conv_bias) {
        auto bias_r = reshape(conv_bias->inner, {1, 1, conv_dim});
        conv_out = add(conv_out, bias_r);
    }

    // SiLU: x * sigmoid(x) — compiled kernel fusion
    MlxArray co_w{conv_out};
    auto conv_output = compiled_silu(co_w)->inner;

    // --- Step 4: Split conv_output into hidden_ssm, B, C ---
    // conv_output: [batch, seq_len, conv_dim]
    // hidden_ssm:  [batch, seq_len, intermediate_size]
    // B, C:        [batch, seq_len, n_groups * ssm_state_size] each
    auto hidden_ssm = slice(conv_output,
                            {0, 0, 0},
                            {batch, seq_len, intermediate_size});
    auto B          = slice(conv_output,
                            {0, 0, intermediate_size},
                            {batch, seq_len, intermediate_size + bc_size});
    auto C          = slice(conv_output,
                            {0, 0, intermediate_size + bc_size},
                            {batch, seq_len, conv_dim});

    // --- Step 5: Reshape inputs for SSM kernel ---
    // x:  [batch, seq_len, num_heads, head_dim]
    // B:  [batch, seq_len, n_groups, ssm_state_size]
    // C:  [batch, seq_len, n_groups, ssm_state_size]
    // dt: [batch, seq_len, num_heads]   (already the right last dim)
    auto x_ssm = reshape(hidden_ssm, {batch, seq_len, num_heads, head_dim});
    auto B_r    = reshape(B,         {batch, seq_len, n_groups,  ssm_state_size});
    auto C_r    = reshape(C,         {batch, seq_len, n_groups,  ssm_state_size});

    // --- Step 6: Fused SSM Metal kernel ---
    // Wrap plain arrays in MlxArray for the kernel call
    MlxArray x_ssm_w{x_ssm};
    MlxArray B_w{B_r};
    MlxArray C_w{C_r};
    MlxArray dt_w{dt_raw};

    std::unique_ptr<MlxArray> ssm_y;
    std::unique_ptr<MlxArray> new_ssm_state;

    // ssm_update_kernel is defined above in this translation unit
    ssm_update_kernel(
        x_ssm_w, A_log, B_w, C_w, D, dt_w, dt_bias,
        ssm_state_in,
        time_step_min, time_step_max,
        ssm_y, new_ssm_state);

    // ssm_y: [batch, 1, num_heads, head_dim] -> [batch, seq_len, intermediate_size]
    if (std::getenv("MLXCEL_TRACE_DTYPE")) {
        mlx::core::eval(ssm_y->inner);
        std::cerr << "[C++ DTYPE] ssm_y: " << ssm_y->inner.dtype() << std::endl;
    }
    auto y = reshape(ssm_y->inner, {batch, seq_len, intermediate_size});

    // --- Step 7: MambaRMSNormGated ---
    // 7a. Gate: y_gated = y * silu(gate)
    // gate_activated = silu(gate), then y_gated = y * gate_activated
    MlxArray gate_w{gate};
    auto y_gated = multiply(y, compiled_silu(gate_w)->inner);

    // 7b. Grouped RMS norm: reshape last dim into [batch, seq_len, n_groups, group_size]
    //     where group_size = intermediate_size / n_groups (NOT head_dim!)
    int norm_group_size = intermediate_size / n_groups;
    auto y_grouped = reshape(y_gated, {batch, seq_len, n_groups, norm_group_size});
    auto unit_weight = mlx::core::ones({norm_group_size}, hidden_states.inner.dtype());
    auto y_normed_grouped = mlx::core::fast::rms_norm(y_grouped, unit_weight, norm_eps);

    // 7c. Flatten back and apply learned norm weight
    auto y_normed_flat = reshape(y_normed_grouped, {batch, seq_len, intermediate_size});
    auto y_normed = multiply(norm_weight.inner, y_normed_flat);

    // --- Step 8: out_proj (quantized matmul, affine mode) ---
    auto y_proj_flat = reshape(y_normed, {batch * seq_len, intermediate_size});
    auto out_flat = out_proj_biases
        ? quantized_matmul(y_proj_flat, out_proj_weight.inner, out_proj_scales.inner,
                          out_proj_biases->inner, true, group_size, bits)
        : quantized_matmul(y_proj_flat, out_proj_weight.inner, out_proj_scales.inner,
                          std::nullopt, true, group_size, bits);
    auto out = reshape(out_flat, {batch, seq_len, hidden});
    if (std::getenv("MLXCEL_TRACE_DTYPE")) {
        mlx::core::eval(out);
        std::cerr << "[C++ DTYPE] final out: " << out.dtype() << std::endl;
    }

    // --- Write outputs ---
    output         = std::make_unique<MlxArray>(std::move(out));
    conv_state_out = std::make_unique<MlxArray>(std::move(new_conv_state));
    ssm_state_out  = std::move(new_ssm_state);
}

}  // namespace mlx_cxx
