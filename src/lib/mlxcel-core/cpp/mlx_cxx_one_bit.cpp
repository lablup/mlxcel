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

// Affine 1-bit quantized matmul and dequantize (issue #1370).
//
// MLX implements affine quantized kernels for 2, 3, 4, 5, 6 and 8 bits only, so
// a checkpoint exported with `quantization.bits == 1` (the prism-ml Bonsai
// family) reaches `quantized_matmul` with a width MLX has no kernel for and the
// first forward pass aborts. The layout is the standard MLX packing at one bit
// per weight:
//
//   weight  uint32 [N, K / 32]   bit j (LSB first) of word c is column 32c + j
//   scales  float  [N, K / G]
//   biases  float  [N, K / G]
//   w[o, i] = bit(o, i) * scales[o, i / G] + biases[o, i / G]
//
// With G a multiple of 32 no packed word straddles a group, so a kernel can
// read one scale / bias pair per word.
//
// Three implementations live here:
//
//   - `one_bit_dequantize_impl`: a plain MLX graph. It is the reference the
//     tests compare against, the path every backend without a kernel port
//     takes, and what `MLXCEL_ONE_BIT_KERNEL=0` forces.
//   - qmv (Metal): small-M matvec. Per (row, 16-column block) it forms the
//     masked sum of activations and their plain sum, so the bias term factors
//     out of the bit mask: y = sum_g scale * masked_sum + bias * total_sum.
//   - qmm (Metal): M >= 16 tiles, 32x32x32 with simdgroup 8x8 fragments; each
//     tile expands its packed bits into `bias + (bit ? scale : 0)` in
//     threadgroup memory and accumulates in f32.
//
// There is no CUDA or ROCm port yet. Those backends take the dequantize graph,
// which is correct and slower; see `one_bit_matmul_impl`.

#include "mlx_cxx_internal.h"
#include "../../mlx-cpp/turbo/kernel_port.h"

#include <optional>
#include <sstream>

namespace mlx_cxx {

namespace {

// Small-M matvec. Threadgroup = 64 threads = 2 simdgroups; each simdgroup owns
// R consecutive output rows. Each lane owns 16 consecutive columns per step
// (block_start = lane * 16, stride 512 = 32 lanes * 16), loads those
// activations once, then for each of its rows reads the matching 16-bit
// half-word of the packed row.
//
// Template args: T (activation / output dtype), G (group size), R (rows per
// simdgroup), K (input width), N (output rows), ALIGNED (N % (2R) == 0, which
// drops the row bounds check).
static const char* ONE_BIT_QMV_METAL_SOURCE = R"(
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint m = threadgroup_position_in_grid.y;
    const uint row0 = (threadgroup_position_in_grid.x * 2u + sg) * (uint)R;
    constexpr uint WORDS = (uint)K / 32u;
    constexpr uint GROUPS = (uint)K / (uint)G;

    float acc[R];
    for (int r = 0; r < R; r++) {
        acc[r] = 0.0f;
    }

    const uint xbase = m * (uint)K;
    for (uint bs = lane * 16u; bs < (uint)K; bs += 512u) {
        float xv[16];
        float total = 0.0f;
        for (int i = 0; i < 16; i++) {
            xv[i] = (float)x[xbase + bs + i];
            total += xv[i];
        }
        const uint word_idx = bs >> 5;
        const uint shift = bs & 31u;
        const uint g = bs / (uint)G;
        for (int r = 0; r < R; r++) {
            const uint o = row0 + (uint)r;
            if (!ALIGNED && o >= (uint)N) {
                break;
            }
            const uint half_word = (w[o * WORDS + word_idx] >> shift) & 0xFFFFu;
            float selected = 0.0f;
            for (int i = 0; i < 16; i++) {
                selected += ((half_word >> i) & 1u) ? xv[i] : 0.0f;
            }
            acc[r] += selected * (float)scales[o * GROUPS + g]
                    + total * (float)biases[o * GROUPS + g];
        }
    }

    for (int r = 0; r < R; r++) {
        acc[r] = simd_sum(acc[r]);
    }
    if (lane == 0u) {
        for (int r = 0; r < R; r++) {
            const uint o = row0 + (uint)r;
            if (ALIGNED || o < (uint)N) {
                out[m * (uint)N + o] = (T)acc[r];
            }
        }
    }
)";

// M >= 16 tiled matmul. 32x32 output tile per threadgroup of 128 threads (4
// simdgroups in a 2x2 layout, each owning a 16x16 quadrant as 2x2 8x8
// fragments), K stepped 32 columns at a time, which is exactly one packed word
// per weight row. `m_size` is a scalar input rather than a template arg so a
// new prompt length does not compile a new kernel.
//
// Template args: T, G, K, N as for qmv.
static const char* ONE_BIT_QMM_METAL_SOURCE = R"(
    threadgroup float Xs[32 * 32];
    threadgroup float Ws[32 * 32];

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = threadgroup_position_in_grid.x * 32u;
    const uint m0 = threadgroup_position_in_grid.y * 32u;
    const uint M = (uint)m_size;
    constexpr uint WORDS = (uint)K / 32u;
    constexpr uint GROUPS = (uint)K / (uint)G;

    const uint sm = (sg / 2u) * 16u;
    const uint sn = (sg % 2u) * 16u;

    simdgroup_matrix<float, 8, 8> c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Load mapping: 128 threads cover 32 rows x 4 column slices of 8.
    const uint lr = tid / 4u;
    const uint lc = (tid % 4u) * 8u;
    const uint gm = m0 + lr;
    const uint gn = n0 + lr;

    for (uint k0 = 0; k0 < (uint)K; k0 += 32u) {
        if (gm < M) {
            for (uint i = 0; i < 8u; i++) {
                Xs[lr * 32u + lc + i] = (float)x[gm * (uint)K + k0 + lc + i];
            }
        } else {
            for (uint i = 0; i < 8u; i++) {
                Xs[lr * 32u + lc + i] = 0.0f;
            }
        }
        if (gn < (uint)N) {
            const uint word = w[gn * WORDS + (k0 >> 5)];
            const uint g = k0 / (uint)G;
            const float s = (float)scales[gn * GROUPS + g];
            const float b = (float)biases[gn * GROUPS + g];
            const float sb = s + b;
            for (uint i = 0; i < 8u; i++) {
                Ws[lr * 32u + lc + i] = ((word >> (lc + i)) & 1u) ? sb : b;
            }
        } else {
            for (uint i = 0; i < 8u; i++) {
                Ws[lr * 32u + lc + i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<float, 8, 8> a0;
            simdgroup_matrix<float, 8, 8> a1;
            simdgroup_matrix<float, 8, 8> b0;
            simdgroup_matrix<float, 8, 8> b1;
            simdgroup_load(a0, Xs + (sm + 0u) * 32u + kk, 32);
            simdgroup_load(a1, Xs + (sm + 8u) * 32u + kk, 32);
            // Ws is [n][k]; the transposed load yields the [k][n] operand.
            simdgroup_load(b0, Ws + (sn + 0u) * 32u + kk, 32, ulong2(0, 0), true);
            simdgroup_load(b1, Ws + (sn + 8u) * 32u + kk, 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(c00, Xs + (sm + 0u) * 32u + sn + 0u, 32);
    simdgroup_store(c01, Xs + (sm + 0u) * 32u + sn + 8u, 32);
    simdgroup_store(c10, Xs + (sm + 8u) * 32u + sn + 0u, 32);
    simdgroup_store(c11, Xs + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gm < M) {
        for (uint i = 0; i < 8u; i++) {
            const uint col = n0 + lc + i;
            if (col < (uint)N) {
                out[gm * (uint)N + col] = (T)Xs[lr * 32u + lc + i];
            }
        }
    }
)";

struct OneBitQmvKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    mlx::core::fast::CustomKernelFunction& get() {
        if (!kernel) {
            kernel = mlx::core::fast::metal_kernel(
                "one_bit_qmv",
                {"x", "w", "scales", "biases"},
                {"out"},
                ONE_BIT_QMV_METAL_SOURCE);
        }
        return *kernel;
    }
};

struct OneBitQmmKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    mlx::core::fast::CustomKernelFunction& get() {
        if (!kernel) {
            kernel = mlx::core::fast::metal_kernel(
                "one_bit_qmm",
                {"x", "w", "scales", "biases", "m_size"},
                {"out"},
                ONE_BIT_QMM_METAL_SOURCE);
        }
        return *kernel;
    }
};

OneBitQmvKernelHolder& get_one_bit_qmv_kernel() {
    static OneBitQmvKernelHolder holder;
    return holder;
}

OneBitQmmKernelHolder& get_one_bit_qmm_kernel() {
    static OneBitQmmKernelHolder holder;
    return holder;
}

// Metal only. A null CUDA / ROCm entry is "no port": those backends take the
// dequantize graph, never a throw (see `one_bit_matmul_impl`).
const mlxcel::KernelPorts& one_bit_qmv_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_one_bit_qmv_kernel().get();
        },
    };
    return ports;
}

const mlxcel::KernelPorts& one_bit_qmm_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_one_bit_qmm_kernel().get();
        },
    };
    return ports;
}

// `MLXCEL_ONE_BIT_KERNEL=0` (or false / off / no) forces the dequantize graph
// for every 1-bit matmul the bridge routes. Read once: the decision must not
// change between two forward passes of one process.
bool one_bit_kernel_env_disabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("MLXCEL_ONE_BIT_KERNEL");
        if (v == nullptr) {
            return false;
        }
        std::string s(v);
        for (auto& c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return s == "0" || s == "false" || s == "off" || s == "no";
    }();
    return disabled;
}

bool is_float_dtype(Dtype d) {
    return d == float16 || d == bfloat16 || d == float32;
}

[[noreturn]] void one_bit_shape_error(const std::string& what) {
    throw std::invalid_argument("[one_bit] " + what);
}

// Validates the packed triple and returns K (the unpacked input width).
int validate_one_bit_triple(
    const array& w, const array& scales, const array& biases, int group_size) {
    if (group_size != 32 && group_size != 64 && group_size != 128) {
        std::ostringstream msg;
        msg << "group_size " << group_size << " is not one of 32, 64, 128";
        one_bit_shape_error(msg.str());
    }
    if (w.dtype() != uint32) {
        one_bit_shape_error("packed weight must be uint32");
    }
    if (w.ndim() < 2 || scales.ndim() != w.ndim() || biases.shape() != scales.shape()) {
        one_bit_shape_error("weight, scales and biases must share rank >= 2 and scales / biases shape");
    }
    if (!is_float_dtype(scales.dtype()) || !is_float_dtype(biases.dtype())) {
        one_bit_shape_error("scales and biases must be float16, bfloat16 or float32");
    }
    for (int i = 0; i + 1 < static_cast<int>(w.ndim()); ++i) {
        if (w.shape(i) != scales.shape(i)) {
            one_bit_shape_error("weight and scales disagree on a leading axis");
        }
    }
    const int64_t k = static_cast<int64_t>(w.shape(-1)) * 32;
    if (k != static_cast<int64_t>(scales.shape(-1)) * group_size || k > INT32_MAX) {
        std::ostringstream msg;
        msg << "packed width " << w.shape(-1) << " * 32 does not match scales width "
            << scales.shape(-1) << " * group_size " << group_size;
        one_bit_shape_error(msg.str());
    }
    return static_cast<int>(k);
}

array one_bit_qmv(const array& x2d, const array& w, const array& scales,
                  const array& biases, int group_size, int k, int n) {
    const int m = x2d.shape(0);
    const int rows = (n <= 64 || k >= 2 * n) ? 4 : 8;
    const int rows_per_tg = 2 * rows;
    const bool aligned = (n % rows_per_tg) == 0;
    const int tgs = (n + rows_per_tg - 1) / rows_per_tg;
    auto T = x2d.dtype();
    auto& kernel = mlxcel::select_kernel_port(
        "one_bit_qmv", "one_bit_dequantize graph", one_bit_qmv_ports());
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> ta = {
        {"T", T},
        {"G", group_size},
        {"R", rows},
        {"K", k},
        {"N", n},
        {"ALIGNED", aligned},
    };
    auto results = kernel(
        {x2d, w, scales, biases}, {Shape{m, n}}, {T},
        std::make_tuple(tgs * 64, m, 1),
        std::make_tuple(64, 1, 1),
        ta, std::nullopt, false, {});
    return results[0];
}

array one_bit_qmm(const array& x2d, const array& w, const array& scales,
                  const array& biases, int group_size, int k, int n) {
    const int m = x2d.shape(0);
    auto T = x2d.dtype();
    auto& kernel = mlxcel::select_kernel_port(
        "one_bit_qmm", "one_bit_dequantize graph", one_bit_qmm_ports());
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> ta = {
        {"T", T},
        {"G", group_size},
        {"K", k},
        {"N", n},
    };
    auto results = kernel(
        {x2d, w, scales, biases, array(m, int32)}, {Shape{m, n}}, {T},
        std::make_tuple(((n + 31) / 32) * 128, (m + 31) / 32, 1),
        std::make_tuple(128, 1, 1),
        ta, std::nullopt, false, {});
    return results[0];
}

// Row threshold at which the tiled kernel takes over from the matvec.
constexpr int ONE_BIT_QMM_MIN_ROWS = 16;

}  // namespace

bool is_one_bit_affine(int32_t bits, rust::Str mode) {
    return bits == 1 && mode.size() == 6 && std::memcmp(mode.data(), "affine", 6) == 0;
}

array one_bit_dequantize_impl(
    const array& w, const array& scales, const array& biases, int group_size) {
    const int k = validate_one_bit_triple(w, scales, biases, group_size);
    // Unpack at byte granularity: on a little-endian host byte b of word c
    // holds columns 32c + 8b .. 32c + 8b + 7, LSB first, so the uint8 view
    // enumerates the bits in column order with a quarter of the temporary a
    // uint32 shift would need.
    auto bytes = mlx::core::view(mlx::core::contiguous(w), uint8);
    auto shifts = mlx::core::arange(0, 8, uint8);
    auto bits = mlx::core::bitwise_and(
        mlx::core::right_shift(mlx::core::expand_dims(bytes, -1), shifts),
        array(static_cast<uint8_t>(1), uint8));
    // [..., N, K] -> [..., N, K / G, G] so scales / biases broadcast per group.
    Shape grouped(w.shape().begin(), w.shape().end() - 1);
    grouped.push_back(k / group_size);
    grouped.push_back(group_size);
    auto mask = mlx::core::not_equal(
        mlx::core::reshape(bits, grouped), array(static_cast<uint8_t>(0), uint8));
    // `bias + scale` is formed in f32 and rounded once to the shipped dtype, so
    // every dequantized weight is either round(scale + bias) or bias exactly.
    auto out_dtype = scales.dtype();
    auto s32 = mlx::core::astype(scales, float32);
    auto b32 = mlx::core::astype(biases, float32);
    auto on = mlx::core::expand_dims(mlx::core::astype(mlx::core::add(s32, b32), out_dtype), -1);
    auto off = mlx::core::expand_dims(mlx::core::astype(biases, out_dtype), -1);
    auto dense = mlx::core::where(mask, on, off);
    Shape flat(w.shape().begin(), w.shape().end() - 1);
    flat.push_back(k);
    return mlx::core::reshape(dense, flat);
}

array one_bit_matmul_impl(
    const array& x, const array& w, const array& scales, const array& biases,
    int group_size, bool transpose, bool force_fallback) {
    const int k = validate_one_bit_triple(w, scales, biases, group_size);
    auto T = x.dtype();
    const bool kernel_shape = transpose && w.ndim() == 2 && x.ndim() >= 1 &&
        x.shape(-1) == k && is_float_dtype(T) && x.size() > 0;
    const bool use_kernel = kernel_shape && !force_fallback &&
        !one_bit_kernel_env_disabled() &&
        mlxcel::has_kernel_port(one_bit_qmv_ports());
    if (!use_kernel) {
        auto dense = mlx::core::astype(one_bit_dequantize_impl(w, scales, biases, group_size), T);
        if (transpose) {
            return mlx::core::matmul(x, mlx::core::swapaxes(dense, -1, -2));
        }
        return mlx::core::matmul(x, dense);
    }

    const int n = w.shape(0);
    const int m = static_cast<int>(x.size() / k);
    auto x2d = mlx::core::reshape(x, {m, k});
    array y = (m >= ONE_BIT_QMM_MIN_ROWS && mlxcel::has_kernel_port(one_bit_qmm_ports()))
        ? one_bit_qmm(x2d, w, scales, biases, group_size, k, n)
        : one_bit_qmv(x2d, w, scales, biases, group_size, k, n);
    Shape out_shape(x.shape().begin(), x.shape().end() - 1);
    out_shape.push_back(n);
    return mlx::core::reshape(y, out_shape);
}

array one_bit_embedding_impl(
    const array& w, const array& scales, const array& biases,
    const array& indices, int group_size) {
    auto flat = mlx::core::reshape(indices, {-1});
    auto rows = one_bit_dequantize_impl(
        mlx::core::take(w, flat, 0),
        mlx::core::take(scales, flat, 0),
        mlx::core::take(biases, flat, 0),
        group_size);
    Shape out_shape = indices.shape();
    out_shape.push_back(rows.shape(-1));
    return mlx::core::reshape(rows, out_shape);
}

std::unique_ptr<MlxArray> one_bit_quantized_matmul(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    int32_t group_size,
    bool force_fallback
) {
    return std::make_unique<MlxArray>(one_bit_matmul_impl(
        x.inner, weight.inner, scales.inner, biases.inner, group_size, true, force_fallback));
}

std::unique_ptr<MlxArray> one_bit_dequantize(
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray& biases,
    int32_t group_size
) {
    return std::make_unique<MlxArray>(
        one_bit_dequantize_impl(weight.inner, scales.inner, biases.inner, group_size));
}

bool one_bit_kernel_available() {
    return mlxcel::has_kernel_port(one_bit_qmv_ports());
}

}  // namespace mlx_cxx
