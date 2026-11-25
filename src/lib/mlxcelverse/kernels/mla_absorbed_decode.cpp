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

#include "mla_absorbed_decode.h"

#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "kernel_port.h"

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Fused absorbed-MLA single-query decode attention (issue #3121). See the
// header for the contract; this file holds the Metal bodies and the launchers.
// The partial pass has two variants that share one body: a dense latent
// window, and an affine-quantized one read in-kernel (issue #3156).
//
// Partial pass. Threadgroup (b, h, half) of 32 simdgroups x 32 lanes. Lane `j`
// holds query elements `[j*DR, (j+1)*DR)` of the latent half and
// `[j*DP, (j+1)*DP)` of the rope half, already multiplied by the scale.
// Simdgroup `s` walks keys `start + s, start + s + 32, ...` of its half with
// an online softmax (running max `m`, running sum `l`, running output `o`),
// so one key costs one `simd_sum`. The 32 simdgroup states are then combined
// through threadgroup memory: the running maxima and sums in one exchange, the
// outputs one `DR` slot at a time, each simdgroup summing one column. The pass
// writes the half's output already divided by its own sum, and the half's
// log-sum-exp (natural log).
//
// Merge pass. One thread per (b, h, element): rescales the two halves by
// `exp(lse - max_lse)` and divides by the summed weights.
//
// Why two halves rather than one pass: with one threadgroup per head the
// machine sees only `B * H` threadgroups (16 for DeepSeek-V2-Lite at batch 1),
// and splitting the keys in two doubled the measured throughput of the
// partial pass at every context from 256 to 8192 keys, while splitting further
// lost again (M5 Max, issue #3121's benchmark record).
//
// The running maximum starts at the most negative finite float, not -inf, so
// `exp(m_old - m_new)` never evaluates `exp(-inf - -inf)`. An empty half (a
// short window whose second half has no keys) leaves `l == 0`, writes a zero
// output and the most negative finite lse, and gets zero weight in the merge.
namespace mlxcel::turbo {

namespace {

// The partial pass is assembled from fragments so the dense and the quantized
// kernels are one body with two window reads (issue #3156). A variant supplies
// `WINDOW` (per-batch-row bases), `ROW` (per-key bases and the rope row `kp`),
// `LOAD` (fill `kv[DR]` from this lane's slice of the latent row and add the
// latent score to `s`), `VALUE` (fold the key into `o`) and `FINISH` (after
// the loop). The rope half, the online softmax, the exchange and the merge are
// shared. The dense fragments concatenate to the pre-split text.
constexpr const char* MLA_DECODE_PARTIAL_HEAD = R"(
    constexpr int BN = 32;
    constexpr int DR = R / 32;
    constexpr int DP = P / 32;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int half_idx = threadgroup_position_in_grid.x;
    const int head = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int H = params[0];
    const int S = params[1];
    const int C = params[2];
    const float sc = scale_arr[0];

    threadgroup float tg_m[BN];
    threadgroup float tg_l[BN];
    threadgroup float tg_o[BN * BN];

    const size_t qrow = (size_t)b * H + head;
    const device T* qa = q_abs + qrow * R + lane * DR;
    const device T* qp = q_pe + qrow * P + lane * DP;
    float q_r[DR];
    float q_p[DP];
    float o[DR];
    for (int i = 0; i < DR; i++) { q_r[i] = sc * float(qa[i]); o[i] = 0.0f; }
    for (int i = 0; i < DP; i++) { q_p[i] = sc * float(qp[i]); }

    const float NEG = -3.4028234663852886e38f;
    float m = NEG;
    float l = 0.0f;
    const int t_end = min(S, (half_idx + 1) * C);
)";

constexpr const char* MLA_DECODE_LOOP_OPEN = R"(    for (int t = half_idx * C + sg; t < t_end; t += BN) {
)";

constexpr const char* MLA_DECODE_ROPE_SCORE = R"(        float kv[DR];
        float s = 0.0f;
        for (int i = 0; i < DP; i++) { s += q_p[i] * float(kp[i]); }
)";

constexpr const char* MLA_DECODE_SOFTMAX = R"(        s = simd_sum(s);
        const float m_new = max(m, s);
        const float a = fast::exp(m - m_new);
        const float p = fast::exp(s - m_new);
        l = l * a + p;
)";

constexpr const char* MLA_DECODE_LOOP_CLOSE = R"(        m = m_new;
    }
)";

constexpr const char* MLA_DECODE_EXCHANGE = R"(
    if (lane == 0) { tg_m[sg] = m; tg_l[sg] = l; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m_lane = tg_m[lane];
    const float M = simd_max(m_lane);
    const float L = simd_sum(tg_l[lane] * fast::exp(m_lane - M));
    const float w_self = fast::exp(m - M);
    const float inv_L = L > 0.0f ? 1.0f / L : 0.0f;

    const size_t prow = qrow * 2 + half_idx;
    device float* out_row = part + prow * R;
    for (int i = 0; i < DR; i++) {
        tg_o[sg * BN + lane] = o[i] * w_self;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Column `sg` of the exchange is output element `sg * DR + i` as held
        // by each of the 32 simdgroups; this simdgroup sums it.
        const float v = simd_sum(tg_o[lane * BN + sg]);
        if (lane == 0) { out_row[sg * DR + i] = v * inv_L; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0 && lane == 0) { lse[prow] = L > 0.0f ? M + log(L) : NEG; }
)";

// Dense latent: `ckv` holds the row in the queries' dtype.
constexpr const char* MLA_DECODE_DENSE_WINDOW = R"(    const device T* ckv_b = ckv + (size_t)b * ckv_strides[0] + lane * DR;
    const device T* kpe_b = kpe + (size_t)b * kpe_strides[0] + lane * DP;
)";

constexpr const char* MLA_DECODE_DENSE_ROW = R"(        const device T* kr = ckv_b + (size_t)t * ckv_strides[2];
        const device T* kp = kpe_b + (size_t)t * kpe_strides[2];
)";

constexpr const char* MLA_DECODE_DENSE_LOAD = R"(        for (int i = 0; i < DR; i++) { kv[i] = float(kr[i]); s += q_r[i] * kv[i]; }
)";

constexpr const char* MLA_DECODE_DENSE_VALUE = R"(        for (int i = 0; i < DR; i++) { o[i] = o[i] * a + p * kv[i]; }
)";

// Affine-quantized 8-bit latent (MLX `quantize` layout): `ckv_q` packs four
// codes per uint32, low byte first, so on this little-endian target byte `e`
// of a row is element `e`, which reads back as `code * scale + bias` with one
// pair per group `e / GS`. A lane's `DR` elements lie inside one group (the
// launcher requires `GS % DR == 0`), so the pair factors out of both
// contractions, the form MLX's own quantized matmul uses:
//
//     q . x          = scale * (q . code) + bias * sum(q)
//     sum_t p_t x_t  = sum_t (p_t scale_t) code_t + sum_t p_t bias_t
//
// `kv` holds raw codes, the value sum weights them by `p * scale`, and the
// bias term is one running scalar per lane, added once after the loop: the
// window is never dequantized and a key costs the dense kernel's work plus
// two scalars. `scales` and `biases` are indexed, not held as pointers,
// because MLX passes an array of fewer than eight elements in the `constant`
// address space. `kpe` has its own element type `TK`: the cache stores it
// fp16 whatever the queries are.
constexpr const char* MLA_DECODE_QUANT_WINDOW = R"(
    const device uchar* ckv_b =
        (const device uchar*)ckv_q + (size_t)b * ckv_q_strides[0] * 4 + lane * DR;
    const size_t scl_b = (size_t)b * scales_strides[0] + lane * DR / GS;
    const size_t bia_b = (size_t)b * biases_strides[0] + lane * DR / GS;
    const device TK* kpe_b = kpe + (size_t)b * kpe_strides[0] + lane * DP;
    float q_sum = 0.0f;
    for (int i = 0; i < DR; i++) { q_sum += q_r[i]; }
    float o_bias = 0.0f;
)";

constexpr const char* MLA_DECODE_QUANT_ROW = R"(
        const device uchar* kr = ckv_b + (size_t)t * ckv_q_strides[2] * 4;
        const float g_scale = float(scales[scl_b + (size_t)t * scales_strides[2]]);
        const float g_bias = float(biases[bia_b + (size_t)t * biases_strides[2]]);
        const device TK* kp = kpe_b + (size_t)t * kpe_strides[2];
)";

constexpr const char* MLA_DECODE_QUANT_LOAD = R"(
        float dot = 0.0f;
        for (int i = 0; i < DR; i++) { kv[i] = float(kr[i]); dot += q_r[i] * kv[i]; }
        s += g_scale * dot + g_bias * q_sum;
)";

constexpr const char* MLA_DECODE_QUANT_VALUE = R"(
        const float pg = p * g_scale;
        for (int i = 0; i < DR; i++) { o[i] = o[i] * a + pg * kv[i]; }
        o_bias = o_bias * a + p * g_bias;
)";

constexpr const char* MLA_DECODE_QUANT_FINISH = R"(
    for (int i = 0; i < DR; i++) { o[i] += o_bias; }
)";

struct PartialVariant {
    const char *window, *row, *load, *value, *finish;
};

constexpr PartialVariant DENSE_VARIANT{MLA_DECODE_DENSE_WINDOW, MLA_DECODE_DENSE_ROW,
    MLA_DECODE_DENSE_LOAD, MLA_DECODE_DENSE_VALUE, ""};
constexpr PartialVariant QUANT_VARIANT{MLA_DECODE_QUANT_WINDOW, MLA_DECODE_QUANT_ROW,
    MLA_DECODE_QUANT_LOAD, MLA_DECODE_QUANT_VALUE, MLA_DECODE_QUANT_FINISH};

std::string partial_source(const PartialVariant& v) {
    return std::string(MLA_DECODE_PARTIAL_HEAD) + v.window + MLA_DECODE_LOOP_OPEN + v.row +
        MLA_DECODE_ROPE_SCORE + v.load + MLA_DECODE_SOFTMAX + v.value + MLA_DECODE_LOOP_CLOSE +
        v.finish + MLA_DECODE_EXCHANGE;
}

constexpr const char* MLA_DECODE_MERGE_SOURCE = R"(
    const int d = thread_position_in_grid.x;
    const int head = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int H = params[0];
    const size_t prow = ((size_t)b * H + head) * 2;
    const float l0 = lse[prow];
    const float l1 = lse[prow + 1];
    const float M = max(l0, l1);
    const float w0 = fast::exp(l0 - M);
    const float w1 = fast::exp(l1 - M);
    const float acc = w0 * part[prow * R + d] + w1 * part[(prow + 1) * R + d];
    out[((size_t)b * H + head) * R + d] = T(acc / (w0 + w1));
)";

struct MlaDecodeKernels {
    std::optional<mlx::core::fast::CustomKernelFunction> partial;
    std::optional<mlx::core::fast::CustomKernelFunction> partial_quantized;
    std::optional<mlx::core::fast::CustomKernelFunction> merge;
    std::once_flag init_flag;

    void init() {
        std::call_once(init_flag, [this] {
            // Not row-contiguous: `ckv` and `kpe` are slices of the cache's
            // preallocated buffer, and a forced copy would re-read the whole
            // window every step. The body indexes them through their strides.
            partial = mlx::core::fast::metal_kernel(
                "mlxcel_mla_absorbed_decode_partial",
                {"q_abs", "q_pe", "ckv", "kpe", "scale_arr", "params"},
                {"part", "lse"},
                partial_source(DENSE_VARIANT),
                "",
                /* ensure_row_contiguous = */ false);
            partial_quantized = mlx::core::fast::metal_kernel(
                "mlxcel_mla_absorbed_decode_partial_quantized",
                {"q_abs", "q_pe", "ckv_q", "scales", "biases", "kpe", "scale_arr", "params"},
                {"part", "lse"},
                partial_source(QUANT_VARIANT),
                "",
                /* ensure_row_contiguous = */ false);
            merge = mlx::core::fast::metal_kernel(
                "mlxcel_mla_absorbed_decode_merge",
                {"part", "lse", "params"},
                {"out"},
                std::string(MLA_DECODE_MERGE_SOURCE));
        });
    }
};

MlaDecodeKernels& kernels() {
    static MlaDecodeKernels holder;
    return holder;
}

// The port table holds the partial pass; the merge exists where it does.
const mlxcel::KernelPorts& mla_absorbed_decode_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            kernels().init();
            return *kernels().partial;
        },
        // No CUDA or ROCm port: the composed path runs there.
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

// The quantized partial pass has its own table and shares the merge. No CUDA
// or ROCm port either: the composed quantized path runs there.
const mlxcel::KernelPorts& mla_absorbed_decode_quantized_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            kernels().init();
            return *kernels().partial_quantized;
        },
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("[mla_absorbed_decode] " + what);
}

bool is_kernel_float(mlx::core::Dtype dt) {
    return dt == mlx::core::float16 || dt == mlx::core::bfloat16 || dt == mlx::core::float32;
}

// What both variants require of the queries.
void check_queries(const mlx::core::array& q_abs, const mlx::core::array& q_pe) {
    if (q_abs.ndim() != 4 || q_pe.ndim() != 4) fail("every input must be 4-D");
    if (q_abs.shape(2) != 1 || q_pe.shape(2) != 1) fail("one query per head only");
    if (q_pe.shape(0) != q_abs.shape(0) || q_pe.shape(1) != q_abs.shape(1)) {
        fail("q_pe disagrees with q_abs");
    }
    const int R = q_abs.shape(3), P = q_pe.shape(3);
    if (R % 32 != 0 || P % 32 != 0 || R > 1024 || R == 0 || P == 0) {
        fail("R and P must be non-zero multiples of 32 with R <= 1024");
    }
}

// A window stream `[B, 1, S, width]` with a contiguous last axis.
void check_stream(const mlx::core::array& a, int B, int S, int width, const char* what) {
    if (a.ndim() != 4 || a.shape(0) != B || a.shape(1) != 1 || a.shape(2) != S ||
        a.shape(3) != width) {
        fail(std::string(what));
    }
    if (a.strides(3) != 1) fail("the cache rows must be contiguous");
}

void check_shapes(
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const mlx::core::array& ckv,
    const mlx::core::array& kpe) {
    check_queries(q_abs, q_pe);
    if (ckv.ndim() != 4 || ckv.shape(2) < 1) fail("the latent window is empty");
    const int B = q_abs.shape(0), S = ckv.shape(2);
    check_stream(ckv, B, S, q_abs.shape(3), "ckv must be [B, 1, S, R]");
    check_stream(kpe, B, S, q_pe.shape(3), "kpe must be [B, 1, S, P]");
    const auto dt = q_abs.dtype();
    if (q_pe.dtype() != dt || ckv.dtype() != dt || kpe.dtype() != dt) fail("inputs must share one dtype");
    if (!is_kernel_float(dt)) fail("dtype must be float16, bfloat16 or float32");
}

// The quantized variant's contract; the rank comes from the queries.
void check_shapes_quantized(
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const mlx::core::array& ckv_q,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    const mlx::core::array& kpe,
    int group_size,
    int bits) {
    // The width dispatch. 8 is the only width a cache mode stores (a 4-bit
    // latent failed the #3155 quality gate), so the rest is refused.
    switch (bits) {
    case 8: break;
    default: fail("only an 8-bit latent is supported, got " + std::to_string(bits) + " bits");
    }
    check_queries(q_abs, q_pe);
    const int B = q_abs.shape(0), R = q_abs.shape(3);
    if (group_size < 1 || R % group_size != 0 || group_size % (R / 32) != 0) {
        fail("R must be whole groups, with each lane's R / 32 elements inside one");
    }
    if (ckv_q.ndim() != 4 || ckv_q.shape(2) < 1) fail("the latent window is empty");
    const int S = ckv_q.shape(2), G = R / group_size;
    check_stream(ckv_q, B, S, R * bits / 32, "ckv_q must be [B, 1, S, R * bits / 32]");
    check_stream(scales, B, S, G, "scales must be [B, 1, S, R / group_size]");
    check_stream(biases, B, S, G, "biases must be [B, 1, S, R / group_size]");
    check_stream(kpe, B, S, q_pe.shape(3), "kpe must be [B, 1, S, P]");
    if (ckv_q.dtype() != mlx::core::uint32) fail("ckv_q must be uint32");
    if (q_pe.dtype() != q_abs.dtype()) fail("the queries must share one dtype");
    if (scales.dtype() != biases.dtype()) fail("scales and biases must share one dtype");
    for (const auto* a : {&q_abs, &scales, &kpe}) {
        if (!is_kernel_float(a->dtype())) fail("dtype must be float16, bfloat16 or float32");
    }
}

// Launch a partial pass and the merge. `window` is the variant's cache inputs
// in its kernel's declared order and `ta` its template arguments.
mlx::core::array launch(
    mlx::core::fast::CustomKernelFunction& partial,
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const std::vector<mlx::core::array>& window,
    int S,
    float scale,
    const std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>& ta) {
    using namespace mlx::core;
    const int B = q_abs.shape(0), H = q_abs.shape(1), R = q_abs.shape(3);
    const auto T = q_abs.dtype();
    // Keys per half, rounded up to whole simdgroup strides.
    const int C = ((S + 1) / 2 + 31) / 32 * 32;

    const array params = array({H, S, C}, Shape{3});
    // The queries are tiny and usually strided views (a transpose and a
    // slice); the body reads them row-major, so make them so here.
    std::vector<array> inputs = {contiguous(q_abs), contiguous(q_pe)};
    inputs.insert(inputs.end(), window.begin(), window.end());
    inputs.push_back(array({scale}, Shape{1}));
    inputs.push_back(params);

    auto partials = partial(
        inputs,
        {Shape{B * H * 2, R}, Shape{B * H * 2}},
        {float32, float32},
        std::make_tuple(2 * 1024, H, B),
        std::make_tuple(1024, 1, 1),
        ta,
        std::nullopt,
        false,
        {});

    auto& merge = *kernels().merge;
    auto merged = merge(
        {partials[0], partials[1], params},
        {Shape{B, H, 1, R}},
        {T},
        std::make_tuple(R, H, B),
        std::make_tuple(R, 1, 1),
        {{"T", T}, {"R", R}},
        std::nullopt,
        false,
        {});
    return merged[0];
}

// Whether `build`'s graph evaluates, remembered per `key` for the process.
//
// Metal compiles a pipeline with a thread limit derived from the kernel's
// register use, and MLX checks the launch against it only at eval (it throws
// before the dispatch is encoded, so the stream stays usable). The partial
// pass cannot be resized without changing its exchange, so the honest answer
// to "does it run here" is to run it: once, on a one-key window, at the exact
// template geometry the caller would launch. The lock is held across the
// probe so concurrent first calls do not compile and evaluate it twice.
bool probe_evaluates(const std::string& key, const std::function<mlx::core::array()>& build) {
    static std::mutex mutex;
    static std::map<std::string, bool> results;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = results.find(key); it != results.end()) {
        return it->second;
    }
    bool evaluates = false;
    try {
        auto out = build();
        mlx::core::eval(out);
        evaluates = true;
    } catch (const std::exception&) {
        evaluates = false;
    }
    results.emplace(key, evaluates);
    return evaluates;
}

std::string dtype_key(mlx::core::Dtype dt) {
    return std::to_string(static_cast<int>(dt.val())) + "/" + std::to_string(dt.size());
}

bool in_rank_contract(int R, int P) {
    return R > 0 && P > 0 && R % 32 == 0 && P % 32 == 0 && R <= 1024;
}

} // namespace

bool mla_absorbed_decode_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu &&
        mlxcel::has_kernel_port(mla_absorbed_decode_ports());
}

mlx::core::array mla_absorbed_decode(
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const mlx::core::array& ckv,
    const mlx::core::array& kpe,
    float scale) {
    auto& partial = mlxcel::select_kernel_port(
        "mla_absorbed_decode", "composed absorbed attention fallback", mla_absorbed_decode_ports());
    check_shapes(q_abs, q_pe, ckv, kpe);

    // R and P size the body's per-lane register arrays; T is the element type.
    return launch(
        partial,
        q_abs,
        q_pe,
        {ckv, kpe},
        ckv.shape(2),
        scale,
        {{"T", q_abs.dtype()}, {"R", q_abs.shape(3)}, {"P", q_pe.shape(3)}});
}

bool mla_absorbed_decode_supported(mlx::core::Dtype dtype, int R, int P) {
    using namespace mlx::core;
    if (!mla_absorbed_decode_available() || !is_kernel_float(dtype) || !in_rank_contract(R, P)) {
        return false;
    }
    const std::string key = "dense:" + dtype_key(dtype) + ":" + std::to_string(R) + ":" +
        std::to_string(P);
    return probe_evaluates(key, [&] {
        const array q_abs = zeros({1, 1, 1, R}, dtype);
        const array q_pe = zeros({1, 1, 1, P}, dtype);
        const array ckv = zeros({1, 1, 1, R}, dtype);
        const array kpe = zeros({1, 1, 1, P}, dtype);
        eval({q_abs, q_pe, ckv, kpe});
        return mla_absorbed_decode(q_abs, q_pe, ckv, kpe, 1.0f);
    });
}

bool mla_absorbed_decode_quantized_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu &&
        mlxcel::has_kernel_port(mla_absorbed_decode_quantized_ports());
}

bool mla_absorbed_decode_quantized_supported(
    mlx::core::Dtype dtype,
    mlx::core::Dtype kpe_dtype,
    mlx::core::Dtype scales_dtype,
    int R,
    int P,
    int group_size,
    int bits) {
    using namespace mlx::core;
    if (!mla_absorbed_decode_quantized_available() || !is_kernel_float(dtype) ||
        !is_kernel_float(kpe_dtype) || !is_kernel_float(scales_dtype) || !in_rank_contract(R, P) ||
        bits <= 0 || 32 % bits != 0 || group_size <= 0 || R % group_size != 0) {
        return false;
    }
    const std::string key = "quantized:" + dtype_key(dtype) + ":" + dtype_key(kpe_dtype) + ":" +
        dtype_key(scales_dtype) + ":" + std::to_string(R) + ":" + std::to_string(P) + ":" +
        std::to_string(group_size) + ":" + std::to_string(bits);
    return probe_evaluates(key, [&] {
        const array q_abs = zeros({1, 1, 1, R}, dtype);
        const array q_pe = zeros({1, 1, 1, P}, dtype);
        const array ckv_q = zeros({1, 1, 1, R * bits / 32}, uint32);
        const array scales = zeros({1, 1, 1, R / group_size}, scales_dtype);
        const array biases = zeros({1, 1, 1, R / group_size}, scales_dtype);
        const array kpe = zeros({1, 1, 1, P}, kpe_dtype);
        eval({q_abs, q_pe, ckv_q, scales, biases, kpe});
        return mla_absorbed_decode_quantized(
            q_abs, q_pe, ckv_q, scales, biases, kpe, 1.0f, group_size, bits);
    });
}

mlx::core::array mla_absorbed_decode_quantized(
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const mlx::core::array& ckv_q,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    const mlx::core::array& kpe,
    float scale,
    int group_size,
    int bits) {
    auto& partial = mlxcel::select_kernel_port(
        "mla_absorbed_decode_quantized",
        "composed quantized absorbed attention fallback",
        mla_absorbed_decode_quantized_ports());
    check_shapes_quantized(q_abs, q_pe, ckv_q, scales, biases, kpe, group_size, bits);
    // `TK` is the rope window's element type; the scale and bias dtype reaches
    // the kernel through its generated signature, and the body reads bytes.
    return launch(
        partial,
        q_abs,
        q_pe,
        {ckv_q, scales, biases, kpe},
        ckv_q.shape(2),
        scale,
        {{"T", q_abs.dtype()},
         {"TK", kpe.dtype()},
         {"R", q_abs.shape(3)},
         {"P", q_pe.shape(3)},
         {"GS", group_size}});
}

} // namespace mlxcel::turbo
