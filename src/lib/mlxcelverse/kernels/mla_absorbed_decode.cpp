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

#include "kernel_port.h"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Fused absorbed-MLA single-query decode attention (issue #3121). See the
// header for the contract; this file holds the two Metal bodies and the
// launcher.
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

constexpr const char* MLA_DECODE_PARTIAL_SOURCE = R"(
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
    const device T* ckv_b = ckv + (size_t)b * ckv_strides[0] + lane * DR;
    const device T* kpe_b = kpe + (size_t)b * kpe_strides[0] + lane * DP;
    for (int t = half_idx * C + sg; t < t_end; t += BN) {
        const device T* kr = ckv_b + (size_t)t * ckv_strides[2];
        const device T* kp = kpe_b + (size_t)t * kpe_strides[2];
        float kv[DR];
        float s = 0.0f;
        for (int i = 0; i < DP; i++) { s += q_p[i] * float(kp[i]); }
        for (int i = 0; i < DR; i++) { kv[i] = float(kr[i]); s += q_r[i] * kv[i]; }
        s = simd_sum(s);
        const float m_new = max(m, s);
        const float a = fast::exp(m - m_new);
        const float p = fast::exp(s - m_new);
        l = l * a + p;
        for (int i = 0; i < DR; i++) { o[i] = o[i] * a + p * kv[i]; }
        m = m_new;
    }

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
                std::string(MLA_DECODE_PARTIAL_SOURCE),
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

// The port table holds the partial pass; the merge is launched alongside it
// and only exists where the partial does.
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

void check_shapes(
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const mlx::core::array& ckv,
    const mlx::core::array& kpe) {
    auto fail = [](const std::string& what) {
        throw std::invalid_argument("[mla_absorbed_decode] " + what);
    };
    if (q_abs.ndim() != 4 || q_pe.ndim() != 4 || ckv.ndim() != 4 || kpe.ndim() != 4) {
        fail("every input must be 4-D");
    }
    const int B = q_abs.shape(0), H = q_abs.shape(1), R = q_abs.shape(3);
    const int P = q_pe.shape(3), S = ckv.shape(2);
    if (q_abs.shape(2) != 1 || q_pe.shape(2) != 1) fail("one query per head only");
    if (q_pe.shape(0) != B || q_pe.shape(1) != H) fail("q_pe disagrees with q_abs");
    if (ckv.shape(0) != B || ckv.shape(1) != 1 || ckv.shape(3) != R) fail("ckv must be [B, 1, S, R]");
    if (kpe.shape(0) != B || kpe.shape(1) != 1 || kpe.shape(2) != S || kpe.shape(3) != P) {
        fail("kpe must be [B, 1, S, P]");
    }
    if (R % 32 != 0 || P % 32 != 0 || R > 1024 || R == 0 || P == 0) {
        fail("R and P must be non-zero multiples of 32 with R <= 1024");
    }
    if (S < 1) fail("the latent window is empty");
    if (ckv.strides(3) != 1 || kpe.strides(3) != 1) fail("the cache rows must be contiguous");
    const auto dt = q_abs.dtype();
    if (q_pe.dtype() != dt || ckv.dtype() != dt || kpe.dtype() != dt) fail("inputs must share one dtype");
    if (dt != mlx::core::float16 && dt != mlx::core::bfloat16 && dt != mlx::core::float32) {
        fail("dtype must be float16, bfloat16 or float32");
    }
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
    using namespace mlx::core;
    auto& partial = mlxcel::select_kernel_port(
        "mla_absorbed_decode", "composed absorbed attention fallback", mla_absorbed_decode_ports());
    check_shapes(q_abs, q_pe, ckv, kpe);

    const int B = q_abs.shape(0), H = q_abs.shape(1), R = q_abs.shape(3);
    const int P = q_pe.shape(3), S = ckv.shape(2);
    const auto T = q_abs.dtype();
    // Keys per half, rounded up to whole simdgroup strides.
    const int C = ((S + 1) / 2 + 31) / 32 * 32;

    // The queries are tiny and usually strided views (a transpose and a
    // slice); the body reads them row-major, so make them so here.
    const array qa = contiguous(q_abs);
    const array qp = contiguous(q_pe);
    const array params = array({H, S, C}, Shape{3});
    const array scale_arr = array({scale}, Shape{1});

    // R and P are template arguments because the body sizes its per-lane
    // register arrays from them; T because the inputs' element type varies.
    std::vector<std::pair<std::string, fast::TemplateArg>> ta = {
        {"T", T},
        {"R", R},
        {"P", P},
    };
    auto partials = partial(
        {qa, qp, ckv, kpe, scale_arr, params},
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

} // namespace mlxcel::turbo
