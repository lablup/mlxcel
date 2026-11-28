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

#include "turbo_packed_attention.h"
#include "kernel_port.h"
#include <mlx/ops.h>
#include <stdexcept>

namespace mlxcel::turbo {
namespace {
using namespace mlx::core;

// One CTA per (batch, query head). Each warp reduces a disjoint stripe of
// tokens and the first warp combines their online-softmax states. The cache
// slices can retain the strides of a larger backing allocation.
fast::CustomKernelFunction& cuda_port() {
    static auto kernel = fast::cuda_kernel("mlxcel_turbo_packed_attention",
        {"q", "k", "kn", "vp", "vr", "hot", "cb", "scale"}, {"out"}, R"CUDA(
        const int lane = threadIdx.x, stripe = threadIdx.y;
        const int batch = blockIdx.z / q_shape[1], head = blockIdx.z % q_shape[1];
        const int kv_head = head / Rep, cold = vp_shape[2];
        const size_t q0 = batch*q_strides[0] + head*q_strides[1];
        const size_t k0 = batch*k_strides[0] + kv_head*k_strides[1];
        const size_t kn0 = batch*kn_strides[0] + kv_head*kn_strides[1];
        const size_t vp0 = batch*vp_strides[0] + kv_head*vp_strides[1];
        const size_t vr0 = batch*vr_strides[0] + kv_head*vr_strides[1];
        const size_t hot0 = batch*hot_strides[0] + kv_head*hot_strides[1];
        float maximum = -INFINITY, mass = 0, acc[D/32] = {}, query[D/32];
        for (int j = 0; j < D/32; ++j)
            query[j] = float(q[q0 + (32*j + lane)*q_strides[3]]);
        __shared__ float maxima[S], masses[S], partial[S][D];
        for (int64_t t = stripe; t < k_shape[2]; t += S) {
            const size_t row = k0 + t*k_strides[2];
            float dot = 0, norm2 = 0;
            for (int j = 0; j < D/32; ++j) {
                int d = 32*j + lane;
                float key;
                if constexpr (PackedK) {
                    unsigned byte = unsigned(k[row + (d/2)*k_strides[3]]);
                    key = cb[(byte >> (4*(d&1))) & 15];
                    norm2 += key*key;
                } else {
                    key = float(k[row + d*k_strides[3]]);
                }
                dot += query[j]*key;
            }
            for (int off = 16; off > 0; off >>= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, off);
                if constexpr (PackedK) norm2 += __shfl_xor_sync(0xffffffff, norm2, off);
            }
            float score = dot*scale[0];
            if constexpr (PackedK)
                score *= float(kn[kn0 + t*kn_strides[2]]) / fmaxf(sqrtf(norm2), 1e-10f);
            float next = fmaxf(maximum, score);
            float retain = expf(maximum - next), weight = expf(score - next);
            mass = mass*retain + weight;
            for (int j = 0; j < D/32; ++j) {
                int d = 32*j + lane;
                float value;
                if (t < cold) {
                    unsigned byte = vp[vp0 + t*vp_strides[2] + (d/2)*vp_strides[3]];
                    value = float(vr[vr0 + t*vr_strides[2]]) * cb[(byte >> (4*(d&1))) & 15];
                } else {
                    value = float(hot[hot0 + (t-cold)*hot_strides[2] + d*hot_strides[3]]);
                }
                acc[j] = acc[j]*retain + weight*value;
            }
            maximum = next;
        }
        for (int j = 0; j < D/32; ++j) partial[stripe][32*j + lane] = acc[j];
        if (lane == 0) { maxima[stripe] = maximum; masses[stripe] = mass; }
        __syncthreads();
        if (stripe == 0) {
            float largest = -INFINITY, total = 0, result[D/32] = {};
            for (int s = 0; s < S; ++s) largest = fmaxf(largest, maxima[s]);
            for (int s = 0; s < S; ++s) {
                float w = expf(maxima[s] - largest);
                total += masses[s]*w;
                for (int j = 0; j < D/32; ++j) result[j] += partial[s][32*j + lane]*w;
            }
            for (int j = 0; j < D/32; ++j)
                out[size_t(blockIdx.z)*D + 32*j + lane] = result[j]/total;
        }
    )CUDA", "", /*ensure_row_contiguous=*/false);
    return kernel;
}

const KernelPorts& packed_ports() {
    static const KernelPorts ports{.metal = nullptr, .cuda = cuda_port, .rocm = nullptr};
    return ports;
}
} // namespace

bool turbo_packed_attention_available() {
    return has_kernel_port(packed_ports());
}

mlx::core::array turbo_packed_attention(
    const array& q, const array& k, const array& kn, const array& vp,
    const array& vr, const array& hot, const array& cb, float scale, bool packed_k) {
    if (q.ndim() != 4 || k.ndim() != 4 || vp.ndim() != 4 || vr.ndim() != 4 ||
        kn.ndim() != 4 || hot.ndim() != 4 || cb.shape() != Shape{16} || cb.dtype() != float32)
        throw std::invalid_argument("turbo_packed_attention: invalid array rank or codebook");
    const int b = q.shape(0), hq = q.shape(1), d = q.shape(3);
    const int hk = k.shape(1), n = k.shape(2), cold = vp.shape(2);
    if (q.shape(2) != 1 || (d != 64 && d != 128 && d != 256) ||
        hk <= 0 || hq <= 0 || hq % hk || int64_t(b)*hq > 65535 || cold > n ||
        k.shape() != Shape{b,hk,n,packed_k ? d/2 : d} ||
        vp.shape() != Shape{b,hk,cold,d/2} || vp.dtype() != uint8 ||
        vr.shape() != Shape{b,hk,cold,1} || vr.dtype() != float16 ||
        (packed_k && (k.dtype() != uint8 || kn.shape() != Shape{b,hk,n,1} || kn.dtype() != float16)) ||
        (n > cold && hot.shape() != Shape{b,hk,n-cold,d}))
        throw std::invalid_argument("turbo_packed_attention: incompatible cache geometry or dtype");
    if (b == 0 || n == 0) return zeros(q.shape(), q.dtype());
    auto& kernel = select_kernel_port("turbo_packed_attention", "dequantize then SDPA", packed_ports());
    constexpr int splits = 32;
    using fast::TemplateArg;
    std::vector<std::pair<std::string, TemplateArg>> template_args = {
         {"D",d},{"Rep",hq/hk},{"S",splits},{"PackedK",packed_k},
         {"QType",q.dtype()},{"KType",k.dtype()},{"HotType",hot.dtype()}};
    return kernel({q,k,kn,vp,vr,hot,cb,array(scale)}, {q.shape()}, {q.dtype()},
        {32,splits,b*hq}, {32,splits,1}, template_args,
        std::nullopt, false, {})[0];
}
} // namespace mlxcel::turbo
