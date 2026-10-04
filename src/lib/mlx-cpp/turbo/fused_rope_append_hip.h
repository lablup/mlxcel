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

#pragma once

// HIP body of the fused q/k RoPE + KV-append-layout kernel (issue #2063).
//
// Kept apart from `fused_rope_append.cpp` only for size: the launch, and the
// `fast::hip_kernel` call that compiles this string, stay in
// `fused_rope_append.cpp` next to the Metal and CUDA launches, so
// `make verify-kernel-dtype-keys` keeps its pinned scope unchanged. This header
// holds data, not a launch.
//
// A line-for-line port of `FUSED_ROPE_APPEND_CUDA_SOURCE`: same thread mapping
// (one thread per element pair of one (batch, token, head) slot), same grid
// `(HeadDim/2, L, B * (Hq + 2*Hkv))` over `(min(HeadDim/2, 64), 1, 1)`, which
// the ROCm `CustomKernel::eval_gpu` ceil-divides into blocks exactly as CUDA
// does (the bounds guard covers the padding), and the same template args.
// `qkv_shape` arrives by value (LOCAL_FIXES item 30), as CUDA's
// `__grid_constant__` shape does.
//
// One substitution, following the rule the CUDA comment in
// `fused_rope_append.cpp` states (#1049): call what MLX's own RoPE kernel for
// the same backend calls, so the fused and unfused paths agree. The overlay's
// `rope.hip` computes the angle as `(scale * position) * inv_freq` with
// `inv_freq = exp2f(-d * log2(base))` and takes both sides of it with one
// `sincosf`, so this body does too, in the same order. Both paths then take
// the trig of a bit-identical fp32 angle with the same device function, which
// is what keeps them together at the large unreduced angles of a long context
// (about 1.3e5 rad for the `p = 0` pair at position 131071).
//
// The two rotation outputs are written as explicit `fmaf` calls, in the
// contraction that reproduces the hipcc-compiled `rope.hip` on gfx1151. Left
// as `x1 * sintheta + x2 * costheta`, hipRTC's contraction of the second
// output differed from the graph's and moved about one f32 element in ten by
// one ulp against `fast_rope` (the first output already matched). Spelled out,
// the port is byte-identical to the graph in f32, f16 and bf16, which
// `fused_rope_append_is_byte_identical_to_the_rocm_graph` pins; a compiler
// change on the graph side would show there first.
//
// No wavefront guard, deliberately, as for `GUMBEL_MAX_SAMPLE_HIP_SOURCE`
// (#2064): no thread reads another thread's value, through a shuffle or
// through shared memory, so the kernel is correct for any wavefront size, and
// an `#error` on wave64 would refuse a correct kernel.

namespace mlxcel::turbo {

inline constexpr const char* FUSED_ROPE_APPEND_HIP_SOURCE = R"(
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    const uint32_t dim = (uint32_t)HeadDim;
    const uint32_t half_dim = dim / 2u;
    const uint32_t rdims = (uint32_t)RopeDims;
    const uint32_t rhalf = rdims / 2u;
    const uint32_t hq = (uint32_t)NHeadsQ;
    const uint32_t hkv = (uint32_t)NHeadsKV;
    const uint32_t batch = (uint32_t)qkv_shape[0];
    const uint32_t seq = (uint32_t)qkv_shape[1];
    const uint32_t slots = hq + 2u * hkv;

    if (p >= half_dim || t >= seq || z >= batch * slots) {
        return;
    }

    uint32_t b = z / slots;
    uint32_t slot = z - b * slots;

    uint32_t kind;   // 0 = q, 1 = k, 2 = v
    uint32_t h;
    uint32_t col;
    if (slot < hq) {
        kind = 0u;
        h = slot;
        col = 0u;
    } else if (slot < hq + hkv) {
        kind = 1u;
        h = slot - hq;
        col = hq * dim;
    } else {
        kind = 2u;
        h = slot - hq - hkv;
        col = (hq + hkv) * dim;
    }

    uint64_t in_base = ((uint64_t)b * seq + t) * (uint64_t)(slots * dim)
        + (uint64_t)col + (uint64_t)h * dim;

    uint64_t out_base;
    if (kind == 0u) {
        out_base = (((uint64_t)b * hq + h) * seq + t) * dim;
    } else if (DestLayout == 0) {
        out_base = (((uint64_t)b * hkv + h) * seq + t) * dim;
    } else {
        out_base = (((uint64_t)b * seq + t) * hkv + h) * dim;
    }

    if (kind == 2u) {
        uint32_t j = 2u * p;
        v_out[out_base + j] = qkv[in_base + j];
        v_out[out_base + j + 1u] = qkv[in_base + j + 1u];
        return;
    }

    bool rotate = p < rhalf;
    uint32_t i1;
    uint32_t i2;
    if (rotate) {
        if (Traditional) {
            i1 = 2u * p;
            i2 = i1 + 1u;
        } else {
            i1 = p;
            i2 = p + rhalf;
        }
    } else {
        i1 = rdims + 2u * (p - rhalf);
        i2 = i1 + 1u;
    }

    float x1 = (float)qkv[in_base + i1];
    float x2 = (float)qkv[in_base + i2];
    float r1 = x1;
    float r2 = x2;
    if (rotate) {
        float d = (float)p / (float)rhalf;
        float inv_freq = exp2f(-d * (float)rope_params[0]);
        float pos = (float)((int)t + positions_base[0]);
        float theta = ((float)rope_params[1] * pos) * inv_freq;
        float sintheta;
        float costheta;
        sincosf(theta, &sintheta, &costheta);
        r1 = fmaf(x1, costheta, -(x2 * sintheta));
        r2 = fmaf(x1, sintheta, x2 * costheta);
    }

    if (kind == 0u) {
        q_out[out_base + i1] = (T)r1;
        q_out[out_base + i2] = (T)r2;
    } else {
        k_out[out_base + i1] = (T)r1;
        k_out[out_base + i2] = (T)r2;
    }
)";

} // namespace mlxcel::turbo
