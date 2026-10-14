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
// The two rotation outputs are written as explicit `fmaf` calls under
// `fp contract(off)`, in the contraction that reproduces the hipcc-compiled
// `rope.hip` on gfx1151, because hipcc did not contract the same expression
// the same way in its two kernels: for one token of one sequence (batch-1
// decode) the graph runs `rope_single_1d`, whose second output matches
// `fma(x2, cos, x1 * sin)`, and otherwise `rope`, whose second output matches
// `fma(x1, sin, x2 * cos)`. The body picks the same form from the same
// condition (`B == 1 && L == 1`; the projection output is row-contiguous, as
// the graph's q and k are there). Left to hipRTC's own contraction, about one
// f32 element in ten moved by one ulp against `fast_rope`, and with only one
// form spelled out the other shape still did. `fp contract(off)` keeps hipRTC
// from contracting the remaining products into neighbouring operations.
//
// f16 needs one more step. The graph's f16 instantiations round each fused
// result to f16 once, as a mixed-precision fma with an f16 destination
// would, while `(T)fmaf(...)` rounds it to f32 first; the double rounding
// disagrees on roughly one element in 2^13, which a 186-token window of 48
// heads always hits. For f16 the body therefore forms the same fused results
// in double (the products of two floats are exact there) and converts once
// (`__half`'s templated constructor converts a double to `_Float16`
// directly). f32 and bf16 keep the `fmaf` value: f32 needs no second
// rounding, and the graph's bf16 store rounds the f32 value too.
//
// With that, a stress run of 432 cases (batch 1 to 3, windows of 1 to 300
// tokens, offsets up to 131000, both conventions, full and partial rotary
// dims, values scaled up to 181x) matched the graph bit for bit in f32, f16
// and bf16, and `fused_rope_append_is_byte_identical_to_the_rocm_graph` pins
// representative cases; a compiler change on the graph side would show there
// first.
//
// No wavefront guard, deliberately, as for `GUMBEL_MAX_SAMPLE_HIP_SOURCE`
// (#2064): no thread reads another thread's value, through a shuffle or
// through shared memory, so the kernel is correct for any wavefront size, and
// an `#error` on wave64 would refuse a correct kernel.

namespace mlxcel::turbo {

inline constexpr const char* FUSED_ROPE_APPEND_HIP_SOURCE = R"(
    // The braces make the pragma the start of a compound statement, which is
    // the only place clang accepts it inside the function MLX wraps around
    // this body (the wrapper may emit declarations ahead of it).
    {
    #pragma clang fp contract(off)
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
    double e1 = x1;
    double e2 = x2;
    if (rotate) {
        float d = (float)p / (float)rhalf;
        float inv_freq = exp2f(-d * (float)rope_params[0]);
        float pos = (float)((int)t + positions_base[0]);
        float theta = ((float)rope_params[1] * pos) * inv_freq;
        float sintheta;
        float costheta;
        sincosf(theta, &sintheta, &costheta);
        // The graph's two kernels fuse different products of the second
        // output; see the header comment. `e1` / `e2` are the same fused
        // results before their f32 rounding (the products are exact in
        // double), for the f16 store below.
        float p1 = x2 * sintheta;
        r1 = fmaf(x1, costheta, -p1);
        e1 = (double)x1 * (double)costheta - (double)p1;
        if (batch == 1u && seq == 1u) {
            float p2 = x1 * sintheta;
            r2 = fmaf(x2, costheta, p2);
            e2 = (double)x2 * (double)costheta + (double)p2;
        } else {
            float p2 = x2 * costheta;
            r2 = fmaf(x1, sintheta, p2);
            e2 = (double)x1 * (double)sintheta + (double)p2;
        }
    }

    T o1;
    T o2;
    if constexpr (__is_same(T, __half)) {
        o1 = (T)e1;
        o2 = (T)e2;
    } else {
        o1 = (T)r1;
        o2 = (T)r2;
    }
    if (kind == 0u) {
        q_out[out_base + i1] = o1;
        q_out[out_base + i2] = o2;
    } else {
        k_out[out_base + i1] = o1;
        k_out[out_base + i2] = o2;
    }
    }
)";

} // namespace mlxcel::turbo
