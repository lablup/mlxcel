// Copyright 2026 Lablup Inc.
//
// Experimental Metal 4 fused attention kernel scaffold.
//
// This file is intentionally kept separate from MLX upstream overlay kernels so
// we can iterate on a mlxcel-specific TensorOps kernel body.
// Runtime wiring still lives in `cpp/mlx_cxx_bridge.cpp`.

#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
    uint batch;
    uint num_heads;
    uint num_kv_heads;
    uint q_len;
    uint kv_len;
    uint head_dim;
    float scale;
    float softcap;
    int window_size;
};

// NOTE:
// - Current runtime path still delegates to MLX SDPA/compiled graphs.
// - This kernel body exists to define the target TensorOps-oriented interface
//   and to stage future direct Metal 4 command encoding.
// - Once the encoder path is wired, this kernel will become the primary M5
//   fused attention implementation.
kernel void fused_attention_metal4(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Placeholder body:
    // keep output deterministic while wiring the dispatch path.
    // Real implementation will:
    // 1) scores = Q @ K^T * scale
    // 2) optional softcap / mask / sliding-window filtering
    // 3) probs = softmax(scores)
    // 4) output = probs @ V
    const uint idx = gid.x;
    const uint total = params.batch * params.num_heads * params.q_len * params.head_dim;
    if (idx < total) {
        output[idx] = half(0.0);
    }
}

