// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! PP-DocLayoutV3 self-attention through MLX's fast SDPA boundary.

use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::detection::rt_detr_v2::hybrid_encoder::SelfAttention;

pub(super) struct FastAttentionOutput {
    pub(super) output: UniquePtr<MlxArray>,
    #[cfg(test)]
    pub(super) query: UniquePtr<MlxArray>,
    #[cfg(test)]
    pub(super) key: UniquePtr<MlxArray>,
    #[cfg(test)]
    pub(super) value: UniquePtr<MlxArray>,
    #[cfg(test)]
    pub(super) sdpa: UniquePtr<MlxArray>,
    #[cfg(test)]
    pub(super) merged: UniquePtr<MlxArray>,
}

/// Match the pinned PP AIFI and decoder `mx.fast` self-attention graph.
pub(super) fn forward_fast(
    attention: &SelfAttention,
    input: &MlxArray,
    position: Option<&MlxArray>,
) -> FastAttentionOutput {
    let shape = mlxcel_core::array_shape(input);
    let (batch, length, hidden) = (shape[0], shape[1], shape[2]);
    let query_key_input = match position {
        Some(position) => mlxcel_core::add(input, position),
        None => mlxcel_core::copy(input),
    };
    let split = |projection: &mlxcel_core::layers::Linear, source: &MlxArray| {
        let projected = projection.forward(source);
        let projected = mlxcel_core::reshape(
            &projected,
            &[batch, length, attention.n_heads, attention.head_dim],
        );
        mlxcel_core::transpose_axes(&projected, &[0, 2, 1, 3])
    };
    let query = split(&attention.q_proj, &query_key_input);
    let key = split(&attention.k_proj, &query_key_input);
    let value = split(&attention.v_proj, input);
    let sdpa = mlxcel_core::layers::attention(&query, &key, &value, attention.scale, None, 0.0, 0);
    let merged = mlxcel_core::transpose_axes(&sdpa, &[0, 2, 1, 3]);
    let merged = mlxcel_core::reshape(&merged, &[batch, length, hidden]);
    let output = attention.out_proj.forward(&merged);
    FastAttentionOutput {
        output,
        #[cfg(test)]
        query,
        #[cfg(test)]
        key,
        #[cfg(test)]
        value,
        #[cfg(test)]
        sdpa,
        #[cfg(test)]
        merged,
    }
}
