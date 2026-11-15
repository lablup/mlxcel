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

//! Shared Llama4 helper functions for mask construction and weight loading.
//!
//! These helpers are reused across the Llama4 attention/cache path and are
//! easier to review and test in isolation than when buried deep in the main
//! model file.

use mlxcel_core::MlxArray;
use mlxcel_core::UniquePtr;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

use crate::models::llama4::TextArgs;

/// Chunked causal attention for a Llama 4 chunked (MoE) layer without a dense
/// mask (issue #3100).
///
/// A query at absolute position `p` attends to the keys of its own
/// `chunk_size` block (`p / chunk_size`) at positions `<= p`, the semantics of
/// [`create_chunked_attention_mask`] and of mlx-lm's `chunk_mask`. That mask is
/// block-diagonal and causal, so instead of building the `[L, S]` array (and
/// running the masked SDPA over every visible key) this splits the queries at
/// chunk boundaries and runs MLX's causal-mode SDPA per piece over that
/// piece's own keys: the piece's chunk start (or `start_position` when the
/// visible window begins later) up to the piece's last query. The causal mode
/// aligns bottom-right, which is what a piece with cached keys of the same
/// chunk in front of it needs, so a prefill piece that starts mid-chunk behind
/// a cache attends to those cached keys plus its own.
///
/// `q` is `[B, Hq, L, D]` for absolute positions `[offset, offset + L)`;
/// `keys` and `values` are `[B, Hkv, S, D]` for absolute positions
/// `[start_position, offset + L)`, which is what
/// [`mlxcel_core::layers::ChunkedKVCache::update_and_fetch`] returns. Returns
/// `[B, Hq, L, D]`. Every batch row shares the one cache, so the split is the
/// same for each row.
///
/// Used by: `models::llama4::Llama4Cache::attend` (chunked layers, unmasked
/// multi-token step).
pub(crate) fn chunked_causal_attention(
    q: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    start_position: i32,
    offset: i32,
    chunk_size: i32,
) -> UniquePtr<MlxArray> {
    let q_shape = mlxcel_core::array_shape(q);
    let k_shape = mlxcel_core::array_shape(keys);
    let v_shape = mlxcel_core::array_shape(values);
    let end = offset + q_shape[2];
    debug_assert_eq!(k_shape[2], end - start_position);
    if chunk_size <= 0 {
        return mlxcel_core::causal_attention(q, keys, values, scale, 0.0, 0);
    }

    let mut pieces = Vec::new();
    let mut q_start = offset;
    while q_start < end {
        let chunk_start = (q_start / chunk_size) * chunk_size;
        let q_end = (chunk_start + chunk_size).min(end);
        let k_start = chunk_start.max(start_position);
        if q_start == offset && q_end == end && k_start == start_position {
            // One piece over every visible key: plain causal, no slicing.
            return mlxcel_core::causal_attention(q, keys, values, scale, 0.0, 0);
        }
        let q_piece = mlxcel_core::slice(
            q,
            &[0, 0, q_start - offset, 0],
            &[q_shape[0], q_shape[1], q_end - offset, q_shape[3]],
        );
        let k_range = (k_start - start_position, q_end - start_position);
        let k_piece = mlxcel_core::slice(
            keys,
            &[0, 0, k_range.0, 0],
            &[k_shape[0], k_shape[1], k_range.1, k_shape[3]],
        );
        let v_piece = mlxcel_core::slice(
            values,
            &[0, 0, k_range.0, 0],
            &[v_shape[0], v_shape[1], k_range.1, v_shape[3]],
        );
        pieces.push(mlxcel_core::causal_attention(
            &q_piece, &k_piece, &v_piece, scale, 0.0, 0,
        ));
        q_start = q_end;
    }
    let refs: Vec<&MlxArray> = pieces.iter().map(|p| p.as_ref().unwrap()).collect();
    mlxcel_core::concatenate_many(&refs, 2)
}

/// Build the additive chunked attention mask used by Llama4 chunked caches.
///
/// The prefill no longer builds it (issue #3100, see
/// [`chunked_causal_attention`]); it stays as the reference the tests compare
/// that path against.
#[cfg(test)]
pub(crate) fn create_chunked_attention_mask(
    seq_len: i32,
    start_position: i32,
    offset: i32,
    chunk_size: usize,
) -> UniquePtr<MlxArray> {
    let end = offset + seq_len;
    let chunk_size = chunk_size as i32;

    let linds = mlxcel_core::arange_i32(start_position, end, 1);
    let rinds = mlxcel_core::arange_i32(offset, end, 1);
    let rinds = mlxcel_core::reshape(&rinds, &[seq_len, 1]);
    let visible_len = end - start_position;

    let chunk_size_f = mlxcel_core::full_f32(&[1], chunk_size as f32, mlxcel_core::dtype::FLOAT32);
    let linds_f = mlxcel_core::astype(&linds, mlxcel_core::dtype::FLOAT32);
    let rinds_f = mlxcel_core::astype(&rinds, mlxcel_core::dtype::FLOAT32);
    let linds_block = mlxcel_core::floor_divide(&linds_f, &chunk_size_f);
    let rinds_block = mlxcel_core::floor_divide(&rinds_f, &chunk_size_f);
    let linds_block = mlxcel_core::reshape(&linds_block, &[1, visible_len]);

    let block_diff = mlxcel_core::subtract(&rinds_block, &linds_block);
    let block_pos = mlxcel_core::abs(&block_diff);
    let zero_f = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
    let same_block = mlxcel_core::equal(&block_pos, &zero_f);

    let linds_reshaped = mlxcel_core::reshape(&linds, &[1, visible_len]);
    let rinds_orig = mlxcel_core::arange_i32(offset, end, 1);
    let rinds_reshaped = mlxcel_core::reshape(&rinds_orig, &[seq_len, 1]);
    let causal = mlxcel_core::less_equal(&linds_reshaped, &rinds_reshaped);
    let bool_mask = mlxcel_core::logical_and(&same_block, &causal);

    let zeros = mlxcel_core::zeros(&[seq_len, visible_len], mlxcel_core::dtype::FLOAT32);
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::where_cond(&bool_mask, &zeros, &neg_inf)
}

/// Get a deep copy of a named weight from the weight map.
pub(crate) fn get_weight_copy(
    weights: &WeightMap,
    name: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight.as_ref().unwrap()))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Load a possibly quantized linear layer from Llama4 weights.
pub(crate) fn load_quantized_linear(
    weights: &WeightMap,
    prefix: &str,
    args: &TextArgs,
) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, args.group_size(), args.bits())
}
