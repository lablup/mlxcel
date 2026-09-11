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

/// Build the additive chunked attention mask used by Llama4 chunked caches.
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
