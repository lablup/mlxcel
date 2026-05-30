// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Common utility functions for mlxcel-core
//!
//! This module provides shared utility functions used across multiple models,
//! reducing code duplication and ensuring consistency.

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use cxx::UniquePtr;
use std::sync::OnceLock;

// Array Slicing Utilities.
/// Slice an array along a specified axis.
///
/// # Arguments
/// * `x` - Input array
/// * `axis` - Axis to slice along (supports negative indexing)
/// * `start` - Start index (supports negative indexing)
/// * `end` - End index. Use -1 to mean "to the end of axis" (Python slice semantics)
///
/// # Example
/// ```ignore
/// // Slice x[:, 0:10, :] along axis 1
/// let sliced = slice_axis(&x, 1, 0, 10);
///
/// // Slice x[:, 5:, :] along axis 1 (5 to end)
/// let sliced = slice_axis(&x, 1, 5, -1);
/// ```
pub fn slice_axis(x: &MlxArray, axis: i32, start: i32, end: i32) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let ndim = shape.len();

    // Handle negative axis
    let axis = if axis < 0 { ndim as i32 + axis } else { axis } as usize;

    let dim_size = shape[axis];

    // Handle end index:
    // - end = -1 means "to the end of axis" (Python slice semantics)
    // - other negative values are relative to end
    let end = if end == -1 {
        dim_size
    } else if end < 0 {
        dim_size + end
    } else {
        end.min(dim_size)
    };

    // Handle negative start
    let start = if start < 0 {
        (dim_size + start).max(0)
    } else {
        start.min(dim_size)
    };

    // Build starts and stops vectors
    let mut starts = vec![0i32; ndim];
    let mut stops: Vec<i32> = shape.clone();
    starts[axis] = start;
    stops[axis] = end;

    ffi::slice(x, &starts, &stops)
}

// Attention Mask Utilities.
/// Create a causal attention mask.
/// Used by: Llama, Qwen, Mixtral, Gemma, Cohere, Phi, OLMo, Exaone, GLM4,
/// MiniCPM, DeepSeek, Hunyuan, StarCoder2 and other causal attention callers
///
/// Creates a lower triangular mask of shape [size, size + offset] where:
/// - 1.0 indicates positions that can be attended to
/// - -inf indicates positions that should be masked
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Offset for KV cache (number of previously cached tokens)
///
/// # Returns
/// Mask of shape [size, size + offset] with -inf in upper triangular region
pub fn create_causal_mask(size: i32, offset: i32) -> UniquePtr<MlxArray> {
    let total_len = size + offset;

    // Create lower triangular mask (1 = attend, 0 = mask)
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mask = ffi::tril(&ones, offset);

    // Convert to attention mask format: where mask=1 -> 0, where mask=0 -> -inf
    // Use where_cond to avoid NaN from 0 * -inf
    // Intentional FP32: additive attention masks carry 0/-inf sentinels and are
    // added to attention scores, not propagated as model activations.
    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    let neg_inf = ffi::full_f32(&[size, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_mask = ffi::greater(&mask, &zeros); // mask > 0 gives bool mask

    ffi::where_cond(&bool_mask, &zeros, &neg_inf)
}

/// Create a causal attention mask with per-sequence left-padding support.
///
/// Mirrors the Python `mlx_lm.models.base.create_causal_mask` with the
/// `left_padding` argument.  Used by [`crate::cache::batch_quant::BatchQuantizedKVCache::make_mask`]
/// and [`crate::cache::batch_quant::BatchTurboQuantKVCache::make_mask`].
///
/// # Arguments
/// * `n` — Number of query tokens in the current step (usually 1 for decode).
/// * `offset` — Actual number of tokens already in the KV buffer (`_idx` in Python
///   terminology, **not** the logical `offset` that starts negative for padded
///   sequences).  The total key length returned is `n + offset`.
/// * `left_padding` — Per-sequence number of leading padding tokens.  The mask
///   zeroes out (sets to −∞) key positions that are padding for each sequence.
///   When empty or all-zero the result is identical to [`create_causal_mask`].
///
/// # Returns
/// Additive mask with 0 for attended positions and −∞ for masked positions.
///
/// # Shape note
/// * **No padding** (`left_padding` empty or all-zero): `[n, n+offset]`, same
///   as [`create_causal_mask`].
/// * **With padding** (`left_padding` has at least one non-zero element):
///   `[B, 1, n, n+offset]` where `B = left_padding.len()`.  The `[B, 1]`
///   leading dims allow broadcasting against a `[B, H, n, n+offset]` score
///   tensor in batched SDPA.
///
/// Used by: BatchQuantizedKVCache, BatchTurboQuantKVCache
pub fn create_causal_mask_with_left_padding(
    n: i32,
    offset: i32,
    left_padding: &[i32],
) -> UniquePtr<MlxArray> {
    let total_len = n + offset;

    // ── Base causal (lower-triangular) mask ─────────────────────────────────
    // Shape: [n, total_len]  (0 = attend, -inf = mask after conversion)
    let ones = ffi::ones(&[n, total_len], dtype::FLOAT32);
    // `tril(ones, offset)` keeps the lower triangle starting `offset` columns
    // to the right of the main diagonal — i.e. the first `offset + q` entries
    // of query row `q`.  That matches the causal condition
    // `q_pos (= q + offset) >= k_pos`.
    let causal_tril = ffi::tril(&ones, offset);

    if left_padding.is_empty() || left_padding.iter().all(|&p| p == 0) {
        // Fast path: no per-sequence padding — identical to create_causal_mask.
        let zeros = ffi::zeros(&[n, total_len], dtype::FLOAT32);
        let neg_inf = ffi::full_f32(&[n, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
        let bool_mask = ffi::greater(&causal_tril, &zeros);
        return ffi::where_cond(&bool_mask, &zeros, &neg_inf);
    }

    // ── Left-padding filter ─────────────────────────────────────────────────
    // For each sequence `b`, key positions `k < left_padding[b]` are padding
    // and must be masked.  We build:
    //
    //   lp_tensor : [B, 1, 1, 1]  — per-sequence padding count
    //   rinds     : [1, 1, 1, total_len]  — key position indices
    //   lp_mask   : [B, 1, 1, total_len]  — True where key pos >= lp[b]
    //
    // Then broadcast-multiply with the causal mask.

    let b = left_padding.len() as i32;

    // Key position indices: 0, 1, …, total_len-1  (shape [1, 1, 1, total_len])
    let rinds_1d = ffi::arange_i32(0, total_len, 1);
    let rinds = ffi::reshape(&rinds_1d, &[1, 1, 1, total_len]);

    // Per-sequence left-padding: shape [B, 1, 1, 1]
    let lp_tensor = ffi::from_slice_i32(left_padding, &[b, 1, 1, 1]);

    // lp_mask[b,0,0,k] = (k >= left_padding[b])  — True = attend, False = mask
    // Using `greater_equal(rinds, lp_tensor)` broadcasts [B,1,1,total_len].
    let lp_mask = ffi::greater_equal(&rinds, &lp_tensor);

    // Causal mask broadcast: [1, 1, n, total_len]
    let causal_4d = ffi::reshape(&causal_tril, &[1, 1, n, total_len]);

    // Cast lp_mask to float for multiply (it is currently bool/int8 from the
    // greater_equal; we need float 0/1 to combine with the causal float mask).
    // Trick: use where_cond with ones/zeros to convert.
    let ones_lp = ffi::ones(&[b, 1, 1, total_len], dtype::FLOAT32);
    let zeros_lp = ffi::zeros(&[b, 1, 1, total_len], dtype::FLOAT32);
    let lp_mask_f32 = ffi::where_cond(&lp_mask, &ones_lp, &zeros_lp);

    // Combined: shape [B, 1, n, total_len]  (causal broadcasts over B)
    let combined = ffi::multiply(&causal_4d, &lp_mask_f32);

    // Convert 0/1 float mask to additive 0 / -inf mask.
    let zeros_out = ffi::zeros(&[b, 1, n, total_len], dtype::FLOAT32);
    let neg_inf_out = ffi::full_f32(&[b, 1, n, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_out = ffi::greater(&combined, &zeros_out);
    ffi::where_cond(&bool_out, &zeros_out, &neg_inf_out)
}

/// Create a boolean causal attention mask.
/// Used by: same as `create_causal_mask` (experimental path)
///
/// Returns a bool mask where `true` means "allowed attention".
pub fn create_causal_bool_mask(size: i32, offset: i32) -> UniquePtr<MlxArray> {
    let total_len = size + offset;
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mask = ffi::tril(&ones, offset);
    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    ffi::greater(&mask, &zeros)
}

/// Create a causal attention mask with sliding window.
/// Used by: Gemma2, Gemma3, Gemma3n, Gemma4, Qwen3, Ministral and other windowed-attention callers
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Offset for KV cache (tokens already in the cache before this call)
/// * `window` - Sliding window size (None for full attention)
///
/// # Returns
/// Mask with sliding window constraint applied, shaped `(size, T_k)` where
/// `T_k = min(size + offset, window)`.
///
/// ## Shape semantics when `size + offset > window`
///
/// A `RotatingKVCache` with `max_size = window` returns at most `window` K tokens
/// (the most recent ones).  The mask must match this T_k dimension so that
/// `mx::fast::scaled_dot_product_attention` can broadcast it against the score
/// tensor `(B, H, T_q, T_k)`.
///
/// When `total_len (= size + offset) > window`, the mask is produced as if we
/// took the last `window` columns of the full `(size, total_len)` causal mask.
/// Cache slot `k_cache` corresponds to logical key position
/// `k_cache + (total_len - window)`, and query row `q` corresponds to logical
/// query position `q + offset`. The causal condition is
/// `q_logical >= k_logical`:
///
/// ```text
/// q + offset >= k_cache + (total_len - window)
/// q + offset >= k_cache + (size + offset - window)
/// q         >= k_cache + (size - window)
/// ```
///
/// Hence the `tril` diagonal offset is `-(size - window) = window - size`,
/// independent of `offset`. The resulting mask shape is `(size, window)`,
/// matching the RotatingKVCache output and allowing broadcast to
/// `(B, H, size, window)`.
///
/// ## Why the window upper-bound term is elided in the capped path
///
/// In the full-length path the `triu` enforces `q <= k + window - 1`.  In the
/// capped path the column range is already restricted to the window; the upper
/// bound is always satisfied, so `triu` is omitted.
pub fn create_causal_mask_with_window(
    size: i32,
    offset: i32,
    window: Option<i32>,
) -> UniquePtr<MlxArray> {
    let uncapped_len = size + offset;

    // When a window is specified and the K sequence would exceed the window
    // (i.e. RotatingKVCache returns fewer than `uncapped_len` tokens), cap the
    // mask width so it matches the actual K dimension returned by the cache.
    //
    // Example: size=4096, offset=0, window=1024
    //   uncapped_len = 4096, which is > 1024.
    //   The cache returns K of shape (B, H, 1024, D).
    //   The score tensor is (B, H, 4096, 1024).
    //   A mask of (4096, 4096) cannot broadcast to (1, 8, 4096, 1024) — SIGABRT.
    //   Fix: produce mask of (4096, 1024) using adjusted tril offset.
    let (total_len, tril_offset) = if let Some(w) = window {
        if uncapped_len > w {
            // Cap: take the last `w` columns of the full (size, uncapped_len) mask.
            // Cache slot k_c holds logical key position k_c + (uncapped_len - w);
            // query row q holds logical query position q + offset. The causal
            // condition q + offset >= k_c + (uncapped_len - w) simplifies to
            // q >= k_c + (size - w), so the tril diagonal offset is
            // -(size - w) = w - size — independent of `offset`.
            (w, w - size)
        } else {
            (uncapped_len, offset)
        }
    } else {
        (uncapped_len, offset)
    };

    // Create lower triangular mask (1 = attend, 0 = mask)
    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mut mask = ffi::tril(&ones, tril_offset);

    // Apply sliding window upper-bound only when the mask is NOT capped.
    // In the capped path the column range is already the window; the upper
    // bound (q <= k + window - 1) is trivially satisfied.
    if let Some(w) = window {
        if uncapped_len <= w {
            // Non-capped path: enforce window upper bound.
            let upper_mask = ffi::triu(&ones, offset - w + 1);
            mask = ffi::multiply(&mask, &upper_mask);
        }
    }

    // Convert to attention mask format: where mask=1 -> 0, where mask=0 -> -inf
    // Use where_cond to avoid NaN from 0 * -inf
    // Intentional FP32: additive attention masks carry 0/-inf sentinels and are
    // added to attention scores, not propagated as model activations.
    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    let neg_inf = ffi::full_f32(&[size, total_len], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_mask = ffi::greater(&mask, &zeros); // mask > 0 gives bool mask

    ffi::where_cond(&bool_mask, &zeros, &neg_inf)
}

/// Create a boolean causal attention mask with optional sliding window.
/// Used by: same as `create_causal_mask_with_window` (experimental path)
///
/// Returns a bool mask where `true` means "allowed attention".
/// Shape is `(size, min(size + offset, window))` when window is specified.
/// See `create_causal_mask_with_window` for the shape-capping rationale.
pub fn create_causal_bool_mask_with_window(
    size: i32,
    offset: i32,
    window: Option<i32>,
) -> UniquePtr<MlxArray> {
    let uncapped_len = size + offset;

    let (total_len, tril_offset) = if let Some(w) = window {
        if uncapped_len > w {
            // See `create_causal_mask_with_window` for the derivation:
            // tril diagonal offset is `w - size`, independent of `offset`.
            (w, w - size)
        } else {
            (uncapped_len, offset)
        }
    } else {
        (uncapped_len, offset)
    };

    let ones = ffi::ones(&[size, total_len], dtype::FLOAT32);
    let mut mask = ffi::tril(&ones, tril_offset);

    if let Some(w) = window {
        if uncapped_len <= w {
            let upper_mask = ffi::triu(&ones, offset - w + 1);
            mask = ffi::multiply(&mask, &upper_mask);
        }
    }

    let zeros = ffi::zeros(&[size, total_len], dtype::FLOAT32);
    ffi::greater(&mask, &zeros)
}

// KV Cache Utilities.
/// Repeat key/value tensors for grouped-query attention.
///
/// When n_kv_heads < n_heads, we need to repeat K and V to match Q dimensions.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, n_kv_heads, seq_len, head_dim]
/// * `n_rep` - Number of times to repeat (n_heads / n_kv_heads)
///
/// # Returns
/// Tensor of shape [batch, n_heads, seq_len, head_dim]
pub fn repeat_kv(x: &MlxArray, n_rep: i32) -> UniquePtr<MlxArray> {
    if n_rep == 1 {
        // No repetition needed — return a zero-copy view via reshape
        let shape = ffi::array_shape(x);
        return ffi::reshape(x, &shape);
    }

    let shape = ffi::array_shape(x);
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // Reshape to [batch, n_kv_heads, 1, seq_len, head_dim]
    let x_exp = ffi::reshape(x, &[batch, n_kv_heads, 1, seq_len, head_dim]);

    // Broadcast to [batch, n_kv_heads, n_rep, seq_len, head_dim]
    let x_broad = ffi::broadcast_to(&x_exp, &[batch, n_kv_heads, n_rep, seq_len, head_dim]);

    // Reshape to [batch, n_kv_heads * n_rep, seq_len, head_dim]
    ffi::reshape(&x_broad, &[batch, n_kv_heads * n_rep, seq_len, head_dim])
}

// Activation Functions.
/// SiLU (Swish) activation: x * sigmoid(x) — compiled kernel fusion
#[inline]
pub fn silu(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::compiled_silu(x)
}

/// GELU activation with sigmoid approximation: x * sigmoid(1.702 * x)
///
/// NOTE: This is NOT the same as exact GELU or tanh-approximate GELU.
/// For exact GELU, use `ffi::gelu()` (re-exported as `mlxcel_core::gelu`).
/// For tanh-approximate GELU, use `gelu_approx()`.
#[inline]
pub fn gelu_sigmoid(x: &MlxArray) -> UniquePtr<MlxArray> {
    let x_dtype = ffi::array_dtype(x);
    let coef = ffi::full_f32(&[1], 1.702, x_dtype);
    let scaled = ffi::multiply(&coef, x);
    let sigmoid_x = ffi::sigmoid(&scaled);
    ffi::multiply(x, &sigmoid_x)
}

/// ReLU squared activation: max(0, x)^2 — compiled kernel fusion
#[inline]
pub fn relu_squared(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::compiled_relu_squared(x)
}

///// Numerically stable softplus activation: log(1 + exp(x)).
/// Uses logaddexp(x, 0) internally to match Python's mx.logaddexp(x, 0).
/// This avoids float16 overflow for values >= ~11.09 (exp(x) > float16 max).
/// Used by: Mamba, Mamba2, Jamba, GatedDelta, RecurrentGemma
#[inline]
pub fn softplus(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::softplus(x)
}

/// GELU approximate activation (erf-based for numerical stability with bf16)
/// Used by many models like Phi
#[inline]
pub fn gelu_approx(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::gelu_approx(x)
}

/// GeGELU activation for Phi3Small
/// Splits input into gelu and linear parts (interleaved), applies gelu to first half,
/// then computes: gelu(x[::2]) * (x[1::2] + 1.0)
///
/// # Arguments
/// * `x` - Input array where last dim will be split into interleaved gelu/linear parts
/// * `limit` - Clipping limit for numerical stability
pub fn gegelu(x: &MlxArray, limit: f32) -> UniquePtr<MlxArray> {
    let x_dtype = ffi::array_dtype(x);
    let shape = ffi::array_shape(x);
    let ndim = shape.len();
    let last_dim = shape[ndim - 1];
    let half_dim = last_dim / 2;

    // Split into gelu part (even indices) and linear part (odd indices)
    // Reshape: [B, L, D] -> [B, L, D/2, 2]
    let mut new_shape = shape.clone();
    new_shape[ndim - 1] = half_dim;
    new_shape.push(2);

    let x_reshaped = ffi::reshape(x, &new_shape);

    // Select gelu_part (index 0) and linear_part (index 1) along last axis
    // Using slice: gelu_part = x_reshaped[..., :, 0], linear_part = x_reshaped[..., :, 1]
    let mut starts = vec![0i32; ndim + 1];
    let mut stops: Vec<i32> = new_shape.clone();

    // gelu_part: slice [..., :, 0:1] then squeeze
    starts[ndim] = 0;
    stops[ndim] = 1;
    let gelu_part = ffi::slice(&x_reshaped, &starts, &stops);
    let gelu_part = ffi::squeeze_axis(&gelu_part, ndim as i32);

    // linear_part: slice [..., :, 1:2] then squeeze
    starts[ndim] = 1;
    stops[ndim] = 2;
    let linear_part = ffi::slice(&x_reshaped, &starts, &stops);
    let linear_part = ffi::squeeze_axis(&linear_part, ndim as i32);

    // Clip both parts for numerical stability
    let neg_limit = ffi::full_f32(&[1], -limit, x_dtype);
    let pos_limit = ffi::full_f32(&[1], limit, x_dtype);

    let a_gelu = ffi::clip(&gelu_part, &neg_limit, &pos_limit);
    let a_linear = ffi::clip(&linear_part, &neg_limit, &pos_limit);

    // Apply GELU approximation: x * sigmoid(1.702 * x)
    let coef = ffi::full_f32(&[1], 1.702, x_dtype);
    let scaled = ffi::multiply(&coef, &a_gelu);
    let sigmoid_x = ffi::sigmoid(&scaled);
    let out_gelu = ffi::multiply(&a_gelu, &sigmoid_x);

    // Compute: out_gelu * (a_linear + 1.0)
    let ones = ffi::full_f32(&[1], 1.0, x_dtype);
    let linear_plus_one = ffi::add(&a_linear, &ones);
    let out = ffi::multiply(&out_gelu, &linear_plus_one);
    if ffi::array_dtype(&out) == x_dtype {
        out
    } else {
        ffi::astype(&out, x_dtype)
    }
}

// Gemma-specific Functions.
/// Softcap function for Gemma 2/3 attention and logits.
///
/// Applies tanh(x / cap) * cap to prevent extreme values.
///
/// # Arguments
/// * `x` - Input array
/// * `cap` - Softcapping value
///
/// # Returns
/// Softcapped array
pub fn softcap(x: &MlxArray, cap: f32) -> UniquePtr<MlxArray> {
    let scaled = crate::divide_scalar(x, cap);
    let tanhed = ffi::tanh(&scaled);
    crate::multiply_scalar(&tanhed, cap)
}

/// Clip residual addition for float16 overflow prevention (Gemma 3).
///
/// When using float16, casts to float32, adds, clips to float16 range,
/// and casts back to float16. For other dtypes, performs normal addition.
///
/// # Arguments
/// * `x` - First input array
/// * `y` - Second input array (to be added to x)
///
/// # Returns
/// Clipped residual sum
pub fn clip_residual_f16(x: &MlxArray, y: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype_code = ffi::array_dtype(x);

    // Check if dtype is float16 (dtype code 9)
    if dtype_code != dtype::FLOAT16 {
        // Not float16, just add normally
        return ffi::add(x, y);
    }

    // float16 max is approximately 65504
    let bound = 65504.0f32;

    // Intentional FP32: the residual is widened only for overflow-safe clipping
    // and is cast back to f16 before returning.
    let x_f32 = ffi::astype(x, dtype::FLOAT32);
    let y_f32 = ffi::astype(y, dtype::FLOAT32);

    // Add
    let sum = ffi::add(&x_f32, &y_f32);

    // Create bound arrays
    let min_bound = ffi::full_f32(&[1], -bound, dtype::FLOAT32);
    let max_bound = ffi::full_f32(&[1], bound, dtype::FLOAT32);

    // Clip
    let clipped = ffi::clip(&sum, &min_bound, &max_bound);

    // Cast back to f16
    ffi::astype(&clipped, dtype::FLOAT16)
}

// Neural Accelerator Tile Alignment Utilities.

/// Tile size for the M5 Neural Accelerator optimal matrix operation.
pub const NA_TILE_SIZE: usize = 32;

/// Align a sequence length up to the nearest multiple of `NA_TILE_SIZE`.
///
/// When the sequence is already aligned (i.e. `len % NA_TILE_SIZE == 0`),
/// the value is returned unchanged. Otherwise it is rounded up so that
/// the prefill input perfectly fills complete 32×32 tiles, enabling peak
/// Neural Accelerator throughput on M5+ hardware.
///
/// # Examples
/// ```ignore
/// assert_eq!(align_to_na_tile(10), 32);
/// assert_eq!(align_to_na_tile(32), 32);
/// assert_eq!(align_to_na_tile(33), 64);
/// assert_eq!(align_to_na_tile(0),   0);
/// ```
#[inline]
pub fn align_to_na_tile(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    len.div_ceil(NA_TILE_SIZE) * NA_TILE_SIZE
}

/// Create a causal attention mask for a tile-aligned padded prefill.
///
/// The input sequence has `actual_len` real tokens followed by `pad_len =
/// padded_len - actual_len` padding tokens. The returned mask has shape
/// `[padded_len, padded_len]` and encodes two constraints:
///
/// 1. **Causal**: query position `q` may only attend to key positions `k ≤ q`.
/// 2. **No padding leakage**: key positions `k ≥ actual_len` are always masked
///    with −∞, even for query positions that are themselves padding tokens.
///
/// This ensures that after the padded forward pass:
/// - The logits at position `actual_len - 1` correctly predict the next token.
/// - Padding tokens do not pollute the KV cache values that will be trimmed.
///
/// # Arguments
/// * `actual_len` - Number of real (non-padding) tokens in the sequence.
/// * `padded_len` - Total sequence length after alignment (≥ `actual_len`).
/// * `offset`     - Number of tokens already in the KV cache (typically 0 for
///   fresh prefill, non-zero for multi-turn continuation).
pub fn create_padded_prefill_mask(
    actual_len: i32,
    padded_len: i32,
    offset: i32,
) -> UniquePtr<MlxArray> {
    let total_kv = padded_len + offset;

    // Step 1: causal lower-triangular mask over the full padded shape.
    let ones = ffi::ones(&[padded_len, total_kv], dtype::FLOAT32);
    let causal = ffi::tril(&ones, offset);

    // Step 2: build a key-padding mask that zeros out positions ≥ actual_len.
    // Shape: [1, total_kv]  (broadcast along the query axis).
    // Value: 1 for valid key positions, 0 for padding key positions.
    let mut valid_mask_data = vec![0f32; total_kv as usize];
    for v in valid_mask_data
        .iter_mut()
        .take((actual_len + offset) as usize)
    {
        *v = 1.0;
    }
    let valid_mask = ffi::from_slice_f32(&valid_mask_data, &[1, total_kv]);

    // Combine: both constraints must hold (multiply, then convert to -inf mask).
    let combined = ffi::multiply(&causal, &valid_mask);

    // Convert to additive mask: 1 → 0.0,  0 → -inf
    // Intentional FP32: additive attention masks carry 0/-inf sentinels and are
    // added to attention scores, not propagated as model activations.
    let zeros = ffi::zeros(&[padded_len, total_kv], dtype::FLOAT32);
    let neg_inf = ffi::full_f32(&[padded_len, total_kv], f32::NEG_INFINITY, dtype::FLOAT32);
    let bool_mask = ffi::greater(&combined, &zeros);
    ffi::where_cond(&bool_mask, &zeros, &neg_inf)
}

// Shape Utilities.
/// Concatenate two arrays along the specified axis.
#[inline]
pub fn concatenate(a: &MlxArray, b: &MlxArray, axis: i32) -> UniquePtr<MlxArray> {
    crate::concatenate(a, b, axis)
}

/// Stack arrays along a new axis.
pub fn stack_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    let ptrs: Vec<*const MlxArray> = arrays
        .iter()
        .map(|a| a.as_ref().unwrap() as *const _)
        .collect();
    unsafe { ffi::stack(&ptrs, axis) }
}

// Pipeline Hint for Layer-Level async_eval

/// Granularity setting for layer-boundary pipeline hints.
///
/// Controlled via the `MLXCEL_PIPELINE_GRANULARITY` environment variable:
/// - `layer`   — call `async_eval` after every transformer layer
/// - `block:N` — call `async_eval` every N layers (e.g. `block:4`)
/// - `off`     — no intermediate eval (default; preserves MLX graph fusion)
#[derive(Debug, Clone, Copy)]
enum PipelineMode {
    /// No intermediate eval — current MLX default behavior.
    Off,
    /// Evaluate after every transformer layer.
    PerLayer,
    /// Evaluate every N layers.
    PerBlock(usize),
}

fn get_pipeline_mode() -> PipelineMode {
    match std::env::var("MLXCEL_PIPELINE_GRANULARITY")
        .as_deref()
        .unwrap_or("off")
    {
        "layer" => PipelineMode::PerLayer,
        s if s.starts_with("block:") => {
            let n = s[6..].parse::<usize>().unwrap_or(4);
            PipelineMode::PerBlock(n.max(1))
        }
        _ => PipelineMode::Off,
    }
}

/// Insert an `async_eval` pipeline hint at a transformer layer boundary.
///
/// Calling this after each layer's `forward()` allows MLX's lazy evaluation
/// engine to begin executing the current layer's compute graph while the next
/// layer's weights are prefetched into L2 cache, hiding memory latency.
///
/// On M5 (Neural Accelerator + GPU shader cores), this can improve throughput
/// by overlapping NA compute for layer N with weight loads for layer N+1.
///
/// Activation is controlled by `MLXCEL_PIPELINE_GRANULARITY`:
/// - `layer`   — hint after every layer
/// - `block:N` — hint every N layers
/// - `off`     — no hints (default; preserves MLX graph fusion)
///
/// # Arguments
/// * `hidden` - The hidden state tensor output from the current layer.
/// * `layer_idx` - Zero-based index of the layer that was just executed.
/// * `total_layers` - Total number of transformer layers in the model.
///
/// Used by: Llama3, Qwen3, Gemma, Gemma2, Gemma3
#[inline]
pub fn pipeline_hint(hidden: &MlxArray, layer_idx: usize, total_layers: usize) {
    static MODE: OnceLock<PipelineMode> = OnceLock::new();
    let mode = MODE.get_or_init(get_pipeline_mode);

    // Never emit a hint after the last layer — the caller will eval the output.
    if layer_idx + 1 >= total_layers {
        return;
    }

    match mode {
        PipelineMode::Off => {}
        PipelineMode::PerLayer => {
            ffi::async_eval(hidden);
        }
        PipelineMode::PerBlock(n) => {
            if (layer_idx + 1).is_multiple_of(*n) {
                ffi::async_eval(hidden);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_axis_basic() {
        // Create a simple test array
        let x = ffi::ones(&[2, 10, 4], dtype::FLOAT32);

        // Slice middle portion
        let sliced = slice_axis(&x, 1, 2, 5);
        let shape = ffi::array_shape(&sliced);
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_slice_axis_end_minus_one() {
        let x = ffi::ones(&[2, 10, 4], dtype::FLOAT32);

        // Slice from index 5 to end using -1
        let sliced = slice_axis(&x, 1, 5, -1);
        let shape = ffi::array_shape(&sliced);
        assert_eq!(shape, vec![2, 5, 4]); // 10 - 5 = 5
    }

    #[test]
    fn test_repeat_kv() {
        let x = ffi::ones(&[1, 4, 10, 64], dtype::FLOAT32);

        // Repeat 2 times (4 heads -> 8 heads)
        let repeated = repeat_kv(&x, 2);
        let shape = ffi::array_shape(&repeated);
        assert_eq!(shape, vec![1, 8, 10, 64]);
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        let x = ffi::ones(&[1, 8, 10, 64], dtype::FLOAT32);

        // No repeat needed
        let repeated = repeat_kv(&x, 1);
        let shape = ffi::array_shape(&repeated);
        assert_eq!(shape, vec![1, 8, 10, 64]);
    }

    #[test]
    fn test_align_to_na_tile_zero() {
        assert_eq!(align_to_na_tile(0), 0);
    }

    #[test]
    fn test_align_to_na_tile_exact() {
        // Already aligned
        assert_eq!(align_to_na_tile(32), 32);
        assert_eq!(align_to_na_tile(64), 64);
        assert_eq!(align_to_na_tile(128), 128);
    }

    #[test]
    fn test_align_to_na_tile_short() {
        // Prompts shorter than one tile
        assert_eq!(align_to_na_tile(1), 32);
        assert_eq!(align_to_na_tile(10), 32);
        assert_eq!(align_to_na_tile(31), 32);
    }

    #[test]
    fn test_align_to_na_tile_cross_boundary() {
        assert_eq!(align_to_na_tile(33), 64);
        assert_eq!(align_to_na_tile(63), 64);
        assert_eq!(align_to_na_tile(65), 96);
    }

    #[test]
    fn test_create_padded_prefill_mask_shape() {
        // actual_len=10, padded_len=32, offset=0
        let mask = create_padded_prefill_mask(10, 32, 0);
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![32, 32]);
    }

    #[test]
    fn test_create_padded_prefill_mask_no_padding() {
        // When actual_len == padded_len, result equals a standard causal mask
        let mask = create_padded_prefill_mask(8, 8, 0);
        let ref_mask = create_causal_mask(8, 0);
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![8, 8]);
        let ref_shape = ffi::array_shape(&ref_mask);
        assert_eq!(ref_shape, vec![8, 8]);
    }

    // --- Sliding window mask shape regression tests -------------

    /// Gemma3-4B trigger: seq_len=4096, window=1024, offset=0.
    ///
    /// Before the fix the mask had shape (4096, 4096).  MLX SDPA falls back to
    /// software when head_dim=256 (not in the Metal fast-kernel list) and its
    /// fallback tried to broadcast (4096, 4096) against score (B, H, 4096, 1024)
    /// → SIGABRT.  After the fix the mask must be (4096, 1024).
    #[test]
    fn test_sliding_window_mask_shape_capped_when_seq_exceeds_window() {
        let mask = create_causal_mask_with_window(4096, 0, Some(1024));
        let shape = ffi::array_shape(&mask);
        // Must be (T_q=4096, T_k=min(4096+0, 1024)=1024), NOT (4096, 4096).
        assert_eq!(
            shape,
            vec![4096, 1024],
            "mask shape must match RotatingKVCache output (4096, 1024); \
             got {shape:?} — broadcast mismatch against score (B,H,4096,1024) would SIGABRT"
        );
    }

    /// When seq_len < window the mask must retain its full (T_q, T_q+offset)
    /// shape — no spurious capping.
    #[test]
    fn test_sliding_window_mask_shape_uncapped_when_seq_within_window() {
        // seq=512, offset=0, window=1024: total=512 < 1024 → no cap
        let mask = create_causal_mask_with_window(512, 0, Some(1024));
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![512, 512]);
    }

    /// When total_len exactly equals window the mask must NOT be capped.
    #[test]
    fn test_sliding_window_mask_shape_at_window_boundary() {
        // seq=512, offset=512, window=1024: total=1024 == window → no cap
        let mask = create_causal_mask_with_window(512, 512, Some(1024));
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![512, 1024]);
    }

    /// In the capped path, queries below the cache start horizon must be fully
    /// masked (-inf).  For seq=4, window=2, offset=0:
    ///   cache holds last 2 of the 4 input tokens (positions 2..3).
    ///   q=0 and q=1 cannot attend to any cached key → all -inf.
    ///   q=2 attends to k=0 (input pos 2≥input pos 2). → row 2, col 0 = 0.
    ///   q=3 attends to k=0,1 (input 3≥2, 3≥3). → row 3, cols 0 and 1 = 0.
    #[test]
    fn test_sliding_window_mask_values_when_capped() {
        // Produce (4, 2) mask: rows=T_q, cols=T_k=window=2
        let mask = create_causal_mask_with_window(4, 0, Some(2));
        let shape = ffi::array_shape(&mask);
        assert_eq!(shape, vec![4, 2]);

        // Extract values (the mask is additive: 0.0 = attend, -inf = block)
        let row0_col0 = ffi::item_f32(&ffi::slice(&mask, &[0, 0], &[1, 1]));
        let row1_col0 = ffi::item_f32(&ffi::slice(&mask, &[1, 0], &[2, 1]));
        let row2_col0 = ffi::item_f32(&ffi::slice(&mask, &[2, 0], &[3, 1]));
        let row3_col0 = ffi::item_f32(&ffi::slice(&mask, &[3, 0], &[4, 1]));
        let row3_col1 = ffi::item_f32(&ffi::slice(&mask, &[3, 1], &[4, 2]));

        // q=0,1 cannot see any cache key (cache starts at input pos 2)
        assert!(
            row0_col0.is_infinite() && row0_col0 < 0.0,
            "row0_col0 should be -inf, got {row0_col0}"
        );
        assert!(
            row1_col0.is_infinite() && row1_col0 < 0.0,
            "row1_col0 should be -inf, got {row1_col0}"
        );
        // q=2 attends to cache-k=0 (input pos 2 ≥ input pos 2)
        assert_eq!(row2_col0, 0.0, "row2_col0 should be 0.0 (attend)");
        // q=3 attends to both cache keys
        assert_eq!(row3_col0, 0.0, "row3_col0 should be 0.0 (attend)");
        assert_eq!(row3_col1, 0.0, "row3_col1 should be 0.0 (attend)");
    }
}
