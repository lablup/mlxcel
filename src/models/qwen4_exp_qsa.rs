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

//! Qwen Sparse Attention (`Qwen4ExpQSAIndexer`, `Qwen4ExpAttention`).
//!
//! A small indexer scores the key history in blocks of `compress_ratio`
//! tokens and keeps the top `budget / compress_ratio` blocks per query; the
//! query then attends to those blocks plus the incomplete block it sits in.
//! While a query sees at most `budget / compress_ratio` complete blocks the
//! selection covers its whole causal prefix, so attention stays dense until a
//! query's prefix reaches `budget + compress_ratio` tokens.
//!
//! Metal uses upstream's indexed online-softmax kernel for prefill and
//! sparse decode. Other backends retain the boolean-mask SDPA fallback.

use mlxcel_core::cache::KVCache;
use mlxcel_core::dtype;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};

use crate::models::qwen3_vl::{InterleavedMRoPE, rotate_half};

use super::config::ResolvedQwen4ExpConfig;
use super::norm::ZeroCenteredRmsNorm;
use super::shapes;

/// Positions and rotary tables of one forward pass, shared by every attention
/// layer and its indexer.
pub(crate) struct RopeContext<'a> {
    /// `[3, B, L]` int32 MRoPE positions.
    pub(crate) positions: UniquePtr<MlxArray>,
    /// `[B, L, rope_dims]` in float32, matching the fused upstream MRoPE kernel.
    pub(crate) cos: UniquePtr<MlxArray>,
    pub(crate) sin: UniquePtr<MlxArray>,
    pub(crate) mrope: &'a InterleavedMRoPE,
}

impl<'a> RopeContext<'a> {
    fn apply(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.mrope
            .apply_metal(x, &self.positions)
            .unwrap_or_else(|| apply_partial_rope(x, &self.cos, &self.sin))
    }

    pub(crate) fn new(mrope: &'a InterleavedMRoPE, positions: UniquePtr<MlxArray>) -> Self {
        let (cos, sin) = mrope.forward(&positions);
        Self {
            cos,
            sin,
            positions,
            mrope,
        }
    }
}

/// `x` is `[B, H, L, D]`, `cos`/`sin` `[B, L, R]` with `R <= D`: rotate the
/// first `R` channels (`rotate_half` pairing) and pass the rest through.
pub(crate) fn apply_partial_rope(
    x: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let rotary = *mlxcel_core::array_shape(cos)
        .last()
        .expect("cos has rank 3");
    let x_dtype = mlxcel_core::array_dtype(x);
    let cos = mlxcel_core::expand_dims(cos, 1);
    let sin = mlxcel_core::expand_dims(sin, 1);
    let mut stop = shape.clone();
    stop[3] = rotary;
    let x_rot = mlxcel_core::slice(x, &[0, 0, 0, 0], &stop);
    let x_rot = mlxcel_core::astype(&x_rot, dtype::FLOAT32);
    let rotated = mlxcel_core::add(
        &mlxcel_core::multiply(&x_rot, &cos),
        &mlxcel_core::multiply(&rotate_half(&x_rot), &sin),
    );
    let rotated = mlxcel_core::astype(&rotated, x_dtype);
    if rotary == shape[3] {
        return rotated;
    }
    let x_pass = mlxcel_core::slice(x, &[0, 0, 0, rotary], &shape);
    concatenate(&rotated, &x_pass, -1)
}

/// Indexer state kept next to an attention layer's KV cache.
#[derive(Default)]
pub(crate) struct QsaIndexCache {
    /// `[B, T, indexer_head_dim]`, before normalization and RoPE.
    pub(crate) raw_keys: Option<UniquePtr<MlxArray>>,
    /// `[3, B, T]` int32.
    pub(crate) positions: Option<UniquePtr<MlxArray>>,
    /// `[B, 1, blocks, indexer_head_dim]`: pooled, normalized and rotated
    /// summaries of the complete blocks seen so far.
    pub(crate) block_keys: Option<UniquePtr<MlxArray>>,
}

impl QsaIndexCache {
    pub(crate) fn len(&self) -> i32 {
        self.raw_keys
            .as_ref()
            .map_or(0, |keys| mlxcel_core::array_shape(keys)[1])
    }

    fn cached_blocks(&self) -> i32 {
        self.block_keys
            .as_ref()
            .map_or(0, |keys| mlxcel_core::array_shape(keys)[2])
    }
}

/// The blocks each query keeps. `selected` is `[B, L, block_topk]` int32
/// with `-1` for unused slots.
pub(crate) struct QsaSelection {
    pub(crate) selected: UniquePtr<MlxArray>,
    /// Exclusive end of each query's causal prefix.
    pub(crate) query_ends: Vec<i32>,
    pub(crate) key_len: i32,
}

pub(crate) struct QsaIndexer {
    index_qk_proj: UnifiedLinear,
    q_layernorm: ZeroCenteredRmsNorm,
    k_layernorm: ZeroCenteredRmsNorm,
    n_heads: i32,
    head_dim: i32,
    compress_ratio: i32,
    block_topk: i32,
}

impl QsaIndexer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let raw = &config.raw;
        if config.rope_dims() > raw.indexer_head_dim {
            return Err(format!(
                "Qwen4-Exp rotary width {} exceeds indexer_head_dim {}",
                config.rope_dims(),
                raw.indexer_head_dim
            ));
        }
        let module = format!("{prefix}.index_qk_proj");
        let (group_size, bits) = config.quantization.for_module(&module);
        shapes::expect_matrix(
            weights,
            &module,
            (group_size, bits),
            &[
                (raw.indexer_n_heads + 1) * raw.indexer_head_dim,
                raw.hidden_size,
            ],
        )?;
        let norm = |name: &str| {
            ZeroCenteredRmsNorm::from_weights(
                weights,
                &format!("{prefix}.{name}"),
                raw.indexer_head_dim,
                None,
                raw.rms_norm_eps,
            )
        };
        Ok(Self {
            index_qk_proj: UnifiedLinear::from_weights(weights, &module, group_size, bits)?,
            q_layernorm: norm("q_layernorm")?,
            k_layernorm: norm("k_layernorm")?,
            n_heads: raw.indexer_n_heads as i32,
            head_dim: raw.indexer_head_dim as i32,
            compress_ratio: raw.indexer_compress_ratio as i32,
            block_topk: (raw.indexer_budget / raw.indexer_compress_ratio) as i32,
        })
    }

    /// Append this call's index keys and positions to `cache`, and select
    /// blocks when any query of the call sees more complete blocks than the
    /// budget holds. `None` means every query attends densely.
    pub(crate) fn select(
        &self,
        hidden: &MlxArray,
        rope: &RopeContext<'_>,
        cache: &mut QsaIndexCache,
    ) -> Option<QsaSelection> {
        let shape = mlxcel_core::array_shape(hidden);
        let (batch, seq_len) = (shape[0], shape[1]);
        let past = cache.len();

        let qk = self.index_qk_proj.forward(hidden);
        let qk = mlxcel_core::reshape(&qk, &[batch, seq_len, self.n_heads + 1, self.head_dim]);
        let raw_keys = mlxcel_core::slice(
            &qk,
            &[0, 0, self.n_heads, 0],
            &[batch, seq_len, self.n_heads + 1, self.head_dim],
        );
        let raw_keys = mlxcel_core::reshape(&raw_keys, &[batch, seq_len, self.head_dim]);
        let raw_keys = match cache.raw_keys.take() {
            Some(previous) => concatenate(&previous, &raw_keys, 1),
            None => raw_keys,
        };
        let positions = match cache.positions.take() {
            Some(previous) => concatenate(&previous, &rope.positions, -1),
            None => mlxcel_core::copy(&rope.positions),
        };
        cache.raw_keys = Some(raw_keys);
        cache.positions = Some(positions);

        let key_len = past + seq_len;
        let max_complete_blocks = key_len / self.compress_ratio;
        if max_complete_blocks <= self.block_topk {
            return None;
        }

        let query = mlxcel_core::slice(
            &qk,
            &[0, 0, 0, 0],
            &[batch, seq_len, self.n_heads, self.head_dim],
        );
        let query = self.q_layernorm.forward(&query);
        let query = mlxcel_core::transpose_axes(&query, &[0, 2, 1, 3]);
        let query = rope.apply(&query);

        let pooled = self.block_keys(cache, batch, max_complete_blocks, rope.mrope);
        let scores = mlxcel_core::matmul(
            &mlxcel_core::astype(&query, dtype::FLOAT32),
            &mlxcel_core::transpose_axes(
                &mlxcel_core::astype(&pooled, dtype::FLOAT32),
                &[0, 1, 3, 2],
            ),
        );
        let zero = mlxcel_core::full_f32(&[1], 0.0, dtype::FLOAT32);
        let scores = mlxcel_core::sum_axis(&mlxcel_core::maximum(&scores, &zero), 1, false);
        let scores = mlxcel_core::divide_scalar(&scores, (self.head_dim as f32).sqrt());

        let query_ends: Vec<i32> = (0..seq_len).map(|i| past + i + 1).collect();
        let complete: Vec<i32> = query_ends
            .iter()
            .map(|end| end / self.compress_ratio)
            .collect();
        let complete_counts = mlxcel_core::from_slice_i32(&complete, &[1, seq_len, 1]);
        let block_ids = mlxcel_core::arange_i32(0, max_complete_blocks, 1);
        let block_ids = mlxcel_core::reshape(&block_ids, &[1, 1, max_complete_blocks]);
        let valid = mlxcel_core::less(&block_ids, &complete_counts);
        let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT32);
        let scores = mlxcel_core::where_cond(&valid, &scores, &neg_inf);

        let width = max_complete_blocks;
        let kth = width - self.block_topk;
        let order = mlxcel_core::argpartition(&scores, kth, -1);
        let selected = mlxcel_core::slice(&order, &[0, 0, kth], &[batch, seq_len, width]);
        let selected = mlxcel_core::astype(&selected, dtype::INT32);
        let unused = mlxcel_core::full_f32(&[1], -1.0, dtype::INT32);
        let selected = mlxcel_core::where_cond(
            &mlxcel_core::less(&selected, &complete_counts),
            &selected,
            &unused,
        );
        Some(QsaSelection {
            selected,
            query_ends,
            key_len,
        })
    }

    /// Summaries of every complete block, reusing the cached prefix of them.
    fn block_keys(
        &self,
        cache: &mut QsaIndexCache,
        batch: i32,
        max_complete_blocks: i32,
        mrope: &InterleavedMRoPE,
    ) -> UniquePtr<MlxArray> {
        let raw_keys = cache.raw_keys.as_ref().expect("select appends keys first");
        let positions = cache
            .positions
            .as_ref()
            .expect("select appends positions first");
        let first_new = cache.cached_blocks().min(max_complete_blocks);
        if first_new == max_complete_blocks {
            return mlxcel_core::copy(cache.block_keys.as_ref().expect("cached blocks"));
        }
        let new_blocks = max_complete_blocks - first_new;
        let pooled = mlxcel_core::slice(
            raw_keys,
            &[0, first_new * self.compress_ratio, 0],
            &[
                batch,
                max_complete_blocks * self.compress_ratio,
                self.head_dim,
            ],
        );
        let pooled = mlxcel_core::reshape(
            &pooled,
            &[batch, new_blocks, self.compress_ratio, self.head_dim],
        );
        let raw_dtype = mlxcel_core::array_dtype(&pooled);
        let pooled =
            mlxcel_core::mean_axis(&mlxcel_core::astype(&pooled, dtype::FLOAT32), 2, false);
        let pooled = self
            .k_layernorm
            .forward(&mlxcel_core::astype(&pooled, raw_dtype));
        let pooled = mlxcel_core::expand_dims(&pooled, 1);

        let starts = mlxcel_core::arange_i32(
            first_new * self.compress_ratio,
            max_complete_blocks * self.compress_ratio,
            self.compress_ratio,
        );
        let starts = mlxcel_core::reshape(&starts, &[1, 1, new_blocks]);
        let block_positions = mlxcel_core::take_along_axis(
            positions,
            &mlxcel_core::broadcast_to(&starts, &[3, batch, new_blocks]),
            2,
        );
        let (cos, sin) = mrope.forward(&block_positions);
        let pooled = mrope
            .apply_metal(&pooled, &block_positions)
            .unwrap_or_else(|| apply_partial_rope(&pooled, &cos, &sin));
        let pooled = match cache.block_keys.take() {
            Some(previous) if first_new > 0 => concatenate(&previous, &pooled, 2),
            _ => pooled,
        };
        cache.block_keys = Some(mlxcel_core::copy(&pooled));
        pooled
    }

    /// `[B, 1, L, key_len]` boolean mask: for a query past the budget, the
    /// selected blocks plus its own incomplete block; otherwise its whole
    /// causal prefix.
    pub(crate) fn build_mask(&self, selection: &QsaSelection) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(&selection.selected);
        let (batch, seq_len, topk) = (shape[0], shape[1], shape[2]);
        let key_len = selection.key_len;
        let ratio = self.compress_ratio;

        let within = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, ratio, 1), &[1, 1, 1, ratio]);
        let starts = mlxcel_core::multiply(
            &mlxcel_core::expand_dims(&selection.selected, -1),
            &mlxcel_core::full_f32(&[1], ratio as f32, dtype::INT32),
        );
        let tokens = mlxcel_core::add(&starts, &within);
        let valid = mlxcel_core::logical_and(
            &mlxcel_core::greater_equal(
                &mlxcel_core::expand_dims(&selection.selected, -1),
                &mlxcel_core::full_f32(&[1], 0.0, dtype::INT32),
            ),
            &mlxcel_core::less(
                &tokens,
                &mlxcel_core::full_f32(&[1], key_len as f32, dtype::INT32),
            ),
        );
        let sentinel = mlxcel_core::full_f32(&[1], key_len as f32, dtype::INT32);
        let tokens = mlxcel_core::where_cond(&valid, &tokens, &sentinel);
        let tokens = mlxcel_core::reshape(&tokens, &[batch, seq_len, topk * ratio]);
        let marks = mlxcel_core::ones(&[batch, seq_len, topk * ratio], dtype::BOOL);
        let selected = mlxcel_core::put_along_axis(
            &mlxcel_core::zeros(&[batch, seq_len, key_len + 1], dtype::BOOL),
            &tokens,
            &marks,
            -1,
        );
        let selected = mlxcel_core::slice(&selected, &[0, 0, 0], &[batch, seq_len, key_len]);

        let ends: Vec<i32> = selection.query_ends.clone();
        let tail_starts: Vec<i32> = ends.iter().map(|end| (end / ratio) * ratio).collect();
        let sparse: Vec<bool> = ends
            .iter()
            .map(|end| end / ratio > self.block_topk)
            .collect();
        let column = |values: &[i32]| mlxcel_core::from_slice_i32(values, &[1, seq_len, 1]);
        let token_ids =
            mlxcel_core::reshape(&mlxcel_core::arange_i32(0, key_len, 1), &[1, 1, key_len]);
        let causal = mlxcel_core::less(&token_ids, &column(&ends));
        let tail = mlxcel_core::logical_and(
            &mlxcel_core::greater_equal(&token_ids, &column(&tail_starts)),
            &causal,
        );
        let sparse_rows: Vec<i32> = sparse.iter().map(|&s| i32::from(s)).collect();
        let sparse_rows = mlxcel_core::astype(&column(&sparse_rows), dtype::BOOL);
        let mask = mlxcel_core::where_cond(
            &sparse_rows,
            &mlxcel_core::logical_or(&selected, &tail),
            &causal,
        );
        mlxcel_core::expand_dims(&mask, 1)
    }
}

pub(crate) struct QsaAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: ZeroCenteredRmsNorm,
    k_norm: ZeroCenteredRmsNorm,
    pub(crate) indexer: QsaIndexer,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl QsaAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let raw = &config.raw;
        let (hidden, head_dim) = (raw.hidden_size, config.head_dim);
        let linear = |name: &str, rows: usize, columns: usize| {
            let module = format!("{prefix}.{name}");
            let (group_size, bits) = config.quantization.for_module(&module);
            shapes::expect_matrix(weights, &module, (group_size, bits), &[rows, columns])?;
            UnifiedLinear::from_weights(weights, &module, group_size, bits)
        };
        let norm = |name: &str| {
            ZeroCenteredRmsNorm::from_weights(
                weights,
                &format!("{prefix}.{name}"),
                head_dim,
                None,
                raw.rms_norm_eps,
            )
        };
        let (query_width, kv_width) = (
            raw.num_attention_heads * head_dim,
            raw.num_key_value_heads * head_dim,
        );
        Ok(Self {
            q_proj: linear("q_proj", 2 * query_width, hidden)?,
            k_proj: linear("k_proj", kv_width, hidden)?,
            v_proj: linear("v_proj", kv_width, hidden)?,
            o_proj: linear("o_proj", hidden, query_width)?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            indexer: QsaIndexer::from_weights(weights, config, &format!("{prefix}.indexer"))?,
            num_heads: raw.num_attention_heads as i32,
            num_kv_heads: raw.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            scale: (config.head_dim as f32).powf(-0.5),
        })
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        rope: &RopeContext<'_>,
        kv: &mut KVCache,
        index: &mut QsaIndexCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let selection = self.indexer.select(x, rope, index);

        let q_proj = self.q_proj.forward(x);
        let q_proj = mlxcel_core::reshape(&q_proj, &[b, l, self.num_heads, 2 * self.head_dim]);
        let queries = mlxcel_core::slice(
            &q_proj,
            &[0, 0, 0, 0],
            &[b, l, self.num_heads, self.head_dim],
        );
        let gate = mlxcel_core::slice(
            &q_proj,
            &[0, 0, 0, self.head_dim],
            &[b, l, self.num_heads, 2 * self.head_dim],
        );
        let gate = mlxcel_core::reshape(&gate, &[b, l, -1]);
        let keys = mlxcel_core::reshape(
            &self.k_proj.forward(x),
            &[b, l, self.num_kv_heads, self.head_dim],
        );
        let values = mlxcel_core::reshape(
            &self.v_proj.forward(x),
            &[b, l, self.num_kv_heads, self.head_dim],
        );

        let queries = mlxcel_core::transpose_axes(&self.q_norm.forward(&queries), &[0, 2, 1, 3]);
        let keys = mlxcel_core::transpose_axes(&self.k_norm.forward(&keys), &[0, 2, 1, 3]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);
        let queries = rope.apply(&queries);
        let keys = rope.apply(&keys);

        let (keys, values) = kv.update_and_fetch(keys, values);

        let sparse = (l > 1 || selection.is_some())
            && mlxcel_core::metal_is_available()
            && mlxcel_core::default_device_is_gpu()
            && matches!(
                mlxcel_core::array_dtype(&queries),
                dtype::FLOAT16 | dtype::BFLOAT16
            )
            && self.head_dim % 32 == 0
            && self.indexer.compress_ratio <= 32
            && mlxcel_core::array_dtype(&keys) == mlxcel_core::array_dtype(&queries)
            && mlxcel_core::array_dtype(&values) == mlxcel_core::array_dtype(&queries);
        let output = if sparse {
            let ends: Vec<i32> = (index.len() - l + 1..=index.len()).collect();
            let ends = mlxcel_core::from_slice_i32(&ends, &[1, l]);
            let ends = mlxcel_core::broadcast_to(&ends, &[b, l]);
            let blocks = match selection.as_ref() {
                Some(selection) => mlxcel_core::copy(&selection.selected),
                None => {
                    // The reference pads short prefixes to the same block budget,
                    // retaining one reduction geometry throughout causal prefill.
                    let ids = mlxcel_core::arange_i32(0, self.indexer.block_topk, 1);
                    let ids = mlxcel_core::reshape(&ids, &[1, 1, self.indexer.block_topk]);
                    let counts = mlxcel_core::floor_divide(
                        &ends,
                        &mlxcel_core::full_f32(
                            &[1],
                            self.indexer.compress_ratio as f32,
                            dtype::INT32,
                        ),
                    );
                    let valid = mlxcel_core::less(&ids, &mlxcel_core::expand_dims(&counts, -1));
                    mlxcel_core::where_cond(
                        &valid,
                        &ids,
                        &mlxcel_core::full_f32(&[1], -1.0, dtype::INT32),
                    )
                }
            };
            mlxcel_core::qwen4_qsa_attention(
                &queries,
                &keys,
                &values,
                &blocks,
                &ends,
                self.scale,
                self.indexer.compress_ratio,
            )
        } else {
            let mask = selection
                .as_ref()
                .map(|selection| self.indexer.build_mask(selection));
            match mask {
                Some(mask) => mlxcel_core::layers::attention(
                    &queries,
                    &keys,
                    &values,
                    self.scale,
                    Some(&mask),
                    0.0,
                    0,
                ),
                None if l > 1 => {
                    mlxcel_core::causal_attention(&queries, &keys, &values, self.scale, 0.0, 0)
                }
                None => mlxcel_core::layers::attention(
                    &queries, &keys, &values, self.scale, None, 0.0, 0,
                ),
            }
        };
        let output = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]),
            &[b, l, -1],
        );
        self.o_proj.forward(&mlxcel_core::multiply(
            &output,
            &mlxcel_core::sigmoid(&gate),
        ))
    }
}
