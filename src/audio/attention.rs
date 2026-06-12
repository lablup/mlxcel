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

//! Gemma4 Conformer chunked local attention with relative position embeddings.
//!
//! Ported from: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/audio.py (AudioAttention)
//!
//! Used by: Gemma4 audio encoder (ConformerBlock)

use super::config::AudioConfig;
use super::encoder::AudioLinear;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Audio weight not found: {key}"))
}

pub(crate) struct AudioAttention {
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_past_horizon: usize,
    max_future_horizon: usize,
    context_size: usize,
    invalid_logits_value: f32,
    softcap: f32,
    q_scale: f32,
    k_scale: f32,

    q_proj: AudioLinear,
    k_proj: AudioLinear,
    v_proj: AudioLinear,
    post: AudioLinear,

    per_dim_scale: UniquePtr<MlxArray>,
    relative_k_proj: UnifiedLinear,
    inv_timescales: UniquePtr<MlxArray>, // [1, 1, hidden/2]
}

impl AudioAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &AudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim();
        let q_scale = (head_dim as f32).powf(-0.5) / 2.0_f32.ln();
        let k_scale = (1.0_f32 + std::f32::consts::E).ln() / 2.0_f32.ln();

        let num_timescales = config.hidden_size / 2;
        let log_timescale_increment =
            (10000.0_f32 / 1.0).ln() / (num_timescales as f32 - 1.0).max(1.0);
        let inv_timescales_vec: Vec<f32> = (0..num_timescales)
            .map(|i| (-log_timescale_increment * i as f32).exp())
            .collect();
        let inv_timescales =
            mlxcel_core::from_slice_f32(&inv_timescales_vec, &[1, 1, num_timescales as i32]);

        Ok(Self {
            num_heads: config.num_attention_heads,
            head_dim,
            chunk_size: config.attention_chunk_size,
            max_past_horizon: config.max_past_horizon(),
            max_future_horizon: config.attention_context_right,
            context_size: config.context_size(),
            invalid_logits_value: config.attention_invalid_logits_value,
            softcap: config.attention_logit_cap,
            q_scale,
            k_scale,

            q_proj: AudioLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                group_size,
                bits,
            )?,
            k_proj: AudioLinear::from_weights(
                weights,
                &format!("{prefix}.k_proj"),
                group_size,
                bits,
            )?,
            v_proj: AudioLinear::from_weights(
                weights,
                &format!("{prefix}.v_proj"),
                group_size,
                bits,
            )?,
            post: AudioLinear::from_weights(weights, &format!("{prefix}.post"), group_size, bits)?,

            per_dim_scale: copy_weight(weights, &format!("{prefix}.per_dim_scale"))?,
            relative_k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.relative_k_proj"),
                group_size,
                bits,
            )?,
            inv_timescales,
        })
    }

    fn pad_dim1(&self, x: &MlxArray, pad_left: i32, pad_right: i32) -> UniquePtr<MlxArray> {
        let ndim = mlxcel_core::array_ndim(x);
        let mut pad_width = vec![0i32; ndim * 2];
        pad_width[2] = pad_left; // dim1 before
        pad_width[3] = pad_right; // dim1 after
        mlxcel_core::pad(x, &pad_width, 0.0)
    }

    /// [B, T, ...] -> [B, num_blocks, chunk_size, ...]
    fn convert_to_block(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let t = shape[1];
        let rest = &shape[2..];
        let num_blocks = (t + self.chunk_size as i32 - 1) / self.chunk_size as i32;
        let pad_len = num_blocks * self.chunk_size as i32 - t;
        let x = if pad_len > 0 {
            self.pad_dim1(x, 0, pad_len)
        } else {
            mlxcel_core::copy(x)
        };
        let mut new_shape = vec![batch, num_blocks, self.chunk_size as i32];
        new_shape.extend_from_slice(rest);
        mlxcel_core::reshape(&x, &new_shape)
    }

    /// [B, T, ...] -> [B, num_blocks, context_size, ...]
    fn extract_block_context(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let pad_left = self.max_past_horizon as i32;
        let pad_right = self.max_future_horizon as i32 + self.chunk_size as i32 - 1;
        let x = self.pad_dim1(x, pad_left, pad_right);

        let padded_shape = mlxcel_core::array_shape(&x);
        let batch = padded_shape[0];
        let t_padded = padded_shape[1];
        let rest = &padded_shape[2..];

        let ctx = self.context_size as i32;
        let chunk = self.chunk_size as i32;
        let num_blocks = (t_padded - ctx) / chunk + 1;

        let starts: Vec<i32> = (0..num_blocks).map(|i| i * chunk).collect();
        let offsets: Vec<i32> = (0..ctx).collect();
        let mut indices = Vec::with_capacity((num_blocks * ctx) as usize);
        for &s in &starts {
            for &o in &offsets {
                indices.push(s + o);
            }
        }
        let indices_arr = mlxcel_core::from_slice_i32(&indices, &[num_blocks, ctx]);

        let mut result_shape = vec![batch, num_blocks, ctx];
        result_shape.extend_from_slice(rest);

        let flat_indices = mlxcel_core::reshape(&indices_arr, &[num_blocks * ctx]);
        let flat_x = mlxcel_core::take(&x, &flat_indices, 1);
        mlxcel_core::reshape(&flat_x, &result_shape)
    }

    /// Compute sinusoidal timing signal for relative position embedding.
    fn get_timing_signal(&self, positions: &[i32], dtype: i32) -> UniquePtr<MlxArray> {
        let n = positions.len() as i32;
        let pos_f32: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
        let pos_arr = mlxcel_core::from_slice_f32(&pos_f32, &[1, n, 1]);

        let scaled = mlxcel_core::multiply(&pos_arr, &self.inv_timescales);
        let sin_part = mlxcel_core::sin(&scaled);
        let cos_part = mlxcel_core::cos(&scaled);
        let signal = mlxcel_core::concatenate(&sin_part, &cos_part, -1);
        mlxcel_core::astype(&signal, dtype)
    }

    /// Relative shift operation for position-aware attention logits.
    fn relative_shift(
        &self,
        term_bd: &MlxArray,
        batch: i32,
        num_heads: i32,
        num_blocks: i32,
        block_size: i32,
        context_size: i32,
        max_span_plus_1: i32,
    ) -> UniquePtr<MlxArray> {
        let pad_amount = (context_size + 1) - max_span_plus_1;
        let ndim = mlxcel_core::array_ndim(term_bd);
        let mut pad_width = vec![0i32; ndim * 2];
        pad_width[ndim * 2 - 1] = pad_amount;
        let term_bd = mlxcel_core::pad(term_bd, &pad_width, 0.0);
        let term_bd = mlxcel_core::reshape(
            &term_bd,
            &[
                batch,
                num_heads,
                num_blocks,
                block_size * (context_size + 1),
            ],
        );
        let term_bd = mlxcel_core::slice(
            &term_bd,
            &[0, 0, 0, 0],
            &[batch, num_heads, num_blocks, block_size * context_size],
        );
        mlxcel_core::reshape(
            &term_bd,
            &[batch, num_heads, num_blocks, block_size, context_size],
        )
    }

    fn compute_relative_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(queries);
        let k_shape = mlxcel_core::array_shape(keys);
        let batch = q_shape[0];
        let u = q_shape[1];
        let w = q_shape[2];
        let n = q_shape[3];
        let h = q_shape[4];
        let c = k_shape[2];

        let max_backward = self.max_past_horizon;
        let max_forward = self.max_future_horizon;

        let pos_indices: Vec<i32> = (0..=(max_backward + max_forward))
            .rev()
            .map(|i| i as i32 - max_forward as i32)
            .collect();
        let max_span_plus_1 = pos_indices.len() as i32;

        let sin_emb = self.get_timing_signal(&pos_indices, mlxcel_core::array_dtype(queries));
        let sin_emb_proj = self.relative_k_proj.forward(&sin_emb);
        let sin_emb = mlxcel_core::reshape(
            &mlxcel_core::astype(&sin_emb_proj, mlxcel_core::array_dtype(queries)),
            &[max_span_plus_1, n, h],
        );

        let queries_p = mlxcel_core::transpose_axes(queries, &[0, 3, 1, 2, 4]);
        let keys_p = mlxcel_core::transpose_axes(keys, &[0, 3, 1, 4, 2]);
        let term_ac = mlxcel_core::matmul(&queries_p, &keys_p);

        let sin_emb_t = mlxcel_core::transpose_axes(&sin_emb, &[1, 2, 0]);
        let q_reshaped = mlxcel_core::reshape(&queries_p, &[batch, n, u * w, h]);
        let term_bd = mlxcel_core::matmul(&q_reshaped, &sin_emb_t);
        let term_bd = mlxcel_core::reshape(&term_bd, &[batch, n, u, w, max_span_plus_1]);

        let term_bd = self.relative_shift(&term_bd, batch, n, u, w, c, max_span_plus_1);

        mlxcel_core::add(&term_ac, &term_bd)
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &MlxArray,
        mask: &MlxArray,
        causal_valid_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];
        let t = shape[1];
        let n = self.num_heads as i32;
        let h = self.head_dim as i32;
        let qkv_shape = [batch, t, n, h];

        let q = mlxcel_core::astype(
            &self.q_proj.forward(hidden_states),
            mlxcel_core::dtype::FLOAT32,
        );
        let q = mlxcel_core::reshape(&q, &qkv_shape);
        let k = mlxcel_core::astype(
            &self.k_proj.forward(hidden_states),
            mlxcel_core::dtype::FLOAT32,
        );
        let k = mlxcel_core::reshape(&k, &qkv_shape);
        let v = mlxcel_core::astype(
            &self.v_proj.forward(hidden_states),
            mlxcel_core::dtype::FLOAT32,
        );
        let v = mlxcel_core::reshape(&v, &qkv_shape);

        // Per-dimension scaling
        let per_dim_scale = mlxcel_core::softplus(&self.per_dim_scale);
        let q_scale_arr = mlxcel_core::full_f32(&[1], self.q_scale, mlxcel_core::dtype::FLOAT32);
        let q_factor = mlxcel_core::multiply(&q_scale_arr, &per_dim_scale);
        let q = mlxcel_core::multiply(&q, &q_factor);

        let k_scale_arr = mlxcel_core::full_f32(&[1], self.k_scale, mlxcel_core::dtype::FLOAT32);
        let k = mlxcel_core::multiply(&k, &k_scale_arr);

        // Chunk queries and extract key/value context windows
        let query_blocks = self.convert_to_block(&q);
        let key_blocks = self.extract_block_context(&k);
        let value_blocks = self.extract_block_context(&v);
        let u = mlxcel_core::array_shape(&query_blocks)[1];

        // Build validity mask
        let valid_mask = mlxcel_core::logical_not(mask);
        let extracted_valid = self.extract_block_context(&valid_mask);
        let extracted_valid = mlxcel_core::reshape(
            &extracted_valid,
            &[batch, 1, u, 1, self.context_size as i32],
        );
        let causal_mask = mlxcel_core::reshape(
            causal_valid_mask,
            &[1, 1, 1, self.chunk_size as i32, self.context_size as i32],
        );
        let condition = mlxcel_core::logical_and(&extracted_valid, &causal_mask);

        // Compute attention logits with relative position
        let logits = self.compute_relative_attention(&query_blocks, &key_blocks);

        // Softcap: tanh(logits / cap) * cap
        let logits = mlxcel_core::multiply_scalar(&logits, 1.0 / self.softcap);
        let logits = mlxcel_core::tanh(&logits);
        let logits = mlxcel_core::multiply_scalar(&logits, self.softcap);

        // Mask invalid positions
        let invalid_val =
            mlxcel_core::full_f32(&[1], self.invalid_logits_value, mlxcel_core::dtype::FLOAT32);
        let logits = mlxcel_core::where_cond(&condition, &logits, &invalid_val);

        // Softmax
        let probs = mlxcel_core::softmax(&logits, -1);

        // Attention: einsum("bnuwc,bucnh->buwnh", probs, value_blocks)
        let probs_ptrs: [*const MlxArray; 2] = [
            probs.as_ref().unwrap() as *const MlxArray,
            value_blocks.as_ref().unwrap() as *const MlxArray,
        ];
        let context = unsafe { mlxcel_core::einsum("bnuwc,bucnh->buwnh", &probs_ptrs) };

        let context = mlxcel_core::reshape(&context, &[batch, u * self.chunk_size as i32, n * h]);
        let context = if u * self.chunk_size as i32 > t {
            mlxcel_core::slice(&context, &[0, 0, 0], &[batch, t, n * h])
        } else {
            context
        };

        self.post.forward(&context)
    }
}
