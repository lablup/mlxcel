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

use super::checked_unified_linear;
use super::config::Gemma3nAudioConfig;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Gemma3n audio weight not found: {key}"))
}

pub(crate) struct Gemma3nAudioAttention {
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_past_horizon: usize,
    max_future_horizon: usize,
    context_size: usize,
    softcap: f32,
    q_scale: f32,
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    per_dim_scale: UniquePtr<MlxArray>,
    pos_proj: UnifiedLinear,
    inv_timescales: UniquePtr<MlxArray>,
}

impl Gemma3nAudioAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma3nAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim();
        let num_timescales = config.hidden_size / 2;
        let increment = 10_000.0f32.ln() / (num_timescales as f32 - 1.0).max(1.0);
        let inv_timescales: Vec<f32> = (0..num_timescales)
            .map(|index| (-increment * index as f32).exp())
            .collect();
        let per_dim_scale = copy_weight(weights, &format!("{prefix}.per_dim_scale"))?;
        let scale_shape = mlxcel_core::array_shape(&per_dim_scale);
        if scale_shape != [head_dim as i32] {
            return Err(format!(
                "{prefix}.per_dim_scale has shape {scale_shape:?}; expected [{head_dim}]"
            ));
        }

        Ok(Self {
            num_heads: config.conf_num_attention_heads,
            head_dim,
            chunk_size: config.conf_attention_chunk_size,
            max_past_horizon: config.max_past_horizon(),
            max_future_horizon: config.conf_attention_context_right,
            context_size: config.context_size(),
            softcap: config.conf_attention_logit_cap,
            // head_dim^-0.5 / softplus(0), exactly as the official model.
            q_scale: (head_dim as f32).powf(-0.5) / 2.0f32.ln(),
            q_proj: checked_unified_linear(
                weights,
                &format!("{prefix}.q_proj"),
                config.hidden_size,
                config.hidden_size,
                group_size,
                bits,
            )?,
            k_proj: checked_unified_linear(
                weights,
                &format!("{prefix}.k_proj"),
                config.hidden_size,
                config.hidden_size,
                group_size,
                bits,
            )?,
            v_proj: checked_unified_linear(
                weights,
                &format!("{prefix}.v_proj"),
                config.hidden_size,
                config.hidden_size,
                group_size,
                bits,
            )?,
            per_dim_scale,
            pos_proj: checked_unified_linear(
                weights,
                &format!("{prefix}.relative_position_embedding.pos_proj"),
                config.hidden_size,
                config.hidden_size,
                group_size,
                bits,
            )?,
            inv_timescales: mlxcel_core::from_slice_f32(
                &inv_timescales,
                &[1, 1, num_timescales as i32],
            ),
        })
    }

    fn pad_dim1(&self, x: &MlxArray, left: i32, right: i32) -> UniquePtr<MlxArray> {
        let mut width = vec![0; mlxcel_core::array_ndim(x) * 2];
        width[2] = left;
        width[3] = right;
        mlxcel_core::pad(x, &width, 0.0)
    }

    fn convert_to_blocks(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let time = shape[1];
        let blocks = (time + self.chunk_size as i32 - 1) / self.chunk_size as i32;
        let padding = blocks * self.chunk_size as i32 - time;
        let x = if padding > 0 {
            self.pad_dim1(x, 0, padding)
        } else {
            mlxcel_core::copy(x)
        };
        let mut result_shape = vec![batch, blocks, self.chunk_size as i32];
        result_shape.extend_from_slice(&shape[2..]);
        mlxcel_core::reshape(&x, &result_shape)
    }

    fn extract_block_context(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let padded = self.pad_dim1(
            x,
            self.max_past_horizon as i32,
            (self.max_future_horizon + self.chunk_size - 1) as i32,
        );
        let shape = mlxcel_core::array_shape(&padded);
        let context = self.context_size as i32;
        let chunk = self.chunk_size as i32;
        let blocks = (shape[1] - context) / chunk + 1;
        let mut indices = Vec::with_capacity((blocks * context) as usize);
        for block in 0..blocks {
            for offset in 0..context {
                indices.push(block * chunk + offset);
            }
        }
        let indices = mlxcel_core::from_slice_i32(&indices, &[blocks * context]);
        let gathered = mlxcel_core::take(&padded, &indices, 1);
        let mut result_shape = vec![shape[0], blocks, context];
        result_shape.extend_from_slice(&shape[2..]);
        mlxcel_core::reshape(&gathered, &result_shape)
    }

    fn timing_signal(&self, positions: &[i32], dtype: i32) -> UniquePtr<MlxArray> {
        let positions: Vec<f32> = positions.iter().map(|position| *position as f32).collect();
        let positions = mlxcel_core::from_slice_f32(&positions, &[1, positions.len() as i32, 1]);
        let scaled = mlxcel_core::multiply(&positions, &self.inv_timescales);
        let signal =
            mlxcel_core::concatenate(&mlxcel_core::sin(&scaled), &mlxcel_core::cos(&scaled), -1);
        mlxcel_core::astype(&signal, dtype)
    }

    fn relative_logits(&self, queries: &MlxArray, keys: &MlxArray) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(queries);
        let k_shape = mlxcel_core::array_shape(keys);
        let (batch, blocks, block_size, heads, head_dim) =
            (q_shape[0], q_shape[1], q_shape[2], q_shape[3], q_shape[4]);
        let context = k_shape[2];
        let positions: Vec<i32> = (0..=self.max_past_horizon + self.max_future_horizon)
            .rev()
            .map(|value| value as i32 - self.max_future_horizon as i32)
            .collect();
        let span = positions.len() as i32;
        let signal = self.timing_signal(&positions, mlxcel_core::array_dtype(queries));
        let signal = self.pos_proj.forward(&signal);
        let signal = mlxcel_core::reshape(&signal, &[span, heads, head_dim]);

        let q = mlxcel_core::transpose_axes(queries, &[0, 3, 1, 2, 4]);
        let k = mlxcel_core::transpose_axes(keys, &[0, 3, 1, 4, 2]);
        let content = mlxcel_core::matmul(&q, &k);
        let signal = mlxcel_core::transpose_axes(&signal, &[1, 2, 0]);
        let q_flat = mlxcel_core::reshape(&q, &[batch, heads, blocks * block_size, head_dim]);
        let position = mlxcel_core::matmul(&q_flat, &signal);
        let position = mlxcel_core::reshape(&position, &[batch, heads, blocks, block_size, span]);
        let position = relative_shift(&position, batch, heads, blocks, block_size, context, span);
        mlxcel_core::add(&content, &position)
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &MlxArray,
        invalid_mask: &MlxArray,
        causal_valid_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let (batch, time) = (shape[0], shape[1]);
        let heads = self.num_heads as i32;
        let head_dim = self.head_dim as i32;
        let qkv_shape = [batch, time, heads, head_dim];

        let query = mlxcel_core::reshape(&self.q_proj.forward(hidden_states), &qkv_shape);
        let key = mlxcel_core::reshape(&self.k_proj.forward(hidden_states), &qkv_shape);
        let value = mlxcel_core::reshape(&self.v_proj.forward(hidden_states), &qkv_shape);
        let scale =
            mlxcel_core::multiply_scalar(&mlxcel_core::softplus(&self.per_dim_scale), self.q_scale);
        let query = mlxcel_core::multiply(&query, &scale);

        let query_blocks = self.convert_to_blocks(&query);
        let key_blocks = self.extract_block_context(&key);
        let value_blocks = self.extract_block_context(&value);
        let blocks = mlxcel_core::array_shape(&query_blocks)[1];

        let valid = mlxcel_core::logical_not(invalid_mask);
        let valid = self.extract_block_context(&valid);
        let valid = mlxcel_core::reshape(&valid, &[batch, 1, blocks, 1, self.context_size as i32]);
        let causal = mlxcel_core::reshape(
            causal_valid_mask,
            &[1, 1, 1, self.chunk_size as i32, self.context_size as i32],
        );
        let condition = mlxcel_core::logical_and(&valid, &causal);

        let logits = self.relative_logits(&query_blocks, &key_blocks);
        let logits = softcap_logits(&logits, self.softcap);
        let logits_dtype = mlxcel_core::array_dtype(&logits);
        let minimum = mlxcel_core::full_f32(&[1], dtype_min(logits_dtype), logits_dtype);
        let logits = mlxcel_core::where_cond(&condition, &logits, &minimum);
        let probabilities = mlxcel_core::softmax(
            &mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32),
            -1,
        );
        let probabilities =
            mlxcel_core::astype(&probabilities, mlxcel_core::array_dtype(&value_blocks));
        let operands: [*const MlxArray; 2] = [
            probabilities.as_ref().unwrap() as *const MlxArray,
            value_blocks.as_ref().unwrap() as *const MlxArray,
        ];
        let context = unsafe { mlxcel_core::einsum("bnuwc,bucnh->buwnh", &operands) };
        let context = mlxcel_core::reshape(
            &context,
            &[batch, blocks * self.chunk_size as i32, heads, head_dim],
        );
        if blocks * self.chunk_size as i32 > time {
            mlxcel_core::slice(&context, &[0, 0, 0, 0], &[batch, time, heads, head_dim])
        } else {
            context
        }
    }
}

fn dtype_min(dtype: i32) -> f32 {
    match dtype {
        d if d == mlxcel_core::dtype::FLOAT16 => -65504.0,
        d if d == mlxcel_core::dtype::BFLOAT16 => -3.38e38,
        _ => f32::MIN,
    }
}

#[allow(clippy::too_many_arguments)]
fn relative_shift(
    value: &MlxArray,
    batch: i32,
    heads: i32,
    blocks: i32,
    block_size: i32,
    context: i32,
    span: i32,
) -> UniquePtr<MlxArray> {
    let mut width = vec![0; mlxcel_core::array_ndim(value) * 2];
    let last = width.len() - 1;
    width[last] = context + 1 - span;
    let value = mlxcel_core::pad(value, &width, 0.0);
    let value = mlxcel_core::reshape(&value, &[batch, heads, blocks, block_size * (context + 1)]);
    let value = mlxcel_core::slice(
        &value,
        &[0, 0, 0, 0],
        &[batch, heads, blocks, block_size * context],
    );
    mlxcel_core::reshape(&value, &[batch, heads, blocks, block_size, context])
}

fn softcap_logits(logits: &MlxArray, cap: f32) -> UniquePtr<MlxArray> {
    let scaled = mlxcel_core::multiply_scalar(logits, 1.0 / cap);
    mlxcel_core::multiply_scalar(&mlxcel_core::tanh(&scaled), cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(array: &MlxArray) -> Vec<f32> {
        mlxcel_core::eval(array);
        mlxcel_core::array_to_raw_bytes(array)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn relative_shift_matches_pinned_jax_layout() {
        let input = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 2, 2]);
        let shifted = relative_shift(&input, 1, 1, 1, 2, 3, 2);
        assert_eq!(mlxcel_core::array_shape(&shifted), vec![1, 1, 1, 2, 3]);
        assert_eq!(values(&shifted), vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn softcap_is_not_an_unbounded_attention_substitute() {
        let logits = mlxcel_core::from_slice_f32(&[-100.0, 0.0, 100.0], &[3]);
        let capped = values(&softcap_logits(&logits, 5.0));
        assert!(capped[0] >= -5.0 && capped[0] < -4.99);
        assert_eq!(capped[1], 0.0);
        assert!(capped[2] <= 5.0 && capped[2] > 4.99);
    }

    #[test]
    fn attention_mask_minimum_remains_finite_in_checkpoint_dtypes() {
        for dtype in [mlxcel_core::dtype::FLOAT16, mlxcel_core::dtype::BFLOAT16] {
            let minimum = mlxcel_core::full_f32(&[1], dtype_min(dtype), dtype);
            let minimum = mlxcel_core::astype(&minimum, mlxcel_core::dtype::FLOAT32);
            assert!(mlxcel_core::item_f32(&minimum).is_finite());
        }
    }
}
