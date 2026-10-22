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

//! Qwen4-Exp GatedDeltaNet (`Qwen4ExpGatedDeltaNet`).
//!
//! The Qwen3.5 layer with two changes: q/k use the Transformers/FLA L2
//! normalization written out in the activation dtype, and the output norm's
//! gate activation follows `output_gate_type`.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::silu;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};

use crate::models::gated_delta::{GatedDeltaCache, gated_delta_update_precise};

use super::config::ResolvedQwen4ExpConfig;
use super::norm::GatedOutputNorm;
use super::shapes;

/// `x * rsqrt(sum(x^2) + 1e-6)` over the last axis, rounding to the input
/// dtype after every op as upstream does.
///
/// Qwen3.5's `normalize_qk` computes the same function through the fused
/// RMSNorm kernel, which accumulates in float32 and rounds once. The two
/// differ in half precision, by far the most in float16, where the squares of
/// small entries lose their low bits.
pub(crate) fn l2_normalize(x: &MlxArray) -> UniquePtr<MlxArray> {
    let x_dtype = mlxcel_core::array_dtype(x);
    let squared = mlxcel_core::square(x);
    let sum = mlxcel_core::sum_axis_native_dtype(&squared, -1, true);
    let eps = mlxcel_core::full_f32(&[1], 1e-6, x_dtype);
    let inv = mlxcel_core::rsqrt(&mlxcel_core::add(&sum, &eps));
    mlxcel_core::multiply(x, &inv)
}

/// Returns `(l2(q) * head_dim^-0.5, l2(k))`.
pub(crate) fn normalize_qk(
    q: &MlxArray,
    k: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let head_dim = *mlxcel_core::array_shape(q).last().expect("q has rank >= 1");
    let scale = (head_dim as f32).powf(-0.5);
    let q = mlxcel_core::multiply_scalar(&l2_normalize(q), scale);
    (q, l2_normalize(k))
}

pub(crate) struct Qwen4ExpGatedDeltaNet {
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    conv_dim: i32,
    conv_kernel_size: i32,
    conv1d_weight: UniquePtr<MlxArray>,
    in_proj_qkv: UnifiedLinear,
    in_proj_z: UnifiedLinear,
    in_proj_b: UnifiedLinear,
    in_proj_a: UnifiedLinear,
    decode_projection: Option<UnifiedLinear>,
    dt_bias: UniquePtr<MlxArray>,
    a_log: UniquePtr<MlxArray>,
    norm: GatedOutputNorm,
    out_proj: UnifiedLinear,
}

impl Qwen4ExpGatedDeltaNet {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let raw = &config.raw;
        if !raw
            .linear_num_value_heads
            .is_multiple_of(raw.linear_num_key_heads)
        {
            return Err(format!(
                "Qwen4-Exp linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
                raw.linear_num_value_heads, raw.linear_num_key_heads
            ));
        }
        let key_dim = raw.linear_key_head_dim * raw.linear_num_key_heads;
        let value_dim = raw.linear_value_head_dim * raw.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let hidden = raw.hidden_size;
        let linear = |name: &str, rows: usize, columns: usize| {
            let module = format!("{prefix}.{name}");
            let (group_size, bits) = config.quantization.for_module(&module);
            shapes::expect_matrix(weights, &module, (group_size, bits), &[rows, columns])?;
            UnifiedLinear::from_weights(weights, &module, group_size, bits)
        };
        let tensor = |name: &str| {
            let key = format!("{prefix}.{name}");
            weights
                .get(&key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Qwen4-Exp checkpoint is missing {key}"))
        };
        let vector = |name: &str, len: usize| {
            shapes::expect_vector(weights, &format!("{prefix}.{name}"), len)?;
            tensor(name)
        };
        let conv1d_weight = tensor("conv1d.weight")?;
        let conv_shape = mlxcel_core::array_shape(&conv1d_weight);
        let expected = [conv_dim as i32, raw.linear_conv_kernel_dim as i32, 1];
        if conv_shape != expected {
            return Err(format!(
                "{prefix}.conv1d.weight has shape {conv_shape:?}, expected {expected:?}"
            ));
        }
        let in_proj_qkv = linear("in_proj_qkv", conv_dim, hidden)?;
        let in_proj_z = linear("in_proj_z", value_dim, hidden)?;
        let in_proj_b = linear("in_proj_b", raw.linear_num_value_heads, hidden)?;
        let in_proj_a = linear("in_proj_a", raw.linear_num_value_heads, hidden)?;
        let decode_projection = super::linear::fused_decode_projection(&[
            &in_proj_qkv,
            &in_proj_z,
            &in_proj_b,
            &in_proj_a,
        ]);
        Ok(Self {
            num_k_heads: raw.linear_num_key_heads as i32,
            num_v_heads: raw.linear_num_value_heads as i32,
            head_k_dim: raw.linear_key_head_dim as i32,
            head_v_dim: raw.linear_value_head_dim as i32,
            key_dim: key_dim as i32,
            conv_dim: conv_dim as i32,
            conv_kernel_size: raw.linear_conv_kernel_dim as i32,
            conv1d_weight,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            decode_projection,
            dt_bias: vector("dt_bias", raw.linear_num_value_heads)?,
            a_log: vector("A_log", raw.linear_num_value_heads)?,
            norm: GatedOutputNorm::new(
                vector("norm.weight", raw.linear_value_head_dim)?,
                raw.rms_norm_eps,
                config.output_gate,
            ),
            out_proj: linear("out_proj", hidden, value_dim)?,
        })
    }

    pub(crate) fn forward(
        &self,
        inputs: &MlxArray,
        cache: Option<&mut GatedDeltaCache>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        let (b, s) = (shape[0], shape[1]);

        let fused = self.decode_projection.as_ref().filter(|linear| {
            s == 1
                && matches!(linear, UnifiedLinear::Quantized { weight, .. }
                if mlxcel_core::array_dtype(&weight.scales) == mlxcel_core::array_dtype(inputs))
        });
        let (qkv, z, b_proj, a) = if let Some(linear) = fused {
            let projected = linear.forward(inputs);
            let widths = [
                self.conv_dim,
                self.num_v_heads * self.head_v_dim,
                self.num_v_heads,
                self.num_v_heads,
            ];
            let mut offset = 0;
            let mut split = |width| {
                let part = mlxcel_core::slice(&projected, &[0, 0, offset], &[b, s, offset + width]);
                offset += width;
                part
            };
            (
                split(widths[0]),
                split(widths[1]),
                split(widths[2]),
                split(widths[3]),
            )
        } else {
            (
                self.in_proj_qkv.forward(inputs),
                self.in_proj_z.forward(inputs),
                super::linear::row_invariant_projection(&self.in_proj_b, inputs),
                super::linear::row_invariant_projection(&self.in_proj_a, inputs),
            )
        };
        let z = mlxcel_core::reshape(&z, &[b, s, self.num_v_heads, self.head_v_dim]);

        let qkv_dtype = mlxcel_core::array_dtype(&qkv);
        let n_keep = self.conv_kernel_size - 1;
        let conv_state = cache
            .as_ref()
            .and_then(|c| c.conv_state.as_ref())
            .filter(|state| mlxcel_core::array_shape(state)[0] == b)
            .map(|state| mlxcel_core::copy(state))
            .unwrap_or_else(|| mlxcel_core::zeros(&[b, n_keep, self.conv_dim], qkv_dtype));
        let conv_input = concatenate(&conv_state, &qkv, 1);

        let mut cache = cache;
        if let Some(c) = cache.as_deref_mut() {
            let conv_len = mlxcel_core::array_shape(&conv_input)[1];
            let tail = mlxcel_core::slice(
                &conv_input,
                &[0, conv_len - n_keep, 0],
                &[b, conv_len, self.conv_dim],
            );
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        let conv_out =
            mlxcel_core::conv1d(&conv_input, &self.conv1d_weight, 1, 0, 1, self.conv_dim);
        let conv_out = silu(&conv_out);
        let split = |start: i32, stop: i32, heads: i32, dim: i32| {
            let part = mlxcel_core::slice(&conv_out, &[0, 0, start], &[b, s, stop]);
            mlxcel_core::reshape(&part, &[b, s, heads, dim])
        };
        let q = split(0, self.key_dim, self.num_k_heads, self.head_k_dim);
        let k = split(
            self.key_dim,
            2 * self.key_dim,
            self.num_k_heads,
            self.head_k_dim,
        );
        let v = split(
            2 * self.key_dim,
            self.conv_dim,
            self.num_v_heads,
            self.head_v_dim,
        );
        let (q, k) = normalize_qk(&q, &k);

        let state = cache
            .as_ref()
            .and_then(|c| c.state_cache.as_ref())
            .filter(|state| mlxcel_core::array_shape(state)[0] == b)
            .map(|state| mlxcel_core::copy(state));
        let (out, new_state) = gated_delta_update_precise(
            &q,
            &k,
            &v,
            &a,
            &b_proj,
            &self.a_log,
            &self.dt_bias,
            state.as_deref(),
            None,
        );
        if let Some(c) = cache {
            c.state_cache = Some(new_state);
            c.advance(s);
        }

        let out = self.norm.forward(&out, &z);
        self.out_proj
            .forward(&mlxcel_core::reshape(&out, &[b, s, -1]))
    }
}
