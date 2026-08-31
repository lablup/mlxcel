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

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::{InklingConfig, InklingLayerCache, InklingTextConfig, weight};
use crate::models::conv_decode::{build_conv_decode_weight, short_conv_decode_step};

pub(crate) struct InklingShortConv {
    weight: UniquePtr<MlxArray>,
    decode_weight: UniquePtr<MlxArray>,
    kernel: i32,
}

impl InklingShortConv {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        name: &str,
        kernel: usize,
    ) -> Result<Self, String> {
        let raw = weight(weights, name)?;
        let shape = mlxcel_core::array_shape(&raw);
        if shape.len() != 3 || shape[1] != kernel as i32 || shape[2] != 1 {
            return Err(format!(
                "{name}: expected [channels, {kernel}, 1], got {shape:?}"
            ));
        }
        let weight = mlxcel_core::astype(&raw, dtype::FLOAT32);
        let decode_weight = build_conv_decode_weight(&weight);
        Ok(Self {
            weight,
            decode_weight,
            kernel: kernel as i32,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
        residual: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let input_dtype = mlxcel_core::array_dtype(input);
        let input_f32 = mlxcel_core::astype(input, dtype::FLOAT32);
        let shape = mlxcel_core::array_shape(&input_f32);
        let keep = self.kernel - 1;
        let padded = match state.as_ref().and_then(|value| value.as_ref()) {
            Some(previous) => mlxcel_core::concatenate(previous, &input_f32, 1),
            None => {
                let zeros = mlxcel_core::zeros(&[shape[0], keep, shape[2]], dtype::FLOAT32);
                mlxcel_core::concatenate(&zeros, &input_f32, 1)
            }
        };
        let padded_len = mlxcel_core::array_shape(&padded)[1];
        let tail = mlxcel_core::utils::slice_axis(&padded, 1, padded_len - keep, padded_len);
        *state = Some(mlxcel_core::contiguous(&tail, false));

        let convolved = if shape[1] == 1 {
            short_conv_decode_step(&padded, &self.decode_weight, dtype::FLOAT32)
        } else {
            mlxcel_core::conv1d(&padded, &self.weight, 1, 0, 1, shape[2])
        };
        let inner = mlxcel_core::add(&convolved, &input_f32);
        let inner = mlxcel_core::astype(&inner, input_dtype);
        match residual {
            Some(residual) => mlxcel_core::add(residual, &inner),
            None => inner,
        }
    }
}

pub(crate) struct InklingAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    r_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    k_sconv: InklingShortConv,
    v_sconv: InklingShortConv,
    rel_proj: UniquePtr<MlxArray>,
    is_sliding: bool,
    n_heads: i32,
    n_kv: i32,
    head_dim: i32,
    d_rel: i32,
    rel_extent: i32,
    window: i32,
    log_floor: Option<f32>,
    log_alpha: f32,
}

impl InklingAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &InklingConfig,
        index: usize,
    ) -> Result<Self, String> {
        let text = &config.text_config;
        let prefix = format!("model.layers.{index}.self_attn");
        let is_sliding = text.layer_is_sliding(index);
        let (n_heads, n_kv, head_dim) = if is_sliding {
            (
                text.swa_num_attention_heads,
                text.swa_num_key_value_heads,
                text.swa_head_dim,
            )
        } else {
            (
                text.num_attention_heads,
                text.num_key_value_heads,
                text.head_dim,
            )
        };
        let (group, bits, _) = config.quantization();
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group, bits)
        };
        let rel_extent = if is_sliding {
            text.sliding_window_size
        } else {
            text.rel_extent
        };
        let rel_proj = weight(weights, &format!("{prefix}.rel_proj"))?;
        let rel_shape = mlxcel_core::array_shape(&rel_proj);
        if rel_shape != [text.d_rel as i32, rel_extent as i32] {
            return Err(format!(
                "{prefix}.rel_proj: expected [{}, {}], got {rel_shape:?}",
                text.d_rel, rel_extent
            ));
        }
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            r_proj: linear("r_proj")?,
            o_proj: linear("o_proj")?,
            q_norm: RMSNorm::new(
                weight(weights, &format!("{prefix}.q_norm.weight"))?,
                text.rms_norm_eps,
            ),
            k_norm: RMSNorm::new(
                weight(weights, &format!("{prefix}.k_norm.weight"))?,
                text.rms_norm_eps,
            ),
            k_sconv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.k_sconv.conv.weight"),
                text.sconv_kernel_size,
            )?,
            v_sconv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.v_sconv.conv.weight"),
                text.sconv_kernel_size,
            )?,
            rel_proj,
            is_sliding,
            n_heads: n_heads as i32,
            n_kv: n_kv as i32,
            head_dim: head_dim as i32,
            d_rel: text.d_rel as i32,
            rel_extent: rel_extent as i32,
            window: text.sliding_window_size as i32,
            log_floor: (!is_sliding)
                .then_some(text.log_scaling_n_floor)
                .flatten()
                .map(|v| v as f32),
            log_alpha: text.log_scaling_alpha,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut InklingLayerCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, length) = (shape[0], shape[1]);
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        let r = self.r_proj.forward(x);
        let k = self.k_sconv.forward(&k, &mut cache.conv[0], None);
        let v = self.v_sconv.forward(&v, &mut cache.conv[1], None);

        let q = mlxcel_core::reshape(&q, &[batch, length, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[batch, length, self.n_kv, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[batch, length, self.n_kv, self.head_dim]);
        let r = mlxcel_core::reshape(&r, &[batch, length, self.n_heads, self.d_rel]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        let (mut keys, mut values) = cache.kv.update_and_fetch(k, v);
        let before = mlxcel_core::array_shape(&keys)[2];
        if self.is_sliding && before > length + self.window - 1 {
            let excess = before - (length + self.window - 1);
            cache.kv.trim_front(excess);
            keys = mlxcel_core::utils::slice_axis(&keys, 2, excess, before);
            values = mlxcel_core::utils::slice_axis(&values, 2, excess, before);
        }
        let source = mlxcel_core::array_shape(&keys)[2];
        let offset = source - length;
        let mut mask = banded_additive_mask(
            &r,
            &self.rel_proj,
            offset,
            source,
            self.is_sliding.then_some(self.window),
            self.rel_extent,
        );
        if let Some(floor) = self.log_floor {
            let tau = log_scaling_tau(length, offset, floor, self.log_alpha);
            let tau4 = mlxcel_core::reshape(&tau, &[1, 1, length, 1]);
            let tau4 = mlxcel_core::astype(&tau4, mlxcel_core::array_dtype(&q));
            q = mlxcel_core::multiply(&q, &tau4);
            let threshold = mlxcel_core::full_like(&mask, -1e29);
            let valid = mlxcel_core::greater(&mask, &threshold);
            let scaled = mlxcel_core::multiply(&mask, &tau4);
            mask = mlxcel_core::where_cond(&valid, &scaled, &mask);
        }
        let mask = mlxcel_core::astype(&mask, mlxcel_core::array_dtype(&q));
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(
                &q,
                &keys,
                &values,
                1.0 / self.head_dim as f32,
                mask.as_ref().unwrap() as *const MlxArray,
            )
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[batch, length, self.n_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }
}

pub(crate) fn banded_additive_mask(
    r: &MlxArray,
    projection: &MlxArray,
    offset: i32,
    source: i32,
    sliding_window: Option<i32>,
    rel_extent: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(r);
    let (batch, length, heads) = (shape[0], shape[1], shape[2]);
    let relative = mlxcel_core::matmul(r, projection);
    let relative = mlxcel_core::transpose_axes(&relative, &[0, 2, 1, 3]);
    let queries = mlxcel_core::arange_i32(offset, offset + length, 1);
    let queries = mlxcel_core::reshape(&queries, &[length, 1]);
    let keys = mlxcel_core::arange_i32(0, source, 1);
    let keys = mlxcel_core::reshape(&keys, &[1, source]);
    let dist = mlxcel_core::subtract(&queries, &keys);
    let dist = mlxcel_core::reshape(&dist, &[1, 1, length, source]);
    let dist = mlxcel_core::broadcast_to(&dist, &[batch, heads, length, source]);
    let low = mlxcel_core::from_slice_i32(&[0], &[1]);
    let high = mlxcel_core::from_slice_i32(&[rel_extent - 1], &[1]);
    let gather = mlxcel_core::clip(&dist, &low, &high);
    let gathered = mlxcel_core::take_along_axis(&relative, &gather, -1);
    let extent = mlxcel_core::from_slice_i32(&[rel_extent], &[1]);
    let past_extent = mlxcel_core::greater_equal(&dist, &extent);
    let zeros = mlxcel_core::zeros_like(&gathered);
    let positional = mlxcel_core::where_cond(&past_extent, &zeros, &gathered);
    let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
    let future = mlxcel_core::less(&dist, &zero);
    let invalid = if let Some(window) = sliding_window {
        let window = mlxcel_core::from_slice_i32(&[window], &[1]);
        mlxcel_core::logical_or(&future, &mlxcel_core::greater_equal(&dist, &window))
    } else {
        future
    };
    let blocked = mlxcel_core::full_like(&positional, -1e30);
    mlxcel_core::where_cond(&invalid, &blocked, &positional)
}

pub(crate) fn log_scaling_tau(
    length: i32,
    offset: i32,
    floor: f32,
    alpha: f32,
) -> UniquePtr<MlxArray> {
    let positions = mlxcel_core::arange_f32((offset + 1) as f32, (offset + length + 1) as f32, 1.0);
    let ratio = mlxcel_core::divide_scalar(&positions, floor);
    let ratio = mlxcel_core::maximum(&ratio, &mlxcel_core::ones_like(&ratio));
    let scaled = mlxcel_core::multiply_scalar(&mlxcel_core::log(&ratio), alpha);
    mlxcel_core::add(&mlxcel_core::ones_like(&scaled), &scaled)
}

#[allow(dead_code)]
fn _config_anchor(_: &InklingTextConfig) {}
