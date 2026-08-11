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

//! Muse Glimmer vision transformer layer primitives.

use crate::models::muse_glimmer::MuseGlimmerVisionConfig;
use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::qwen2_vl::concat_many;

pub(crate) struct MuseVisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    proj: Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl MuseVisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            q_proj: load_linear(
                weights,
                &format!("{prefix}.attn.q_proj"),
                BiasMode::Required,
            )?,
            k_proj: load_linear(
                weights,
                &format!("{prefix}.attn.k_proj"),
                BiasMode::Required,
            )?,
            v_proj: load_linear(
                weights,
                &format!("{prefix}.attn.v_proj"),
                BiasMode::Required,
            )?,
            proj: load_linear(weights, &format!("{prefix}.attn.proj"), BiasMode::Required)?,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        rotary_freqs: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let seq_length = mlxcel_core::array_shape(hidden_states)[0];
        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);

        let q = mlxcel_core::reshape(&q, &[seq_length, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[seq_length, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[seq_length, self.num_heads, self.head_dim]);
        let q = apply_muse_vision_rope(&q, rotary_freqs);
        let k = apply_muse_vision_rope(&k, rotary_freqs);

        let q = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&q, &[1, 0, 2]), 0);
        let k = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&k, &[1, 0, 2]), 0);
        let v = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&v, &[1, 0, 2]), 0);
        let mut outputs = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));
        for pair in cu_seqlens.windows(2) {
            let start = pair[0];
            let end = pair[1];
            let q_seg = mlxcel_core::slice(
                &q,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let k_seg = mlxcel_core::slice(
                &k,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let v_seg = mlxcel_core::slice(
                &v,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            // SAFETY: q_seg, k_seg, and v_seg are freshly sliced MLX arrays
            // with matching [1, heads, seq, head_dim] shapes and no mask.
            outputs.push(unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_seg,
                    &k_seg,
                    &v_seg,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            });
        }

        let output = if outputs.len() == 1 {
            outputs.remove(0)
        } else {
            concat_many(&outputs, 2)
        };
        let output = mlxcel_core::squeeze_axis(&output, 0);
        let output = mlxcel_core::transpose_axes(&output, &[1, 0, 2]);
        let output = mlxcel_core::reshape(&output, &[seq_length, self.num_heads * self.head_dim]);
        self.proj.forward(&output)
    }
}

pub(crate) struct MuseVisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl MuseVisionMlp {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc1: load_linear(weights, &format!("{prefix}.mlp.fc1"), BiasMode::Required)?,
            fc2: load_linear(weights, &format!("{prefix}.mlp.fc2"), BiasMode::Required)?,
        })
    }

    fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(hidden_states);
        let h = mlxcel_core::gelu(&h);
        self.fc2.forward(&h)
    }
}

pub(crate) struct MuseVisionLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: MuseVisionAttention,
    mlp: MuseVisionMlp,
    window_attention: bool,
}

impl MuseVisionLayer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerVisionConfig,
        layer_idx: usize,
        root: &str,
    ) -> Result<Self, String> {
        let prefix = format!("{root}.layers.{layer_idx}");
        Ok(Self {
            norm1: load_layer_norm(weights, &format!("{prefix}.norm1"), config.layer_norm_eps)?,
            norm2: load_layer_norm(weights, &format!("{prefix}.norm2"), config.layer_norm_eps)?,
            attn: MuseVisionAttention::from_weights(weights, config, &prefix)?,
            mlp: MuseVisionMlp::from_weights(weights, &prefix)?,
            window_attention: config.is_window_layer(layer_idx),
        })
    }

    pub(crate) fn is_window_attention(&self) -> bool {
        self.window_attention
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        rotary_freqs: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.norm1.forward(hidden_states);
        let attn = self.attn.forward(&normed, cu_seqlens, rotary_freqs);
        let h = mlxcel_core::add(hidden_states, &attn);
        let normed = self.norm2.forward(&h);
        let mlp = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp)
    }
}

pub(crate) fn load_layer_norm(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight = get_weight_copy(weights, &format!("{prefix}.weight"))?;
    let bias = get_weight_copy(weights, &format!("{prefix}.bias"))?;
    Ok(LayerNorm::new(weight, Some(bias), eps))
}

pub(crate) fn load_linear(
    weights: &WeightMap,
    prefix: &str,
    bias_mode: BiasMode,
) -> Result<Linear, String> {
    ensure_dense_linear(weights, prefix)?;
    let linear = Linear::from_weights(weights, prefix)?;
    match (bias_mode, linear.bias.is_some()) {
        (BiasMode::Required, false) => {
            Err(format!("Muse Glimmer linear requires bias: {prefix}.bias"))
        }
        (BiasMode::Forbidden, true) => Err(format!(
            "Muse Glimmer linear must not have bias: {prefix}.bias"
        )),
        _ => Ok(linear),
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BiasMode {
    Required,
    Forbidden,
}

fn ensure_dense_linear(weights: &WeightMap, prefix: &str) -> Result<(), String> {
    for suffix in [".scales", ".biases", ".global_scale"] {
        let key = format!("{prefix}{suffix}");
        if weights.contains_key(&key) {
            return Err(format!(
                "Muse Glimmer phase 2 vision path only supports canonical dense BF16 linears; found quantization sidecar {key}"
            ));
        }
    }
    Ok(())
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

fn apply_muse_vision_rope(tensor: &MlxArray, freqs: &MlxArray) -> UniquePtr<MlxArray> {
    let orig_dtype = mlxcel_core::array_dtype(tensor);
    let tensor_f32 = mlxcel_core::astype(tensor, mlxcel_core::dtype::FLOAT32);
    let cos_vals = mlxcel_core::expand_dims(&mlxcel_core::cos(freqs), 1);
    let sin_vals = mlxcel_core::expand_dims(&mlxcel_core::sin(freqs), 1);
    let rotated = rotate_half(&tensor_f32);
    let term1 = mlxcel_core::multiply(&tensor_f32, &cos_vals);
    let term2 = mlxcel_core::multiply(&rotated, &sin_vals);
    mlxcel_core::astype(&mlxcel_core::add(&term1, &term2), orig_dtype)
}

fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let half = shape[ndim - 1] / 2;
    let mut starts = vec![0i32; ndim];
    let mut stops = shape.clone();
    stops[ndim - 1] = half;
    let x1 = mlxcel_core::slice(x, &starts, &stops);
    starts[ndim - 1] = half;
    stops[ndim - 1] = shape[ndim - 1];
    let x2 = mlxcel_core::slice(x, &starts, &stops);
    mlxcel_core::concatenate(&mlxcel_core::negative(&x2), &x1, ndim as i32 - 1)
}
