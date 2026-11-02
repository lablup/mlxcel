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

use super::muse_glimmer_cache::MuseCache;
use super::muse_glimmer_config::MuseGlimmerTextConfig;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct MuseRmsNorm {
    weight: Option<UniquePtr<MlxArray>>,
    eps: f32,
    centered: bool,
}

impl MuseRmsNorm {
    pub fn no_weight(eps: f32) -> Self {
        Self {
            weight: None,
            eps,
            centered: false,
        }
    }

    pub fn standard(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self {
            weight: Some(weight),
            eps,
            centered: false,
        }
    }

    pub fn centered(weight: UniquePtr<MlxArray>, eps: f32) -> Self {
        Self {
            weight: Some(weight),
            eps,
            centered: true,
        }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match (&self.weight, self.centered) {
            (Some(weight), false) => mlxcel_core::fast_rms_norm(x, weight, self.eps),
            (Some(weight), true) => {
                let ones = mlxcel_core::ones(
                    &[mlxcel_core::array_shape(weight)[0]],
                    mlxcel_core::array_dtype(weight),
                );
                let adjusted = mlxcel_core::add(&ones, weight);
                mlxcel_core::fast_rms_norm(x, &adjusted, self.eps)
            }
            (None, _) => mlxcel_core::fast_rms_norm_no_weight(x, self.eps),
        }
    }
}

pub struct MuseGlimmerAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    gate_proj: UnifiedLinear,
    q_norm: MuseRmsNorm,
    k_norm: MuseRmsNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    /// `None` on a NoPE (full-attention) layer.
    rope_theta: Option<f32>,
    /// Position scale for the rotation (`1 / factor` for `linear`, else 1).
    rope_scale: f32,
    /// `llama3` / `yarn` frequency table from `rope_parameters` (#2384).
    rope_freqs: Option<UniquePtr<MlxArray>>,
    /// YaRN attention-magnitude multiplier for Q and K; 1.0 otherwise.
    rope_mscale: f32,
    sliding_window: i32,
}

impl MuseGlimmerAttention {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut MuseCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let offset = cache.offset();

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let (q, k) = match (self.rope_theta, self.rope_freqs.as_ref()) {
            (None, _) => (q, k),
            (Some(_), Some(freqs)) => {
                let (q, k) = if self.rope_mscale != 1.0 {
                    (
                        mlxcel_core::multiply_scalar(&q, self.rope_mscale),
                        mlxcel_core::multiply_scalar(&k, self.rope_mscale),
                    )
                } else {
                    (q, k)
                };
                (
                    mlxcel_core::fast_rope_with_freqs(
                        &q,
                        self.head_dim,
                        false,
                        self.rope_scale,
                        offset,
                        freqs,
                    ),
                    mlxcel_core::fast_rope_with_freqs(
                        &k,
                        self.head_dim,
                        false,
                        self.rope_scale,
                        offset,
                        freqs,
                    ),
                )
            }
            (Some(theta), None) => (
                mlxcel_core::fast_rope(&q, self.head_dim, false, theta, self.rope_scale, offset),
                mlxcel_core::fast_rope(&k, self.head_dim, false, theta, self.rope_scale, offset),
            ),
        };

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);
        let attn_out = if mask.is_none() {
            mlxcel_core::causal_attention(
                &q,
                &cache_k,
                &cache_v,
                self.scale,
                0.0,
                self.sliding_window,
            )
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            // SAFETY: q/cache_k/cache_v are MLX arrays with compatible
            // attention shapes, and mask_ptr is either null or points to the
            // mask borrowed for the duration of this call.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.sliding_window,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let gate = self.gate_proj.forward(x);
        let gate = mlxcel_core::reshape(&gate, &[b, l, self.num_heads, self.head_dim]);
        let gate = mlxcel_core::sigmoid(&gate);
        let attn_out = mlxcel_core::multiply(&attn_out, &gate);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerTextConfig,
        layer_idx: usize,
        prefix: &str,
        sliding_cache_window: i32,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();
        let rope = config.rope_scaling_kind_for_layer(layer_idx);
        let rope_theta = config
            .rope_theta_for_layer(layer_idx)
            .map(crate::models::rope_overrides::resolve_base);
        let rope_scale = rope.as_ref().map_or(1.0, |kind| kind.scale());
        let rope_mscale = rope.as_ref().map_or(1.0, |kind| kind.attn_scale());
        let rope_freqs = match rope {
            Some(
                crate::models::rope_utils::RopeScalingKind::Llama3 { freqs }
                | crate::models::rope_utils::RopeScalingKind::Yarn { freqs, .. },
            ) => Some(freqs),
            _ => None,
        };
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                group_size,
                bits,
            )?,
            k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.k_proj"),
                group_size,
                bits,
            )?,
            v_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.v_proj"),
                group_size,
                bits,
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.o_proj"),
                group_size,
                bits,
            )?,
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                group_size,
                bits,
            )?,
            q_norm: MuseRmsNorm::no_weight(config.rms_norm_eps),
            k_norm: MuseRmsNorm::no_weight(config.rms_norm_eps),
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            scale: (config.head_dim as f32).powf(-0.5) * config.qk_scale_factor,
            rope_theta,
            rope_scale,
            rope_freqs,
            rope_mscale,
            sliding_window: if config.is_sliding_layer(layer_idx) {
                sliding_cache_window
            } else {
                0
            },
        })
    }
}

pub struct MuseGlimmerMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MuseGlimmerMlp {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.up_proj"),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                group_size,
                bits,
            )?,
        })
    }
}

pub struct MuseGlimmerDecoderLayer {
    self_attn: MuseGlimmerAttention,
    mlp: MuseGlimmerMlp,
    input_layernorm: MuseRmsNorm,
    post_attention_layernorm: MuseRmsNorm,
    pre_feedforward_layernorm: MuseRmsNorm,
    post_feedforward_layernorm: MuseRmsNorm,
    pub(crate) use_sliding: bool,
}

impl MuseGlimmerDecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut MuseCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(x);
        let h = self.input_layernorm.forward(x);
        let h = self.self_attn.forward(&h, cache, mask);
        let h = self.post_attention_layernorm.forward(&h);
        let h = mlxcel_core::add(&residual, &h);

        let residual = mlxcel_core::copy(&h);
        let h = self.pre_feedforward_layernorm.forward(&h);
        let h = self.mlp.forward(&h);
        let h = self.post_feedforward_layernorm.forward(&h);
        mlxcel_core::add(&residual, &h)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerTextConfig,
        layer_idx: usize,
        model_prefix: &str,
        load_weight: impl Fn(&WeightMap, &str) -> Result<UniquePtr<MlxArray>, String>,
        sliding_cache_window: i32,
    ) -> Result<Self, String> {
        let prefix = format!("{model_prefix}.layers.{layer_idx}");
        let self_attn = MuseGlimmerAttention::from_weights(
            weights,
            config,
            layer_idx,
            &format!("{prefix}.self_attn"),
            sliding_cache_window,
        )?;
        let mlp = MuseGlimmerMlp::from_weights(weights, config, &format!("{prefix}.mlp"))?;
        let load = |suffix: &str| load_weight(weights, &format!("{prefix}.{suffix}.weight"));
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: MuseRmsNorm::centered(load("input_layernorm")?, config.rms_norm_eps),
            post_attention_layernorm: MuseRmsNorm::centered(
                load("post_attention_layernorm")?,
                config.post_norm_eps,
            ),
            pre_feedforward_layernorm: MuseRmsNorm::centered(
                load("pre_feedforward_layernorm")?,
                config.rms_norm_eps,
            ),
            post_feedforward_layernorm: MuseRmsNorm::centered(
                load("post_feedforward_layernorm")?,
                config.post_norm_eps,
            ),
            use_sliding: config.is_sliding_layer(layer_idx),
        })
    }
}
