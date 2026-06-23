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

//! GptOss MoE model implementation using mlxcel-core
//!
//! Key features:
//! - Sparse MoE with SwitchGLU experts and custom SwiGLU activation
//!   (alpha=1.702, clamp at swiglu_limit)
//! - Alternating sliding_attention / full_attention layers (from config layer_types)
//! - Per-head attention sinks (learned bias for first key position)
//! - YarnRoPE positional encoding
//! - RotatingKVCache for sliding layers, KVCache for full attention layers
//! - All projections use bias (attention_bias=True)
//!
//! Reference: mlx-lm gpt_oss.py

use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use crate::distributed::pipeline::LayerFilter;
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::partial_loading::filter_weight_map;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,

    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,

    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

/// Replace layer number with 0 for quantization config lookup
fn regex_replace_layer(prefix: &str) -> String {
    // "model.layers.15.self_attn.q_proj" -> "model.layers.0.self_attn.q_proj"
    if let Some(start) = prefix.find("layers.") {
        let rest = &prefix[start + 7..]; // after "layers."
        if let Some(dot) = rest.find('.') {
            return format!("{}layers.0{}", &prefix[..start], &rest[dot..]);
        }
    }
    prefix.to_string()
}

fn default_swiglu_limit() -> f32 {
    7.0
}

/// Per-component quantization config with per-layer overrides
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Quantization {
    /// Full quantization config with per-layer overrides
    Full(HashMap<String, serde_json::Value>),
}

impl Quantization {
    /// Get (group_size, bits, mode) for a specific weight prefix
    fn params_for(&self, prefix: &str) -> (i32, i32, String) {
        let Quantization::Full(map) = self;
        // Check for per-layer override (e.g., "model.layers.0.self_attn.q_proj")
        if let Some(v) = map.get(prefix)
            && let Some(obj) = v.as_object()
        {
            let gs = obj.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
            let b = obj.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;
            let mode = obj
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("affine")
                .to_string();
            return (gs, b, mode);
        }
        // Fall back to top-level defaults
        self.defaults()
    }

    /// Top-level defaults
    fn defaults(&self) -> (i32, i32, String) {
        let Quantization::Full(map) = self;
        let gs = map.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
        let b = map.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;
        let mode = map
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("affine")
            .to_string();
        (gs, b, mode)
    }
}

impl ModelArgs {
    /// Default group_size (top-level)
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.defaults().0)
            .unwrap_or(64)
    }

    /// Default bits (top-level)
    pub fn bits(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.defaults().1)
            .unwrap_or(4)
    }

    /// Get quantization params (group_size, bits, mode) for a specific weight prefix.
    /// Tries exact match first, then tries layer 0 as a pattern (since all layers
    /// in gpt_oss use the same quantization per component type).
    fn quant_for(&self, prefix: &str) -> (i32, i32, String) {
        self.quantization
            .as_ref()
            .map(|q| {
                // Try exact match first
                let result = q.params_for(prefix);
                // If no exact match (returns default), try substituting layer number with 0
                let default = q.defaults();
                if result == default {
                    // Replace ".layers.N." with ".layers.0." to find the pattern
                    let pattern = regex_replace_layer(prefix);
                    if pattern != prefix {
                        return q.params_for(&pattern);
                    }
                }
                result
            })
            .unwrap_or_else(|| (64, 4, "affine".to_string()))
    }

    /// Get the layer_types list, defaulting to alternating sliding/full pattern
    pub fn layer_types_list(&self) -> Vec<String> {
        self.layer_types.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|i| {
                    if i % 2 == 0 {
                        "sliding_attention".to_string()
                    } else {
                        "full_attention".to_string()
                    }
                })
                .collect()
        })
    }

    /// Compute YarnRoPE frequencies
    pub(crate) fn compute_yarn_freqs(&self) -> Option<(UniquePtr<MlxArray>, f32)> {
        let rope_scaling = self.rope_scaling.as_ref()?;
        let rope_type = rope_scaling
            .get("rope_type")
            .or_else(|| rope_scaling.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        if rope_type != "yarn" {
            return None;
        }

        let factor = rope_scaling
            .get("factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let original_max_pos = rope_scaling
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as f32;
        let beta_fast = rope_scaling
            .get("beta_fast")
            .and_then(|v| v.as_f64())
            .unwrap_or(32.0) as f32;
        let beta_slow = rope_scaling
            .get("beta_slow")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let mscale = rope_scaling
            .get("mscale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let mscale_all_dim = rope_scaling
            .get("mscale_all_dim")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let dims = self.head_dim as f32;
        let base = self.rope_theta;
        let half_dims = self.head_dim / 2;

        // yarn_find_correction_dim
        let find_correction_dim = |num_rotations: f32| -> f32 {
            (dims * (original_max_pos / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * base.ln())
        };

        // yarn_find_correction_range
        let low = find_correction_dim(beta_fast).floor().max(0.0) as usize;
        let high = find_correction_dim(beta_slow).ceil().min(dims - 1.0) as usize;

        // yarn_get_mscale
        let get_mscale = |scale: f32, ms: f32| -> f32 {
            if scale <= 1.0 {
                1.0
            } else {
                0.1 * ms * scale.ln() + 1.0
            }
        };

        let rope_mscale = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim);

        // Compute frequencies
        let mut freqs_data = vec![0.0f32; half_dims];
        for (i, freq_out) in freqs_data.iter_mut().enumerate().take(half_dims) {
            let freq_extra = base.powf((2 * i) as f32 / dims);
            let freq_inter = factor * freq_extra;

            // yarn_linear_ramp_mask
            let ramp_min = low as f32;
            let ramp_max = if high == low {
                high as f32 + 0.001
            } else {
                high as f32
            };
            let ramp = ((i as f32 - ramp_min) / (ramp_max - ramp_min)).clamp(0.0, 1.0);
            let freq_mask = 1.0 - ramp;

            // Interpolate: (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1 - freq_mask))
            *freq_out = (freq_inter * freq_extra)
                / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask));
        }

        let freqs = mlxcel_core::from_slice_f32(&freqs_data, &[half_dims as i32]);
        Some((freqs, rope_mscale))
    }
}

// Expert Linear Layer for GptOss
// Handles both affine quantization (with quant biases) and MXFP4 (E8M0 scales, no quant biases),
// plus optional per-expert linear bias.
enum ExpertLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        quant_biases: Option<UniquePtr<MlxArray>>,
        linear_bias: Option<UniquePtr<MlxArray>>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
        linear_bias: Option<UniquePtr<MlxArray>>,
    },
}

impl ExpertLinear {
    fn forward(&self, x: &MlxArray, indices: &MlxArray, sorted: bool) -> UniquePtr<MlxArray> {
        let out = match self {
            Self::Quantized {
                weight,
                scales,
                quant_biases,
                group_size,
                bits,
                mode,
                ..
            } => {
                let biases_ptr = quant_biases
                    .as_ref()
                    .map(|b| b.as_ref().unwrap() as *const _)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        biases_ptr,
                        std::ptr::null(), // lhs_indices
                        indices as *const _,
                        true,
                        *group_size,
                        *bits,
                        sorted,
                        mode,
                    )
                }
            }
            Self::Regular { weight, .. } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                unsafe {
                    mlxcel_core::gather_mm(x, &wt, std::ptr::null(), indices as *const _, sorted)
                }
            }
        };
        // Add per-expert linear bias if present
        let linear_bias = match self {
            Self::Quantized { linear_bias, .. } | Self::Regular { linear_bias, .. } => {
                linear_bias.as_ref()
            }
        };
        if let Some(bias) = linear_bias {
            let gathered_bias = mlxcel_core::take(bias, indices, 0);
            let gathered_bias = mlxcel_core::expand_dims(&gathered_bias, -2);
            mlxcel_core::add(&out, &gathered_bias)
        } else {
            out
        }
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing weight: {}", prefix))?;

        // Per-expert linear bias (e.g., gate_proj.bias [num_experts, out_dim])
        let linear_bias = weights
            .get(&format!("{}.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            let scales = weights
                .get(&scales_key)
                .map(|w| mlxcel_core::copy(w))
                .unwrap();

            // Quantized biases (only present for affine mode)
            let quant_biases = weights
                .get(&format!("{}.biases", prefix))
                .map(|w| mlxcel_core::copy(w));

            Ok(Self::Quantized {
                weight,
                scales,
                quant_biases,
                linear_bias,
                group_size,
                bits,
                mode: mode.to_string(),
            })
        } else {
            Ok(Self::Regular {
                weight,
                linear_bias,
            })
        }
    }
}

// GptOss custom SwiGLU activation
// swiglu(x_linear, x_glu, alpha=1.702, limit=7.0):
//   x_glu = clamp(x_glu, max=limit)
//   x_linear = clamp(x_linear, min=-limit, max=limit)
//   out_glu = x_glu * sigmoid(alpha * x_glu)
//   return out_glu * (x_linear + 1)
fn gpt_oss_swiglu(x_linear: &MlxArray, x_glu: &MlxArray, limit: f32) -> UniquePtr<MlxArray> {
    if (limit - 7.0).abs() <= f32::EPSILON {
        return mlxcel_core::compiled_gpt_oss_swiglu_activation(x_linear, x_glu);
    }

    let input_dtype = mlxcel_core::array_dtype(x_linear);
    let alpha = 1.702f32;

    // Clamp values
    let neg_limit = mlxcel_core::full_f32(&[1], -limit, input_dtype);
    let pos_limit = mlxcel_core::full_f32(&[1], limit, input_dtype);
    let x_glu = mlxcel_core::minimum(x_glu, &pos_limit);
    let x_linear = mlxcel_core::maximum(x_linear, &neg_limit);
    let x_linear = mlxcel_core::minimum(&x_linear, &pos_limit);

    // glu_scaled = alpha * x_glu -> sigmoid -> out_glu = x_glu * sig
    let alpha_arr = mlxcel_core::full_f32(&[1], alpha, input_dtype);
    let glu_scaled = mlxcel_core::multiply(&alpha_arr, &x_glu);
    let sig = mlxcel_core::sigmoid(&glu_scaled);
    let out_glu = mlxcel_core::multiply(&x_glu, &sig);

    // (x_linear + 1) * out_glu
    let one = mlxcel_core::full_f32(&[1], 1.0, input_dtype);
    let x_linear_plus_1 = mlxcel_core::add(&x_linear, &one);
    let result = mlxcel_core::multiply(&out_glu, &x_linear_plus_1);
    mlxcel_core::astype(&result, input_dtype)
}

// SwitchGLU for GptOss (custom activation, MXFP4 support)
struct GptOssSwitchGLU {
    gate_proj: ExpertLinear,
    up_proj: ExpertLinear,
    down_proj: ExpertLinear,
    swiglu_limit: f32,
}

impl GptOssSwitchGLU {
    fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];
        let total = n_tokens * top_k;
        let do_sort = total >= 64;
        let hidden_size = mlxcel_core::array_shape(x)[1];

        // Python writes this as `mx.expand_dims(x, (-2, -3))`, producing
        // [tokens, 1, 1, hidden]. The input here is already flattened to rank
        // 2, so a reshape is equivalent and avoids two decode-hot shape ops.
        let x_exp = mlxcel_core::reshape(x, &[n_tokens, 1, 1, hidden_size]);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) =
                crate::models::switch_layers::gather_sort(&x_exp, indices);
            let x_up = self.up_proj.forward(&sorted_x, &sorted_idx, true);
            let x_gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            // Python SwitchGLU: activation(x_up, x_gate) → swiglu(x_linear=x_up, x_glu=x_gate)
            let activated = gpt_oss_swiglu(&x_up, &x_gate, self.swiglu_limit);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);
            scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            let x_up = self.up_proj.forward(&x_exp, indices, false);
            let x_gate = self.gate_proj.forward(&x_exp, indices, false);
            // Python SwitchGLU: activation(x_up, x_gate) → swiglu(x_linear=x_up, x_glu=x_gate)
            let activated = gpt_oss_swiglu(&x_up, &x_gate, self.swiglu_limit);
            let output = self.down_proj.forward(&activated, indices, false);
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        swiglu_limit: f32,
        mode: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: ExpertLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
                mode,
            )?,
            up_proj: ExpertLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
                mode,
            )?,
            down_proj: ExpertLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
                mode,
            )?,
            swiglu_limit,
        })
    }
}

/// Unsort tokens back to original order
fn scatter_unsort(x: &MlxArray, inv_order: &MlxArray, orig_shape: &[i32]) -> UniquePtr<MlxArray> {
    let unsorted = mlxcel_core::take(x, inv_order, 0);
    let x_shape = mlxcel_core::array_shape(&unsorted);
    let n_tokens = orig_shape[0];
    let top_k = orig_shape[1];
    let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
    mlxcel_core::squeeze_axis(&reshaped, 2)
}

// MoE MLP Block.
struct MLPBlock {
    router: UnifiedLinear,
    experts: GptOssSwitchGLU,
    num_experts_per_tok: usize,
}

impl MLPBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Router logits
        let logits = self.router.forward(&x_flat);

        // Top-k selection using argpartition (emulates torch.topk)
        let k = self.num_experts_per_tok as i32;
        let n_experts = mlxcel_core::array_shape(&logits)[1];
        let kth = n_experts - k;

        let indices = mlxcel_core::argpartition(&logits, kth, -1);
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // Softmax over top-k logits
        let topk_logits = mlxcel_core::take_along_axis(&logits, &topk_indices, -1);
        let scores = mlxcel_core::softmax_precise(&topk_logits, -1);

        // Apply experts -> [n_tokens, k, hidden]
        let expert_out = self.experts.forward(&x_flat, &topk_indices);

        let result = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Reshape back
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        // Router has its own quantization params (typically group_size=64, bits=8)
        let router_prefix = format!("{}.router", prefix);
        let (r_gs, r_bits, r_mode) = args.quant_for(&router_prefix);
        let router =
            UnifiedLinear::from_weights_with_mode(weights, &router_prefix, r_gs, r_bits, &r_mode)?;

        // Experts use the default quantization mode (typically MXFP4 for this model)
        let (exp_gs, exp_bits, exp_mode) = args
            .quantization
            .as_ref()
            .map(|q| q.defaults())
            .unwrap_or((64, 4, "affine".to_string()));
        let experts = GptOssSwitchGLU::from_weights(
            weights,
            &format!("{}.experts", prefix),
            exp_gs,
            exp_bits,
            args.swiglu_limit,
            &exp_mode,
        )?;

        Ok(Self {
            router,
            experts,
            num_experts_per_tok: args.num_experts_per_tok,
        })
    }
}

// Attention with sinks.
struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    sinks: UniquePtr<MlxArray>,
    rope_freqs: Option<UniquePtr<MlxArray>>,
    rope_mscale: f32,
    rope_dims: i32,
    rope_base: f32,
}

impl Attention {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset();

        // Apply RoPE (with optional Yarn frequencies and mscale)
        let (q, k) = if let Some(ref freqs) = self.rope_freqs {
            // YarnRoPE: scale Q/K by mscale before applying RoPE with custom frequencies
            let q = if (self.rope_mscale - 1.0).abs() > 1e-6 {
                // Python: x[..., :dims] = mscale * x[..., :dims]
                // Since rope_dims == head_dim, this scales all of q
                mlxcel_core::multiply_scalar(&q, self.rope_mscale)
            } else {
                q
            };
            let k = if (self.rope_mscale - 1.0).abs() > 1e-6 {
                mlxcel_core::multiply_scalar(&k, self.rope_mscale)
            } else {
                k
            };
            let q =
                mlxcel_core::fast_rope_with_freqs(&q, self.rope_dims, false, 1.0, offset, freqs);
            let k =
                mlxcel_core::fast_rope_with_freqs(&k, self.rope_dims, false, 1.0, offset, freqs);
            (q, k)
        } else {
            let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
            let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);
            (q, k)
        };

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Fast SDPA with sinks support (MLX kernel-fused path)
        // Used by: GptOss
        let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
        let sinks_ptr = self.sinks.as_ref().unwrap() as *const _;
        let attn_out = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
                &q, &cache_k, &cache_v, self.scale, mask_ptr, sinks_ptr,
            )
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        rope_freqs: Option<&MlxArray>,
        rope_mscale: f32,
    ) -> Result<Self, String> {
        // Use per-component quantization params (attention uses affine, not MXFP4)
        let q_prefix = format!("{}.q_proj", prefix);
        let (q_gs, q_bits, q_mode) = args.quant_for(&q_prefix);
        let q_proj =
            UnifiedLinear::from_weights_with_mode(weights, &q_prefix, q_gs, q_bits, &q_mode)?;

        let k_prefix = format!("{}.k_proj", prefix);
        let (k_gs, k_bits, k_mode) = args.quant_for(&k_prefix);
        let k_proj =
            UnifiedLinear::from_weights_with_mode(weights, &k_prefix, k_gs, k_bits, &k_mode)?;

        let v_prefix = format!("{}.v_proj", prefix);
        let (v_gs, v_bits, v_mode) = args.quant_for(&v_prefix);
        let v_proj =
            UnifiedLinear::from_weights_with_mode(weights, &v_prefix, v_gs, v_bits, &v_mode)?;

        let o_prefix = format!("{}.o_proj", prefix);
        let (o_gs, o_bits, o_mode) = args.quant_for(&o_prefix);
        let o_proj =
            UnifiedLinear::from_weights_with_mode(weights, &o_prefix, o_gs, o_bits, &o_mode)?;

        let head_dim = args.head_dim as i32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Load sinks (per-head, shape [num_attention_heads])
        let sinks = weights
            .get(&format!("{}.sinks", prefix))
            .map(|w| mlxcel_core::copy(w))
            .unwrap_or_else(|| {
                mlxcel_core::zeros(
                    &[args.num_attention_heads as i32],
                    mlxcel_core::dtype::FLOAT32,
                )
            });

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale,
            sinks,
            rope_freqs: rope_freqs.map(mlxcel_core::copy),
            rope_mscale,
            rope_dims: head_dim,
            rope_base: args.rope_theta,
        })
    }
}

// Transformer Block.
pub(crate) struct TransformerBlock {
    self_attn: Attention,
    mlp: MLPBlock,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(x);
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(&residual, &attn_out);

        let residual = mlxcel_core::copy(&h);
        let normed = self.post_attention_layernorm.forward(&h);
        let moe_out = self.mlp.forward(&normed);
        mlxcel_core::add(&residual, &moe_out)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        rope_freqs: Option<&MlxArray>,
        rope_mscale: f32,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(
            weights,
            args,
            &format!("{}.self_attn", prefix),
            rope_freqs,
            rope_mscale,
        )?;
        let mlp = MLPBlock::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Cache Interface (same pattern as Gemma3).
// Used by: GptOss
pub(crate) trait CacheInterface {
    fn offset(&self) -> i32;
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);
}

impl CacheInterface for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

impl CacheInterface for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }
    fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(k, v)
    }
}

pub(crate) enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    pub(crate) fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Cache::Standard(c) => c,
            Cache::Rotating(c) => c,
        }
    }
}

pub(crate) fn gpt_oss_cache_offset(cache: &Cache) -> i32 {
    match cache {
        Cache::Standard(cache) => cache.offset,
        Cache::Rotating(cache) => cache.offset,
    }
}

// GptOss Model.
pub struct GptOssModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    layer_types: Vec<String>,
    sliding_window: usize,
}

impl GptOssModel {
    fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[1];

        let mut h = self.embed_tokens.forward(input_ids);

        // Find indices for full and sliding attention layers
        let full_idx = self
            .layer_types
            .iter()
            .position(|t| t == "full_attention")
            .unwrap_or(1);
        let swa_idx = self
            .layer_types
            .iter()
            .position(|t| t == "sliding_attention")
            .unwrap_or(0);

        // Python create_attention_mask returns None for single-token input (N=1)
        // Only create masks for multi-token prefill
        if seq_len > 1 {
            let global_offset = caches[full_idx].as_interface().offset();
            let full_mask = create_causal_mask(seq_len, global_offset);

            let sliding_offset = caches[swa_idx].as_interface().offset();
            let max_cache = self.sliding_window as i32;
            // Full-width windowed mask for a fresh single-pass prefill that
            // exceeds the window (RotatingKVCache keeps all prefill keys),
            // clamped mask otherwise. See issue #408.
            let sliding_mask =
                create_sliding_window_prefill_mask(seq_len, sliding_offset, max_cache);

            for (i, layer) in self.layers.iter().enumerate() {
                let mask = if self.layer_types[i] == "full_attention" {
                    Some(full_mask.as_ref().unwrap() as &MlxArray)
                } else {
                    Some(sliding_mask.as_ref().unwrap() as &MlxArray)
                };
                h = layer.forward(&h, caches[i].as_interface(), mask);
            }
        } else {
            // Single token: no mask needed (Python returns None for N=1)
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h, caches[i].as_interface(), None);
            }
        }

        let h = self.norm.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    fn make_caches(&self) -> Vec<Cache> {
        self.layer_types
            .iter()
            .map(|lt| {
                if lt == "full_attention" {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                }
            })
            .collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        // Embedding may have different quantization than the top-level default
        let (embed_gs, embed_bits, _embed_mode) = args.quant_for("model.embed_tokens");
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", embed_gs, embed_bits)?;

        // Compute Yarn RoPE frequencies
        let yarn_result = args.compute_yarn_freqs();
        let (yarn_freqs_ref, rope_mscale) = match &yarn_result {
            Some((freqs, mscale)) => (Some(freqs.as_ref().unwrap() as &MlxArray), *mscale),
            None => (None, 1.0),
        };

        let layer_types = args.layer_types_list();

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer =
                TransformerBlock::from_weights(weights, args, i, yarn_freqs_ref, rope_mscale)?;
            layers.push(layer);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        let lm_head = if !args.tie_word_embeddings {
            let (lm_gs, lm_bits, lm_mode) = args.quant_for("lm_head");
            Some(UnifiedLinear::from_weights_with_mode(
                weights, "lm_head", lm_gs, lm_bits, &lm_mode,
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            layer_types,
            sliding_window: args.sliding_window,
        })
    }
}

// Wrapper for LanguageModel trait (manages internal mixed caches).
// Used by: LoadedModel::GptOss
pub struct GptOssWrapper {
    model: GptOssModel,
    caches: RefCell<Vec<Cache>>,
}

impl GptOssWrapper {
    pub fn new(model: GptOssModel) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            caches: RefCell::new(caches),
        }
    }

    fn reset_caches(&self) {
        *self.caches.borrow_mut() = self.model.make_caches();
    }
}

impl mlxcel_core::generate::LanguageModel for GptOssWrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut caches = self.caches.borrow_mut();
        self.model.forward_with_caches(input_ids, &mut caches)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.reset_caches();
        // Return dummy caches (internal mixed caches are used instead)
        (0..self.model.layers.len())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn supports_batching(&self) -> bool {
        false // Mixed internal caches not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![200002] // gpt_oss eos_token_id
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

pub(crate) struct GptOssStageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    layers: Vec<TransformerBlock>,
    norm: Option<RMSNorm>,
    lm_head: Option<UnifiedLinear>,
    layer_types: Vec<String>,
    sliding_window: usize,
}

impl GptOssStageModel {
    pub(crate) fn load(
        model_dir: &Path,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse {}: {}", config_path.display(), e))?;

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        let mut effective_filter = filter.clone();
        if args.tie_word_embeddings && filter.has_lm_head {
            effective_filter.has_embedding = true;
        }
        filter_weight_map(&mut weights, &effective_filter);
        Self::from_filtered_weights(&weights, &args, filter, stage_index)
    }

    fn from_filtered_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let (embed_gs, embed_bits, _embed_mode) = args.quant_for("model.embed_tokens");
        let embed_tokens = if filter.has_embedding {
            Some(UnifiedEmbedding::from_weights(
                weights,
                "model.embed_tokens",
                embed_gs,
                embed_bits,
            )?)
        } else {
            None
        };

        let yarn_result = args.compute_yarn_freqs();
        let (yarn_freqs_ref, rope_mscale) = match &yarn_result {
            Some((freqs, mscale)) => (Some(freqs.as_ref().unwrap() as &MlxArray), *mscale),
            None => (None, 1.0),
        };

        let layer_types = args.layer_types_list()[filter.layer_range.clone()].to_vec();

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            let layer = TransformerBlock::from_weights(
                weights,
                args,
                layer_idx,
                yarn_freqs_ref,
                rope_mscale,
            )?;
            layers.push(layer);
        }

        if layers.is_empty() {
            return Err(format!(
                "stage {} did not load any layers from range {}..{}",
                stage_index, filter.layer_range.start, filter.layer_range.end
            ));
        }

        let (norm, lm_head) = if filter.has_lm_head {
            let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
            let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);
            let lm_head = if !args.tie_word_embeddings {
                let (lm_gs, lm_bits, lm_mode) = args.quant_for("lm_head");
                Some(UnifiedLinear::from_weights_with_mode(
                    weights, "lm_head", lm_gs, lm_bits, &lm_mode,
                )?)
            } else {
                None
            };
            (Some(norm), lm_head)
        } else {
            (None, None)
        };

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            layer_types,
            sliding_window: args.sliding_window,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.layer_types
            .iter()
            .map(|lt| {
                if lt == "full_attention" {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                }
            })
            .collect()
    }

    pub(crate) fn execute_from_token_ids(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        let hidden = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| {
                "stage does not host embeddings; hidden-state input required".to_string()
            })?
            .forward(input_ids);
        self.execute_hidden(hidden, caches)
    }

    pub(crate) fn execute_from_hidden_states(
        &self,
        hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if self.filter.has_embedding {
            return Err("entry stage expects token IDs, not hidden states".to_string());
        }
        self.execute_hidden(hidden, caches)
    }

    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if caches.len() != self.layers.len() {
            return Err(format!(
                "stage cache count mismatch: expected {}, got {}",
                self.layers.len(),
                caches.len()
            ));
        }

        let seq_len = mlxcel_core::array_shape(hidden.as_ref().unwrap())[1];
        let full_mask = if seq_len > 1 {
            self.first_layer_offset(caches, "full_attention")
                .map(|offset| create_causal_mask(seq_len, offset))
        } else {
            None
        };
        let sliding_mask = if seq_len > 1 {
            self.first_layer_offset(caches, "sliding_attention")
                .map(|offset| {
                    // See issue #408: full-width windowed mask for a fresh
                    // >window prefill, clamped otherwise.
                    let max_cache = self.sliding_window as i32;
                    create_sliding_window_prefill_mask(seq_len, offset, max_cache)
                })
        } else {
            None
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            let mask = match self.layer_types[idx].as_str() {
                "full_attention" => full_mask.as_deref(),
                "sliding_attention" => sliding_mask.as_deref(),
                _ => None,
            };
            hidden = layer.forward(hidden.as_ref().unwrap(), caches[idx].as_interface(), mask);
        }

        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                let hidden = norm.forward(hidden.as_ref().unwrap());
                Ok(StageExecutionOutput::Logits(lm_head.forward(&hidden)))
            }
            (Some(norm), None) if self.filter.has_lm_head => {
                let hidden = norm.forward(hidden.as_ref().unwrap());
                let embed_tokens = self
                    .embed_tokens
                    .as_ref()
                    .ok_or_else(|| "tied-word-embedding stage missing embeddings".to_string())?;
                Ok(StageExecutionOutput::Logits(
                    embed_tokens.as_linear(&hidden),
                ))
            }
            _ => Ok(StageExecutionOutput::HiddenStates(hidden)),
        }
    }

    fn first_layer_offset(&self, caches: &mut [Cache], layer_type: &str) -> Option<i32> {
        self.layer_types
            .iter()
            .position(|kind| kind == layer_type)
            .map(|idx| caches[idx].as_interface().offset())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_oss_swiglu_fallback_preserves_bf16_and_f16_dtype() {
        for dtype in [mlxcel_core::dtype::BFLOAT16, mlxcel_core::dtype::FLOAT16] {
            let x_linear = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&[-4.0, -1.0, 2.0, 4.0], &[1, 4]),
                dtype,
            );
            let x_glu = mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&[-2.0, 0.5, 2.0, 5.0], &[1, 4]),
                dtype,
            );

            let out = gpt_oss_swiglu(&x_linear, &x_glu, 6.0);
            mlxcel_core::eval(&out);

            assert_eq!(mlxcel_core::array_shape(&out), vec![1, 4]);
            assert_eq!(mlxcel_core::array_dtype(&out), dtype);
        }
    }
}
