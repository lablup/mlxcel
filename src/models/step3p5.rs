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

//! Step 3.5 model implementation using mlxcel-core
//!
//! Key features:
//! - ZeroCenteredRMSNorm (equivalent to regular RMSNorm at runtime)
//! - Head-wise attention gate (g_proj with sigmoid)
//! - ClampedSwiGLU activation with per-layer limits
//! - Sigmoid-based MoE gate with router_bias
//! - Per-layer sliding window vs full attention
//! - Per-layer rope_theta and partial_rotary_factor
//! - SwitchGLU experts with shared expert

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Step3p5Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_attention_groups: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// Can be a single float or an array of floats (per-layer)
    #[serde(default = "default_rope_theta_value")]
    pub rope_theta: serde_json::Value,

    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,

    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    #[serde(default)]
    pub yarn_only_types: Option<Vec<String>>,

    #[serde(default)]
    pub partial_rotary_factors: Option<Vec<f64>>,

    #[serde(default)]
    pub attention_other_setting: Option<AttentionOtherSetting>,

    #[serde(default = "default_true")]
    pub use_head_wise_attn_gate: bool,

    #[serde(default = "default_moe_num_experts")]
    pub moe_num_experts: usize,

    #[serde(default = "default_moe_top_k")]
    pub moe_top_k: usize,

    #[serde(default)]
    pub moe_intermediate_size: usize,

    #[serde(default)]
    pub share_expert_dim: usize,

    #[serde(default)]
    pub moe_layers_enum: Option<String>,

    #[serde(default = "default_moe_router_scaling")]
    pub moe_router_scaling_factor: f32,

    #[serde(default = "default_true")]
    pub norm_expert_weight: bool,

    /// Per-layer SwiGLU limits for routed experts (nullable entries → 0.0)
    #[serde(default)]
    pub swiglu_limits: serde_json::Value,

    /// Per-layer SwiGLU limits for shared experts and dense MLP (nullable entries → 0.0)
    #[serde(default)]
    pub swiglu_limits_shared: serde_json::Value,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttentionOtherSetting {
    pub num_attention_heads: usize,
    pub num_attention_groups: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "step3p5".to_string()
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta_value() -> serde_json::Value {
    serde_json::Value::Number(serde_json::Number::from_f64(10000.0).unwrap())
}
fn default_max_position_embeddings() -> usize {
    262144
}
fn default_sliding_window() -> i32 {
    512
}
fn default_true() -> bool {
    true
}
fn default_moe_num_experts() -> usize {
    288
}
fn default_moe_top_k() -> usize {
    8
}
fn default_moe_router_scaling() -> f32 {
    3.0
}

impl Step3p5Config {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// Get rope_theta for a given layer (handles scalar or per-layer array)
    pub fn get_rope_theta(&self, layer_idx: usize) -> f32 {
        match &self.rope_theta {
            serde_json::Value::Number(n) => n.as_f64().unwrap_or(10000.0) as f32,
            serde_json::Value::Array(arr) => arr
                .get(layer_idx)
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32,
            _ => 10000.0,
        }
    }

    pub fn get_partial_rotary_factor(&self, layer_idx: usize) -> f64 {
        self.partial_rotary_factors
            .as_ref()
            .and_then(|v| v.get(layer_idx).copied())
            .unwrap_or(1.0)
    }

    /// Get swiglu limit for routed experts at given layer
    pub fn get_swiglu_limit(&self, layer_idx: usize) -> f32 {
        get_limit_from_value(&self.swiglu_limits, layer_idx)
    }

    /// Get swiglu limit for shared experts / dense MLP at given layer
    pub fn get_swiglu_limit_shared(&self, layer_idx: usize) -> f32 {
        get_limit_from_value(&self.swiglu_limits_shared, layer_idx)
    }

    pub fn get_layer_type(&self, layer_idx: usize) -> &str {
        self.layer_types
            .as_ref()
            .and_then(|v| v.get(layer_idx).map(|s| s.as_str()))
            .unwrap_or("full_attention")
    }

    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        if let Some(ref layer_types) = self.layer_types {
            layer_types
                .get(layer_idx)
                .map(|t| t == "sliding_attention")
                .unwrap_or(false)
        } else {
            layer_idx.is_multiple_of(2)
        }
    }

    /// Get (num_heads, num_kv_heads) for a given layer
    pub fn attention_heads_for_layer(&self, layer_idx: usize) -> (usize, usize) {
        if self.is_sliding(layer_idx)
            && let Some(ref other) = self.attention_other_setting
        {
            return (other.num_attention_heads, other.num_attention_groups);
        }
        (self.num_attention_heads, self.num_attention_groups)
    }

    /// Get the set of MoE layer indices
    pub fn moe_layer_indices(&self) -> HashSet<usize> {
        if let Some(ref enum_str) = self.moe_layers_enum {
            enum_str
                .trim()
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect()
        } else {
            // Default: all layers except 0
            (1..self.num_hidden_layers).collect()
        }
    }
}

fn get_limit_from_value(val: &serde_json::Value, layer_idx: usize) -> f32 {
    match val {
        serde_json::Value::Array(arr) if layer_idx < arr.len() => {
            arr[layer_idx].as_f64().unwrap_or(0.0) as f32
        }
        _ => 0.0,
    }
}

/// Parse eos_token_id from config JSON (handles both int and array)
fn parse_eos_token_ids(config: &serde_json::Value) -> Vec<i32> {
    if let Some(id) = config.get("eos_token_id") {
        match id {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    return vec![i as i32];
                }
            }
            serde_json::Value::Array(arr) => {
                return arr
                    .iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect();
            }
            _ => {}
        }
    }
    vec![2] // Default EOS
}

// Clamped SwiGLU activation.
/// clip(silu(gate), max=limit) * clip(x, -limit, limit)
fn clamped_swiglu(x: &MlxArray, gate: &MlxArray, limit: f32) -> UniquePtr<MlxArray> {
    let silu_gate = mlxcel_core::silu(gate);
    let x_dtype = mlxcel_core::array_dtype(x);
    let limit_arr = mlxcel_core::full_f32(&[1], limit, x_dtype);
    let neg_limit_arr = mlxcel_core::full_f32(&[1], -limit, x_dtype);
    let clamped_gate = mlxcel_core::minimum(&silu_gate, &limit_arr);
    let clamped_x = mlxcel_core::clip(x, &neg_limit_arr, &limit_arr);
    mlxcel_core::multiply(&clamped_gate, &clamped_x)
}

// Attention.
pub struct Step3p5Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    g_proj: Option<UnifiedLinear>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    rope_base: f32,
    scale: f32,
    window_size: i32,
}

impl Step3p5Attention {
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

        // Reshape and apply Q/K norms per-head
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let q = self.q_norm.forward(&q);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let k = self.k_norm.forward(&k);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);

        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Apply RoPE
        let offset = cache.offset();
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, true, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, true, self.rope_base, 1.0, offset);

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &cache_k,
                    &cache_v,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.window_size,
                )
            }
        } else {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, self.window_size)
        };

        // Transpose back: [B, H, L, D] -> [B, L, H, D]
        let mut output = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);

        // Head-wise attention gate
        if let Some(ref g_proj) = self.g_proj {
            let g = g_proj.forward(x); // [B, L, num_heads]
            let g = mlxcel_core::sigmoid(&g);
            let g = mlxcel_core::expand_dims(&g, -1); // [B, L, num_heads, 1]
            output = mlxcel_core::multiply(&output, &g);
        }

        // Reshape and output projection
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Step3p5Config,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let (num_heads, num_kv_heads) = args.attention_heads_for_layer(layer_idx);

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;
        let q_norm = RMSNorm::new(q_norm_weight, args.rms_norm_eps);
        let k_norm = RMSNorm::new(k_norm_weight, args.rms_norm_eps);

        let g_proj = if args.use_head_wise_attn_gate {
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{}.g_proj", prefix),
                group_size,
                bits,
            )?)
        } else {
            None
        };

        let partial_rotary_factor = args.get_partial_rotary_factor(layer_idx);
        let rope_dims = (args.head_dim as f64 * partial_rotary_factor) as i32;
        let rope_base = args.get_rope_theta(layer_idx);
        let scale = (args.head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            g_proj,
            num_heads: num_heads as i32,
            num_kv_heads: num_kv_heads as i32,
            head_dim: args.head_dim as i32,
            rope_dims,
            rope_base,
            scale,
            window_size: if args.is_sliding(layer_idx) {
                args.sliding_window
            } else {
                0
            },
        })
    }
}

// Dense MLP (with optional clamped SwiGLU).
pub struct Step3p5MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    swiglu_limit: Option<f32>,
}

impl Step3p5MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        let activated = if let Some(limit) = self.swiglu_limit {
            clamped_swiglu(&up, &gate, limit)
        } else {
            mlxcel_core::compiled_swiglu_activation(&gate, &up)
        };

        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Step3p5Config,
        prefix: &str,
        swiglu_limit: f32,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        let limit = if swiglu_limit > 0.0 {
            Some(swiglu_limit)
        } else {
            None
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            swiglu_limit: limit,
        })
    }
}

// MoE Gate (sigmoid-based with router_bias).
pub struct Step3p5MoEGate {
    gate: UnifiedLinear,
    router_bias: UniquePtr<MlxArray>,
    top_k: i32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
}

impl Step3p5MoEGate {
    /// Returns (topk_indices, topk_weights)
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let gates = self.gate.forward(x);

        // Sigmoid scoring (in native dtype — f16 sigmoid saturation at |x|>5.5
        // preserves relative ordering for top-k selection, safe for MoE gating)
        let scores = mlxcel_core::sigmoid(&gates);

        // Add router_bias for selection
        let corrected_scores = mlxcel_core::add(&scores, &self.router_bias);

        // Get top-k indices from corrected scores
        let neg_corrected = mlxcel_core::negative(&corrected_scores);
        let all_indices = mlxcel_core::argpartition(&neg_corrected, self.top_k - 1, -1);

        // Slice last dim to get top-k
        let shape = mlxcel_core::array_shape(&all_indices);
        let ndim = shape.len();
        let starts: Vec<i32> = vec![0; ndim];
        let mut ends: Vec<i32> = shape.to_vec();
        ends[ndim - 1] = self.top_k;
        let topk_indices = mlxcel_core::slice(&all_indices, &starts, &ends);

        // Get weights from original (uncorrected) scores
        let topk_weights = mlxcel_core::take_along_axis(&scores, &topk_indices, -1);

        // Normalize if configured
        let topk_weights = if self.norm_topk_prob {
            let w_dtype = mlxcel_core::array_dtype(&topk_weights);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, w_dtype);
            let sum = mlxcel_core::sum_axis(&topk_weights, -1, true);
            let sum_eps = mlxcel_core::add(&sum, &eps);
            mlxcel_core::divide(&topk_weights, &sum_eps)
        } else {
            topk_weights
        };

        // Scale
        let scale = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::array_dtype(&topk_weights),
        );
        let topk_weights = mlxcel_core::multiply(&topk_weights, &scale);

        (topk_indices, topk_weights)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Step3p5Config,
        prefix: &str,
    ) -> Result<Self, String> {
        // MoE gate uses 8-bit quantization, group_size 64
        let gate = UnifiedLinear::from_weights(weights, &format!("{}.gate", prefix), 64, 8)?;
        let router_bias = get_weight_copy(weights, &format!("{}.router_bias", prefix))?;

        Ok(Self {
            gate,
            router_bias,
            top_k: args.moe_top_k as i32,
            routed_scaling_factor: args.moe_router_scaling_factor,
            norm_topk_prob: args.norm_expert_weight,
        })
    }
}

// SwitchGLU (MoE experts with optional clamped activation).
pub struct Step3p5SwitchGLU {
    gate_weight: UniquePtr<MlxArray>,
    gate_scales: UniquePtr<MlxArray>,
    gate_biases: UniquePtr<MlxArray>,
    up_weight: UniquePtr<MlxArray>,
    up_scales: UniquePtr<MlxArray>,
    up_biases: UniquePtr<MlxArray>,
    down_weight: UniquePtr<MlxArray>,
    down_scales: UniquePtr<MlxArray>,
    down_biases: UniquePtr<MlxArray>,
    group_size: i32,
    bits: i32,
    swiglu_limit: Option<f32>,
}

impl Step3p5SwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        // Expand x for gather_qmm
        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

        let gate_bias_ptr = self
            .gate_biases
            .as_ref()
            .map(|b| b as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let up_bias_ptr = self
            .up_biases
            .as_ref()
            .map(|b| b as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let down_bias_ptr = self
            .down_biases
            .as_ref()
            .map(|b| b as *const MlxArray)
            .unwrap_or(std::ptr::null());

        // Gate projection
        let x_gate = unsafe {
            mlxcel_core::gather_qmm(
                &x_exp,
                &self.gate_weight,
                &self.gate_scales,
                gate_bias_ptr,
                std::ptr::null(),
                indices as *const _,
                true,
                self.group_size,
                self.bits,
                false,
                "affine",
            )
        };

        // Up projection
        let x_up = unsafe {
            mlxcel_core::gather_qmm(
                &x_exp,
                &self.up_weight,
                &self.up_scales,
                up_bias_ptr,
                std::ptr::null(),
                indices as *const _,
                true,
                self.group_size,
                self.bits,
                false,
                "affine",
            )
        };

        // Activation: standard SwiGLU or clamped
        let activated = if let Some(limit) = self.swiglu_limit {
            clamped_swiglu(&x_up, &x_gate, limit)
        } else {
            mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up)
        };

        // Down projection
        let output = unsafe {
            mlxcel_core::gather_qmm(
                &activated,
                &self.down_weight,
                &self.down_scales,
                down_bias_ptr,
                std::ptr::null(),
                indices as *const _,
                true,
                self.group_size,
                self.bits,
                false,
                "affine",
            )
        };

        // Squeeze extra dim
        mlxcel_core::squeeze_axis(&output, -2)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
        swiglu_limit: Option<f32>,
    ) -> Result<Self, String> {
        let gate_weight = get_weight_copy(weights, &format!("{}.gate_proj.weight", prefix))?;
        let gate_scales = get_weight_copy(weights, &format!("{}.gate_proj.scales", prefix))?;
        let gate_biases = get_weight_copy(weights, &format!("{}.gate_proj.biases", prefix))?;

        let up_weight = get_weight_copy(weights, &format!("{}.up_proj.weight", prefix))?;
        let up_scales = get_weight_copy(weights, &format!("{}.up_proj.scales", prefix))?;
        let up_biases = get_weight_copy(weights, &format!("{}.up_proj.biases", prefix))?;

        let down_weight = get_weight_copy(weights, &format!("{}.down_proj.weight", prefix))?;
        let down_scales = get_weight_copy(weights, &format!("{}.down_proj.scales", prefix))?;
        let down_biases = get_weight_copy(weights, &format!("{}.down_proj.biases", prefix))?;

        Ok(Self {
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            down_weight,
            down_scales,
            down_biases,
            group_size,
            bits,
            swiglu_limit,
        })
    }
}

// MoE Block.
pub struct Step3p5MoE {
    gate: Step3p5MoEGate,
    switch_mlp: Step3p5SwitchGLU,
    share_expert: Step3p5MLP,
}

impl Step3p5MoE {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (indices, scores) = self.gate.forward(x);

        // Expert computation
        let y = self.switch_mlp.forward(x, &indices);

        let x_dtype = mlxcel_core::array_dtype(x);
        let routed_output = crate::models::switch_layers::moe_weighted_sum(&y, &scores, x_dtype);

        // Add shared expert
        let shared_output = self.share_expert.forward(x);
        mlxcel_core::add(&routed_output, &shared_output)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Step3p5Config,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate = Step3p5MoEGate::from_weights(weights, args, &format!("{}.gate", prefix))?;

        let swiglu_limit = args.get_swiglu_limit(layer_idx);
        let limit = if swiglu_limit > 0.0 {
            Some(swiglu_limit)
        } else {
            None
        };

        let switch_mlp = Step3p5SwitchGLU::from_weights(
            weights,
            &format!("{}.switch_mlp", prefix),
            group_size,
            bits,
            limit,
        )?;

        let swiglu_limit_shared = args.get_swiglu_limit_shared(layer_idx);
        let share_expert = Step3p5MLP::from_weights(
            weights,
            args,
            &format!("{}.share_expert", prefix),
            swiglu_limit_shared,
        )?;

        Ok(Self {
            gate,
            switch_mlp,
            share_expert,
        })
    }
}

trait CacheInterface {
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

enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Cache::Standard(cache) => cache,
            Cache::Rotating(cache) => cache,
        }
    }
}

// MLP Type (Dense or MoE).
pub enum MLPType {
    Dense(Step3p5MLP),
    MoE(Step3p5MoE),
}

// Decoder Layer.
pub struct Step3p5DecoderLayer {
    self_attn: Step3p5Attention,
    mlp: MLPType,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    is_sliding: bool,
}

impl Step3p5DecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Attention with residual
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed),
            MLPType::MoE(moe) => moe.forward(&normed),
        };
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Step3p5Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Step3p5Attention::from_weights(
            weights,
            args,
            layer_idx,
            &format!("{}.self_attn", prefix),
        )?;

        let is_sliding = args.is_sliding(layer_idx);
        let moe_layers = args.moe_layer_indices();
        let is_moe = moe_layers.contains(&layer_idx);

        let mlp = if is_moe {
            MLPType::MoE(Step3p5MoE::from_weights(
                weights,
                args,
                layer_idx,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            let swiglu_limit = args.get_swiglu_limit_shared(layer_idx);
            MLPType::Dense(Step3p5MLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
                swiglu_limit,
            )?)
        };

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
            is_sliding,
        })
    }
}

// Step3p5 Model.
pub struct Step3p5Model {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<Step3p5DecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    config: Step3p5Config,
    /// Index of first sliding window layer (for mask creation)
    swa_idx: Option<usize>,
    /// Index of first full attention layer (for mask creation)
    full_idx: Option<usize>,
    eos_token_ids: Vec<i32>,
    internal_caches: RefCell<Vec<Cache>>,
}

impl Step3p5Model {
    fn build_layer_caches(config: &Step3p5Config) -> Vec<Cache> {
        (0..config.num_hidden_layers)
            .map(|layer_idx| {
                if config.is_sliding(layer_idx) {
                    Cache::Rotating(RotatingKVCache::new(config.sliding_window))
                } else {
                    Cache::Standard(KVCache::new())
                }
            })
            .collect()
    }

    fn reset_internal_caches(&self) {
        *self.internal_caches.borrow_mut() = Self::build_layer_caches(&self.config);
    }

    fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1] as i32;

        // Create masks for prefill
        let full_mask = if seq_len > 1 {
            self.full_idx.map(|idx| {
                let offset = caches[idx].as_interface().offset();
                create_causal_mask(seq_len, offset)
            })
        } else {
            None
        };

        let swa_mask = if seq_len > 1 {
            self.swa_idx.map(|idx| {
                // Full-width windowed mask for a fresh single-pass prefill that
                // exceeds the window (RotatingKVCache keeps all prefill keys),
                // clamped mask otherwise. See issue #408.
                let offset = caches[idx].as_interface().offset();
                create_sliding_window_prefill_mask(seq_len, offset, self.config.sliding_window)
            })
        } else {
            None
        };

        // Pass through layers
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.is_sliding {
                swa_mask.as_ref().map(|m| m.as_ref().unwrap() as &MlxArray)
            } else {
                full_mask.as_ref().map(|m| m.as_ref().unwrap() as &MlxArray)
            };
            h = layer.forward(&h, caches[i].as_interface(), mask);
        }

        // Final norm and lm_head
        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Step3p5Config), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: Step3p5Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Parse EOS token from raw JSON
        let config_value: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config JSON: {}", e))?;
        let eos_token_ids = parse_eos_token_ids(&config_value);

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = Self::sanitize_weights(weights, &config);

        let mut model = Self::from_weights(&weights, &config)?;
        model.eos_token_ids = eos_token_ids;

        Ok((model, config))
    }

    fn sanitize_weights(mut weights: WeightMap, config: &Step3p5Config) -> WeightMap {
        let remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ];

        // Check if weights need remapping (vanilla format)
        let is_vanilla = weights.keys().any(|k| {
            remappings
                .iter()
                .any(|(src, dst)| k.contains(src) && !k.contains(dst))
        });

        let old_keys: Vec<String> = weights.keys().cloned().collect();
        let mut new_weights = WeightMap::new();

        for k in old_keys {
            // Skip MTP layers
            if k.contains(".mtp") {
                continue;
            }

            // Skip layers beyond num_hidden_layers
            if k.contains("model.layers.") {
                let parts: Vec<&str> = k.split('.').collect();
                if parts.len() > 2
                    && let Ok(idx) = parts[2].parse::<usize>()
                    && idx >= config.num_hidden_layers
                {
                    continue;
                }
            }

            let mut new_key = k.clone();
            for (src, dst) in &remappings {
                if new_key.contains(src) && !new_key.contains(dst) {
                    new_key = new_key.replace(src, dst);
                    break;
                }
            }

            if let Some(v) = weights.remove(&k) {
                // Add +1 to norm weights for vanilla format (ZeroCenteredRMSNorm)
                if is_vanilla && new_key.ends_with(".weight") && new_key.contains("norm") {
                    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&v));
                    let adjusted = mlxcel_core::add(&v, &one);
                    new_weights.insert(new_key, adjusted);
                } else {
                    new_weights.insert(new_key, v);
                }
            }
        }

        new_weights
    }

    pub fn from_weights(weights: &WeightMap, args: &Step3p5Config) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = Step3p5DecoderLayer::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        let lm_head = if args.tie_word_embeddings {
            // When tie_word_embeddings, lm_head uses embed_tokens weight
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
        };

        // Find first sliding and full attention layer indices
        let swa_idx = layers.iter().position(|l| l.is_sliding);
        let full_idx = layers.iter().position(|l| !l.is_sliding);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: args.clone(),
            swa_idx,
            full_idx,
            eos_token_ids: vec![2], // Default, overridden in load()
            internal_caches: RefCell::new(Self::build_layer_caches(args)),
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for Step3p5Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut internal_caches = self.internal_caches.borrow_mut();
        self.forward_with_caches(input_ids, &mut internal_caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.reset_internal_caches();
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_batching(&self) -> bool {
        false // Step3p5 uses internal RefCell caches, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "step3p5_tests.rs"]
mod tests;
