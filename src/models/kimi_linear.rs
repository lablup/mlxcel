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

//! KimiLinear: Hybrid MLA (Multi-head Latent Attention) + GatedDeltaNet model
//!
//! Key Features:
//! - KimiMLAAttention: Multi-head Latent Attention with KV compression
//! - KimiDeltaAttention: Linear attention via GatedDeltaNet with ShortConv1d
//! - Sparse MoE with grouped top-k routing (sigmoid/softmax)
//! - Per-head MultiLinear projections for MLA
//!
//! Reference: mlx-lm/mlx_lm/models/kimi_linear.py

use crate::models::gated_delta::{gated_delta_update, scaled_fast_rms_norm_no_weight};
use crate::models::switch_layers::SwitchGLU;
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, silu, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttnConfig {
    pub kda_layers: Vec<usize>,
    pub num_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    pub short_conv_kernel_size: usize,
}

fn default_conv_kernel() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize)]
pub struct KimiLinearConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub linear_attn_config: LinearAttnConfig,
    pub num_experts: usize,
    pub moe_intermediate_size: usize,
    pub kv_lora_rank: usize,

    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub qk_nope_head_dim: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub v_head_dim: Option<usize>,
    #[serde(default)]
    pub mla_use_nope: bool,
    #[serde(default = "default_one_usize")]
    pub num_experts_per_token: usize,
    #[serde(default)]
    pub num_shared_experts: usize,
    #[serde(default = "default_sigmoid")]
    pub moe_router_activation_func: String,
    #[serde(default = "default_true")]
    pub moe_renormalize: bool,
    #[serde(default = "default_one_f32")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_one_usize")]
    pub moe_layer_freq: usize,
    #[serde(default = "default_true")]
    pub use_grouped_topk: bool,
    #[serde(default = "default_one_usize")]
    pub num_expert_group: usize,
    #[serde(default = "default_one_usize")]
    pub topk_group: usize,
    pub quantization: Option<Quantization>,
}

fn default_one_usize() -> usize {
    1
}
fn default_one_f32() -> f32 {
    1.0
}
fn default_sigmoid() -> String {
    "sigmoid".to_string()
}
fn default_true() -> bool {
    true
}

impl KimiLinearConfig {
    pub fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }
    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }
    pub fn qk_nope(&self) -> usize {
        self.qk_nope_head_dim.unwrap_or(self.head_dim)
    }
    pub fn qk_rope(&self) -> usize {
        self.qk_rope_head_dim.unwrap_or(0)
    }
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope() + self.qk_rope()
    }
    pub fn v_head(&self) -> usize {
        self.v_head_dim.unwrap_or(self.head_dim)
    }
    pub fn is_linear_layer(&self, idx: usize) -> bool {
        self.linear_attn_config.kda_layers.contains(&(idx + 1))
    }
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.num_experts > 0
            && idx >= self.first_k_dense_replace
            && idx.is_multiple_of(self.moe_layer_freq)
    }
    pub fn delta_num_heads(&self) -> usize {
        self.linear_attn_config.num_heads
    }
    pub fn delta_head_dim(&self) -> usize {
        self.linear_attn_config.head_dim
    }
    pub fn delta_projection_dim(&self) -> usize {
        self.delta_num_heads() * self.delta_head_dim()
    }
}

// MultiLinear - Per-head matrix multiply (dense or quantized).
/// Per-head linear projection used in MLA.
/// Weight shape: [num_heads, output_dims, input_dims]
///
/// Used by: KimiLinear (MLA attention)
struct MultiLinear {
    weight: UniquePtr<MlxArray>,
    scales: Option<UniquePtr<MlxArray>>,
    biases: Option<UniquePtr<MlxArray>>,
    group_size: i32,
    bits: i32,
    is_quantized: bool,
}

impl MultiLinear {
    fn forward(&self, x: &MlxArray, transpose: bool) -> UniquePtr<MlxArray> {
        if self.is_quantized {
            let biases_ptr = self
                .biases
                .as_ref()
                .map(|b| b.as_ref().unwrap() as *const _)
                .unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::quantized_matmul(
                    x,
                    &self.weight,
                    self.scales.as_ref().unwrap(),
                    biases_ptr,
                    transpose,
                    self.group_size,
                    self.bits,
                    "affine",
                )
            }
        } else if transpose {
            let w = mlxcel_core::swap_axes(&self.weight, -1, -2);
            mlxcel_core::matmul(x, &w)
        } else {
            mlxcel_core::matmul(x, &self.weight)
        }
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing MultiLinear weight: {}", prefix))?;

        let scales = weights
            .get(&format!("{}.scales", prefix))
            .map(|w| mlxcel_core::copy(w));
        let biases = weights
            .get(&format!("{}.biases", prefix))
            .map(|w| mlxcel_core::copy(w));

        let is_quantized = scales.is_some();

        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            is_quantized,
        })
    }
}

// ShortConv1d - Depthwise convolution with state.
/// Short depthwise convolution with manual state management.
/// Used for preprocessing Q, K, V in delta attention layers.
///
/// Used by: KimiLinear (Delta attention)
struct ShortConv1d {
    conv_weight: UniquePtr<MlxArray>,
    kernel_size: usize,
    channels: usize,
}

impl ShortConv1d {
    fn forward(
        &self,
        x: &MlxArray,
        state: Option<&MlxArray>,
        mask: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];

        // Apply mask: zero out masked positions
        let x = if let Some(m) = mask {
            let m_exp = mlxcel_core::expand_dims(m, -1);
            let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::array_dtype(x));
            mlxcel_core::where_cond(&m_exp, x, &zero)
        } else {
            mlxcel_core::copy(x)
        };

        // Get or create state
        let state = if let Some(s) = state {
            mlxcel_core::copy(s)
        } else {
            mlxcel_core::zeros(
                &[b, (self.kernel_size - 1) as i32, self.channels as i32],
                mlxcel_core::array_dtype(&x),
            )
        };

        // Concatenate [state, x] along time axis
        let conv_input = concatenate(&state, &x, 1);

        // Apply depthwise conv1d + SiLU
        let conv_out = mlxcel_core::conv1d(
            &conv_input,
            &self.conv_weight,
            1, // stride
            0, // padding
            1, // dilation
            self.channels as i32,
        );
        let conv_out = silu(&conv_out);

        // Extract new state: last kernel_size-1 positions from conv_input.
        // Wrap slice in contiguous() to force MLX to materialize a fresh,
        // independent buffer. Without this, the slice is a lazy view that
        // retains a reference to the full conv_input allocation, causing a
        // memory leak proportional to the sequence length.
        let n_keep = (self.kernel_size - 1) as i32;
        let conv_shape = mlxcel_core::array_shape(&conv_input);
        let total_len = conv_shape[1];
        let tail = mlxcel_core::slice(
            &conv_input,
            &[0, total_len - n_keep, 0],
            &[b, total_len, self.channels as i32],
        );
        let new_state = mlxcel_core::contiguous(&tail, false);

        (conv_out, new_state)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &KimiLinearConfig,
    ) -> Result<Self, String> {
        let channels = config.delta_projection_dim();
        let kernel_size = config.linear_attn_config.short_conv_kernel_size;

        let conv_weight = weights
            .get(&format!("{}.conv.weight", prefix))
            .map(|w| {
                let shape = mlxcel_core::array_shape(w);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    mlxcel_core::swap_axes(w, -1, -2)
                } else {
                    mlxcel_core::copy(w)
                }
            })
            .ok_or_else(|| format!("Missing conv weight: {}", prefix))?;

        Ok(Self {
            conv_weight,
            kernel_size,
            channels,
        })
    }
}

// Cache Types.
/// Cache for KimiDeltaAttention layers (4 elements).
pub struct KimiDeltaCache {
    pub q_conv_state: Option<UniquePtr<MlxArray>>,
    pub k_conv_state: Option<UniquePtr<MlxArray>>,
    pub v_conv_state: Option<UniquePtr<MlxArray>>,
    pub ssm_state: Option<UniquePtr<MlxArray>>,
    pub offset: i32,
}

impl KimiDeltaCache {
    pub fn new() -> Self {
        Self {
            q_conv_state: None,
            k_conv_state: None,
            v_conv_state: None,
            ssm_state: None,
            offset: 0,
        }
    }

    pub fn advance(&mut self, step: i32) {
        self.offset += step;
    }
}

impl Default for KimiDeltaCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixed cache type for KimiLinear layers
pub enum KimiLinearCache {
    MLA(KVCache),
    Delta(KimiDeltaCache),
}

impl KimiLinearCache {
    pub fn offset(&self) -> i32 {
        match self {
            KimiLinearCache::MLA(kv) => kv.offset,
            KimiLinearCache::Delta(d) => d.offset,
        }
    }
}

// KimiMLAAttention - Multi-head Latent Attention.
struct KimiMLAAttention {
    q_proj: UnifiedLinear,
    kv_a_proj_with_mqa: UnifiedLinear,
    kv_a_layernorm: RMSNorm,
    embed_q: MultiLinear,
    unembed_out: MultiLinear,
    o_proj: UnifiedLinear,

    num_heads: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    q_head_dim: i32,
    _v_head_dim: i32,
    kv_lora_rank: i32,
    scale: f32,
}

impl KimiMLAAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KVCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Q projection: [B, L, hidden] -> [B, L, num_heads, q_head_dim]
        let q = self.q_proj.forward(x);
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]); // [B, H, L, q_head_dim]

        // Split q into nope and pe components
        let q_nope = mlxcel_core::slice(
            &q,
            &[0, 0, 0, 0],
            &[b, self.num_heads, l, self.qk_nope_head_dim],
        );
        let q_pe = mlxcel_core::slice(
            &q,
            &[0, 0, 0, self.qk_nope_head_dim],
            &[b, self.num_heads, l, self.q_head_dim],
        );

        // KV compression: [B, L, hidden] -> [B, L, kv_lora_rank + qk_rope_head_dim]
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let compressed = mlxcel_core::slice(&compressed_kv, &[0, 0, 0], &[b, l, self.kv_lora_rank]);
        let k_pe = mlxcel_core::slice(
            &compressed_kv,
            &[0, 0, self.kv_lora_rank],
            &[b, l, self.kv_lora_rank + self.qk_rope_head_dim],
        );

        // k_pe: [B, L, qk_rope] -> [B, 1, L, qk_rope]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // kv_latent: layernorm then expand to [B, 1, L, kv_lora_rank]
        let kv_latent = self.kv_a_layernorm.forward(&compressed);
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);

        // Update KV cache (stores kv_latent as "keys" and k_pe as "values")
        let (cached_latent, cached_k_pe) = cache.update_and_fetch(kv_latent, k_pe);

        // PE scoring: (q_pe * scale) @ k_pe^T -> [B, H, L, S]
        let scale_arr = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(&q_pe));
        let q_pe_scaled = mlxcel_core::multiply(&q_pe, &scale_arr);
        let k_pe_t = mlxcel_core::swap_axes(&cached_k_pe, -1, -2);
        let mut pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);

        // Apply causal mask to pe_scores
        if let Some(m) = mask {
            let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, dtype::FLOAT16);
            pe_scores = mlxcel_core::where_cond(m, &pe_scores, &neg_inf);
        }

        // Attention computation (different paths for generation vs prefill)
        let output = if l == 1 {
            // Generation: work in latent space
            let q_nope = self.embed_q.forward(&q_nope, true); // [B, H, 1, kv_lora_rank]
            // k = v = cached_latent: [B, 1, S, kv_lora_rank]

            // Manual attention: scores = (q_nope * scale) @ k^T + pe_scores
            let q_nope_scaled = mlxcel_core::multiply(&q_nope, &scale_arr);
            let k_t = mlxcel_core::swap_axes(&cached_latent, -1, -2);
            let nope_scores = mlxcel_core::matmul(&q_nope_scaled, &k_t);
            let scores = mlxcel_core::add(&nope_scores, &pe_scores);
            let weights = mlxcel_core::softmax(&scores, -1);
            let attn_out = mlxcel_core::matmul(&weights, &cached_latent);

            // Project from latent to output space
            self.unembed_out.forward(&attn_out, true) // [B, H, 1, v_head_dim]
        } else {
            // Prefill: expand KV from latent to full dimension
            let k = self.embed_q.forward(&cached_latent, false); // [B, H, L, qk_nope_head_dim]
            let v = self.unembed_out.forward(&cached_latent, true); // [B, H, L, v_head_dim]

            // Manual attention: scores = (q_nope * scale) @ k^T + pe_scores
            let q_nope_scaled = mlxcel_core::multiply(&q_nope, &scale_arr);
            let k_t = mlxcel_core::swap_axes(&k, -1, -2);
            let nope_scores = mlxcel_core::matmul(&q_nope_scaled, &k_t);
            let scores = mlxcel_core::add(&nope_scores, &pe_scores);
            let weights = mlxcel_core::softmax(&scores, -1);
            mlxcel_core::matmul(&weights, &v) // [B, H, L, v_head_dim]
        };

        // Transpose and reshape: [B, H, L, v_head_dim] -> [B, L, H*v_head_dim]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);

        self.o_proj.forward(&output)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &KimiLinearConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let q_proj = UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), gs, bits)?;
        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{}.kv_a_proj_with_mqa", prefix),
            gs,
            bits,
        )?;

        let kv_norm_weight = weights
            .get(&format!("{}.kv_a_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing kv_a_layernorm weight: {}", prefix))?;

        let embed_q = MultiLinear::from_weights(weights, &format!("{}.embed_q", prefix), gs, bits)?;
        let unembed_out =
            MultiLinear::from_weights(weights, &format!("{}.unembed_out", prefix), gs, bits)?;
        let o_proj = UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), gs, bits)?;

        Ok(Self {
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm: RMSNorm::new(kv_norm_weight, config.rms_norm_eps),
            embed_q,
            unembed_out,
            o_proj,
            num_heads: config.num_attention_heads as i32,
            qk_nope_head_dim: config.qk_nope() as i32,
            qk_rope_head_dim: config.qk_rope() as i32,
            q_head_dim: config.q_head_dim() as i32,
            _v_head_dim: config.v_head() as i32,
            kv_lora_rank: config.kv_lora_rank as i32,
            scale: (config.q_head_dim() as f32).powf(-0.5),
        })
    }
}

// KimiDeltaAttention - Linear Attention via GatedDeltaNet.
struct KimiDeltaAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    q_conv: ShortConv1d,
    k_conv: ShortConv1d,
    v_conv: ShortConv1d,
    f_a_proj: UnifiedLinear,
    f_b_proj: UnifiedLinear,
    b_proj: UnifiedLinear,
    g_a_proj: UnifiedLinear,
    g_b_proj: UnifiedLinear,
    a_log: UniquePtr<MlxArray>,
    dt_bias: UniquePtr<MlxArray>,
    o_norm: RMSNorm,
    o_proj: UnifiedLinear,

    num_heads: i32,
    head_dim: i32,
    _projection_dim: i32,
    scale: f32,
}

impl KimiDeltaAttention {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut KimiDeltaCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let t = shape[1];

        // Apply Q, K, V projections then ShortConv1d
        let q_proj = self.q_proj.forward(x);
        let k_proj = self.k_proj.forward(x);
        let v_proj = self.v_proj.forward(x);

        let (q_conv, q_state) = self
            .q_conv
            .forward(&q_proj, cache.q_conv_state.as_deref(), mask);
        let (k_conv, k_state) = self
            .k_conv
            .forward(&k_proj, cache.k_conv_state.as_deref(), mask);
        let (v_conv, v_state) = self
            .v_conv
            .forward(&v_proj, cache.v_conv_state.as_deref(), mask);

        // Update conv states in cache
        cache.q_conv_state = Some(q_state);
        cache.k_conv_state = Some(k_state);
        cache.v_conv_state = Some(v_state);

        // Reshape to per-head: [B, T, projection_dim] -> [B, T, num_heads, head_dim]
        let q = mlxcel_core::reshape(&q_conv, &[b, t, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k_conv, &[b, t, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v_conv, &[b, t, self.num_heads, self.head_dim]);

        // RMS normalize Q and K (without learned weight). Reference mlx-lm
        // uses mx.fast.rms_norm here; keep it fused instead of expanding it.
        let inv_scale = self.scale;
        let q = scaled_fast_rms_norm_no_weight(&q, inv_scale * inv_scale, 1e-6);
        let k = scaled_fast_rms_norm_no_weight(&k, inv_scale, 1e-6);

        // Compute gating logits
        let a_logits = self.f_b_proj.forward(&self.f_a_proj.forward(x));
        let a_logits = mlxcel_core::reshape(&a_logits, &[b, t, self.num_heads, self.head_dim]);
        let b_logits = self.b_proj.forward(x);
        let b_logits = mlxcel_core::reshape(&b_logits, &[b, t, self.num_heads]);

        // Reshape A_log and dt_bias for gated_delta_update
        let a_log = mlxcel_core::reshape(&self.a_log, &[self.num_heads, 1]);
        let dt_bias = mlxcel_core::reshape(&self.dt_bias, &[self.num_heads, self.head_dim]);

        // Run gated delta update
        let ssm_state = cache.ssm_state.as_deref();
        let (out, new_ssm_state) = gated_delta_update(
            &q, &k, &v, &a_logits, &b_logits, &a_log, &dt_bias, ssm_state, None,
        );

        // Update SSM state and advance cache
        cache.ssm_state = Some(new_ssm_state);
        cache.advance(t);

        // Output gating: o_norm(out) * sigmoid(gate)
        let gate = self.g_b_proj.forward(&self.g_a_proj.forward(x));
        let gate = mlxcel_core::reshape(&gate, &[b, t, self.num_heads, self.head_dim]);
        let out = mlxcel_core::reshape(&out, &[b, t, self.num_heads, self.head_dim]);
        let out_normed = self.o_norm.forward(&out);
        let gate_sig = mlxcel_core::sigmoid(&gate);
        let out = mlxcel_core::multiply(&out_normed, &gate_sig);
        let out = mlxcel_core::reshape(&out, &[b, t, -1]);

        self.o_proj.forward(&out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &KimiLinearConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let q_proj = UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), gs, bits)?;
        let k_proj = UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), gs, bits)?;
        let v_proj = UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), gs, bits)?;

        let q_conv = ShortConv1d::from_weights(weights, &format!("{}.q_conv", prefix), config)?;
        let k_conv = ShortConv1d::from_weights(weights, &format!("{}.k_conv", prefix), config)?;
        let v_conv = ShortConv1d::from_weights(weights, &format!("{}.v_conv", prefix), config)?;

        let f_a_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.f_a_proj", prefix), gs, bits)?;
        let f_b_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.f_b_proj", prefix), gs, bits)?;
        let b_proj = UnifiedLinear::from_weights(weights, &format!("{}.b_proj", prefix), gs, bits)?;
        let g_a_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.g_a_proj", prefix), gs, bits)?;
        let g_b_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.g_b_proj", prefix), gs, bits)?;

        let a_log = weights
            .get(&format!("{}.A_log", prefix))
            .map(|w| {
                let shape = mlxcel_core::array_shape(w);
                let total: i32 = shape.iter().product();
                mlxcel_core::reshape(w, &[total])
            })
            .ok_or_else(|| format!("Missing A_log: {}", prefix))?;

        let dt_bias = weights
            .get(&format!("{}.dt_bias", prefix))
            .map(|w| {
                let shape = mlxcel_core::array_shape(w);
                if shape.len() > 1 {
                    let total: i32 = shape.iter().product();
                    mlxcel_core::reshape(w, &[total])
                } else {
                    mlxcel_core::copy(w)
                }
            })
            .ok_or_else(|| format!("Missing dt_bias: {}", prefix))?;

        let o_norm_weight = weights
            .get(&format!("{}.o_norm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing o_norm weight: {}", prefix))?;

        let o_proj = UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), gs, bits)?;

        let num_heads = config.delta_num_heads() as i32;
        let head_dim = config.delta_head_dim() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            q_conv,
            k_conv,
            v_conv,
            f_a_proj,
            f_b_proj,
            b_proj,
            g_a_proj,
            g_b_proj,
            a_log,
            dt_bias,
            o_norm: RMSNorm::new(o_norm_weight, config.rms_norm_eps),
            o_proj,
            num_heads,
            head_dim,
            _projection_dim: (config.delta_projection_dim()) as i32,
            scale: (config.delta_head_dim() as f32).powf(-0.5),
        })
    }
}

// MLP.
struct KimiMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl KimiMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        let gated = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&gated)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }
}

// KimiSparseMoE.
struct KimiSparseMoE {
    gate: UnifiedLinear,
    switch_mlp: SwitchGLU,
    e_score_correction_bias: Option<UniquePtr<MlxArray>>,
    shared_experts: Option<KimiMLP>,
    num_experts_per_token: usize,
    _num_expert_group: usize,
    _topk_group: usize,
    routed_scaling_factor: f32,
    renormalize: bool,
    score_function: String,
}

impl KimiSparseMoE {
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

        // Router scores
        let logits = self.gate.forward(&x_flat);

        // Compute scores based on activation function
        let scores = if self.score_function == "sigmoid" {
            mlxcel_core::sigmoid(&logits)
        } else {
            mlxcel_core::softmax(&logits, -1)
        };
        let orig_scores = mlxcel_core::copy(&scores);

        // Add bias if present
        let scores = if let Some(ref bias) = self.e_score_correction_bias {
            mlxcel_core::add(&scores, bias)
        } else {
            scores
        };

        // Top-k selection (simplified: assumes num_expert_group == 1)
        let k = self.num_experts_per_token as i32;
        let scores_shape = mlxcel_core::array_shape(&scores);
        let _n_experts = scores_shape[1];

        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = mlxcel_core::slice(&indices, &[0, 0], &[scores_shape[0], k]);

        // Get original scores for top-k
        let mut topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        // Renormalize
        if k > 1 && self.renormalize {
            let s_dtype = mlxcel_core::array_dtype(&topk_scores);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, s_dtype);
            let sum = mlxcel_core::add(&mlxcel_core::sum_axis(&topk_scores, -1, true), &eps);
            topk_scores = mlxcel_core::divide(&topk_scores, &sum);
        }

        // Apply scaling factor
        if self.routed_scaling_factor != 1.0 {
            let scale = mlxcel_core::full_f32(
                &[1],
                self.routed_scaling_factor,
                mlxcel_core::array_dtype(&topk_scores),
            );
            topk_scores = mlxcel_core::multiply(&topk_scores, &scale);
        }

        // Expert computation
        let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);
        let mut y = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &topk_scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Shared experts
        if let Some(ref shared) = self.shared_experts {
            let shared_y = shared.forward(&x_flat);
            y = mlxcel_core::add(&y, &shared_y);
        }

        // Reshape back
        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&y, &orig_shape)
        } else {
            y
        }
    }

    fn from_weights(
        weights: &WeightMap,
        config: &KimiLinearConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        // MoE gate uses 8-bit quantization
        let gate = UnifiedLinear::from_weights(weights, &format!("{}.gate", prefix), 64, 8)?;

        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{}.switch_mlp", prefix), gs, bits)?;

        let e_score_correction_bias = weights
            .get(&format!("{}.e_score_correction_bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        let shared_experts = if config.num_shared_experts > 0 {
            Some(KimiMLP::from_weights(
                weights,
                &format!("{}.shared_experts", prefix),
                gs,
                bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            switch_mlp,
            e_score_correction_bias,
            shared_experts,
            num_experts_per_token: config.num_experts_per_token,
            _num_expert_group: config.num_expert_group,
            _topk_group: config.topk_group,
            routed_scaling_factor: config.routed_scaling_factor,
            renormalize: config.moe_renormalize,
            score_function: config.moe_router_activation_func.clone(),
        })
    }
}

// MLP Variant.
enum MLPVariant {
    Dense(KimiMLP),
    MoE(KimiSparseMoE),
}

impl MLPVariant {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MLPVariant::Dense(mlp) => mlp.forward(x),
            MLPVariant::MoE(moe) => moe.forward(x),
        }
    }
}

// Attention Variant.
enum AttentionVariant {
    MLA(KimiMLAAttention),
    Delta(KimiDeltaAttention),
}

// Decoder Layer.
struct KimiDecoderLayer {
    is_linear: bool,
    self_attn: AttentionVariant,
    mlp: MLPVariant,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl KimiDecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        attn_mask: Option<&MlxArray>,
        ssm_mask: Option<&MlxArray>,
        cache: &mut KimiLinearCache,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);

        let mask = if self.is_linear { ssm_mask } else { attn_mask };

        let r = match (&self.self_attn, cache) {
            (AttentionVariant::Delta(attn), KimiLinearCache::Delta(c)) => {
                attn.forward(&normed, mask, c)
            }
            (AttentionVariant::MLA(attn), KimiLinearCache::MLA(c)) => {
                attn.forward(&normed, mask, c)
            }
            _ => panic!("Cache type mismatch"),
        };

        let h = mlxcel_core::add(x, &r);
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &mlp_out)
    }

    fn from_weights(
        weights: &WeightMap,
        config: &KimiLinearConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        let is_linear = config.is_linear_layer(layer_idx);

        let self_attn = if is_linear {
            AttentionVariant::Delta(KimiDeltaAttention::from_weights(
                weights,
                config,
                &format!("{}.self_attn", prefix),
            )?)
        } else {
            AttentionVariant::MLA(KimiMLAAttention::from_weights(
                weights,
                config,
                &format!("{}.self_attn", prefix),
            )?)
        };

        let mlp = if config.is_moe_layer(layer_idx) {
            MLPVariant::MoE(KimiSparseMoE::from_weights(
                weights,
                config,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            MLPVariant::Dense(KimiMLP::from_weights(
                weights,
                &format!("{}.mlp", prefix),
                config.group_size(),
                config.bits(),
            )?)
        };

        let input_norm = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing input_layernorm: {}", prefix))?;
        let post_norm = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Missing post_attention_layernorm: {}", prefix))?;

        Ok(Self {
            is_linear,
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_norm, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm, config.rms_norm_eps),
        })
    }
}

// KimiLinear Model.
pub struct KimiLinearModel {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<KimiDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub tie_word_embeddings: bool,
    _ssm_layer_idx: usize,
    attn_layer_idx: usize,
}

impl KimiLinearModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KimiLinearCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        let shape = mlxcel_core::array_shape(&h);
        let l = shape[1];

        // Create masks
        let attn_mask = if l > 1 {
            let offset = caches[self.attn_layer_idx].offset();
            Some(create_causal_mask(l, offset))
        } else {
            None
        };

        // SSM mask: for prefill, all-true boolean; for generation, None
        // (SSM layers process all tokens; mask is for variable-length batches)
        let ssm_mask: Option<UniquePtr<MlxArray>> = None;

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, attn_mask.as_deref(), ssm_mask.as_deref(), cache);
        }

        let h = self.norm.forward(&h);

        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn make_kimi_caches(&self) -> Vec<KimiLinearCache> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_linear {
                    KimiLinearCache::Delta(KimiDeltaCache::new())
                } else {
                    KimiLinearCache::MLA(KVCache::new())
                }
            })
            .collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, KimiLinearConfig), String> {
        let model_dir = model_dir.as_ref();

        println!("[KimiLinear] Loading config...");
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: KimiLinearConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let n_linear = (0..config.num_hidden_layers)
            .filter(|&i| config.is_linear_layer(i))
            .count();
        let n_moe = (0..config.num_hidden_layers)
            .filter(|&i| config.is_moe_layer(i))
            .count();
        println!(
            "[KimiLinear] Config loaded: {} layers ({} MLA, {} delta, {} MoE)",
            config.num_hidden_layers,
            config.num_hidden_layers - n_linear,
            n_linear,
            n_moe
        );

        println!("[KimiLinear] Loading weights...");
        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = Self::sanitize_weights(weights, &config);

        println!("[KimiLinear] Building model...");
        let model = Self::from_weights(&weights, &config)?;

        println!("[KimiLinear] Model loaded successfully");
        Ok((model, config))
    }

    pub fn sanitize_weights(mut weights: WeightMap, config: &KimiLinearConfig) -> WeightMap {
        // Remove mtp weights
        let mtp_keys: Vec<String> = weights
            .keys()
            .filter(|k| k.starts_with("model.mtp"))
            .cloned()
            .collect();
        for k in mtp_keys {
            weights.remove(&k);
        }

        // Remove lm_head if tied
        if config.tie_word_embeddings {
            weights.remove("lm_head.weight");
        }

        // MoE weight stacking
        for l in 0..config.num_hidden_layers {
            if !config.is_moe_layer(l) {
                continue;
            }

            let src_prefix = format!("model.layers.{}.block_sparse_moe", l);
            let dst_prefix = format!("model.layers.{}.mlp", l);

            // Stack expert weights: w1->gate_proj, w2->down_proj, w3->up_proj
            for (src, dst) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                let mut expert_weights: Vec<UniquePtr<MlxArray>> = Vec::new();
                let mut expert_scales: Vec<UniquePtr<MlxArray>> = Vec::new();
                let mut expert_biases: Vec<UniquePtr<MlxArray>> = Vec::new();

                let mut e = 0;
                while let Some(w) =
                    weights.remove(&format!("{}.experts.{}.{}.weight", src_prefix, e, src))
                {
                    expert_weights.push(w);
                    if let Some(s) =
                        weights.remove(&format!("{}.experts.{}.{}.scales", src_prefix, e, src))
                    {
                        expert_scales.push(s);
                    }
                    if let Some(b) =
                        weights.remove(&format!("{}.experts.{}.{}.biases", src_prefix, e, src))
                    {
                        expert_biases.push(b);
                    }
                    e += 1;
                }

                if !expert_weights.is_empty() {
                    let stacked = stack_arrays(&expert_weights, 0);
                    weights.insert(format!("{}.switch_mlp.{}.weight", dst_prefix, dst), stacked);
                    if !expert_scales.is_empty() {
                        weights.insert(
                            format!("{}.switch_mlp.{}.scales", dst_prefix, dst),
                            stack_arrays(&expert_scales, 0),
                        );
                    }
                    if !expert_biases.is_empty() {
                        weights.insert(
                            format!("{}.switch_mlp.{}.biases", dst_prefix, dst),
                            stack_arrays(&expert_biases, 0),
                        );
                    }
                }
            }

            // Rename shared experts
            for name in ["gate_proj", "up_proj", "down_proj"] {
                let src_key = format!("{}.shared_experts.{}.weight", src_prefix, name);
                if let Some(w) = weights.remove(&src_key) {
                    weights.insert(format!("{}.shared_experts.{}.weight", dst_prefix, name), w);
                }
                // Also handle quantized shared experts
                for suffix in ["scales", "biases"] {
                    let src_key = format!("{}.shared_experts.{}.{}", src_prefix, name, suffix);
                    if let Some(w) = weights.remove(&src_key) {
                        weights.insert(
                            format!("{}.shared_experts.{}.{}", dst_prefix, name, suffix),
                            w,
                        );
                    }
                }
            }

            // Rename gate
            let gate_key = format!("{}.gate.weight", src_prefix);
            if let Some(w) = weights.remove(&gate_key) {
                weights.insert(format!("{}.gate.weight", dst_prefix), w);
            }
            for suffix in ["scales", "biases"] {
                let src_key = format!("{}.gate.{}", src_prefix, suffix);
                if let Some(w) = weights.remove(&src_key) {
                    weights.insert(format!("{}.gate.{}", dst_prefix, suffix), w);
                }
            }

            // Rename e_score_correction_bias
            let bias_key = format!("{}.gate.e_score_correction_bias", src_prefix);
            if let Some(w) = weights.remove(&bias_key) {
                weights.insert(format!("{}.e_score_correction_bias", dst_prefix), w);
            }
        }

        // Conv1d weight sanitization for delta attention layers
        for l in 0..config.num_hidden_layers {
            if !config.is_linear_layer(l) {
                continue;
            }

            let attn_prefix = format!("model.layers.{}.self_attn", l);

            // Rename q_conv1d -> q_conv, k_conv1d -> k_conv, v_conv1d -> v_conv
            for (src_name, dst_name) in [
                ("q_conv1d", "q_conv"),
                ("k_conv1d", "k_conv"),
                ("v_conv1d", "v_conv"),
            ] {
                let src_key = format!("{}.{}.weight", attn_prefix, src_name);
                if let Some(w) = weights.remove(&src_key) {
                    let shape = mlxcel_core::array_shape(&w);
                    let w = if shape.len() == 3 {
                        mlxcel_core::swap_axes(&w, 1, 2) // moveaxis(2, 1)
                    } else {
                        w
                    };
                    weights.insert(format!("{}.{}.conv.weight", attn_prefix, dst_name), w);
                }
            }

            // Flatten dt_bias if needed
            let dt_key = format!("{}.dt_bias", attn_prefix);
            if let Some(w) = weights.get(&dt_key) {
                let shape = mlxcel_core::array_shape(w);
                if shape.len() > 1 {
                    let total: i32 = shape.iter().product();
                    let w = mlxcel_core::reshape(w, &[total]);
                    weights.insert(dt_key, w);
                }
            }
        }

        // kv_b_proj decomposition for MLA layers
        for l in 0..config.num_hidden_layers {
            if config.is_linear_layer(l) {
                continue;
            }

            let attn_prefix = format!("model.layers.{}.self_attn", l);
            let kv_b_key = format!("{}.kv_b_proj.weight", attn_prefix);

            if weights.contains_key(&kv_b_key) {
                let qk_nope = config.qk_nope() as i32;
                let v_head = config.v_head() as i32;
                let num_heads = config.num_attention_heads as i32;

                let is_quantized =
                    weights.contains_key(&format!("{}.kv_b_proj.scales", attn_prefix));

                let v = if is_quantized {
                    // Dequantize first
                    let w = weights.remove(&kv_b_key).unwrap();
                    let scales = weights
                        .remove(&format!("{}.kv_b_proj.scales", attn_prefix))
                        .unwrap();
                    let biases = weights
                        .remove(&format!("{}.kv_b_proj.biases", attn_prefix))
                        .unwrap();
                    let w_shape = mlxcel_core::array_shape(&w);
                    let dims = config.kv_lora_rank as i32;
                    let bits = (w_shape[w_shape.len() - 1] * 32) / dims;
                    let s_shape = mlxcel_core::array_shape(&scales);
                    let group_size = dims / s_shape[s_shape.len() - 1];
                    unsafe {
                        mlxcel_core::dequantize(
                            &w,
                            &scales,
                            &*biases as *const _,
                            group_size,
                            bits,
                            "affine",
                        )
                    }
                } else {
                    weights.remove(&kv_b_key).unwrap()
                };

                // Reshape to [num_heads, qk_nope + v_head, kv_lora_rank]
                let v = mlxcel_core::reshape(&v, &[num_heads, qk_nope + v_head, -1]);

                // Split: wk = v[:, :qk_nope, :].swapaxes(-1, -2), wv = v[:, qk_nope:, :]
                // Note: MLX slice stop=-1 means dim_size-1 (excludes last), not "to end"
                let v_last_dim = mlxcel_core::array_shape(&v)[2];
                let wk = mlxcel_core::slice(&v, &[0, 0, 0], &[num_heads, qk_nope, v_last_dim]);
                let wk = mlxcel_core::swap_axes(&wk, -1, -2); // [num_heads, kv_lora_rank, qk_nope]
                let wv = mlxcel_core::slice(
                    &v,
                    &[0, qk_nope, 0],
                    &[num_heads, qk_nope + v_head, v_last_dim],
                );

                // Store as dense MultiLinear weights (no re-quantization)
                weights.insert(format!("{}.embed_q.weight", attn_prefix), wk);
                weights.insert(format!("{}.unembed_out.weight", attn_prefix), wv);
            }
        }

        weights
    }

    pub fn from_weights(weights: &WeightMap, config: &KimiLinearConfig) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(KimiDecoderLayer::from_weights(weights, config, i)?);
        }

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Missing model.norm.weight".to_string())?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?)
        };

        // Find representative layer indices for mask computation
        let kda_layers = &config.linear_attn_config.kda_layers;
        let ssm_layer_idx = if !kda_layers.is_empty() {
            kda_layers[0] - 1 // Convert from 1-indexed
        } else {
            0
        };
        let attn_layer_idx = (0..config.num_hidden_layers)
            .find(|i| !config.is_linear_layer(*i))
            .unwrap_or(0);

        Ok(Self {
            embed_tokens,
            layers,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            _ssm_layer_idx: ssm_layer_idx,
            attn_layer_idx,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for KimiLinearModel {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt DeltaCache recurrent state
    }

    fn supports_batching(&self) -> bool {
        false // KimiLinear uses internal mixed caches (MLA + DeltaCache), not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![2] // Default EOS
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Return dummy KV caches for trait compatibility
        // KimiLinear uses mixed cache types (MLA KVCache + DeltaCache) internally
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // KimiLinear manages its own mixed caches internally
        let mut caches = self.make_kimi_caches();
        self.forward(input_ids, &mut caches)
    }
}
