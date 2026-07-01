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

//! DeepSeek-V3 model implementation using mlxcel-core
//!
//! Key features:
//! - MLA (Multi-head Latent Attention) with Q/KV LoRA compression
//! - Complex MoE with sigmoid scoring and group-based expert selection
//! - Shared experts + routed experts
//! - First k layers are dense (no MoE)

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, MultiLinear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV3Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub n_routed_experts: Option<usize>,

    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,

    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,

    #[serde(default = "default_topk_method")]
    pub topk_method: String,

    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    #[serde(default = "default_n_group")]
    pub n_group: usize,

    #[serde(default = "default_topk_group")]
    pub topk_group: usize,

    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,

    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,

    #[serde(default)]
    pub first_k_dense_replace: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "deepseek_v3".to_string()
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_topk_method() -> String {
    "noaux_tc".to_string()
}
fn default_scoring_func() -> String {
    "sigmoid".to_string()
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_n_group() -> usize {
    1
}
fn default_topk_group() -> usize {
    1
}
fn default_num_experts_per_tok() -> usize {
    1
}
fn default_moe_layer_freq() -> usize {
    1
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl DeepSeekV3Config {
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some()
            && layer_idx >= self.first_k_dense_replace
            && layer_idx.is_multiple_of(self.moe_layer_freq)
    }

    pub fn get_attention_scale(&self) -> f32 {
        let q_head_dim = self.q_head_dim() as f32;
        let mut scale = q_head_dim.powf(-0.5);

        // Adjust scale for mscale_all_dim
        if let Some(ref rope_scaling) = self.rope_scaling
            && let (Some(mscale_all_dim), Some(factor)) = (
                rope_scaling.get("mscale_all_dim").and_then(|v| v.as_f64()),
                rope_scaling.get("factor").and_then(|v| v.as_f64()),
            )
            && mscale_all_dim > 0.0
            && factor > 1.0
        {
            let s = (0.1 * mscale_all_dim * factor.ln() + 1.0) as f32;
            scale = scale * s * s;
        }

        scale
    }
}

// DeepSeek-V3 MLA (Multi-head Latent Attention).
pub struct DeepSeekV3Attention {
    // Q projection with LoRA
    pub q_a_proj: UnifiedLinear,
    pub q_a_layernorm: RMSNorm,
    pub q_b_proj: UnifiedLinear,

    // KV projection with compression
    pub kv_a_proj_with_mqa: UnifiedLinear,
    pub kv_a_layernorm: RMSNorm,

    // MLA: embed_q and unembed_out replace kv_b_proj
    // embed_q: projects Q_nope into latent space (decode) or latent to K (prefill)
    // unembed_out: projects latent to V (prefill) or attention output back (decode)
    pub embed_q: MultiLinear,
    pub unembed_out: MultiLinear,

    pub o_proj: UnifiedLinear,

    pub num_heads: i32,
    pub q_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub qk_nope_head_dim: i32,
    pub v_head_dim: i32,
    pub kv_lora_rank: i32,
    pub scale: f32,
    pub rope_base: f32,
}

impl DeepSeekV3Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Compute Q with LoRA: q_a_proj → norm → q_b_proj
        let q_a = self.q_a_proj.forward(x);
        let q_a_norm = self.q_a_layernorm.forward(&q_a);
        let q = self.q_b_proj.forward(&q_a_norm);

        // Reshape Q to [batch, seq, heads, head_dim] then transpose
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Split Q into nope and pe parts
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        // Compute KV with compression
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);

        // Split into compressed and k_pe
        let compressed = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);

        // Reshape k_pe to [batch, 1, seq, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // kv_latent = layernorm(compressed)
        let kv_latent = self.kv_a_layernorm.forward(&compressed);

        // Apply RoPE to pe parts
        let offset = cache.offset;
        let q_pe = mlxcel_core::fast_rope(
            &q_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );
        let k_pe = mlxcel_core::fast_rope(
            &k_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );

        // Expand kv_latent for caching: [B, L, kv_lora_rank] → [B, 1, L, kv_lora_rank]
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);

        // Cache stores (kv_latent, k_pe) for memory efficiency
        let (kv_latent, k_pe) = cache.update_and_fetch(kv_latent, k_pe);

        // Compute positional encoding scores: pe_scores = (q_pe * scale) @ k_pe.T
        let scale_scalar = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(&q_pe));
        let q_pe_scaled = mlxcel_core::multiply(&q_pe, &scale_scalar);
        let k_pe_t = mlxcel_core::transpose_axes(&k_pe, &[0, 1, 3, 2]);
        let pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);

        // Apply causal mask to pe_scores (for prefill)
        let pe_scores = if let Some(m) = mask {
            // mask has 0 for attend, -inf for blocked
            mlxcel_core::add(&pe_scores, m)
        } else {
            pe_scores
        };

        // MLA attention: different paths for decode vs prefill
        let output = if l == 1 {
            // Decode: project Q into latent space, use kv_latent as K=V
            let q_projected = self.embed_q.forward(&q_nope);
            let pe_mask_ptr = &*pe_scores as *const MlxArray;
            let output = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_projected,
                    &kv_latent,
                    &kv_latent,
                    self.scale,
                    pe_mask_ptr,
                    0.0,
                    0,
                )
            };
            // Project output from latent space to v_head_dim
            self.unembed_out.forward(&output)
        } else {
            // Prefill: project kv_latent to K and V
            let k = self.embed_q.forward_no_transpose(&kv_latent);
            let v = self.unembed_out.forward(&kv_latent);
            let pe_mask_ptr = &*pe_scores as *const MlxArray;
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_nope,
                    &k,
                    &v,
                    self.scale,
                    pe_mask_ptr,
                    0.0,
                    0,
                )
            }
        };

        // Transpose back and reshape
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * self.v_head_dim]);

        // Output projection
        self.o_proj.forward(&output)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let q_head_dim = args.q_head_dim() as i32;

        let q_a_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.q_a_proj", prefix),
            group_size,
            bits,
        )?;
        let q_b_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.q_b_proj", prefix),
            group_size,
            bits,
        )?;
        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{}.kv_a_proj_with_mqa", prefix),
            group_size,
            bits,
        )?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        // Load embed_q and unembed_out (decomposed from kv_b_proj by sanitize_weights)
        let embed_q =
            MultiLinear::from_weights(weights, &format!("{}.embed_q", prefix), group_size, bits)?;
        let unembed_out = MultiLinear::from_weights(
            weights,
            &format!("{}.unembed_out", prefix),
            group_size,
            bits,
        )?;

        let q_a_norm_weight =
            get_weight_copy(weights, &format!("{}.q_a_layernorm.weight", prefix))?;
        let kv_a_norm_weight =
            get_weight_copy(weights, &format!("{}.kv_a_layernorm.weight", prefix))?;

        let q_a_layernorm = RMSNorm::new(q_a_norm_weight, 1e-6);
        let kv_a_layernorm = RMSNorm::new(kv_a_norm_weight, 1e-6);

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            embed_q,
            unembed_out,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            q_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim as i32,
            qk_nope_head_dim: args.qk_nope_head_dim as i32,
            v_head_dim: args.v_head_dim as i32,
            kv_lora_rank: args.kv_lora_rank as i32,
            scale: args.get_attention_scale(),
            rope_base: args.rope_theta,
        })
    }
}

// Dense MLP.
pub struct DenseMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl DenseMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
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

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

// MoE Gate.
pub struct MoEGate {
    pub weight: UniquePtr<MlxArray>,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub top_k: i32,
    pub n_routed_experts: i32,
    pub routed_scaling_factor: f32,
    pub n_group: i32,
    pub topk_group: i32,
    pub norm_topk_prob: bool,
}

impl MoEGate {
    /// Forward pass returns (expert_indices, expert_scores)
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // gates = x @ weight.T
        let weight_t = mlxcel_core::transpose(&self.weight);
        let gates = mlxcel_core::matmul(x, &weight_t);

        // Sigmoid scoring
        let scores = mlxcel_core::sigmoid(&gates);
        let orig_scores = mlxcel_core::copy(&scores);

        // Add correction bias
        let scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Group-based expert masking (zero out non-selected groups)
        let scores = if self.n_group > 1 {
            super::switch_layers::group_mask_scores(&scores, self.n_group, self.topk_group)
        } else {
            scores
        };

        // Get top-k expert indices using argpartition
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, self.top_k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, self.top_k);

        // Get scores from orig_scores
        let topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        // Normalize if needed
        let topk_scores = if self.top_k > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
            mlxcel_core::divide(&topk_scores, &sum)
        } else {
            topk_scores
        };

        // Scale scores
        let scale = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::array_dtype(&topk_scores),
        );
        let topk_scores = mlxcel_core::multiply(&topk_scores, &scale);

        (topk_indices, topk_scores)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let n_routed = args.n_routed_experts.unwrap_or(1) as i32;

        let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        let e_score_correction_bias =
            get_weight_copy(weights, &format!("{}.e_score_correction_bias", prefix))?;

        Ok(Self {
            weight,
            e_score_correction_bias,
            top_k: args.num_experts_per_tok as i32,
            n_routed_experts: n_routed,
            routed_scaling_factor: args.routed_scaling_factor,
            n_group: args.n_group as i32,
            topk_group: args.topk_group as i32,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// SwitchGLU (MoE expert layer using gather_qmm).
pub struct SwitchGLU {
    // Expert weights: [num_experts, output_dim, input_dim]
    pub gate_weight: UniquePtr<MlxArray>,
    pub gate_scales: UniquePtr<MlxArray>,
    pub gate_biases: UniquePtr<MlxArray>,
    pub up_weight: UniquePtr<MlxArray>,
    pub up_scales: UniquePtr<MlxArray>,
    pub up_biases: UniquePtr<MlxArray>,
    pub down_weight: UniquePtr<MlxArray>,
    pub down_scales: UniquePtr<MlxArray>,
    pub down_biases: UniquePtr<MlxArray>,
    pub group_size: i32,
    pub bits: i32,
}

impl SwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];

        // Expand x for gather_qmm: [n_tokens, hidden] -> [n_tokens, 1, 1, hidden]
        let x_exp = mlxcel_core::expand_dims(x, -2);
        let x_exp = mlxcel_core::expand_dims(&x_exp, -3);

        // Get bias pointers (biases field is UniquePtr, need to get raw pointer)
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

        // Gate projection with gather_qmm
        let x_gate = unsafe {
            mlxcel_core::gather_qmm(
                &x_exp,
                &self.gate_weight,
                &self.gate_scales,
                gate_bias_ptr,
                std::ptr::null(), // lhs_indices
                indices as *const _,
                true, // transpose
                self.group_size,
                self.bits,
                false, // sorted_indices
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

        // SwiGLU activation
        let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);

        // Down projection
        // activated is [n_tokens, top_k, 1, intermediate_size]
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

        // Squeeze: [n_tokens, top_k, 1, hidden] -> [n_tokens, top_k, hidden]
        let output = mlxcel_core::squeeze_axis(&output, -2);
        mlxcel_core::reshape(&output, &[n_tokens, top_k, -1])
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load stacked expert weights
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
        })
    }
}

// MoE Block.
pub struct MoEBlock {
    pub gate: MoEGate,
    pub switch_mlp: SwitchGLU,
    pub shared_experts: Option<DenseMLP>,
}

impl MoEBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (indices, scores) = self.gate.forward(x);

        // Expert computation
        let y = self.switch_mlp.forward(x, &indices);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &y,
            &scores,
            mlxcel_core::array_dtype(x),
        );

        // Add shared experts if present
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x);
            result = mlxcel_core::add(&result, &shared_out);
        }

        result
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &DeepSeekV3Config,
        prefix: &str,
    ) -> Result<Self, String> {
        let gate = MoEGate::from_weights(weights, args, &format!("{}.gate", prefix))?;
        let switch_mlp = SwitchGLU::from_weights(weights, args, &format!("{}.switch_mlp", prefix))?;

        let shared_experts = if args.n_shared_experts.is_some() {
            Some(DenseMLP::from_weights(
                weights,
                args,
                &format!("{}.shared_experts", prefix),
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            switch_mlp,
            shared_experts,
        })
    }
}

// MLP Type (Dense or MoE).
pub enum MLPType {
    Dense(DenseMLP),
    MoE(MoEBlock),
}

// Decoder Layer.
pub struct DecoderLayer {
    pub self_attn: DeepSeekV3Attention,
    pub mlp: MLPType,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
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
        args: &DeepSeekV3Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            DeepSeekV3Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;

        let mlp = if args.is_moe_layer(layer_idx) {
            MLPType::MoE(MoEBlock::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            MLPType::Dense(DenseMLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_attn_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// DeepSeek-V3 Model.
pub struct DeepSeekV3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
    pub config: DeepSeekV3Config,
}

impl DeepSeekV3Model {
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        // Preserve any caller-provided mask (for padded prefill) but let the
        // attention stack use its implicit causal path for standard prefill.
        let mask = _mask.map(mlxcel_core::copy);

        // Pass through layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask.as_deref());
        }

        // Final norm and lm_head
        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Text-embedding lookup, exposed so a VLM wrapper can build merged
    /// text/vision embeddings before calling [`Self::forward_impl_with_embeddings`].
    ///
    /// Used by: `vision::kimi_vl::KimiVLModel`.
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward pass that optionally consumes precomputed input embeddings
    /// (the VLM image path) instead of embedding `input_ids`. When
    /// `input_embeddings` is `None` this is identical to [`Self::forward_impl`].
    ///
    /// Used by: `vision::kimi_vl::KimiVLModel`.
    pub fn forward_impl_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embed_tokens.forward(input_ids),
        };

        let mask = mask.map(mlxcel_core::copy);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask.as_deref());
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn make_caches_impl(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, DeepSeekV3Config), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: DeepSeekV3Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Sanitize weights (stack expert weights if needed)
        let weights = Self::sanitize_weights(weights, &config);

        // Create model
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    /// Public wrapper over [`Self::sanitize_weights`] for callers outside of
    /// `src/models/deepseek_v3.rs` (e.g. the pipeline-parallel stage executor
    /// that needs to sanitise the full weight map before partial-loading
    /// filters out layers from other stages).
    ///
    /// Used by: DeepSeek V3 pipeline stage executor
    pub fn sanitize_weights_with_args(weights: WeightMap, config: &DeepSeekV3Config) -> WeightMap {
        Self::sanitize_weights(weights, config)
    }

    fn sanitize_weights(mut weights: WeightMap, config: &DeepSeekV3Config) -> WeightMap {
        // Remap pre-stacked weight names that omit the `.mlp.` segment.
        //
        // Some HuggingFace MLX shards for DeepSeek-V3 store pre-stacked MoE expert
        // weights under keys like `model.layers.{l}.switch_mlp.*` rather than
        // `model.layers.{l}.mlp.switch_mlp.*`. Similarly, the router gate weight may
        // appear as `model.layers.{l}.mlp.gate_weight` instead of the expected
        // `model.layers.{l}.mlp.gate.weight` / `model.layers.{l}.mlp.gate.e_score_correction_bias`.
        //
        // This remapping step normalises these alternative naming conventions before
        // the rest of sanitize_weights runs, so all subsequent logic can assume the
        // canonical key structure.
        {
            let num_layers = config.num_hidden_layers.saturating_sub(1);

            // Collect renames to avoid borrowing `weights` mutably while iterating it.
            let mut renames: Vec<(String, String)> = Vec::new();

            for l in 0..num_layers {
                let prefix = format!("model.layers.{l}");

                // Alias `{prefix}.switch_mlp.*` → `{prefix}.mlp.switch_mlp.*`
                for m in ["gate_proj", "down_proj", "up_proj"] {
                    for k in ["weight", "scales", "biases"] {
                        let src = format!("{prefix}.switch_mlp.{m}.{k}");
                        let dst = format!("{prefix}.mlp.switch_mlp.{m}.{k}");
                        if weights.contains_key(&src) && !weights.contains_key(&dst) {
                            renames.push((src, dst));
                        }
                    }
                }

                // Alias `{prefix}.mlp.gate_weight` → `{prefix}.mlp.gate.weight`
                let gate_weight_src = format!("{prefix}.mlp.gate_weight");
                let gate_weight_dst = format!("{prefix}.mlp.gate.weight");
                if weights.contains_key(&gate_weight_src) && !weights.contains_key(&gate_weight_dst)
                {
                    renames.push((gate_weight_src, gate_weight_dst));
                }

                // Alias `{prefix}.mlp.gate.e_score_correction_bias` if stored at
                // `{prefix}.mlp.e_score_correction_bias` (some checkpoints omit the sub-key)
                let bias_src = format!("{prefix}.mlp.e_score_correction_bias");
                let bias_dst = format!("{prefix}.mlp.gate.e_score_correction_bias");
                if weights.contains_key(&bias_src) && !weights.contains_key(&bias_dst) {
                    renames.push((bias_src, bias_dst));
                }
            }

            for (src, dst) in renames {
                if let Some(v) = weights.remove(&src) {
                    weights.insert(dst, v);
                }
            }
        }

        // Check if weights need stacking (individual experts.N format) or are already stacked (switch_mlp format)
        if let Some(n_routed) = config.n_routed_experts {
            let num_layers = config.num_hidden_layers.saturating_sub(1);

            for l in 0..num_layers {
                let prefix = format!("model.layers.{}", l);

                // Check if this layer has individual expert weights that need stacking
                let first_key = format!("{}.mlp.experts.0.gate_proj.weight", prefix);
                if weights.contains_key(&first_key) {
                    // Stack expert weights from individual experts to [num_experts, ...] tensors
                    for m in ["gate_proj", "down_proj", "up_proj"] {
                        for k in ["weight", "scales", "biases"] {
                            let check_key = format!("{}.mlp.experts.0.{}.{}", prefix, m, k);
                            if weights.contains_key(&check_key) {
                                let mut expert_arrays = Vec::new();
                                for e in 0..n_routed {
                                    let key = format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
                                    if let Some(w) = weights.get(&key) {
                                        expert_arrays.push(mlxcel_core::copy(w));
                                    }
                                }

                                if !expert_arrays.is_empty() {
                                    let stacked = stack_arrays(&expert_arrays, 0);
                                    let new_key = format!("{}.mlp.switch_mlp.{}.{}", prefix, m, k);
                                    weights.insert(new_key, stacked);

                                    for e in 0..n_routed {
                                        let key =
                                            format!("{}.mlp.experts.{}.{}.{}", prefix, e, m, k);
                                        weights.remove(&key);
                                    }
                                }
                            }
                        }
                    }
                }
                // If switch_mlp weights already exist, they're already stacked - no action needed
            }
        }

        // Decompose kv_b_proj into embed_q and unembed_out (if not already decomposed)
        let num_layers = config.num_hidden_layers.saturating_sub(1);
        let num_heads = config.num_attention_heads as i32;
        let head_dim = (config.qk_nope_head_dim + config.v_head_dim) as i32;
        let qk_nope_head_dim = config.qk_nope_head_dim as i32;

        for l in 0..num_layers {
            let prefix = format!("model.layers.{}.self_attn", l);
            let kv_b_key = format!("{}.kv_b_proj.weight", prefix);
            let embed_q_key = format!("{}.embed_q.weight", prefix);

            // Skip if already decomposed
            if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
                continue;
            }

            // Check if quantized
            let scales_key = format!("{}.kv_b_proj.scales", prefix);
            let is_quantized = weights.contains_key(&scales_key);

            // Get the kv_b_proj weight
            let w = weights.remove(&kv_b_key).unwrap();

            let w_full = if is_quantized {
                // Dequantize first
                let s = weights
                    .remove(&format!("{}.kv_b_proj.scales", prefix))
                    .unwrap();
                let b = weights
                    .remove(&format!("{}.kv_b_proj.biases", prefix))
                    .unwrap();

                let w_shape = mlxcel_core::array_shape(&w);
                let s_shape = mlxcel_core::array_shape(&s);
                let kv_lora_rank = config.kv_lora_rank as i32;
                let inferred_bits = (w_shape[w_shape.len() - 1] * 32) / kv_lora_rank;
                let inferred_gs = kv_lora_rank / s_shape[s_shape.len() - 1];

                unsafe {
                    mlxcel_core::dequantize(
                        &w,
                        &s,
                        &*b as *const _,
                        inferred_gs,
                        inferred_bits,
                        "affine",
                    )
                }
            } else {
                mlxcel_core::copy(&w)
            };

            // Reshape: [num_heads * head_dim, kv_lora_rank] → [num_heads, head_dim, kv_lora_rank]
            let w_3d = mlxcel_core::reshape(&w_full, &[num_heads, head_dim, -1]);

            // Split: wk = [:, :qk_nope_head_dim, :], wv = [:, qk_nope_head_dim:, :]
            let wk = slice_axis(&w_3d, 1, 0, qk_nope_head_dim);
            let wv = slice_axis(&w_3d, 1, qk_nope_head_dim, -1);

            // embed_q weight: wk.swapaxes(-1, -2) = [num_heads, kv_lora_rank, qk_nope_head_dim]
            let wk = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);
            // Make contiguous
            let wk = mlxcel_core::copy(&wk);
            let wv = mlxcel_core::copy(&wv);

            // Store as non-quantized (dequantized) MultiLinear weights
            weights.insert(format!("{}.embed_q.weight", prefix), wk);
            weights.insert(format!("{}.unembed_out.weight", prefix), wv);
        }

        // Remove multi-token prediction layer (last layer) and rotary freqs
        let keys_to_remove: Vec<String> = weights
            .keys()
            .filter(|k| {
                k.starts_with(&format!(
                    "model.layers.{}",
                    config.num_hidden_layers.saturating_sub(1)
                )) || k.contains("rotary_emb.inv_freq")
            })
            .cloned()
            .collect();

        for key in keys_to_remove {
            weights.remove(&key);
        }

        weights
    }

    pub fn from_weights(weights: &WeightMap, args: &DeepSeekV3Config) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers (excluding the last multi-token prediction layer)
        let num_layers = args.num_hidden_layers.saturating_sub(1);
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = DecoderLayer::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load lm_head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: args.clone(),
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
impl LanguageModel for DeepSeekV3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.make_caches_impl()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![1] // DeepSeek-V3 EOS token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal DeepSeekV3Config for testing weight sanitization.
    fn test_config_moe() -> DeepSeekV3Config {
        DeepSeekV3Config {
            model_type: "deepseek_v3".to_string(),
            vocab_size: 1,
            hidden_size: 1,
            intermediate_size: 1,
            moe_intermediate_size: 1,
            num_hidden_layers: 3, // 2 real layers + 1 MTP layer (last is dropped)
            num_attention_heads: 16,
            num_key_value_heads: 1,
            n_shared_experts: None,
            n_routed_experts: Some(4),
            routed_scaling_factor: 1.0,
            kv_lora_rank: 32,
            q_lora_rank: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 64,
            qk_nope_head_dim: 64,
            topk_method: "greedy".to_string(),
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: false,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 1,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            max_position_embeddings: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            quantization: None,
        }
    }

    /// Insert a dummy (empty) weight entry — sufficient for key-presence checks.
    fn insert_dummy(weights: &mut WeightMap, key: &str) {
        weights.insert(
            key.to_string(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
        );
    }

    #[test]
    fn sanitize_weights_remaps_switch_mlp_without_mlp_segment() {
        let config = test_config_moe();
        let mut weights = WeightMap::new();

        // Simulate pre-stacked weights stored WITHOUT the `.mlp.` segment
        for l in 0..2usize {
            for m in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    insert_dummy(
                        &mut weights,
                        &format!("model.layers.{l}.switch_mlp.{m}.{k}"),
                    );
                }
            }
        }

        let out = DeepSeekV3Model::sanitize_weights(weights, &config);

        // After sanitization, keys must have the `.mlp.` segment
        for l in 0..2usize {
            for m in ["gate_proj", "down_proj", "up_proj"] {
                let canonical_key = format!("model.layers.{l}.mlp.switch_mlp.{m}.weight");
                assert!(
                    out.contains_key(&canonical_key),
                    "expected key '{canonical_key}' not found after remapping"
                );
                // Old key must be gone
                let old_key = format!("model.layers.{l}.switch_mlp.{m}.weight");
                assert!(
                    !out.contains_key(&old_key),
                    "old key '{old_key}' should have been removed"
                );
            }
        }
    }

    #[test]
    fn sanitize_weights_does_not_overwrite_existing_canonical_keys() {
        let config = test_config_moe();
        let mut weights = WeightMap::new();

        // Both old and canonical keys present — canonical should survive
        for l in 0..2usize {
            for m in ["gate_proj"] {
                {
                    let k = "weight";
                    insert_dummy(
                        &mut weights,
                        &format!("model.layers.{l}.switch_mlp.{m}.{k}"),
                    );
                    insert_dummy(
                        &mut weights,
                        &format!("model.layers.{l}.mlp.switch_mlp.{m}.{k}"),
                    );
                }
            }
        }

        let out = DeepSeekV3Model::sanitize_weights(weights, &config);

        // Canonical key must still be present
        for l in 0..2usize {
            assert!(
                out.contains_key(&format!("model.layers.{l}.mlp.switch_mlp.gate_proj.weight")),
                "canonical key should survive when both old and canonical were present"
            );
        }
    }
}
