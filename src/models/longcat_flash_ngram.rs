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

//! LongcatFlash and LongcatFlashNgram model implementation
//!
//! Key Features:
//! - MLA (Multi-head Latent Attention) with Q/KV LoRA compression
//! - Dual sub-layer decoder architecture (2x attn + 2x MLP per layer)
//! - MoE with identity zero experts and SwitchGLU
//! - NgramEmbedding with n-gram hash lookups (ngram variant)
//! - CacheList per layer (2 KVCaches) + ArraysCache for ngram context
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/longcat_flash.py, longcat_flash_ngram.py

use crate::models::switch_layers::SwitchGLU;
use mlxcel_core::dtype;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{
    Embedding, KVCache, MultiLinear, RMSNorm, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::{create_causal_mask, slice_axis, stack_arrays};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LongcatFlashNgramConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub moe_topk: usize,
    pub expert_ffn_hidden_size: usize,
    pub n_routed_experts: usize,
    pub zero_expert_num: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub mla_scale_q_lora: bool,
    #[serde(default)]
    pub mla_scale_kv_lora: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_zero_expert_type")]
    pub zero_expert_type: String,
    #[serde(default = "default_ngram_vocab_size_ratio")]
    pub ngram_vocab_size_ratio: usize,
    #[serde(default = "default_emb_neighbor_num")]
    pub emb_neighbor_num: usize,
    #[serde(default = "default_emb_split_num")]
    pub emb_split_num: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub router_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_zero_expert_type() -> String {
    "identity".to_string()
}
fn default_ngram_vocab_size_ratio() -> usize {
    78
}
fn default_emb_neighbor_num() -> usize {
    4
}
fn default_emb_split_num() -> usize {
    4
}

impl LongcatFlashNgramConfig {
    fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    fn get_attention_scale(&self) -> f32 {
        let qk_head_dim = self.qk_head_dim() as f32;
        let mut scale = qk_head_dim.powf(-0.5);

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

    fn is_ngram_model(&self) -> bool {
        self.model_type == "longcat_flash_ngram"
    }
}

// Helper: get weight with copy.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Missing weight: {name}"))
}

// LongcatFlashMLA - Multi-head Latent Attention.
// Used by: LongcatFlash, LongcatFlashNgram.
struct LongcatFlashMLA {
    // Q projection with LoRA
    q_a_proj: Option<UnifiedLinear>,
    q_a_layernorm: Option<RMSNorm>,
    q_b_proj: Option<UnifiedLinear>,
    // Q projection without LoRA
    q_proj: Option<UnifiedLinear>,

    // KV projection with compression
    kv_a_proj_with_mqa: UnifiedLinear,
    kv_a_layernorm: RMSNorm,

    // MLA: embed_q and unembed_out replace kv_b_proj
    embed_q: MultiLinear,
    unembed_out: MultiLinear,

    o_proj: UnifiedLinear,

    num_heads: i32,
    qk_head_dim: i32,
    qk_rope_head_dim: i32,
    qk_nope_head_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
    scale: f32,
    rope_base: f32,
    mla_scale_q_lora: Option<f32>,
    mla_scale_kv_lora: Option<f32>,
}

impl LongcatFlashMLA {
    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Compute Q
        let q = if let (Some(q_a_proj), Some(q_a_norm), Some(q_b_proj)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a = q_a_proj.forward(x);
            let q_a_norm = q_a_norm.forward(&q_a);
            q_b_proj.forward(&q_a_norm)
        } else {
            self.q_proj.as_ref().unwrap().forward(x)
        };

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.qk_head_dim]);
        let mut q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Apply Q LoRA scaling
        if let Some(scale) = self.mla_scale_q_lora {
            let s = mlxcel_core::full_f32(&[1], scale, mlxcel_core::array_dtype(&q));
            q = mlxcel_core::multiply(&q, &s);
        }

        // Split Q into nope and pe
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        // Compute compressed KV
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let compressed = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);

        // Reshape k_pe to [B, 1, L, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        // KV latent with layernorm and optional scaling
        let mut kv_latent = self.kv_a_layernorm.forward(&compressed);
        if let Some(scale) = self.mla_scale_kv_lora {
            let s = mlxcel_core::full_f32(&[1], scale, mlxcel_core::array_dtype(&kv_latent));
            kv_latent = mlxcel_core::multiply(&kv_latent, &s);
        }

        // Apply RoPE
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

        // Expand kv_latent to [B, 1, L, kv_lora_rank] for cache
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);

        // Cache stores (kv_latent, k_pe)
        let (kv_latent, k_pe) = cache.update_and_fetch(kv_latent, k_pe);

        // Compute PE scores: (q_pe * scale) @ k_pe^T
        let scale_arr = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(&q_pe));
        let scaled_q_pe = mlxcel_core::multiply(&q_pe, &scale_arr);
        let k_pe_t = mlxcel_core::transpose_axes(&k_pe, &[0, 1, 3, 2]);
        let mut pe_scores = mlxcel_core::matmul(&scaled_q_pe, &k_pe_t);

        // Apply causal mask to pe_scores (additive: 0 for attend, -inf for don't attend)
        if let Some(m) = mask {
            pe_scores = mlxcel_core::add(&pe_scores, m);
        }

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

        let attn_out = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.v_head_dim]);

        self.o_proj.forward(&attn_out)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        // Q projection: with or without LoRA
        let has_q_lora = weights.contains_key(&format!("{prefix}.q_a_proj.weight"))
            || weights.contains_key(&format!("{prefix}.q_a_proj.scales"));

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if has_q_lora {
            let q_a =
                UnifiedLinear::from_weights(weights, &format!("{prefix}.q_a_proj"), gs, bits)?;
            let q_a_norm_w = get_weight_copy(weights, &format!("{prefix}.q_a_layernorm.weight"))?;
            let q_a_norm = RMSNorm::new(q_a_norm_w, args.rms_norm_eps);
            let q_b =
                UnifiedLinear::from_weights(weights, &format!("{prefix}.q_b_proj"), gs, bits)?;
            (Some(q_a), Some(q_a_norm), Some(q_b), None)
        } else {
            let q = UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), gs, bits)?;
            (None, None, None, Some(q))
        };

        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.kv_a_proj_with_mqa"),
            gs,
            bits,
        )?;
        let kv_a_norm_w = get_weight_copy(weights, &format!("{prefix}.kv_a_layernorm.weight"))?;
        let kv_a_layernorm = RMSNorm::new(kv_a_norm_w, args.rms_norm_eps);

        // MLA: embed_q and unembed_out (decomposed from kv_b_proj by sanitize_weights)
        let embed_q = MultiLinear::from_weights(weights, &format!("{prefix}.embed_q"), gs, bits)?;
        let unembed_out =
            MultiLinear::from_weights(weights, &format!("{prefix}.unembed_out"), gs, bits)?;

        let o_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.o_proj"), gs, bits)?;

        let mla_scale_q_lora = if args.mla_scale_q_lora {
            Some((args.hidden_size as f32 / args.q_lora_rank as f32).sqrt())
        } else {
            None
        };
        let mla_scale_kv_lora = if args.mla_scale_kv_lora {
            Some((args.hidden_size as f32 / args.kv_lora_rank as f32).sqrt())
        } else {
            None
        };

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            embed_q,
            unembed_out,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            qk_head_dim: args.qk_head_dim() as i32,
            qk_rope_head_dim: args.qk_rope_head_dim as i32,
            qk_nope_head_dim: args.qk_nope_head_dim as i32,
            v_head_dim: args.v_head_dim as i32,
            kv_lora_rank: args.kv_lora_rank as i32,
            scale: args.get_attention_scale(),
            rope_base: args.rope_theta,
            mla_scale_q_lora,
            mla_scale_kv_lora,
        })
    }
}

// LongcatFlashMLP - Dense MLP (used for per-sublayer MLPs).
struct LongcatFlashMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl LongcatFlashMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let _ = hidden_size; // Used by Python for is_expert branch; we just load from weights
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.up_proj"), gs, bits)?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.down_proj"),
                gs,
                bits,
            )?,
        })
    }
}

// LongcatFlashTopkRouter.
struct LongcatFlashTopkRouter {
    classifier: UnifiedLinear,
    e_score_correction_bias: UniquePtr<MlxArray>,
    top_k: usize,
    _n_routed_experts: usize, // includes zero_expert_num
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
}

impl LongcatFlashTopkRouter {
    /// Returns (topk_indices, topk_weights)
    fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let logits = self.classifier.forward(x);
        let scores = mlxcel_core::softmax(&logits, -1);

        // Add correction bias
        let corrected = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Top-k selection via argpartition
        let neg_corrected = mlxcel_core::negative(&corrected);
        let indices = mlxcel_core::argpartition(&neg_corrected, self.top_k as i32 - 1, -1);

        let shape = mlxcel_core::array_shape(&indices);
        let topk_indices = mlxcel_core::slice(&indices, &[0, 0], &[shape[0], self.top_k as i32]);

        // Gather original scores (not corrected) for topk
        let topk_weights = mlxcel_core::take_along_axis(&scores, &topk_indices, -1);

        // Normalize if needed
        let topk_weights = if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_weights, -1, true);
            let w_dtype = mlxcel_core::array_dtype(&topk_weights);
            let eps = mlxcel_core::full_f32(&[1], 1e-20, w_dtype);
            let denom = mlxcel_core::add(&sum, &eps);
            mlxcel_core::divide(&topk_weights, &denom)
        } else {
            topk_weights
        };

        // Scale
        let scale_arr = mlxcel_core::full_f32(
            &[1],
            self.routed_scaling_factor,
            mlxcel_core::array_dtype(&topk_weights),
        );
        let topk_weights = mlxcel_core::multiply(&topk_weights, &scale_arr);

        // Cast back to input dtype
        let input_dtype = mlxcel_core::array_dtype(x);
        let topk_weights = mlxcel_core::astype(&topk_weights, input_dtype);

        (topk_indices, topk_weights)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let total_experts = args.n_routed_experts + args.zero_expert_num;

        // classifier with special quantization (8-bit, group_size 64)
        let classifier = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.classifier"),
            64, // quant_predicate: group_size=64, bits=8 for classifier
            8,
        )?;

        let bias_key = format!("{prefix}.e_score_correction_bias");
        let e_score_correction_bias = if let Some(w) = weights.get(&bias_key) {
            mlxcel_core::copy(w)
        } else {
            mlxcel_core::zeros(&[total_experts as i32], dtype::FLOAT32)
        };

        let _ = (gs, bits); // router uses special quantization

        Ok(Self {
            classifier,
            e_score_correction_bias,
            top_k: args.moe_topk,
            _n_routed_experts: total_experts,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// LongcatFlashMoE - MoE with identity zero experts.
struct LongcatFlashMoE {
    router: LongcatFlashTopkRouter,
    switch_mlp: SwitchGLU,
    n_routed_experts: usize, // number of actual routed experts (not including zero)
}

impl LongcatFlashMoE {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (topk_indices, topk_weights) = self.router.forward(x);

        // Mask indices that point to zero (identity) experts
        let n_routed = mlxcel_core::from_slice_i32(&[self.n_routed_experts as i32], &[1]);
        let mask = mlxcel_core::greater_equal(&topk_indices, &n_routed);

        // Replace identity expert indices with 0 (they won't contribute)
        let zero = mlxcel_core::zeros_like(&topk_indices);
        let clamped_indices = mlxcel_core::where_cond(&mask, &zero, &topk_indices);

        // Zero out weights for identity experts in regular path
        let zero_f = mlxcel_core::zeros_like(&topk_weights);
        let regular_weights = mlxcel_core::where_cond(&mask, &zero_f, &topk_weights);

        // Run SwitchGLU for regular experts
        let regular_outputs = self.switch_mlp.forward(x, &clamped_indices);

        // Weight and sum expert outputs
        let weights_exp = mlxcel_core::expand_dims(&regular_weights, -1);
        let weighted = mlxcel_core::multiply(&regular_outputs, &weights_exp);
        let mut output = mlxcel_core::sum_axis(&weighted, -2, false);

        // Add identity expert contribution: x * sum(identity_weights)
        let identity_weights = mlxcel_core::where_cond(&mask, &topk_weights, &zero_f);
        let identity_sum = mlxcel_core::sum_axis(&identity_weights, -1, true);
        let identity_contrib = mlxcel_core::multiply(x, &identity_sum);
        output = mlxcel_core::add(&output, &identity_contrib);

        output
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        let router =
            LongcatFlashTopkRouter::from_weights(weights, args, &format!("{prefix}.router"))?;
        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{prefix}.switch_mlp"), gs, bits)?;

        Ok(Self {
            router,
            switch_mlp,
            n_routed_experts: args.n_routed_experts,
        })
    }
}

// LongcatFlashDecoderLayer - dual sub-layer architecture.
// Each layer has 2 attention + 2 MLP sub-layers plus a shared MoE MLP.
struct LongcatFlashDecoderLayer {
    self_attn: [LongcatFlashMLA; 2],
    mlps: [LongcatFlashMLP; 2],
    input_layernorm: [RMSNorm; 2],
    post_attention_layernorm: [RMSNorm; 2],
    moe_mlp: LongcatFlashMoE,
}

#[allow(clippy::needless_range_loop)]
impl LongcatFlashDecoderLayer {
    fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        caches: &mut [&mut KVCache; 2],
    ) -> UniquePtr<MlxArray> {
        let mut hidden = mlxcel_core::copy(x);
        let mut shortcut_mlp_output: Option<UniquePtr<MlxArray>> = None;

        for i in 0..2 {
            let residual = mlxcel_core::copy(&hidden);

            // Self-attention
            let normed = self.input_layernorm[i].forward(&hidden);
            let attn_out = self.self_attn[i].forward(&normed, caches[i], mask);
            hidden = mlxcel_core::add(&residual, &attn_out);

            // MLP
            let residual = mlxcel_core::copy(&hidden);
            let normed = self.post_attention_layernorm[i].forward(&hidden);

            // Compute MoE on first sub-layer's post-attn norm
            if i == 0 {
                shortcut_mlp_output = Some(self.moe_mlp.forward(&normed));
            }

            let mlp_out = self.mlps[i].forward(&normed);
            hidden = mlxcel_core::add(&residual, &mlp_out);

            // Add shortcut MoE output after second sub-layer
            if i == 1
                && let Some(ref shortcut) = shortcut_mlp_output
            {
                hidden = mlxcel_core::add(&hidden, shortcut);
            }
        }

        hidden
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        let self_attn_0 =
            LongcatFlashMLA::from_weights(weights, args, &format!("{prefix}.self_attn.0"))?;
        let self_attn_1 =
            LongcatFlashMLA::from_weights(weights, args, &format!("{prefix}.self_attn.1"))?;

        let mlp_0 = LongcatFlashMLP::from_weights(
            weights,
            &format!("{prefix}.mlps.0"),
            args.hidden_size,
            gs,
            bits,
        )?;
        let mlp_1 = LongcatFlashMLP::from_weights(
            weights,
            &format!("{prefix}.mlps.1"),
            args.hidden_size,
            gs,
            bits,
        )?;

        let in_norm_0_w = get_weight_copy(weights, &format!("{prefix}.input_layernorm.0.weight"))?;
        let in_norm_1_w = get_weight_copy(weights, &format!("{prefix}.input_layernorm.1.weight"))?;
        let post_norm_0_w = get_weight_copy(
            weights,
            &format!("{prefix}.post_attention_layernorm.0.weight"),
        )?;
        let post_norm_1_w = get_weight_copy(
            weights,
            &format!("{prefix}.post_attention_layernorm.1.weight"),
        )?;

        let moe_mlp = LongcatFlashMoE::from_weights(weights, args, &format!("{prefix}.mlp"))?;

        Ok(Self {
            self_attn: [self_attn_0, self_attn_1],
            mlps: [mlp_0, mlp_1],
            input_layernorm: [
                RMSNorm::new(in_norm_0_w, args.rms_norm_eps),
                RMSNorm::new(in_norm_1_w, args.rms_norm_eps),
            ],
            post_attention_layernorm: [
                RMSNorm::new(post_norm_0_w, args.rms_norm_eps),
                RMSNorm::new(post_norm_1_w, args.rms_norm_eps),
            ],
            moe_mlp,
        })
    }
}

// NgramEmbedding - n-gram hash embeddings.
struct NgramEmbedder {
    embedding: UnifiedEmbedding,
    post_proj: UnifiedLinear,
    _vocab_dim: usize,
}

struct NgramEmbedding {
    word_embeddings: UnifiedEmbedding,
    embedders: Vec<NgramEmbedder>,
    vocab_mods: HashMap<(usize, usize), Vec<i64>>,
    _vocab_size: usize,
    m: usize, // ngram_vocab_size_ratio * vocab_size
    k: usize, // emb_split_num
    n: usize, // emb_neighbor_num
}

impl NgramEmbedding {
    fn compute_vocab_mods(
        n: usize,
        k: usize,
        m: usize,
        vocab_size: usize,
    ) -> HashMap<(usize, usize), Vec<i64>> {
        let mut vocab_mods = HashMap::new();
        for i in 2..=n {
            for j in 0..k {
                let index = (i - 2) * k + j;
                let emb_vocab_dim = (m + index * 2 + 1) as i64;
                let mut mods = Vec::new();
                let mut power_mod: i64 = 1;
                for _ in 0..i - 1 {
                    power_mod = (power_mod * vocab_size as i64) % emb_vocab_dim;
                    mods.push(power_mod);
                }
                vocab_mods.insert((i, j), mods);
            }
        }
        vocab_mods
    }

    fn shift_right(x: &MlxArray, n: usize) -> UniquePtr<MlxArray> {
        if n == 0 {
            return mlxcel_core::copy(x);
        }
        let shape = mlxcel_core::array_shape(x);
        let batch_size = shape[0];
        let seq_len = shape[1];
        if seq_len <= n as i32 {
            return mlxcel_core::zeros_like(x);
        }
        let zeros = mlxcel_core::zeros(&[batch_size, n as i32], dtype::INT64);
        let kept = mlxcel_core::slice(x, &[0, 0], &[batch_size, seq_len - n as i32]);
        mlxcel_core::concatenate(&zeros, &kept, -1)
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        context: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[shape.len() - 1];

        let input_i64 = mlxcel_core::astype(input_ids, dtype::INT64);

        // Update context for autoregressive decoding
        let cur_context = if let Some(prev_ctx) = context.as_ref() {
            let prev = prev_ctx.as_ref().unwrap();
            let joined = mlxcel_core::concatenate(prev, &input_i64, -1);
            let ctx_shape = mlxcel_core::array_shape(&joined);
            let ctx_len = ctx_shape[ctx_shape.len() - 1];
            let keep = (self.n as i32 - 1).max(0);
            if ctx_len > keep {
                let start = ctx_len - keep;
                mlxcel_core::slice(&joined, &[0, start], &[-1, ctx_len])
            } else {
                joined
            }
        } else {
            mlxcel_core::copy(&input_i64)
        };

        // Save context for next call
        let ctx_shape = mlxcel_core::array_shape(&cur_context);
        let ctx_len = ctx_shape[ctx_shape.len() - 1];
        let keep = (self.n as i32 - 1).max(0);
        *context = Some(if ctx_len > keep {
            let start = ctx_len - keep;
            mlxcel_core::slice(&cur_context, &[0, start], &[-1, ctx_len])
        } else {
            mlxcel_core::copy(&cur_context)
        });

        // Word embeddings
        let mut x = self.word_embeddings.forward(input_ids);

        // Compute shifted IDs
        let mut shifted_ids: HashMap<usize, UniquePtr<MlxArray>> = HashMap::new();
        for i in 2..=self.n {
            shifted_ids.insert(i, Self::shift_right(&cur_context, i - 1));
        }

        // Add n-gram embeddings
        let mut emb_idx = 0;
        for i in 2..=self.n {
            for j in 0..self.k {
                let emb_vocab_dim = (self.m + emb_idx * 2 + 1) as i64;
                let vocab_mods = &self.vocab_mods[&(i, j)];

                // Compute ngram_ids = context + sum(shifted * power_mod)
                let mut ngram_ids = mlxcel_core::copy(&cur_context);
                for k_idx in 2..=i {
                    let shifted = &shifted_ids[&k_idx];
                    let mod_val = vocab_mods[k_idx - 2];
                    let mod_arr = mlxcel_core::from_slice_i64(&[mod_val], &[1]);
                    let term = mlxcel_core::multiply(shifted, &mod_arr);
                    ngram_ids = mlxcel_core::add(&ngram_ids, &term);
                }

                // new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                let dim_arr = mlxcel_core::from_slice_i64(&[emb_vocab_dim], &[1]);
                let modded = mlxcel_core::remainder(&ngram_ids, &dim_arr);
                let modded_shape = mlxcel_core::array_shape(&modded);
                let modded_len = modded_shape[modded_shape.len() - 1];
                let new_ids = if modded_len > seq_len {
                    mlxcel_core::slice(&modded, &[0, modded_len - seq_len], &[-1, modded_len])
                } else {
                    modded
                };

                // Cast to int32 for embedding lookup
                let new_ids_i32 = mlxcel_core::astype(&new_ids, dtype::INT32);

                // Lookup embedding and project
                let x_ngram = self.embedders[emb_idx].embedding.forward(&new_ids_i32);
                let x_proj = self.embedders[emb_idx].post_proj.forward(&x_ngram);
                x = mlxcel_core::add(&x, &x_proj);

                emb_idx += 1;
            }
        }

        // Normalize by (1 + k * (n-1))
        let divisor = (1 + self.k * (self.n - 1)) as f32;
        let div_arr = mlxcel_core::full_f32(&[1], divisor, mlxcel_core::array_dtype(&x));
        mlxcel_core::divide(&x, &div_arr)
    }

    fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();
        let m = args.ngram_vocab_size_ratio * args.vocab_size;
        let k = args.emb_split_num;
        let n = args.emb_neighbor_num;
        let num_embedders = k * (n - 1);
        let _emb_dim = args.hidden_size / num_embedders;

        let word_emb_w = get_weight_copy(weights, &format!("{prefix}.word_embeddings.weight"))?;
        let word_embeddings = UnifiedEmbedding::Regular(Embedding::new(word_emb_w));

        let mut embedders = Vec::with_capacity(num_embedders);
        for i in 0..num_embedders {
            let emb_vocab_size = m + i * 2 + 1;
            let emb_key = format!("{prefix}.embedders.{i}.weight");
            let emb_w = get_weight_copy(weights, &emb_key)?;
            let embedding = UnifiedEmbedding::Regular(Embedding::new(emb_w));
            let post_proj = UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.post_projs.{i}"),
                gs,
                bits,
            )?;
            embedders.push(NgramEmbedder {
                embedding,
                post_proj,
                _vocab_dim: emb_vocab_size,
            });
        }

        let vocab_mods = Self::compute_vocab_mods(n, k, m, args.vocab_size);

        Ok(Self {
            word_embeddings,
            embedders,
            vocab_mods,
            _vocab_size: args.vocab_size,
            m,
            k,
            n,
        })
    }
}

// LongcatFlashNgramModel - top-level model.
pub struct LongcatFlashNgramModel {
    // For ngram model
    ngram_embeddings: Option<NgramEmbedding>,
    // For plain longcat_flash model
    embed_tokens: Option<UnifiedEmbedding>,

    layers: Vec<LongcatFlashDecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    num_layers: usize,
    _is_ngram: bool,
}

impl LongcatFlashNgramModel {
    pub fn load(model_path: &str) -> Result<(Self, usize), String> {
        let path = Path::new(model_path);
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: LongcatFlashNgramConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config: {e}"))?;

        let mut weights = crate::models::load_text_weights(path, None)?;
        weights = sanitize_weights(weights, &args);

        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args.vocab_size))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &LongcatFlashNgramConfig,
    ) -> Result<Self, String> {
        let gs = args.group_size();
        let bits = args.bits();

        // Embeddings
        let (ngram_embeddings, embed_tokens) = if args.is_ngram_model() {
            let ngram = NgramEmbedding::from_weights(weights, args, "model.ngram_embeddings")?;
            (Some(ngram), None)
        } else {
            let emb_w = get_weight_copy(weights, "model.embed_tokens.weight")?;
            (None, Some(UnifiedEmbedding::Regular(Embedding::new(emb_w))))
        };

        // Layers
        let mut layers = Vec::with_capacity(args.num_layers);
        for i in 0..args.num_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(LongcatFlashDecoderLayer::from_weights(
                weights, args, &prefix,
            )?);
        }

        // Final norm
        let norm_w = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_w, args.rms_norm_eps);

        // LM head
        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?;

        Ok(Self {
            ngram_embeddings,
            embed_tokens,
            layers,
            norm,
            lm_head,
            num_layers: args.num_layers,
            _is_ngram: args.is_ngram_model(),
        })
    }
}

impl LongcatFlashNgramModel {
    /// Internal forward pass with proper cache management
    fn internal_forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [[KVCache; 2]],
        ngram_context: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        // Compute embeddings
        let h = if let Some(ref ngram) = self.ngram_embeddings {
            ngram.forward(input_ids, ngram_context)
        } else {
            self.embed_tokens.as_ref().unwrap().forward(input_ids)
        };

        // Create causal mask using first sub-layer cache of first layer
        let h_shape = mlxcel_core::array_shape(&h);
        let seq_len = h_shape[h_shape.len() - 2];
        let offset = caches[0][0].offset;
        let mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        // Forward through layers
        let mask_ref = mask.as_ref().map(|m| m.as_ref().unwrap());
        let mut hidden = h;
        for (i, layer) in self.layers.iter().enumerate() {
            let [ref mut c0, ref mut c1] = caches[i];
            hidden = layer.forward(&hidden, mask_ref, &mut [c0, c1]);
        }

        // Final norm and LM head
        let normed = self.norm.forward(&hidden);
        self.lm_head.forward(&normed)
    }
}

impl LanguageModel for LongcatFlashNgramModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Create fresh internal caches (2 KVCaches per layer for dual sub-layer arch)
        let mut internal_caches: Vec<[KVCache; 2]> = (0..self.num_layers)
            .map(|_| [KVCache::new(), KVCache::new()])
            .collect();
        let mut ngram_context = None;

        self.internal_forward(input_ids, &mut internal_caches, &mut ngram_context)
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn supports_batching(&self) -> bool {
        false // LongcatFlash uses internal dual-layer caches, not compatible with per-sequence KV isolation
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Return dummy caches; real caches are managed internally
        (0..self.num_layers).map(|_| KVCache::new()).collect()
    }
}

// Weight sanitization.
pub fn sanitize_weights(mut weights: WeightMap, args: &LongcatFlashNgramConfig) -> WeightMap {
    // Stack MoE expert weights into SwitchGLU format
    for l in 0..args.num_layers {
        let prefix = format!("model.layers.{l}");
        for (old_name, new_name) in [
            ("gate_proj", "gate_proj"),
            ("down_proj", "down_proj"),
            ("up_proj", "up_proj"),
        ] {
            for weight_kind in ["weight", "scales", "biases"] {
                let first_key = format!("{prefix}.mlp.experts.0.{old_name}.{weight_kind}");
                if weights.contains_key(&first_key) {
                    let mut to_join = Vec::new();
                    for e in 0..args.n_routed_experts {
                        let key = format!("{prefix}.mlp.experts.{e}.{old_name}.{weight_kind}");
                        if let Some(w) = weights.remove(&key) {
                            to_join.push(w);
                        }
                    }
                    if !to_join.is_empty() {
                        let stacked = stack_arrays(&to_join, 0);
                        let new_key = format!("{prefix}.mlp.switch_mlp.{new_name}.{weight_kind}");
                        weights.insert(new_key, stacked);
                    }
                }
            }
        }
    }

    // Decompose kv_b_proj into embed_q and unembed_out for MLA
    let num_heads = args.num_attention_heads as i32;
    let head_dim = (args.qk_nope_head_dim + args.v_head_dim) as i32;
    let qk_nope_head_dim = args.qk_nope_head_dim as i32;

    for l in 0..args.num_layers {
        for i in 0..2 {
            let prefix = format!("model.layers.{l}.self_attn.{i}");
            let kv_b_key = format!("{prefix}.kv_b_proj.weight");
            let embed_q_key = format!("{prefix}.embed_q.weight");

            // Skip if already decomposed or no kv_b_proj
            if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
                continue;
            }

            // Check if quantized
            let scales_key = format!("{prefix}.kv_b_proj.scales");
            let is_quantized = weights.contains_key(&scales_key);

            let w = weights.remove(&kv_b_key).unwrap();

            let w_full = if is_quantized {
                let s = weights
                    .remove(&format!("{prefix}.kv_b_proj.scales"))
                    .unwrap();
                let b = weights
                    .remove(&format!("{prefix}.kv_b_proj.biases"))
                    .unwrap();
                let w_shape = mlxcel_core::array_shape(&w);
                let s_shape = mlxcel_core::array_shape(&s);
                let kv_lora_rank = args.kv_lora_rank as i32;
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

            // embed_q: wk.swapaxes(-1, -2) = [num_heads, kv_lora_rank, qk_nope_head_dim]
            let wk = mlxcel_core::transpose_axes(&wk, &[0, 2, 1]);
            let wk = mlxcel_core::copy(&wk);
            let wv = mlxcel_core::copy(&wv);

            weights.insert(format!("{prefix}.embed_q.weight"), wk);
            weights.insert(format!("{prefix}.unembed_out.weight"), wv);
        }
    }

    // Remove MTP (multi-token prediction) weights
    weights.retain(|k, _| !k.starts_with("model.mtp"));

    // For ngram model: rename embed_tokens → ngram_embeddings.word_embeddings
    if args.is_ngram_model()
        && let Some(w) = weights.remove("model.embed_tokens.weight")
    {
        weights.insert(
            "model.ngram_embeddings.word_embeddings.weight".to_string(),
            w,
        );
    }

    weights
}
