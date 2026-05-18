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

//! GLM4 MoE Lite model implementation using mlxcel-core
//!
//! GLM4 MoE Lite architecture features:
//! - MLA (Multi-head Latent Attention) with LoRA-style KV compression
//! - Split head dimensions: qk_nope_head_dim, qk_rope_head_dim, v_head_dim
//! - Compressed Q projection: q_a_proj -> q_a_layernorm -> q_b_proj
//! - Compressed KV projection with embed_q and unembed_out per-head linear layers
//! - Sparse MoE with grouped expert selection
//! - Sigmoid routing with e_score_correction_bias

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, MultiLinear, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    // MLA dimensions
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,

    // MoE parameters
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_group: usize,
    #[serde(default)]
    pub topk_group: usize,
    #[serde(default = "default_routed_scaling")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_moe_freq")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub group_size: Option<i32>,
    #[serde(default)]
    pub bits: Option<i32>,
}

fn default_routed_scaling() -> f32 {
    1.0
}

fn default_moe_freq() -> usize {
    1
}

impl ModelArgs {
    /// Get Q head dimension (qk_nope + qk_rope)
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Check if a layer should use MoE (vs dense MLP)
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.n_routed_experts.is_some()
            && layer_idx >= self.first_k_dense_replace
            && layer_idx.is_multiple_of(self.moe_layer_freq)
    }

    pub fn group_size(&self) -> i32 {
        self.group_size.unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.bits.unwrap_or(4)
    }
}

// MLA Attention (Multi-head Latent Attention).
pub struct MlaAttention {
    // Q projection (LoRA compressed or direct)
    pub q_a_proj: Option<UnifiedLinear>,
    pub q_a_layernorm: Option<RMSNorm>,
    pub q_b_proj: Option<UnifiedLinear>,
    pub q_proj: Option<UnifiedLinear>,

    // KV projection (LoRA compressed)
    pub kv_a_proj_with_mqa: UnifiedLinear,
    pub kv_a_layernorm: RMSNorm,

    // MLA: embed_q and unembed_out replace kv_b_proj
    pub embed_q: MultiLinear,
    pub unembed_out: MultiLinear,

    pub o_proj: UnifiedLinear,

    pub num_heads: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub kv_lora_rank: i32,
    pub q_head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
}

impl MlaAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Q projection
        let q = if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a_out = q_a.forward(x);
            let q_a_normed = q_a_ln.forward(&q_a_out);
            q_b.forward(&q_a_normed)
        } else if let Some(q_proj) = &self.q_proj {
            q_proj.forward(x)
        } else {
            return mlxcel_core::zeros(
                &[b, l, self.num_heads * self.q_head_dim],
                mlxcel_core::dtype::FLOAT16,
            );
        };

        // Reshape Q: [B, L, heads * q_head_dim] -> [B, heads, L, q_head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);

        // Split Q into nope and rope parts
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        // KV projection with MQA rope
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);

        // Split into kv_latent and k_pe
        let kv_latent = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);

        // Normalize KV latent
        let kv_latent = self.kv_a_layernorm.forward(&kv_latent);

        // Reshape k_pe: [B, L, rope_dim] -> [B, 1, L, rope_dim]
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

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

        // Expand kv_latent: [B, L, kv_lora_rank] -> [B, 1, L, kv_lora_rank]
        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);

        // Cache stores (kv_latent, k_pe) for memory efficiency
        let (kv_latent, k_pe) = cache.update_and_fetch(kv_latent, k_pe);

        // Compute positional encoding scores: pe_scores = (q_pe * scale) @ k_pe.T
        let scale_scalar = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(&q_pe));
        let q_pe_scaled = mlxcel_core::multiply(&q_pe, &scale_scalar);
        let k_pe_t = mlxcel_core::transpose_axes(&k_pe, &[0, 1, 3, 2]);
        let pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);

        // Apply causal mask to pe_scores
        let pe_scores = if let Some(m) = mask {
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

        // Transpose and reshape: [B, heads, L, v_head_dim] -> [B, L, heads * v_head_dim]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * self.v_head_dim]);

        self.o_proj.forward(&output)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Q projection
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if args.q_lora_rank.is_some() {
            let q_a = UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_a_proj", prefix),
                group_size,
                bits,
            )?;
            let q_a_ln_weight =
                get_weight_copy(weights, &format!("{}.q_a_layernorm.weight", prefix))?;
            let q_a_ln = RMSNorm::new(q_a_ln_weight, args.rms_norm_eps);
            let q_b = UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_b_proj", prefix),
                group_size,
                bits,
            )?;
            (Some(q_a), Some(q_a_ln), Some(q_b), None)
        } else {
            let q = UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_proj", prefix),
                group_size,
                bits,
            )?;
            (None, None, None, Some(q))
        };

        // KV projection
        let kv_a_proj_with_mqa = UnifiedLinear::from_weights(
            weights,
            &format!("{}.kv_a_proj_with_mqa", prefix),
            group_size,
            bits,
        )?;
        let kv_a_ln_weight =
            get_weight_copy(weights, &format!("{}.kv_a_layernorm.weight", prefix))?;
        let kv_a_layernorm = RMSNorm::new(kv_a_ln_weight, args.rms_norm_eps);

        // MLA per-head projections (embed_q and unembed_out)
        let embed_q =
            MultiLinear::from_weights(weights, &format!("{}.embed_q", prefix), group_size, bits)?;
        let unembed_out = MultiLinear::from_weights(
            weights,
            &format!("{}.unembed_out", prefix),
            group_size,
            bits,
        )?;

        // Output projection
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let q_head_dim = args.q_head_dim() as i32;
        let scale = 1.0 / (q_head_dim as f32).sqrt();

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
            qk_nope_head_dim: args.qk_nope_head_dim as i32,
            qk_rope_head_dim: args.qk_rope_head_dim as i32,
            v_head_dim: args.v_head_dim as i32,
            kv_lora_rank: args.kv_lora_rank as i32,
            q_head_dim,
            scale,
            rope_base: args.rope_theta,
        })
    }
}

// MLP.
pub struct MLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
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

// SwitchLinear for MoE experts.
/// Stacked linear layers for MoE experts (Variant B: no num_experts field).
/// Supports both quantized (gather_qmm) and non-quantized (gather_mm) forward paths.
pub enum SwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
    },
}

impl SwitchLinear {
    pub fn forward(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        sorted_indices: bool,
    ) -> UniquePtr<MlxArray> {
        match self {
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                // SAFETY: weight/scales/biases are valid UniquePtr-owned MlxArray values.
                unsafe {
                    mlxcel_core::gather_qmm(
                        x,
                        weight,
                        scales,
                        biases
                            .as_ref()
                            .map(|b| b as *const _)
                            .unwrap_or(std::ptr::null()),
                        std::ptr::null(),
                        indices as *const _,
                        true,
                        *group_size,
                        *bits,
                        sorted_indices,
                        "affine",
                    )
                }
            }
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
                // SAFETY: wt and indices are valid MlxArray values in scope.
                unsafe {
                    mlxcel_core::gather_mm(
                        x,
                        &wt,
                        std::ptr::null(),
                        indices as *const _,
                        sorted_indices,
                    )
                }
            }
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            let scales = mlxcel_core::copy(weights.get(&scales_key).unwrap());
            let biases = get_weight_copy(weights, &format!("{}.biases", prefix))?;
            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            })
        } else {
            Ok(Self::Regular { weight })
        }
    }
}

// SwitchGLU for MoE.
pub struct SwitchGLU {
    pub gate_proj: SwitchLinear,
    pub up_proj: SwitchLinear,
    pub down_proj: SwitchLinear,
}

impl SwitchGLU {
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];
        let total = n_tokens * top_k;
        let do_sort = total >= 64;

        let x_expanded = mlxcel_core::expand_dims(x, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) = self.gather_sort(&x_expanded, indices);
            let gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            let up = self.up_proj.forward(&sorted_x, &sorted_idx, true);
            let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);
            self.scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            let gate = self.gate_proj.forward(&x_expanded, indices, false);
            let up = self.up_proj.forward(&x_expanded, indices, false);
            let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
            let output = self.down_proj.forward(&activated, indices, false);
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    fn gather_sort(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let indices_shape = mlxcel_core::array_shape(indices);
        let top_k = indices_shape[indices_shape.len() - 1];

        let flat_indices = mlxcel_core::reshape(indices, &[-1]);
        let order = mlxcel_core::argsort(&flat_indices, -1);
        let inv_order = mlxcel_core::argsort(&order, -1);

        let x_shape = mlxcel_core::array_shape(x);
        let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);

        let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
        let token_indices = mlxcel_core::divide(&order, &top_k_arr);
        let token_indices = mlxcel_core::astype(&token_indices, mlxcel_core::dtype::INT32);

        let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);
        let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);

        (sorted_x, sorted_indices, inv_order)
    }

    fn scatter_unsort(
        &self,
        x: &MlxArray,
        inv_order: &MlxArray,
        orig_shape: &[i32],
    ) -> UniquePtr<MlxArray> {
        let unsorted = mlxcel_core::take(x, inv_order, 0);
        let x_shape = mlxcel_core::array_shape(&unsorted);
        let n_tokens = orig_shape[0];
        let top_k = orig_shape[1];
        let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
        mlxcel_core::squeeze_axis(&reshaped, 2)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }
}

// MoE Layer.
pub struct MoELayer {
    pub gate_weight: UniquePtr<MlxArray>,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub switch_mlp: SwitchGLU,
    pub shared_experts: Option<MLP>,
    pub num_experts_per_tok: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}

impl MoELayer {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Router: x @ gate_weight.T
        let gate_transposed = mlxcel_core::transpose_axes(&self.gate_weight, &[1, 0]);
        let logits = mlxcel_core::matmul(&x_flat, &gate_transposed);

        // Sigmoid scoring
        let scores = mlxcel_core::sigmoid(&logits);
        let orig_scores = mlxcel_core::copy(&scores);
        let scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Group-based expert masking (zero out non-selected groups)
        let scores = if self.n_group > 1 {
            super::switch_layers::group_mask_scores(&scores, self.n_group, self.topk_group)
        } else {
            scores
        };

        // Top-k selection
        let k = self.num_experts_per_tok;
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, k);

        let mut topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        if self.num_experts_per_tok > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
            topk_scores = mlxcel_core::divide(&topk_scores, &sum);
        }

        let scale = mlxcel_core::from_slice_f32(&[self.routed_scaling_factor], &[1]);
        let topk_scores = mlxcel_core::multiply(&topk_scores, &scale);

        // Apply experts
        let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &topk_scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Add shared expert
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(&x_flat);
            result = mlxcel_core::add(&result, &shared_out);
        }

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&result, &orig_shape)
        } else {
            result
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate_weight = get_weight_copy(weights, &format!("{}.gate.weight", prefix))?;
        let e_score_correction_bias =
            get_weight_copy(weights, &format!("{}.gate.e_score_correction_bias", prefix))?;

        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{}.switch_mlp", prefix), group_size, bits)?;

        let shared_experts = if let Some(n_shared) = args.n_shared_experts {
            if n_shared > 0 {
                Some(MLP::from_weights(
                    weights,
                    &format!("{}.shared_experts", prefix),
                    group_size,
                    bits,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            gate_weight,
            e_score_correction_bias,
            switch_mlp,
            shared_experts,
            num_experts_per_tok: args.num_experts_per_tok as i32,
            n_group: args.n_group as i32,
            topk_group: args.topk_group as i32,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// FFN Enum.
pub enum FFN {
    Dense(MLP),
    MoE(MoELayer),
}

impl FFN {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FFN::Dense(mlp) => mlp.forward(x),
            FFN::MoE(moe) => moe.forward(x),
        }
    }
}

// Transformer Block.
pub struct TransformerBlock {
    pub self_attn: MlaAttention,
    pub mlp: FFN,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);
        let group_size = args.group_size();
        let bits = args.bits();

        let self_attn =
            MlaAttention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;

        let mlp = if args.is_moe_layer(layer_idx) {
            FFN::MoE(MoELayer::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            FFN::Dense(MLP::from_weights(
                weights,
                &format!("{}.mlp", prefix),
                group_size,
                bits,
            )?)
        };

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_weight, args.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_attn_norm_weight, args.rms_norm_eps),
        })
    }
}

// Model.
pub struct Glm4MoeLiteModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl Glm4MoeLiteModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[1];

        let mask = if seq_len > 1 {
            let offset = caches.first().map(|c| c.seq_len()).unwrap_or(0);
            Some(create_causal_mask(seq_len, offset))
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask.as_deref());
        }

        let h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
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
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        let lm_head = UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
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
impl LanguageModel for Glm4MoeLiteModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Glm4MoeLiteModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Glm4MoeLiteModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // GLM4 MoE Lite EOS tokens from config.json
        vec![154820, 154827, 154829]
    }
}
