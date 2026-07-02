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

//! GLM4 MoE model implementation using mlxcel-core
//!
//! GLM4 MoE architecture features:
//! - Sparse MoE with grouped expert selection
//! - Sigmoid routing with e_score_correction_bias
//! - Shared experts (optional)
//! - First K layers use dense MLP instead of MoE
//! - Partial RoPE (partial_rotary_factor)
//! - Optional Q/K normalization
//! - 4 RMSNorm layers per block (input, post_self_attn, post_attention, post_mlp)
//! - Fused gate_up_proj for experts (fused at load from separate, already
//!   expert-stacked switch_mlp.gate_proj/up_proj when the checkpoint is not
//!   pre-fused, e.g. real GLM-4.5V MoE)

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

fn default_one() -> usize {
    1
}

// Configuration.
// Used by: GLM4 MoE, Solar Open
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub moe_intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,

    // head_dim can be explicit or computed from split dims (glm4_moe_lite)
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    pub v_head_dim: Option<usize>,

    // MoE parameters
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    // Used by: GLM4 MoE, Solar Open
    #[serde(default = "default_one")]
    pub n_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub first_k_dense_replace: usize,

    #[serde(default)]
    pub n_shared_experts: Option<usize>,

    #[serde(default)]
    pub use_qk_norm: bool,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,

    #[serde(default = "default_topk_method")]
    pub topk_method: String,

    #[serde(default)]
    pub group_size: Option<i32>,

    #[serde(default)]
    pub bits: Option<i32>,

    /// Nested quantization config (used by solar_open, auto_round, etc.)
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    #[serde(default)]
    pub group_size: Option<i32>,
    #[serde(default)]
    pub bits: Option<i32>,
}

fn default_scoring_func() -> String {
    "sigmoid".to_string()
}

fn default_topk_method() -> String {
    "noaux_tc".to_string()
}

impl ModelArgs {
    /// Get the head dimension (explicit or computed from split dims)
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or_else(|| {
            // Compute from split dims if available (GLM4 MoE Lite style)
            let qk_nope = self.qk_nope_head_dim.unwrap_or(0);
            let qk_rope = self.qk_rope_head_dim.unwrap_or(0);
            if qk_nope > 0 && qk_rope > 0 {
                qk_nope + qk_rope
            } else {
                // Fallback: compute from hidden_size / num_heads
                self.hidden_size / self.num_attention_heads
            }
        })
    }

    /// Get the number of dimensions to apply RoPE to (partial RoPE)
    pub fn rope_dims(&self) -> usize {
        (self.partial_rotary_factor * self.head_dim() as f32) as usize
    }

    /// Check if a layer should use MoE (vs dense MLP)
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_k_dense_replace
    }

    pub fn group_size(&self) -> i32 {
        self.group_size
            .or_else(|| self.quantization_config.as_ref().and_then(|q| q.group_size))
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.bits
            .or_else(|| self.quantization_config.as_ref().and_then(|q| q.bits))
            .unwrap_or(4)
    }
}

// Attention with Partial RoPE and Optional Q/K Norm.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32, // Partial RoPE dimensions
    pub rope_base: f32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let mut q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let mut k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Apply Q/K norm if enabled (per-head normalization)
        if let Some(ref q_norm) = self.q_norm {
            q = q_norm.forward(&q);
        }
        if let Some(ref k_norm) = self.k_norm {
            k = k_norm.forward(&k);
        }

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply partial RoPE (only to first rope_dims dimensions)
        // GLM4 uses traditional=true
        let q = mlxcel_core::fast_rope(
            &q,
            self.rope_dims,
            true, // traditional
            self.rope_base,
            1.0, // scale
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.rope_dims,
            true, // traditional
            self.rope_base,
            1.0, // scale
            offset,
        );

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            // Single token or explicit mask
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let q_norm = if args.use_qk_norm {
            let weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
            Some(RMSNorm::new(weight, args.rms_norm_eps))
        } else {
            None
        };

        let k_norm = if args.use_qk_norm {
            let weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;
            Some(RMSNorm::new(weight, args.rms_norm_eps))
        } else {
            None
        };

        let head_dim = args.head_dim() as i32;
        let rope_dims = args.rope_dims() as i32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims,
            rope_base: args.rope_theta,
        })
    }
}

// Dense MLP for first K layers.
pub struct DenseMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl DenseMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SwiGLU: silu(gate) * up
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
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

// SwitchLinear: Stacked expert weights for MoE.
/// Stacked linear layers for MoE experts
/// Weights shape: [num_experts, output_dim, input_dim_packed]
/// Supports both quantized (gather_qmm) and non-quantized (gather_mm) forward paths.
pub enum SwitchLinear {
    Quantized {
        weight: UniquePtr<MlxArray>,
        scales: UniquePtr<MlxArray>,
        biases: UniquePtr<MlxArray>,
        group_size: i32,
        bits: i32,
        num_experts: usize,
    },
    Regular {
        weight: UniquePtr<MlxArray>,
        num_experts: usize,
    },
}

impl SwitchLinear {
    /// Return the number of experts this layer holds.
    pub fn num_experts(&self) -> usize {
        match self {
            Self::Quantized { num_experts, .. } => *num_experts,
            Self::Regular { num_experts, .. } => *num_experts,
        }
    }

    /// Forward pass: gather_qmm for quantized weights, gather_mm for regular weights.
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
                ..
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
                        std::ptr::null(), // lhs_indices
                        indices as *const _,
                        true, // transpose
                        *group_size,
                        *bits,
                        sorted_indices,
                        "affine",
                    )
                }
            }
            Self::Regular { weight, .. } => {
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
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        let scales_key = format!("{}.scales", prefix);
        if weights.contains_key(&scales_key) {
            let scales = mlxcel_core::copy(weights.get(&scales_key).unwrap());
            let biases = get_weight_copy(weights, &format!("{}.biases", prefix))?;
            let shape = mlxcel_core::array_shape(&weight);
            let num_experts = shape[0] as usize;
            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                group_size: args.group_size(),
                bits: args.bits(),
                num_experts,
            })
        } else {
            let shape = mlxcel_core::array_shape(&weight);
            let num_experts = shape[0] as usize;
            Ok(Self::Regular {
                weight,
                num_experts,
            })
        }
    }

    /// Fuse separately-stored, already-expert-stacked `gate_proj` and `up_proj`
    /// tensors into the fused `gate_up_proj` layout the model consumes.
    ///
    /// Real GLM-4(.5)V MoE checkpoints ship the routed experts as two tensors,
    /// `switch_mlp.gate_proj` and `switch_mlp.up_proj`, each shaped
    /// `[num_experts, out_features, in_features]` (the last axis is the
    /// group-packed input for quantized checkpoints). The forward path expects a
    /// single fused `gate_up_proj` whose output-feature axis holds gate rows
    /// first, then up rows (see `SwitchGLU::forward`, which splits the result at
    /// `intermediate_size`). Fusion is therefore a concatenation along the
    /// output-feature axis (axis 1).
    ///
    /// This is quant-aware: 4/8-bit weights pack along the *input* axis (the last
    /// axis), so concatenating along the output axis never splits a packed word,
    /// and the matching per-output-row `scales`/`biases` concatenate on the same
    /// axis. No dequantization is required.
    fn from_separate_gate_up(
        weights: &WeightMap,
        args: &ModelArgs,
        gate_prefix: &str,
        up_prefix: &str,
    ) -> Result<Self, String> {
        // Output-feature axis of the `[num_experts, out_features, in]` stack.
        const EXPERT_OUT_AXIS: i32 = 1;

        let gate_weight = get_weight_ref(weights, &format!("{}.weight", gate_prefix))?;
        let up_weight = get_weight_ref(weights, &format!("{}.weight", up_prefix))?;
        let weight = mlxcel_core::concatenate(gate_weight, up_weight, EXPERT_OUT_AXIS);
        let num_experts = mlxcel_core::array_shape(&weight)[0] as usize;

        let gate_scales_key = format!("{}.scales", gate_prefix);
        if weights.contains_key(&gate_scales_key) {
            let gate_scales = get_weight_ref(weights, &gate_scales_key)?;
            let up_scales = get_weight_ref(weights, &format!("{}.scales", up_prefix))?;
            let scales = mlxcel_core::concatenate(gate_scales, up_scales, EXPERT_OUT_AXIS);

            let gate_biases = get_weight_ref(weights, &format!("{}.biases", gate_prefix))?;
            let up_biases = get_weight_ref(weights, &format!("{}.biases", up_prefix))?;
            let biases = mlxcel_core::concatenate(gate_biases, up_biases, EXPERT_OUT_AXIS);

            Ok(Self::Quantized {
                weight,
                scales,
                biases,
                group_size: args.group_size(),
                bits: args.bits(),
                num_experts,
            })
        } else {
            Ok(Self::Regular {
                weight,
                num_experts,
            })
        }
    }
}

// SwitchGLU: SwiGLU with fused gate_up projection.
/// SwitchGLU: SwiGLU activation with stacked expert weights for MoE
/// Uses fused gate_up projection that is split at intermediate_size
pub struct SwitchGLU {
    pub gate_up_proj: SwitchLinear, // Fused: outputs 2 * intermediate_size
    pub down_proj: SwitchLinear,
    pub intermediate_size: i32,
}

impl SwitchGLU {
    /// Forward pass with kernel-fused SwiGLU activation
    /// x: [n_tokens, hidden]
    /// indices: [n_tokens, top_k]
    /// Returns: [n_tokens, top_k, hidden]
    pub fn forward(&self, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];

        // Check if we should use sorted_indices optimization (>= 64 tokens)
        let total_elements = n_tokens * top_k;
        let do_sort = total_elements >= 64;

        // Expand x for broadcasting: [n_tokens, hidden] -> [n_tokens, 1, 1, hidden]
        let x_expanded = mlxcel_core::expand_dims(x, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);

        if do_sort {
            // Sort tokens by expert for better memory access
            let (sorted_x, sorted_idx, inv_order) = self.gather_sort(&x_expanded, indices);

            // Apply fused gate_up projection
            let fused = self.gate_up_proj.forward(&sorted_x, &sorted_idx, true);

            // Split into gate and up
            let gate = mlxcel_core::slice_last_dim(&fused, 0, self.intermediate_size);
            let up = mlxcel_core::slice_last_dim(
                &fused,
                self.intermediate_size,
                2 * self.intermediate_size,
            );

            // Kernel-fused SwiGLU: silu(gate) * up
            let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

            // Down projection
            let output = self.down_proj.forward(&activated, &sorted_idx, true);

            // Restore original order
            self.scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            // Direct path without sorting
            let fused = self.gate_up_proj.forward(&x_expanded, indices, false);

            // Split into gate and up
            let gate = mlxcel_core::slice_last_dim(&fused, 0, self.intermediate_size);
            let up = mlxcel_core::slice_last_dim(
                &fused,
                self.intermediate_size,
                2 * self.intermediate_size,
            );

            // Kernel-fused SwiGLU: silu(gate) * up
            let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

            // Down projection
            let output = self.down_proj.forward(&activated, indices, false);

            // Squeeze: [n_tokens, top_k, 1, hidden] -> [n_tokens, top_k, hidden]
            mlxcel_core::squeeze_axis(&output, -2)
        }
    }

    /// Sort tokens by expert index for better memory access
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

        // Flatten indices: [n_tokens, top_k] -> [n_tokens * top_k]
        let flat_indices = mlxcel_core::reshape(indices, &[-1]);

        // Sort indices by expert
        let order = mlxcel_core::argsort(&flat_indices, -1);
        let inv_order = mlxcel_core::argsort(&order, -1);

        // x is [n_tokens, 1, 1, hidden]
        // Flatten: [n_tokens, 1, hidden]
        let x_shape = mlxcel_core::array_shape(x);
        let x_flat = mlxcel_core::reshape(x, &[x_shape[0], 1, x_shape[3]]);

        // Divide order by top_k to get token indices
        let top_k_arr = mlxcel_core::from_slice_i32(&[top_k], &[1]);
        let token_indices = mlxcel_core::divide(&order, &top_k_arr);
        let token_indices = mlxcel_core::astype(&token_indices, mlxcel_core::dtype::INT32);

        // Take x rows in sorted order
        let sorted_x = mlxcel_core::take(&x_flat, &token_indices, 0);

        // Get sorted expert indices
        let sorted_indices = mlxcel_core::take(&flat_indices, &order, 0);

        (sorted_x, sorted_indices, inv_order)
    }

    /// Restore original order after sorted expert computation
    fn scatter_unsort(
        &self,
        x: &MlxArray,
        inv_order: &MlxArray,
        orig_shape: &[i32],
    ) -> UniquePtr<MlxArray> {
        // x has shape [n_sorted, 1, hidden]
        // Reorder by inv_order
        let unsorted = mlxcel_core::take(x, inv_order, 0);

        // Unflatten and squeeze
        let x_shape = mlxcel_core::array_shape(&unsorted);
        let n_tokens = orig_shape[0];
        let top_k = orig_shape[1];

        let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
        mlxcel_core::squeeze_axis(&reshaped, 2)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        // Prefer a checkpoint that already ships the fused `gate_up_proj` expert
        // stack; otherwise fuse the separate `gate_proj` / `up_proj` tensors that
        // real GLM-4(.5)V MoE checkpoints provide (concatenated gate-then-up
        // along the expert output-feature axis, quant-aware).
        let gate_up_prefix = format!("{}.gate_up_proj", prefix);
        let gate_up_proj = if weights.contains_key(&format!("{}.weight", gate_up_prefix)) {
            SwitchLinear::from_weights(weights, args, &gate_up_prefix)?
        } else {
            SwitchLinear::from_separate_gate_up(
                weights,
                args,
                &format!("{}.gate_proj", prefix),
                &format!("{}.up_proj", prefix),
            )?
        };

        Ok(Self {
            gate_up_proj,
            down_proj: SwitchLinear::from_weights(weights, args, &format!("{}.down_proj", prefix))?,
            intermediate_size: args.moe_intermediate_size as i32,
        })
    }
}

// Shared Expert MLP.
/// Standard MLP with SwiGLU activation for shared expert
pub struct SharedExpertMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl SharedExpertMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        // SwiGLU: silu(gate) * up
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
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

// MoE Block with Sigmoid Routing.
/// GLM4 MoE layer with sigmoid routing, grouped selection, and optional shared experts
pub struct Glm4Moe {
    pub router_weight: UniquePtr<MlxArray>,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub experts: SwitchGLU,
    pub shared_expert: Option<SharedExpertMLP>,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}

impl Glm4Moe {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten to [n_tokens, hidden]
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Compute gate logits: x @ router_weight.T
        let router_transposed = mlxcel_core::transpose_axes(&self.router_weight, &[1, 0]);
        let logits = mlxcel_core::matmul(&x_flat, &router_transposed);

        // Sigmoid scoring
        let scores = mlxcel_core::sigmoid(&logits);
        let orig_scores = mlxcel_core::copy(&scores);

        // Add e_score_correction_bias
        let scores = mlxcel_core::add(&scores, &self.e_score_correction_bias);

        // Group-based expert masking (zero out non-selected groups)
        let scores = if self.n_group > 1 {
            super::switch_layers::group_mask_scores(
                &scores,
                self.n_group as i32,
                self.topk_group as i32,
            )
        } else {
            scores
        };

        // Top-k selection using argpartition
        let k = self.num_experts_per_tok as i32;
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = slice_axis(&indices, -1, 0, k);

        // Get scores from original (before bias correction)
        let mut topk_scores = mlxcel_core::take_along_axis(&orig_scores, &topk_indices, -1);

        // Normalize if needed
        if self.num_experts_per_tok > 1 && self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&topk_scores, -1, true);
            topk_scores = mlxcel_core::divide(&topk_scores, &sum);
        }

        // Apply scaling factor
        let scale = mlxcel_core::from_slice_f32(&[self.routed_scaling_factor], &[1]);
        let topk_scores = mlxcel_core::multiply(&topk_scores, &scale);

        // Apply experts - returns [n_tokens, k, hidden]
        let expert_out = self.experts.forward(&x_flat, &topk_indices);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &topk_scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Add shared expert if present
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(&x_flat);
            result = mlxcel_core::add(&result, &shared_out);
        }

        // Reshape back to original shape
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
        let router_weight = get_weight_copy(weights, &format!("{}.gate.weight", prefix))?;
        let e_score_correction_bias =
            get_weight_copy(weights, &format!("{}.gate.e_score_correction_bias", prefix))?;

        let experts = SwitchGLU::from_weights(weights, args, &format!("{}.switch_mlp", prefix))?;

        let shared_expert = if args.n_shared_experts.is_some() {
            Some(SharedExpertMLP::from_weights(
                weights,
                args,
                &format!("{}.shared_experts", prefix),
            )?)
        } else {
            None
        };

        Ok(Self {
            router_weight,
            e_score_correction_bias,
            experts,
            shared_expert,
            num_experts_per_tok: args.num_experts_per_tok,
            n_group: args.n_group,
            topk_group: args.topk_group,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// FFN Enum: Dense or MoE.
pub enum FFN {
    Dense(DenseMLP),
    Moe(Glm4Moe),
}

impl FFN {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FFN::Dense(mlp) => mlp.forward(x),
            FFN::Moe(moe) => moe.forward(x),
        }
    }
}

// Transformer Block with 4 RMSNorm layers.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: FFN,
    pub input_layernorm: RMSNorm,
    pub post_self_attn_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub post_mlp_layernorm: RMSNorm,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let attn_out = self.post_self_attn_layernorm.forward(&attn_out);
        let h = mlxcel_core::add(x, &attn_out);

        // Post-attention norm
        let h = self.post_attention_layernorm.forward(&h);

        // MLP
        let mlp_out = self.mlp.forward(&h);
        let mlp_out = self.post_mlp_layernorm.forward(&mlp_out);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;

        let mlp = if args.is_moe_layer(layer_idx) {
            FFN::Moe(Glm4Moe::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            FFN::Dense(DenseMLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_self_attn_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_self_attn_layernorm.weight", prefix),
        )?;
        let post_attention_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;
        let post_mlp_norm_weight =
            get_weight_copy(weights, &format!("{}.post_mlp_layernorm.weight", prefix))?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_self_attn_layernorm = RMSNorm::new(post_self_attn_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_attention_norm_weight, args.rms_norm_eps);
        let post_mlp_layernorm = RMSNorm::new(post_mlp_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_self_attn_layernorm,
            post_attention_layernorm,
            post_mlp_layernorm,
        })
    }
}

// GLM4 MoE Model.
pub struct Glm4MoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl Glm4MoeModel {
    /// Forward pass through the entire model
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.embed_tokens.forward(input_ids);

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load model from directory
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
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

/// Borrow a weight without copying (used when fusing tensors at load).
fn get_weight_ref<'a>(
    weights: &'a WeightMap,
    name: &str,
) -> Result<&'a UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for Glm4MoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Glm4MoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Glm4MoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // GLM4 MoE EOS tokens
        vec![2, 151329, 151336, 151338] // </s>, <|endoftext|>, <|user|>, <|observation|>
    }
}

// Tests.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_args() {
        let json = r#"{
            "model_type": "glm4_moe",
            "vocab_size": 151552,
            "hidden_size": 4096,
            "intermediate_size": 13696,
            "max_position_embeddings": 8192,
            "moe_intermediate_size": 1408,
            "num_attention_heads": 32,
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "n_routed_experts": 16,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": false,
            "first_k_dense_replace": 1,
            "n_shared_experts": 2,
            "use_qk_norm": false,
            "attention_bias": true
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();

        assert_eq!(args.vocab_size, 151552);
        assert_eq!(args.hidden_size, 4096);
        assert_eq!(args.num_hidden_layers, 40);
        assert_eq!(args.n_routed_experts, 16);
        assert_eq!(args.num_experts_per_tok, 2);
        assert_eq!(args.first_k_dense_replace, 1);

        // Test partial RoPE calculation
        let rope_dims = args.rope_dims();
        assert_eq!(rope_dims, 64); // 128 * 0.5 = 64

        // Test is_moe_layer
        assert!(!args.is_moe_layer(0)); // Layer 0 should be dense
        assert!(args.is_moe_layer(1)); // Layer 1+ should be MoE
    }

    #[test]
    fn test_solar_open_config_with_quantization_config() {
        let json = r#"{
            "model_type": "solar_open",
            "vocab_size": 196608,
            "hidden_size": 4096,
            "intermediate_size": 10240,
            "max_position_embeddings": 131072,
            "moe_intermediate_size": 1280,
            "num_attention_heads": 64,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000,
            "partial_rotary_factor": 1.0,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": true,
            "first_k_dense_replace": 0,
            "n_shared_experts": 1,
            "quantization_config": {
                "bits": 4,
                "group_size": 128
            }
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();

        assert_eq!(args.vocab_size, 196608);
        assert_eq!(args.num_hidden_layers, 48);
        assert_eq!(args.n_routed_experts, 128);
        assert_eq!(args.num_experts_per_tok, 8);
        assert_eq!(args.first_k_dense_replace, 0);

        // quantization_config fallback
        assert_eq!(args.group_size(), 128);
        assert_eq!(args.bits(), 4);

        // All layers should be MoE (first_k_dense_replace = 0)
        assert!(args.is_moe_layer(0));

        // Full RoPE (partial_rotary_factor = 1.0)
        assert_eq!(args.rope_dims(), 128);
    }

    /// Minimal GLM-4V MoE text args: only the fields the fusion path reads
    /// (`moe_intermediate_size`, quant `group_size`/`bits` defaults) matter.
    fn tiny_moe_args(moe_intermediate_size: usize) -> ModelArgs {
        let json = format!(
            r#"{{
                "model_type": "glm4v_moe_text",
                "vocab_size": 151552,
                "hidden_size": 16,
                "intermediate_size": 32,
                "max_position_embeddings": 8192,
                "moe_intermediate_size": {moe_intermediate_size},
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "routed_scaling_factor": 1.0,
                "norm_topk_prob": true,
                "first_k_dense_replace": 1
            }}"#
        );
        serde_json::from_str(&json).unwrap()
    }

    /// Real GLM-4(.5)V MoE checkpoints ship the routed experts as separate,
    /// already expert-stacked `switch_mlp.gate_proj` / `switch_mlp.up_proj`
    /// (4-bit: `.weight` U32 + `.scales`/`.biases`), not a fused
    /// `switch_mlp.gate_up_proj`. `SwitchGLU::from_weights` must fuse them into
    /// the fused layout by concatenating gate-then-up along the expert
    /// output-feature axis (axis 1), carrying the packed weight and its scales
    /// and biases together, while `down_proj` is loaded unchanged.
    #[test]
    fn fuses_separate_switch_mlp_gate_up_experts() {
        // Small dims that mirror the real 3D layout
        // `[num_experts, out_features, in_features(packed)]`.
        let e = 4i32; // n_routed_experts
        let i = 8i32; // moe_intermediate_size (gate/up out_features)
        let kp = 2i32; // packed input columns (hidden / 8 for 4-bit)
        let g = 1i32; // quant groups along the input
        let h = 16i32; // hidden (down_proj out_features)
        let down_kp = 1i32; // down_proj packed input columns

        let args = tiny_moe_args(i as usize);

        let mut weights = WeightMap::new();
        let prefix = "model.layers.1.mlp.switch_mlp";
        // Separate, already expert-stacked gate_proj / up_proj (4-bit quantized).
        for proj in ["gate_proj", "up_proj"] {
            weights.insert(
                format!("{}.{}.weight", prefix, proj),
                mlxcel_core::ones(&[e, i, kp], mlxcel_core::dtype::UINT32),
            );
            weights.insert(
                format!("{}.{}.scales", prefix, proj),
                mlxcel_core::ones(&[e, i, g], mlxcel_core::dtype::BFLOAT16),
            );
            weights.insert(
                format!("{}.{}.biases", prefix, proj),
                mlxcel_core::ones(&[e, i, g], mlxcel_core::dtype::BFLOAT16),
            );
        }
        // down_proj is consumed as-is (no fusion).
        weights.insert(
            format!("{}.down_proj.weight", prefix),
            mlxcel_core::ones(&[e, h, down_kp], mlxcel_core::dtype::UINT32),
        );
        weights.insert(
            format!("{}.down_proj.scales", prefix),
            mlxcel_core::ones(&[e, h, g], mlxcel_core::dtype::BFLOAT16),
        );
        weights.insert(
            format!("{}.down_proj.biases", prefix),
            mlxcel_core::ones(&[e, h, g], mlxcel_core::dtype::BFLOAT16),
        );

        let glu = SwitchGLU::from_weights(&weights, &args, prefix)
            .expect("fusing separate gate_proj/up_proj should succeed");

        assert_eq!(glu.intermediate_size, i);

        match &glu.gate_up_proj {
            SwitchLinear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                num_experts,
            } => {
                assert_eq!(*num_experts, e as usize);
                assert_eq!(*group_size, args.group_size());
                assert_eq!(*bits, args.bits());
                // Output-feature axis is doubled (gate rows then up rows); the
                // packed input axis and the quant groups are unchanged.
                assert_eq!(mlxcel_core::array_shape(weight), vec![e, 2 * i, kp]);
                assert_eq!(mlxcel_core::array_shape(scales), vec![e, 2 * i, g]);
                assert_eq!(mlxcel_core::array_shape(biases), vec![e, 2 * i, g]);
            }
            SwitchLinear::Regular { .. } => panic!("expected a quantized fused gate_up_proj"),
        }

        match &glu.down_proj {
            SwitchLinear::Quantized {
                weight,
                num_experts,
                ..
            } => {
                assert_eq!(*num_experts, e as usize);
                assert_eq!(mlxcel_core::array_shape(weight), vec![e, h, down_kp]);
            }
            SwitchLinear::Regular { .. } => panic!("expected a quantized down_proj"),
        }
    }

    /// Backward compatibility: a checkpoint that already provides the fused
    /// `switch_mlp.gate_up_proj` stack must still load through the fused path.
    #[test]
    fn loads_prefused_switch_mlp_gate_up_experts() {
        let e = 4i32;
        let i = 8i32;
        let kp = 2i32;
        let g = 1i32;
        let h = 16i32;
        let down_kp = 1i32;

        let args = tiny_moe_args(i as usize);

        let mut weights = WeightMap::new();
        let prefix = "model.layers.1.mlp.switch_mlp";
        // Pre-fused gate_up_proj: out_features already 2 * moe_intermediate_size.
        weights.insert(
            format!("{}.gate_up_proj.weight", prefix),
            mlxcel_core::ones(&[e, 2 * i, kp], mlxcel_core::dtype::UINT32),
        );
        weights.insert(
            format!("{}.gate_up_proj.scales", prefix),
            mlxcel_core::ones(&[e, 2 * i, g], mlxcel_core::dtype::BFLOAT16),
        );
        weights.insert(
            format!("{}.gate_up_proj.biases", prefix),
            mlxcel_core::ones(&[e, 2 * i, g], mlxcel_core::dtype::BFLOAT16),
        );
        weights.insert(
            format!("{}.down_proj.weight", prefix),
            mlxcel_core::ones(&[e, h, down_kp], mlxcel_core::dtype::UINT32),
        );
        weights.insert(
            format!("{}.down_proj.scales", prefix),
            mlxcel_core::ones(&[e, h, g], mlxcel_core::dtype::BFLOAT16),
        );
        weights.insert(
            format!("{}.down_proj.biases", prefix),
            mlxcel_core::ones(&[e, h, g], mlxcel_core::dtype::BFLOAT16),
        );

        let glu = SwitchGLU::from_weights(&weights, &args, prefix)
            .expect("loading a pre-fused gate_up_proj should succeed");

        match &glu.gate_up_proj {
            SwitchLinear::Quantized {
                weight,
                num_experts,
                ..
            } => {
                assert_eq!(*num_experts, e as usize);
                assert_eq!(mlxcel_core::array_shape(weight), vec![e, 2 * i, kp]);
            }
            SwitchLinear::Regular { .. } => panic!("expected a quantized fused gate_up_proj"),
        }
    }

    #[test]
    fn test_top_level_quantization_takes_precedence() {
        let json = r#"{
            "model_type": "glm4_moe",
            "vocab_size": 151552,
            "hidden_size": 4096,
            "intermediate_size": 13696,
            "max_position_embeddings": 8192,
            "moe_intermediate_size": 1408,
            "num_attention_heads": 32,
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "n_routed_experts": 16,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": false,
            "first_k_dense_replace": 1,
            "group_size": 64,
            "bits": 4,
            "quantization_config": {
                "bits": 8,
                "group_size": 256
            }
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();

        // Top-level values should take precedence over quantization_config
        assert_eq!(args.group_size(), 64);
        assert_eq!(args.bits(), 4);
    }
}
