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

//! Solar Open model implementation
//!
//! Solar Open (Upstage) shares the GLM4-MoE architecture:
//! - Sparse MoE with sigmoid routing and e_score_correction_bias
//! - Shared experts (optional)
//! - All layers are MoE (first_k_dense_replace = 0)
//! - Full RoPE (partial_rotary_factor = 1.0), traditional=false
//! - 2 RMSNorm layers per block (input_layernorm, post_attention_layernorm)
//! - Separate gate_proj/up_proj for experts (not fused gate_up_proj)
//!
//! Architecture reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4_moe.py
//! Weight format: supports both MLX native and GPTQ (auto_round) quantization

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::stack_arrays;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

use super::switch_layers::{SwitchGLU, group_mask_scores};
// Re-use the config struct from glm4_moe (compatible fields)
pub use super::glm4_moe::ModelArgs;

// ============================================================================
// Weight Sanitization (GPTQ conversion + expert stacking)
// ============================================================================

/// Convert GPTQ/auto_round weights to MLX native format and stack per-expert weights.
///
/// Handles:
/// 1. GPTQ qweight/qzeros/scales → MLX weight/scales/biases conversion
/// 2. Per-expert individual weights → stacked SwitchLinear format
pub fn sanitize_weights(mut weights: WeightMap, config: &ModelArgs) -> WeightMap {
    // Step 1: Convert GPTQ format to MLX format (qweight/qzeros/scales → weight/biases/scales)
    convert_gptq_to_mlx(&mut weights, config);

    // Step 2: Stack per-expert weights into SwitchLinear format
    stack_expert_weights(&mut weights, config);

    // Step 3: Remove multi-token prediction layers if present
    let mtp_prefix = format!("model.layers.{}.", config.num_hidden_layers);
    weights.retain(|k, _| !k.starts_with(&mtp_prefix));

    // Step 4: Materialize all converted weights to collapse the lazy graph
    println!("[SolarOpen] Materializing converted weights...");
    let ptrs: Vec<*const MlxArray> = weights
        .values()
        .map(|v| v.as_ref().unwrap() as *const MlxArray)
        .collect();
    if !ptrs.is_empty() {
        unsafe { mlxcel_core::eval_all(&ptrs) };
    }

    weights
}

/// Convert GPTQ/auto_round quantized weights to MLX native format.
///
/// GPTQ format: qweight (packed int32), qzeros (packed int32), scales (float16)
/// MLX format:  weight (packed uint32), biases (float16), scales (float16)
///
/// For symmetric quantization (sym=true):
///   - biases = -zero_point * scales (typically zero_point = 2^(bits-1) = 8 for 4-bit)
///   - weight values need bit reordering from GPTQ to MLX packing
fn convert_gptq_to_mlx(weights: &mut WeightMap, config: &ModelArgs) {
    let bits = config.bits();
    let group_size = config.group_size();

    // Collect all keys that need conversion (have qweight but no weight)
    let qweight_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".qweight"))
        .cloned()
        .collect();

    if qweight_keys.is_empty() {
        return;
    }

    println!(
        "[SolarOpen] Converting {} GPTQ layers to MLX format (bits={}, group_size={})",
        qweight_keys.len(),
        bits,
        group_size
    );

    for qweight_key in qweight_keys {
        let prefix = &qweight_key[..qweight_key.len() - 8]; // Remove ".qweight"
        let scales_key = format!("{}.scales", prefix);
        let qzeros_key = format!("{}.qzeros", prefix);
        let weight_key = format!("{}.weight", prefix);
        let biases_key = format!("{}.biases", prefix);

        // Skip if already converted
        if weights.contains_key(&weight_key) {
            continue;
        }

        let Some(qweight) = weights.remove(&qweight_key) else {
            continue;
        };

        // Get scales (keep a copy for biases computation)
        let Some(scales) = weights.remove(&scales_key) else {
            continue;
        };

        // Get qzeros if present
        let qzeros = weights.remove(&qzeros_key);

        // Convert GPTQ packed weights to MLX format
        let (mlx_weight, mlx_scales, mlx_biases) =
            gptq_to_mlx_tensors(&qweight, &scales, qzeros.as_ref(), bits, group_size);

        weights.insert(weight_key, mlx_weight);
        weights.insert(scales_key, mlx_scales);
        weights.insert(biases_key, mlx_biases);
    }
}

/// Convert a single GPTQ weight tensor to MLX format.
///
/// auto_gptq packing (used by auto_round):
///   qweight: [in_features / pack_factor, out_features]  (packed along rows)
///   scales:  [n_groups, out_features]
///   qzeros:  [n_groups, out_features / pack_factor]      (packed along cols)
///
/// MLX native format:
///   weight:  [out_features, in_features / pack_factor]   (packed along cols)
///   scales:  [out_features, n_groups]
///   biases:  [out_features, n_groups]
fn gptq_to_mlx_tensors(
    qweight: &MlxArray,
    scales: &MlxArray,
    qzeros: Option<&UniquePtr<MlxArray>>,
    bits: i32,
    _group_size: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let pack_factor = 32 / bits;
    let qw_shape = mlxcel_core::array_shape(qweight);
    // auto_gptq: qweight is [in_features / pack_factor, out_features]
    let in_features = qw_shape[0] * pack_factor;
    let out_features = qw_shape[1];

    // Step 1: Unpack qweight along first dimension
    // [in/8, out] → [in/8, out, 8] → [in/8, 8, out] → [in, out]
    let unpacked = unpack_gptq_rows(qweight, pack_factor);

    // Step 2: Transpose to MLX layout: [out, in]
    let transposed = mlxcel_core::transpose_axes(&unpacked, &[1, 0]);

    // Step 3: Repack in MLX format (packed along cols): [out, in/8]
    let mlx_weight = pack_mlx_4bit(&transposed, in_features, out_features);

    // Step 4: Transpose scales from [n_groups, out] to [out, n_groups]
    // Explicitly cast to FLOAT16 (scales may be loaded as native fp16, but
    // astype ensures the correct dtype for quantized_matmul/gather_qmm paths)
    let mlx_scales = mlxcel_core::transpose_axes(scales, &[1, 0]);
    let mlx_scales = mlxcel_core::astype(&mlx_scales, mlxcel_core::dtype::FLOAT16);
    let mlx_scales = mlxcel_core::contiguous(&mlx_scales, false);

    // Step 5: Compute biases from qzeros and scales (in FLOAT16)
    let scales_dtype = mlxcel_core::dtype::FLOAT16;
    let mlx_biases = if let Some(qz) = qzeros {
        compute_mlx_biases_from_qzeros(qz, &mlx_scales, pack_factor, scales_dtype)
    } else {
        // Symmetric quantization: zero_point = 2^(bits-1)
        let zero_point = (1 << (bits - 1)) as f32;
        let neg_zp = mlxcel_core::from_slice_f32(&[-zero_point], &[1]);
        mlxcel_core::multiply(&mlx_scales, &neg_zp)
    };

    (mlx_weight, mlx_scales, mlx_biases)
}

/// Unpack auto_gptq 4-bit weights packed along the FIRST dimension.
///
/// qweight shape: [in/pack_factor, out]
/// Each int32 at [i, j] stores pack_factor values for inputs [i*8..i*8+7] at output j.
/// Returns: [in, out]
fn unpack_gptq_rows(packed: &MlxArray, pack_factor: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(packed);
    let packed_rows = shape[0];
    let cols = shape[1];

    // Shifts for 4-bit extraction: [0, 4, 8, 12, 16, 20, 24, 28]
    let shifts: Vec<i32> = (0..pack_factor).map(|i| i * 4).collect();
    let shifts_arr = mlxcel_core::from_slice_i32(&shifts, &[1, 1, pack_factor]);
    let mask = mlxcel_core::from_slice_i32(&[0xF], &[1]);

    // Expand for broadcasting: [packed_rows, cols, 1]
    let expanded = mlxcel_core::expand_dims(packed, -1);
    let expanded = mlxcel_core::astype(&expanded, mlxcel_core::dtype::INT32);

    // Shift and mask: [packed_rows, cols, pack_factor]
    let shifted = mlxcel_core::right_shift(&expanded, &shifts_arr);
    let values = mlxcel_core::bitwise_and(&shifted, &mask);

    // Transpose to [packed_rows, pack_factor, cols] then reshape to [in, cols]
    let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1]);
    mlxcel_core::reshape(&values, &[packed_rows * pack_factor, cols])
}

/// Unpack GPTQ 4-bit values packed along the LAST dimension.
///
/// Used for qzeros: [n_groups, out/pack_factor]
/// Returns: [n_groups, out]
fn unpack_gptq_cols(packed: &MlxArray, pack_factor: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(packed);
    let rows = shape[0];
    let packed_cols = shape[1];

    let shifts: Vec<i32> = (0..pack_factor).map(|i| i * 4).collect();
    let shifts_arr = mlxcel_core::from_slice_i32(&shifts, &[1, 1, pack_factor]);
    let mask = mlxcel_core::from_slice_i32(&[0xF], &[1]);

    let expanded = mlxcel_core::expand_dims(packed, -1);
    let expanded = mlxcel_core::astype(&expanded, mlxcel_core::dtype::INT32);

    let shifted = mlxcel_core::right_shift(&expanded, &shifts_arr);
    let values = mlxcel_core::bitwise_and(&shifted, &mask);

    // Reshape to [rows, packed_cols * pack_factor]
    mlxcel_core::reshape(&values, &[rows, packed_cols * pack_factor])
}

/// Pack values into MLX 4-bit uint32 format (packed along last dimension).
///
/// Input: [out_features, in_features]
/// Output: [out_features, in_features / pack_factor]
fn pack_mlx_4bit(values: &MlxArray, in_features: i32, out_features: i32) -> UniquePtr<MlxArray> {
    let pack_factor = 8; // 32 / 4 bits
    let packed_in = in_features / pack_factor;

    // Reshape to [out_features, packed_in, pack_factor]
    let reshaped = mlxcel_core::reshape(values, &[out_features, packed_in, pack_factor]);
    let reshaped = mlxcel_core::astype(&reshaped, mlxcel_core::dtype::UINT32);

    // MLX packing: value_i at bits [4*i : 4*(i+1)]
    let shifts: Vec<i32> = (0..pack_factor).map(|i| i * 4).collect();
    let shifts_arr = mlxcel_core::from_slice_i32(&shifts, &[1, 1, pack_factor]);
    let shifts_arr = mlxcel_core::astype(&shifts_arr, mlxcel_core::dtype::UINT32);

    let shifted = mlxcel_core::left_shift(&reshaped, &shifts_arr);

    // Sum along pack dimension to combine bits
    let packed = mlxcel_core::sum_axis(&shifted, -1, false);
    mlxcel_core::astype(&packed, mlxcel_core::dtype::UINT32)
}

/// Compute MLX biases from GPTQ qzeros.
///
/// qzeros: [n_groups, out/pack_factor] (GPTQ format)
/// scales: [out, n_groups] (already in MLX format)
/// Returns biases: [out, n_groups]
fn compute_mlx_biases_from_qzeros(
    qzeros: &MlxArray,
    scales: &MlxArray,
    pack_factor: i32,
    target_dtype: i32,
) -> UniquePtr<MlxArray> {
    // Unpack qzeros along last dim: [n_groups, out/8] → [n_groups, out]
    let unpacked_zeros = unpack_gptq_cols(qzeros, pack_factor);

    // Transpose to [out, n_groups] to match scales layout
    let zeros_t = mlxcel_core::transpose_axes(&unpacked_zeros, &[1, 0]);
    let zeros_f = mlxcel_core::astype(&zeros_t, target_dtype);

    // MLX biases = -zero_point * scales
    let neg_zeros = mlxcel_core::negative(&zeros_f);

    // Ensure shapes match (truncate if needed)
    let scales_shape = mlxcel_core::array_shape(scales);
    let zeros_shape = mlxcel_core::array_shape(&neg_zeros);

    let neg_zeros = if zeros_shape[0] != scales_shape[0] {
        mlxcel_core::utils::slice_axis(&neg_zeros, 0, 0, scales_shape[0])
    } else {
        neg_zeros
    };

    mlxcel_core::multiply(&neg_zeros, scales)
}

/// Stack per-expert weights into SwitchLinear stacked format.
///
/// Converts: model.layers.L.mlp.experts.E.{gate,up,down}_proj.{weight,scales,biases}
/// Into:     model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
fn stack_expert_weights(weights: &mut WeightMap, config: &ModelArgs) {
    let n_experts = config.n_routed_experts;

    for l in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}", l);

        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for suffix in ["weight", "scales", "biases"] {
                let first_key = format!("{}.mlp.experts.0.{}.{}", prefix, proj, suffix);
                if !weights.contains_key(&first_key) {
                    continue;
                }

                let mut expert_tensors: Vec<UniquePtr<MlxArray>> = Vec::new();
                for e in 0..n_experts {
                    let key = format!("{}.mlp.experts.{}.{}.{}", prefix, e, proj, suffix);
                    if let Some(w) = weights.remove(&key) {
                        expert_tensors.push(w);
                    }
                }

                if expert_tensors.len() == n_experts {
                    let stacked = stack_arrays(&expert_tensors, 0);
                    let stacked_key = format!("{}.mlp.switch_mlp.{}.{}", prefix, proj, suffix);
                    weights.insert(stacked_key, stacked);
                }
            }
        }
    }
}

// ============================================================================
// Attention with RoPE (traditional=false)
// ============================================================================

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
    pub rope_dims: i32,
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

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let mut q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let mut k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        if let Some(ref q_norm) = self.q_norm {
            q = q_norm.forward(&q);
        }
        if let Some(ref k_norm) = self.k_norm {
            k = k_norm.forward(&k);
        }

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Solar Open / Python glm4_moe uses traditional=false
        let q = mlxcel_core::fast_rope(
            &q,
            self.rope_dims,
            false, // traditional=false (matches Python)
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.rope_dims,
            false, // traditional=false
            self.rope_base,
            1.0,
            offset,
        );

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

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

// ============================================================================
// Dense MLP (for first_k_dense_replace layers, if any)
// ============================================================================

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
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

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

// ============================================================================
// Shared Expert MLP
// ============================================================================

pub struct SharedExpertMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl SharedExpertMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
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

// ============================================================================
// MoE Block with Sigmoid Routing
// ============================================================================

pub struct MoE {
    pub router_weight: UniquePtr<MlxArray>,
    pub e_score_correction_bias: UniquePtr<MlxArray>,
    pub switch_mlp: SwitchGLU,
    pub shared_expert: Option<SharedExpertMLP>,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
}

impl MoE {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

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

        // Group-based expert masking
        let scores = if self.n_group > 1 {
            group_mask_scores(&scores, self.n_group as i32, self.topk_group as i32)
        } else {
            scores
        };

        // Top-k selection
        let k = self.num_experts_per_tok as i32;
        let neg_scores = mlxcel_core::negative(&scores);
        let indices = mlxcel_core::argpartition(&neg_scores, k - 1, -1);
        let topk_indices = mlxcel_core::utils::slice_axis(&indices, -1, 0, k);

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

        // Apply experts via SwitchGLU
        let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);

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

        // Reshape back
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

        let router_weight = get_weight_copy(weights, &format!("{}.gate.weight", prefix))?;
        let e_score_correction_bias =
            get_weight_copy(weights, &format!("{}.gate.e_score_correction_bias", prefix))?;

        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{}.switch_mlp", prefix), group_size, bits)?;

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
            switch_mlp,
            shared_expert,
            num_experts_per_tok: args.num_experts_per_tok,
            n_group: args.n_group,
            topk_group: args.topk_group,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// ============================================================================
// FFN Enum: Dense or MoE
// ============================================================================

pub enum FFN {
    Dense(DenseMLP),
    Moe(MoE),
}

impl FFN {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            FFN::Dense(mlp) => mlp.forward(x),
            FFN::Moe(moe) => moe.forward(x),
        }
    }
}

// ============================================================================
// Decoder Layer (2-norm pre-LN, matching Python glm4_moe.py)
// ============================================================================

pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: FFN,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    /// Forward pass: standard pre-LN with 2 norms
    ///   r = self_attn(input_layernorm(x))
    ///   h = x + r
    ///   r = mlp(post_attention_layernorm(h))
    ///   return h + r
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

        let self_attn = Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;

        let mlp = if args.is_moe_layer(layer_idx) {
            FFN::Moe(MoE::from_weights(
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

// ============================================================================
// Solar Open Model
// ============================================================================

pub struct SolarOpenModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl SolarOpenModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let eval_layer_outputs = should_eval_layer_outputs(input_ids);
        let mut h = self.embed_tokens.forward(input_ids);

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
            // Keep the graph bounded for multi-token prefill. For single-token
            // decode, final-logits evaluation flushes the graph once; forcing
            // one sync per layer costs 48 GPU synchronizations per token.
            if eval_layer_outputs {
                let ptrs = [h.as_ref().unwrap() as *const MlxArray];
                unsafe { mlxcel_core::eval_all(&ptrs) };
            }
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

        println!(
            "[SolarOpen] Loading model: {} layers, {} experts, top-{}, group_size={}, bits={}",
            args.num_hidden_layers,
            args.n_routed_experts,
            args.num_experts_per_tok,
            args.group_size(),
            args.bits()
        );

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let weights = sanitize_weights(weights, &args);

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
            let layer = DecoderLayer::from_weights(weights, args, i)?;
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

// ============================================================================
// LanguageModel trait implementation
// ============================================================================

impl LanguageModel for SolarOpenModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        SolarOpenModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        SolarOpenModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Solar Open EOS tokens (from generation_config.json: [2, 24, 25])
        vec![2, 24, 25]
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn should_eval_layer_outputs(input_ids: &MlxArray) -> bool {
    mlxcel_core::array_shape(input_ids).last().copied() != Some(1)
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_open_config_parsing() {
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
        assert_eq!(args.group_size(), 128);
        assert_eq!(args.bits(), 4);
        assert!(args.is_moe_layer(0)); // All layers are MoE
        assert_eq!(args.rope_dims(), 128); // Full RoPE
    }

    #[test]
    fn solar_open_skips_per_layer_eval_for_decode_step() {
        let decode_ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
        let prefill_ids = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);

        assert!(!should_eval_layer_outputs(&decode_ids));
        assert!(should_eval_layer_outputs(&prefill_ids));
    }
}
