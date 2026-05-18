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

//! Hunyuan MoE model implementation using mlxcel-core
//!
//! Key features:
//! - DynamicNTK-Alpha RoPE: base adjusted with alpha at init time
//! - Optional Q/K normalization (query_layernorm, key_layernorm)
//! - Sparse MoE with SwitchGLU + optional shared_mlp (use_mixed_mlp_moe)
//! - Per-layer moe_intermediate_size and moe_topk
//! - Cross-Layer Attention (CLA) support (optional KV sharing)

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
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
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub num_experts: usize,

    #[serde(default)]
    pub moe_topk: MoeTopk,

    #[serde(default)]
    pub moe_intermediate_size: MoeIntermediateSize,

    #[serde(default)]
    pub num_shared_expert: NumSharedExpert,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default = "default_use_qk_norm")]
    pub use_qk_norm: bool,

    #[serde(default = "default_use_mixed_mlp_moe")]
    pub use_mixed_mlp_moe: bool,

    #[serde(default)]
    pub use_cla: bool,

    #[serde(default = "default_cla_share_factor")]
    pub cla_share_factor: usize,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub alpha: f32,
    pub factor: f32,
    #[serde(rename = "type")]
    pub scaling_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

// Support both single int and array for moe_topk
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MoeTopk {
    Single(usize),
    Array(Vec<usize>),
}

impl Default for MoeTopk {
    fn default() -> Self {
        MoeTopk::Single(8)
    }
}

impl MoeTopk {
    pub fn get(&self, layer_idx: usize) -> usize {
        match self {
            MoeTopk::Single(v) => *v,
            MoeTopk::Array(arr) => arr.get(layer_idx).copied().unwrap_or(8),
        }
    }
}

// Support both single int and array for moe_intermediate_size
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MoeIntermediateSize {
    Single(usize),
    Array(Vec<usize>),
}

impl Default for MoeIntermediateSize {
    fn default() -> Self {
        MoeIntermediateSize::Single(3072)
    }
}

impl MoeIntermediateSize {
    pub fn get(&self, layer_idx: usize) -> usize {
        match self {
            MoeIntermediateSize::Single(v) => *v,
            MoeIntermediateSize::Array(arr) => arr.get(layer_idx).copied().unwrap_or(3072),
        }
    }
}

// Support both single int and array for num_shared_expert
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum NumSharedExpert {
    Single(usize),
    Array(Vec<usize>),
}

impl Default for NumSharedExpert {
    fn default() -> Self {
        NumSharedExpert::Single(1)
    }
}

impl NumSharedExpert {
    pub fn get(&self, layer_idx: usize) -> usize {
        match self {
            NumSharedExpert::Single(v) => *v,
            NumSharedExpert::Array(arr) => arr.get(layer_idx).copied().unwrap_or(1),
        }
    }
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_max_position_embeddings() -> usize {
    32768
}

fn default_use_qk_norm() -> bool {
    true
}

fn default_use_mixed_mlp_moe() -> bool {
    true
}

fn default_cla_share_factor() -> usize {
    2
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
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

    pub fn scaling_alpha(&self) -> f32 {
        self.rope_scaling.as_ref().map(|s| s.alpha).unwrap_or(1.0)
    }

    /// Check if this layer has its own KV projections (for CLA support)
    pub fn has_kv_proj(&self, layer_idx: usize) -> bool {
        !self.use_cla || layer_idx.is_multiple_of(self.cla_share_factor)
    }
}

// SwitchLinear: Stacked expert weights for MoE.
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
            } => unsafe {
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
            },
            Self::Regular { weight } => {
                let wt = mlxcel_core::swap_axes(weight, -1, -2);
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

// SwitchGLU: SwiGLU with stacked expert weights.
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

        // Check if we should use sorted_indices optimization
        let total_elements = n_tokens * top_k;
        let do_sort = total_elements >= 64;

        // Expand x for broadcasting: [n_tokens, hidden] -> [n_tokens, 1, 1, hidden]
        let x_expanded = mlxcel_core::expand_dims(x, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) = self.gather_sort(&x_expanded, indices);

            let x_gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            let x_up = self.up_proj.forward(&sorted_x, &sorted_idx, true);

            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);

            self.scatter_unsort(&output, &inv_order, &indices_shape)
        } else {
            let x_gate = self.gate_proj.forward(&x_expanded, indices, false);
            let x_up = self.up_proj.forward(&x_expanded, indices, false);

            let activated = mlxcel_core::compiled_swiglu_activation(&x_gate, &x_up);
            let output = self.down_proj.forward(&activated, indices, false);

            // Squeeze: [n_tokens, top_k, 1, hidden] -> [n_tokens, top_k, hidden]
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
        let gate_proj = SwitchLinear::from_weights(
            weights,
            &format!("{}.gate_proj", prefix),
            group_size,
            bits,
        )?;
        let up_proj =
            SwitchLinear::from_weights(weights, &format!("{}.up_proj", prefix), group_size, bits)?;
        let down_proj = SwitchLinear::from_weights(
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

// Dense MLP (shared_mlp).
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

// Gate (Router).
pub struct Gate {
    pub wg: UnifiedLinear,
}

impl Gate {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.wg.forward(x)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let wg = UnifiedLinear::from_weights(weights, &format!("{}.wg", prefix), group_size, bits)?;
        Ok(Self { wg })
    }
}

// MoE Block.
pub struct MoeBlock {
    pub shared_mlp: Option<MLP>,
    pub gate: Gate,
    pub switch_mlp: SwitchGLU,
    pub top_k: usize,
}

impl MoeBlock {
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

        // Get router logits and apply softmax
        let logits = self.gate.forward(&x_flat);
        let gates = mlxcel_core::softmax(&logits, -1);

        // Top-k selection
        let k = self.top_k as i32;
        let n_experts = mlxcel_core::array_shape(&logits)[1];
        let kth = n_experts - k;

        let indices = mlxcel_core::argpartition(&logits, kth, -1);
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // Get scores for top-k experts (already normalized from softmax, no re-normalization)
        let scores = mlxcel_core::take_along_axis(&gates, &topk_indices, -1);

        // Apply experts
        let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);

        let mut result = crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        // Add shared expert output if present
        if let Some(ref shared_mlp) = self.shared_mlp {
            let shared_out = shared_mlp.forward(&x_flat);
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
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load shared MLP if use_mixed_mlp_moe
        let shared_mlp = if args.use_mixed_mlp_moe {
            Some(MLP::from_weights(
                weights,
                &format!("{}.shared_mlp", prefix),
                group_size,
                bits,
            )?)
        } else {
            None
        };

        // Load gate
        let gate = Gate::from_weights(weights, &format!("{}.gate", prefix), group_size, bits)?;

        // Load switch_mlp (MoE experts)
        let switch_mlp =
            SwitchGLU::from_weights(weights, &format!("{}.switch_mlp", prefix), group_size, bits)?;

        let top_k = args.moe_topk.get(layer_idx);

        Ok(Self {
            shared_mlp,
            gate,
            switch_mlp,
            top_k,
        })
    }
}

// Attention.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: Option<UnifiedLinear>,
    pub v_proj: Option<UnifiedLinear>,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_freqs: UniquePtr<MlxArray>,
    pub use_qk_norm: bool,
    pub query_layernorm: Option<RMSNorm>,
    pub key_layernorm: Option<RMSNorm>,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q
        let q = self.q_proj.forward(x);

        // Get K, V from projections or shared state
        let (k, v, kv_out) = if let Some((shared_k, shared_v)) = shared_kv {
            (
                mlxcel_core::copy(shared_k),
                mlxcel_core::copy(shared_v),
                None,
            )
        } else if let (Some(k_proj), Some(v_proj)) = (&self.k_proj, &self.v_proj) {
            let k = k_proj.forward(x);
            let v = v_proj.forward(x);
            let k_copy = mlxcel_core::copy(&k);
            let v_copy = mlxcel_core::copy(&v);
            (k, v, Some((k_copy, v_copy)))
        } else {
            panic!("Attention layer missing KV projections and no shared KV provided");
        };

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let mut q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let mut k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE with custom freqs
        let q = mlxcel_core::fast_rope_with_freqs(
            &q,
            self.head_dim,
            false,
            1.0,
            offset,
            &self.rope_freqs,
        );
        let k = mlxcel_core::fast_rope_with_freqs(
            &k,
            self.head_dim,
            false,
            1.0,
            offset,
            &self.rope_freqs,
        );

        // Apply Q/K normalization if enabled
        let q = if let Some(ref q_norm) = self.query_layernorm {
            q_norm.forward(&q)
        } else {
            q
        };
        let k = if let Some(ref k_norm) = self.key_layernorm {
            k_norm.forward(&k)
        } else {
            k
        };

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention (handles GQA expansion internally)
        let attn_out = if l > 1 {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        } else {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        };

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        // Output projection
        (self.o_proj.forward(&attn_out), kv_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;

        // Only load K/V projections if this layer has them
        let (k_proj, v_proj) = if args.has_kv_proj(layer_idx) {
            let k = UnifiedLinear::from_weights(
                weights,
                &format!("{}.k_proj", prefix),
                group_size,
                bits,
            )?;
            let v = UnifiedLinear::from_weights(
                weights,
                &format!("{}.v_proj", prefix),
                group_size,
                bits,
            )?;
            (Some(k), Some(v))
        } else {
            (None, None)
        };

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute DynamicNTK-Alpha RoPE frequencies
        let scaling_alpha = args.scaling_alpha();
        let dims = head_dim as f32;
        let adjusted_base = args.rope_theta * scaling_alpha.powf(dims / (dims - 2.0));

        // Compute freqs: adjusted_base ** (arange(0, dims, 2) / dims)
        let half_dims = head_dim / 2;
        let mut freqs_data = vec![0.0f32; half_dims as usize];
        for i in 0..half_dims {
            let exp = (2 * i) as f32 / dims;
            freqs_data[i as usize] = adjusted_base.powf(exp);
        }
        let rope_freqs = mlxcel_core::from_slice_f32(&freqs_data, &[half_dims]);

        // Load Q/K normalization layers if enabled
        let (query_layernorm, key_layernorm) = if args.use_qk_norm {
            let q_norm_weight =
                get_weight_copy(weights, &format!("{}.query_layernorm.weight", prefix))?;
            let k_norm_weight =
                get_weight_copy(weights, &format!("{}.key_layernorm.weight", prefix))?;

            (
                Some(RMSNorm::new(q_norm_weight, args.rms_norm_eps)),
                Some(RMSNorm::new(k_norm_weight, args.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale,
            rope_freqs,
            use_qk_norm: args.use_qk_norm,
            query_layernorm,
            key_layernorm,
        })
    }
}

// Transformer Block.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MoeBlock,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub has_kv_proj: bool,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let (attn_out, kv_out) = self.self_attn.forward(&normed, cache, mask, shared_kv);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN (MoE)
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        (mlxcel_core::add(&h, &ff_out), kv_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            Attention::from_weights(weights, args, layer_idx, &format!("{}.self_attn", prefix))?;
        let mlp = MoeBlock::from_weights(weights, args, layer_idx, &format!("{}.mlp", prefix))?;

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
            has_kv_proj: args.has_kv_proj(layer_idx),
        })
    }
}

// Hunyuan MoE Model.
pub struct HunyuanMoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub use_cla: bool,
    pub cla_share_factor: usize,
}

impl HunyuanMoeModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        let mask = mask.map(mlxcel_core::copy);

        // Track shared KV states for CLA
        let mut shared_kv: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            // Determine if we should use shared KV
            let use_shared = self.use_cla && (i % self.cla_share_factor) != 0;

            let kv_ref = if use_shared {
                shared_kv.as_ref().and_then(|(k, v)| {
                    let k_ref = k.as_ref()?;
                    let v_ref = v.as_ref()?;
                    Some((k_ref, v_ref))
                })
            } else {
                None
            };

            let (new_h, kv_out) = layer.forward(&h, &mut caches[i], mask.as_deref(), kv_ref);
            h = new_h;

            // Update shared KV if this layer produced new KV states
            if let Some((k, v)) = kv_out {
                shared_kv = Some((k, v));
            }
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        if let Some(head) = &self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
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

        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            use_cla: args.use_cla,
            cla_share_factor: args.cla_share_factor,
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
impl LanguageModel for HunyuanMoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        HunyuanMoeModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        HunyuanMoeModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![127960] // Hunyuan EOS token from config
    }
}
