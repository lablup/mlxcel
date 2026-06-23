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

//! Mellum (Mellum 2) model implementation using mlxcel-core.
//!
//! Mellum 2 is JetBrains' sliding/full hybrid-attention MoE code model.
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mellum.py
//!
//! Key features:
//! - QK-RMSNorm: `q_norm`/`k_norm` are `RMSNorm(head_dim)` applied to the
//!   reshaped per-head q/k BEFORE RoPE. All projections are bias-free.
//! - Hybrid attention driven by `layer_types[i]`:
//!   - `"full_attention"` uses a causal mask + standard `KVCache` and a
//!     **YaRN**-scaled RoPE.
//!   - `"sliding_attention"` uses a windowed mask + `RotatingKVCache` and a
//!     default RoPE.
//! - Per-layer RoPE built from the `rope_parameters` dict keyed by layer_type.
//! - Sparse MoE (softmax-routed top-k, `norm_topk_prob`) via the shared
//!   `SwitchGLU`. `mlp_layer_types` selects MoE vs dense per layer (every layer
//!   is sparse in the released checkpoint).
//! - Pre-norm residual decoder layer.
//! - Untied or tied LM head, driven by `tie_word_embeddings`.

use crate::models::switch_layers::{SwitchGLU, fused_moe_enabled, moe_weighted_sum};
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

const FULL_ATTENTION: &str = "full_attention";
const SLIDING_ATTENTION: &str = "sliding_attention";

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,

    pub layer_types: Vec<String>,

    /// Per-layer-type RoPE parameters, keyed by `"full_attention"` /
    /// `"sliding_attention"`. Each value is a dict with `rope_theta` plus the
    /// optional YaRN scaling fields.
    #[serde(default)]
    pub rope_parameters: HashMap<String, serde_json::Value>,

    /// Per-layer MLP kind (`"sparse"` for MoE, anything else for dense). All
    /// layers are `"sparse"` in the released Mellum 2 checkpoint; kept so a
    /// future dense/sparse mix dispatches correctly.
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<String>>,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_tie_word_embeddings() -> bool {
    false
}

fn default_max_position_embeddings() -> usize {
    131072
}

fn default_sliding_window() -> usize {
    1024
}

impl ModelArgs {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// Whether layer `idx` is a sliding-window attention layer.
    pub(crate) fn is_sliding(&self, idx: usize) -> bool {
        self.layer_types
            .get(idx)
            .map(|t| t == SLIDING_ATTENTION)
            .unwrap_or(false)
    }

    /// Whether layer `idx` is a sparse MoE layer. Defaults to MoE when
    /// `mlp_layer_types` is absent (the released checkpoint is fully sparse).
    pub(crate) fn is_moe_layer(&self, idx: usize) -> bool {
        match &self.mlp_layer_types {
            Some(types) => types.get(idx).map(|t| t == "sparse").unwrap_or(true),
            None => true,
        }
    }

    /// RoPE base frequency for the given layer type (default 500000 when the
    /// `rope_parameters` entry is missing).
    pub(crate) fn rope_theta_for(&self, layer_type: &str) -> f32 {
        self.rope_parameters
            .get(layer_type)
            .and_then(|p| p.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(500_000.0) as f32
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                n.as_i64().map(|id| vec![id as i32]).unwrap_or_default()
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Pre-computed YaRN RoPE frequencies plus the mscale applied to Q/K before the
/// rotation. Mirrors mlx-lm's `YarnRoPE`
/// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py).
pub(crate) struct YarnRope {
    pub(crate) freqs: UniquePtr<MlxArray>,
    pub(crate) mscale: f32,
}

/// Build the YaRN frequencies for a `full_attention` layer from its
/// `rope_parameters` entry. Returns `None` when the entry is not a YaRN config
/// (then the layer falls back to a default RoPE).
pub(crate) fn compute_yarn_rope(head_dim: usize, params: &serde_json::Value) -> Option<YarnRope> {
    let rope_type = params
        .get("rope_type")
        .or_else(|| params.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    if rope_type != "yarn" {
        return None;
    }

    let base = params
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(500_000.0) as f32;
    let factor = params.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let original_max_pos = params
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_f64())
        .unwrap_or(4096.0) as f32;
    let beta_fast = params
        .get("beta_fast")
        .and_then(|v| v.as_f64())
        .unwrap_or(32.0) as f32;
    let beta_slow = params
        .get("beta_slow")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let mscale = params.get("mscale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let mscale_all_dim = params
        .get("mscale_all_dim")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let dims = head_dim as f32;
    let half_dims = head_dim / 2;

    // yarn_find_correction_dim / yarn_find_correction_range.
    let find_correction_dim = |num_rotations: f32| -> f32 {
        (dims * (original_max_pos / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * base.ln())
    };
    let low = find_correction_dim(beta_fast).floor().max(0.0) as usize;
    let high = find_correction_dim(beta_slow).ceil().min(dims - 1.0) as usize;

    // yarn_get_mscale. mlx-lm derives the attention factor from mscale /
    // mscale_all_dim; the config's explicit `attention_factor` equals this value.
    let get_mscale = |scale: f32, ms: f32| -> f32 {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * ms * scale.ln() + 1.0
        }
    };
    let rope_mscale = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim);

    // yarn_linear_ramp_mask + frequency interpolation.
    let ramp_min = low as f32;
    let ramp_max = if high == low {
        high as f32 + 0.001
    } else {
        high as f32
    };
    let mut freqs_data = vec![0.0f32; half_dims];
    for (i, freq_out) in freqs_data.iter_mut().enumerate() {
        let freq_extra = base.powf((2 * i) as f32 / dims);
        let freq_inter = factor * freq_extra;
        let ramp = ((i as f32 - ramp_min) / (ramp_max - ramp_min)).clamp(0.0, 1.0);
        let freq_mask = 1.0 - ramp;
        *freq_out =
            (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask));
    }

    Some(YarnRope {
        freqs: mlxcel_core::from_slice_f32(&freqs_data, &[half_dims as i32]),
        mscale: rope_mscale,
    })
}

// Attention with QK-RMSNorm and per-layer RoPE.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub is_sliding: bool,
    pub window_size: i32,
    /// Default-RoPE base for sliding layers (ignored when `rope_freqs` is set).
    pub rope_base: f32,
    /// YaRN frequencies for full-attention layers (None = default RoPE).
    pub rope_freqs: Option<UniquePtr<MlxArray>>,
    pub rope_mscale: f32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Project Q, K, V.
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, n_heads, head_dim].
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // QK-RMSNorm over head_dim, applied BEFORE transpose and RoPE.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Transpose to [batch, n_heads, seq_len, head_dim].
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset();

        // Apply RoPE: YaRN frequencies for full-attention layers, default RoPE
        // for sliding layers.
        let (q, k) = if let Some(ref freqs) = self.rope_freqs {
            let (q, k) = if (self.rope_mscale - 1.0).abs() > 1e-6 {
                (
                    mlxcel_core::multiply_scalar(&q, self.rope_mscale),
                    mlxcel_core::multiply_scalar(&k, self.rope_mscale),
                )
            } else {
                (q, k)
            };
            let q = mlxcel_core::fast_rope_with_freqs(&q, self.head_dim, false, 1.0, offset, freqs);
            let k = mlxcel_core::fast_rope_with_freqs(&k, self.head_dim, false, 1.0, offset, freqs);
            (q, k)
        } else {
            let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
            let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);
            (q, k)
        };

        // Update KV cache and get sliced views.
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention (handles GQA expansion internally).
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

        // Transpose back and reshape.
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
        rope_freqs: Option<UniquePtr<MlxArray>>,
        rope_mscale: f32,
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

        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;

        let head_dim = args.head_dim as i32;
        let is_sliding = args.is_sliding(layer_idx);
        let layer_type = if is_sliding {
            SLIDING_ATTENTION
        } else {
            FULL_ATTENTION
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RMSNorm::new(q_norm_weight, args.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, args.rms_norm_eps),
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            is_sliding,
            window_size: if is_sliding {
                args.sliding_window as i32
            } else {
                0
            },
            rope_base: args.rope_theta_for(layer_type),
            rope_freqs,
            rope_mscale,
        })
    }
}

// Sparse MoE block (softmax-routed top-k experts).
pub struct SparseMoeBlock {
    pub gate: UnifiedLinear,
    pub switch_mlp: SwitchGLU,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];

        // Flatten to [n_tokens, hidden].
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        // Router logits -> softmax routing probabilities. Use the precise
        // (f32-accumulation) softmax to match upstream mellum.py, which routes
        // with `mx.softmax(..., precise=True)` for router stability (same as olmoe).
        let logits = self.gate.forward(&x_flat);
        let gates = mlxcel_core::softmax_precise(&logits, -1);

        // Top-k selection via argpartition (top-k of logits == top-k of softmax).
        let k = self.num_experts_per_tok as i32;
        let n_experts = mlxcel_core::array_shape(&logits)[1];
        let kth = n_experts - k;
        let indices = mlxcel_core::argpartition(&logits, kth, -1);
        let indices_shape = mlxcel_core::array_shape(&indices);
        let topk_indices =
            mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

        // Gather scores and normalize.
        let mut scores = mlxcel_core::take_along_axis(&gates, &topk_indices, -1);
        if self.norm_topk_prob {
            let sum = mlxcel_core::sum_axis(&scores, -1, true);
            scores = mlxcel_core::divide(&scores, &sum);
        }

        // Single-token decode: try the fused MoE expert kernel, fall back to the
        // gather path for any unsupported config (e.g. non-quantized weights).
        let result = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1 && fused_moe_enabled() {
                self.switch_mlp
                    .forward_fused_kernel(&x_flat, &topk_indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.switch_mlp.forward(&x_flat, &topk_indices);
                    moe_weighted_sum(&expert_out, &scores, mlxcel_core::array_dtype(&x_flat))
                }
            }
        };

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
        let gate = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate", prefix),
            args.group_size(),
            args.bits(),
        )?;
        let switch_mlp = SwitchGLU::from_weights(
            weights,
            &format!("{}.switch_mlp", prefix),
            args.group_size(),
            args.bits(),
        )?;
        Ok(Self {
            gate,
            switch_mlp,
            num_experts_per_tok: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
        })
    }
}

// Dense MLP (SwiGLU). Only used when `mlp_layer_types` marks a layer non-sparse.
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

pub enum MLPType {
    Dense(MLP),
    MoE(SparseMoeBlock),
}

impl MLPType {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            MLPType::Dense(mlp) => mlp.forward(x),
            MLPType::MoE(moe) => moe.forward(x),
        }
    }
}

// Decoder layer (pre-norm residual).
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: MLPType,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
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
        rope_freqs: Option<UniquePtr<MlxArray>>,
        rope_mscale: f32,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Attention::from_weights(
            weights,
            args,
            &format!("{}.self_attn", prefix),
            layer_idx,
            rope_freqs,
            rope_mscale,
        )?;

        let mlp = if args.is_moe_layer(layer_idx) {
            MLPType::MoE(SparseMoeBlock::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            MLPType::Dense(MLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        };

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_weight, args.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm_weight, args.rms_norm_eps),
        })
    }
}

// Cache interface (KVCache / RotatingKVCache polymorphism).
pub trait CacheInterface {
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

pub enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Cache::Standard(c) => c,
            Cache::Rotating(c) => c,
        }
    }
}

// Mellum model.
pub struct MellumModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub sliding_window: usize,
    pub eos_token_ids: Vec<i32>,
}

impl MellumModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        mask_full: Option<&MlxArray>,
        mask_sliding: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.self_attn.is_sliding {
                mask_sliding
            } else {
                mask_full
            };
            h = layer.forward(&h, caches[i].as_interface(), mask);
        }

        let h = self.norm.forward(&h);
        if let Some(head) = &self.lm_head {
            head.forward(&h)
        } else {
            self.embed_tokens.as_linear(&h)
        }
    }

    pub fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = shape[1];

        // Decode (seq_len == 1) needs no explicit mask; the fused SDPA handles
        // single-token causal/windowed attention via `window_size`.
        if seq_len == 1 {
            return self.forward(input_ids, caches, None, None);
        }

        // Prefill: build a full causal mask and a windowed mask, keyed off the
        // first cache of each kind so offsets stay correct across turns.
        let full_idx = self
            .layers
            .iter()
            .position(|l| !l.self_attn.is_sliding)
            .unwrap_or(0);
        let sliding_idx = self.layers.iter().position(|l| l.self_attn.is_sliding);

        let full_offset = caches[full_idx].as_interface().offset();
        let mask_full = Some(create_causal_mask(seq_len, full_offset));

        let mask_sliding = sliding_idx.map(|idx| {
            // Full-width windowed mask for a fresh single-pass prefill that
            // exceeds the window (the RotatingKVCache keeps all prefill keys),
            // clamped mask otherwise. See issue #408.
            let sliding_offset = caches[idx].as_interface().offset();
            let max_cache = self.sliding_window as i32;
            create_sliding_window_prefill_mask(seq_len, sliding_offset, max_cache)
        });

        self.forward(
            input_ids,
            caches,
            mask_full.as_ref().map(|m| m.as_ref().unwrap()),
            mask_sliding.as_ref().map(|m| m.as_ref().unwrap()),
        )
    }

    /// Create per-layer caches: `KVCache` for full-attention layers,
    /// `RotatingKVCache(max_size=sliding_window)` for sliding layers.
    pub fn make_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.self_attn.is_sliding {
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window as i32))
                } else {
                    Cache::Standard(KVCache::new())
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

        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        // Stack per-expert tensors and drop a tied lm_head in place (no copy) so
        // the constructor sees the same pre-stacked layout as upstream.
        Self::sanitize(&mut weights, &args);
        let model = Self::from_weights(&weights, &args)?;
        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Compute YaRN frequencies once from the full-attention rope params, then
        // clone per full-attention layer.
        let yarn = args
            .rope_parameters
            .get(FULL_ATTENTION)
            .and_then(|p| compute_yarn_rope(args.head_dim, p));

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let (rope_freqs, rope_mscale) = if args.is_sliding(i) {
                (None, 1.0)
            } else {
                match &yarn {
                    Some(y) => (Some(mlxcel_core::copy(&y.freqs)), y.mscale),
                    None => (None, 1.0),
                }
            };
            layers.push(DecoderLayer::from_weights(
                weights,
                args,
                i,
                rope_freqs,
                rope_mscale,
            )?);
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

        let eos_token_ids = args.eos_token_ids();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: args.sliding_window,
            eos_token_ids,
        })
    }

    /// Mirror upstream `Model.sanitize`: drop a tied `lm_head.weight`, then stack
    /// per-expert `model.layers.{l}.mlp.experts.{e}.{up,down,gate}_proj.*`
    /// tensors into `...mlp.switch_mlp.{proj}.*`. A no-op when the experts are
    /// already pre-stacked (no `experts.0` weight present).
    pub fn sanitize(weights: &mut WeightMap, args: &ModelArgs) {
        if args.tie_word_embeddings {
            weights.remove("lm_head.weight");
        }

        let probe = "model.layers.0.mlp.experts.0.up_proj.weight";
        if !weights.contains_key(probe) {
            return;
        }

        for l in 0..args.num_hidden_layers {
            if !args.is_moe_layer(l) {
                continue;
            }
            let prefix = format!("model.layers.{}", l);
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                stack_experts(weights, &prefix, proj, args.num_experts);
            }
        }
    }
}

/// Stack `{prefix}.mlp.experts.{e}.{proj}.{suffix}` across experts into
/// `{prefix}.mlp.switch_mlp.{proj}.{suffix}` for weight/scales/biases.
fn stack_experts(weights: &mut WeightMap, prefix: &str, proj: &str, num_experts: usize) {
    for suffix in ["weight", "scales", "biases"] {
        let first = format!("{}.mlp.experts.0.{}.{}", prefix, proj, suffix);
        if !weights.contains_key(&first) {
            continue;
        }
        let mut parts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let key = format!("{}.mlp.experts.{}.{}.{}", prefix, e, proj, suffix);
            match weights.remove(&key) {
                Some(w) => parts.push(w),
                None => return,
            }
        }
        let stacked = mlxcel_core::stack_owned(&parts, 0);
        weights.insert(
            format!("{}.mlp.switch_mlp.{}.{}", prefix, proj, suffix),
            stacked,
        );
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel wrapper (owns the mixed full/sliding caches internally).
pub struct MellumWrapper {
    model: MellumModel,
    caches: RefCell<Vec<Cache>>,
}

impl MellumWrapper {
    pub fn new(model: MellumModel) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            caches: RefCell::new(caches),
        }
    }

    pub fn reset_caches(&self) {
        *self.caches.borrow_mut() = self.model.make_caches();
    }
}

impl mlxcel_core::generate::LanguageModel for MellumWrapper {
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
        // Reset the internal mixed caches; the returned slice is unused (the
        // model owns its full/sliding cache state).
        self.reset_caches();
        (0..self.model.layers.len())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn supports_batching(&self) -> bool {
        // Mellum keeps mixed full/sliding caches in a RefCell, which is not
        // compatible with per-sequence KV isolation.
        false
    }

    fn reset_runtime_state(&self) {
        self.reset_caches();
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }
}
