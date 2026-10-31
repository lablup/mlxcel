// Copyright 2025-2026 Lablup Inc.
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

use crate::models::gpt_oss_common::{
    ExpertLinear, YarnRopeParams, gpt_oss_swiglu, yarn_rope_freqs,
};
use crate::models::model_owned::{KvCacheLayerModes, LayerKv, ModelOwnedSequenceState};
use mlxcel_core::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, route_batched_decode_with_ids};
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::distributed::pipeline::LayerFilter;
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::partial_loading::filter_weight_map;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
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

mlxcel_core::impl_rope_lifting_deserialize!(ModelArgs);

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

    /// Reject any `quantization` entry that would abort an MLX kernel.
    ///
    /// gpt-oss is the one family whose `quantization` block is a map of
    /// per-weight-prefix overrides rather than a single triple, so a bound on
    /// the top-level defaults alone says nothing about what an individual
    /// projection will load with, and vice versa. This walks the top-level pair
    /// and every override in one pass, so a hostile pair scoped to a single
    /// prefix is refused during config validation even when the defaults are
    /// valid, before any tensor is touched (issue #958).
    ///
    /// Entries whose value is not an object are the top-level scalars
    /// themselves (`group_size`, `bits`, `mode`) and are covered by the
    /// `defaults()` check rather than iterated as overrides.
    ///
    /// The declared `mode` is walked the same way (issue #973). gpt-oss is the
    /// only family that reads `quantization.mode` out of `config.json` and
    /// threads it into a loader, so this is the only place a hostile mode can be
    /// blamed on the prefix that declared it. A mode-less override keeps
    /// resolving to `"affine"` through `params_for` rather than inheriting the
    /// top-level mode, which is what the 24 router overrides in the canonical
    /// mxfp4 checkpoint rely on.
    fn validate(&self) -> Result<(), String> {
        let Quantization::Full(map) = self;
        let (gs, bits, mode) = self.defaults();
        mlxcel_core::layers::validate_quantization_params(gs, bits)
            .map_err(|e| format!("GptOss config.json quantization: {e}"))?;
        mlxcel_core::layers::validate_quantization_mode(&mode)
            .map_err(|e| format!("GptOss config.json quantization: {e}"))?;
        for key in map.keys() {
            let Some(value) = map.get(key) else { continue };
            if !value.is_object() {
                continue;
            }
            let (gs, bits, mode) = self.params_for(key);
            mlxcel_core::layers::validate_quantization_params(gs, bits)
                .map_err(|e| format!("GptOss config.json quantization.{key}: {e}"))?;
            mlxcel_core::layers::validate_quantization_mode(&mode)
                .map_err(|e| format!("GptOss config.json quantization.{key}: {e}"))?;
        }
        Ok(())
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
    /// Reject a `quantization` block that would abort an MLX kernel.
    ///
    /// Family-level early diagnostic, in the same shape as the BailingMoe,
    /// GptNeoX and Helium ones added by #929: it names gpt-oss and fires during
    /// config validation, before any tensor is touched. The authoritative bound
    /// still lives at each storage point (`ExpertLinear::from_weights` and the
    /// shared reconciler behind `UnifiedLinear` / `UnifiedEmbedding`); this
    /// exists because gpt-oss is the only family whose per-prefix overrides can
    /// each carry their own hostile pair (issue #958) or their own hostile
    /// `mode` (issue #973).
    pub fn validate_quantization(&self) -> Result<(), String> {
        let Some(quantization) = self.quantization.as_ref() else {
            return Ok(());
        };
        quantization.validate()
    }

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

    /// The sliding window as the `i32` the rotating caches take, checked at
    /// load when `layer_types` has a sliding layer (issue #2790); a config
    /// with none keeps any window, unused.
    pub(crate) fn validated_sliding_window(&self) -> Result<i32, String> {
        let has_sliding_layer = self
            .layer_types_list()
            .iter()
            .any(|layer_type| layer_type != "full_attention");
        crate::models::model_owned::cache_window("GPT-OSS", self.sliding_window, has_sliding_layer)
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

        let read = |key: &str, default: f64| -> f32 {
            rope_scaling
                .get(key)
                .and_then(|v| v.as_f64())
                .unwrap_or(default) as f32
        };
        let params = YarnRopeParams {
            dims: self.head_dim,
            base: self.rope_theta,
            factor: read("factor", 1.0),
            original_max_position_embeddings: rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as f32,
            beta_fast: read("beta_fast", 32.0),
            beta_slow: read("beta_slow", 1.0),
            mscale: read("mscale", 1.0),
            mscale_all_dim: read("mscale_all_dim", 0.0),
            // mlx-lm's YarnRoPE always floors / ceils the correction range.
            truncate: true,
        };
        let (freqs_data, rope_mscale) = yarn_rope_freqs(&params);
        let freqs = mlxcel_core::from_slice_f32(&freqs_data, &[freqs_data.len() as i32]);
        Some((freqs, rope_mscale))
    }
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

    fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [Vec<Cache>],
        layer_idx: usize,
    ) -> UniquePtr<MlxArray> {
        let b = caches.len() as i32;
        let project = |projection: &UnifiedLinear, heads| {
            let projected = projection.forward(x);
            let projected = mlxcel_core::reshape(&projected, &[b, 1, heads, self.head_dim]);
            mlxcel_core::transpose_axes(&projected, &[0, 2, 1, 3])
        };
        let q = project(&self.q_proj, self.num_heads);
        let k = project(&self.k_proj, self.num_kv_heads);
        let v = project(&self.v_proj, self.num_kv_heads);
        let offsets: Vec<_> = caches
            .iter()
            .map(|row| gpt_oss_cache_offset(&row[layer_idx]))
            .collect();
        let rope = |x: &MlxArray| {
            if let Some(freqs) = &self.rope_freqs {
                let scaled = if (self.rope_mscale - 1.0).abs() > 1e-6 {
                    mlxcel_core::multiply_scalar(x, self.rope_mscale)
                } else {
                    mlxcel_core::copy(x)
                };
                mlxcel_core::fast_rope_batched_with_freqs(
                    &scaled,
                    self.rope_dims,
                    false,
                    1.0,
                    &offsets,
                    freqs,
                )
            } else {
                mlxcel_core::fast_rope_batched(
                    x,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    1.0,
                    &offsets,
                )
            }
        };
        let q = rope(&q);
        let k = rope(&k);
        let attended = self.attend_rows_with_sinks(&q, &k, &v, caches, layer_idx);
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        let attended = mlxcel_core::reshape(&attended, &[b, 1, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attended)
    }

    // KvAttention::attend / attend_batched_rows cannot carry learned sinks.
    // Keep the sink-aware SDPA and mixed-cache updates per row, while all
    // projections and the rest of the transformer run once for the batch.
    fn attend_rows_with_sinks(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        caches: &mut [Vec<Cache>],
        layer_idx: usize,
    ) -> UniquePtr<MlxArray> {
        let rows: Vec<_> = caches
            .iter_mut()
            .enumerate()
            .map(|(row, caches)| {
                let slice_row = |x: &MlxArray, heads| {
                    mlxcel_core::slice(
                        x,
                        &[row as i32, 0, 0, 0],
                        &[row as i32 + 1, heads, 1, self.head_dim],
                    )
                };
                let q = slice_row(q, self.num_heads);
                let k = slice_row(k, self.num_kv_heads);
                let v = slice_row(v, self.num_kv_heads);
                let (k, v) = caches[layer_idx].as_interface().update_and_fetch(k, v);
                // SAFETY: sinks is a live per-head array; the null mask requests
                // unmasked width-one decode over this row's visible KV only.
                unsafe {
                    mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
                        &q,
                        &k,
                        &v,
                        self.scale,
                        std::ptr::null(),
                        self.sinks.as_ref().unwrap() as *const _,
                    )
                }
            })
            .collect();
        mlxcel_core::concatenate_many(
            &rows
                .iter()
                .map(|row| row.as_ref().unwrap())
                .collect::<Vec<_>>(),
            0,
        )
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

    fn visible_len(&self) -> usize {
        match self {
            Cache::Standard(cache) => cache.live_len().max(0) as usize,
            Cache::Rotating(cache) => cache.visible_len().max(0) as usize,
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
    /// Validated at load ([`ModelArgs::validated_sliding_window`]).
    sliding_window: i32,
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
            let max_cache = self.sliding_window;
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

    fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for (idx, layer) in self.layers.iter().enumerate() {
            let normed = layer.input_layernorm.forward(&h);
            let attention = layer.self_attn.forward_batched_decode(&normed, caches, idx);
            h = mlxcel_core::add(&h, &attention);
            let normed = layer.post_attention_layernorm.forward(&h);
            h = mlxcel_core::add(&h, &layer.mlp.forward(&normed));
        }
        let h = self.norm.forward(&h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }

    /// One cache per layer: `KVCache` on full-attention layers and
    /// `RotatingKVCache(sliding_window)` elsewhere, each stored in its
    /// resolved KV cache mode (`None` keeps all layers FP16).
    fn make_caches_with_modes(&self, modes: Option<&KvCacheLayerModes>) -> Vec<Cache> {
        self.layer_types
            .iter()
            .enumerate()
            .map(|(layer_idx, lt)| {
                if lt == "full_attention" {
                    let mode =
                        modes.map_or(KVCacheMode::Fp16, |modes| modes.mode_for_layer(layer_idx));
                    Cache::Standard(KVCache::new_with_mode(mode))
                } else {
                    let window = self.sliding_window;
                    let mode = modes.map_or(KVCacheMode::Fp16, |modes| {
                        modes.rotating_mode_for_layer(layer_idx, window)
                    });
                    Cache::Rotating(RotatingKVCache::new_with_mode(window, mode))
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
        args.validate_quantization()?;

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

        let sliding_window = args.validated_sliding_window()?;
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
            sliding_window,
        })
    }
}

// Wrapper for LanguageModel trait (manages internal mixed caches).
// Used by: LoadedModel::GptOss
pub struct GptOssWrapper {
    model: GptOssModel,
    sequence_state: ModelOwnedSequenceState<Cache>,
    kv_cache_layer_modes: KvCacheLayerModes,
}

impl GptOssWrapper {
    #[cfg(test)]
    pub(crate) fn sequence_kv_modes(&self, id: SequenceId) -> Vec<KVCacheMode> {
        self.sequence_state
            .with_sequence_state_ref(id, |caches| {
                caches
                    .iter()
                    .map(|cache| match cache {
                        Cache::Standard(c) => c.mode,
                        Cache::Rotating(c) => c.mode,
                    })
                    .collect()
            })
            .expect("test sequence exists")
    }

    pub fn new(model: GptOssModel) -> Self {
        let caches = model.make_caches_with_modes(None);
        Self {
            model,
            sequence_state: ModelOwnedSequenceState::new(caches),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
        }
    }

    /// Fresh caches in the KV cache modes the server or CLI injected.
    fn fresh_caches(&self) -> Vec<Cache> {
        self.model
            .make_caches_with_modes(Some(&self.kv_cache_layer_modes))
    }

    fn reset_caches(&self) {
        self.sequence_state.replace_internal(self.fresh_caches());
    }

    fn assert_prepared_sequence_ids(&self, seq_ids: &[SequenceId]) {
        self.sequence_state
            .validate_sequence_states(seq_ids)
            .unwrap_or_else(|err| panic!("gpt-oss batched decode: {err}"));
    }
}

impl mlxcel_core::generate::LanguageModel for GptOssWrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            None,
            || self.fresh_caches(),
            |caches| self.model.forward_with_caches(input_ids, caches),
        )
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

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.reset_caches();
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    /// Sliding layers hold a `RotatingKVCache`, which stores FP16 for every
    /// mode but an aligned `Turbo4Asym` (issue #2783).
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<KVCacheMode>>> {
        let window = self.model.sliding_window;
        self.kv_cache_layer_modes.storage_modes(|layer_idx| {
            self.model
                .layer_types
                .get(layer_idx)
                .is_some_and(|layer_type| layer_type != "full_attention")
                .then_some(window)
                .map_or(LayerKv::Full, LayerKv::Sliding)
        })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.model.layers.len())
    }

    fn supports_padded_prefill(&self) -> bool {
        // Rotating caches cannot trim padding back out of model-owned state.
        false
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.fresh_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.fresh_caches(),
            |caches| self.model.forward_with_caches(input_ids, caches),
        )
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_existing_sequence_state(seq_id, |caches| {
                let visible_lens: Vec<_> = caches.iter().map(Cache::visible_len).collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &visible_lens)
            })?
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_ids(input_ids, None, batch_caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(shape.len(), 2, "gpt-oss batched input must be rank 2");
        assert_eq!(
            shape[0] as usize,
            batch_caches.len(),
            "gpt-oss input batch must match cache rows"
        );
        if let Some(ids) = seq_ids {
            assert_eq!(
                ids.len(),
                batch_caches.len(),
                "gpt-oss requires one sequence id per row"
            );
        }

        route_batched_decode_with_ids(self, input_ids, seq_ids, batch_caches, |batch_caches| {
            let ids = seq_ids.expect("gpt-oss model-owned batched decode requires sequence ids");
            if shape[1] != 1 || mask.is_some() {
                self.assert_prepared_sequence_ids(ids);
                let rows: Vec<_> = batch_caches
                    .iter_mut()
                    .enumerate()
                    .map(|(row, caches)| {
                        let input = mlxcel_core::slice(
                            input_ids,
                            &[row as i32, 0],
                            &[row as i32 + 1, shape[1]],
                        );
                        self.forward_with_sequence_id(&input, Some(ids[row]), caches, mask)
                    })
                    .collect();
                return mlxcel_core::concatenate_many(
                    &rows
                        .iter()
                        .map(|row| row.as_ref().unwrap())
                        .collect::<Vec<_>>(),
                    0,
                );
            }
            self.sequence_state
                .with_batched_sequence_states(ids, |caches| {
                    self.model
                        .forward_batched_decode_with_caches(input_ids, caches)
                })
                .unwrap_or_else(|err| panic!("gpt-oss batched decode: {err}"))
        })
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
    /// Validated at load ([`ModelArgs::validated_sliding_window`]).
    sliding_window: i32,
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
        args.validate_quantization()?;

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

        let sliding_window = args.validated_sliding_window()?;
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
            sliding_window,
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
                    Cache::Rotating(RotatingKVCache::new(self.sliding_window))
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
                    let max_cache = self.sliding_window;
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

    /// `ExpertLinear` stores the declared pair and hands it to `gather_qmm`
    /// without ever reaching `reconcile_quantization_layout`, so a hostile pair
    /// used to abort the process at the first routed forward pass rather than
    /// failing at load (issue #958). This drives the real loader rather than the
    /// pure bounds helper, so the guard is exercised where a checkpoint actually
    /// reaches it. The assertions are on the load result and no forward pass is
    /// run, so a regression fails cleanly here rather than aborting the test
    /// binary.
    /// Issue #1806: gpt-oss threads its declared mode into the experts, so an
    /// NVFP4 expert plane on a backend with no NVFP4 kernel must fail the load
    /// naming the layer rather than abort in `gather_qmm`.
    /// gpt-oss calls `gather_qmm` without a tensor scale, so an expert plane
    /// that carries one (resolved from mlx-lm's `global_scale: true`, or bare)
    /// fails the load instead of decoding every expert wrong (issue #2351).
    #[test]
    fn gpt_oss_expert_linear_refuses_a_per_expert_global_scale() {
        let prefix = "model.layers.0.mlp.experts.down_proj";
        for leaf in ["native_global_scale", "global_scale"] {
            let mut plane = crate::models::switch_layers::nvfp4_expert_plane(prefix);
            plane.insert(
                format!("{prefix}.{leaf}"),
                mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0], &[3]),
            );
            let err = ExpertLinear::from_weights(&plane, prefix, 16, 4, "nvfp4")
                .err()
                .expect("a tensor scale has no slot in this expert path");
            assert!(err.contains(prefix) && err.contains(leaf), "{err}");
        }
    }

    #[test]
    fn gpt_oss_expert_linear_refuses_a_mode_the_backend_cannot_run() {
        use mlxcel_core::hardware::{QuantMode, QuantModeSupport, quant_mode_support};
        let prefix = "model.layers.0.mlp.experts.down_proj";
        let plane = crate::models::switch_layers::nvfp4_expert_plane(prefix);
        let loaded = ExpertLinear::from_weights(&plane, prefix, 16, 4, "nvfp4");
        match quant_mode_support(QuantMode::Nvfp4) {
            QuantModeSupport::Native => {
                loaded.expect("a backend with native NVFP4 must keep loading it");
            }
            _ => {
                let err = loaded
                    .err()
                    .expect("NVFP4 experts must be refused on this backend");
                assert!(err.contains(prefix) && err.contains("nvfp4"), "{err}");
            }
        }
    }

    #[test]
    fn gpt_oss_expert_linear_rejects_params_that_would_abort_gather_qmm() {
        let prefix = "model.layers.0.mlp.experts.gate_up_proj";
        let mut weights = WeightMap::new();
        crate::models::switch_layers::insert_stacked_quantized_expert_plane(
            &mut weights,
            prefix,
            3,
            4,
            8,
            1,
        );

        // Positive control first, so a guard that rejects everything quantized
        // cannot pass this test.
        ExpertLinear::from_weights(&weights, prefix, 64, 4, "affine")
            .expect("an honest affine pair must still load");

        for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
            let err = ExpertLinear::from_weights(&weights, prefix, group_size, bits, "affine")
                .err()
                .unwrap_or_else(|| {
                    panic!("ExpertLinear must reject group_size {group_size} / bits {bits}")
                });
            assert!(err.contains(field), "unhelpful error: {err}");
        }

        // A bf16 expert plane carries no packing at all, so the params are
        // irrelevant there and must not be enforced.
        let mut regular = WeightMap::new();
        regular.insert(
            format!("{prefix}.weight"),
            mlxcel_core::from_slice_f32(&vec![0.0; 3 * 4 * 8], &[3, 4, 8]),
        );
        ExpertLinear::from_weights(&regular, prefix, 0, 0, "affine")
            .expect("a non-quantized expert plane must not be gated on quantization params");
    }

    /// The third value in the same `quantization` block. `ExpertLinear` stores
    /// the declared mode and hands it to `gather_qmm`, and gpt-oss is the only
    /// family that puts a `config.json` string there at all, so this is where a
    /// hostile mode actually reaches a kernel (issue #973). No forward pass is
    /// run, because MLX's throw would abort the test binary rather than fail it.
    #[test]
    fn gpt_oss_expert_linear_rejects_a_mode_that_would_abort_gather_qmm() {
        let prefix = "model.layers.0.mlp.experts.gate_up_proj";
        let mut affine = WeightMap::new();
        crate::models::switch_layers::insert_stacked_quantized_expert_plane(
            &mut affine,
            prefix,
            3,
            4,
            8,
            1,
        );

        // Positive control first, so a guard that rejects everything quantized
        // cannot pass this test.
        ExpertLinear::from_weights(&affine, prefix, 64, 4, "affine")
            .expect("an honest affine plane must still load");

        for mode in crate::models::switch_layers::HOSTILE_QUANT_MODES {
            let err = ExpertLinear::from_weights(&affine, prefix, 64, 4, mode)
                .err()
                .unwrap_or_else(|| panic!("ExpertLinear must reject mode {mode:?}"));
            assert!(
                err.contains("mode") && err.contains("gate_up_proj"),
                "the message must name the field and the prefix: {err}"
            );
        }

        // `from_weights` enters the quantized branch on `.scales` presence alone
        // and makes `.biases` an `Option` that becomes a null pointer when
        // absent, so a plane with scales and no zero points under a declared
        // `affine` mode used to be built without complaint and abort in
        // `gather_qmm`. This is MLX's own precondition, so it can only refuse a
        // load that was already going to abort.
        let mut no_biases = WeightMap::new();
        for (leaf, last, value) in [("weight", 8, 0.0), ("scales", 1, 1.0)] {
            no_biases.insert(
                format!("{prefix}.{leaf}"),
                mlxcel_core::from_slice_f32(&vec![value; (3 * 4 * last) as usize], &[3, 4, last]),
            );
        }
        // The shape the canonical mxfp4 checkpoint actually ships: 72 expert
        // planes with scales, no zero points, on the top-level block-float mode.
        ExpertLinear::from_weights(&no_biases, prefix, 32, 4, "mxfp4")
            .expect("the canonical mxfp4 expert plane must still load");
        let err = ExpertLinear::from_weights(&no_biases, prefix, 64, 4, "affine")
            .err()
            .unwrap_or_else(|| panic!("affine over a plane with no zero points must be rejected"));
        assert!(
            err.contains("biases") && err.contains("gate_up_proj"),
            "unhelpful error: {err}"
        );

        // And the mirror, which MLX rejects just as hard.
        let err = ExpertLinear::from_weights(&affine, prefix, 32, 4, "mxfp4")
            .err()
            .unwrap_or_else(|| panic!("block-float over a .biases plane must be rejected"));
        assert!(err.contains("biases"), "unhelpful error: {err}");

        // A non-quantized expert plane carries no mode at all.
        let mut regular = WeightMap::new();
        regular.insert(
            format!("{prefix}.weight"),
            mlxcel_core::from_slice_f32(&vec![0.0; 3 * 4 * 8], &[3, 4, 8]),
        );
        ExpertLinear::from_weights(&regular, prefix, 0, 0, "optiq")
            .expect("a non-quantized expert plane must not be gated on the declared mode");
    }

    /// gpt-oss is the one family whose `quantization` block is a map of
    /// per-weight-prefix overrides, so a bound on the top-level defaults says
    /// nothing about what an individual projection loads with, and vice versa.
    /// The canonical mxfp4 checkpoint carries a valid `model.embed_tokens`
    /// override, which is exactly what let a hostile top-level pair reach the
    /// experts alone while every reconciler-guarded loader saw a good one.
    #[test]
    fn gpt_oss_config_validation_walks_every_per_prefix_override() {
        let args_from = |quantization: serde_json::Value| -> ModelArgs {
            serde_json::from_value(serde_json::json!({
                "model_type": "gpt_oss",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
                "rms_norm_eps": 1e-5,
                "rope_theta": 150000.0,
                "sliding_window": 8,
                "quantization": quantization,
            }))
            .expect("test config must parse")
        };

        // Positive control: the real gpt-oss-20b-mxfp4 shape, mxfp4 experts on
        // the top-level defaults with affine overrides for everything else.
        args_from(serde_json::json!({
            "group_size": 32,
            "bits": 4,
            "mode": "mxfp4",
            "model.embed_tokens": { "group_size": 64, "bits": 4, "mode": "affine" },
            "model.layers.0.mlp.router": { "group_size": 64, "bits": 8 },
        }))
        .validate_quantization()
        .expect("the canonical gpt-oss quantization block must still validate");

        // A hostile pair scoped to a single prefix must be refused even though
        // the top-level defaults are perfectly valid.
        for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
            let err = args_from(serde_json::json!({
                "group_size": 32,
                "bits": 4,
                "mode": "mxfp4",
                "model.layers.0.mlp.experts.gate_up_proj": {
                    "group_size": group_size,
                    "bits": bits,
                },
            }))
            .validate_quantization()
            .expect_err("a hostile per-prefix override must be rejected");
            assert!(
                err.contains(field) && err.contains("gate_up_proj"),
                "the message must name both the field and the override, got: {err}"
            );

            // And the mirror case: valid overrides, hostile top-level defaults.
            // This is the shape that actually reaches the experts, because they
            // read `Quantization::defaults` rather than `quant_for`.
            let err = args_from(serde_json::json!({
                "group_size": group_size,
                "bits": bits,
                "model.embed_tokens": { "group_size": 64, "bits": 4, "mode": "affine" },
            }))
            .validate_quantization()
            .expect_err("hostile top-level defaults must be rejected");
            assert!(err.contains(field), "unhelpful error: {err}");
        }

        // A config with no `quantization` block at all is not gated.
        let args: ModelArgs = serde_json::from_value(serde_json::json!({
            "model_type": "gpt_oss",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 150000.0,
            "sliding_window": 8,
        }))
        .expect("test config must parse");
        args.validate_quantization()
            .expect("a bf16 checkpoint declares no quantization and must not be gated");
    }

    /// The declared `mode` has to be walked in both directions for the same
    /// reason the pair does. The live gap is the top-level one: `MLPBlock`
    /// reads `Quantization::defaults` for the experts rather than `quant_for`,
    /// so a hostile top-level mode reaches `gather_qmm` while every guarded
    /// sibling loader sees a good per-prefix mode, which is the shape every real
    /// gpt-oss checkpoint ships (issue #973).
    #[test]
    fn gpt_oss_config_validation_walks_every_declared_mode() {
        let args_from = |quantization: serde_json::Value| -> ModelArgs {
            serde_json::from_value(serde_json::json!({
                "model_type": "gpt_oss",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
                "rms_norm_eps": 1e-5,
                "rope_theta": 150000.0,
                "sliding_window": 8,
                "quantization": quantization,
            }))
            .expect("test config must parse")
        };

        // Positive control: the exact shape `models/gpt-oss-20b-mxfp4` ships, a
        // non-affine top-level mode with affine per-prefix overrides and a
        // mode-less override that must keep resolving to affine rather than
        // inheriting the top-level mxfp4.
        let canonical = args_from(serde_json::json!({
            "group_size": 32,
            "bits": 4,
            "mode": "mxfp4",
            "model.embed_tokens": { "group_size": 64, "bits": 4, "mode": "affine" },
            "model.layers.0.mlp.router": { "group_size": 64, "bits": 8 },
        }));
        canonical
            .validate_quantization()
            .expect("the canonical gpt-oss quantization block must still validate");
        assert_eq!(canonical.quant_for("model.layers.0.mlp.router").2, "affine");
        assert_eq!(canonical.quant_for("model.layers.0.mlp.experts").2, "mxfp4");

        // Every mode MLX accepts still validates at the top level.
        for mode in mlxcel_core::layers::SUPPORTED_QUANTIZATION_MODES {
            args_from(serde_json::json!({ "group_size": 32, "bits": 4, "mode": mode }))
                .validate_quantization()
                .unwrap_or_else(|e| panic!("MLX accepts {mode}, so the config must too: {e}"));
        }

        for mode in crate::models::switch_layers::HOSTILE_QUANT_MODES {
            // The live gap: a hostile top-level mode with a valid per-prefix
            // override, which is what actually reaches the experts.
            let err = args_from(serde_json::json!({
                "group_size": 32,
                "bits": 4,
                "mode": mode,
                "model.embed_tokens": { "group_size": 64, "bits": 4, "mode": "affine" },
            }))
            .validate_quantization()
            .expect_err("a hostile top-level mode must be rejected");
            assert!(err.contains("mode"), "unhelpful error: {err}");

            // And the mirror: a hostile per-prefix mode under valid top-level
            // defaults, which reaches the `UnifiedLinear` surface instead.
            let err = args_from(serde_json::json!({
                "group_size": 32,
                "bits": 4,
                "mode": "mxfp4",
                "model.layers.0.self_attn.q_proj": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": mode,
                },
            }))
            .validate_quantization()
            .expect_err("a hostile per-prefix mode must be rejected");
            assert!(
                err.contains("mode") && err.contains("q_proj"),
                "the message must name both the field and the override, got: {err}"
            );
        }
    }
}

#[cfg(test)]
#[path = "gpt_oss_tests.rs"]
pub(super) mod batching_tests;
