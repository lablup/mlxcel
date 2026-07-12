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

//! Command MoE (Cohere2 MoE, `model_type: "cohere2_moe"`) text-only LLM.
//!
//! This is the Cohere2 parallel-residual backbone (interleaved sliding/global
//! attention, conditional RoPE, logit scaling, tied embeddings) with the dense
//! FFN replaced by a sparse mixture-of-experts block:
//!
//! - Router: `x @ gate^T` (gate stored `[num_experts, hidden]`, no bias),
//!   activated in f32 by sigmoid (default) or softmax, top-k routing that
//!   gathers the ACTIVATED values at the selected experts (not a fresh softmax
//!   over the k selected logits), optional `norm_topk_prob` renormalization with
//!   a `1e-12` denominator clamp.
//! - Always-on shared experts (`moe_num_shared_experts`): one dense SwiGLU MLP
//!   combined by `"average"` (default `(y + y_s) / 2`) or `"sum"` (`y + y_s`).
//! - Optional dense-FFN prefix of the first `first_k_dense_replace` layers.
//! - Optional RMSNorm mode when `rms_norm_eps` is present (else LayerNorm).
//!
//! Attention, prefill-mask machinery, and the parallel block are adapted from
//! `src/models/cohere2.rs`; the routed-MoE dispatch mirrors
//! `src/models/olmoe.rs::SparseMoeBlock::forward`.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{
    FusedQKVLinear, KVCache, LayerNorm, RMSNorm, UnifiedEmbedding, UnifiedLinear,
};
use mlxcel_core::utils::{create_causal_mask, create_sliding_window_prefill_mask_dense};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
/// Cohere2 MoE model configuration.
///
/// Every key with a documented default is a serde default so real sparse
/// checkpoints (which omit backbone defaults) parse. `head_dim` is independent
/// of `hidden_size / num_attention_heads`. Alias keys resolve at read time:
/// `num_shared_experts` overrides `moe_num_shared_experts`, `expert_selection_fn`
/// overrides `moe_gate_act`, `prefix_dense_intermediate_size` defaults to
/// `intermediate_size`.
#[derive(Debug, Clone, Deserialize)]
pub struct Cohere2MoeConfig {
    pub model_type: String,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// When present, all norms are RMSNorm with this eps instead of LayerNorm.
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,

    #[serde(default = "default_logit_scale")]
    pub logit_scale: f32,

    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub layer_norm_bias: bool,

    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    /// Optional per-layer `"sliding_attention"` / `"full_attention"` list;
    /// overrides the pattern when present.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_moe_gate_act")]
    pub moe_gate_act: String,
    /// Alias for `moe_gate_act`; overrides it when present.
    #[serde(default)]
    pub expert_selection_fn: Option<String>,

    #[serde(default = "default_moe_num_shared_experts")]
    pub moe_num_shared_experts: usize,
    /// Alias for `moe_num_shared_experts`; overrides it when present.
    #[serde(default)]
    pub num_shared_experts: Option<usize>,
    #[serde(default = "default_shared_combination")]
    pub shared_expert_combination_strategy: String,

    /// The first `k` layers use a dense FFN instead of the MoE block.
    #[serde(default)]
    pub first_k_dense_replace: usize,
    /// Dense-FFN width for the prefix layers; defaults to `intermediate_size`.
    #[serde(default)]
    pub prefix_dense_intermediate_size: Option<usize>,
    /// When 1 (the default), prefix dense (global-attention) layers get RoPE.
    #[serde(default = "default_prefix_dense_sliding_window_pattern")]
    pub prefix_dense_sliding_window_pattern: usize,

    #[serde(default)]
    pub bos_token_id: Option<i32>,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub pad_token_id: Option<i32>,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

/// `eos_token_id` may be a single int or a list of ints (reused shape from
/// `src/models/cohere2.rs`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(i32),
    Multiple(Vec<i32>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_head_dim() -> usize {
    128
}
fn default_num_hidden_layers() -> usize {
    36
}
fn default_intermediate_size() -> usize {
    1024
}
fn default_num_attention_heads() -> usize {
    64
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_rope_theta() -> f32 {
    50000.0
}
fn default_vocab_size() -> usize {
    256000
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_logit_scale() -> f32 {
    0.0625
}
fn default_sliding_window() -> usize {
    4096
}
fn default_sliding_window_pattern() -> usize {
    4
}
fn default_num_experts() -> usize {
    128
}
fn default_num_experts_per_tok() -> usize {
    8
}
fn default_true() -> bool {
    true
}
fn default_moe_gate_act() -> String {
    "sigmoid".to_string()
}
fn default_moe_num_shared_experts() -> usize {
    4
}
fn default_shared_combination() -> String {
    "average".to_string()
}
fn default_prefix_dense_sliding_window_pattern() -> usize {
    1
}

impl Cohere2MoeConfig {
    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// Effective number of always-on shared experts (`num_shared_experts` alias
    /// overrides `moe_num_shared_experts`). 0 disables the shared expert.
    pub fn shared_expert_count(&self) -> usize {
        self.num_shared_experts
            .unwrap_or(self.moe_num_shared_experts)
    }

    /// Dense-FFN width used by the prefix layers.
    pub fn prefix_dense_intermediate(&self) -> usize {
        self.prefix_dense_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    /// Router activation: sigmoid (default) unless the resolved gate act is
    /// `"softmax"`. `expert_selection_fn` overrides `moe_gate_act` when present.
    pub fn gate_is_sigmoid(&self) -> bool {
        let act = self
            .expert_selection_fn
            .as_deref()
            .unwrap_or(self.moe_gate_act.as_str());
        !act.eq_ignore_ascii_case("softmax")
    }

    /// Shared-expert combine averages (`(y + y_s) / 2`) unless the strategy is
    /// `"sum"` (`y + y_s`).
    pub fn shared_combine_average(&self) -> bool {
        !self
            .shared_expert_combination_strategy
            .eq_ignore_ascii_case("sum")
    }

    /// A dense-FFN prefix layer (`i < first_k_dense_replace`).
    pub fn is_dense_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.first_k_dense_replace
    }

    /// Layer classification, evaluated in order:
    /// 1. `i < first_k_dense_replace` => global (never sliding).
    /// 2. else if `layer_types` present => sliding iff `layer_types[i] ==
    ///    "sliding_attention"`.
    /// 3. else => sliding iff `(i + 1) % sliding_window_pattern != 0`.
    pub fn is_sliding_window_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.first_k_dense_replace {
            return false;
        }
        if let Some(ref types) = self.layer_types {
            return types
                .get(layer_idx)
                .map(|t| t == "sliding_attention")
                .unwrap_or(false);
        }
        !(layer_idx + 1).is_multiple_of(self.sliding_window_pattern)
    }

    /// RoPE is applied iff the layer is sliding, or the layer is a dense prefix
    /// layer and `prefix_dense_sliding_window_pattern == 1`. Non-prefix global
    /// layers use no positional encoding.
    pub fn layer_uses_rope(&self, layer_idx: usize) -> bool {
        self.is_sliding_window_layer(layer_idx)
            || (layer_idx < self.first_k_dense_replace
                && self.prefix_dense_sliding_window_pattern == 1)
    }
}

// Norm wrapper (RMSNorm when `rms_norm_eps` present, else LayerNorm).
enum Cohere2MoeNorm {
    Rms(RMSNorm),
    Layer(LayerNorm),
}

impl Cohere2MoeNorm {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Rms(n) => n.forward(x),
            Self::Layer(n) => n.forward(x),
        }
    }

    fn from_weights(
        weights: &WeightMap,
        args: &Cohere2MoeConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
        if let Some(eps) = args.rms_norm_eps {
            Ok(Self::Rms(RMSNorm::new(weight, eps)))
        } else {
            // Load the LayerNorm bias when the checkpoint carries it.
            let bias = weights
                .get(&format!("{}.bias", prefix))
                .map(|b| mlxcel_core::copy(b));
            Ok(Self::Layer(LayerNorm::new(
                weight,
                bias,
                args.layer_norm_eps,
            )))
        }
    }
}

// Dense SwiGLU MLP (prefix layers and shared experts; widths differ by prefix).
pub struct Cohere2MoeMLP {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl Cohere2MoeMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Cohere2MoeConfig,
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

// Cohere2 MoE attention (conditional RoPE + sliding/global interleave).
pub struct Cohere2MoeAttention {
    /// Fused QKV projection: Q, K, V weights concatenated along the output dim.
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    /// Sliding layer, or a prefix dense layer forcing RoPE. Global non-prefix
    /// layers set this false and use no positional encoding.
    pub use_rope: bool,
    /// Sliding-window width; 0 for global layers (drives K/V windowing only).
    pub window_size: i32,
}

impl Cohere2MoeAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let (q, k, v) = self.qkv_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Traditional (interleaved-pair) RoPE, applied only when the layer type
        // requires it: sliding layers, and forced on prefix dense layers.
        let (q, k) = if self.use_rope {
            let q_rope =
                mlxcel_core::fast_rope(&q, self.rope_dims, true, self.rope_base, 1.0, offset);
            let k_rope =
                mlxcel_core::fast_rope(&k, self.rope_dims, true, self.rope_base, 1.0, offset);
            (q_rope, k_rope)
        } else {
            (q, k)
        };

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let attn_out = if l > 1 {
            // Prefill: slice K/V to the mask's key axis. A full mask keeps every
            // key, a clamped sliding mask drops the oldest (mirrors
            // `src/models/cohere2.rs`).
            let k_shape = mlxcel_core::array_shape(&cache_k);
            let k_len = k_shape[2];
            let mask_klen = mask
                .map(|m| *mlxcel_core::array_shape(m).last().unwrap_or(&k_len))
                .unwrap_or(k_len);
            let (k_used, v_used) = if self.window_size > 0 && k_len > mask_klen {
                let v_shape = mlxcel_core::array_shape(&cache_v);
                let start = k_len - mask_klen;
                (
                    Some(mlxcel_core::slice(
                        &cache_k,
                        &[0, 0, start, 0],
                        &[k_shape[0], k_shape[1], k_len, k_shape[3]],
                    )),
                    Some(mlxcel_core::slice(
                        &cache_v,
                        &[0, 0, start, 0],
                        &[v_shape[0], v_shape[1], k_len, v_shape[3]],
                    )),
                )
            } else {
                (None, None)
            };
            let k_ref: &MlxArray = k_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_k.as_ref().unwrap());
            let v_ref: &MlxArray = v_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_v.as_ref().unwrap());

            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    k_ref,
                    v_ref,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.window_size,
                )
            }
        } else {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, self.window_size)
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Cohere2MoeConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let use_sliding_window = args.is_sliding_window_layer(layer_idx);
        let use_rope = args.layer_uses_rope(layer_idx);

        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_key_value_heads as i32;
        let head_dim = args.head_dim as i32;

        // FusedQKVLinear auto-detects q/k/v linear bias, so `attention_bias`
        // needs no explicit plumbing here.
        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights,
            prefix,
            group_size,
            bits,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_dims: head_dim,
            rope_base: args.rope_theta,
            use_rope,
            window_size: if use_sliding_window {
                args.sliding_window as i32
            } else {
                0
            },
        })
    }
}

// MoE router scoring.
/// Compute the top-k expert indices and their router scores from raw router
/// logits.
///
/// Expert selection is by argpartition on the raw logits: sigmoid and softmax
/// are strictly monotonic, so the top-k set is identical to argpartition on the
/// activated values. The scores are the ACTIVATED values gathered at the
/// selected experts (`take_along_axis`), NOT a fresh softmax over the k selected
/// logits. Activation runs in f32. When `norm_topk_prob`, scores are divided by
/// `max(sum, 1e-12)`; the clamp matters for sigmoid gating, where all k scores
/// can be near zero. Returns `(topk_indices, scores)`, both `[n_tokens, k]` and
/// aligned so `scores[t, j]` is the weight for expert `topk_indices[t, j]`.
fn router_topk_scores(
    logits: &MlxArray,
    k: i32,
    gate_is_sigmoid: bool,
    norm_topk_prob: bool,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let n_experts = mlxcel_core::array_shape(logits)[1];
    let kth = n_experts - k;

    // Top-k selection: indices[..., kth:] after argpartition on the raw logits.
    let indices = mlxcel_core::argpartition(logits, kth, -1);
    let indices_shape = mlxcel_core::array_shape(&indices);
    let topk_indices =
        mlxcel_core::slice(&indices, &[0, kth], &[indices_shape[0], indices_shape[1]]);

    // Activate over ALL experts in f32, then gather at the selected experts.
    let logits_f32 = mlxcel_core::astype(logits, mlxcel_core::dtype::FLOAT32);
    let p = if gate_is_sigmoid {
        mlxcel_core::sigmoid(&logits_f32)
    } else {
        mlxcel_core::softmax_precise(&logits_f32, -1)
    };
    let mut scores = mlxcel_core::take_along_axis(&p, &topk_indices, -1);

    if norm_topk_prob {
        let sum = mlxcel_core::sum_axis(&scores, -1, true);
        // Clamp the denominator to 1e-12 (matters for sigmoid where all k can
        // be near zero).
        let clamp = mlxcel_core::full_f32(&[1], 1e-12, mlxcel_core::array_dtype(&sum));
        let denom = mlxcel_core::maximum(&sum, &clamp);
        scores = mlxcel_core::divide(&scores, &denom);
    }

    (topk_indices, scores)
}

/// Combine the routed-expert output with the shared-expert output. `average`
/// halves the sum (`(y + y_s) / 2`); otherwise it is a plain sum (`y + y_s`).
fn combine_shared_expert(
    routed: &MlxArray,
    shared: &MlxArray,
    average: bool,
) -> UniquePtr<MlxArray> {
    let sum = mlxcel_core::add(routed, shared);
    if average {
        mlxcel_core::multiply_scalar(&sum, 0.5)
    } else {
        sum
    }
}

// Sparse MoE block (router + routed experts + optional shared experts).
pub struct Cohere2MoeSparseBlock {
    pub router: UnifiedLinear,
    pub experts: crate::models::switch_layers::SwitchGLU,
    pub shared_expert: Option<Cohere2MoeMLP>,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub gate_is_sigmoid: bool,
    pub shared_combine_average: bool,
}

impl Cohere2MoeSparseBlock {
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

        let logits = self.router.forward(&x_flat);
        let k = self.num_experts_per_tok as i32;
        let (topk_indices, scores) =
            router_topk_scores(&logits, k, self.gate_is_sigmoid, self.norm_topk_prob);

        // Routed experts. Fused single-token decode kernel on by default;
        // otherwise (or when the kernel declines the config) the proven
        // SwitchGLU + moe_weighted_sum path.
        let routed = {
            let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1
                && crate::models::switch_layers::fused_moe_enabled()
            {
                self.experts
                    .forward_fused_kernel(&x_flat, &topk_indices, &scores)
                    .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
            } else {
                None
            };
            match fused {
                Some(out) => out,
                None => {
                    let expert_out = self.experts.forward(&x_flat, &topk_indices);
                    crate::models::switch_layers::moe_weighted_sum(
                        &expert_out,
                        &scores,
                        mlxcel_core::array_dtype(&x_flat),
                    )
                }
            }
        };

        // Always-on shared experts on the same flattened input.
        let combined = if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(&x_flat);
            combine_shared_expert(&routed, &shared_out, self.shared_combine_average)
        } else {
            routed
        };

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&combined, &orig_shape)
        } else {
            combined
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Cohere2MoeConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        // Router gate: `[num_experts, hidden]`, no bias (leaf `gate`, distinct
        // from `gate_proj`). UnifiedLinear detects quantization per tensor, so
        // an unquantized gate in an otherwise-quantized export still loads.
        let router =
            UnifiedLinear::from_weights(weights, &format!("{}.gate", prefix), group_size, bits)?;

        // `SwitchLinear::from_weights_with_mode` resolves BOTH the pre-stacked
        // `...switch_mlp.{proj}.weight` layout and the per-expert
        // `...experts.{e}.{proj}.weight` layout (stacking the latter), so no
        // redundant expert-stacking pass is needed.
        let experts = crate::models::switch_layers::SwitchGLU::from_weights(
            weights,
            &format!("{}.switch_mlp", prefix),
            group_size,
            bits,
        )?;

        let shared_expert = if args.shared_expert_count() > 0 {
            Some(Cohere2MoeMLP::from_weights(
                weights,
                args,
                &format!("{}.shared_experts", prefix),
            )?)
        } else {
            None
        };

        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts_per_tok: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
            gate_is_sigmoid: args.gate_is_sigmoid(),
            shared_combine_average: args.shared_combine_average(),
        })
    }
}

// Per-layer FFN: dense (prefix) or sparse MoE.
enum Cohere2MoeFFN {
    Dense(Cohere2MoeMLP),
    Sparse(Cohere2MoeSparseBlock),
}

impl Cohere2MoeFFN {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(block) => block.forward(x),
        }
    }
}

// Parallel-residual decoder layer (single norm; attention and FFN read it).
pub struct Cohere2MoeDecoderLayer {
    self_attn: Cohere2MoeAttention,
    ffn: Cohere2MoeFFN,
    input_layernorm: Cohere2MoeNorm,
}

impl Cohere2MoeDecoderLayer {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // h = norm(x); out = attn(h) + ffn(h) + x.
        let h = self.input_layernorm.forward(x);
        let attn_h = self.self_attn.forward(&h, cache, mask);
        let ff_h = self.ffn.forward(&h);
        let sum = mlxcel_core::add(&attn_h, &ff_h);
        mlxcel_core::add(&sum, x)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &Cohere2MoeConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn = Cohere2MoeAttention::from_weights(
            weights,
            args,
            layer_idx,
            &format!("{}.self_attn", prefix),
        )?;

        let ffn = if args.is_dense_layer(layer_idx) {
            Cohere2MoeFFN::Dense(Cohere2MoeMLP::from_weights(
                weights,
                args,
                &format!("{}.mlp", prefix),
            )?)
        } else {
            Cohere2MoeFFN::Sparse(Cohere2MoeSparseBlock::from_weights(
                weights, args, layer_idx,
            )?)
        };

        let input_layernorm =
            Cohere2MoeNorm::from_weights(weights, args, &format!("{}.input_layernorm", prefix))?;

        Ok(Self {
            self_attn,
            ffn,
            input_layernorm,
        })
    }
}

// Cohere2 MoE model.
pub struct Cohere2MoeModel {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<Cohere2MoeDecoderLayer>,
    norm: Cohere2MoeNorm,
    pub lm_head: UnifiedLinear,
    pub logit_scale: f32,
    pub config: Cohere2MoeConfig,
    // First sliding and first global attention layer indices, used to size the
    // two prefill masks from the matching cache's live window.
    swa_idx: usize,
    ga_idx: usize,
}

impl Cohere2MoeModel {
    fn run_layers(
        &self,
        mut h: UniquePtr<MlxArray>,
        caches: &mut [KVCache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(&h);
        let l = shape[1] as usize;

        // Size the prefill masks from each cache's live window (`live_len()`),
        // not the monotonic `offset`, so `--max-kv-size` trimming stays
        // consistent with the returned K/V (mirrors `src/models/cohere2.rs`).
        let (full_mask, sliding_mask) = if l > 1 {
            let ga_live_len = caches[self.ga_idx].live_len();
            let swa_live_len = caches[self.swa_idx].live_len();
            let full = Some(create_causal_mask(l as i32, ga_live_len));
            let sliding = Some(create_sliding_window_prefill_mask_dense(
                l as i32,
                swa_live_len,
                self.config.sliding_window as i32,
            ));
            (full, sliding)
        } else {
            (None, None)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.config.is_sliding_window_layer(i) {
                sliding_mask.as_ref().map(|m| m.as_ref().unwrap())
            } else {
                full_mask.as_ref().map(|m| m.as_ref().unwrap())
            };
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.norm.forward(&h);
        let logits = self.lm_head.forward(&h);
        let scale_arr =
            mlxcel_core::full_f32(&[1], self.logit_scale, mlxcel_core::array_dtype(&logits));
        mlxcel_core::multiply(&logits, &scale_arr)
    }

    /// Forward pass over token ids.
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        self.run_layers(h, caches)
    }

    /// Get token embeddings (for VLM merge / speculative paths).
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward with pre-computed embeddings.
    pub fn forward_with_embeddings_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };
        self.run_layers(h, caches)
    }

    /// Load model from a directory containing safetensors files and config.json.
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, Cohere2MoeConfig), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: Cohere2MoeConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&weights, &config)?;

        Ok((model, config))
    }

    /// Create model from loaded weights.
    pub fn from_weights(weights: &WeightMap, args: &Cohere2MoeConfig) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // First sliding and first global layer indices (from the new
        // classification). Both fall back to 0 when absent; the mask keyed to a
        // missing layer type is simply never consumed.
        let swa_idx = (0..args.num_hidden_layers)
            .find(|&i| args.is_sliding_window_layer(i))
            .unwrap_or(0);
        let ga_idx = (0..args.num_hidden_layers)
            .find(|&i| !args.is_sliding_window_layer(i))
            .unwrap_or(0);

        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(Cohere2MoeDecoderLayer::from_weights(weights, args, i)?);
        }

        let norm = Cohere2MoeNorm::from_weights(weights, args, "model.norm")?;

        // Tied embeddings: build the LM head from `model.embed_tokens`. Keep a
        // load-`lm_head`-if-present fallback for robustness.
        let lm_head = if weights.contains_key("lm_head.weight") {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            logit_scale: args.logit_scale,
            config: args.clone(),
            swa_idx,
            ga_idx,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for Cohere2MoeModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        match &self.config.eos_token_id {
            Some(EosTokenId::Single(id)) => vec![*id],
            Some(EosTokenId::Multiple(ids)) => ids.clone(),
            None => vec![255001], // Default Cohere EOS token.
        }
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

#[cfg(test)]
#[path = "cohere2_moe_tests.rs"]
mod tests;
