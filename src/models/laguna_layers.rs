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

//! Laguna decoder building blocks: the mixed full/sliding cache, the gated
//! QK-norm attention with per-layer head counts and partial YaRN rotation,
//! the sigmoid / sqrt-softplus routed MoE with correction bias and shared
//! expert, the dense SwiGLU MLP, and the pre-norm decoder layer.
//!
//! The model shell that assembles these lives in [`crate::models::laguna`].

use crate::models::laguna::{LayerRope, ModelArgs, QuantSpec, RouterScoreFunc};
use crate::models::switch_layers::{SwitchGLU, fused_moe_enabled, moe_weighted_sum};
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_causal_mask_with_window_full};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-layer cache: dense `KVCache` on full-attention layers, rotating
/// window cache on sliding layers.
pub enum LagunaCache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl LagunaCache {
    /// Number of positions the cache has consumed so far.
    pub fn offset(&self) -> i32 {
        match self {
            Self::Standard(c) => c.offset,
            Self::Rotating(c) => c.offset,
        }
    }

    pub fn is_rotating(&self) -> bool {
        matches!(self, Self::Rotating(_))
    }

    pub fn update_and_fetch(
        &mut self,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Self::Standard(c) => c.update_and_fetch(k, v),
            Self::Rotating(c) => c.update_and_fetch(k, v),
        }
    }

    /// Rewind the last `n` positions. Returns the number actually trimmed.
    ///
    /// Used by: speculative rollback (issue #1351).
    pub fn trim(&mut self, n: i32) -> i32 {
        match self {
            Self::Standard(c) => c.trim(n),
            Self::Rotating(c) => c.trim(n),
        }
    }
}

/// How the attention output gate is applied (`gating` in `config.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateMode {
    /// No `g_proj`; the attention output is used as is.
    None,
    /// `g_proj` emits one logit per query head (`gating: true` or
    /// `"per-head"`).
    PerHead,
    /// `g_proj` emits one logit per output feature (any other truthy string).
    PerElement,
}

impl GateMode {
    pub(crate) fn from_config(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Bool(true) => Self::PerHead,
            serde_json::Value::String(s) if s == "per-head" => Self::PerHead,
            serde_json::Value::String(s) if !s.is_empty() => Self::PerElement,
            _ => Self::None,
        }
    }
}

/// Attention with QK-RMSNorm, a per-layer query-head count, partial-rotary
/// YaRN on full layers, plain RoPE on sliding layers, an optional per-head
/// attention sink, and a softplus output gate.
pub struct Attention {
    pub q_proj: UnifiedLinear,
    pub k_proj: UnifiedLinear,
    pub v_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub g_proj: Option<UnifiedLinear>,
    pub gate_mode: GateMode,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub is_sliding: bool,
    pub window_size: i32,
    /// Plain-RoPE base, used when `rope_freqs` is `None`.
    pub rope_base: f32,
    /// Number of leading head dims that are rotated (`head_dim *
    /// partial_rotary_factor`); the remainder passes through untouched.
    pub rope_dims: i32,
    /// YaRN frequencies for full-attention layers (`None` = plain RoPE).
    pub rope_freqs: Option<UniquePtr<MlxArray>>,
    pub rope_mscale: f32,
    /// `[head_dim]` f32 multiplier: `mscale` on the rotated dims and 1 on the
    /// pass-through dims. `None` when `mscale == 1`.
    mscale_vec: Option<UniquePtr<MlxArray>>,
    /// Per-head attention sink logits (`self_attn.sink`, sliding layers with
    /// `swa_attention_sink_enabled`).
    pub sinks: Option<UniquePtr<MlxArray>>,
}

impl Attention {
    /// Attention over `x` (`[B, L, H]`) against `cache`.
    ///
    /// A multi-token append builds its own additive mask from the key count
    /// the cache actually returns: a plain causal band on full layers, a
    /// causal band clipped to `sliding_window` on sliding layers. Deriving
    /// the key axis from the returned tensor rather than from the cache
    /// offset keeps the mask aligned with both the plain rotating cache
    /// (which exposes `min(offset, window - 1)` prior keys) and the buffered
    /// variant speculative rollback arms (which exposes up to `window +
    /// buffer` prior keys). A single-token decode passes no mask.
    pub fn forward(&self, x: &MlxArray, cache: &mut LagunaCache) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // QK-RMSNorm over head_dim, before transpose and RoPE.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset();
        let (q, k) = self.apply_rope(q, k, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let mask = if l > 1 {
            let k_len = mlxcel_core::array_shape(&cache_k)[2];
            let prior = (k_len - l).max(0);
            Some(if self.is_sliding {
                create_causal_mask_with_window_full(l, prior, Some(self.window_size))
            } else {
                create_causal_mask(l, prior)
            })
        } else {
            None
        };
        let mask_ptr = mask
            .as_ref()
            .map(|m| m.as_ref().unwrap() as *const _)
            .unwrap_or(std::ptr::null());
        let attn_out = if let Some(sinks) = &self.sinks {
            let sinks_ptr = sinks.as_ref().unwrap() as *const _;
            unsafe {
                mlxcel_core::fast_scaled_dot_product_attention_with_sinks(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, sinks_ptr,
                )
            }
        } else if l > 1 {
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

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        let gated = self.apply_gate(x, attn_out, b, l);
        self.o_proj.forward(&gated)
    }

    /// Rotate the first `rope_dims` features of `q` / `k`. YaRN layers scale
    /// those features by `mscale` first; the pass-through tail is untouched.
    fn apply_rope(
        &self,
        q: UniquePtr<MlxArray>,
        k: UniquePtr<MlxArray>,
        offset: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match &self.rope_freqs {
            Some(freqs) => {
                let (q, k) = match &self.mscale_vec {
                    Some(vec) => {
                        let vec = mlxcel_core::astype(vec, mlxcel_core::array_dtype(&q));
                        (
                            mlxcel_core::multiply(&q, &vec),
                            mlxcel_core::multiply(&k, &vec),
                        )
                    }
                    None => (q, k),
                };
                (
                    mlxcel_core::fast_rope_with_freqs(
                        &q,
                        self.rope_dims,
                        false,
                        1.0,
                        offset,
                        freqs,
                    ),
                    mlxcel_core::fast_rope_with_freqs(
                        &k,
                        self.rope_dims,
                        false,
                        1.0,
                        offset,
                        freqs,
                    ),
                )
            }
            None => (
                mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset),
                mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset),
            ),
        }
    }

    /// `o * softplus(g_proj(x))`, per head or per element. `x` is the
    /// attention input (the post-`input_layernorm` hidden state).
    fn apply_gate(
        &self,
        x: &MlxArray,
        o: UniquePtr<MlxArray>,
        b: i32,
        l: i32,
    ) -> UniquePtr<MlxArray> {
        let Some(g_proj) = &self.g_proj else {
            return o;
        };
        let gate = g_proj.forward(x);
        let gate =
            mlxcel_core::utils::softplus(&mlxcel_core::astype(&gate, mlxcel_core::dtype::FLOAT32));
        let gate = mlxcel_core::astype(&gate, mlxcel_core::array_dtype(&o));
        match self.gate_mode {
            GateMode::PerHead => {
                let o4 = mlxcel_core::reshape(&o, &[b, l, self.num_heads, self.head_dim]);
                let g4 = mlxcel_core::expand_dims(&gate, -1);
                let gated = mlxcel_core::multiply(&o4, &g4);
                mlxcel_core::reshape(&gated, &[b, l, self.num_heads * self.head_dim])
            }
            GateMode::PerElement => mlxcel_core::multiply(&o, &gate),
            GateMode::None => o,
        }
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        quant: &QuantSpec,
        prefix: &str,
        layer_idx: usize,
        rope: &LayerRope,
    ) -> Result<Self, String> {
        let linear = |leaf: &str| {
            UnifiedLinear::from_weights_with_mode(
                weights,
                &format!("{prefix}.{leaf}"),
                quant.group_size,
                quant.bits,
                quant.mode,
            )
        };
        let q_proj = linear("q_proj")?;
        let k_proj = linear("k_proj")?;
        let v_proj = linear("v_proj")?;
        let o_proj = linear("o_proj")?;

        let head_dim = args.head_dim() as i32;
        let num_heads = args.num_heads_for_layer(layer_idx)? as i32;
        let num_kv_heads = args.num_key_value_heads as i32;
        if num_kv_heads <= 0 || num_heads % num_kv_heads != 0 {
            return Err(format!(
                "{prefix}: {num_heads} query heads do not form whole GQA groups over \
                 {num_kv_heads} key/value heads"
            ));
        }

        let gate_mode = GateMode::from_config(&args.gating);
        let g_proj = match gate_mode {
            GateMode::None => None,
            GateMode::PerHead | GateMode::PerElement => {
                let g_out = weights
                    .get(&format!("{prefix}.g_proj.weight"))
                    .map(|w| mlxcel_core::array_shape(w)[0])
                    .ok_or_else(|| format!("Weight not found: {prefix}.g_proj.weight"))?;
                let expected = if gate_mode == GateMode::PerHead {
                    num_heads
                } else {
                    num_heads * head_dim
                };
                if g_out != expected {
                    return Err(format!(
                        "{prefix}.g_proj emits {g_out} gate logits but `gating` = {:?} needs \
                         {expected} for {num_heads} heads of {head_dim}",
                        args.gating
                    ));
                }
                Some(linear("g_proj")?)
            }
        };

        let q_norm_weight = get_weight_copy(weights, &format!("{prefix}.q_norm.weight"))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{prefix}.k_norm.weight"))?;

        let is_sliding = args.is_sliding(layer_idx);
        let sinks = if is_sliding && args.swa_attention_sink_enabled {
            Some(get_weight_copy(weights, &format!("{prefix}.sink"))?)
        } else {
            None
        };

        let rope_dims = rope.rotated_dims as i32;
        let (rope_freqs, rope_mscale) = match &rope.yarn {
            Some(y) => (Some(mlxcel_core::copy(&y.freqs)), y.mscale),
            None => (None, 1.0),
        };
        let mscale_vec = if (rope_mscale - 1.0).abs() > 1e-6 {
            let mut vec = vec![1.0f32; head_dim as usize];
            vec[..rope.rotated_dims].fill(rope_mscale);
            Some(mlxcel_core::from_slice_f32(&vec, &[head_dim]))
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            g_proj,
            gate_mode,
            q_norm: RMSNorm::new(q_norm_weight, args.rms_norm_eps),
            k_norm: RMSNorm::new(k_norm_weight, args.rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            is_sliding,
            window_size: if is_sliding {
                args.sliding_window as i32
            } else {
                0
            },
            rope_base: rope.base,
            rope_dims,
            rope_freqs,
            rope_mscale,
            mscale_vec,
            sinks,
        })
    }
}

/// Router scores from raw (already soft-capped) f32 logits.
///
/// Used by: `SparseMoeBlock::forward`, `laguna_tests`
pub(crate) fn router_scores(logits: &MlxArray, score_func: RouterScoreFunc) -> UniquePtr<MlxArray> {
    match score_func {
        RouterScoreFunc::Sigmoid => mlxcel_core::sigmoid(logits),
        RouterScoreFunc::SqrtSoftplus => mlxcel_core::sqrt(&mlxcel_core::utils::softplus(logits)),
        RouterScoreFunc::Softmax => mlxcel_core::softmax_precise(logits, -1),
    }
}

/// Top-k expert selection: the choice is made on `scores + bias`, the
/// returned weights come from the bias-free `scores`, optionally renormalized
/// over the selected experts and scaled by `routed_scaling_factor`.
///
/// Returns `(indices [N, k], weights [N, k])`.
///
/// Used by: `SparseMoeBlock::forward`, `laguna_tests`
pub(crate) fn router_select(
    scores: &MlxArray,
    bias: Option<&MlxArray>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let selection = match bias {
        Some(b) => mlxcel_core::add(scores, b),
        None => mlxcel_core::copy(scores),
    };
    let shape = mlxcel_core::array_shape(&selection);
    let n_experts = shape[shape.len() - 1];
    // The expert count comes from the real score row and the pivot is clamped
    // into it, so a `num_experts_per_tok` that disagrees with the router width
    // cannot reach `argpartition` out of range. MLX throws on an out-of-range
    // `kth`, and that throw crosses the cxx bridge as an uncatchable abort on a
    // token rather than a load error. `ModelArgs::validate` and
    // `validate_router_geometry` reject the configs and checkpoints that get
    // here; this keeps the kernel call in range regardless of who built the
    // block. An unclamped `n_experts - k` also slices a *negative* start out of
    // `order` when `k > n_experts`, which silently returns fewer than k experts
    // instead of failing.
    let k = i32::try_from(num_experts_per_tok)
        .unwrap_or(i32::MAX)
        .clamp(1, n_experts.max(1));
    let kth = (n_experts - k).max(0);
    let order = mlxcel_core::argpartition(&selection, kth, -1);
    let order_shape = mlxcel_core::array_shape(&order);
    let indices = mlxcel_core::slice(&order, &[0, kth], &[order_shape[0], order_shape[1]]);

    let mut weights = mlxcel_core::take_along_axis(scores, &indices, -1);
    if norm_topk_prob {
        let denom = mlxcel_core::sum_axis(&weights, -1, true);
        weights = mlxcel_core::divide(&weights, &denom);
    }
    if (routed_scaling_factor - 1.0).abs() > 1e-6 {
        weights = mlxcel_core::multiply_scalar(&weights, routed_scaling_factor);
    }
    (indices, weights)
}

/// Sparse MoE block: `gate.proj` router, correction bias, routed `SwitchGLU`
/// experts, and an always-on shared SwiGLU expert.
pub struct SparseMoeBlock {
    pub gate: UnifiedLinear,
    /// `e_score_correction_bias`, kept f32 `[num_experts]`.
    pub e_score_correction_bias: Option<UniquePtr<MlxArray>>,
    pub switch_mlp: SwitchGLU,
    pub shared_expert: Option<MLP>,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub softcap: f32,
    pub score_func: RouterScoreFunc,
}

/// Reject an expert plane the router is able to index past.
///
/// [`router_select`] derives the expert count from the router's own score row,
/// so the gather axis of `gather_qmm` (and of the per-expert `.global_scale`
/// `take` behind it) is the stacked plane count while the index range is the
/// router width. MLX's gather adds the axis size to a negative index but does
/// not range-check a positive one, so a checkpoint whose `switch_mlp` planes
/// are shorter than the router turns an ordinary token into an out-of-bounds
/// read whose result reaches the logits. This is the hazard AFMoE guards in
/// `validate_stacked_experts`, and Laguna reaches it from `config.json` alone:
/// [`crate::models::laguna_sanitize`] stacks exactly `num_experts` planes, and
/// `num_experts` is a config key while the router width is a tensor shape.
///
/// Used by: `SparseMoeBlock::from_weights`
fn validate_router_geometry(
    weights: &WeightMap,
    prefix: &str,
    num_experts_per_tok: usize,
) -> Result<(), String> {
    let router_key = format!("{prefix}.gate.proj.weight");
    let router = weights
        .get(&router_key)
        .ok_or_else(|| format!("Weight not found: {router_key}"))?;
    let router_shape = mlxcel_core::array_shape(router);
    let router_width = router_shape.first().copied().unwrap_or(0);
    if router_shape.len() != 2 || router_width < 1 {
        return Err(format!(
            "unexpected {router_key} shape {router_shape:?}: expected a 2-D [num_experts, hidden] \
             router weight routing over at least one expert"
        ));
    }
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let key = format!("{prefix}.switch_mlp.{proj}.weight");
        // Only the already-stacked layout is checked here; a checkpoint that
        // still carries per-expert planes is stacked by `SwitchLinear` itself,
        // which runs its own shape reconciliation.
        let Some(plane) = weights.get(&key) else {
            continue;
        };
        let shape = mlxcel_core::array_shape(plane);
        if shape.len() != 3 {
            return Err(format!(
                "unexpected {key} shape {shape:?}: expected a 3-D [num_experts, out, in] stacked \
                 expert tensor"
            ));
        }
        if shape[0] < router_width {
            return Err(format!(
                "{key} carries {} expert planes but {router_key} routes over {router_width}. The \
                 router emits indices below {router_width} and the gather behind gather_qmm does \
                 not range-check a positive index, so the missing planes would be read out of \
                 bounds and the result would reach the logits.",
                shape[0]
            ));
        }
    }
    let bias_key = format!("{prefix}.gate.e_score_correction_bias");
    if let Some(bias) = weights.get(&bias_key) {
        let bias_shape = mlxcel_core::array_shape(bias);
        if bias_shape.len() != 1 || bias_shape[0] != router_width {
            return Err(format!(
                "unexpected {bias_key} shape {bias_shape:?}: expected [{router_width}]; it is \
                 added to a row of that many router scores, and MLX throws on a broadcast it \
                 cannot resolve"
            ));
        }
    }
    if num_experts_per_tok == 0 || num_experts_per_tok > router_width as usize {
        return Err(format!(
            "{prefix}: num_experts_per_tok ({num_experts_per_tok}) must be between 1 and the \
             {router_width} experts {router_key} routes over"
        ));
    }
    Ok(())
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];
        let x_flat = if orig_shape.len() > 2 {
            let n: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[n, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        let logits = self.gate.forward(&x_flat);
        let mut logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
        if self.softcap > 0.0 {
            let scaled = mlxcel_core::divide_scalar(&logits, self.softcap);
            logits = mlxcel_core::multiply_scalar(&mlxcel_core::tanh(&scaled), self.softcap);
        }
        let scores = router_scores(&logits, self.score_func);
        let (indices, weights) = router_select(
            &scores,
            self.e_score_correction_bias.as_deref(),
            self.num_experts_per_tok,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        );

        // Single-token decode tries the fused affine kernel; NVFP4 experts and
        // every other unsupported shape fall back to the gather path.
        let fused = if mlxcel_core::array_shape(&x_flat)[0] == 1 && fused_moe_enabled() {
            self.switch_mlp
                .forward_fused_kernel(&x_flat, &indices, &weights)
                .map(|out| mlxcel_core::reshape(&out, &[1, hidden_dim]))
        } else {
            None
        };
        let mut routed = match fused {
            Some(out) => out,
            None => {
                let expert_out = self.switch_mlp.forward(&x_flat, &indices);
                moe_weighted_sum(&expert_out, &weights, mlxcel_core::array_dtype(&x_flat))
            }
        };

        if let Some(shared) = &self.shared_expert {
            routed = mlxcel_core::add(&routed, &shared.forward(&x_flat));
        }

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&routed, &orig_shape)
        } else {
            routed
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        quant: &QuantSpec,
        prefix: &str,
    ) -> Result<Self, String> {
        validate_router_geometry(weights, prefix, args.num_experts_per_tok)?;
        let gate = UnifiedLinear::from_weights_with_mode(
            weights,
            &format!("{prefix}.gate.proj"),
            quant.group_size,
            quant.bits,
            quant.mode,
        )?;
        let e_score_correction_bias = weights
            .get(&format!("{prefix}.gate.e_score_correction_bias"))
            .map(|w| mlxcel_core::astype(w, mlxcel_core::dtype::FLOAT32));
        let switch_mlp = SwitchGLU::from_weights_with_mode(
            weights,
            &format!("{prefix}.switch_mlp"),
            quant.group_size,
            quant.bits,
            quant.mode,
        )?;
        let shared_prefix = format!("{prefix}.shared_expert");
        let shared_expert = if weights.contains_key(&format!("{shared_prefix}.gate_proj.weight")) {
            Some(MLP::from_weights(weights, quant, &shared_prefix)?)
        } else {
            None
        };
        Ok(Self {
            gate,
            e_score_correction_bias,
            switch_mlp,
            shared_expert,
            num_experts_per_tok: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
            routed_scaling_factor: args.moe_routed_scaling_factor,
            softcap: args.moe_router_logit_softcapping,
            score_func: args.router_score_func()?,
        })
    }
}

/// Dense SwiGLU MLP (layer 0, and the MoE shared expert).
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
        quant: &QuantSpec,
        prefix: &str,
    ) -> Result<Self, String> {
        let linear = |leaf: &str| {
            UnifiedLinear::from_weights_with_mode(
                weights,
                &format!("{prefix}.{leaf}"),
                quant.group_size,
                quant.bits,
                quant.mode,
            )
        };
        Ok(Self {
            gate_proj: linear("gate_proj")?,
            up_proj: linear("up_proj")?,
            down_proj: linear("down_proj")?,
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

/// Pre-norm residual decoder layer.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: MLPType,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub fn forward(&self, x: &MlxArray, cache: &mut LagunaCache) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache);
        let h = mlxcel_core::add(x, &attn_out);
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        quant: &QuantSpec,
        layer_idx: usize,
        rope: &LayerRope,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let self_attn = Attention::from_weights(
            weights,
            args,
            quant,
            &format!("{prefix}.self_attn"),
            layer_idx,
            rope,
        )?;
        let mlp = if args.is_moe_layer(layer_idx) {
            MLPType::MoE(SparseMoeBlock::from_weights(
                weights,
                args,
                quant,
                &format!("{prefix}.mlp"),
            )?)
        } else {
            MLPType::Dense(MLP::from_weights(weights, quant, &format!("{prefix}.mlp"))?)
        };
        let input_norm_weight =
            get_weight_copy(weights, &format!("{prefix}.input_layernorm.weight"))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: RMSNorm::new(input_norm_weight, args.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_norm_weight, args.rms_norm_eps),
        })
    }
}

pub(crate) fn get_weight_copy(
    weights: &WeightMap,
    name: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}
