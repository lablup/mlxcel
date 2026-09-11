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

//! IQuest-Coder Loop (`iquestloopcoder`) building blocks.
//!
//! The architecture is a Llama-shaped dense decoder that runs its whole layer
//! stack **twice over the same tokens with the same weights** (`loop_num = 2`).
//! Only the second pass differs: each layer's attention output is a per-head
//! sigmoid-gated mix of a *global* branch, which attends the keys and values
//! the **first** pass produced for that layer, and a *local* branch, which
//! attends the second pass's own keys and values through a sliding window of
//! `loop_window_size` (64) tokens.
//!
//! ```text
//! pass 1, layer i:  h = block_i(h)                       -> stores (K1_i, V1_i)
//! pass 2, layer i:  q2, k2, v2 = proj_i(norm_i(h))       (same weights, same positions)
//!                   g          = sigmoid(q2 . gate_w_i + gate_b_i)   per head
//!                   out        = g * sdpa(q2, K1_i, V1_i) + (1 - g) * sdpa(q2, K2_i, V2_i)
//! ```
//!
//! Three properties decide correctness and each has a test in
//! `iquestloopcoder_tests.rs`:
//!
//! 1. The global branch reads the K/V that pass 1's `update_and_fetch`
//!    **returned in this same forward call**, so it already contains the tokens
//!    being processed now. Re-reading the pass-1 cache after pass 2 has run, or
//!    writing pass-2 K/V into the pass-1 cache, both produce fluent but wrong
//!    output (`pass2_uses_pass1_kv`).
//! 2. The gate is a function of the pass-2 query **after** RoPE, per head
//!    (`gate_is_per_head_sigmoid`).
//! 3. The local branch is windowed. Below `loop_window_size` tokens a windowed
//!    and an unwindowed implementation agree exactly, so only a prompt longer
//!    than the window can tell them apart (`window_limits_local_branch`).
//!
//! # Cache layout
//!
//! Every layer owns **two** caches: a dense [`KVCache`] for pass 1 and a
//! [`RotatingKVCache`] of `loop_window_size` entries for pass 2. They are
//! carried together in [`LayerCaches`], one per layer, and the pair is owned by
//! the model wrapper rather than by the generator's external `Vec<KVCache>`
//! slice; see [`crate::models::iquestloopcoder_model`] for what that implies
//! for the server.
//!
//! # Reference
//!
//! `modeling_iquestloopcoder.py`, shipped inside the checkpoint. This module
//! implements its `IQuestLoopCoderModel._forward_loop`, which is both the
//! training path and the prefill path, extended to an incremental cache. Two
//! deliberate divergences from that file's *cached decode* path
//! (`_forward_with_cache`) are recorded in
//! [`crate::models::iquestloopcoder_model::IQuestLoopCoderModel`].

use mlxcel_core::layers::{
    FusedQKVLinear, KVCache, RMSNorm, RotatingKVCache, UnifiedLinear, fused_add_rms_norm,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::models::llama3::MLP;
use crate::models::rope_utils::{RopeScalingKind, RopeScalingSpec};

/// The only `loop_num` this module implements.
///
/// Every published IQuest-Coder Loop checkpoint declares `2`. A checkpoint
/// declaring anything else would need a third set of caches and a second gate
/// application, so [`ModelArgs::validate`] refuses it at load rather than
/// silently running two passes over a model trained for more.
pub const SUPPORTED_LOOP_NUM: usize = 2;

fn default_rope_theta() -> f32 {
    500_000.0
}

fn default_loop_num() -> usize {
    SUPPORTED_LOOP_NUM
}

fn default_loop_window_size() -> usize {
    64
}

fn default_quantization_mode() -> String {
    "affine".to_string()
}

/// `eos_token_id` as either a scalar or a list.
///
/// IQuest-Coder Loop publishes a list (`[2, 75864, 75869]`) and all three must
/// stop generation, but the scalar spelling is accepted so a re-exported
/// config does not fail to load.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenIdField {
    Single(i32),
    Multiple(Vec<i32>),
}

impl TokenIdField {
    fn ids(&self) -> Vec<i32> {
        match self {
            Self::Single(id) => vec![*id],
            Self::Multiple(ids) => ids.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
    #[serde(default = "default_quantization_mode")]
    pub mode: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,

    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingSpec>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// How many times the layer stack is run over the same tokens.
    ///
    /// See [`SUPPORTED_LOOP_NUM`]: only `2` loads.
    #[serde(default = "default_loop_num")]
    pub loop_num: usize,

    /// Sliding-window length of the pass-2 local attention branch, in tokens.
    #[serde(default = "default_loop_window_size")]
    pub loop_window_size: usize,

    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,

    #[serde(skip)]
    pub checkpoint_label: Option<String>,
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
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

    pub fn quant_mode(&self) -> String {
        self.quantization
            .as_ref()
            .map(|q| q.mode.clone())
            .unwrap_or_else(default_quantization_mode)
    }

    /// EOS ids the checkpoint declares, or the published 40B-Loop set.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdField::ids)
            .filter(|ids| !ids.is_empty())
            .unwrap_or_else(|| vec![2, 75864, 75869])
    }

    pub fn set_checkpoint_label(&mut self, model_dir: &std::path::Path) {
        self.checkpoint_label = model_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty());
    }

    pub fn model_label(&self) -> &str {
        self.checkpoint_label.as_deref().unwrap_or(&self.model_type)
    }

    /// Reject a config this module cannot serve, before any weight is read.
    ///
    /// `loop_num != 2` is the substantive one (see [`SUPPORTED_LOOP_NUM`]). A
    /// non-positive `loop_window_size` is also refused because it would build a
    /// `RotatingKVCache` that stores nothing, leaving every local-branch
    /// attention row empty and softmaxing to NaN at the first token rather than
    /// at load.
    pub fn validate(&self) -> Result<(), String> {
        if self.loop_num != SUPPORTED_LOOP_NUM {
            return Err(format!(
                "iquestloopcoder: loop_num = {} is not supported (only {} is implemented); \
                 the checkpoint would need a third cache set per layer",
                self.loop_num, SUPPORTED_LOOP_NUM
            ));
        }
        if self.loop_window_size == 0 {
            return Err(
                "iquestloopcoder: loop_window_size must be positive; 0 leaves the pass-2 local \
                 attention branch with no keys"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn rope_scaling_kind(&self) -> RopeScalingKind {
        RopeScalingKind::resolve(
            self.rope_scaling.as_ref(),
            self.head_dim(),
            self.rope_theta,
            self.max_position_embeddings.map(|n| n as f32),
            self.model_label(),
        )
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

/// The per-layer pass-2 gate, `model.gate_projections.{i}`.
///
/// One `head_dim`-long vector and one scalar bias per attention head. The gate
/// value for head `h` at token `t` is `sigmoid(q2[h, t] . w[h] + b[h])`, a
/// number in `(0, 1)` that weights the global branch against the local one.
///
/// `weight` is stored pre-reshaped to `[1, H, D, 1]` and `bias` to `[1, H, 1,
/// 1]` so [`Self::forward`] is one broadcast `matmul` plus one broadcast `add`
/// against a `[B, H, L, D]` query, with no per-call reshaping. The leading `1`
/// makes the batch broadcast explicit rather than relying on rank promotion.
///
/// These tensors are never quantized in any published checkpoint: they arrive
/// as plain `[H, D]` / `[H]` arrays and are widened with the other dense
/// weights at the load boundary.
pub struct LoopGate {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
}

impl LoopGate {
    /// `q`: `[B, H, L, D]` post-RoPE pass-2 query. Returns `[B, H, L, 1]`.
    pub fn forward(&self, q: &MlxArray) -> UniquePtr<MlxArray> {
        let logits = mlxcel_core::matmul(q, &self.weight);
        let logits = mlxcel_core::add(&logits, &self.bias);
        mlxcel_core::sigmoid(&logits)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<Self, String> {
        let weight = get_weight_copy(weights, &format!("{prefix}.weight"))?;
        let shape = mlxcel_core::array_shape(&weight);
        if shape != vec![num_heads, head_dim] {
            return Err(format!(
                "{prefix}.weight has shape {shape:?}, expected [{num_heads}, {head_dim}]"
            ));
        }
        let bias = get_weight_copy(weights, &format!("{prefix}.bias"))?;
        let bias_shape = mlxcel_core::array_shape(&bias);
        if bias_shape != vec![num_heads] {
            return Err(format!(
                "{prefix}.bias has shape {bias_shape:?}, expected [{num_heads}]"
            ));
        }

        Ok(Self {
            weight: mlxcel_core::reshape(&weight, &[1, num_heads, head_dim, 1]),
            bias: mlxcel_core::reshape(&bias, &[1, num_heads, 1, 1]),
        })
    }

    /// Build a gate directly from arrays. Used by the tests.
    #[cfg(test)]
    pub(crate) fn from_arrays(
        weight: &MlxArray,
        bias: &MlxArray,
        num_heads: i32,
        head_dim: i32,
    ) -> Self {
        Self {
            weight: mlxcel_core::reshape(weight, &[1, num_heads, head_dim, 1]),
            bias: mlxcel_core::reshape(bias, &[1, num_heads, 1, 1]),
        }
    }
}

/// Attention split into the three stages the loop needs separately.
///
/// [`Attention::forward`] on the Llama path fuses projection, cache update and
/// SDPA into one call keyed on a single cache. Pass 2 has one query and *two*
/// key/value sets living in two different caches, so it needs the projection
/// (with RoPE) and the SDPA to be callable apart: [`Self::get_qkv`],
/// [`Self::attend`], [`Self::project_out`].
pub struct Attention {
    pub qkv_proj: FusedQKVLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub rope_freqs: Option<UniquePtr<MlxArray>>,
    pub rope_mscale: f32,
}

impl Attention {
    /// Project, split into heads and rotate.
    ///
    /// Returns `(q [B, H, L, D], k [B, Hkv, L, D], v [B, Hkv, L, D])` with `q`
    /// and `k` rotated at absolute positions `offset .. offset + L`. Both
    /// passes call this with the **same** `offset`, which is what makes pass 2
    /// see the positions pass 1 saw.
    pub fn get_qkv(
        &self,
        x: &MlxArray,
        offset: i32,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let (q, k, v) = self.qkv_proj.forward(x);
        let q = self.split_heads(&q, b, l, self.num_heads);
        let k = self.split_heads(&k, b, l, self.num_kv_heads);
        let v = self.split_heads(&v, b, l, self.num_kv_heads);
        let (q, k) = self.apply_rope(&q, &k, offset);
        (q, k, v)
    }

    fn split_heads(&self, x: &MlxArray, b: i32, l: i32, heads: i32) -> UniquePtr<MlxArray> {
        let reshaped = mlxcel_core::reshape(x, &[b, l, heads, self.head_dim]);
        mlxcel_core::transpose_axes(&reshaped, &[0, 2, 1, 3])
    }

    /// Rotate Q and K, honouring a `rope_scaling` frequency table when the
    /// checkpoint carries one. Mirrors the same decision in
    /// [`crate::models::llama3::Attention`]: MLX takes a base or a table,
    /// never both.
    fn apply_rope(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offset: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self.rope_freqs.as_ref() {
            Some(freqs) => {
                let (q_scaled, k_scaled);
                let (q, k): (&MlxArray, &MlxArray) = if self.rope_mscale != 1.0 {
                    q_scaled = mlxcel_core::multiply_scalar(q, self.rope_mscale);
                    k_scaled = mlxcel_core::multiply_scalar(k, self.rope_mscale);
                    (&q_scaled, &k_scaled)
                } else {
                    (q, k)
                };
                (
                    mlxcel_core::fast_rope_with_freqs(
                        q,
                        self.rope_dims,
                        false,
                        self.rope_scale,
                        offset,
                        freqs,
                    ),
                    mlxcel_core::fast_rope_with_freqs(
                        k,
                        self.rope_dims,
                        false,
                        self.rope_scale,
                        offset,
                        freqs,
                    ),
                )
            }
            None => (
                mlxcel_core::fast_rope(
                    q,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    self.rope_scale,
                    offset,
                ),
                mlxcel_core::fast_rope(
                    k,
                    self.rope_dims,
                    false,
                    self.rope_base,
                    self.rope_scale,
                    offset,
                ),
            ),
        }
    }

    /// Causal SDPA of `q` over the supplied K/V, returning `[B, H, L, D]`.
    ///
    /// `window` is `0` for the full-history branches and `loop_window_size` for
    /// the pass-2 local branch. [`mlxcel_core::causal_attention`] aligns
    /// causality bottom-right (`offset = k_len - q_len`), which is what both
    /// branches want: the K/V handed in already carry the history, and the
    /// query rows are the newest `L` positions. It is also the helper that
    /// already pairs correctly with [`RotatingKVCache`] over a window, keeping
    /// every prefill key and enforcing the window with a full-width mask rather
    /// than by slicing K/V (which would strand the earliest query rows).
    pub fn attend(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        window: i32,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::causal_attention(q, k, v, self.scale, 0.0, window)
    }

    /// Merge heads and apply `o_proj`. `attn`: `[B, H, L, D]` -> `[B, L, H*D]`.
    pub fn project_out(&self, attn: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(attn);
        let (b, l) = (shape[0], shape[2]);
        let merged = mlxcel_core::transpose_axes(attn, &[0, 2, 1, 3]);
        let merged = mlxcel_core::reshape(&merged, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&merged)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        rope: &RopeScalingKind,
    ) -> Result<Self, String> {
        let head_dim = args.head_dim() as i32;
        let num_heads = args.num_attention_heads as i32;
        let num_kv_heads = args.num_kv_heads() as i32;
        let mode = args.quant_mode();

        let rope = rope.duplicate();
        let rope_scale = rope.scale();
        let rope_mscale = rope.attn_scale();
        let rope_freqs = match rope {
            RopeScalingKind::Llama3 { freqs } | RopeScalingKind::Yarn { freqs, .. } => Some(freqs),
            _ => None,
        };

        let qkv_proj = FusedQKVLinear::from_weights_separate_with_mode(
            weights,
            prefix,
            args.group_size(),
            args.bits(),
            num_heads,
            num_kv_heads,
            head_dim,
            &mode,
        )?;
        let o_proj = UnifiedLinear::from_weights_with_mode(
            weights,
            &format!("{prefix}.o_proj"),
            args.group_size(),
            args.bits(),
            &mode,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: head_dim,
            rope_base: crate::models::rope_overrides::resolve_base(args.rope_theta),
            rope_scale,
            rope_freqs,
            rope_mscale,
        })
    }
}

/// The two caches one layer owns.
///
/// `pass1` grows with the sequence; `pass2` is bounded at `loop_window_size`.
/// Keeping them in one struct (rather than two parallel `Vec`s, or a doubled
/// flat list) makes it impossible to hand a layer the other pass's cache, which
/// is the mistake `pass2_uses_pass1_kv` guards against.
pub struct LayerCaches {
    pub pass1: KVCache,
    pub pass2: RotatingKVCache,
}

impl LayerCaches {
    pub fn new(loop_window_size: i32) -> Self {
        Self {
            pass1: KVCache::new(),
            pass2: RotatingKVCache::new(loop_window_size),
        }
    }
}

/// One decoder layer. The weights are shared by both passes; only the pass-2
/// gate lives outside the block (in the model, keyed by layer index) because
/// that is where the checkpoint puts it: `model.gate_projections.{i}`, not
/// `model.layers.{i}.*`.
pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
    /// Pass 1: plain causal attention over the full history.
    ///
    /// Returns the layer output and the K/V that `update_and_fetch`
    /// **returned**, that is past history plus the tokens just written. Pass 2
    /// attends exactly those arrays; see the module docs.
    pub fn forward_pass1(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        offset: i32,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let normed = self.input_layernorm.forward(x);
        let (q, k, v) = self.self_attn.get_qkv(&normed, offset);
        let (k_all, v_all) = cache.update_and_fetch(k, v);
        let attn = self.self_attn.attend(&q, &k_all, &v_all, 0);
        let attn_out = self.self_attn.project_out(&attn);
        let h = self.feed_forward(x, &attn_out);
        (h, k_all, v_all)
    }

    /// Pass 2: gated mix of the global (pass-1 K/V) and local (windowed pass-2
    /// K/V) branches.
    ///
    /// `offset` is the *pass-1* offset captured before any cache was touched
    /// this call, so the pass-2 query is rotated at the same absolute positions
    /// the pass-1 query was.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pass2(
        &self,
        x: &MlxArray,
        pass1_keys: &MlxArray,
        pass1_values: &MlxArray,
        gate: &LoopGate,
        cache: &mut RotatingKVCache,
        offset: i32,
        window: i32,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let (q2, k2, v2) = self.self_attn.get_qkv(&normed, offset);

        // Computed from the post-RoPE query, which is the tensor that attends.
        let g = gate.forward(&q2);

        let attn_global = self.self_attn.attend(&q2, pass1_keys, pass1_values, 0);
        let (k2_all, v2_all) = cache.update_and_fetch(k2, v2);
        let attn_local = self.self_attn.attend(&q2, &k2_all, &v2_all, window);

        let mixed = mix_gated(&g, &attn_global, &attn_local);
        let attn_out = self.self_attn.project_out(&mixed);
        self.feed_forward(x, &attn_out)
    }

    /// Residual join, post-attention norm, SwiGLU MLP, second residual join.
    /// Identical in both passes.
    ///
    /// `pub(crate)` so `iquestloopcoder_tests` can build a reference layer
    /// output to compare [`Self::forward_pass2`] against. The differential tests
    /// have to call the real `forward_pass2`, so their reference side needs the
    /// same tail.
    pub(crate) fn feed_forward(&self, x: &MlxArray, attn_out: &MlxArray) -> UniquePtr<MlxArray> {
        let (normed, h) = fused_add_rms_norm(&self.post_attention_layernorm, attn_out, x);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        rope: &RopeScalingKind,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let mode = args.quant_mode();

        let self_attn =
            Attention::from_weights(weights, args, &format!("{prefix}.self_attn"), rope)?;

        let mlp_prefix = format!("{prefix}.mlp");
        let mlp = MLP {
            gate_proj: UnifiedLinear::from_weights_with_mode(
                weights,
                &format!("{mlp_prefix}.gate_proj"),
                args.group_size(),
                args.bits(),
                &mode,
            )?,
            up_proj: UnifiedLinear::from_weights_with_mode(
                weights,
                &format!("{mlp_prefix}.up_proj"),
                args.group_size(),
                args.bits(),
                &mode,
            )?,
            down_proj: UnifiedLinear::from_weights_with_mode(
                weights,
                &format!("{mlp_prefix}.down_proj"),
                args.group_size(),
                args.bits(),
                &mode,
            )?,
        };

        let input_layernorm = RMSNorm::new(
            get_weight_copy(weights, &format!("{prefix}.input_layernorm.weight"))?,
            args.rms_norm_eps,
        );
        let post_attention_layernorm = RMSNorm::new(
            get_weight_copy(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            args.rms_norm_eps,
        );

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

/// `g * global + (1 - g) * local`, with `1` materialized in the gate's own
/// dtype so the mix stays in the model's activation precision instead of being
/// promoted to f32 by an f32 literal.
pub(crate) fn mix_gated(g: &MlxArray, global: &MlxArray, local: &MlxArray) -> UniquePtr<MlxArray> {
    let one = mlxcel_core::full_f32(&[1, 1, 1, 1], 1.0, mlxcel_core::array_dtype(g));
    let inv_g = mlxcel_core::subtract(&one, g);
    mlxcel_core::add(
        &mlxcel_core::multiply(g, global),
        &mlxcel_core::multiply(&inv_g, local),
    )
}
