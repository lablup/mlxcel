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

//! ERNIE-4.5 MoE VL language model (modality-split MoE + interleaved 3D MRoPE).
//!
//! The text decoder of ERNIE-4.5-VL differs from the in-tree
//! [`crate::models::ernie4_5_moe`] backbone in two load-bearing ways:
//!
//! - **Modality-split expert banks.** Each MoE layer holds two independent
//!   routed banks: a text bank (`gate`, `switch_mlp`, `e_score_correction_bias`)
//!   and a multimodal bank (`gate_1`, `switch_mlp_1`,
//!   `e_score_correction_bias_1`). Routing per bank is softmax over the router
//!   logits in f32, top-k selected over `probs + bias` (the correction bias
//!   influences *selection only*), and mixing weights are the uncorrected probs
//!   gathered at the selected experts, normalized by
//!   `max(sum, moe_norm_min)`. Per token the text-bank output is used where the
//!   token type is text and the multimodal-bank output where the token is an
//!   image placeholder; text-only prompts and all decode steps take the text
//!   bank alone. A fused shared-experts SwiGLU is added for every token.
//! - **Interleaved 3D MRoPE.** Positions are 3D `[T, H, W]`. For frequency-pair
//!   index `i` in `0..head_dim/2` with `hw = mrope_section[0] + mrope_section[1]`:
//!   the angle uses `p_H` when `i < hw` and `i` is even, `p_W` when `i < hw` and
//!   `i` is odd, and `p_T` when `i >= hw`; `f_i = rope_theta^(-2i/head_dim)`.
//!   cos/sin are repeated at adjacent slots and applied by rotating adjacent
//!   (even, odd) element pairs in f32. When all three axes carry the same
//!   scalar position this degenerates exactly to traditional interleaved RoPE
//!   (`fast_rope` with `traditional = true`), which is unit-tested below.
//!
//! Reuses [`crate::models::qwen_mrope_state::MRopeState`] for per-sequence
//! MRoPE delta tracking and the [`crate::models::ernie4_5_moe`] SwitchGLU
//! expert-bank machinery (the checkpoint ships pre-fused stacked expert
//! tensors).
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/language.py>.

use crate::models::ernie4_5_moe::{DenseMLP, SwitchGLU, SwitchLinear};
use crate::models::qwen_mrope_state::MRopeState;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

// Config.

/// MoE fields arrive as an int on text-only exports and as a 2-list
/// `[text, multimodal]` on VL checkpoints.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum IntOrPair {
    Int(usize),
    List(Vec<usize>),
}

impl IntOrPair {
    pub fn first(&self) -> usize {
        match self {
            Self::Int(v) => *v,
            Self::List(v) => v.first().copied().unwrap_or(0),
        }
    }
    pub fn second(&self) -> Option<usize> {
        match self {
            Self::Int(_) => None,
            Self::List(v) => v.get(1).copied(),
        }
    }
    pub fn min(&self) -> usize {
        match self {
            Self::Int(v) => *v,
            Self::List(v) => v.iter().copied().min().unwrap_or(0),
        }
    }
    pub fn max(&self) -> usize {
        match self {
            Self::Int(v) => *v,
            Self::List(v) => v.iter().copied().max().unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ernie45MoeVlTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub moe_num_experts: Option<IntOrPair>,
    #[serde(default)]
    pub moe_intermediate_size: Option<IntOrPair>,
    #[serde(default = "default_moe_k")]
    pub moe_k: usize,
    #[serde(default)]
    pub moe_layer_start_index: Option<IntOrPair>,
    #[serde(default)]
    pub moe_layer_end_index: Option<IntOrPair>,
    #[serde(default = "default_moe_layer_interval")]
    pub moe_layer_interval: usize,
    #[serde(default)]
    pub moe_num_shared_experts: usize,
    #[serde(default = "default_moe_norm_min")]
    pub moe_norm_min: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub freq_allocation: Option<i32>,
    #[serde(default = "default_im_patch_id")]
    pub im_patch_id: i32,
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    500_000.0
}
fn default_moe_k() -> usize {
    6
}
fn default_moe_layer_interval() -> usize {
    1
}
fn default_moe_norm_min() -> f32 {
    1e-12
}
fn default_im_patch_id() -> i32 {
    100_295
}
fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

impl Ernie45MoeVlTextConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(0)
    }
    fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(0)
    }
    /// `[H, W, T]` frequency allocation; falls back to deriving from
    /// `freq_allocation` (T count) when `rope_scaling.mrope_section` is absent.
    fn mrope_section(&self) -> Vec<i32> {
        if let Some(rs) = &self.rope_scaling
            && rs.mrope_section.len() == 3
        {
            return rs.mrope_section.clone();
        }
        let half = (self.head_dim() / 2) as i32;
        let t = self.freq_allocation.unwrap_or(20);
        let hw_each = (half - t) / 2;
        vec![hw_each, hw_each, t]
    }
    /// Layer `l` is MoE iff `(l+1) % interval == 0` and
    /// `min(start) <= l <= max(end)` (end defaults to the last layer).
    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        let Some(experts) = &self.moe_num_experts else {
            return false;
        };
        if experts.first() == 0 {
            return false;
        }
        let start = self
            .moe_layer_start_index
            .as_ref()
            .map(|v| v.min())
            .unwrap_or(0);
        let end = self
            .moe_layer_end_index
            .as_ref()
            .map(|v| v.max())
            .unwrap_or(self.num_hidden_layers - 1);
        (layer_idx + 1).is_multiple_of(self.moe_layer_interval.max(1))
            && layer_idx >= start
            && layer_idx <= end
    }
}

// Interleaved 3D MRoPE.

/// Precomputed per-axis frequency tables. `inv_freq[i] = base^(-2i / head_dim)`;
/// the H table holds even `i < hw`, the W table odd `i < hw`, the T table
/// `i >= hw` (`hw = mrope_section[0] + mrope_section[1]`).
struct ErnieMRoPE {
    inv_freq_h: Vec<f32>,
    inv_freq_w: Vec<f32>,
    inv_freq_t: Vec<f32>,
    head_dim: i32,
}

impl ErnieMRoPE {
    fn new(head_dim: usize, base: f32, mrope_section: &[i32]) -> Self {
        let half = head_dim / 2;
        let hw = (mrope_section[0] + mrope_section[1]) as usize;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            inv_freq.push(base.powf(-((2 * i) as f32) / head_dim as f32));
        }
        let inv_freq_h: Vec<f32> = inv_freq[..hw].iter().step_by(2).copied().collect();
        let inv_freq_w: Vec<f32> = inv_freq[..hw].iter().skip(1).step_by(2).copied().collect();
        let inv_freq_t: Vec<f32> = inv_freq[hw..].to_vec();
        Self {
            inv_freq_h,
            inv_freq_w,
            inv_freq_t,
            head_dim: head_dim as i32,
        }
    }

    /// `position_ids`: `[3, batch, seq]` int (axis 0 ordered T, H, W).
    /// Returns `(cos, sin)`, each `[batch, seq, head_dim]` in f32 with each
    /// angle repeated at adjacent slots (`[a0, a0, a1, a1, ...]`).
    fn forward(&self, position_ids: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(position_ids);
        let (batch, seq) = (shape[1], shape[2]);

        let axis_pos = |idx: i32| {
            let sl = mlxcel_core::slice(position_ids, &[idx, 0, 0], &[idx + 1, batch, seq]);
            let sl = mlxcel_core::reshape(&sl, &[batch, seq, 1]);
            mlxcel_core::astype(&sl, mlxcel_core::dtype::FLOAT32)
        };
        let p_t = axis_pos(0);
        let p_h = axis_pos(1);
        let p_w = axis_pos(2);

        let table = |v: &[f32]| {
            let arr = mlxcel_core::from_slice_f32(v, &[1, 1, v.len() as i32]);
            mlxcel_core::astype(&arr, mlxcel_core::dtype::FLOAT32)
        };
        let ang_h = mlxcel_core::multiply(&p_h, &table(&self.inv_freq_h)); // [b, s, hw/2]
        let ang_w = mlxcel_core::multiply(&p_w, &table(&self.inv_freq_w)); // [b, s, hw/2]
        let ang_t = mlxcel_core::multiply(&p_t, &table(&self.inv_freq_t)); // [b, s, t]

        // Interleave H and W back to per-frequency order: [h0, w0, h1, w1, ...].
        let n_hw = self.inv_freq_h.len() as i32;
        let h_e = mlxcel_core::expand_dims(&ang_h, -1);
        let w_e = mlxcel_core::expand_dims(&ang_w, -1);
        let hw = mlxcel_core::concatenate(&h_e, &w_e, -1); // [b, s, hw/2, 2]
        let hw = mlxcel_core::reshape(&hw, &[batch, seq, 2 * n_hw]);
        let angles = mlxcel_core::concatenate(&hw, &ang_t, -1); // [b, s, head_dim/2]

        // Repeat each angle at adjacent slots: [a0, a0, a1, a1, ...].
        let repeat_adjacent = |x: &MlxArray| {
            let e = mlxcel_core::expand_dims(x, -1);
            let d = mlxcel_core::concatenate(&e, &e, -1);
            mlxcel_core::reshape(&d, &[batch, seq, self.head_dim])
        };
        let cos = repeat_adjacent(&mlxcel_core::cos(&angles));
        let sin = repeat_adjacent(&mlxcel_core::sin(&angles));
        (cos, sin)
    }
}

/// Rotate adjacent (even, odd) pairs: `[-x1, x0, -x3, x2, ...]`.
fn rotate_interleaved(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let last = shape[ndim - 1];
    let mut pair_shape = shape.clone();
    pair_shape[ndim - 1] = last / 2;
    pair_shape.push(2);
    let xr = mlxcel_core::reshape(x, &pair_shape);

    let mut starts = vec![0i32; pair_shape.len()];
    let mut stops = pair_shape.clone();
    stops[pair_shape.len() - 1] = 1;
    let even = mlxcel_core::slice(&xr, &starts, &stops);
    starts[pair_shape.len() - 1] = 1;
    stops[pair_shape.len() - 1] = 2;
    let odd = mlxcel_core::slice(&xr, &starts, &stops);

    let neg_odd = mlxcel_core::negative(&odd);
    let rotated = mlxcel_core::concatenate(&neg_odd, &even, -1);
    mlxcel_core::reshape(&rotated, &shape)
}

/// Apply the interleaved rotation to q and k in f32; cos/sin are
/// `[batch, seq, head_dim]` and broadcast over the heads axis.
fn apply_interleaved_rope(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let cos4 = mlxcel_core::expand_dims(cos, 1); // [b, 1, s, d]
    let sin4 = mlxcel_core::expand_dims(sin, 1);

    let rotate = |x: &MlxArray| {
        let orig_dtype = mlxcel_core::array_dtype(x);
        let xf = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let t1 = mlxcel_core::multiply(&xf, &cos4);
        let r = rotate_interleaved(&xf);
        let t2 = mlxcel_core::multiply(&r, &sin4);
        let out = mlxcel_core::add(&t1, &t2);
        mlxcel_core::astype(&out, orig_dtype)
    };
    (rotate(q), rotate(k))
}

// Attention (ERNIE projections + interleaved MRoPE).

struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    mrope: ErnieMRoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        config: &Ernie45MoeVlTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        let head_dim = config.head_dim();
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.self_attn.q_proj"),
                gs,
                bits,
            )?,
            k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.self_attn.k_proj"),
                gs,
                bits,
            )?,
            v_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.self_attn.v_proj"),
                gs,
                bits,
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.self_attn.o_proj"),
                gs,
                bits,
            )?,
            mrope: ErnieMRoPE::new(head_dim, config.rope_theta, &config.mrope_section()),
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_kv_heads() as i32,
            head_dim: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let (cos, sin) = self.mrope.forward(position_ids);
        let (q, k) = apply_interleaved_rope(&q, &k, &cos, &sin);

        let (k, v) = cache.update_and_fetch(k, v);

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            mlxcel_core::utils::repeat_kv(&k, n_rep)
        } else {
            mlxcel_core::copy(&k)
        };
        let v = if n_rep > 1 {
            mlxcel_core::utils::repeat_kv(&v, n_rep)
        } else {
            mlxcel_core::copy(&v)
        };

        let mask_ptr = mask
            .map(|m| m as *const MlxArray)
            .unwrap_or(std::ptr::null());
        // SAFETY: q/k/v are valid arrays; mask_ptr is null or a valid array ref.
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, self.scale, mask_ptr, 0.0, 0)
        };

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }
}

// Modality-split MoE.

fn load_switch_linear(
    weights: &WeightMap,
    prefix: &str,
    gs: i32,
    bits: i32,
) -> Result<SwitchLinear, String> {
    let weight_key = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {weight_key}"))?;
    let num_experts = mlxcel_core::array_shape(&weight)[0] as usize;
    let scales_key = format!("{prefix}.scales");
    if let Some(scales) = weights.get(&scales_key) {
        let biases_key = format!("{prefix}.biases");
        let biases = weights
            .get(&biases_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {biases_key}"))?;
        Ok(SwitchLinear::Quantized {
            weight,
            scales: mlxcel_core::copy(scales),
            biases,
            group_size: gs,
            bits,
            num_experts,
        })
    } else {
        Ok(SwitchLinear::Regular {
            weight,
            num_experts,
        })
    }
}

fn load_switch_glu(
    weights: &WeightMap,
    prefix: &str,
    gs: i32,
    bits: i32,
) -> Result<SwitchGLU, String> {
    Ok(SwitchGLU {
        gate_proj: load_switch_linear(weights, &format!("{prefix}.gate_proj"), gs, bits)?,
        up_proj: load_switch_linear(weights, &format!("{prefix}.up_proj"), gs, bits)?,
        down_proj: load_switch_linear(weights, &format!("{prefix}.down_proj"), gs, bits)?,
    })
}

fn load_dense_mlp(
    weights: &WeightMap,
    prefix: &str,
    gs: i32,
    bits: i32,
) -> Result<DenseMLP, String> {
    Ok(DenseMLP {
        gate_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.gate_proj"), gs, bits)?,
        up_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.up_proj"), gs, bits)?,
        down_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.down_proj"), gs, bits)?,
    })
}

/// One routed bank: router, selection-only correction bias, stacked experts.
struct ExpertBank {
    gate: UnifiedLinear,
    correction_bias: UniquePtr<MlxArray>, // [E] f32
    switch_mlp: SwitchGLU,
}

impl ExpertBank {
    /// Softmax routing with selection-only correction bias.
    /// `x`: `[n_tokens, hidden]` -> `(indices [n, k], scores [n, k])`.
    fn route(
        &self,
        x: &MlxArray,
        top_k: i32,
        norm_min: f32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let logits = self.gate.forward(x);
        let logits = mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32);
        let probs = mlxcel_core::softmax(&logits, -1);

        // Selection over probs + bias; mixing weights stay uncorrected.
        let biased = mlxcel_core::add(&probs, &self.correction_bias);
        let neg = mlxcel_core::negative(&biased);
        let parted = mlxcel_core::argpartition(&neg, top_k - 1, -1);
        let parted_shape = mlxcel_core::array_shape(&parted);
        let indices = mlxcel_core::slice(&parted, &[0, 0], &[parted_shape[0], top_k]);

        let scores = mlxcel_core::take_along_axis(&probs, &indices, -1);
        let score_sum = mlxcel_core::sum_axis(&scores, -1, true);
        let floor = mlxcel_core::full_f32(&[1], norm_min, mlxcel_core::dtype::FLOAT32);
        let score_sum = mlxcel_core::maximum(&score_sum, &floor);
        let scores = mlxcel_core::divide(&scores, &score_sum);
        (indices, scores)
    }

    /// Routed forward: `[n_tokens, hidden]` -> `[n_tokens, hidden]`.
    fn forward(&self, x: &MlxArray, top_k: i32, norm_min: f32) -> UniquePtr<MlxArray> {
        let (indices, scores) = self.route(x, top_k, norm_min);
        let expert_out = self.switch_mlp.forward(x, &indices);
        crate::models::switch_layers::moe_weighted_sum(
            &expert_out,
            &scores,
            mlxcel_core::array_dtype(x),
        )
    }
}

/// Dual-bank MoE block with per-token modality dispatch.
struct DualMoeBlock {
    text_bank: ExpertBank,
    mm_bank: Option<ExpertBank>,
    shared_experts: Option<DenseMLP>,
    top_k: i32,
    norm_min: f32,
}

impl DualMoeBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &Ernie45MoeVlTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        let mlp = format!("{prefix}.mlp");

        let load_bias = |key: &str| -> Result<UniquePtr<MlxArray>, String> {
            let bias = weights
                .get(key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            Ok(mlxcel_core::astype(&bias, mlxcel_core::dtype::FLOAT32))
        };

        let text_bank = ExpertBank {
            gate: UnifiedLinear::from_weights(weights, &format!("{mlp}.gate"), gs, bits)?,
            correction_bias: load_bias(&format!("{mlp}.e_score_correction_bias"))?,
            switch_mlp: load_switch_glu(weights, &format!("{mlp}.switch_mlp"), gs, bits)?,
        };

        let has_mm = config
            .moe_num_experts
            .as_ref()
            .and_then(|e| e.second())
            .map(|n| n > 0)
            .unwrap_or(false)
            && weights.contains_key(&format!("{mlp}.gate_1.weight"));
        let mm_bank = if has_mm {
            Some(ExpertBank {
                gate: UnifiedLinear::from_weights(weights, &format!("{mlp}.gate_1"), gs, bits)?,
                correction_bias: load_bias(&format!("{mlp}.e_score_correction_bias_1"))?,
                switch_mlp: load_switch_glu(weights, &format!("{mlp}.switch_mlp_1"), gs, bits)?,
            })
        } else {
            None
        };

        // Fused (unindexed) shared-experts SwiGLU, added for every token.
        let shared_experts = if config.moe_num_shared_experts > 0 {
            Some(load_dense_mlp(
                weights,
                &format!("{mlp}.shared_experts"),
                gs,
                bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            text_bank,
            mm_bank,
            shared_experts,
            top_k: config.moe_k as i32,
            norm_min: config.moe_norm_min,
        })
    }

    /// `x`: `[batch, seq, hidden]`; `mm_mask`: `[batch, seq]` (1 at image
    /// placeholder positions), `None` for text-only prompts and decode steps.
    fn forward(&self, x: &MlxArray, mm_mask: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden = orig_shape[orig_shape.len() - 1];
        let n_tokens: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
        let x_flat = mlxcel_core::reshape(x, &[n_tokens, hidden]);

        let y_text = self.text_bank.forward(&x_flat, self.top_k, self.norm_min);

        let mut result = match (self.mm_bank.as_ref(), mm_mask) {
            (Some(mm), Some(mask)) => {
                let y_mm = mm.forward(&x_flat, self.top_k, self.norm_min);
                // y = y_text + m * (y_mm - y_text), m broadcast over hidden.
                let m = mlxcel_core::reshape(mask, &[n_tokens, 1]);
                let m = mlxcel_core::astype(&m, mlxcel_core::array_dtype(&y_text));
                let diff = mlxcel_core::subtract(&y_mm, &y_text);
                let sel = mlxcel_core::multiply(&m, &diff);
                mlxcel_core::add(&y_text, &sel)
            }
            _ => y_text,
        };

        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(&x_flat);
            result = mlxcel_core::add(&result, &shared_out);
        }

        mlxcel_core::reshape(&result, &orig_shape)
    }
}

enum MlpVariant {
    Dense(DenseMLP),
    Moe(DualMoeBlock),
}

// Decoder layer.

struct DecoderLayer {
    attn: Attention,
    mlp: MlpVariant,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &Ernie45MoeVlTextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let gs = config.group_size();
        let bits = config.bits();

        let mlp = if config.is_moe_layer(layer_idx) {
            MlpVariant::Moe(DualMoeBlock::from_weights(weights, config, &prefix)?)
        } else {
            MlpVariant::Dense(load_dense_mlp(weights, &format!("{prefix}.mlp"), gs, bits)?)
        };

        Ok(Self {
            attn: Attention::from_weights(weights, config, &prefix)?,
            mlp,
            input_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.input_layernorm"),
                config.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{prefix}.post_attention_layernorm"),
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: &MlxArray,
        mm_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let r = self
            .attn
            .forward(&self.input_layernorm.forward(x), cache, mask, position_ids);
        let h = mlxcel_core::add(x, &r);
        let normed = self.post_attention_layernorm.forward(&h);
        let r = match &self.mlp {
            MlpVariant::Dense(mlp) => mlp.forward(&normed),
            MlpVariant::Moe(moe) => moe.forward(&normed, mm_mask),
        };
        mlxcel_core::add(&h, &r)
    }
}

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let key = format!("{prefix}.weight");
    let weight = weights
        .get(&key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    Ok(RMSNorm::new(weight, eps))
}

// Full language model.

pub struct Ernie45MoeVlTextModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    pub eos_token_ids: Vec<i32>,
    im_patch_id: i32,
    mrope_state: MRopeState,
}

impl Ernie45MoeVlTextModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Ernie45MoeVlTextConfig,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(weights, config, i)?);
        }

        let norm = load_rms_norm(weights, "model.norm", config.rms_norm_eps)?;

        let lm_head = if config.tie_word_embeddings {
            UnifiedLinear::from_weights(weights, "model.embed_tokens", gs, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", gs, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            eos_token_ids: vec![2],
            im_patch_id: config.im_patch_id,
            mrope_state: MRopeState::new(),
        })
    }

    /// Store MRoPE state for the legacy/non-server caller (CLI generate and the
    /// vision wrapper before a `SequenceId` is plumbed).
    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        self.mrope_state.set_fallback(position_ids, rope_deltas);
    }

    /// Store MRoPE state for a specific server sequence.
    pub fn set_mrope_state_for_sequence(
        &self,
        seq_id: SequenceId,
        position_ids: UniquePtr<MlxArray>,
        rope_deltas: i32,
    ) {
        self.mrope_state
            .set_for_sequence(seq_id, position_ids, rope_deltas);
    }

    /// Clear the legacy/fallback MRoPE state (new image/video).
    pub fn clear_mrope_state(&self) {
        self.mrope_state.clear_fallback();
    }

    /// Drop a server sequence's MRoPE entry.
    pub fn release_mrope_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.release_sequence(seq_id);
    }

    /// Move the fallback slot into the per-sequence map under `seq_id`.
    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.bind_fallback_to_sequence(seq_id);
    }

    /// Per-token multimodal dispatch mask, `[batch, seq]` (1 at image
    /// placeholder positions). `None` when the window carries no placeholder
    /// (text-only prompts and every decode step), so the multimodal bank is
    /// skipped entirely on those paths.
    fn compute_mm_mask(&self, input_ids: &MlxArray, seq_len: i32) -> Option<UniquePtr<MlxArray>> {
        if seq_len <= 1 {
            return None;
        }
        let patch = mlxcel_core::from_slice_i32(&[self.im_patch_id], &[1]);
        let is_image = mlxcel_core::equal(input_ids, &patch);
        let is_image = mlxcel_core::astype(&is_image, mlxcel_core::dtype::INT32);
        let count = mlxcel_core::sum_all(&is_image);
        mlxcel_core::eval(&count);
        if mlxcel_core::item_i32(&count) == 0 {
            return None;
        }
        Some(is_image)
    }

    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, input_embeddings, caches, mask, None)
    }

    pub(crate) fn forward_for_sequence(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        let ids_shape = mlxcel_core::array_shape(input_ids);
        let batch = ids_shape[0];
        let seq_len = ids_shape[1];
        let cache_offset = caches[0].offset;

        let position_ids = self.mrope_state.with_entry(seq_id, |entry| {
            if let Some(ref stored_pos) = entry.position_ids {
                let pos_shape = mlxcel_core::array_shape(stored_pos);
                if pos_shape.len() == 3
                    && pos_shape[1] == batch
                    && pos_shape[2] >= cache_offset + seq_len
                {
                    return mlxcel_core::slice(
                        stored_pos,
                        &[0, 0, cache_offset],
                        &[pos_shape[0], pos_shape[1], cache_offset + seq_len],
                    );
                }
                Self::compute_position_ids_with_delta(
                    entry.rope_deltas.unwrap_or(0),
                    batch,
                    seq_len,
                    cache_offset,
                )
            } else if cache_offset > 0 {
                Self::compute_position_ids_with_delta(
                    entry.rope_deltas.unwrap_or(0),
                    batch,
                    seq_len,
                    cache_offset,
                )
            } else {
                let pos = mlxcel_core::arange_i32(0, seq_len, 1);
                let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
                let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
                let pos = mlxcel_core::expand_dims(&pos, 0);
                mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len])
            }
        });

        let mm_mask = self.compute_mm_mask(input_ids, seq_len);

        let auto_mask;
        let mask = if mask.is_some() {
            mask
        } else {
            auto_mask = mlxcel_core::utils::create_causal_mask(seq_len, caches[0].live_len());
            Some(auto_mask.as_ref().unwrap() as &MlxArray)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask, &position_ids, mm_mask.as_deref());
        }

        h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    fn compute_position_ids_with_delta(
        delta: i32,
        batch: i32,
        seq_len: i32,
        cache_offset: i32,
    ) -> UniquePtr<MlxArray> {
        let offset = cache_offset + delta;
        let pos = mlxcel_core::arange_i32(offset, offset + seq_len, 1);
        let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
        let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
        let pos = mlxcel_core::expand_dims(&pos, 0);
        mlxcel_core::broadcast_to(&pos, &[3, batch, seq_len])
    }

    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

impl mlxcel_core::generate::LanguageModel for Ernie45MoeVlTextModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, None, caches, mask, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, input_embeddings, caches, mask, seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.release_mrope_sequence(seq_id);
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Ernie45MoeVlTextModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With p_T = p_H = p_W = p (every text token), the interleaved 3D MRoPE
    /// must reduce exactly to traditional interleaved RoPE, i.e. `fast_rope`
    /// with `traditional = true` and the same base.
    #[test]
    fn text_only_mrope_degenerates_to_traditional_rope() {
        let head_dim = 128usize;
        let base = 500_000.0f32;
        let (batch, heads, seq) = (1i32, 2i32, 5i32);

        // Deterministic pseudo-random input.
        let n = (batch * heads * seq * head_dim as i32) as usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.618).sin() * 0.5).collect();
        let q = mlxcel_core::from_slice_f32(&data, &[batch, heads, seq, head_dim as i32]);

        // Scalar positions 0..seq on all three axes.
        let mrope = ErnieMRoPE::new(head_dim, base, &[22, 22, 20]);
        let pos = mlxcel_core::arange_i32(0, seq, 1);
        let pos = mlxcel_core::reshape(&pos, &[1, 1, seq]);
        let pos = mlxcel_core::broadcast_to(&pos, &[3, batch, seq]);
        let (cos, sin) = mrope.forward(&pos);
        let (q_ours, _) = apply_interleaved_rope(&q, &q, &cos, &sin);

        let q_ref = mlxcel_core::fast_rope(&q, head_dim as i32, true, base, 1.0, 0);

        let diff = mlxcel_core::subtract(&q_ours, &q_ref);
        let diff = mlxcel_core::multiply(&diff, &diff);
        let max_sq = mlxcel_core::max_all(&diff);
        mlxcel_core::eval(&max_sq);
        let max_err = mlxcel_core::item_f32(&max_sq).sqrt();
        assert!(
            max_err < 1e-3,
            "interleaved 3D MRoPE must match fast_rope(traditional) for scalar positions, max err {max_err}"
        );
    }

    #[test]
    fn int_or_pair_parses_both_forms() {
        #[derive(Deserialize)]
        struct T {
            a: IntOrPair,
            b: IntOrPair,
        }
        let t: T = serde_json::from_str(r#"{ "a": 128, "b": [64, 64] }"#).unwrap();
        assert_eq!(t.a.first(), 128);
        assert_eq!(t.a.second(), None);
        assert_eq!(t.b.first(), 64);
        assert_eq!(t.b.second(), Some(64));
        assert_eq!(t.b.max(), 64);
    }

    #[test]
    fn moe_layer_rule_matches_checkpoint_geometry() {
        // The 28B-A3B checkpoint: 28 layers, start [1,1], end [29,28], interval 1
        // -> layer 0 dense, layers 1..=27 MoE.
        let config: Ernie45MoeVlTextConfig = serde_json::from_str(
            r#"{
                "hidden_size": 2560, "intermediate_size": 12288,
                "num_hidden_layers": 28, "num_attention_heads": 20,
                "num_key_value_heads": 4, "vocab_size": 103424,
                "moe_num_experts": [64, 64], "moe_intermediate_size": [1536, 512],
                "moe_k": 6, "moe_layer_start_index": [1, 1],
                "moe_layer_end_index": [29, 28], "moe_layer_interval": 1,
                "moe_num_shared_experts": 2
            }"#,
        )
        .unwrap();
        assert!(!config.is_moe_layer(0));
        for l in 1..28 {
            assert!(config.is_moe_layer(l), "layer {l} must be MoE");
        }
        assert_eq!(config.mrope_section(), vec![22, 22, 20]);
        assert_eq!(config.head_dim(), 128);
    }

    /// The correction bias must change which experts are selected without
    /// changing the mixing weights (which come from the uncorrected probs).
    #[test]
    fn correction_bias_is_selection_only() {
        // 1 token, hidden 4, 3 experts. Gate weight rows chosen so the raw
        // probs order experts as [e0 > e1 > e2]; a large bias on e2 must pull
        // it into the top-2 selection while its mixing weight stays the raw
        // (uncorrected, renormalized) prob.
        let gate_w = mlxcel_core::from_slice_f32(
            &[
                1.0, 0.0, 0.0, 0.0, // e0 logit = x[0]
                0.5, 0.0, 0.0, 0.0, // e1 logit = 0.5 x[0]
                0.0, 0.0, 0.0, 0.0, // e2 logit = 0
            ],
            &[3, 4],
        );
        let gate = UnifiedLinear::Regular(mlxcel_core::layers::Linear::new(gate_w, None));

        // Identity-ish experts are unnecessary: route() alone is under test.
        let bias = mlxcel_core::from_slice_f32(&[0.0, 0.0, 10.0], &[3]);
        let bank = ExpertBank {
            gate,
            correction_bias: bias,
            switch_mlp: SwitchGLU {
                gate_proj: SwitchLinear::Regular {
                    weight: mlxcel_core::from_slice_f32(&[0.0; 3 * 4 * 4], &[3, 4, 4]),
                    num_experts: 3,
                },
                up_proj: SwitchLinear::Regular {
                    weight: mlxcel_core::from_slice_f32(&[0.0; 3 * 4 * 4], &[3, 4, 4]),
                    num_experts: 3,
                },
                down_proj: SwitchLinear::Regular {
                    weight: mlxcel_core::from_slice_f32(&[0.0; 3 * 4 * 4], &[3, 4, 4]),
                    num_experts: 3,
                },
            },
        };

        let x = mlxcel_core::from_slice_f32(&[2.0, 0.0, 0.0, 0.0], &[1, 4]);
        let (indices, scores) = bank.route(&x, 2, 1e-12);
        mlxcel_core::eval(&indices);
        mlxcel_core::eval(&scores);

        // Selected experts: e2 (bias-boosted) and e0 (highest raw prob).
        let idx0 = mlxcel_core::slice(&indices, &[0, 0], &[1, 1]);
        let idx1 = mlxcel_core::slice(&indices, &[0, 1], &[1, 2]);
        mlxcel_core::eval(&idx0);
        mlxcel_core::eval(&idx1);
        let mut selected = vec![mlxcel_core::item_i32(&idx0), mlxcel_core::item_i32(&idx1)];
        selected.sort_unstable();
        assert_eq!(selected, vec![0, 2], "bias must pull e2 into the selection");

        // Mixing weights are renormalized raw probs of the selected experts:
        // softmax([2, 1, 0]) = [0.6652, 0.2447, 0.0900]; picked {e2, e0} ->
        // weights [p2, p0] / (p2 + p0).
        let s = mlxcel_core::reshape(&scores, &[2]);
        let s0 = mlxcel_core::slice(&s, &[0], &[1]);
        let s1 = mlxcel_core::slice(&s, &[1], &[2]);
        mlxcel_core::eval(&s0);
        mlxcel_core::eval(&s1);
        let v0 = mlxcel_core::item_f32(&s0);
        let v1 = mlxcel_core::item_f32(&s1);
        let (p0, p2) = (0.66524096f32, 0.09003057f32);
        let expected: Vec<f32> = vec![p2 / (p0 + p2), p0 / (p0 + p2)];
        // scores follow the selection order of `indices`; compare as a set of
        // (index, weight) pairs.
        let got: Vec<f32> = vec![v0, v1];
        let mut got_sorted = got.clone();
        got_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected_sorted = expected.clone();
        expected_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (g, e) in got_sorted.iter().zip(expected_sorted.iter()) {
            assert!(
                (g - e).abs() < 1e-4,
                "mixing weights must be uncorrected renormalized probs, got {got:?} expected {expected:?}"
            );
        }
    }
}
