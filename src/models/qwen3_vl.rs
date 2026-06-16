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

//! Qwen3-VL Language Model with Interleaved MRoPE and DeepStack
//!
//! Differs from Qwen2-VL language model:
//! - Interleaved MRoPE (step-3 slicing) instead of chunked sections
//! - q_norm/k_norm (RMSNorm on head_dim) before RoPE
//! - No attention bias (Qwen2-VL has bias)
//! - DeepStack visual feature injection in decoder layers
//!
//! Used by: Qwen3-VL
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl/language.py

use crate::models::qwen_mrope_state::MRopeState;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{FusedQKVLinear, KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::cell::RefCell;

// Config.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VLConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool, // false for Qwen3-VL
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<i32>,
    #[serde(rename = "type", default)]
    pub scaling_type: String,
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
    1000000.0
}
fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

impl Qwen3VLConfig {
    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    fn head_dim(&self) -> usize {
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
    fn mrope_section(&self) -> Vec<i32> {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_else(|| vec![24, 20, 20])
    }
}

// Interleaved MRoPE.
pub(crate) struct InterleavedMRoPE {
    inv_freq: Vec<f32>,
    mrope_section: Vec<i32>,
}

impl InterleavedMRoPE {
    pub(crate) fn new(dim: usize, base: f32, mrope_section: Vec<i32>) -> Self {
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            inv_freq.push(1.0 / base.powf(i as f32 / dim as f32));
        }
        Self {
            inv_freq,
            mrope_section,
        }
    }

    /// Compute cos/sin for interleaved MRoPE
    /// position_ids: [3, batch, seq_len] for multimodal, or [batch, seq_len] for text-only
    /// Returns (cos, sin) each [batch, seq_len, head_dim]
    pub(crate) fn forward(
        &self,
        position_ids: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let pos_shape = mlxcel_core::array_shape(position_ids);

        // If 2D, broadcast to [3, batch, seq_len]
        let position_ids_3d = if pos_shape.len() == 2 {
            let expanded = mlxcel_core::expand_dims(position_ids, 0);
            mlxcel_core::broadcast_to(&expanded, &[3, pos_shape[0], pos_shape[1]])
        } else {
            mlxcel_core::copy(position_ids)
        };

        let pos_shape = mlxcel_core::array_shape(&position_ids_3d);
        let batch = pos_shape[1];
        let seq_len = pos_shape[2];
        let half_dim = self.inv_freq.len() as i32;

        // inv_freq: [half_dim] -> [1, 1, half_dim, 1]
        let inv_freq_arr = mlxcel_core::from_slice_f32(&self.inv_freq, &[half_dim]);
        let inv_freq_arr = mlxcel_core::astype(&inv_freq_arr, mlxcel_core::dtype::FLOAT32);
        let inv_freq_4d = mlxcel_core::reshape(&inv_freq_arr, &[1, 1, half_dim, 1]);
        let inv_freq_4d = mlxcel_core::broadcast_to(&inv_freq_4d, &[3, batch, half_dim, 1]);

        // position_ids: [3, batch, seq_len] -> [3, batch, 1, seq_len]
        let pos_expanded = mlxcel_core::reshape(&position_ids_3d, &[3, batch, 1, seq_len]);
        let pos_expanded = mlxcel_core::astype(&pos_expanded, mlxcel_core::dtype::FLOAT32);

        // freqs = inv_freq @ position_ids: [3, batch, half_dim, seq_len]
        let freqs = mlxcel_core::matmul(&inv_freq_4d, &pos_expanded);
        // Transpose: [3, batch, seq_len, half_dim]
        let freqs = mlxcel_core::transpose_axes(&freqs, &[0, 1, 3, 2]);

        // Apply interleaved MRoPE section mixing
        let freqs = self.apply_interleaved_mrope(&freqs);
        // freqs: [batch, seq_len, half_dim]

        // Double the frequencies: [batch, seq_len, head_dim]
        let emb = mlxcel_core::concatenate(&freqs, &freqs, -1);

        let cos = mlxcel_core::cos(&emb);
        let sin = mlxcel_core::sin(&emb);

        (cos, sin)
    }

    /// Apply interleaved MRoPE: reorganize from chunked [TTT...HHH...WWW] to
    /// interleaved [THTHWHTHW...TT]
    /// freqs: [3, batch, seq_len, half_dim]
    /// Returns: [batch, seq_len, half_dim]
    fn apply_interleaved_mrope(&self, freqs: &MlxArray) -> UniquePtr<MlxArray> {
        let freqs_shape = mlxcel_core::array_shape(freqs);
        let _batch = freqs_shape[1];
        let _seq_len = freqs_shape[2];
        let half_dim = freqs_shape[3];

        // Build a per-column source-dimension index: for each position in [0..half_dim],
        // determine which of the 3 MRoPE dimensions (T=0, H=1, W=2) it comes from.
        // Pattern with mrope_section=[s0, s1, s2]:
        //   columns 0..s0*3       use dim 0 (T)
        //   within s0*3..s0*3+s1*3: step-3 starting at offset 1 → dim 1 (H)
        //   within s0*3..s0*3+s2*3: step-3 starting at offset 2 → dim 2 (W)
        // T is the default; H/W overwrite at their interleaved positions.
        let mut dim_indices: Vec<i32> = vec![0; half_dim as usize]; // default: T
        for (dim_idx, &section_len) in self.mrope_section[1..].iter().enumerate() {
            let src_dim = dim_idx as i32 + 1;
            let offset = dim_idx as i32 + 1;
            let length = section_len * 3;
            let mut idx = offset;
            while idx < length {
                if (idx as usize) < dim_indices.len() {
                    dim_indices[idx as usize] = src_dim;
                }
                idx += 3;
            }
        }

        // Vectorized gather: for each column, take from the appropriate dimension.
        // freqs shape: [3, batch, seq_len, half_dim]
        // We gather along axis 0 using dim_indices broadcast to [1, 1, 1, half_dim].
        let idx_arr = mlxcel_core::from_slice_i32(&dim_indices, &[1, 1, 1, half_dim]);
        let result = mlxcel_core::take_along_axis(freqs, &idx_arr, 0);
        // Squeeze the dimension axis: [1, batch, seq_len, half_dim] → [batch, seq_len, half_dim]
        mlxcel_core::squeeze_axis(&result, 0)
    }
}

/// Apply MRoPE to Q and K tensors
/// q, k: [batch, heads, seq, head_dim]
/// cos, sin: [batch, seq, head_dim]
pub(crate) fn apply_multimodal_rotary_pos_emb(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Expand: [batch, seq, head_dim] -> [batch, 1, seq, head_dim]
    let cos = mlxcel_core::expand_dims(cos, 1);
    let sin = mlxcel_core::expand_dims(sin, 1);

    let q_embed = {
        let t1 = mlxcel_core::multiply(q, &cos);
        let r = rotate_half(q);
        let t2 = mlxcel_core::multiply(&r, &sin);
        mlxcel_core::add(&t1, &t2)
    };
    let k_embed = {
        let t1 = mlxcel_core::multiply(k, &cos);
        let r = rotate_half(k);
        let t2 = mlxcel_core::multiply(&r, &sin);
        mlxcel_core::add(&t1, &t2)
    };

    (q_embed, k_embed)
}

pub(crate) fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let half = shape[shape.len() - 1] / 2;
    let ndim = shape.len();

    let mut starts = vec![0i32; ndim];
    let mut stops = shape.clone();
    stops[ndim - 1] = half;
    let x1 = mlxcel_core::slice(x, &starts, &stops);

    starts[ndim - 1] = half;
    stops[ndim - 1] = shape[ndim - 1];
    let x2 = mlxcel_core::slice(x, &starts, &stops);

    let neg_x2 = mlxcel_core::negative(&x2);
    mlxcel_core::concatenate(&neg_x2, &x1, ndim as i32 - 1)
}

// Attention with q_norm/k_norm and Interleaved MRoPE.
struct Attention {
    qkv_proj: FusedQKVLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    mrope: InterleavedMRoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_base: f32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3VLConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads as i32;
        let num_kv_heads = config.num_kv_heads() as i32;
        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights,
            &format!("{}.self_attn", prefix),
            gs,
            bits,
            num_heads,
            num_kv_heads,
            head_dim as i32,
        )?;

        let o_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.o_proj", prefix),
            gs,
            bits,
        )?;

        let q_norm = load_rms_norm(
            weights,
            &format!("{}.self_attn.q_norm", prefix),
            config.rms_norm_eps,
        )?;
        let k_norm = load_rms_norm(
            weights,
            &format!("{}.self_attn.k_norm", prefix),
            config.rms_norm_eps,
        )?;

        let mrope = InterleavedMRoPE::new(head_dim, config.rope_theta, config.mrope_section());

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            mrope,
            num_heads,
            num_kv_heads,
            head_dim: head_dim as i32,
            rope_base: config.rope_theta,
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
        let b = shape[0];
        let l = shape[1];

        let (q, k, v) = self.qkv_proj.forward(x);

        // Reshape: [B, L, dim] -> [B, L, heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Apply q_norm/k_norm BEFORE RoPE and transpose
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Transpose: [B, L, heads, head_dim] -> [B, heads, L, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        // Apply interleaved MRoPE
        let (cos, sin) = self.mrope.forward(position_ids);
        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin);

        // KV cache
        let (k, v) = cache.update_and_fetch(k, v);

        // Repeat KV heads if GQA
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

        // Attention
        let output = if let Some(m) = mask {
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    m as *const MlxArray,
                    0.0,
                    0,
                )
            }
        } else {
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            }
        };

        // [B, heads, L, head_dim] -> [B, L, dim]
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }

    fn forward_text_only(
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

        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

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

        let output = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &k, &v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(&q, &k, &v, self.scale, mask_ptr, 0.0, 0)
            }
        };

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }
}

// MLP (SwiGLU, same as Qwen2-VL/Llama).
struct MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLP {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3VLConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.gate_proj", prefix),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.up_proj", prefix),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.down_proj", prefix),
                gs,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// Decoder Layer.
struct DecoderLayer {
    attn: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3VLConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            attn: Attention::from_weights(weights, config, prefix)?,
            mlp: MLP::from_weights(weights, config, prefix)?,
            input_layernorm: load_rms_norm(
                weights,
                &format!("{}.input_layernorm", prefix),
                config.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_attention_layernorm", prefix),
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
    ) -> UniquePtr<MlxArray> {
        let r = self
            .attn
            .forward(&self.input_layernorm.forward(x), cache, mask, position_ids);
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &r)
    }

    fn forward_text_only(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let r = self
            .attn
            .forward_text_only(&self.input_layernorm.forward(x), cache, mask);
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &r)
    }
}

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let key = format!("{}.weight", prefix);
    let weight = weights
        .get(&key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", key))?;
    Ok(RMSNorm::new(weight, eps))
}

// Qwen3VLModel - Full language model with DeepStack support.
pub struct Qwen3VLModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    _config: Qwen3VLConfig,
    /// Per-sequence MRoPE state (mlx-vlm PR #1095). Each row
    /// in a server batch needs its own delta — the legacy fallback slot
    /// preserves CLI/single-row behavior when no `SequenceId` is plumbed.
    mrope_state: MRopeState,
    /// DeepStack state: visual position masks and visual embeddings
    visual_pos_masks: RefCell<Option<UniquePtr<MlxArray>>>,
    deepstack_visual_embeds: RefCell<Option<Vec<UniquePtr<MlxArray>>>>,
}

impl Qwen3VLModel {
    pub fn from_weights(weights: &WeightMap, config: &Qwen3VLConfig) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                config,
                &format!("model.layers.{}", i),
            )?);
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
            _config: config.clone(),
            mrope_state: MRopeState::new(),
            visual_pos_masks: RefCell::new(None),
            deepstack_visual_embeds: RefCell::new(None),
        })
    }

    /// Set MRoPE state for the legacy/non-server caller. Used by the CLI
    /// generate path and by the vision wrapper when a `SequenceId` is not
    /// (yet) available.
    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        self.mrope_state.set_fallback(position_ids, rope_deltas);
    }

    /// Set MRoPE state for a specific server-side sequence so the cached
    /// per-sequence delta no longer leaks across requests.
    pub fn set_mrope_state_for_sequence(
        &self,
        seq_id: SequenceId,
        position_ids: UniquePtr<MlxArray>,
        rope_deltas: i32,
    ) {
        self.mrope_state
            .set_for_sequence(seq_id, position_ids, rope_deltas);
    }

    /// Clear the legacy/fallback MRoPE state (for new image/video).
    pub fn clear_mrope_state(&self) {
        self.mrope_state.clear_fallback();
    }

    /// Drop a server sequence's MRoPE entry so the per-sequence map
    /// does not grow without bound across requests.
    pub fn release_mrope_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.release_sequence(seq_id);
    }

    /// Move whatever the fallback slot holds into the per-sequence map
    /// under `seq_id`. Called by the scheduler right after the vision
    /// wrapper's `get_input_embeddings` has populated the fallback slot,
    /// so subsequent decode steps for this sequence resolve the MRoPE
    /// state by id instead of by leaky scalar.
    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.bind_fallback_to_sequence(seq_id);
    }

    /// Remove and return the per-sequence MRoPE entry under `seq_id`
    /// without dropping the contained position-id tensor. Used by the
    /// server preemption path so the entry can survive an evict-and-
    /// reallocate cycle (follow-up).
    pub(crate) fn take_mrope_entry(
        &self,
        seq_id: SequenceId,
    ) -> Option<crate::models::qwen_mrope_state::MRopeEntry> {
        self.mrope_state.take_for_sequence(seq_id)
    }

    /// Re-install a previously taken MRoPE entry under a (possibly new)
    /// `seq_id`. Used to rebind state across preemption.
    pub(crate) fn install_mrope_entry(
        &self,
        seq_id: SequenceId,
        entry: crate::models::qwen_mrope_state::MRopeEntry,
    ) {
        self.mrope_state.bind_for_sequence(seq_id, entry);
    }

    /// Set DeepStack state after vision processing
    pub fn set_deepstack_state(
        &self,
        visual_pos_masks: UniquePtr<MlxArray>,
        deepstack_visual_embeds: Vec<UniquePtr<MlxArray>>,
    ) {
        *self.visual_pos_masks.borrow_mut() = Some(visual_pos_masks);
        *self.deepstack_visual_embeds.borrow_mut() = Some(deepstack_visual_embeds);
    }

    /// Clear DeepStack state
    pub fn clear_deepstack_state(&self) {
        *self.visual_pos_masks.borrow_mut() = None;
        *self.deepstack_visual_embeds.borrow_mut() = None;
    }

    fn can_use_text_only_fast_path(&self, seq_id: Option<SequenceId>) -> bool {
        let has_mrope_state = self.mrope_state.with_entry(seq_id, |entry| {
            entry.position_ids.is_some() || entry.rope_deltas.unwrap_or(0) != 0
        });
        if has_mrope_state {
            return false;
        }

        let has_visual_mask = self.visual_pos_masks.borrow().is_some();
        let has_deepstack_embeds = self
            .deepstack_visual_embeds
            .borrow()
            .as_ref()
            .is_some_and(|embeds| !embeds.is_empty());
        !has_visual_mask && !has_deepstack_embeds
    }

    /// DeepStack: add visual features at image positions in hidden states
    fn deepstack_process(
        h: &MlxArray,
        visual_pos_masks: &MlxArray,
        visual_embeds: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        // visual_pos_masks: [batch, seq_len] bool mask of image positions
        // visual_embeds: [total_image_tokens, hidden_size]
        // h: [batch, seq_len, hidden_size]
        let h_shape = mlxcel_core::array_shape(h);
        let batch = h_shape[0];

        if batch == 1 {
            // Fast path for batch_size=1
            let mask_1d = mlxcel_core::slice(visual_pos_masks, &[0, 0], &[1, h_shape[1]]);
            let mask_1d = mlxcel_core::squeeze_axis(&mask_1d, 0);

            // Find image positions from mask
            mlxcel_core::eval(&mask_1d);
            let mask_shape = mlxcel_core::array_shape(&mask_1d);
            let seq_len = mask_shape[0] as usize;

            // Read mask values - convert to i32 first for reliable reading of bool arrays
            let mask_i32 = mlxcel_core::astype(&mask_1d, mlxcel_core::dtype::INT32);
            mlxcel_core::eval(&mask_i32);

            let mut image_positions = Vec::new();
            for i in 0..seq_len {
                let val = mlxcel_core::slice(&mask_i32, &[i as i32], &[i as i32 + 1]);
                mlxcel_core::eval(&val);
                if mlxcel_core::item_i32(&val) != 0 {
                    image_positions.push(i as i32);
                }
            }

            if image_positions.is_empty() {
                return mlxcel_core::copy(h);
            }

            // Extract the batch slice: [seq_len, hidden_size]
            let batch_h = mlxcel_core::slice(h, &[0, 0, 0], &[1, h_shape[1], h_shape[2]]);
            let batch_h = mlxcel_core::squeeze_axis(&batch_h, 0);

            // Create index array for image positions
            let idx_arr =
                mlxcel_core::from_slice_i32(&image_positions, &[image_positions.len() as i32]);

            // Add visual_embeds at image positions
            // h[image_positions] += visual_embeds
            let current_vals = mlxcel_core::take(&batch_h, &idx_arr, 0);
            let n_img = image_positions.len() as i32;
            let visual_slice = mlxcel_core::slice(visual_embeds, &[0, 0], &[n_img, h_shape[2]]);
            let updated_vals = mlxcel_core::add(&current_vals, &visual_slice);

            // Build result by scatter
            // For simplicity, use the full tensor approach:
            // result = h.copy(), then update positions
            let result = mlxcel_core::copy(&batch_h);
            for (local_idx, &pos) in image_positions.iter().enumerate() {
                let val = mlxcel_core::slice(
                    &updated_vals,
                    &[local_idx as i32, 0],
                    &[local_idx as i32 + 1, h_shape[2]],
                );
                mlxcel_core::slice_update(&result, &val, &[pos, 0], &[pos + 1, h_shape[2]]);
            }

            mlxcel_core::expand_dims(&result, 0)
        } else {
            // General batch path (unlikely for inference but handle it)
            mlxcel_core::copy(h)
        }
    }

    /// Forward pass with DeepStack support
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_for_sequence(input_ids, input_embeddings, caches, mask, None)
    }

    /// Internal forward path that takes an optional `SequenceId` so the
    /// cached MRoPE state is resolved per row.
    pub(crate) fn forward_for_sequence(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        let cache_offset = caches[0].offset;
        if input_embeddings.is_none() && cache_offset == 0 {
            self.clear_mrope_state();
            self.clear_deepstack_state();
        }

        if input_embeddings.is_none() && self.can_use_text_only_fast_path(seq_id) {
            return self.forward_text_only(input_ids, caches, mask);
        }

        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        let ids_shape = mlxcel_core::array_shape(input_ids);
        let batch = ids_shape[0];
        let seq_len = ids_shape[1];

        // Compute position_ids using this sequence's MRoPE entry.
        let position_ids = self.mrope_state.with_entry(seq_id, |entry| {
            if let Some(ref stored_pos) = entry.position_ids {
                // Sufficiency check: reuse cached entry when it covers the needed range,
                // including during chunked prefill where cache_offset > 0.
                // This matches upstream mlx-vlm PR #1048 (commit 1bf7742) which relaxed
                // the equality guard to shape[-1] >= cache_offset + seq_length.
                //
                // (upstream mlx-vlm PR #1040, commit 58e2435): also validate
                // pos_shape[1] == batch so sequential requests with different batch_sizes
                // do not reuse stale position IDs and crash on broadcast_shapes.
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

        // Create causal mask if needed
        let auto_mask;
        let mask = if mask.is_some() {
            mask
        } else {
            auto_mask = mlxcel_core::utils::create_causal_mask(seq_len, cache_offset);
            Some(auto_mask.as_ref().unwrap() as &MlxArray)
        };

        // Get deepstack state references
        let ds_masks = self.visual_pos_masks.borrow();
        let ds_embeds = self.deepstack_visual_embeds.borrow();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[layer_idx], mask, &position_ids);

            // DeepStack: inject visual features after this layer
            if let (Some(masks), Some(embeds)) = (&*ds_masks, &*ds_embeds)
                && layer_idx < embeds.len()
                && cache_offset == 0
            {
                h = Self::deepstack_process(&h, masks, &embeds[layer_idx]);
            }
        }

        h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    fn forward_text_only(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward_text_only(&h, &mut caches[layer_idx], mask);
        }

        h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Compute `[3, batch, seq_len]` position ids by adding `delta` to a
    /// sequential range starting at `cache_offset`.
    fn compute_position_ids_with_delta(
        delta: i32,
        batch: i32,
        seq_len: i32,
        cache_offset: i32,
    ) -> UniquePtr<MlxArray> {
        let offset = cache_offset + delta;

        // Fast path for single-token decode (seq_len=1, batch=1):
        // avoid arange+reshape+broadcast chain; directly construct [3, 1, 1].
        if seq_len == 1 && batch == 1 {
            return mlxcel_core::from_slice_i32(&[offset, offset, offset], &[3, 1, 1]);
        }

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
}

// LanguageModel trait implementation.
impl mlxcel_core::generate::LanguageModel for Qwen3VLModel {
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
        Qwen3VLModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![151645, 151643] // Qwen EOS tokens
    }
}
