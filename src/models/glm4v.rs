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

//! GLM-4V text backbone with sectioned even/odd MRoPE.
//!
//! Based on the GLM-4 architecture (partial RoPE, fused `gate_up_proj`, four
//! RMSNorm layers per block, attention bias) but driven by 3D MRoPE position
//! IDs `[T, H, W]` instead of scalar RoPE so image tokens carry spatial
//! structure. GLM-4V uses the `sectioned_even_odd` MRoPE style: the first
//! `partial_rotary_factor * head_dim` dimensions receive an interleaved
//! (even/odd, GPT-J style) rotation where rotary pair `j` is driven by the
//! position axis selected from `mrope_section` (chunked `[T, H, W]`).
//!
//! Used by: GLM-4V, GLM-4V MoE (shared MRoPE machinery)
//! Reference: references/mlx-vlm/mlx_vlm/models/glm4v/language.py

use crate::models::qwen_mrope_state::MRopeState;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

// Config.
#[derive(Debug, Clone, Deserialize)]
pub struct Glm4vTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub eos_token_id: Option<Vec<i32>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
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

fn default_num_kv_heads() -> usize {
    2
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_partial_rotary_factor() -> f32 {
    0.5
}
fn default_attention_bias() -> bool {
    true
}
fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

impl Glm4vTextConfig {
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
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| vec![8, 12, 12])
    }
    /// Number of dimensions RoPE is applied to (partial rotary).
    fn rope_dims(&self) -> usize {
        // Even count required for the even/odd pairing.
        let dims = (self.partial_rotary_factor * self.head_dim() as f32) as usize;
        dims - (dims % 2)
    }
}

/// Rotary pairing used when applying sectioned MRoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Glm4vRopePairing {
    /// GPT-J style: rotary pair `(2j, 2j+1)`, `cos`/`sin` interleave-repeated.
    /// Used by GLM-4V (`sectioned_even_odd`).
    EvenOdd,
    /// GPT-NeoX style: rotary pair `(d, d + half)`, `cos`/`sin` tiled. Used by
    /// GLM-4V MoE (`sectioned_half_split`).
    HalfSplit,
}

/// Sectioned MRoPE (`sectioned_even_odd` / `sectioned_half_split`).
///
/// Precomputes the inverse-frequency table and the per-pair position-axis
/// selector so `cos`/`sin` can be built directly from 3D position IDs. The
/// rotation is applied to the first `rope_dims` head dimensions; the remaining
/// dimensions pass through. The `pairing` selects even/odd vs half-split layout.
pub(crate) struct Glm4vMRoPE {
    /// `rope_dims / 2` inverse frequencies.
    inv_freq: Vec<f32>,
    /// Position axis (0=T, 1=H, 2=W) selected for each rotary pair.
    axis_selector: Vec<i32>,
    rope_dims: i32,
    pairing: Glm4vRopePairing,
}

impl Glm4vMRoPE {
    pub(crate) fn new(
        base: f32,
        rope_dims: usize,
        mrope_section: &[i32],
        pairing: Glm4vRopePairing,
    ) -> Self {
        let half = rope_dims / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            inv_freq.push(1.0 / base.powf((2 * i) as f32 / rope_dims as f32));
        }
        // Chunked position selection: the j-th pair takes the position axis
        // whose cumulative `mrope_section` window contains j (matching the
        // reference `_chunked_position_selector`, clamped to `half`).
        let mut axis_selector = vec![0i32; half];
        let mut offset = 0usize;
        for (axis, &length) in mrope_section.iter().enumerate() {
            let end = (offset + length as usize).min(half);
            for slot in axis_selector.iter_mut().take(end).skip(offset) {
                *slot = axis as i32;
            }
            offset = end;
        }
        Self {
            inv_freq,
            axis_selector,
            rope_dims: rope_dims as i32,
            pairing,
        }
    }

    /// Build `cos`/`sin` of shape `[batch, 1, seq, rope_dims]` from
    /// `position_ids` `[3, batch, seq]` (or `[batch, seq]` for text-only).
    pub(crate) fn cos_sin(
        &self,
        position_ids: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let pos_shape = mlxcel_core::array_shape(position_ids);
        // Normalize to [3, batch, seq].
        let pos3 = if pos_shape.len() == 2 {
            let expanded = mlxcel_core::expand_dims(position_ids, 0);
            mlxcel_core::broadcast_to(&expanded, &[3, pos_shape[0], pos_shape[1]])
        } else {
            mlxcel_core::copy(position_ids)
        };
        let pos_shape = mlxcel_core::array_shape(&pos3);
        let batch = pos_shape[1];
        let seq = pos_shape[2];
        let half = self.inv_freq.len() as i32;

        // freqs_all[3, batch, seq, half] = position_ids[..., None] * inv_freq.
        let pos_f = mlxcel_core::astype(&pos3, mlxcel_core::dtype::FLOAT32);
        let pos_e = mlxcel_core::expand_dims(&pos_f, 3);
        let inv = mlxcel_core::from_slice_f32(&self.inv_freq, &[half]);
        let inv = mlxcel_core::reshape(&inv, &[1, 1, 1, half]);
        let freqs_all = mlxcel_core::multiply(&pos_e, &inv);

        // Select the axis frequency for each pair via partition masks.
        let mut angles: Option<UniquePtr<MlxArray>> = None;
        for axis in 0..3i32 {
            let fa =
                mlxcel_core::slice(&freqs_all, &[axis, 0, 0, 0], &[axis + 1, batch, seq, half]);
            let fa = mlxcel_core::squeeze_axis(&fa, 0);
            let mask: Vec<f32> = self
                .axis_selector
                .iter()
                .map(|&s| if s == axis { 1.0 } else { 0.0 })
                .collect();
            let mask = mlxcel_core::from_slice_f32(&mask, &[1, 1, half]);
            let contrib = mlxcel_core::multiply(&fa, &mask);
            angles = Some(match angles {
                None => contrib,
                Some(acc) => mlxcel_core::add(&acc, &contrib),
            });
        }
        let angles = angles.unwrap();

        let cos_h = mlxcel_core::cos(&angles);
        let sin_h = mlxcel_core::sin(&angles);
        // Expand `[batch, seq, half]` to `[batch, seq, rope_dims]`: even/odd
        // interleave-repeats each frequency; half-split tiles the block.
        let (cos_f, sin_f) = match self.pairing {
            Glm4vRopePairing::EvenOdd => (
                mlxcel_core::repeat(&cos_h, 2, 2),
                mlxcel_core::repeat(&sin_h, 2, 2),
            ),
            Glm4vRopePairing::HalfSplit => (
                mlxcel_core::concatenate(&cos_h, &cos_h, 2),
                mlxcel_core::concatenate(&sin_h, &sin_h, 2),
            ),
        };
        // Add a broadcast head axis: [batch, 1, seq, rope_dims].
        let cos_f = mlxcel_core::expand_dims(&cos_f, 1);
        let sin_f = mlxcel_core::expand_dims(&sin_f, 1);
        (cos_f, sin_f)
    }

    /// Apply the sectioned rotation to `x` `[batch, heads, seq, head_dim]`.
    pub(crate) fn apply(
        &self,
        x: &MlxArray,
        cos_f: &MlxArray,
        sin_f: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let h = shape[1];
        let l = shape[2];
        let head_dim = shape[3];
        let rope_dims = self.rope_dims;
        let half = rope_dims / 2;

        let x_rot = mlxcel_core::slice(x, &[0, 0, 0, 0], &[b, h, l, rope_dims]);

        let rotated = match self.pairing {
            Glm4vRopePairing::EvenOdd => {
                // rotate_half_even_odd: (a, b) pairs -> (-b, a).
                let x5 = mlxcel_core::reshape(&x_rot, &[b, h, l, half, 2]);
                let even = mlxcel_core::slice(&x5, &[0, 0, 0, 0, 0], &[b, h, l, half, 1]);
                let even = mlxcel_core::squeeze_axis(&even, 4);
                let odd = mlxcel_core::slice(&x5, &[0, 0, 0, 0, 1], &[b, h, l, half, 2]);
                let odd = mlxcel_core::squeeze_axis(&odd, 4);
                let neg_odd = mlxcel_core::negative(&odd);
                let rot5 = mlxcel_core::stack_owned(&[neg_odd, even], -1);
                mlxcel_core::reshape(&rot5, &[b, h, l, rope_dims])
            }
            Glm4vRopePairing::HalfSplit => {
                // rotate_half: [-x2, x1] with x1 = first half, x2 = second half.
                let x1 = mlxcel_core::slice(&x_rot, &[0, 0, 0, 0], &[b, h, l, half]);
                let x2 = mlxcel_core::slice(&x_rot, &[0, 0, 0, half], &[b, h, l, rope_dims]);
                let neg_x2 = mlxcel_core::negative(&x2);
                mlxcel_core::concatenate(&neg_x2, &x1, 3)
            }
        };

        let term1 = mlxcel_core::multiply(&x_rot, cos_f);
        let term2 = mlxcel_core::multiply(&rotated, sin_f);
        let x_embed = mlxcel_core::add(&term1, &term2);

        if head_dim > rope_dims {
            let x_pass = mlxcel_core::slice(x, &[0, 0, 0, rope_dims], &[b, h, l, head_dim]);
            mlxcel_core::concatenate(&x_embed, &x_pass, 3)
        } else {
            x_embed
        }
    }
}

// Attention with sectioned even/odd MRoPE.
struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    mrope: Glm4vMRoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        config: &Glm4vTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        let q_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.q_proj", prefix),
            gs,
            bits,
        )?;
        let k_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.k_proj", prefix),
            gs,
            bits,
        )?;
        let v_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.v_proj", prefix),
            gs,
            bits,
        )?;
        let o_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.self_attn.o_proj", prefix),
            gs,
            bits,
        )?;

        let head_dim = config.head_dim();
        let mrope = Glm4vMRoPE::new(
            config.rope_theta,
            config.rope_dims(),
            &config.mrope_section(),
            Glm4vRopePairing::EvenOdd,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            mrope,
            num_heads: config.num_attention_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        cos_f: &MlxArray,
        sin_f: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let q = self.mrope.apply(&q, cos_f, sin_f);
        let k = self.mrope.apply(&k, cos_f, sin_f);

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

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }
}

// MLP with fused gate_up_proj (SwiGLU).
struct MLP {
    gate_up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
    intermediate_size: i32,
}

impl MLP {
    fn from_weights(
        weights: &WeightMap,
        config: &Glm4vTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        Ok(Self {
            gate_up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.gate_up_proj", prefix),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.down_proj", prefix),
                gs,
                bits,
            )?,
            intermediate_size: config.intermediate_size as i32,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let fused = self.gate_up_proj.forward(x);
        let gate = mlxcel_core::slice_last_dim(&fused, 0, self.intermediate_size);
        let up =
            mlxcel_core::slice_last_dim(&fused, self.intermediate_size, 2 * self.intermediate_size);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// Decoder layer with four RMSNorm layers (GLM-4 block).
struct DecoderLayer {
    attn: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_self_attn_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    post_mlp_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &Glm4vTextConfig,
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
            post_self_attn_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_self_attn_layernorm", prefix),
                config.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_attention_layernorm", prefix),
                config.rms_norm_eps,
            )?,
            post_mlp_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_mlp_layernorm", prefix),
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        cos_f: &MlxArray,
        sin_f: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.attn.forward(&normed, cache, mask, cos_f, sin_f);
        let attn_out = self.post_self_attn_layernorm.forward(&attn_out);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        let mlp_out = self.post_mlp_layernorm.forward(&mlp_out);
        mlxcel_core::add(&h, &mlp_out)
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

/// GLM-4V text model (language backbone with MRoPE).
pub struct Glm4vTextModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    mrope: Glm4vMRoPE,
    eos_token_ids: Vec<i32>,
    mrope_state: MRopeState,
}

impl Glm4vTextModel {
    pub fn from_weights(weights: &WeightMap, config: &Glm4vTextConfig) -> Result<Self, String> {
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

        let mrope = Glm4vMRoPE::new(
            config.rope_theta,
            config.rope_dims(),
            &config.mrope_section(),
            Glm4vRopePairing::EvenOdd,
        );

        let eos_token_ids = config
            .eos_token_id
            .clone()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![151329, 151336, 151338, 151348]);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mrope,
            eos_token_ids,
            mrope_state: MRopeState::new(),
        })
    }

    /// Store MRoPE position IDs for the legacy/single-row caller.
    pub fn set_mrope_state(&self, position_ids: UniquePtr<MlxArray>, rope_deltas: i32) {
        self.mrope_state.set_fallback(position_ids, rope_deltas);
    }

    pub fn clear_mrope_state(&self) {
        self.mrope_state.clear_fallback();
    }

    pub fn release_mrope_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.release_sequence(seq_id);
    }

    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.mrope_state.bind_fallback_to_sequence(seq_id);
    }

    pub(crate) fn take_mrope_entry(
        &self,
        seq_id: SequenceId,
    ) -> Option<crate::models::qwen_mrope_state::MRopeEntry> {
        self.mrope_state.take_for_sequence(seq_id)
    }

    pub(crate) fn install_mrope_entry(
        &self,
        seq_id: SequenceId,
        entry: crate::models::qwen_mrope_state::MRopeEntry,
    ) {
        self.mrope_state.bind_for_sequence(seq_id, entry);
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

        // Resolve MRoPE position IDs for this sequence (mirrors Qwen2-VL).
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

        let (cos_f, sin_f) = self.mrope.cos_sin(&position_ids);

        let auto_mask;
        let mask = if mask.is_some() {
            mask
        } else {
            auto_mask = mlxcel_core::utils::create_causal_mask(seq_len, caches[0].live_len());
            Some(auto_mask.as_ref().unwrap() as &MlxArray)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask, &cos_f, &sin_f);
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
}

impl mlxcel_core::generate::LanguageModel for Glm4vTextModel {
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
        Glm4vTextModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[path = "glm4v_tests.rs"]
mod tests;
