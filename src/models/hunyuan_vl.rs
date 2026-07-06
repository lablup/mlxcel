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

//! Hunyuan-VL language model (Hunyuan dense decoder + XD-RoPE).
//!
//! The decoder is the Hunyuan dense stack (GQA with an explicit `head_dim`
//! larger than `hidden / heads`, per-head Q/K RMSNorm applied *after* the
//! rotation, SwiGLU MLP, DynamicNTK-alpha rope base adjusted at init:
//! `base * alpha^(d / (d - 2))`), but position encoding at prefill is XD-RoPE:
//! 4D `[P, T, H, W]` position ids where the `head_dim / 2` frequency dims are
//! split into `xdrope_section` chunks (default `[16, 16, 16, 16]`), chunk `a`
//! taking its positions from axis `a`. Text tokens carry the same sequential
//! position on all four axes, which degenerates exactly to the standard
//! concat-half rotation (`fast_rope` with `traditional = false`), and decode
//! steps use sequential positions (`rope_deltas = 0`).
//!
//! Reuses [`crate::models::qwen_mrope_state::MRopeState`] for per-sequence
//! prefill position-id storage.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/hunyuan_vl/language.py>.

use crate::models::qwen_mrope_state::MRopeState;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

// Config.
#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanVlTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_use_qk_norm")]
    pub use_qk_norm: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub quantization: Option<QuantConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(rename = "type", default)]
    pub scaling_type: String,
    #[serde(default)]
    pub xdrope_section: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_use_qk_norm() -> bool {
    true
}
fn default_tie() -> bool {
    true
}
fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

impl HunyuanVlTextConfig {
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
    /// DynamicNTK-alpha adjusted base: `theta * alpha^(d / (d - 2))` when
    /// `rope_scaling.type` is `xdrope` / `dynamic` and alpha is set.
    fn effective_rope_base(&self) -> f32 {
        let d = self.head_dim() as f32;
        match &self.rope_scaling {
            Some(rs)
                if (rs.scaling_type == "xdrope" || rs.scaling_type == "dynamic")
                    && rs.alpha.is_some() =>
            {
                self.rope_theta * rs.alpha.unwrap().powf(d / (d - 2.0))
            }
            _ => self.rope_theta,
        }
    }
    fn xdrope_section(&self) -> Vec<i32> {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.xdrope_section.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| vec![16, 16, 16, 16])
    }
}

// XD-RoPE.

/// Frequency table split into per-axis chunks. `inv_freq[j] = base^(-2j/d)`;
/// chunk `a` (widths from `xdrope_section`, axes ordered `[P, T, H, W]`) takes
/// its positions from position axis `a`.
struct XdRope {
    /// Per axis: the inv_freq sub-table for that axis's frequency chunk.
    chunk_freqs: Vec<Vec<f32>>,
    head_dim: i32,
}

impl XdRope {
    fn new(head_dim: usize, base: f32, xdrope_section: &[i32]) -> Self {
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for j in 0..half {
            inv_freq.push(base.powf(-((2 * j) as f32) / head_dim as f32));
        }
        let mut chunk_freqs = Vec::with_capacity(xdrope_section.len());
        let mut offset = 0usize;
        for &w in xdrope_section {
            let end = (offset + w as usize).min(half);
            chunk_freqs.push(inv_freq[offset..end].to_vec());
            offset = end;
        }
        Self {
            chunk_freqs,
            head_dim: head_dim as i32,
        }
    }

    /// `position_ids`: `[num_axes, batch, seq]` int. Returns `(cos, sin)`, each
    /// `[batch, seq, head_dim]` in f32 with the concat-half layout
    /// (`[a_0..a_{d/2-1}, a_0..a_{d/2-1}]`).
    fn forward(&self, position_ids: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(position_ids);
        let (batch, seq) = (shape[1], shape[2]);

        let mut half: Option<UniquePtr<MlxArray>> = None;
        for (axis, freqs) in self.chunk_freqs.iter().enumerate() {
            let a = axis as i32;
            let pos = mlxcel_core::slice(position_ids, &[a, 0, 0], &[a + 1, batch, seq]);
            let pos = mlxcel_core::reshape(&pos, &[batch, seq, 1]);
            let pos = mlxcel_core::astype(&pos, mlxcel_core::dtype::FLOAT32);
            let table = mlxcel_core::from_slice_f32(freqs, &[1, 1, freqs.len() as i32]);
            let ang = mlxcel_core::multiply(&pos, &table);
            half = Some(match half {
                None => ang,
                Some(acc) => mlxcel_core::concatenate(&acc, &ang, -1),
            });
        }
        let half = half.expect("xdrope_section must be non-empty");
        let emb = mlxcel_core::concatenate(&half, &half, -1); // [b, s, head_dim]
        debug_assert_eq!(mlxcel_core::array_shape(&emb)[2], self.head_dim);

        (mlxcel_core::cos(&emb), mlxcel_core::sin(&emb))
    }
}

/// Concat-half rotate: `[-x2, x1]` over the last-axis halves.
fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let half = shape[ndim - 1] / 2;

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

/// Apply the concat-half rotation in f32; cos/sin `[batch, seq, head_dim]`
/// broadcast over the heads axis.
fn apply_rope(
    q: &MlxArray,
    k: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let cos4 = mlxcel_core::expand_dims(cos, 1);
    let sin4 = mlxcel_core::expand_dims(sin, 1);
    let rot = |x: &MlxArray| {
        let orig = mlxcel_core::array_dtype(x);
        let xf = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let t1 = mlxcel_core::multiply(&xf, &cos4);
        let r = rotate_half(&xf);
        let t2 = mlxcel_core::multiply(&r, &sin4);
        let out = mlxcel_core::add(&t1, &t2);
        mlxcel_core::astype(&out, orig)
    };
    (rot(q), rot(k))
}

// Attention.
struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    query_layernorm: Option<RMSNorm>,
    key_layernorm: Option<RMSNorm>,
    rope: XdRope,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        config: &HunyuanVlTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        let head_dim = config.head_dim();
        let load_norm = |name: &str| -> Result<Option<RMSNorm>, String> {
            if !config.use_qk_norm {
                return Ok(None);
            }
            let key = format!("{prefix}.self_attn.{name}.weight");
            let w = weights
                .get(&key)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Weight not found: {key}"))?;
            Ok(Some(RMSNorm::new(w, config.rms_norm_eps)))
        };
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
            query_layernorm: load_norm("query_layernorm")?,
            key_layernorm: load_norm("key_layernorm")?,
            rope: XdRope::new(
                head_dim,
                config.effective_rope_base(),
                &config.xdrope_section(),
            ),
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

        let (cos, sin) = self.rope.forward(position_ids);
        let (q, k) = apply_rope(&q, &k, &cos, &sin);

        // Per-head Q/K RMSNorm runs AFTER the rotation (upstream order).
        let q = match &self.query_layernorm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.key_layernorm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

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
        // SAFETY: q/k/v valid; mask_ptr null or a valid array reference.
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, self.scale, mask_ptr, 0.0, 0)
        };

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, -1]);
        self.o_proj.forward(&output)
    }
}

// MLP (SwiGLU).
struct MLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl MLP {
    fn from_weights(
        weights: &WeightMap,
        config: &HunyuanVlTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.up_proj"),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.down_proj"),
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

// Decoder layer.
struct DecoderLayer {
    attn: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &HunyuanVlTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            attn: Attention::from_weights(weights, config, prefix)?,
            mlp: MLP::from_weights(weights, config, prefix)?,
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
    ) -> UniquePtr<MlxArray> {
        let r = self
            .attn
            .forward(&self.input_layernorm.forward(x), cache, mask, position_ids);
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
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
pub struct HunyuanVlTextModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: UnifiedLinear,
    pub eos_token_ids: Vec<i32>,
    mrope_state: MRopeState,
    num_axes: i32,
}

impl HunyuanVlTextModel {
    pub fn from_weights(weights: &WeightMap, config: &HunyuanVlTextConfig) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();

        let embed_tokens = UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                config,
                &format!("model.layers.{i}"),
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
            eos_token_ids: vec![120_007, 120_020],
            mrope_state: MRopeState::new(),
            num_axes: config.xdrope_section().len() as i32,
        })
    }

    /// Store prefill position ids for the legacy/non-server caller.
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
        let num_axes = self.num_axes;

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
            }
            // Sequential positions on every axis (text prefill and all decode
            // steps; decode continues at the cache offset, rope_deltas = 0).
            let delta = entry.rope_deltas.unwrap_or(0);
            Self::sequential_position_ids(num_axes, batch, seq_len, cache_offset + delta)
        });

        let auto_mask;
        let mask = if mask.is_some() {
            mask
        } else {
            auto_mask = mlxcel_core::utils::create_causal_mask(seq_len, caches[0].live_len());
            Some(auto_mask.as_ref().unwrap() as &MlxArray)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask, &position_ids);
        }

        h = self.norm.forward(&h);
        self.lm_head.forward(&h)
    }

    fn sequential_position_ids(
        num_axes: i32,
        batch: i32,
        seq_len: i32,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let pos = mlxcel_core::arange_i32(offset, offset + seq_len, 1);
        let pos = mlxcel_core::reshape(&pos, &[1, seq_len]);
        let pos = mlxcel_core::broadcast_to(&pos, &[batch, seq_len]);
        let pos = mlxcel_core::expand_dims(&pos, 0);
        mlxcel_core::broadcast_to(&pos, &[num_axes, batch, seq_len])
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

impl mlxcel_core::generate::LanguageModel for HunyuanVlTextModel {
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
        HunyuanVlTextModel::make_caches(self)
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

    /// With the same sequential position on all four axes (every text token),
    /// XD-RoPE must reduce exactly to the standard concat-half rotation, i.e.
    /// `fast_rope` with `traditional = false` and the same (alpha-adjusted)
    /// base.
    #[test]
    fn text_only_xdrope_degenerates_to_standard_rope() {
        let head_dim = 128usize;
        let base = 10_000.0f32 * 1000.0f32.powf(128.0 / 126.0);
        let (batch, heads, seq) = (1i32, 2i32, 5i32);

        let n = (batch * heads * seq * head_dim as i32) as usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
        let q = mlxcel_core::from_slice_f32(&data, &[batch, heads, seq, head_dim as i32]);

        let rope = XdRope::new(head_dim, base, &[16, 16, 16, 16]);
        let pos = mlxcel_core::arange_i32(0, seq, 1);
        let pos = mlxcel_core::reshape(&pos, &[1, 1, seq]);
        let pos = mlxcel_core::broadcast_to(&pos, &[4, batch, seq]);
        let (cos, sin) = rope.forward(&pos);
        let (q_ours, _) = apply_rope(&q, &q, &cos, &sin);

        let q_ref = mlxcel_core::fast_rope(&q, head_dim as i32, false, base, 1.0, 0);

        let diff = mlxcel_core::subtract(&q_ours, &q_ref);
        let diff = mlxcel_core::multiply(&diff, &diff);
        let max_sq = mlxcel_core::max_all(&diff);
        mlxcel_core::eval(&max_sq);
        let max_err = mlxcel_core::item_f32(&max_sq).sqrt();
        assert!(
            max_err < 1e-3,
            "XD-RoPE with equal axes must match fast_rope(concat-half), max err {max_err}"
        );
    }

    #[test]
    fn alpha_scaling_adjusts_base() {
        let config: HunyuanVlTextConfig = serde_json::from_str(
            r#"{
                "hidden_size": 1024, "num_hidden_layers": 24,
                "intermediate_size": 3584, "num_attention_heads": 16,
                "num_key_value_heads": 8, "vocab_size": 120818,
                "head_dim": 128, "rope_theta": 10000.0,
                "rope_scaling": { "type": "xdrope", "alpha": 1000.0,
                                   "xdrope_section": [16, 16, 16, 16] }
            }"#,
        )
        .unwrap();
        let expected = 10_000.0f32 * 1000.0f32.powf(128.0 / 126.0);
        assert!((config.effective_rope_base() - expected).abs() < 1.0);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.xdrope_section(), vec![16, 16, 16, 16]);
    }
}
