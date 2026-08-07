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

//! Jina VLM (`model_type: "jvlm"`) text decoder.
//!
//! Qwen2-class backbone with an OLMo-style tensor layout, so neither
//! `models::qwen2` (separate `q_proj`/`k_proj`/`v_proj`, `gate_proj`/`up_proj`)
//! nor `models::molmo2` (`att_proj`/`ff_proj` under `blocks.N`) can load it
//! unchanged. Concretely, per layer the checkpoint ships:
//!
//! ```text
//! language_model.layers.N.attn.qkv        [4096, 2048]  fused Q|K|V, no bias
//! language_model.layers.N.attn.q_norm     [128]         per-head RMSNorm
//! language_model.layers.N.attn.k_norm     [128]         per-head RMSNorm
//! language_model.layers.N.attn.out        [2048, 2048]
//! language_model.layers.N.attn_norm       [2048]
//! language_model.layers.N.ffn.gate_up     [12288, 2048] fused up|gate
//! language_model.layers.N.ffn.down        [2048, 6144]
//! language_model.layers.N.ffn_norm        [2048]
//! ```
//!
//! Two details are easy to get backwards and produce fluent nonsense rather
//! than a crash:
//!
//! 1. The fused `ffn.gate_up` is `[up, gate]`, not `[gate, up]`: the reference
//!    computes `up, gate = split(gate_up, 2); down(silu(gate) * up)`. This is
//!    the same convention Molmo v1/v2 use, so
//!    [`mlxcel_core::compiled_swiglu_activation`] is called with the *second*
//!    half as the gate.
//! 2. The Q/K RMSNorms are per-head (`qkv_lnorm_on_heads: true`), i.e. applied
//!    over `head_dim` after the `[B, L, heads, head_dim]` reshape and before
//!    the head transpose.
//!
//! The embedding table is split into a frozen base (`embedding.embedding`,
//! 151936 rows) plus a 128-row extension (`embedding.new_embedding`) holding
//! `<im_start>`/`<im_end>`/`<im_patch>`/`<im_col>`/`<|image|>`/`<im_slice>`.
//! That is exactly the Molmo dual table, so [`Molmo2Embedding`] is reused.
//!
//! RoPE is the rotate-half variant at theta 1e6 over the full head
//! (`partial_rotary_factor: 1.0`), and the LM head is untied
//! (`language_model.lm_head`, 4-bit in the released MLX conversion).
//!
//! Reference: references/mlx-vlm/mlx_vlm/models/jina_vlm/language.py

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

// The dual embedding table is identical to Molmo's; share it so the two
// families cannot drift.
pub use super::molmo2::{Molmo2Embedding, Quantization};

/// Default EOS/stop ids: `<|endoftext|>` (also the BOS/pad) and `<|im_end|>`.
pub const JINA_VLM_DEFAULT_EOS_IDS: [i32; 2] = [151643, 151645];

/// Text-side configuration, parsed from the nested `text_config` block.
///
/// The checkpoint's schema is OLMo-flavoured (`n_layers`, `block_config.
/// attn_config.n_heads`, ...) rather than HF-flat, so
/// [`JinaVlmTextConfig::from_json`] does the flattening and this struct only
/// carries the resolved values.
#[derive(Debug, Clone)]
pub struct JinaVlmTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub additional_vocab_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// `q_lnorm` / `k_lnorm`: whether per-head Q/K RMSNorms are present.
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub quantization: Option<Quantization>,
}

impl Default for JinaVlmTextConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            additional_vocab_size: 128,
            intermediate_size: 6144,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            use_qk_norm: true,
            tie_word_embeddings: false,
            quantization: None,
        }
    }
}

impl JinaVlmTextConfig {
    /// Parse the nested `text_config` object.
    ///
    /// Every field falls back to the released `jinaai/jina-vlm` value so a
    /// partially specified config still loads, and both the OLMo spelling
    /// (`n_layers`, `block_config.attn_config.n_heads`, ...) and the HF spelling
    /// (`num_hidden_layers`, `num_attention_heads`, ...) are accepted because
    /// the released checkpoint carries both for some keys and only the OLMo one
    /// for the rest. The nested OLMo spelling always wins where both are
    /// present, so the released checkpoint never reaches an alias.
    ///
    /// The aliases matter because the fallback is silent: a variant that ships a
    /// flat HF `text_config` with no `block_config` would otherwise resolve
    /// `num_hidden_layers` correctly and then take the released 3B head geometry
    /// for everything else, which produces wrong output rather than an error.
    /// The private `validate_fused_qkv_geometry` below is the backstop for
    /// whatever the aliases still miss.
    pub fn from_json(text_config: &serde_json::Value) -> Self {
        let d = Self::default();
        let block = text_config.get("block_config");
        let attn = block.and_then(|b| b.get("attn_config"));
        let ffn = block.and_then(|b| b.get("ffn_config"));
        // The layer-norm eps lives on the attention sub-block in the released
        // config; fall back to the block-level norm and then to the default.
        let lnorm = attn
            .and_then(|a| a.get("lnorm_config"))
            .or_else(|| block.and_then(|b| b.get("lnorm_config")));

        let usize_at = |v: Option<&serde_json::Value>, key: &str, default: usize| -> usize {
            v.and_then(|o| o.get(key))
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
        };

        let num_hidden_layers = text_config
            .get("n_layers")
            .or_else(|| text_config.get("num_hidden_layers"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(d.num_hidden_layers);

        Self {
            hidden_size: usize_at(Some(text_config), "hidden_size", d.hidden_size),
            num_hidden_layers,
            num_attention_heads: usize_at(
                attn,
                "n_heads",
                usize_at(
                    Some(text_config),
                    "num_attention_heads",
                    d.num_attention_heads,
                ),
            ),
            num_key_value_heads: usize_at(
                attn,
                "n_kv_heads",
                usize_at(
                    Some(text_config),
                    "num_key_value_heads",
                    d.num_key_value_heads,
                ),
            ),
            head_dim: usize_at(
                attn,
                "head_dim",
                usize_at(Some(text_config), "head_dim", d.head_dim),
            ),
            vocab_size: usize_at(Some(text_config), "vocab_size", d.vocab_size),
            additional_vocab_size: usize_at(
                Some(text_config),
                "additional_vocab_size",
                d.additional_vocab_size,
            ),
            intermediate_size: usize_at(
                ffn,
                "size",
                usize_at(Some(text_config), "intermediate_size", d.intermediate_size),
            ),
            rms_norm_eps: lnorm
                .and_then(|l| l.get("eps"))
                .and_then(|v| v.as_f64())
                .unwrap_or(d.rms_norm_eps as f64) as f32,
            rope_theta: text_config
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(d.rope_theta as f64) as f32,
            use_qk_norm: attn
                .and_then(|a| a.get("q_lnorm"))
                .and_then(|v| v.as_bool())
                .unwrap_or(d.use_qk_norm),
            tie_word_embeddings: text_config
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(d.tie_word_embeddings),
            quantization: None,
        }
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
}

/// Top-level `quantization` block, shared with the vision tower and connector.
#[derive(Debug, Clone, Deserialize)]
pub struct JinaVlmQuantization {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

/// Cross-check the fused `attn.qkv` tensor against the geometry `config.json`
/// declares, before anything reaches MLX.
///
/// Every geometry field in [`JinaVlmTextConfig::from_json`] falls back to a
/// released default when its key is missing, `null`, a string, or negative, so a
/// config that contradicts the checkpoint does not fail to parse: it parses to
/// the 3B numbers. [`JinaVlmAttention::forward`] then slices Q/K/V out of the
/// fused projection at `q_dim`/`k_dim` offsets derived from those numbers. When
/// the config understates the geometry every slice is still in range, so MLX
/// clamps nothing and throws nothing, and the following reshape is
/// self-consistent by construction. Attention simply runs on the wrong parts of
/// Q/K/V and the model emits fluent nonsense with no error, no panic, and no
/// warning. Comparing against the loaded tensor is the only place the
/// disagreement is observable.
///
/// The output (row) axis carries the head geometry and is never the packed axis:
/// affine quantization compresses the input axis only, so the released 4-bit
/// `[4096, 256]` `uint32` plane still reports 4096 rows exactly like the bf16
/// `[4096, 2048]` layer-0 tensor. The row check therefore covers the dense and
/// the quantized case alike. The input width is the one that needs the two cases
/// split: dense compares the axis directly, quantized goes through
/// [`mlxcel_core::layers::validate_quantized_packing`], which reconstructs the
/// width as `scales.shape(-1) * group_size` the way MLX itself does and whose
/// mismatch would otherwise abort the process at the first forward pass.
fn validate_fused_qkv_geometry(
    weights: &WeightMap,
    config: &JinaVlmTextConfig,
    prefix: &str,
) -> Result<(), String> {
    let weight_name = format!("{prefix}.qkv.weight");
    // A missing tensor is `UnifiedLinear::from_weights`'s error to report.
    let Some(weight) = weights.get(&weight_name) else {
        return Ok(());
    };
    let shape = mlxcel_core::array_shape(weight);
    if shape.len() != 2 {
        return Err(format!(
            "Jina VLM {weight_name} has shape {shape:?}, but the fused QKV projection must be \
             2-D [out, in]"
        ));
    }

    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    if heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err(format!(
            "Jina VLM {prefix} needs a positive head geometry, but config.json gives \
             num_attention_heads {heads}, num_key_value_heads {kv_heads}, head_dim {head_dim}"
        ));
    }

    // Checked all the way: the row equality is what bounds `q_dim` and `k_dim`
    // to an i32 further down, so it must not be reached through a wrapped
    // product.
    let expected = kv_heads
        .checked_mul(2)
        .and_then(|kv| heads.checked_add(kv))
        .and_then(|fused_heads| fused_heads.checked_mul(head_dim));
    let rows = usize::try_from(shape[0]).unwrap_or(0);
    if expected != Some(rows) {
        let expected = expected
            .map(|v| v.to_string())
            .unwrap_or_else(|| "an overflowing row count".to_string());
        return Err(format!(
            "Jina VLM {prefix}.qkv has {rows} output rows but the config implies \
             (num_attention_heads {heads} + 2 * num_key_value_heads {kv_heads}) * head_dim \
             {head_dim} = {expected}; the checkpoint and config.json disagree"
        ));
    }

    let hidden_size = config.hidden_size;
    if let Some(scales) = weights.get(&format!("{prefix}.qkv.scales")) {
        let scales_shape = mlxcel_core::array_shape(scales);
        let biases_shape = weights
            .get(&format!("{prefix}.qkv.biases"))
            .map(|b| mlxcel_core::array_shape(b));
        let group_size = config.group_size();
        let bits = config.bits();
        let mode =
            mlxcel_core::layers::infer_quantization_mode(biases_shape.is_some(), group_size, bits);
        mlxcel_core::layers::validate_quantized_packing(
            &format!("{prefix}.qkv"),
            &mlxcel_core::layers::QuantizedTensorShapes {
                weight: &shape,
                scales: &scales_shape,
                biases: biases_shape.as_deref(),
            },
            hidden_size,
            group_size,
            bits,
            mode,
        )?;
    } else if usize::try_from(shape[1]).unwrap_or(0) != hidden_size {
        return Err(format!(
            "Jina VLM {weight_name} has shape {shape:?} but the config says hidden_size \
             {hidden_size}; the checkpoint and config.json disagree"
        ));
    }

    Ok(())
}

/// Fused-QKV attention with per-head Q/K RMSNorm and rotate-half RoPE.
pub struct JinaVlmAttention {
    pub qkv: UnifiedLinear,
    pub out: UnifiedLinear,
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub q_dim: i32,
    pub k_dim: i32,
}

impl JinaVlmAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let qkv = self.qkv.forward(x);
        let q = mlxcel_core::slice_last_dim(&qkv, 0, self.q_dim);
        let k = mlxcel_core::slice_last_dim(&qkv, self.q_dim, self.q_dim + self.k_dim);
        let v =
            mlxcel_core::slice_last_dim(&qkv, self.q_dim + self.k_dim, self.q_dim + self.k_dim * 2);

        // Per-head norm runs on [B, L, heads, head_dim] so RMSNorm reduces over
        // head_dim, matching `qkv_lnorm_on_heads: true`.
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let attn_out = if l > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        self.out.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &JinaVlmTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        validate_fused_qkv_geometry(weights, config, prefix)?;

        let qkv = UnifiedLinear::from_weights(weights, &format!("{prefix}.qkv"), group_size, bits)?;
        let out = UnifiedLinear::from_weights(weights, &format!("{prefix}.out"), group_size, bits)?;

        let (q_norm, k_norm) = if config.use_qk_norm {
            let q_weight = get_weight_copy(weights, &format!("{prefix}.q_norm.weight"))?;
            let k_weight = get_weight_copy(weights, &format!("{prefix}.k_norm.weight"))?;
            (
                Some(RMSNorm::new(q_weight, config.rms_norm_eps)),
                Some(RMSNorm::new(k_weight, config.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        let head_dim = config.head_dim as i32;
        let num_heads = config.num_attention_heads as i32;
        let num_kv_heads = config.num_key_value_heads as i32;

        Ok(Self {
            qkv,
            out,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: config.rope_theta,
            q_dim: num_heads * head_dim,
            k_dim: num_kv_heads * head_dim,
        })
    }
}

/// Fused SwiGLU MLP. The fused projection is ordered `[up, gate]`.
pub struct JinaVlmMLP {
    pub gate_up: UnifiedLinear,
    pub down: UnifiedLinear,
}

impl JinaVlmMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.gate_up.forward(x);
        let shape = mlxcel_core::array_shape(&projected);
        let half = shape[shape.len() - 1] / 2;

        let up = mlxcel_core::slice_last_dim(&projected, 0, half);
        let gate = mlxcel_core::slice_last_dim(&projected, half, half * 2);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &JinaVlmTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();
        let gate_up =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.gate_up"), group_size, bits)?;
        let down =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.down"), group_size, bits)?;
        Ok(Self { gate_up, down })
    }
}

/// Pre-norm decoder block.
pub struct JinaVlmBlock {
    pub attn: JinaVlmAttention,
    pub ffn: JinaVlmMLP,
    pub attn_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
}

impl JinaVlmBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.attn_norm.forward(x);
        let attn_out = self.attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.ffn_norm.forward(&h);
        let ffn_out = self.ffn.forward(&normed);
        mlxcel_core::add(&h, &ffn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &JinaVlmTextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let layer_prefix = format!("{prefix}.layers.{layer_idx}");

        let attn =
            JinaVlmAttention::from_weights(weights, config, &format!("{layer_prefix}.attn"))?;
        let ffn = JinaVlmMLP::from_weights(weights, config, &format!("{layer_prefix}.ffn"))?;

        let attn_norm_weight =
            get_weight_copy(weights, &format!("{layer_prefix}.attn_norm.weight"))?;
        let ffn_norm_weight = get_weight_copy(weights, &format!("{layer_prefix}.ffn_norm.weight"))?;

        Ok(Self {
            attn,
            ffn,
            attn_norm: RMSNorm::new(attn_norm_weight, config.rms_norm_eps),
            ffn_norm: RMSNorm::new(ffn_norm_weight, config.rms_norm_eps),
        })
    }
}

/// Jina VLM text decoder (`language_model.*`).
pub struct JinaVlmTextModel {
    pub embedding: Molmo2Embedding,
    pub layers: Vec<JinaVlmBlock>,
    pub ln_f: RMSNorm,
    pub lm_head: UnifiedLinear,
    pub eos_token_ids: Vec<i32>,
}

impl JinaVlmTextModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings(input_ids, None, caches, mask)
    }

    pub fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = match input_embeddings {
            Some(embeds) => mlxcel_core::copy(embeds),
            None => self.embedding.forward(input_ids),
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        let h = self.ln_f.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &JinaVlmTextConfig,
        prefix: &str,
        eos_token_ids: Vec<i32>,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let embedding = Molmo2Embedding::from_weights(weights, &format!("{prefix}.embedding"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(JinaVlmBlock::from_weights(weights, config, i, prefix)?);
        }

        let ln_f_weight = get_weight_copy(weights, &format!("{prefix}.ln_f.weight"))?;
        let ln_f = RMSNorm::new(ln_f_weight, config.rms_norm_eps);

        let lm_head =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.lm_head"), group_size, bits)?;

        Ok(Self {
            embedding,
            layers,
            ln_f,
            lm_head,
            eos_token_ids,
        })
    }
}

impl LanguageModel for JinaVlmTextModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        JinaVlmTextModel::forward(self, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        JinaVlmTextModel::forward_with_embeddings(self, input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embedding.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        JinaVlmTextModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

#[cfg(test)]
#[path = "jina_vlm_tests.rs"]
pub(crate) mod tests;
