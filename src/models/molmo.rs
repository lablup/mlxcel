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

//! Molmo v1 (`model_type: "molmo"`) text model implementation using mlxcel-core.
//!
//! The decoder is the original OLMo-style backbone used by Molmo-7B. It is
//! architecturally very close to Molmo2 but with three concrete differences
//! that prevent direct reuse of `Molmo2TransformerBlock`:
//!
//! 1. **No per-head QK normalization.** Molmo2 applies `RMSNorm(head_dim)` to
//!    Q and K after the reshape; Molmo v1 does not (the v1 tensors have no
//!    `q_norm`/`k_norm`).
//! 2. **Interleaved RoPE.** v1 config sets `rope_impl: "interleave"`, which maps
//!    to `traditional=true` in the MLX fast RoPE kernel (Molmo2 uses the
//!    rotate-half variant, `traditional=false`).
//! 3. **LM head location.** v1 stores the (untied) output projection at
//!    `language_model.model.ff_out`, not a separate `language_model.lm_head`.
//!
//! Shared with Molmo2: the dual embedding table (`Molmo2Embedding` = base
//! `embedding` + `new_embedding`) and the fused-QKV / SwiGLU shapes. The fused
//! `att_proj` carries a bias (`qkv_bias: true`) and is 4-bit quantized; the
//! `UnifiedLinear` loader picks up the optional `.bias` automatically.
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo/language.py

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

// Reuse the dual-embedding table and the quantization struct from Molmo2 so the
// two families stay in lockstep (additive reuse, no behavioral change).
pub use super::molmo2::{Molmo2Embedding, Quantization};

// Configuration.
//
// Molmo-7B ships a *flat* config (text fields at the top level, vision fields
// under a minimal `vision_config`). Every text field has a serde default drawn
// from the reference `TextConfig` dataclass so the parser tolerates the flat
// schema as well as a nested `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct MolmoTextConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size", alias = "d_model")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers", alias = "n_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_intermediate_size", alias = "mlp_hidden_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_attention_heads", alias = "n_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads", alias = "n_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_embedding_size")]
    pub embedding_size: usize,
    #[serde(default = "default_additional_vocab_size")]
    pub additional_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// `rope_impl: "interleave"` → traditional/interleaved RoPE.
    #[serde(default = "default_rope_impl")]
    pub rope_impl: String,
    #[serde(default = "default_qkv_bias")]
    pub qkv_bias: bool,
    #[serde(default)]
    pub weight_tying: bool,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_model_type() -> String {
    "molmo".to_string()
}
fn default_hidden_size() -> usize {
    3584
}
fn default_num_hidden_layers() -> usize {
    28
}
fn default_intermediate_size() -> usize {
    37888
}
fn default_num_attention_heads() -> usize {
    28
}
fn default_num_kv_heads() -> usize {
    4
}
fn default_vocab_size() -> usize {
    152064
}
fn default_embedding_size() -> usize {
    152064
}
fn default_additional_vocab_size() -> usize {
    128
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_rope_impl() -> String {
    "interleave".to_string()
}
fn default_qkv_bias() -> bool {
    true
}

impl MolmoTextConfig {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether RoPE uses the interleaved ("traditional") layout.
    pub fn rope_traditional(&self) -> bool {
        self.rope_impl == "interleave"
    }
}

// Molmo v1 Attention (fused biased QKV, NO QK norm, interleaved RoPE).
pub struct MolmoAttention {
    pub att_proj: UnifiedLinear, // Fused QKV projection (with bias)
    pub attn_out: UnifiedLinear, // Output projection
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    pub rope_traditional: bool,
    pub q_dim: i32, // num_heads * head_dim
    pub k_dim: i32, // num_kv_heads * head_dim
}

impl MolmoAttention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Fused QKV projection (att_proj carries its bias internally).
        let qkv = self.att_proj.forward(x);

        // Split into Q, K, V along the last dim.
        let q = mlxcel_core::slice_last_dim(&qkv, 0, self.q_dim);
        let k = mlxcel_core::slice_last_dim(&qkv, self.q_dim, self.q_dim + self.k_dim);
        let v =
            mlxcel_core::slice_last_dim(&qkv, self.q_dim + self.k_dim, self.q_dim + self.k_dim * 2);

        // Reshape to [B, L, heads, head_dim]. Unlike Molmo2 there is NO QK norm.
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Transpose to [B, heads, L, head_dim].
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Interleaved ("traditional") RoPE for Molmo v1.
        let q = mlxcel_core::fast_rope(
            &q,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );
        let k = mlxcel_core::fast_rope(
            &k,
            self.head_dim,
            self.rope_traditional,
            self.rope_base,
            1.0,
            offset,
        );

        // Update KV cache.
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention (GQA handled by the kernel).
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

        // Transpose back and merge heads.
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.attn_out.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &MolmoTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let att_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.att_proj", prefix),
            group_size,
            bits,
        )?;
        let attn_out = UnifiedLinear::from_weights(
            weights,
            &format!("{}.attn_out", prefix),
            group_size,
            bits,
        )?;

        let head_dim = config.head_dim() as i32;
        let num_heads = config.num_attention_heads as i32;
        let num_kv_heads = config.num_key_value_heads as i32;

        Ok(Self {
            att_proj,
            attn_out,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: config.rope_theta,
            rope_traditional: config.rope_traditional(),
            q_dim: num_heads * head_dim,
            k_dim: num_kv_heads * head_dim,
        })
    }
}

// Molmo v1 MLP (fused SwiGLU: ff_proj → split → silu(gate)*x → ff_out).
pub struct MolmoMLP {
    pub ff_proj: UnifiedLinear, // hidden → intermediate*2 (fused gate+up)
    pub ff_out: UnifiedLinear,  // intermediate → hidden
}

impl MolmoMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.ff_proj.forward(x);
        let shape = mlxcel_core::array_shape(&projected);
        let half = shape[shape.len() - 1] / 2;

        // Reference splits as (x, gate) and returns silu(gate) * x.
        let x_part = mlxcel_core::slice_last_dim(&projected, 0, half);
        let gate = mlxcel_core::slice_last_dim(&projected, half, half * 2);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &x_part);

        self.ff_out.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &MolmoTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let ff_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.ff_proj", prefix), group_size, bits)?;
        let ff_out =
            UnifiedLinear::from_weights(weights, &format!("{}.ff_out", prefix), group_size, bits)?;

        Ok(Self { ff_proj, ff_out })
    }
}

// Molmo v1 Transformer Block (pre-norm; attn_norm/ff_norm are RMSNorm).
pub struct MolmoBlock {
    pub self_attn: MolmoAttention,
    pub mlp: MolmoMLP,
    pub attn_norm: RMSNorm,
    pub ff_norm: RMSNorm,
}

impl MolmoBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.attn_norm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.ff_norm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &MolmoTextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let layer_prefix = format!("{}.blocks.{}", prefix, layer_idx);

        let self_attn = MolmoAttention::from_weights(weights, config, &layer_prefix)?;
        let mlp = MolmoMLP::from_weights(weights, config, &layer_prefix)?;

        let attn_norm_weight =
            get_weight_copy(weights, &format!("{}.attn_norm.weight", layer_prefix))?;
        let ff_norm_weight = get_weight_copy(weights, &format!("{}.ff_norm.weight", layer_prefix))?;

        let attn_norm = RMSNorm::new(attn_norm_weight, config.layer_norm_eps);
        let ff_norm = RMSNorm::new(ff_norm_weight, config.layer_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            attn_norm,
            ff_norm,
        })
    }
}

// Molmo v1 text model.
pub struct MolmoModel {
    pub wte: Molmo2Embedding,
    pub blocks: Vec<MolmoBlock>,
    pub ln_f: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl MolmoModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.wte.forward(input_ids);

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &mut caches[i], mask);
        }

        let h = self.ln_f.forward(&h);
        self.lm_head.forward(&h)
    }

    /// Forward pass with pre-computed embeddings (for the VLM merge path).
    pub fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.wte.forward(input_ids)
        };

        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &mut caches[i], mask);
        }

        let h = self.ln_f.forward(&h);
        self.lm_head.forward(&h)
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.blocks.len()).map(|_| KVCache::new()).collect()
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &MolmoTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        let wte = Molmo2Embedding::from_weights(weights, &format!("{}.wte", prefix))?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            blocks.push(MolmoBlock::from_weights(weights, config, i, prefix)?);
        }

        let ln_f_weight = get_weight_copy(weights, &format!("{}.ln_f.weight", prefix))?;
        let ln_f = RMSNorm::new(ln_f_weight, config.layer_norm_eps);

        // Untied LM head lives at `{prefix}.ff_out` (i.e.
        // `language_model.model.ff_out`), distinct from the per-block `ff_out`.
        let lm_head =
            UnifiedLinear::from_weights(weights, &format!("{}.ff_out", prefix), group_size, bits)?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for MolmoModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        MolmoModel::forward(self, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        MolmoModel::forward_with_embeddings(self, input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.wte.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        MolmoModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Molmo-7B uses the Qwen2 tokenizer: <|endoftext|>, <|im_end|>.
        vec![151643, 151645]
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}
