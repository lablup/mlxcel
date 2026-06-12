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

//! Molmo2 text model implementation using mlxcel-core
//!
//! Key differences from OLMo2:
//! - Fused QKV projection (att_proj) instead of separate q/k/v projections
//! - Per-head QK normalization: RMSNorm(head_dim) applied AFTER reshape to [B,L,heads,head_dim]
//! - Pre-norm architecture (norm BEFORE attention/MLP, unlike OLMo2's post-norm)
//! - Fused SwiGLU MLP: ff_proj → split → silu(gate)*x → ff_out
//! - Dual embedding: base embedding (151936) + new_embedding (128)
//! - RoPE theta=5000000
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo2/language.py

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Molmo2TextConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_additional_vocab_size")]
    pub additional_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub qkv_bias: bool,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_model_type() -> String {
    "molmo2".to_string()
}
fn default_hidden_size() -> usize {
    2560
}
fn default_num_hidden_layers() -> usize {
    36
}
fn default_intermediate_size() -> usize {
    9728
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_kv_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_vocab_size() -> usize {
    151936
}
fn default_additional_vocab_size() -> usize {
    128
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    5000000.0
}

impl Molmo2TextConfig {
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

// Molmo2 Embedding (base + new_embedding).
pub struct Molmo2Embedding {
    pub embedding: UniquePtr<MlxArray>, // [vocab_size, hidden_size]
    pub new_embedding: UniquePtr<MlxArray>, // [additional_vocab_size, hidden_size]
}

impl Molmo2Embedding {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Concatenate base and new embedding tables, then index
        let full_embedding = mlxcel_core::concatenate(&self.embedding, &self.new_embedding, 0);
        mlxcel_core::embedding(&full_embedding, x)
    }

    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let embedding = weights
            .get(&format!("{}.embedding", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}.embedding", prefix))?;
        let new_embedding = weights
            .get(&format!("{}.new_embedding", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}.new_embedding", prefix))?;
        Ok(Self {
            embedding,
            new_embedding,
        })
    }
}

// Molmo2 Attention (fused QKV + per-head QK norm).
pub struct Molmo2Attention {
    pub att_proj: UnifiedLinear, // Fused QKV projection
    pub attn_out: UnifiedLinear, // Output projection
    pub q_norm: RMSNorm,         // Per-head Q norm (dims=head_dim)
    pub k_norm: RMSNorm,         // Per-head K norm (dims=head_dim)
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
    // Fused dimensions for splitting QKV
    pub q_dim: i32, // num_heads * head_dim
    pub k_dim: i32, // num_kv_heads * head_dim
}

impl Molmo2Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        // Fused QKV projection
        let qkv = self.att_proj.forward(x);

        // Split into Q, K, V
        let q = mlxcel_core::slice_last_dim(&qkv, 0, self.q_dim);
        let k = mlxcel_core::slice_last_dim(&qkv, self.q_dim, self.q_dim + self.k_dim);
        let v =
            mlxcel_core::slice_last_dim(&qkv, self.q_dim + self.k_dim, self.q_dim + self.k_dim * 2);

        // Reshape to [B, L, heads, head_dim] THEN apply per-head QK norm
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        // Apply per-head QK normalization (RMSNorm on head_dim dimension)
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        // Transpose to [B, heads, L, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, offset);

        // Update KV cache
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
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

        // Transpose back and reshape
        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);

        self.attn_out.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &Molmo2TextConfig,
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

        // Per-head QK norms (dims=head_dim)
        let q_norm_weight = get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_weight = get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?;
        let q_norm = RMSNorm::new(q_norm_weight, config.layer_norm_eps);
        let k_norm = RMSNorm::new(k_norm_weight, config.layer_norm_eps);

        let head_dim = config.head_dim as i32;
        let num_heads = config.num_attention_heads as i32;
        let num_kv_heads = config.num_key_value_heads as i32;

        Ok(Self {
            att_proj,
            attn_out,
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

// Molmo2 MLP (fused SwiGLU).
pub struct Molmo2MLP {
    pub ff_proj: UnifiedLinear, // hidden → intermediate*2 (fused gate+up)
    pub ff_out: UnifiedLinear,  // intermediate → hidden
}

impl Molmo2MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let projected = self.ff_proj.forward(x);
        let shape = mlxcel_core::array_shape(&projected);
        let half = shape[shape.len() - 1] / 2;

        // Split into [x, gate]
        let x_part = mlxcel_core::slice_last_dim(&projected, 0, half);
        let gate = mlxcel_core::slice_last_dim(&projected, half, half * 2);

        // silu(gate) * x
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &x_part);

        self.ff_out.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &Molmo2TextConfig,
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

// Molmo2 Transformer Block (pre-norm).
pub struct Molmo2TransformerBlock {
    pub self_attn: Molmo2Attention,
    pub mlp: Molmo2MLP,
    pub attn_norm: RMSNorm,
    pub ff_norm: RMSNorm,
}

impl Molmo2TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm: attn_norm → attention → residual
        let normed = self.attn_norm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm: ff_norm → MLP → residual
        let normed = self.ff_norm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &Molmo2TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let layer_prefix = format!("{}.blocks.{}", prefix, layer_idx);

        let self_attn =
            Molmo2Attention::from_weights(weights, config, &format!("{}.self_attn", layer_prefix))?;
        let mlp = Molmo2MLP::from_weights(weights, config, &format!("{}.mlp", layer_prefix))?;

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

// Molmo2 Model.
pub struct Molmo2Model {
    pub wte: Molmo2Embedding,
    pub blocks: Vec<Molmo2TransformerBlock>,
    pub ln_f: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl Molmo2Model {
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

    /// Forward pass with pre-computed embeddings (for VLM)
    pub fn forward_with_embeddings(
        &self,
        _input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeds) = input_embeddings {
            mlxcel_core::copy(embeds)
        } else {
            self.wte.forward(_input_ids)
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
        config: &Molmo2TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Load dual embedding
        let wte = Molmo2Embedding::from_weights(weights, &format!("{}.wte", prefix))?;

        // Load transformer blocks
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = Molmo2TransformerBlock::from_weights(weights, config, i, prefix)?;
            blocks.push(block);
        }

        // Final layer norm
        let ln_f_weight = get_weight_copy(weights, &format!("{}.ln_f.weight", prefix))?;
        let ln_f = RMSNorm::new(ln_f_weight, config.layer_norm_eps);

        // LM head (not tied for Molmo2)
        let lm_head_prefix = prefix
            .strip_suffix(".model")
            .map(|p| format!("{}.lm_head", p))
            .unwrap_or_else(|| "language_model.lm_head".to_string());

        let lm_head = UnifiedLinear::from_weights(weights, &lm_head_prefix, group_size, bits)?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for Molmo2Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Molmo2Model::forward(self, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Molmo2Model::forward_with_embeddings(self, input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.wte.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Molmo2Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Molmo2 uses Qwen2 tokenizer EOS tokens
        vec![151645, 151643] // <|im_end|>, <|endoftext|>
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}
