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

//! Moondream2 text model implementation.
//!
//! Moondream2 uses a Phi-1.5-style dense decoder:
//! - fused QKV projection with bias
//! - partial rotary embedding (`partial_rotary_factor` of `head_dim`)
//! - a single LayerNorm per block feeding attention and MLP in parallel
//!   (`x + attn(ln(x)) + mlp(ln(x))`)
//! - tanh-approximate GELU MLP
//! - a final LayerNorm followed by an untied `lm_head` with bias
//!
//! The port mirrors the checkpoint's own reference implementation
//! (`text.py`, `rope.py` and `layers.py` shipped in `vikhyatk/moondream2`);
//! `references/mlx-vlm/mlx_vlm/models/moondream2/language.py` implements the
//! same math for the pre-2025 key layout. Note that the reference RoPE reads
//! the 32 rotary dims in NeoX half-split pairs and writes the rotated pairs
//! back interleaved; because q and k receive the same fixed output
//! permutation, attention scores are invariant to it, so `fast_rope` with
//! `traditional = false` (half-split in, half-split out) is exact.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub dim: usize,
    #[serde(default = "default_intermediate_size")]
    pub ff_dim: usize,
    #[serde(default = "default_num_layers")]
    pub n_layers: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_context")]
    pub max_context: usize,
    #[serde(default = "default_num_heads")]
    pub n_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub n_kv_heads: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: i32,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: i32,
}

fn default_model_type() -> String {
    "moondream2".to_string()
}

fn default_hidden_size() -> usize {
    2048
}

fn default_intermediate_size() -> usize {
    8192
}

fn default_num_layers() -> usize {
    24
}

fn default_vocab_size() -> usize {
    51200
}

fn default_max_context() -> usize {
    2048
}

fn default_num_heads() -> usize {
    32
}

fn default_num_kv_heads() -> usize {
    32
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_partial_rotary_factor() -> f32 {
    0.5
}

fn default_layer_norm_eps() -> f32 {
    1e-5
}

fn default_group_size() -> i32 {
    64
}

fn default_bits() -> i32 {
    4
}

/// Default begin/end-of-text id, matching the LEGACY (2025-01-09 ..
/// 2025-04-14) moondream2 revisions whose GPT-2/CodeGen tokenizer uses
/// `<|endoftext|>` (id 50256) as bos, eos and unk. Starmie-era checkpoints
/// (2025-06-21+) use id 0 with the `moondream/starmie-v1` tokenizer, exactly
/// like Moondream3; the loader overrides both ids per detected era, so these
/// serde defaults only cover configs built outside the loader.
fn default_eos_token_id() -> i32 {
    50256
}

fn default_bos_token_id() -> i32 {
    50256
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Number of leading head dimensions RoPE is applied to.
    pub fn rope_dims(&self) -> i32 {
        (self.partial_rotary_factor * self.head_dim() as f32) as i32
    }

    /// Fused QKV output width: `(n_heads + 2 * n_kv_heads) * head_dim`.
    pub fn qkv_dim(&self) -> i32 {
        (self.n_heads as i32 + 2 * self.n_kv_heads as i32) * self.head_dim() as i32
    }
}

struct Moondream2Attention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    rope_base: f32,
    scale: f32,
}

impl Moondream2Attention {
    fn from_weights(weights: &WeightMap, prefix: &str, config: &ModelArgs) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            qkv: UnifiedLinear::from_weights(
                weights,
                &format!("{}.qkv", prefix),
                config.group_size,
                config.bits,
            )?,
            proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.proj", prefix),
                config.group_size,
                config.bits,
            )?,
            num_heads: config.n_heads as i32,
            num_kv_heads: config.n_kv_heads as i32,
            head_dim,
            rope_dims: config.rope_dims(),
            rope_base: config.rope_theta,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        let qkv = self.qkv.forward(x);
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = mlxcel_core::slice_last_dim(&qkv, 0, q_dim);
        let k = mlxcel_core::slice_last_dim(&qkv, q_dim, q_dim + kv_dim);
        let v = mlxcel_core::slice_last_dim(&qkv, q_dim + kv_dim, q_dim + 2 * kv_dim);

        let q = mlxcel_core::reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Partial rotary embedding: non-interleaved (halves) layout matches
        // `nn.RoPE(traditional=False)` in the reference.
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);

        let (cache_k, cache_v) = cache.update_and_fetch(k, v);
        let attn = if seq_len > 1 && mask.is_none() {
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[batch, seq_len, self.num_heads * self.head_dim]);
        self.proj.forward(&attn)
    }
}

struct Moondream2Mlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl Moondream2Mlp {
    fn from_weights(weights: &WeightMap, prefix: &str, config: &ModelArgs) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(
                weights,
                &format!("{}.fc1", prefix),
                config.group_size,
                config.bits,
            )?,
            fc2: UnifiedLinear::from_weights(
                weights,
                &format!("{}.fc2", prefix),
                config.group_size,
                config.bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = self.fc1.forward(x);
        // nn.GELU(approx="tanh")
        let hidden = mlxcel_core::gelu_approx(&hidden);
        self.fc2.forward(&hidden)
    }
}

struct Moondream2Block {
    ln: LayerNorm,
    attn: Moondream2Attention,
    mlp: Moondream2Mlp,
}

impl Moondream2Block {
    fn from_weights(
        weights: &WeightMap,
        config: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("text.blocks.{}", layer_idx);
        Ok(Self {
            ln: load_layer_norm(weights, &format!("{}.ln", prefix), config.layer_norm_eps)?,
            attn: Moondream2Attention::from_weights(weights, &format!("{}.attn", prefix), config)?,
            mlp: Moondream2Mlp::from_weights(weights, &format!("{}.mlp", prefix), config)?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Parallel residual on a single shared LayerNorm.
        let normed = self.ln.forward(x);
        let attn = self.attn.forward(&normed, cache, mask);
        let mlp = self.mlp.forward(&normed);
        let residual = mlxcel_core::add(x, &attn);
        mlxcel_core::add(&residual, &mlp)
    }
}

pub struct Moondream2Model {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<Moondream2Block>,
    pub post_ln: LayerNorm,
    pub lm_head: UnifiedLinear,
    pub config: ModelArgs,
}

impl Moondream2Model {
    pub fn from_weights(weights: &WeightMap, config: &ModelArgs) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.n_layers);
        for idx in 0..config.n_layers {
            layers.push(Moondream2Block::from_weights(weights, config, idx)?);
        }

        Ok(Self {
            embed_tokens: UnifiedEmbedding::from_weights(
                weights,
                "text.wte",
                config.group_size,
                config.bits,
            )?,
            post_ln: load_layer_norm(weights, "text.post_ln", config.layer_norm_eps)?,
            lm_head: UnifiedLinear::from_weights(
                weights,
                "text.lm_head",
                config.group_size,
                config.bits,
            )?,
            layers,
            config: config.clone(),
        })
    }

    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut hidden = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &mut caches[idx], mask);
        }

        let hidden = self.post_ln.forward(&hidden);
        self.lm_head.forward(&hidden)
    }

    fn build_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.config.eos_token_id]
    }

    pub fn bos_token_id(&self) -> i32 {
        self.config.bos_token_id
    }
}

impl LanguageModel for Moondream2Model {
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

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.build_caches()
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids()
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Missing weight: {}", name))
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
    let bias = weights
        .get(&format!("{}.bias", prefix))
        .map(|value| mlxcel_core::copy(value));
    Ok(LayerNorm::new(weight, bias, eps))
}

#[cfg(test)]
#[path = "moondream2_tests.rs"]
mod tests;
