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

//! Moondream3 text model implementation.
//!
//! This ports the query/caption text path of Moondream3:
//! - fused QKV attention with per-head tau scaling
//! - LayerNorm-based blocks
//! - dense GELU MLP in the early layers
//! - sparse expert MLP in the later layers

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::models::switch_layers::SwitchLinear;

#[derive(Debug, Clone, Deserialize)]
pub struct Moondream3MoeConfig {
    #[serde(default = "default_moe_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_moe_start_layer")]
    pub start_layer: usize,
    #[serde(default = "default_moe_experts_per_token")]
    pub experts_per_token: usize,
    #[serde(default = "default_moe_inner_dim")]
    pub expert_inner_dim: usize,
    #[serde(default = "default_group_size")]
    pub expert_group_size: i32,
}

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
    #[serde(default = "default_prefix_attention")]
    pub prefix_attn: usize,
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: i32,
    #[serde(default = "default_moe_config")]
    pub moe: Option<Moondream3MoeConfig>,
}

fn default_model_type() -> String {
    "moondream3".to_string()
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
    4096
}

fn default_num_heads() -> usize {
    32
}

fn default_num_kv_heads() -> usize {
    32
}

fn default_prefix_attention() -> usize {
    730
}

fn default_group_size() -> i32 {
    128
}

fn default_bits() -> i32 {
    4
}

fn default_eos_token_id() -> i32 {
    0
}

fn default_moe_config() -> Option<Moondream3MoeConfig> {
    Some(Moondream3MoeConfig {
        num_experts: default_moe_num_experts(),
        start_layer: default_moe_start_layer(),
        experts_per_token: default_moe_experts_per_token(),
        expert_inner_dim: default_moe_inner_dim(),
        expert_group_size: default_group_size(),
    })
}

fn default_moe_num_experts() -> usize {
    64
}

fn default_moe_start_layer() -> usize {
    4
}

fn default_moe_experts_per_token() -> usize {
    8
}

fn default_moe_inner_dim() -> usize {
    1024
}

impl ModelArgs {
    fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

struct Moondream3Attention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    tau_wq: UniquePtr<MlxArray>,
    tau_wv: UniquePtr<MlxArray>,
    tau_alpha: UniquePtr<MlxArray>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    scale: f32,
}

impl Moondream3Attention {
    fn from_weights(weights: &WeightMap, prefix: &str, config: &ModelArgs) -> Result<Self, String> {
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
            tau_wq: get_weight_copy(weights, &format!("{}.tau.wq", prefix))?,
            tau_wv: get_weight_copy(weights, &format!("{}.tau.wv", prefix))?,
            tau_alpha: get_weight_copy(weights, &format!("{}.tau.alpha", prefix))?,
            num_heads: config.n_heads as i32,
            num_kv_heads: config.n_kv_heads as i32,
            head_dim: config.head_dim() as i32,
            rope_dims: (config.head_dim() / 2) as i32,
            scale: 1.0 / (config.head_dim() as f32).sqrt(),
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

        let q = self.apply_tau_to_q(&qkv, &q, offset);
        let v = self.apply_tau_to_v(&qkv, &v, offset);

        // Moondream3 uses non-interleaved (halves) RoPE layout → traditional=false
        let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, 1500000.0, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, 1500000.0, 1.0, offset);

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

    fn apply_tau_to_q(&self, qkv: &MlxArray, q: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let tau = self.tau_values(qkv, &self.tau_wq, offset);
        mlxcel_core::multiply(q, &tau)
    }

    fn apply_tau_to_v(&self, qkv: &MlxArray, v: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let tau = self.tau_values(qkv, &self.tau_wv, offset);
        mlxcel_core::multiply(v, &tau)
    }

    fn tau_values(&self, qkv: &MlxArray, weight: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let qkv_shape = mlxcel_core::array_shape(qkv);
        let seq_len = qkv_shape[1] as usize;

        // Moondream3 tau uses exact GELU (F.gelu), not tanh-approximate
        let tok_feat = mlxcel_core::gelu(qkv);
        let tau_weight = mlxcel_core::swap_axes(weight, -1, -2);
        let tok = mlxcel_core::matmul(&tok_feat, &tau_weight);
        let tok = mlxcel_core::tanh(&tok);
        let tok = mlxcel_core::transpose_axes(&tok, &[0, 2, 1]);

        let positions: Vec<f32> = (0..seq_len)
            .map(|idx| (offset + idx as i32 + 1) as f32)
            .collect();
        let positions = mlxcel_core::from_slice_f32(&positions, &[1, seq_len as i32]);
        let positions = mlxcel_core::log(&positions);

        let alpha = mlxcel_core::reshape(&self.tau_alpha, &[self.num_heads, 1]);
        let tau_pos = mlxcel_core::multiply(&alpha, &positions);
        let tau_pos = mlxcel_core::sigmoid(&tau_pos);
        let qkv_dtype = mlxcel_core::array_dtype(qkv);
        let half = mlxcel_core::full_f32(&[1], 0.5, qkv_dtype);
        let one = mlxcel_core::full_f32(&[1], 1.0, qkv_dtype);
        let tau_pos = mlxcel_core::subtract(&tau_pos, &half);
        let tau_pos = mlxcel_core::add(&tau_pos, &one);
        let tau = mlxcel_core::expand_dims(&tok, -1);
        let tau_pos = mlxcel_core::expand_dims(&tau_pos, 0);
        let tau_pos = mlxcel_core::expand_dims(&tau_pos, -1);
        let tau = mlxcel_core::add(&tau, &tau_pos);
        mlxcel_core::astype(&tau, mlxcel_core::array_dtype(qkv))
    }
}

struct DenseMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl DenseMlp {
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
        let hidden = mlxcel_core::gelu_approx(&hidden);
        self.fc2.forward(&hidden)
    }
}

struct SparseMoeMlp {
    router: UnifiedLinear,
    fc1: SwitchLinear,
    fc2: SwitchLinear,
    experts_per_token: usize,
    expert_inner_dim: usize,
}

impl SparseMoeMlp {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &ModelArgs,
        moe: &Moondream3MoeConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            router: UnifiedLinear::from_weights(weights, &format!("{}.router", prefix), 64, 4)?,
            fc1: SwitchLinear::from_weights(
                weights,
                &format!("{}.fc1", prefix),
                moe.expert_group_size,
                config.bits,
            )?,
            fc2: SwitchLinear::from_weights(
                weights,
                &format!("{}.fc2", prefix),
                moe.expert_group_size,
                config.bits,
            )?,
            experts_per_token: moe.experts_per_token,
            expert_inner_dim: moe.expert_inner_dim,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let hidden_dim = orig_shape[orig_shape.len() - 1];
        let x_flat = if orig_shape.len() > 2 {
            let token_count: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            mlxcel_core::reshape(x, &[token_count, hidden_dim])
        } else {
            mlxcel_core::copy(x)
        };

        let logits = self.router.forward(&x_flat);
        let k = self.experts_per_token as i32;
        let expert_count = mlxcel_core::array_shape(&logits)[1];
        let topk = mlxcel_core::argpartition(&logits, expert_count - k, -1);
        let topk_shape = mlxcel_core::array_shape(&topk);
        let topk = mlxcel_core::slice(
            &topk,
            &[0, expert_count - k],
            &[topk_shape[0], topk_shape[1]],
        );
        let topk_logits = mlxcel_core::take_along_axis(&logits, &topk, -1);
        let scores = mlxcel_core::softmax(&topk_logits, -1);

        let x_expanded = mlxcel_core::expand_dims(&x_flat, -2);
        let x_expanded = mlxcel_core::expand_dims(&x_expanded, -3);
        let hidden = self.fc1.forward(&x_expanded, &topk, false);
        let hidden = mlxcel_core::squeeze_axis(&hidden, -2);

        let h = mlxcel_core::slice_last_dim(&hidden, 0, self.expert_inner_dim as i32);
        let g = mlxcel_core::slice_last_dim(
            &hidden,
            self.expert_inner_dim as i32,
            (self.expert_inner_dim * 2) as i32,
        );
        // Moondream3 MoE GeGLU uses exact GELU (F.gelu), not tanh-approximate
        let h = mlxcel_core::gelu(&h);
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&g));
        let g = mlxcel_core::add(&g, &one);
        let hidden = mlxcel_core::multiply(&h, &g);
        let hidden = mlxcel_core::expand_dims(&hidden, -2);
        let hidden = self.fc2.forward(&hidden, &topk, false);
        let hidden = mlxcel_core::squeeze_axis(&hidden, -2);

        let combined = crate::models::switch_layers::moe_weighted_sum(
            &hidden,
            &scores,
            mlxcel_core::array_dtype(&x_flat),
        );

        if orig_shape.len() > 2 {
            mlxcel_core::reshape(&combined, &orig_shape)
        } else {
            combined
        }
    }
}

enum MlpKind {
    Dense(DenseMlp),
    Moe(SparseMoeMlp),
}

impl MlpKind {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Moe(mlp) => mlp.forward(x),
        }
    }
}

struct Moondream3Block {
    ln: LayerNorm,
    attn: Moondream3Attention,
    mlp: MlpKind,
}

impl Moondream3Block {
    fn from_weights(
        weights: &WeightMap,
        config: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("text.blocks.{}", layer_idx);
        let mlp = if let Some(moe) = &config.moe {
            if layer_idx >= moe.start_layer {
                MlpKind::Moe(SparseMoeMlp::from_weights(
                    weights,
                    &format!("{}.mlp", prefix),
                    config,
                    moe,
                )?)
            } else {
                MlpKind::Dense(DenseMlp::from_weights(
                    weights,
                    &format!("{}.mlp", prefix),
                    config,
                )?)
            }
        } else {
            MlpKind::Dense(DenseMlp::from_weights(
                weights,
                &format!("{}.mlp", prefix),
                config,
            )?)
        };

        Ok(Self {
            ln: load_layer_norm(weights, &format!("{}.ln", prefix), 1e-5)?,
            attn: Moondream3Attention::from_weights(weights, &format!("{}.attn", prefix), config)?,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let normed = self.ln.forward(x);
        let attn = self.attn.forward(&normed, cache, mask);
        let mlp = self.mlp.forward(&normed);
        let residual = mlxcel_core::add(x, &attn);
        mlxcel_core::add(&residual, &mlp)
    }
}

pub struct Moondream3Model {
    pub embed_tokens: UnifiedEmbedding,
    layers: Vec<Moondream3Block>,
    pub post_ln: LayerNorm,
    pub lm_head: UnifiedLinear,
    pub config: ModelArgs,
}

impl Moondream3Model {
    pub fn from_weights(weights: &WeightMap, config: &ModelArgs) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.n_layers);
        for idx in 0..config.n_layers {
            layers.push(Moondream3Block::from_weights(weights, config, idx)?);
        }

        Ok(Self {
            embed_tokens: UnifiedEmbedding::from_weights(
                weights,
                "text.wte",
                config.group_size,
                config.bits,
            )?,
            post_ln: load_layer_norm(weights, "text.post_ln", 1e-5)?,
            lm_head: UnifiedLinear::from_weights(weights, "text.lm_head", 64, 4)?,
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
        0
    }
}

impl LanguageModel for Moondream3Model {
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
#[path = "moondream3_tests.rs"]
mod tests;
