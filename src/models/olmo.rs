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

//! OLMo model implementation using mlxcel-core
//!
//! Key differences from Llama:
//! - Uses LayerNorm (without affine parameters) instead of RMSNorm
//! - Uses fused QKV projection (att_proj) instead of separate Q/K/V
//! - Uses fused gate/up projection (ff_proj) for SwiGLU MLP
//! - Different parameter naming: wte, blocks, att_norm, ff_norm

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{Embedding, KVCache, LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub d_model: usize,
    pub n_layers: usize,
    #[serde(default)]
    pub mlp_hidden_size: Option<usize>,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub embedding_size: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_traditional: bool,

    #[serde(default)]
    pub weight_tying: bool,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_mlp_ratio() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Get mlp_hidden_size, calculating from d_model * mlp_ratio if not specified
    pub fn get_mlp_hidden_size(&self) -> usize {
        self.mlp_hidden_size
            .unwrap_or(self.d_model * self.mlp_ratio)
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

// Attention with fused QKV projection.
pub struct OlmoAttention {
    pub att_proj: UnifiedLinear, // Fused QKV projection
    pub attn_out: UnifiedLinear,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_base: f32,
}

impl OlmoAttention {
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

        // Split into Q, K, V (each is d_model size)
        let d_model = self.num_heads * self.head_dim;
        let q = mlxcel_core::slice_last_dim(&qkv, 0, d_model);
        let k = mlxcel_core::slice_last_dim(&qkv, d_model, 2 * d_model);
        let v = mlxcel_core::slice_last_dim(&qkv, 2 * d_model, 3 * d_model);

        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_heads, self.head_dim]);

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let offset = cache.offset;

        // Apply RoPE
        let q = mlxcel_core::fast_rope(
            &q,
            self.head_dim,
            false, // not traditional
            self.rope_base,
            1.0,
            offset,
        );
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

        // Output projection
        self.attn_out.forward(&attn_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

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

        let head_dim = args.head_dim() as i32;

        Ok(Self {
            att_proj,
            attn_out,
            num_heads: args.n_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base: args.rope_theta,
        })
    }
}

// MLP with fused gate/up projection (SwiGLU).
pub struct OlmoMLP {
    pub ff_proj: UnifiedLinear, // Fused gate+up projection
    pub ff_out: UnifiedLinear,
}

impl OlmoMLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Fused gate+up projection
        let projected = self.ff_proj.forward(x);

        // Get shape to determine split point
        let shape = mlxcel_core::array_shape(&projected);
        let total_size = shape[shape.len() - 1];
        let half = total_size / 2;

        // Split into x1 (up) and x2 (gate)
        // Python: x1, x2 = split(...); swiglu(x2, x1) = silu(x2) * x1
        let up = mlxcel_core::slice_last_dim(&projected, 0, half);
        let gate = mlxcel_core::slice_last_dim(&projected, half, total_size);

        // SwiGLU: silu(gate) * up = silu(x2) * x1
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.ff_out.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let ff_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.ff_proj", prefix), group_size, bits)?;
        let ff_out =
            UnifiedLinear::from_weights(weights, &format!("{}.ff_out", prefix), group_size, bits)?;

        Ok(Self { ff_proj, ff_out })
    }
}

// Transformer Block.
pub struct OlmoTransformerBlock {
    pub self_attn: OlmoAttention,
    pub mlp: OlmoMLP,
    pub att_norm: LayerNorm,
    pub ff_norm: LayerNorm,
}

impl OlmoTransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.att_norm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN
        let normed = self.ff_norm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.transformer.blocks.{}", layer_idx);

        // OLMo weights don't have .self_attn or .mlp prefix - load directly from block
        let self_attn = OlmoAttention::from_weights(weights, args, &prefix)?;
        let mlp = OlmoMLP::from_weights(weights, args, &prefix)?;

        // OLMo uses LayerNorm without affine parameters (no weight/bias in weights)
        // Create identity LayerNorm with ones for weight and no bias
        let att_norm_weight =
            mlxcel_core::ones(&[args.d_model as i32], mlxcel_core::dtype::FLOAT32);
        let att_norm = LayerNorm::new(att_norm_weight, None, 1e-5);
        let ff_norm_weight = mlxcel_core::ones(&[args.d_model as i32], mlxcel_core::dtype::FLOAT32);
        let ff_norm = LayerNorm::new(ff_norm_weight, None, 1e-5);

        Ok(Self {
            self_attn,
            mlp,
            att_norm,
            ff_norm,
        })
    }
}

// OLMo Model.
pub struct OlmoModel {
    pub wte: Embedding,
    pub blocks: Vec<OlmoTransformerBlock>,
    pub norm: LayerNorm,
    pub ff_out: Option<UnifiedLinear>,
    pub args: ModelArgs,
}

impl OlmoModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Embed tokens
        let mut h = self.wte.forward(input_ids);

        // Pass through transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // Output projection
        if let Some(ff_out) = &self.ff_out {
            ff_out.forward(&h)
        } else {
            // Use embedding weights (tied weights)
            self.wte.as_linear(&h)
        }
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.blocks.len()).map(|_| KVCache::new()).collect()
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        // Load weights
        let weights = crate::models::load_text_weights(model_dir, None)?;

        // Create model
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load embedding (non-quantized, weight names have "model." prefix)
        let wte = Embedding::from_weights(weights, "model.transformer.wte")?;

        // Load blocks
        let mut blocks = Vec::with_capacity(args.n_layers);
        for i in 0..args.n_layers {
            let block = OlmoTransformerBlock::from_weights(weights, args, i)?;
            blocks.push(block);
        }

        // Load final norm (without affine)
        let norm_weight = mlxcel_core::ones(&[args.d_model as i32], mlxcel_core::dtype::FLOAT32);
        let norm = LayerNorm::new(norm_weight, None, 1e-5);

        // Load output projection if not tied
        let ff_out = if !args.weight_tying {
            Some(UnifiedLinear::from_weights(
                weights,
                "model.transformer.ff_out",
                group_size,
                bits,
            )?)
        } else {
            None
        };

        Ok(Self {
            wte,
            blocks,
            norm,
            ff_out,
            args: args.clone(),
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for OlmoModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        OlmoModel::forward(self, input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        OlmoModel::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // OLMo standard EOS token
        vec![50279]
    }
}
