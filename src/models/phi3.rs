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

//! Phi3 model implementation using mlxcel-core
//!
//! Key differences from standard Llama:
//! - Fused QKV projection (single linear outputs Q, K, V concatenated)
//! - Fused gate_up projection in MLP (single linear outputs gate, up concatenated)
//! - Optional partial RoPE for Phi4MM-style checkpoints

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::path::Path;

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,

    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    #[serde(default)]
    pub quantization: Option<Quantization>,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type", default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// SuScaledRoPE / longrope: per-dimension scaling factors for long sequences.
    #[serde(default)]
    pub long_factor: Option<Vec<f64>>,
    /// SuScaledRoPE / longrope: per-dimension scaling factors for short sequences.
    #[serde(default)]
    pub short_factor: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_max_position_embeddings() -> usize {
    131072
}

impl ModelArgs {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn rope_dims(&self) -> usize {
        ((self.head_dim() as f32) * self.partial_rotary_factor) as usize
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

// Phi3 Attention (with fused QKV).
pub struct Phi3Attention {
    // Phi3 uses fused QKV projection: outputs [Q, K, V] concatenated
    pub qkv_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub rope_base: f32,
    /// SuScaledRoPE: pre-computed frequencies (long_factor * base_freqs).
    /// Used by: Phi4MM, Phi3V (longrope scaling)
    pub su_rope_freqs: Option<UniquePtr<MlxArray>>,
    /// SuScaledRoPE: input scaling factor applied to Q/K before rotation.
    /// = sqrt(1 + log(max_pos/orig_max_pos) / log(orig_max_pos))
    pub su_rope_scale: f32,
    /// Pre-computed scale array to avoid per-token allocation in SuScaledRoPE forward
    su_rope_scale_arr: Option<UniquePtr<MlxArray>>,
}

impl Phi3Attention {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let offset = cache.offset;

        // Fused QKV split + RoPE preparation. The SuScaledRoPE quantized path
        // mirrors mlx-lm/mlx-vlm by scaling only the rotary prefix and keeps
        // the projection/split/reshape/transpose/RoPE chain in one bridge call.
        let (q, k, v) = if let Some(ref freqs) = self.su_rope_freqs {
            if let Some((q, k, v)) = self.qkv_proj.forward_fused_qkv_split_su_scaled_rope(
                x,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rope_dims,
                freqs,
                self.su_rope_scale,
                offset,
            ) {
                (q, k, v)
            } else {
                self.prepare_qkv_with_rope(x, b, l, offset, Some(freqs))
            }
        } else {
            self.prepare_qkv_with_rope(x, b, l, offset, None)
        };

        // Update KV cache and get sliced views
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        // Scaled dot-product attention
        let attn_out = if l > 1 && mask.is_none() {
            // Prefill: use causal masking
            mlxcel_core::causal_attention(&q, &cache_k, &cache_v, self.scale, 0.0, 0)
        } else {
            // Single token or explicit mask
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
        self.o_proj.forward(&attn_out)
    }

    fn prepare_qkv_with_rope(
        &self,
        x: &MlxArray,
        b: i32,
        l: i32,
        offset: i32,
        su_freqs: Option<&MlxArray>,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let qkv = self.qkv_proj.forward(x);
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;

        let q = mlxcel_core::slice_last_dim(&qkv, 0, q_size);
        let k = mlxcel_core::slice_last_dim(&qkv, q_size, q_size + kv_size);
        let v = mlxcel_core::slice_last_dim(&qkv, q_size + kv_size, q_size + 2 * kv_size);

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);

        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let (q, k) = if let Some(freqs) = su_freqs {
            let (q, k) = self.scale_su_rope_rotary_prefix(&q, &k);
            let q =
                mlxcel_core::fast_rope_with_freqs(&q, self.rope_dims, false, 1.0, offset, freqs);
            let k =
                mlxcel_core::fast_rope_with_freqs(&k, self.rope_dims, false, 1.0, offset, freqs);
            (q, k)
        } else {
            let q = mlxcel_core::fast_rope(&q, self.rope_dims, false, self.rope_base, 1.0, offset);
            let k = mlxcel_core::fast_rope(&k, self.rope_dims, false, self.rope_base, 1.0, offset);
            (q, k)
        };

        (q, k, v)
    }

    fn scale_su_rope_rotary_prefix(
        &self,
        q: &MlxArray,
        k: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let Some(ref scale_arr) = self.su_rope_scale_arr else {
            return (
                mlxcel_core::reshape(q, &mlxcel_core::array_shape(q)),
                mlxcel_core::reshape(k, &mlxcel_core::array_shape(k)),
            );
        };

        let scale_rotary = |x: &MlxArray| {
            let shape = mlxcel_core::array_shape(x);
            let last = shape.len() - 1;
            let mut rotary_end = shape.clone();
            rotary_end[last] = self.rope_dims;
            let rotary = mlxcel_core::slice(x, &vec![0; shape.len()], &rotary_end);
            let rotary = mlxcel_core::multiply(&rotary, scale_arr);
            mlxcel_core::slice_update(x, &rotary, &vec![0; shape.len()], &rotary_end)
        };

        (scale_rotary(q), scale_rotary(k))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Fused QKV projection
        let qkv_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.qkv_proj", prefix),
            group_size,
            bits,
        )?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.o_proj", prefix), group_size, bits)?;

        let head_dim = args.head_dim() as i32;

        let mut attn = Self {
            qkv_proj,
            o_proj,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_kv_heads() as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_dims: args.rope_dims() as i32,
            rope_base: args.rope_theta,
            su_rope_freqs: None,
            su_rope_scale: 1.0,
            su_rope_scale_arr: None,
        };

        // Configure SuScaledRoPE if longrope scaling is present
        if let Some(ref scaling) = args.rope_scaling {
            let rope_type = scaling.rope_type.as_deref().unwrap_or("");
            if rope_type == "longrope" || rope_type == "su" {
                attn.configure_su_rope(args);
            }
        }

        Ok(attn)
    }

    /// Configure SuScaledRoPE from model config.
    /// Used by: Phi4MM, Phi3V (longrope/su scaling)
    fn configure_su_rope(&mut self, args: &ModelArgs) {
        let scaling = match &args.rope_scaling {
            Some(s) => s,
            None => return,
        };
        let long_factor = match &scaling.long_factor {
            Some(f) => f,
            None => return,
        };

        let dims = self.rope_dims as usize;
        let half_dims = dims / 2;
        if long_factor.len() < half_dims {
            return;
        }

        // Compute modified frequencies: long_factor[i] * base^(2i/dims)
        let base = self.rope_base as f64;
        let mut freqs = vec![0.0f32; half_dims];
        for i in 0..half_dims {
            let exponent = (2 * i) as f64 / dims as f64;
            freqs[i] = (long_factor[i] * base.powf(exponent)) as f32;
        }
        self.su_rope_freqs = Some(mlxcel_core::from_slice_f32(&freqs, &[half_dims as i32]));

        // Compute scaling factor
        let orig_max = scaling.original_max_position_embeddings.unwrap_or(4096) as f64;
        let max_pos = args.max_position_embeddings as f64;
        let factor = max_pos / orig_max;
        let scale = if factor <= 1.0 {
            1.0
        } else {
            (1.0 + factor.ln() / orig_max.ln()).sqrt() as f32
        };
        self.su_rope_scale = scale;
        // Pre-compute scale array to avoid per-token from_slice_f32 allocation
        if (scale - 1.0).abs() > 1e-6 {
            self.su_rope_scale_arr = Some(mlxcel_core::from_slice_f32(&[scale], &[1]));
        }
    }
}

// Phi3 MLP (with fused gate_up).
pub struct Phi3MLP {
    // Fused gate_up projection: outputs [gate, up] concatenated
    pub gate_up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    pub hidden_size: i32,
}

impl Phi3MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Fused gate_up projection
        let gate_up = self.gate_up_proj.forward(x);

        // Split gate and up
        let gate = mlxcel_core::slice_last_dim(&gate_up, 0, self.hidden_size);
        let up = mlxcel_core::slice_last_dim(&gate_up, self.hidden_size, 2 * self.hidden_size);

        // SwiGLU: down_proj(silu(gate) * up)
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);

        self.down_proj.forward(&activated)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        let gate_up_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.gate_up_proj", prefix),
            group_size,
            bits,
        )?;
        let down_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.down_proj", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            hidden_size: args.intermediate_size as i32,
        })
    }
}

// Phi3 Transformer Block.
pub struct Phi3TransformerBlock {
    pub self_attn: Phi3Attention,
    pub mlp: Phi3MLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl Phi3TransformerBlock {
    pub fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Pre-norm attention
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm FFN
        let normed = self.post_attention_layernorm.forward(&h);
        let ff_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &ff_out)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{}", layer_idx);

        let self_attn =
            Phi3Attention::from_weights(weights, args, &format!("{}.self_attn", prefix))?;
        let mlp = Phi3MLP::from_weights(weights, args, &format!("{}.mlp", prefix))?;

        let input_norm_weight =
            get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?;
        let post_norm_weight = get_weight_copy(
            weights,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?;

        let input_layernorm = RMSNorm::new(input_norm_weight, args.rms_norm_eps);
        let post_attention_layernorm = RMSNorm::new(post_norm_weight, args.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

// Phi3 Model.
pub struct Phi3Model {
    pub embed_tokens: UnifiedEmbedding,
    pub layers: Vec<Phi3TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: UnifiedLinear,
}

impl Phi3Model {
    /// Forward pass through the entire model
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_impl(input_ids, None, caches, mask)
    }

    /// Forward with optional pre-computed embeddings (for VLM prefill).
    /// Used by: Phi4MM VLM, Phi4-SigLIP VLM
    pub fn forward_impl(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            self.embed_tokens.forward(input_ids)
        };

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[i], mask);
        }

        // Final norm
        let h = self.norm.forward(&h);

        // LM head
        self.lm_head.forward(&h)
    }

    /// Get raw token embeddings (for VLM embedding merge).
    /// Used by: Phi4MM VLM, Phi4-SigLIP VLM
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.forward(input_ids)
    }

    /// Create KV caches for all layers
    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Load model from directory
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

    /// Create model from loaded weights
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let group_size = args.group_size();
        let bits = args.bits();

        // Load quantized embedding
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", group_size, bits)?;

        // Load layers
        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            let layer = Phi3TransformerBlock::from_weights(weights, args, i)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = get_weight_copy(weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        // Load LM head
        let lm_head = if args.tie_word_embeddings {
            // Use embedding weights for lm_head
            UnifiedLinear::from_weights(weights, "model.embed_tokens", group_size, bits)?
        } else {
            UnifiedLinear::from_weights(weights, "lm_head", group_size, bits)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

// LanguageModel trait implementation.
impl LanguageModel for Phi3Model {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        Phi3Model::forward(self, input_ids, caches, mask)
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
        Some(self.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Phi3Model::make_caches(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Phi3 EOS tokens
        vec![32000, 32007] // <|end|>, <|endoftext|>
    }
}

#[cfg(test)]
#[path = "phi3_tests.rs"]
mod tests;
